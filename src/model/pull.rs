/// HuggingFace Hub model pull support.
///
/// Parses model references of the form:
///   `<owner>/<repo>:<quantization>`  e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M`
///   `<owner>/<repo>/<filename.gguf>` e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF/model-Q4_K_M.gguf`
///
/// Features:
/// - Streams download progress via a channel (SSE-friendly)
/// - Resume interrupted downloads via HTTP Range requests
/// - HuggingFace token support for private/gated models (`HF_TOKEN` env or request field)
/// - SHA-256 verified, content-addressed blob store
#[cfg(feature = "hf")]
pub mod hf {
    use futures::StreamExt;
    use reqwest::Client;
    use tokio::io::AsyncWriteExt;

    use crate::dirs;
    use crate::error::{PowerError, Result};
    use crate::model::manifest::{ModelFormat, ModelManifest};
    use crate::model::storage;

    /// HuggingFace Hub base URL.
    const HF_BASE: &str = "https://huggingface.co";

    /// Progress event emitted during a pull operation.
    #[derive(Debug, Clone)]
    pub enum PullProgress {
        /// Download in progress: bytes completed out of total (total=0 if unknown).
        Downloading { completed: u64, total: u64 },
        /// Resuming a previous interrupted download from `offset` bytes.
        Resuming { offset: u64, total: u64 },
        /// SHA-256 verification in progress.
        Verifying,
        /// Pull completed successfully.
        Done,
    }

    /// Parsed HuggingFace model reference.
    #[derive(Debug, Clone)]
    pub struct HfRef {
        /// HuggingFace repo id, e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF`.
        pub repo: String,
        /// Exact filename to download, e.g. `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.
        pub filename: String,
    }

    impl HfRef {
        /// Parse a model reference string into a `HfRef`.
        ///
        /// Supported formats:
        /// - `owner/repo:Q4_K_M`  — resolves filename via HF API
        /// - `owner/repo/file.gguf` — direct filename
        pub fn parse(name: &str) -> Result<Self> {
            // Format: owner/repo/filename.gguf
            let parts: Vec<&str> = name.splitn(3, '/').collect();
            if parts.len() == 3 {
                return Ok(HfRef {
                    repo: format!("{}/{}", parts[0], parts[1]),
                    filename: parts[2].to_string(),
                });
            }

            // Format: owner/repo:quantization
            if let Some(colon) = name.rfind(':') {
                let repo = &name[..colon];
                let quant = &name[colon + 1..];
                if repo.contains('/') {
                    return Ok(HfRef {
                        repo: repo.to_string(),
                        filename: quant.to_string(), // resolved later via API
                    });
                }
            }

            Err(PowerError::Server(format!(
                "invalid model reference '{}': expected 'owner/repo:quantization' or 'owner/repo/file.gguf'",
                name
            )))
        }

        /// Build the direct download URL for this model file.
        pub fn download_url(&self) -> String {
            format!("{HF_BASE}/{}/resolve/main/{}", self.repo, self.filename)
        }

        /// Build the HF API URL to list repo files (used to resolve quantization → filename).
        pub fn api_files_url(&self) -> String {
            format!("{HF_BASE}/api/models/{}/tree/main", self.repo)
        }

        /// Stable partial-download filename derived from the download URL.
        ///
        /// Using a deterministic name (SHA-256 of the URL) means a resumed pull
        /// after a crash or restart will find the same partial file.
        pub fn partial_filename(&self) -> String {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(self.download_url().as_bytes());
            let digest = h.finalize();
            format!("partial-{:x}", digest)
        }
    }

    /// Resolve the effective HF token: explicit arg → `HF_TOKEN` env var → None.
    fn resolve_token(token: Option<&str>) -> Option<String> {
        token
            .map(|t| t.to_string())
            .or_else(|| std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()))
    }

    /// Build a reqwest `Client` with the given optional bearer token.
    fn build_client(token: Option<&str>) -> Result<(Client, Option<String>)> {
        let effective_token = resolve_token(token);
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref tok) = effective_token {
            let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {tok}"))
                .map_err(|e| PowerError::Server(format!("invalid HF token: {e}")))?;
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
        let client = Client::builder()
            .user_agent("a3s-power/0.2")
            .default_headers(headers)
            .build()
            .map_err(|e| PowerError::Server(format!("failed to build HTTP client: {e}")))?;
        Ok((client, effective_token))
    }

    /// Resolve a quantization tag (e.g. `Q4_K_M`) to an actual filename by
    /// querying the HuggingFace repo file listing.
    async fn resolve_filename(client: &Client, hf_ref: &HfRef) -> Result<String> {
        let quant = &hf_ref.filename;

        // If it already looks like a filename (has extension), use as-is.
        if quant.contains('.') {
            return Ok(quant.clone());
        }

        let url = hf_ref.api_files_url();
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| PowerError::Server(format!("HF API request failed: {e}")))?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED
            || resp.status() == reqwest::StatusCode::FORBIDDEN
        {
            return Err(PowerError::Server(format!(
                "HF API returned {}: access denied — set HF_TOKEN or pass 'token' in the request body",
                resp.status()
            )));
        }

        if !resp.status().is_success() {
            return Err(PowerError::Server(format!(
                "HF API returned {}: {}",
                resp.status(),
                url
            )));
        }

        let files: Vec<serde_json::Value> = resp
            .json()
            .await
            .map_err(|e| PowerError::Server(format!("HF API response parse failed: {e}")))?;

        // Find a GGUF file whose name contains the quantization tag (case-insensitive).
        let quant_upper = quant.to_uppercase();
        let matched = files
            .iter()
            .filter_map(|f| f["path"].as_str())
            .filter(|p| p.ends_with(".gguf"))
            .find(|p| p.to_uppercase().contains(&quant_upper))
            .map(|s| s.to_string());

        matched.ok_or_else(|| {
            PowerError::Server(format!(
                "no GGUF file matching quantization '{}' found in repo '{}'",
                quant, hf_ref.repo
            ))
        })
    }

    /// Download a model from HuggingFace Hub, streaming progress via `tx`.
    ///
    /// Supports:
    /// - Resume: if a partial file exists from a previous interrupted download,
    ///   sends `Range: bytes=<offset>-` to continue from where it left off.
    /// - Auth: `token` is used as `Authorization: Bearer <token>`. If `None`,
    ///   falls back to the `HF_TOKEN` environment variable.
    ///
    /// Returns a `ModelManifest` on success. The caller is responsible for
    /// registering the manifest with the model registry.
    pub async fn pull(
        name: &str,
        token: Option<&str>,
        tx: tokio::sync::mpsc::Sender<PullProgress>,
    ) -> Result<ModelManifest> {
        let (client, _effective_token) = build_client(token)?;

        let mut hf_ref = HfRef::parse(name)?;

        // Resolve quantization tag → actual filename if needed.
        if !hf_ref.filename.contains('.') {
            hf_ref.filename = resolve_filename(&client, &hf_ref).await?;
        }

        let url = hf_ref.download_url();
        tracing::info!(url = %url, "Pulling model from HuggingFace Hub");

        // Stable partial file path — same across restarts for the same URL.
        let blobs_dir = dirs::blobs_dir();
        std::fs::create_dir_all(&blobs_dir).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "failed to create blobs dir: {e}"
            )))
        })?;
        let tmp_path = blobs_dir.join(hf_ref.partial_filename());

        // Check for an existing partial file to resume from.
        let resume_offset = if tmp_path.exists() {
            std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        // HEAD request to get total content-length (always without Range).
        let head: reqwest::Response = client
            .head(&url)
            .send()
            .await
            .map_err(|e| PowerError::Server(format!("HEAD request failed: {e}")))?;

        if head.status() == reqwest::StatusCode::UNAUTHORIZED
            || head.status() == reqwest::StatusCode::FORBIDDEN
        {
            return Err(PowerError::Server(format!(
                "access denied ({}): set HF_TOKEN or pass 'token' in the request body",
                head.status()
            )));
        }

        let total = head
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v: &reqwest::header::HeaderValue| v.to_str().ok())
            .and_then(|s: &str| s.parse::<u64>().ok())
            .unwrap_or(0);

        // If partial file is already complete, skip download.
        if resume_offset > 0 && total > 0 && resume_offset >= total {
            tracing::info!(path = %tmp_path.display(), "Partial file is complete, skipping download");
        } else {
            // Build GET request, adding Range header if resuming.
            let mut req = client.get(&url);
            if resume_offset > 0 {
                req = req.header(reqwest::header::RANGE, format!("bytes={resume_offset}-"));
                tracing::info!(offset = resume_offset, total, "Resuming download");
                let _ = tx
                    .send(PullProgress::Resuming {
                        offset: resume_offset,
                        total,
                    })
                    .await;
            }

            let resp: reqwest::Response = req
                .send()
                .await
                .map_err(|e| PowerError::Server(format!("GET request failed: {e}")))?;

            // 206 Partial Content = server supports resume; 200 = server ignored Range.
            let server_supports_resume = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
            if !resp.status().is_success() {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Err(PowerError::Server(format!(
                    "download failed with HTTP {}: {}",
                    resp.status(),
                    url
                )));
            }

            // If server returned 200 instead of 206, it doesn't support Range —
            // truncate the partial file and start over.
            let (mut tmp_file, mut completed) = if server_supports_resume && resume_offset > 0 {
                let f = tokio::fs::OpenOptions::new()
                    .append(true)
                    .open(&tmp_path)
                    .await
                    .map_err(|e| {
                        PowerError::Io(std::io::Error::other(format!(
                            "failed to open partial file for append: {e}"
                        )))
                    })?;
                (f, resume_offset)
            } else {
                let f = tokio::fs::File::create(&tmp_path).await.map_err(|e| {
                    PowerError::Io(std::io::Error::other(format!(
                        "failed to create temp file: {e}"
                    )))
                })?;
                (f, 0u64)
            };

            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk =
                    chunk.map_err(|e| PowerError::Server(format!("download error: {e}")))?;
                tmp_file.write_all(&chunk).await.map_err(|e| {
                    PowerError::Io(std::io::Error::other(format!("write error: {e}")))
                })?;
                completed += chunk.len() as u64;
                let _ = tx
                    .send(PullProgress::Downloading { completed, total })
                    .await;
            }

            tmp_file
                .flush()
                .await
                .map_err(|e| PowerError::Io(std::io::Error::other(format!("flush error: {e}"))))?;
        }

        // Verify and store in content-addressed blob store.
        let _ = tx.send(PullProgress::Verifying).await;
        let (blob_path, sha256) = storage::store_blob_from_temp(&tmp_path).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp_path);
        })?;

        let size = std::fs::metadata(&blob_path).map(|m| m.len()).unwrap_or(0);

        let manifest = ModelManifest {
            name: name.to_string(),
            format: ModelFormat::Gguf,
            size,
            sha256,
            parameters: None,
            created_at: chrono::Utc::now(),
            path: blob_path,
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };

        let _ = tx.send(PullProgress::Done).await;
        Ok(manifest)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serial_test::serial;

        #[test]
        fn test_parse_repo_filename() {
            let r = HfRef::parse("bartowski/Llama-3.2-3B-Instruct-GGUF/model-Q4_K_M.gguf").unwrap();
            assert_eq!(r.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
            assert_eq!(r.filename, "model-Q4_K_M.gguf");
        }

        #[test]
        fn test_parse_repo_quantization() {
            let r = HfRef::parse("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M").unwrap();
            assert_eq!(r.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
            assert_eq!(r.filename, "Q4_K_M");
        }

        #[test]
        fn test_parse_invalid() {
            assert!(HfRef::parse("no-slash-here").is_err());
            assert!(HfRef::parse("").is_err());
        }

        #[test]
        fn test_download_url() {
            let r = HfRef {
                repo: "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
                filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
            };
            assert_eq!(
                r.download_url(),
                "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            );
        }

        #[test]
        fn test_api_files_url() {
            let r = HfRef {
                repo: "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
                filename: "Q4_K_M".to_string(),
            };
            assert_eq!(
                r.api_files_url(),
                "https://huggingface.co/api/models/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main"
            );
        }

        #[test]
        fn test_filename_with_extension_skips_resolve() {
            let r = HfRef::parse("owner/repo/file.gguf").unwrap();
            assert_eq!(r.filename, "file.gguf");
            assert!(r.filename.contains('.'));
        }

        #[test]
        fn test_partial_filename_is_deterministic() {
            let r = HfRef {
                repo: "owner/repo".to_string(),
                filename: "model.gguf".to_string(),
            };
            assert_eq!(r.partial_filename(), r.partial_filename());
            assert!(r.partial_filename().starts_with("partial-"));
        }

        #[test]
        fn test_partial_filename_differs_for_different_urls() {
            let r1 = HfRef {
                repo: "owner/repo-a".to_string(),
                filename: "model.gguf".to_string(),
            };
            let r2 = HfRef {
                repo: "owner/repo-b".to_string(),
                filename: "model.gguf".to_string(),
            };
            assert_ne!(r1.partial_filename(), r2.partial_filename());
        }

        #[test]
        fn test_resolve_token_explicit() {
            // Explicit token takes priority over env var.
            std::env::set_var("HF_TOKEN", "env-token");
            let tok = resolve_token(Some("explicit-token"));
            assert_eq!(tok, Some("explicit-token".to_string()));
            std::env::remove_var("HF_TOKEN");
        }

        #[test]
        fn test_resolve_token_from_env() {
            std::env::set_var("HF_TOKEN", "my-env-token");
            let tok = resolve_token(None);
            assert_eq!(tok, Some("my-env-token".to_string()));
            std::env::remove_var("HF_TOKEN");
        }

        #[test]
        fn test_resolve_token_none() {
            std::env::remove_var("HF_TOKEN");
            let tok = resolve_token(None);
            assert_eq!(tok, None);
        }

        #[test]
        fn test_resolve_token_empty_env_ignored() {
            std::env::set_var("HF_TOKEN", "");
            let tok = resolve_token(None);
            assert_eq!(tok, None);
            std::env::remove_var("HF_TOKEN");
        }

        #[test]
        #[serial]
        fn test_resume_offset_from_existing_partial() {
            let dir = tempfile::tempdir().unwrap();
            std::env::set_var("A3S_POWER_HOME", dir.path());

            let blobs_dir = dirs::blobs_dir();
            std::fs::create_dir_all(&blobs_dir).unwrap();

            let r = HfRef {
                repo: "owner/repo".to_string(),
                filename: "model.gguf".to_string(),
            };
            let partial_path = blobs_dir.join(r.partial_filename());
            // Write 1024 bytes of fake partial data.
            std::fs::write(&partial_path, vec![0u8; 1024]).unwrap();

            let offset = std::fs::metadata(&partial_path)
                .map(|m| m.len())
                .unwrap_or(0);
            assert_eq!(offset, 1024);

            std::env::remove_var("A3S_POWER_HOME");
        }
    }
}

// Re-export for non-hf builds so callers can always reference the module.
#[cfg(not(feature = "hf"))]
pub mod hf {
    /// Stub: HuggingFace pull is not available without the `hf` feature.
    pub fn not_available() -> &'static str {
        "compile with --features hf to enable HuggingFace model pull"
    }
}
