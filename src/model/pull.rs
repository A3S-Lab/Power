/// Model hub pull support (default: ModelScope).
///
/// Parses model references of the form:
///   `<owner>/<repo>:<quantization>`  e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M`
///   `<owner>/<repo>/<filename.gguf>` e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF/model-Q4_K_M.gguf`
///
/// Features:
/// - Streams download progress via a channel (SSE-friendly)
/// - Resume interrupted downloads via HTTP Range requests
/// - Hub token support for private/gated models (`MODELSCOPE_TOKEN`/`HF_TOKEN` env or request field)
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

    /// Supported model hubs for remote pull.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum HubSource {
        ModelScope,
        HuggingFace,
    }

    impl HubSource {
        /// Select hub source from env var, defaulting to ModelScope.
        fn from_env() -> Self {
            match std::env::var("A3S_POWER_MODEL_SOURCE")
                .ok()
                .unwrap_or_else(|| "modelscope".to_string())
                .to_lowercase()
                .as_str()
            {
                "hf" | "huggingface" => Self::HuggingFace,
                _ => Self::ModelScope,
            }
        }

        fn base_url(self) -> &'static str {
            match self {
                Self::ModelScope => "https://modelscope.cn",
                Self::HuggingFace => "https://huggingface.co",
            }
        }

        fn resolve_revision(self) -> &'static str {
            match self {
                Self::ModelScope => "master",
                Self::HuggingFace => "main",
            }
        }
    }

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

    async fn send_progress(tx: &tokio::sync::mpsc::Sender<PullProgress>, progress: PullProgress) {
        if tx.send(progress).await.is_err() {
            tracing::debug!("Model pull progress receiver closed");
        }
    }

    async fn cleanup_temp_download(path: &std::path::Path, reason: &str) {
        match tokio::fs::remove_file(path).await {
            Ok(()) => tracing::debug!(
                path = %path.display(),
                reason,
                "Removed temporary download"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => tracing::debug!(
                path = %path.display(),
                reason,
                "Temporary download was already removed"
            ),
            Err(e) => tracing::warn!(
                path = %path.display(),
                reason,
                error = %e,
                "Failed to remove temporary download"
            ),
        }
    }

    fn file_size(path: &std::path::Path, purpose: &str) -> Result<u64> {
        std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .map_err(|e| {
                PowerError::Io(std::io::Error::other(format!(
                    "failed to inspect {purpose} {}: {e}",
                    path.display()
                )))
            })
    }

    fn existing_file_size(path: &std::path::Path, purpose: &str) -> Result<u64> {
        match std::fs::metadata(path) {
            Ok(metadata) => Ok(metadata.len()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(PowerError::Io(std::io::Error::other(format!(
                "failed to inspect {purpose} {}: {e}",
                path.display()
            )))),
        }
    }

    fn content_length_or_unknown(
        headers: &reqwest::header::HeaderMap,
        purpose: &str,
    ) -> Result<u64> {
        let Some(value) = headers.get(reqwest::header::CONTENT_LENGTH) else {
            return Ok(0);
        };

        let value = value.to_str().map_err(|e| {
            PowerError::Server(format!(
                "{purpose} returned invalid Content-Length header: {e}"
            ))
        })?;

        value.parse::<u64>().map_err(|e| {
            PowerError::Server(format!(
                "{purpose} returned invalid Content-Length value '{value}': {e}"
            ))
        })
    }

    /// Parsed model reference.
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
        fn download_url(&self, source: HubSource) -> String {
            match source {
                HubSource::ModelScope => format!(
                    "{}/models/{}/resolve/{}/{}",
                    source.base_url(),
                    self.repo,
                    source.resolve_revision(),
                    self.filename
                ),
                HubSource::HuggingFace => format!(
                    "{}/{}/resolve/{}/{}",
                    source.base_url(),
                    self.repo,
                    source.resolve_revision(),
                    self.filename
                ),
            }
        }

        /// Build the HF API URL to list repo files (used to resolve quantization → filename).
        fn api_files_url(&self, source: HubSource) -> String {
            match source {
                HubSource::ModelScope => format!(
                    "{}/api/v1/models/{}/repo/files?Revision={}",
                    source.base_url(),
                    self.repo,
                    source.resolve_revision()
                ),
                HubSource::HuggingFace => format!(
                    "{}/api/models/{}/tree/{}",
                    source.base_url(),
                    self.repo,
                    source.resolve_revision()
                ),
            }
        }

        /// Stable partial-download filename derived from the download URL.
        ///
        /// Using a deterministic name (SHA-256 of the URL) means a resumed pull
        /// after a crash or restart will find the same partial file.
        pub fn partial_filename(&self) -> String {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            let source = HubSource::from_env();
            h.update(self.download_url(source).as_bytes());
            let digest = h.finalize();
            format!("partial-{:x}", digest)
        }
    }

    /// Resolve effective hub token.
    /// Priority: explicit arg -> MODELSCOPE_TOKEN -> A3S_POWER_HUB_TOKEN -> HF_TOKEN.
    fn resolve_token(token: Option<&str>) -> Option<String> {
        token
            .map(|t| t.to_string())
            .or_else(|| {
                std::env::var("MODELSCOPE_TOKEN")
                    .ok()
                    .filter(|t| !t.is_empty())
            })
            .or_else(|| {
                std::env::var("A3S_POWER_HUB_TOKEN")
                    .ok()
                    .filter(|t| !t.is_empty())
            })
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
            .user_agent(concat!("a3s-power/", env!("CARGO_PKG_VERSION")))
            .default_headers(headers)
            .build()
            .map_err(|e| PowerError::Server(format!("failed to build HTTP client: {e}")))?;
        Ok((client, effective_token))
    }

    /// Resolve a quantization tag (e.g. `Q4_K_M`) to an actual filename by
    /// querying the HuggingFace repo file listing.
    async fn resolve_filename(
        client: &Client,
        hf_ref: &HfRef,
        source: HubSource,
    ) -> Result<String> {
        let quant = &hf_ref.filename;

        // If it already looks like a filename (has extension), use as-is.
        if quant.contains('.') {
            return Ok(quant.clone());
        }

        let url = hf_ref.api_files_url(source);
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

        let payload: serde_json::Value = resp.json().await.map_err(|e| match source {
            HubSource::ModelScope => {
                PowerError::Server(format!("ModelScope API response parse failed: {e}"))
            }
            HubSource::HuggingFace => {
                PowerError::Server(format!("HF API response parse failed: {e}"))
            }
        })?;

        let paths = extract_file_paths(source, &payload)?;
        find_matching_gguf_file(&paths, quant, &hf_ref.repo)
    }

    fn extract_file_paths(source: HubSource, payload: &serde_json::Value) -> Result<Vec<String>> {
        let files = match source {
            HubSource::ModelScope => payload
                .get("Data")
                .and_then(|data| data.get("Files"))
                .and_then(|files| files.as_array())
                .ok_or_else(|| {
                    PowerError::Server(
                        "ModelScope API response missing Data.Files array".to_string(),
                    )
                })?,
            HubSource::HuggingFace => payload.as_array().ok_or_else(|| {
                PowerError::Server("HF API response missing file list array".to_string())
            })?,
        };

        Ok(files
            .iter()
            .filter_map(|file| {
                file.get("path")
                    .and_then(|path| path.as_str())
                    .or_else(|| file.get("Path").and_then(|path| path.as_str()))
            })
            .map(str::to_string)
            .collect())
    }

    fn find_matching_gguf_file(paths: &[String], quant: &str, repo: &str) -> Result<String> {
        let quant_upper = quant.to_uppercase();
        paths
            .iter()
            .filter(|path| path.ends_with(".gguf"))
            .find(|path| path.to_uppercase().contains(&quant_upper))
            .cloned()
            .ok_or_else(|| {
                PowerError::Server(format!(
                    "no GGUF file matching quantization '{}' found in repo '{}'",
                    quant, repo
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
        let source = HubSource::from_env();

        let mut hf_ref = HfRef::parse(name)?;

        // Resolve quantization tag → actual filename if needed.
        if !hf_ref.filename.contains('.') {
            hf_ref.filename = resolve_filename(&client, &hf_ref, source).await?;
        }

        let url = hf_ref.download_url(source);
        tracing::info!(url = %url, source = ?source, "Pulling model from remote hub");

        // Stable partial file path — same across restarts for the same URL.
        let blobs_dir = dirs::blobs_dir();
        std::fs::create_dir_all(&blobs_dir).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "failed to create blobs dir: {e}"
            )))
        })?;
        let tmp_path = blobs_dir.join(hf_ref.partial_filename());

        // Check for an existing partial file to resume from.
        let resume_offset = existing_file_size(&tmp_path, "partial download")?;

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

        let total = content_length_or_unknown(head.headers(), "model pull HEAD request")?;

        // If partial file is already complete, skip download.
        if resume_offset > 0 && total > 0 && resume_offset >= total {
            tracing::info!(path = %tmp_path.display(), "Partial file is complete, skipping download");
        } else {
            // Build GET request, adding Range header if resuming.
            let mut req = client.get(&url);
            if resume_offset > 0 {
                req = req.header(reqwest::header::RANGE, format!("bytes={resume_offset}-"));
                tracing::info!(offset = resume_offset, total, "Resuming download");
                send_progress(
                    &tx,
                    PullProgress::Resuming {
                        offset: resume_offset,
                        total,
                    },
                )
                .await;
            }

            let resp: reqwest::Response = req
                .send()
                .await
                .map_err(|e| PowerError::Server(format!("GET request failed: {e}")))?;

            // 206 Partial Content = server supports resume; 200 = server ignored Range.
            let server_supports_resume = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
            if !resp.status().is_success() {
                cleanup_temp_download(&tmp_path, "download failed").await;
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
                send_progress(&tx, PullProgress::Downloading { completed, total }).await;
            }

            tmp_file
                .flush()
                .await
                .map_err(|e| PowerError::Io(std::io::Error::other(format!("flush error: {e}"))))?;
        }

        // Verify and store in content-addressed blob store.
        send_progress(&tx, PullProgress::Verifying).await;
        let (blob_path, sha256) = match storage::store_blob_from_temp(&tmp_path) {
            Ok(stored) => stored,
            Err(e) => {
                cleanup_temp_download(&tmp_path, "blob store failed").await;
                return Err(e);
            }
        };

        let size = file_size(&blob_path, "stored model blob")?;

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

        send_progress(&tx, PullProgress::Done).await;
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
                r.download_url(HubSource::ModelScope),
                "https://modelscope.cn/models/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/master/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            );
            assert_eq!(
                r.download_url(HubSource::HuggingFace),
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
                r.api_files_url(HubSource::ModelScope),
                "https://modelscope.cn/api/v1/models/bartowski/Llama-3.2-3B-Instruct-GGUF/repo/files?Revision=master"
            );
            assert_eq!(
                r.api_files_url(HubSource::HuggingFace),
                "https://huggingface.co/api/models/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main"
            );
        }

        #[test]
        fn test_extract_modelscope_file_paths() {
            let payload = serde_json::json!({
                "Data": {
                    "Files": [
                        { "Path": "model-Q4_K_M.gguf" },
                        { "Path": "README.md" }
                    ]
                }
            });

            let paths = extract_file_paths(HubSource::ModelScope, &payload).unwrap();

            assert_eq!(paths, vec!["model-Q4_K_M.gguf", "README.md"]);
        }

        #[test]
        fn test_extract_modelscope_file_paths_reports_missing_files_array() {
            let payload = serde_json::json!({
                "Data": {
                    "Items": []
                }
            });

            let err = extract_file_paths(HubSource::ModelScope, &payload).unwrap_err();

            assert!(err.to_string().contains("Data.Files array"), "error: {err}");
        }

        #[test]
        fn test_extract_huggingface_file_paths_reports_non_array_payload() {
            let payload = serde_json::json!({
                "error": "temporarily unavailable"
            });

            let err = extract_file_paths(HubSource::HuggingFace, &payload).unwrap_err();

            assert!(err.to_string().contains("file list array"), "error: {err}");
        }

        #[test]
        fn test_find_matching_gguf_file_matches_quantization_case_insensitively() {
            let paths = vec![
                "README.md".to_string(),
                "nested/model-q4_k_m.gguf".to_string(),
                "nested/model-Q8_0.gguf".to_string(),
            ];

            let matched = find_matching_gguf_file(&paths, "Q4_K_M", "owner/repo").unwrap();

            assert_eq!(matched, "nested/model-q4_k_m.gguf");
        }

        #[test]
        fn test_find_matching_gguf_file_reports_missing_match() {
            let paths = vec!["README.md".to_string(), "model-Q8_0.gguf".to_string()];

            let err = find_matching_gguf_file(&paths, "Q4_K_M", "owner/repo").unwrap_err();

            assert!(
                err.to_string()
                    .contains("no GGUF file matching quantization"),
                "error: {err}"
            );
        }

        #[test]
        fn test_file_size_reports_metadata_errors() {
            let dir = tempfile::tempdir().unwrap();
            let missing = dir.path().join("missing.gguf");

            let err = file_size(&missing, "stored model blob").unwrap_err();

            assert!(
                err.to_string()
                    .contains("failed to inspect stored model blob"),
                "error: {err}"
            );
            assert!(err.to_string().contains("missing.gguf"), "error: {err}");
        }

        #[test]
        fn test_existing_file_size_missing_returns_zero() {
            let dir = tempfile::tempdir().unwrap();
            let missing = dir.path().join("partial-missing");

            let size = existing_file_size(&missing, "partial download").unwrap();

            assert_eq!(size, 0);
        }

        #[test]
        fn test_existing_file_size_existing_returns_size() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("partial");
            std::fs::write(&path, b"partial").unwrap();

            let size = existing_file_size(&path, "partial download").unwrap();

            assert_eq!(size, 7);
        }

        #[test]
        fn test_content_length_missing_is_unknown() {
            let headers = reqwest::header::HeaderMap::new();

            let total = content_length_or_unknown(&headers, "test HEAD").unwrap();

            assert_eq!(total, 0);
        }

        #[test]
        fn test_content_length_parses_u64() {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(reqwest::header::CONTENT_LENGTH, "4096".parse().unwrap());

            let total = content_length_or_unknown(&headers, "test HEAD").unwrap();

            assert_eq!(total, 4096);
        }

        #[test]
        fn test_content_length_reports_invalid_value() {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::CONTENT_LENGTH,
                "definitely-not-a-size".parse().unwrap(),
            );

            let err = content_length_or_unknown(&headers, "test HEAD").unwrap_err();

            assert!(
                err.to_string().contains("invalid Content-Length value"),
                "error: {err}"
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

            let offset = existing_file_size(&partial_path, "partial download").unwrap();
            assert_eq!(offset, 1024);

            std::env::remove_var("A3S_POWER_HOME");
        }
    }
}

// Re-export for non-hf builds so callers can always reference the module.
#[cfg(not(feature = "hf"))]
pub mod hf {
    /// Stub: remote model pull is not available without the `hf` feature.
    pub fn not_available() -> &'static str {
        "compile with --features hf to enable model hub pull"
    }
}
