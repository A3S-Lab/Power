use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::config::TeePolicyMode;
use crate::model::manifest::{ModelFormat, ModelManifest};
use crate::server::state::AppState;
use crate::tee::attestation::{
    AttestationClaimsV2, ModelDigestClaim, ModelDigestKind, RuntimePolicyClaim,
};

/// Query parameters for `GET /v1/attestation`.
#[derive(Debug, Deserialize)]
pub struct AttestationQuery {
    /// Optional client-supplied hex-encoded nonce to bind into the attestation report.
    pub nonce: Option<String>,
    /// Optional model name to bind through AttestationClaimsV2.
    /// Ties the attestation to the specific model being served.
    pub model: Option<String>,
}

/// Decode a hex string to bytes, returning an error response on invalid input.
fn decode_hex_nonce(hex: &str) -> Result<Vec<u8>, (StatusCode, axum::Json<serde_json::Value>)> {
    if !hex.len().is_multiple_of(2) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "nonce must be an even-length hex string",
                    "type": "invalid_request_error",
                    "code": "invalid_nonce"
                }
            })),
        ));
    }
    hex::decode(hex).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("nonce contains invalid hex: {e}"),
                    "type": "invalid_request_error",
                    "code": "invalid_nonce"
                }
            })),
        )
    })
}

fn error_json(
    status: StatusCode,
    message: impl Into<String>,
    code: &'static str,
) -> (StatusCode, axum::Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "code": code
            }
        })),
    )
}

fn decode_sha256_hex(
    hash: &str,
    code: &'static str,
) -> Result<Vec<u8>, (StatusCode, axum::Json<serde_json::Value>)> {
    let hex = hash.strip_prefix("sha256:").unwrap_or(hash);
    if hex.len() != 64 {
        return Err(error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("model hash must be 64 hex characters, got {}", hex.len()),
            code,
        ));
    }
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            "model hash contains non-hex characters",
            code,
        ));
    }
    hex::decode(hex).map_err(|e| {
        error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to decode model hash: {e}"),
            code,
        )
    })
}

fn expected_model_hash(
    state: &AppState,
    model_name: &str,
    manifest: &ModelManifest,
) -> Result<(Vec<u8>, bool), (StatusCode, axum::Json<serde_json::Value>)> {
    if manifest.format == ModelFormat::Remote {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            format!("model '{model_name}' is remote and has no local weights to attest"),
            "model_not_attestable",
        ));
    }

    let configured_hash = state
        .config
        .model_hashes
        .get(model_name)
        .map(String::as_str);
    let hash_str = configured_hash.unwrap_or(manifest.sha256.as_str());

    if hash_str.trim().is_empty() {
        return Err(error_json(
            StatusCode::CONFLICT,
            format!("model '{model_name}' has no pinned SHA-256 hash"),
            "model_hash_missing",
        ));
    }

    decode_sha256_hex(hash_str, "invalid_model_hash")
        .map(|digest| (digest, configured_hash.is_some()))
}

async fn model_key_for_attestation(
    state: &AppState,
    model_name: &str,
) -> Result<Option<[u8; 32]>, (StatusCode, axum::Json<serde_json::Value>)> {
    if let Some(ref key_provider) = state.key_provider {
        return key_provider.get_key().await.map(Some).map_err(|e| {
            error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("key provider failed for model '{model_name}': {e}"),
                "model_key_failed",
            )
        });
    }

    let Some(key_source) = state.config.model_key_source.as_ref() else {
        return Ok(None);
    };

    crate::tee::encrypted_model::load_key(key_source)
        .map(Some)
        .map_err(|e| {
            error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to load model key for '{model_name}': {e}"),
                "model_key_failed",
            )
        })
}

async fn runtime_model_digest(
    state: &AppState,
    model_name: &str,
    manifest: &ModelManifest,
    configured_pin: bool,
) -> Result<(ModelDigestClaim, Vec<u8>), (StatusCode, axum::Json<serde_json::Value>)> {
    if !manifest.path.exists() {
        return Err(error_json(
            StatusCode::CONFLICT,
            format!(
                "model '{model_name}' file is missing: {}",
                manifest.path.display()
            ),
            "model_file_missing",
        ));
    }
    if manifest.path.is_dir() {
        let actual =
            crate::model::storage::compute_sha256_directory(&manifest.path).map_err(|e| {
                error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to hash current model directory for '{model_name}': {e}"),
                    "model_hash_failed",
                )
            })?;
        let digest = decode_sha256_hex(&actual, "invalid_runtime_model_hash")?;
        return Ok((
            ModelDigestClaim {
                name: model_name.to_string(),
                kind: ModelDigestKind::DirectoryManifestSha256,
                digest: digest.clone(),
                plaintext_digest: None,
                ciphertext_digest: None,
            },
            digest,
        ));
    }
    if !manifest.path.is_file() {
        return Err(error_json(
            StatusCode::CONFLICT,
            format!("model '{model_name}' uses an unsupported non-file artifact"),
            "model_hash_unsupported",
        ));
    }
    if manifest
        .path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("enc"))
    {
        let ciphertext =
            crate::model::storage::compute_sha256_file(&manifest.path).map_err(|e| {
                error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to hash encrypted model artifact for '{model_name}': {e}"),
                    "model_hash_failed",
                )
            })?;
        let ciphertext_digest = decode_sha256_hex(&ciphertext, "invalid_runtime_model_hash")?;
        let key = model_key_for_attestation(state, model_name).await?;
        let plaintext_digest = match key {
            Some(key) => {
                let plaintext =
                    crate::tee::encrypted_model::compute_plaintext_sha256(&manifest.path, &key)
                        .map_err(|e| {
                            error_json(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("failed to hash encrypted model plaintext for '{model_name}': {e}"),
                        "model_hash_failed",
                    )
                        })?;
                Some(decode_sha256_hex(&plaintext, "invalid_runtime_model_hash")?)
            }
            None if configured_pin => {
                return Err(error_json(
                    StatusCode::CONFLICT,
                    format!(
                        "model '{model_name}' is encrypted and has a configured model hash, but no model key is available to verify plaintext"
                    ),
                    "model_key_missing",
                ));
            }
            None => None,
        };

        if configured_pin {
            let plaintext_digest = plaintext_digest.ok_or_else(|| {
                error_json(
                    StatusCode::CONFLICT,
                    format!("model '{model_name}' plaintext digest is unavailable"),
                    "model_hash_failed",
                )
            })?;
            return Ok((
                ModelDigestClaim {
                    name: model_name.to_string(),
                    kind: ModelDigestKind::PlaintextWeightsSha256,
                    digest: plaintext_digest.clone(),
                    plaintext_digest: Some(plaintext_digest.clone()),
                    ciphertext_digest: Some(ciphertext_digest),
                },
                plaintext_digest,
            ));
        }

        return Ok((
            ModelDigestClaim {
                name: model_name.to_string(),
                kind: ModelDigestKind::CiphertextArtifactSha256,
                digest: ciphertext_digest.clone(),
                plaintext_digest,
                ciphertext_digest: Some(ciphertext_digest.clone()),
            },
            ciphertext_digest,
        ));
    }

    let actual = crate::model::storage::compute_sha256_path(&manifest.path).map_err(|e| {
        error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to hash current model artifact for '{model_name}': {e}"),
            "model_hash_failed",
        )
    })?;

    let digest = decode_sha256_hex(&actual, "invalid_runtime_model_hash")?;
    Ok((
        ModelDigestClaim {
            name: model_name.to_string(),
            kind: ModelDigestKind::PlaintextWeightsSha256,
            digest: digest.clone(),
            plaintext_digest: None,
            ciphertext_digest: None,
        },
        digest,
    ))
}

async fn resolve_model_claims(
    state: &AppState,
    model_name: &str,
) -> Result<
    (ModelDigestClaim, Option<RuntimePolicyClaim>),
    (StatusCode, axum::Json<serde_json::Value>),
> {
    let manifest = state.registry.get(model_name).map_err(|_| {
        error_json(
            StatusCode::NOT_FOUND,
            format!("model '{model_name}' not found"),
            "model_not_found",
        )
    })?;

    let (expected, configured_pin) = expected_model_hash(state, model_name, &manifest)?;
    let (model, actual) =
        runtime_model_digest(state, model_name, &manifest, configured_pin).await?;
    if expected != actual {
        return Err(error_json(
            StatusCode::CONFLICT,
            format!("model '{model_name}' current SHA-256 does not match pinned hash"),
            "model_hash_mismatch",
        ));
    }

    let runtime = crate::api::prompt_policy::runtime_policy_claim_with_gpu_config(
        &manifest,
        Some(&state.config.gpu),
    )
    .map_err(|e| {
        error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to build runtime policy claim: {e}"),
            "runtime_policy_failed",
        )
    })?;

    Ok((model, runtime))
}

/// GET /v1/attestation — generate and return a TEE attestation report.
///
/// Optional `?nonce=<hex>` query parameter binds a client nonce into the report
/// to prevent replay attacks.
///
/// Optional `?model=<name>` query parameter emits AttestationClaimsV2 and binds
/// `sha256(canonical_claims_v2)` into CPU TEE report_data.
///
/// Returns 503 if TEE mode is not enabled or no TEE provider is configured.
pub async fn handler(
    State(state): State<AppState>,
    Query(params): Query<AttestationQuery>,
) -> impl IntoResponse {
    let provider = match &state.tee_provider {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "TEE mode is not enabled",
                        "type": "tee_unavailable",
                        "code": "tee_not_enabled"
                    }
                })),
            )
                .into_response();
        }
    };

    // Decode nonce if provided
    let nonce_bytes = match params.nonce.as_deref() {
        Some(hex) => match decode_hex_nonce(hex) {
            Ok(b) => Some(b),
            Err(resp) => return resp.into_response(),
        },
        None => None,
    };

    // Look up model and runtime policy claims if model name provided.
    let model_claims: Option<(ModelDigestClaim, Option<RuntimePolicyClaim>)> =
        if let Some(ref model_name) = params.model {
            match resolve_model_claims(&state, model_name).await {
                Ok(claims) => Some(claims),
                Err(resp) => return resp.into_response(),
            }
        } else {
            None
        };

    let (model_digest, runtime_claim) = match model_claims {
        Some((model, runtime)) => (Some(model), runtime),
        None => (None, None),
    };

    let gpu_claim = if matches!(state.config.tee_policy_mode, TeePolicyMode::GpuConfidential) {
        let Some(nonce) = nonce_bytes.as_deref().filter(|bytes| !bytes.is_empty()) else {
            return error_json(
                StatusCode::BAD_REQUEST,
                "gpu-confidential attestation requires a non-empty nonce so CPU TEE and NVIDIA GPU evidence share the same freshness value",
                "gpu_confidential_nonce_required",
            )
            .into_response();
        };
        if nonce.len() != 32 {
            return error_json(
                StatusCode::BAD_REQUEST,
                format!(
                    "gpu-confidential attestation requires a 32-byte nonce (64 hex characters), got {} bytes",
                    nonce.len()
                ),
                "gpu_confidential_nonce_length",
            )
            .into_response();
        }

        let Some(gpu_provider) = &state.gpu_evidence_provider else {
            return error_json(
                StatusCode::SERVICE_UNAVAILABLE,
                "NVIDIA GPU confidential-computing evidence provider is not configured",
                "gpu_attestation_not_configured",
            )
            .into_response();
        };

        match gpu_provider.evidence_claim_for_nonce(Some(nonce)).await {
            Ok(claim) => Some(claim),
            Err(e) => {
                return error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to load NVIDIA GPU confidential-computing evidence: {e}"),
                    "gpu_attestation_failed",
                )
                .into_response();
            }
        }
    } else {
        None
    };

    let report_result = if model_digest.is_some() || gpu_claim.is_some() || runtime_claim.is_some()
    {
        let mut claims =
            AttestationClaimsV2::new(provider.tee_type()).with_nonce(nonce_bytes.as_deref());
        if let Some(model) = model_digest {
            claims = claims.with_model(model);
        }
        if let Some(gpu) = gpu_claim {
            claims = claims.with_gpu(gpu);
        }
        if let Some(runtime) = runtime_claim {
            claims = claims.with_runtime(runtime);
        }
        provider.attestation_report_with_claims(claims).await
    } else {
        provider.attestation_report(nonce_bytes.as_deref()).await
    };

    match report_result {
        Ok(report) => {
            state.metrics.increment_tee_attestation();
            (StatusCode::OK, Json(report)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Failed to generate attestation report: {e}"),
                    "type": "tee_error",
                    "code": "attestation_failed"
                }
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendRegistry;
    use crate::config::{PowerConfig, TeePolicyMode};
    use crate::model::manifest::ManifestMessage;
    use crate::model::registry::ModelRegistry;
    use crate::model::storage;
    use crate::tee::attestation::{
        build_claims_report_data, AttestationReport, DefaultTeeProvider, GpuEvidenceClaim, TeeType,
    };
    use crate::tee::gpu::StaticGpuEvidenceProvider;
    use axum::extract::State;
    use std::{collections::HashMap, path::Path, sync::Arc};

    fn no_nonce() -> Query<AttestationQuery> {
        Query(AttestationQuery {
            nonce: None,
            model: None,
        })
    }

    fn with_nonce(hex: &str) -> Query<AttestationQuery> {
        Query(AttestationQuery {
            nonce: Some(hex.to_string()),
            model: None,
        })
    }

    fn nonce32() -> Vec<u8> {
        vec![0x11; 32]
    }

    fn nonce32_hex() -> String {
        hex::encode(nonce32())
    }

    fn with_model(model: &str) -> Query<AttestationQuery> {
        Query(AttestationQuery {
            nonce: None,
            model: Some(model.to_string()),
        })
    }

    fn test_state_no_tee() -> AppState {
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
    }

    fn test_state_simulated() -> AppState {
        let provider = DefaultTeeProvider::with_type(TeeType::Simulated);
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                tee_mode: true,
                ..Default::default()
            }),
        )
        .with_tee_provider(Arc::new(provider))
    }

    fn local_manifest(name: &str, path: &Path, sha256: &str) -> ModelManifest {
        ModelManifest {
            name: name.to_string(),
            format: ModelFormat::Gguf,
            size: std::fs::metadata(path).map(|m| m.len()).unwrap_or_default(),
            sha256: sha256.to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: path.to_path_buf(),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: Vec::new(),
            family: None,
            families: None,
        }
    }

    fn test_state_with_manifest(manifest: ModelManifest, config: PowerConfig) -> AppState {
        let registry = Arc::new(ModelRegistry::new());
        registry.register_transient(manifest).unwrap();
        AppState::new(registry, Arc::new(BackendRegistry::new()), Arc::new(config))
            .with_tee_provider(Arc::new(DefaultTeeProvider::with_type(TeeType::Simulated)))
    }

    fn test_state_gpu_confidential(gpu_provider: bool) -> AppState {
        let mut state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                tee_mode: true,
                tee_policy_mode: TeePolicyMode::GpuConfidential,
                ..Default::default()
            }),
        )
        .with_tee_provider(Arc::new(DefaultTeeProvider::with_type(TeeType::Simulated)));

        if gpu_provider {
            state = state.with_gpu_evidence_provider(Arc::new(StaticGpuEvidenceProvider::new(
                GpuEvidenceClaim::new("nvidia-nras", vec![0x11; 32])
                    .with_verdict_digest(vec![0x22; 32]),
            )));
        }

        state
    }

    async fn response_json(resp: axum::response::Response) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    async fn response_report(resp: axum::response::Response) -> AttestationReport {
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn test_attestation_no_tee_returns_503() {
        let state = test_state_no_tee();
        let resp = handler(State(state), no_nonce()).await.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "tee_not_enabled");
    }

    #[tokio::test]
    async fn test_attestation_with_simulated_tee() {
        let state = test_state_simulated();
        let resp = handler(State(state), no_nonce()).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = response_json(resp).await;
        assert_eq!(json["tee_type"], "simulated");
        assert!(json["report_data"].is_string());
        assert!(json["measurement"].is_string());
        assert!(json["timestamp"].is_string());
        // No nonce in response when not supplied
        assert!(json["nonce"].is_null());
        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[tokio::test]
    async fn test_attestation_report_has_correct_fields() {
        let state = test_state_simulated();
        let resp = handler(State(state), no_nonce()).await.into_response();
        let json = response_json(resp).await;

        // Simulated report has 64 bytes of 0xAA for report_data
        let report_data = json["report_data"].as_str().unwrap();
        assert_eq!(report_data.len(), 128); // 64 bytes = 128 hex chars
        assert!(report_data.chars().all(|c| c == 'a'));

        // Simulated report has 48 bytes of 0xBB for measurement
        let measurement = json["measurement"].as_str().unwrap();
        assert_eq!(measurement.len(), 96); // 48 bytes = 96 hex chars
        assert!(measurement.chars().all(|c| c == 'b'));

        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[tokio::test]
    async fn test_attestation_error_json_structure() {
        let state = test_state_no_tee();
        let resp = handler(State(state), no_nonce()).await.into_response();
        let json = response_json(resp).await;
        // Verify OpenAI-style error structure
        assert!(json["error"]["message"].is_string());
        assert!(json["error"]["type"].is_string());
        assert!(json["error"]["code"].is_string());
    }

    #[tokio::test]
    async fn test_attestation_with_nonce_binds_to_report() {
        let state = test_state_simulated();
        // nonce = [0x01, 0x02, 0x03] → hex "010203"
        let resp = handler(State(state), with_nonce("010203"))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = response_json(resp).await;
        // nonce reflected in response
        assert_eq!(json["nonce"], "010203");
        // report_data starts with nonce bytes
        let report_data = json["report_data"].as_str().unwrap();
        assert!(report_data.starts_with("010203"));
        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[tokio::test]
    async fn test_attestation_invalid_hex_nonce_returns_400() {
        let state = test_state_simulated();
        let resp = handler(State(state), with_nonce("xyz"))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "invalid_nonce");
        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[tokio::test]
    async fn test_attestation_with_model_binds_runtime_hash() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"model-v1").unwrap();
        let hash = storage::compute_sha256(b"model-v1");
        let state = test_state_with_manifest(
            local_manifest("test-model", &model_path, &hash),
            PowerConfig {
                tee_mode: true,
                ..Default::default()
            },
        );

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        let model = claims.model.as_ref().unwrap();
        assert_eq!(model.name, "test-model");
        assert_eq!(hex::encode(&model.digest), hash);
        assert_eq!(
            report.report_data,
            build_claims_report_data(claims).unwrap()
        );
    }

    #[tokio::test]
    async fn test_attestation_with_model_uses_configured_pin_over_manifest_hash() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"model-v1").unwrap();
        let hash = storage::compute_sha256(b"model-v1");
        let manifest = local_manifest(
            "test-model",
            &model_path,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        );
        let state = test_state_with_manifest(
            manifest,
            PowerConfig {
                tee_mode: true,
                model_hashes: HashMap::from([("test-model".to_string(), hash.clone())]),
                ..Default::default()
            },
        );

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        let model = claims.model.as_ref().unwrap();
        assert_eq!(hex::encode(&model.digest), hash);
        assert_eq!(
            report.report_data,
            build_claims_report_data(claims).unwrap()
        );
    }

    #[tokio::test]
    async fn test_attestation_with_model_and_nonce_uses_v2_claims() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"model-v1").unwrap();
        let hash = storage::compute_sha256(b"model-v1");
        let state = test_state_with_manifest(
            local_manifest("test-model", &model_path, &hash),
            PowerConfig {
                tee_mode: true,
                ..Default::default()
            },
        );

        let resp = handler(
            State(state),
            Query(AttestationQuery {
                nonce: Some("010203".to_string()),
                model: Some("test-model".to_string()),
            }),
        )
        .await
        .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        assert_eq!(claims.nonce, Some(vec![0x01, 0x02, 0x03]));
        assert_eq!(report.nonce, Some(vec![0x01, 0x02, 0x03]));
        assert_eq!(
            report.report_data,
            build_claims_report_data(claims).unwrap()
        );
    }

    #[tokio::test]
    async fn test_attestation_with_model_binds_applied_runtime_policy_claims() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"model-v1").unwrap();
        let hash = storage::compute_sha256(b"model-v1");
        let mut manifest = local_manifest("test-model", &model_path, &hash);
        manifest.template_override = Some("{{ .System }}\n{{ .Prompt }}".to_string());
        manifest.system_prompt = Some("You are A3S Power.".to_string());
        manifest.messages = vec![ManifestMessage {
            role: "system".to_string(),
            content: "Pinned message".to_string(),
        }];
        manifest.default_parameters = Some(HashMap::from([
            ("temperature".to_string(), serde_json::json!(0.2)),
            ("top_p".to_string(), serde_json::json!(0.9)),
            ("stop".to_string(), serde_json::json!(["</s>"])),
        ]));
        let state = test_state_with_manifest(
            manifest,
            PowerConfig {
                tee_mode: true,
                gpu: crate::config::GpuConfig {
                    gpu_layers: -1,
                    main_gpu: 0,
                    tensor_split: vec![0.5, 0.5],
                },
                ..Default::default()
            },
        );
        let expected_gpu_execution_digest =
            crate::api::prompt_policy::canonical_gpu_execution_digest(&state.config.gpu).unwrap();

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        let runtime = claims.runtime.as_ref().unwrap();
        let prompt = runtime.prompt.as_ref().unwrap();
        assert_eq!(
            prompt.chat_template_source.as_deref(),
            Some("manifest.template_override")
        );
        assert_eq!(
            prompt.chat_template_sha256.as_ref().map(hex::encode),
            Some(storage::compute_sha256(
                "{{ .System }}\n{{ .Prompt }}".as_bytes()
            ))
        );
        assert!(prompt.system_prompt_sha256.is_none());
        assert!(prompt.messages_sha256.is_none());
        assert!(runtime.decoding.as_ref().is_none());
        assert_eq!(
            runtime
                .execution
                .as_ref()
                .map(|execution| execution.gpu_sha256.clone()),
            Some(expected_gpu_execution_digest)
        );
        assert_eq!(
            report.report_data,
            build_claims_report_data(claims).unwrap()
        );
    }

    #[tokio::test]
    async fn test_gpu_confidential_attestation_binds_gpu_claim_without_model() {
        let state = test_state_gpu_confidential(true);
        let nonce = nonce32();
        let resp = handler(State(state), with_nonce(&nonce32_hex()))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        let gpu = claims.gpu.as_ref().unwrap();
        assert_eq!(gpu.provider, "nvidia-nras");
        assert_eq!(gpu.evidence_digest, vec![0x11; 32]);
        assert_eq!(gpu.verdict_digest, Some(vec![0x22; 32]));
        assert_eq!(claims.nonce, Some(nonce));
        assert_eq!(
            report.report_data,
            build_claims_report_data(claims).unwrap()
        );
    }

    #[tokio::test]
    async fn test_gpu_confidential_attestation_requires_provider() {
        let state = test_state_gpu_confidential(false);
        let resp = handler(State(state), with_nonce(&nonce32_hex()))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "gpu_attestation_not_configured");
    }

    #[tokio::test]
    async fn test_gpu_confidential_attestation_requires_nonce() {
        let state = test_state_gpu_confidential(true);
        let resp = handler(State(state), no_nonce()).await.into_response();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "gpu_confidential_nonce_required");
    }

    #[tokio::test]
    async fn test_gpu_confidential_attestation_requires_32_byte_nonce() {
        let state = test_state_gpu_confidential(true);
        let resp = handler(State(state), with_nonce("010203"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "gpu_confidential_nonce_length");
    }

    #[tokio::test]
    async fn test_attestation_with_model_rejects_tampered_runtime_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"model-v1").unwrap();
        let original_hash = storage::compute_sha256(b"model-v1");
        let state = test_state_with_manifest(
            local_manifest("test-model", &model_path, &original_hash),
            PowerConfig {
                tee_mode: true,
                ..Default::default()
            },
        );
        std::fs::write(&model_path, b"model-v2").unwrap();

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "model_hash_mismatch");
    }

    #[tokio::test]
    async fn test_attestation_with_model_rejects_missing_hash() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, b"model-v1").unwrap();
        let state = test_state_with_manifest(
            local_manifest("test-model", &model_path, ""),
            PowerConfig {
                tee_mode: true,
                ..Default::default()
            },
        );

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "model_hash_missing");
    }

    #[tokio::test]
    async fn test_attestation_with_model_binds_directory_manifest_hash() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("hf-model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), br#"{"model_type":"bert"}"#).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), b"weights-v1").unwrap();
        let hash = storage::compute_sha256_directory(&model_dir).unwrap();
        let mut manifest = local_manifest("test-model", &model_dir, &hash);
        manifest.format = ModelFormat::HuggingFace;
        let state = test_state_with_manifest(
            manifest,
            PowerConfig {
                tee_mode: true,
                ..Default::default()
            },
        );

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        let model = claims.model.as_ref().unwrap();
        assert_eq!(model.kind, ModelDigestKind::DirectoryManifestSha256);
        assert_eq!(hex::encode(&model.digest), hash);
        assert_eq!(
            report.report_data,
            build_claims_report_data(claims).unwrap()
        );
    }

    #[tokio::test]
    async fn test_attestation_with_model_rejects_tampered_directory_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("hf-model");
        std::fs::create_dir(&model_dir).unwrap();
        let weights_path = model_dir.join("model.safetensors");
        std::fs::write(model_dir.join("config.json"), br#"{"model_type":"bert"}"#).unwrap();
        std::fs::write(&weights_path, b"weights-v1").unwrap();
        let hash = storage::compute_sha256_directory(&model_dir).unwrap();
        let mut manifest = local_manifest("test-model", &model_dir, &hash);
        manifest.format = ModelFormat::HuggingFace;
        let state = test_state_with_manifest(
            manifest,
            PowerConfig {
                tee_mode: true,
                ..Default::default()
            },
        );
        std::fs::write(&weights_path, b"weights-v2").unwrap();

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "model_hash_mismatch");
    }

    #[tokio::test]
    async fn test_attestation_with_model_binds_encrypted_plaintext_when_pinned() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        std::fs::write(&plain_path, b"plaintext-model-v1").unwrap();
        let key = [0x42; 32];
        let model_path =
            crate::tee::encrypted_model::encrypt_model_file(&plain_path, &key).unwrap();
        let key_path = dir.path().join("model.key");
        std::fs::write(&key_path, hex::encode(key)).unwrap();
        let plaintext_hash = storage::compute_sha256(b"plaintext-model-v1");
        let ciphertext_hash = storage::compute_sha256_file(&model_path).unwrap();
        let state = test_state_with_manifest(
            local_manifest("test-model", &model_path, &ciphertext_hash),
            PowerConfig {
                tee_mode: true,
                model_key_source: Some(crate::tee::encrypted_model::KeySource::File(key_path)),
                model_hashes: HashMap::from([("test-model".to_string(), plaintext_hash.clone())]),
                ..Default::default()
            },
        );

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        let model = claims.model.as_ref().unwrap();
        assert_eq!(model.kind, ModelDigestKind::PlaintextWeightsSha256);
        assert_eq!(hex::encode(&model.digest), plaintext_hash);
        assert_eq!(
            model.plaintext_digest.as_ref().map(hex::encode),
            Some(plaintext_hash)
        );
        assert_eq!(
            model.ciphertext_digest.as_ref().map(hex::encode),
            Some(ciphertext_hash)
        );
    }

    #[tokio::test]
    async fn test_attestation_with_model_binds_encrypted_ciphertext_without_key() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        std::fs::write(&plain_path, b"plaintext-model-v1").unwrap();
        let key = [0x42; 32];
        let model_path =
            crate::tee::encrypted_model::encrypt_model_file(&plain_path, &key).unwrap();
        let ciphertext_hash = storage::compute_sha256_file(&model_path).unwrap();
        let state = test_state_with_manifest(
            local_manifest("test-model", &model_path, &ciphertext_hash),
            PowerConfig {
                tee_mode: true,
                ..Default::default()
            },
        );

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::OK);
        let report = response_report(resp).await;
        let claims = report.claims.as_ref().unwrap();
        let model = claims.model.as_ref().unwrap();
        assert_eq!(model.kind, ModelDigestKind::CiphertextArtifactSha256);
        assert_eq!(hex::encode(&model.digest), ciphertext_hash);
        assert!(model.plaintext_digest.is_none());
        assert_eq!(
            model.ciphertext_digest.as_ref().map(hex::encode),
            Some(ciphertext_hash)
        );
    }

    #[tokio::test]
    async fn test_attestation_with_model_rejects_encrypted_plaintext_pin_without_key() {
        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        std::fs::write(&plain_path, b"plaintext-model-v1").unwrap();
        let key = [0x42; 32];
        let model_path =
            crate::tee::encrypted_model::encrypt_model_file(&plain_path, &key).unwrap();
        let ciphertext_hash = storage::compute_sha256_file(&model_path).unwrap();
        let plaintext_hash = storage::compute_sha256(b"plaintext-model-v1");
        let state = test_state_with_manifest(
            local_manifest("test-model", &model_path, &ciphertext_hash),
            PowerConfig {
                tee_mode: true,
                model_hashes: HashMap::from([("test-model".to_string(), plaintext_hash)]),
                ..Default::default()
            },
        );

        let resp = handler(State(state), with_model("test-model"))
            .await
            .into_response();

        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let json = response_json(resp).await;
        assert_eq!(json["error"]["code"], "model_key_missing");
    }

    #[test]
    fn test_decode_hex_nonce_valid() {
        let bytes = decode_hex_nonce("deadbeef").unwrap();
        assert_eq!(bytes, vec![0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn test_decode_hex_nonce_odd_length_fails() {
        assert!(decode_hex_nonce("abc").is_err());
    }

    #[test]
    fn test_decode_hex_nonce_invalid_chars_fails() {
        assert!(decode_hex_nonce("zzzz").is_err());
    }

    #[test]
    fn test_decode_hex_nonce_unicode_fails_without_panicking() {
        let result = std::panic::catch_unwind(|| decode_hex_nonce("🙂"));

        assert!(result.is_ok(), "unicode nonce input must not panic");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_decode_hex_nonce_empty_is_valid() {
        let bytes = decode_hex_nonce("").unwrap();
        assert!(bytes.is_empty());
    }
}
