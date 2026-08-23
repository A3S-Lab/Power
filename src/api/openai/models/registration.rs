use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use crate::model::manifest::{
    AuxiliaryModelArtifact, ExternalDraftArtifact, ExternalDraftKind, ModelFormat, ModelManifest,
};
use crate::server::state::AppState;

/// Request body for POST /v1/models - register a local model file.
#[derive(Debug, Deserialize)]
pub struct RegisterModelRequest {
    /// Display name for the model.
    pub name: String,
    /// Absolute path to the model file on disk.
    pub path: String,
    /// Model format: "gguf" (default), "safetensors", or "huggingface".
    #[serde(default)]
    pub format: Option<String>,
    /// Optional content-addressed DFlash or DSpark GGUF draft artifact.
    #[serde(default)]
    pub external_draft: Option<RegisterExternalDraftRequest>,
    /// Optional LoRA/QLoRA adapter captured by Power as a content-addressed artifact.
    #[serde(default)]
    pub adapter: Option<RegisterAuxiliaryArtifactRequest>,
    /// Optional multimodal projector captured by Power as a content-addressed artifact.
    #[serde(default)]
    pub projector: Option<RegisterAuxiliaryArtifactRequest>,
    /// Unknown registration fields are preserved so handlers can reject
    /// unsupported model policy instead of silently dropping it.
    #[serde(default, flatten)]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

/// Client-supplied location for an auxiliary inference artifact.
///
/// Power measures size and SHA-256; callers cannot inject integrity metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct RegisterAuxiliaryArtifactRequest {
    pub path: String,
    #[serde(default, flatten)]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl RegisterAuxiliaryArtifactRequest {
    fn unsupported_fields_message(&self, field: &str) -> Option<String> {
        if self.unsupported.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported {field} field(s): {}; the only supported field is path",
                self.unsupported
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    }
}

/// Client-supplied external-draft location and provenance.
///
/// Power measures the draft size and digest and binds it to the measured
/// target digest. Integrity fields are deliberately not accepted here.
#[derive(Debug, Clone, Deserialize)]
pub struct RegisterExternalDraftRequest {
    pub kind: ExternalDraftKind,
    pub path: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default, flatten)]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl RegisterExternalDraftRequest {
    fn unsupported_fields_message(&self) -> Option<String> {
        if self.unsupported.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported external_draft field(s): {}; supported fields are kind, path, source, revision, and license",
                self.unsupported.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        }
    }
}

impl RegisterModelRequest {
    fn unsupported_fields_message(&self) -> Option<String> {
        if self.unsupported.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported model registration field(s): {}; supported fields are name, path, format, external_draft, adapter, and projector",
                self.unsupported.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        }
    }
}

fn parse_model_format(format: Option<&str>) -> Result<ModelFormat, String> {
    match format.unwrap_or("gguf") {
        "gguf" => Ok(ModelFormat::Gguf),
        "safetensors" => Ok(ModelFormat::SafeTensors),
        "huggingface" => Ok(ModelFormat::HuggingFace),
        unsupported => Err(format!(
            "unsupported model format '{unsupported}'; supported values are gguf, safetensors, and huggingface"
        )),
    }
}

#[derive(Debug)]
struct InspectionError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl InspectionError {
    fn bad_request(code: &'static str, message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message,
        }
    }

    fn server_error(code: &'static str, message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code,
            message,
        }
    }

    fn into_response(self) -> Response {
        error_response(self.status, self.code, self.message)
    }
}

fn inspect_target(
    path: &Path,
    format: &ModelFormat,
    require_valid_gguf: bool,
) -> Result<(u64, String), InspectionError> {
    if !path.exists() {
        return Err(InspectionError::bad_request(
            "file_not_found",
            format!("path not found: {}", path.display()),
        ));
    }

    if format == &ModelFormat::HuggingFace {
        if !path.is_dir() {
            return Err(InspectionError::bad_request(
                "not_a_directory",
                format!(
                    "huggingface model path must be a directory: {}",
                    path.display()
                ),
            ));
        }
        let size = dir_size(path).map_err(|error| {
            InspectionError::bad_request(
                "path_unreadable",
                format!(
                    "failed to inspect huggingface model directory {}: {error}",
                    path.display()
                ),
            )
        })?;
        return Ok((size, String::new()));
    }

    let metadata = std::fs::metadata(path).map_err(|error| {
        InspectionError::bad_request(
            "path_unreadable",
            format!("failed to inspect model file {}: {error}", path.display()),
        )
    })?;
    if !metadata.is_file() {
        return Err(InspectionError::bad_request(
            "not_a_file",
            format!("model path must be a file: {}", path.display()),
        ));
    }

    if require_valid_gguf {
        crate::model::gguf::read_metadata(path).map_err(|error| {
            InspectionError::bad_request(
                "invalid_target_gguf",
                format!("external_draft target is not a valid GGUF file: {error}"),
            )
        })?;
    }

    let sha256 = crate::model::storage::compute_sha256_file(path).map_err(|error| {
        InspectionError::server_error("hash_failed", format!("failed to hash model file: {error}"))
    })?;
    Ok((metadata.len(), sha256))
}

fn error_response(status: StatusCode, code: &'static str, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": if status.is_server_error() {
                    "server_error"
                } else {
                    "invalid_request_error"
                },
                "code": code
            }
        })),
    )
        .into_response()
}

/// POST /v1/models - Register a local model and optional external draft.
pub async fn register_handler(
    State(state): State<AppState>,
    Json(req): Json<RegisterModelRequest>,
) -> impl IntoResponse {
    if let Some(message) = req.unsupported_fields_message() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_request_fields",
            message,
        );
    }
    if let Some(message) = req
        .external_draft
        .as_ref()
        .and_then(RegisterExternalDraftRequest::unsupported_fields_message)
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_request_fields",
            message,
        );
    }
    for (field, artifact) in [
        ("adapter", req.adapter.as_ref()),
        ("projector", req.projector.as_ref()),
    ] {
        if let Some(message) =
            artifact.and_then(|artifact| artifact.unsupported_fields_message(field))
        {
            return error_response(
                StatusCode::BAD_REQUEST,
                "unsupported_request_fields",
                message,
            );
        }
    }

    let format = match parse_model_format(req.format.as_deref()) {
        Ok(format) => format,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, "unsupported_model_format", message);
        }
    };
    if req.external_draft.is_some() && format != ModelFormat::Gguf {
        return error_response(
            StatusCode::BAD_REQUEST,
            "external_draft_requires_gguf",
            "external_draft is supported only for GGUF target models".to_string(),
        );
    }
    if (req.adapter.is_some() || req.projector.is_some()) && format != ModelFormat::Gguf {
        return error_response(
            StatusCode::BAD_REQUEST,
            "auxiliary_artifact_requires_gguf",
            "adapter and projector artifacts are supported only for GGUF target models".to_string(),
        );
    }

    let path = PathBuf::from(&req.path);
    let inspection_path = path.clone();
    let inspection_format = format.clone();
    let require_valid_gguf =
        req.external_draft.is_some() || req.adapter.is_some() || req.projector.is_some();
    let (size, sha256) = match tokio::task::spawn_blocking(move || {
        inspect_target(&inspection_path, &inspection_format, require_valid_gguf)
    })
    .await
    {
        Ok(Ok(identity)) => identity,
        Ok(Err(error)) => return error.into_response(),
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_inspection_failed",
                format!("model inspection task failed: {error}"),
            );
        }
    };

    let external_draft = if let Some(draft) = req.external_draft {
        let target_sha256 = sha256.clone();
        match tokio::task::spawn_blocking(move || {
            ExternalDraftArtifact::capture_for_target(
                draft.kind,
                PathBuf::from(draft.path),
                target_sha256,
                draft.source,
                draft.revision,
                draft.license,
            )
        })
        .await
        {
            Ok(Ok(artifact)) => Some(artifact),
            Ok(Err(error)) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_external_draft",
                    error.to_string(),
                );
            }
            Err(error) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "external_draft_inspection_failed",
                    format!("external draft inspection task failed: {error}"),
                );
            }
        }
    } else {
        None
    };

    let adapter_request = req.adapter;
    let projector_request = req.projector;
    let (adapter_artifact, projector_artifact) = match tokio::task::spawn_blocking(move || {
        let adapter = adapter_request
            .map(|artifact| AuxiliaryModelArtifact::capture(PathBuf::from(artifact.path)))
            .transpose()
            .map_err(|error| {
                crate::error::PowerError::Config(format!("Failed to capture LoRA adapter: {error}"))
            })?;
        let projector = projector_request
            .map(|artifact| AuxiliaryModelArtifact::capture(PathBuf::from(artifact.path)))
            .transpose()
            .map_err(|error| {
                crate::error::PowerError::Config(format!(
                    "Failed to capture multimodal projector: {error}"
                ))
            })?;
        Ok::<_, crate::error::PowerError>((adapter, projector))
    })
    .await
    {
        Ok(Ok(artifacts)) => artifacts,
        Ok(Err(error)) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_auxiliary_artifact",
                error.to_string(),
            );
        }
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "auxiliary_artifact_inspection_failed",
                format!("auxiliary artifact inspection task failed: {error}"),
            );
        }
    };

    let manifest = ModelManifest {
        name: req.name.clone(),
        format,
        size,
        sha256,
        parameters: None,
        created_at: chrono::Utc::now(),
        path,
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        adapter_artifact,
        external_draft,
        projector_path: None,
        projector_artifact,
        messages: vec![],
        family: None,
        families: None,
    };

    match state.registry.register(manifest) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "id": req.name,
                "object": "model",
                "created": chrono::Utc::now().timestamp(),
                "owned_by": "local"
            })),
        )
            .into_response(),
        Err(error) => error_response(
            StatusCode::CONFLICT,
            "model_already_exists",
            error.to_string(),
        ),
    }
}

/// Compute total size of a directory by summing all file sizes.
pub(super) fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0_u64;
    for entry in std::fs::read_dir(path)? {
        let path = entry?.path();
        let metadata = std::fs::metadata(&path)?;
        let size = if metadata.is_dir() {
            dir_size(&path)?
        } else {
            metadata.len()
        };
        total = total
            .checked_add(size)
            .ok_or_else(|| std::io::Error::other("directory size overflow"))?;
    }
    Ok(total)
}
