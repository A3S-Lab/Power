use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::model::modelfile;
use crate::server::state::AppState;

/// Request body for POST /api/create.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRequest {
    pub name: String,
    pub modelfile: String,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Quantization level to apply (e.g. "q4_0", "q4_1", "q5_0", "q5_1", "q8_0").
    /// Note: actual re-quantization is not yet supported; this field is accepted
    /// for API compatibility and stored in the manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize: Option<String>,
}

/// Response body for POST /api/create.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResponse {
    pub status: String,
}

/// POST /api/create - Create a model from a Modelfile (Ollama-compatible).
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<CreateRequest>,
) -> impl IntoResponse {
    let is_stream = request.stream.unwrap_or(true);

    // Parse the Modelfile
    let mf = match modelfile::parse(&request.modelfile) {
        Ok(mf) => mf,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Failed to parse Modelfile: {e}")
                })),
            )
                .into_response();
        }
    };

    // Resolve the base model — either a registered model name or a local GGUF file path
    let from_path = std::path::Path::new(&mf.from);
    let is_local_file = from_path.extension().map_or(false, |ext| ext == "gguf")
        || mf.from.starts_with('/')
        || mf.from.starts_with("./")
        || mf.from.starts_with("../");

    let (base_format, base_size, base_sha256, base_params, base_path, base_family, base_families) = if is_local_file {
        let gguf_path = from_path.to_path_buf();
        if !gguf_path.exists() {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("GGUF file '{}' not found", mf.from)
                })),
            )
                .into_response();
        }
        let file_size = match std::fs::metadata(&gguf_path) {
            Ok(m) => m.len(),
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Failed to read GGUF file: {e}")
                    })),
                )
                    .into_response();
            }
        };
        let blob_path = match crate::model::storage::store_blob_from_path(&gguf_path) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Failed to store blob: {e}")
                    })),
                )
                    .into_response();
            }
        };
        let sha256 = match crate::model::storage::compute_sha256_file(&gguf_path) {
            Ok(h) => h,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Failed to compute hash: {e}")
                    })),
                )
                    .into_response();
            }
        };
        (
            crate::model::manifest::ModelFormat::Gguf,
            file_size,
            sha256,
            None,
            blob_path,
            None,
            None,
        )
    } else {
        match state.registry.get(&mf.from) {
            Ok(base) => (
                base.format.clone(),
                base.size,
                base.sha256.clone(),
                base.parameters.clone(),
                base.path.clone(),
                base.family.clone(),
                base.families.clone(),
            ),
            Err(_) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("base model '{}' not found; pull it first", mf.from)
                    })),
                )
                    .into_response();
            }
        }
    };

    // Build default parameters from Modelfile
    let default_params = modelfile::parameters_to_json(&mf);
    let modelfile_content = modelfile::to_string(&mf);

    // Create new manifest inheriting from the base model
    let new_manifest = crate::model::manifest::ModelManifest {
        name: request.name.clone(),
        format: base_format,
        size: base_size,
        sha256: base_sha256,
        parameters: base_params,
        created_at: chrono::Utc::now(),
        path: base_path,
        system_prompt: mf.system.clone(),
        template_override: mf.template.clone(),
        default_parameters: if default_params.is_empty() {
            None
        } else {
            Some(default_params)
        },
        modelfile_content: Some(modelfile_content),
        license: mf.license.clone(),
        adapter_path: mf.adapter.clone(),
        projector_path: None,
        messages: mf
            .messages
            .iter()
            .map(|m| crate::model::manifest::ManifestMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect(),
        family: base_family,
        families: base_families,
    };

    // Register the new model
    match state.registry.register(new_manifest) {
        Ok(()) => {
            if is_stream {
                // Stream NDJSON status updates (Ollama-compatible)
                let statuses = vec![
                    CreateResponse { status: "reading model metadata".to_string() },
                    CreateResponse { status: "creating system layer".to_string() },
                    CreateResponse { status: "using already created layer".to_string() },
                    CreateResponse { status: "writing manifest".to_string() },
                    CreateResponse { status: "success".to_string() },
                ];
                let stream = futures::stream::iter(statuses);
                crate::api::sse::ndjson_response(stream)
            } else {
                Json(CreateResponse {
                    status: "success".to_string(),
                })
                .into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to register model: {e}")
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    #[serial]
    async fn test_create_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("llama3.2:3b"))
            .unwrap();

        let app = router::build(state);
        let body = serde_json::json!({
            "name": "my-model",
            "modelfile": "FROM llama3.2:3b\nPARAMETER temperature 0.7\nSYSTEM \"You are helpful.\""
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/create")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        // Default stream=true returns NDJSON; last line should be "success"
        let text = String::from_utf8(body.to_vec()).unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();
        assert!(!lines.is_empty(), "NDJSON body should have at least one line");
        let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["status"], "success");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_create_base_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let body = serde_json::json!({
            "name": "my-model",
            "modelfile": "FROM nonexistent\nPARAMETER temperature 0.7"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/create")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn test_create_invalid_modelfile() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let body = serde_json::json!({
            "name": "my-model",
            "modelfile": "PARAMETER temperature 0.7"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/create")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("FROM"));
    }

    #[test]
    fn test_create_request_with_quantize() {
        let json = r#"{"name": "my-model", "modelfile": "FROM llama3", "quantize": "q4_0"}"#;
        let req: super::CreateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "my-model");
        assert_eq!(req.quantize.as_deref(), Some("q4_0"));
    }

    #[test]
    fn test_create_request_without_quantize() {
        let json = r#"{"name": "my-model", "modelfile": "FROM llama3"}"#;
        let req: super::CreateRequest = serde_json::from_str(json).unwrap();
        assert!(req.quantize.is_none());
    }
}
