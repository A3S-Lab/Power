use axum::extract::State;
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
    // Parse the Modelfile
    let mf = match modelfile::parse(&request.modelfile) {
        Ok(mf) => mf,
        Err(e) => {
            return Json(serde_json::json!({
                "error": format!("Failed to parse Modelfile: {e}")
            }))
            .into_response();
        }
    };

    // Resolve the base model
    let base_manifest = match state.registry.get(&mf.from) {
        Ok(m) => m,
        Err(_) => {
            return Json(serde_json::json!({
                "error": format!("base model '{}' not found; pull it first", mf.from)
            }))
            .into_response();
        }
    };

    // Build default parameters from Modelfile
    let default_params = modelfile::parameters_to_json(&mf);
    let modelfile_content = modelfile::to_string(&mf);

    // Create new manifest inheriting from the base model
    let new_manifest = crate::model::manifest::ModelManifest {
        name: request.name.clone(),
        format: base_manifest.format.clone(),
        size: base_manifest.size,
        sha256: base_manifest.sha256.clone(),
        parameters: base_manifest.parameters.clone(),
        created_at: chrono::Utc::now(),
        path: base_manifest.path.clone(),
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
        messages: mf
            .messages
            .iter()
            .map(|m| crate::model::manifest::ManifestMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect(),
    };

    // Register the new model
    match state.registry.register(new_manifest) {
        Ok(()) => Json(CreateResponse {
            status: "success".to_string(),
        })
        .into_response(),
        Err(e) => Json(serde_json::json!({
            "error": format!("Failed to register model: {e}")
        }))
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
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "success");

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
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("FROM"));
    }
}
