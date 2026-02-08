pub mod blobs;
pub mod chat;
pub mod copy;
pub mod create;
pub mod embed;
pub mod embeddings;
pub mod generate;
pub mod models;
pub mod ps;
pub mod pull;
pub mod push;
pub mod version;

use axum::routing::{delete, get, head, post};
use axum::Router;

use crate::server::state::AppState;

/// Build the native (Ollama-compatible) API routes.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/generate", post(generate::handler))
        .route("/chat", post(chat::handler))
        .route("/pull", post(pull::handler))
        .route("/push", post(push::handler))
        .route("/tags", get(models::list_handler))
        .route("/show", post(models::show_handler))
        .route("/delete", delete(models::delete_handler))
        .route("/embeddings", post(embeddings::handler))
        .route("/version", get(version::handler))
        .route("/ps", get(ps::handler))
        .route("/copy", post(copy::handler))
        .route("/create", post(create::handler))
        .route("/embed", post(embed::handler))
        .route(
            "/blobs/:digest",
            head(blobs::check_handler)
                .post(blobs::upload_handler)
                .get(blobs::download_handler)
                .delete(blobs::delete_handler),
        )
}
