mod contract;
mod handlers;
mod request;
mod stream;

pub use contract::*;

use axum::extract::DefaultBodyLimit;
use axum::routing::post;
use axum::Router;

use crate::server::state::AppState;

const MAX_DISTRIBUTED_REQUEST_BYTES: usize = 8 * 1024 * 1024;

/// Closed machine-to-machine routes used by Gateway's request orchestrator.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/decode/prepare", post(handlers::prepare_decode))
        .route("/prefill/execute", post(handlers::execute_prefill))
        .route("/decode/execute", post(handlers::execute_decode))
        .route("/abort", post(handlers::abort_execution))
        .layer(DefaultBodyLimit::max(MAX_DISTRIBUTED_REQUEST_BYTES))
}
