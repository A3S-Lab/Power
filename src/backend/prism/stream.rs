//! Prism observation policy over the shared OpenAI response protocol.
use super::{PrismProfile, UpstreamSpecMetrics, UpstreamTimings};
use crate::backend::types::ChatResponseChunk;
use crate::error::{PowerError, Result};
use futures::{Stream, StreamExt};
use std::sync::Arc;
use tokio::sync::Mutex;

/// The shared stream owns HTTP startup/body; this wrapper only records the
/// backend's speculative observation after a complete protocol termination.
pub(super) fn chat_response_stream(
    response: reqwest::Response,
    model: String,
    profile: PrismProfile,
    metrics: Arc<Mutex<UpstreamSpecMetrics>>,
    tool_policy: crate::backend::openai_compat::ChatToolPolicy,
) -> impl Stream<Item = Result<ChatResponseChunk>> + Send {
    crate::backend::openai_compat::chat_response_stream(response, false, tool_policy)
        .then(move |result| {
            let metrics = Arc::clone(&metrics);
            let model = model.clone();
            async move {
                let mut chunk = result?;
                if let Some(value) = chunk.upstream_timings.as_ref().and_then(|v|v.get("timings")) {
                    let timings: UpstreamTimings = serde_json::from_value(value.clone())
                        .map_err(|e|PowerError::InferenceFailed(format!("prism upstream protocol error: invalid timings: {e}")))?;
                    chunk.prompt_eval_duration_ns = timings.prompt_eval_duration_ns();
                    if chunk.done {
                        if profile.expects_speculation_timings() && !timings.speculation_engaged() {
                            tracing::warn!(model = %model, profile = profile.as_str(),
                                "prism speculation profile selected but upstream timings have no draft_n");
                        }
                        metrics.lock().await.observe(&timings);
                    }
                }
                Ok(chunk)
            }
        })
}
