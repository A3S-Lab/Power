use std::convert::Infallible;

use axum::body::{Body, Bytes};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, X_CONTENT_TYPE_OPTIONS};
use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;
use futures::{stream, StreamExt};
use uuid::Uuid;

use crate::serving::{PhaseResponseChunk, PhaseResponseStream, TerminalFailureReason};

use super::{
    DistributedDecodeStreamEvent, DistributedDecodeStreamFrame, DistributedResponseChunk,
    DISTRIBUTED_SERVING_STREAM_SCHEMA,
};

enum StreamStage {
    Ready,
    Streaming,
    Done,
}

struct DecodeStreamState {
    execution_id: Uuid,
    worker_epoch: Uuid,
    execution_profile_sha256: String,
    response: PhaseResponseStream,
    next_sequence: u64,
    stage: StreamStage,
}

pub(super) fn decode_stream_response(
    execution_id: Uuid,
    worker_epoch: Uuid,
    execution_profile_sha256: String,
    response: PhaseResponseStream,
) -> Response {
    let state = DecodeStreamState {
        execution_id,
        worker_epoch,
        execution_profile_sha256,
        response,
        next_sequence: 0,
        stage: StreamStage::Ready,
    };
    let body_stream = stream::unfold(state, |mut state| async move {
        let event = match state.stage {
            StreamStage::Ready => {
                state.stage = StreamStage::Streaming;
                DistributedDecodeStreamEvent::Ready
            }
            StreamStage::Streaming => match state.response.next().await {
                Some(Ok(chunk)) => {
                    let sequence = state.next_sequence;
                    state.next_sequence = state.next_sequence.saturating_add(1);
                    DistributedDecodeStreamEvent::Chunk {
                        sequence,
                        response: match chunk {
                            PhaseResponseChunk::Chat(chunk) => {
                                DistributedResponseChunk::ChatCompletions(chunk)
                            }
                            PhaseResponseChunk::Completion(chunk) => {
                                DistributedResponseChunk::Completions(chunk)
                            }
                        },
                    }
                }
                Some(Err(_)) => {
                    state.stage = StreamStage::Done;
                    DistributedDecodeStreamEvent::Failed {
                        sequence: state.next_sequence,
                        reason: TerminalFailureReason::ExecutionFailed,
                    }
                }
                None => {
                    state.stage = StreamStage::Done;
                    DistributedDecodeStreamEvent::Completed {
                        sequence: state.next_sequence,
                    }
                }
            },
            StreamStage::Done => return None,
        };
        let frame = state.frame(event);
        let bytes = match encode_frame(&frame) {
            Ok(bytes) => bytes,
            Err(()) => {
                state.stage = StreamStage::Done;
                encode_fallback_failure(&state)
            }
        };
        Some((Ok::<Bytes, Infallible>(bytes), state))
    });

    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
        .headers_mut()
        .insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    response
}

impl DecodeStreamState {
    fn frame(&self, payload: DistributedDecodeStreamEvent) -> DistributedDecodeStreamFrame {
        DistributedDecodeStreamFrame {
            schema: DISTRIBUTED_SERVING_STREAM_SCHEMA.to_string(),
            execution_id: self.execution_id,
            worker_epoch: self.worker_epoch,
            execution_profile_sha256: self.execution_profile_sha256.clone(),
            payload,
        }
    }
}

fn encode_frame(frame: &DistributedDecodeStreamFrame) -> Result<Bytes, ()> {
    let mut bytes = serde_json::to_vec(frame).map_err(|_| ())?;
    bytes.push(b'\n');
    Ok(Bytes::from(bytes))
}

fn encode_fallback_failure(state: &DecodeStreamState) -> Bytes {
    let frame = state.frame(DistributedDecodeStreamEvent::Failed {
        sequence: state.next_sequence,
        reason: TerminalFailureReason::ExecutionFailed,
    });
    match serde_json::to_vec(&frame) {
        Ok(mut bytes) => {
            bytes.push(b'\n');
            Bytes::from(bytes)
        }
        Err(_) => Bytes::from_static(b"{\"event\":\"failed\"}\n"),
    }
}
