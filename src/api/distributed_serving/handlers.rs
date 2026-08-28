use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use uuid::Uuid;

use crate::error::PowerError;
use crate::server::state::AppState;
use crate::serving::{
    DecodePhaseRequest, DistributedServingRuntime, PhaseDecision, PrefillPhaseRequest,
};

use super::stream::decode_stream_response;
use super::{
    AbortDistributedExecutionRequest, AbortDistributedExecutionResponse, DecodeExecuteRequest,
    DecodePrepareRequest, DistributedPhaseDecision, DistributedPhaseResponse,
    DistributedProtocolErrorCode, DistributedProtocolErrorResponse, PrefillExecuteRequest,
    PreparedDecodeResult, PublishedPrefillResult, DISTRIBUTED_SERVING_SCHEMA,
};

#[derive(Clone)]
struct ProtocolBinding {
    execution_id: Uuid,
    worker_epoch: Uuid,
    execution_profile_sha256: String,
}

pub(super) struct ProtocolError {
    status: StatusCode,
    code: DistributedProtocolErrorCode,
    message: &'static str,
}

impl ProtocolError {
    fn new(status: StatusCode, code: DistributedProtocolErrorCode, message: &'static str) -> Self {
        Self {
            status,
            code,
            message,
        }
    }

    fn invalid_body(rejection: JsonRejection) -> Self {
        let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            StatusCode::PAYLOAD_TOO_LARGE
        } else {
            StatusCode::BAD_REQUEST
        };
        Self::new(
            status,
            DistributedProtocolErrorCode::InvalidRequest,
            "request body does not match the distributed-serving v1 contract",
        )
    }
}

impl IntoResponse for ProtocolError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(DistributedProtocolErrorResponse {
                schema: DISTRIBUTED_SERVING_SCHEMA.to_string(),
                code: self.code,
                message: self.message.to_string(),
            }),
        )
            .into_response()
    }
}

pub(super) async fn prepare_decode(
    State(state): State<AppState>,
    payload: Result<Json<DecodePrepareRequest>, JsonRejection>,
) -> Result<Response, ProtocolError> {
    let request = payload.map_err(ProtocolError::invalid_body)?.0;
    let (runtime, binding) = validate_binding(
        &state,
        &request.schema,
        request.execution_id,
        request.worker_epoch,
        &request.execution_profile_sha256,
    )?;
    let (model, phase_request) = request
        .request
        .into_phase_request(&state.config)
        .map_err(protocol_runtime_error)?;
    let decision = runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: binding.execution_id,
            model,
            request: phase_request,
            expires_at: request.expires_at,
        })
        .await
        .map_err(protocol_runtime_error)?;

    Ok(decision_response(binding, decision, |prepared| {
        PreparedDecodeResult {
            target: prepared.target,
        }
    }))
}

pub(super) async fn execute_prefill(
    State(state): State<AppState>,
    payload: Result<Json<PrefillExecuteRequest>, JsonRejection>,
) -> Result<Response, ProtocolError> {
    let request = payload.map_err(ProtocolError::invalid_body)?.0;
    let (runtime, binding) = validate_binding(
        &state,
        &request.schema,
        request.execution_id,
        request.worker_epoch,
        &request.execution_profile_sha256,
    )?;
    let (model, phase_request) = request
        .request
        .into_phase_request(&state.config)
        .map_err(protocol_runtime_error)?;
    let decision = runtime
        .execute_prefill(PrefillPhaseRequest {
            execution_id: binding.execution_id,
            model,
            request: phase_request,
            target: request.target,
            expires_at: request.expires_at,
        })
        .await
        .map_err(protocol_runtime_error)?;

    Ok(decision_response(binding, decision, |published| {
        PublishedPrefillResult {
            source: published.source,
        }
    }))
}

pub(super) async fn execute_decode(
    State(state): State<AppState>,
    payload: Result<Json<DecodeExecuteRequest>, JsonRejection>,
) -> Result<Response, ProtocolError> {
    let request = payload.map_err(ProtocolError::invalid_body)?.0;
    let (runtime, binding) = validate_binding(
        &state,
        &request.schema,
        request.execution_id,
        request.worker_epoch,
        &request.execution_profile_sha256,
    )?;
    let decision = runtime
        .execute_decode(binding.execution_id, request.source)
        .await
        .map_err(protocol_runtime_error)?;

    Ok(match decision {
        PhaseDecision::Ready(stream) => decode_stream_response(
            binding.execution_id,
            binding.worker_epoch,
            binding.execution_profile_sha256,
            stream,
        ),
        PhaseDecision::Recompute { reason } => {
            decision_response(binding, PhaseDecision::<()>::Recompute { reason }, |()| ())
        }
        PhaseDecision::RetryableUnavailable {
            reason,
            retry_after_ms,
        } => decision_response(
            binding,
            PhaseDecision::<()>::RetryableUnavailable {
                reason,
                retry_after_ms,
            },
            |()| (),
        ),
        PhaseDecision::TerminalFailure { reason } => decision_response(
            binding,
            PhaseDecision::<()>::TerminalFailure { reason },
            |()| (),
        ),
    })
}

pub(super) async fn abort_execution(
    State(state): State<AppState>,
    payload: Result<Json<AbortDistributedExecutionRequest>, JsonRejection>,
) -> Result<Response, ProtocolError> {
    let request = payload.map_err(ProtocolError::invalid_body)?.0;
    let (runtime, binding) = validate_binding(
        &state,
        &request.schema,
        request.execution_id,
        request.worker_epoch,
        &request.execution_profile_sha256,
    )?;
    runtime
        .abort(binding.execution_id)
        .await
        .map_err(protocol_runtime_error)?;

    Ok(Json(AbortDistributedExecutionResponse {
        schema: DISTRIBUTED_SERVING_SCHEMA.to_string(),
        execution_id: binding.execution_id,
        worker_epoch: binding.worker_epoch,
        execution_profile_sha256: binding.execution_profile_sha256,
        accepted: true,
    })
    .into_response())
}

fn validate_binding(
    state: &AppState,
    schema: &str,
    execution_id: Uuid,
    worker_epoch: Uuid,
    execution_profile_sha256: &str,
) -> Result<(Arc<DistributedServingRuntime>, ProtocolBinding), ProtocolError> {
    if schema != DISTRIBUTED_SERVING_SCHEMA {
        return Err(ProtocolError::new(
            StatusCode::BAD_REQUEST,
            DistributedProtocolErrorCode::UnsupportedSchema,
            "unsupported distributed-serving schema",
        ));
    }
    if execution_id.is_nil() {
        return Err(ProtocolError::new(
            StatusCode::BAD_REQUEST,
            DistributedProtocolErrorCode::InvalidRequest,
            "distributed execution identifier is invalid",
        ));
    }
    if worker_epoch != state.worker_epoch() {
        return Err(ProtocolError::new(
            StatusCode::CONFLICT,
            DistributedProtocolErrorCode::StaleWorker,
            "worker epoch is stale",
        ));
    }
    let runtime = state.distributed_serving.clone().ok_or_else(|| {
        ProtocolError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            DistributedProtocolErrorCode::Unavailable,
            "distributed-serving runtime is unavailable",
        )
    })?;
    let active_profile_sha256 = runtime.profile().sha256().map_err(protocol_runtime_error)?;
    if execution_profile_sha256 != active_profile_sha256 {
        return Err(ProtocolError::new(
            StatusCode::CONFLICT,
            DistributedProtocolErrorCode::ProfileMismatch,
            "execution profile does not match this worker",
        ));
    }

    Ok((
        runtime,
        ProtocolBinding {
            execution_id,
            worker_epoch,
            execution_profile_sha256: active_profile_sha256,
        },
    ))
}

fn protocol_runtime_error(error: PowerError) -> ProtocolError {
    match error {
        PowerError::InvalidRequest(_) | PowerError::InvalidFormat(_) => ProtocolError::new(
            StatusCode::BAD_REQUEST,
            DistributedProtocolErrorCode::InvalidRequest,
            "distributed phase request is invalid",
        ),
        PowerError::BackendNotAvailable(_)
        | PowerError::InferenceQueueFull { .. }
        | PowerError::ModelSessionPoolFull { .. } => ProtocolError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            DistributedProtocolErrorCode::Unavailable,
            "distributed phase service is unavailable",
        ),
        PowerError::InferenceDeadlineExceeded | PowerError::InferenceCancelled => {
            ProtocolError::new(
                StatusCode::REQUEST_TIMEOUT,
                DistributedProtocolErrorCode::Unavailable,
                "distributed phase deadline was exceeded",
            )
        }
        PowerError::Unauthorized(_) => ProtocolError::new(
            StatusCode::UNAUTHORIZED,
            DistributedProtocolErrorCode::InvalidRequest,
            "distributed phase request is unauthorized",
        ),
        PowerError::PolicyViolation(_) => ProtocolError::new(
            StatusCode::FORBIDDEN,
            DistributedProtocolErrorCode::InvalidRequest,
            "distributed phase request violates worker policy",
        ),
        PowerError::ModelNotFound(_)
        | PowerError::InferenceFailed(_)
        | PowerError::Io(_)
        | PowerError::Server(_)
        | PowerError::Config(_)
        | PowerError::Serialization(_)
        | PowerError::Acl(_)
        | PowerError::IntegrityCheckFailed { .. }
        | PowerError::SignatureVerificationFailed { .. }
        | PowerError::AttestationVerificationFailed(_) => ProtocolError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            DistributedProtocolErrorCode::Internal,
            "distributed phase execution failed",
        ),
    }
}

fn decision_response<T, U>(
    binding: ProtocolBinding,
    decision: PhaseDecision<T>,
    map_ready: impl FnOnce(T) -> U,
) -> Response
where
    U: Serialize,
{
    let (status, outcome) = match decision {
        PhaseDecision::Ready(value) => (
            StatusCode::OK,
            DistributedPhaseDecision::Ready {
                result: map_ready(value),
            },
        ),
        PhaseDecision::Recompute { reason } => (
            StatusCode::CONFLICT,
            DistributedPhaseDecision::Recompute { reason },
        ),
        PhaseDecision::RetryableUnavailable {
            reason,
            retry_after_ms,
        } => (
            StatusCode::SERVICE_UNAVAILABLE,
            DistributedPhaseDecision::RetryableUnavailable {
                reason,
                retry_after_ms,
            },
        ),
        PhaseDecision::TerminalFailure { reason } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            DistributedPhaseDecision::TerminalFailure { reason },
        ),
    };
    (
        status,
        Json(DistributedPhaseResponse {
            schema: DISTRIBUTED_SERVING_SCHEMA.to_string(),
            execution_id: binding.execution_id,
            worker_epoch: binding.worker_epoch,
            execution_profile_sha256: binding.execution_profile_sha256,
            outcome,
        }),
    )
        .into_response()
}
