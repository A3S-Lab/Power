pub(crate) mod support;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use chrono::{Duration, Utc};
use futures::StreamExt;
use uuid::Uuid;

use super::*;
use crate::error::PowerError;
use support::*;
use support::{CorruptConsumeReceipt, DecodeExecuteFixture, TransferHooks};

#[tokio::test]
async fn decode_prepares_destination_then_consumes_before_returning_stream() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime = runtime(&profile, epoch, calls.clone());
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(80);
    let mut stream = start_decode(&runtime, epoch, execution_id, expires_at).await;
    assert!(stream.next().await.unwrap().is_ok());
    assert!(stream.next().await.is_none());

    wait_for_count(&calls.phase_aborts, 1).await;
    assert_eq!(
        calls.values()[..4],
        [
            "phase.prepare",
            "transfer.prepare",
            "transfer.consume",
            "phase.execute"
        ]
    );
    assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn explicit_abort_cancels_an_active_decode_stream_and_is_idempotent() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime = runtime(&profile, epoch, calls.clone());
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(80);
    let mut stream = start_decode(&runtime, epoch, execution_id, expires_at).await;

    runtime.abort(execution_id).await.unwrap();
    runtime.abort(execution_id).await.unwrap();

    let item = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(item, Err(PowerError::BackendNotAvailable(_))));
    assert!(stream.next().await.is_none());
    assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn mismatched_executor_lifetime_is_rejected_and_cleaned_synchronously() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime =
        runtime_with_expiry_delta(&profile, epoch, calls.clone(), Duration::milliseconds(1));
    let result = runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: Uuid::new_v4(),
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at: Utc::now() + Duration::milliseconds(80),
        })
        .await;
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("expected the mismatched lifetime to fail"),
    };

    assert!(matches!(error, PowerError::InvalidRequest(_)));
    assert_eq!(calls.values(), ["phase.prepare", "phase.abort"]);
    assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1);
    assert!(runtime.accepts_work());
}

#[tokio::test]
async fn phase_preparation_deadline_reclaims_the_runtime_lease() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let calls = Arc::new(Calls::default());
    let runtime = runtime_with_behavior(
        &profile,
        Uuid::new_v4(),
        calls.clone(),
        FixtureBehavior {
            block_prepare: true,
            ..FixtureBehavior::default()
        },
    );
    let result = runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: Uuid::new_v4(),
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at: Utc::now() + Duration::milliseconds(80),
        })
        .await;

    assert!(matches!(result, Err(PowerError::BackendNotAvailable(_))));
    wait_for_count(&calls.phase_aborts, 1).await;
    assert!(runtime.accepts_work());
}

#[tokio::test]
async fn retryable_phase_decision_is_forwarded_only_after_cleanup() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let calls = Arc::new(Calls::default());
    let runtime = runtime_with_behavior(
        &profile,
        Uuid::new_v4(),
        calls.clone(),
        FixtureBehavior {
            retryable_prepare: true,
            ..FixtureBehavior::default()
        },
    );
    let decision = runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: Uuid::new_v4(),
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at: Utc::now() + Duration::milliseconds(80),
        })
        .await
        .unwrap();

    assert!(matches!(
        decision,
        PhaseDecision::RetryableUnavailable {
            reason: RetryableUnavailableReason::AdmissionPressure,
            retry_after_ms: Some(5)
        }
    ));
    assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn unconfirmed_cleanup_taints_the_runtime_and_suppresses_new_work() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let calls = Arc::new(Calls::default());
    let runtime = runtime_with_behavior(
        &profile,
        Uuid::new_v4(),
        calls.clone(),
        FixtureBehavior {
            retryable_prepare: true,
            fail_abort: true,
            ..FixtureBehavior::default()
        },
    );
    let result = runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: Uuid::new_v4(),
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at: Utc::now() + Duration::milliseconds(80),
        })
        .await;

    assert!(matches!(result, Err(PowerError::BackendNotAvailable(_))));
    assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1);
    assert!(!runtime.accepts_work());
}

#[tokio::test]
async fn prefill_publishes_state_and_compensating_abort_reclaims_both_owners() {
    let profile = profile(DisaggregatedServingRole::Prefill, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime = runtime(&profile, epoch, calls.clone());
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(80);
    let published = runtime
        .execute_prefill(PrefillPhaseRequest {
            execution_id,
            model: "internal/model-v1".to_string(),
            request: request(),
            target: StateTransferTarget {
                schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
                transfer_id: execution_id,
                destination_worker_epoch: Uuid::new_v4(),
                binding: binding(),
                protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
                prepared_at: Utc::now(),
                expires_at,
                ticket: "target-ticket".to_string(),
            },
            expires_at,
        })
        .await
        .unwrap();
    let source = match published {
        PhaseDecision::Ready(PublishedPrefillState { source }) => source,
        _ => panic!("expected a published prefill source"),
    };
    assert_eq!(source.transfer_id, execution_id);
    assert_eq!(
        calls.values()[..3],
        ["phase.prepare", "phase.execute", "transfer.publish"]
    );

    runtime.abort(execution_id).await.unwrap();
    assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1);
    assert_eq!(calls.transfer_aborts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn transfer_receipt_then_non_ready_execute_never_opens_a_decode_stream() {
    let cases = [
        (
            DecodeExecuteFixture::Recompute(RecomputeReason::StateStale),
            "recompute",
        ),
        (
            DecodeExecuteFixture::RetryableUnavailable {
                reason: RetryableUnavailableReason::ResourcePressure,
                retry_after_ms: Some(25),
            },
            "retryable",
        ),
        (
            DecodeExecuteFixture::TerminalFailure(TerminalFailureReason::ExecutionFailed),
            "terminal",
        ),
    ];

    for (decode_execute, label) in cases {
        let profile = profile(DisaggregatedServingRole::Decode, 100);
        let epoch = Uuid::new_v4();
        let calls = Arc::new(Calls::default());
        let runtime = runtime_with_behavior(
            &profile,
            epoch,
            calls.clone(),
            FixtureBehavior {
                decode_execute,
                ..FixtureBehavior::default()
            },
        );
        let execution_id = Uuid::new_v4();
        let expires_at = Utc::now() + Duration::milliseconds(80);
        let source = prepare_decode_source(&runtime, epoch, execution_id, expires_at).await;
        let decision = runtime
            .execute_decode(execution_id, source)
            .await
            .unwrap_or_else(|error| panic!("{label}: execute should return a decision: {error}"));

        match (decode_execute, decision) {
            (DecodeExecuteFixture::Recompute(expected), PhaseDecision::Recompute { reason }) => {
                assert_eq!(reason, expected, "{label}")
            }
            (
                DecodeExecuteFixture::RetryableUnavailable {
                    reason: expected_reason,
                    retry_after_ms: expected_delay,
                },
                PhaseDecision::RetryableUnavailable {
                    reason,
                    retry_after_ms,
                },
            ) => {
                assert_eq!(reason, expected_reason, "{label}");
                assert_eq!(retry_after_ms, expected_delay, "{label}");
            }
            (
                DecodeExecuteFixture::TerminalFailure(expected),
                PhaseDecision::TerminalFailure { reason },
            ) => assert_eq!(reason, expected, "{label}"),
            _ => panic!("{label}: expected the injected non-Ready decision"),
        }

        assert_eq!(
            calls.values()[..4],
            [
                "phase.prepare",
                "transfer.prepare",
                "transfer.consume",
                "phase.execute"
            ],
            "{label}: consume must precede execute"
        );
        wait_for_count(&calls.phase_aborts, 1).await;
        assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1, "{label}");
        assert!(runtime.accepts_work(), "{label}: cleanup confirmed");
        runtime.abort(execution_id).await.unwrap();
        assert_eq!(
            calls.phase_aborts.load(Ordering::SeqCst),
            1,
            "{label}: abort after decision cleanup is idempotent"
        );
    }
}

#[tokio::test]
async fn post_consume_non_ready_cleanup_failure_taints_readiness() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime = runtime_with_behavior(
        &profile,
        epoch,
        calls.clone(),
        FixtureBehavior {
            decode_execute: DecodeExecuteFixture::Recompute(RecomputeReason::StateCorrupt),
            fail_abort: true,
            ..FixtureBehavior::default()
        },
    );
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(80);
    let source = prepare_decode_source(&runtime, epoch, execution_id, expires_at).await;
    let error = match runtime.execute_decode(execution_id, source).await {
        Err(error) => error,
        Ok(_) => panic!("unconfirmed cleanup must not surface a ready stream"),
    };

    assert!(matches!(error, PowerError::BackendNotAvailable(_)));
    assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1);
    assert!(!runtime.accepts_work());
}

#[tokio::test]
async fn corrupt_source_ticket_bytes_fail_closed_without_opening_a_decode_stream() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime = runtime(&profile, epoch, calls.clone());
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(80);
    let mut source = prepare_decode_source(&runtime, epoch, execution_id, expires_at).await;
    source.ticket = "source\tticket".to_string();

    let error = match runtime.execute_decode(execution_id, source).await {
        Err(error) => error,
        Ok(_) => panic!("corrupt ticket must never open a decode stream"),
    };
    assert!(matches!(error, PowerError::InvalidRequest(_)));
    wait_for_count(&calls.phase_aborts, 1).await;
    assert!(
        !calls
            .values()
            .iter()
            .any(|value| *value == "transfer.consume"),
        "corrupt ticket must fail before consume"
    );
    assert!(
        !calls.values().iter().any(|value| *value == "phase.execute"),
        "corrupt ticket must never reach phase execute"
    );
    assert!(runtime.accepts_work());
}

#[tokio::test]
async fn corrupt_consume_receipt_bytes_fail_closed_with_compensating_cleanup() {
    let cases = [
        CorruptConsumeReceipt::ShortBytes,
        CorruptConsumeReceipt::InvalidIntegrityDigest,
    ];

    for mode in cases {
        let profile = profile(DisaggregatedServingRole::Decode, 100);
        let epoch = Uuid::new_v4();
        let calls = Arc::new(Calls::default());
        let runtime = runtime_with_behavior(
            &profile,
            epoch,
            calls.clone(),
            FixtureBehavior {
                corrupt_consume_receipt: mode,
                ..FixtureBehavior::default()
            },
        );
        let execution_id = Uuid::new_v4();
        let expires_at = Utc::now() + Duration::milliseconds(80);
        let source = prepare_decode_source(&runtime, epoch, execution_id, expires_at).await;
        let error = match runtime.execute_decode(execution_id, source).await {
            Err(error) => error,
            Ok(_) => panic!("{mode:?}: corrupt receipt must never decode as success"),
        };
        assert!(
            matches!(error, PowerError::InvalidRequest(_)),
            "{mode:?}: expected InvalidRequest, got {error:?}"
        );
        assert!(
            calls
                .values()
                .iter()
                .any(|value| *value == "transfer.consume"),
            "{mode:?}: consume must run before integrity rejection"
        );
        assert!(
            !calls.values().iter().any(|value| *value == "phase.execute"),
            "{mode:?}: corrupt receipt must never reach phase execute"
        );
        wait_for_count(&calls.phase_aborts, 1).await;
        assert_eq!(calls.phase_aborts.load(Ordering::SeqCst), 1, "{mode:?}");
        assert!(runtime.accepts_work(), "{mode:?}");
    }
}

#[tokio::test]
async fn inflight_capacity_pressure_fails_closed_and_reclaims_after_abort() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime = runtime(&profile, epoch, calls.clone());
    let expires_at = Utc::now() + Duration::milliseconds(80);
    let first = Uuid::new_v4();
    let second = Uuid::new_v4();
    let third = Uuid::new_v4();

    runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: first,
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at,
        })
        .await
        .unwrap();
    runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: second,
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at,
        })
        .await
        .unwrap();

    let pressure = runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: third,
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at,
        })
        .await;
    assert!(matches!(
        pressure,
        Err(PowerError::BackendNotAvailable(message))
            if message.contains("capacity is exhausted")
    ));
    assert!(runtime.accepts_work());

    runtime.abort(first).await.unwrap();
    runtime
        .prepare_decode(DecodePhaseRequest {
            execution_id: third,
            model: "internal/model-v1".to_string(),
            request: request(),
            expires_at: Utc::now() + Duration::milliseconds(80),
        })
        .await
        .unwrap();
    runtime.abort(second).await.unwrap();
    runtime.abort(third).await.unwrap();
    assert!(runtime.accepts_work());
}

#[tokio::test]
async fn resource_pressure_phase_decision_cleans_up_without_a_decode_stream() {
    let profile = profile(DisaggregatedServingRole::Decode, 100);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let runtime = runtime_with_behavior(
        &profile,
        epoch,
        calls.clone(),
        FixtureBehavior {
            decode_execute: DecodeExecuteFixture::RetryableUnavailable {
                reason: RetryableUnavailableReason::ResourcePressure,
                retry_after_ms: Some(15),
            },
            ..FixtureBehavior::default()
        },
    );
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(80);
    let source = prepare_decode_source(&runtime, epoch, execution_id, expires_at).await;
    let decision = runtime.execute_decode(execution_id, source).await.unwrap();
    assert!(matches!(
        decision,
        PhaseDecision::RetryableUnavailable {
            reason: RetryableUnavailableReason::ResourcePressure,
            retry_after_ms: Some(15)
        }
    ));
    wait_for_count(&calls.phase_aborts, 1).await;
    assert!(runtime.accepts_work());
}

#[tokio::test]
async fn caller_abort_during_transfer_consume_never_opens_a_decode_stream() {
    let profile = profile(DisaggregatedServingRole::Decode, 250);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let hooks = Arc::new(TransferHooks::default());
    hooks.block_consume.store(true, Ordering::SeqCst);
    let runtime = runtime_with_behavior(
        &profile,
        epoch,
        calls.clone(),
        FixtureBehavior {
            transfer_hooks: Arc::clone(&hooks),
            ..FixtureBehavior::default()
        },
    );
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(200);
    let source = prepare_decode_source(&runtime, epoch, execution_id, expires_at).await;
    let running = {
        let runtime = runtime.clone();
        tokio::spawn(async move { runtime.execute_decode(execution_id, source).await })
    };
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        hooks.consume_started.notified(),
    )
    .await
    .expect("consume must start before abort");

    runtime.abort(execution_id).await.unwrap();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), running)
        .await
        .expect("abort must unblock consume")
        .unwrap();
    assert!(
        matches!(outcome, Err(PowerError::BackendNotAvailable(_))),
        "mid-consume abort must not return Ready/NDJSON success"
    );
    assert!(
        !calls.values().iter().any(|value| *value == "phase.execute"),
        "cancelled consume must never reach phase execute"
    );
    wait_for_count(&calls.phase_aborts, 1).await;
    wait_for_count(&calls.transfer_aborts, 1).await;
    assert!(runtime.accepts_work());
    hooks.block_consume.store(false, Ordering::SeqCst);
    assert_lease_capacity_reclaimed(&runtime, 250).await;
}

#[tokio::test]
async fn deadline_abort_during_transfer_consume_never_opens_a_decode_stream() {
    let profile = profile(DisaggregatedServingRole::Decode, 40);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let hooks = Arc::new(TransferHooks::default());
    hooks.block_consume.store(true, Ordering::SeqCst);
    let runtime = runtime_with_behavior(
        &profile,
        epoch,
        calls.clone(),
        FixtureBehavior {
            transfer_hooks: Arc::clone(&hooks),
            ..FixtureBehavior::default()
        },
    );
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(40);
    let source = prepare_decode_source(&runtime, epoch, execution_id, expires_at).await;
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        runtime.execute_decode(execution_id, source),
    )
    .await
    .expect("deadline must abort blocked consume");
    assert!(
        matches!(outcome, Err(PowerError::BackendNotAvailable(_))),
        "deadline mid-consume must not return Ready/NDJSON success"
    );
    assert!(!calls.values().iter().any(|value| *value == "phase.execute"));
    wait_for_count(&calls.phase_aborts, 1).await;
    wait_for_count(&calls.transfer_aborts, 1).await;
    assert!(runtime.accepts_work());
    hooks.block_consume.store(false, Ordering::SeqCst);
    assert_lease_capacity_reclaimed(&runtime, 40).await;
}

#[tokio::test]
async fn caller_abort_during_destination_prepare_reclaims_without_success() {
    let profile = profile(DisaggregatedServingRole::Decode, 250);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let hooks = Arc::new(TransferHooks::default());
    hooks
        .block_prepare_destination
        .store(true, Ordering::SeqCst);
    let runtime = runtime_with_behavior(
        &profile,
        epoch,
        calls.clone(),
        FixtureBehavior {
            transfer_hooks: Arc::clone(&hooks),
            ..FixtureBehavior::default()
        },
    );
    let execution_id = Uuid::new_v4();
    let running = {
        let runtime = runtime.clone();
        tokio::spawn(async move {
            runtime
                .prepare_decode(DecodePhaseRequest {
                    execution_id,
                    model: "internal/model-v1".to_string(),
                    request: request(),
                    expires_at: Utc::now() + Duration::milliseconds(200),
                })
                .await
        })
    };
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        hooks.prepare_destination_started.notified(),
    )
    .await
    .expect("destination prepare must start before abort");

    runtime.abort(execution_id).await.unwrap();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), running)
        .await
        .expect("abort must unblock destination prepare")
        .unwrap();
    assert!(matches!(outcome, Err(PowerError::BackendNotAvailable(_))));
    wait_for_count(&calls.phase_aborts, 1).await;
    wait_for_count(&calls.transfer_aborts, 1).await;
    assert!(runtime.accepts_work());
    hooks
        .block_prepare_destination
        .store(false, Ordering::SeqCst);
    assert_lease_capacity_reclaimed(&runtime, 250).await;
}

#[tokio::test]
async fn caller_abort_mid_decode_stream_never_emits_success_after_cancel() {
    let profile = profile(DisaggregatedServingRole::Decode, 250);
    let epoch = Uuid::new_v4();
    let calls = Arc::new(Calls::default());
    let hooks = Arc::new(TransferHooks::default());
    let runtime = runtime_with_behavior(
        &profile,
        epoch,
        calls.clone(),
        FixtureBehavior {
            decode_execute: DecodeExecuteFixture::HangBeforeFirstChunk,
            transfer_hooks: Arc::clone(&hooks),
            ..FixtureBehavior::default()
        },
    );
    let execution_id = Uuid::new_v4();
    let expires_at = Utc::now() + Duration::milliseconds(200);
    let mut stream = start_decode(&runtime, epoch, execution_id, expires_at).await;
    let waiting = tokio::spawn(async move { stream.next().await });
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        hooks.stream_started.notified(),
    )
    .await
    .expect("decode stream must start before abort");

    runtime.abort(execution_id).await.unwrap();
    let item = tokio::time::timeout(std::time::Duration::from_secs(1), waiting)
        .await
        .expect("abort must unblock hung stream")
        .unwrap()
        .expect("cancelled stream must surface one terminal error frame");
    assert!(
        matches!(item, Err(PowerError::BackendNotAvailable(_))),
        "mid-stream abort must never emit a success token chunk, got {item:?}"
    );
    // Consume already finished before the stream opened, so transfer abort is
    // idempotent and may not invoke the driver again.
    wait_for_count(&calls.phase_aborts, 1).await;
    assert!(runtime.accepts_work());
    assert_lease_capacity_reclaimed(&runtime, 250).await;
}

#[test]
fn distributed_runtime_and_wire_free_commands_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<DistributedServingRuntime>();
    assert_send_sync::<PreparedDecodeTransfer>();
    assert_send_sync::<PublishedPrefillState>();
}
