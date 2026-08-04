use super::microbatch::planner;
use super::*;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn binding() -> ExecutionBatchBinding {
    ExecutionBatchBinding::new(digest('1'), digest('2'), digest('3')).unwrap()
}

fn cpu_snapshot(available_bytes: u64) -> HardwareMemorySnapshot {
    HardwareMemorySnapshot::new(
        "cpu",
        MemoryPoolSnapshot::new(
            2_000,
            available_bytes,
            MemoryDiscoverySource::WindowsGlobalMemoryStatus,
            false,
        )
        .unwrap(),
        None,
    )
    .unwrap()
}

fn unified_snapshot(available_bytes: u64) -> HardwareMemorySnapshot {
    HardwareMemorySnapshot::new(
        "metal:0",
        MemoryPoolSnapshot::new(
            2_000,
            available_bytes,
            MemoryDiscoverySource::MachHostStatistics,
            false,
        )
        .unwrap(),
        Some(
            MemoryPoolSnapshot::new(
                2_000,
                available_bytes,
                MemoryDiscoverySource::MetalRecommendedWorkingSet,
                true,
            )
            .unwrap(),
        ),
    )
    .unwrap()
}

fn limits() -> InferenceLimits {
    InferenceLimits {
        max_concurrent_requests: 1,
        max_input_bytes: 1_000,
        max_tensor_elements: 100,
        max_state_bytes: 1_000,
        max_graph_nodes: 16,
        ..InferenceLimits::default()
    }
}

async fn wait_for<F>(condition: F)
where
    F: Fn() -> bool,
{
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("microbatch state did not converge");
}

fn candidate(character: char, host_peak_bytes: u64) -> MicrobatchCandidate {
    MicrobatchCandidate::new(digest(character), 100, 10, 50, host_peak_bytes, 0).unwrap()
}

fn cpu_plan(
    available_bytes: u64,
    candidates: Vec<MicrobatchCandidate>,
) -> crate::error::Result<MicrobatchPlan> {
    planner::plan(
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
        },
        cpu_snapshot(available_bytes),
        binding(),
        None,
        limits(),
        MicrobatchPolicy::new(3, 10_000, 0)
            .unwrap()
            .with_host_reserve_bytes(100)
            .with_base_memory(100, 0),
        candidates,
    )
}

#[test]
fn planning_is_contiguous_deterministic_and_self_validating() {
    let candidates = vec![
        candidate('a', 250),
        candidate('b', 250),
        candidate('c', 250),
        candidate('d', 250),
        candidate('e', 250),
    ];
    let first = cpu_plan(800, candidates.clone()).unwrap();
    let second = cpu_plan(800, candidates).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.host_budget_bytes, 600);
    assert_eq!(first.batches.len(), 3);
    assert_eq!(first.batches[0].slots.len(), 2);
    assert_eq!(first.batches[1].slots.len(), 2);
    assert_eq!(first.batches[2].slots.len(), 1);
    assert_eq!(
        first
            .batches
            .iter()
            .flat_map(|batch| batch.slots.iter().map(|slot| slot.source_index))
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    first.validate().unwrap();
    let round_trip: MicrobatchPlan =
        serde_json::from_slice(&serde_json::to_vec(&first).unwrap()).unwrap();
    assert_eq!(round_trip, first);
    round_trip.validate().unwrap();
}

#[test]
fn malformed_duplicates_and_individually_oversized_slots_fail_closed() {
    assert!(cpu_plan(800, vec![candidate('a', 250), candidate('a', 250)]).is_err());
    assert!(cpu_plan(300, vec![candidate('a', 250)]).is_err());
    assert!(MicrobatchCandidate::new(digest('a'), 100, 10, 50, 149, 0).is_err());

    let device_on_cpu = MicrobatchCandidate::new(digest('a'), 100, 10, 50, 100, 50).unwrap();
    assert!(cpu_plan(800, vec![device_on_cpu]).is_err());
}

#[test]
fn unified_memory_is_counted_once_across_host_and_device() {
    let candidates = vec![
        MicrobatchCandidate::new(digest('a'), 100, 10, 50, 400, 100).unwrap(),
        MicrobatchCandidate::new(digest('b'), 100, 10, 50, 400, 100).unwrap(),
    ];
    let plan = planner::plan(
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Metal,
            ordinal: Some(0),
        },
        unified_snapshot(900),
        binding(),
        None,
        limits(),
        MicrobatchPolicy::new(3, 10_000, 10_000).unwrap(),
        candidates,
    )
    .unwrap();

    assert_eq!(plan.shared_budget_bytes, Some(900));
    assert_eq!(plan.batches.len(), 2);
    assert_eq!(plan.batches[0].host_peak_bytes, 400);
    assert_eq!(plan.batches[0].device_peak_bytes, 100);
}

#[test]
fn stale_memory_pressure_rejects_only_when_an_existing_batch_no_longer_fits() {
    let plan = cpu_plan(800, vec![candidate('a', 250), candidate('b', 250)]).unwrap();
    plan.revalidate_pressure(&cpu_snapshot(800)).unwrap();
    assert!(plan.revalidate_pressure(&cpu_snapshot(600)).is_err());

    let changed_topology = HardwareMemorySnapshot::new(
        "cpu",
        MemoryPoolSnapshot::new(
            3_000,
            800,
            MemoryDiscoverySource::WindowsGlobalMemoryStatus,
            false,
        )
        .unwrap(),
        None,
    )
    .unwrap();
    assert!(plan.revalidate_pressure(&changed_topology).is_err());
}

#[test]
fn serialized_plan_tampering_is_detected() {
    let plan = cpu_plan(800, vec![candidate('a', 250), candidate('b', 250)]).unwrap();
    let mut digest_tamper = plan.clone();
    digest_tamper.declaration_sha256 = digest('f');
    assert!(digest_tamper.validate().is_err());

    let mut aggregate_tamper = plan;
    aggregate_tamper.batches[0].host_peak_bytes += 1;
    assert!(aggregate_tamper.validate().is_err());
}

#[test]
fn session_binding_changes_plan_identity_and_debug_is_content_free() {
    let base = planner::plan(
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
        },
        cpu_snapshot(800),
        binding(),
        None,
        limits(),
        MicrobatchPolicy::new(3, 10_000, 0).unwrap(),
        vec![candidate('a', 250)],
    )
    .unwrap();
    let pooled = planner::plan(
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
        },
        cpu_snapshot(800),
        binding(),
        Some(digest('f')),
        limits(),
        MicrobatchPolicy::new(3, 10_000, 0).unwrap(),
        vec![candidate('a', 250)],
    )
    .unwrap();
    assert_ne!(base.declaration_sha256, pooled.declaration_sha256);
    assert!(!format!("{pooled:?}").contains(&digest('a')));
}

#[test]
fn microbatch_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MicrobatchCandidate>();
    assert_send_sync::<MicrobatchPolicy>();
    assert_send_sync::<MicrobatchPlan>();
    assert_send_sync::<MicrobatchExecution>();
}

#[tokio::test]
async fn pooled_execution_revalidates_admits_and_emits_a_digest_only_receipt() {
    let model = ModelIdentity::new("test-model", "revision-1", digest('a'));
    let limits = InferenceLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 1,
        ..InferenceLimits::default()
    };
    let session_spec = ModelSessionSpec::new(
        ModelSessionBinding::new(model.clone(), digest('e')),
        limits,
        32,
    )
    .unwrap();
    let pool = ModelSessionPool::new(
        DevicePreference::Cpu,
        ModelSessionPoolPolicy::new(1, 32, 1, 1).unwrap(),
    )
    .unwrap();
    let session = pool
        .get_or_load(
            session_spec,
            &tokio_util::sync::CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(7_u32) },
        )
        .await
        .unwrap();
    let input = ExecutionDigest::token_ids(&[1, 2]);
    let plan = session
        .plan_microbatches(
            ExecutionBatchBinding::new(digest('a'), digest('2'), digest('3')).unwrap(),
            MicrobatchPolicy::new(1, 10_000, 0).unwrap(),
            vec![MicrobatchCandidate::new(
                digest('b'),
                input.byte_length,
                input.item_count,
                0,
                input.byte_length as u64,
                0,
            )
            .unwrap()],
        )
        .unwrap();
    let cancellation = tokio_util::sync::CancellationToken::new();
    let execution = session
        .begin_microbatch(&plan, 0, &cancellation)
        .await
        .unwrap();
    assert_eq!(session.runtime().admission_snapshot().active, 1);
    assert!(session
        .runtime()
        .begin_microbatch(&plan, 0, &cancellation)
        .await
        .is_err());

    assert!(execution
        .receipt(
            ModelIdentity::new("wrong", "revision-1", digest('a')),
            input.clone(),
            ExecutionDigest::token_ids(&[9]),
        )
        .is_err());
    assert!(execution
        .receipt(
            model.clone(),
            ExecutionDigest::token_ids(&[1]),
            ExecutionDigest::token_ids(&[9]),
        )
        .is_err());
    let receipt = execution
        .receipt(model, input, ExecutionDigest::token_ids(&[9]))
        .unwrap();
    assert_eq!(receipt.schema, ExecutionReceipt::MICROBATCH_SCHEMA);
    assert!(receipt.accelerator.is_none());
    let evidence = receipt.microbatch.as_ref().unwrap();
    evidence.validate().unwrap();
    assert_eq!(
        evidence.session_declaration_sha256.as_deref(),
        Some(session.declaration_sha256())
    );
    assert_eq!(evidence.plan_sha256, plan.declaration_sha256);
    assert_eq!(evidence.batch_index, 0);
    assert_eq!(evidence.batch_count, 1);
    assert_eq!(evidence.slot_count, 1);
    assert!(!evidence.model_admission_queued);
    assert!(!evidence.device_admission_queued);
    let encoded = serde_json::to_string(&receipt).unwrap();
    assert!(!encoded.contains("availableBytes"));
    assert!(!encoded.contains("slotSha256"));
    drop(execution);
    assert_eq!(session.runtime().admission_snapshot().active, 0);
}

#[tokio::test]
async fn receipt_records_when_execution_waited_for_its_model_session() {
    let model = ModelIdentity::new("test-model", "revision-1", digest('a'));
    let limits = InferenceLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 1,
        ..InferenceLimits::default()
    };
    let pool = ModelSessionPool::new(
        DevicePreference::Cpu,
        ModelSessionPoolPolicy::new(1, 32, 2, 1).unwrap(),
    )
    .unwrap();
    let session = pool
        .get_or_load(
            ModelSessionSpec::new(
                ModelSessionBinding::new(model.clone(), digest('e')),
                limits,
                32,
            )
            .unwrap(),
            &tokio_util::sync::CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(7_u32) },
        )
        .await
        .unwrap();
    let input = ExecutionDigest::token_ids(&[1, 2]);
    let plan = session
        .plan_microbatches(
            ExecutionBatchBinding::new(digest('a'), digest('2'), digest('3')).unwrap(),
            MicrobatchPolicy::new(1, 10_000, 0).unwrap(),
            vec![MicrobatchCandidate::new(
                digest('b'),
                input.byte_length,
                input.item_count,
                0,
                input.byte_length as u64,
                0,
            )
            .unwrap()],
        )
        .unwrap();
    let active = session
        .begin_microbatch(&plan, 0, &tokio_util::sync::CancellationToken::new())
        .await
        .unwrap();
    let waiting = tokio::spawn({
        let session = session.clone();
        let plan = plan.clone();
        let model = model.clone();
        let input = input.clone();
        async move {
            let execution = session
                .begin_microbatch(&plan, 0, &tokio_util::sync::CancellationToken::new())
                .await?;
            execution.receipt(model, input, ExecutionDigest::token_ids(&[9]))
        }
    });
    wait_for(|| session.runtime().admission_snapshot().waiting == 1).await;
    drop(active);

    let receipt = waiting.await.unwrap().unwrap();
    let evidence = receipt.microbatch.unwrap();
    assert!(evidence.model_admission_queued);
    assert!(!evidence.device_admission_queued);
}

#[tokio::test]
async fn receipt_records_when_execution_waited_for_the_shared_device() {
    let first_model = ModelIdentity::new("first-model", "revision-1", digest('a'));
    let second_model = ModelIdentity::new("second-model", "revision-1", digest('c'));
    let limits = InferenceLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 1,
        ..InferenceLimits::default()
    };
    let pool = ModelSessionPool::new(
        DevicePreference::Cpu,
        ModelSessionPoolPolicy::new(2, 64, 1, 1).unwrap(),
    )
    .unwrap();
    let first = pool
        .get_or_load(
            ModelSessionSpec::new(
                ModelSessionBinding::new(first_model.clone(), digest('e')),
                limits.clone(),
                32,
            )
            .unwrap(),
            &tokio_util::sync::CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(1_u32) },
        )
        .await
        .unwrap();
    let second = pool
        .get_or_load(
            ModelSessionSpec::new(
                ModelSessionBinding::new(second_model.clone(), digest('f')),
                limits,
                32,
            )
            .unwrap(),
            &tokio_util::sync::CancellationToken::new(),
            |_runtime, _cancellation| async { Ok(2_u32) },
        )
        .await
        .unwrap();
    let first_input = ExecutionDigest::token_ids(&[1]);
    let second_input = ExecutionDigest::token_ids(&[2]);
    let first_plan = first
        .plan_microbatches(
            ExecutionBatchBinding::new(digest('a'), digest('2'), digest('3')).unwrap(),
            MicrobatchPolicy::new(1, 10_000, 0).unwrap(),
            vec![MicrobatchCandidate::new(
                digest('b'),
                first_input.byte_length,
                first_input.item_count,
                0,
                first_input.byte_length as u64,
                0,
            )
            .unwrap()],
        )
        .unwrap();
    let second_plan = second
        .plan_microbatches(
            ExecutionBatchBinding::new(digest('c'), digest('2'), digest('3')).unwrap(),
            MicrobatchPolicy::new(1, 10_000, 0).unwrap(),
            vec![MicrobatchCandidate::new(
                digest('d'),
                second_input.byte_length,
                second_input.item_count,
                0,
                second_input.byte_length as u64,
                0,
            )
            .unwrap()],
        )
        .unwrap();
    let active = first
        .begin_microbatch(&first_plan, 0, &tokio_util::sync::CancellationToken::new())
        .await
        .unwrap();
    let waiting = tokio::spawn({
        let second = second.clone();
        let second_model = second_model.clone();
        let second_input = second_input.clone();
        async move {
            let execution = second
                .begin_microbatch(&second_plan, 0, &tokio_util::sync::CancellationToken::new())
                .await?;
            execution.receipt(second_model, second_input, ExecutionDigest::token_ids(&[9]))
        }
    });
    wait_for(|| pool.snapshot().device_admission.waiting == 1).await;
    drop(active);

    let receipt = waiting.await.unwrap().unwrap();
    let evidence = receipt.microbatch.unwrap();
    assert!(!evidence.model_admission_queued);
    assert!(evidence.device_admission_queued);
}
