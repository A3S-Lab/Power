use tokio_util::sync::CancellationToken;

use super::{
    DevicePreference, EmbeddedRuntime, ExecutionBatchBinding, ExecutionBatchMemberBinding,
    ExecutionBatchMemberSpec, ExecutionBatchRowOutcome, ExecutionBatchRowSpec, ExecutionDigest,
    InferenceLimits,
};

fn digest(byte: char) -> String {
    byte.to_string().repeat(64)
}

fn limits() -> InferenceLimits {
    InferenceLimits {
        max_concurrent_requests: 3,
        max_state_bytes: 512,
        max_input_bytes: 256,
        max_tensor_elements: 64,
        max_graph_nodes: 8,
        max_context_tokens: 64,
        max_generated_tokens: 16,
        ..InferenceLimits::default()
    }
}

fn batch_binding() -> ExecutionBatchBinding {
    ExecutionBatchBinding::new(digest('1'), digest('2'), digest('3')).unwrap()
}

fn member(
    limits: &InferenceLimits,
    request: &[u8],
    state: &[u8],
    position: usize,
    state_bytes: u64,
) -> (ExecutionBatchMemberBinding, ExecutionBatchMemberSpec) {
    let binding = ExecutionBatchMemberBinding::for_identifiers(request, state, limits).unwrap();
    let spec = ExecutionBatchMemberSpec::new(binding.clone(), position, 0, 8, state_bytes);
    (binding, spec)
}

fn row(
    member: &ExecutionBatchMemberBinding,
    position: usize,
    values: &[u32],
) -> ExecutionBatchRowSpec {
    ExecutionBatchRowSpec::new(
        member.member_id_sha256(),
        position,
        vec![1, values.len()],
        ExecutionDigest::token_ids(values),
    )
}

fn continuing(
    member: &ExecutionBatchMemberBinding,
    next_position: usize,
    next_generated: usize,
    next_state_bytes: u64,
    value: u32,
) -> ExecutionBatchRowOutcome {
    ExecutionBatchRowOutcome::continuing(
        member.member_id_sha256(),
        next_position,
        next_generated,
        next_state_bytes,
        ExecutionDigest::token_ids(&[value]),
    )
}

fn completed(
    member: &ExecutionBatchMemberBinding,
    next_position: usize,
    next_generated: usize,
    next_state_bytes: u64,
    value: u32,
) -> ExecutionBatchRowOutcome {
    ExecutionBatchRowOutcome::completed(
        member.member_id_sha256(),
        next_position,
        next_generated,
        next_state_bytes,
        ExecutionDigest::token_ids(&[value]),
    )
}

#[tokio::test]
async fn batch_waiting_admission_reuses_the_runtime_queue_and_cancellation() {
    let limits = InferenceLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 1,
        ..limits()
    };
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let held = runtime.begin(&CancellationToken::new()).unwrap();
    let cancellation = CancellationToken::new();
    let (_, waiting_spec) = member(&limits, b"waiting", b"waiting-state", 0, 16);
    let waiter = tokio::spawn({
        let batch = batch.clone();
        let cancellation = cancellation.clone();
        async move { batch.admit_wait(waiting_spec, cancellation).await }
    });
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while runtime.admission_snapshot().waiting != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    cancellation.cancel();
    assert!(waiter.await.unwrap().is_err());
    assert_eq!(batch.active_member_count(), 0);
    assert_eq!(runtime.admission_snapshot().waiting, 0);
    drop(held);
    assert_eq!(runtime.admission_snapshot().active, 0);
}

#[test]
fn continuous_ragged_rounds_are_canonical_and_new_members_join_next_round() {
    let limits = limits();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation_a = CancellationToken::new();
    let cancellation_b = CancellationToken::new();
    let cancellation_c = CancellationToken::new();
    let (member_a, spec_a) = member(&limits, b"request-a", b"state-a", 4, 32);
    let (member_b, spec_b) = member(&limits, b"request-b", b"state-b", 9, 48);
    let (member_c, spec_c) = member(&limits, b"request-c", b"state-c", 2, 24);

    batch
        .admit(
            spec_a,
            runtime.begin(&cancellation_a).unwrap(),
            cancellation_a,
        )
        .unwrap();
    batch
        .admit(
            spec_b,
            runtime.begin(&cancellation_b).unwrap(),
            cancellation_b,
        )
        .unwrap();

    // Caller order is deliberately reversed and row shapes differ. Power
    // restores admission order without interpreting either row.
    let step = batch
        .begin_step(vec![
            row(&member_b, 9, &[20, 21, 22]),
            row(&member_a, 4, &[10, 11]),
        ])
        .unwrap();
    assert_eq!(step.rows().len(), 2);
    assert_eq!(
        step.rows()[0].member_id_sha256(),
        member_a.member_id_sha256()
    );
    assert_eq!(
        step.rows()[1].member_id_sha256(),
        member_b.member_id_sha256()
    );
    assert_eq!(step.rows()[0].shape(), &[1, 2]);
    assert_eq!(step.rows()[1].shape(), &[1, 3]);
    assert_eq!(step.rows()[0].canonical_index(), 0);
    assert_eq!(step.rows()[1].canonical_index(), 1);
    assert!(step.rows()[0].permit().belongs_to(&runtime));

    // Continuous admission is allowed while the current immutable roster is
    // executing. The new member cannot appear until the following step.
    batch
        .admit(
            spec_c,
            runtime.begin(&cancellation_c).unwrap(),
            cancellation_c,
        )
        .unwrap();
    assert_eq!(step.rows().len(), 2);

    let first = step
        .commit(vec![
            continuing(&member_b, 10, 1, 52, 201),
            continuing(&member_a, 5, 1, 36, 101),
        ])
        .unwrap();
    assert_eq!(first.step, 0);
    assert_eq!(first.row_count, 2);
    assert_eq!(first.continued_members, 2);
    assert_eq!(first.completed_members, 0);
    assert_eq!(first.cancelled_members, 0);
    assert_eq!(first.active_members_after, 3);
    assert_eq!(first.input_sha256.len(), 64);
    assert_eq!(first.output_sha256.len(), 64);

    let second_step = batch
        .begin_step(vec![
            row(&member_c, 2, &[30]),
            row(&member_a, 5, &[12]),
            row(&member_b, 10, &[23, 24]),
        ])
        .unwrap();
    assert_eq!(
        second_step
            .rows()
            .iter()
            .map(|row| row.member_id_sha256())
            .collect::<Vec<_>>(),
        vec![
            member_a.member_id_sha256(),
            member_b.member_id_sha256(),
            member_c.member_id_sha256(),
        ]
    );
    let second = second_step
        .commit(vec![
            completed(&member_c, 3, 1, 28, 301),
            completed(&member_b, 11, 2, 56, 202),
            completed(&member_a, 6, 2, 40, 102),
        ])
        .unwrap();
    assert_eq!(second.completed_members, 3);
    assert_eq!(second.active_members_after, 0);

    let evidence = batch.finish().unwrap();
    assert_eq!(evidence.schema, "a3s.power.execution-batch-lifecycle.v1");
    assert_eq!(evidence.admitted_members, 3);
    assert_eq!(evidence.completed_members, 3);
    assert_eq!(evidence.cancelled_members, 0);
    assert_eq!(evidence.committed_steps, 2);
    assert_eq!(evidence.processed_rows, 5);
    assert_eq!(evidence.max_active_members, 3);
    assert_eq!(evidence.peak_state_bytes, 124);
    assert_eq!(evidence.declaration_sha256, batch.declaration_sha256());
    assert_eq!(evidence.transcript_sha256.len(), 64);
}

#[test]
fn cancelled_rows_are_discarded_without_rolling_back_other_members() {
    let mut limits = limits();
    limits.max_concurrent_requests = 2;
    limits.max_state_bytes = 80;
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation_a = CancellationToken::new();
    let cancellation_b = CancellationToken::new();
    let (member_a, spec_a) = member(&limits, b"request-a", b"reusable-state", 3, 40);
    let (member_b, spec_b) = member(&limits, b"request-b", b"state-b", 7, 40);
    batch
        .admit(
            spec_a,
            runtime.begin(&cancellation_a).unwrap(),
            cancellation_a.clone(),
        )
        .unwrap();
    batch
        .admit(
            spec_b,
            runtime.begin(&cancellation_b).unwrap(),
            cancellation_b,
        )
        .unwrap();

    let step = batch
        .begin_step(vec![row(&member_a, 3, &[1]), row(&member_b, 7, &[2])])
        .unwrap();
    cancellation_a.cancel();
    let evidence = step
        .commit(vec![continuing(&member_b, 8, 1, 40, 22)])
        .unwrap();
    assert_eq!(evidence.cancelled_members, 1);
    assert_eq!(evidence.continued_members, 1);
    assert_eq!(batch.active_member_count(), 1);

    // Cancellation released both the admission slot and the opaque state
    // identity, so a later request can safely reuse that model-owned slot.
    let replacement_cancellation = CancellationToken::new();
    let (replacement, replacement_spec) =
        member(&limits, b"request-replacement", b"reusable-state", 0, 40);
    batch
        .admit(
            replacement_spec,
            runtime.begin(&replacement_cancellation).unwrap(),
            replacement_cancellation,
        )
        .unwrap();
    let step = batch
        .begin_step(vec![row(&replacement, 0, &[3]), row(&member_b, 8, &[4])])
        .unwrap();
    step.commit(vec![
        completed(&replacement, 1, 1, 40, 33),
        completed(&member_b, 9, 2, 40, 44),
    ])
    .unwrap();
    let final_evidence = batch.finish().unwrap();
    assert_eq!(final_evidence.admitted_members, 3);
    assert_eq!(final_evidence.completed_members, 2);
    assert_eq!(final_evidence.cancelled_members, 1);
}

#[test]
fn lifecycle_reuses_runtime_admission_and_releases_it_on_completion() {
    let mut limits = limits();
    limits.max_concurrent_requests = 1;
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation = CancellationToken::new();
    let (member, spec) = member(&limits, b"request", b"state", 0, 16);
    batch
        .admit(
            spec,
            runtime.begin(&cancellation).unwrap(),
            cancellation.clone(),
        )
        .unwrap();
    assert!(runtime.begin(&CancellationToken::new()).is_err());

    batch
        .begin_step(vec![row(&member, 0, &[1])])
        .unwrap()
        .commit(vec![completed(&member, 1, 1, 16, 2)])
        .unwrap();
    assert!(runtime.begin(&CancellationToken::new()).is_ok());
    batch.finish().unwrap();
}

#[test]
fn explicit_cancellation_and_dropped_steps_are_safe_boundaries() {
    let mut limits = limits();
    limits.max_concurrent_requests = 2;
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation_a = CancellationToken::new();
    let cancellation_b = CancellationToken::new();
    let (member_a, spec_a) = member(&limits, b"request-a", b"state-a", 0, 16);
    let (member_b, spec_b) = member(&limits, b"request-b", b"state-b", 0, 16);
    batch
        .admit(
            spec_a,
            runtime.begin(&cancellation_a).unwrap(),
            cancellation_a,
        )
        .unwrap();
    batch
        .admit(
            spec_b,
            runtime.begin(&cancellation_b).unwrap(),
            cancellation_b.clone(),
        )
        .unwrap();

    let step = batch
        .begin_step(vec![row(&member_a, 0, &[1]), row(&member_b, 0, &[2])])
        .unwrap();
    cancellation_b.cancel();
    drop(step);
    assert_eq!(batch.active_member_count(), 1);
    let snapshot = batch.active_members();
    assert_eq!(snapshot[0].member_id_sha256(), member_a.member_id_sha256());
    assert_eq!(snapshot[0].position(), 0);

    batch.cancel(member_a.member_id_sha256()).unwrap();
    assert_eq!(batch.active_member_count(), 0);
    let evidence = batch.finish().unwrap();
    assert_eq!(evidence.cancelled_members, 2);
    assert_eq!(evidence.committed_steps, 0);
}

#[test]
fn evidence_and_debug_output_do_not_expose_member_or_state_identifiers() {
    let mut limits = limits();
    limits.max_concurrent_requests = 1;
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation = CancellationToken::new();
    let plaintext_request = b"tenant/customer/document/request-42";
    let plaintext_state = b"private-kv-slot-7";
    let (member, spec) = member(&limits, plaintext_request, plaintext_state, 11, 32);
    batch
        .admit(spec, runtime.begin(&cancellation).unwrap(), cancellation)
        .unwrap();

    let step = batch.begin_step(vec![row(&member, 11, &[7])]).unwrap();
    let debug = format!("{batch:?} {step:?} {:?}", step.rows()[0]);
    assert!(!debug.contains("tenant"));
    assert!(!debug.contains("private-kv"));
    assert!(!debug.contains(member.member_id_sha256()));
    assert!(!debug.contains(member.state_id_sha256()));
    let step_evidence = step.commit(vec![completed(&member, 12, 1, 36, 8)]).unwrap();
    let final_evidence = batch.finish().unwrap();
    let serialized = serde_json::to_string(&(step_evidence, final_evidence)).unwrap();
    assert!(!serialized.contains("tenant"));
    assert!(!serialized.contains("private-kv"));
    assert!(!serialized.contains(member.member_id_sha256()));
    assert!(!serialized.contains(member.state_id_sha256()));
}

#[test]
fn execution_batch_public_guards_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExecutionBatchBinding>();
    assert_send_sync::<ExecutionBatchMemberBinding>();
    assert_send_sync::<ExecutionBatchMemberSpec>();
    assert_send_sync::<ExecutionBatchRowSpec>();
    assert_send_sync::<ExecutionBatchRowOutcome>();
    assert_send_sync::<super::ExecutionBatchLifecycle>();
    assert_send_sync::<super::ExecutionBatchStep>();
    assert_send_sync::<super::ExecutionBatchStepEvidence>();
    assert_send_sync::<super::ExecutionBatchLifecycleEvidence>();
}
