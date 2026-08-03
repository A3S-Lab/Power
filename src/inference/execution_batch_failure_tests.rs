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
        max_concurrent_requests: 2,
        max_state_bytes: 64,
        max_input_bytes: 32,
        max_tensor_elements: 6,
        max_graph_nodes: 2,
        max_context_tokens: 16,
        max_generated_tokens: 4,
        ..InferenceLimits::default()
    }
}

fn batch_binding() -> ExecutionBatchBinding {
    ExecutionBatchBinding::new(digest('1'), digest('2'), digest('3')).unwrap()
}

fn member_binding(
    limits: &InferenceLimits,
    request: &[u8],
    state: &[u8],
) -> ExecutionBatchMemberBinding {
    ExecutionBatchMemberBinding::for_identifiers(request, state, limits).unwrap()
}

fn spec(
    member: ExecutionBatchMemberBinding,
    position: usize,
    generated: usize,
    maximum: usize,
    state_bytes: u64,
) -> ExecutionBatchMemberSpec {
    ExecutionBatchMemberSpec::new(member, position, generated, maximum, state_bytes)
}

fn row(
    member: &ExecutionBatchMemberBinding,
    position: usize,
    shape: Vec<usize>,
    values: &[u32],
) -> ExecutionBatchRowSpec {
    ExecutionBatchRowSpec::new(
        member.member_id_sha256(),
        position,
        shape,
        ExecutionDigest::token_ids(values),
    )
}

fn continuing(
    member: &ExecutionBatchMemberBinding,
    position: usize,
    generated: usize,
    state_bytes: u64,
) -> ExecutionBatchRowOutcome {
    ExecutionBatchRowOutcome::continuing(
        member.member_id_sha256(),
        position,
        generated,
        state_bytes,
        ExecutionDigest::token_ids(&[1]),
    )
}

#[test]
fn binding_and_member_admission_fail_closed() {
    assert!(ExecutionBatchBinding::new("bad", digest('2'), digest('3')).is_err());
    assert!(ExecutionBatchMemberBinding::new("bad", digest('4')).is_err());

    let limits = limits();
    assert!(ExecutionBatchMemberBinding::for_identifiers(b"", b"state", &limits).is_err());
    assert!(ExecutionBatchMemberBinding::for_identifiers(b"request", b"", &limits).is_err());
    let oversized = vec![0_u8; limits.max_graph_name_bytes + 1];
    assert!(ExecutionBatchMemberBinding::for_identifiers(&oversized, b"state", &limits).is_err());

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let foreign = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let member_a = member_binding(&limits, b"a", b"state-a");
    let cancellation = CancellationToken::new();
    assert!(batch
        .admit(
            spec(member_a.clone(), 0, 0, 2, 16),
            foreign.begin(&cancellation).unwrap(),
            cancellation.clone(),
        )
        .is_err());

    let permit = runtime.begin(&cancellation).unwrap();
    batch
        .admit(
            spec(member_a.clone(), 0, 0, 2, 16),
            permit.clone(),
            cancellation.clone(),
        )
        .unwrap();
    assert!(batch
        .admit(
            spec(member_binding(&limits, b"b", b"state-b"), 0, 0, 2, 16,),
            permit,
            CancellationToken::new(),
        )
        .is_err());

    let permit_b = runtime.begin(&CancellationToken::new()).unwrap();
    assert!(batch
        .admit(
            spec(member_a.clone(), 0, 0, 2, 16),
            permit_b,
            CancellationToken::new(),
        )
        .is_err());

    // A different member may not alias the same opaque state identity.
    let same_state =
        ExecutionBatchMemberBinding::new(digest('a'), member_a.state_id_sha256().to_string())
            .unwrap();
    let permit_b = runtime.begin(&CancellationToken::new()).unwrap();
    assert!(batch
        .admit(
            spec(same_state, 0, 0, 2, 16),
            permit_b,
            CancellationToken::new(),
        )
        .is_err());
}

#[test]
fn admission_bounds_context_generation_state_and_cancellation() {
    let limits = limits();

    let cases = [
        spec(
            member_binding(&limits, b"context", b"state-context"),
            limits.max_context_tokens + 1,
            0,
            1,
            1,
        ),
        spec(
            member_binding(&limits, b"generated", b"state-generated"),
            0,
            2,
            1,
            1,
        ),
        spec(
            member_binding(&limits, b"maximum", b"state-maximum"),
            0,
            0,
            limits.max_generated_tokens + 1,
            1,
        ),
        spec(
            member_binding(&limits, b"state", b"state-state"),
            0,
            0,
            1,
            limits.max_state_bytes + 1,
        ),
    ];
    for invalid in cases {
        let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
        let batch = runtime.execution_batch(batch_binding()).unwrap();
        let cancellation = CancellationToken::new();
        let permit = runtime.begin(&cancellation).unwrap();
        assert!(batch.admit(invalid, permit, cancellation).is_err());
    }

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancelled = CancellationToken::new();
    let permit = runtime.begin(&cancelled).unwrap();
    cancelled.cancel();
    assert!(batch
        .admit(
            spec(
                member_binding(&limits, b"cancelled", b"state-cancelled"),
                0,
                0,
                1,
                1,
            ),
            permit,
            cancelled,
        )
        .is_err());

    let cancellation_a = CancellationToken::new();
    let member_a = member_binding(&limits, b"a", b"state-a");
    batch
        .admit(
            spec(member_a, 0, 0, 2, 40),
            runtime.begin(&cancellation_a).unwrap(),
            cancellation_a,
        )
        .unwrap();
    let cancellation_b = CancellationToken::new();
    assert!(batch
        .admit(
            spec(member_binding(&limits, b"b", b"state-b"), 0, 0, 2, 25,),
            runtime.begin(&cancellation_b).unwrap(),
            cancellation_b,
        )
        .is_err());
}

#[test]
fn step_roster_is_exact_and_rows_are_bounded_before_launch() {
    let limits = limits();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation_a = CancellationToken::new();
    let cancellation_b = CancellationToken::new();
    let member_a = member_binding(&limits, b"a", b"state-a");
    let member_b = member_binding(&limits, b"b", b"state-b");
    batch
        .admit(
            spec(member_a.clone(), 3, 0, 3, 16),
            runtime.begin(&cancellation_a).unwrap(),
            cancellation_a,
        )
        .unwrap();
    batch
        .admit(
            spec(member_b.clone(), 7, 0, 3, 16),
            runtime.begin(&cancellation_b).unwrap(),
            cancellation_b,
        )
        .unwrap();

    assert!(batch
        .begin_step(vec![row(&member_a, 3, vec![1], &[1])])
        .is_err());
    assert!(batch
        .begin_step(vec![
            row(&member_a, 3, vec![1], &[1]),
            row(&member_a, 3, vec![1], &[1]),
        ])
        .is_err());
    assert!(batch
        .begin_step(vec![
            row(&member_a, 2, vec![1], &[1]),
            row(&member_b, 7, vec![1], &[2]),
        ])
        .is_err());
    assert!(batch
        .begin_step(vec![
            row(&member_a, 3, vec![1, 1, 1], &[1]),
            row(&member_b, 7, vec![1], &[2]),
        ])
        .is_err());

    let mut bad_digest = ExecutionDigest::token_ids(&[1]);
    bad_digest.sha256 = "bad".to_string();
    assert!(batch
        .begin_step(vec![
            ExecutionBatchRowSpec::new(member_a.member_id_sha256(), 3, vec![1], bad_digest),
            row(&member_b, 7, vec![1], &[2]),
        ])
        .is_err());

    // Both rows are individually bounded but overflow the aggregate tensor
    // and input limits.
    assert!(batch
        .begin_step(vec![
            row(&member_a, 3, vec![4], &[1, 2, 3, 4]),
            row(&member_b, 7, vec![4], &[5, 6, 7, 8]),
        ])
        .is_err());

    let mut input_limited = limits.clone();
    input_limited.max_input_bytes = 12;
    input_limited.max_tensor_elements = 8;
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, input_limited.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let ca = CancellationToken::new();
    let cb = CancellationToken::new();
    let a = member_binding(&input_limited, b"input-a", b"input-state-a");
    let b = member_binding(&input_limited, b"input-b", b"input-state-b");
    batch
        .admit(spec(a.clone(), 0, 0, 2, 8), runtime.begin(&ca).unwrap(), ca)
        .unwrap();
    batch
        .admit(spec(b.clone(), 0, 0, 2, 8), runtime.begin(&cb).unwrap(), cb)
        .unwrap();
    assert!(batch
        .begin_step(vec![
            row(&a, 0, vec![2], &[1, 2]),
            row(&b, 0, vec![2], &[3, 4]),
        ])
        .is_err());
}

#[test]
fn failed_commit_is_atomic_and_the_same_positions_can_retry() {
    let limits = limits();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation_a = CancellationToken::new();
    let cancellation_b = CancellationToken::new();
    let member_a = member_binding(&limits, b"a", b"state-a");
    let member_b = member_binding(&limits, b"b", b"state-b");
    batch
        .admit(
            spec(member_a.clone(), 3, 0, 3, 24),
            runtime.begin(&cancellation_a).unwrap(),
            cancellation_a,
        )
        .unwrap();
    batch
        .admit(
            spec(member_b.clone(), 7, 0, 3, 24),
            runtime.begin(&cancellation_b).unwrap(),
            cancellation_b,
        )
        .unwrap();

    let step = batch
        .begin_step(vec![
            row(&member_a, 3, vec![1], &[1]),
            row(&member_b, 7, vec![1], &[2]),
        ])
        .unwrap();
    assert!(step
        .commit(vec![
            continuing(&member_a, 4, 1, 48),
            continuing(&member_b, 8, 1, 24),
        ])
        .is_err());

    let snapshots = batch.active_members();
    assert_eq!(snapshots[0].position(), 3);
    assert_eq!(snapshots[0].generated_items(), 0);
    assert_eq!(snapshots[0].state_bytes(), 24);
    assert_eq!(snapshots[1].position(), 7);

    let step = batch
        .begin_step(vec![
            row(&member_a, 3, vec![1], &[1]),
            row(&member_b, 7, vec![1], &[2]),
        ])
        .unwrap();
    let mut bad = continuing(&member_a, 4, 1, 24);
    bad.output.sha256 = "bad".to_string();
    assert!(step
        .commit(vec![bad, continuing(&member_b, 8, 1, 24)])
        .is_err());

    batch
        .begin_step(vec![
            row(&member_a, 3, vec![1], &[1]),
            row(&member_b, 7, vec![1], &[2]),
        ])
        .unwrap()
        .commit(vec![
            ExecutionBatchRowOutcome::completed(
                member_a.member_id_sha256(),
                4,
                1,
                24,
                ExecutionDigest::token_ids(&[3]),
            ),
            ExecutionBatchRowOutcome::completed(
                member_b.member_id_sha256(),
                8,
                1,
                24,
                ExecutionDigest::token_ids(&[4]),
            ),
        ])
        .unwrap();
    batch.finish().unwrap();
}

#[test]
fn outcomes_must_advance_monotonically_and_respect_member_caps() {
    let limits = limits();
    for invalid_outcome in [
        (3, 1, 16),
        (4, 0, 16),
        (4, 3, 16),
        (limits.max_context_tokens + 1, 1, 16),
        (4, 1, limits.max_state_bytes + 1),
    ] {
        let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
        let batch = runtime.execution_batch(batch_binding()).unwrap();
        let cancellation = CancellationToken::new();
        let member = member_binding(&limits, b"request", b"state");
        batch
            .admit(
                spec(member.clone(), 3, 0, 2, 16),
                runtime.begin(&cancellation).unwrap(),
                cancellation,
            )
            .unwrap();
        let step = batch
            .begin_step(vec![row(&member, 3, vec![1], &[1])])
            .unwrap();
        assert!(step
            .commit(vec![continuing(
                &member,
                invalid_outcome.0,
                invalid_outcome.1,
                invalid_outcome.2,
            )])
            .is_err());
    }

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation = CancellationToken::new();
    let member = member_binding(&limits, b"request", b"state");
    batch
        .admit(
            spec(member.clone(), 3, 0, 1, 16),
            runtime.begin(&cancellation).unwrap(),
            cancellation,
        )
        .unwrap();
    let step = batch
        .begin_step(vec![row(&member, 3, vec![1], &[1])])
        .unwrap();
    assert!(step.commit(vec![continuing(&member, 4, 1, 16)]).is_err());
}

#[test]
fn only_one_step_can_be_in_flight_and_finish_requires_a_terminal_lifecycle() {
    let limits = limits();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let batch = runtime.execution_batch(batch_binding()).unwrap();
    let cancellation = CancellationToken::new();
    let member = member_binding(&limits, b"request", b"state");
    batch
        .admit(
            spec(member.clone(), 0, 0, 2, 16),
            runtime.begin(&cancellation).unwrap(),
            cancellation,
        )
        .unwrap();
    let step = batch
        .begin_step(vec![row(&member, 0, vec![1], &[1])])
        .unwrap();
    assert!(batch
        .begin_step(vec![row(&member, 0, vec![1], &[1])])
        .is_err());
    assert!(batch.finish().is_err());
    drop(step);
    assert!(batch.finish().is_err());
    batch.cancel(member.member_id_sha256()).unwrap();
    assert!(batch.finish().is_ok());
    assert!(batch.finish().is_err());
    assert!(batch.cancel(member.member_id_sha256()).is_err());
}
