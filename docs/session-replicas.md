# Model-Neutral Session Replicas

`ModelSessionPool<T>` can retain a finite set of independently mutable backend
contexts for one exact model identity. This is a runtime resource contract, not
a language-model or Qwen execution path. Power never interprets `T`, selects a
model architecture, or assigns model semantics to a replica.

## Contract

| Constraint | Enforced behavior |
| --- | --- |
| Model and execution bytes must be exact | The entry key binds family, revision, weight SHA-256, and model-owned execution SHA-256. The family is opaque identity rather than runtime dispatch. |
| Mutable state cannot be shared accidentally | `ModelSessionReplica<T>` is a non-cloneable exclusive lease. Stateful backends can load `Mutex<Context>` or another synchronization primitive as `T`. One exact entry cannot mix shared `ModelSession<T>` access with exclusive replica access. |
| Replica concurrency is finite | `max_replicas_per_session` defaults to one, must be in `1..=256`, and backs a bounded admission controller. The per-entry waiting limit comes from `InferenceLimits::max_queued_requests`. |
| Residency cannot grow after admission | `ModelSessionSpec::resident_bytes` is the declared cost of one replica. Power checks and reserves `resident_bytes * max_replicas_per_session` before invoking any loader. |
| Device work has one authority | Every replica receives the same pool-created `EmbeddedRuntime`, which contains the same physical-device admission controller. A loader cannot receive a slot-specific scheduler. |
| Initialization remains lazy | Each finite slot has one cancellation-safe initialization cell. A released initialized slot is reused without calling its loader again. |
| Failure cannot strand capacity | Cancellation, loader error, or future drop returns the replica permit and free-list entry. If no replica initialized, the pool also removes the empty entry and releases its byte reservation. |
| Queue time is caller-bounded | `acquire_replica_until` accepts one monotonic `tokio::time::Instant`. Expiry returns a typed error, returns all acquired capacity, and never becomes a wall-clock timestamp. |
| Operations remain private | The loader receives no ordinal. Debug output redacts model identities, the declaration digest contains no ordinal, and snapshots report only aggregate reserved, ready, leased, and waiting counts. |

Language decoders, vision encoders, OCR graphs, embedding models, and multimodal
pipelines therefore use the same mechanism. Model crates still own context
layout, KV or recurrent state, preprocessing, kernels, and all state
transitions.

## Lifecycle

```text
exact model + execution + limits + per-replica bytes
                         |
                         v
          reserve worst-case replica residency
                         |
                         v
           enter bounded per-entry admission
                         |
                         v
          lease one anonymous finite replica slot
                         |
                         v
      initialize once or reuse its ready backend state
                         |
                         v
       execute through the shared runtime/device gate
                         |
                         v
       drop lease -> return slot and queue capacity
```

## Embedded use

Configure replica mode explicitly:

```rust
let policy = ModelSessionPoolPolicy::new(
    4,                    // exact session identities
    24 * 1024 * 1024 * 1024,
    2,                    // physical-device executions
    16,                   // device waiters
)?
.with_max_replicas_per_session(3)?;
```

Then use `ModelSessionPool::acquire_replica` instead of `get_or_load`. For a
backend context that needs `&mut self` across an async call, load it inside a
`tokio::sync::Mutex` and keep the non-cloneable replica lease for the complete
request:

```rust,ignore
let replica = pool
    .acquire_replica(spec, cancellation, |runtime, cancellation| async move {
        let context = BackendContext::load(runtime, cancellation).await?;
        Ok(tokio::sync::Mutex::new(context))
    })
    .await?;

let mut context = replica.value().lock().await;
context.infer(input, cancellation).await?;
```

Use `acquire_replica_until` when queueing must have a finite latency budget:

```rust,ignore
let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(50);
let replica = pool
    .acquire_replica_until(spec, cancellation, deadline, load_context)
    .await?;
```

The deadline covers admission only. Once a replica is admitted, the owning
model crate retains execution and cancellation policy. The same absolute
deadline is used by `EmbeddedRuntime::begin_wait_until` for its model and
physical-device queues, so time spent at the first gate cannot reset the budget
at the second gate. `AdmissionSnapshot::deadline_expirations` and
`ModelSessionPoolSnapshot::expired_replica_requests` are cumulative aggregate
evidence; neither exposes a deadline value, request bytes, or replica identity.

The configured replica count is part of the replica declaration SHA-256. A
deployment changing that count or its resulting total reservation therefore
changes the evidence identity. JSON policies created before replica support
remain compatible and deserialize with one replica.

`get_or_load` remains the cloneable shared-session API and is accepted only
when the pool maximum is one. Whichever access style first registers an exact
identity owns that entry; attempting to use the other style fails explicitly.

## Reproduce the contract tests

From the Power repository root:

```bash
cargo test --locked --no-default-features --features embedded-inference \
  --lib inference::session_replica_tests
cargo test --locked --no-default-features --features embedded-inference \
  --lib inference::session_pool_tests
cargo clippy --locked --no-default-features --features embedded-inference \
  --lib -- -D warnings
```

The focused tests cover independent lazy state, shared device admission,
worst-case byte reservation before loader execution, bounded policy parsing,
opaque language/vision/embedding identities, queue cancellation, aborted
initialization, monotonic expiry across sequential gates, persistent aggregate
expiry evidence, ready-slot reuse, shared/exclusive isolation, privacy-safe
debug output, and `Send + Sync` public leases.
