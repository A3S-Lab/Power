# Embedded Inference Architecture

A3S Power provides a model-neutral native Rust inference substrate. Model
architectures do not belong in Power: OCR networks belong in `a3s-ocr`, and
other product models remain in their owning crates. Power supplies the common
execution, placement, integrity, resource, and privacy mechanisms those model
crates share.

This design borrows systems ideas from
[A3S-Lab/colibri](https://github.com/A3S-Lab/colibri/tree/b085b48888a88d9a1c00b151a9979774b72cdbfd)
at pinned revision `b085b48888a88d9a1c00b151a9979774b72cdbfd`, while
retaining Power's TEE, encrypted-model, signature, attestation, resource-bound,
and receipt features. No Colibri source code is copied into Power.

## Ownership Boundary

| Concern | Owner |
| --- | --- |
| Tensor kernels, typed devices, admission, cancellation, limits | `a3s-power` |
| Static graph validation and reviewed operator execution | `a3s-power` |
| Storage/RAM/device weight placement and telemetry policy | `a3s-power` |
| Model family, topology, graph identity, and revision | Model crate |
| Tokenizer, preprocessing, postprocessing, and generation loop | Model crate |
| Product orchestration and document structure | Product crate |

Power therefore contains no PP-OCR, Unlimited-OCR, or other model assets,
tokenizers, conversion tools, or revision hashes.

## Runtime Shape

```text
model crate
  ├─ model-owned reviewed plans and native control flow
  ├─ one EmbeddedRuntime per model session
  └─ one ExecutionPermit per logical request
       │
       ├─ GraphExecutor ───────── validated dense/static graphs
       ├─ RoutedExpertBatch ───── exact batch union, no route changes
       └─ WeightHierarchy
            ├─ route coupling ─── private bounded hints, measured against truth
            ├─ device cache ───── bounded, typed device, exact dtype
            ├─ host cache ─────── per-layer LFRU/LRU + provenance-aware pins
            ├─ prefetch pool ───── bounded tasks and blocking workers
            ├─ staged batches ──── current-layer readiness + canonical order
            ├─ residency plan ──── replaceable and stably adapted atomic groups
            └─ SafeTensors ─────── verified mmap/positional weighted sources
```

A logical request holds one permit across every component graph. A multimodal
model must not create independent admission, device, hash, receipt, or cache
systems for its vision encoder, projector, dense layers, and routed experts.

## Colibri Ideas Adapted as Generic Mechanisms

- Storage, host RAM, and accelerator memory form one typed weight hierarchy.
  Placement changes latency only; tensor dtype and shape are checked after each
  transfer and are never silently converted.
- Routed weights load on demand. The default layer-local LFRU policy uses
  frequency as the primary signal and bounded recency as a tie-breaker, with
  periodic heat decay so a stale workload does not dominate forever. Plain LRU
  remains selectable. Both policies are bounded by entry caps and byte budgets.
- Explicit pins and residency-plan pins have separate provenance. A new hot-set
  plan can transactionally replace the prior plan at a request-safe boundary
  without releasing caller-owned pins; a failed replacement restores the
  complete prior cache and plan.
- `RoutedExpertBatch` unions repeated expert IDs across batch positions so each
  unique expert can be staged once. Original expert order, gate weight, and
  top-k selection remain intact.
- Explicitly paired, position-aligned route batches can teach a bounded
  cross-layer coupling table. For each position, Power sums raw learned
  co-occurrence counts over the current exact routed set, ranks target experts
  by score with expert ID as the deterministic tie-breaker, and returns a
  deduplicated batch union. The model crate alone maps those IDs to its tensor
  names and may pass the resulting requests to the existing prefetch pool.
  Predictions contain no gate values and cannot be used by Power as router
  selections.
- Route hints must be evaluated against the later actual target batch. Power
  reports exact predicted, actual, and matched selection counts rather than
  assuming speculative I/O helped. Learned relationships can overfit a
  workload and a low-recall hint can waste bandwidth, so recording and use are
  explicit instead of an automatic default.
- `start_prefetch` starts bounded blocking I/O immediately. The hierarchy caps
  active tasks and workers, unions duplicate requests, and propagates
  cancellation when the task is aborted or dropped. A model can prefetch layer
  N+1, compute layer N, then await the task, providing the one-layer-ahead
  overlap used by routed models.
- `start_staged_batch` adapts Colibri's current-layer pipeline without moving
  model execution into Power. It validates every non-empty atomic group,
  duplicate key, layer, tensor descriptor, item count, and byte total before
  touching cache heat. Fully resident groups are available synchronously;
  exact misses use the same prefetch admission, bounded Tokio blocking workers,
  source router, cache, per-key load serialization, and cancellation path.
  Early groups carry an implicit canonical input index, and final completion
  always restores the original order. Model crates compute into indexed output
  slots and retain their exact gate application and reduction order.
- Prefetch bookkeeping distinguishes a cache hit at prefetch time, a materialized
  weight later consumed by demand, and a materialized weight evicted unused.
  Aggregate telemetry reports useful and unused counts and bytes, allowing the
  policy to be evaluated end to end instead of treating every queued read as a
  win.
- Per-key load serialization prevents a demand load and a concurrent prefetch
  from materializing the same tensor twice.
- `plan_residency` converts model-supplied atomic weight groups and measured heat
  into a deterministic device/host/storage plan. It respects exact byte budgets
  and per-layer entry limits, never splits a group, and binds the plan to the
  weight digest, runtime device, and policy. Applying a plan reconciles the
  active plan transactionally while leaving manual pins intact.
- `plan_residency_adaptation` follows Colibri's lossless live re-pin principle
  at an explicit model-owned safe boundary. The default policy requires a
  challenger to exceed the incumbent by more than 25% plus four heat units and
  permits at most four replacements per pass. Power swaps only atomic groups
  with identical byte and per-layer entry footprints, so host/device ledgers
  remain exact even for model families whose groups are not uniform. Each
  ephemeral adaptation is bound to its active base plan; stale application
  fails before any cache mutation. Application reuses the transactional plan
  and pin path rather than creating another cache or residency mechanism.
- `EmbeddedRuntime::plan_residency_budget` discovers host memory through bounded
  native Linux, macOS, or Windows APIs and device memory through the selected
  CUDA or Metal handle. The caller supplies explicit fractions, reserves, caps,
  and host/device allocation order. The resulting plan is capped by
  `InferenceLimits::max_resident_weight_bytes`; Metal unified memory is counted
  once, and a failed or incomplete discovery fails closed instead of guessing.
  Discovery is opt-in and does not change the zero-cache default.
- `WeightStoreConfig` accepts a primary collection plus bounded, read-only
  replicas. Complete replicas must match the primary aggregate digest. An
  explicitly partial replica may contain a non-empty subset of primary
  SafeTensors files; every present relative file, byte length, SHA-256 digest,
  tensor name, dtype, shape, and byte count must match before mapping. A stable
  bandwidth-weighted hash selects only among sources that contain the requested
  tensor, and recoverable replica errors fall back to primary. This extends the
  existing `WeightStore` rather than creating a second model cache or integrity
  path.
- Every verified tensor has one exact source-file index, absolute byte range,
  dtype, shape, and byte count. `Mmap` remains the default. The opt-in buffered
  positional path does not retain a collection-wide mmap and uses a 1 MiB
  full-read loop with interruption, cancellation, overflow, truncation, and
  honest short-read handling. Its plaintext buffers zeroize on drop.
- Direct positional reads reuse that index, source router, primary fallback,
  demand path, and prefetch path. Linux uses aligned `O_DIRECT`; Windows queries
  the storage transfer alignment and uses `FILE_FLAG_NO_BUFFERING`. Unsupported
  platforms or filesystems fail explicitly. Power never reports direct I/O
  after a buffered fallback.
- Storage weights remain explicitly configured by default. The opt-in
  `ValidationThroughput` policy derives bounded relative weights from throughput
  observed during the mandatory integrity hash pass, so automatic weighting
  does not scan a multi-gigabyte model twice. The observations are available
  only from the explicit source descriptor API and are never logged or exported
  as placement telemetry automatically.
- Model crates may pass opaque positive file-benefit scores to
  `plan_partial_mirror`. Power ranks complete verified SafeTensors files by
  benefit density, selects a deterministic whole-file subset under an explicit
  byte budget, and checks native free space against a caller-owned reserve.
  `stage_partial_mirror_blocking` reuses exact completed files and copies each
  missing file through a same-directory temporary, re-hashes it against the
  admitted primary descriptor, syncs it, and publishes it atomically without
  replacement. This extends `WeightStore`; it is not another cache, router, or
  integrity implementation.
- CPU, CUDA, and Metal are explicit device choices. An unavailable explicit
  device fails instead of silently moving execution elsewhere.
- Runtime limits bound graph plans, tensor elements, resident weights, model
  state, context, generation, and concurrency. Model-owned KV or recurrent
  state must call `checked_state_bytes` before allocation.
- Placement and routing telemetry are controlled by `TelemetryMode`. It is off
  by default. Detailed expert heat can reveal input semantics, is never logged
  or persisted automatically, and must remain inside the TEE unless policy
  explicitly authorizes export.

### Ordered Current-Layer Staging

The model crate supplies groups of exact weight requests in its canonical
compute order. Power neither interprets a group as an expert nor performs the
model operation:

```rust
let mut batch = hierarchy.start_staged_batch(groups, &permit, cancellation.clone())?;

// Model-owned compute can start while exact cache misses load in the shared
// bounded background path. Each output is written by canonical_index.
for group in batch.ready_groups() {
    compute_into_slot(group.canonical_index(), group.weights())?;
}

let completion = batch.wait().await?;
for group in completion.groups {
    if output_slot_is_empty(group.canonical_index()) {
        compute_into_slot(group.canonical_index(), group.weights())?;
    }
}
reduce_slots_in_canonical_order()?;
# Ok::<(), a3s_power::error::PowerError>(())
```

A group is exposed only when all of its weights are ready. Different groups may
become ready out of order, but `StagedWeightBatchCompletion::groups` is strictly
canonical. Staging is exact current-layer demand, so it does not inflate
speculative-prefetch usefulness. The report separates cumulative worker service
time, background wall time, and the time spent inside the caller's final
`wait`. Timing and aggregate placement counters remain zero under
`TelemetryMode::Disabled`; enabled staged counters contain no layer, tensor,
route, expert, or tensor-value identity. Dropping or aborting the batch
propagates cancellation and releases the same admission capacity used by
prefetch.

### Verified Storage Topology

Replica weights are relative bandwidth hints, not precision or routing knobs:

```rust
use a3s_power::inference::{
    InferenceLimits, WeightSourceConfig, WeightSourceWeighting, WeightStore,
    WeightStoreConfig,
};

let config = WeightStoreConfig::new("/models/primary")
    .with_partial_replica(WeightSourceConfig::new("/models/replica"))
    .with_source_weighting(WeightSourceWeighting::ValidationThroughput);
let store = WeightStore::open_config(&config, &InferenceLimits::default())?;
# Ok::<(), a3s_power::error::PowerError>(())
```

A complete source must match the aggregate digest. Every file in a partial
source must match the corresponding primary file digest, and tensors not
covered by that source deterministically stay on another eligible source.
Source indices, reads, bytes, and fallback counts appear only when aggregate or
detailed telemetry is explicitly enabled; filesystem paths and measured
validation throughput are not included in placement telemetry.

### Positional Reads and Storage Evidence

`TensorStorageDescriptor` exposes an explicitly requested, path-free tensor
range descriptor. `TensorRead` owns a zeroizing buffer and reports only its
strategy, selected source index, and whether primary fallback occurred. Demand
loads and prefetches call the same cancellable materialization path, so a new
strategy cannot bypass coverage-aware routing or create a second cache.

`a3s-power-storage-bench` exercises this path without constructing a graph,
model architecture, tokenizer, listener, subprocess, or inference backend. It
records integrity-open time separately, performs output-digest validation
outside the measured read interval, and emits no model paths or tensor names.
Linux cold runs synchronize each involved file, apply `POSIX_FADV_DONTNEED`
after integrity-open, and use `mincore` to prove that every requested page
across primary and replica sources is non-resident. Other platforms currently
refuse the verified cold label.
See [Storage Benchmark Protocol](storage-benchmark.md).

### Usage-Ranked Partial Mirror Staging

Routing heat remains model-owned. A model crate can reduce it to bounded opaque
file benefits and explicitly authorize a caller-managed plaintext destination:

```rust
use a3s_power::inference::{
    InferenceLimits, WeightMirrorCandidate, WeightMirrorConfidentiality,
    WeightMirrorPolicy, WeightStore,
};
use tokio_util::sync::CancellationToken;

let store = WeightStore::open("/models/primary", &InferenceLimits::default())?;
let candidates = [
    WeightMirrorCandidate::new("model-00003-of-00008.safetensors", 8_400),
    WeightMirrorCandidate::new("model-00005-of-00008.safetensors", 5_100),
];
let policy = WeightMirrorPolicy::new(32 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)?
    .with_confidentiality(WeightMirrorConfidentiality::CallerManagedPlaintext);
let plan = store.plan_partial_mirror("/fast-storage/model", &candidates, &policy)?;
if plan.admitted {
    store.stage_partial_mirror_blocking(
        "/fast-storage/model",
        &candidates,
        &policy,
        &CancellationToken::new(),
    )?;
}
# Ok::<(), a3s_power::error::PowerError>(())
```

Both calls perform blocking filesystem work and belong on a blocking worker in
async applications. The destination must be a dedicated directory separate
from the primary collection. Existing exact selected files are resumable;
conflicts, unselected SafeTensors files, symlinks, changed sources, and
insufficient reserves fail closed and are never overwritten or deleted.
Completed copies can be admitted through the existing partial-replica
`WeightStoreConfig` path. No usage history, plan, path, capacity snapshot,
telemetry, or receipt is logged or persisted automatically. TEE deployments
must keep `DenyPlaintext` unless their policy protects the destination.

### Hardware-Aware Residency Budget

Hardware discovery never spawns `nvidia-smi`, `vm_stat`, or another process.
Power uses bounded native OS and selected-device APIs, then derives a
reproducible plan from caller-owned policy:

```rust
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyBudgetPolicy,
    ResidencyPolicy,
};

let runtime = EmbeddedRuntime::new(DevicePreference::Auto, InferenceLimits::default())?;
let budget = ResidencyBudgetPolicy::new(5_000, 5_000)?
    .with_host_reserve_bytes(2 * 1024 * 1024 * 1024)
    .with_device_reserve_bytes(512 * 1024 * 1024);
let plan = runtime.plan_residency_budget(&budget)?;
let residency = plan.apply_to(&ResidencyPolicy::default())?;
# Ok::<(), a3s_power::error::PowerError>(())
```

The serialized snapshot and plan are available for explicit operator review,
but Power never logs, persists, exports through placement telemetry, or binds
them into execution receipts automatically. A TEE guest therefore plans only
from the memory visible inside that guest, while policy still controls whether
the result may leave the trust boundary.

### Lossless Live Residency Adaptation

The model owner decides when no model work is in flight and supplies opaque
session heat for exactly the groups in the active plan. Power neither derives
model topology from IDs nor persists this heat:

```rust
use a3s_power::inference::ResidencyAdaptationPolicy;

let policy = ResidencyAdaptationPolicy::default();
let adaptation = hierarchy.plan_residency_adaptation(&live_candidates, &policy)?;
if !adaptation.is_noop() {
    hierarchy.apply_residency_adaptation(&adaptation, &permit, &cancellation)?;
}
# Ok::<(), a3s_power::error::PowerError>(())
```

The default policy mirrors Colibri's 25% plus four-unit hysteresis and four-swap
limit, but Power generalizes the mechanism around atomic `ResidencyCandidate`
groups. A replacement changes tier only. It cannot change the exact weights in
a group, router output, gate mass, dtype, precision, or execution receipt. The
adaptation and replacement identities are intentionally not serializable; a
TEE caller that wants history across sessions must keep that state in its own
authorized sealed store and submit a fresh heat snapshot.

### Value-Preserving Cross-Layer Prefetch Hints

The model crate supplies exact router output and keeps topology-to-weight
mapping ownership. Power learns and scores only opaque expert IDs:

```rust
// ResidencyPolicy::telemetry must be explicitly set to Detailed.
hierarchy.record_route_transition(&current_routes, &actual_next_routes)?;
let hints = hierarchy.route_prefetch_hints(
    &current_routes,
    actual_next_routes.layer(),
    6,
)?;

// The model owner maps hints.experts() to its own WeightKey values and passes
// those requests through WeightHierarchy::start_prefetch.
let evidence = hierarchy.evaluate_route_prefetch_hints(
    &hints,
    &actual_next_routes,
)?;
assert!(evidence.recall() <= 1.0);
# Ok::<(), a3s_power::error::PowerError>(())
```

`route_coupling_history` returns a versioned snapshot bound to the admitted
weight SHA-256 and recorded layer geometry. Restoring a malformed, oversized,
cross-model, or geometrically inconsistent history fails atomically. Power
does not write a route trace or sidecar; a TEE deployment must use a
caller-owned sealed store if cross-session learning is authorized.

## Integrity and TEE Invariants

- `WeightStore` hashes every SafeTensors file and a deterministic aggregate
  manifest before serving it through mmap or positional reads. Model crates pin
  the aggregate digest with `verify_integrity` and may reuse Power's Ed25519
  model seal verification with `verify_signature`.
- Replica selection never changes dtype, shape, bytes, routing, or precision.
  Complete sources require the exact aggregate digest; partial sources require
  exact per-file digests and tensor descriptors. Source count is bounded by
  `InferenceLimits::max_weight_sources`.
- Embedded inference does not bind a socket, start a Web server, download a
  model, invoke Python, or spawn an inference service.
- The server, API, CLI, model registry, remote clients, and Web dependencies are
  behind the default `server` feature. An embedded-only build disables default
  features, so its dependency closure contains no HTTP server/client stack and
  does not enable Tokio networking, process, or signal support.
- Existing encrypted model loading, remote attestation, privacy redaction,
  request receipts, and zeroizing sensitive request buffers remain independent
  security controls; the embedded runtime does not replace them.
- Positional plaintext buffers and aligned direct-I/O scratch allocations are
  zeroized on drop. Read strategies, storage benchmark data, source paths, and
  tensor ranges are not added to attestation claims or execution receipts.
- Routing history is bound to the exact weight digest. Power only returns a
  serializable value; persistence must use a model-owned encrypted or sealed
  store. Plaintext sidecar files are not created.
- Cross-layer coupling history additionally binds expert geometry, requires
  detailed telemetry, restores atomically under explicit bounds, and is never
  logged or persisted automatically. Ephemeral predictions are not
  serializable by the Power API.
- Placement telemetry contains no tensor values. Detailed routing identifiers
  are still sensitive metadata and are opt-in.
- Staged-batch telemetry contains aggregate counts and timing only. Group
  indices, layer IDs, tensor names, tensors, and model-owned outputs are never
  logged, persisted, added to receipts, or added to attestation claims by
  Power. Timing remains zero when telemetry is disabled.
- Residency candidates and plans can reveal the learned hot set. Power returns
  them to the caller but never logs or persists them automatically.
- Live residency adaptations are ephemeral, non-serializable, bound to the
  active base plan, and applied through the same transactional cache path.
  Caller-supplied heat and replacement identities are not placed in telemetry,
  logs, receipts, or attestation claims.
- Hardware memory snapshots can reveal deployment capacity. Automatic budgeting
  is explicit, snapshots are never logged or placed in execution receipts, and
  callers must apply their own TEE export policy.

## Colibri Adoption Boundaries

| Colibri mechanism | Power treatment |
| --- | --- |
| VRAM/RAM/storage as one hierarchy | Implemented generically with exact dtype and shape checks |
| Layer-local LFRU/LRU and learned hot pins | Implemented with decaying frequency, bounded recency, separate manual/plan pins, and transactional hot-set replacement |
| Lossless live tier adaptation | Implemented at explicit safe boundaries with caller-owned heat, default 25% + 4 hysteresis, a four-replacement cap, exact-footprint swaps, stale-plan rejection, and the existing transactional pin/cache path |
| Batched expert union | Implemented without changing router order, top-k, or gate weights |
| One-layer-ahead I/O overlap | Implemented with bounded, cancellable Tokio blocking workers and useful/unused prefetch measurement |
| Current-layer resident/cold overlap | Implemented with atomic staged groups, immediate resident readiness, exact-demand miss loading through shared admission/cache/source routing, and canonical completion |
| Cross-layer coupling hints | Implemented as bounded, digest/geometry-bound, detailed-telemetry-only co-occurrence learning with deterministic per-position scores, batch union, and measured recall; hints never alter router output |
| Hardware-aware placement | Native Linux/macOS/Windows host discovery, selected CUDA/Metal device discovery, unified-memory accounting, deterministic capped budget planning, and integrity-read storage weighting are implemented without subprocesses |
| Multi-drive weighted mirrors and direct I/O | Exact complete/partial replicas, usage-ranked budgeted staging, coverage-aware weighted routing, primary fallback, bounded positional reads, and aligned Linux/Windows direct reads share one `WeightStore`; mmap remains default pending end-to-end wins |
| Cold-storage microbenchmark | Standalone path-free reports separate integrity-open, output validation, and measured demand reads; Linux cold labels require `FADV_DONTNEED` plus `mincore`, while unsupported platforms refuse the claim |
| Routing-history sidecar | Plaintext automatic persistence is intentionally not adopted; TEE policy owns sealed storage |
| Cache-aware expert substitution | Not enabled because it changes model semantics; exact routing is the default invariant |
| Speculative decoding and KV policy | Model control flow remains in the model crate; Power supplies shared state bounds and receipts |
| Web dashboard | Not part of embedded inference; an explicit external consumer may receive policy-approved aggregate telemetry |

## Deep Colibri Adoption Sequence

Power adopts Colibri mechanisms only when they can remain model-neutral,
value-preserving, bounded, and compatible with the existing TEE trust path.
The current sequence is:

| Priority | Colibri lesson | Power-level treatment | Required evidence |
| --- | --- | --- | --- |
| 1 | Deferred current-layer cold I/O while resident experts compute | **Power substrate complete:** ordered atomic staged batches reuse existing admission, cache, source routing, key locks, cancellation, and privacy-gated telemetry; model crates retain canonical computation | Power tests cover tensor/order parity, immediate readiness, cancellation, shared materialization, and separated timing; each model integration must still publish end-to-end output parity and workload gains |
| 2 | Hardware/model-specific measured tuning | Add an ephemeral, digest-bound comparison profile for lossless execution knobs only; model crates own teacher-forced calibration and any sealed persistence | Repeated baseline/winner reversal, minimum throughput gain, hit-rate parity, and bounded p99 regression |
| 3 | Lossless compressed expert tiers | Add representations behind the existing verified `WeightStore` index and receipts; never quantize or reinterpret tensors during placement | Whole-artifact digest validation, decoded byte parity, bounded scratch, zeroization, and end-to-end wins |
| 4 | Accelerator-wide residency declarations and fused batches | Extend typed CUDA/Metal devices without bypassing Candle safety, admission, cancellation, or confidential-GPU claims | Kernel parity, explicit fallback identity, memory-pressure tests, and named-hardware results |
| 5 | Warm model state across sessions | Keep KV/recurrent topology in the model crate; Power supplies bounds, digest binding, zeroization, and a sealed-state envelope rather than plaintext sidecars | Interrupted-write recovery, wrong-model rejection, rollback policy, and TEE export authorization |

Cache-aware expert substitution remains outside the default path because it
changes model semantics. Grammar drafts, MTP/speculative control flow,
tokenizers, and model-specific KV layouts remain with the owning model crate.
Power will not add a Web dashboard or embed model assets to imitate Colibri's
product surface.

## Validation Gates

Every model integration must publish reproducible evidence for:

1. output parity against a pinned upstream implementation;
2. exact model revision, graph-plan digest, and weight digest;
3. cold and warm latency, peak host/device memory, per-source bytes read, cache
   hit rate, and useful/unused prefetch rate on named hardware;
4. identical outputs with caching, prefetch, and current-layer staging disabled
   versus enabled;
5. cancellation, resource-limit, malformed-plan, and wrong-digest failures;
6. TEE regression tests, including telemetry-off behavior and no plaintext
   persistence.

An optimization is not enabled by default from a microbenchmark alone. It must
preserve model semantics and improve an end-to-end workload under a documented
hardware and cache state.

## Deliberate Follow-up Work

The manual hosted-runner workflow can now capture Linux verified-cold/direct
and Windows warm/direct reports against a hash-pinned public SafeTensors
workload. A Linux host that cannot evict every requested page produces an
explicit path-free capability limitation instead of a false cold report. Those
ephemeral runners do not provide independent storage controllers, and their
hardware may change between runs. The current foundation therefore still needs
reviewed stable-hardware results, independent-controller multi-source evidence,
end-to-end model workload wins before direct I/O can become a default,
encrypted persistence for model-owned routing state, and broader cross-model
benchmarks. These must extend the same runtime and integrity primitives rather
than introduce parallel model-specific systems.
