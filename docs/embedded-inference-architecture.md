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
            ├─ device cache ───── bounded, typed device, exact dtype
            ├─ host cache ─────── per-layer LFRU/LRU + provenance-aware pins
            ├─ prefetch pool ───── bounded tasks and blocking workers
            ├─ residency plan ──── replaceable atomic heat-ranked groups
            └─ SafeTensors ─────── verified weighted storage replicas
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
- `start_prefetch` starts bounded blocking I/O immediately. The hierarchy caps
  active tasks and workers, unions duplicate requests, and propagates
  cancellation when the task is aborted or dropped. A model can prefetch layer
  N+1, compute layer N, then await the task, providing the one-layer-ahead
  overlap used by routed models.
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

## Integrity and TEE Invariants

- `WeightStore` hashes every SafeTensors file and a deterministic aggregate
  manifest before mapping it. Model crates pin the aggregate digest with
  `verify_integrity` and may reuse Power's Ed25519 model seal verification with
  `verify_signature`.
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
- Routing history is bound to the exact weight digest. Power only returns a
  serializable value; persistence must use a model-owned encrypted or sealed
  store. Plaintext sidecar files are not created.
- Placement telemetry contains no tensor values. Detailed routing identifiers
  are still sensitive metadata and are opt-in.
- Residency candidates and plans can reveal the learned hot set. Power returns
  them to the caller but never logs or persists them automatically.
- Hardware memory snapshots can reveal deployment capacity. Automatic budgeting
  is explicit, snapshots are never logged or placed in execution receipts, and
  callers must apply their own TEE export policy.

## Colibri Adoption Boundaries

| Colibri mechanism | Power treatment |
| --- | --- |
| VRAM/RAM/storage as one hierarchy | Implemented generically with exact dtype and shape checks |
| Layer-local LFRU/LRU and learned hot pins | Implemented with decaying frequency, bounded recency, separate manual/plan pins, and transactional hot-set replacement |
| Batched expert union | Implemented without changing router order, top-k, or gate weights |
| One-layer-ahead I/O overlap | Implemented with bounded, cancellable Tokio blocking workers and useful/unused prefetch measurement |
| Hardware-aware placement | Native Linux/macOS/Windows host discovery, selected CUDA/Metal device discovery, unified-memory accounting, deterministic capped budget planning, and integrity-read storage weighting are implemented without subprocesses |
| Multi-drive weighted mirrors and direct I/O | Exact complete/partial replicas, usage-ranked budgeted staging, coverage-aware weighted routing, source telemetry, and primary fallback are implemented under `WeightStore`; direct range I/O remains follow-up work |
| Routing-history sidecar | Plaintext automatic persistence is intentionally not adopted; TEE policy owns sealed storage |
| Cache-aware expert substitution | Not enabled because it changes model semantics; exact routing is the default invariant |
| Speculative decoding and KV policy | Model control flow remains in the model crate; Power supplies shared state bounds and receipts |
| Web dashboard | Not part of embedded inference; an explicit external consumer may receive policy-approved aggregate telemetry |

## Validation Gates

Every model integration must publish reproducible evidence for:

1. output parity against a pinned upstream implementation;
2. exact model revision, graph-plan digest, and weight digest;
3. cold and warm latency, peak host/device memory, per-source bytes read, cache
   hit rate, and useful/unused prefetch rate on named hardware;
4. identical outputs with caching and prefetch disabled versus enabled;
5. cancellation, resource-limit, malformed-plan, and wrong-digest failures;
6. TEE regression tests, including telemetry-off behavior and no plaintext
   persistence.

An optimization is not enabled by default from a microbenchmark alone. It must
preserve model semantics and improve an end-to-end workload under a documented
hardware and cache state.

## Deliberate Follow-up Work

The current foundation does not yet implement direct range I/O, provide an
independent cold-storage benchmark, persist encrypted model state, or run
cross-model benchmarks. These should extend the same runtime and integrity
primitives rather than introduce parallel model-specific systems.
