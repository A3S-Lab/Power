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
| Bounded exact model/device session pools and memory-aware microbatch plans | `a3s-power` |
| Static graph validation and reviewed operator execution | `a3s-power` |
| Storage/RAM/device weight placement and telemetry policy | `a3s-power` |
| Opaque state sealing, bounds, export authorization, and crash recovery | `a3s-power` |
| Model family, topology, graph identity, and revision | Model crate |
| KV/recurrent state layout, serialization, and semantic parity | Model crate |
| Tokenizer, preprocessing, postprocessing, and generation loop | Model crate |
| Product orchestration and document structure | Product crate |

Power therefore contains no PP-OCR, Unlimited-OCR, or other model assets,
tokenizers, conversion tools, or revision hashes.

## Runtime Shape

```text
model crate
  ├─ model-owned reviewed plans and native control flow
  ├─ model-owned KV/recurrent layout ── opaque bytes to SealedStateEnvelope
  ├─ SealedStateStore ──────────────── authenticated primary/backup recovery
  ├─ one EmbeddedRuntime per model session
  ├─ one ExecutionPermit per logical request
  ├─ ExecutionBatchLifecycle ───────── distinct existing permit per member
  │    └─ canonical ragged steps ───── atomic state/evidence boundary
  ├─ HardwareEvidenceBundle ────────── storage + tuning + parity replay
  └─ shared request-scoped execution
       │
       ├─ GraphExecutor ───────── validated dense/static graphs
       ├─ RoutedExpertBatch ───── exact batch union, no route changes
       └─ WeightHierarchy
            ├─ route coupling ─── private bounded hints, measured against truth
            ├─ device cache ───── bounded, typed device, exact dtype
            ├─ host cache ─────── per-layer LFRU/LRU + provenance-aware pins
            ├─ load window ─────── shared worker + canonical-byte admission
            ├─ staged batches ──── event readiness + canonical order
            ├─ residency plan ──── replaceable and stably adapted atomic groups
            ├─ fused batches ───── attested device-only groups + explicit fallback
            └─ SafeTensors ─────── canonical primary + verified weighted representations
```

A logical request holds one permit across every component graph. A multimodal
model must not create independent admission, device, hash, receipt, or cache
systems for its vision encoder, projector, dense layers, and routed experts.
Multiple exact model sessions may instead share one device-bound
`ModelSessionPool<T>`. The pool gives each entry its own bounded
`EmbeddedRuntime` and one shared physical-device gate; it deduplicates lazy
initialization and retains ready entries until pool drop without adding an
eviction or persistence policy.

## Colibri Ideas Adapted as Generic Mechanisms

- Finite model and device queues reuse one cancellation-aware admission
  controller. Queue slots, active permits, and unfinished model-load slots are
  RAII-owned, so cancellation or future drop cannot strand capacity.
- Deterministic microbatch planning preserves caller order while enforcing
  explicit input/state bounds and point-in-time host/device peak-memory
  budgets. Execution refreshes memory pressure and exact topology before
  admission. One execution permit covers the resulting aggregate model call;
  `max_concurrent_requests` limits concurrent calls, not slots within that
  call.
- `TensorInput::stack_leading` and `TensorOutput::split_leading` provide one
  canonical model-neutral boundary for fused leading-axis calls. They validate
  compatible trailing shapes, exact positive partitions, finite values, caller
  order, and the shared tensor limit. The model crate still owns padding,
  valid extents, bucketing, and per-slot semantics.
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
- `ExecutionBatchLifecycle` adapts Colibri's continuously batched multi-slot
  decode boundary without importing its model loop. Its global declaration
  binds exact weights, the model-owned state layout, scheduler semantics,
  runtime device, and relevant limits. Each member consumes a distinct permit
  from the existing runtime admission controller and cannot alias another
  active member's model-state identity.
- Every execution step contains all members present at that boundary exactly
  once, canonically ordered by admission while allowing different positions and
  row shapes. Members admitted while arithmetic is running enter only the next
  step. Power validates aggregate input, shape, context, generation, and state
  bounds before launch and before commit; it does not choose tokens, batch
  shapes, KV layouts, kernels, or completion policy.
- Step commits are atomic. Cancellation discards only the affected row, an
  invalid commit changes no lifecycle position or state and can be retried, and
  completion releases the same permit that admitted the member. Step and final
  evidence expose aggregate counts plus canonical input/output/transcript
  digests, never member/state identities or model-owned bytes.
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
  active tasks, workers, total bytes, and bytes concurrently owned by workers;
  it unions duplicate requests and propagates cancellation when the task is
  aborted or dropped. A model can prefetch layer N+1, compute layer N, then
  await the task, providing the one-layer-ahead overlap used by routed models.
- `start_staged_batch` adapts Colibri's current-layer pipeline without moving
  model execution into Power. It validates every non-empty atomic group,
  duplicate key, layer, tensor descriptor, item count, and byte total before
  touching cache heat. Fully resident groups are available synchronously;
  exact misses use the same prefetch admission, bounded Tokio blocking workers,
  canonical-byte flight window, source router, cache, per-key load
  serialization, and cancellation path. `next_ready_group().await` exposes
  each newly complete group without a polling loop. Early groups carry an
  implicit canonical input index, and final completion always restores the
  original order. Model crates compute into indexed output slots and retain
  their exact gate application and reduction order.
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
- Accelerator declarations select canonical groups from the already active
  residency plan. Every selected group must be wholly in the device tier, and
  the declaration binds its exact membership and order to the weight digest,
  active-plan digest, typed device, fused-kernel digest, exact-fallback digest
  and target, runtime limits, and security mode. Resolution performs an atomic
  device-cache-only acquisition: it never reloads a missing tensor, promotes a
  host entry, or relabels an exact fallback as accelerator execution.
- `AcceleratorFusedBatch` keeps the existing `ExecutionPermit`, checks
  cancellation before and after model-owned Candle compute, and rejects
  cross-device or over-limit input/output tensors. A kernel can report only
  the typed `Unavailable` outcome to select its declared exact fallback;
  arithmetic and policy errors remain failures. Fallback execution retains the
  permit and its own declared CPU or runtime-device target.
- Embedded receipt v2 commits to the declaration, implementation, input,
  output, declared device, actual execution device, fallback reason, and an
  optional canonical attestation-claims digest. It contains no tensor names,
  group IDs, values, or raw evidence. Confidential-GPU declarations require a
  hardware CPU-TEE report whose existing canonical v2 claims bind the exact
  model digest, NVIDIA NRAS evidence/verdict, and declaration policy digest.
  Power structurally matches that already-verified report; it does not create a
  second hardware verifier or attestation schema.
- Warm-state envelopes adapt Colibri's incremental, commit-last recovery idea
  but never adopt its plaintext, model-shaped `.coli_kv` file. The owning model
  crate serializes exact KV/recurrent topology into opaque bytes and supplies a
  layout digest. Power authenticates weight, layout, hashed state identifier,
  generation, lengths, scope, and nonce in one fixed AES-256-GCM header.
- `SealedStateStore` reuses Power's directory durability helper and publishes a
  synchronized same-directory pending file only after preserving the prior
  committed primary as a synchronized backup. Recovery never opens pending
  data; it authenticates primary and backup and selects the highest generation
  allowed by a caller-owned rollback floor. Power does not invent a hardware
  monotonic counter or silently lower that floor.
- TEE-local envelopes expose no byte-export method. Authorized export requires
  the same canonical hardware-report claim matcher used by confidential GPU
  binding, an exact attested model digest, and a caller-approved export-policy
  digest. State keys, opened plaintext, imported ciphertext, and claim
  serialization buffers use zeroizing owners.
- `EmbeddedRuntime::plan_residency_budget` discovers host memory through bounded
  native Linux, macOS, or Windows APIs and device memory through the selected
  CUDA or Metal handle. The caller supplies explicit fractions, safety
  reserves, caps, host/device allocation order, and model-neutral fixed-state
  plus peak-scratch byte reservations. Power applies the fraction first and
  removes declared runtime bytes before assigning hot-weight cache capacity.
  The resulting plan is capped by
  `InferenceLimits::max_resident_weight_bytes`; Metal unified memory is counted
  once while both host and device runtime allocations consume that shared pool.
  `EmbeddedRuntime::apply_residency_budget` refreshes the native snapshot and
  fails before cache-policy mutation when availability or exact pool topology
  no longer supports the plan. Plans with non-zero runtime reservations cannot
  use the offline-only `ResidencyBudgetPlan::apply_to` shortcut. Discovery is
  opt-in, incomplete discovery fails closed instead of guessing, and the
  zero-cache default does not change.
- `WeightSourceConfig` may combine one anchor root with explicit disjoint
  `shard_roots`. Every physical root must resolve to a real, non-empty
  directory; duplicate, nested, or overlapping roots and duplicate logical
  relative paths fail closed. Discovery, file/byte bounds, hashing, indexing,
  mmap or positional readers, lossless admission, and partial-mirror staging
  operate on the combined logical collection. The canonical digest covers
  stable relative paths, lengths, and bytes, not physical roots, so moving
  unchanged shards across volumes does not change model identity, signatures,
  TEE binding, or receipts.
- `WeightStoreConfig` accepts a primary collection plus bounded, read-only
  replicas. Complete replicas must match the primary aggregate digest. An
  explicitly partial replica may contain a non-empty subset of primary
  SafeTensors files; every present relative file, byte length, SHA-256 digest,
  tensor name, dtype, shape, and byte count must match before mapping. A stable
  bandwidth-weighted hash selects only among sources that contain the requested
  tensor, and recoverable replica errors fall back to primary. This extends the
  existing `WeightStore` rather than creating a second model cache or integrity
  path.
- The primary is always canonical SafeTensors. A typed lossless replica may
  store opaque U8 records under the original tensor names, using Power's
  pure-Rust `rans-nibble-256-v1` codec: one shard-local 16-symbol static table,
  256 round-robin streams, exact derived framing, and mandatory zero padding.
  The representation stamp is required because compressed length is
  data-dependent; Power never guesses a codec from dtype or record size. The
  complete physical collection SHA-256 is verified before metadata parsing or
  decoded allocation. Every admitted record must be smaller than its canonical
  tensor, decode within the existing state-memory bound, pass framing,
  amplification, stream-state, cancellation, and consumption checks, and
  byte-match the canonical primary before routing. Complete coverage or a
  proper non-empty tensor subset is accepted. Runtime decode reuses the same
  mmap/positional reader, source router, primary fallback, prefetch/staging,
  residency cache, telemetry, and benchmark path.
- Every verified tensor has one exact source-file index, absolute byte range,
  dtype, shape, and byte count. `Mmap` remains the default. The opt-in buffered
  positional path does not retain a collection-wide mmap and uses a 1 MiB
  full-read loop with interruption, cancellation, overflow, truncation, and
  honest short-read handling. Its plaintext buffers zeroize on drop.
- The opt-in macOS cache-bypass path applies `F_NOCACHE` to the mandatory
  integrity hash and exact positional range handles while reusing the same
  index, source router, primary fallback, cancellation, zeroizing buffers, and
  benchmark evidence. It is not labeled direct I/O and does not prove that
  pages populated by another handle were absent, so it cannot mint a verified
  cold-cache claim.
- Direct positional reads reuse that index, source router, primary fallback,
  demand path, and prefetch path. Linux uses aligned `O_DIRECT`; Windows queries
  the storage transfer alignment and uses `FILE_FLAG_NO_BUFFERING`. Unsupported
  platforms or filesystems fail explicitly. Power never reports direct I/O
  after a buffered fallback.
- Storage weights remain explicitly configured by default. The opt-in
  `ValidationThroughput` policy derives bounded relative weights from throughput
  observed during the mandatory source-validation pass. For canonical sources
  this is the integrity hash; for lossless sources it also includes required
  decode admission and canonical comparison. Automatic weighting performs no
  extra probe. The observations are available only from the explicit source
  descriptor API and are never logged or exported as placement telemetry
  automatically.
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
while let Some(group) = batch.next_ready_group().await? {
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
time, background wall time, event-driven readiness wait, and the time spent
inside the caller's final `wait`. It also reports peak active-worker count and
canonical bytes. `ResidencyPolicy::max_background_inflight_bytes` is shared by
prefetch and staging, is capped by the total batch byte limit, and rejects an
individual tensor that cannot fit; the scheduler may issue a smaller later
item into spare capacity without reordering the pending queue. Timing and
aggregate placement counters remain zero under
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
    .with_primary_shard_root("/models/primary-overflow")
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
    ResidencyPolicy, RuntimeMemoryReservations,
};

let runtime = EmbeddedRuntime::new(DevicePreference::Auto, InferenceLimits::default())?;
// Counts future allocations not already reflected in the planning snapshot.
let reservations = RuntimeMemoryReservations::default()
    .with_host_fixed_bytes(512 * 1024 * 1024)
    .with_host_scratch_bytes(256 * 1024 * 1024);
let budget = ResidencyBudgetPolicy::new(5_000, 5_000)?
    .with_host_reserve_bytes(2 * 1024 * 1024 * 1024)
    .with_device_reserve_bytes(512 * 1024 * 1024)
    .with_runtime_reservations(reservations)?;
let plan = runtime.plan_residency_budget(&budget)?;
let residency = runtime.apply_residency_budget(&plan, &ResidencyPolicy::default())?;
# Ok::<(), a3s_power::error::PowerError>(())
```

Fixed bytes remain live for the execution; scratch is the caller's maximum
additional simultaneous working set. The owning model crate computes those
four byte counts without exposing KV, recurrent, activation, or model topology
to Power. The serialized reservations, snapshot, and plan are available for
explicit operator review, but Power never logs, persists, exports them through
placement telemetry, or binds them into execution receipts automatically. A
TEE guest therefore plans only from memory visible inside that guest, while
policy still controls whether the result may leave the trust boundary.

### Attestation-Bound Fused Accelerator Batches

Fused execution references the active plan instead of declaring another
weight registry:

```rust
use a3s_power::inference::{
    AcceleratorBatchResolution, AcceleratorFallbackMode,
    AcceleratorFusedBatchSpec, AcceleratorKernelOutcome,
};

let spec = AcceleratorFusedBatchSpec::new(
    fused_kernel_sha256,
    exact_fallback_sha256,
    vec!["model-owned-group-0".to_string()],
)
.with_fallback_mode(AcceleratorFallbackMode::AllowExact);
let declaration = hierarchy.declare_accelerator_residency(&spec)?;

match hierarchy.resolve_accelerator_batch(
    &declaration,
    confidential_gpu_binding.as_ref(),
    &permit,
    &cancellation,
)? {
    AcceleratorBatchResolution::Ready(batch) => {
        let result = batch.execute_or_fallback(
            &input,
            &cancellation,
            |input, groups, cancellation| {
                model_owned_fused_candle_kernel(input, groups, cancellation)
                    .map(AcceleratorKernelOutcome::Output)
            },
        )?;
        consume_fused_or_exact_fallback(result)?;
    }
    AcceleratorBatchResolution::Fallback(fallback) => {
        run_model_owned_exact_fallback(fallback)?;
    }
}
# Ok::<(), a3s_power::error::PowerError>(())
```

`ConfidentialGpuBinding::from_verified_attestation_report` accepts only a
hardware SEV-SNP/TDX report with canonical v2 claims and matching model,
NVIDIA NRAS evidence/verdict, and execution-policy digest. Its name is an
explicit trust contract: callers must first use Power's existing strict
hardware verifier or obtain the report inside the currently attested runtime.
The binding stores digests only.

### Attestation-Bound Heterogeneous Device Meshes

Colibri assigns each layer or resident expert group a home device, moves the
residual stream explicitly, and has measured cases where multi-GPU P2P traffic
costs more than it saves. Power adopts the enforceable substrate without
moving graph partitioning into the runtime or promising a speedup:

```rust
use a3s_power::inference::{
    AcceleratorBatchResolution, AcceleratorDeviceMesh, AcceleratorKernelOutcome,
    AcceleratorMeshDevice, AcceleratorPeerTransferOutcome,
    AcceleratorPeerTransferSpec, DevicePreference, RuntimeDevice,
};

let peer = RuntimeDevice::resolve(DevicePreference::Cuda { ordinal: 1 })?;
let mesh = AcceleratorDeviceMesh::new(
    "home",
    vec![
        AcceleratorMeshDevice::new("home", hierarchy.runtime().device().clone()),
        AcceleratorMeshDevice::new("expert-peer", peer),
    ],
    vec![
        AcceleratorPeerTransferSpec::new("home", "expert-peer", 8 * 1024 * 1024, 1),
        AcceleratorPeerTransferSpec::new("expert-peer", "home", 8 * 1024 * 1024, 1),
    ],
    16 * 1024 * 1024,
)?;
let declaration = hierarchy.declare_accelerator_mesh_residency(&spec, &mesh)?;

if let AcceleratorBatchResolution::Ready(batch) =
    hierarchy.resolve_accelerator_mesh_batch(
        &declaration,
        &mesh,
        confidential_gpu_binding.as_ref(),
        &permit,
        &cancellation,
    )?
{
    let result = batch.execute_mesh_or_fallback(
        &input,
        &cancellation,
        |input, groups, mesh, cancellation| {
            let AcceleratorPeerTransferOutcome::Transferred(peer_input) =
                mesh.transfer("home", "expert-peer", input)?
            else {
                return Ok(AcceleratorKernelOutcome::Unavailable);
            };
            let peer_output = model_owned_peer_kernel(
                &peer_input,
                groups,
                cancellation,
            )?;
            let AcceleratorPeerTransferOutcome::Transferred(home_output) =
                mesh.transfer("expert-peer", "home", &peer_output)?
            else {
                return Ok(AcceleratorKernelOutcome::Unavailable);
            };
            Ok(AcceleratorKernelOutcome::Output(home_output))
        },
    )?;
    consume_fused_or_exact_fallback(result)?;
}
# Ok::<(), a3s_power::error::PowerError>(())
```

The mesh contains at most 16 unique typed runtime devices and every node must
be reachable from and able to return to the primary node. Each directed edge
has a maximum tensor size and launch count; the mesh also has a total byte
budget. Power reuses `InferenceLimits::max_graph_nodes` for topology count,
`max_input_bytes` for one copy, and `max_state_bytes` for aggregate traffic.
Copies are synchronous at this substrate boundary, so one Power-managed copy
is in flight at a time; returned tensors remain model-owned state within the
aggregate bound. Cancellation is checked before and after every copy.

The active residency plan remains the only weight placement source of truth.
Resolution atomically acquires already-pinned primary-device weights and never
creates another cache, residency registry, or planner. The model kernel may
move an activation or selected resident tensor through the guarded edges and
continues to own graph partitioning, reduction order, and arithmetic. A
single-device mesh is valid specifically so model integrations can establish
exact parity before adding peer edges.

In confidential-GPU mode every CUDA node carries an explicit NVIDIA
claim-array index. Power requires the declaration's CUDA indices and fabric
indices to equal the verified NRAS GPU/NVSwitch claim sets exactly. A CUDA
ordinal is never inferred to be a claim index or UEID. NRAS proves the claimed
devices/fabric are present; it does not prove that a particular directed edge
is usable, so the actual Candle copy remains the fail-closed availability
check. Backend unavailability can select only the digest-declared exact
fallback. Receipt v3 exposes the mesh digest, canonical actual devices, and a
transfer-trace digest; private node names, edges, group IDs, tensor names, and
values stay outside receipts and automatic telemetry.

### Bounded Sealed Warm State

Power treats cross-session state as opaque model-owned bytes. It does not know
whether the payload contains MLA latents, conventional KV tensors, recurrent
state, token history, or routing heat:

```rust
use a3s_power::inference::{
    SealedStateBinding, SealedStateEnvelope, SealedStateKey,
    SealedStateRollbackPolicy, SealedStateScope, SealedStateStore,
};

let binding = SealedStateBinding::for_identifier(
    weights_sha256,
    model_owned_state_layout_sha256,
    session_identifier,
    &limits,
)?;
let key = SealedStateKey::from_bytes(tee_derived_state_key);
let scope = SealedStateScope::TeeLocal;
let rollback = SealedStateRollbackPolicy::new(trusted_generation_floor);
let envelope = SealedStateEnvelope::seal(
    &binding,
    next_generation,
    &model_owned_serialized_state,
    &key,
    scope,
    &limits,
    &cancellation,
)?;

let store = SealedStateStore::new(state_path)?;
store.commit(
    &envelope,
    &binding,
    &key,
    scope,
    rollback,
    &limits,
    &cancellation,
)?;
if let Some(recovered) = store.load(
    &binding,
    &key,
    scope,
    rollback,
    &limits,
    &cancellation,
)? {
    model_owned_restore(recovered.as_bytes())?;
}
# Ok::<(), a3s_power::error::PowerError>(())
```

`commit` and `load` perform blocking filesystem and cryptographic work; async
model code must call them from its existing bounded blocking path. One store
instance serializes writers, while cross-process writer ownership remains an
operator responsibility. The primary and backup are both encrypted and
authenticated. An interrupted pending file is uncommitted even if its bytes are
complete. A deployment that needs rollback resistance keeps
`trusted_generation_floor` in a TEE monotonic source or trusted control plane;
the files alone cannot prove freshness against an attacker able to replace the
whole directory.

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
TEE caller that wants history across sessions must serialize opaque state
through the shared authorized `SealedStateEnvelope` path and submit a fresh
heat snapshot.

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
does not write a route trace or sidecar; a TEE deployment uses the shared
authorized `SealedStateEnvelope` path if cross-session learning is authorized.

### Digest-Bound Lossless Tuning Evidence

Colibri demonstrates why machine-specific execution settings must be measured
instead of guessed: an unsafe thread, NUMA, I/O, or accelerator setting can
regress badly on unfamiliar hardware. Power therefore keeps its defaults and
evaluates only model-owned aggregate evidence through
`evaluate_tuning_profile`.

The model crate creates one teacher-forced calibration sequence and gives every
execution the same hidden-state inputs. It owns the candidate sweep, the
mapping from opaque configuration digest to reviewed lossless settings, and
the application of those settings. Power requires every submitted run to bind
to the same:

1. weight collection SHA-256;
2. reviewed graph/source SHA-256;
3. typed calibration-workload digest;
4. runtime, device, and environment SHA-256 identities;
5. baseline or candidate configuration SHA-256; and
6. typed output digest.

Each candidate contains at least two complete rounds. Every round records both
baseline→candidate and candidate→baseline order, and each adjacent pair must
report the same completed work, latency-sample count, and logical cache-request
count. A candidate is eligible only when both order-specific lower medians meet
the configured throughput gain, every cache-hit delta stays within tolerance,
every p99 stays within the regression cap, and every output digest is identical
across the complete submission. Selection uses the lower of the two
order-specific medians. Exact winner ties retain the baseline instead of using
a digest tie-breaker.

Evidence is bounded to 32 candidates and 64 rounds per candidate. Digests must
be canonical lowercase SHA-256 values; mixed bindings, duplicate candidate or
round IDs, malformed measurements, insufficient samples, wrong order, output
mismatch, and arithmetic overflow fail closed. The serializable decision
contains only digests, thresholds, and aggregate statistics. It contains no
workload text, tensor, path, topology, or configuration value, and Power does
not log, apply, or persist it. If persistence is authorized, the model crate
uses the shared `SealedStateEnvelope` mechanism with its own layout digest.

### Canonical Hardware Evidence Bundles

Colibri's measurements also show that a locally plausible optimization may
regress on another machine or workload. Power therefore composes its existing
evidence instead of adding a second benchmark, telemetry stream, receipt, or
attestation format.

`HardwareEvidenceBinding::new` takes a Power version and exact Git revision,
weight and reviewed graph/source digests, one typed `RuntimeDeviceIdentity`, and
the path-free `StorageBenchmarkSystem`. Domain-separated canonical hashes derive
the runtime, device, and named environment identities. Its `tuning_binding`
method is the single mapping into `TuningProfileBinding`; model integrations do
not invent parallel platform identities for the same review run.

`HardwareEvidenceBundle::build` accepts:

1. 2–128 bounded raw storage reports from one revision, model, deterministic
   tensor sequence, and exact named system;
2. the existing raw `TuningProfileEvidence` plus `TuningProfilePolicy`;
3. 1–256 digest-only `ModelParityArtifact` summaries, each bound to the same
   platform and the configuration selected by tuning; and
4. the SHA-256 of each model-owned detailed parity artifact, without its path or
   contents.

Power canonicalizes report, candidate, round, and parity-artifact order. It
re-runs `compare_storage_benchmarks` and `evaluate_tuning_profile` rather than
trusting supplied summaries, requires at least two distinct storage groups with
exact output-byte parity, requires exact typed reference/tested output parity,
and rejects duplicate cases or artifacts. Storage, tuning, and model parity
must agree on the Power revision, weights, reviewed graph/source, runtime,
typed device, named environment, and selected configuration. A negative result
is preserved: retaining the baseline still produces valid review evidence.

The self-contained aggregate payload is capped at 32 MiB. `verify` replays all
derivations and checks its canonical SHA-256 after deserialization;
`verify_pinned` also requires an expected digest from a caller-owned trust root.
The digest proves mutation relative to that pin, not authorship. A signed
release, attestation, or equivalent policy must authenticate it. Construction
performs no I/O and never logs, persists, uploads, seals, or serves the bundle.
Named hardware, aggregate timing, workload/output hashes, and artifact hashes
are deliberate review data and may be correlatable, so TEE deployments retain
authority over export.

## Integrity and TEE Invariants

- `WeightStore` hashes every SafeTensors file and a deterministic aggregate
  manifest before serving it through mmap or positional reads. Model crates pin
  the aggregate digest with `verify_integrity` and may reuse Power's Ed25519
  model seal verification with `verify_signature`.
- Physical multi-root placement is configuration, not model identity. Each
  logical source has one canonical relative-file namespace; every root is
  non-empty and non-overlapping, duplicate paths fail, and the combined file
  count remains bounded. Path-free storage evidence binds only the aggregate
  `rootCount`; actual roots never enter telemetry, receipts, hardware bundles,
  or attestation claims.
- Replica selection never changes dtype, shape, bytes, routing, or precision.
  Complete canonical sources require the primary aggregate digest; partial
  canonical sources require exact per-file digests and tensor descriptors.
  Lossless sources require their own physical artifact pin plus an exact
  decoded comparison for every covered canonical tensor. Source count is
  bounded by `InferenceLimits::max_weight_sources`.
- Embedded inference does not bind a socket, start a Web server, download a
  model, invoke Python, or spawn an inference service.
- The server, API, CLI, model registry, remote clients, and Web dependencies are
  behind the default `server` feature. An embedded-only build disables default
  features, so its dependency closure contains no HTTP server/client stack and
  does not enable Tokio networking, process, or signal support.
- Existing encrypted model loading, remote attestation, privacy redaction,
  request receipts, and zeroizing sensitive request buffers remain independent
  security controls; the embedded runtime does not replace them.
- Sealed warm-state headers expose only exact digests, generation, sizes,
  export scope, and nonce. AES-256-GCM authenticates the complete fixed header
  and opaque payload. Keys and opened plaintext zeroize on drop; model-owned
  serializers must likewise clear their source buffers after sealing.
- Local sealed envelopes cannot be exported through Power's API. Hardware-TEE
  export tokens are model- and policy-bound and consume an already verified
  SEV-SNP/TDX report; structural claim matching is shared with confidential GPU
  execution and is not represented as a second signature verifier.
- Atomic state recovery ignores pending files, authenticates primary and
  backup, and obeys the caller's minimum generation. Retaining that minimum in
  a trusted monotonic source and serializing cross-process writers remain
  deployment responsibilities; encrypted files alone do not prevent rollback.
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
- Lossless tuning evidence and decisions contain opaque digests and aggregate
  counters only. Power never receives candidate values or workload content,
  never applies the selected digest, and never creates a tuning sidecar.
- Hardware evidence bundles intentionally contain path-free named-system data,
  aggregate measurements, and model-owned workload/output/artifact digests.
  Their debug view omits named hardware and detailed artifact identities. Power
  does not log, persist, upload, add them to receipts/attestation claims, or
  authorize TEE export automatically; a bundle digest is not a signature.
- Lossless representation admission and reads zeroize encoded, decoded,
  canonical-comparison, and integrity-hash heap buffers. Artifact and table
  metadata stays inside the pinned collection; typed representation and
  artifact identities may appear in explicitly requested descriptors and
  benchmark evidence. Tensor bytes, names, and paths do not appear in normal
  debug output, placement telemetry, receipts, or automatic persistence.
- Residency candidates and plans can reveal the learned hot set. Power returns
  them to the caller but never logs or persists them automatically.
- Live residency adaptations are ephemeral, non-serializable, bound to the
  active base plan, and applied through the same transactional cache path.
  Caller-supplied heat and replacement identities are not placed in telemetry,
  logs, receipts, or attestation claims.
- Hardware memory snapshots can reveal deployment capacity. Automatic budgeting
  is explicit; snapshots, runtime reservations, and pressure decisions are
  never logged or placed in telemetry or execution receipts, and callers must
  apply their own TEE export policy.
- Accelerator declarations can reveal group and tensor membership, so Power
  never logs or persists them. Normal telemetry receives no new identities.
  Receipt evidence contains declaration and claims digests only and always
  distinguishes the requested accelerator from the actual execution device.

## Colibri Adoption Boundaries

| Colibri mechanism | Power treatment |
| --- | --- |
| VRAM/RAM/storage as one hierarchy | Implemented generically with exact dtype and shape checks |
| N-drive shard split for capacity aggregation | Implemented as explicit disjoint physical roots inside one logical `WeightStore` source; stable relative files produce the same collection digest as one root, all existing integrity/read/cache/replica/mirror paths are reused, and evidence exposes only root count |
| Layer-local LFRU/LRU and learned hot pins | Implemented with decaying frequency, bounded recency, separate manual/plan pins, and transactional hot-set replacement |
| Lossless live tier adaptation | Implemented at explicit safe boundaries with caller-owned heat, default 25% + 4 hysteresis, a four-replacement cap, exact-footprint swaps, stale-plan rejection, and the existing transactional pin/cache path |
| Batched expert union | Implemented without changing router order, top-k, or gate weights |
| One-layer-ahead I/O overlap | Implemented with cancellable Tokio blocking workers, shared count/byte flight admission, useful/unused measurement, and peak-flight evidence |
| Current-layer resident/cold overlap | Implemented with atomic staged groups, event-driven issue/take readiness, exact-demand miss loading through the same admission/cache/source routing, canonical completion, and separate event/final wait evidence |
| Cross-layer coupling hints | Implemented as bounded, digest/geometry-bound, detailed-telemetry-only co-occurrence learning with deterministic per-position scores, batch union, and measured recall; hints never alter router output |
| Hardware-aware placement | Native Linux/macOS/Windows host discovery, selected CUDA/Metal device discovery, caller-owned fixed-state/scratch reservations, current-pressure revalidation, unified-memory accounting, deterministic capped budget planning, and integrity-read storage weighting are implemented without subprocesses |
| Multi-drive weighted mirrors and cache-bypass/direct I/O | Exact complete/partial replicas, usage-ranked budgeted staging, coverage-aware weighted routing, primary fallback, bounded positional reads, explicit macOS `F_NOCACHE`, and aligned Linux/Windows direct reads share one `WeightStore`; mmap remains default pending end-to-end wins |
| Cold-storage microbenchmark | Standalone path-free reports separate integrity-open, output validation, and measured demand reads; Linux cold labels require `FADV_DONTNEED` plus `mincore`, while unsupported platforms refuse the claim |
| Measured machine/model tuning | Implemented as bounded digest-only AB/BA evidence evaluation for model-owned lossless knobs; Power keeps defaults on insufficient gain, parity regression, or a winner tie and never applies or persists a profile |
| Reproducible review envelope | Implemented as one bounded canonical bundle that replays existing storage comparison and tuning evaluation, binds model-owned exact-parity artifact pins to the selected configuration and one named platform, preserves negative results, and requires an external authenticity pin; no new collector, upload path, receipt, or attestation schema is introduced |
| Lossless entropy-coded weight tier | Implemented as optional digest-pinned read-only replicas behind the existing `WeightStore`: mandatory stamps, shard-local 256-stream nibble rANS tables, exact framing/state checks, bounded zeroizing decode, full canonical byte admission, complete/proper-partial coverage, and primary fallback; canonical SafeTensors remains mandatory |
| Accelerator resident registries and fused groups | Implemented as digest-bound selections from the active device plan, atomic device-cache-only acquisition, bounded model-owned Candle execution, typed kernel-unavailable fallback, actual-device receipts, and optional canonical confidential-GPU claim binding; no second registry or kernel backend is introduced |
| Multi-device home placement and peer traffic | Implemented as a canonical typed mesh capped at 16 devices, strongly connected directed edges, synchronous bounded Candle copies, explicit backend-unavailable fallback, exact confidential GPU/NVSwitch claim-index sets, and digest-only receipt evidence; the active residency cache remains the sole weight registry and model crates own graph partitioning |
| Continuous multi-slot decode | Implemented as a model-neutral continuous/ragged lifecycle using distinct existing runtime permits, admission-order complete rosters, late-member next-step admission, bounded atomic commits, row-local cancellation, and digest-only evidence; model crates retain sequence policy, KV/recurrent topology, tensors, kernels, and arithmetic |
| Cross-session inference pooling and memory-aware microbatching | Implemented with exact model/execution/resource declarations, bounded model and shared-device queues, lazy-load deduplication, cancellation-safe slot cleanup, deterministic contiguous plans, current-pressure revalidation, unified-memory accounting, and digest-only receipt v4 evidence; model crates own slot semantics and tensor assembly |
| Persistent warm conversations | Implemented as model-neutral opaque AES-256-GCM envelopes bound to exact weights, model-owned layout, hashed state identity, generation, bounds, and export scope; synchronized primary/backup recovery adapts commit-last durability without adopting plaintext `.coli_kv` or moving KV topology into Power |
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
| 2 | Hardware/model-specific measured tuning | **Power substrate complete:** bounded digest-only evidence requires repeated baseline→candidate and candidate→baseline rounds, typed output parity, dual-order median gain, cache-hit parity, and bounded p99; model crates retain calibration, candidate application, and sealed persistence | Power tests cover binding, reversal, thresholds, malformed/duplicate/overflow evidence, deterministic selection, tie rejection, privacy-safe serialization, and `Send + Sync`; each model integration must still publish named-hardware end-to-end evidence |
| 3 | Lossless compressed expert tiers | **Power substrate complete:** optional `rans-nibble-256-v1` replicas remain behind the existing verified `WeightStore`; canonical SafeTensors is mandatory, every physical artifact is pinned before parsing, and every record is admitted only after exact canonical decode parity | Power tests cover deterministic multi-symbol round trips, stamps/tables, artifact pins, malformed framing/state/amplification, complete/proper-partial coverage, mmap/positional reads, scratch, cancellation, fallback, benchmark identity/parity, privacy-safe debug, and `Send + Sync`; model integrations must still publish named-hardware end-to-end wins before use by default |
| 4 | Accelerator-wide residency declarations and fused batches | **Power substrate complete:** exact active-plan device groups, fused/fallback implementation digests, typed actual-device evidence, permit/cancellation/limit checks, and existing confidential-GPU claim binding share one residency and receipt path; model crates own kernels | Power tests cover deterministic binding, Candle fused/reference parity, pressure and kernel-unavailable fallback identity, malformed/stale/device/permit/cancellation/limit failures, confidential-claim mismatch, privacy-safe receipts, and `Send + Sync`; each model integration must still publish named-hardware kernel parity and gains |
| 5 | Warm model state across sessions | **Power substrate complete:** model-owned opaque bytes are bounded and sealed with authenticated model/layout/state/generation/scope binding; synchronized primary/backup publication, caller-pinned rollback floors, zeroization, and explicit hardware-TEE export authorization share existing attestation and filesystem durability paths | Power tests cover round trips, fresh nonces, wrong key/model/layout/state, tamper/truncation/oversize, interruption recovery, monotonic saves, rollback floors, cancellation, export mismatch, privacy-safe debug, and `Send + Sync`; model integrations must publish uninterrupted-vs-restored output parity and named-workload warm-start gains |
| 6 | Async issue/take and staged loading rounds | **Power substrate complete:** prefetch and current-layer staging share one task, worker, item, total-byte, and in-flight-byte admission path; `next_ready_group` wakes on atomic readiness without polling, while final completion keeps canonical order | Power tests cover event wakeup, out-of-order readiness with canonical completion, cancellation/failure termination, shared materialization, byte-window scheduling, peak-flight evidence, telemetry-off privacy, serde compatibility, and `Send + Sync`; model integrations must publish end-to-end overlap gains |
| 7 | Resource planning that accounts for fixed runtime state before hot weights | **Power substrate complete:** the existing budget policy accepts typed host/device fixed-state and peak-scratch reservations, subtracts them before cache capacity, accounts for both sets once on unified memory, and revalidates a fresh native snapshot before applying cache bytes | Power tests cover checked discrete/unified arithmetic, unavailable pools, overflow/malformed input, serde compatibility, exact topology and stale-pressure rejection, runtime application, privacy boundaries, and `Send + Sync`; model integrations must still publish named-hardware peak-memory evidence |
| 8 | Heterogeneous multi-device resident pipelines | **Power substrate complete:** canonical typed meshes reuse the active residency declaration/cache, cap topology at 16 devices, require bidirectional reachability to the primary, bound every synchronous peer copy and aggregate traffic, and bind exact confidential GPU/NVSwitch claim-index sets; model crates retain graph partitioning and kernels | Power tests cover exact single-device Candle parity, canonical topology, stale/malformed/wrong-mesh rejection, per-edge and aggregate byte/count bounds, cancellation, typed fallback, exact attested claim sets, privacy-safe receipt v3, and `Send + Sync`; model integrations must still publish named-hardware parity and end-to-end gains before enabling a mesh policy |
| 9 | Continuous and ragged execution batches | **Power substrate complete:** a digest-bound lifecycle gives every member a distinct existing runtime permit, includes all boundary members exactly once in admission order, admits late members only to the next step, supports ragged position/shape metadata, and commits bounded state atomically with row-local cancellation; model crates retain sequence scheduling, KV/recurrent topology, tensors, kernels, and arithmetic | Power tests cover binding/isolation, permit and state-identity alias refusal, fair canonical rosters, late admission, ragged and aggregate bounds, cancellation, atomic retry, state accounting, digest-only evidence, and `Send + Sync`; each model integration must publish unbatched-versus-batched output parity and named-workload throughput/latency evidence |
| 10 | Reproducible hardware evidence bundles | **Power substrate complete:** one 32 MiB-bounded canonical envelope embeds raw path-free storage and tuning evidence, replays the existing comparison/evaluator, binds typed runtime/device/named-environment identities plus selected-configuration model parity artifact pins, preserves baseline-retained negative results, and performs no automatic I/O | Power tests cover canonical ordering/serde/digest pins, replayed derivations, exact storage/model parity, mixed revision/model/hardware/runtime/device/environment/configuration refusal, nested tamper, collection bounds, privacy-safe debug, negative results, and `Send + Sync`; model integrations must still publish externally authenticated bundles from reproducible named-hardware runs |
| 11 | Cache-bypass reads must remain measurable rather than presumed faster | **Power substrate complete:** explicit macOS `F_NOCACHE` extends the same integrity and positional-range handles, cannot claim direct or verified-cold I/O, and leaves mmap as default after negative PP-OCRv6 evidence | Power tests cover byte parity, cancellation, fallback, unsupported platforms, cache-claim refusal, path-free evidence, and `Send + Sync`; model integrations must retain negative results and promote a strategy only after named-hardware end-to-end wins |
| 12 | N-drive shard split can aggregate capacity without duplicating the model | **Power substrate complete:** canonical and lossless sources may span explicit disjoint roots under one logical relative-file namespace and one collection identity; the existing readers, source router, cache, partial-mirror staging, TEE binding, and hardware bundle are unchanged | Power tests cover one-root identity parity, mmap/positional loading, canonical and lossless replicas, duplicate/nested/cross-source root refusal, partial-mirror source resolution, path redaction, root-count evidence separation, serde compatibility, and existing TEE bundle validation; model integrations must publish named-volume capacity and performance evidence |
| 13 | Cross-session device sharing and memory-shaped request batches | **Power substrate complete:** exact model sessions share one bounded resolved-device gate, finite cancellation-aware model/device queues, and deterministic contiguous microbatch plans that are revalidated against live memory before execution; receipt v4 binds scheduling digests without slot identities or snapshots | Power tests cover count/byte limits, concurrent load deduplication, cancellation and future-drop cleanup, model/device queue evidence, CPU/discrete/unified-memory planning, stale pressure/topology, tamper/wrong-session refusal, serde compatibility, privacy-safe debug, and `Send + Sync`; OCR and parser integrations must still publish stage parity, partial-failure, bounded-memory, and named-hardware throughput evidence |

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
4. identical outputs with caching, prefetch, current-layer staging, and
   continuous/ragged batching disabled versus enabled;
5. cancellation, resource-limit, malformed-plan, and wrong-digest failures;
6. TEE regression tests, including telemetry-off behavior and no plaintext
   persistence; and
7. an externally authenticated `HardwareEvidenceBundle` digest tying storage,
   tuning, parity artifacts, and the named platform together.

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
model-owned integration evidence for sealed KV/routing state, and broader
cross-model benchmarks. Multi-device integrations also need controlled
single-device versus mesh A/Bs that report peer bytes/hops and negative results;
Colibri demonstrates that P2P can erase a resident-pipeline gain. These must
extend the same runtime and integrity primitives rather than introduce parallel
model-specific systems.
