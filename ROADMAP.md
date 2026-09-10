# A3S Power Roadmap

This roadmap covers the model-neutral embedded inference substrate. Model
architectures, model assets, preprocessing, decoding, OCR geometry, document
semantics, and application scheduling belong to their owning crates.

The batching workstream was reviewed against TurboOCR `main` at
`ed01c3ea2a3c7011bc361c2985215444918409b8` (release `v3.5.0`). TurboOCR is an
implementation reference, not a dependency. Power does not adopt its
TensorRT/ONNX Runtime stack, service protocols, OCR models, or OCR-specific
kernels.

## Non-negotiable boundaries

- Embedded sessions never bind a Web port or start another process.
- Every optimized path reuses the existing admission, device, residency,
  cancellation, TEE policy, attestation, and receipt mechanisms.
- Power may validate tensor layouts and aggregate resource declarations, but
  it never chooses padding, width buckets, detection thresholds, tokenization,
  decoding, page windows, or model semantics.
- Canonical weights, execution declarations, and confidential-computing claims
  remain digest-bound. A faster path may not weaken or omit them.
- Registry publication must match the tag to the crate version, then package
  and rebuild the exact lockfile-resolved crate. Release automation never
  bypasses Cargo package verification.

## Execution milestones

### P0 — Bounded execution foundation

- [x] Exact model/device session pooling with finite load and execution queues.
- [x] Deterministic, current-pressure-aware contiguous microbatch planning.
- [x] Cancellation-safe admission and digest-only receipt-v4 batch evidence.
- [x] Model-owned continuous/ragged execution lifecycles with atomic commits.
- [x] CPU, CUDA, and Metal device identity plus TEE/confidential accelerator
      evidence and explicit fallback identity.
- [x] Digest-pinned seekable AES-256-GCM weight collections with bounded
      authentication, positional decryption, cancellation, and unchanged
      `WeightHierarchy` residency semantics.
- [x] Content-address every local auxiliary inference artifact and bind its
      portable identity through strict startup, backend load, signatures,
      attestation, receipts, and caller-owned verifier policy.
- [x] Bind the resolved model-neutral server inference policy into attestation
      and request receipts, including speculative/MTP/FR controls,
      prompt-cache bounds, memory loading, threads, Flash Attention, and
      parallel slots; require independent pins on strict declared-policy paths.

### P1 — Canonical tensor batch layout

- [x] Stack compatible owned F32 tensors along the leading axis while
      preserving exact caller order and enforcing the shared tensor limit.
- [x] Split an output tensor into a complete sequence of positive leading-axis
      partitions with exact shape/value validation.
- [x] Keep padding, valid extents, bucketing, and slot failure meaning in the
      model crate. The generic API exposes no OCR vocabulary.
- [x] Release eager static-graph intermediates after their final declared
      consumer while retaining constants and the graph output.
- [x] Lower CUDA multiplier-one F32 depthwise convolution to one fused kernel
      per node, with optional fused bias and exact padded, dilated, strided,
      arbitrary-layout parity coverage. Explicit round-to-nearest arithmetic
      preserves the prior accumulation order; other layouts retain the generic
      fallback.
- [x] Fuse adjacent single-consumer F32 `HardSigmoid`-to-`Mul` CUDA pairs for
      equal rank-four tensors and exact contiguous NCHW channel gates. Preserve
      the original affine/clamp/multiply arithmetic byte-for-byte, keep the
      graph schema unchanged, and retain node-by-node fallback for every
      unreviewed device, dtype, shape, or layout.
- [x] Fuse adjacent single-consumer F32
      `Div`-`Erf`-`Add`-`Mul`-`Mul` CUDA chains with scalar initializers. Capture
      scalar values once at model load, preserve all five rounding boundaries
      byte-for-byte, and retain ordinary execution for every unmatched graph,
      device, dtype, layout, or shared value.
- [x] Fold exact contiguous NCHW channel-bias `Add` nodes into reviewed CUDA
      ReLU, error-function GELU, and gated HardSigmoid activation windows after
      two-input convolutions. Keep the convolution backend unchanged, require
      private bounded intermediates, use launch-bounded 32-bit indexing, and
      retain byte-exact arithmetic plus node-by-node fallback.
- [x] Fuse exact private decomposed LayerNorm
      `Add(epsilon)`-`Sqrt`-`Div`-`Mul(scale)`-`Add(bias)` CUDA tails while
      retaining the original reductions, centering, and squaring. Require
      exact last-axis broadcast shapes, preserve every pointwise F32 rounding
      boundary byte-for-byte, and keep node-by-node fallback for all
      unreviewed topology, devices, dtypes, or layouts.
- [x] Preserve canonical v1 tensor and token receipt bytes while hashing
      contiguous little-endian inputs directly, with a bounded canonical
      staging fallback on big-endian hosts.
- [x] Allow a model-owned deterministic output projection to execute on the
      graph device before bounded host materialization. The caller must bind
      the projection into its execution identity; Power retains permit,
      cancellation, same-device, dtype, finite-value, and tensor bounds.
- [x] Add benchmark evidence for allocation count and host-copy cost on named
      hardware before claiming a throughput improvement. The generic Windows
      CPU/RTX 4090 fixture capture retains raw paired samples, exact output
      parity, distinct runtime-artifact pins, and the CPU negative result.

### P2 — Shape-profile execution evidence

- [x] Add a model-owned, digest-bound shape-profile declaration for a finite
      set of batch/shape classes and an explicit dynamic fallback.
- [x] Record selected profile identity and fallback reason without exposing
      tensor values, source identities, or model-private geometry.
- [x] Reject stale profiles when weights, graph identity, device topology,
      scratch bounds, or TEE policy change.

This adapts TurboOCR's useful static `(batch, width)` profile discipline without
importing TensorRT profiles or moving shape selection into Power.

### P3 — Bounded replicas and deadline-aware admission

- [x] Allow a policy-bounded number of independently mutable session replicas
      for one exact model identity while retaining one shared device gate and
      resident-byte budget.
- [x] Add monotonic admission deadlines, queue-expiry evidence, and
      cancellation-safe cleanup. No request bytes or slot identities enter
      telemetry.
- [x] Add health-driven replica retirement and lazy reconstruction at a safe
      request boundary; do not introduce an OCR-local watchdog or pool.

This is the model-neutral counterpart of TurboOCR pipeline replicas,
deadline-drop, and recycle behavior.

### P4 — Device-resident batch boundaries

- [x] Add bounded device-resident input/output handles for adjacent reviewed
      graph calls, with exact dtype/shape/device validation and owned fallback
      copies.
- [x] Preserve cancellation checks and receipt digests across fused or retained
      buffers.
- [x] Expose only generic reviewed operators. OCR resize/normalize, ROI warp,
      DB postprocessing, and CTC decoding remain in A3S OCR.

### P5 — Confidential performance release gate

The model-neutral, fail-closed evidence schema, platform-specific profile/TEE
bindings, strict four-platform policy, and isolated complete-contract collector
are implemented. Clean-revision Windows CPU/CUDA captures now replay the full
contract and form one verified two-platform partial bundle. A checked-in
external capture runbook now covers named Metal collection and nonce-bound,
proof-backed confidential-GPU promotion while preserving raw vendor evidence.
The strict bundle builder now derives the four platform bindings and writes the
bundle and digest pin as one no-overwrite operation. The release workflow uses a
machine-checked source-parent/evidence-child protocol so checked-in evidence can
bind the source revision without a self-referential commit hash. Operational
release state is determined by the tagged evidence child rather than a mutable
roadmap checkbox: an absent or invalid four-platform bundle blocks publication.

- [x] Drive the real resident graph, cancellation lifecycle, bounded queue,
      replica pool, and explicit shape fallback from both a generic calibration
      fixture and any caller-owned reviewed graph with an independent typed
      reference output.

- [x] Make exact-revision CPU, Metal, CUDA, and supported confidential-GPU
      captures a machine-enforced publication prerequisite. The tagged evidence
      child records whether a concrete release satisfies it.
- [x] Publish clean-revision CPU and CUDA complete-contract captures with raw
      JSON, byte-stable policy input, exact artifact hashes, negative results,
      and replay commands.
- [x] Require an opaque successful strict-verification proof when promoting a
      local CUDA capture to confidential-GPU evidence; the proof borrows the
      exact authenticated report, and a raw report or caller label cannot mint
      the production security class.
- [x] Preserve the verifier-pinned inference-execution policy and optional
      auxiliary-artifact set as explicit confidential release fields, and
      reject missing, malformed, or mutated digests during bundle replay.
- [x] Preserve the accepted 48-byte CPU TEE launch measurement and SHA-256 of
      the exact raw signed report as explicit confidential release fields.
- [x] Add the external Metal/confidential-GPU capture runbook that collects vendor
      evidence, invokes strict proof-backed promotion, and preserves the raw
      report plus trust-root material for the same immutable release revision.
- [x] Unify SafeTensors startup, attestation, embedded-runtime, and accelerator
      declaration digests; add deterministic persistent fixture weights and an
      atomic local-CUDA-source/declaration capture command for cross-host proof.
- [x] Assemble typed four-platform captures with a machine-enforced CLI that
      derives the strict policy, verifies the exact version/revision, writes a
      create-new bundle and digest pin, and attaches both to non-`0.x` releases.
- [x] Expose a bounded single-capture verifier that rejects digest corruption,
      platform relabeling, and revision drift before cross-host bundle assembly
      without overclaiming strict four-platform release eligibility.
- [x] Bind every staged cross-host artifact into an exact portable file
      inventory with bounded streaming hashes. Reject missing, added, mutated,
      traversing, symlinked/reparse, or relabeled handoffs while keeping
      authorship and four-platform eligibility as separate gates.
- [x] Require non-`0.x` tags to point to a single-parent evidence commit whose
      complete diff adds only the bundle and pin; build and publish artifacts
      from its frozen source parent. The child must be reachable from `main` and
      the annotated tag must have a GitHub-verified signature.
- [x] Add one pre-tag candidate command that fails closed on a dirty checkout,
      pre-v1 or pending changelog state, invalid evidence-child layout, missing
      main containment, or failed strict four-platform bundle replay. Release CI
      uses the same command instead of duplicating the semantic gate inline.
- [x] Implement reviewed Intel DCAP Quote generation and QVL verification, or
      explicitly exclude TDX from the v1 production support matrix. A local
      TDREPORT now fails closed and is not treated as a PCK-signed Quote. The
      strict v1 bundle verifier additionally requires a typed SEV-SNP binding.
- [x] Require every release capture to replay scalar/batch numerical
      equivalence, bounded peak host/device memory, cancellation, queue expiry,
      replica recovery, and explicit fallback.
- [x] Bind benchmark artifacts to weights, graph declarations, runtime/device,
      TEE policy, and build revision. Third-party headline numbers are never
      reused as A3S measurements.
- [x] Separate the active Qwen3.8 Q6_K quality profile from archived
      mixed-quantization and external-draft experiments. The default repeated
      matrix resolves both control and optimized modes to one unchanged Q6_K
      artifact and exposes its resolved contract before model loading.
- [x] Keep long Windows quality captures under continuous process-level GPU
      observation after admission. A new foreign PID above the configured SM
      threshold invalidates the evidence and is retained in the environment
      receipt instead of being mistaken for an inference regression.
- [x] Publish a clean, path-free Q6_K-only 3x100 quality package that pins the
      target, source, tools, reports, task/config inputs, and continuous GPU
      monitor logs; keep its lossless-default gate closed when output or strict
      score parity is not established.
- [x] Add a typed native DFlash2 binding and reviewed pinned-llama.cpp runtime
      port, then publish a clean paired capture with the unchanged Q6_K target,
      exact artifact and binary identities, idle/runtime GPU telemetry, five
      deterministic outputs, and offline comparison. Keep DFlash2 opt-in until
      representative complete-output parity and cross-model evidence pass.

### P6 — Distributed serving execution boundary (in progress)

This milestone preserves the useful execution-side outcomes of distributed
inference systems without making Power a router, deployment controller, or
model-semantics owner.

- [x] Publish one versioned, model-neutral worker capability and observation
  contract over the existing Service health/metrics boundary. It covers phase
  readiness, admission queue depth, active execution, prompt-cache
  occupancy/pressure, transfer health, observation generation and age without
  exposing prompts, tokens, KV bytes, tenant identity, or unbounded labels.
- [x] Publish one fail-closed, service-authenticated internal request-flow
  protocol for decode preparation, prefill execution, decode execution, and
  idempotent abort. Bind every call and stream frame to the process epoch and
  immutable execution profile, preserve backpressure through versioned NDJSON,
  and pin the closed JSON shapes with golden fixtures.
- [x] Accept an immutable `aggregated` or `prefill-decode` execution profile only
  through closed A3S ACL. Power validates the exact model, backend, device,
  layout, peer, generation, byte, time, cancellation, privacy and attestation
  bindings before a phase or state-transfer operation.
  The closed static profile, canonical digest, transfer and phase-executor
  bindings, fail-closed startup gate, and health-gated observation projection
  are implemented.
- [ ] Reuse the existing bounded admission, session replicas, weight hierarchy,
  sealed-state envelope, telemetry and receipt mechanisms. Device/host/local
  storage and peer tiers must not create a second cache or persistence format.
  A process-bound transfer lifecycle now enforces fail-fast capacity,
  idempotent leases, content-free counters, monotonic expiry, bounded abort and
  fail-closed cleanup health around every injected data-path adapter. The
  wrapper also accounts declared adapter-owned registration bytes per lease,
  rejects a second lease for an already-registered opaque handle, and reclaims
  that registration on consume/abort/timeout without copying KV into Power.
  Inflight capacity for both `BoundedStateTransferService` and
  `DistributedServingRuntime` now reuses the shared `AdmissionController` in
  fail-fast mode (`waiting_limit == 0`) bound to the ACL
  `max_inflight_transfers` limit; construction refuses a waiting queue or
  mismatched active limit as a second admission policy, and capacity rejections
  project from that controller. When a matching runtime is composed, worker
  observation projects that same fail-fast phase admission snapshot rather than
  inventing a second capacity story from the HTTP `max_concurrent_requests`
  limiter; generation stays monotonic after cancel/taint while readiness
  clears.   When `embedded-inference` is enabled,
  host-buffered transfer payloads reuse `SealedStateEnvelope` via
  `seal_transfer_host_buffer` / `open_transfer_host_buffer` (domain-separated
  binding over transfer identity, profile generation, and privacy /
  attestation policy digests — `transfer-host-buffer.v2`). Wire tickets remain
  opaque adapter metadata by design and are intentionally not force-sealed with
  `SealedStateEnvelope` (host buffers seal; tickets do not). Descriptor
  validation fail-closes if tickets carry sealed-model-state schema or envelope
  MAGIC (ASCII / Base64 / hex), control characters, or oversized payloads so
  tickets cannot become a second persistence or KV-byte channel.
  DistributedServingRuntime and the
  phase-executor port do not construct a second session-replica pool or weight
  hierarchy; the immutable prefill/decode profile and phase-executor
  capabilities now bind `weight_cache = shared-weight-hierarchy` (unknown
  private-cache identities fail closed) and may pin `residency_policy_sha256`.
  When `embedded-inference` is enabled, `validate_weight_hierarchy` requires
  that digest to match `ResidencyPolicy::sha256` before a process hierarchy is
  accepted. Prefill/decode profiles also bind
  `session_pool = shared-session-pool` (unknown private-pool identities fail
  closed) and may pin `session_pool_policy_sha256`. When `embedded-inference` is
  enabled, `validate_session_pool` requires that digest to match
  `ModelSessionPoolPolicy::sha256` before a process pool is accepted, so P/D
  cannot mint a second session-replica pool.   Matching distributed runtimes now
  project content-free transfer and fail-fast phase-admission counters through
  the existing Service `GET /metrics` text format as label-free series (no
  transfer/execution/tenant/model labels); aggregated profiles omit them.
  Optional digest-only `DistributedOperationEvidence`
  (`a3s.power.distributed-operation.v1`) domain-separates a validated
  `StateTransferReceipt` without folding transfer proofs into microbatch
  receipt-v4. State-transfer target/source/receipt schemas are now
  `a3s.power.state-transfer-*.v2` and carry `ServingDeploymentIdentity`
  (`generation`, `peer_set_sha256`) so peer publish/consume fail closed on
  stale Cloud deployment generation or foreign peer set even when model /
  execution / layout bindings still match; process epoch alone is not enough.
  Wire tickets stay opaque adapter metadata by design and are intentionally
  **not** force-sealed with `SealedStateEnvelope` (host buffers seal; tickets
  do not); descriptor validation fail-closes on sealed-model-state schema /
  MAGIC (ASCII, Base64, hex), control characters, and oversized tickets.
  Production-adapter / high-speed-transport evidence and full session-replica
  lifecycle reuse under live P/D execution remain open. A
  request-level runtime now composes that lifecycle with phase execution under
  one bounded execution lease and is the server's single source of distributed
  readiness.   A deterministic conformance test launches independent prefill and
  decode Power processes with ACL
  `serving_execution.transport = "buffered-host-loopback"`, installs the
  product `BufferedHostLoopback` pair (not a test-only fixture adapter), moves
  opaque conformance state over the product authenticated encrypted loopback
  data path, and verifies the public HTTP lifecycle (including stale Cloud
  deployment generation / foreign peer-set rejection, peer loss, and restart)
  as loopback conformance only—not HSN or model-semantic evidence.
- [ ] Keep tokenization, KV/recurrent layout, serialization, phase arithmetic and
  semantic parity in the owning model/backend adapter. Power moves only opaque,
  bounded authenticated state and never claims a cache hit or successful
  decode from transport completion alone.
  Authenticated unit and HTTP evidence now proves a successful transfer
  consume/receipt followed by non-`Ready` `execute` returns a typed decision
  (never an NDJSON token stream) and runs compensating cleanup; production
  backend phase executors and high-speed transport remain open. An injectable
  product-surface buffered-host loopback transfer adapter now exists for
  software composition; it does not close model-semantic ownership.
- [ ] Report a typed recompute, retryable-unavailable, or terminal-failure outcome
  before response generation. Endpoint choice, flow control, request replay,
  desired replicas, placement, rollout and autoscaling remain Gateway or Cloud
  responsibilities.
  The closed pre-response decision contract and Gateway-facing Power endpoint
  are implemented. Cross-process orchestration now has executable success,
  peer-loss, cleanup, restart and stale-epoch evidence. Post-consume
  `Recompute` / `RetryableUnavailable` / `TerminalFailure` mapping over the
  authenticated HTTP boundary is covered by first-principles fixture tests;
  concrete production backend phase executors remain open (buffered-host
  loopback transfer injection is available without claiming backend readiness).
  A matching product-surface `BufferedHostLoopbackPhaseExecutor` now pairs with
  the loopback transfer for opaque conformance composition; it still does not
  close model-semantic or HSN readiness. A separate product-surface
  `BackendOwnedPhaseExecutor` (`phase_executor = backend-owned` with buffered-host
  transport) is the Injected port for real layout/KV binding: Empty ownership
  stays Unavailable; matching `BackendPhaseStateOwnership` becomes Eligible
  after fail-closed layout validation, still without Ready execute.
  `ProfileBoundBackendPhaseStateOwnership` is the interim digest-only bind
  surface (not a real backend). ACL `state_ownership = profile-bound` with
  `phase_executor = backend-owned` installs it at composition (Eligible,
  never Ready / never advertised). Default Empty ownership remains
  Unavailable. It does not close that checkbox.
- [ ] Require real high-speed-network, cancellation, peer loss, stale generation,
  corrupt state, resource pressure, process restart and cleanup evidence before
  advertising cross-node or prefill/decode support. The product-pair loopback
  conformance suite covers peer loss, process restart, stale process epochs,
  stale Cloud deployment generation / foreign peer set over the HTTP
  orchestrator boundary, and graceful cleanup, but it is not high-speed-network
  or model-semantic evidence.
  First-principles fixture evidence now covers corrupt authenticated
  ticket/receipt bytes and resource-pressure / in-flight capacity / admission
  pressure fail-closed outcomes (typed `InvalidRequest`,
  `RetryableUnavailable`, or equivalent `BackendNotAvailable`) with
  compensating cleanup. The distributed runtime now applies the same
  caller-cancel and deadline abort contract to in-flight transfer
  prepare/publish/consume as to phase work; fixture evidence covers mid-transfer
  and mid-stream abort without Ready/NDJSON success, with compensating cleanup
  and reclaimed leases. Peer publish/consume fail closed on stale deployment
  generation or foreign peer set carried by transfer descriptors (beyond
  process-epoch checks), including cross-process HTTP evidence that prefill
  rejects a tampered target and decode refuses a tampered source without
  Ready/NDJSON.   High-speed-network transport and production adapters remain
  open. A typed `ProductionAdapterContract` now documents the required adapter
  memory ownership (`AdapterOwnedRegistration`), transport integrity, and
  confirmed-reclaim cleanup obligations; Empty/Unavailable
  `EmptyStateTransferService` / `EmptyServingPhaseExecutor` placeholders refuse
  work and fail composition, bounded-transfer wrap, and distributed-runtime
  construction until a concrete adapter is injected. Matching runtimes also
  require Injected provision plus the required production contract inside
  `accepts_work`, so worker observation never lists prefill/decode readiness
  for Empty or non-required contracts. That contract strengthens the P6
  injection boundary only and is not high-speed-network or production-adapter
  evidence. Power now also ships an injectable product-surface
  `BufferedHostLoopbackStateTransfer` for profiles that pin
  `BufferedHostMemoryPullV1` and `AuthenticatedEncryptedTransport`: opaque
  adapter-owned host buffers move over authenticated AES-GCM loopback TCP under
  `Injected` + `ProductionAdapterContract::REQUIRED`, with confirmed abort
  reclaim. This is a real composition path (also exercised by the cross-process
  ACL-transport conformance suite) and still not high-speed-network or
  model-backend evidence; concrete production phase executors
  and HSN remain open. Aggregated defaults still refuse transfer injection and
  never advertise P/D. Power now also ships a matching product-surface
  `BufferedHostLoopbackPhaseExecutor` that pairs with the loopback transfer
  (`paired_for_profile` / `pair_with`), owns opaque conformance fixture handles
  (not model-semantic layout), and refuses Ready decode until adapter-owned
  bytes verify after consume (`Recompute` on missing/corrupt). Transfer-only
  or Empty-phase compositions still fail closed. This completes an injectable
  product pair for buffered-host loopback conformance composition only; HSN
  and real backend/llama.cpp P/D remain open. Power now also ships a named
  product-surface DirectDeviceMemoryPull pair
  (`DirectDeviceMemoryPullStateTransfer` +
  `DirectDeviceMemoryPullPhaseExecutor`, ACL
  `transport = "direct-device-memory-pull"`) that pins
  `DirectDeviceMemoryPullV1`, reports Injected + required contract so
  composition can install it instead of Empty, and stays Unavailable with
  refused data-path work until a real high-speed adapter is bound.
  `accepts_work` / worker `ready_phases` refuse to advertise P/D for that
  composition transport. This is the HSN product port, not HSN evidence:
  cancellation, peer loss, stale generation, corrupt state, resource pressure,
  process restart, and cleanup still need a real adapter on a high-speed path
  before this checkbox can close. Force-sealing wire tickets with
  `SealedStateEnvelope` is an intentional non-goal (opaque metadata + fail-closed
  sealed-persistence rejection; host buffers seal instead).
  ACL/composition now accepts honest opt-ins
  `serving_execution.transport = "buffered-host-loopback"` (or
  `PowerServerBuilder::with_buffered_host_loopback_transport`) and
  `serving_execution.transport = "direct-device-memory-pull"` (or
  `with_direct_device_memory_pull_transport`). Loopback wires the working
  product pair when protocol/privacy match. DirectDeviceMemoryPull wires the
  Unavailable HSN product port when protocol is `DirectDeviceMemoryPullV1` and
  never advertises P/D. Protocol alone never auto-wires; incomplete pairs and
  builder+transport mixes fail closed. Aggregated defaults still advertise no
  P/D and make no HSN claim.
  Cross-process distributed-serving conformance now loads that same ACL
  transport opt-in and exercises the product pair end-to-end (success stream,
  peer-loss, restart, stale deployment / peer-set) instead of a test-only
  fixture adapter; claims remain loopback conformance only.
  Power now also ships a product-surface `BackendOwnedPhaseExecutor` for the
  open concrete backend phase path: ACL
  `transport = "buffered-host-loopback"` + `phase_executor = "backend-owned"`
  (or `with_backend_owned_phase_on_buffered_host_loopback`) installs Ready
  buffered-host loopback transfer with an Injected + required-contract phase
  port. Default [`EmptyBackendPhaseStateOwnership`] keeps health Unavailable.
  Binding a non-Empty [`BackendPhaseStateOwnership`] validates profile
  `layout_sha256` via opaque `state_layout_sha256` (plus optional related
  model/backend/execution digests) fail-closed before becoming Eligible;
  mismatched layout fails closed at bind time. Eligible still refuses Ready
  prepare/execute until a real execute adapter path exists—matching layout
  registration alone is never cache-hit or decode success, and
  `accepts_work` / `ready_phases` never advertise P/D. Import/export hooks on
  the trait are opaque byte↔handle only; this does not invent llama.cpp KV
  semantics.   Power now also ships
  [`ProfileBoundBackendPhaseStateOwnership`]: an honest interim product
  surface that mirrors exact closed profile digests (layout, model, closed
  backend artifact, execution) so the backend-owned executor can bind to
  Eligible without a real KV owner. ACL opt-in
  `state_ownership = "profile-bound"` (with `phase_executor = "backend-owned"`)
  wires that surface at composition; absent keeps Empty → Unavailable.
  Opaque import/export on that surface fail closed; it is **not** a
  llama.cpp / picolm ownership adapter and does not claim model-semantic P/D.
  Eligible still refuses `accepts_work` and worker `ready_phases` never
  advertise P/D. Power now also ships product-surface
  [`BackendPhaseExecution`]: default [`EmptyBackendPhaseExecution`] keeps
  Eligible refusing Ready. ACL opt-in `phase_execution = "pending"` (with
  `phase_executor = backend-owned`) installs
  [`PendingBackendPhaseExecution`] so Eligible ownership can advance to Ready
  health and delegate prepare/execute. Pending unlocks Ready health only;
  prepare/execute/abort still fail closed and do not invent KV or claim
  model-semantic decode. Backend-owned composition continues to suppress
  worker `ready_phases` via `may_advertise_prefill_decode`. This advances
  the named backend phase product port toward real backends; concrete
  llama.cpp / picolm layout + execute implementors remain open.
  The product loopback transfer AAD (v2) now also binds privacy mode,
  `privacy_policy_sha256`, and optional `attestation_policy_sha256` so peers
  with matching model/layout bindings but mismatched privacy or attestation
  policies fail closed on consume; this is product-pair wire binding, not HSN
  or attested-fabric readiness. The sealed host-buffer helper
  (`transfer-host-buffer.v2`) binds the same privacy/attestation digests into
  the `SealedStateEnvelope` state-id so reopen with a drifted policy fails
  closed without claiming attested fabric or TEE-export readiness.

### P6 open-checkbox exit criteria (not claimed here)

Each open checkbox closes only when its evidence below exists. Interim
product ports (`profile-bound`, `phase_execution = pending`, Unavailable
HSN, AAD/host-buffer attestation digests) never close a checkbox alone.

**Reuse (admission / replicas / weight hierarchy / sealed envelopes /
telemetry / receipts):** closes when a concrete llama.cpp or picolm
`BackendPhaseExecution` drives live P/D under the already-bound shared
session-pool and weight-hierarchy ports, with process-bound transfer
leases and digest-only receipts on that path. Software reuse of the ports
is already bound; live executor lifecycle evidence is not.

**Opaque state (tokenization / KV / layout stay model-owned):** closes
when a real `BackendPhaseStateOwnership` implementor imports/exports
opaque adapter-owned state bytes for llama.cpp or picolm (layout digest
match + byte hooks only in Power). `ProfileBoundBackendPhaseStateOwnership`
mirrors closed profile digests to Eligible without KV and does **not**
close this checkbox. Power must not invent a second KV format.

**Typed outcomes (recompute / retryable-unavailable / terminal-failure
before response generation):** closes when that same concrete
`BackendPhaseExecution` produces Ready prepare/execute (or typed
non-Ready decisions) over the authenticated HTTP boundary after transfer
consume. `EmptyBackendPhaseExecution` keeps Eligible from Ready;
`PendingBackendPhaseExecution` unlocks Ready health only and fails closed
on prepare/execute—neither closes this checkbox. Gateway/Cloud retain
placement and autoscaling.

**HSN evidence (cross-node / prefill-decode advertisement):** closes when
a real DirectDeviceMemoryPull adapter on a high-speed path proves
cancellation, peer loss, stale generation, corrupt state, resource
pressure, process restart, and cleanup, and only then may
`may_advertise_prefill_decode` / `accepts_work` / worker `ready_phases`
list P/D for that transport. The named Unavailable product port is not
that evidence. Buffered-host loopback remains conformance-only.

**Attestation (attested-private-fabric readiness):** product-pair AAD and
sealed host-buffer digests already bind optional
`attestation_policy_sha256` fail-closed. Closing attested-fabric
readiness still requires TEE-export / fabric attestation evidence beyond
digest wire binding; mismatched-policy consume rejection alone does not
close it.

**Composition fail-closed (always required, not a checkbox closer):**
`phase_execution = pending` + `state_ownership = profile-bound` under
`phase_executor = backend-owned` may reach executor Ready health, but
`may_advertise_prefill_decode` stays false, so
`DistributedServingRuntime::accepts_work` stays false and worker
`ready_phases` never list prefill/decode.

## Cross-repository delivery order

1. Power publishes model-neutral execution contracts.
2. A3S OCR pins that revision and owns PP-OCRv6 batch assembly and geometry.
3. A3S Parser pins the compatible OCR revision and owns document/page windows,
   persistence, reconciliation, and overlays.
4. A3S Cloud pins a certified Power Service profile; A3S Gateway consumes only
   the matching bounded observations and never imports Power internals.

No milestone is complete merely because another repository can emulate it with
a second scheduler, cache, pool, or receipt format.
