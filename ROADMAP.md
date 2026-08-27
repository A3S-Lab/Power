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
- [x] Give each CUDA model-session replica one isolated Candle device identity
      and stream, and omit redundant per-activation cross-stream events only
      inside that host-bounded lane. Keep ordinary runtime and accelerator-mesh
      event tracking, with a real-device boundary test. A downstream 29-page
      RTX 4090 full-stage gate improved from 4,030.936 to 3,672.701 ms median
      with exact text and semantic fingerprints.
- [x] Bind one lifetime-owned, 256-byte-aligned cuBLAS workspace to every CUDA
      model-session device: 32 MiB for Hopper compute capability 9.x and 4 MiB
      for other supported architectures; general CUDA runtimes retain the
      vendor default pool. Fail session construction on allocation or binding
      error. Teardown synchronizes the same stream and resets the handle before
      freeing the buffer, including an escaped-device reuse regression. Focused
      real-CUDA binding passes, and two independent 64-page OCR processes are
      exact to the same-source static-session control without a process-global
      workspace setting. Named-hardware throughput remains a downstream gate.
- [x] Deterministic, current-pressure-aware contiguous microbatch planning.
- [x] Cancellation-safe admission and digest-only receipt-v4 batch evidence.
- [x] Model-owned continuous/ragged execution lifecycles with atomic commits.
- [x] CPU, CUDA, and Metal device identity plus TEE/confidential accelerator
      evidence and explicit fallback identity.
- [x] Digest-pinned seekable AES-256-GCM weight collections with bounded
      authentication, positional decryption, cancellation, and unchanged
      `WeightHierarchy` residency semantics.

### P1 — Canonical tensor batch layout

- [x] Stack compatible owned F32 tensors along the leading axis while
      preserving exact caller order and enforcing the shared tensor limit.
- [x] Split an output tensor into a complete sequence of positive leading-axis
      partitions with exact shape/value validation.
- [x] Keep padding, valid extents, bucketing, and slot failure meaning in the
      model crate. The generic API exposes no OCR vocabulary.
- [x] Release eager static-graph intermediates after their final declared
      consumer while retaining constants and the graph output.
- [x] Fold private constant contiguous tensor `Reshape` views once at executor
      construction after full validation. Preserve graph outputs, scalar
      constants, tensor limits, and runtime fallback; select only from topology,
      constant availability, layout, and resource bounds. The current reviewed
      layout trace removes exactly 16 of 28 executed reshapes and exposes the
      existing convolution channel-bias fusion with exact 64-page Parser/Office,
      table-cell, and seal-position parity. Stable timing remains open: the
      first comparison used unequal tracing and the first strict trace-free
      cohort stopped with zero samples when unrelated compilers appeared.
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
- [x] Lower exact private CUDA F32 sigmoid products without a graph- or
      model-specific selector. Fuse equal-shape `Sigmoid -> Sigmoid -> Mul`
      triples, or reuse an already materialized same-shape, NCHW per-channel, or
      NCHW per-spatial multiplier while fusing the full-shape `Sigmoid -> Mul`
      pair. Generic offset/broadcast tests are byte-exact; the current official
      layout graph removes four Mul execution boundaries per call, and the
      five-document/64-page Parser cache plus all 74 Office artifacts remain
      exact. Stable end-to-end timing remains open because the first post-parity
      resource snapshot exceeded the strict GPU-idleness gate.
- [x] Apply an exact private adjacent F32
      `BatchNormalization -> Sigmoid` edge in the CPU or CUDA normalization
      output pass. Preserve Swish precedence, require one private consumer, and
      do not extend or reorder convolution. Full graph suites pass; the official
      layout graph removes four more execution boundaries per call with exact
      64-page cache and 74-artifact parity. The unguarded 10.665-pages/s
      diagnostic is not a stable performance claim.
- [x] Prepare static BatchNormalization `[mean, stddev]` channel statistics once
      per graph executor. Preserve the original CPU F32 expression and CUDA
      round-to-nearest addition plus `sqrtf`, then reuse the result in ordinary,
      depthwise-convolution, and spatial-convolution output paths without
      changing `sub -> div -> mul -> add`. Focused real-GPU tests, complete
      CPU/CUDA graph suites, five normalized 64-page caches, and all 74 Office
      artifacts are exact. The unguarded 9.741-pages/s correctness run is not a
      performance result; the frozen fixed-order quiet-host A/B remains open.
- [x] Evaluate and reject reuse of a fresh private bias-free CUDA F32 pointwise-
      convolution output for adjacent BatchNormalization. One mutable values
      pointer correctly records the cross-stream write, and focused CUDA,
      complete CPU/CUDA, 64-page OCR, and 74-file Office gates are exact. After
      two fully discarded contaminated attempts, the 300-second quiet-qualified
      eight-run A/B measured control/candidate means of 6,185.25/6,213.5 ms,
      medians of 6,146.5/6,268.5 ms, and two candidate wins in four pairs.
      Production graph dispatch keeps the ordinary two-output path globally;
      retain only the mutable-argument event contract and its regression.
- [x] Fuse adjacent single-consumer F32
      `Div`-`Erf`-`Add`-`Mul`-`Mul` CUDA chains with scalar initializers. Capture
      scalar values once at model load, preserve all five rounding boundaries
      byte-for-byte, and retain ordinary execution for every unmatched graph,
      device, dtype, layout, or shared value.
- [x] Fuse an exact private contiguous F32
      `BatchNormalization -> Div -> Erf -> Add -> Mul -> Mul` window into one
      CPU or CUDA output pass. Require constant exact-channel parameters,
      finite scalar activation initializers, exact formula-local use counts,
      and no retained or shared intermediate; preserve every normalization and
      activation rounding boundary and retain node-by-node fallback otherwise.
- [x] Let an exact rank-three CUDA F32 last-two-axis transpose view feed a
      contiguous rank-two matrix directly through strided GEMM. Match only
      device, dtype, rank, contiguity, nonzero compatible dimensions, and the
      exact transpose stride; retain materialization otherwise. Bitwise tests
      cover three unrelated geometries. The OCR recognition probe removes all
      386 matching transpose launches, while nine interleaved real-document
      samples per side show only a contention-sensitive 0.51% mean reduction,
      so no stable throughput claim is attached. The downstream strict semantic
      golden remains unchanged and open.
- [x] Fuse an exact private contiguous CUDA F32
      `Add(last-axis bias)`-`Sigmoid`-`Mul` formula into one pass. Match only
      normalized topology, use counts, rank, exact last-axis geometry, device,
      dtype, layout, cancellation, and declared bounds; retain ordinary
      execution otherwise. Generic unrelated-geometry tests are bit-exact. A
      launch-blocked OCR trace removes 284 dynamic pointwise launches, and two
      precommitted independent interleaved cohorts improve separately. Ten
      29-page samples per binary reduce combined mean latency by 4.46% and
      median latency by 3.14%, from 9.337 to 9.757 mean pages/s, with identical
      2,518-block text and semantic fingerprints. Stable 10-pages/s and the
      downstream semantic-golden gates remain open.
- [x] Reuse the output of an exact private CUDA F32 `MatMul` for a rank-one
      last-axis bias and compose the retained Swish tail when present. Match
      only topology, liveness, arbitrary nonempty prefix geometry, contiguous
      layout, same-device facts, cancellation, and bounds. Generic rank-two
      through rank-four and storage-offset parity is byte-exact. Eight internal
      recognition windows remove eight allocation/free pairs without reducing
      the preceding baseline's kernel count. Ten 29-page samples per binary
      improve combined mean latency 1.56%, but the reverse cohort improves only
      0.34% by mean and regresses 0.66% by median; the isolated-graph median also
      regresses. Retain as deterministic work removal, not stable 10-pages/s
      evidence, and record the superseded non-composing attempt as a negative
      result.
- [x] Preserve leading-axis CUDA reduction reproducibility for direct F32
      pointwise and spatial convolution by partitioning only their batched GEMM
      phase into fixed at-most-32 groups. Spatial im2col executes once and every
      group writes directly into one final allocation. A full 481-node
      batch-128 probe compares 95,795,200 values across all four partitions
      with zero differing bits. Do not infer an OCR batch default from this
      executor quantum: a one-pass `64..256` sweep was non-monotonic, all 41
      explicit cuBLAS algorithms failed exact parity, and a four-lane follow-up
      regressed complete-stage throughput before being reverted.
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
- [x] Execute contiguous CPU F32 group-one pointwise/spatial and multiplier-one
      depthwise convolutions through direct NCHW kernels with explicit
      topology, dtype, layout, geometry, tensor-limit, and fallback gates.
- [x] Apply exact private pointwise/spatial channel-bias plus ReLU/GELU
      post-operations in the convolution output buffer, preserving graph
      arithmetic and single-consumer liveness.
- [x] Apply an exact private CPU `Conv -> BatchNormalization` edge, with its
      optional private HardSwish tail, in the direct pointwise, spatial, or
      depthwise output buffer. Match only topology, single-consumer liveness,
      constant F32 channel parameters, layout, geometry, device, tensor limits,
      and cancellation; preserve convolution bias and all graph rounding
      boundaries. Bitwise operator coverage plus real table, seal-position,
      seal-precision, and bidirectional Text/Seal A/B gates retain the semantic
      baseline without a model, corpus, content, or measured-shape selector.
- [x] Fold every exact private convolution channel-bias add into the existing
      convolution bias path, including group-one forms without an activation.
      For direct CPU pointwise convolution, also fold one exact-shape private
      residual add into the same output buffer. Preserve
      `(convolution + bias) + residual`, reject broadcasting and shared edges,
      and select only from topology, use counts, dtype, layout, geometry,
      device, cancellation, and resource limits. Bitwise geometry tests and a
      content-free small recognition-graph probe retain the exact output while
      moving the recorded batch-16 median from 370.671 ms to 323.096 ms.
- [x] Avoid nested full-pool pointwise GEMM/post work when the graph already
      runs on a Rayon worker, with bitwise standalone/outer-parallel coverage;
      remove only the redundant second clear of fresh depthwise output.
- [x] Enable `gemm`'s x86-v4 runtime microkernel dispatch. AVX-512F is selected
      only by CPUID; unsupported hosts retain the existing FMA/SIMD/scalar
      path, and builds remain portable rather than using `target-cpu=native`.
      Bidirectional same-binary Text, Table, and Seal A/B runs improve while
      retaining the same semantic evidence.
- [x] Vectorize contiguous stride-one CPU depthwise interiors with one
      AVX2/FMA eight-output row accumulator when the host exposes both ISA
      features. Preserve each lane's scalar FMA traversal, keep the scalar
      tail and all unsupported layouts unchanged, and add optional bias only
      after the full accumulation chain. The gate is hardware and geometry,
      not a model, corpus, or measured-shape threshold.
- [x] Reject combined-batch pointwise GEMM, GEMM bias-prefill, nested-serial
      depthwise, custom pointwise AVX row accumulation, pointwise weight
      pretransposition, four-accumulator depthwise AVX unrolling, redundant
      MaxPool compare removal, uninitialized pointwise/spatial GEMM output
      allocation, 3x3 depthwise specialization, and worker-occupancy pointwise
      scheduling after neutral-shape or end-to-end regressions. The last
      candidate cut one isolated geometry to 0.885 ms but regressed a bracketed
      29-page Text run to 31.071 seconds versus 30.670/30.874-second baselines.
      No model, corpus, batch-count, or measured-shape threshold rescues them.
- [x] Keep downstream OCR fingerprint investigation out of Power dispatch.
      Removing direct spatial execution, terminal projection, outer OCR
      parallelism, or biased-activation fusion one at a time retains the same
      unreviewed result, while the historical result belongs to a different OCR
      planning identity. Do not add model names, tensor values, documents,
      fingerprints, or observed shapes as runtime selectors.
- [x] Reject implicit F16/BF16 graph conversion on CPU. A temporary four-shape,
      content-free matrix probe measured F16 1.85--2.18x slower than F32 and
      found BF16 matrix multiplication unsupported by the active backend. The
      probe was removed; any future low-precision path must be an explicit
      typed graph/backend capability with independent numerical evidence.
- [x] Preserve canonical v1 tensor and token receipt bytes while hashing
      contiguous little-endian inputs directly, with a bounded canonical
      staging fallback on big-endian hosts.
- [x] Allow a model-owned deterministic output projection to execute on the
      graph device before bounded host materialization. The caller must bind
      the projection into its execution identity; Power retains permit,
      cancellation, same-device, dtype, finite-value, and tensor bounds.
- [x] Materialize one row-coalesced terminal-classifier projection per execution
      window and partition the validated host values into their original shapes
      and order. Same-tensor CPU/CUDA tests are bit-exact, complete graph suites
      pass, all five normalized 64-page caches remain exact, and all 74 Office
      artifacts match byte-for-byte.
- [x] Upload a finite multi-input CUDA execution window through one bounded flat
      F32 allocation, restoring every original shape and order as a contiguous
      read-only view. CPU, Metal, single-input, and empty-value paths remain
      unchanged. Exact offset and classifier-window CUDA tests pass; all five
      normalized 64-page caches and all 74 Office artifacts remain exact.
- [x] Compose consecutive private `Transpose` permutations only for explicit
      complete same-rank permutations with one private consumer and no retained
      intermediate. Fanout and unsupported forms preserve ordinary execution;
      a composed identity becomes zero-copy only after runtime rank validation.
      Generic planner tests and graph suites `129/0/9` and `133/0/43` pass;
      deterministic CPU output, all five normalized 64-page caches, and all 74
      Office artifacts remain exact. The unguarded 10.511-pages/s diagnostic is
      not promotion evidence.
- [x] Add opt-in CUDA-event attribution on each static graph executor's owning
      model-session stream. It reports only operator/tensor geometry, bounded
      counts, host submission time, and attributed/unattributed stream
      intervals; unsupported devices and driver failures fail closed. The mode
      synchronizes graphs and is diagnostic only, never a throughput or runtime
      selector. Full-corpus bottleneck attribution remains an external Parser
      evidence gate.
- [ ] Add benchmark evidence for allocation count and host-copy cost on named
      hardware, including the frozen fixed-order Parser A/Bs, before claiming a
      throughput improvement. First compare the one-upload baseline
      (`945ee546912f5afa26f54e1beee8d1e9bc136aecec86cdb38555ce4d7e1ee653`)
      with the Transpose-fold candidate
      (`681e2f60981b205a3832bccdd800410808dd521435e71f26bc7bc8669d29cb57`).
      The 10.347-, 10.435-, and 10.511-pages/s correctness diagnostics were
      unguarded and are not promotion evidence.

### P2 — Shape-profile execution evidence

- [ ] Add a model-owned, digest-bound shape-profile declaration for a finite
      set of batch/shape classes and an explicit dynamic fallback.
- [ ] Record selected profile identity and fallback reason without exposing
      tensor values, source identities, or model-private geometry.
- [ ] Reject stale profiles when weights, graph identity, device topology,
      scratch bounds, or TEE policy change.

This adapts TurboOCR's useful static `(batch, width)` profile discipline without
importing TensorRT profiles or moving shape selection into Power.

### P3 — Bounded replicas and deadline-aware admission

- [ ] Allow a policy-bounded number of independently mutable session replicas
      for one exact model identity while retaining one shared device gate and
      resident-byte budget.
- [ ] Add monotonic admission deadlines, queue-expiry evidence, and
      cancellation-safe cleanup. No request bytes or slot identities enter
      telemetry.
- [ ] Add health-driven replica retirement and lazy reconstruction at a safe
      request boundary; do not introduce an OCR-local watchdog or pool.

This is the model-neutral counterpart of TurboOCR pipeline replicas,
deadline-drop, and recycle behavior.

### P4 — Device-resident batch boundaries

- [ ] Add bounded device-resident input/output handles for adjacent reviewed
      graph calls, with exact dtype/shape/device validation and owned fallback
      copies.
- [ ] Preserve cancellation checks and receipt digests across fused or retained
      buffers.
- [ ] Expose only generic reviewed operators. OCR resize/normalize, ROI warp,
      DB postprocessing, and CTC decoding remain in A3S OCR.

### P5 — Confidential performance release gate

- [ ] Publish CPU, Metal, CUDA, and supported confidential-GPU captures from a
      clean immutable revision.
- [ ] Prove scalar/batch numerical equivalence, bounded peak host/device memory,
      cancellation, queue expiry, replica recovery, and explicit fallback.
- [ ] Bind benchmark artifacts to weights, graph declarations, runtime/device,
      TEE policy, and build revision. Third-party headline numbers are never
      reused as A3S measurements.

## Cross-repository delivery order

1. Power publishes model-neutral execution contracts.
2. A3S OCR pins that revision and owns PP-OCRv6 batch assembly and geometry.
3. A3S Parser pins the compatible OCR revision and owns document/page windows,
   persistence, reconciliation, and overlays.

No milestone is complete merely because another repository can emulate it with
a second scheduler, cache, pool, or receipt format.
