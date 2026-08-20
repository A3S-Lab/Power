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
