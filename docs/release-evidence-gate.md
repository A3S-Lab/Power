# Production Release Evidence Gate

The P5 release gate answers a narrow question: does one immutable Power revision
have complete, internally consistent runtime evidence for every platform class
required by its release policy?

It does not identify model architecture from a name. It does not interpret
tokens, images, audio, document geometry, vocabularies, quantization schemes, or
container formats. Those semantics remain in the integrating crate. Power sees
only typed devices, bounded counters, exact output digests, and immutable
artifact identities.

## Coverage model

`ReleaseEvidencePolicy::strict_v1` requires exactly four distinct platform
bindings and captures:

| Platform class | Device requirement | Platform-specific binding |
| --- | --- | --- |
| CPU | `RuntimeDeviceKind::Cpu` | Shape-profile declaration, TEE policy, local execution |
| CUDA | `RuntimeDeviceKind::Cuda` | Shape-profile declaration, TEE policy, local execution |
| Metal | `RuntimeDeviceKind::Metal` | Shape-profile declaration, TEE policy, local execution |
| Confidential GPU | `RuntimeDeviceKind::Cuda` | Shape-profile declaration, TEE policy, verified claims and accelerator declaration |

Missing, duplicate, or undeclared classes fail. Captures are canonically ordered
and one tensor benchmark digest cannot be reused for two platform classes. This
prevents an ordinary CUDA measurement from being relabeled as confidential-GPU
performance.

Shape-profile declarations are platform-specific by construction: each one
commits to a typed device, topology, and host/device memory reservations. TEE
policies may also differ between local and confidential execution. Policy schema
v2 therefore keeps Power revision, weights, and graph identities in the common
`ReleaseRevisionBinding`, while `ReleasePlatformBinding` pins the profile and
TEE digests for exactly one platform. Requiring one shared profile digest would
make honest CPU/CUDA/Metal coverage impossible.

After replaying a capture, `ReleaseRevisionBinding::from_capture` and
`ReleaseCapture::platform_binding` project the common and platform-specific
identities for explicit policy review without manual nested-field copying.

The Rust construction path does not accept loose confidential digest strings or
a raw report. `verify_confidential_gpu_attestation` applies the fixed production
profile and returns a non-serializable proof tied to the exact authenticated
report. `ReleaseCapture::promote_confidential_gpu` consumes that proof plus the
matching accelerator declaration, verifies the source is a valid local CUDA
capture, projects the verified claims, and rebuilds the capture under the
confidential-GPU class. `ReleaseCapture::build` accepts only local captures, so
a caller-supplied label cannot invoke the trusted mint path. Deserialized
bundles remain evidence inputs and still require the external signed trust-root
check described below.

Projects that are not preparing a v1 production release may construct an
explicit narrower policy, for example a CPU-only development policy. Such a
policy does not satisfy `strict_v1` and must not be described as full P5
coverage.

## Evidence flow

```text
common revision: Power + weights + reviewed graph
                         |
                         v
release policy: CPU/CUDA/Metal/confidential platform bindings
                |          |          |          |
                v          v          v          v
             capture    capture    capture    capture
                \__________|__________|__________/
                           |
                           v
                canonical bundle SHA-256
                           |
                           v
                 caller-owned trust root
```

The bundle digest detects mutation. It does not establish authorship. A release
system must pin that digest in a signed release, verified attestation, or an
equivalent caller-owned trust root.

## What every capture proves

Each `ReleaseCapture` embeds a replayable
`TensorBatchBenchmarkReport`, a `ShapeProfileBinding`, and one
`ReleaseContractEvidence` value.

### Scalar/batch equivalence

The tensor report contains alternating individual and leading-batch raw samples.
Verification reconstructs its named-hardware binding and median summaries and
requires exact ordered output-digest parity. It does not infer that batching is
faster; negative results remain valid evidence.

### Bounded peak memory

Host evidence uses continuous allocator live-byte accounting or a process
resident-set sampler. Accelerator evidence must use sampled device-pool
availability and records its interval and sample count. Verification requires:

- a positive observed peak;
- peak additional bytes no greater than the shape binding's fixed-plus-scratch
  reservation;
- final additional bytes no greater than the fixed reservation; and
- device evidence for CUDA and Metal, with no device-memory claim for CPU.

Sampled measurements are identified as sampled. They are not presented as exact
allocator peaks.

### Cancellation cleanup

The execution lifecycle must contain admitted, completed, and cancelled members,
at least one committed step, processed rows, and a bounded non-zero peak state.
After cancellation, both active and waiting admission counts must be zero, as
must resident tensor handles and bytes.

### Queue expiry

Before and after snapshots must come from one bounded, quiescent admission
controller. Counters cannot move backwards, and the deadline-expiration counter
must increase.

### Replica recovery

Three snapshots preserve the transition rather than only its final result:

1. a healthy, quiescent replica set;
2. an observed retirement with fewer ready replicas and reconstruction pending;
3. a quiescent recovered set with the original ready capacity and resident-byte
   reservation restored.

The device, pool limits, registered sessions, and reserved replica count remain
stable across the sequence.

### Explicit fallback

The shape-profile evidence must select a typed `DynamicFallback` path. Its
implementation digest, declaration, weights, shape binding, and runtime device
are checked, and its typed output digest must exactly equal the reference output.
An implicit device move or an unlabeled fallback is not accepted.

## Immutable binding

The policy and captures jointly bind:

- Power version and a 40- or 64-character lowercase Git revision;
- the exact runtime executable SHA-256 for each build;
- verified weights;
- reviewed graph source and graph declaration;
- the finite shape-profile declaration and device topology;
- host and device memory reservations;
- the resolved typed device and named hardware environment;
- the TEE policy; and
- verified confidential-GPU claims, CPU TEE type, and accelerator declaration
  when required.

The outer bundle, each capture, the tensor report, and the shape binding use
domain-separated canonical digests. Deserialization denies unknown fields.

For v1, the confidential binding must name AMD SEV-SNP. Intel TDX remains a
typed generic schema value for future policy revisions, but
`ReleaseEvidenceBundle::verify_strict_v1_release` rejects it until Power has a
reviewed DCAP Quote/QVL path. The exact support boundary is recorded in the
[v1 Production Support Matrix](v1-support-matrix.md).

## Tagged-release verification

The v1 tag points to a single-parent evidence commit, not directly to the source
commit measured by the captures. Its parent is the frozen source revision. The
evidence commit may add only the two regular files under
`release/v<crate-version>/`; that directory must not exist in the source parent.
This two-commit layout avoids the impossible requirement for a checked-in
bundle to contain the hash of the commit that contains the bundle.

First derive and validate the frozen source revision, then pass that revision to
the isolated read-only verifier:

```bash
source_commit="$(bash tools/verify-release-evidence-commit.sh 1.0.0 HEAD)"

a3s-power-tensor-batch-bench verify-release-bundle \
  --bundle release/v1.0.0/release-evidence.json \
  --expected-sha256-file release/v1.0.0/release-evidence.sha256 \
  --power-version 1.0.0 \
  --power-commit "${source_commit}"
```

It bounded-reads both files, denies unknown bundle fields, replays every nested
digest and contract, requires the exact four-platform v1 policy, checks the
external pin, matches the requested version and source revision, and enforces
the SEV-SNP confidential boundary. The pin file must contain one lowercase
SHA-256 digest with only an optional LF or CRLF line ending.

Release CI runs both checks for every non-`0.x` tag before artifacts can be
published. It builds binaries and publishes the crate from the frozen source
parent while attaching the evidence pair from the tagged child. Missing,
pre-existing, or mixed source/evidence changes therefore block v1 instead of
silently degrading to the available local platforms.

## Capture runners

The isolated `a3s-power-tensor-batch-bench` process now has two complete
contract paths:

- `release-fixture` creates a temporary generic Add graph for runtime and
  hardware calibration;
- `release-run` accepts caller-owned verified weights, a reviewed graph, at
  least two compatible F32 inputs, opaque profile/fallback implementation
  digests, and one independently produced typed reference output.

Both paths first record alternating scalar/batch evidence, then drive the real
resident graph, execution-batch lifecycle, bounded admission queue, session
replica pool, and shape-profile selector. Host peak memory comes from the
process-global live-byte allocator in that isolated process. CUDA and Metal use
sampled device-pool availability. Declared fixed and scratch reservations are
parsed before the workload is constructed; exceeding them fails the capture
instead of rewriting the bounds after measurement.

The reference file is read only to compute an `ExecutionDigest`; paths, tensor
values, model family, and graph role are absent from the emitted capture. The
caller remains responsible for proving that the reference was produced by an
independent reviewed implementation rather than copying the tested output.
Complete commands and input formats are in the
[Tensor Batch Cost Benchmark Protocol](tensor-batch-benchmark.md).
The external [Metal and Confidential-GPU Release Capture](external-release-capture.md)
guide carries the same contract onto named hardware, preserves raw vendor
evidence, and invokes strict proof-backed promotion without adding a
model-specific release path.

After all four captures exist, `build-release-bundle` derives their common
revision and platform-specific bindings instead of asking a release operator to
copy digest fields. It requires each capture under its typed CPU, CUDA, Metal,
or confidential-GPU argument, constructs `ReleaseEvidencePolicy::strict_v1`,
replays the final version and commit check, and creates the bundle and digest
pin as a no-overwrite pair. An ordinary write failure rolls back any newly
created half of the pair.

## Model and backend boundary

Architecture-specific implementations are adapters behind the evidence hashes.
A Llama, Qwen, Mistral, vision encoder, embedding model, OCR network, audio
model, or scientific graph may use a different loader and operator plan while
presenting the same release contract. GGUF, SafeTensors, and caller-owned formats
are likewise outside the gate.

Optimizations such as Flash Attention, fused kernels, speculative decoding, MTP,
reduced projections, tensor sharing, or device-resident chains must be included
in the reviewed graph and runtime artifact identities. None receives a special
verification branch.

## Current evidence status

The [2026-08-21 Windows CPU/CUDA pre-captures](benchmarks/release-gate-windows-20260821/README.md)
were produced from clean revision
`1a9504e58fc2751e016efede2fc006615a0b8cc2`. They replay exact scalar/batch
parity and retain the CPU negative result. They predate the complete runtime
contract capture and therefore cannot form a strict v1 bundle.

The [2026-08-21 complete Windows CPU/CUDA captures](benchmarks/release-contract-windows-20260821/README.md)
were generated from clean revision
`6b7d6e5265b34c3e9e812c830ce22cc4a35940e5`. Both replay every runtime
contract, and their platform-specific profile bindings form one verified
two-platform partial bundle. The raw JSON, policy bytes, artifact hashes,
negative CPU batching result, and exact reproduction commands are checked in.

Metal and confidential-GPU results cannot be inferred from this Windows RTX
4090 host. They remain explicit release blockers until captured on appropriate
hardware from the same immutable release revision. The Rust API now implements
proof-backed confidential promotion, and a raw attestation report or
caller-provided label is insufficient. The checked-in
[external capture workflow](external-release-capture.md) now defines the exact
Metal runner, nonce-bound NVIDIA evidence collection, configured raw-evidence
binding, model-owned accelerator declaration, strict verifier pins, create-new
promotion, and artifact inventory. That implementation does not replace the
missing hardware results: the raw Metal and confidential-GPU captures plus an
authenticated final bundle still have to be produced from one immutable
revision.
