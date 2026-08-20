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

`ReleaseEvidencePolicy::strict_v1` requires exactly four distinct captures:

| Platform class | Device requirement | Additional binding |
| --- | --- | --- |
| CPU | `RuntimeDeviceKind::Cpu` | Local execution |
| CUDA | `RuntimeDeviceKind::Cuda` | Local execution |
| Metal | `RuntimeDeviceKind::Metal` | Local execution |
| Confidential GPU | `RuntimeDeviceKind::Cuda` | Verified claims digest and accelerator declaration digest |

Missing, duplicate, or undeclared classes fail. Captures are canonically ordered
and one tensor benchmark digest cannot be reused for two platform classes. This
prevents an ordinary CUDA measurement from being relabeled as confidential-GPU
performance.

The Rust construction path does not accept loose confidential digest strings.
`ReleaseCaptureSecurity::from_verified_confidential_gpu` requires a
`ConfidentialGpuBinding`, which Power can create only after matching an already
verified attestation report to the exact accelerator declaration, weights,
execution policy, device, and optional mesh. Deserialized bundles still require
the external signed trust-root check described below.

Projects that are not preparing a v1 production release may construct an
explicit narrower policy, for example a CPU-only development policy. Such a
policy does not satisfy `strict_v1` and must not be described as full P5
coverage.

## Evidence flow

```text
ReleaseRevisionBinding
  ├─ Power version + immutable commit
  ├─ weights + graph source + graph declaration
  ├─ finite shape-profile declaration
  └─ TEE policy
             │
             ▼
ReleaseEvidencePolicy ── exact required platform set
             │
             ├───────────────┬───────────────┬──────────────────┐
             ▼               ▼               ▼                  ▼
        CPU capture     CUDA capture     Metal capture   confidential capture
             │               │               │                  │
             └───────────────┴───────────────┴──────────────────┘
                                     │
                                     ▼
                         ReleaseEvidenceBundle SHA-256
                                     │
                                     ▼
                        caller-owned signed trust root
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
- verified confidential-GPU claims and accelerator declaration when required.

The outer bundle, each capture, the tensor report, and the shape binding use
domain-separated canonical digests. Deserialization denies unknown fields.

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

Metal and confidential-GPU results cannot be inferred from this Windows RTX
4090 host. They remain explicit release blockers until captured on appropriate
hardware from the same immutable release revision.
