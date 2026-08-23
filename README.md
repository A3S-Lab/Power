# A3S Power

<p align="center">
  <a href="https://a3s-lab.github.io/Power/"><img src="./site/docs/public/a3s-os-logo.png" width="72" alt="A3S OS"></a>
</p>

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S Power connects embedded and OpenAI-compatible inference to one bounded runtime, canonical execution receipts, and independent verification">
</p>

<p align="center">
  <strong>A model-neutral Rust runtime for bounded, verifiable inference - embedded or OpenAI-compatible.</strong>
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI status"></a>
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/pages.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/pages.yml?branch=main&amp;style=flat-square&amp;label=docs" alt="Documentation deployment status"></a>
  <a href="https://a3s-lab.github.io/Power/"><img src="https://img.shields.io/badge/docs-简体中文-2864e8?style=flat-square" alt="A3S Power Chinese documentation"></a>
  <a href="https://a3s-lab.github.io/Power/en/"><img src="https://img.shields.io/badge/docs-English-2864e8?style=flat-square" alt="A3S Power English documentation"></a>
  <a href="https://crates.io/crates/a3s-power"><img src="https://img.shields.io/crates/v/a3s-power?style=flat-square&amp;color=2864e8" alt="a3s-power on crates.io"></a>
  <a href="https://docs.rs/a3s-power"><img src="https://img.shields.io/docsrs/a3s-power?style=flat-square" alt="a3s-power API documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#why-power-exists">First principles</a> &middot;
  <a href="#measured-boundary">Measured boundary</a> &middot;
  <a href="#start-in-three-steps">Quick start</a> &middot;
  <a href="#one-core-three-surfaces">Architecture</a> &middot;
  <a href="#verification-is-a-client-decision">Verification</a> &middot;
  <a href="https://a3s-lab.github.io/Power/">Documentation</a>
</p>

A3S Power makes inference infrastructure explicit. Model repositories keep
ownership of topology, tokenization, preprocessing, state semantics, and quality
policy. Power owns the execution boundary around them: devices, admission,
placement, cancellation, artifact integrity, privacy, and evidence.

That separation supports a listener-free Rust library, an optional hosted
service, and a safe artifact provisioner without hiding model behavior behind a
second model abstraction.

## Why Power exists

Start from three constraints:

| First principle | Engineering consequence | Power mechanism |
| --- | --- | --- |
| Inference consumes finite memory, compute, queue, and transfer capacity. | Every request must enter through explicit limits and remain cancellable; adjacent graph calls should not bounce unchanged tensors through host memory. | Bounded admission, deterministic microbatching, model-owned finite shape profiles, aggregate resident-tensor budgets, placement plans, session pools, and cancellation-safe queues. |
| A model name does not identify the bytes or policy that produced an answer. | Execution identity must bind artifacts, runtime policy, device path, input, and output. | SHA-256 identities, verified mirrors, accelerator evidence, and canonical receipts. |
| The server operator cannot be the root of trust for its own claims. | Acceptance policy belongs to the client or verifier. | Nonce-bound CPU TEE reports, optional confidential-GPU claims, RA-TLS, and an independent verifier CLI. |

The shared runtime is deliberately model-neutral: Power contains no product
model assets, and its execution contracts do not dispatch on language, vision,
OCR, embedding, or any other model family. Architecture-specific adapters may
live behind those contracts without changing the core scheduler or verifier.

## Measured boundary

Power publishes machine-readable captures from its real streaming API, not only
a standalone llama.cpp microbenchmark.

The Qwen3.8 table below is one pinned backend/model/hardware integration, not
the scope of the engine. Power's public runtime, graph, scheduling, memory,
copy-cost, and evidence contracts do not branch on Qwen or any model family;
language, vision, OCR, embedding, and multimodal crates retain their own
topology and semantics.

The active performance and quality gate tests only the unchanged Q6_K target
artifact. A Q4 file appears below only when an external speculative decoder
needs an auxiliary proposer; it is never scored or reported as the target
model. Mixed-quantization and Q8 captures remain historical research outside
the current acceptance decision.

| Qwen3.8-27B Q6_K target mode | Fixed-task quality proxy | Request-wide throughput | Median steady decode |
| --- | --- | ---: | ---: |
| Untouched Q6_K, autoregressive | 67/100 lenient; 60/100 strict (100 tasks, 3x) | 30.883 token/s | 35.5793 token/s (earlier capture) |
| Untouched Q6_K, paired DFlash2 calibration control | 9/12 lenient and strict in every repetition | 29.702 token/s mean | 35.380 token/s |
| **Untouched Q6_K + external DFlash2 Q4 proposer, K7** | 9/12 lenient and strict; **12/12 answer parity, 7/12 complete-output parity** | **45.143 token/s mean** (1.520x paired control) | **108.429 token/s** |
| Untouched Q6_K, DSpark acceptance control | Exact 256-token greedy output in paired 3x capture | 25.171 token/s median | 32.249 token/s |
| **Untouched Q6_K + external DSpark Q4, K10/S6 (peak prompt)** | Exact paired 256-token output and receipt hashes | **65.825 token/s median** | **169.324 token/s** |
| Untouched Q6_K + external DSpark Q4, K10/S6 (100 tasks, 3x) | 73/100 lenient; 59/100 strict; **54/100 exact-output parity** versus target-only | **32.678 token/s** (1.445x paired control) | - |
| **Untouched Q6_K + adaptive external DSpark Q4, K10/S6 (controlled peak)** | Identical output and receipt hashes across all 3 samples | **63.535 token/s median** | **164.756 token/s median; 160.881 minimum** |
| Untouched Q6_K + adaptive external DSpark Q4, K10/S6 (100 tasks, 1x) | 69/100 lenient; 56/100 strict; **55/100 exact-output parity** versus 67/100 and 58/100 target-only | **31.052 token/s** (1.358x paired control) | - |
| Adaptive DSpark loss-focused follow-up (5 selected tasks) | **5/5 answer parity and 0 losses** at 512 tokens × 3; **5/5 untruncated parity** at 1,024 tokens | **30.521 vs 24.967 token/s** at 512 tokens (1.222x) | - |
| **Untouched Q6_K + prefix-FR8192, fixed K6/S6, B8** | 9/12 lenient and strict in both paired modes (1x; 3 truncated) | **46.923 token/s** | - |
| **Untouched Q6_K + prefix-FR8192, fixed K7/S6, B11, high-priority CUDA** | Fixed peak prompt retained the control digest | - | **172.835 token/s** on a contended desktop |
| Untouched Q6_K, full-vocabulary MTP, K7/S7 | Fixed peak prompt has exact greedy parity | - | 147.0207 token/s |
| Untouched Q6_K, full-vocabulary MTP, K7/S6 | 5/12 lenient; 3/12 strict (1x calibration; 11 truncated) | **47.032 token/s** | - |
| **Untouched Q6_K + prefix-FR8192 MTP, K7/S6** | 4/12 lenient; 3/12 strict (1x calibration; 11 truncated) | 37.290 token/s | **176.6109 token/s** |

The native external-DSpark acceptance capture used one clean CUDA commit and
the same batch-12, context-512, 256-token greedy request for both rows. DSpark
recorded 169.561, 167.102, and 169.324 token/s, so its 169.324 median and
167.102 minimum passed the 160 token/s all-sample gate. It accepted 90.873% of
proposals, committed 9.8077 tokens per target pass, performed zero replay, and
matched the target-only output byte for byte. The 5.250x decode speedup is a
short-context single-request boundary, not a universal service or quality
claim.

The separate 600-request MMLU/GSM8K/C-Eval capture measured the same K10/S6
profile at 32.678 token/s request-wide versus 22.618 token/s for target-only.
Both modes were deterministic across three repetitions. DSpark scored 73/100
lenient and 59/100 strict versus 67/100 and 58/100 for the paired control, so
no score decrease was observed. However, only 54/100 complete outputs matched
byte for byte, and every DSpark request exercised exact fallback replay. This
is useful workload evidence, but it fails Power's lossless production-default
gate; K10/S6 remains an explicit benchmark profile while graph-shape parity and
rollback width are recalibrated.

The current request-local controller replaces that unconditional wide start
with one rollback-safe probe. It begins at K6, jumps directly to K10 only when
the first K6 proposal is fully accepted, preserves stable graph shapes for
healthy partial rounds, and opens a one-way target-only circuit after sustained
low yield. On clean commit `cbdb3f673446b3532c9683dabc816a149ae27b1f`, the
controlled peak produced 166.988, 160.881, and 164.756 token/s. Its 164.756
median and 160.881 minimum passed the 160 token/s median and all-sample gates;
all three output and receipt hashes matched. The peak retained 92.713%
acceptance, 9.8077 tokens per target pass, and zero replay.

The paired 100-task capture reached 31.052 token/s versus 22.872 token/s for
target-only, a 1.358x request-wide gain. It accepted 62.878% of proposals,
committed 3.373 tokens per target pass, switched 24 requests to target-only,
and recorded zero fallback replay and zero rollback-guard activation. The
candidate moved from 67/58 to 69/56 lenient/strict answers, with five lenient
gains, three lenient losses, one strict gain, three strict losses, 89/100
answer parity, and 55/100 complete-output parity. All 57 tasks untruncated in
both modes retained the same extracted answer. A clean loss-focused follow-up
then selected every observed lenient or strict loss plus one positive control.
Across three alternating 512-token repetitions it retained 5/5 paired answers
with zero gains and zero losses while reaching 30.521 versus 24.967 token/s.
At a 1,024-token override, all five tasks ended normally and retained 5/5
paired-answer parity; both modes scored 4/5. This localizes the earlier losses
to cutoff-sensitive trajectories, but complete outputs still differed 0/5.
The profile is therefore faster and answer-noninferior on the selected
follow-up, not a lossless default, a general-intelligence proof, or a 175
token/s service floor.

DFlash, DFlash2, and DSpark are alternative external-draft contracts, not an
additive mode. A genuine DFlash v1 GGUF has not yet been accepted on this host.
Forcing the DSpark artifact through a DFlash-shaped diagnostic produced only
1.031% acceptance and remains negative compatibility evidence.

DFlash2 was tested separately with the unchanged Q6_K target and a 1.14 GB Q4
auxiliary proposer. The three-order 12-task calibration retained 9/12 in both
modes, 12/12 extracted answers, and 7/12 complete outputs; mean request-wide
throughput rose from 29.702 to 45.143 token/s. A repetitive-prompt boundary
reached 108.429 token/s median at 98.230% acceptance, not 175 token/s. This is
an exact upstream llama.cpp PR 27342 standalone capture. Power validates the
DFlash2 artifact contract but its pinned `llama-cpp-rs` backend intentionally
rejects execution until a reviewed binding update lands, so the profile is
experimental and not a native production result.

The current untouched 22,884,408,288-byte Q6_K artifact sustained a 176.6109
token/s median across nine 1,024-token samples, with a 173.2630 minimum and
seven samples at or above 175. Prefix-FR8192 improved steady decode by 20.13%
over its 147.0207 token/s full-vocabulary K7/S7 control, and both emitted the
same deterministic output digest. No model weight was requantized.

The 2026-08-22 execution-only follow-up kept those exact bytes and separated
two workload shapes. The peak K7/S6/B11 profile disables Flash Attention only
for short-batch decode and uses high-priority CUDA streams; its clean nine-run
capture reached 172.835 token/s median and 171.298 minimum while the Windows
display GPU already had 5--8% background utilization. The mixed-task K6/S6/B8
profile reached 46.923 token/s versus a paired 28.713 token/s target-only
control, a 63.42% gain, with
the same 12 final answers, the same 9/12 score, and zero replay. CUDA graphs are
essential: disabling them fell to 133.876 token/s on the peak workload.

That capture crosses 175 as a median boundary; it does **not** establish a
175 token/s service floor because two samples were below it. New acceptance
runs can independently require the median and every measured sample, and the
generic GGUF runner can require a configurable pre-start quiet window and
retains an input-bound preflight receipt when the idle-GPU ceiling is exceeded.
On this shared Windows display GPU, exclusive scheduling is part of the
requirement, not an inference flag.

That FR profile is a long, high-coverage peak, **not** a universal default or a
service floor. On the one-pass 12-task calibration, full-vocabulary K7/S6
reached 47.032 token/s request-wide with 52.30% proposal acceptance, while
prefix-FR8192 reached 37.290 token/s with 24.82% acceptance. Eleven tasks per
mode hit the 128-token cap, so this calibration diagnoses workload sensitivity;
the repeated 100-task matrix remains the primary quality evidence.

The quality values are task-accuracy proxies, not general intelligence or IQ
measurements. Exact target verification and a fixed-task score do not by
themselves prove unchanged prose, general-intelligence equivalence, or a
universal production default.

The lossless prefix-cache path was measured separately with the unchanged
Q6_K model, target-only decoding, and Flash Attention enabled. Five fresh
cold/warm pairs reduced median backend prefill from 786.1375 ms to 33.4102 ms
(23.5299x) and median TTFT from 950.0142 ms to 72.1932 ms (13.1593x), while
reusing 9,740 prompt tokens. This is repeated-context latency evidence, not a
steady-decode or external-draft claim.

- [Current Q6_K benchmark, raw samples, and limitations](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md)
- [Q6_K prefix-cache cold/warm capture and exact reproduction](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/prompt-cache/README.md)
- [Untouched-Q6_K 176.61 token/s boundary and dynamic-quantization analysis](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/PURE-Q6.md)
- [Repeated quality matrix and reproducible environment](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md)
- [Step-by-step reproduction guide](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md)
- [Native DSpark Q4 peak and 600-request quality captures](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/README.md)
- [Q6_K-only DFlash2 peak, quality boundary, and path-free evidence](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2/README.md)
- [Adaptive DSpark path-free peak and paired-quality evidence](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/adaptive/evidence.json)
- [Adaptive DSpark truncation follow-up and offline evidence](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/quality/README.md#adaptive-truncation-follow-up)

## Start in three steps

### 1. Choose the boundary

Use the embedded runtime when your crate owns the model graph. Use the hosted
service when clients need an OpenAI-compatible network API. Both paths reuse the
same integrity, resource, and evidence contracts.

### 2. Embed the model-neutral runtime

```toml
[dependencies]
a3s-power = { version = "1.0.0", default-features = false, features = ["embedded-inference"] }
```

```rust
use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};

fn main() -> Result<(), a3s_power::error::PowerError> {
    let runtime = EmbeddedRuntime::new(
        DevicePreference::Auto,
        InferenceLimits::default(),
    )?;

    println!("execution device: {}", runtime.device().name());
    Ok(())
}
```

The caller supplies a reviewed model plan and retains semantic state. Power
supplies the bounded execution boundary and evidence contracts; constructing the
embedded runtime never opens a listener or downloads a model.

### 3. Host only when the product needs a service

```bash
cargo install a3s-power
a3s-power serve --host 127.0.0.1 --port 11434
```

In another terminal:

```bash
a3s-power models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
a3s-power chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
```

Models and content-addressed blobs live under `~/.a3s/power` by default. Set
`A3S_POWER_HOME` to move the store.

## One core, three surfaces

```text
revision-locked bundle       model-owned graph       API client
          |                         |                     |
    provisioner                 embedded               service
          \                         |                    /
           +----------- shared runtime core -----------+
                                  |
              admission / placement / cancellation
                                  |
                    devices / weights / state
                                  |
                   evidence / canonical receipt
                                  |
                      independent verification
```

| Surface | Use it when | Network behavior |
| --- | --- | --- |
| **Artifact provisioner** | A product owns a revision-locked bundle and needs safe first-use installation. | Policy-gated download; verified artifacts are reusable offline. |
| **Embedded runtime** | A model crate owns its graph and needs shared execution, placement, state, and evidence. | No listener, model hub, or child process. |
| **Hosted service** | Clients need chat, completions, embeddings, lifecycle, metrics, and attestation. | Explicit HTTP, RA-TLS, or vsock transport. |
| **Minimal TEE service** | A constrained enclave needs a pure-Rust, layer-streaming GGUF path. | Transport remains an explicit feature choice. |

### Responsibility boundary

| Power owns | Model-owning crates own |
| --- | --- |
| Typed CPU, CUDA, and Metal devices; bounded graph execution and opaque resident-tensor boundaries | Architecture, topology, layers, kernels, and arithmetic |
| Admission, session pools, microbatching, cancellation, and limits | Tokenizer, preprocessing, postprocessing, and generation policy |
| Artifact identity, mirrors, placement, and residency | Asset revisions, conversion, tensor contracts, and quality gates |
| TEE privacy, attestation binding, sealed state, and receipts | KV/recurrent layout and semantic state |

Read the [embedded inference architecture](docs/embedded-inference-architecture.md)
for the complete contract model.

### Finite profiles do not give Power model semantics

A model crate may publish up to 256 opaque, digest-bound shape classes for
reviewed optimized implementations. Power checks only aggregate batch,
tensor-element, scratch, device-topology, artifact, and TEE-policy bounds. The
model still chooses what a class means and how an input maps to it. Missing or
oversized classes either fail or use an explicitly digest-bound dynamic
implementation; receipt v5 records the path and reason without dimensions or
class data. See [Model-Owned Shape Profiles](docs/shape-profiles.md).

### Mutable replicas do not give Power model semantics

Stateful backends may request up to 256 lazy, independently initialized replicas
for one exact model/execution identity. Every replica is held by a non-cloneable
lease, but all replicas reuse one resolved runtime and one physical-device
admission gate. Power reserves the declared per-replica resident bytes for the
entire configured replica set before invoking a loader, so concurrency cannot
silently overcommit memory.

The family string is opaque identity, not a dispatch key: language decoders,
vision encoders, OCR graphs, embedding models, and multimodal pipelines use the
same pool. The loader receives no replica ordinal or model-family switch, and
snapshots contain only aggregate counts. Cancellation or future drop returns
the lease and removes a never-initialized empty entry. Optional monotonic
deadlines cover the complete model/device admission path without serializing
wall-clock time; the typed expiry maps to HTTP 408 at the hosted boundary
and increments only content-free aggregate counters. A model crate can consume
an exclusive lease with `retire()` after it determines that mutable state is no
longer reusable; Power replaces that anonymous generation before releasing the
slot and reconstructs it lazily without changing the declaration identity. See
[Model-Neutral Session Replicas](docs/session-replicas.md).

### Resident graph chains do not give Power a model architecture

`GraphExecutor::run_to_resident` can keep one reviewed graph output on the
resolved runtime device, and `run_resident` can consume it in the next reviewed
graph under the same request permit. The opaque `ResidentGraphTensor` is
non-cloneable, F32-only, exact-shape checked, bound to one runtime/device, and
charged to a shared aggregate byte budget derived from `max_tensor_elements`.
It contains no tokenizer, layer, attention, image, OCR, or model-family switch.

The first owned tensor is hashed before upload. `materialize` performs the one
final owned output copy, rechecks cancellation, and returns both canonical v1
tensor digests plus aggregate boundary counts. Incompatible runtimes never
trigger an implicit copy: materialize explicitly, move the owned output through
`TensorOutput::into_input`, and acquire the target permit afterward. See
[Device-Resident Reviewed Graph Chains](docs/device-resident-graphs.md).

### The release gate verifies contracts, not model names

`ReleaseEvidenceBundle` applies one fail-closed policy to named-hardware
captures. The strict v1 policy requires distinct CPU, CUDA, Metal, and
confidential-GPU evidence; a local CUDA report cannot be reused as the
confidential capture. Every capture must replay exact scalar/batch parity and
prove bounded host/device peak memory, active-work cancellation cleanup, queue
deadline expiry, replica retirement and reconstruction, and an explicit exact
fallback.

The policy separates identities that can be shared honestly from those that
cannot. Power revision, weights, and reviewed graph are common; each platform
binds its own finite shape-profile declaration and TEE policy because device
topology and memory reservations differ. The exact runtime executable remains
capture-specific, and confidential GPU claims are required where applicable.

Local captures can be constructed only with the local security class. A
confidential capture must pass the fixed
`verify_confidential_gpu_attestation` profile and carry its opaque,
exact-report proof into `ReleaseCapture::promote_confidential_gpu`; raw reports,
deserialized labels, and caller-authored verification booleans cannot mint that
class. The resulting digest-bound capture is still evidence, not authorship, so
the release system must authenticate its bundle digest.

For v1, the confidential binding is explicitly SEV-SNP-only. The release
workflow requires a tag to identify an evidence-only child of the frozen source
commit, bounded-reads its checked-in four-platform bundle, verifies the external
SHA-256 pin plus exact crate version and source commit, and builds release
binaries from that parent. This avoids self-referential commit identity and
blocks every non-`0.x` tag when evidence is missing, mixed with source changes,
not reachable from `main`, carried by a lightweight or unverified tag, or
TDX-backed. Strict construction also rejects Metal captures that do not name a
native Apple Silicon/macOS host or that disclose virtual, emulated, translated,
fallback, software-rendered, or unnamed hardware; hosted paravirtual Metal is
preflight evidence, not production evidence. See the
[v1 Production Support Matrix](docs/v1-support-matrix.md).

`a3s-power-tensor-batch-bench release-run` applies the same collector to any
caller-owned reviewed graph, typed tensors, opaque profile identities, and
independent reference output. `release-fixture` is only a reproducible Add-graph
calibration path. Its persistent cross-host weights are created with
`materialize-release-fixture-weights`; `release-confidential-fixture` then
writes a local CUDA source plus active residency declaration as a validated
create-new pair. `verify-release-capture` independently bounded-reads a received
capture, recomputes its canonical digest, and checks its exact platform,
version, and source revision before bundle assembly; its receipt explicitly
states that the strict four-platform bundle is still required. SafeTensors
startup pins, attestation, embedded execution, and
accelerator declarations share one canonical collection digest. Neither the
collector nor the evidence schema contains a
tokenizer, container format, generation mode, model family, or architecture
dispatch key. Qwen is one possible workload; language, vision, embedding,
audio, scientific, and custom graphs use the same contract. See
[Production Release Evidence Gate](docs/release-evidence-gate.md). The
[external hardware capture guide](docs/external-release-capture.md) defines the
real Metal run and nonce-bound confidential-GPU promotion without introducing a
Qwen-, GGUF-, or backend-specific release path.

## Backends are capabilities, not architecture

| Feature | Role | Native dependency |
| --- | --- | --- |
| `mistralrs` | Default Candle-based GGUF, SafeTensors, vision, and embedding backend | No C++ inference engine |
| `llamacpp` | Mature GGUF backend with native MTP support | CMake, C++ compiler, and libclang |
| `llamacpp-cuda` | CUDA execution for llama.cpp | CUDA toolkit |
| `llamacpp-external-draft` | Verified external DFlash or DSpark execution; typed, fail-closed DFlash2 admission pending a binding update | Reviewed external-draft patch to the pinned llama-cpp-rs source |
| `llamacpp-mtp-fr` | Experimental reduced-vocabulary MTP draft projection | Reviewed patch to the pinned llama.cpp source |
| `picolm` | Pure-Rust layer-streaming GGUF backend for constrained TEE memory | No C/C++ inference engine |
| `embedded-cuda` / `embedded-metal` | Accelerators for model-owned embedded graphs | Platform toolkit |
| `tls` / `vsock` | RA-TLS and A3S Box guest-host transports | Platform-specific |
| `hw-verify` | AMD SEV-SNP signature verification; Intel TDX fails closed pending DCAP Quote/QVL support | Platform crypto dependencies and AMD KDS access |

Cargo resolves the pinned optional Git dependencies before feature selection.
On Windows, enable Git's long-path support once before the first build so the
pinned llama.cpp source can be checked out intact:

```powershell
git config --global core.longpaths true
```

```bash
# Default hosted service
cargo build --locked --release

# Listener-free embedded runtime
cargo build --locked --release --no-default-features --features embedded-inference

# Pure-Rust layer-streaming TEE service
cargo build --locked --release --no-default-features --features tee-minimal

# llama.cpp with CUDA
cargo build --locked --release --no-default-features --features llamacpp-cuda

# Strict verifier plus model-neutral confidential release promotion
cargo build --locked --release --no-default-features \
  --features server,embedded-inference,hw-verify \
  --bin a3s-power-verify
```

The FR-Spec-inspired path is separate because it patches the pinned llama.cpp
source. Follow the reviewed procedure in
[Model-neutral Speculative Decoding](docs/speculative-decoding.md); ordinary
`llamacpp` builds do not need that patch.

## Speculation remains exact

Power exposes model-neutral strategy selection, capability negotiation,
adaptive draft lengths, exact target verification, rollback, and metrics.
Backends advertise support; an explicitly requested unsupported strategy fails
closed.

Available strategies are `off`, `prompt-lookup`, `ngram-context`, `draft-model`,
`mtp`, `dflash`, `dflash2`, and `dspark`; `auto` selects a backend-supported
default.

llama.cpp can now load one verified external DFlash or DSpark GGUF beside its
target. Registration hashes both files, validates the artifact-specific tensor
contract, binds the draft to the target digest, and fails closed on a kind or
identity mismatch. It does not load native MTP and an external drafter at the
same time. DFlash2 registration has its own selector/convolution tensor
contract, but the current pinned binding does not expose its executor; explicit
or automatic DFlash2 selection therefore fails closed instead of being
relabeled as DFlash v1.

```acl
spec_mode = "mtp"
spec_draft_max = 7
spec_mtp_recurrent_snapshots = 7

# Experimental compact draft projection. Omit for full vocabulary.
# spec_mtp_fr_vocab_size = 8192
```

K7/S7 keeps a resident rollback point for every proposal and is the balanced
mixed-workload profile. Reduced-vocabulary FR changes only the draft head, but
its acceptance is domain- and language-sensitive. The current quality-gated
profile therefore uses full-vocabulary drafting.

## Configure policy with ACL

The service reads `~/.a3s/power/config.acl`, or the path supplied to
`a3s-power serve --config`.

```acl
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
prompt_cache_max_entries = 1
prompt_cache_ttl_seconds = 300
keep_alive = "5m"

flash_attention = true
num_parallel = 1

gpu {
  gpu_layers = -1
  main_gpu = 0
}
```

TEE deployments add verifier-owned pins and strict policy:

```acl
tee_mode = true
tee_policy_mode = "strict"
redact_logs = true

expected_measurement "sev-snp" {
  digest = "<96-character measurement hex>"
}

model_hash "your-model" {
  digest = "sha256:<64-character artifact digest>"
}
```

Invalid ACL, ranges, strategies, hashes, or unsupported explicit backends fail
before inference. Strict mode rejects simulated attestation.

## OpenAI-compatible API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness, loaded models, backend capabilities, and TEE status |
| `POST` | `/v1/chat/completions` | Chat, tools, structured output, vision, and SSE streaming |
| `POST` | `/v1/completions` | Text completion and SSE streaming |
| `POST` | `/v1/embeddings` | Embedding inference |
| `GET` | `/v1/models` | Registered models |
| `POST` | `/v1/models` | Register local weights and optional auxiliary artifacts |
| `POST` | `/v1/models/pull` | Resumable ModelScope or Hugging Face pull |
| `GET` | `/v1/attestation` | Nonce- and model-bound TEE evidence |
| `GET` | `/metrics` | Prometheus metrics |

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Explain capability-based security."}],
    "prompt_cache_key": "shared-agent-prefix-v1",
    "stream": true
  }'
```

For GGUF models, registration accepts optional typed `adapter`, `projector`,
or `external_draft` objects. Callers provide artifact locations; Power reads
the files and records their exact byte lengths and SHA-256 identities instead
of trusting client-supplied hashes. Strict TEE startup rejects older path-only
adapter or projector manifests.

Chat and completion responses include an `attestation_receipt` and its SHA-256
digest. Streaming responses emit the receipt before `[DONE]`.

`prompt_cache_key` is an explicit A3S extension. The llama.cpp text path reuses
a token prefix only when its KV and recurrent state can roll back exactly;
unsupported backends return
`prompt_cache_unsupported` instead of ignoring the field. Power scopes the key
by authenticated identity, endpoint, and model, bounds resident contexts by
LRU capacity and TTL, exposes hit/miss/token/eviction metrics, and binds only a
key digest into the request receipt. Native MTP and cached llama.cpp sessions
do not compose yet; explicit MTP fails closed and `auto` chooses target-only
decoding for keyed requests. See [Keyed Prompt-Prefix Cache](docs/prompt-prefix-cache.md).

## Verification is a client decision

```text
model bytes + runtime policy
             |
             v
canonical claims digest + fresh nonce
             |
             v
CPU TEE report + optional GPU evidence
             |
             v
request, effective prompt, and output receipt
             |
             v
independent client verifier accepts or rejects
```

```bash
a3s-power-verify \
  --url http://127.0.0.1:11434 \
  --nonce <client-nonce-hex> \
  --model-hash <artifact-sha256-hex> \
  --inference-execution-digest <resolved-power-policy-sha256> \
  --auxiliary-artifacts-digest <portable-auxiliary-set-sha256> \
  --expected-measurement <launch-measurement-hex>
```

Derive the server policy pin from the reviewed ACL. When the model also uses a
draft, LoRA adapter, or multimodal projector, derive its path-independent pin
from the reviewed deployment manifest:

```bash
a3s-power-verify --print-inference-execution-digest power.acl
a3s-power-verify --print-auxiliary-artifacts-digest model-manifest.json
```

New local-model reports always declare the inference policy, so strict clients
pin its digest. Strict verification requires the auxiliary pin whenever the
attested runtime declares auxiliary artifacts. That digest commits to roles, decoder contracts, sizes,
artifact hashes, and external-draft target binding; host-local paths are not
part of the identity.

The inference-execution digest commits to the normalized speculative mode and
MTP/FR controls, prompt-cache bounds, model residency, mmap/mlock, thread
count, Flash Attention, and parallel request slots from the fully resolved ACL
configuration. Environment overrides therefore change the digest. Use an
explicit `spec_mode` when a verifier must prove one exact decoder; `auto`
honestly commits to backend selection rather than pretending it selected MTP,
DFlash, or DSpark in advance.

The verifier selects acceptable launch measurements, artifact hashes, runtime
policy, GPU evidence, and receipt fields. The server does not get to weaken
those conditions.

Power verifies AMD SEV-SNP signatures, binds policy-visible fields to the exact
signed raw report, and checks nonce freshness, canonical model/runtime claims,
RA-TLS binding, request/response digests, and optional NVIDIA GPU/NVSwitch
topology and NRAS verdicts. Intel TDX currently emits a local TDREPORT and fails
strict verification until a reviewed DCAP Quote/QVL path exists. See
[Hardware Verifier Operations](docs/hardware-verifier-operations.md) for the
current production boundary and failure policy.

### Security boundaries

- Simulated TEE mode is development-only and fails strict verification.
- CPU TEE placement does not make GPU offload confidential; use verified
  `gpu-confidential` policy.
- Effective-prompt digests exist only for deterministic text paths. Opaque
  renderer paths leave the claim absent instead of fabricating it.
- Mixed quantization and reduced-vocabulary FR are experimental performance
  techniques, not universal quality guarantees.

## Documentation

| Guide | Scope |
| --- | --- |
| [Documentation home - 简体中文](https://a3s-lab.github.io/Power/) | Default `next` documentation in Simplified Chinese |
| [Documentation home - English](https://a3s-lab.github.io/Power/en/) | English `next` documentation |
| [v1.0.0 release documentation](https://a3s-lab.github.io/Power/v1.0.0/) | Versioned bilingual production-release snapshot |
| [v0.9.0 release documentation](https://a3s-lab.github.io/Power/v0.9.0/) | Versioned bilingual release snapshot |
| [Optimization playbook](docs/optimization-playbook.md) | Complete model-neutral map of graph, tensor, speculation, scheduling, storage, residency, accelerator, and evidence techniques |
| [Embedded Inference Architecture](docs/embedded-inference-architecture.md) | Graph execution, placement, scheduling, state, and receipts |
| [Model-Owned Shape Profiles](docs/shape-profiles.md) | Finite opaque classes, stale-binding rejection, fallback, and receipt v5 |
| [Model-Neutral Session Replicas](docs/session-replicas.md) | Exclusive mutable contexts, shared device admission, residency bounds, and cancellation |
| [Device-Resident Reviewed Graph Chains](docs/device-resident-graphs.md) | Same-request opaque handles, exact boundary validation, digest continuity, and explicit owned fallback |
| [Production Release Evidence Gate](docs/release-evidence-gate.md) | Strict platform coverage, immutable bindings, runtime failure proofs, and trust-root requirements |
| [v1 Production Support Matrix](docs/v1-support-matrix.md) | Required execution platforms, SEV-SNP boundary, TDX exclusion, and the machine-enforced release artifact contract |
| [External Metal and Confidential-GPU Capture](docs/external-release-capture.md) | Clean-revision hardware commands, raw vendor evidence, strict proof-backed promotion, and artifact inventory |
| [Model-neutral Speculative Decoding](docs/speculative-decoding.md) | Strategies, native MTP, patching, protocol, and acceptance |
| [Keyed Prompt-Prefix Cache](docs/prompt-prefix-cache.md) | Explicit API contract, tenant isolation, bounded KV lifecycle, metrics, MTP boundary, and paired cold/warm benchmark |
| [Qwen3.8-27B Q6_K benchmark](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md) | Performance gates, artifact identity, quality, and raw evidence |
| [Q6_K-only DFlash2 experiment](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2/README.md) | Exact upstream prototype, paired quality, offline evidence, and reproduction |
| [Reproduction guide](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) | CUDA build, pinned inputs, replay, audit, and validation |
| [Hardware Verifier Operations](docs/hardware-verifier-operations.md) | Production hardware-signature verification |
| [Supply-chain Audit](docs/supply-chain.md) | Feature profiles, native code, and threat model |
| [Storage Benchmark](docs/storage-benchmark.md) | Verified storage and residency measurements |
| [Tensor Batch Cost Benchmark](docs/tensor-batch-benchmark.md) | Model-neutral allocation, host-boundary copy cost, parity, and named-hardware reproduction |
| [Windows CPU/CUDA complete contract captures](docs/benchmarks/release-contract-windows-20260821/README.md) | Clean-revision peak memory, cancellation, queue expiry, replica recovery, fallback parity, raw JSON, and exact reproduction |
| [Windows CPU/CUDA P5 pre-captures](docs/benchmarks/release-gate-windows-20260821/README.md) | Clean-revision raw samples, hashes, negative evidence, reproduction, and explicit remaining gaps |
| [Roadmap](ROADMAP.md) | Acceptance gates and remaining work |
| [Changelog](CHANGELOG.md) | Released behavior |

Rust API documentation is published at
[docs.rs/a3s-power](https://docs.rs/a3s-power).

## Development

Run checks from this crate rather than the monorepo root:

```bash
cargo fmt --all -- --check
cargo test --locked --lib
cargo test --locked --no-default-features --features embedded-inference --lib
cargo test --locked --no-default-features --features picolm --lib
cargo clippy --locked --all-targets -- -D warnings

npm ci --prefix site
npm run typecheck --prefix site
npm run build --prefix site
npm run check:site --prefix site
```

CI validates formatting, Clippy feature matrices, tests, the listener-free
embedded boundary, release targets, and the GitHub Pages artifact.

## A3S ecosystem

| Project | Relationship |
| --- | --- |
| [A3S Box](https://github.com/A3S-Lab/Box) | Hosts Power inside SEV-SNP or TDX MicroVMs |
| [A3S Gateway](https://github.com/A3S-Lab/Gateway) | Routes inference traffic |
| [A3S Runtime](https://github.com/A3S-Lab/Runtime) | Supplies deployment-unit contracts |
| [A3S Code](https://github.com/A3S-Lab/Code) | Consumes local inference and verified bundles |
| [A3S Event](https://github.com/A3S-Lab/Event) | Distributes platform events |

Questions and design discussions are welcome on
[Discord](https://discord.gg/XVg6Hu6H).

## License

[MIT](LICENSE)
