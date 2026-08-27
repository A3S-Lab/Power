# A3S Power

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S Power provides a listener-free embedded inference library and an optional TEE-aware OpenAI-compatible server over one model-neutral runtime">
</p>

<p align="center">
  <strong>Model-neutral Rust inference with verifiable execution, TEE privacy, and canonical receipts.</strong>
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI status"></a>
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/release.yml?style=flat-square&amp;label=release" alt="Release status"></a>
  <a href="https://crates.io/crates/a3s-power"><img src="https://img.shields.io/crates/v/a3s-power?style=flat-square&amp;color=5420bd" alt="a3s-power on crates.io"></a>
  <a href="https://docs.rs/a3s-power"><img src="https://img.shields.io/docsrs/a3s-power?style=flat-square" alt="a3s-power documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#choose-the-surface">Surfaces</a> ·
  <a href="#responsibility-boundary">Boundary</a> ·
  <a href="#trust-chain">Trust chain</a> ·
  <a href="#embedded-inference-runtime">Embedded runtime</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#development">Development</a>
</p>

---

A3S Power is one model-neutral inference foundation with two deliberately
separate delivery surfaces: a listener-free embedded Rust library for
model-owning crates, and an optional OpenAI-compatible service for hosted LLM
inference. Both reuse the same typed devices, admission, weight integrity and
residency, cancellation, privacy controls, and execution evidence.

TEE deployment adds hardware-backed confidentiality and remote attestation.
Clients can bind a nonce, model/runtime policy, and request receipt to a
hardware report instead of relying only on an operator promise.

## Choose the surface

| Surface | Enable | Use it when | Network behavior |
| --- | --- | --- | --- |
| Embedded library | `default-features = false`, `embedded-inference` | An OCR, vision, language, or other model crate owns the graph and needs Power's shared runtime | Opens no listener; excludes Power's server and model hub |
| OpenAI-compatible service | Default `server` + `mistralrs` features | You need hosted chat, completions, embeddings, model lifecycle, metrics, attestation, TLS, or vsock | Explicit server process and configured transports |
| Minimal TEE service | `tee-minimal` | A small auditable GGUF layer-streaming deployment inside a constrained enclave | Server/TLS/vsock are explicit feature choices |

Embedded and server builds are not two inference engines. They are two entry
points into shared integrity, privacy, and execution contracts.

## Responsibility boundary

Power owns generic inference infrastructure. Model repositories keep model
semantics.

| Power owns | Model-owning crates own |
| --- | --- |
| Typed CPU/CUDA/Metal devices and bounded execution | Architecture, graph topology, layers, kernels, and exact arithmetic |
| Admission, cancellation, tensor and batch limits | Tokenizer, preprocessing, postprocessing, and generation policy |
| SafeTensors identity, replicas, mirrors, multi-root placement, and residency | Model assets, revision pins, tensor-name/shape contracts, and conversion |
| TEE integrity/privacy primitives, attestation binding, sealed state, and receipts | KV/recurrent state layout and any semantic interpretation of opaque state |
| Privacy-reviewed telemetry and hardware evidence envelopes | Product-level quality, parity, and named-hardware acceptance policy |

Power therefore embeds no PP-OCRv6, Unlimited-OCR, parser, or other product
model. A3S OCR owns its OCR models; A3S Parser consumes OCR and Office evidence
at the document layer.

## First successful build

Embed only the model-neutral runtime:

```toml
[dependencies]
a3s-power = { version = "0.8.0", default-features = false, features = ["embedded-inference"] }
```

```rust
use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};

fn main() -> Result<(), a3s_power::error::PowerError> {
    let _runtime = EmbeddedRuntime::new(
        DevicePreference::Auto,
        InferenceLimits::default(),
    )?;
    Ok(())
}
```

Or inspect the service surface:

```bash
cargo install a3s-power
a3s-power --help
```

## Trust model

Ordinary hosted inference asks the data owner to trust the machine operator.
That may be acceptable for many deployments, but it is insufficient when the
operator and data owner are different parties and prompts, responses, or model
weights require hardware-enforced confidentiality.

Power is designed for an explicit verifier: it checks the hardware report,
launch measurement, nonce, model/runtime claims, and request-level receipt.
Simulated TEE mode remains a development aid and is rejected by strict
verification.

## Trust chain

A3S Power runs LLM inference inside **Trusted Execution Environments** (AMD SEV-SNP / Intel TDX). The CPU encrypts all memory. The infrastructure operator **cannot** read prompts, responses, or model weights — the hardware enforces it.

But hardware isolation alone isn't enough. You need to **verify** it. Power provides a complete chain of cryptographic proof:

```
┌─────────────────────────────────────────────────────────────────────┐
│  a3s-box MicroVM (AMD SEV-SNP / Intel TDX)                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  a3s-power                                                    │  │
│  │                                                               │  │
│  │  1. Verify model integrity (SHA-256 + Ed25519 signature)      │  │
│  │  2. Bind model hash into hardware attestation report          │  │
│  │  3. Serve inference via OpenAI-compatible API                 │  │
│  │  4. Redact all inference content from logs and metrics            │
│  │  5. Zero all memory on model unload                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  Hardware-encrypted memory — host cannot read                       │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼  Client verifies independently:
┌─────────────────────────────────────────────────────────────────────┐
│  a3s-power-verify                                                    │
│  ✓ Nonce binding (prevents replay)                                   │
│  ✓ Model hash binding (proves which model is running)                │
│  ✓ Hardware signature (AMD KDS P-384 / Intel PCS P-256)              │
│  ✓ Platform measurement (proves unmodified code)                     │
└─────────────────────────────────────────────────────────────────────┘
```

The verifier, not the server operator, decides which measurement, model,
runtime policy, GPU evidence, and receipt fields are acceptable.

## What verification adds

| Evidence | What the verifier can check |
| --- | --- |
| CPU TEE report | SEV-SNP or TDX hardware signature, nonce freshness, and pinned launch measurement |
| Canonical claims | Model artifact, applied runtime policy, prompt/template policy, and optional confidential-GPU evidence |
| RA-TLS certificate | The attestation report carried in the TLS handshake |
| Inference receipt | Prompt-bearing request policy, decoding controls, effective prompt when available, and output digest |
| Encrypted model state | Ciphertext provenance plus configured plaintext integrity/signature binding |
| Private runtime behavior | Log redaction, optional token-count suppression, zeroization, cancellation, and bounded sealed state |

## Service overview

**A3S Power** is a privacy-preserving LLM inference server designed to run inside Trusted Execution Environments (TEE). It provides an OpenAI-compatible API for chat completions, text completions, and embeddings — with hardware-enforced memory encryption, model integrity verification, and automatic log redaction.

Power is built to run inside [a3s-box](https://github.com/A3S-Lab/Box) MicroVMs with AMD SEV-SNP or Intel TDX, ensuring that inference data (prompts, responses, model weights) never leaves the encrypted enclave.

## Capabilities

### Trust and verification

The service and verifier surfaces include:

- **TEE-Aware Runtime**: Auto-detects AMD SEV-SNP (`/dev/sev-guest`) and Intel TDX (`/dev/tdx_guest`) at startup; simulated mode for development (`A3S_TEE_SIMULATE=1`)
- **Remote Attestation**: Real hardware ioctl — AMD `SNP_GET_REPORT` and Intel `TDX_CMD_GET_REPORT0` — generates firmware-signed proof that inference runs in a genuine TEE; full raw reports included for client verification
- **Model/Runtime/GPU-Attestation Binding**: `GET /v1/attestation?model=<name>` re-hashes the current local model artifact (file, deterministic directory manifest, or encrypted artifact), emits an `AttestationClaimsV2` claim set, and binds `sha256(canonical_claims_v2)` into CPU TEE `report_data`; encrypted model pins cover decrypted plaintext when configured and include ciphertext provenance digests; model-bound claims include applied chat-template digests plus a canonical GPU execution/offload digest, and `tee_policy_mode = "gpu-confidential"` additionally binds NVIDIA GPU CC evidence, NRAS verdict digests, and structured NVIDIA device identity/freshness claims from live `nvattest-cli` collection or direct NRAS REST attestation using the same request nonce
- **Request-Level Inference Receipts**: Chat and text completion responses include `attestation_receipt` plus `attestation_receipt_sha256`, covering prompt-bearing API input, model runtime chat-template/GPU execution policy claims, exposed decoding parameters, streaming request options, stop tokens, response format, tools including function `strict` schema flags, tool choice, and parallel tool-call policy; local renderers, mistralrs text tokenization, and opt-in proxy upstreams can add an `effective_prompt` digest; streaming responses emit the receipt in a final SSE event before `[DONE]`
- **RA-TLS Transport**: TLS certificate embeds the attestation report as a custom X.509 extension (OID `1.3.6.1.4.1.56560.1.1`) — clients verify the TEE during the TLS handshake itself, no separate API call needed
- **Hardware Signature Verification**: Client-side SDK has `VerificationPolicy::strict()` / `verify_report_strict()` for fail-closed verification with mandatory hardware signatures, operator-pinned launch measurement verification, simulated-report rejection, optional required GPU evidence/device-claim/runtime policy checks, NVIDIA NRAS verdict digest pinning, NVIDIA GPU provider/format/evidence-count, exact GPU/NVSwitch topology, claims schema version, and GPU plus NVSwitch UEID/OEM ID/hwmodel/firmware pinning, GPU driver pinning, request receipt shape/digest/policy helpers, attestation-to-receipt runtime policy binding, and effective-prompt digest pinning when a receipt exposes it
- **Client Verification CLI**: `a3s-power-verify` defaults to strict verification with mandatory `--expected-measurement`; skipping hardware signatures and measurement pinning requires the explicit `--allow-offline` development/offline flag; `--hw-cert-cache-ttl-secs` tunes AMD KDS / Intel PCS certificate cache duration for strict verifier processes; NVIDIA GPU confidential-computing deployments can use `--gpu-confidential` to require v2 claims, top-level GPU evidence nonce freshness, structured device nonce freshness, pinned NVIDIA NRAS verdict digest binding, verifier-pinned GPU provider/format/count policy, structured NVIDIA device claims, verifier-pinned exact GPU topology, claims schema version, and identity/version policy, runtime policy, and a pinned GPU execution digest; individual GPU/runtime pins remain available with `--require-gpu-evidence`, `--require-gpu-device-claims`, `--gpu-provider`, `--gpu-evidence-format`, `--gpu-verdict-format`, `--gpu-evidence-count`, `--gpu-count`, `--nvswitch-count`, `--gpu-claims-version`, `--gpu-ueid`, `--gpu-oemid`, `--gpu-hwmodel`, `--gpu-driver-version`, `--gpu-firmware-version`, `--nvswitch-claims-version`, `--nvswitch-ueid`, `--nvswitch-oemid`, `--nvswitch-hwmodel`, `--nvswitch-firmware-version`, `--require-runtime-policy`, and `--gpu-execution-digest`; `--print-gpu-execution-digest` computes GPU execution pins with Power's canonicalizer
- **Encrypted Model Loading**: AES-256-GCM file-backed `DecryptedModel` loading with zero-overwrite cleanup; `in_memory_decrypt = true` loads verified plaintext directly from `MemoryDecryptedModel` locked RAM when the backend supports it (`picolm` GGUF), otherwise fails closed; `streaming_decrypt = true` passes `LayerStreamingDecryptedModel` plaintext to supporting backends (`picolm` GGUF), with unsupported backends failing closed. Configured encrypted-model integrity pins and signatures are checked against decrypted plaintext SHA-256, with ciphertext SHA-256 exposed separately in attestation claims
- **KeyProvider Trait**: Abstract key loading for HSM integration; `StaticKeyProvider` (file/env) + `RotatingKeyProvider` (zero-downtime rotation)
- **Deep Log Redaction**: Strips inference content from all log output — 10 sensitive JSON keys (`content`, `prompt`, `text`, `arguments`, `input`, `delta`, `system`, `message`, `query`, `instruction`); `sanitize_error()` strips prompt fragments from error messages; `suppress_token_metrics` rounds token counts to nearest 10 to prevent side-channel inference
- **Memory Zeroing**: `SensitiveString` wrapper auto-zeroizes on drop; all inference buffers cleared via `zeroize` crate — the operator cannot recover prompts or responses from memory dumps
- **Model Integrity**: SHA-256 hash verification at startup + Ed25519 publisher signatures; fails fast on tampering
- **Model-neutral speculative decoding**: Public `a3s_power::speculative` proposal, adaptive-length, and exact-acceptance primitives are shared across model backends. Model crates provide trained draft/target adapters and transactional architecture state, so [DSpark](https://arxiv.org/abs/2607.05147) integration is not tied to a model name. picolm supplies zero-weight baselines, while compatible llama.cpp models can use native MTP; explicit unsupported strategies fail closed. The `a3s-power-speculative-bench` client captures digest-pinned baseline/candidate evidence through Power's real SSE API and rejects output, workload, runtime-setting, or hardware drift. See [Model-neutral Speculative Decoding](docs/speculative-decoding.md).
- **picolm Layer-Streaming**: Pure Rust GGUF inference with true O(layer_size) peak RAM via `madvise(DONTNEED)` page release after each layer. Real transformer ops: multi-head/GQA attention, SwiGLU/GeGLU FFN, RoPE, RMSNorm. FP16 KV cache with fused f16 dot/accumulate (no intermediate buffer). Fused dequant+dot kernels. NEON SIMD (aarch64) + AVX2 (x86_64). Rayon parallel matmul. Pre-computed RoPE tables. Batch prefill, tool calling, grammar-constrained output. Selectable speculative-decoding modes (`spec_mode`: `auto` / `off` / `prompt-lookup` / `ngram-context`; `auto` resolves to prompt lookup) with **batched layer-streaming verify** — a draft block is verified in one weight-streaming pass instead of one pass per token — adaptive draft length, and lossless acceptance (output matches plain decoding for the same seed). Zero-alloc hot path. 14+ tok/s decode on Apple Silicon. Enables 7B+ models inside 512MB TEE EPC. No C/C++ inference backend, ~4,500 lines of fully auditable Rust.
- **Pure Rust Inference Path**: Default backend via `mistralrs` (candle) — no C++ inference engine in the trusted computing base; the `tee-minimal` build (~1,220 dep tree lines) is the smallest auditable LLM inference stack that exists

### Server inference

The optional server surface includes:

- **OpenAI-Compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` — works with any OpenAI SDK
- **True Token-by-Token Streaming**: Per-token SSE delivery via `stream_chat_request`; opted-in text-completion streams expose exact final usage plus server-side first/last-token timing for controlled performance evidence when token-metric suppression is disabled
- **Multiple Backends**: mistralrs (pure Rust, default), llama.cpp (C++ bindings, optional), picolm (TEE layer-streaming, optional), proxy (forwards to an upstream OpenAI-compatible server — vLLM/TGI/SGLang/OpenAI — so Power can front an existing accelerated engine)
- **Model Formats**: GGUF, SafeTensors (ISQ quantization), Vision/Multimodal (LLaVA, Phi-3-Vision), HuggingFace Embeddings (Qwen3, GTE, NomicBert)
- **Embedded Inference Runtime**: Model-neutral Rust library primitives for reviewed static graphs, bounded admission, continuous ragged execution lifecycles, exact SafeTensors integrity, multi-root capacity aggregation, mmap-default or bounded positional tensor reads, independently authenticated seekable AES-256-GCM weight collections, complete or partial weighted read-only replicas, digest-pinned and canonically verified pure-Rust lossless rANS representations, usage-ranked verified partial-mirror staging, validation-throughput source weighting, typed devices, fixed-state/scratch-aware host/CUDA/Metal cache budgets with current-pressure revalidation, LFRU/LRU residency, hysteresis-bounded live hot-tier adaptation, privacy-gated cross-layer route hints, event-driven current-layer staged batches with a shared worker-and-byte flight window, attestation-bound accelerator residency declarations, bounded heterogeneous device meshes and peer copies, fused Candle batches with explicit exact-fallback identity, bounded sealed warm-state envelopes, digest-bound lossless tuning evidence, canonical privacy-reviewed hardware evidence bundles, measurable prefetch, cancellation, and execution receipts. Model architectures and KV/recurrent layouts live in their owning crates; embedded sessions never bind a Web port
- **GPU Acceleration**: Auto-detection of Apple Metal and NVIDIA CUDA; configurable layer offloading, multi-GPU support
- **Tool/Function Calling**: Structured tool definitions with XML, Mistral, and JSON output parsing
- **JSON Schema Structured Output**: Constrain local llama.cpp output via JSON Schema → GBNF grammar conversion; unsupported local backend/schema combinations fail closed instead of silently ignoring output policy
- **Thinking & Reasoning**: Streaming `<think>` block parser for DeepSeek-R1, QwQ reasoning models
- **Chat Template Engine**: Jinja2-compatible rendering via `minijinja` (Llama 3, ChatML, Phi, Gemma, custom); model-provided raw templates fail closed on render errors instead of silently switching prompt formats
- **KV Cache Reuse**: Prefix matching across multi-turn requests for conversation speedup
- **Remote Model Hub Pull**: `POST /v1/models/pull` with SSE progress, Range resume, concurrent dedup, source-specific token auth for ModelScope or HuggingFace Hub

## Embedded inference runtime

The `embedded-inference` feature is a library path, separate from Power's HTTP
server and backend registry. It provides a shared runtime, reviewed static-graph
execution, exact SafeTensors inventories, typed devices, hard resource bounds,
cancellation, canonical execution receipts, bounded parallel prefetch with
useful/unused accounting, event-driven current-layer staging with canonical
group reassembly, deterministic heat-driven weight placement, and
digest-verified bandwidth-weighted storage replicas. One logical source may
span an anchor root plus explicit disjoint `shard_roots`, allowing immutable
SafeTensors shards to use aggregate capacity across multiple volumes without a
complete copy on any one volume. Every root must be non-empty, real, and
non-overlapping; relative paths must be unique across the source. File and byte
limits apply to the combined collection, while its canonical digest remains
identical to the same relative files under one root. This is capacity
aggregation inside the existing `WeightStore`, not another source router,
cache, integrity path, or model format. The runtime can also discover bounded
host, CUDA, and Metal memory snapshots and derive an explicitly requested cache
budget with caller-owned safety reserves, available-memory fractions, caps,
allocation order, and typed host/device fixed-state plus peak-scratch
reservations. Those runtime bytes are removed before hot-weight cache capacity;
Metal unified memory counts both reservation sets against one physical pool.
`EmbeddedRuntime::apply_residency_budget` takes a fresh native snapshot and
fails before changing cache policy if availability or pool topology no longer
supports the plan; plans with non-zero runtime reservations cannot use the
offline-only `ResidencyBudgetPlan::apply_to` shortcut. Automatic budgeting
remains opt-in, never changes the zero-cache default, and never logs
reservations/snapshots or adds them to receipts. A replica may explicitly cover
the complete collection or an exact subset of primary SafeTensors files; source
weights can reuse throughput observed during the mandatory integrity read
without scanning the model again. Hot plans can be replaced transactionally
without releasing explicit caller pins. At a model-defined safe boundary,
caller-owned live heat can derive an ephemeral incremental plan with a bounded
replacement count and deterministic hysteresis. Only groups with identical
byte and per-layer entry footprints exchange tiers, and a stale base plan fails
closed. This changes placement only; routing, gate values, dtype, precision,
and tensor contents stay exact. A model crate may also provide opaque positive
benefits for complete SafeTensors files and ask the same `WeightStore` to plan
and stage a usage-ranked partial mirror. Staging is denied by default, requires explicit
caller-managed plaintext authority, respects a byte budget and free-space
reserve, resumes exact files, detects source mutation, and atomically publishes
only digest-verified files without replacement. Power never persists the
benefits, plan, destination, or a mirror receipt automatically. It never
downloads a model, starts another process, or opens a listener.

Each CUDA model-session `RuntimeDevice` owns one cuBLAS handle/stream and one
lifetime-bound, 256-byte-aligned user workspace. General CUDA runtimes retain
the vendor default pool. Power allocates the CUDA 12.6 recommended
32 MiB on Hopper (compute capability 9.x) and 4 MiB on other supported
architectures, then fails device construction if allocation, alignment, or
binding fails. This replaces process-global workspace configuration and keeps
concurrent model-session replicas independently reproducible without inspecting
a model, graph, tensor geometry, input, or timing. The fixed allocation is made
before later native-memory snapshots, and Power never reassigns the handle's
stream during execution because cuBLAS would reset the workspace. At teardown,
Power synchronizes that same stream and deliberately resets the handle to its
default pool before freeing the buffer, so an independently retained Candle
device cannot inherit a dangling workspace pointer. Two independent 64-page
downstream OCR processes now reproduce the static-session control
exactly after removing only the declared elapsed field; guarded throughput is a
separate downstream gate.

Static-graph execution releases eager intermediate tensors immediately after
their last declared consumer while retaining constants and the declared graph
output. After full graph validation and private-`Identity` elision, executor
construction also resolves a private `Reshape` once when both inputs are
constants, its tensor is contiguous and nonempty, its output is not published,
and the result stays inside the declared tensor-element bound. A supported
reshape is metadata-only and preserves scalar-constant identity; dynamic,
published, non-contiguous, malformed, or oversized forms remain executable
nodes with the existing runtime behavior. This topology-only rewrite can expose
an adjacent generic convolution channel-bias window without inspecting model,
node, source, value, corpus, or measured-shape identity.

On the current reviewed layout graph, an opt-in trace reduces executed
`Reshape` nodes from 28 to 12 and sends exactly 16 following channel-bias adds
through the existing convolution fusion. Generic CPU/CUDA tests and the full
64-page Parser corpus retain exact current output, reconstruction, table-cell,
and seal-position evidence. The first full-corpus timing comparison used
unequal tracing, and the first strict trace-free cohort stopped before launching
either binary when unrelated compilers appeared. The work removal is therefore
retained without a stable throughput claim; Parser's canonical negative-result
ledger records both invalid attempts and the pre-existing strict text golden
that still fails identically on the frozen baseline.

Power also privately lowers exact CUDA F32 sigmoid products. Two adjacent
private Sigmoids with equal contiguous shapes and their terminal Mul may execute
in one pass. More generally, a private full-shape `Sigmoid -> Mul` edge may
execute in one pass while reusing an already materialized multiplier with the
same shape, an NCHW per-channel `[N, C, 1, 1]` shape, or an NCHW per-spatial
`[N, 1, H, W]` shape. The broadcast form deliberately retains the smaller
gate's earlier Sigmoid instead of recomputing its exponential across channels.
Admission uses only topology, use counts, dtype, device, contiguity, tensor
geometry, cancellation, and declared element bounds; unsupported forms keep
ordinary node execution.

Generic CUDA tests are byte-exact across unrelated shapes, both product operand
orders, and nonzero storage offsets. The current PicoDet-L trace consumes four
active `Sigmoid -> Mul` pairs per graph call, changing Sigmoid accounting from
`12 executions / 12 nodes` to `12 / 16` and Mul accounting from `14 / 18` to
`10 / 14`. Parser validation retained exact normalized caches for five PDFs and
64 distinct pages, then reproduced all 74 A3S Office reconstruction artifacts
byte-for-byte, including `68/68` positioned merged-table cells and `10/10`
positioned rider seals. A guarded throughput comparison has not yet admitted a
clean sample, so this is deterministic work removal without a speed claim.

The follow-up lowers one exact private `BatchNormalization -> Sigmoid` edge in
the normalization output pass. Swish is matched first and remains a distinct
three-node formula; the Sigmoid-only path requires exactly one normalized-value
consumer and does not extend or reorder CUDA pointwise convolution. CPU and CUDA
formula tests are byte-exact, and the complete graph suites pass. On the same
official Layout graph, BatchNormalization accounts for `52 / 152`
executions/source nodes, Sigmoid falls to `8 / 12`, and Mul remains `10 / 14`,
removing four more execution boundaries per call. The combined 64-page cache and all 74
Office files remain exact. Its unguarded 10.665-pages/s diagnostic is not a
throughput claim; the following strict snapshot exceeded the GPU-idleness gate
before either frozen binary ran.

BatchNormalization now prepares each static channel's exact
`sqrt(variance + epsilon)` once when the graph executor is created. CPU
preparation retains Rust's existing F32 expression; CUDA uses the same
round-to-nearest add and `sqrtf` operations that previously ran for every output
element. Ordinary BatchNormalization and its depthwise and spatial convolution
post-operations then preserve the existing `sub -> div -> mul -> add` sequence.
Focused real-GPU tests are byte-exact, both complete graph suites retain their
prior counts, all five normalized 64-page Parser caches match the preceding
candidate, and all 74 Office artifacts match by SHA-256. The unguarded
correctness run reported 9.741 pages/s, which is excluded from performance
attribution; the fixed-order quiet-host comparison remains open.

An experiment let an exact private, bias-free CUDA F32 pointwise `Conv ->
BatchNormalization` edge normalize the freshly allocated GEMM result in place.
The low-level kernel uses one mutable values pointer so cudarc records a write
for a consumer on another stream. A dedicated two-stream regression, all five
real-CUDA BatchNormalization cases, and the complete CPU/CUDA suites pass. The
frozen downstream Parser executable, SHA-256
`4377c5edcf09b7095e7f300761e21e6f087bfd75ed0dff149c31728eb56345ab`,
also retained the exact normalized 64-page cache and all 74 A3S Office files.
After two fully discarded compiler-contaminated attempts, a 300-second
quiet-qualified fixed-order eight-run A/B measured control/candidate means of
6,185.25/6,213.5 ms, medians of 6,146.5/6,268.5 ms, and two candidate wins in
four adjacent pairs. The experiment failed every promotion condition, so graph
execution does not select this in-place pointwise post-operation. The mutable
argument/event-ordering implementation remains covered as correctness
infrastructure; production keeps the ordinary two-output path for every model
and geometry.

On CUDA, multiplier-one F32 depthwise `Conv` nodes execute one fused
kernel per node instead of materializing one device-wide multiply and add for
every kernel position. The kernel supports independent spatial strides,
dilation, arbitrary NCHW storage strides, and optional fused bias. Explicit
round-to-nearest multiply/add operations preserve the prior accumulation order;
an RTX 4090 parity gate is byte-exact against the reviewed scalar accumulation.
Adjacent single-consumer F32 `HardSigmoid`-to-`Mul` pairs also execute as one
CUDA kernel when both operands have the same rank-four shape or the activation
is an exact contiguous `[N, C, 1, 1]` channel gate over `[N, C, H, W]`. The
specialized path preserves the original two affine stages, ordered clamp, and
final multiplication with explicit round-to-nearest operations. It removes
four intermediate activation buffers and four launches per matched pair
without changing the admitted graph or its public schema. Adjacent,
single-consumer F32 `Div`-`Erf`-`Add`-`Mul`-`Mul` chains with three scalar
initializers are likewise lowered to one CUDA kernel. Scalar initializer values
are captured once while the reviewed weights are loaded; explicit
round-to-nearest division, addition, and both multiplication stages retain the
five-node result byte-for-byte while avoiding four intermediate buffers and
four launches per matched chain. When the same exact private activation chain
consumes a contiguous F32 `BatchNormalization` result, CPU and CUDA execution
can apply the activation in the normalization output pass. The matcher requires
the normalized value's exact two uses inside the GELU formula, finite scalar
initializers, constant channel parameters, and no retained or shared
intermediate. This removes the normalization-to-activation buffer and one
launch while preserving every normalization and activation rounding boundary.
When a two-input `Conv` feeds an exact
`[1, C, 1, 1]` channel-bias `Add` through at most two private identities into
ReLU, the reviewed error-function activation, or a gated HardSigmoid multiply,
the executor keeps the convolution on its ordinary backend and folds the bias
into the activation kernel. This removes one additional full-tensor buffer and
launch. The CUDA kernel uses bounded 32-bit channel indexing under Power's
existing `u32` element-count gate and preserves the original F32 addition and
activation rounding boundaries byte-for-byte.
The pointwise tail of a decomposed last-axis LayerNorm can also execute as one
CUDA kernel when its private topology is exactly
`Add(epsilon)`-`Sqrt`-`Div`-`Mul(scale)`-`Add(bias)`, epsilon is one positive
F32 scalar initializer, and scale/bias are exact contiguous feature vectors.
Both mean reductions, centering, and squaring stay on the ordinary graph path,
so reduction order is unchanged. Explicit round-to-nearest arithmetic retains
all five pointwise rounding boundaries byte-for-byte while removing four
intermediate buffers and four launches per matched tail.
On CPU, contiguous F32 group-one pointwise and spatial convolutions use direct
NCHW kernels, and multiplier-one depthwise convolution partitions fresh output
by batch/channel while retaining kernel-row/kernel-column FMA order. Exact
private pointwise or spatial `Conv -> channel bias -> ReLU/GELU` windows apply
their post-operation in the convolution output buffer. If a graph job is
already executing on a Rayon worker, pointwise GEMM and its post pass stay
inside that job instead of requesting the full pool again; standalone calls
retain internal parallelism. A bitwise test compares both execution contexts.
An exact private `Conv -> BatchNormalization` edge can likewise normalize the
direct pointwise, spatial, or depthwise output in that buffer, optionally
followed by the existing private HardSwish form. Constant contiguous F32
parameters must match the output-channel count, convolution bias remains before
normalization, and every subtraction, square root, division, multiplication,
addition, clamp, and final multiplication keeps its graph order. Shared edges,
retained intermediate outputs, unsupported layouts, dtypes, devices, or direct
convolution forms keep node-by-node execution.
An exact private `Conv -> channel bias` edge without an activation now uses the
same convolution bias path for grouped and group-one convolution. A group-one
pointwise window may additionally consume one private identity chain and one
exact-output-shaped F32 residual add in the convolution output buffer. Channel
bias must be `[C]` or `[1,C,1,1]`; residual broadcasting, shared values,
non-contiguous layouts, and non-pointwise forms retain ordinary execution. The
addition order remains `(convolution + channel bias) + residual`.
The `gemm` dependency includes its x86-v4 microkernels, selected only by its
runtime CPUID dispatch. AVX-512F hosts can therefore use the wider reviewed
kernel while all other x86 and non-x86 hosts retain the existing FMA, SIMD, or
scalar implementation; the crate is not compiled for the build host with
`target-cpu=native`.
The depthwise path also relies on the output allocation's existing zero state
instead of clearing each untouched interior segment a second time. On x86 and
x86-64, a contiguous stride-one interior row uses AVX2/FMA when the host exposes
both instructions and the row contains at least one eight-lane hardware vector.
Each lane retains the scalar kernel-row/kernel-column FMA sequence, the scalar
tail is unchanged, and an optional channel bias is added only after the complete
accumulation chain before the final store. Other strides, shorter interiors,
architectures, and instruction sets retain the scalar path. The eight-value
boundary is the ISA width, not a measured model or corpus threshold. These
choices use only device, execution-context, topology, dtype, layout, and tensor
geometry facts; no model label, input value, or caller identity participates.
Other devices, dtypes, and convolution layouts retain the existing Candle path.
Unsupported gated-activation shapes and layouts likewise retain ordinary
node-by-node execution. Unsupported error-function chains, constants, devices,
dtypes, layouts, graph outputs, channel biases, identity depths, LayerNorm
broadcast shapes, and shared intermediates do the same. CPU fallback, full
graph execution on an RTX 4090, and byte-exact CUDA comparisons cover the
reviewed fused forms.

On the Xeon w5-2445, bidirectional retained-binary A/B runs kept exact OCR
evidence while x86-v4 reduced 21-page Text full-source latency by 9.9--13.6%
and window latency by 4.6--14.9%. The six-page Text-plus-Table gate fell from
6.040/6.287 seconds to 5.485/5.547 seconds, and the 29-page Text-plus-Seal
diagnostic fell from 40.016/41.492 seconds to 36.247/36.891 seconds. This is
same-host development evidence, not a cross-machine throughput claim.
For the later private Conv-BatchNorm slice, four alternating-order 29-page
Seal-only runs measured 6.696--7.214 seconds for the candidate and
7.755--9.006 seconds for its retained baseline; the first two adjacent pairs
reduced latency by 13.4--13.7%. A traced pair moved the timed stage from 7.652
to 6.575 seconds and summed graph time from 14.044 to 12.310 seconds, with no
independent BatchNormalization execution left in the matched graph. Two
opposite-order Text-only pairs measured 30.533 versus 31.280 seconds and
31.809 versus 32.559 seconds, a 2.3--2.4% reduction. Bitwise operator tests,
the exact six-page cross-page-table gate, the 29-page reviewed seal-position
gate, and the independent 35-page seal-precision gate passed. The existing
full rider Text fingerprint remains outside its reviewed golden set, but the
candidate and retained baseline produce the same fingerprint; the golden was
not relaxed. These are same-host development measurements, not a complete
document-throughput or cross-machine claim.

For the later private `BatchNormalization -> erf-GELU` slice, an alternating
retained-binary 29-page full-stage A/B run measured a 3,232.997 ms baseline
mean and a 3,156.856 ms candidate mean, a 2.36% reduction. A recognition-graph
profile measured the fused normalization/activation work about 37.5% faster,
saving 182--192 microseconds in the reviewed batch-128 graph. Both binaries
produced 2,518 text blocks, text SHA-256
`8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8`, and
semantic SHA-256
`49efe252b380179b4385eb114adf1897a9cbe7f1b949a7e9b6037958158cd1ff`.
These are same-host development measurements under variable shared-GPU load,
not a stable 10-page/s or cross-machine throughput claim.

A subsequent model-neutral CUDA layout path lets an exact rank-three F32
last-two-axis transpose view feed a contiguous rank-two right-hand matrix
directly through cuBLAS strided GEMM. The gate checks only device identity,
dtype, rank, contiguity, nonzero compatible dimensions, and the exact
transpose stride; every other layout still materializes through the reviewed
fallback. A recognition probe reduced 386 matching transpose kernels totaling
989.750 microseconds to zero and remained bit-exact across three unrelated
matrix geometries. Two precommitted interleaved 29-page cohorts also retained
2,518 blocks and the text/semantic SHA-256 values above. The three-sample-per-
side cohort moved from a 3,014.101 ms baseline mean to 2,981.494 ms (1.08%),
while the later six-sample-per-side cohort under heavier shared-GPU load moved
from 3,549.776 to 3,540.427 ms (0.26%). Across all nine samples per side, means
were 3,371.218 and 3,354.116 ms (0.51%) and medians were 3,412.336 and
3,373.865 ms (1.13%). This is a small same-host work-elimination result under
variable contention, not a stable throughput or cross-machine claim. The
strict downstream rider gate still rejects the shared semantic fingerprint
against an older reviewed golden; no expectation was relaxed for performance.

A further model-neutral lowering combines an exact private
`Add(last-axis bias) -> Sigmoid -> Mul` formula into one contiguous CUDA F32
pass. Matching depends only on normalized topology, use counts, rank, exact
last-axis geometry, device, dtype, layout, cancellation, and the declared
element bound; every unsupported form retains ordinary graph execution. An OCR
inventory gate locks two source windows without passing model or node identity
into Power. A launch-blocked full-stage trace observed 142 dynamic matches and
removed all 142 standalone Sigmoid launches plus 142 of the 284 Mul launches.
Two precommitted interleaved 29-page A/B cohorts improved independently. Six
samples per binary moved from a 3,081.185 ms baseline mean to 2,932.748 ms,
and a reverse-order four samples per binary moved from 3,162.325 to 3,037.512
ms. Across all ten samples per binary, mean latency fell from 3,113.641 to
2,974.654 ms (4.46%), median latency from 3,045.915 to 2,950.303 ms (3.14%),
and mean throughput rose from 9.337 to 9.757 pages/s. Every run retained 2,518
blocks and the text/semantic SHA-256 values above. Individual samples crossed
10 pages/s, but this is not stable 10-page/s or cross-machine evidence; the
older downstream semantic golden remains unchanged.

The current follow-up begins that post-operation at an exact private `MatMul`
boundary and reuses the GEMM output. Admission uses normalized topology,
liveness, F32 rank/layout/geometry, same-device CUDA facts, cancellation, and
declared bounds only. It accepts arbitrary nonempty left prefixes with a
rank-two right matrix and an exact rank-one output-column bias, and it composes
the already-retained Swish formula when that exact private tail follows. The
OCR inventory locks nine source `MatMul -> Add` windows; its existing terminal
classifier projection still owns one, so six internal bias windows and two
internal bias-plus-Swish windows remove eight allocation/free pairs per graph.
The kernel count and explicit F32 rounding sequence are unchanged from the
preceding biased-Swish baseline. Rank-two through rank-four and nonzero-offset
CUDA tests are byte-exact.

The first six-sample-per-binary full-stage cohort improved 2.35% by mean and
3.95% by median. A reverse-order four-sample cohort improved only 0.34% by mean
and regressed 0.66% by median. Combined means moved from 3,024.159 to 2,976.892
ms (1.56%), medians from 2,999.025 to 2,885.866 ms (3.77%), and mean throughput
from 9.589 to 9.742 pages/s; seven of ten interleaved pairs favored the
candidate and every run retained 2,518 blocks plus both hashes above. A noisy
20-sample-per-binary isolated-graph comparison improved mean and 10% trimmed
mean by 3.94% and 1.77%, but regressed median from 9.634 to 10.128 ms. The
generic allocation removal is retained with this mixed evidence; it is not a
stable 10-pages/s claim. The preceding standalone MatMul-bias attempt was not
timed or promoted because it intercepted the Add nodes and re-exposed two
Sigmoid and two Mul executions per graph. Parser's negative-result ledger
records that failed composition and all other known unpromoted attempts; Power
does not recover any of them with a model, node, document, value, fingerprint,
corpus, or measured-shape selector.

Large leading-axis CUDA graph calls exposed a separate numerical boundary: one
unpartitioned batch-128 launch changed earlier F32 convolution values relative
to batches of at most 32, despite identical logical rows. The first generic
drift appeared at a spatial convolution; batch 33 remained exact, while batch
34 and larger did not. Power therefore applies a fixed 32-item leading-axis
reduction quantum to direct pointwise GEMM and the GEMM phase of direct spatial
convolution. Spatial lowering still runs once, and every reduction partition
writes directly into one final allocation. The quantum is an executor
reproducibility and workspace bound selected from tensor shape only; it is not
an OCR, model, document, content, or measured-shape dispatch rule. A complete
481-node batch-128 recognition probe compared all four partitions and found
zero differing bits across 95,795,200 F32 values.

The throughput evidence remains mixed. One direct-output 29-page pair moved
from 3,042.835 ms at batch 32 to 2,995.660 ms at batch 128, or 9.531 to 9.681
pages/s, while a one-pass batch `64..256` sweep was non-monotonic and fell as
low as 7.061 pages/s at batch 160. All 41 available explicit cuBLAS algorithm
choices failed to reproduce the at-most-32 result for a generic pointwise
case, each changing 8,104 F32 words. A four-lane follow-up retained exact OCR
hashes but regressed to 9.190 pages/s at batch 32 and 8.848 pages/s at batch
128, so it was reverted. OCR's public batch default remains unchanged; these
results do not establish stable 10 pages/s.

For the later channel-bias/residual slice, a content-free linear-input
recognition-graph probe on the same Xeon reduced the retained small-profile
`[16,3,48,320]` median from 370.671 ms to 323.096 ms across the recorded
before/after runs. The graph trace no longer contains 13 standalone channel-bias
adds or 11 exact-shape convolution residual adds, and its bounded output kept
SHA-256 `61bccf1df2c55a8809f32f93a1ac1349efeaa449f04281f4ad6ee15789ce2aea`.
The tiny-profile diagnostic moved from about 103.430 ms to 84.539 ms at the
same geometry. These are model-executor diagnostics with mathematical inputs,
not OCR accuracy, document throughput, or support evidence.

Downstream one-variable diagnostics also removed direct spatial convolution,
terminal output projection, outer OCR graph-job parallelism, and CPU
convolution-bias activation separately. Every build retained the same
unreviewed rider Text fingerprint; serializing the outer jobs instead regressed
29-page latency to 74.554 seconds. The historical reviewed CPU fingerprint was
produced under a different OCR width-planning identity, and restoring that
planner allowance on the current stack did not reproduce it. Power therefore
does not add model-, document-, value-, or fingerprint-driven dispatch to chase
an OCR corpus result. Accuracy ownership remains with the digest-bound OCR model,
planner, and independent reconstruction gates.
A worker-occupancy pointwise candidate reduced one content-free batch geometry
from 1.654--2.076 ms to 0.885 ms and passed the bitwise pointwise suite, but the
29-page Text stage took 31.071 seconds between retained baseline runs of 30.670
and 30.874 seconds. It was reverted without adding a batch-count or
measured-shape exception.
On the same host, a temporary content-free CPU matrix probe covered four broad
geometries: F32 took 0.693--0.957 ms, F16 took 1.421--1.774 ms, and the active
backend did not support BF16 matrix multiplication. The probe was removed.
Power therefore retains the declared F32 graph boundary; host ISA support alone
does not authorize implicit conversion or document-dependent precision.
Canonical F32-tensor and token-ID receipt payloads hash their contiguous
read-only bytes directly on little-endian hosts; big-endian hosts retain the
same v1 little-endian byte protocol through bounded staging. Both paths preserve
the exact receipt digest while avoiding one hasher update call per value.
`GraphExecutor::run_with_output_projection` additionally lets the owning model
apply a deterministic same-device projection before the bounded F32 result is
copied to the host. The model must bind that arithmetic into its execution
identity; Power still enforces the original permit and cancellation boundary,
rejects a device change, and applies the normal tensor, dtype, and finite-value
checks to the projected result.
For a row-coalesced terminal classifier window, Power now materializes the
already contiguous `[total_rows, projected_width]` result on the host once and
then partitions the validated values into the original leading shapes and
order. The former path materialized every device slice independently. This
removes `window_outputs - 1` device-to-host materializations and possible
synchronization boundaries without changing projection arithmetic, model or
scheduler authority, or the returned F32 values. Same-tensor CPU and CUDA tests
are bit-exact, the complete graph suites pass `121/0/9` and `125/0/42`, all five
normalized 64-page Parser cache wires match the preceding candidate, and all 74
A3S Office artifacts match byte-for-byte. The unguarded correctness run emitted
about 10.347 pages/s; it is not a throughput claim, and the frozen fixed-order
comparison remains pending because the latest preflight found unrelated
compiler activity.
The corresponding CUDA input boundary now performs one bounded upload for a
multi-input execution window. After aggregate element and byte validation,
Power concatenates the already owned finite F32 values, uploads one flat
tensor, and restores every original shape and order as a contiguous read-only
view. Single-input, CPU, Metal, and empty-value behavior retain their existing
paths. This removes `window_inputs - 1` CUDA allocations and host-to-device
upload-return synchronization opportunities without changing graph execution.
An explicit CUDA test locks exact shapes, values, and nonzero storage offsets;
an independent terminal-classifier window matches individually uploaded calls
bit-for-bit. Complete graph suites pass `121/0/9` and `125/0/43`; all five
normalized 64-page Parser caches and all 74 Office artifacts remain exact. The
unguarded OCR diagnostic was about 10.435 pages/s, not promotion evidence. The
frozen Parser candidate has SHA-256
`945ee546912f5afa26f54e1beee8d1e9bc136aecec86cdb38555ce4d7e1ee653`;
its first strict preflight admitted zero samples because unrelated compilers
were active.
The graph planner also composes consecutive private `Transpose` permutations
when both are explicit complete same-rank permutations, the intermediate has
one consumer, and no retained output owns it. Fanout, retained values, default
or invalid permutations keep the ordinary path; the planner never assumes
runtime rank. An inverse pair becomes one identity-permutation node, then
returns the input as a zero-copy view only after executor rank/permutation
validation. Six generic planner tests and the complete CPU/CUDA graph suites
pass (`129/0/9` and `133/0/43`). The current recognition trace changes from
three `Transpose` plan executions to two; deterministic CPU output, all five
normalized 64-page OCR
caches, and all 74 Office artifacts remain exact. The frozen Parser candidate
is SHA-256
`681e2f60981b205a3832bccdd800410808dd521435e71f26bc7bc8669d29cb57`.
Its unguarded 10.511-pages/s diagnostic is not promotion evidence; the strict
immediate one-upload-versus-folded A/B remains pending because unrelated
compiler processes are active.
The preceding transfer choices do not alter graph topology or model-owned
preprocessing. Neither transfer nor planning changes execution-receipt
semantics, and tensor-limit failures report both the observed and configured
element counts.

Embedded scheduling uses the same runtime admission primitive at two bounded
levels. `InferenceLimits::max_queued_requests` caps each model runtime's async
waiting path (`begin_wait`); zero keeps that path fail-fast. A device-bound
`ModelSessionPool<T>` additionally caps exact model sessions, declared resident
bytes, concurrent device work, and the cross-session device queue. Concurrent
callers initialize one exact model/execution/resource declaration once, while
cancellation or future drop releases an unfinished pool slot. Ready entries are
retained until the pool is dropped; Power does not add eviction, persistence,
networking, or model semantics.

Each CUDA-backed pool entry owns one Candle device identity and one execution
stream, and its public tensor boundary remains host-owned. Power therefore
omits cudarc's per-activation cross-stream event tracking only for these
isolated model-session lanes. Ordinary `RuntimeDevice` users and accelerator
meshes retain event tracking. This changes scheduling overhead, not graph
operators or arithmetic. In a downstream 29-page RTX 4090
Orientation+Text+Table+Seal gate, six interleaved cold-process samples per side
reduced the median from 4,030.936 to 3,672.701 ms (7.194 to 7.896 pages/s) and
the p90 from 4,161.525 to 3,895.917 ms, while exact text and non-receipt
semantic SHA-256 values were unchanged. This is decoded-raster integration
evidence, not complete PDF parsing or reconstruction throughput.

`plan_microbatches` accepts model-owned digest identities and explicit per-slot
input, state, host-peak, and device-peak resource declarations. It groups
contiguous slots deterministically under caller policy, runtime limits, and a
fresh host/device memory snapshot, counting Metal unified memory only once.
`begin_microbatch` revalidates topology and current headroom before waiting
through both bounded admission gates. Receipt v4 records only the exact session
and plan digests, aggregate batch position/counts, and queue-path booleans; slot
digests and raw memory snapshots remain in caller-owned plan data and are never
copied into a receipt automatically.

Prefetch and current-layer staging share one background-load admission path.
`max_prefetch_workers` bounds tasks by count, while
`max_background_inflight_bytes` independently bounds the sum of canonical
tensor bytes owned by active workers; the effective byte window cannot exceed
the existing total-batch `max_prefetch_bytes`. A staged batch exposes
`next_ready_group().await`, so a model can issue exact work as soon as an atomic
group arrives without a polling loop. Out-of-order compute still writes into
canonical slots, and final completion restores canonical group order. Reports
and opt-in aggregate telemetry expose only peak in-flight counts/bytes and
event/final wait time, never layer, tensor, route, expert, or value identities.

Canonical SafeTensors remains the mandatory primary and recovery source. A
model-owned artifact tool may provide an optional
`LosslessRansNibble256V1` replica with an explicit complete-collection SHA-256.
Power verifies that pin before parsing SafeTensors metadata, requires the
format stamp and one shard-local static table, then decodes and byte-compares
every covered tensor with the primary before the replica becomes routable.
The pure-Rust codec uses 256 round-robin nibble rANS streams and exact framing,
padding, amplification, stream-state, and full-consumption checks. Encoded plus
decoded scratch is bounded before allocation and heap buffers zeroize on drop.
Only genuinely smaller records are admitted, with either complete coverage or
a proper non-empty tensor subset. Demand, prefetch, staging, residency,
fallback, telemetry, and storage evidence continue through the existing
`WeightStore`; Power does not quantize, reinterpret, auto-pack, or persist model
weights.

For routed models, aligned exact route batches can explicitly teach a bounded
cross-layer coupling table. Power sums learned co-occurrence counts over each
position's current routed set, returns deterministic target-expert hints and a
deduplicated batch union, and measures recall against the later actual router
output. These are prefetch hints only: Power never substitutes an expert,
changes a gate, or maps an expert ID to model-owned tensor names. Expert-level
history requires detailed telemetry, is bound to the admitted weight digest
and layer geometry, and is never logged or persisted automatically. When
authorized, a model owner may serialize it as opaque bytes through the shared
`SealedStateEnvelope` path.

Hardware-specific tuning is also evidence-only. A model crate generates one
teacher-forced calibration workload, applies its own lossless candidate knobs,
and submits aggregate `TuningProfileEvidence`. Every run carries exact weight,
graph/source, workload, runtime, device, environment, configuration, and output
digests. Power requires at least two complete baseline→candidate and
candidate→baseline rounds, then checks both order-specific median throughput
gains, per-run cache-hit parity, and per-run p99 bounds. It retains the baseline
when no candidate passes or the best conservative gain is tied. The resulting
`TuningProfileDecision` contains only digests, policy thresholds, and aggregate
measurements. Power neither applies a candidate nor persists a profile; a model
crate may map the selected digest to reviewed settings and persist opaque
profile bytes through the same authorized `SealedStateEnvelope` path.

Path-free storage evidence records the physical root count for each logical
source, so single-volume and capacity-aggregated measurements cannot be merged
silently. Actual roots remain absent from reports, telemetry, receipts, and
attestation claims.

`HardwareEvidenceBundle` turns those isolated measurements into one canonical,
reviewable envelope without adding another telemetry or receipt system. A
`HardwareEvidenceBinding` derives exact runtime, typed-device, and named-system
digests from a Power version/commit and path-free `StorageBenchmarkSystem`.
Bundle construction embeds and canonically orders 2–128 source reports, re-runs
the existing storage comparison and tuning evaluator, requires at least two
storage groups with byte-exact output parity, and binds 1–256 model-owned parity
artifact pins for the configuration the tuning decision actually selected.
Storage, tuning, and every parity artifact must share the same weights,
graph/source, runtime, device, and environment binding. Negative evidence is
valid: a bundle remains reviewable when policy retains the baseline.

`verify()` replays every derivation and checks the bounded canonical envelope;
`verify_pinned()` additionally checks a caller-owned expected digest. The
bundle SHA-256 detects mutation but is not a signature, so publication must pin
it in an attestation, signed release, or another trust root. Named hardware,
aggregate timing, workload/output digests, and artifact hashes are explicit
review data and may be correlatable. Power never logs, persists, uploads, or
serves a bundle automatically, and TEE deployments retain authority over
whether it may leave the attested boundary.

Continuous and ragged execution adapts Colibri's multi-slot decode lifecycle
without moving its model scheduler into Power. An `ExecutionBatchLifecycle`
binds exact weight, model-owned state-layout, and scheduler digests. Every member
holds a distinct permit from the existing runtime admission controller and a
non-aliasing model-state identity. At each step boundary, every member already
present appears exactly once in canonical admission order, while positions and
row shapes may differ; members admitted during arithmetic join only the next
step. Aggregate input bytes/elements, row rank, context, generation, and state
bytes remain under existing runtime limits. Step commits are atomic, cancelled
rows are discarded without rolling back other rows, and an invalid commit
leaves lifecycle positions and state unchanged for an exact retry. Public
evidence contains only aggregate counters and canonical transcript digests.
The model crate still owns token selection, KV or recurrent topology, tensors,
kernels, and uninterrupted-output parity.

Accelerator fusion extends the same residency and receipt path rather than
creating a backend-specific registry. A model crate names canonical groups in
the active residency plan and supplies digests for its fused Candle kernel and
exact fallback. Power binds the declaration to the weight digest, active plan,
typed CUDA/Metal device, runtime limits, fallback target, and security mode.
Resolution returns only already-pinned device weights; it never promotes a
miss and calls it accelerated. Residency loss or a typed kernel-unavailable
result produces a separate fallback token when policy permits it. Execution
evidence records both the declared device and the actual fallback device. In
confidential-GPU mode, execution requires an existing hardware-verified v2
attestation report whose model and `runtime.execution.gpu_sha256` claims match
the declaration.

The same declaration can bind a canonical heterogeneous device mesh. Power
caps it at 16 unique resolved devices, requires every node to be reachable from
and able to return to the primary device, and checks every directed copy
against per-edge count/byte limits plus one aggregate byte budget. These reuse
the existing graph, input, and state limits. A model-owned kernel retains graph
partitioning and arithmetic, while `AcceleratorMeshExecution` performs and
accounts explicit Candle device copies. Backend copy failure is a typed
unavailability result; malformed topology, over-budget traffic, and
cancellation fail closed. Power does not add a per-device cache or residency
registry: weights still come only from the active plan and existing device
cache.

Confidential meshes bind each CUDA node to an explicit NVIDIA claim-array
index and bind the exact GPU/NVSwitch claim sets. CUDA ordinals are never
treated as claim indices or UEIDs, and NVSwitch presence is not presented as
proof that any particular edge works. Runtime copies remain the availability
test. Embedded receipt v2 covers single-device fusion; receipt v3 adds only the
mesh digest, canonical actual-device identities, and a peer-transfer trace
digest—not node names, edges, tensor names, group IDs, values, or attestation
evidence bytes. As Colibri's multi-GPU measurements show, peer traffic may
erase a pipeline gain, so no mesh policy is enabled by default without
model-owned end-to-end evidence on named hardware.

Warm-state persistence adapts Colibri's commit-last recovery principle without
adopting its plaintext `.coli_kv` format. A model crate serializes its own KV or
recurrent topology into opaque bytes. Power bounds those bytes, seals them with
AES-256-GCM, and authenticates a fixed header containing the exact weight,
state-layout, and hashed session identities, generation, lengths, and export
scope. `SealedStateStore` publishes through a synchronized same-directory
pending file while retaining the prior committed generation as a recovery
candidate. Load authenticates primary and backup, ignores pending data, and
selects the highest valid generation permitted by a caller-pinned rollback
floor. Power does not claim a hardware monotonic counter; deployments requiring
rollback resistance must retain that floor in their own trusted source.

TEE-local envelopes have no public raw-byte export API. State leaving the
attested boundary requires a digest-only `TeeStateExportAuthorization` derived
from an already hardware-verified SEV-SNP/TDX report, the exact attested model,
and an explicit export-policy digest. This reuses the canonical v2 attestation
claim matcher used by confidential acceleration. Keys and opened plaintext are
owned by zeroizing types, paths and state identifiers stay out of debug output,
and persistence remains caller-invoked blocking filesystem work rather than a
background service or listener.

The same `WeightStore` now offers four explicit materialization strategies.
`Mmap` remains the default. `PositionalBuffered` avoids mapping the complete
collection and reads each indexed tensor through bounded, cancellable
positional reads. `PositionalCacheBypass` applies macOS `F_NOCACHE` to both
integrity hashing and tensor-range handles without claiming direct I/O or a
verified cold cache. `PositionalDirect` uses aligned `O_DIRECT` on Linux and
`FILE_FLAG_NO_BUFFERING` on Windows. Unsupported combinations fail explicitly
instead of silently falling back, and mmap remains the default until a
documented end-to-end workload proves another strategy on named hardware.

For confidential sparse-weight execution, `encrypt_seekable_weight_collection`
converts a validated SafeTensors collection into
`a3s.power.seekable-encrypted-weights.v1`. Each source file uses fixed-size,
independently authenticated AES-256-GCM chunks; its header and chunk index are
authenticated additional data. `WeightStore::open_seekable_encrypted` requires
the caller-pinned manifest SHA-256, authenticates every ciphertext chunk, and
reconstructs the original logical collection digest before it can serve a
tensor. Subsequent positional reads decrypt only covering chunks into
zeroizing buffers and flow through the existing demand, prefetch, staging,
cache, cancellation, and telemetry paths. No plaintext weight file or
collection-wide plaintext allocation is created. Key acquisition and binding
the manifest digest into TEE attestation remain deployment policy.

Model-owning crates provide graph identities and plans, network control flow,
tokenizers, preprocessing, postprocessing, and revision policy. Consequently,
Power does not embed PP-OCR, Unlimited-OCR, or any other product model.

```bash
cargo build --no-default-features --features embedded-inference
```

Static-graph timing is opt-in diagnostic output. Set
`A3S_POWER_TRACE_GRAPH_TIMINGS=1` for host-side submission intervals or
`A3S_POWER_TRACE_GRAPH_TOTALS=1` for host totals only. An `embedded-cuda`
build additionally accepts `A3S_POWER_TRACE_CUDA_GRAPH_TIMINGS=1`; it records
timing-enabled CUDA events on each model session's owning stream and reports
only input geometry, operator geometry, execution/node counts, host submission
time, attributed stream intervals, and unattributed stream intervals. It fails
closed on a non-CUDA device. CUDA event mode synchronizes each completed graph
and adds boundary events, so it is for bottleneck attribution only and must be
disabled for throughput comparisons. None of these values selects graph,
model, request, or parsing behavior.

That feature does not enable Power's `server` feature. Its normal dependency
closure excludes Axum, Hyper, Tower, HTTP clients, ONNX Runtime, and browser
automation. Tokio networking, process, and signal features also remain behind
`server`. Verify the boundary with:

```bash
just check-embedded
```

See [Embedded Inference Architecture](docs/embedded-inference-architecture.md)
for the ownership boundary, Colibri-inspired weight hierarchy, exact routing,
event-driven current-layer staging, verified lossless weight representations,
digest-bound lossless tuning, cross-layer hint learning, stable live hot-store
adaptation, prefetch semantics, TEE invariants, and model parity gates. See
[Storage Benchmark Protocol](docs/storage-benchmark.md) for the
standalone mmap/positional comparison tool, cache-state proof rules, platform
limitations, and measured PP-OCRv6 results.

## Server operations

- **Content-Addressed Storage**: Model blobs stored by SHA-256 hash with automatic deduplication
- **Automatic Model Lifecycle**: LRU eviction, configurable keep-alive, background reaper for idle models
- **Rate Limiting & Admission Control**: Per-second token-bucket on `/v1/*` returns `429` with an OpenAI-style error; concurrency (`max_concurrent_requests`) uses vLLM-style backpressure — excess requests **queue** for an admission permit (held across the streamed body) rather than being rejected, with a `power_requests_waiting` gauge
- **Prometheus Metrics**: 16 metric groups — HTTP, inference, TTFT, GPU, TEE attestations, model decryptions, log redactions
- **Audit Logging**: JSONL / Encrypted / Async / Noop; flushed on graceful shutdown
- **Vsock Transport**: AF_VSOCK for a3s-box MicroVM guest-host communication (Linux only)
- **A3S ACL Configuration**: Closed, typed configuration parsed and generated by `a3s-acl`
- **A3S Runtime Service Profile**: Compiles Power into a digest-pinned, Box-hosted Runtime Service with secret-backed ACL, health checks, resource limits, and restart policy

## Architecture

A3S Power is organized into 6 layers. Each layer has a clear responsibility and communicates only with adjacent layers through trait-based interfaces.

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              a3s-power                                      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  API Layer                                                            │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐  │  │
│  │  │ /v1/chat/    │ │ /v1/models   │ │ /v1/embed  │ │ /v1/attest   │  │  │
│  │  │ completions  │ │ /v1/models/  │ │ dings      │ │ ation        │  │  │
│  │  │              │ │ pull         │ │            │ │              │  │  │
│  │  │ /v1/         │ │ /v1/models/  │ │            │ │ /health      │  │  │
│  │  │ completions  │ │ :name        │ │            │ │ /metrics     │  │  │
│  │  └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └──────┬───────┘  │  │
│  │         │                │               │               │          │  │
│  │  ┌──────┴────────────────┴───────────────┴───────────────┘          │  │
│  │  │  autoload: LRU eviction → decrypt → integrity check → load      │  │
│  │  └──────┬──────────────────────────────────────────────────         │  │
│  └─────────┼─────────────────────────────────────────────────────────┘  │
│            │                                                             │
│  ┌─────────┼─────────────────────────────────────────────────────────┐  │
│  │  Server │Layer                                                     │  │
│  │  ┌──────┴───────────────────────────────────────────────────────┐  │  │
│  │  │  Middleware Stack (outermost → innermost)                     │  │  │
│  │  │  RateLimiter → RequestID → Metrics → Tracing → CORS → Auth  │  │  │
│  │  └──────────────────────────┬───────────────────────────────────┘  │  │
│  │                             │                                      │  │
│  │  ┌──────────┐ ┌─────────┐ ┌┴─────────┐ ┌──────────┐ ┌─────────┐  │  │
│  │  │ AppState │ │  Auth   │ │  Audit   │ │ Metrics  │ │Transport│  │  │
│  │  │ (model   │ │ (Bearer │ │ (JSONL/  │ │(Promethe │ │TCP/TLS/ │  │  │
│  │  │lifecycle,│ │  SHA256 │ │ encrypt/ │ │ us, 16   │ │ Vsock)  │  │  │
│  │  │ LRU,     │ │  const- │ │ async/   │ │ metric   │ │         │  │  │
│  │  │ privacy) │ │  time)  │ │ noop)    │ │ groups)  │ │         │  │  │
│  │  └──────┬───┘ └─────────┘ └──────────┘ └──────────┘ └─────────┘  │  │
│  └─────────┼─────────────────────────────────────────────────────────┘  │
│            │                                                             │
│  ┌─────────┼─────────────────────────────────────────────────────────┐  │
│  │  Backend│Layer                                                     │  │
│  │  ┌──────┴───────────────────────────────────────────────────────┐  │  │
│  │  │  BackendRegistry (priority-based, TEE-aware routing)         │  │  │
│  │  │  ┌─────────────────────┬─────────────────┬────────────────┐  │  │  │
│  │  │  │ MistralRsBackend ★  │ LlamaCppBackend │ PicolmBackend  │  │  │  │
│  │  │  │ pure Rust (candle)  │ C++ bindings    │ pure Rust      │  │  │  │
│  │  │  │ GGUF/SafeTensors/   │ GGUF only       │ layer-stream   │  │  │  │
│  │  │  │ HuggingFace/Vision  │ KV cache, LoRA  │ O(layer_size)  │  │  │  │
│  │  │  │ ISQ quantization    │ grammar, vision │ TEE-optimized  │  │  │  │
│  │  │  └─────────────────────┴─────────────────┴────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │  Shared: chat_template · gpu · json_schema · tool_parser    │  │  │
│  │  │          think_parser · gguf_stream                         │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Model Layer                                                      │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌─────────────┐  │  │
│  │  │ ModelRegistry│ │ BlobStorage  │ │ GgufMeta │ │ HfPull      │  │  │
│  │  │ (RwLock<Map>)│ │ (SHA-256     │ │ (parser, │ │ (Range      │  │  │
│  │  │ manifest     │ │  content-    │ │  memory  │ │  resume,    │  │  │
│  │  │ persistence) │ │  addressed)  │ │  estim.) │ │  SSE prog.) │  │  │
│  │  └──────────────┘ └──────────────┘ └──────────┘ └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  TEE Layer (cross-cutting security)                               │  │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌─────────────────┐  │  │
│  │  │Attestation │ │ Encrypted  │ │ Privacy  │ │  Model Seal     │  │  │
│  │  │(TeeProvider│ │ Model      │ │(Provider │ │  (SHA-256 +     │  │  │
│  │  │ SEV-SNP,   │ │ AES-256-   │ │ redact,  │ │   Ed25519 sig)  │  │  │
│  │  │ TDX, ioctl)│ │ GCM, 3     │ │ zeroize, │ │                 │  │  │
│  │  │            │ │ modes)     │ │ suppress)│ │                 │  │  │
│  │  └────────────┘ └────────────┘ └──────────┘ └─────────────────┘  │  │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌─────────────────┐  │  │
│  │  │KeyProvider │ │ TeePolicy  │ │   EPC    │ │  RA-TLS Cert    │  │  │
│  │  │(Static,    │ │(allowlist, │ │(memory   │ │  (X.509 +       │  │  │
│  │  │ Rotating,  │ │ measure-   │ │ detect,  │ │   attestation   │  │  │
│  │  │ HSM ext.)  │ │ ment pin)  │ │ routing) │ │   extension)    │  │  │
│  │  └────────────┘ └────────────┘ └──────────┘ └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Verify Layer (client-side SDK)                                   │  │
│  │  ┌──────────────────────────────┐ ┌─────────────────────────────┐ │  │
│  │  │ verify_report()              │ │ HardwareVerifier trait       │ │  │
│  │  │ · nonce binding (const-time) │ │ · SevSnpVerifier (AMD KDS)  │ │  │
│  │  │ · model hash binding         │ │ · TdxVerifier (Intel PCS)   │ │  │
│  │  │ · measurement check          │ │ · ECDSA P-384 / P-256       │ │  │
│  │  └──────────────────────────────┘ └─────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Infrastructure: config.rs (A3S ACL) · dirs.rs · error.rs (14 var.)  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core vs Extension

Power follows the Minimal Core + External Extensions pattern. Core components are stable and non-replaceable; extensions are trait-based and swappable.

```
Core (7)                              Extensions (8 trait-based)
─────────────────────────             ──────────────────────────────────────
AppState (model lifecycle)            Backend: MistralRs / LlamaCpp / Picolm
BackendRegistry + Backend trait       TeeProvider: SEV-SNP / TDX / Simulated
ModelRegistry + ModelManifest         PrivacyProvider: redaction policy
PowerConfig (A3S ACL)                     TeePolicy: allowlist + measurement pin
PowerError (14 variants → HTTP)       KeyProvider: Static / Rotating / KMS
Router + middleware stack             AuthProvider: API key (SHA-256)
RequestContext (per-request)          AuditLogger: JSONL / Encrypted / Async / Noop
                                      HardwareVerifier: AMD KDS / Intel PCS
```

### Request Flow: Chat Completion

```
Client
  │
  │  POST /v1/chat/completions
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Middleware Stack                                                 │
│ RateLimiter ─► RequestID ─► Metrics ─► Tracing ─► CORS ─► Auth │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ chat::handler()                                                  │
│                                                                  │
│  1. Build RequestContext (request_id, auth_id)                   │
│  2. Privacy: sanitize_log() if redaction enabled                 │
│  3. ModelRegistry.get(model) → ModelManifest                     │
│  4. BackendRegistry.find_for_manifest(manifest) → Backend        │
│                                                                  │
│  5. autoload::ensure_loaded()                                    │
│     ├─ LRU eviction if at max_loaded_models                     │
│     ├─ If .enc: KeyProvider.get_key() → AES-256-GCM decrypt     │
│     │   ├─ MemoryDecryptedModel (mlock RAM, zeroize on drop)    │
│     │   ├─ DecryptedModel (temp file, secure wipe on drop)      │
│     │   └─ LayerStreamingDecryptedModel (chunk-by-chunk)        │
│     ├─ model_seal: verify SHA-256 integrity                     │
│     ├─ model_seal: verify Ed25519 signature (if configured)     │
│     └─ Backend.load(manifest)                                   │
│                                                                  │
│  6. Backend.chat(model, request) → Stream<ChatResponseChunk>     │
│  7. Streaming SSE: role → content chunks (TTFT) → usage → DONE  │
│  8. Privacy: zeroize buffers, round token counts                 │
│  9. Timing padding (±20% jitter) if configured                  │
│ 10. Audit: log event, Metrics: record duration/tokens            │
│ 11. If keep_alive=0: Backend.unload() → RAII secure wipe        │
└─────────────────────────────────────────────────────────────────┘
```

### TEE Security Integration

The TEE layer is cross-cutting — it integrates at every layer of the stack:

```
Layer           TEE Integration
──────────────  ──────────────────────────────────────────────────────
API             Log redaction, buffer zeroization, token rounding,
                timing padding, attestation endpoint (nonce + model bind)

Server          Encrypted audit logs (AES-256-GCM), constant-time auth,
                RAII decrypted model storage, RA-TLS cert with attestation
                X.509 extension, TEE-specific Prometheus counters

Backend         EPC-aware routing (auto picolm when model > 75% EPC),
                KV cache isolation per request, mlock weight pinning

Model           Content-addressed SHA-256 storage, GGUF memory estimation
                for EPC budget planning

TEE             Attestation (SEV-SNP/TDX ioctl), AES-256-GCM encryption
                (file-backed loading plus RAM/streaming primitives), Ed25519 model signatures,
                key rotation, policy enforcement, log redaction (10 keys),
                SensitiveString (auto-zeroize), EPC memory detection

Verify          Client-side: nonce binding, model hash binding,
                measurement check (all constant-time), hardware signature
                verification via AMD KDS / Intel PCS certificate chains
```

### Encrypted Model Decryption Modes

```
                    ┌─────────────────────────────────────────┐
                    │         KeyProvider.get_key()            │
                    │    Static ─── Rotating ─── (HSM ext.)   │
                    └──────────────────┬──────────────────────┘
                                       │ AES-256-GCM key
                    ┌──────────────────┼──────────────────────┐
                    │                  │                       │
              ┌─────┴──────┐   ┌──────┴───────┐   ┌──────────┴──────────┐
              │ DecryptedMo│   │ MemoryDecrypt│   │ LayerStreamingDecry │
              │ del (file) │   │ edModel (RAM)│   │ ptedModel (chunks)  │
              │            │   │              │   │                     │
              │ Temp .dec  │   │ mlock-pinned │   │ Chunked plaintext   │
              │ file on    │   │ RAM buffer,  │   │ access primitive    │
              │ disk, zero │   │ zeroize on   │   │ with Zeroizing      │
              │ overwrite  │   │ drop         │   │ chunk buffers       │
              │ + delete   │   │              │   │                     │
              │ on drop    │   │              │   │                     │
              └────────────┘   └──────────────┘   └─────────────────────┘
              End-to-end       End-to-end with        End-to-end with
              backend path      picolm GGUF           picolm GGUF;
              today             memory loading        unsupported
                                                        backends fail closed
```

Encrypted-model autoload supports `in_memory_decrypt = true` only when the
selected backend explicitly accepts locked plaintext buffers. Today that means
`picolm` for GGUF models; other backends fail closed before load.
`streaming_decrypt = true` similarly requires a backend that explicitly accepts
`LayerStreamingDecryptedModel`; today that path is `picolm` for GGUF models.

### Backend Trait

Three backends are available, each feature-gated:

- **`mistralrs`** (default): Pure Rust inference via candle. GGUF, SafeTensors, HuggingFace, Vision formats. ISQ on-load quantization. No C++ inference toolchain. Ideal for TEE supply-chain auditing.
- **`llamacpp`** (optional): C++ llama.cpp via `llama-cpp-2` bindings. GGUF only. Session KV cache with prefix matching, LoRA adapters, MTMD multimodal, grammar constraints, mirostat sampling, and native MTP for compatible models. Enable `llamacpp-cuda` instead when NVIDIA CUDA execution is required.
- **`picolm`** (optional): Pure Rust layer-streaming. GGUF only. Real transformer inference (multi-head/GQA attention, SwiGLU/GeGLU FFN, RoPE, RMSNorm). Peak RAM = O(layer_size) not O(model_size) via `madvise(DONTNEED)` page release. FP16 KV cache with fused f16 dot/accumulate. Fused dequant+dot kernels (Q4_K, Q6_K, Q8_0). NEON SIMD (aarch64) + AVX2 (x86_64). Rayon parallel matmul. Batch prefill, speculative decoding, tool calling, grammar-constrained output. 14+ tok/s decode on Apple Silicon. Enables 7B+ models in 512MB TEE EPC. No C/C++ inference backend — ~4,500 lines of fully auditable Rust.

The `BackendRegistry` selects backends by priority and exact model manifest.
Format-only backends inherit the existing behavior, while architecture-aware
backends override `supports_manifest()` so multiple implementations of the same
container format can coexist. In TEE environments,
`find_for_tee_manifest()` retains exact manifest matching while preferring
picolm when a compatible model exceeds 75% of available EPC memory.

Without any backend feature enabled, Power can manage models but returns "backend not available" for inference.

```rust
#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn supports(&self, format: &ModelFormat) -> bool;
    fn supports_manifest(&self, manifest: &ModelManifest) -> bool {
        self.supports(&manifest.format)
    }
    async fn load(&self, manifest: &ModelManifest) -> Result<()>;
    async fn unload(&self, model_name: &str) -> Result<()>;
    async fn chat(&self, model_name: &str, request: ChatRequest)
        -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>>;
    async fn complete(&self, model_name: &str, request: CompletionRequest)
        -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>>;
    async fn embed(&self, model_name: &str, request: EmbeddingRequest)
        -> Result<EmbeddingResponse>;
}
```

Downstream model crates can compose the HTTP service without making Power
depend on the model implementation:

```rust,no_run
use std::sync::Arc;
use a3s_power::server::PowerServerBuilder;

# async fn run(
#     config: a3s_power::config::PowerConfig,
#     backend: Arc<dyn a3s_power::backend::Backend>,
# ) -> a3s_power::error::Result<()> {
PowerServerBuilder::new(config)
    .with_backend(backend)
    .start()
    .await
# }
```

Injected backends have priority over built-ins. Calling
`without_default_backends()` makes the injected registry complete and avoids
starting any built-in inference backend.

### Extension Points

All extension points are trait-based with working default implementations — the system works out of the box:

```rust
/// Remote attestation provider (TEE hardware abstraction).
pub trait TeeProvider: Send + Sync {
    async fn attestation_report(&self, nonce: Option<&[u8]>) -> Result<AttestationReport>;
    async fn attestation_report_with_model(
        &self, nonce: Option<&[u8]>, model_hash: Option<&[u8]>
    ) -> Result<AttestationReport>;
    fn is_tee_environment(&self) -> bool;
    fn tee_type(&self) -> TeeType;  // SevSnp | Tdx | Simulated | None
}

/// Privacy protection for inference logs.
pub trait PrivacyProvider: Send + Sync {
    fn should_redact(&self) -> bool;
    fn sanitize_log(&self, msg: &str) -> String;
    fn sanitize_error(&self, err: &str) -> String;
    fn should_suppress_token_metrics(&self) -> bool;
}

/// Model decryption key management (extensible to HSM/KMS).
pub trait KeyProvider: Send + Sync {
    async fn get_key(&self) -> Result<[u8; 32]>;
    async fn rotate_key(&self) -> Result<[u8; 32]>;
    fn provider_name(&self) -> &str;
}

/// Authentication mechanism.
pub trait AuthProvider: Send + Sync {
    fn authenticate(&self, token: &str) -> Result<AuthId>;
}

/// Audit trail persistence.
pub trait AuditLogger: Send + Sync {
    fn log(&self, event: AuditEvent);
    async fn flush(&self);
}

/// TEE policy enforcement.
pub trait TeePolicy: Send + Sync {
    fn is_allowed(&self, tee_type: TeeType) -> bool;
    fn validate_measurement(&self, measurement: &[u8]) -> bool;
}

/// Client-side hardware attestation signature verification.
pub trait HardwareVerifier: Send + Sync {
    async fn verify(&self, report: &AttestationReport) -> Result<()>;
}
```

## Installation

### Cargo (cross-platform)

```bash
# Default: pure Rust inference via mistral.rs (no C++ toolchain needed)
cargo install a3s-power

# With llama.cpp inference backend (requires C++ compiler + CMake)
cargo install a3s-power --no-default-features --features llamacpp

# With llama.cpp and NVIDIA CUDA
cargo install a3s-power --no-default-features --features llamacpp-cuda

# Model management only (no inference)
cargo install a3s-power --no-default-features
```

### Build from Source

```bash
git clone https://github.com/A3S-Lab/Power.git
cd Power

# Default: pure Rust inference via mistral.rs
cargo build --release

# With llama.cpp inference instead
cargo build --release --no-default-features --features llamacpp

# With llama.cpp and NVIDIA CUDA
cargo build --release --no-default-features --features llamacpp-cuda

# Binary at target/release/a3s-power
```

### Homebrew (macOS)

```bash
brew tap a3s-lab/tap https://github.com/A3S-Lab/homebrew-tap
brew install a3s-power
```

## Configuration

Configuration is read from `~/.a3s/power/config.acl` (A3S ACL format):

```acl
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
keep_alive = "5m"

# TEE privacy protection
tee_mode = true
tee_policy_mode = "strict"
redact_logs = true

# Production TEE launch measurement pinning. Strict/gpu-confidential policy
# requires the detected hardware TEE type to have a 48-byte measurement pin.
expected_measurement "sev-snp" {
  digest = "<96-char-measurement-hex>"
}
# Repeat with label "tdx" and its MRTD digest when TDX is allowed.

# Model integrity verification (required by strict policy when tee_mode = true)
model_hash "llama3.2:3b" {
  digest = "sha256:abc123..."
}
model_hash "qwen2.5:7b" {
  digest = "sha256:def456..."
}

# GPU acceleration
gpu {
  gpu_layers = -1    # -1 = offload all layers, 0 = CPU only
  main_gpu   = 0
}

# NVIDIA GPU confidential-computing evidence binding
# Required only when tee_policy_mode = "gpu-confidential".
gpu_attestation {
  # Preferred live path: invoke NVIDIA nvattest for each attestation request.
  source = "nvattest-cli"
  provider = "nvidia-nras"
  nvattest_path = "/usr/local/bin/nvattest"
  nvattest_verifier = "remote"
  nvattest_gpu_evidence_source = "nvml"
  # nras_url = "https://<your-nras-endpoint>"

  # Alternative compatibility path for external evidence pipelines:
  # The configured verdict must be NVIDIA nvattest/NRAS JSON whose eat_nonce
  # matches each /v1/attestation nonce; stale verdicts fail closed.
  # source = "configured"
  # evidence_path = "/run/a3s/nvidia-gpu-evidence.json"
  # verdict_path  = "/run/a3s/nvidia-nras-verdict.json"

  # Direct NRAS REST path for deployments that collect DeviceEvidenceV2 JSON
  # through their own NVIDIA evidence collector:
  # source = "nras-rest"
  # evidence_path = "/run/a3s/nvidia-gpu-evidence-list.json"
  # nras_url = "https://nras.attestation.nvidia.com" # no embedded credentials
  # nras_gpu_architecture = "HOPPER" # or "BLACKWELL"
  # nras_claims_version = "3.0"
  # nras_bearer_token_env = "NRAS_BEARER_TOKEN"
}

# Optional proxy backend integration.
# When enabled, Power asks the upstream for the exact rendered prompt digest
# before proxied chat inference. If required = true, missing support fails closed.
# proxy_upstream "llama-70b" {
#   url = "http://power-member:8000"
# }
# proxy_effective_prompt_digest = true
# proxy_effective_prompt_digest_required = false
# proxy_effective_prompt_digest_path = "/v1/chat/effective-prompt-digest"
```

### A3S Box Runtime Service

`PowerRuntimeServiceProfile` compiles a Power deployment into the shared
`a3s-runtime` `RuntimeUnitSpec` Service contract. The resulting generation uses
a digest-pinned OCI artifact, a provider-resolved `config.acl` secret, declared
resources, a named HTTP endpoint, `/health`, graceful process shutdown, and the
Runtime restart policy. A3S Box is the execution provider.

Power does not implement placement, accelerator leasing, rollout, queues,
authorization, or Gateway routing. A3S Cloud owns those concerns and supplies
the immutable generation, resource lease, artifact, secret references, and
semantics digest before handing the standard Service to Runtime and Box.

### Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `127.0.0.1` | HTTP server bind address |
| `port` | `11434` | HTTP server port |
| `data_dir` | `~/.a3s/power` | Base directory for model storage |
| `max_loaded_models` | `1` | Maximum models loaded concurrently |
| `keep_alive` | `"5m"` | Auto-unload idle models (`"0"` = immediate, `"-1"` = never); invalid config or request values fail closed |
| `spec_mode` | `"auto"` | Model-neutral strategy: `"off"`, `"prompt-lookup"`, `"ngram-context"`, `"draft-model"`, `"mtp"`, `"dflash"`, or `"dspark"`; backends reject unsupported explicit choices |
| `spec_draft_max` | adapter default | Maximum draft tokens per target verification pass (`1..64`) |
| `spec_draft_min` | `0` | Minimum draft tokens required from a model-backed adapter |
| `spec_draft_p_min` | `0.0` | Minimum model-backed draft confidence (`0.0..1.0`) |
| `use_mlock` | `false` | Lock model weights in memory (prevent swapping) |
| `num_thread` | auto | Thread count for inference |
| `flash_attention` | `false` | Enable flash attention |
| `num_parallel` | `1` | Concurrent inference slots |
| `tee_mode` | `false` | Enable TEE: attestation, integrity checks, log redaction |
| `tee_policy_mode` | `"strict"` | TEE attestation policy: `"strict"` for production, `"development"` for simulated/local tests, `"gpu-confidential"` for NVIDIA GPU confidential-computing deployments with bound GPU evidence |
| `expected_measurement "TEE"` | none | Repeatable launch-measurement block with a required `digest`; strict and GPU-confidential policy require the detected hardware TEE (`"sev-snp"` measurement or `"tdx"` MRTD) |
| `redact_logs` | `false` | Redact inference content from logs |
| `audit_log` | `false` | Enable structured audit logging |
| `audit_log_path` | `null` | Audit log path; defaults to `$A3S_POWER_HOME/audit.jsonl`; startup fails closed if the path cannot be opened while `audit_log = true` |
| `audit_log_encrypt` | `false` | Encrypt audit log entries at rest; requires `audit_key_source` and fails configuration validation when missing |
| `audit_key_source` | none | Singleton block with either `type = "file"` or `type = "env"` plus `value`; supplies the AES-256-GCM key source for encrypted audit logs |
| `model_hash "MODEL"` | none | Repeatable model-integrity block with a required SHA-256 `digest` |
| `model_signing_key` | `null` | Valid 32-byte Ed25519 public key (hex) for verifying model `.sig` signatures; invalid values fail configuration validation; `/v1/attestation?model=...` re-verifies the current runtime digest signature when no explicit `model_hashes` pin is configured |
| `gpu.gpu_layers` | `0` | GPU layer offloading (`-1` = all) |
| `gpu.main_gpu` | `0` | Primary GPU index |
| `gpu_attestation.source` | `"configured"` | GPU CC evidence source: `"configured"` for file/hex bytes, `"nvattest-cli"` for live NVIDIA `nvattest`, or `"nras-rest"` for direct NVIDIA NRAS REST attestation |
| `gpu_attestation.provider` | `"nvidia-nras"` | Provider label for NVIDIA GPU confidential-computing evidence claims; `gpu-confidential` production policy requires `"nvidia-nras"` |
| `gpu_attestation.evidence_path` | `null` | Path to raw NVIDIA GPU CC evidence bytes; mutually exclusive with `evidence_hex` and fails configuration validation if both are set; required when `source = "nras-rest"`; `gpu-confidential` production policy requires an absolute path to an existing non-empty regular file when file-backed evidence is configured; configured evidence sources are capped at 64 MiB |
| `gpu_attestation.evidence_hex` | `null` | Hex-encoded raw NVIDIA GPU CC evidence bytes; mutually exclusive with `evidence_path` and fails configuration validation if both are set; required when `source = "nras-rest"` unless `evidence_path` is set; configured evidence sources are capped at 64 MiB |
| `gpu_attestation.verdict_path` | `null` | Path to raw NVIDIA NRAS verdict bytes; mutually exclusive with `verdict_hex` and fails configuration validation if both are set; must be unset when `source = "nras-rest"` because NRAS REST obtains the verdict directly; `gpu-confidential` production policy requires an absolute path to an existing non-empty regular file when configured evidence uses a file-backed verdict; configured verdict sources are capped at 64 MiB |
| `gpu_attestation.verdict_hex` | `null` | Hex-encoded raw NVIDIA NRAS verdict bytes; mutually exclusive with `verdict_path` and fails configuration validation if both are set; must be unset when `source = "nras-rest"` because NRAS REST obtains the verdict directly; configured verdict sources are capped at 64 MiB |
| `gpu_attestation.nvattest_path` | `"nvattest"` | Path to NVIDIA's `nvattest` CLI when `source = "nvattest-cli"`; `gpu-confidential` production policy requires an absolute path to an existing executable file |
| `gpu_attestation.nvattest_verifier` | `"remote"` | `nvattest attest --verifier` value; must be `"remote"` or `"local"` when `source = "nvattest-cli"`; `gpu-confidential` mode requires `"remote"` for NRAS |
| `gpu_attestation.nvattest_gpu_evidence_source` | `"nvml"` | `nvattest collect-evidence --gpu-evidence-source`; must be `"nvml"` or `"corelib"` when `source = "nvattest-cli"`; use `"nvml"` for H100 confidential-computing deployments |
| `gpu_attestation.nvattest_gpu_architecture` | `null` | GPU architecture value required when `source = "nvattest-cli"` and `nvattest_gpu_evidence_source = "corelib"` |
| `gpu_attestation.nras_url` | `null` | Optional NRAS URL. For `nvattest-cli`, passed to `nvattest attest --nras-url`; for `nras-rest`, may be a service root/base path or full `/v4/attest/gpu` endpoint. In `gpu-confidential` production policy, custom NRAS URLs must use HTTPS and must not include embedded credentials |
| `gpu_attestation.nras_gpu_architecture` | `null` | GPU architecture required when `source = "nras-rest"`: `"HOPPER"` or `"BLACKWELL"` |
| `gpu_attestation.nras_claims_version` | `"3.0"` | NVIDIA NRAS REST claims version (`"2.0"` or `"3.0"`); invalid values fail configuration validation when `source = "nras-rest"` |
| `gpu_attestation.nras_bearer_token_env` | `null` | Optional environment variable name containing a bearer token for direct NRAS REST calls; the name is trimmed and must be a portable ASCII identifier (`[A-Za-z_][A-Za-z0-9_]*`); use this instead of embedding credentials in `nras_url` |
| `gpu_attestation.nras_timeout_secs` | `30` | Timeout for each direct NRAS REST request; must be greater than zero when `source = "nras-rest"` |
| `gpu_attestation.rim_url` | `null` | Optional RIM URL passed to `nvattest attest --rim-url`; `gpu-confidential` production policy requires HTTPS when configured |
| `gpu_attestation.ocsp_url` | `null` | Optional OCSP URL passed to `nvattest attest --ocsp-url`; `gpu-confidential` production policy requires HTTPS when configured |
| `gpu_attestation.relying_party_policy_path` | `null` | Optional relying-party policy file for `nvattest attest`; `gpu-confidential` production policy requires an absolute path to an existing non-empty regular file when configured |
| `gpu_attestation.nvattest_timeout_secs` | `30` | Timeout for each `nvattest` command; must be greater than zero when `source = "nvattest-cli"` |
| `model_key_source` | none | Singleton block with either `type = "file"` or `type = "env"` plus `value`; supplies the decryption key for `.enc` model files |
| `key_provider` | `"static"` | Key provider type: `"static"` (uses `model_key_source`) or `"rotating"` (uses ordered `key_rotation_source` blocks); unknown values fail validation |
| `key_rotation_source` | none | Repeatable ordered block with `type` and `value`; required when `key_provider = "rotating"` |
| `in_memory_decrypt` | `false` | Load encrypted GGUF plaintext from locked RAM when the selected backend supports it (`picolm`); unsupported backends fail closed |
| `streaming_decrypt` | `false` | Load encrypted GGUF plaintext through `LayerStreamingDecryptedModel` when the selected backend supports it (`picolm`); unsupported backends fail closed |
| `suppress_token_metrics` | `false` | Round token counts in responses to nearest 10 (prevents exact token-count side-channel) |
| `rate_limit_rps` | `0` | Max requests per second for `/v1/*` endpoints (`0` = unlimited) |
| `max_concurrent_requests` | `0` | Max concurrent in-flight inference requests; excess **queue** for an admission permit held across the streamed response (`0` = unlimited) |
| `proxy_upstream "MODEL"` | none | Repeatable block whose `url` points to an OpenAI-compatible Power member or backend. Proxied inference runs on that upstream, outside this process's TEE |
| `proxy_effective_prompt_digest` | `false` | Ask proxy upstreams for a rendered chat prompt digest before inference and include it in receipts when returned |
| `proxy_effective_prompt_digest_required` | `false` | Fail closed when a proxy upstream does not support or cannot return the rendered prompt digest |
| `proxy_effective_prompt_digest_path` | `"/v1/chat/effective-prompt-digest"` | Upstream endpoint path for proxy rendered prompt digest requests |
| `tls_port` | `null` | TLS server port; when set, a TLS server starts in parallel; configuration validation fails unless the binary was built with the `tls` feature |
| `tls_sans` | `[]` | Additional DNS names or IP addresses for the TLS certificate; invalid entries fail closed instead of being skipped |
| `ra_tls` | `false` | Embed TEE attestation in TLS cert (RA-TLS); fails configuration validation unless `tls_port` and `tee_mode = true` are set, and startup fails closed if no attestation report can be embedded |
| `vsock_port` | `null` | Vsock port for guest-host communication (`vsock` feature, Linux only) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `A3S_POWER_HOME` | Base directory for all Power data (default: `~/.a3s/power`) |
| `A3S_POWER_HOST` | Server bind address |
| `A3S_POWER_PORT` | Server port; invalid values fail closed |
| `A3S_POWER_DATA_DIR` | Model storage directory |
| `A3S_POWER_MAX_MODELS` | Max concurrent loaded models; invalid values fail closed |
| `A3S_POWER_KEEP_ALIVE` | Default keep-alive duration |
| `A3S_POWER_SPEC_MODE` | Model-neutral speculative strategy; invalid or backend-unsupported explicit values fail closed |
| `A3S_POWER_SPEC_DRAFT_MAX` | Maximum draft tokens per verification pass (`1..64`) |
| `A3S_POWER_SPEC_DRAFT_MIN` | Minimum draft tokens required from a model-backed adapter |
| `A3S_POWER_SPEC_DRAFT_P_MIN` | Minimum model-backed draft confidence (`0.0..1.0`) |
| `A3S_POWER_MODEL_SOURCE` | Remote model hub source for pull (`"modelscope"`, `"hf"`, or `"huggingface"`); invalid configured values fail closed |
| `A3S_POWER_HUB_TOKEN` | Generic bearer token fallback for remote model hub pulls |
| `A3S_POWER_GPU_LAYERS` | GPU layer offloading; invalid values fail closed |
| `A3S_POWER_GPU_ATTESTATION_SOURCE` | GPU CC evidence source (`"configured"`, `"nvattest-cli"`, or `"nras-rest"`); invalid values fail closed |
| `A3S_POWER_GPU_ATTESTATION_PROVIDER` | Provider label for NVIDIA GPU CC evidence claims |
| `A3S_POWER_GPU_ATTESTATION_EVIDENCE_PATH` | Path to raw NVIDIA GPU CC evidence bytes |
| `A3S_POWER_GPU_ATTESTATION_EVIDENCE_HEX` | Hex-encoded raw NVIDIA GPU CC evidence bytes |
| `A3S_POWER_GPU_ATTESTATION_VERDICT_PATH` | Path to raw NVIDIA NRAS verdict bytes |
| `A3S_POWER_GPU_ATTESTATION_VERDICT_HEX` | Hex-encoded raw NVIDIA NRAS verdict bytes |
| `A3S_POWER_GPU_ATTESTATION_NVATTEST_PATH` | Path to NVIDIA's `nvattest` CLI |
| `A3S_POWER_GPU_ATTESTATION_NVATTEST_VERIFIER` | `nvattest attest --verifier` value |
| `A3S_POWER_GPU_ATTESTATION_NVATTEST_GPU_EVIDENCE_SOURCE` | Live GPU evidence source (`"nvml"` or `"corelib"`) |
| `A3S_POWER_GPU_ATTESTATION_NVATTEST_GPU_ARCHITECTURE` | Architecture value for `corelib` evidence collection |
| `A3S_POWER_GPU_ATTESTATION_NRAS_URL` | Optional NRAS URL |
| `A3S_POWER_GPU_ATTESTATION_NRAS_GPU_ARCHITECTURE` | GPU architecture for direct NRAS REST (`"HOPPER"` or `"BLACKWELL"`) |
| `A3S_POWER_GPU_ATTESTATION_NRAS_CLAIMS_VERSION` | Claims version for direct NRAS REST (`"2.0"` or `"3.0"`) |
| `A3S_POWER_GPU_ATTESTATION_NRAS_BEARER_TOKEN_ENV` | Environment variable containing an optional NRAS REST bearer token |
| `A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS` | Timeout for each direct NRAS REST request; invalid values fail closed |
| `A3S_POWER_GPU_ATTESTATION_RIM_URL` | Optional RIM URL |
| `A3S_POWER_GPU_ATTESTATION_OCSP_URL` | Optional OCSP URL |
| `A3S_POWER_GPU_ATTESTATION_RELYING_PARTY_POLICY_PATH` | Optional relying-party policy file |
| `A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS` | Timeout for each `nvattest` command; invalid values fail closed |
| `A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST` | Enable proxy upstream rendered prompt digest requests; invalid values fail closed |
| `A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED` | Require proxy upstream rendered prompt digest support and fail closed when missing; invalid values fail closed |
| `A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_PATH` | Upstream endpoint path for rendered prompt digest requests |
| `A3S_POWER_TEE_MODE` | Enable TEE mode (`"1"` or `"true"`); invalid values fail closed |
| `A3S_POWER_TEE_POLICY_MODE` | Set TEE policy mode (`"strict"`, `"development"`, or `"gpu-confidential"`); invalid values fail closed |
| `A3S_POWER_TEE_STRICT` | Legacy shortcut: `"1"` selects strict policy and removes simulated TEE from the allowlist |
| `A3S_POWER_REDACT_LOGS` | Enable log redaction (`"1"` or `"true"`); invalid values fail closed |
| `A3S_POWER_TLS_PORT` | TLS server port (`tls` feature required); invalid values fail closed |
| `A3S_POWER_RA_TLS` | Enable RA-TLS attestation embedding (`"1"` or `"true"`); invalid values fail closed |
| `A3S_POWER_AUDIT_LOG` | Enable structured audit logging (`"1"` or `"true"`); invalid values fail closed |
| `A3S_POWER_VSOCK_PORT` | Vsock port (`vsock` feature, Linux only); invalid values fail closed |
| `A3S_TEE_SIMULATE` | Simulate TEE environment for development (`"1"`) |

## TEE Privacy Protection

### Model Integrity Verification

When `tee_mode = true`, Power uses `tee_policy_mode = "strict"` by default. Strict policy refuses to start unless the detected hardware TEE type has a 48-byte `expected_measurements` launch-measurement pin and local models are pinned with `model_hashes` or covered by `model_signing_key`; it also rejects simulated TEE evidence. Use `tee_policy_mode = "development"` only for local tests that intentionally rely on `A3S_TEE_SIMULATE=1`.

`tee_policy_mode = "gpu-confidential"` is for deployments that require NVIDIA GPU confidential-computing evidence to be bound into the CPU TEE attestation. Power supports ordinary CUDA acceleration separately, but ordinary CUDA is not a GPU confidential-computing attestation claim. In GPU confidential mode, startup requires final `gpu.gpu_layers != 0` so a CPU-only execution policy cannot be paired with GPU evidence; use `tee_policy_mode = "strict"` for CPU-only TEE deployments. On NVIDIA CUDA hosts, the default GPU auto-configuration sets `gpu_layers = -1` unless the operator overrides it. In GPU confidential mode, `/v1/attestation` requires a 32-byte nonce, encoded as 64 hex characters. Power gives that same nonce to the GPU evidence provider, hashes the raw GPU evidence and NRAS verdict bytes, emits a `GpuEvidenceClaim`, and binds it together with the CPU nonce into `sha256(canonical_claims_v2)` in CPU TEE `report_data`. For live `nvattest-cli`, direct `nras-rest`, and configured NVIDIA verdict JSON paths that expose claims, the GPU claim also includes structured NVIDIA device claims extracted from the verdict: device type, `eat_nonce`, hardware model, UEID/OEM ID, driver and firmware versions, measurement result, secure-boot/debug status, and normalized validation booleans for report signature, nonce match, FWID match, RIM schema validation, RIM signature, version match, and measurement availability. Power fails closed if an NVIDIA GPU/NVSwitch claim reports `measres != "success"`, omits `secboot` or sets it to `false`, omits `dbgstat`, or reports a debug state other than `disabled`.

When `gpu_attestation.source = "configured"`, Power reads externally produced
GPU evidence and verdict bytes from file or hex configuration. Startup verifies
that evidence and verdict sources exist, and file-backed sources must use
absolute paths to existing non-empty regular files in the production profile.
Configured evidence and verdict byte sources are capped at 64 MiB.
Each nonce-bound `/v1/attestation` call then requires the configured verdict to
be NVIDIA nvattest or NRAS JSON whose device `eat_nonce` matches the request
nonce; stale or non-parseable verdicts fail closed instead of being rebound to a
fresh CPU nonce.

When `gpu_attestation.source = "nvattest-cli"`, Power invokes NVIDIA's
`nvattest` binary for collection and attestation. Configure remote NRAS
credentials, service keys, and relying-party policy according to the NVIDIA
`nvattest` deployment instructions; Power does not store NRAS service keys in
A3S ACL. In `gpu-confidential` policy mode, startup requires
`provider = "nvidia-nras"`, an absolute executable `nvattest_path`, and
`nvattest_verifier = "remote"` so NVIDIA NRAS verifies the GPU evidence; local
`nvattest` verification and PATH-resolved CLI lookup are reserved for
development or non-production experiments outside the production profile. If a
relying-party policy path is configured, it must be an absolute path to an
existing non-empty regular file. If custom `nras_url`, `rim_url`, or `ocsp_url`
values are configured in the production profile, they must be HTTPS URLs. The
temporary evidence file passed from `nvattest collect-evidence` to
`nvattest attest` is created with exclusive create semantics and owner-only
permissions on Unix. Power reads `nvattest` stdout/stderr with bounded buffers:
evidence and verdict stdout are capped at 64 MiB, and stderr diagnostics are
capped at 1 MiB.

When `gpu_attestation.source = "nras-rest"`, Power reads configured
DeviceEvidenceV2 JSON from `evidence_path` or `evidence_hex`, posts
`nonce`, `arch`, `evidence_list`, and `claims_version` to NVIDIA NRAS
`/v4/attest/gpu`, then binds the returned detached EAT JSON as the verdict.
File-backed evidence must use an absolute path to an existing non-empty regular
file in the production profile.
GPU confidential mode requires clients to send a 64-character hex nonce to
`/v1/attestation`. The evidence JSON may be a single
`{ evidence, certificate, firmware_version? }` object, an `evidence_list`
array, or an `nvattest collect-evidence` wrapper whose embedded nonce matches
the attestation nonce. Power validates `evidence` and `certificate` locally as
non-empty base64/base64url before posting to NRAS, and rejects evidence lists
or structured verdict device-claim lists with more than 1024 entries. In the
production profile, the default NVIDIA NRAS endpoint is used when `nras_url` is
omitted. Direct
`nras-rest` overrides must use HTTPS and may be a service root/base path that
Power expands to `/v4/attest/gpu`, or the exact full `/v4/attest/gpu`
endpoint; query strings, fragments, embedded credentials, and unsupported
versioned paths are rejected during `gpu-confidential` startup and before
direct NRAS REST requests. Direct NRAS REST response bodies are capped at 16
MiB before JSON or detached-EAT parsing. Detached EAT values must appear in
explicit EAT fields, are capped at 1024 tokens, and must contain JWTs with
base64url JSON payloads; unrelated version strings elsewhere in the response
are not treated as token candidates.
If `nras_bearer_token_env` is configured, it names the environment variable
holding the bearer token; Power trims the name and rejects empty or non-portable
names before making NRAS requests. The token value itself is trimmed, must be
non-empty, and must contain only visible ASCII characters.

Verifiers can pin the exact GPU deployment identity with
`ExpectedGpuEvidence` and `ExpectedGpuDevices` in the SDK or with CLI flags
such as `--gpu-provider`, `--gpu-evidence-format`, `--gpu-verdict-format`,
`--gpu-evidence-count`, `--gpu-count`, `--nvswitch-count`,
`--gpu-claims-version`, `--gpu-ueid`, `--gpu-oemid`, `--gpu-hwmodel`,
`--gpu-driver-version`, `--gpu-firmware-version`, `--nvswitch-claims-version`,
`--nvswitch-ueid`, `--nvswitch-oemid`, `--nvswitch-hwmodel`, and
`--nvswitch-firmware-version`. UEID pinning is an exact device-set check;
GPU/NVSwitch counts pin the attested topology; claims version, OEM ID, hwmodel,
driver version, and firmware version pins are allow-lists applied to every
attested NVIDIA GPU or NVSwitch claim for the matching device type. OEM ID is
supplemental and does not replace exact UEID pinning or deployment-specific
model/version pins in the production profile.
For production NVIDIA GPU confidential-computing deployments, prefer
`a3s-power-verify --gpu-confidential`: it requires a 32-byte `--nonce`,
`--gpu-verdict-digest`, and bundles v2 claims, top-level GPU evidence nonce
freshness, structured device nonce freshness, NVIDIA NRAS verdict binding,
required GPU provider/format/count pins, structured NVIDIA device claims,
required exact GPU topology plus claims-version and identity/version pins,
runtime policy, and GPU
execution/offload digest checks into one verifier profile. The production
profile requires `--gpu-claims-version` plus either an exact `--gpu-ueid` set or
`--gpu-count` together with at least one of `--gpu-hwmodel`,
`--gpu-driver-version`, or `--gpu-firmware-version`, and it rejects NVIDIA
device claims unless secure boot is enabled and debug is disabled. When
`--nvswitch-count` is greater than zero, the production profile also requires
`--nvswitch-claims-version` plus either an exact `--nvswitch-ueid` set or at
least one of `--nvswitch-hwmodel` or `--nvswitch-firmware-version`.
Verifiers can also pin the model execution/offload policy with
`--require-runtime-policy --gpu-execution-digest <64-char-hex>`, which checks
the attested `runtime.execution.gpu_sha256` digest over canonical
`gpu_layers`, `main_gpu`, and `tensor_split` values. To compute that value
without reimplementing Power's canonical JSON semantics, use:

```bash
a3s-power-verify --print-gpu-execution-digest \
  --gpu-layers <N> \
  --main-gpu <N> \
  --tensor-split <CSV>
```

For CPU TEE hardware-signature operations, including `hw-verify` builds,
AMD KDS / Intel PCS outbound access, raw-report requirements, and production
failure handling, see
[`docs/hardware-verifier-operations.md`](docs/hardware-verifier-operations.md).

```acl
tee_mode = true
tee_policy_mode = "gpu-confidential"
model_hash "llama3.2:3b" {
  digest = "sha256:a1b2c3d4e5f6..."
}

gpu_attestation {
  source = "nvattest-cli"
  provider = "nvidia-nras"
  nvattest_path = "/usr/local/bin/nvattest"
  nvattest_verifier = "remote"
  nvattest_gpu_evidence_source = "nvml"
  # nras_url = "https://<your-nras-endpoint>"
}
```

```
INFO TEE mode enabled tee_type="sev-snp"
INFO Model integrity verified model="llama3.2:3b"
INFO All model integrity checks passed count=1
```

### Remote Attestation

The `TeeProvider` detects the TEE environment and generates attestation reports:

| TEE Type | Detection | Description |
|----------|-----------|-------------|
| AMD SEV-SNP | `/dev/sev-guest` | Hardware memory encryption + attestation |
| Intel TDX | `/dev/tdx_guest` | Trust Domain Extensions |
| Simulated | `A3S_TEE_SIMULATE=1` | Development/testing mode; rejected by strict production policy |
| None | (default) | No TEE detected |

The `/health` endpoint exposes TEE status:

```json
{
  "status": "ok",
  "version": "0.4.0",
  "uptime_seconds": 120,
  "loaded_models": 1,
  "tee": {
    "enabled": true,
    "type": "sev-snp",
    "models_verified": true
  }
}
```

### Log Redaction

When `redact_logs = true`, the `PrivacyProvider` automatically strips inference content from all log output:

```
// Before redaction:
{"content": "tell me a secret", "model": "llama3"}

// After redaction:
{"content": "[REDACTED]", "model": "llama3"}
```

Redacted JSON keys: `"content"`, `"prompt"`, `"text"`, `"arguments"`, `"input"`, `"delta"`, `"system"`, `"message"`, `"query"`, `"instruction"` — covering chat messages, tool call arguments, streaming deltas, system prompts, and completion requests.

Error messages that echo prompt content are also sanitized via `sanitize_error()`. When `suppress_token_metrics = true`, token counts in responses are rounded to the nearest 10 to prevent exact token-count side-channel inference.

## API Reference

### Server Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with TEE status, version, uptime, loaded models |
| `GET` | `/metrics` | Prometheus metrics (requests, durations, tokens, inference, TTFT, model memory, GPU) |

### OpenAI-Compatible API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming/non-streaming, vision, tools, thinking) |
| `POST` | `/v1/completions` | Text completion (streaming/non-streaming) |
| `POST` | `/v1/embeddings` | Generate embeddings |
| `GET` | `/v1/models` | List all registered models |
| `GET` | `/v1/models/:name` | Get a single model by name |
| `POST` | `/v1/models` | Register a local model artifact (`name`, `path`, optional `format` of `gguf`, `safetensors`, or `huggingface`); unsupported formats and unknown fields fail closed |
| `DELETE` | `/v1/models/:name` | Unload and deregister a model |
| `POST` | `/v1/models/pull` | Pull a GGUF model from HuggingFace Hub (`name`, optional `force` and `token` body fields); unknown fields fail closed; streams SSE progress events; requires `hf` feature; concurrent pulls of the same model are deduplicated |
| `GET` | `/v1/models/pull/:name/status` | Get persisted pull progress for a model (`status`, `completed`, `total`, `error`); URL-encode names that contain `/` or `:` |
| `GET` | `/v1/attestation` | TEE attestation report (returns 503 if TEE not enabled); optional `?nonce=<hex>` binds client nonce; optional `?model=<name>` emits v2 model/runtime claims and binds the claims digest into `report_data`; unknown query parameters fail closed; `gpu-confidential` mode also binds GPU evidence claims and requires a 32-byte `?nonce=<64-hex>` |

The `a3s-power models show` and `a3s-power models rm` commands encode model
names as URL path segments automatically. Manual HTTP clients must percent-encode
path parameters that contain `/`, `:`, spaces, or query-special characters.

Chat and text completion responses include an `attestation_receipt` object and
`attestation_receipt_sha256`. For streaming calls, Power emits a final SSE event
with those fields before `[DONE]`; when `stream_options.include_usage = true` or
`suppress_token_metrics = true`, that final event also includes `usage`.
Power rejects `stream_options` on non-streaming requests because those options
do not affect a non-streaming response shape, and currently supports only
`stream_options.include_usage`; other stream option fields fail closed.
Embedding requests accept only implemented top-level fields; unsupported fields
such as `user` fail closed instead of being silently dropped.

The v2 receipt covers prompt-bearing API input, model runtime
chat-template/GPU execution policy claims, request decoding parameters
including extended local sampling controls, streaming request options, stop
tokens, response format, tools including function `strict` schema flags, tool
choice, and parallel tool-call policy. Unknown top-level chat/text completion
fields and unknown nested message/content, response-format, tool definition, and
tool-choice fields fail closed instead of being silently dropped before proxying
or receipt hashing. Local chat backends reject unsupported message roles instead
of coercing them to `user`; remote/proxy models preserve roles for upstream
enforcement. Chat receipts also include
`effective_prompt` when the selected backend can expose the exact prompt
representation it submits to the model. llama.cpp and picolm text-only chat
emit `kind = "chat.rendered-prompt"` for post-template prompt bytes. mistralrs
text chat emits `kind = "chat.prompt-token-ids"` for a domain-separated SHA-256
over the token ID sequence produced by mistralrs' own chat tokenization path;
vision and multimodal llama.cpp/picolm/mistralrs requests leave the field absent.
Proxy backends leave the field absent by default, but can include an
upstream-declared digest when
`proxy_effective_prompt_digest = true` and the upstream implements the configured
digest endpoint. The proxy sends the same OpenAI-compatible chat body used for
inference, including structured multimodal content, tools, tool choice,
parallel tool-call policy, response format, and sampling controls, with
`stream = false`; the endpoint should return either
`{ "sha256": "<64 hex>", "kind": "chat.rendered-prompt", "backend": "..." }` or
the same object nested under `effective_prompt`. Unsupported proxy endpoints are
ignored unless `proxy_effective_prompt_digest_required = true`; malformed
digests fail closed.

`a3s-power-verify` can bind a saved receipt back to a saved or fetched
attestation report. Receipt verification first checks the receipt schema,
request type/input-kind pairing, and all receipt digest fields before comparing
the receipt runtime policy with the attested runtime policy. Verifiers can also
pin receipt-level policy with `--receipt-model`, `--receipt-request-type`,
`--receipt-input-digest`, `--receipt-decoding-parameters-digest`,
`--receipt-stream-options-digest`, `--receipt-stop-tokens-digest`,
`--receipt-response-format-digest`, `--receipt-tools-digest`,
`--receipt-tool-choice-digest`, `--effective-prompt-digest` (all digest pins
are 64-character SHA-256 hex values),
`--require-effective-prompt-absent`, `--effective-prompt-backend`, and
`--effective-prompt-kind`. When the original request JSON is available,
`--receipt-chat-request-file` or `--receipt-completion-request-file` recomputes
and compares every request-derived receipt field:

```bash
a3s-power-verify --file report.json \
  --receipt-file receipt.json \
  --receipt-chat-request-file chat-request.json \
  --receipt-digest <64-char-hex> \
  --receipt-model llama3 \
  --receipt-request-type chat-completion \
  --receipt-input-digest <64-char-hex> \
  --receipt-decoding-parameters-digest <64-char-hex> \
  --receipt-stream-options-digest <64-char-hex> \
  --receipt-stop-tokens-digest <64-char-hex> \
  --allow-offline
```

Use `--effective-prompt-digest <64-char-hex>` when receipt policy pins the
exact rendered-prompt or prompt-token-ID digest exposed in `effective_prompt`.
Use `--require-effective-prompt-absent` for opaque multimodal paths where the
receipt must prove that Power did not overclaim a post-template prompt digest.
When `--receipt-chat-request-file` points to an image-bearing request,
`a3s-power-verify` applies that absence requirement by default unless the
verifier explicitly pins an effective-prompt digest, backend, or kind.
Use `--require-runtime-policy --gpu-execution-digest <64-char-hex>` when
verifier policy pins the exact GPU execution/offload configuration used by the
attested server.

SDK callers that still have the original request can use
`verify_receipt_matches_chat_request()` or
`verify_receipt_matches_completion_request()` to recompute and compare all
request-derived receipt fields before separately checking attestation runtime
policy or `effective_prompt` pins.

Use this command to calculate the GPU execution pin with Power's own canonicalizer:

```bash
a3s-power-verify --print-gpu-execution-digest \
  --gpu-layers <N> \
  --main-gpu <N> \
  --tensor-split <CSV>
```

### Examples

#### Chat Completion

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Streaming Chat

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### Text Completion

```bash
curl http://localhost:11434/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Once upon a time"
  }'
```

#### Tool/Function Calling

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

#### Structured Output (JSON Schema)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "List 3 colors with hex codes"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "color_list",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "colors": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "hex": {"type": "string"}
                },
                "required": ["name", "hex"],
                "additionalProperties": false
              }
            }
          },
          "required": ["colors"],
          "additionalProperties": false
        }
      }
    }
  }'
```

Local JSON Schema enforcement requires a backend that can apply the requested
grammar. Power rejects unsupported local backend/schema combinations instead of
silently ignoring `response_format`; remote models preserve the OpenAI wire
shape for upstream enforcement.

Per-request `keep_alive` overrides must use the same validated duration format
as configuration. Invalid values are rejected instead of falling back to the
server default.

#### List Models

```bash
curl http://localhost:11434/v1/models
```

Each model entry includes `context_length` when the registered manifest
declares an exact token window. The field is omitted when unknown; Power does
not invent a backend default.

#### Pull a Model from a Remote Hub

Requires the `hf` feature (`cargo build --features hf`). Power pulls from
ModelScope by default; set `A3S_POWER_MODEL_SOURCE=hf` or
`A3S_POWER_MODEL_SOURCE=huggingface` to use HuggingFace Hub. Any other
configured source value fails closed instead of silently falling back. Streams
SSE progress:

```bash
# By quantization tag (resolves filename via HF API)
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"}'

# By exact filename
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf"}'

# Private/gated model with a hub token
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "meta-llama/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "token": "hf_..."}'

# Force re-download
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "force": true}'

# Check persisted progress for a pull; manual HTTP clients must URL-encode names
# containing "/", ":", spaces, or query-special characters.
curl http://localhost:11434/v1/models/pull/bartowski%2FLlama-3.2-3B-Instruct-GGUF%3AQ4_K_M/status

# The CLI encodes the model name automatically.
a3s-power models status bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
```

SSE response stream:
```
data: {"status":"resuming","offset":104857600,"total":2147483648}   ← if resuming
data: {"status":"downloading","completed":209715200,"total":2147483648}
data: {"status":"verifying"}
data: {"status":"success","id":"bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M","object":"model","created":1234567890}
```

Interrupted downloads resume automatically on retry — the partial file is identified by a SHA-256 of the canonical download URL and picked up via HTTP `Range` requests. Hub download and file-list URLs are built with a URL parser, preserving intended repo/file subdirectories while percent-encoding spaces and query-special characters. Set the selected hub's token env var (`MODELSCOPE_TOKEN` or `HF_TOKEN`) or `A3S_POWER_HUB_TOKEN` as an alternative to passing `token` in the request body.

#### Health Check (with TEE status)

```bash
curl http://localhost:11434/health
```

## Model Storage

Models are stored in `~/.a3s/power/` (override with `$A3S_POWER_HOME`):

```
~/.a3s/power/
├── config.acl               # A3S ACL configuration
└── models/
    ├── manifests/            # JSON manifest files
    │   ├── llama3.2-3b.json
    │   └── qwen2.5-7b.json
    └── blobs/                # Content-addressed model files
        ├── sha256-abc123...
        └── sha256-def456...
```

Model files are stored by SHA-256 hash, enabling deduplication and integrity verification.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `mistralrs` | ✅ enabled | Pure Rust inference backend via `mistralrs` (candle-based). No C++ inference toolchain required. Ideal for TEE auditing. |
| `llamacpp` | ❌ disabled | CPU llama.cpp inference backend via `llama-cpp-2`. Requires C++ compiler + CMake. Includes KV cache, LoRA, grammar, mirostat, and native MTP for compatible GGUF models. |
| `llamacpp-cuda` | ❌ disabled | Composite `llamacpp` build with the `llama-cpp-2/cuda` backend. Requires a compatible NVIDIA CUDA toolchain. |
| `picolm` | ❌ disabled | Pure Rust layer-streaming GGUF inference. Real transformer ops (multi-head attention, SwiGLU FFN, RoPE, RMSNorm). Peak RAM = O(layer_size) not O(model_size) via `madvise(DONTNEED)`. FP16 KV cache with fused f16 dot/accumulate. Fused dequant+dot kernels. NEON SIMD (aarch64) + AVX2 (x86_64). Batch prefill, speculative decoding, tool calling, grammar-constrained output. 14+ tok/s decode on Apple Silicon. Enables 7B+ models in 512MB TEE EPC. No C/C++ inference backend. ~4,500 lines of pure Rust. |
| `hf` | ❌ disabled | Remote model hub pull (`POST /v1/models/pull`). Range resume, SSE progress, source-specific hub token auth. |
| `tls` | ❌ disabled | RA-TLS transport: TLS server with self-signed cert + optional attestation X.509 extension. Adds `axum-server`, `rcgen`, `time` deps. |
| `vsock` | ❌ disabled | Vsock transport for a3s-box MicroVM guest-host HTTP. **Linux only** — requires `AF_VSOCK` kernel support. Adds `tokio-vsock` and `hyper-util` deps. |
| `hw-verify` | ❌ disabled | Hardware attestation signature verification. AMD KDS (ECDSA P-384) + Intel PCS (ECDSA P-256) certificate chain validation. |
| `tee-minimal` | ❌ disabled | Composite: `picolm` + `tls` + `vsock`. Smallest auditable TEE build — no mistralrs/candle and no C++ inference engine. TLS/crypto still uses native `ring`/`aws-lc-sys` build dependencies. |

Without a backend feature (`mistralrs`, `llamacpp`, or `picolm`), Power can manage models but inference calls return "backend not available".

## TEE Deployment

For production TEE deployments (AMD SEV-SNP / Intel TDX), use the `tee-minimal` build profile:

```bash
cargo build --release --no-default-features --features tee-minimal
```

### Why `tee-minimal`?

Inside a TEE, every crate in the inference path is part of the trusted computing base.
The `tee-minimal` profile minimizes this surface:

| Profile | Inference backend | Dep tree lines | Native inference deps | Other native deps |
|---------|------------------|----------------|-----------------------|-------------------|
| `default` | mistralrs (candle) | ~2,000 | None | TLS/HTTP crypto crates may build C crypto helpers |
| `tee-minimal` | picolm (pure Rust) | ~1,220 | None | `ring`/`aws-lc-sys` via TLS/RA-TLS crypto |
| `llamacpp` | llama.cpp | ~1,800+ | Yes (C++) | C++ compiler + CMake |

### What `tee-minimal` includes

- **picolm backend**: Pure Rust layer-streaming GGUF inference (~4,500 lines, fully auditable). Real transformer ops, 14+ tok/s decode, FP16 KV cache, true O(layer_size) peak RAM.
- **Full TEE stack**: attestation, model integrity (SHA-256), log redaction, memory zeroing
- **Encrypted model loading**: AES-256-GCM file-backed loading plus `picolm` GGUF loading from locked plaintext RAM or `LayerStreamingDecryptedModel`; unsupported backends fail closed before load
- **RA-TLS transport**: attestation embedded in X.509 cert
- **Vsock transport**: for a3s-box MicroVM guest-host communication

### Layer-Streaming Inference (picolm) — How It Works

Traditional LLM inference loads the entire model into RAM before generating a single token. A 7B Q4_K_M model needs ~4 GB. Inside a TEE, the Encrypted Page Cache (EPC) is often limited to 512 MB–1 GB. The model simply doesn't fit.

picolm solves this with **layer-streaming**: instead of loading all weights at once, it memory-maps the GGUF file and processes one transformer layer at a time. Only the current layer's weights occupy physical RAM. After processing, the OS reclaims those pages.

#### Memory Model

```
Traditional (mistralrs / llama.cpp):
┌──────────────────────────────────────────────────┐
│  All 32 layers loaded in RAM simultaneously       │
│  Peak RAM ≈ model_size (e.g. 4 GB for 7B Q4_K_M) │
└──────────────────────────────────────────────────┘

picolm layer-streaming:
┌──────────────────────────────────────────────────┐
│  mmap(model.gguf)  ← virtual address space only   │
│                       no physical RAM allocated    │
│                                                    │
│  for layer in 0..n_layers:                         │
│    ┌─────────────────────────┐                     │
│    │ blk.{layer}.* tensors   │ ← OS pages in       │
│    │ (~120 MB for 7B Q4_K_M) │   weights on demand  │
│    └─────────────────────────┘                     │
│    forward_pass(hidden_state, layer_weights)        │
│    madvise(MADV_DONTNEED) ← release physical pages │
│                                                    │
│  Peak RAM ≈ layer_size + KV cache (FP16)           │
│           ≈ 120 MB + 44 MB (7B, 2048 ctx)         │
└──────────────────────────────────────────────────┘
```

#### Technical Architecture

The implementation has two components:

**1. `gguf_stream.rs` — Zero-Copy GGUF Parser**

Opens the GGUF file via `mmap(MAP_PRIVATE | PROT_READ)`. Parses the header (v2/v3), metadata, and tensor descriptors — but does **not** load any weight data. Each tensor is recorded as an `(offset, size)` pair into the mmap region.

When picolm requests a layer's weights, `tensor_bytes(name)` returns a `&[u8]` slice directly into the mmap — zero copy, zero allocation. The OS kernel pages in the data on first access and can evict it under memory pressure.

```
GGUF file on disk:
┌────────┬──────────┬──────────────────────────────────┐
│ Header │ Metadata │ Tensor Data (aligned)              │
│ 8 bytes│ variable │ blk.0.attn_q | blk.0.attn_k | ... │
└────────┴──────────┴──────────────────────────────────┘
                          ↑
                    mmap returns &[u8] slice
                    directly into this region
                    (no memcpy, no allocation)
```

**2. `picolm.rs` + `picolm_ops/` — Layer-Streaming Forward Pass**

Iterates `blk.0.*` through `blk.{n-1}.*`, applying each layer's weights to the hidden state. After processing layer N, `madvise(MADV_DONTNEED)` explicitly releases the physical pages. The OS is guaranteed to reclaim them before layer N+1 is paged in — this is what makes peak RAM truly O(layer_size).

Key optimizations:
- **TensorCache**: All tensor byte slices and types resolved once at load time into a flat array. The hot path indexes by `layer * 10 + slot` — zero string formatting, zero HashMap lookups.
- **ForwardBuffers**: All working buffers (q, k, v, gate, up, down, normed, logits, scores, attn_out) pre-allocated once. Zero heap allocation during inference.
- **Fused vec_dot**: Dequant+dot in a single pass per row — no intermediate f32 buffer. Dedicated kernels for Q4_K, Q6_K, Q8_0.
- **Rayon parallel matmul**: Multi-threaded row parallelism for matrices with >64 rows.
- **FP16 KV cache**: Keys and values stored as `f16`, converted on read. Halves KV cache memory.
- **Pre-computed RoPE**: cos/sin tables built at load time. No transcendental functions in the hot path.

```rust
// Simplified flow (actual code in src/backend/picolm.rs)
let gguf = GgufFile::open("model.gguf")?;  // mmap, parse header only
let tc = TensorCache::build(&gguf, n_layers)?;  // resolve tensor pointers once
let rope_table = RopeTable::new(max_seq, head_dim, rope_dim, theta);
let mut hidden = vec![0.0f32; n_embd];
let mut buf = ForwardBuffers::new(/* pre-allocate all working buffers */);

for layer in 0..n_layers {
    attention_layer(&mut hidden, &tc, layer, pos, kv_cache, &rope_table, &mut buf)?;
    ffn_layer(&mut hidden, &tc, layer, activation, &mut buf)?;
    tc.release_layer(&gguf, layer);  // madvise(DONTNEED) — free physical pages
}
```

#### Encrypted Model Support

For encrypted models (`.enc`), `LayerStreamingDecryptedModel` exposes chunked
plaintext access where each returned chunk is wrapped in `Zeroizing<Vec<u8>>`.
`streaming_decrypt = true` passes this source to backends that explicitly support
it. Today that means `picolm` for GGUF models; unsupported backends fail closed
before load.

- Chunk buffers are zeroized when dropped
- The full decrypted plaintext is still held in locked memory because the current
  AES-GCM artifact format is not independently seekable
- End-to-end inference from chunked plaintext requires a backend loader that
  consumes this source directly

```
Encrypted layer-streaming:
┌─────────────────────────────────────────────────────┐
│  model.gguf.enc (AES-256-GCM encrypted on disk)      │
│                                                       │
│  after AES-GCM authentication + decrypt to locked RAM: │
│  for each requested range:                             │
│    chunk = read_chunk(layer_offset, layer_len)         │
│    chunk: Zeroizing<Vec<u8>>  ← auto-zeroed on drop   │
│    // future backend path consumes chunk directly      │
│    // chunk dropped → chunk memory zeroed immediately  │
└─────────────────────────────────────────────────────┘
```

#### Real-World Memory Comparison

| Model | Traditional | picolm Layer-Streaming | Reduction |
|-------|------------|----------------------|-----------|
| 0.5B Q4_K_M (~350 MB) | ~350 MB | ~15 MB + KV | 23× |
| 3B Q4_K_M (~2 GB) | ~2 GB | ~60 MB + KV | 33× |
| 7B Q4_K_M (~4 GB) | ~4 GB | ~120 MB + KV | 33× |
| 13B Q4_K_M (~7 GB) | ~7 GB | ~200 MB + KV | 35× |
| 70B Q4_K_M (~40 GB) | ~40 GB | ~1.1 GB + KV | 36× |

KV cache uses FP16 storage (half the memory of F32). For 7B at 2048 context: ~44 MB.

#### Current Status

picolm is a **production-ready** pure Rust inference engine. The full transformer forward pass is implemented:

- **Attention**: Multi-head attention with Grouped-Query Attention (GQA), Q/K/V bias support (Qwen, Phi)
- **FFN**: SwiGLU (LLaMA, Mistral, Phi) and GeGLU (Gemma) activation variants
- **RoPE**: Pre-computed cos/sin tables with partial-dimension support
- **RMSNorm**: On-the-fly dequantization per layer (output norm pre-dequantized)
- **Dequantization**: Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, F16, F32
- **Fused vec_dot**: Dequant+dot in a single pass — no intermediate f32 buffer
- **Parallel matmul**: Rayon multi-threaded row parallelism for large matrices
- **FP16 KV cache**: Half-precision storage with fused f16→f32 dot product and accumulate — no intermediate buffer in attention
- **Tensor cache**: Pre-resolved tensor pointers — zero HashMap lookups in the hot path
- **Pre-allocated buffers**: Zero heap allocation during inference (including sampler probs/indices)
- **True layer-streaming**: `madvise(MADV_DONTNEED)` releases physical pages after each layer
- **BPE tokenizer**: Full GPT-style byte-pair encoding with ChatML template support
- **Batch prefill**: Process prompt tokens in batch for faster time-to-first-token
- **Speculative decoding**: Prompt-lookup draft for faster decode throughput
- **Tool/function calling**: OpenAI-compatible `tool_calls` with auto-dispatch
- **Grammar-constrained output**: JSON Schema enforcement during generation
- **Repeat/frequency/presence penalty**: Configurable repetition control (zero-alloc, stack-based dedup)

Performance on Qwen 2.5 0.5B Q4_K_M (Apple Silicon):
- **Decode**: 14+ tok/s
- **Prefill**: 15+ tok/s
- **900+ tests** across unit, integration, and real-model validation profiles

#### Performance Optimization Status

Profiling breakdown of the decode hot path (per token):

| Stage | % Time | Status |
|-------|--------|--------|
| Embedding lookup | 0.3% | ✅ Optimized |
| Attention (Q·K scores + V weighted sum) | 22.1% | ✅ Fused f16 KV dot/accumulate, NEON softmax |
| FFN (gate + up + down matvec) | 63.4% | ✅ Fused vec_dot, Rayon parallel, NEON SiLU/residual |
| Logit projection | 9.1% | ✅ Rayon parallel matmul |
| Sampling | 0.3% | ✅ Zero-alloc (pre-allocated probs/indices) |

Completed optimizations:
- ✅ NEON SIMD for softmax, RMSNorm, SiLU, add_residual (aarch64)
- ✅ AVX2 SIMD for Q4_K, Q6_K vec_dot kernels (x86_64)
- ✅ Q4_K NEON kernel — register-based nibble extraction via `vld1_lane_u32` + `vand`/`vshr`
- ✅ Fused f16 KV attention — `k_dot()` and `v_accumulate()` skip intermediate f32 buffer
- ✅ Zero-alloc sampler — pre-allocated `probs_buf` and `indices_buf` in `ForwardBuffers`
- ✅ Zero-alloc repeat penalty — stack-based `[(u32, u32); 64]` dedup, no HashMap
- ✅ Pre-computed RoPE cos/sin tables — no transcendental functions in hot path
- ✅ TensorCache — flat array indexed by `layer * SLOTS + slot`, zero HashMap lookups
- ✅ ForwardBuffers — all working buffers pre-allocated, zero heap allocation per token
- ✅ FP16 KV cache — halves memory via `half` crate batch SIMD conversion
- ✅ Rayon parallel matmul — multi-threaded row parallelism for matrices with >64 rows
- ✅ Decode profiling instrumentation — per-stage timing breakdown for continuous optimization

Remaining optimization opportunities (diminishing returns):
- 🔲 Block-wise quantized matmul — process multiple output rows per pass for better cache locality
- 🔲 Integer-only Q4_K accumulation — accumulate in i32, avoid f32 conversion overhead
- 🔲 Tiled matmul with explicit prefetch hints — improve L1/L2 cache utilization
- 🔲 Fused gate+up projection — single matmul pass if weight layout permits
- 🔲 AMX/SME acceleration — Apple Silicon matrix coprocessor (requires nightly Rust)

#### Configuration

```acl
# config.acl — TEE deployment with file-backed encrypted-model loading
tee_mode        = true
redact_logs     = true

# File-backed DecryptedModel loading works with file-based backends.
# in_memory_decrypt works for GGUF models when the selected backend is picolm;
# other backends fail closed rather than reading the encrypted path.

# Direct plaintext-buffer mode:
# in_memory_decrypt = true
# LayerStreamingDecryptedModel mode; requires a supporting backend such as picolm GGUF:
# streaming_decrypt = true
```

### Supply-chain audit

See [`docs/supply-chain.md`](docs/supply-chain.md) for:
- Full dependency listing per feature profile
- Audit status for each crate in the `tee-minimal` inference path
- Security properties of `LayerStreamingDecryptedModel`
- How to reproduce dependency counts and audit unsafe blocks

See [`docs/hardware-verifier-operations.md`](docs/hardware-verifier-operations.md)
for strict AMD SEV-SNP / Intel TDX verifier operations.

### Building with RA-TLS

```bash
# Build with TLS support
cargo build --features tls

# Test TLS cert generation
cargo test --features tls -p a3s-power tee::cert
```

To enable RA-TLS, set `tls_port` and `ra_tls = true` alongside `tee_mode = true`:

```acl
tee_mode = true
tls_port = 11443
ra_tls   = true
```

At startup, the TLS server binds on the configured port with a fresh self-signed ECDSA P-256 certificate. When `ra_tls = true`, startup first requires a TEE provider to generate an attestation report and embeds it as OID extension `1.3.6.1.4.1.56560.1.1`; report generation failures abort startup before the TLS listener is bound. Clients can extract and verify this extension to confirm they are communicating with a genuine TEE before trusting inference output.

## Development

### Build & Test

```bash
# Build
cargo build -p a3s-power                          # Debug (default: mistralrs)
cargo build -p a3s-power --release                 # Release
cargo build -p a3s-power --no-default-features --features llamacpp  # With llama.cpp

# Test (900+ tests across current validation profiles)
cargo test -p a3s-power --lib -- --test-threads=1
cargo test -p a3s-power --test integration

# Test with TLS feature
cargo test -p a3s-power --features tls --lib -- --test-threads=1

# Lint
cargo clippy -p a3s-power -- -D warnings
cargo fmt -p a3s-power -- --check

# Run
cargo run -p a3s-power                             # Start server
```

### Project Structure

```
power/
├── Cargo.toml
├── justfile                     # Build, test, coverage, lint, CI targets
├── README.md
└── src/
    ├── main.rs                  # Entry point: load A3S ACL config → server::start()
    ├── lib.rs                   # Module declarations
    ├── config.rs                # PowerConfig types and defaults
    ├── config/                  # Closed ACL schema/codec, behavior, and focused tests
    ├── runtime_service.rs       # Standard a3s-runtime Service profile for A3S Box
    ├── dirs.rs                  # Platform paths (~/.a3s/power/{manifests,blobs,pulls})
    ├── error.rs                 # PowerError enum (14 variants) + HTTP status mapping
    │
    ├── api/                     # API layer — OpenAI-compatible HTTP handlers
    │   ├── mod.rs               # Shared utilities, timestamp helpers
    │   ├── types.rs             # OpenAI request/response types (chat, completion, embedding)
    │   ├── receipt.rs           # Request-level attestation receipt hashing
    │   ├── health.rs            # GET /health (TEE status, version, uptime, loaded models)
    │   ├── autoload.rs          # Model lifecycle: LRU eviction → decrypt → verify → load
    │   └── openai/              # OpenAI-compatible endpoint handlers
    │       ├── mod.rs           # Route definitions, openai_error() helper
    │       ├── chat.rs          # POST /v1/chat/completions (streaming SSE + JSON)
    │       ├── completions.rs   # POST /v1/completions
    │       ├── embeddings.rs    # POST /v1/embeddings
    │       ├── models.rs        # GET/POST/DELETE /v1/models, POST /v1/models/pull
    │       └── attestation.rs   # GET /v1/attestation (nonce + model hash binding)
    │
    ├── backend/                 # Backend layer — inference engine abstraction
    │   ├── mod.rs               # Backend trait + manifest-aware BackendRegistry (priority, TEE routing)
    │   ├── types.rs             # ChatRequest, ChatResponseChunk, EmbeddingRequest, Tool, ToolCall
    │   ├── mistralrs_backend.rs # Pure Rust: GGUF/SafeTensors/HF/Vision, ISQ (feature: mistralrs) ★
    │   ├── llamacpp.rs          # C++ bindings: KV cache, LoRA, MTMD, grammar, native MTP (feature: llamacpp)
    │   ├── llamacpp/            # llama.cpp backend-specific speculative runtime
    │   ├── picolm.rs            # Pure Rust layer-streaming, O(layer_size) RAM (feature: picolm)
    │   ├── picolm_ops/          # picolm transformer ops (~4,500 lines, pure Rust)
    │   │   ├── attention.rs     # Multi-head / GQA attention with Q/K/V bias support
    │   │   ├── buffers.rs       # Pre-allocated working buffers (zero heap alloc in hot path)
    │   │   ├── dequant.rs       # Dequantization kernels (Q4_K, Q5_K, Q6_K, Q8_0, F16, F32)
    │   │   ├── ffn.rs           # SwiGLU / GeGLU feed-forward network
    │   │   ├── kv_cache.rs      # FP16 KV cache (half memory vs F32)
    │   │   ├── matmul.rs        # Fused vec_dot + rayon parallel matmul
    │   │   ├── norm.rs          # RMSNorm (raw + pre-dequantized weights)
    │   │   ├── rope.rs          # RoPE with pre-computed cos/sin tables
    │   │   ├── tensor_cache.rs  # Per-layer tensor pointer cache (zero HashMap lookups)
    │   │   ├── tokenizer.rs     # BPE tokenizer with ChatML template support
    │   │   └── vec_dot.rs       # Fused dequant+dot kernels (Q4_K, Q6_K, Q8_0)
    │   ├── chat_template.rs     # Jinja2 chat template rendering (ChatML/Llama/Phi/Generic)
    │   ├── gpu.rs               # Metal + CUDA detection, auto gpu_layers config
    │   ├── json_schema.rs       # JSON Schema → GBNF grammar for constrained output
    │   ├── tool_parser.rs       # Tool call parsing (XML/Hermes, Mistral, raw JSON)
    │   ├── think_parser.rs      # Streaming <think> block extraction (DeepSeek-R1, QwQ)
    │   ├── gguf_stream.rs       # GGUF v2/v3 mmap reader for picolm layer-streaming
    │   └── test_utils.rs        # MockBackend for testing
    │
    ├── model/                   # Model layer — storage, registry, pull
    │   ├── mod.rs               # Module declarations
    │   ├── manifest.rs          # ModelManifest, ModelFormat (Gguf/SafeTensors/HuggingFace/Vision)
    │   ├── registry.rs          # ModelRegistry (RwLock<HashMap>, JSON manifest persistence)
    │   ├── storage.rs           # Content-addressed blob store (SHA-256 naming, prune)
    │   ├── gguf.rs              # GGUF metadata reader, memory estimation (KV cache + compute)
    │   ├── pull.rs              # HuggingFace Hub pull with Range resume, SSE progress (feature: hf)
    │   └── pull_state.rs        # Persistent pull state (Pulling/Done/Failed) as JSON
    │
    ├── server/                  # Server layer — transport, auth, metrics, audit
    │   ├── mod.rs               # Server startup orchestration (TCP/TLS/Vsock), graceful shutdown
    │   ├── state.rs             # AppState: model lifecycle, LRU, decrypted model RAII, privacy
    │   ├── router.rs            # Axum router + middleware: rate limit, request ID, metrics, auth
    │   ├── auth.rs              # AuthProvider trait, ApiKeyAuth (SHA-256, constant-time)
    │   ├── audit.rs             # AuditLogger trait: JSONL / Encrypted / Async / Noop
    │   ├── metrics.rs           # Prometheus metrics (16 groups: HTTP, inference, TTFT, GPU, TEE)
    │   ├── request_context.rs   # Per-request context (request_id, auth_id, created_at)
    │   ├── lock.rs              # Shared RwLock helpers
    │   └── vsock.rs             # AF_VSOCK transport (feature: vsock, Linux only)
    │
    ├── tee/                     # TEE layer — cross-cutting security
    │   ├── mod.rs               # Module entry
    │   ├── attestation.rs       # TeeProvider trait, SEV-SNP/TDX ioctl, report_data binding
    │   ├── encrypted_model.rs   # AES-256-GCM: DecryptedModel / MemoryDecrypted / LayerStreaming
    │   ├── key_provider.rs      # KeyProvider trait: StaticKeyProvider + RotatingKeyProvider
    │   ├── model_seal.rs        # SHA-256 integrity + Ed25519 signature verification
    │   ├── policy.rs            # TeePolicy trait: allowlist + measurement pinning
    │   ├── privacy.rs           # PrivacyProvider: log redaction (10 keys), SensitiveString, zeroize
    │   ├── epc.rs               # EPC memory detection (/proc/meminfo), 75% threshold routing
    │   └── cert.rs              # RA-TLS X.509 cert with attestation extension (feature: tls)
    │
    ├── verify/                  # Verify layer — client-side attestation SDK
    │   ├── mod.rs               # verify_report(), nonce/hash/measurement binding (constant-time)
    │   └── hw_verify.rs         # SevSnpVerifier (AMD KDS) + TdxVerifier (Intel PCS)
    │
    └── bin/
        └── a3s-power-verify.rs  # CLI for strict attestation report verification
```

## A3S Ecosystem

A3S Power is the inference engine of the A3S privacy-preserving AI platform. It runs inside a3s-box MicroVMs to provide hardware-isolated LLM inference.

```
┌──────────────────────────────────────────────────────────────────┐
│                         A3S Ecosystem                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  a3s-box MicroVM (AMD SEV-SNP / Intel TDX)               │    │
│  │  ┌────────────────────────────────────────────────────┐  │    │
│  │  │  a3s-power                                         │  │    │
│  │  │  OpenAI API ← Vsock/RA-TLS → host                 │  │    │
│  │  └────────────────────────────────────────────────────┘  │    │
│  │  Hardware-encrypted memory — host cannot read             │    │
│  └──────────────────────────────────────────────────────────┘    │
│       ▲ Vsock                                                     │
│       │                                                           │
│  ┌────┴─────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  a3s-gateway │  │  a3s-event   │  │  a3s-code              │  │
│  │  (API route) │  │  (event bus) │  │  (AI coding agent)     │  │
│  └──────────────┘  └──────────────┘  └────────────────────────┘  │
│                                                                   │
│  Client-side:                                                     │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  a3s-power verify SDK                                     │    │
│  │  Nonce binding · Model hash binding · HW signature check  │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

| Component | Relationship to Power |
|-----------|----------------------|
| **a3s-box** | Hosts Power inside TEE-enabled MicroVMs (AMD SEV-SNP / Intel TDX) |
| **a3s-code** | Uses Power as a local inference backend |
| **a3s-gateway** | Routes inference requests to Power instances |
| **a3s-event** | Distributes inference events across the platform |
| **verify SDK** | Client-side attestation verification (nonce, model hash, HW signature) |

## Roadmap

The acceptance-oriented embedded-runtime plan, including the TurboOCR-derived
batching workstream and its TEE release gates, lives in
[`ROADMAP.md`](ROADMAP.md). The checklist below also tracks the broader Power
server and inference product.

### Completed

- [x] Core inference engine (llama.cpp, chat templates, tool calling, structured output, thinking)
- [x] Model-neutral speculative control plane with picolm zero-weight baselines and native llama.cpp MTP capability negotiation, exact verification, rollback, and metrics
- [x] Path-free speculative benchmark harness with exact opted-in SSE usage/timings, registered GGUF digest checks, receipt verification, deterministic output parity, and controlled baseline/candidate comparison
- [ ] Capture and publish the real Qwen3.5-27B Q6_K CUDA result at or above 100 generated token/s through Power's streaming API
- [x] Bounded embedded model/device session pooling, cancellation-safe waiting queues, current-pressure-aware deterministic microbatch plans, canonical leading-axis tensor stacking/splitting, and digest-only receipt v4 scheduling evidence
- [x] Model-neutral embedded inference substrate — exact SafeTensors integrity, disjoint multi-root capacity aggregation, bounded continuous/ragged execution lifecycles on shared request admission, mmap-default and opt-in bounded positional/direct tensor reads, storage/host/device residency, native fixed-state/scratch-aware cache budgets with live pressure revalidation and unified-memory accounting, LFRU hot sets, atomic plans, hysteresis-bounded live hot-tier adaptation, batched expert unions, privacy-gated cross-layer route hints, event-driven current-layer staged batches with shared count/byte admission, attestation-bound accelerator residency declarations, canonical 16-device meshes with bounded peer copies and exact confidential GPU/NVSwitch claim-set binding, fused Candle batches with explicit actual-device/fallback identity, AES-256-GCM sealed warm-state envelopes with authenticated recovery and explicit TEE export authorization, digest-bound AB/BA lossless tuning evidence, canonical self-verifying hardware evidence bundles, complete/partial weighted replicas, optional artifact-pinned and canonical-byte-verified pure-Rust rANS representations, usage-ranked verified partial-mirror staging, integrity-read throughput weighting, private telemetry, canonical receipts, a standalone storage benchmark, and a manual Linux/Windows hosted-runner evidence workflow without an embedded Web listener
- [x] Pure Rust inference backend — `mistralrs` feature (default): GGUF inference via candle, no C++ dependency; ideal for TEE supply-chain auditing
- [x] OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`)
- [x] Content-addressed model storage with SHA-256
- [x] GPU auto-detection and acceleration (Metal, CUDA, multi-GPU)
- [x] KV cache reuse with prefix matching
- [x] Prometheus metrics and health endpoint
- [x] TEE refactoring — removed Ollama compatibility layer (~6,900 lines deleted)
- [x] A3S ACL-only configuration (removed TOML)
- [x] TEE awareness — `TeeProvider` trait, `DefaultTeeProvider` (SEV-SNP, TDX, Simulated)
- [x] Model integrity verification — SHA-256 at startup
- [x] Privacy protection — `PrivacyProvider` trait, log redaction
- [x] TEE status in `/health` endpoint

- [x] Attestation endpoint — `GET /v1/attestation` for clients to verify TEE
- [x] Memory zeroing — `zeroize` crate, `SensitiveString` auto-zeroize wrapper
- [x] Encrypted model loading — AES-256-GCM, `DecryptedModel` RAII secure wipe, key from file/env
- [x] PrivacyProvider integrated into inference chain — prompt/response wrapped in `SensitiveString`, `sanitize_log` applied at every log site
- [x] EncryptedModel integrated into autoload — `.enc` models auto-detected, decrypted, RAII cleanup on unload/eviction
- [x] TEE metrics — Prometheus counters for attestation reports, model decryptions, and log redactions
- [x] Attestation nonce — `?nonce=<hex>` binds client nonce into `report_data` to prevent replay attacks
- [x] RA-TLS transport — `tls` feature: self-signed ECDSA P-256 cert; `ra_tls = true` embeds JSON attestation report as custom X.509 extension (OID 1.3.6.1.4.1.56560.1.1); TLS server spawned in parallel with plain HTTP
- [x] Vsock transport — `vsock` feature (Linux only): AF_VSOCK server for a3s-box MicroVM guest-host HTTP communication; uses same axum router as TCP; no network config required inside the VM
- [x] SEV-SNP ioctl — real `/dev/sev-guest` ioctl (`SNP_GET_REPORT`) for hardware attestation reports; extracts `report_data` (64 bytes) and `measurement` (48 bytes) from firmware response; full raw report included for client-side verification
- [x] TDX ioctl — real `/dev/tdx-guest` ioctl (`TDX_CMD_GET_REPORT0`) for hardware attestation reports; extracts `reportdata` (64 bytes) and `mrtd` (48 bytes) from TDREPORT; supports both `/dev/tdx-guest` and `/dev/tdx_guest` device paths
- [x] KeyProvider trait — `StaticKeyProvider` (wraps file/env key source) + `RotatingKeyProvider` (multiple keys, zero-downtime rotation via `rotate_key()`); initialized on server startup; `AppState.key_provider` field
- [x] Deep log redaction — `PrivacyProvider` covers 10 sensitive JSON keys; `sanitize_error()` strips prompt fragments from error messages
- [x] Token metric suppression — `suppress_token_metrics` config rounds token counts to nearest 10 to prevent side-channel inference
- [x] In-memory encrypted-model backend loading — `in_memory_decrypt` decrypts into `MemoryDecryptedModel` locked RAM and loads GGUF plaintext through `picolm`; unsupported backends fail closed before load
- [x] Rate limiting — token-bucket middleware (`rate_limit_rps`) + concurrency cap (`max_concurrent_requests`) on `/v1/*`; returns `429` with OpenAI-style error
- [x] Model/runtime/GPU-attestation binding — `AttestationClaimsV2` + `sha256(canonical_claims_v2)` in CPU TEE `report_data`; `GET /v1/attestation?model=<name>` re-hashes the current local model artifact, including deterministic directory manifests and encrypted plaintext/ciphertext claims, and fails on missing or stale hashes; model-bound claims include applied chat-template digests plus canonical GPU execution/offload digests; `gpu-confidential` mode binds NVIDIA GPU CC evidence, NRAS verdict digests, and structured NVIDIA device identity/freshness claims from live `nvattest-cli` collection or direct `nras-rest` attestation and requires a 32-byte nonce
- [x] Embedding model support — `ModelFormat::HuggingFace` variant; `MistralRsBackend` loads HF embedding models via `EmbeddingModelBuilder` with local path; `POST /v1/embeddings` fully functional; register with `format=huggingface`
- [x] SafeTensors inference — `ModelFormat::SafeTensors` variant; `MistralRsBackend` loads local safetensors chat models via `TextModelBuilder` with ISQ on-load quantization; ISQ type configurable via `default_parameters.isq` (Q4_0, Q4K, Q6K, Q8_0, HQQ4, HQQ8, etc.); omitted ISQ defaults to Q8_0, while explicit invalid ISQ values fail closed; register with `format=safetensors`
- [x] Client attestation verification SDK — `verify` module with `verify_report()`, `verify_report_strict()`, `VerificationPolicy`, `ExpectedGpuEvidence`, `ExpectedGpuDevices`, `ExpectedReceipt`, `verify_nonce_binding()`, `verify_model_hash_binding()`, `verify_claims_gpu_evidence_binding()`, `verify_claims_expected_gpu_evidence()`, `verify_claims_gpu_device_claims()`, `verify_claims_expected_gpu_devices()`, `verify_claims_runtime_policy_binding()`, `verify_receipt_well_formed()`, `verify_receipt_policy()`, `verify_receipt_matches_chat_request()`, `verify_receipt_matches_completion_request()`, `verify_receipt_against_attestation()`, `verify_receipt_digest_hex()`, `verify_receipt_effective_prompt_digest_hex()`, and `verify_measurement()`; `HardwareVerifier` trait for pluggable hardware signature verification; strict verification requires hardware signatures and an expected launch measurement; `VerificationPolicy::gpu_confidential()` and `a3s-power-verify --gpu-confidential` bundle production NVIDIA GPU confidential-computing checks and require a 32-byte nonce, top-level GPU evidence nonce, `--gpu-verdict-digest`, GPU provider/format/count, exact GPU/NVSwitch topology, claims schema version, and identity/version pins; `a3s-power-verify` defaults to strict mode, requires `--expected-measurement`, requires `--allow-offline` to skip hardware signatures/measurement pinning, supports hardware certificate cache TTL tuning with `--hw-cert-cache-ttl-secs`, GPU provider/format/count, GPU execution digest, exact GPU/NVSwitch count pins, GPU/NVSwitch claims version pins, and device identity pins including UEID/OEM ID plus `--receipt-file` / `--receipt-digest` / `--receipt-model` / `--receipt-request-type` / `--receipt-chat-request-file` / `--receipt-completion-request-file` / `--receipt-input-digest` / receipt decoding, stream-options, and output-policy digest pins / `--effective-prompt-digest` for attestation-to-receipt verification, and requires `--nonce` when GPU evidence, device-claim, or identity pinning is used
- [x] Graceful shutdown — SIGTERM + Ctrl-C handled via `shutdown_signal()`; unloads all models (triggers RAII zeroize of decrypted weights); flushes audit log via `AuditLogger::flush()` before exit; `AsyncJsonLinesAuditLogger` flush uses oneshot channel to wait for background writer to drain
- [x] Remote model hub pull — `hf` feature: `POST /v1/models/pull` downloads GGUF models from ModelScope or HuggingFace Hub; supports `owner/repo:Q4_K_M` (resolves filename via hub API) and `owner/repo/file.gguf` (direct); streams SSE progress events (`resuming`, `downloading`, `verifying`, `success`); resume interrupted downloads via HTTP Range requests (deterministic partial filename = SHA-256 of the canonical URL); hub/API URLs percent-encode repo, filename, and query components while preserving intended subdirectories; source-specific token auth for private/gated models via `token` request field, `MODELSCOPE_TOKEN`/`HF_TOKEN`, or `A3S_POWER_HUB_TOKEN`; stores in content-addressed blob store; SHA-256 verified; `force` flag for re-download
- [x] Pull concurrent control — `Mutex<HashSet>` in `AppState` deduplicates concurrent pulls of the same model; returns `409 Conflict` if a pull is already in progress
- [x] Pull progress persistence — JSON state files in `~/.a3s/power/pulls/`; `GET /v1/models/pull/:name/status` returns `{status, completed, total, error}` and accepts URL-encoded model names; survives server restarts; throttled writes (every 5%) to minimize disk I/O
- [x] True token-by-token streaming — `stream_chat_request` replaces non-streaming path; each `Response::Chunk` forwarded immediately via mpsc channel; `Response::Done` sets `finish_reason`
- [x] Request-level inference receipts — `/v1/chat/completions` and `/v1/completions` return v2 `attestation_receipt` plus `attestation_receipt_sha256`; receipts include model runtime chat-template/GPU execution policy claims, request decoding/output policy digests, and stream-options digests; streaming responses emit the receipt in a final SSE event before `[DONE]`
- [x] Effective prompt digest coverage for deterministic chat paths — llama.cpp and picolm text-only chat return local rendered-prompt digests; mistralrs text chat returns a domain-separated prompt-token-ID digest; proxy backends can include an upstream-declared digest through the opt-in `/v1/chat/effective-prompt-digest` contract
- [ ] Effective prompt digest coverage for remaining opaque renderers — llama.cpp, picolm, and mistralrs vision/multimodal paths must either expose exact prompt representations or continue leaving `effective_prompt` absent
- [x] Vision/multimodal inference — `ModelFormat::Vision` variant; `MistralRsBackend` loads vision models via `VisionModelBuilder` with ISQ; base64 images accepted via `images` field or OpenAI `image_url` content parts; decoded with `image` + `base64` crates
- [x] picolm backend — pure Rust layer-streaming GGUF inference (`picolm` feature); real transformer forward pass (multi-head/GQA attention, SwiGLU/GeGLU FFN, RoPE, RMSNorm); fused dequant+dot kernels (Q4_K, Q6_K, Q8_0); rayon parallel matmul; FP16 KV cache; pre-computed RoPE tables; tensor cache (zero HashMap lookups); pre-allocated buffers (zero heap allocation in hot path); true O(layer_size) peak RAM via `madvise(MADV_DONTNEED)` page release; BPE tokenizer with ChatML template; 14+ tok/s decode on Apple Silicon; ~4,500 lines of pure Rust; no C/C++ inference backend
- [x] picolm features — batch prefill (faster time-to-first-token); speculative decoding via prompt-lookup; tool/function calling (OpenAI-compatible `tool_calls`); grammar-constrained structured output (JSON Schema enforcement); repeat/frequency/presence penalty
- [x] picolm SIMD — NEON (aarch64): softmax, RMSNorm, SiLU, add_residual, Q4_K nibble extraction; AVX2 (x86_64): Q4_K, Q6_K vec_dot kernels
- [x] picolm performance — fused f16 KV attention (`k_dot`/`v_accumulate` skip intermediate f32 buffer); zero-alloc sampler (pre-allocated probs/indices in ForwardBuffers); zero-alloc repeat penalty (stack-based `[(u32,u32); 64]` dedup); Q4_K NEON register-based nibble extraction; decode profiling instrumentation (per-stage timing breakdown); 900+ tests across current validation profiles
- [x] EPC memory detection — `tee::epc` module reads `/proc/meminfo`; `BackendRegistry::find_for_tee()` auto-routes to picolm when model exceeds 75% of available EPC
- [x] `LayerStreamingDecryptedModel` primitive — chunked access to AES-256-GCM encrypted models; each returned chunk is `Zeroizing<Vec<u8>>`; `streaming_decrypt = true` passes this plaintext source to supporting backends and fails closed for unsupported backends
- [x] End-to-end chunked encrypted-model backend loading — `picolm` GGUF consumes `LayerStreamingDecryptedModel` plaintext for streaming decrypt mode instead of loading the encrypted path; the current AES-GCM artifact format is still non-seekable, so full plaintext remains locked in RAM while the handle is live
- [x] `tee-minimal` feature profile — `picolm` + `tls` + `vsock`; smallest auditable TEE build (~1,220 dep tree lines vs ~2,000 for default); no mistralrs/candle and no C++ inference engine; TLS/crypto still brings native `ring`/`aws-lc-sys` build dependencies
- [x] Supply-chain audit document — `docs/supply-chain.md`; per-profile dependency listing, audit status table, threat model

## CI/CD

Automated via GitHub Actions:

- **CI** (`.github/workflows/ci.yml`): Format check, Clippy (6 feature combos across all targets), unit tests, cross-build (4 platforms)
- **Release** (`.github/workflows/release.yml`): CI gate → 4-platform build → GitHub Release → crates.io → Homebrew formula update

### Supported Platforms

| Target | OS | Cross |
|--------|----|-------|
| `aarch64-apple-darwin` | macOS (Apple Silicon) | Native |
| `x86_64-apple-darwin` | macOS (Intel) | Native |
| `aarch64-unknown-linux-gnu` | Linux (ARM64) | `cross` |
| `x86_64-unknown-linux-gnu` | Linux (x86_64) | Native |

### Release Process

```bash
# 1. Bump version in Cargo.toml
# 2. Commit and tag
git add -A && git commit -m "chore: release v0.x.y"
git tag v0.x.y && git push origin main --tags
# 3. GitHub Actions builds, publishes to crates.io, creates GitHub Release, updates Homebrew formula
```

## Community

Join us on [Discord](https://discord.gg/XVg6Hu6H) for questions, discussions, and updates.

## License

MIT
