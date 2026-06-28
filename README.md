# A3S Power

<p align="center">
  <strong>The Only LLM Inference Server You Don't Have to Trust</strong>
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml"><img src="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/release.yml"><img src="https://github.com/A3S-Lab/Power/actions/workflows/release.yml/badge.svg" alt="Release"></a>
  <a href="https://crates.io/crates/a3s-power"><img src="https://img.shields.io/crates/v/a3s-power.svg" alt="crates.io"></a>
  <a href="https://github.com/A3S-Lab/Power/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
</p>

<p align="center">
  <em>Cryptographically prove that a specific model runs unmodified inside hardware-encrypted memory — without trusting the infrastructure operator.</em>
</p>

<p align="center">
  <a href="#the-problem">The Problem</a> •
  <a href="#how-power-solves-it">How Power Solves It</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#layer-streaming-inference-picolm--how-it-works">Layer-Streaming</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#development">Development</a>
</p>

---

## The Problem

Every LLM inference server — Ollama, vLLM, llama.cpp, TGI, LocalAI — was designed for a world where you **trust the machine**. You send your prompts to a server and hope the operator doesn't look at them. That's a policy promise, not a technical guarantee.

For healthcare (HIPAA), finance (SOX/GLBA), government (classified data), and any multi-tenant AI deployment where the infrastructure operator is a different party than the data owner — "we promise not to look" is not enough.

## How Power Solves It

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

The difference: every other inference server asks you to **trust**. Power lets you **verify**.

## Why Not Just Use Ollama / vLLM / TGI?

| Capability | Ollama | vLLM | TGI | Power |
|---|:---:|:---:|:---:|:---:|
| OpenAI-compatible API | ✅ | ✅ | ✅ | ✅ |
| GPU acceleration | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| TEE hardware isolation (SEV-SNP / TDX) | ❌ | ❌ | ❌ | ✅ |
| Remote attestation (hardware-signed proof) | ❌ | ❌ | ❌ | ✅ |
| Model-attestation binding (prove which model runs) | ❌ | ❌ | ❌ | ✅ |
| RA-TLS (attestation in TLS handshake) | ❌ | ❌ | ❌ | ✅ |
| Encrypted model loading (AES-256-GCM, 3 modes) | ❌ | ❌ | ❌ | ✅ |
| Deep log redaction (10 keys + error sanitization) | ❌ | ❌ | ❌ | ✅ |
| Memory zeroing (zeroize on drop) | ❌ | ❌ | ❌ | ✅ |
| Client-side verification SDK | ❌ | ❌ | ❌ | ✅ |
| Hardware signature verification (AMD KDS / Intel PCS) | ❌ | ❌ | ❌ | ✅ |
| Layer-streaming for memory-constrained TEE | ❌ | ❌ | ❌ | ✅ |
| Pure Rust inference (fully auditable, no C++) | ❌ | ❌ | ❌ | ✅ |

The bottom half of this table is Power's moat. No other inference server has a threat model. They all assume you trust the machine.

## Overview

**A3S Power** is a privacy-preserving LLM inference server designed to run inside Trusted Execution Environments (TEE). It provides an OpenAI-compatible API for chat completions, text completions, and embeddings — with hardware-enforced memory encryption, model integrity verification, and automatic log redaction.

Power is built to run inside [a3s-box](https://github.com/AI45Lab/Box) MicroVMs with AMD SEV-SNP or Intel TDX, ensuring that inference data (prompts, responses, model weights) never leaves the encrypted enclave.

## Features

### Trust & Verification (The Moat)

These features exist in no other LLM inference server:

- **TEE-Aware Runtime**: Auto-detects AMD SEV-SNP (`/dev/sev-guest`) and Intel TDX (`/dev/tdx_guest`) at startup; simulated mode for development (`A3S_TEE_SIMULATE=1`)
- **Remote Attestation**: Real hardware ioctl — AMD `SNP_GET_REPORT` and Intel `TDX_CMD_GET_REPORT0` — generates firmware-signed proof that inference runs in a genuine TEE; full raw reports included for client verification
- **Model-Attestation Binding**: `GET /v1/attestation?model=<name>` embeds the model's SHA-256 hash into `report_data` alongside the nonce — layout `[nonce(32)][model_sha256(32)]` — cryptographically tying the attestation to the specific model being served; you can't swap the model without invalidating the attestation
- **RA-TLS Transport**: TLS certificate embeds the attestation report as a custom X.509 extension (OID `1.3.6.1.4.1.56560.1.1`) — clients verify the TEE during the TLS handshake itself, no separate API call needed
- **Hardware Signature Verification**: Client-side SDK verifies attestation signatures against AMD KDS (ECDSA P-384) and Intel PCS (ECDSA P-256) certificate chains — closes the loop from hardware root of trust to client
- **Client Verification CLI**: `a3s-power-verify` independently verifies nonce binding, model hash binding, platform measurement, and hardware signatures from any running Power server
- **Encrypted Model Loading**: AES-256-GCM with 3 modes — `DecryptedModel` (temp file, zero-overwrite on drop), `MemoryDecryptedModel` (mlock-pinned RAM, never touches disk), `LayerStreamingDecryptedModel` (chunk-by-chunk for picolm); the infrastructure operator cannot read model weights from disk or swap
- **KeyProvider Trait**: Abstract key loading for HSM integration; `StaticKeyProvider` (file/env) + `RotatingKeyProvider` (zero-downtime rotation)
- **Deep Log Redaction**: Strips inference content from all log output — 10 sensitive JSON keys (`content`, `prompt`, `text`, `arguments`, `input`, `delta`, `system`, `message`, `query`, `instruction`); `sanitize_error()` strips prompt fragments from error messages; `suppress_token_metrics` rounds token counts to nearest 10 to prevent side-channel inference
- **Memory Zeroing**: `SensitiveString` wrapper auto-zeroizes on drop; all inference buffers cleared via `zeroize` crate — the operator cannot recover prompts or responses from memory dumps
- **Model Integrity**: SHA-256 hash verification at startup + Ed25519 publisher signatures; fails fast on tampering
- **picolm Layer-Streaming**: Pure Rust GGUF inference with true O(layer_size) peak RAM via `madvise(DONTNEED)` page release after each layer. Real transformer ops: multi-head/GQA attention, SwiGLU/GeGLU FFN, RoPE, RMSNorm. FP16 KV cache with fused f16 dot/accumulate (no intermediate buffer). Fused dequant+dot kernels. NEON SIMD (aarch64) + AVX2 (x86_64). Rayon parallel matmul. Pre-computed RoPE tables. Batch prefill, tool calling, grammar-constrained output. Selectable speculative-decoding modes (`spec_mode`: `off` / `prompt-lookup` / DSpark-like `ngram-context`) with **batched layer-streaming verify** — a draft block is verified in one weight-streaming pass instead of one pass per token — adaptive draft length, and lossless rejection-sampling acceptance (output matches plain decoding for the same seed). Zero-alloc hot path. 14+ tok/s decode on Apple Silicon. Enables 7B+ models inside 512MB TEE EPC. No C/C++ inference backend, ~4,500 lines of fully auditable Rust.
- **Pure Rust Inference Path**: Default backend via `mistralrs` (candle) — no C++ inference engine in the trusted computing base; the `tee-minimal` build (~1,220 dep tree lines) is the smallest auditable LLM inference stack that exists

### Inference Engine

Full-featured LLM inference, competitive with any standalone server:

- **OpenAI-Compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` — works with any OpenAI SDK
- **True Token-by-Token Streaming**: Per-token SSE delivery via `stream_chat_request`
- **Multiple Backends**: mistralrs (pure Rust, default), llama.cpp (C++ bindings, optional), picolm (TEE layer-streaming, optional), proxy (forwards to an upstream OpenAI-compatible server — vLLM/TGI/SGLang/OpenAI — so Power can front an existing accelerated engine)
- **Model Formats**: GGUF, SafeTensors (ISQ quantization), Vision/Multimodal (LLaVA, Phi-3-Vision), HuggingFace Embeddings (Qwen3, GTE, NomicBert)
- **GPU Acceleration**: Auto-detection of Apple Metal and NVIDIA CUDA; configurable layer offloading, multi-GPU support
- **Tool/Function Calling**: Structured tool definitions with XML, Mistral, and JSON output parsing
- **JSON Schema Structured Output**: Constrain model output via JSON Schema → GBNF grammar conversion
- **Thinking & Reasoning**: Streaming `<think>` block parser for DeepSeek-R1, QwQ reasoning models
- **Chat Template Engine**: Jinja2-compatible rendering via `minijinja` (Llama 3, ChatML, Phi, Gemma, custom)
- **KV Cache Reuse**: Prefix matching across multi-turn requests for conversation speedup
- **HuggingFace Hub Pull**: `POST /v1/models/pull` with SSE progress, Range resume, concurrent dedup, HF token auth

### Operations

- **Content-Addressed Storage**: Model blobs stored by SHA-256 hash with automatic deduplication
- **Automatic Model Lifecycle**: LRU eviction, configurable keep-alive, background reaper for idle models
- **Rate Limiting & Admission Control**: Per-second token-bucket on `/v1/*` returns `429` with an OpenAI-style error; concurrency (`max_concurrent_requests`) uses vLLM-style backpressure — excess requests **queue** for an admission permit (held across the streamed body) rather than being rejected, with a `power_requests_waiting` gauge
- **Prometheus Metrics**: 16 metric groups — HTTP, inference, TTFT, GPU, TEE attestations, model decryptions, log redactions
- **Audit Logging**: JSONL / Encrypted / Async / Noop; flushed on graceful shutdown
- **Vsock Transport**: AF_VSOCK for a3s-box MicroVM guest-host communication (Linux only)
- **HCL Configuration**: HashiCorp Configuration Language for all settings

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
│  │  Infrastructure: config.rs (HCL) · dirs.rs · error.rs (14 var.)  │  │
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
PowerConfig (HCL)                     TeePolicy: allowlist + measurement pin
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
│  4. BackendRegistry.find_for_format(format) → Backend            │
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
                (3 modes: file/RAM/streaming), Ed25519 model signatures,
                key rotation, policy enforcement, log redaction (9 keys),
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
              │ Temp .dec  │   │ mlock-pinned │   │ Chunk-by-chunk      │
              │ file on    │   │ RAM buffer,  │   │ Zeroizing<Vec<u8>>  │
              │ disk, zero │   │ never touches│   │ per layer, for      │
              │ overwrite  │   │ disk, zeroize│   │ picolm streaming    │
              │ + delete   │   │ on drop      │   │ O(layer_size) peak  │
              │ on drop    │   │              │   │                     │
              └────────────┘   └──────────────┘   └─────────────────────┘
                  Any              Any                  picolm only
                backend          backend
```

### Backend Trait

Three backends are available, each feature-gated:

- **`mistralrs`** (default): Pure Rust inference via candle. GGUF, SafeTensors, HuggingFace, Vision formats. ISQ on-load quantization. No C++ inference toolchain. Ideal for TEE supply-chain auditing.
- **`llamacpp`** (optional): C++ llama.cpp via `llama-cpp-2` bindings. GGUF only. Session KV cache with prefix matching, LoRA adapters, MTMD multimodal, grammar constraints, mirostat sampling.
- **`picolm`** (optional): Pure Rust layer-streaming. GGUF only. Real transformer inference (multi-head/GQA attention, SwiGLU/GeGLU FFN, RoPE, RMSNorm). Peak RAM = O(layer_size) not O(model_size) via `madvise(DONTNEED)` page release. FP16 KV cache with fused f16 dot/accumulate. Fused dequant+dot kernels (Q4_K, Q6_K, Q8_0). NEON SIMD (aarch64) + AVX2 (x86_64). Rayon parallel matmul. Batch prefill, speculative decoding, tool calling, grammar-constrained output. 14+ tok/s decode on Apple Silicon. Enables 7B+ models in 512MB TEE EPC. No C/C++ inference backend — ~4,500 lines of fully auditable Rust.

The `BackendRegistry` selects backends by priority and model format. In TEE environments, `find_for_tee()` auto-routes to picolm when the model exceeds 75% of available EPC memory.

Without any backend feature enabled, Power can manage models but returns "backend not available" for inference.

```rust
#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn supports(&self, format: &ModelFormat) -> bool;
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

# Binary at target/release/a3s-power
```

### Homebrew (macOS)

```bash
brew tap a3s-lab/tap https://github.com/A3S-Lab/homebrew-tap
brew install a3s-power
```

## Configuration

Configuration is read from `~/.a3s/power/config.hcl` (HCL format):

```hcl
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
keep_alive = "5m"

# TEE privacy protection
tee_mode = true
redact_logs = true

# Model integrity verification (checked at startup when tee_mode = true)
model_hashes = {
  "llama3.2:3b" = "sha256:abc123..."
  "qwen2.5:7b"  = "sha256:def456..."
}

# GPU acceleration
gpu {
  gpu_layers = -1    # -1 = offload all layers, 0 = CPU only
  main_gpu   = 0
}
```

### Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `127.0.0.1` | HTTP server bind address |
| `port` | `11434` | HTTP server port |
| `data_dir` | `~/.a3s/power` | Base directory for model storage |
| `max_loaded_models` | `1` | Maximum models loaded concurrently |
| `keep_alive` | `"5m"` | Auto-unload idle models (`"0"` = immediate, `"-1"` = never) |
| `use_mlock` | `false` | Lock model weights in memory (prevent swapping) |
| `num_thread` | auto | Thread count for inference |
| `flash_attention` | `false` | Enable flash attention |
| `num_parallel` | `1` | Concurrent inference slots |
| `tee_mode` | `false` | Enable TEE: attestation, integrity checks, log redaction |
| `redact_logs` | `false` | Redact inference content from logs |
| `model_hashes` | `{}` | Expected SHA-256 hashes for model verification |
| `model_signing_key` | `null` | Ed25519 public key (hex) for verifying model `.sig` signatures |
| `gpu.gpu_layers` | `0` | GPU layer offloading (`-1` = all) |
| `gpu.main_gpu` | `0` | Primary GPU index |
| `model_key_source` | `null` | Decryption key for `.enc` model files: `{ file = "/path/to/key.hex" }` or `{ env = "MY_KEY_VAR" }` |
| `key_provider` | `"static"` | Key provider type: `"static"` (uses `model_key_source`) or `"rotating"` (uses `key_rotation_sources`) |
| `key_rotation_sources` | `[]` | For rotating provider: list of key sources in rotation order |
| `in_memory_decrypt` | `false` | Decrypt `.enc` models entirely in RAM with `mlock` (never writes plaintext to disk) |
| `suppress_token_metrics` | `false` | Round token counts in responses to nearest 10 (prevents exact token-count side-channel) |
| `rate_limit_rps` | `0` | Max requests per second for `/v1/*` endpoints (`0` = unlimited) |
| `max_concurrent_requests` | `0` | Max concurrent in-flight inference requests; excess **queue** for an admission permit held across the streamed response (`0` = unlimited) |
| `proxy_upstreams` | `{}` | Map of model name → upstream base URL to proxy to an OpenAI-compatible server (vLLM/TGI/SGLang/OpenAI), e.g. `{ "llama-70b" = "http://vllm:8000" }`. Proxied inference runs on the upstream, outside any TEE |
| `tls_port` | `null` | TLS server port; when set, a TLS server starts in parallel (`tls` feature required) |
| `ra_tls` | `false` | Embed TEE attestation in TLS cert (RA-TLS); requires `tls_port` + `tee_mode` |
| `vsock_port` | `null` | Vsock port for guest-host communication (`vsock` feature, Linux only) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `A3S_POWER_HOME` | Base directory for all Power data (default: `~/.a3s/power`) |
| `A3S_POWER_HOST` | Server bind address |
| `A3S_POWER_PORT` | Server port |
| `A3S_POWER_DATA_DIR` | Model storage directory |
| `A3S_POWER_MAX_MODELS` | Max concurrent loaded models |
| `A3S_POWER_KEEP_ALIVE` | Default keep-alive duration |
| `A3S_POWER_GPU_LAYERS` | GPU layer offloading |
| `A3S_POWER_TEE_MODE` | Enable TEE mode (`"1"` or `"true"`) |
| `A3S_POWER_REDACT_LOGS` | Enable log redaction (`"1"` or `"true"`) |
| `A3S_POWER_TLS_PORT` | TLS server port (`tls` feature required) |
| `A3S_POWER_RA_TLS` | Enable RA-TLS attestation embedding (`"1"` or `"true"`) |
| `A3S_POWER_VSOCK_PORT` | Vsock port (`vsock` feature, Linux only) |
| `A3S_TEE_SIMULATE` | Simulate TEE environment for development (`"1"`) |

## TEE Privacy Protection

### Model Integrity Verification

When `tee_mode = true` and `model_hashes` is configured, Power verifies every model file's SHA-256 hash at startup. If any model fails verification, the server refuses to start.

```hcl
tee_mode = true
model_hashes = {
  "llama3.2:3b" = "sha256:a1b2c3d4e5f6..."
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
| Simulated | `A3S_TEE_SIMULATE=1` | Development/testing mode |
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
| `POST` | `/v1/models` | Register a local model file (`name`, `path` body fields) |
| `DELETE` | `/v1/models/:name` | Unload and deregister a model |
| `POST` | `/v1/models/pull` | Pull a GGUF model from HuggingFace Hub (`name`, `force` body fields); streams SSE progress events; requires `hf` feature; concurrent pulls of the same model are deduplicated |
| `GET` | `/v1/models/pull/:name/status` | Get persisted pull progress for a model (`status`, `completed`, `total`, `error`); URL-encode names that contain `/` or `:` |
| `GET` | `/v1/attestation` | TEE attestation report (returns 503 if TEE not enabled); optional `?nonce=<hex>` binds client nonce; optional `?model=<name>` binds model SHA-256 into `report_data` |

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
        "type": "object",
        "properties": {
          "colors": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "hex": {"type": "string"}
              }
            }
          }
        }
      }
    }
  }'
```

#### List Models

```bash
curl http://localhost:11434/v1/models
```

#### Pull a Model from HuggingFace Hub

Requires the `hf` feature (`cargo build --features hf`). Streams SSE progress:

```bash
# By quantization tag (resolves filename via HF API)
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"}'

# By exact filename
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf"}'

# Private/gated model with HF token
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "meta-llama/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "token": "hf_..."}'

# Force re-download
curl -N http://localhost:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "force": true}'

# Check persisted progress for a pull; URL-encode names containing "/" or ":"
curl http://localhost:11434/v1/models/pull/bartowski%2FLlama-3.2-3B-Instruct-GGUF%3AQ4_K_M/status
```

SSE response stream:
```
data: {"status":"resuming","offset":104857600,"total":2147483648}   ← if resuming
data: {"status":"downloading","completed":209715200,"total":2147483648}
data: {"status":"verifying"}
data: {"status":"success","id":"bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M","object":"model","created":1234567890}
```

Interrupted downloads resume automatically on retry — the partial file is identified by a SHA-256 of the download URL and picked up via HTTP `Range` requests. Set `HF_TOKEN` env var as an alternative to passing `token` in the request body.

#### Health Check (with TEE status)

```bash
curl http://localhost:11434/health
```

## Model Storage

Models are stored in `~/.a3s/power/` (override with `$A3S_POWER_HOME`):

```
~/.a3s/power/
├── config.hcl               # HCL configuration
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
| `llamacpp` | ❌ disabled | llama.cpp inference backend via `llama-cpp-2`. Requires C++ compiler + CMake. Full-featured (KV cache, LoRA, grammar, mirostat). |
| `picolm` | ❌ disabled | Pure Rust layer-streaming GGUF inference. Real transformer ops (multi-head attention, SwiGLU FFN, RoPE, RMSNorm). Peak RAM = O(layer_size) not O(model_size) via `madvise(DONTNEED)`. FP16 KV cache with fused f16 dot/accumulate. Fused dequant+dot kernels. NEON SIMD (aarch64) + AVX2 (x86_64). Batch prefill, speculative decoding, tool calling, grammar-constrained output. 14+ tok/s decode on Apple Silicon. Enables 7B+ models in 512MB TEE EPC. No C/C++ inference backend. ~4,500 lines of pure Rust. |
| `hf` | ❌ disabled | HuggingFace Hub model pull (`POST /v1/models/pull`). Range resume, SSE progress, HF_TOKEN auth. |
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
- **Encrypted model loading**: AES-256-GCM with `in_memory_decrypt` or `streaming_decrypt`
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

For encrypted models (`.enc`), `LayerStreamingDecryptedModel` decrypts one chunk at a time. Each chunk is wrapped in `Zeroizing<Vec<u8>>` — automatically zeroed when dropped. This means:

- Plaintext weights for only one layer exist in RAM at any moment
- Each chunk is cryptographically erased after use
- The infrastructure operator cannot recover weights from memory dumps

```
Encrypted layer-streaming:
┌─────────────────────────────────────────────────────┐
│  model.gguf.enc (AES-256-GCM encrypted on disk)      │
│                                                       │
│  for each layer:                                      │
│    chunk = decrypt_chunk(key, layer_offset, layer_len)│
│    chunk: Zeroizing<Vec<u8>>  ← auto-zeroed on drop   │
│    forward_pass(hidden_state, &chunk)                  │
│    // chunk dropped → memory zeroed immediately        │
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

```hcl
# config.hcl — TEE deployment with layer-streaming
tee_mode        = true
redact_logs     = true

# For encrypted models: decrypt one layer at a time (requires picolm feature)
streaming_decrypt = true

# Or: decrypt full model into mlock RAM (compatible with all backends)
# in_memory_decrypt = true
```

### Supply-chain audit

See [`docs/supply-chain.md`](docs/supply-chain.md) for:
- Full dependency listing per feature profile
- Audit status for each crate in the `tee-minimal` inference path
- Security properties of `LayerStreamingDecryptedModel`
- How to reproduce dependency counts and audit unsafe blocks

### Building with RA-TLS

```bash
# Build with TLS support
cargo build --features tls

# Test TLS cert generation
cargo test --features tls -p a3s-power tee::cert
```

To enable RA-TLS, set `tls_port` and `ra_tls = true` alongside `tee_mode = true`:

```hcl
tee_mode = true
tls_port = 11443
ra_tls   = true
```

At startup, the TLS server binds on the configured port with a fresh self-signed ECDSA P-256 certificate. When `ra_tls = true` and a TEE provider is active, the certificate includes the attestation report as OID extension `1.3.6.1.4.1.56560.1.1`. Clients can extract and verify this extension to confirm they are communicating with a genuine TEE before trusting inference output.

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
    ├── main.rs                  # Entry point: load HCL config → server::start()
    ├── lib.rs                   # Module declarations
    ├── config.rs                # PowerConfig (HCL deserialization + env overrides)
    ├── dirs.rs                  # Platform paths (~/.a3s/power/{manifests,blobs,pulls})
    ├── error.rs                 # PowerError enum (14 variants) + HTTP status mapping
    │
    ├── api/                     # API layer — OpenAI-compatible HTTP handlers
    │   ├── mod.rs               # Shared utilities, timestamp helpers
    │   ├── types.rs             # OpenAI request/response types (chat, completion, embedding)
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
    │   ├── mod.rs               # Backend trait (8 methods) + BackendRegistry (priority, TEE routing)
    │   ├── types.rs             # ChatRequest, ChatResponseChunk, EmbeddingRequest, Tool, ToolCall
    │   ├── mistralrs_backend.rs # Pure Rust: GGUF/SafeTensors/HF/Vision, ISQ (feature: mistralrs) ★
    │   ├── llamacpp.rs          # C++ bindings: KV cache, LoRA, MTMD vision, grammar (feature: llamacpp)
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
    │   ├── privacy.rs           # PrivacyProvider: log redaction (9 keys), SensitiveString, zeroize
    │   ├── epc.rs               # EPC memory detection (/proc/meminfo), 75% threshold routing
    │   └── cert.rs              # RA-TLS X.509 cert with attestation extension (feature: tls)
    │
    ├── verify/                  # Verify layer — client-side attestation SDK
    │   ├── mod.rs               # verify_report(), nonce/hash/measurement binding (constant-time)
    │   └── hw_verify.rs         # SevSnpVerifier (AMD KDS) + TdxVerifier (Intel PCS)
    │
    └── bin/
        └── a3s-power-verify.rs  # CLI for offline attestation report verification
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

### Completed

- [x] Core inference engine (llama.cpp, chat templates, tool calling, structured output, thinking)
- [x] Pure Rust inference backend — `mistralrs` feature (default): GGUF inference via candle, no C++ dependency; ideal for TEE supply-chain auditing
- [x] OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`)
- [x] Content-addressed model storage with SHA-256
- [x] GPU auto-detection and acceleration (Metal, CUDA, multi-GPU)
- [x] KV cache reuse with prefix matching
- [x] Prometheus metrics and health endpoint
- [x] TEE refactoring — removed Ollama compatibility layer (~6,900 lines deleted)
- [x] HCL-only configuration (removed TOML)
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
- [x] In-memory decryption config — `in_memory_decrypt` field; `MemoryDecryptedModel` decrypts into `mlock`-pinned RAM, never writes plaintext to disk
- [x] Rate limiting — token-bucket middleware (`rate_limit_rps`) + concurrency cap (`max_concurrent_requests`) on `/v1/*`; returns `429` with OpenAI-style error
- [x] Model-attestation binding — `build_report_data(nonce, model_hash)` layout `[nonce(32)][sha256(32)]`; `TeeProvider::attestation_report_with_model()` default impl; `GET /v1/attestation?model=<name>` ties attestation to specific model
- [x] Embedding model support — `ModelFormat::HuggingFace` variant; `MistralRsBackend` loads HF embedding models via `EmbeddingModelBuilder` with local path; `POST /v1/embeddings` fully functional; register with `format=huggingface`
- [x] SafeTensors inference — `ModelFormat::SafeTensors` variant; `MistralRsBackend` loads local safetensors chat models via `TextModelBuilder` with ISQ on-load quantization; ISQ type configurable via `default_parameters.isq` (Q4_0, Q4K, Q6K, Q8_0, HQQ4, HQQ8, etc.); defaults to Q8_0; register with `format=safetensors`
- [x] Client attestation verification SDK — `verify` module with `verify_report()`, `verify_nonce_binding()`, `verify_model_hash_binding()`, `verify_measurement()`; `HardwareVerifier` trait for pluggable hardware signature verification; `a3s-power-verify` CLI binary
- [x] Graceful shutdown — SIGTERM + Ctrl-C handled via `shutdown_signal()`; unloads all models (triggers RAII zeroize of decrypted weights); flushes audit log via `AuditLogger::flush()` before exit; `AsyncJsonLinesAuditLogger` flush uses oneshot channel to wait for background writer to drain
- [x] HuggingFace Hub model pull — `hf` feature: `POST /v1/models/pull` downloads GGUF models from HuggingFace Hub; supports `owner/repo:Q4_K_M` (resolves filename via HF API) and `owner/repo/file.gguf` (direct); streams SSE progress events (`resuming`, `downloading`, `verifying`, `success`); resume interrupted downloads via HTTP Range requests (deterministic partial filename = SHA-256 of URL); HF token auth for private/gated models via `token` request field or `HF_TOKEN` env var; stores in content-addressed blob store; SHA-256 verified; `force` flag for re-download
- [x] Pull concurrent control — `Mutex<HashSet>` in `AppState` deduplicates concurrent pulls of the same model; returns `409 Conflict` if a pull is already in progress
- [x] Pull progress persistence — JSON state files in `~/.a3s/power/pulls/`; `GET /v1/models/pull/:name/status` returns `{status, completed, total, error}` and accepts URL-encoded model names; survives server restarts; throttled writes (every 5%) to minimize disk I/O
- [x] True token-by-token streaming — `stream_chat_request` replaces non-streaming path; each `Response::Chunk` forwarded immediately via mpsc channel; `Response::Done` sets `finish_reason`
- [x] Vision/multimodal inference — `ModelFormat::Vision` variant; `MistralRsBackend` loads vision models via `VisionModelBuilder` with ISQ; base64 images accepted via `images` field or OpenAI `image_url` content parts; decoded with `image` + `base64` crates
- [x] picolm backend — pure Rust layer-streaming GGUF inference (`picolm` feature); real transformer forward pass (multi-head/GQA attention, SwiGLU/GeGLU FFN, RoPE, RMSNorm); fused dequant+dot kernels (Q4_K, Q6_K, Q8_0); rayon parallel matmul; FP16 KV cache; pre-computed RoPE tables; tensor cache (zero HashMap lookups); pre-allocated buffers (zero heap allocation in hot path); true O(layer_size) peak RAM via `madvise(MADV_DONTNEED)` page release; BPE tokenizer with ChatML template; 14+ tok/s decode on Apple Silicon; ~4,500 lines of pure Rust; no C/C++ inference backend
- [x] picolm features — batch prefill (faster time-to-first-token); speculative decoding via prompt-lookup; tool/function calling (OpenAI-compatible `tool_calls`); grammar-constrained structured output (JSON Schema enforcement); repeat/frequency/presence penalty
- [x] picolm SIMD — NEON (aarch64): softmax, RMSNorm, SiLU, add_residual, Q4_K nibble extraction; AVX2 (x86_64): Q4_K, Q6_K vec_dot kernels
- [x] picolm performance — fused f16 KV attention (`k_dot`/`v_accumulate` skip intermediate f32 buffer); zero-alloc sampler (pre-allocated probs/indices in ForwardBuffers); zero-alloc repeat penalty (stack-based `[(u32,u32); 64]` dedup); Q4_K NEON register-based nibble extraction; decode profiling instrumentation (per-stage timing breakdown); 900+ tests across current validation profiles
- [x] EPC memory detection — `tee::epc` module reads `/proc/meminfo`; `BackendRegistry::find_for_tee()` auto-routes to picolm when model exceeds 75% of available EPC
- [x] `LayerStreamingDecryptedModel` — chunk-by-chunk access to AES-256-GCM encrypted models; each chunk returned as `Zeroizing<Vec<u8>>`, zeroized on drop; `streaming_decrypt` config field
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
