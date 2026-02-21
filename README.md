# A3S Power

<p align="center">
  <strong>Privacy-Preserving LLM Inference for TEE Environments</strong>
</p>

<p align="center">
  <em>Run large language models inside Trusted Execution Environments with hardware-enforced privacy, model integrity verification, and log redaction.</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#tee-privacy-protection">TEE Privacy</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#development">Development</a>
</p>

---

## Overview

**A3S Power** is a privacy-preserving LLM inference server designed to run inside Trusted Execution Environments (TEE). It provides an OpenAI-compatible API for chat completions, text completions, and embeddings — with hardware-enforced memory encryption, model integrity verification, and automatic log redaction.

Power is built to run inside [a3s-box](https://github.com/A3S-Lab/Box) MicroVMs with AMD SEV-SNP or Intel TDX, ensuring that inference data (prompts, responses, model weights) never leaves the encrypted enclave.

### How It Works

```
┌─────────────────────────────────────────────────────┐
│  a3s-box MicroVM (AMD SEV-SNP / Intel TDX)          │
│  ┌───────────────────────────────────────────────┐  │
│  │  a3s-power                                    │  │
│  │                                               │  │
│  │  1. Verify model integrity (SHA-256)          │  │
│  │  2. Generate remote attestation report        │  │
│  │  3. Serve inference via OpenAI API            │  │
│  │  4. Redact all inference content from logs    │  │
│  └───────────────────────────────────────────────┘  │
│  Hardware-encrypted memory — host cannot read        │
└─────────────────────────────────────────────────────┘
```

## Features

- **TEE-Aware Runtime**: Auto-detects AMD SEV-SNP (`/dev/sev-guest`) and Intel TDX (`/dev/tdx_guest`) at startup; simulated mode for development (`A3S_TEE_SIMULATE=1`)
- **Remote Attestation**: `TeeProvider` trait with `AttestationReport` generation — cryptographic proof that inference runs in a genuine TEE; AMD SEV-SNP uses real `/dev/sev-guest` ioctl (`SNP_GET_REPORT`), Intel TDX uses real `/dev/tdx-guest` ioctl (`TDX_CMD_GET_REPORT0`); full raw reports included for client verification
- **Model Integrity Verification**: SHA-256 hash verification of all model files at startup against configured expected hashes; fails fast on tampering
- **Deep Log Redaction**: `PrivacyProvider` trait strips inference content from all log output in TEE mode — covers 10 sensitive JSON keys (`"content"`, `"prompt"`, `"text"`, `"arguments"`, `"input"`, `"delta"`, `"system"`, `"message"`, `"query"`, `"instruction"`); `sanitize_error()` strips prompt fragments from error messages; `suppress_token_metrics` rounds token counts to nearest 10 to prevent side-channel inference
- **Memory Zeroing**: `SensitiveString` wrapper auto-zeroizes on drop; `zeroize_string()` / `zeroize_bytes()` utilities for clearing inference buffers via `zeroize` crate
- **Encrypted Model Loading**: AES-256-GCM encryption/decryption of model files; `DecryptedModel` RAII wrapper securely wipes temp files on drop; `MemoryDecryptedModel` decrypts entirely in RAM with `mlock` (never touches disk); key from file or env var
- **KeyProvider Trait**: Abstract key loading for HSM integration and zero-downtime key rotation; `StaticKeyProvider` wraps existing file/env key source; `RotatingKeyProvider` holds multiple keys and advances on `rotate_key()` — deploy new key, rotate, remove old
- **Rate Limiting**: Token-bucket rate limiter middleware (`rate_limit_rps`) and concurrency cap (`max_concurrent_requests`) applied to all `/v1/*` endpoints; returns `429 Too Many Requests` with OpenAI-style error body
- **Model-Attestation Binding**: `GET /v1/attestation?model=<name>` embeds the model's SHA-256 hash into `report_data` alongside the nonce — layout `[nonce(32)][model_sha256(32)]` — cryptographically tying the attestation to the specific model being served
- **Health + TEE Status**: `GET /health` reports TEE type, attestation status, and model verification state
- **OpenAI-Compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` — works with any OpenAI SDK
- **Pure Rust Inference (default)**: GGUF model inference via `mistralrs` (built on candle) — no C++ dependency, ideal for TEE auditing
- **SafeTensors Inference**: HuggingFace SafeTensors chat models loaded via `TextModelBuilder` with ISQ on-load quantization (default Q8_0); register with `format=safetensors`, configure ISQ via `default_parameters.isq` (e.g. `Q4K`, `Q6K`, `Q8_0`)
- **Vision/Multimodal Inference**: Vision models (e.g. LLaVA, Phi-3-Vision) loaded via `VisionModelBuilder`; register with `format=vision`; pass base64-encoded images via `images` field or OpenAI-style `content` parts (`image_url`); ISQ quantization supported
- **True Token-by-Token Streaming**: Chat completions use `stream_chat_request` for per-token SSE delivery; each `Response::Chunk` is forwarded immediately as it is generated
- **Embedding Models**: HuggingFace-format embedding models (e.g. Qwen3-Embedding, GTE, NomicBert) loaded via `EmbeddingModelBuilder`; register with `format=huggingface`, call `POST /v1/embeddings`; empty-input fast path returns immediately
- **llama.cpp Backend (optional)**: GGUF inference via `llama-cpp-2` Rust bindings (feature-gated, requires C++ toolchain)
- **GPU Acceleration**: Auto-detection of Apple Metal and NVIDIA CUDA; configurable layer offloading, multi-GPU support
- **Chat Template Engine**: Jinja2-compatible template rendering via `minijinja` (Llama 3, ChatML, Phi, Gemma, custom)
- **Tool/Function Calling**: Structured tool definitions with XML, Mistral, and JSON output parsing
- **JSON Schema Structured Output**: Constrain model output via JSON Schema → GBNF grammar conversion
- **Thinking & Reasoning**: Streaming `<think>` block parser for DeepSeek-R1, QwQ reasoning models
- **KV Cache Reuse**: Prefix matching across multi-turn requests for conversation speedup
- **Content-Addressed Storage**: Model blobs stored by SHA-256 hash with automatic deduplication
- **Automatic Model Lifecycle**: LRU eviction, configurable keep-alive, background reaper for idle models
- **TEE Metrics**: Prometheus counters for attestation reports (`power_tee_attestations_total`), model decryptions (`power_tee_model_decryptions_total`), and log redactions (`power_tee_redactions_total`)
- **RA-TLS Transport**: Feature-gated (`tls`) TLS server with self-signed ECDSA P-256 certificate; when `ra_tls = true`, the TEE attestation report is embedded as a custom X.509 extension (OID 1.3.6.1.4.1.56560.1.1) so clients can cryptographically verify the server is running inside a genuine TEE before trusting inference
- **Vsock Transport**: Feature-gated (`vsock`, Linux only) AF_VSOCK server for a3s-box MicroVM guest-host communication; exposes the same API as the TCP listener without requiring any network configuration inside the VM
- **Prometheus Metrics**: Request counts, durations, tokens, inference timing, TTFT, model memory, GPU utilization
- **HCL Configuration**: HashiCorp Configuration Language for all settings

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     a3s-power                         │
│                                                       │
│  TEE Layer                                            │
│  ┌────────────────┐ ┌──────────────┐ ┌────────────┐  │
│  │  attestation   │ │  model_seal  │ │  privacy   │  │
│  │  (TeeProvider) │ │  (SHA-256)   │ │  (redact)  │  │
│  └────────┬───────┘ └──────┬───────┘ └─────┬──────┘  │
│           │                │               │          │
│  Server Layer              │               │          │
│  ┌─────────────────────────┴───────────────┴───────┐  │
│  │  Axum Router                                    │  │
│  │  /health  /metrics  /v1/*                       │  │
│  └──────────────────────┬──────────────────────────┘  │
│                         │                             │
│  Backend Layer          │                             │
│  ┌──────────────────────┴──────────────────────────┐  │
│  │  BackendRegistry                                │  │
│  │  ┌────────────────────────────────────────────┐ │  │
│  │  │  MistralRsBackend (feature: mistralrs) ★   │ │  │
│  │  │  pure Rust · candle · GGUF                 │ │  │
│  │  ├────────────────────────────────────────────┤ │  │
│  │  │  LlamaCppBackend (feature: llamacpp)       │ │  │
│  │  │  chat_template · json_schema · tool_parser │ │  │
│  │  └────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  Model Layer                                          │
│  ┌─────────────────────────────────────────────────┐  │
│  │  ModelRegistry                                  │  │
│  │  manifest · storage (SHA-256) · gguf parser     │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Backend Trait

The `Backend` trait abstracts inference engines. Two backends are available:

- **`mistralrs`** (default): Pure Rust inference via `mistralrs` (candle-based). No C++ toolchain required. Ideal for TEE environments where supply-chain auditability matters.
- **`llamacpp`** (optional): C++ llama.cpp via `llama-cpp-2` Rust bindings. Mature, full-featured (KV cache reuse, LoRA, grammar sampling, mirostat).

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

### TEE Extension Points

TEE capabilities are implemented as traits with default implementations:

```rust
/// Remote attestation provider.
pub trait TeeProvider: Send + Sync {
    /// Generate attestation report. Optional nonce is bound into report_data
    /// to prevent replay attacks.
    async fn attestation_report(&self, nonce: Option<&[u8]>) -> Result<AttestationReport>;
    /// Generate attestation report bound to a specific model hash.
    /// report_data layout: [nonce(32 bytes)][model_sha256(32 bytes)]
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

/// Model decryption key provider.
pub trait KeyProvider: Send + Sync {
    async fn get_key(&self) -> Result<[u8; 32]>;
    async fn rotate_key(&self) -> Result<[u8; 32]>;  // default: Err (not supported)
    fn provider_name(&self) -> &str;
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
| `max_concurrent_requests` | `0` | Max concurrent requests for `/v1/*` endpoints (`0` = unlimited) |
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
  "version": "0.2.0",
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
| `GET` | `/v1/models/pull/:name/status` | Get persisted pull progress for a model (`status`, `completed`, `total`, `error`); requires `hf` feature |
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
| `mistralrs` | ✅ enabled | Pure Rust inference backend via `mistralrs` (candle-based). No C++ toolchain required. Ideal for TEE auditing. |
| `llamacpp` | ❌ disabled | llama.cpp inference backend via `llama-cpp-2`. Requires C++ compiler + CMake. Full-featured (KV cache, LoRA, grammar, mirostat). |
| `tls` | ❌ disabled | RA-TLS transport: TLS server with self-signed cert + optional attestation X.509 extension. Adds `axum-server`, `rcgen`, `time` deps. |
| `vsock` | ❌ disabled | Vsock transport for a3s-box MicroVM guest-host HTTP. **Linux only** — requires `AF_VSOCK` kernel support. Adds `tokio-vsock` dep. |

Without a backend feature (`mistralrs` or `llamacpp`), Power can manage models but inference calls return "backend not available".

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

# Test (755+ tests)
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
├── README.md
└── src/
    ├── main.rs              # Entry point (start server)
    ├── lib.rs               # Module declarations
    ├── error.rs             # PowerError enum + Result<T>
    ├── config.rs            # HCL configuration (TEE-native)
    ├── dirs.rs              # Platform paths (~/.a3s/power/)
    │
    ├── tee/                 # TEE privacy protection layer
    │   ├── mod.rs           # Module entry
    │   ├── attestation.rs   # TeeProvider trait + remote attestation + build_report_data
    │   ├── cert.rs          # CertManager: RA-TLS cert generation (feature: tls)
    │   ├── encrypted_model.rs # AES-256-GCM model encryption/decryption + MemoryDecryptedModel
    │   ├── key_provider.rs  # KeyProvider trait + StaticKeyProvider + RotatingKeyProvider
    │   ├── model_seal.rs    # Model integrity verification (SHA-256)
    │   ├── policy.rs        # TEE policy enforcement
    │   └── privacy.rs       # PrivacyProvider + deep log redaction + memory zeroing
    │
    ├── backend/             # Inference engines
    │   ├── mod.rs           # Backend trait + BackendRegistry
    │   ├── types.rs         # ChatRequest, CompletionRequest, EmbeddingRequest
    │   ├── mistralrs_backend.rs # mistral.rs backend — pure Rust (feature: mistralrs, default)
    │   ├── llamacpp.rs      # llama.cpp backend (feature: llamacpp)
    │   ├── chat_template.rs # Jinja2 chat template rendering
    │   ├── json_schema.rs   # JSON Schema → GBNF grammar
    │   ├── tool_parser.rs   # Tool call output parser
    │   ├── think_parser.rs  # Thinking block parser
    │   ├── gpu.rs           # GPU auto-detection
    │   └── test_utils.rs    # MockBackend for testing
    │
    ├── model/               # Model management (local only)
    │   ├── mod.rs           # Module declarations
    │   ├── manifest.rs      # ModelManifest metadata
    │   ├── registry.rs      # In-memory model index
    │   ├── storage.rs       # Content-addressed blob store
    │   └── gguf.rs          # GGUF metadata parser
    │
    ├── server/              # HTTP server
    │   ├── mod.rs           # Server startup + TEE init + key provider init + keep_alive reaper
    │   ├── state.rs         # Shared AppState with LRU tracking + key_provider field
    │   ├── router.rs        # Axum router + rate limiting middleware
    │   ├── lock.rs          # Shared RwLock helpers (read_lock/write_lock)
    │   ├── metrics.rs       # Prometheus metrics
    │   ├── auth.rs          # API key authentication
    │   ├── audit.rs         # Structured audit logging (JSONL)
    │   ├── request_context.rs # Request-scoped context (ID, timing)
    │   └── vsock.rs         # Vsock transport (feature: vsock, Linux only)
    │
    └── api/                 # HTTP API handlers
        ├── mod.rs           # Shared utilities
        ├── health.rs        # GET /health (with TEE status)
        ├── autoload.rs      # Model auto-loading on first request
        ├── sse.rs           # SSE streaming utilities
        ├── types.rs         # Request/response types
        └── openai/          # OpenAI-compatible API
            ├── mod.rs       # Route group
            ├── chat.rs      # POST /v1/chat/completions
            ├── completions.rs # POST /v1/completions
            ├── models.rs    # GET /v1/models
            ├── embeddings.rs # POST /v1/embeddings
            └── attestation.rs # GET /v1/attestation (nonce + model binding)
```

## A3S Ecosystem

A3S Power is the inference engine of the A3S privacy-preserving AI platform. It runs inside a3s-box MicroVMs to provide hardware-isolated LLM inference.

```
┌──────────────────────────────────────────────────────┐
│                    A3S Ecosystem                      │
│                                                       │
│  Runtime:     a3s-box      (MicroVM with TEE)         │
│                  │                                     │
│  Inference:   a3s-power ◄── runs inside a3s-box       │
│                  │                                     │
│  Application: a3s-code     (AI coding agent)          │
│                  │                                     │
│  Platform:    a3s-gateway  (API gateway)               │
│               a3s-event    (event bus)                 │
└──────────────────────────────────────────────────────┘
```

| Component | Relationship to Power |
|-----------|----------------------|
| **a3s-box** | Hosts Power inside TEE-enabled MicroVMs (AMD SEV-SNP / Intel TDX) |
| **a3s-code** | Uses Power as a local inference backend |
| **a3s-gateway** | Routes inference requests to Power instances |

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
- [x] Pull progress persistence — JSON state files in `~/.a3s/power/pulls/`; `GET /v1/models/pull/:name/status` returns `{status, completed, total, error}`; survives server restarts; throttled writes (every 5%) to minimize disk I/O
- [x] True token-by-token streaming — `stream_chat_request` replaces non-streaming path; each `Response::Chunk` forwarded immediately via mpsc channel; `Response::Done` sets `finish_reason`
- [x] Vision/multimodal inference — `ModelFormat::Vision` variant; `MistralRsBackend` loads vision models via `VisionModelBuilder` with ISQ; base64 images accepted via `images` field or OpenAI `image_url` content parts; decoded with `image` + `base64` crates

## Community

Join us on [Discord](https://discord.gg/XVg6Hu6H) for questions, discussions, and updates.

## License

MIT
