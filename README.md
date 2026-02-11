# A3S Power

<p align="center">
  <strong>Local Model Management & Serving</strong>
</p>

<p align="center">
  <em>Infrastructure layer — CLI + HTTP server for downloading, managing, and running local LLM models</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#development">Development</a>
</p>

---

## Overview

**A3S Power** is an Ollama-compatible CLI tool and HTTP server for local model management and inference. It provides both an Ollama-compatible native API and an OpenAI-compatible API, so existing tools, SDKs, and frontends work out of the box.

### Basic Usage

```bash
# Pull a model by name (resolves from Ollama registry, built-in registry, or HuggingFace)
a3s-power pull llama3.2:3b

# Pull from a direct URL
a3s-power pull https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Interactive chat
a3s-power run llama3.2:3b

# Single prompt
a3s-power run llama3.2:3b --prompt "Explain quicksort in one paragraph"

# Push a model to a remote registry
a3s-power push llama3.2:3b --destination https://registry.example.com

# Start HTTP server
a3s-power serve
```

## Features

- **CLI Model Management**: Pull, list, show, delete, and push models from the command line
- **Ollama Registry Integration**: Pull any model from `registry.ollama.ai` by name (`llama3.2:3b`) — primary resolution source with built-in registry and HuggingFace fallback
- **Interactive Chat**: Multi-turn conversation with streaming token output
- **Vision/Multimodal Support**: Accept base64 images (Ollama `images` field) and image URLs (OpenAI `content` array format); projector auto-downloaded from Ollama registry; image processing requires vision model with projector (e.g. llava)
- **Tool/Function Calling**: Structured tool definitions, tool choice, and tool call responses (OpenAI-compatible)
- **JSON Schema Structured Output**: Constrain model output to match JSON Schema via GBNF grammar generation — supports `"json"`, `{"type":"json_object"}`, or full JSON Schema objects
- **Chat Template Auto-Detection**: Detects ChatML, Llama, Phi, and Generic templates from GGUF metadata
- **Jinja2 Template Engine**: Renders arbitrary Jinja2 chat templates via `minijinja` (Llama 3, Gemma, ChatML, Phi, custom) with hardcoded fallback
- **KV Cache Reuse**: Persists `LlamaContext` across requests with prefix matching — skips re-evaluating shared prompt tokens for multi-turn speedup
- **Tool Call Parsing**: Parses model output into structured `tool_calls` — supports `<tool_call>` XML, `[TOOL_CALLS]` prefix, and raw JSON formats
- **Modelfile Support**: Create custom models with `FROM`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, `ADAPTER` (LoRA/QLoRA), `LICENSE`, and `MESSAGE` (pre-seeded conversations) directives
- **Multiple Concurrent Models**: Load multiple models with LRU eviction at configurable capacity
- **Automatic Model Unloading**: Background keep_alive reaper unloads idle models after configurable timeout (default 5m)
- **GPU Acceleration**: Configurable GPU layer offloading via `[gpu]` config section with automatic GPU detection (Metal/CUDA), multi-GPU support (`main_gpu`), and per-request `num_gpu` override
- **GPU Auto-Detection**: Automatically detects Apple Metal and NVIDIA CUDA GPUs at server startup, sets optimal `gpu_layers` when not explicitly configured
- **Memory Estimation**: Estimates VRAM requirements before loading a model (model weights + KV cache + compute overhead) and logs warnings
- **Full Ollama Options**: All Ollama generation options supported — `repeat_last_n`, `penalize_newline`, `num_batch`, `num_thread`, `num_thread_batch`, `use_mmap`, `use_mlock`, `numa`, `flash_attention`, `num_gpu`, `main_gpu` — in addition to standard sampling parameters
- **Embedding Support**: Real embedding generation with automatic model reload in embedding mode
- **HTTP Server**: Axum-based server with CORS, tracing, and metrics middleware
- **Ollama-Compatible API**: `/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`, `/api/push`, `/api/show`, `/api/delete`, `/api/embeddings`, `/api/embed`, `/api/ps`, `/api/copy`, `/api/version`, `/api/blobs/:digest`
- **OpenAI-Compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **Blob Management API**: Check, upload, and download content-addressed blobs via REST
- **Push API**: Upload models to remote registries with progress reporting
- **NDJSON Streaming**: Native API endpoints stream as `application/x-ndjson` (Ollama wire format); OpenAI endpoints use SSE
- **Context Token Return**: `/api/generate` returns token IDs in `context` field for conversation continuity
- **Prometheus Metrics**: `GET /metrics` endpoint with request counts, durations, tokens, model gauges, inference duration, TTFT, cost, evictions, model memory, and GPU metrics
- **Usage Dashboard**: `GET /v1/usage` endpoint with date range and model filtering for cost tracking
- **GGUF Metadata Reader**: Lightweight binary parser for GGUF file headers — extracts architecture metadata and tensor descriptors without loading weights
- **Verbose Show**: `/api/show` with `verbose: true` returns full GGUF metadata and tensor information
- **Per-Layer Pull Progress**: Pull progress shows per-layer digest identifiers (`pulling sha256:abc...`) matching Ollama's output format
- **Content-Addressed Storage**: Model blobs stored by SHA-256 hash with automatic deduplication
- **llama.cpp Backend**: GGUF inference via `llama-cpp-2` Rust bindings (optional feature flag)
- **Health Check**: `GET /health` endpoint with uptime, version, and loaded model count
- **Model Auto-Loading**: Models are automatically loaded on first inference request with LRU eviction
- **TOML Configuration**: User-configurable host, port, GPU settings, keep_alive, and storage settings
- **Ollama Environment Variables**: `OLLAMA_HOST`, `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_GPU`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_DEBUG`, `OLLAMA_ORIGINS`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_TMPDIR`, `OLLAMA_NOPRUNE`, `OLLAMA_SCHED_SPREAD` for drop-in compatibility
- **Download Resumption**: Interrupted model downloads resume automatically via HTTP Range requests
- **Async-First**: Built on Tokio for high-performance async operations

## Ollama Compatibility Status

> Compared against Ollama source at [github.com/ollama/ollama](https://github.com/ollama/ollama) (latest main).

### ✅ Fully Aligned

| Category | Status |
|----------|--------|
| Native API (14 endpoints) | `/api/generate`, `/api/chat`, `/api/pull`, `/api/push`, `/api/tags`, `/api/show`, `/api/delete`, `/api/copy`, `/api/embed`, `/api/embeddings`, `/api/ps`, `/api/version`, `/api/create`, `/api/blobs/:digest` |
| OpenAI API (4 endpoints) | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` |
| CLI commands (12) | `run`, `pull`, `list/ls`, `show`, `delete/rm`, `serve`, `create`, `push`, `cp`, `ps`, `stop`, `help` |
| Streaming | NDJSON for native API, SSE for OpenAI API |
| Modelfile | `FROM`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, `ADAPTER`, `LICENSE`, `MESSAGE` + heredoc |
| Sampling parameters | temperature, top_p, top_k, min_p, repeat_penalty, frequency/presence_penalty, seed, typical_p, num_keep, stop |
| Runner options | num_ctx, num_predict, num_batch, num_gpu, num_thread, use_mmap |
| Keep-alive | String + numeric, per-request + global config, `"0"` / `"-1"` special values |
| Tool/Function calling | Both native `/api/chat` and OpenAI `/v1/chat/completions`, XML/Mistral/JSON parsing |
| JSON structured output | `"json"`, `{"type":"json_object"}`, full JSON Schema → GBNF grammar |
| Ollama registry | Pull from `registry.ollama.ai` with template/system/params/license extraction |
| KV cache reuse | Prefix matching across multi-turn requests |
| LoRA adapters | `ADAPTER` directive, loaded at inference |
| GPU auto-detection | Metal + CUDA, auto `gpu_layers`, multi-GPU |
| Blob management | HEAD/POST/GET/DELETE `/api/blobs/:digest` |
| Context return | `/api/generate` returns `context` token array |
| `done_reason` | Returned in generate/chat responses |
| `raw` mode | Skip template formatting in `/api/generate` |
| `suffix` field | Fill-in-the-middle in `/api/generate` |
| CORS | Configurable origins with `OLLAMA_ORIGINS` |

### 🔴 Remaining Gaps (vs Ollama latest)

#### API Request/Response Fields

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `think` parameter | Critical | `api/types.go:109,173` | `ThinkValue` (bool or `"high"/"medium"/"low"`) in generate/chat requests — enables reasoning models (DeepSeek-R1, QwQ). Not implemented. |
| `thinking` response field | Critical | `api/types.go:216,856` | `Message.Thinking` and `GenerateResponse.Thinking` — returns thinking content separately from response. Not implemented. |
| Thinking parser | Critical | `thinking/parser.go` | Streaming parser that separates `<think>...</think>` blocks from content in real-time. Infers tags from template. Not implemented. |
| `logprobs` / `top_logprobs` | Important | `api/types.go:123-129,187-193` | Log probability support in generate/chat requests + `Logprob`/`TokenLogprob` response types. Not implemented. |
| `truncate` field (generate/chat) | Important | `api/types.go:112,176` | Truncate prompt when exceeding context length instead of erroring. Not implemented. |
| `shift` field (generate/chat) | Important | `api/types.go:117,180` | Shift context window when hitting limit instead of erroring. Not implemented. |
| `_debug_render_only` | Nice-to-have | `api/types.go:121,185` | Debug mode that returns rendered template without calling model. Not implemented. |
| `tool_calls` in GenerateResponse | Moderate | `api/types.go:870` | `/api/generate` can also return `tool_calls` (not just `/api/chat`). Not implemented. |

#### OpenAI API Gaps

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `GET /v1/models/:model` | Important | `routes.go:1610` | Retrieve single model details. Not implemented (only `GET /v1/models` list). |
| `POST /v1/responses` | Moderate | `routes.go:1611` | OpenAI Responses API compatibility. Not implemented. |
| `POST /v1/messages` | Moderate | `routes.go:1617` | Anthropic Messages API compatibility via middleware. Not implemented. |
| `POST /v1/images/generations` | Nice-to-have | `routes.go:1613` | Image generation endpoint. Not implemented. |
| `POST /v1/images/edits` | Nice-to-have | `routes.go:1614` | Image editing endpoint. Not implemented. |
| `reasoning` / `reasoning_effort` | Important | `openai/openai.go:94-96,112-113` | OpenAI reasoning effort (`"high"/"medium"/"low"`) mapped to `think`. Not implemented. |
| `stream_options.include_usage` | Moderate | `openai/openai.go:90-92` | Return usage stats in final streaming chunk when requested. Not implemented. |
| `encoding_format` (embeddings) | Moderate | `openai/openai.go:87` | `"float"` or `"base64"` encoding for embedding responses. Not implemented. |
| `dimensions` (embeddings) | Moderate | `api/types.go:626` | Truncate output embeddings to specified dimension. Not implemented. |

#### ShowResponse Fields

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `capabilities` | Important | `api/types.go:755` | List of model capabilities (`completion`, `tools`, `vision`, `thinking`, `embedding`, `insert`, `image`). Not implemented. |
| `renderer` / `parser` | Moderate | `api/types.go:746-747` | Custom renderer/parser names for model. Not implemented. |
| `projector_info` | Moderate | `api/types.go:753` | Projector metadata for vision models. Not implemented. |
| `remote_model` / `remote_host` | Moderate | `api/types.go:750-751` | Remote model proxy info. Not implemented. |
| `requires` | Nice-to-have | `api/types.go:757` | Minimum Ollama version required. Not implemented. |
| `messages` | Moderate | `api/types.go:749` | Pre-seeded messages in show response. Not implemented. |

#### ProcessResponse (ps) Fields

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `size_vram` | Moderate | `api/types.go:829` | VRAM usage per loaded model. Not implemented. |
| `context_length` | Moderate | `api/types.go:830` | Active context length per loaded model. Not implemented. |

#### Create API

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| New structured Create API | Important | `api/types.go:663-709` | Ollama's new `from`, `files`, `adapters`, `template`, `system`, `parameters`, `messages`, `license` fields (replacing Modelfile-only approach). a3s-power only supports Modelfile-based create. |
| Re-quantization | Important | `server/create.go` | `create --quantize q4_K_M` actually quantizes the model. a3s-power accepts but no-ops. |
| SafeTensors conversion | Moderate | `convert/` | Convert SafeTensors → GGUF during create. Not implemented. |

#### Environment Variables

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `OLLAMA_KV_CACHE_TYPE` | Important | `envconfig/config.go:278` | KV cache quantization type (default: f16). Not implemented. |
| `OLLAMA_GPU_OVERHEAD` | Moderate | `envconfig/config.go:279` | Reserve VRAM per GPU (bytes). Not implemented. |
| `OLLAMA_LOAD_TIMEOUT` | Moderate | `envconfig/config.go:283` | Stall detection timeout for model loads (default 5m). Not implemented. |
| `OLLAMA_MAX_QUEUE` | Moderate | `envconfig/config.go:285` | Maximum queued requests. Not implemented. |
| `OLLAMA_NOHISTORY` | Nice-to-have | `envconfig/config.go:287` | Disable readline history. Not implemented. |
| `OLLAMA_MULTIUSER_CACHE` | Nice-to-have | `envconfig/config.go:292` | Optimize prompt caching for multi-user. Not implemented. |
| `OLLAMA_CONTEXT_LENGTH` | Important | `envconfig/config.go:293` | Global default context length override. Not implemented. |
| `OLLAMA_REMOTES` | Moderate | `envconfig/config.go:295` | Allowed hosts for remote models. Not implemented. |
| `OLLAMA_LLM_LIBRARY` | Nice-to-have | `envconfig/config.go:282` | Override LLM library autodetection. Not applicable (Rust bindings). |

#### Auth & Account

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `signin` / `signout` CLI | Moderate | `cmd/cmd.go:666,697` | Sign in/out of ollama.com account. Not implemented. |
| `POST /api/me` | Moderate | `routes.go:1583` | Whoami endpoint. Not implemented. |
| `POST /api/signout` | Moderate | `routes.go:1585` | Signout endpoint. Not implemented. |
| Registry auth (push) | Important | `auth/auth.go` | Keypair-based auth for pushing to `registry.ollama.ai`. Not implemented. |

#### CLI Flags

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `run --think` | Critical | `cmd/cmd.go:2069` | Enable thinking mode from CLI. Not implemented. |
| `run --hidethinking` | Important | `cmd/cmd.go:2071` | Hide thinking output in CLI. Not implemented. |
| `run --truncate` | Moderate | `cmd/cmd.go:2072` | Truncate embeddings input. Not implemented. |
| `run --dimensions` | Moderate | `cmd/cmd.go:2073` | Truncate output embeddings dimension. Not implemented. |
| `run --nowordwrap` | Nice-to-have | `cmd/cmd.go:2067` | Disable word wrapping in CLI. Not implemented. |
| `show --license` | Nice-to-have | `cmd/cmd.go:2049` | Show only license. Not implemented (shows all). |
| `show --modelfile` | Nice-to-have | `cmd/cmd.go:2050` | Show only modelfile. Not implemented. |
| `show --parameters` | Nice-to-have | `cmd/cmd.go:2051` | Show only parameters. Not implemented. |
| `show --template` | Nice-to-have | `cmd/cmd.go:2052` | Show only template. Not implemented. |
| `show --system` | Nice-to-have | `cmd/cmd.go:2053` | Show only system message. Not implemented. |
| `run --experimental` | Nice-to-have | `cmd/cmd.go:2074` | Experimental agent loop with tools. Not implemented. |

#### Server/Runtime

| Gap | Severity | Ollama Source | Description |
|-----|----------|---------------|-------------|
| `GET /` and `HEAD /` | Nice-to-have | `routes.go:1570-1571` | Returns `"Ollama is running"` string. Not implemented (a3s-power has `/health`). |
| Experimental aliases API | Nice-to-have | `routes.go:1594-1596` | `GET/POST/DELETE /api/experimental/aliases`. Not implemented. |
| Request queuing | Moderate | `envconfig:OLLAMA_MAX_QUEUE` | Queue requests when all model slots busy. Not implemented. |
| `num_parallel` wiring | Moderate | — | Concurrent request slots per loaded model. Config exists but unclear if wired to llama.cpp. |

#### Extra Options (a3s-power has but Ollama removed)

Note: a3s-power supports some options that Ollama has **removed** from their latest `Options` struct:
- `mirostat`, `mirostat_tau`, `mirostat_eta` — removed from Ollama
- `tfs_z` — removed from Ollama
- `main_gpu` — removed from Ollama Runner
- `use_mlock` — removed from Ollama Runner
- `flash_attention` — removed from Ollama Runner (now env-only via `OLLAMA_FLASH_ATTENTION`)
- `num_thread_batch` — removed from Ollama Runner
- `penalize_newline` — removed from Ollama
- `numa` — removed from Ollama

These are kept in a3s-power for backward compatibility but may diverge from Ollama's current behavior.

## Quality Metrics

### Test Coverage

**878 unit tests** with **90.11% region coverage** across 59 source files:

| Module | Lines | Coverage | Functions | Coverage |
|--------|-------|----------|-----------|----------|
| api/health.rs | 62 | 100.00% | 10 | 100.00% |
| api/mod.rs | 27 | 100.00% | 5 | 100.00% |
| api/native/mod.rs | 22 | 100.00% | 1 | 100.00% |
| api/native/ps.rs | 149 | 100.00% | 17 | 100.00% |
| api/native/version.rs | 21 | 100.00% | 6 | 100.00% |
| api/openai/mod.rs | 30 | 100.00% | 4 | 100.00% |
| api/openai/usage.rs | 384 | 100.00% | 27 | 100.00% |
| backend/llamacpp.rs | 186 | 100.00% | 26 | 100.00% |
| backend/test_utils.rs | 130 | 100.00% | 18 | 100.00% |
| cli/delete.rs | 102 | 100.00% | 5 | 100.00% |
| cli/list.rs | 88 | 100.00% | 7 | 100.00% |
| error.rs | 93 | 100.00% | 19 | 100.00% |
| model/manifest.rs | 164 | 100.00% | 19 | 100.00% |
| server/router.rs | 209 | 100.00% | 33 | 100.00% |
| backend/json_schema.rs | 389 | 98.97% | 53 | 100.00% |
| backend/tool_parser.rs | 347 | 99.14% | 43 | 100.00% |
| model/modelfile.rs | 552 | 99.28% | 42 | 100.00% |
| server/state.rs | 266 | 99.25% | 37 | 97.30% |
| api/sse.rs | 95 | 98.95% | 16 | 93.75% |
| api/types.rs | 613 | 98.37% | 52 | 100.00% |
| server/metrics.rs | 607 | 98.35% | 54 | 96.30% |
| backend/chat_template.rs | 349 | 98.28% | 32 | 100.00% |
| backend/mod.rs | 65 | 98.46% | 15 | 100.00% |
| dirs.rs | 55 | 98.18% | 12 | 91.67% |
| backend/types.rs | 261 | 98.08% | 23 | 95.65% |
| api/native/chat.rs | 735 | 94.42% | 32 | 100.00% |
| api/native/generate.rs | 709 | 95.77% | 32 | 100.00% |
| api/native/models.rs | 457 | 96.06% | 32 | 100.00% |
| config.rs | 475 | 96.84% | 60 | 96.67% |
| api/openai/embeddings.rs | 187 | 95.72% | 9 | 100.00% |
| api/native/blobs.rs | 212 | 94.81% | 15 | 100.00% |
| api/autoload.rs | 220 | 94.09% | 24 | 100.00% |
| api/native/embed.rs | 158 | 93.04% | 9 | 100.00% |
| model/gguf.rs | 746 | 93.43% | 80 | 80.00% |
| api/openai/models.rs | 118 | 93.22% | 9 | 100.00% |
| api/native/embeddings.rs | 133 | 96.24% | 7 | 100.00% |
| api/native/copy.rs | 60 | 91.67% | 6 | 100.00% |
| cli/mod.rs | 340 | 91.18% | 34 | 100.00% |
| api/native/create.rs | 340 | 90.00% | 19 | 94.74% |
| api/openai/chat.rs | 531 | 88.14% | 23 | 78.26% |
| model/registry.rs | 308 | 87.99% | 42 | 83.33% |
| model/storage.rs | 331 | 87.31% | 31 | 83.87% |
| cli/show.rs | 234 | 84.19% | 15 | 100.00% |
| api/openai/completions.rs | 394 | 82.99% | 14 | 78.57% |
| backend/gpu.rs | 281 | 82.92% | 38 | 92.11% |
| model/resolve.rs | 341 | 75.66% | 54 | 79.63% |
| api/native/push.rs | 187 | 75.40% | 10 | 80.00% |
| cli/push.rs | 43 | 74.42% | 10 | 90.00% |
| model/ollama_registry.rs | 530 | 73.21% | 57 | 70.18% |
| cli/ps.rs | 152 | 70.39% | 22 | 81.82% |
| cli/serve.rs | 34 | 70.59% | 4 | 50.00% |
| cli/stop.rs | 102 | 70.59% | 12 | 75.00% |
| server/mod.rs | 84 | 65.48% | 12 | 66.67% |
| model/push.rs | 151 | 62.91% | 27 | 81.48% |
| cli/pull.rs | 72 | 62.50% | 6 | 83.33% |
| api/native/pull.rs | 269 | 50.19% | 16 | 81.25% |
| cli/run.rs | 845 | 48.88% | 57 | 85.96% |
| model/pull.rs | 384 | 48.70% | 36 | 63.89% |
| **TOTAL** | **15429** | **87.94%** | **1430** | **91.47%** |

> **Overall: 90.11% region coverage, 91.47% function coverage, 87.94% line coverage**

Run coverage report:
```bash
LLVM_COV=/opt/homebrew/Cellar/llvm/21.1.8/bin/llvm-cov \
LLVM_PROFDATA=/opt/homebrew/Cellar/llvm/21.1.8/bin/llvm-profdata \
cargo llvm-cov --lib -p a3s-power --summary-only
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│                  a3s-power                       │
│                                                  │
│  CLI Layer                                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │ run  │ │ pull │ │ list │ │ push │ │serve │ │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ │
│     │        │        │        │        │      │
│  Model Layer          │                  │      │
│  ┌────────────────────┴────────┐         │      │
│  │      ModelRegistry          │         │      │
│  │  ┌──────────┐ ┌──────────┐ │         │      │
│  │  │ manifest │ │ storage  │ │         │      │
│  │  └──────────┘ └──────────┘ │         │      │
│  └─────────────────────────────┘         │      │
│                                          │      │
│  Backend Layer                           │      │
│  ┌─────────────────────────────┐         │      │
│  │    BackendRegistry          │         │      │
│  │  ┌──────────────────────┐  │         │      │
│  │  │ LlamaCppBackend      │  │         │      │
│  │  │ (feature: llamacpp)  │  │         │      │
│  │  └──────────────────────┘  │         │      │
│  └─────────────────────────────┘         │      │
│                                          │      │
│  Server Layer ◄──────────────────────────┘      │
│  ┌─────────────────────────────────────┐        │
│  │  Axum Router                        │        │
│  │  ┌────────────┐ ┌────────────────┐  │        │
│  │  │ /api/*     │ │ /v1/*          │  │        │
│  │  │ (Ollama)   │ │ (OpenAI)       │  │        │
│  │  └────────────┘ └────────────────┘  │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
```

### Backend Trait

The `Backend` trait abstracts inference engines. The llama.cpp backend is feature-gated; without the `llamacpp` feature, Power can still manage models but returns "backend not available" for inference calls.

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

## Installation

### Homebrew (macOS)

```bash
brew install a3s-lab/tap/a3s-power
```

### Cargo (cross-platform)

```bash
# Model management only
cargo install a3s-power

# With llama.cpp inference backend (requires C++ compiler + CMake)
cargo install a3s-power --features llamacpp
```

### Pre-built Binary (macOS Apple Silicon)

```bash
curl -LO https://github.com/A3S-Lab/Power/releases/download/v0.1.2/a3s-power-v0.1.2-aarch64-apple-darwin.tar.gz
tar xzf a3s-power-v0.1.2-aarch64-apple-darwin.tar.gz
sudo mv a3s-power /usr/local/bin/
```

### Build from Source

```bash
git clone https://github.com/A3S-Lab/Power.git
cd Power

# Without inference backend (model management only)
cargo build --release

# With llama.cpp inference (requires C++ compiler + CMake)
cargo build --release --features llamacpp

# Binary at target/release/a3s-power
```

## Quick Start

### Model Management

```bash
# Pull a model by name (Ollama registry → built-in registry → HuggingFace fallback)
a3s-power pull llama3.2:3b

# Pull from a direct URL
a3s-power pull https://example.com/model.gguf

# List local models
a3s-power list

# Show model details
a3s-power show my-model

# Delete a model
a3s-power delete my-model

# Push a model to a remote registry
a3s-power push my-model --destination https://registry.example.com
```

### Interactive Chat

```bash
# Start interactive chat session
a3s-power run my-model

# Send a single prompt
a3s-power run my-model --prompt "What is Rust?"
```

### HTTP Server

```bash
# Start server on default port (127.0.0.1:11434)
a3s-power serve

# Custom host and port
a3s-power serve --host 0.0.0.0 --port 8080
```

## API Reference

### Server

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (status, version, uptime, loaded models) |
| `GET` | `/metrics` | Prometheus metrics (requests, durations, tokens, inference, TTFT, cost, evictions, model memory, GPU) |

### Native API (Ollama-Compatible)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Text generation (streaming/non-streaming) |
| `POST` | `/api/chat` | Chat completion with vision & tool support (streaming/non-streaming) |
| `POST` | `/api/pull` | Download a model by name or URL (streaming progress) |
| `POST` | `/api/push` | Push a model to a remote registry |
| `GET` | `/api/tags` | List local models |
| `POST` | `/api/show` | Show model details |
| `DELETE` | `/api/delete` | Delete a model |
| `POST` | `/api/embeddings` | Generate embeddings |
| `POST` | `/api/embed` | Batch embedding generation |
| `GET` | `/api/ps` | List running/loaded models |
| `POST` | `/api/copy` | Copy/alias a model |
| `GET` | `/api/version` | Server version |
| `HEAD` | `/api/blobs/:digest` | Check if a blob exists |
| `POST` | `/api/blobs/:digest` | Upload a blob with SHA-256 verification |
| `GET` | `/api/blobs/:digest` | Download a blob |
| `DELETE` | `/api/blobs/:digest` | Delete a blob |

### OpenAI-Compatible API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming/non-streaming) |
| `POST` | `/v1/completions` | Text completion (streaming/non-streaming) |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/embeddings` | Generate embeddings |
| `GET` | `/v1/usage` | Usage and cost dashboard data (date range + model filter) |

### Examples

#### List Models

```bash
# OpenAI-compatible
curl http://localhost:11434/v1/models

# Ollama-compatible
curl http://localhost:11434/api/tags
```

#### Chat Completion (OpenAI)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Chat Completion with Streaming

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### Text Generation (Ollama)

```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Why is the sky blue?"
  }'
```

#### Text Completion (OpenAI)

```bash
curl http://localhost:11434/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Once upon a time"
  }'
```

#### Vision/Multimodal (OpenAI)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava:7b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }]
  }'
```

#### Tool/Function Calling (OpenAI)

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
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

#### Push Model

```bash
curl -X POST http://localhost:11434/api/push \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:3b", "destination": "https://registry.example.com"}'
```

#### Structured Output (JSON Schema)

```bash
# Constrain output to match a JSON Schema
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "List 3 colors with hex codes",
    "format": {
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
            "required": ["name", "hex"]
          }
        }
      },
      "required": ["colors"]
    }
  }'
```

#### Blob Management

```bash
# Check if blob exists
curl -I http://localhost:11434/api/blobs/sha256:abc123...

# Upload blob
curl -X POST http://localhost:11434/api/blobs/sha256:abc123... \
  --data-binary @model.gguf

# Download blob
curl http://localhost:11434/api/blobs/sha256:abc123... -o downloaded.gguf
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `a3s-power run <model> [--prompt <text>]` | Load model and start interactive chat, or send a single prompt |
| `a3s-power pull <name_or_url>` | Download a model by name (`llama3.2:3b`) or direct URL |
| `a3s-power push <model> --destination <url>` | Push a model to a remote registry |
| `a3s-power list` | List all locally available models |
| `a3s-power show <model>` | Show model details (format, size, parameters) |
| `a3s-power delete <model>` | Delete a model from local storage |
| `a3s-power create <name> -f <modelfile>` | Create a custom model from a Modelfile |
| `a3s-power cp <source> <destination>` | Copy/alias a model to a new name |
| `a3s-power ps` | List running (loaded) models on the server |
| `a3s-power stop <model>` | Stop (unload) a running model from the server |
| `a3s-power serve [--host <addr>] [--port <port>]` | Start HTTP server (default: `127.0.0.1:11434`) |

## Model Storage

Models are stored in `~/.a3s/power/` (override with `$A3S_POWER_HOME`):

```
~/.a3s/power/
├── config.toml              # User configuration
└── models/
    ├── manifests/           # JSON manifest files
    │   ├── llama-2-7b.json
    │   └── qwen2.5-7b.json
    └── blobs/               # Content-addressed model files
        ├── sha256-abc123...
        └── sha256-def456...
```

### Content-Addressed Storage

Model files are stored by their SHA-256 hash, enabling:
- **Deduplication**: Identical files share storage
- **Integrity verification**: Blobs can be verified against their hash
- **Clean deletion**: Remove manifest + blob independently

## Configuration

Configuration is read from `~/.a3s/power/config.toml`:

```toml
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
keep_alive = "5m"    # auto-unload idle models ("0"=immediate, "-1"=never, "5m", "1h")

[gpu]
gpu_layers = -1   # offload all layers to GPU (-1=all, 0=CPU only)
main_gpu = 0      # primary GPU index
```

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `127.0.0.1` | HTTP server bind address |
| `port` | `11434` | HTTP server port |
| `data_dir` | `~/.a3s/power` | Base directory for model storage |
| `max_loaded_models` | `1` | Maximum models loaded in memory concurrently |
| `keep_alive` | `"5m"` | Auto-unload idle models after this duration (`"0"`=immediate, `"-1"`=never, `"5m"`, `"1h"`, `"30s"`) |
| `gpu.gpu_layers` | `0` | Number of layers to offload to GPU (0=CPU, -1=all) |
| `gpu.main_gpu` | `0` | Index of the primary GPU to use |

All fields are optional and have sensible defaults.

### Environment Variables (Ollama-Compatible)

Environment variables override config file values for drop-in Ollama compatibility:

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLAMA_HOST` | Server bind address (`host:port` or `host`) | `0.0.0.0:11434` |
| `OLLAMA_MODELS` | Model storage directory | `/data/models` |
| `OLLAMA_KEEP_ALIVE` | Default keep-alive duration | `10m`, `-1`, `0` |
| `OLLAMA_MAX_LOADED_MODELS` | Max concurrent loaded models | `3` |
| `OLLAMA_NUM_GPU` | GPU layers to offload (-1 = all) | `-1` |
| `A3S_POWER_HOME` | Base directory for all Power data | `~/.a3s/power` |

`OLLAMA_HOST` supports scheme prefixes (e.g. `http://0.0.0.0:8080`).

## Feature Flags

| Flag | Description |
|------|-------------|
| `llamacpp` | Enable llama.cpp inference backend via `llama-cpp-2`. Requires a C++ compiler and CMake. |

Without any feature flags, Power can manage models (pull, list, delete) and serve API responses, but inference calls will return a "backend not available" error.

## Development

### Build Commands

```bash
# Build
cargo build -p a3s-power                          # Debug build
cargo build -p a3s-power --release                 # Release build
cargo build -p a3s-power --features llamacpp       # With llama.cpp

# Test
cargo test -p a3s-power --lib -- --test-threads=1  # All 878 tests

# Lint
cargo clippy -p a3s-power -- -D warnings           # Clippy
cargo fmt -p a3s-power -- --check                   # Format check

# Run
cargo run -p a3s-power -- list                      # CLI
cargo run -p a3s-power -- serve                     # Server
```

### Project Structure

```
power/
├── Cargo.toml
├── README.md
├── LICENSE
├── .gitignore
└── src/
    ├── main.rs              # Binary entry point (CLI dispatch)
    ├── lib.rs               # Library root (module re-exports)
    ├── error.rs             # PowerError enum + Result<T> alias
    ├── config.rs            # TOML configuration (host, port, data_dir)
    ├── dirs.rs              # Platform-specific paths (~/.a3s/power/)
    ├── cli/
    │   ├── mod.rs           # Cli struct + Commands enum (clap)
    │   ├── run.rs           # Interactive chat + single prompt
    │   ├── pull.rs          # Download with progress bar
    │   ├── push.rs          # Push model to remote registry
    │   ├── list.rs          # Tabular model listing
    │   ├── show.rs          # Model detail display
    │   ├── delete.rs        # Model + blob deletion
    │   ├── ps.rs            # List running models (queries server)
    │   ├── stop.rs          # Stop/unload a running model
    │   └── serve.rs         # HTTP server startup
    ├── model/
    │   ├── manifest.rs      # ModelManifest, ModelFormat, ModelParameters
    │   ├── registry.rs      # In-memory index backed by disk manifests
    │   ├── storage.rs       # Content-addressed blob store (SHA-256)
    │   ├── pull.rs          # HTTP download with progress callback
    │   ├── push.rs          # Push model to remote registry
    │   ├── resolve.rs       # Name-based model resolution (Ollama registry → built-in → HuggingFace)
    │   ├── ollama_registry.rs # Ollama registry client (fetch manifests, metadata, blob URLs)
    │   ├── modelfile.rs     # Modelfile parser (FROM, PARAMETER, SYSTEM, TEMPLATE, etc.)
    │   └── known_models.json# Built-in registry of popular GGUF models (offline fallback)
    ├── backend/
    │   ├── mod.rs           # Backend trait + BackendRegistry
    │   ├── types.rs         # Inference types (vision, tools, chat, completion, embedding)
    │   ├── llamacpp.rs      # llama.cpp backend (feature-gated, multi-model, KV cache reuse)
    │   ├── chat_template.rs # Chat template detection, Jinja2 rendering (minijinja), and fallback formatting
    │   ├── json_schema.rs  # JSON Schema → GBNF grammar converter for structured output
    │   ├── tool_parser.rs   # Tool call output parser (XML, Mistral, JSON formats)
    │   └── test_utils.rs    # MockBackend for testing
    ├── server/
    │   ├── mod.rs           # Server startup (bind, listen)
    │   ├── state.rs         # Shared AppState with LRU model tracking
    │   ├── router.rs        # Axum router with CORS + tracing + metrics
    │   └── metrics.rs       # Prometheus metrics collection and /metrics handler
    └── api/
        ├── autoload.rs      # Model auto-loading on first inference
        ├── health.rs        # GET /health endpoint
        ├── types.rs         # OpenAI + Ollama request/response types
        ├── sse.rs           # Streaming utilities (NDJSON for native API, SSE for OpenAI API)
        ├── native/
        │   ├── mod.rs       # Ollama-compatible route group
        │   ├── generate.rs  # POST /api/generate
        │   ├── chat.rs      # POST /api/chat (vision + tools)
        │   ├── models.rs    # GET /api/tags, POST /api/show, DELETE /api/delete
        │   ├── pull.rs      # POST /api/pull (streaming progress)
        │   ├── push.rs      # POST /api/push (push to registry)
        │   ├── blobs.rs     # HEAD/POST/GET /api/blobs/:digest
        │   ├── embeddings.rs# POST /api/embeddings
        │   ├── embed.rs     # POST /api/embed (batch embeddings)
        │   ├── ps.rs        # GET /api/ps (running models)
        │   ├── copy.rs      # POST /api/copy (model aliasing)
        │   ├── create.rs    # POST /api/create (from Modelfile)
        │   └── version.rs   # GET /api/version
        └── openai/
            ├── mod.rs       # OpenAI-compatible route group + shared helpers
            ├── chat.rs      # POST /v1/chat/completions
            ├── completions.rs # POST /v1/completions
            ├── models.rs    # GET /v1/models
            └── embeddings.rs# POST /v1/embeddings
```

## A3S Ecosystem

A3S Power is an **infrastructure component** of the A3S ecosystem — a standalone model server that enables local LLM inference for other A3S tools.

```
┌──────────────────────────────────────────────────────────┐
│                    A3S Ecosystem                          │
│                                                           │
│  Infrastructure:  a3s-box     (MicroVM sandbox runtime)   │
│                   a3s-power   (local model serving)       │
│                      │            ▲                        │
│  Application:     a3s-code    ────┘  (AI coding agent)    │
│                    /   \                                   │
│  Utilities:   a3s-lane  a3s-context                       │
│                         (memory/knowledge)                 │
│                                                           │
│               a3s-power ◄── You are here                  │
└──────────────────────────────────────────────────────────┘
```

| Project | Package | Relationship |
|---------|---------|--------------|
| **box** | `a3s-box-*` | Can use Power for local model inference |
| **code** | `a3s-code` | Uses Power as a local model backend |
| **lane** | `a3s-lane` | Independent utility (no direct relationship) |
| **context** | `a3s-context` | Independent utility (no direct relationship) |

**Standalone Usage**: `a3s-power` works independently as a local model server for any application:
- Drop-in Ollama replacement with identical API and NDJSON wire format
- Pull any model from Ollama registry by name (`llama3.2:3b`, `qwen2.5:7b`, etc.)
- OpenAI SDK compatible for seamless integration
- Local-first inference with no cloud dependency

## Roadmap

### Phase 1: Core ✅

- [x] CLI model management (pull, list, show, delete)
- [x] Content-addressed storage with SHA-256
- [x] Model manifest system with JSON persistence
- [x] TOML configuration
- [x] Platform-specific directory resolution
- [x] Comprehensive unit test foundation

### Phase 2: Backend & Inference ✅

- [x] Backend trait abstraction
- [x] llama.cpp backend via `llama-cpp-2` (feature-gated)
- [x] Streaming token generation via channels
- [x] Interactive chat with conversation history
- [x] Single prompt mode

### Phase 3: HTTP Server ✅

- [x] Axum-based HTTP server with CORS + tracing
- [x] Ollama-compatible native API (12 endpoints + blob management)
- [x] OpenAI-compatible API (4 endpoints)
- [x] SSE streaming for all inference endpoints
- [x] Non-streaming response collection

### Phase 4: Polish & Production ✅

- [x] Model registry resolution (name-based pulls with Ollama registry → built-in registry → HuggingFace fallback)
- [x] Embedding generation support (automatic reload with embedding mode)
- [x] Multiple concurrent model loading (HashMap storage with LRU eviction)
- [x] Model auto-loading on first API request
- [x] GPU acceleration configuration (`[gpu]` config with layer offloading)
- [x] Chat template auto-detection from GGUF metadata (ChatML, Llama, Phi, Generic)
- [x] Health check endpoint (`/health`)
- [x] Prometheus metrics endpoint (`/metrics` with request/token/model counters)

### Phase 5: Full Ollama Parity ✅

- [x] Vision/Multimodal support (`MessageContent` enum with text + image URL parts)
- [x] Tool/Function calling (tool definitions, tool choice, tool call responses)
- [x] Push API + CLI with streaming progress (`POST /api/push`, `a3s-power push`)
- [x] Blob management API (`HEAD/POST/GET/DELETE /api/blobs/:digest`)
- [x] Generate API: `system`, `template`, `raw`, `suffix`, `context`, `images` fields
- [x] Native chat `images` field (Ollama base64 format)
- [x] CLI `cp` command for model aliasing
- [x] New error variants (`UploadFailed`, `InvalidDigest`, `BlobNotFound`)

### Phase 6: Observability & Cost Tracking ✅

End-to-end observability for LLM inference:

- [x] **OpenTelemetry-Ready Metrics**: Instrument inference pipeline with Prometheus metrics
  - `power_inference_duration_seconds{model}` summary (count + sum)
  - `power_ttft_seconds{model}` summary (time to first token)
  - Per-model inference instrumentation across all 4 inference endpoints
- [x] **Token & Cost Metrics**: Per-call recording via Prometheus
  - `power_inference_tokens_total{model, type=input|output}` counter
  - `power_cost_dollars{model}` counter
  - `power_inference_duration_seconds{model}` summary
  - `power_ttft_seconds{model}` summary (time to first token)
- [x] **Cost Dashboard Data**: Aggregate cost by model / day
  - JSON export endpoint: `GET /v1/usage` with date range and model filter
- [x] **Model Lifecycle Metrics**: Load time, memory usage, eviction count
  - `power_model_load_duration_seconds{model}` summary
  - `power_model_memory_bytes{model}` gauge
  - `power_model_evictions_total` counter
- [x] **GPU Utilization Metrics**: GPU memory, compute utilization per device
  - `power_gpu_memory_bytes{device}` gauge
  - `power_gpu_utilization{device}` gauge

### Phase 7: Ollama Drop-in Compatibility ✅

Wire-format and runtime compatibility for seamless Ollama replacement:

- [x] **Ollama Registry Integration**: Pull any model from `registry.ollama.ai` by name — primary resolution source with template, system prompt, params, and license metadata
- [x] **NDJSON Streaming**: Native API endpoints (`/api/generate`, `/api/chat`, `/api/pull`, `/api/push`) stream as `application/x-ndjson` (Ollama wire format); OpenAI endpoints keep SSE
- [x] **Automatic Model Unloading**: Background keep_alive reaper checks every 5s and unloads idle models (configurable: `"5m"`, `"1h"`, `"0"`, `"-1"`)
- [x] **Context Token Return**: `/api/generate` returns token IDs in `context` field for conversation continuity
- [x] 878 comprehensive unit tests

### Phase 8: Advanced Compatibility ✅

- [x] **Jinja2/Go Template Engine**: Render arbitrary Jinja2 chat templates via `minijinja` (Llama 3, Gemma, ChatML, Phi, custom) with hardcoded fallback; prefers Ollama registry `template_override` over GGUF metadata
- [x] **KV Cache Reuse**: Persist `LlamaContext` across requests with prefix matching — skips re-evaluating shared prompt tokens for multi-turn conversation speedup
- [x] **Tool Call Parsing**: Parse model output into structured `tool_calls` — supports `<tool_call>` XML (Hermes/Qwen), `[TOOL_CALLS]` prefix (Mistral), and raw JSON formats; zero overhead when no tools in request
- [x] **JSON Schema Structured Output**: Support `format: {"type":"object","properties":{...}}` via JSON Schema → GBNF grammar conversion; accepts `"json"`, `{"type":"json_object"}`, or full JSON Schema objects
- [x] **Vision Inference**: Multimodal vision pipeline — accepts base64 images in Ollama `images` field and OpenAI `image_url` content parts; projector auto-downloaded from Ollama registry; uses llama.cpp `mtmd` API for image encoding when projector available
- [x] **ADAPTER Support**: LoRA/QLoRA adapter loading at inference time — Modelfile `ADAPTER` directive parsed, adapter file loaded via `llama_lora_adapter_init`, applied to context with `lora_adapter_set` at scale 1.0
- [x] **MESSAGE Directive**: Pre-seeded conversation history via Modelfile `MESSAGE` directive; messages stored in manifest and automatically prepended to chat requests
- [x] 878 comprehensive unit tests

### Phase 9: Operational Parity ✅

Runtime and CLI parity for production Ollama replacement:

- [x] **Default Port 11434**: Matches Ollama's default port for zero-config drop-in replacement
- [x] **`ps` CLI Command**: List running (loaded) models via `a3s-power ps` (queries server `GET /api/ps`)
- [x] **`stop` CLI Command**: Unload a running model via `a3s-power stop <model>` (sends `keep_alive: 0`)
- [x] **Ollama Environment Variables**: `OLLAMA_HOST`, `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_GPU` — override config file for container/script compatibility
- [x] **Download Resumption**: Interrupted model downloads resume automatically via HTTP Range requests with partial file tracking
- [x] 878 comprehensive unit tests

### Phase 10: Intelligence & Observability ✅

GPU auto-detection, memory estimation, verbose model inspection, and per-layer pull progress:

- [x] **GPU Auto-Detection**: Detect Apple Metal (via `system_profiler`) and NVIDIA CUDA (via `nvidia-smi`) GPUs at server startup; auto-set `gpu_layers = -1` when GPU available and user hasn't explicitly configured
- [x] **Memory Estimation**: Estimate VRAM requirements before loading (model weights + KV cache + compute overhead); log estimates to help users right-size their hardware
- [x] **GGUF Metadata Reader**: Lightweight binary parser for GGUF v2/v3 file headers — extracts all key-value metadata and tensor descriptors without loading weights into memory
- [x] **Verbose Show**: `/api/show` with `verbose: true` returns full GGUF metadata (architecture, context length, embedding dimensions, etc.) and tensor information (name, shape, type, element count)
- [x] **Per-Layer Pull Progress**: Streaming pull progress shows per-layer digest identifiers (`pulling sha256:abc123...`) matching Ollama's output format; resolves model before download to extract layer digests
- [x] 878 comprehensive unit tests

### Phase 11: Full Options Parity ✅

Complete Ollama generation options support and multi-GPU wiring:

- [x] **Missing Generation Options**: Added `repeat_last_n`, `penalize_newline`, `num_batch`, `num_thread`, `num_thread_batch`, `use_mmap`, `use_mlock`, `numa`, `flash_attention`, `num_gpu`, `main_gpu` to `GenerateOptions`
- [x] **Backend Wiring**: All new options flow through API → backend `CompletionRequest`/`ChatRequest` → llama.cpp context params and sampler
- [x] **Flash Attention**: Wired to `LlamaContextParams::with_flash_attention_policy(Enabled)` when `flash_attention: true`
- [x] **Multi-GPU**: `main_gpu` config wired to `LlamaModelParams::with_main_gpu()`; per-request `num_gpu`/`main_gpu` override supported
- [x] **Memory Lock**: `use_mlock` config wired to `LlamaModelParams::with_use_mlock(true)` to prevent model swapping
- [x] **Thread Control**: `num_thread` and `num_thread_batch` wired to `LlamaContextParams::with_n_threads()` and `with_n_threads_batch()`
- [x] **Batch Size**: `num_batch` wired to `LlamaContextParams::with_n_batch()`
- [x] **Repeat Penalty Window**: `repeat_last_n` wired to `LlamaSampler::penalties()` first argument (was hardcoded to 64)
- [x] **Config Extensions**: Added `use_mlock`, `num_thread`, `flash_attention` to `PowerConfig` with TOML support
- [x] 878 comprehensive unit tests

### Phase 12: CLI Run Options Parity ✅

Complete Ollama CLI `run` command options — all 14/14 options now implemented:

- [x] **`--format`**: JSON output format constraint (accepts `"json"` or JSON schema object)
- [x] **`--system`**: Override system prompt per session (prepended as system message)
- [x] **`--template`**: Override chat template (reserved for template engine integration)
- [x] **`--keep-alive`**: Model keep-alive duration (e.g. `"5m"`, `"1h"`, `"-1"` for never unload)
- [x] **`--verbose`**: Show timing and token statistics after each generation (prompt eval count/rate, eval count, total duration, tokens/s)
- [x] **`--insecure`**: Skip TLS verification flag for registry operations
- [x] 878 comprehensive unit tests

### Phase 13: Environment Variables & CLI Polish ✅

Complete Ollama environment variable parity and CLI enhancements:

- [x] **`OLLAMA_NUM_PARALLEL`**: Number of parallel request slots (concurrent inference)
- [x] **`OLLAMA_DEBUG`**: Enable debug logging (sets `RUST_LOG=debug` if not already set)
- [x] **`OLLAMA_ORIGINS`**: Custom CORS origins (comma-separated); empty = permissive
- [x] **`OLLAMA_FLASH_ATTENTION`**: Global flash attention override (`"1"` or `"true"`)
- [x] **`OLLAMA_TMPDIR`**: Custom temporary directory for downloads and scratch files
- [x] **CLI `show --verbose`**: Display full GGUF metadata (keys, values, tensor list) from CLI
- [x] **CLI `pull --insecure`**: Skip TLS verification for pull operations
- [x] **CLI `push --insecure`**: Skip TLS verification for push operations
- [x] **Interactive `/help`**: Show available slash commands in interactive chat
- [x] **Interactive `/clear`**: Clear conversation history (preserves system prompt)
- [x] **Interactive `/show`**: Display model name, message counts, and current settings
- [x] **Interactive `"""`**: Multi-line input support with triple-quote delimiters
- [x] **CORS Configuration**: Server respects `OLLAMA_ORIGINS` for restricted CORS; defaults to permissive
- [x] 878 comprehensive unit tests

### Phase 14: Final Ollama Parity ✅

Complete remaining Ollama feature gaps — `help` subcommand, blob pruning, GPU scheduling:

- [x] **`help` subcommand**: `a3s-power help [command]` prints help for any subcommand (replaces clap's built-in)
- [x] **Blob pruning**: `prune_unused_blobs()` removes orphaned blob files not referenced by any manifest; returns count and bytes freed
- [x] **`OLLAMA_NOPRUNE`**: Disable automatic blob pruning (`"1"` or `"true"`)
- [x] **`OLLAMA_SCHED_SPREAD`**: Spread model layers across all available GPUs (`"1"` or `"true"`)
- [x] 878 comprehensive unit tests

### Phase 15: Thinking & Reasoning 🚧

Critical for DeepSeek-R1, QwQ, and other reasoning models:

- [ ] **`think` parameter**: `ThinkValue` type (bool or `"high"/"medium"/"low"`) in generate/chat requests
- [ ] **`thinking` response field**: Separate thinking content from response in `Message.thinking` and `GenerateResponse.thinking`
- [ ] **Thinking parser**: Streaming parser that separates `<think>...</think>` blocks from content; infer tags from template
- [ ] **`run --think` CLI flag**: Enable thinking mode from interactive chat
- [ ] **`run --hidethinking` CLI flag**: Hide thinking output in CLI display
- [ ] **OpenAI `reasoning` / `reasoning_effort`**: Map to `think` parameter in `/v1/chat/completions`

### Phase 16: Logprobs & Context Control 🚧

Log probabilities and context window management:

- [ ] **`logprobs` / `top_logprobs`**: Return log probabilities in generate/chat responses with `Logprob`/`TokenLogprob` types
- [ ] **`truncate` field**: Truncate prompt when exceeding context length instead of erroring
- [ ] **`shift` field**: Shift context window when hitting limit instead of erroring
- [ ] **`OLLAMA_CONTEXT_LENGTH`**: Global default context length override env var
- [ ] **`OLLAMA_KV_CACHE_TYPE`**: KV cache quantization type (f16/q8_0/q4_0)

### Phase 17: OpenAI API Parity 🚧

Additional OpenAI-compatible endpoints and fields:

- [ ] **`GET /v1/models/:model`**: Retrieve single model details
- [ ] **`POST /v1/responses`**: OpenAI Responses API compatibility
- [ ] **`POST /v1/messages`**: Anthropic Messages API compatibility via middleware
- [ ] **`stream_options.include_usage`**: Return usage stats in final streaming chunk
- [ ] **`encoding_format`**: `"float"` or `"base64"` for embedding responses
- [ ] **`dimensions`**: Truncate output embeddings to specified dimension

### Phase 18: Create API & Model Management 🚧

Align with Ollama's new structured Create API:

- [ ] **Structured Create API**: Support `from`, `files`, `adapters`, `template`, `system`, `parameters`, `messages`, `license` fields (not just Modelfile)
- [ ] **Re-quantization**: Integrate llama.cpp quantization for `create --quantize`
- [ ] **SafeTensors conversion**: Convert SafeTensors → GGUF during create
- [ ] **ShowResponse fields**: Add `capabilities`, `renderer`, `parser`, `projector_info`, `messages`, `remote_model`, `remote_host`
- [ ] **ProcessResponse fields**: Add `size_vram`, `context_length` to `/api/ps`
- [ ] **`tool_calls` in GenerateResponse**: Return tool calls from `/api/generate` (not just `/api/chat`)

### Phase 19: Auth & Registry Push 🚧

Account management and registry push:

- [ ] **Registry push (OCI auth)**: Push to `registry.ollama.ai` with keypair-based auth
- [ ] **`signin` / `signout` CLI**: Sign in/out of ollama.com account
- [ ] **`POST /api/me`**: Whoami endpoint
- [ ] **`POST /api/signout`**: Signout endpoint

### Phase 20: Environment Variables & CLI Polish 🚧

Remaining env vars and CLI flags:

- [ ] **`OLLAMA_GPU_OVERHEAD`**: Reserve VRAM per GPU (bytes)
- [ ] **`OLLAMA_LOAD_TIMEOUT`**: Stall detection timeout for model loads
- [ ] **`OLLAMA_MAX_QUEUE`**: Maximum queued requests
- [ ] **`OLLAMA_NOHISTORY`**: Disable readline history
- [ ] **`OLLAMA_MULTIUSER_CACHE`**: Optimize prompt caching for multi-user
- [ ] **`OLLAMA_REMOTES`**: Allowed hosts for remote models
- [ ] **`show --license/--modelfile/--parameters/--template/--system`**: Show individual sections
- [ ] **`run --nowordwrap`**: Disable word wrapping in CLI
- [ ] **`run --truncate` / `--dimensions`**: Embedding-specific CLI flags
- [ ] **`_debug_render_only`**: Debug mode returning rendered template
- [ ] **`GET /` and `HEAD /`**: Return `"Ollama is running"` for compatibility checks
- [ ] **Request queuing**: Queue requests when all model slots busy (`OLLAMA_MAX_QUEUE`)
- [ ] **`num_parallel` wiring**: Wire to llama.cpp `n_parallel` for concurrent request slots

## License

MIT
