# A3S Power

<p align="center">
  <strong>Local Model Management & Serving</strong>
</p>

<p align="center">
  <em>Infrastructure layer — CLI + HTTP server for downloading, managing, and running local LLM models</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
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
- **Vision/Multimodal Support**: Accept image URLs in chat messages (OpenAI-compatible `content` array format)
- **Tool/Function Calling**: Structured tool definitions, tool choice, and tool call responses (OpenAI-compatible)
- **Chat Template Auto-Detection**: Detects ChatML, Llama, Phi, and Generic templates from GGUF metadata
- **Jinja2 Template Engine**: Renders arbitrary Jinja2 chat templates via `minijinja` (Llama 3, Gemma, ChatML, Phi, custom) with hardcoded fallback
- **KV Cache Reuse**: Persists `LlamaContext` across requests with prefix matching — skips re-evaluating shared prompt tokens for multi-turn speedup
- **Tool Call Parsing**: Parses model output into structured `tool_calls` — supports `<tool_call>` XML, `[TOOL_CALLS]` prefix, and raw JSON formats
- **Multiple Concurrent Models**: Load multiple models with LRU eviction at configurable capacity
- **Automatic Model Unloading**: Background keep_alive reaper unloads idle models after configurable timeout (default 5m)
- **GPU Acceleration**: Configurable GPU layer offloading via `[gpu]` config section
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
- **Content-Addressed Storage**: Model blobs stored by SHA-256 hash with automatic deduplication
- **llama.cpp Backend**: GGUF inference via `llama-cpp-2` Rust bindings (optional feature flag)
- **Health Check**: `GET /health` endpoint with uptime, version, and loaded model count
- **Model Auto-Loading**: Models are automatically loaded on first inference request with LRU eviction
- **TOML Configuration**: User-configurable host, port, GPU settings, keep_alive, and storage settings
- **Async-First**: Built on Tokio for high-performance async operations

## Quality Metrics

### Test Coverage

**400 unit tests** with comprehensive coverage across 50+ source files:

| Module | Lines | Coverage | Functions | Coverage |
|--------|-------|----------|-----------|----------|
| api/health.rs | 62 | 100.00% | 10 | 100.00% |
| api/native/mod.rs | 22 | 100.00% | 1 | 100.00% |
| api/native/ps.rs | 71 | 100.00% | 11 | 100.00% |
| api/native/version.rs | 21 | 100.00% | 6 | 100.00% |
| api/openai/mod.rs | 30 | 100.00% | 4 | 100.00% |
| api/openai/usage.rs | 238 | 100.00% | 18 | 100.00% |
| backend/llamacpp.rs | 143 | 100.00% | 23 | 100.00% |
| backend/test_utils.rs | 114 | 100.00% | 18 | 100.00% |
| error.rs | 93 | 100.00% | 19 | 100.00% |
| model/manifest.rs | 79 | 100.00% | 12 | 100.00% |
| server/router.rs | 107 | 100.00% | 16 | 100.00% |
| api/types.rs | 406 | 99.75% | 33 | 100.00% |
| api/native/embeddings.rs | 111 | 99.10% | 7 | 100.00% |
| server/metrics.rs | 607 | 98.35% | 54 | 96.30% |
| api/native/models.rs | 170 | 98.82% | 18 | 100.00% |
| backend/mod.rs | 65 | 98.46% | 15 | 100.00% |
| dirs.rs | 55 | 98.18% | 12 | 91.67% |
| backend/types.rs | 226 | 97.79% | 20 | 95.00% |
| api/sse.rs | 37 | 97.30% | 9 | 88.89% |
| backend/chat_template.rs | 195 | 96.92% | 19 | 100.00% |
| api/openai/embeddings.rs | 187 | 95.72% | 9 | 100.00% |
| api/native/create.rs | 112 | 95.54% | 7 | 100.00% |
| api/native/blobs.rs | 210 | 94.76% | 15 | 100.00% |
| server/state.rs | 186 | 94.62% | 26 | 92.31% |
| config.rs | 184 | 92.93% | 26 | 96.15% |
| api/autoload.rs | 148 | 91.89% | 14 | 100.00% |
| api/native/copy.rs | 60 | 91.67% | 6 | 100.00% |
| cli/mod.rs | 106 | 90.57% | 12 | 100.00% |
| model/modelfile.rs | 171 | 89.47% | 13 | 100.00% |
| model/storage.rs | 127 | 88.98% | 15 | 86.67% |
| api/openai/models.rs | 66 | 87.88% | 6 | 100.00% |
| model/registry.rs | 248 | 85.08% | 36 | 80.56% |
| api/openai/chat.rs | 447 | 83.89% | 19 | 68.42% |
| api/native/generate.rs | 475 | 82.95% | 20 | 75.00% |
| api/native/embed.rs | 77 | 81.82% | 7 | 100.00% |
| api/native/chat.rs | 497 | 79.68% | 21 | 71.43% |
| api/openai/completions.rs | 317 | 78.55% | 12 | 66.67% |
| api/native/push.rs | 150 | 66.67% | 9 | 66.67% |
| model/resolve.rs | 187 | 65.78% | 27 | 66.67% |
| model/push.rs | 112 | 50.00% | 17 | 70.59% |
| model/pull.rs | 116 | 46.55% | 17 | 64.71% |
| cli/delete.rs | 7 | 0.00% | 1 | 0.00% |
| cli/list.rs | 21 | 0.00% | 1 | 0.00% |
| cli/pull.rs | 31 | 0.00% | 3 | 0.00% |
| cli/push.rs | 15 | 0.00% | 3 | 0.00% |
| cli/run.rs | 168 | 0.00% | 6 | 0.00% |
| cli/serve.rs | 10 | 0.00% | 2 | 0.00% |
| cli/show.rs | 31 | 0.00% | 1 | 0.00% |
| server/mod.rs | 24 | 0.00% | 4 | 0.00% |
| api/native/pull.rs | 108 | 0.00% | 5 | 0.00% |
| **TOTAL** | **7450** | **84.40%** | **685** | **87.30%** |

Run coverage report:
```bash
just test-cov    # Pretty coverage with module breakdown
just cov-table   # Detailed file-by-file table
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

## Quick Start

### Build

```bash
# Build without inference backend (model management only)
cargo build -p a3s-power

# Build with llama.cpp inference (requires C++ compiler + CMake)
cargo build -p a3s-power --features llamacpp
```

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
# Start server on default port (127.0.0.1:11435)
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
curl http://localhost:11435/v1/models

# Ollama-compatible
curl http://localhost:11435/api/tags
```

#### Chat Completion (OpenAI)

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Chat Completion with Streaming

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### Text Generation (Ollama)

```bash
curl http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Why is the sky blue?"
  }'
```

#### Text Completion (OpenAI)

```bash
curl http://localhost:11435/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Once upon a time"
  }'
```

#### Vision/Multimodal (OpenAI)

```bash
curl http://localhost:11435/v1/chat/completions \
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
curl http://localhost:11435/v1/chat/completions \
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
curl -X POST http://localhost:11435/api/push \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:3b", "destination": "https://registry.example.com"}'
```

#### Blob Management

```bash
# Check if blob exists
curl -I http://localhost:11435/api/blobs/sha256:abc123...

# Upload blob
curl -X POST http://localhost:11435/api/blobs/sha256:abc123... \
  --data-binary @model.gguf

# Download blob
curl http://localhost:11435/api/blobs/sha256:abc123... -o downloaded.gguf
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
| `a3s-power serve [--host <addr>] [--port <port>]` | Start HTTP server (default: `127.0.0.1:11435`) |

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
port = 11435
max_loaded_models = 1
keep_alive = "5m"    # auto-unload idle models ("0"=immediate, "-1"=never, "5m", "1h")

[gpu]
gpu_layers = -1   # offload all layers to GPU (-1=all, 0=CPU only)
main_gpu = 0      # primary GPU index
```

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `127.0.0.1` | HTTP server bind address |
| `port` | `11435` | HTTP server port |
| `data_dir` | `~/.a3s/power` | Base directory for model storage |
| `max_loaded_models` | `1` | Maximum models loaded in memory concurrently |
| `keep_alive` | `"5m"` | Auto-unload idle models after this duration (`"0"`=immediate, `"-1"`=never, `"5m"`, `"1h"`, `"30s"`) |
| `gpu.gpu_layers` | `0` | Number of layers to offload to GPU (0=CPU, -1=all) |
| `gpu.main_gpu` | `0` | Index of the primary GPU to use |

All fields are optional and have sensible defaults.

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
cargo test -p a3s-power --lib -- --test-threads=1  # All 400 tests

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

- [x] Model registry resolution (name-based pulls with built-in registry + HuggingFace fallback)
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
- [x] 400 comprehensive unit tests

### Phase 8: Advanced Compatibility 🚧

- [x] **Jinja2/Go Template Engine**: Render arbitrary Jinja2 chat templates via `minijinja` (Llama 3, Gemma, ChatML, Phi, custom) with hardcoded fallback; prefers Ollama registry `template_override` over GGUF metadata
- [x] **KV Cache Reuse**: Persist `LlamaContext` across requests with prefix matching — skips re-evaluating shared prompt tokens for multi-turn conversation speedup
- [x] **Tool Call Parsing**: Parse model output into structured `tool_calls` — supports `<tool_call>` XML (Hermes/Qwen), `[TOOL_CALLS]` prefix (Mistral), and raw JSON formats; zero overhead when no tools in request
- [ ] **JSON Schema Structured Output**: Support `format: {"type":"object","properties":{...}}` via GBNF grammar generation
- [ ] **Vision Inference**: Actual multimodal inference with llava/clip projector support
- [ ] **ADAPTER Support**: Load LoRA/QLoRA adapters via Modelfile `ADAPTER` directive
- [ ] **MESSAGE Directive**: Pre-seeded conversation history in Modelfile

## License

MIT
