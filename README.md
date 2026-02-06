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
# Pull a model
a3s-power pull https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Interactive chat
a3s-power run llama-2-7b.Q4_K_M

# Single prompt
a3s-power run llama-2-7b.Q4_K_M --prompt "Explain quicksort in one paragraph"

# Start HTTP server
a3s-power serve
```

## Features

- **CLI Model Management**: Pull, list, show, and delete models from the command line
- **Interactive Chat**: Multi-turn conversation with streaming token output
- **HTTP Server**: Axum-based server with CORS and tracing middleware
- **Ollama-Compatible API**: `/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`, `/api/show`, `/api/delete`, `/api/embeddings`
- **OpenAI-Compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **SSE Streaming**: All inference and pull endpoints support server-sent events
- **Content-Addressed Storage**: Model blobs stored by SHA-256 hash with automatic deduplication
- **llama.cpp Backend**: GGUF inference via `llama-cpp-2` Rust bindings (optional feature flag)
- **TOML Configuration**: User-configurable host, port, and storage settings
- **Async-First**: Built on Tokio for high-performance async operations

## Quality Metrics

### Test Coverage

**96 unit tests** with **50.6% line coverage** / **67.2% function coverage** (via `cargo llvm-cov`):

| File | Line Coverage | Function Coverage |
|------|--------------|-------------------|
| `api/types.rs` | 100.0% | 100.0% |
| `api/openai/mod.rs` | 100.0% | 100.0% |
| `api/native/mod.rs` | 100.0% | 100.0% |
| `api/sse.rs` | 72.0% | 66.7% |
| `api/openai/models.rs` | 51.9% | 66.7% |
| `api/native/models.rs` | 14.9% | 18.2% |
| `backend/llamacpp.rs` | 100.0% | 100.0% |
| `backend/types.rs` | 100.0% | 100.0% |
| `backend/mod.rs` | 83.7% | 83.3% |
| `model/manifest.rs` | 100.0% | 100.0% |
| `model/registry.rs` | 77.5% | 70.0% |
| `model/storage.rs` | 78.4% | 75.0% |
| `model/pull.rs` | 51.4% | 64.7% |
| `config.rs` | 88.6% | 92.3% |
| `dirs.rs` | 88.5% | 81.8% |
| `error.rs` | 100.0% | 100.0% |
| `server/router.rs` | 100.0% | 100.0% |
| `server/state.rs` | 100.0% | 100.0% |
| **TOTAL** | **50.6%** | **67.2%** |

> CLI handlers (`cli/*`), HTTP handlers (`api/native/{chat,generate,pull,embeddings}.rs`, `api/openai/{chat,completions,embeddings}.rs`), and `server/mod.rs` have 0% coverage — these require integration tests with live backends and are excluded from the unit-test library target.

Run tests:
```bash
cargo test -p a3s-power --lib -- --test-threads=1
```

Run coverage:
```bash
cargo llvm-cov -p a3s-power --lib --summary-only -- --test-threads=1
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│                  a3s-power                       │
│                                                  │
│  CLI Layer                                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │ run  │ │ pull │ │ list │ │ show │ │serve │ │
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
# Pull a model from a direct URL
a3s-power pull https://example.com/model.gguf

# List local models
a3s-power list

# Show model details
a3s-power show my-model

# Delete a model
a3s-power delete my-model
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

### Native API (Ollama-Compatible)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Text generation (streaming/non-streaming) |
| `POST` | `/api/chat` | Chat completion (streaming/non-streaming) |
| `POST` | `/api/pull` | Download a model (streaming progress) |
| `GET` | `/api/tags` | List local models |
| `POST` | `/api/show` | Show model details |
| `DELETE` | `/api/delete` | Delete a model |
| `POST` | `/api/embeddings` | Generate embeddings |

### OpenAI-Compatible API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming/non-streaming) |
| `POST` | `/v1/completions` | Text completion (streaming/non-streaming) |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/embeddings` | Generate embeddings |

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

### CLI Commands

| Command | Description |
|---------|-------------|
| `a3s-power run <model> [--prompt <text>]` | Load model and start interactive chat, or send a single prompt |
| `a3s-power pull <url>` | Download a model from a direct URL |
| `a3s-power list` | List all locally available models |
| `a3s-power show <model>` | Show model details (format, size, parameters) |
| `a3s-power delete <model>` | Delete a model from local storage |
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
```

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `127.0.0.1` | HTTP server bind address |
| `port` | `11435` | HTTP server port |
| `data_dir` | `~/.a3s/power` | Base directory for model storage |
| `max_loaded_models` | `1` | Maximum models loaded in memory |

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
cargo test -p a3s-power --lib -- --test-threads=1  # All 96 tests

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
    │   ├── list.rs          # Tabular model listing
    │   ├── show.rs          # Model detail display
    │   ├── delete.rs        # Model + blob deletion
    │   └── serve.rs         # HTTP server startup
    ├── model/
    │   ├── manifest.rs      # ModelManifest, ModelFormat, ModelParameters
    │   ├── registry.rs      # In-memory index backed by disk manifests
    │   ├── storage.rs       # Content-addressed blob store (SHA-256)
    │   └── pull.rs          # HTTP download with progress callback
    ├── backend/
    │   ├── mod.rs           # Backend trait + BackendRegistry
    │   ├── types.rs         # Inference request/response types
    │   └── llamacpp.rs      # llama.cpp backend (feature-gated)
    ├── server/
    │   ├── mod.rs           # Server startup (bind, listen)
    │   ├── state.rs         # Shared AppState
    │   └── router.rs        # Axum router with CORS + tracing
    └── api/
        ├── types.rs         # OpenAI + Ollama request/response types
        ├── sse.rs           # SSE streaming utilities
        ├── native/
        │   ├── mod.rs       # Ollama-compatible route group
        │   ├── generate.rs  # POST /api/generate
        │   ├── chat.rs      # POST /api/chat
        │   ├── models.rs    # GET /api/tags, POST /api/show, DELETE /api/delete
        │   ├── pull.rs      # POST /api/pull (streaming progress)
        │   └── embeddings.rs# POST /api/embeddings
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
- Drop-in Ollama replacement with identical API
- OpenAI SDK compatible for seamless integration
- Local-first inference with no cloud dependency

## Roadmap

### Phase 1: Core ✅

- [x] CLI model management (pull, list, show, delete)
- [x] Content-addressed storage with SHA-256
- [x] Model manifest system with JSON persistence
- [x] TOML configuration
- [x] Platform-specific directory resolution
- [x] 96 comprehensive unit tests

### Phase 2: Backend & Inference ✅

- [x] Backend trait abstraction
- [x] llama.cpp backend via `llama-cpp-2` (feature-gated)
- [x] Streaming token generation via channels
- [x] Interactive chat with conversation history
- [x] Single prompt mode

### Phase 3: HTTP Server ✅

- [x] Axum-based HTTP server with CORS + tracing
- [x] Ollama-compatible native API (7 endpoints)
- [x] OpenAI-compatible API (4 endpoints)
- [x] SSE streaming for all inference endpoints
- [x] Non-streaming response collection

### Phase 4: Polish & Production 🚧

- [ ] Model registry resolution (name-based pulls, not just URLs)
- [ ] Embedding generation support (model loaded with embedding mode)
- [ ] Multiple concurrent model loading
- [ ] Model auto-loading on first API request
- [ ] GPU acceleration configuration
- [ ] Chat template auto-detection from GGUF metadata
- [ ] Health check endpoint (`/health`)
- [ ] Prometheus metrics endpoint

## License

MIT
