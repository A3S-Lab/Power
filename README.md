# A3S Power

Local model management and serving with OpenAI-compatible API.

Power is a CLI tool and HTTP server for downloading, managing, and running local LLM models. It provides both an Ollama-compatible native API and an OpenAI-compatible API, so existing tools and SDKs work out of the box.

## Features

- **CLI model management** - pull, list, show, delete models from the command line
- **Interactive chat** - multi-turn conversation with streaming token output
- **HTTP server** - axum-based server with CORS and tracing
- **Ollama-compatible API** - `/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`, etc.
- **OpenAI-compatible API** - `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **SSE streaming** - all inference and pull endpoints support server-sent events
- **Content-addressed storage** - model blobs stored by SHA-256 hash with deduplication
- **llama.cpp backend** - GGUF inference via `llama-cpp-2` (optional feature flag)

## Quick Start

```bash
# Build without inference backend (model management only)
cargo build -p a3s-power

# Build with llama.cpp inference (requires C++ toolchain)
cargo build -p a3s-power --features llamacpp

# Pull a model (direct URL)
a3s-power pull https://example.com/model.gguf

# List local models
a3s-power list

# Start interactive chat
a3s-power run my-model

# Single prompt
a3s-power run my-model --prompt "Explain quicksort"

# Start HTTP server
a3s-power serve
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `a3s-power run <model> [--prompt <text>]` | Load model and start interactive chat, or send a single prompt |
| `a3s-power pull <url>` | Download a model from a direct URL |
| `a3s-power list` | List all locally available models |
| `a3s-power show <model>` | Show model details (format, size, parameters) |
| `a3s-power delete <model>` | Delete a model from local storage |
| `a3s-power serve [--host <addr>] [--port <port>]` | Start HTTP server (default: `127.0.0.1:11435`) |

## API Endpoints

### Native API (Ollama-compatible)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Text generation (streaming/non-streaming) |
| `POST` | `/api/chat` | Chat completion (streaming/non-streaming) |
| `POST` | `/api/pull` | Download a model (streaming progress) |
| `GET` | `/api/tags` | List local models |
| `POST` | `/api/show` | Show model details |
| `DELETE` | `/api/delete` | Delete a model |
| `POST` | `/api/embeddings` | Generate embeddings |

### OpenAI-compatible API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming/non-streaming) |
| `POST` | `/v1/completions` | Text completion (streaming/non-streaming) |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/embeddings` | Generate embeddings |

### Examples

```bash
# Start the server
a3s-power serve

# List models (OpenAI-compatible)
curl http://localhost:11435/v1/models

# List models (Ollama-compatible)
curl http://localhost:11435/api/tags

# Chat completion (OpenAI-compatible)
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Chat completion with streaming
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# Text generation (Ollama-compatible)
curl http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Why is the sky blue?"
  }'
```

## Architecture

```
src/
├── main.rs              # Binary entry point
├── lib.rs               # Library root
├── error.rs             # PowerError enum
├── config.rs            # TOML configuration
├── dirs.rs              # Platform-specific paths (~/.a3s/power/)
├── cli/                 # CLI command handlers
├── model/
│   ├── manifest.rs      # ModelManifest, ModelFormat, ModelParameters
│   ├── registry.rs      # In-memory model index backed by disk
│   ├── storage.rs       # Content-addressed blob store (SHA-256)
│   └── pull.rs          # HTTP download with progress
├── backend/
│   ├── mod.rs           # Backend trait + BackendRegistry
│   ├── types.rs         # Inference request/response types
│   └── llamacpp.rs      # llama.cpp backend (feature-gated)
├── server/
│   ├── mod.rs           # Server startup
│   ├── state.rs         # Shared AppState
│   └── router.rs        # Axum router
└── api/
    ├── types.rs         # OpenAI + Ollama request/response types
    ├── sse.rs           # SSE streaming utilities
    ├── native/          # Ollama-compatible endpoints
    └── openai/          # OpenAI-compatible endpoints
```

## Model Storage

Models are stored in `~/.a3s/power/` (override with `$A3S_POWER_HOME`):

```
~/.a3s/power/
├── config.toml
└── models/
    ├── manifests/       # JSON manifest files
    └── blobs/           # Content-addressed model files (sha256-...)
```

## Configuration

Configuration is read from `~/.a3s/power/config.toml`:

```toml
host = "127.0.0.1"
port = 11435
max_loaded_models = 1
```

All fields are optional and have sensible defaults.

## Feature Flags

| Flag | Description |
|------|-------------|
| `llamacpp` | Enable llama.cpp inference backend via `llama-cpp-2`. Requires a C++ compiler and CMake. |

Without any feature flags, Power can manage models (pull, list, delete) and serve API responses, but inference calls will return a "backend not available" error.

## Tests

```bash
# Run all unit tests (38 tests)
cargo test -p a3s-power --lib -- --test-threads=1
```

## License

MIT
