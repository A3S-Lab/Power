---
title: Operations
description: A3S Power backend capabilities, build profiles, service endpoints, artifact storage, and production boundaries.
---

# Operations

Backends are capability providers behind Power's shared execution and evidence
contracts. Choose them by model format, platform, and trust boundary rather
than treating the backend name as the architecture.

## Build profiles

| Feature | Role | Native dependency |
| --- | --- | --- |
| `mistralrs` | Default Candle-based GGUF, SafeTensors, vision, and embedding backend | No C++ inference engine |
| `llamacpp` | Mature GGUF backend with native MTP support | CMake, C++ compiler, and libclang |
| `llamacpp-cuda` | CUDA execution for llama.cpp | CUDA toolkit |
| `llamacpp-external-draft` | Verified external DFlash or DSpark execution; typed, fail-closed DFlash2 admission pending a binding update | Reviewed external-draft patch to pinned llama-cpp-rs source |
| `llamacpp-mtp-fr` | Experimental reduced-vocabulary MTP draft projection | Reviewed patch to pinned llama.cpp source |
| `picolm` | Pure-Rust, layer-streaming GGUF backend for constrained TEE memory | No C/C++ inference engine |
| `embedded-cuda` / `embedded-metal` | Accelerators for model-owned embedded graphs | Platform toolkit |
| `tls` / `vsock` | RA-TLS and A3S Box guest-host transports | Platform-specific |
| `hw-verify` | AMD SEV-SNP verification; Intel TDX fails closed pending DCAP Quote/QVL support | Platform crypto dependencies and AMD KDS access |

```bash
# Default hosted service
cargo build --release

# Listener-free embedded runtime
cargo build --release --no-default-features --features embedded-inference

# Pure-Rust layer-streaming TEE service
cargo build --release --no-default-features --features tee-minimal

# llama.cpp with CUDA
cargo build --release --no-default-features --features llamacpp-cuda

# Strict verifier with confidential release promotion
cargo build --locked --release --no-default-features \
  --features server,embedded-inference,hw-verify \
  --bin a3s-power-verify
```

`llamacpp-external-draft` and `llamacpp-mtp-fr` are intentionally separate from
the ordinary `llamacpp` profile because they expose reviewed additions to the
pinned source. The MTP-FR profile includes the external-draft binding so the
documented CUDA benchmark build command remains sufficient. Ordinary
`llamacpp` builds require neither patch.

## Service endpoints

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

Health and model inspection endpoints expose effective, non-secret settings so
benchmark and deployment automation can reject configuration drift.

## Keyed prompt-prefix reuse

Add `prompt_cache_key` to a text chat or completion request when later
requests will share a long prefix. The llama.cpp path reuses only a token prefix
whose KV and recurrent state can roll back exactly, then evaluates the suffix;
an unprovable hybrid-state rollback becomes a measured miss. mistral.rs,
picolm, proxy, and multimodal
requests currently return `prompt_cache_unsupported`; the field is never
silently ignored.

```text
prompt_cache_max_entries = 1
prompt_cache_ttl_seconds = 300
```

Exact cache benchmarks require both `redact_logs = false` and
`suppress_token_metrics = false` in an isolated process. Log redaction
deliberately activates metric suppression; `/health` reports the effective
policy so the benchmark client fails before accepting rounded evidence.

Power hashes and scopes each key by authenticated identity, endpoint, and model.
The raw key is not stored in the backend or receipt. `/health` reports support
and bounds; `/metrics` reports requests, hits, misses, reused/evaluated tokens,
evictions, and resident entries. Opted-in completion streams expose backend
prompt-evaluation time separately from TTFT; the canonical benchmark client
checks both timings against exact miss/hit counter deltas.

The checked-in RTX 4090 Q6_K capture measured five cold/warm pairs: median
backend prefill fell from 786.1375 ms to 33.4102 ms (23.5299x), median TTFT fell
from 950.0142 ms to 72.1932 ms (13.1593x), and 9,740 prompt tokens were reused.
[Inspect the raw report and replay commands](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/prompt-cache).

Native llama.cpp MTP does not yet share a state transaction with a cached
context. Explicit MTP plus a cache key fails closed; `auto` selects exact
target-only decoding. Prefix caching improves repeated prefill and TTFT, not
steady decode token/s. See the [canonical cache contract](https://github.com/A3S-Lab/Power/blob/main/docs/prompt-prefix-cache.md).

## Artifact installation

The artifact provisioner requires an expected filename, maximum byte length,
and SHA-256 digest. It streams into a private staging file, verifies the exact
bytes, and commits atomically under a cross-process lock. Offline policy fails
closed when a previously verified artifact is unavailable.

The hosted model store is content-addressed under `~/.a3s/power` by default.
Model aliases point to manifests rather than weakening blob identity.

GGUF registration accepts typed `adapter`, `projector`, and `external_draft`
locations. Power measures size and SHA-256 itself, verifies the exact bytes
again before load, and binds the portable auxiliary-artifacts identity into
attestation and request receipts. Strict TEE mode rejects legacy path-only
adapter and projector references.

## Production boundaries

- Bind development servers to loopback unless a reviewed transport policy says
  otherwise.
- Use RA-TLS or vsock deliberately; constructing the embedded runtime never
  chooses a transport for the caller.
- Treat simulated TEE mode as development-only.
- Do not claim confidential GPU execution from CPU TEE placement alone.
- Promote confidential release captures only with the opaque proof returned by
  strict confidential-GPU verification; raw reports are evidence inputs, not
  authorization tokens.
- Preserve raw report fields when saving attestation evidence.
- Preserve unchanged NVIDIA evidence and verdict bytes, and use the
  [external capture workflow](https://github.com/A3S-Lab/Power/blob/main/docs/external-release-capture.md)
  for strict `--promote-capture` release evidence.
- Treat mixed quantization and vocabulary-reduced drafting as quality-gated,
  workload-specific techniques.
- Keep model bytes, ACL, binary hashes, drivers, and host controls with every
  performance acceptance record.

## Supply chain and storage

The pure-Rust `tee-minimal` path reduces native inference dependencies, while
the llama.cpp path trades a larger native toolchain for mature GGUF and CUDA
capabilities. Audit the feature profile that will actually ship.

- [Supply-chain audit](https://github.com/A3S-Lab/Power/blob/main/docs/supply-chain.md)
- [Verified storage benchmark protocol](https://github.com/A3S-Lab/Power/blob/main/docs/storage-benchmark.md)
- [Project roadmap](https://github.com/A3S-Lab/Power/blob/main/ROADMAP.md)
- [Release changelog](https://github.com/A3S-Lab/Power/blob/main/CHANGELOG.md)

For Rust API types and feature flags, use
[docs.rs/a3s-power](https://docs.rs/a3s-power).
