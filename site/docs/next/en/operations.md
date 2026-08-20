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
| `llamacpp-mtp-fr` | Experimental reduced-vocabulary MTP draft projection | Reviewed patch to pinned llama.cpp source |
| `picolm` | Pure-Rust, layer-streaming GGUF backend for constrained TEE memory | No C/C++ inference engine |
| `embedded-cuda` / `embedded-metal` | Accelerators for model-owned embedded graphs | Platform toolkit |
| `tls` / `vsock` | RA-TLS and A3S Box guest-host transports | Platform-specific |
| `hw-verify` | AMD KDS and Intel PCS signature verification | Platform crypto dependencies |

```bash
# Default hosted service
cargo build --release

# Listener-free embedded runtime
cargo build --release --no-default-features --features embedded-inference

# Pure-Rust layer-streaming TEE service
cargo build --release --no-default-features --features tee-minimal

# llama.cpp with CUDA
cargo build --release --no-default-features --features llamacpp-cuda
```

The `llamacpp-mtp-fr` profile is intentionally separate because it modifies the
pinned source. Ordinary `llamacpp` builds do not need the experimental patch.

## Service endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness, loaded models, backend capabilities, and TEE status |
| `POST` | `/v1/chat/completions` | Chat, tools, structured output, vision, and SSE streaming |
| `POST` | `/v1/completions` | Text completion and SSE streaming |
| `POST` | `/v1/embeddings` | Embedding inference |
| `GET` | `/v1/models` | Registered models |
| `POST` | `/v1/models/pull` | Resumable ModelScope or Hugging Face pull |
| `GET` | `/v1/attestation` | Nonce- and model-bound TEE evidence |
| `GET` | `/metrics` | Prometheus metrics |

Health and model inspection endpoints expose effective, non-secret settings so
benchmark and deployment automation can reject configuration drift.

## Artifact installation

The artifact provisioner requires an expected filename, maximum byte length,
and SHA-256 digest. It streams into a private staging file, verifies the exact
bytes, and commits atomically under a cross-process lock. Offline policy fails
closed when a previously verified artifact is unavailable.

The hosted model store is content-addressed under `~/.a3s/power` by default.
Model aliases point to manifests rather than weakening blob identity.

## Production boundaries

- Bind development servers to loopback unless a reviewed transport policy says
  otherwise.
- Use RA-TLS or vsock deliberately; constructing the embedded runtime never
  chooses a transport for the caller.
- Treat simulated TEE mode as development-only.
- Do not claim confidential GPU execution from CPU TEE placement alone.
- Preserve raw report fields when saving attestation evidence.
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
