# Changelog

All notable changes to A3S Power will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-21

### Added

- **3 inference backends**: mistralrs (pure Rust, default), llamacpp (C++ bindings), picolm (experimental, layer-streaming for TEE)
- **OpenAI-compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/attestation`
- **TEE security stack**: AMD SEV-SNP / Intel TDX attestation (ioctl), AES-256-GCM model encryption (3 modes: file/RAM/streaming), Ed25519 model signatures, log redaction, timing padding, EPC-aware backend routing
- **Model management**: content-addressed blob store (SHA-256), HuggingFace Hub pull with Range resume, GGUF metadata reader with memory estimation
- **Server infrastructure**: Axum HTTP server, TLS (RA-TLS with attestation X.509 extension), Vsock transport, API key auth (constant-time SHA-256), rate limiting, Prometheus metrics (16 groups), structured audit logging (plaintext/encrypted/async)
- **Client-side verification SDK**: nonce binding, model hash binding, measurement check, AMD KDS / Intel PCS hardware signature verification
- **Key management**: static and rotating key providers, SIGHUP-triggered key rotation
- **Privacy**: 10 sensitive JSON keys redacted, error sanitization, SensitiveString (auto-zeroize), token count rounding
- **Configuration**: HCL-first with `A3S_POWER_*` env var overrides
- **CLI**: `a3s-power` server binary with `--version`/`--help`, `a3s-power-verify` for offline attestation verification
- **GPU**: Metal + CUDA auto-detection, automatic gpu_layers configuration, VRAM metrics
- **Chat templates**: Jinja2 rendering (ChatML/Llama/Phi/Generic) with fuel-limited execution
- **Tool calling**: streaming parser for XML/Hermes, Mistral, and raw JSON formats
- **Reasoning**: streaming `<think>` block extraction for DeepSeek-R1/QwQ models
- **Structured output**: JSON Schema → GBNF grammar conversion for constrained generation
- **787+ unit tests** covering all modules

### Known Limitations

- **picolm backend is experimental**: forward pass uses stub arithmetic (not real transformer ops), tokenizer uses byte-fallback (not BPE). Infrastructure is production-ready but inference output is placeholder.
- **llamacpp vision**: URL-based images not supported (base64 data URIs work)
- **GPU utilization metric**: reports detected VRAM at startup but no real-time utilization polling (requires NVML/ROCm)

## [0.1.0] - 2025-12-01

### Added

- Initial release with mistralrs backend and basic model management.
