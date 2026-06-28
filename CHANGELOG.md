# Changelog

All notable changes to A3S Power will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Proxy backend — front any OpenAI-compatible upstream (vLLM / TGI / SGLang / OpenAI).** Configure `proxy_upstreams` (model name → base URL) and Power registers each as a `ModelFormat::Remote` model, forwarding chat (streamed), completions and embeddings to the upstream while applying its own routing, auth, rate-limiting and log-redaction. This lets Power *replace vLLM in the stack* without reimplementing CUDA kernels — it absorbs the accelerated engine as a swappable backend. Trust boundary: proxied inference runs on the upstream, outside any TEE (non-confidential fast path; no hardware attestation over proxied content).

- **vLLM-style admission control for `max_concurrent_requests`.** Concurrency limiting moved out of the rate-limit middleware into a `ConcurrencyLimiter` (Tokio semaphore) inside the inference handlers. Excess requests now **queue** for a permit (backpressure, like vLLM's `max_num_seqs`) instead of being rejected with `429`, and the permit is held across the whole streamed response body — released on completion *or* early client disconnect. New `power_requests_waiting` and `power_requests_running` Prometheus gauges (vLLM-style `num_requests_waiting` / `num_requests_running` observability). The per-second `rate_limit_rps` token bucket still returns `429`.

### Fixed

- Fixed `max_concurrent_requests` not actually bounding streaming generation: the old rate-limit middleware released the concurrency slot when the handler returned the response, which for streaming (SSE) happens *before* the body is generated — so concurrent streamed completions were effectively uncapped. The new handler-level permit spans the full stream.

- **Selectable speculative-decoding modes for picolm** via `spec_mode` config (env `A3S_POWER_SPEC_MODE`): `off` (plain autoregressive), `prompt-lookup` (default, suffix n-gram matched against the prompt), and `ngram-context` (DSpark-like self-speculation — an online n-gram model over the full running sequence, so free-form generation is accelerated too, not just input-overlapping output). A new `Drafter` trait is the seam where a trained draft head can drop in later.
- **Batched layer-streaming speculative verify.** Draft blocks are now verified in a single layer-outer/token-inner pass (each layer's weights loaded once for the whole block) instead of re-streaming every layer per draft token — turning K drafts into ~one weight-streaming pass on the memory-bandwidth-bound path. Acceptance uses lossless rejection sampling (respecting temperature/top-p/penalties), so output matches plain decoding for the same seed.
- **Adaptive draft length** (DSpark's load-aware-scheduler analogue): an EMA of the per-round acceptance ratio grows the draft block when speculation pays off and shrinks it on a bad streak, bounding wasted verify work.

### Fixed

- Fixed a picolm speculative-decoding correctness bug: on *partial* draft acceptance the carried-forward hidden state desynced from the truncated KV cache (it was left as the last draft's output regardless of how many drafts were accepted). The new verify forwards the accepted prefix plus a lossless correction token, keeping hidden state and KV cache consistent.
- Made mistral.rs backend capability tests respect the active feature set, so picolm-only TEE builds no longer expect HuggingFace/Vision support from the disabled mistralrs backend.
- Cleaned up test-only warnings in lean feature profiles.
- Updated the llama.cpp multimodal bitmap path for the current `llama-cpp-2` MTMD API.
- Removed the obsolete `box_integration` example and unused `a3s-box-sdk` dev-dependency after the Box SDK dropped the legacy Sandbox API.
- Expanded CI clippy coverage to all targets for default, HuggingFace, llama.cpp, picolm, and tee-minimal feature profiles.
- Fixed the Linux `vsock` server path to avoid pulling `tokio-vsock`'s Axum 0.8 adapter into Power's Axum 0.7 server stack.
- Tightened the release gate to run all-target clippy plus tee-minimal clippy/tests before publishing.
- Added route-level coverage for pull-status lookups with URL-encoded HuggingFace model names.
- Fixed the all-features backend registry test to account for the optional picolm backend, and added all-features lib tests to CI/release gates.
- Corrected tee-minimal documentation and CI setup to distinguish the pure-Rust inference path from native TLS/crypto build dependencies.
- Replaced server startup signal-handler panics with logged fallbacks for graceful shutdown and key rotation.
- Removed production `unwrap()` paths from attestation/pull-status JSON responses and recovered poisoned server mutexes instead of cascading panics.
- Hardened hardware-verifier certificate cache locking so poisoned locks return contextual verification errors instead of panicking.
- Hardened picolm session KV cache handoff so poisoned locks surface as inference errors or warnings instead of blocking-task panics.
- Hardened llama.cpp session, LoRA, and MTMD mutex handling so poisoned locks no longer panic inference workers.
- Replaced llama.cpp context-size unwrap fallbacks with explicit zero handling and a safe default.
- Removed tool-call parser argument unwraps by centralizing OpenAI-compatible argument serialization.
- Replaced TEE model signature array-conversion unwraps with contextual verification errors.
- Removed model-pull SSE JSON unwraps and added duplicate-pull response coverage.
- Replaced picolm tensor-cache and JSON grammar invariant unwraps with inference errors or invalid-character rejection.
- Centralized OpenAI SSE JSON encoding so serialization failures produce structured error events instead of empty data frames.
- Closed the model-pull in-flight race by making `start_pull` the authoritative duplicate guard and recovering its mutex after poison.
- Hardened audit loggers so poisoned file locks recover and write, flush, enqueue, serialization, and encryption failures are logged instead of silently dropping audit evidence.
- Preserved loaded-model state when request-scoped `keep_alive=0` unloads fail, and logged the backend unload error instead of marking the model unloaded prematurely.
- Prevented loaded model deletion from removing registry/state entries when backend unload fails.
- Added logging around model-pull temporary file cleanup and closed progress streams so download cleanup failures are observable.
- Improved CLI HTTP error reporting so failed response body reads are surfaced instead of being shown as empty errors.
- Preserved keep-alive reaper loaded-model state when no backend is available or backend unload fails.
- Logged request cleanup failures instead of silently ignoring backend `cleanup_request` errors.
- Logged pull-state deletion failures instead of silently ignoring cleanup errors.
- Recovered llama.cpp streaming tool-call text locks after poison instead of silently dropping accumulated tool-call output.
- Logged model-pull state persistence failures and marked pulls failed when manifest registration fails after download.
- Logged unreadable or invalid pull-state files instead of treating every load failure as a missing state.
- Preserved shutdown loaded-model state when no backend is available or backend unload fails.
- Logged decrypted-model secure wipe and delete failures during TEE file cleanup.
- Allowed boolean `A3S_POWER_*` environment overrides to disable enabled config values and logged invalid override values.
- Kept TEE in-memory plaintext in the same allocation that is passed to `mlock`, and logged `munlock` failures instead of silently ignoring cleanup errors.
- Rejected malformed mistral.rs multimodal image inputs with contextual errors instead of silently falling back to text-only requests.
- Reported malformed hub file-list API responses during model pulls instead of misclassifying them as missing GGUF quantization matches.
- Rejected unreadable HuggingFace model directories during local registration instead of silently recording a zero-byte manifest size.
- Reported model-pull partial-download and stored-blob metadata errors instead of silently treating unreadable files as zero bytes.
- Reported blob metadata errors during unused-blob pruning instead of silently counting unreadable blobs as zero bytes freed.
- Reported invalid model-pull `Content-Length` headers instead of silently treating malformed sizes as unknown.
- Rejected non-file local model paths during GGUF/SafeTensors registration instead of failing later as hash errors, and preserved exact file sizes in manifests.
- Logged `/v1/logs` live-stream lag events instead of silently dropping lagged broadcast entries.
- Rejected unsupported remote image URLs in llama.cpp chat requests instead of silently dropping those image parts.
- Logged closed llama.cpp completion receivers and stopped worker inference instead of silently discarding channel send failures.
- Rejected llama.cpp image requests when no multimodal projector is loaded instead of falling back to text-only inference.
- Reported llama.cpp generation batch/decode failures to completion streams instead of silently ending worker inference.
- Logged closed picolm chat receivers and stopped inference instead of silently discarding channel send failures.
- Logged picolm layer page-release and `madvise(MADV_DONTNEED)` failures instead of silently ignoring memory-pressure cleanup errors.
- Reported unrepresentable model keep-alive expiry timestamps as absent instead of silently showing an immediate expiry time.
- Logged closed mistral.rs chat receivers and stopped stream forwarding instead of silently discarding channel send failures.
- Treated overflowing `keep_alive` minute/hour values as invalid instead of panicking in debug builds or wrapping in release builds.
- Rejected GGUF files with missing, empty, or malformed tokenizer token metadata instead of silently constructing an empty picolm vocabulary.
- Rejected GGUF headers with invalid alignment, overflowing tensor dimensions, tensor byte ranges, or derived feed-forward sizes instead of panicking in debug builds or wrapping offsets in release builds.
- Hardened GGUF binary readers so malformed string, array, and cursor lengths return format errors instead of overflowing reader bounds.
- Hardened GGUF model metadata registration and memory estimation against oversized counts, dimensions, strings, arrays, and overflowing KV-cache estimates.
- Hardened picolm GGUF streaming metadata parsing against oversized counts, strings, arrays, tensor dimensions, and wrapping numeric metadata conversions.
- Rejected GGUF model metadata with overflowing tensor element counts during registration instead of surfacing saturated tensor sizes in model details.
- Rejected GGUF model metadata with overflowing tensor byte sizes or tensor offset ranges during registration.
- Rejected malformed GGUF quantized tensor descriptors whose first dimension or element count is not aligned to the type's block size.
- Rejected malformed picolm GGUF tokenizer score/type arrays instead of silently replacing invalid numeric values with zeroes.
- Rejected out-of-range picolm GGUF BOS/EOS token IDs during metadata parsing instead of allowing integer wrapping.
- Rejected non-finite or non-f32-compatible picolm GGUF scalar float metadata during parsing.
- Rejected malformed picolm GGUF scalar integer metadata instead of silently falling back to default model dimensions or alignment.
- Rejected malformed picolm GGUF scalar string metadata instead of silently falling back to default architecture or dropping chat templates.
- Rejected invalid picolm GGUF model shapes during load, including zero or non-divisible attention heads, invalid RoPE dimensions, non-finite numeric metadata, negative token IDs, and overflowing derived allocation estimates.

### Documentation

- Updated this changelog to reflect the completed picolm production-readiness, performance, hardening, and ecosystem work through `0.4.2`.
- Fixed a corrupted box-drawing line in the README verification diagram.
- Documented URL encoding for pull-status model names that contain `/` or `:`.

## [0.4.2] - 2026-02-23

### Fixed

- Resolved clippy warnings in the current feature set.
- Applied formatting cleanup in `main.rs`.

## [0.4.1] - 2026-02-23

### Added

- Added the full CLI surface with `serve`, `models`, `chat`, and `ps` subcommands.

### Changed

- Optimized release binary size with size-focused release profile settings, reducing the release binary from roughly 29 MB to 15 MB.
- Refreshed README documentation for the v0.4.0 picolm optimization status.

## [0.4.0] - 2026-02-22

### Added

- Completed picolm production-readiness work: multi-turn session KV cache, GGUF chat-template loading, configurable context length, stop sequences, and integration tests.
- Added picolm TEE hardening: timing padding on streaming/non-streaming paths, memory zeroization for forward/KV buffers, and startup self-tests for core math kernels.
- Added SIMD-accelerated picolm kernels: AVX2/FMA paths for F32, Q8_0, Q4_K, and Q6_K, plus NEON paths for Apple Silicon.
- Added batch prefill for lower time-to-first-token in layer-streaming inference.
- Added prompt-lookup speculative decoding with KV rollback.
- Added picolm tool/function calling support.
- Added grammar-constrained JSON structured output for picolm.
- Added repeat, frequency, and presence penalties for picolm generation.

### Changed

- Removed the unused `candle-core` dependency from the `picolm` feature; picolm now depends only on `memmap2`, `half`, and `rayon`.
- Improved picolm hot-path performance with fused f16 KV attention, zero-allocation sampling, pre-dequantized layer norms, and dual gate/up matvec.
- Updated planning and README documentation to mark phases 4 through 7 complete.

## [0.3.0] - 2026-02-22

### Added

- Implemented the pure-Rust picolm GGUF inference backend with real transformer operations.
- Added true layer-streaming inference with memory-mapped GGUF reads and page release after each layer, enabling O(layer_size) peak RAM instead of O(model_size).
- Added synthetic-GGUF end-to-end integration tests for picolm.
- Added Qwen GPT-style tokenizer support and attention-bias handling.
- Added a README technical deep dive for picolm layer streaming.

### Fixed

- Corrected Q4_K, Q5_K, and Q6_K dequantization.
- Fixed a test race condition in README/CI documentation work.

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
