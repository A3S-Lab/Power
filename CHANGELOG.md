# Changelog

All notable changes to A3S Power will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added model-owned finite shape-profile declarations for the embedded runtime.
  Opaque class digests bind exact implementations to aggregate execution and
  scratch bounds under the current weights, graph, device topology, memory
  reservations, and TEE policy. Explicit dynamic fallback records a typed
  reason and implementation digest; receipt v5 exposes no dimensions, class
  data, filesystem paths, or tensor values.

## [0.9.0] - 2026-08-19

### Added

- Added an independent `artifact-provisioning` feature for consumer-owned,
  revision-locked model bundles. It provides bounded HTTPS downloads,
  per-file and whole-bundle size admission, SHA-256 verification,
  cross-process locking, atomic commits, deterministic URL-free receipts,
  tamper detection, and strict offline reuse without enabling a Power server
  or inference backend.
- Added model-neutral speculative-decoding strategy and capability negotiation,
  shared exact block-verification metrics, configurable draft bounds, and
  fail-closed backend selection. picolm preserves its zero-weight lookup modes,
  while llama.cpp 0.1.154 adds native MTP execution for compatible GGUF models
  with target/draft cache rollback and optional CUDA compilation.
- Added `a3s-power-speculative-bench` for controlled baseline/candidate runs
  through Power's streaming completion API. Reports verify the registered GGUF
  SHA-256 and inference receipt, bind non-secret runtime and hardware settings,
  omit request paths/content, require deterministic output parity, and enforce
  an explicit median decode-throughput threshold.
- Added a seekable AES-256-GCM weight representation for embedded inference.
  Each SafeTensors file is split into independently authenticated chunks under
  a digest-pinned collection manifest; opening verifies the complete logical
  plaintext identity with bounded zeroizing buffers, while demand, prefetch,
  and staged reads decrypt only the chunks covering an indexed tensor.
- Added manifest-aware backend capability matching. Backends may override
  `supports_manifest` to select an exact model family while format-only
  implementations retain their existing behavior. Request, unload, reaper,
  and TEE routing paths now preserve that exact selection.
- Added the typed `PowerServerBuilder` composition root. Downstream model crates
  can inject backend objects ahead of Power's built-ins or explicitly run with
  only their supplied registry without configuring raw backend names.
- Added process-local model manifest injection to `PowerServerBuilder`, so a
  downstream backend and its exact manifest can be composed without mutating
  Power's global model registry on disk.
- Added exact positional-staging coverage for model-owned packed `U8` weight
  records, demonstrating that one atomic record can carry adjacent expert
  matrices and scales through the existing hierarchy without a second cache.

## [0.8.0] - 2026-08-04

### Added

- Added Colibri-inspired N-drive capacity aggregation to the existing
  `WeightStore`. Canonical and lossless logical sources can span explicit,
  disjoint, non-empty physical roots while retaining the same relative-file
  collection digest, integrity/signature identity, source router, residency
  cache, partial-mirror staging, TEE binding, and receipt path.
- Added fail-closed duplicate, nested, overlapping, empty-root, duplicate-file,
  and cross-source checks; the storage benchmark now accepts repeated
  `--primary-shard` arguments and binds only the path-free physical root count
  into source-profile and hardware evidence.
- Added finite, cancellation-aware embedded inference queues with explicit
  active/waiting bounds, fail-fast overflow, RAII cleanup, and content-free
  admission snapshots. Existing fail-fast admission remains available.
- Added a device-bound, model-neutral `ModelSessionPool<T>` with hard session
  count, declared resident-byte, device concurrency, and device queue bounds.
  Exact model/execution/resource declarations share one lazily initialized
  value and one resolved device; abandoned initialization releases its slot.
- Added deterministic memory-aware microbatch plans with caller-declared
  per-slot input, state, host-peak, and device-peak resources. Plans preserve
  contiguous order, account for unified memory once, and revalidate current
  memory pressure before bounded admission.
- Added embedded receipt v4 for admitted microbatches. It binds the exact pool
  session (when present), plan digest, batch position, slot count, and whether
  model or device admission queued without adding model input, slot identity,
  or raw memory snapshots to receipt evidence.
- Added model-neutral leading-axis tensor assembly and exact ordered output
  partitioning with finite-value, shape, resource-limit, and full-partition
  validation for OCR and other model-owned batched graphs.
- Added a Colibri-inspired macOS `PositionalCacheBypass` weight strategy using
  `F_NOCACHE` for both integrity hashing and exact tensor-range handles. It
  reuses the existing `WeightStore` index, replica routing, fallback,
  cancellation, zeroizing buffers, telemetry policy, and benchmark evidence;
  unsupported platforms fail explicitly and it is never mislabeled direct or
  verified-cold I/O.
- Added canonical, self-verifying hardware evidence bundles that compose the
  existing path-free storage reports/comparison, raw lossless tuning evidence,
  replayed tuning decision, and model-owned exact-parity artifact pins under
  one model/runtime/device/environment binding.
- Added deterministic ordering, a 32 MiB canonical envelope bound, pinned
  bundle verification, exact re-derivation of storage and tuning summaries,
  and preservation of reviewed negative results when the baseline remains
  selected.
- Added Colibri-inspired, model-neutral continuous and ragged execution-batch
  lifecycles. Every member holds a distinct permit from the existing runtime
  admission controller, every step covers the complete boundary roster in
  canonical admission order, and members admitted during arithmetic join only
  the next step.
- Added atomic batch-step commits with row-local cancellation, bounded ragged
  shape/input/context/generation/state accounting, non-aliasing model-owned
  state identities, exact retry after invalid outcomes, and aggregate-only
  step/lifecycle evidence with canonical transcript digests.
- Added Colibri-inspired heterogeneous accelerator meshes with at most 16
  unique resolved devices, a canonical primary/home device, strongly connected
  directed transfer edges, per-edge count/byte bounds, and one aggregate byte
  budget. Model crates retain graph partitioning and kernels; Power performs
  explicit Candle copies through `AcceleratorMeshExecution` and reuses the
  active residency plan, device cache, execution permit, cancellation, and
  exact fallback path.
- Added mesh execution evidence and embedded receipt v3. Receipts expose only
  the mesh digest, canonical actual-device identities, and a digest of the
  bounded transfer trace.
- Added model-neutral host/device fixed-state and peak-scratch reservations to
  the existing hardware residency budget policy. Cache budgets now account for
  these bytes after applying caller-owned available-memory fractions, while
  unified host/device memory remains one physical pool.
- Added current-pressure revalidation for serialized residency plans and an
  `EmbeddedRuntime::apply_residency_budget` path that refreshes native memory
  availability before applying cache bytes.
- Added Colibri-inspired event-driven current-layer staging through
  `StagedWeightBatch::next_ready_group`, allowing model-owned compute to consume
  newly complete atomic groups without polling while preserving canonical final
  order.
- Added one shared background load window for prefetch and staging. It bounds
  active workers and their canonical in-flight bytes, reports peak flight and
  event/final wait evidence, and looks past a temporarily blocked large item to
  use safe spare capacity.

- Added privacy-gated, Colibri-inspired cross-layer route coupling for
  value-preserving prefetch hints. Exact aligned route batches produce bounded
  co-occurrence history, deterministic per-position predictions and a batch
  union for the existing prefetch path, plus exact recall evaluation against
  actual router output.

### Changed

- Kept mmap as the default after two forward/reverse PP-OCRv6 storage-only runs
  on Apple M2 Pro showed exact output-byte parity but cache-bypass p50 latency
  regressions of 6.6% for detection and 10.1% for recognition weights.

### Security

- Model-session keys and declarations bind exact model, execution, resolved
  device, limits, and resident bytes. Microbatch plans fail closed on duplicate
  slot digests, overflow, wrong session/device/limits, tampering, stale memory
  headroom, and malformed topology. Pool and receipt debug output is
  content-free; raw memory snapshots remain caller-owned plan data.
- Hardware evidence construction rejects mixed revisions, models, named
  hardware, runtime devices, tuning bindings, selected configurations, and
  parity artifacts. Bundle SHA-256 is mutation evidence rather than a
  signature; callers must pin it in an attestation, signed release, or another
  trust root. Construction never logs, persists, uploads, or opens a listener.
- Bound execution batches to exact weight, model-owned state-layout, scheduler,
  runtime-device, and limit digests. Member/state identities and row data remain
  out of public evidence and debug output; Power retains no model state bytes
  and adds no scheduler, persistence path, listener, or second admission queue.
- Confidential meshes bind every CUDA node to an explicit NVIDIA claim-array
  index and require exact GPU/NVSwitch claim-index sets from the existing
  verified NRAS evidence. CUDA ordinals are not treated as claim indices or
  UEIDs, NVSwitch presence is not treated as proof of edge connectivity, and
  backend copy failure enters only the declared exact fallback. Mesh node/edge
  names and transfer details are not logged, persisted, or placed in receipts.
- Runtime reservations, memory snapshots, and pressure decisions remain
  caller-owned and are never logged, persisted, placed in telemetry, bound into
  receipts, or exported from a TEE automatically. Overflow, unavailable pools,
  changed topology, and stale headroom fail closed before cache policy changes.
- Kept the new scheduling path inside existing admission, cancellation,
  per-key serialization, cache, source routing, telemetry, and TEE boundaries.
  Background byte limits fail before I/O, and telemetry-off mode exposes no
  timing or peak-flight counters.

- Bound route coupling by lookahead, position, entry, and hint limits; require
  detailed telemetry; bind restored history to exact weight SHA-256 and layer
  geometry; and keep histories, predictions, expert IDs, and evaluations out
  of automatic logs, persistence, attestation claims, and receipts.

## [0.7.0] - 2026-08-03

### Added

- Added exact verified SafeTensors range indexes and opt-in buffered positional
  reads without retaining a collection-wide mmap. Demand and prefetch reuse the
  existing coverage-aware weighted source route and primary fallback.
- Added aligned direct reads through Linux `O_DIRECT` and Windows
  `FILE_FLAG_NO_BUFFERING`, with explicit unsupported results instead of a
  buffered fallback.
- Added `a3s-power-storage-bench`, a standalone path-redacted benchmark that
  separates integrity-open, output validation, and demand-read timing; compares
  cold/warm and single/multi-source reports; and verifies output digest parity.

### Changed

- Kept mmap as the default after official PP-OCRv6 warm-storage measurements on
  Apple M2 Pro showed positional buffered p50 latency regressions of 9.6% for
  detection weights and 10.6% for recognition weights. macOS direct reads and
  verified cold-cache labels remain explicitly unsupported.

### Security

- Zeroized positional tensor buffers and aligned direct-I/O scratch memory,
  preserved cancellation and TEE behavior, and kept paths, tensor names,
  ranges, source choices, benchmark reports, and hardware labels out of
  automatic logs, telemetry, persistence, attestation claims, and execution
  receipts.

## [0.6.0] - 2026-08-02

### Added

- Added model-neutral, usage-ranked partial SafeTensors mirror planning and
  staging on the existing `WeightStore` integrity path. Selection is
  deterministic and budget-bound, native filesystem capacity honors an
  explicit reserve, exact completed files resume without copying, and every
  new file is SHA-256 verified, synced, and atomically published without
  replacement.

### Security

- Plaintext mirror staging is denied by default and requires the typed
  `CallerManagedPlaintext` authority. Staging is cancellable, never overwrites
  a conflicting destination, refuses symlink/path escape, detects source
  mutation after store admission, and does not automatically persist routing
  history, plans, paths, telemetry, or execution receipts.

## [0.5.1] - 2026-08-02

### Added

- Added opt-in hardware-aware residency budgets with bounded native
  Linux/macOS/Windows host-memory discovery, selected CUDA/Metal device-memory
  discovery, explicit reserves/fractions/caps, deterministic allocation order,
  runtime-limit enforcement, and single-count Metal unified-memory planning.

### Security

- Kept automatic cache planning disabled by default, failed closed on incomplete
  hardware discovery, spawned no probe process, and excluded capacity snapshots
  from automatic logs, placement telemetry, persistence, and execution receipts.

## [0.5.0] - 2026-08-02

### Added

- Added Colibri-inspired partial SafeTensors replicas with exact per-file and
  per-tensor validation, coverage-aware deterministic source selection, and
  primary fallback for tensors outside the mirrored subset.
- Added opt-in validation-throughput source weighting that reuses the mandatory
  integrity hash pass instead of reading multi-gigabyte weight sources twice.

### Security

- Kept complete and partial replica admission fail-closed, bounded by the
  existing source/file/model limits, with measured storage throughput available
  only through an explicit descriptor and excluded from automatic telemetry.

## [0.4.5] - 2026-08-02

### Added

- Deepened the shared Colibri-inspired weight hierarchy with frequency-first LFRU eviction and decay, transactional hot-plan replacement with separate manual/plan pin ownership, useful-versus-unused prefetch accounting, and exact bandwidth-weighted SafeTensors replicas with deterministic source selection, per-source aggregate telemetry, and primary fallback.

### Security

- Kept detailed route heat private-by-default while fully hashing every configured weight replica against the primary and bounding source count through the existing embedded inference limits.

## [0.4.4] - 2026-08-02

### Added

- Added `PowerRuntimeServiceProfile`, which compiles a digest-pinned Power deployment into the shared `a3s-runtime` Service contract for execution by A3S Box.
- Added a model-neutral embedded Rust inference substrate with reviewed static graphs, typed CPU/Metal/CUDA devices, bounded admission and cancellation, exact SafeTensors inventories, canonical execution receipts, and strict isolation from the HTTP server feature.
- Added Colibri-inspired storage/host/device weight residency, layer-local LRU, explicit hot pins, deterministic heat-based planning, batched expert unions, bounded layer-ahead prefetch, and privacy-gated placement/routing telemetry.
- Added embedded tensor operators and integrity hooks without embedding PP-OCR, Unlimited-OCR, tokenizers, model assets, or model revision policy in Power.

### Changed

- Replaced the previous product configuration codec with closed, typed A3S ACL parsed and generated by `a3s-acl`; the default configuration path is now `~/.a3s/power/config.acl`.
- Kept TEE encrypted-model, signature, attestation, zeroization, privacy, and request-receipt controls intact across the new embedded execution path.

### Fixed

- Materialized valid strided MatMul operands before Candle execution so reviewed graphs can consume transposed and batched transposed views without `MatMulUnexpectedStriding`.

## [0.4.3] - 2026-07-19

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
- **Configuration**: file-first with `A3S_POWER_*` environment overrides
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
