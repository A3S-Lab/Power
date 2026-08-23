# A3S Power — Next Phase Development Plan

## First Principles Analysis

### Core Mission
A3S Power provides a **model-neutral, bounded, and verifiable inference runtime**.
Language, vision, OCR, embedding, audio, multimodal, and scientific model crates
retain their own topology and semantics. Power owns shared execution concerns:
artifact integrity, devices, admission, placement, mutable state, cancellation,
privacy, evidence, and independent verification. Hardware TEE support is one
deployment boundary, not a restriction to LLMs or to a particular model family.

### What Makes Power Unique (The Moat)
1. **Hardware-enforced privacy** — not policy promises, but CPU-level memory encryption
2. **Verifiable inference** — clients can bind exact artifacts, reviewed graphs,
   runtime policy, devices, inputs, and outputs
3. **tee-minimal build** — ~1,220 deps, no C++ inference engine, reduced auditable supply chain
4. **Layer-streaming** — O(layer_size) peak RAM, runs 7B+ models in 512MB EPC
5. **Model-neutral execution contracts** — one bounded runtime serves language,
   vision, OCR, embedding, audio, multimodal, and custom reviewed graphs without
   family dispatch in the core

### Current State (v1.0.0 release-candidate line)
- 3 backends (mistralrs, llamacpp, picolm) — all functional
- TEE runtime stack (attestation collection, encrypted models, RA-TLS, privacy,
  audit); production verifier coverage is qualified below
- OpenAI-compatible API with streaming
- Listener-free embedded runtime with typed CPU, CUDA, and Metal devices
- Model-owned reviewed graphs, finite shape profiles, session replicas,
  cancellation-safe execution batches, and device-resident graph boundaries
- A model-neutral release-evidence gate and complete-contract collector;
  a production tag is admitted only when its evidence-only child authenticates
  exact-revision CPU, CUDA, Metal, and confidential-GPU captures
- One canonical SafeTensors collection identity now joins startup integrity,
  model-bound attestation, embedded execution, persistent cross-host fixtures,
  and confidential accelerator declarations; mismatched directory hashing can
  no longer make a real promotion impossible
- Local speculative drafts, LoRA adapters, and multimodal projectors now have
  content identities independent of host paths. Strict startup, backend load,
  signature verification, model-bound attestation, request receipts, and the
  independent verifier all bind the same canonical auxiliary-artifacts digest.
- Strict SEV-SNP verification now binds policy fields to the exact signed raw
  report. Intel TDX fails closed until a DCAP Quote/QVL path is implemented.

### 2026 Shared Document Inference Substrate (TO2)

The model-neutral Power layer now provides the scheduling substrate required by
`a3s-ocr` and `a3s-parser`: finite cancellation-aware model queues, an exact
device/model session pool with declared resident-byte bounds, a shared physical
device gate, policy-bounded exclusive mutable session replicas with worst-case
residency admission, safe-boundary health retirement with lazy reconstruction,
monotonic queue deadlines with aggregate expiry evidence, deterministic
memory-aware microbatch plans with live-pressure revalidation, same-request
device-resident reviewed-graph chains with aggregate handle-byte admission, and
digest-only receipt evidence. Language,
vision, OCR, embedding, and multimodal crates use the same runtime contracts.
OCR stage semantics, image
pre/postprocessing, document target identity, cross-page structure, and retry
policy remain in their owning crates. Ready pool entries intentionally have no
implicit eviction; the owning service controls pool lifetime.
Resident graph handles contain no family dispatch or model topology: they
validate only reviewed F32 shape contracts, runtime/device/permit identity,
aggregate bytes, cancellation, and canonical initial/final tensor digests.

### 2026 Attestation Soundness Reopen

The implementation has a production attestation hardening follow-up after an
external review identified fail-open and claim-binding gaps in the current TEE
flow. The detailed remediation plan is tracked in
[`docs/attestation-hardening-plan.md`](docs/attestation-hardening-plan.md).
The current remediation has added strict fail-closed policy defaults, v2
CPU/GPU/runtime attestation claims, strict verifier defaults, and request-level
inference receipts. Encrypted GGUF models can now load from locked
`MemoryDecryptedModel` plaintext and `LayerStreamingDecryptedModel` plaintext
through `picolm`, while unsupported backends fail closed. Local deterministic chat
renderers can include rendered-prompt `effective_prompt` digests in receipts;
mistralrs text chat can include a domain-separated prompt-token-ID digest, and
proxy backends can include an upstream-declared digest through an explicit
opt-in endpoint. NVIDIA GPU confidential-computing support now has configured
evidence/verdict binding, a live `nvattest-cli` provider, and a direct
`nras-rest` provider that share the CPU/GPU attestation nonce, extract
structured NVIDIA device identity/freshness claims from the NRAS/NVAT verdict,
exposes verifier policy checks for those device claims, and supports
deployment-specific GPU/NVSwitch topology, NVIDIA claims-version, UEID,
OEM ID, hwmodel, driver, firmware, secure-boot, debug-state, and NVSwitch
identity/version pinning, with RIM schema validation enforced for accepted
device claims. A native NRAS SDK client and remaining opaque multimodal renderer
prompt digests still remain to be implemented. Until those
paths are complete, Phase 6 should be read as local TEE runtime hardening plus
substantial attestation remediation, not as a complete production attestation
soundness claim.

### First Principles Question: What Should We Build Next?

Every candidate feature must pass this filter:
1. Does it directly strengthen the privacy/verifiability moat?
2. Does it make Power deployable in real TEE production environments?
3. Does it close a gap that blocks actual adoption?

Features that fail this filter get rejected, no matter how "nice to have" they are.

---

## Phase 4: TEE Runtime Components Complete; Remote Verification In Progress

**Goal**: Make Power deployable in real AMD SEV-SNP and Intel TDX environments
with independently verifiable evidence. The runtime components below are
implemented. Production SEV-SNP still needs release evidence, and TDX remains
blocked on DCAP Quote generation and QVL verification.

### 4.1 — ✅ picolm Multi-Turn Session KV Cache
- `Arc<Mutex<Option<KvCache>>>` return path via `tokio::spawn` background task
- KV cache positions correctly maintained across turns
- Session map insert/remove with eviction on unload

### 4.2 — ✅ picolm Chat Template from GGUF Metadata
- Read `tokenizer.chat_template` from GGUF metadata
- Jinja2 rendering via minijinja with ChatML fallback
- Tested with Llama 3 and invalid template fallback

### 4.3 — ✅ picolm Configurable Context Length
- Read `context_length` from GGUF metadata, capped at 32K
- KV cache allocation scaled accordingly

### 4.4 — ✅ picolm Stop Sequence Support
- Check generated text against `stop` sequences after each token
- Trim output at stop boundary, set `finish_reason: "stop"`

### 4.5 — ✅ Remove Unused `candle-core` from picolm Feature
- Removed from Cargo.toml, picolm feature now: `["dep:memmap2", "dep:half", "dep:rayon"]`

### 4.6 — ✅ Integration Tests
- `tests/integration.rs`: 14 tests (HTTP API, router, registry, auth, error paths)
- `tests/picolm_tee.rs`: 8 tests (load/unload cycle, TEE mode, deterministic output)
- `tests/picolm_real.rs`: Real model inference (gated by model file presence)

---

## Phase 5: Performance & Scalability ✅

**Goal**: Close the remaining performance gaps vs. llama.cpp for TEE-constrained environments.

### 5.1 — ✅ SIMD-Accelerated vec_dot Kernels
- AVX2+FMA kernels for F32, Q8_0, Q4_K, Q6_K
- Runtime feature detection via `is_x86_feature_detected!`
- Scalar fallback for non-AVX2 platforms (aarch64)
- Parity tests: AVX2 vs scalar for all kernel types

### 5.2 — ✅ NEON-Accelerated vec_dot for Apple Silicon
- `#[cfg(target_arch = "aarch64")]` NEON paths for F32, Q8_0, Q4_K, Q6_K
- F32: vfmaq_f32 4-wide accumulation
- Q8_0: vmull_s8 + vpadalq_s16 with scalar scale accumulation
- Q4_K: nibble extract + vmull with min/scale dequant
- Q6_K: 6-bit reconstruct (ql low4 + qh high2) + vmull
- Parity tests for all 4 kernel types

### 5.3 — ✅ Batch Prefill
- Layer-outer, token-inner loop ordering: O(n_layers) page faults instead of O(n_layers × n_tokens)
- Each layer's mmap pages loaded once for all tokens, then released
- `matmul_batch` function for batched matrix-vector multiply
- Hidden states matrix `[n_tokens × n_embd]` flows through layers

### 5.4 — ✅ Speculative Decoding (Prompt-Lookup)
- Prompt-lookup decoding: matches n-grams (2–5) from generated text against input tokens
- Zero draft cost — no extra model or layer-skipping needed
- `count_accepted` greedy verification against full-model logits
- KV cache rollback via `truncate()` for rejected draft tokens
- Wired into decode loop with hidden-state backup/restore
- Disabled during grammar-constrained generation (structured output)
- Works well for tasks where output overlaps input (summarization, JSON, code completion)

---

## Phase 6: TEE Hardening ✅

**Goal**: Close security gaps for production TEE deployment.

### 6.1 — ✅ Timing Side-Channel Mitigation
- `timing_padding_ms` wired into both streaming and non-streaming chat paths
- ±20% jitter via existing `timing_padding()` method

### 6.2 — ✅ Memory Zeroization Audit
- `Drop` impl for `ForwardBuffers`: zeroizes all 12 Vec<f32> buffers
- `Drop` impl for `LayerKvCache`: zeroizes K/V f16 data
- `KvCache::clear()` now zeroizes data instead of just resetting length

### 6.3 — ✅ Startup Self-Test
- Embedded test vectors for rms_norm, vec_dot_f32, vec_dot_q8_0
- Runs at model load time, fails fast with clear error on mismatch
- Catches memory corruption in TEE before inference begins

---

## Phase 7: Ecosystem Integration ✅

**Goal**: Make Power useful in the broader A3S platform.

### 7.1 — ✅ picolm Tool/Function Calling
- Wire `tool_parser::parse_tool_calls()` into picolm response stream
- Accumulate full generated text, parse tool calls on final chunk (EOS/stop/max_tokens)
- `has_tools` flag in GenerateParams, set from ChatRequest.tools
- Matches llamacpp/mistralrs backend pattern

### 7.2 — ✅ picolm Structured Output (JSON Grammar)
- `JsonGrammarSampler`: stack-based JSON validator for grammar-constrained sampling
- Tracks structural state (object/array/string/number/keyword nesting)
- `mask_logits`: filters tokens whose first character violates grammar
- Wired into decode loop via `response_format` field from ChatRequest
- Auto-stops generation when complete JSON value is produced

### 7.3 — ✅ picolm Repeat/Frequency Penalty
- 64-token ring buffer tracking recent generated tokens
- `repeat_penalty` (multiplicative, llama.cpp style)
- `frequency_penalty` (proportional to count, OpenAI style)
- `presence_penalty` (flat if appeared, OpenAI style)
- Applied to logits before sampling

---

## What We Will NOT Build (First Principles Rejection)

### ❌ GPU Support inside picolm
**Why not**: picolm is the CPU/EPC-constrained pure Rust backend. Adding GPU memory to picolm itself would undermine that backend's minimal CPU TEE security model. Power still supports NVIDIA GPU acceleration through the other backends, and production NVIDIA GPU Confidential Computing support is tracked separately through `tee_policy_mode = "gpu-confidential"` and bound GPU evidence claims.

### ❌ Embeddings in picolm
**Why not**: Embedding models are small, don't need layer-streaming, and don't process sensitive user prompts (they process documents at indexing time, not query time). Use mistralrs for embeddings. Adding embedding support to picolm adds complexity without strengthening the privacy moat.

### ❌ Vision/Multimodal in picolm
**Why not**: Vision models require image encoders (ViT) that are architecturally different from text transformers. The complexity cost is high, and the TEE use case for vision is niche. Use mistralrs for vision.

### ❌ Model Quantization in picolm
**Why not**: Quantization is a one-time offline operation. It doesn't need to run inside TEE. Users quantize models before deployment.

### ❌ LoRA/Adapter Support in picolm
**Why not**: LoRA adds complexity to the forward pass and the supply-chain audit story. In TEE, you want to verify one specific model — not a base + N adapters. If needed, merge LoRA into the base model before deployment.

---

## Completion Summary

The picolm, performance, hardening-component, and ecosystem milestones above
are implemented and covered by the repository test profiles. They are not a
claim that Power is ready to tag v1.0.0.

The Rust release API now requires an opaque exact-report proof for confidential
promotion, and the checked-in external capture runbook carries that API through
raw vendor-evidence preservation and create-new CLI output. The generic runner
also materializes persistent fixture weights and creates the local CUDA
source/declaration pair atomically; caller-owned graphs retain their model-owned
declaration path. The release CLI now also verifies each transferred capture
against an exact platform, version, and source revision before assembly while
marking that result as single-capture scope rather than production eligibility.
A production release
exists only when its signed evidence child proves that all of these gates pass:

1. one immutable revision supplies CPU, CUDA, Metal, and confidential-GPU
   complete-contract captures;
2. the checked-in external capture runbook is executed on named hardware and
   its raw vendor evidence, report, declaration, environment, artifact hashes,
   promoted capture, and release trust-root material are preserved;
3. Intel TDX either gains a reviewed DCAP Quote/QVL path or remains explicitly
   unsupported by the v1 production support matrix;
4. the full default, embedded, accelerator, verifier, documentation, and release
   checks pass for the frozen source parent and its evidence-only child; and
5. the GitHub-verified annotated tag and root-monorepo gitlink bind those exact
   revisions.

See [ROADMAP.md](ROADMAP.md) and
[Production Release Evidence Gate](docs/release-evidence-gate.md) for the
authoritative remaining acceptance evidence.
