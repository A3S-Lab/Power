# A3S Power — Next Phase Development Plan

## First Principles Analysis

### Core Mission
A3S Power exists to solve one problem: **LLM inference where the infrastructure operator cannot see your data**. Hardware TEE (SEV-SNP / TDX) enforces memory encryption; Power provides the cryptographic proof chain that lets clients verify it.

### What Makes Power Unique (The Moat)
1. **Hardware-enforced privacy** — not policy promises, but CPU-level memory encryption
2. **Verifiable inference** — client can cryptographically prove which model ran, unmodified
3. **tee-minimal build** — ~1,220 deps, zero C, fully auditable supply chain
4. **Layer-streaming** — O(layer_size) peak RAM, runs 7B+ models in 512MB EPC

### Current State (v0.3.0)
- 3 backends (mistralrs, llamacpp, picolm) — all functional
- Full TEE stack (attestation, encrypted models, RA-TLS, privacy, audit)
- OpenAI-compatible API with streaming
- picolm: 14+ tok/s decode, pure Rust, layer-streaming, 868 tests

### First Principles Question: What Should We Build Next?

Every candidate feature must pass this filter:
1. Does it directly strengthen the privacy/verifiability moat?
2. Does it make Power deployable in real TEE production environments?
3. Does it close a gap that blocks actual adoption?

Features that fail this filter get rejected, no matter how "nice to have" they are.

---

## Phase 4: Production TEE Readiness ✅

**Goal**: Make Power deployable in real AMD SEV-SNP / Intel TDX environments with confidence.

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

### ❌ GPU Support for picolm
**Why not**: TEE memory (EPC) is CPU memory. GPU memory is outside the trust boundary. Adding GPU to picolm would undermine the entire security model. Use mistralrs/llamacpp for GPU inference outside TEE.

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

```
Phase 4 (Production TEE Readiness):          ✅ ALL COMPLETE
  4.5 Remove candle-core from picolm         ✅
  4.1 Multi-turn KV cache                    ✅
  4.2 Chat template from GGUF                ✅
  4.4 Stop sequence support                  ✅
  4.3 Configurable context length            ✅
  4.6 Integration tests                      ✅

Phase 6 (TEE Hardening):                     ✅ ALL COMPLETE
  6.1 Timing side-channel mitigation         ✅
  6.2 Memory zeroization audit               ✅
  6.3 Startup self-test                      ✅

Phase 5 (Performance):                       ✅ ALL COMPLETE
  5.1 AVX2/AVX-512 vec_dot                   ✅ (F32, Q8_0, Q4_K, Q6_K)
  5.2 NEON vec_dot                           ✅ (F32, Q8_0, Q4_K, Q6_K)
  5.3 Batch prefill                          ✅ (layer-outer loop, matmul_batch)
  5.4 Speculative decoding                   ✅ (prompt-lookup, KV rollback)

Phase 7 (Ecosystem):                         ✅ ALL COMPLETE
  7.3 Repeat penalty                         ✅
  7.1 Tool/function calling                  ✅
  7.2 Structured output                      ✅ (JsonGrammarSampler)
```

**895 tests total** (unit + integration + real model).
**All phases complete.** Phases 4–7 fully implemented.
