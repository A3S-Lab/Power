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
- picolm: 14+ tok/s decode, pure Rust, layer-streaming, 858 tests

### First Principles Question: What Should We Build Next?

Every candidate feature must pass this filter:
1. Does it directly strengthen the privacy/verifiability moat?
2. Does it make Power deployable in real TEE production environments?
3. Does it close a gap that blocks actual adoption?

Features that fail this filter get rejected, no matter how "nice to have" they are.

---

## Phase 4: Production TEE Readiness 🎯

**Goal**: Make Power deployable in real AMD SEV-SNP / Intel TDX environments with confidence.

### 4.1 — picolm Multi-Turn Session KV Cache (Priority: HIGH)

**Why**: Without KV cache reuse across turns, every message in a conversation re-processes the entire history. This is a 10-50x latency penalty for multi-turn use cases — the primary use case for chat inference. The TODO already exists in the code.

**What**:
- Fix the `Arc<Mutex<Option<KvCache>>>` return path after stream completes
- Verify KV cache positions are correctly maintained across turns
- Add integration test: 3-turn conversation, verify 2nd/3rd turn skip prefill of prior tokens
- Add KV cache eviction when session expires

**Scope**: ~100 lines, 1 file (picolm.rs)

### 4.2 — picolm Chat Template from GGUF Metadata (Priority: HIGH)

**Why**: Hardcoded ChatML breaks models that use Llama 3, Phi, Gemma, or other templates. Users get silently wrong outputs with no error. This is a correctness bug, not a feature.

**What**:
- Read `tokenizer.chat_template` from GGUF metadata (already parsed by gguf_stream.rs)
- Use minijinja (already a dependency) to render the template
- Fall back to ChatML if no template found
- Test with Llama 3 template format (`<|begin_of_text|><|start_header_id|>...`)

**Scope**: ~150 lines, 2 files (picolm.rs, tokenizer.rs)

### 4.3 — picolm Configurable Context Length (Priority: MEDIUM)

**Why**: Hardcoded `max_seq_len: 2048` silently truncates models that support 8K-128K context. For TEE deployments processing documents (healthcare records, legal contracts), this is a blocker.

**What**:
- Read `context_length` from GGUF metadata (already in `ModelConfig`)
- Add `max_context_length` config field to `PowerConfig`
- Use `min(model_context_length, config_max, available_memory_estimate)`
- Scale KV cache allocation accordingly

**Scope**: ~50 lines, 2 files (picolm.rs, config.rs)

### 4.4 — picolm Stop Sequence Support (Priority: MEDIUM)

**Why**: Without stop sequences, tool calling and structured output are impossible. The `stop` field is accepted in requests but silently ignored.

**What**:
- Check generated text against `stop` sequences after each token
- Trim the stop sequence from output
- Set `finish_reason: "stop"` when triggered

**Scope**: ~40 lines, 1 file (picolm.rs)

### 4.5 — Remove Unused `candle-core` from picolm Feature (Priority: LOW)

**Why**: picolm doesn't use candle — it has its own kernels. This dependency adds ~200 crates to the `tee-minimal` build, directly undermining the supply-chain audit story.

**What**:
- Remove `dep:candle-core` from picolm feature in Cargo.toml
- Verify `tee-minimal` still builds
- Update dep tree count in README

**Scope**: 1 line in Cargo.toml, verify build

### 4.6 — Integration Tests (Priority: HIGH)

**Why**: 858 unit tests but zero integration tests. The test files exist but are empty. End-to-end server behavior (start → load → infer → verify attestation) is untested. For a security-critical system, this is unacceptable.

**What**:
- `tests/integration.rs`: Start server, load model, chat completion, verify response format
- `tests/picolm_tee.rs`: Simulated TEE mode, attestation endpoint, nonce binding, model hash binding
- `tests/picolm_real.rs`: Real model inference (gated by env var `A3S_TEST_MODEL_PATH`)
- Test error paths: invalid model, auth failure, rate limit, malformed request

**Scope**: ~400 lines across 3 test files

---

## Phase 5: Performance & Scalability

**Goal**: Close the remaining performance gaps vs. llama.cpp for TEE-constrained environments.

### 5.1 — SIMD-Accelerated vec_dot Kernels (Priority: HIGH)

**Why**: The fused vec_dot kernels are scalar. On x86-64 (the TEE platform — SEV-SNP/TDX are x86 only), AVX2/AVX-512 can give 4-8x speedup on the dot product inner loop. This is the single biggest performance win available.

**What**:
- `#[cfg(target_arch = "x86_64")]` SIMD paths for Q4_K, Q6_K, Q8_0
- Use `std::arch::x86_64` intrinsics (no external deps)
- Runtime feature detection via `is_x86_feature_detected!`
- Scalar fallback for non-SIMD platforms (aarch64 TEE doesn't exist yet)
- Benchmark: target 40+ tok/s decode on x86-64

**Scope**: ~500 lines in vec_dot.rs

### 5.2 — NEON-Accelerated vec_dot for Apple Silicon (Priority: LOW)

**Why**: Development/testing convenience. Not TEE-relevant (no ARM TEE in production), but makes local development faster.

**What**:
- `#[cfg(target_arch = "aarch64")]` NEON paths
- Same kernels as 5.1 but with NEON intrinsics

**Scope**: ~400 lines in vec_dot.rs

### 5.3 — Batch Prefill (Priority: MEDIUM)

**Why**: Current prefill processes one token at a time through the full transformer. For long prompts (1K+ tokens), batch matmul would be significantly faster.

**What**:
- Batch Q/K/V projections: matmul instead of matvec for prefill
- Batch attention score computation
- Only for prefill phase; decode stays single-token

**Scope**: ~300 lines, new batch_matmul.rs + changes to attention.rs

### 5.4 — Speculative Decoding (Priority: LOW)

**Why**: Use a small draft model to propose tokens, verify with the main model in a single batch forward pass. Can give 2-3x decode speedup.

**What**:
- Load a draft model (e.g., 0.5B) alongside the main model
- Draft N tokens, verify in batch
- Accept matching prefix, reject and resample from main model

**Scope**: ~500 lines, significant architectural change. Evaluate after 5.1-5.3.

---

## Phase 6: TEE Hardening

**Goal**: Close security gaps for production TEE deployment.

### 6.1 — Timing Side-Channel Mitigation (Priority: HIGH)

**Why**: `timing_padding_ms` is defined in config but not wired into the request handler. Token generation timing can leak information about prompt content. For a privacy-focused system, this is a real threat.

**What**:
- Wire `timing_padding_ms` into the response handler
- Add ±20% jitter to response timing
- Constant-time token count reporting (already partially done with rounding)

**Scope**: ~30 lines in server handler

### 6.2 — Memory Zeroization Audit (Priority: HIGH)

**Why**: Verify that all sensitive data (prompts, responses, KV cache, intermediate activations) is zeroized on drop. The `zeroize` crate is used but coverage may have gaps after the picolm optimizations (ForwardBuffers, TensorCache).

**What**:
- Audit ForwardBuffers: ensure `Drop` impl zeroizes all buffers
- Audit KV cache: ensure session cleanup zeroizes FP16 storage
- Audit TensorCache: verify no dangling references to model weights after unload
- Add `#[derive(Zeroize, ZeroizeOnDrop)]` where missing

**Scope**: ~50 lines across 3-4 files

### 6.3 — Startup Self-Test (Priority: MEDIUM)

**Why**: In TEE, you can't attach a debugger. If inference produces wrong results (e.g., due to memory corruption), there's no way to know. A self-test at startup catches this.

**What**:
- Embed a tiny test vector (known input → known output) for each supported quant type
- Run at model load time, verify output matches expected
- Fail fast with clear error if mismatch

**Scope**: ~100 lines in picolm.rs

---

## Phase 7: Ecosystem Integration

**Goal**: Make Power useful in the broader A3S platform.

### 7.1 — picolm Tool/Function Calling (Priority: MEDIUM)

**Why**: Tool calling is the foundation for AI agents (a3s-code). Without it in picolm, TEE-deployed agents can't use tools.

**What**:
- Wire `tool_parser.rs` into picolm response stream
- Parse tool call JSON from model output
- Return structured `tool_calls` in response chunks

**Scope**: ~100 lines in picolm.rs

### 7.2 — picolm Structured Output (JSON Schema) (Priority: MEDIUM)

**Why**: Structured output is required for reliable API integrations. `json_schema.rs` (JSON Schema → GBNF grammar) exists but isn't connected to picolm.

**What**:
- Implement grammar-constrained sampling in picolm
- Wire `json_schema.rs` grammar into the token sampling loop
- Constrain token selection to valid grammar continuations

**Scope**: ~300 lines, new grammar_sampler.rs

### 7.3 — picolm Repeat/Frequency Penalty (Priority: LOW)

**Why**: Without repeat penalty, models tend to loop. Currently accepted in request but silently ignored.

**What**:
- Track recent token IDs in a ring buffer
- Apply penalty to logits before sampling

**Scope**: ~40 lines in picolm.rs

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

## Priority Order (Recommended Execution Sequence)

```
Phase 4 (Production TEE Readiness):
  4.5 Remove candle-core from picolm     [1 hour]   ← quick win, shrinks dep tree
  4.1 Multi-turn KV cache                [1 day]    ← biggest UX impact
  4.2 Chat template from GGUF            [1 day]    ← correctness fix
  4.4 Stop sequence support              [0.5 day]  ← enables tool calling later
  4.3 Configurable context length        [0.5 day]  ← unblocks long-context use cases
  4.6 Integration tests                  [2 days]   ← security-critical system needs this

Phase 6 (TEE Hardening):
  6.1 Timing side-channel mitigation     [0.5 day]  ← config field exists, just wire it
  6.2 Memory zeroization audit           [1 day]    ← security audit
  6.3 Startup self-test                  [0.5 day]  ← TEE reliability

Phase 5 (Performance):
  5.1 AVX2/AVX-512 vec_dot              [2 days]   ← biggest perf win for x86 TEE
  5.3 Batch prefill                      [2 days]   ← long-prompt speedup
  5.2 NEON vec_dot                       [1 day]    ← dev convenience
  5.4 Speculative decoding               [3 days]   ← evaluate after 5.1-5.3

Phase 7 (Ecosystem):
  7.1 Tool/function calling              [1 day]    ← enables a3s-code in TEE
  7.2 Structured output                  [2 days]   ← grammar-constrained sampling
  7.3 Repeat penalty                     [0.5 day]  ← quality improvement
```

**Total estimated effort**: ~18 days for all phases.
**Recommended v0.4.0 scope**: Phase 4 + Phase 6.1-6.2 (~7 days) — production-ready TEE deployment.
