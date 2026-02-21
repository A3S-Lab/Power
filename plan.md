# A3S Power — Development Plan

> Inspired by [picolm](https://github.com/RightNow-AI/picolm) · Based on first-principles analysis · Date: 2026-02-21

---

## First-Principles Review

**A3S Power's core mission**: Run LLM inference inside TEE (AMD SEV-SNP / Intel TDX) with hardware-enforced privacy guarantees.

**What picolm teaches us**: picolm is a ~2,500-line C11 inference engine that runs 1B-parameter models on 256MB RAM via layer-by-layer streaming. Its two most relevant ideas for Power are:

1. **Layer streaming** — only one transformer layer lives in RAM at a time; the model can be larger than available memory
2. **Zero-dependency philosophy** — only libc/libm/pthread; the entire engine is auditable in an afternoon

These map directly to two real problems in TEE inference:

- **EPC memory wall**: AMD SEV-SNP Encrypted Page Cache is typically 512MB–2GB. Current Power loads entire model weights into EPC. A 7B Q4 model is ~4GB — it simply won't fit.
- **Supply-chain auditability**: mistralrs pulls in the full candle ecosystem (hundreds of crates). In a TEE where every dependency is a potential attack vector, a smaller, auditable inference path has real security value.

Everything else in picolm (subprocess model, $10 hardware targets, grammar engine, tokenizer) either duplicates what Power already has or doesn't fit the TEE server architecture.

---

## What We Are NOT Doing

Before the plan, explicit rejections to prevent scope creep:

| Idea | Reason rejected |
|------|----------------|
| Adopt picolm's subprocess stdin/stdout model | Power is an HTTP service; IPC adds latency and complexity with no benefit |
| Target $10 / 256MB embedded hardware | Power's deployment target is TEE servers, not edge devices |
| Replace mistralrs/llamacpp with picolm | Three backends serve different needs; picolm is additive, not a replacement |
| Re-implement grammar/sampler from picolm | Power already has GBNF + mistralrs sampler; duplication is waste |
| Port picolm's tokenizer | mistralrs tokenizer is more complete; no reason to replace |

---

## Phase 1 — TEE Memory Efficiency (P0)

**Goal**: Run a 7B Q4 model inside a 512MB EPC.

**The problem today**: `MemoryDecryptedModel` decrypts the entire model into mlock-pinned RAM before inference begins. For a 7B Q4 model (~4GB), this exceeds typical EPC budgets. The server either OOMs or refuses to start.

**The solution**: A new `picolm` backend that streams transformer layers one at a time through RAM, keeping peak memory proportional to a single layer (~50–200MB) rather than the full model.

### 1.1 — picolm Backend (`backend/picolm.rs`)

Add a third inference backend that wraps the picolm C11 engine via FFI.

**New files**:
```
src/backend/picolm.rs          ← Backend trait impl wrapping picolm FFI
src/backend/picolm_stream.rs   ← Layer streaming protocol (load/infer/unload)
build.rs                       ← cc crate compiles picolm C sources
vendor/picolm/                 ← picolm C11 source (vendored, pinned commit)
```

**Feature gate**: `--features picolm` (disabled by default; opt-in for TEE deployments)

**Backend trait implementation**:
- `supports()` → GGUF format, LLaMA architecture
- `load()` → memory-maps model file, does NOT load weights into RAM
- `chat()` / `complete()` → streams layers through RAM one at a time during forward pass
- `unload()` → releases mmap, no zeroize needed (weights never in plaintext RAM)
- `embed()` → not supported (returns `BackendNotAvailable`)

**Key constraint**: picolm only supports LLaMA-architecture GGUF models (Q2_K through Q8_0). This is intentional — it's the common case for TEE deployments where model choice is controlled.

**Tests**:
```rust
// tests/picolm_backend.rs
test_backend_trait_compliance()     // load/chat/unload cycle
test_layer_streaming_memory()       // peak RSS stays under 2x layer size
test_gguf_format_detection()        // supports() returns true for GGUF
test_non_llama_rejected()           // unsupported arch returns clear error
```

### 1.2 — Memory-Aware Backend Routing

`BackendRegistry` currently selects backends by model format. Extend it to consider available EPC memory when in TEE mode.

**Change in `backend/mod.rs`**:
```rust
// New method on BackendRegistry
pub fn find_for_tee(
    &self,
    format: &ModelFormat,
    model_size_bytes: u64,
    epc_available_bytes: Option<u64>,  // None = not in TEE
) -> Option<&dyn Backend>
```

**Routing logic**:
- If not in TEE mode → existing behavior (priority order)
- If in TEE mode and `model_size > epc_available * 0.75` → prefer picolm backend if available
- If picolm not available and model won't fit → return `Err` with clear message: `"Model too large for TEE EPC ({}MB available, {}MB required). Enable the picolm feature for layer streaming."`

**EPC detection** (`tee/epc.rs`):
```rust
pub fn available_epc_bytes() -> Option<u64>
// Reads /sys/kernel/mm/hugepages or /proc/meminfo on Linux
// Returns None on non-Linux or non-TEE environments
```

**Tests**:
```rust
test_routes_to_picolm_when_model_exceeds_epc()
test_falls_back_to_mistralrs_when_model_fits()
test_clear_error_when_model_too_large_and_picolm_unavailable()
```

### 1.3 — Streaming Decryption for Encrypted Models

Today `MemoryDecryptedModel` decrypts the entire `.enc` model into mlock-pinned RAM. Combined with layer streaming, we can do better: decrypt one layer at a time, zeroize immediately after use.

**New type in `tee/encrypted_model.rs`**:
```rust
pub struct LayerStreamingDecryptedModel {
    // Decrypts AES-256-GCM blocks on demand
    // Each layer: decrypt → pass to picolm → zeroize
    // Peak plaintext in RAM: 1 layer at a time
}
```

**Constraint**: Only usable with the picolm backend (which controls the layer iteration loop). mistralrs and llamacpp load the full model before inference begins — streaming decryption is incompatible with them.

**Config addition**:
```hcl
# Existing
in_memory_decrypt = true   # full model in mlock RAM

# New (requires picolm feature)
streaming_decrypt = true   # one layer at a time, zeroize after use
```

**Tests**:
```rust
test_streaming_decrypt_peak_memory()    // peak plaintext < 2x layer size
test_streaming_decrypt_zeroize()        // layer buffer zeroized after each forward pass
test_streaming_decrypt_integrity()      // SHA-256 of reassembled plaintext matches expected
```

---

## Phase 2 — Supply-Chain Auditability (P1)

**Goal**: Provide a TEE deployment build profile where the entire inference path is auditable.

**The problem today**: The default `mistralrs` feature pulls in candle, which has a large dependency tree. In a TEE where supply-chain integrity matters, "auditable" means a human can read every line of inference code. mistralrs is excellent but not auditable in this sense.

### 2.1 — `tee-minimal` Feature Profile

Add a feature combination that produces the smallest, most auditable TEE build:

```toml
# Cargo.toml addition
[features]
# Minimal TEE build: picolm inference + full TEE stack, no C++ or large Rust deps
tee-minimal = ["picolm", "tls", "vsock"]
```

**What this gives you**:
- Inference: picolm (~2,500 lines C11, fully auditable)
- TEE: full attestation, model integrity, log redaction, encrypted model loading
- Transport: RA-TLS + Vsock (TEE-native transports)
- Excluded: mistralrs (candle ecosystem), llamacpp (C++ toolchain), hf (reqwest)

**Build command for TEE deployment**:
```bash
cargo build --release --no-default-features --features tee-minimal
```

**Documentation**: Add a "TEE Deployment" section to README.md explaining when to use `tee-minimal` vs default build.

### 2.2 — Dependency Audit Document

Create `docs/supply-chain.md` listing:
- Every crate in each feature combination's dependency tree
- Audit status for each (audited / not audited / vendored)
- How to reproduce the dependency count: `cargo tree --features tee-minimal | wc -l`
- Comparison table: `tee-minimal` vs `default` vs `llamacpp`

This is documentation, not code — but it's what enterprise TEE customers will ask for.

---

## Phase 3 — Benchmarks & Validation (P2)

**Goal**: Prove that layer streaming actually solves the EPC memory wall, with numbers.

### 3.1 — TEE Memory Benchmarks

```
benches/
  tee_memory.rs     ← peak RSS comparison: picolm vs mistralrs for same model
  layer_stream.rs   ← throughput (tok/s) vs model size for picolm backend
  epc_pressure.rs   ← inference under simulated EPC pressure (cgroups memory limit)
```

**Key metrics to capture**:
- Peak RSS during model load (picolm vs mistralrs)
- Peak RSS during inference (picolm vs mistralrs)
- Tokens/second at various EPC limits (512MB, 1GB, 2GB)
- Time-to-first-token under memory pressure

### 3.2 — Integration Tests for New Features

```
tests/
  picolm_tee.rs           ← full load/infer/unload cycle with picolm + TEE mode
  streaming_decrypt.rs    ← encrypted model + layer streaming end-to-end
  tee_minimal_build.rs    ← smoke test: tee-minimal feature compiles and serves requests
  epc_routing.rs          ← BackendRegistry routes correctly under EPC constraints
```

---

## Implementation Order

```
Week 1-2:  Phase 1.1 — picolm backend FFI + build.rs + vendor/
Week 2-3:  Phase 1.2 — EPC detection + memory-aware backend routing
Week 3-4:  Phase 1.3 — LayerStreamingDecryptedModel
Week 4:    Phase 2.1 — tee-minimal feature profile
Week 5:    Phase 2.2 — supply-chain.md
Week 5-6:  Phase 3   — benchmarks + integration tests
```

---

## Architecture After This Plan

```
BackendRegistry
├── MistralRsBackend   (default)  — full-featured, pure Rust, large dep tree
├── LlamaCppBackend    (llamacpp) — full-featured, C++, mature
└── PicolmBackend      (picolm)   — minimal, auditable, layer streaming ← NEW
      │
      └── LayerStreamingDecryptedModel ← NEW
            decrypt one layer → forward pass → zeroize → next layer
```

**Backend selection in TEE mode**:
```
model_size > epc * 0.75?
  yes → PicolmBackend (if available) → LayerStreamingDecryptedModel (if .enc)
  no  → MistralRsBackend (default)   → MemoryDecryptedModel (if .enc)
```

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| 7B Q4 model runs in 512MB EPC | ✓ with picolm backend |
| Peak plaintext RAM for encrypted 7B model | < 300MB (1-2 layers) |
| `tee-minimal` dependency count vs default | < 30% of default |
| All existing 755+ tests still pass | ✓ no regressions |
| New tests added | ≥ 40 (backend + routing + decrypt + bench) |
| `cargo build --no-default-features --features tee-minimal` succeeds | ✓ |
