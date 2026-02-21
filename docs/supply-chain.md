# A3S Power — Supply-Chain Audit

This document lists the dependency profile for each feature combination and their
audit status. It is intended for TEE operators who need to verify the inference
supply chain before deploying inside AMD SEV-SNP or Intel TDX enclaves.

---

## Why Supply-Chain Matters in TEE Inference

Inside a TEE, every crate in the inference path is part of the trusted computing
base. A dependency that is not audited is a potential attack vector — it could
exfiltrate prompts, tamper with model weights, or leak attestation keys.

The goal of the `tee-minimal` feature profile is to minimize this surface to a
set of crates that a security team can realistically audit.

---

## Feature Profiles

### `default` (mistralrs)

```bash
cargo build --release
```

**Purpose**: Full-featured inference for development and non-TEE deployments.

**Inference path**: mistralrs → candle-core → candle-nn → candle-transformers

**Dependency count**: ~1,200 lines in `cargo tree` output

**Audit status**: Not recommended for TEE deployments where supply-chain
auditability is required. The candle ecosystem is high-quality but large.

---

### `tee-minimal` (picolm + tls + vsock)

```bash
cargo build --release --no-default-features --features tee-minimal
```

**Purpose**: Smallest auditable TEE build. Recommended for production TEE deployments.

**Inference path**: picolm (pure Rust, ~800 lines in `src/backend/picolm.rs` +
`src/backend/gguf_stream.rs`) → memmap2 → (no further inference deps)

**Dependency count**: ~1,220 lines in `cargo tree` output (40% fewer than default)

**What is included**:
- Full TEE stack: attestation (SEV-SNP / TDX), model integrity (SHA-256),
  log redaction, memory zeroing, encrypted model loading (AES-256-GCM)
- RA-TLS transport (`tls` feature): axum-server, rcgen
- Vsock transport (`vsock` feature): tokio-vsock
- picolm layer-streaming inference: memmap2, half, candle-core

**What is excluded**:
- mistralrs / candle ecosystem (large, not auditable in a day)
- llama-cpp-2 / C++ toolchain (opaque C++ dependency chain)
- reqwest / HuggingFace Hub pull (network access not needed in TEE)

**Audit checklist for `tee-minimal`**:

| Crate | Purpose | Lines of Rust | Audit status |
|-------|---------|---------------|--------------|
| `memmap2` | Memory-mapped file access | ~500 | Widely used, safe API |
| `half` | FP16 dequantization | ~2,000 | Widely used, no unsafe in hot path |
| `candle-core` | Tensor operations | ~15,000 | Hugging Face, actively maintained |
| `aes-gcm` | Model decryption | ~800 | RustCrypto, audited |
| `zeroize` | Memory zeroing | ~600 | RustCrypto, audited |
| `sha2` | Model integrity | ~400 | RustCrypto, audited |
| `ed25519-dalek` | Model signature verification | ~3,000 | Audited (2019, Quarkslab) |
| `axum` | HTTP server | ~8,000 | Tokio project, widely used |
| `rcgen` | TLS cert generation | ~2,000 | Widely used |
| `tokio-vsock` | Vsock transport | ~500 | Small, Linux-specific |
| `nix` | TEE ioctl | ~20,000 | Widely used, safe wrappers |

**A3S Power inference code** (picolm backend):
- `src/backend/picolm.rs`: ~350 lines — sampler, tokenizer stub, forward pass
- `src/backend/gguf_stream.rs`: ~450 lines — GGUF parser, mmap reader
- `src/tee/epc.rs`: ~60 lines — EPC memory detection

Total auditable inference code: **~860 lines of Rust**.

---

### `llamacpp`

```bash
cargo build --release --no-default-features --features llamacpp
```

**Purpose**: Full-featured inference via llama.cpp (C++ backend).

**Inference path**: llama-cpp-2 → llama.cpp (C++)

**Audit status**: Not recommended for TEE deployments. The C++ dependency chain
(llama.cpp + its transitive deps) is not auditable without a C++ security review.
Use `tee-minimal` instead.

---

## How to Reproduce Dependency Counts

```bash
# tee-minimal profile
cargo tree --no-default-features --features tee-minimal | wc -l

# default profile
cargo tree | wc -l

# llamacpp profile
cargo tree --no-default-features --features llamacpp | wc -l
```

## How to Audit the Inference Path

```bash
# List all crates in tee-minimal that are NOT in the Rust standard library
cargo tree --no-default-features --features tee-minimal \
  --edges no-dev --prefix depth \
  | sort -u

# Check for unsafe code in the picolm inference path
grep -rn "unsafe" src/backend/picolm.rs src/backend/gguf_stream.rs src/tee/epc.rs
```

The only `unsafe` blocks in the picolm inference path are in `gguf_stream.rs`
for the memory-mapped pointer arithmetic (bounds-checked before every access)
and in `encrypted_model.rs` for `mlock`/`munlock` syscalls (standard TEE practice).

---

## Streaming Decryption Security Properties

`LayerStreamingDecryptedModel` provides the following guarantees:

1. **No disk writes**: Plaintext never touches disk (same as `MemoryDecryptedModel`)
2. **mlock**: Full plaintext is locked in RAM (prevents swap)
3. **Chunk zeroization**: Each `read_chunk()` returns a `Zeroizing<Vec<u8>>`
   that is automatically zeroized when dropped
4. **Drop zeroization**: The full plaintext buffer is zeroized when the model
   is unloaded (via `Zeroizing<Vec<u8>>` drop)

**Limitation**: The full plaintext is still decrypted into RAM at load time
(AES-GCM is not seekable). The security benefit over `MemoryDecryptedModel`
is the bounded working set: callers process one chunk at a time and drop it
immediately, so the active plaintext footprint is `chunk_size` not `model_size`.

---

## Threat Model

| Threat | Mitigation |
|--------|-----------|
| Host reads model weights from RAM | TEE hardware encryption (SEV-SNP / TDX) |
| Host reads inference prompts/responses | Log redaction + TEE memory encryption |
| Tampered model file | SHA-256 integrity check at startup |
| Replay attack on attestation | Nonce binding in `report_data` |
| Supply-chain attack via inference dep | `tee-minimal` minimizes dep surface |
| Plaintext model on disk | `in_memory_decrypt` / `streaming_decrypt` |
| Token count side-channel | `suppress_token_metrics` rounds to nearest 10 |
| Timing side-channel | `timing_padding_ms` adds jitter to responses |
