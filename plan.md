# picolm LLM Inference Backend — Implementation Plan

## Status Quo

The picolm backend has working infrastructure:
- **GGUF parser** (`gguf_stream.rs`): mmap-based, parses header/metadata/tensor descriptors, provides zero-copy `tensor_bytes()` and `layer_tensor_names()`.
- **Backend trait impl** (`picolm.rs`): load/unload/chat/complete wiring, streaming via `mpsc` channel, sampler (top-p + temperature), byte-fallback tokenizer stub.
- **Forward pass**: Stub arithmetic — mixes raw weight bytes into hidden state with no real transformer math.

What's missing: dequantization, RMSNorm, RoPE attention, SwiGLU FFN, BPE tokenizer, KV cache.

---

## File Structure

```
src/backend/
├── mod.rs                  # (modify) — add `pub mod picolm_ops;`
├── picolm.rs               # (modify) — replace stub forward pass, wire tokenizer + KV cache
├── gguf_stream.rs          # (modify) — extend GgufMeta with n_ff, rope_theta, norm_eps; extract tokenizer vocab
└── picolm_ops/
    ├── mod.rs              # Re-exports all submodules
    ├── dequant.rs          # Dequantization kernels: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32
    ├── norm.rs             # RMSNorm
    ├── matmul.rs           # Matrix-vector multiply (dequantize-on-the-fly from raw &[u8])
    ├── rope.rs             # Rotary Position Embeddings
    ├── attention.rs        # Multi-head / Grouped-Query Attention (single-token decode path)
    ├── ffn.rs              # SwiGLU Feed-Forward Network
    ├── tokenizer.rs        # BPE tokenizer loaded from GGUF metadata
    └── kv_cache.rs         # Per-layer KV cache compatible with layer-streaming
```

**Rationale**: A `picolm_ops/` submodule keeps the math isolated and independently testable. Each file is <400 lines, auditable for TEE supply-chain review.

---

## Implementation Order & Dependencies

```
Phase 0 (P0-a): Dequantization + MatVec          ← foundation, no dependencies
Phase 0 (P0-b): RMSNorm                          ← depends on nothing
Phase 0 (P0-c): RoPE                             ← depends on nothing
Phase 0 (P0-d): Attention                        ← depends on P0-a, P0-b, P0-c
Phase 0 (P0-e): SwiGLU FFN                       ← depends on P0-a, P0-b
Phase 0 (P0-f): Forward pass integration          ← depends on P0-a through P0-e
Phase 1 (P1-a): BPE tokenizer from GGUF          ← depends on gguf_stream metadata extraction
Phase 1 (P1-b): KV cache                         ← depends on P0-d (attention shapes)
Phase 1 (P1-c): Wire everything into picolm.rs   ← depends on all above
```

---

## Phase 0-a: Dequantization Kernels (`dequant.rs`)

### GGML Block Formats

All quantized formats pack weights into fixed-size blocks. Each block contains a scale factor (and sometimes a minimum/offset) plus quantized nibbles or bytes.

```rust
/// Dequantize a raw GGUF tensor slice into f32.
/// Dispatches based on ggml_type. Output length = n_elements.
pub fn dequantize(raw: &[u8], ggml_type: u32, n_elements: usize, out: &mut [f32]);
```

#### Q4_0 (type 2) — 32 elements per block, 18 bytes/block
```
Layout: [f16 scale (2B)] [16 bytes of 4-bit nibbles]
Each byte holds 2 elements: lo nibble = elem[2i], hi nibble = elem[2i+1]
value = (nibble - 8) * scale
```

```rust
fn dequant_q4_0(block: &[u8; 18], out: &mut [f32; 32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..16 {
        let byte = block[2 + j];
        out[j]      = ((byte & 0x0F) as f32 - 8.0) * scale;
        out[j + 16] = ((byte >> 4)   as f32 - 8.0) * scale;
    }
}
```

#### Q8_0 (type 8) — 32 elements per block, 34 bytes/block
```
Layout: [f16 scale (2B)] [32 × i8 quants]
value = quant * scale
```

```rust
fn dequant_q8_0(block: &[u8; 34], out: &mut [f32; 32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..32 {
        out[j] = (block[2 + j] as i8) as f32 * scale;
    }
}
```

#### Q4_K (type 12) — 256 elements per block, 144 bytes/block
```
Layout: [f16 d (2B)] [f16 dmin (2B)] [12B scales] [4B high-bits of scales] [128B nibbles]
Super-block of 8 sub-blocks × 32 elements each.
Each sub-block has a 6-bit scale and 6-bit minimum extracted from the 16-byte scale section.
value = d * scale_i * nibble - dmin * min_i
```

#### Q5_K (type 13) — 256 elements per block, 176 bytes/block
```
Similar to Q4_K but 5-bit quantization. Extra high-bit byte per sub-block.
Layout: [f16 d (2B)] [f16 dmin (2B)] [12B scales] [4B high-bits] [32B high-bits of quants] [128B lo-nibbles]
```

#### Q6_K (type 14) — 256 elements per block, 210 bytes/block
```
Layout: [128B low nibbles] [64B high bits] [16B scales (i8)] [f16 d (2B)]
6-bit quantization, signed. value = d * scale_i * (quant - 32)
```

#### F16 (type 1)
```rust
fn dequant_f16(raw: &[u8], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = f16::from_le_bytes([raw[2*i], raw[2*i+1]]).to_f32();
    }
}
```

#### F32 (type 0)
```rust
fn dequant_f32(raw: &[u8], out: &mut [f32]) {
    // Direct reinterpret — same layout
    let src = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, out.len()) };
    out.copy_from_slice(src);
}
```

### Key Design Decision: No Intermediate Buffer for MatVec

The dequantization functions are designed to be called **block-at-a-time** inside the matrix-vector multiply loop, not to dequantize entire tensors. This keeps peak RAM at O(block_size) = 256 floats = 1KB, not O(tensor_size).


---

## Phase 0-a (cont): Matrix-Vector Multiply (`matmul.rs`)

The core compute primitive. Every attention projection and FFN layer is a matrix-vector product where the matrix is a quantized GGUF tensor and the vector is f32.

```rust
/// Matrix-vector multiply: out[i] = dot(row_i(weight), x) for i in 0..out_rows.
/// `weight_raw` is the raw GGUF bytes for a 2D tensor [out_rows × in_cols].
/// Dequantizes one block at a time — never materializes the full weight matrix.
pub fn matvec(
    weight_raw: &[u8],
    ggml_type: u32,
    out_rows: usize,
    in_cols: usize,
    x: &[f32],
    out: &mut [f32],
);
```

**Implementation strategy**: For each row, iterate over blocks within that row, dequantize the block into a small stack buffer (`[f32; 256]`), and accumulate the dot product. This is the "dequantize-on-the-fly" pattern.

```rust
pub fn matvec(weight_raw: &[u8], ggml_type: u32, out_rows: usize, in_cols: usize, x: &[f32], out: &mut [f32]) {
    let block_size = ggml_block_size(ggml_type);       // 32 or 256
    let block_bytes = ggml_block_bytes(ggml_type);      // e.g. 18 for Q4_0
    let blocks_per_row = in_cols / block_size;
    let row_bytes = blocks_per_row * block_bytes;

    for row in 0..out_rows {
        let row_data = &weight_raw[row * row_bytes..(row + 1) * row_bytes];
        let mut sum = 0.0f32;
        let mut buf = [0.0f32; 256]; // max block size

        for blk in 0..blocks_per_row {
            let blk_data = &row_data[blk * block_bytes..(blk + 1) * block_bytes];
            let col_offset = blk * block_size;
            dequantize_block(blk_data, ggml_type, &mut buf[..block_size]);
            for j in 0..block_size {
                sum += buf[j] * x[col_offset + j];
            }
        }
        out[row] = sum;
    }
}
```

**Helper functions**:
```rust
/// Block size (number of elements) for a GGML quantization type.
pub fn ggml_block_size(ggml_type: u32) -> usize;   // Q4_0/Q8_0 → 32, Q4_K/Q5_K/Q6_K → 256, F16/F32 → 1

/// Block byte size for a GGML quantization type.
pub fn ggml_block_bytes(ggml_type: u32) -> usize;  // matches gguf_stream::ggml_type_size logic

/// Dequantize a single block into f32 output buffer.
fn dequantize_block(block: &[u8], ggml_type: u32, out: &mut [f32]);
```

---

## Phase 0-b: RMSNorm (`norm.rs`)

LLaMA uses RMSNorm (not LayerNorm). Applied before attention and before FFN in each layer.

```rust
/// In-place RMSNorm: x[i] = x[i] / rms(x) * weight[i]
/// where rms(x) = sqrt(mean(x²) + eps)
///
/// `weight_raw` is the raw GGUF bytes for the norm weight tensor (1D, n_embd elements).
pub fn rms_norm(
    x: &mut [f32],
    weight_raw: &[u8],
    weight_ggml_type: u32,
    eps: f32,
);
```

```rust
pub fn rms_norm(x: &mut [f32], weight_raw: &[u8], weight_ggml_type: u32, eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms_inv = 1.0 / (ss + eps).sqrt();

    // Dequantize norm weights (typically F32, but handle any type)
    let mut w = vec![0.0f32; n];
    dequantize(weight_raw, weight_ggml_type, n, &mut w);

    for i in 0..n {
        x[i] = x[i] * rms_inv * w[i];
    }
}
```

**Note**: The norm weight is small (n_embd floats = 16KB for 4096-dim), so full dequantization is fine here.

---

## Phase 0-c: Rotary Position Embeddings (`rope.rs`)

RoPE encodes position information by rotating pairs of dimensions in Q and K vectors.

```rust
/// Apply RoPE to a vector of shape [n_heads, head_dim] stored as flat [n_heads * head_dim].
/// Only rotates the first `rope_dim` dimensions of each head (typically head_dim).
///
/// `pos` is the absolute token position in the sequence.
/// `theta_base` is the RoPE base frequency (default 10000.0, Llama3 uses 500000.0).
pub fn apply_rope(
    qk: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta_base: f32,
);
```

```rust
pub fn apply_rope(qk: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta_base: f32) {
    let rope_dim = head_dim; // LLaMA rotates all dimensions
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in (0..rope_dim).step_by(2) {
            let freq = 1.0 / theta_base.powf(i as f32 / rope_dim as f32);
            let angle = pos as f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();
            let x0 = qk[offset + i];
            let x1 = qk[offset + i + 1];
            qk[offset + i]     = x0 * cos_val - x1 * sin_val;
            qk[offset + i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}
```

**Architecture variants**:
- LLaMA 2/3: `theta_base = 10000.0` / `500000.0` (read from GGUF `rope.freq_base`)
- Mistral: same as LLaMA
- Phi: may use partial rotation (`rope_dim < head_dim`) — handle via metadata
- Gemma: same as LLaMA but may differ in theta


---

## Phase 0-d: Multi-Head Attention (`attention.rs`)

Single-token decode path (autoregressive generation). During generation we only compute attention for the **last token** — the KV cache holds all previous tokens.

### Tensor Names (LLaMA convention in GGUF)

Per layer `L`:
- `blk.L.attn_q.weight` — Q projection [n_embd × n_embd]
- `blk.L.attn_k.weight` — K projection [n_embd × n_kv_embd] (GQA: n_kv_embd = n_kv_heads × head_dim)
- `blk.L.attn_v.weight` — V projection [n_embd × n_kv_embd]
- `blk.L.attn_output.weight` — output projection [n_embd × n_embd]
- `blk.L.attn_norm.weight` — pre-attention RMSNorm [n_embd]

### Function Signature

```rust
/// Single-token attention for one transformer layer.
///
/// Reads Q/K/V/O weight tensors from GGUF, applies RMSNorm, projects,
/// applies RoPE, updates KV cache, computes scaled dot-product attention,
/// and projects output back to hidden dimension.
///
/// `hidden` is the residual stream [n_embd], modified in-place (residual add).
pub fn attention_layer(
    hidden: &mut [f32],
    gguf: &GgufFile,
    layer: u32,
    pos: usize,
    kv_cache: &mut LayerKvCache,
    cfg: &ModelConfig,
);
```

### ModelConfig (extracted from GgufMeta)

```rust
/// Model hyperparameters needed by the forward pass.
/// Extracted once from GgufMeta at load time.
pub struct ModelConfig {
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,       // n_embd / n_heads
    pub n_layers: u32,
    pub n_ff: usize,           // FFN intermediate size
    pub vocab_size: usize,
    pub norm_eps: f32,         // RMSNorm epsilon (default 1e-5)
    pub rope_theta: f32,       // RoPE base frequency (default 10000.0)
    pub context_length: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}
```

### Attention Algorithm (single-token decode)

```
1. norm_hidden = rms_norm(hidden, attn_norm_weight)
2. q = matvec(attn_q_weight, norm_hidden)          → [n_heads × head_dim]
3. k = matvec(attn_k_weight, norm_hidden)          → [n_kv_heads × head_dim]
4. v = matvec(attn_v_weight, norm_hidden)          → [n_kv_heads × head_dim]
5. apply_rope(q, n_heads, head_dim, pos, theta)
6. apply_rope(k, n_kv_heads, head_dim, pos, theta)
7. kv_cache.store(k, v, pos)
8. For each query head h:
   a. kv_group = h / (n_heads / n_kv_heads)       // GQA head mapping
   b. For each cached position p in 0..=pos:
      score[p] = dot(q_h, kv_cache.k[kv_group][p]) / sqrt(head_dim)
   c. softmax(score[0..=pos])
   d. attn_out_h = sum(score[p] * kv_cache.v[kv_group][p])
9. output = matvec(attn_output_weight, attn_out)   → [n_embd]
10. hidden += output                                 // residual connection
```

### Grouped-Query Attention (GQA)

LLaMA 2 70B, LLaMA 3, Mistral all use GQA where `n_kv_heads < n_heads`. Multiple query heads share the same KV head. The mapping is: `kv_head_index = query_head_index * n_kv_heads / n_heads`.

---

## Phase 0-e: SwiGLU FFN (`ffn.rs`)

LLaMA uses SwiGLU (Swish-Gated Linear Unit) instead of standard ReLU FFN.

### Tensor Names

Per layer `L`:
- `blk.L.ffn_gate.weight` — gate projection [n_ff × n_embd]
- `blk.L.ffn_up.weight` — up projection [n_ff × n_embd]
- `blk.L.ffn_down.weight` — down projection [n_embd × n_ff]
- `blk.L.ffn_norm.weight` — pre-FFN RMSNorm [n_embd]

### Function Signature

```rust
/// SwiGLU FFN for one transformer layer.
///
/// Reads gate/up/down weight tensors from GGUF, applies RMSNorm,
/// computes SwiGLU, and adds residual.
///
/// `hidden` is the residual stream [n_embd], modified in-place.
pub fn ffn_layer(
    hidden: &mut [f32],
    gguf: &GgufFile,
    layer: u32,
    cfg: &ModelConfig,
);
```

### SwiGLU Algorithm

```
1. norm_hidden = rms_norm(hidden, ffn_norm_weight)
2. gate = matvec(ffn_gate_weight, norm_hidden)     → [n_ff]
3. up   = matvec(ffn_up_weight, norm_hidden)       → [n_ff]
4. for i in 0..n_ff:
     gate[i] = gate[i] * sigmoid(gate[i])          // SiLU = x * sigmoid(x)
     gate[i] = gate[i] * up[i]                     // element-wise gate
5. down = matvec(ffn_down_weight, gate)             → [n_embd]
6. hidden += down                                    // residual connection
```

**Memory note**: `gate` and `up` are each `n_ff` floats. For LLaMA 7B, `n_ff = 11008`, so 2 × 11008 × 4 = ~86KB. This is the largest per-layer allocation besides the KV cache.


---

## Phase 0-f: Forward Pass Integration

Replace the stub `forward_pass_streaming` in `picolm.rs` with real transformer math.

### Full Forward Pass (single token, autoregressive)

```
For each generation step:
  1. Embed: hidden = embedding_lookup(token_id)
     - tensor: `token_embd.weight` [vocab_size × n_embd]
     - Extract row `token_id` via dequantize (single row, not full matrix)

  2. For layer in 0..n_layers:
     a. attention_layer(hidden, gguf, layer, pos, &mut kv_cache[layer], cfg)
        - Reads blk.{layer}.attn_* tensors from mmap
        - All tensor slices dropped after this call
     b. ffn_layer(hidden, gguf, layer, cfg)
        - Reads blk.{layer}.ffn_* tensors from mmap
        - All tensor slices dropped after this call

  3. Final norm: rms_norm(hidden, output_norm_weight)
     - tensor: `output_norm.weight` [n_embd]

  4. Logit projection: logits = matvec(output_weight, hidden)
     - tensor: `output.weight` [vocab_size × n_embd]
     - NOTE: For memory, compute logits in chunks or use the existing
       sample_token on the full logit vector (vocab_size × 4 bytes = ~128KB for 32K vocab)

  5. Sample next token from logits
  6. Decode token → text, send via channel
  7. pos += 1
```

### Embedding Lookup (row extraction)

```rust
/// Extract a single row from the embedding matrix without dequantizing the full tensor.
/// Returns [n_embd] floats.
pub fn embed_token(
    embd_raw: &[u8],
    ggml_type: u32,
    vocab_size: usize,
    n_embd: usize,
    token_id: u32,
    out: &mut [f32],
);
```

This extracts only the bytes for row `token_id` and dequantizes that single row. For Q8_0 with n_embd=4096, that's 4096/32 × 34 = 4352 bytes read, not the full embedding table.

### Prompt Processing (prefill)

For the initial prompt tokens, we need to process all tokens sequentially (no batched prefill in v1):

```
for (i, &token_id) in input_ids.iter().enumerate() {
    embed_token(..., token_id, &mut hidden);
    for layer in 0..n_layers {
        attention_layer(&mut hidden, gguf, layer, i, &mut kv_cache[layer], &cfg);
        ffn_layer(&mut hidden, gguf, layer, &cfg);
    }
    rms_norm(&mut hidden, output_norm_raw, output_norm_type, cfg.norm_eps);
    // Don't need logits for prompt tokens (except the last one)
}
// Now generate from the last prompt position
```

**Performance note**: Prefill is O(prompt_len × n_layers × layer_cost). For a 7B model with 32 layers and 512-token prompt, this means 32 × 512 = 16,384 layer passes. Each layer pass reads ~200MB from mmap (for Q4_K_M). The OS page cache makes repeated mmap reads fast, but prefill will still be slow compared to batched implementations. This is acceptable for TEE use cases where security > throughput.

---

## Phase 1-a: BPE Tokenizer (`tokenizer.rs`)

### GGUF Metadata Keys for Tokenizer

```
tokenizer.ggml.model          → "llama" | "gpt2" | ...
tokenizer.ggml.tokens         → Array[String] — the vocabulary (token → string)
tokenizer.ggml.scores         → Array[F32] — merge priority scores
tokenizer.ggml.token_type     → Array[I32] — 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte
tokenizer.ggml.bos_token_id   → U32
tokenizer.ggml.eos_token_id   → U32
tokenizer.ggml.merges         → Array[String] — BPE merge rules (optional, gpt2-style)
```

### Data Structures

```rust
/// BPE tokenizer loaded from GGUF metadata.
pub struct BpeTokenizer {
    /// Token ID → string piece
    vocab: Vec<String>,
    /// Token ID → merge priority score (lower = merge first)
    scores: Vec<f32>,
    /// Token ID → type (normal, byte, control, etc.)
    token_types: Vec<i32>,
    /// String piece → token ID (for encoding)
    piece_to_id: HashMap<String, u32>,
    /// Byte fallback tokens: byte value → token ID
    byte_tokens: [Option<u32>; 256],
    pub bos_id: u32,
    pub eos_id: u32,
}
```

### Encoding Algorithm (SentencePiece-style BPE)

LLaMA uses SentencePiece BPE (not GPT-2 BPE). The algorithm:

```
1. Prepend space to input: " " + text (SentencePiece convention)
2. Convert to UTF-8 bytes
3. Initialize: each byte → byte fallback token (or single-char token if in vocab)
4. Repeat until no more merges:
   a. For each adjacent pair (tokens[i], tokens[i+1]):
      - candidate = vocab[tokens[i]] + vocab[tokens[i+1]]
      - if candidate is in piece_to_id: record (i, merged_id, score)
   b. Find the pair with the lowest score (highest priority)
   c. Merge: replace tokens[i..i+2] with [merged_id]
5. Return token IDs
```

### Decoding Algorithm

```rust
impl BpeTokenizer {
    pub fn decode(&self, token_id: u32) -> Option<String> {
        if token_id as usize >= self.vocab.len() { return None; }
        let piece = &self.vocab[token_id as usize];
        // Handle byte tokens: <0xHH> → single byte
        if piece.starts_with("<0x") && piece.ends_with('>') {
            let hex = &piece[3..piece.len()-1];
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                return Some(String::from(byte as char));
            }
        }
        // Handle SentencePiece space: ▁ (U+2581) → ' '
        Some(piece.replace('▁', " "))
    }
}
```

### Changes to `gguf_stream.rs`

Extend `GgufMeta` and the parser to extract tokenizer arrays:

```rust
// Add to GgufMeta:
pub n_ff: u32,
pub norm_eps: f32,
pub rope_theta: f32,
pub rope_dim: Option<u32>,       // for Phi partial rotation
pub vocab_tokens: Vec<String>,   // tokenizer.ggml.tokens
pub vocab_scores: Vec<f32>,      // tokenizer.ggml.scores
pub vocab_types: Vec<i32>,       // tokenizer.ggml.token_type
```

**Memory note**: For a 32K vocab, `vocab_tokens` is ~32K strings averaging ~6 bytes each = ~200KB. `vocab_scores` = 128KB. This is negligible compared to the KV cache.


---

## Phase 1-b: KV Cache (`kv_cache.rs`)

### The Design Challenge: KV Cache × Layer Streaming

The core tension: layer-streaming means we process one layer at a time and discard its weights. But attention needs the KV cache from **all previous positions** for **this layer**. The KV cache must persist across generation steps.

**Key insight**: The KV cache is NOT part of the "weights" that get streamed. Weights are read-only (from mmap) and discarded. The KV cache is mutable state that grows with sequence length. It must be kept in RAM.

### Memory Budget Analysis

For a 7B LLaMA model (n_layers=32, n_kv_heads=32, head_dim=128, context=4096):

```
KV cache per layer = 2 × n_kv_heads × head_dim × context_length × 4 bytes
                   = 2 × 32 × 128 × 4096 × 4
                   = 128 MB per layer (at full context)

Total KV cache    = 128 MB × 32 layers = 4 GB  ← TOO MUCH for 512MB EPC
```

For GQA models (LLaMA 3 8B: n_kv_heads=8):
```
KV cache per layer = 2 × 8 × 128 × 4096 × 4 = 32 MB per layer
Total KV cache     = 32 MB × 32 = 1 GB  ← still large
```

### Solution: Bounded KV Cache with Sliding Window

For TEE environments, we bound the KV cache to fit within the EPC budget:

```rust
/// Per-layer KV cache for one transformer layer.
pub struct LayerKvCache {
    /// K cache: [n_kv_heads × head_dim × max_seq_len], stored as [max_seq_len][n_kv_heads * head_dim]
    k: Vec<f32>,
    /// V cache: same shape as K
    v: Vec<f32>,
    /// Number of positions currently stored
    len: usize,
    /// Maximum sequence length (bounded)
    max_seq_len: usize,
    /// Dimensions
    n_kv_heads: usize,
    head_dim: usize,
}

/// Full KV cache for all layers.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
}
```

### KV Cache API

```rust
impl LayerKvCache {
    /// Create a new empty KV cache for one layer.
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self;

    /// Store K and V vectors for position `pos`.
    /// If pos >= max_seq_len, shifts the window (drops oldest position).
    pub fn store(&mut self, k: &[f32], v: &[f32], pos: usize);

    /// Get K vector for a specific cached position and KV head.
    /// Returns &[head_dim].
    pub fn k_at(&self, pos: usize, kv_head: usize) -> &[f32];

    /// Get V vector for a specific cached position and KV head.
    pub fn v_at(&self, pos: usize, kv_head: usize) -> &[f32];

    /// Number of positions currently cached.
    pub fn len(&self) -> usize;
}

impl KvCache {
    /// Create KV cache for all layers.
    pub fn new(n_layers: u32, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self;

    /// Get mutable reference to a specific layer's cache.
    pub fn layer_mut(&mut self, layer: u32) -> &mut LayerKvCache;

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize;

    /// Clear all cached positions (for new conversation).
    pub fn clear(&mut self);
}
```

### Memory Layout

Store KV as position-major for cache-friendly sequential access during attention:

```
k_data layout: [pos_0: [kv_head_0: [head_dim floats], kv_head_1: [...], ...],
                pos_1: [...],
                ...]
```

This means `k_at(pos, kv_head)` is a contiguous slice at offset `pos * n_kv_heads * head_dim + kv_head * head_dim`.

### Interaction with Layer Streaming

The forward pass loop becomes:

```rust
fn forward_pass(gguf: &GgufFile, token_id: u32, pos: usize, kv_cache: &mut KvCache, cfg: &ModelConfig, hidden: &mut [f32]) {
    // 1. Embed
    let embd_raw = gguf.tensor_bytes("token_embd.weight")?;
    embed_token(embd_raw, embd_type, cfg.vocab_size, cfg.n_embd, token_id, hidden);
    // embd_raw dropped (mmap slice, no RAM freed, but no longer referenced)

    // 2. Layer-streaming loop
    for layer in 0..cfg.n_layers {
        // Attention: reads blk.{layer}.attn_* from mmap, updates kv_cache.layer_mut(layer)
        attention_layer(hidden, gguf, layer, pos, kv_cache.layer_mut(layer), cfg);
        // All attn weight mmap slices dropped here

        // FFN: reads blk.{layer}.ffn_* from mmap
        ffn_layer(hidden, gguf, layer, cfg);
        // All FFN weight mmap slices dropped here
    }

    // 3. Final norm
    let norm_raw = gguf.tensor_bytes("output_norm.weight")?;
    rms_norm(hidden, norm_raw, norm_type, cfg.norm_eps);
}
```

**Peak RAM during inference**:
- `hidden`: n_embd × 4 = 16 KB (4096-dim)
- Attention scratch: q (16KB) + k (4KB for GQA) + v (4KB) + scores (context × 4) + attn_out (16KB) ≈ 56KB + context×4
- FFN scratch: gate (44KB) + up (44KB) + down (16KB) ≈ 104KB (for n_ff=11008)
- KV cache: the only large persistent allocation (see budget above)
- Mmap slices: zero-copy pointers, no RAM cost (OS page cache handles it)

**Total peak RAM (excluding KV cache)**: ~200KB of scratch buffers. The KV cache dominates.

### Session-Aware KV Cache

The existing `ChatRequest.session_id` field enables KV cache reuse across turns:

```rust
// In LoadedModel, add:
struct LoadedModel {
    gguf: Arc<GgufFile>,
    cfg: ModelConfig,
    /// Session-keyed KV caches. Each session gets its own cache.
    kv_caches: HashMap<String, KvCache>,
    /// Anonymous (no session_id) requests get a transient cache, cleared after each request.
    transient_kv: KvCache,
}
```

---

## Phase 1-c: Wire Everything into `picolm.rs`

### Changes to `picolm.rs`

1. **Replace `encode_prompt` / `decode_token`** with `BpeTokenizer` loaded from GGUF metadata
2. **Replace `forward_pass_streaming`** with real transformer forward pass
3. **Add `ModelConfig`** extraction from `GgufMeta` in `load()`
4. **Add `KvCache`** to `LoadedModel`
5. **Add `BpeTokenizer`** to `LoadedModel`
6. **Update `cleanup_request`** to clear session KV cache when requested

### Changes to `gguf_stream.rs`

1. **Extend `GgufMeta`** with: `n_ff`, `norm_eps`, `rope_theta`, `rope_dim`, `vocab_tokens`, `vocab_scores`, `vocab_types`
2. **Extract these from GGUF metadata** in `parse_gguf_header`:
   - `{arch}.feed_forward_length` → `n_ff`
   - `{arch}.attention.layer_norm_rms_epsilon` → `norm_eps`
   - `{arch}.rope.freq_base` → `rope_theta`
   - `{arch}.rope.dimension_count` → `rope_dim`
   - `tokenizer.ggml.tokens` → `vocab_tokens`
   - `tokenizer.ggml.scores` → `vocab_scores`
   - `tokenizer.ggml.token_type` → `vocab_types`
3. **Add `MetaValue::as_string_array()`** and **`as_f32_array()`** helpers

### Changes to `mod.rs`

1. Add `pub mod picolm_ops;` (feature-gated)

---

## Testing Strategy

### Unit Tests (per module, no model file needed)

| Module | Tests |
|--------|-------|
| `dequant.rs` | Known-answer tests for each quant type: hand-compute expected f32 from known bytes |
| `norm.rs` | RMSNorm of a known vector against numpy reference |
| `rope.rs` | RoPE at pos=0 (identity for cos=1,sin=0), pos=1 against reference |
| `matmul.rs` | Small 4×4 F32 matvec, then Q8_0 matvec against dequant+naive multiply |
| `attention.rs` | Single-head attention with identity weights, verify output = input |
| `ffn.rs` | SwiGLU with known gate/up/down, verify against manual computation |
| `tokenizer.rs` | Encode/decode roundtrip with synthetic vocab |
| `kv_cache.rs` | Store/retrieve, sliding window eviction, memory accounting |

### Integration Tests (require a small GGUF model)

- Download a tiny test model (e.g., TinyLlama 1.1B Q4_K_M, ~600MB) in CI
- Verify: load → encode prompt → generate 10 tokens → decode → non-empty coherent text
- Verify: KV cache reuse across two turns produces different output than fresh start
- Verify: peak RSS stays under expected bound

### Benchmarks

- `layer_stream` bench (already exists): update to use real forward pass
- Add `dequant` bench: throughput of each quant type in GB/s
- Add `matvec` bench: GFLOPS for Q4_K matvec at typical dimensions

---

## Architecture Variant Handling

| Feature | LLaMA 2 | LLaMA 3 | Mistral | Phi-3 | Gemma |
|---------|---------|---------|---------|-------|-------|
| Attention | MHA/GQA | GQA | GQA+sliding | MHA | MHA |
| FFN | SwiGLU | SwiGLU | SwiGLU | SwiGLU | GeGLU |
| Norm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| RoPE theta | 10000 | 500000 | 10000 | 10000 | 10000 |
| RoPE dim | full | full | full | partial | full |
| Vocab | 32K | 128K | 32K | 32K | 256K |

All variants are handled by reading hyperparameters from GGUF metadata rather than hardcoding. The `ModelConfig` struct captures all variant-specific values.

**Gemma GeGLU note**: GeGLU uses `GELU(gate) * up` instead of `SiLU(gate) * up`. Add a `ffn_act` field to `ModelConfig` and branch in `ffn.rs`. This is a 3-line change.

**Mistral sliding window note**: Mistral uses a sliding window for attention (window_size from metadata). The KV cache `store()` method already supports this via the `max_seq_len` bound. For Mistral, set `max_seq_len = min(context_length, sliding_window_size)`.

---

## Dependency Summary

No new crate dependencies needed. The existing `picolm` feature deps are sufficient:
- `half` — f16 ↔ f32 conversion (used in dequantization)
- `memmap2` — mmap for GGUF (already used)
- `candle-core` — **can be dropped** after this implementation (was a placeholder dep; we implement everything from scratch)

**Recommendation**: Remove `candle-core` from the `picolm` feature after implementation. It's a large dependency tree that contradicts the "zero C dependencies, fully auditable" goal. The `half` crate alone provides the f16 support we need.

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Prefill too slow (sequential token processing) | Poor UX for long prompts | Accept for v1; add batched prefill in v2 (process multiple tokens per layer pass) |
| KV cache exceeds EPC budget | OOM in TEE | Bounded max_seq_len; log warning when approaching limit |
| Numerical divergence from reference impl | Wrong outputs | Validate against llama.cpp output for same model+prompt+seed |
| Q4_K/Q5_K dequant bugs | Silent wrong results | Extensive known-answer tests; compare block-by-block against llama.cpp's dequant |
| Tokenizer edge cases (Unicode, special tokens) | Encoding errors | Test with multilingual inputs; fall back to byte tokens for unknown chars |
