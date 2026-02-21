//! Synthetic GGUF v3 file builder for integration tests.
//!
//! Builds a minimal but valid GGUF file with real F32 tensor data so the
//! picolm forward pass can run end-to-end without a real model download.

use std::path::Path;

// GGUF constants
const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

/// Tiny model config for testing.
pub struct TinyModelConfig {
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_ff: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
}

impl Default for TinyModelConfig {
    fn default() -> Self {
        Self {
            n_embd: 8,
            n_heads: 2,
            n_kv_heads: 2,
            n_ff: 16,
            n_layers: 1,
            vocab_size: 8,
        }
    }
}

/// Build a synthetic GGUF file and write it to `path`.
pub fn build_tiny_gguf(path: &Path, cfg: &TinyModelConfig) {
    let head_dim = cfg.n_embd / cfg.n_heads;
    let kv_dim = cfg.n_kv_heads * head_dim;

    // Collect all tensors: (name, shape_rows, shape_cols)
    // All tensors are F32 (ggml_type=0)
    let mut tensor_defs: Vec<(&str, String, usize, usize)> = Vec::new();

    // Global tensors
    tensor_defs.push(("global", "token_embd.weight".into(), cfg.vocab_size, cfg.n_embd));
    tensor_defs.push(("global", "output_norm.weight".into(), 1, cfg.n_embd));

    // Per-layer tensors
    for l in 0..cfg.n_layers {
        let layer = |name: &str| format!("blk.{l}.{name}");
        tensor_defs.push(("layer", layer("attn_norm.weight"), 1, cfg.n_embd));
        tensor_defs.push(("layer", layer("attn_q.weight"), cfg.n_heads * head_dim, cfg.n_embd));
        tensor_defs.push(("layer", layer("attn_k.weight"), kv_dim, cfg.n_embd));
        tensor_defs.push(("layer", layer("attn_v.weight"), kv_dim, cfg.n_embd));
        tensor_defs.push(("layer", layer("attn_output.weight"), cfg.n_embd, cfg.n_heads * head_dim));
        tensor_defs.push(("layer", layer("ffn_norm.weight"), 1, cfg.n_embd));
        tensor_defs.push(("layer", layer("ffn_gate.weight"), cfg.n_ff, cfg.n_embd));
        tensor_defs.push(("layer", layer("ffn_up.weight"), cfg.n_ff, cfg.n_embd));
        tensor_defs.push(("layer", layer("ffn_down.weight"), cfg.n_embd, cfg.n_ff));
    }

    // Build vocab tokens: ["<unk>", "<s>", "</s>", "a", "b", "c", "d", "e"]
    let vocab: Vec<String> = (0..cfg.vocab_size)
        .map(|i| match i {
            0 => "<unk>".into(),
            1 => "<s>".into(),
            2 => "</s>".into(),
            _ => format!("{}", (b'a' + (i - 3) as u8) as char),
        })
        .collect();

    // ── Build metadata KV pairs ──
    let mut meta_buf: Vec<u8> = Vec::new();
    let mut n_kv: u64 = 0;

    // Helper closures
    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        write_gguf_string(buf, val);
    }

    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_U32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, vals: &[String]) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            write_gguf_string(buf, v);
        }
    }

    fn write_kv_f32_array(buf: &mut Vec<u8>, key: &str, vals: &[f32]) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn write_kv_i32_array(buf: &mut Vec<u8>, key: &str, vals: &[i32]) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_I32.to_le_bytes());
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    // Architecture
    write_kv_string(&mut meta_buf, "general.architecture", "llama"); n_kv += 1;
    write_kv_u32(&mut meta_buf, "llama.block_count", cfg.n_layers as u32); n_kv += 1;
    write_kv_u32(&mut meta_buf, "llama.embedding_length", cfg.n_embd as u32); n_kv += 1;
    write_kv_u32(&mut meta_buf, "llama.attention.head_count", cfg.n_heads as u32); n_kv += 1;
    write_kv_u32(&mut meta_buf, "llama.attention.head_count_kv", cfg.n_kv_heads as u32); n_kv += 1;
    write_kv_u32(&mut meta_buf, "llama.context_length", 64); n_kv += 1;
    write_kv_u32(&mut meta_buf, "llama.feed_forward_length", cfg.n_ff as u32); n_kv += 1;
    write_kv_f32(&mut meta_buf, "llama.attention.layer_norm_rms_epsilon", 1e-5); n_kv += 1;
    write_kv_f32(&mut meta_buf, "llama.rope.freq_base", 10000.0); n_kv += 1;

    // Tokenizer
    write_kv_u32(&mut meta_buf, "tokenizer.ggml.bos_token_id", 1); n_kv += 1;
    write_kv_u32(&mut meta_buf, "tokenizer.ggml.eos_token_id", 2); n_kv += 1;
    write_kv_string_array(&mut meta_buf, "tokenizer.ggml.tokens", &vocab); n_kv += 1;

    let scores: Vec<f32> = (0..cfg.vocab_size).map(|i| -(i as f32)).collect();
    write_kv_f32_array(&mut meta_buf, "tokenizer.ggml.scores", &scores); n_kv += 1;

    // token types: 0=unk, 1=normal for BOS, 3=control for EOS, 1=normal for rest
    let token_types: Vec<i32> = (0..cfg.vocab_size)
        .map(|i| match i {
            0 => 0, // unknown
            1 => 3, // control (BOS)
            2 => 3, // control (EOS)
            _ => 1, // normal
        })
        .collect();
    write_kv_i32_array(&mut meta_buf, "tokenizer.ggml.token_type", &token_types); n_kv += 1;

    // ── Build tensor descriptors ──
    let n_tensors = tensor_defs.len() as u64;
    let mut tensor_desc_buf: Vec<u8> = Vec::new();
    let mut tensor_data_buf: Vec<u8> = Vec::new();
    let mut running_offset: u64 = 0;

    // Simple PRNG for deterministic "random" weights
    let mut rng: u32 = 0xDEAD_BEEF;

    for (_category, name, rows, cols) in &tensor_defs {
        let n_elements = if *rows == 1 { *cols } else { rows * cols };
        let is_1d = *rows == 1;

        // Write tensor descriptor
        write_gguf_string(&mut tensor_desc_buf, name);
        if is_1d {
            tensor_desc_buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            tensor_desc_buf.extend_from_slice(&(*cols as u64).to_le_bytes());
        } else {
            tensor_desc_buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
            tensor_desc_buf.extend_from_slice(&(*cols as u64).to_le_bytes()); // dim0 = cols (inner)
            tensor_desc_buf.extend_from_slice(&(*rows as u64).to_le_bytes()); // dim1 = rows (outer)
        }
        tensor_desc_buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32
        tensor_desc_buf.extend_from_slice(&running_offset.to_le_bytes());

        // Write tensor data (F32)
        // For norm weights, use 1.0 (identity norm)
        // For other weights, use small random values
        for _ in 0..n_elements {
            let val = if name.contains("norm") {
                1.0f32
            } else {
                // xorshift32
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                // Map to [-0.1, 0.1]
                (rng as f32 / u32::MAX as f32) * 0.2 - 0.1
            };
            tensor_data_buf.extend_from_slice(&val.to_le_bytes());
        }

        running_offset += (n_elements * 4) as u64;
    }

    // ── Assemble the file ──
    let mut file_buf: Vec<u8> = Vec::new();

    // Header
    file_buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    file_buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    file_buf.extend_from_slice(&n_tensors.to_le_bytes());
    file_buf.extend_from_slice(&n_kv.to_le_bytes());

    // Metadata KV
    file_buf.extend_from_slice(&meta_buf);

    // Tensor descriptors
    file_buf.extend_from_slice(&tensor_desc_buf);

    // Align to 32 bytes
    let alignment = 32;
    let padding = (alignment - (file_buf.len() % alignment)) % alignment;
    file_buf.resize(file_buf.len() + padding, 0);

    // Tensor data
    file_buf.extend_from_slice(&tensor_data_buf);

    std::fs::write(path, &file_buf).unwrap();
}
