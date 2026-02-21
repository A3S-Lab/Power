//! Matrix-vector multiply with on-the-fly dequantization.
//!
//! The core compute primitive for transformer inference. Every attention
//! projection and FFN layer is a matvec where the matrix is a quantized
//! GGUF tensor and the vector is f32.
//!
//! Dequantizes one block at a time — never materializes the full weight matrix.
//! Peak buffer = 256 floats = 1 KB.

use super::dequant::{block_bytes, block_size, dequantize_block};

/// Matrix-vector multiply: `out[i] = dot(row_i(weight), x)` for `i` in `0..out_rows`.
///
/// `weight_raw` is the raw GGUF bytes for a 2D tensor `[out_rows × in_cols]`.
/// Dequantizes one block at a time — never materializes the full weight matrix.
pub fn matvec(
    weight_raw: &[u8],
    ggml_type: u32,
    out_rows: usize,
    in_cols: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let bs = block_size(ggml_type);
    let bb = block_bytes(ggml_type);

    // For element-wise types (F32, F16), bs=1 and we process one element at a time.
    // For block types, in_cols must be a multiple of block_size.
    let blocks_per_row = if bs == 1 { in_cols } else { in_cols / bs };
    let row_bytes = blocks_per_row * bb;

    let mut buf = [0.0f32; 256]; // max block size

    for (row, out_val) in out[..out_rows].iter_mut().enumerate() {
        let row_start = row * row_bytes;
        let row_data = &weight_raw[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;

        for blk in 0..blocks_per_row {
            let blk_start = blk * bb;
            let blk_data = &row_data[blk_start..blk_start + bb];
            let col_offset = blk * bs;

            dequantize_block(blk_data, ggml_type, &mut buf[..bs]);

            // Dot product of dequantized block with corresponding x slice
            for j in 0..bs {
                sum += buf[j] * x[col_offset + j];
            }
        }
        *out_val = sum;
    }
}

/// Extract a single row from a 2D quantized tensor and dequantize it.
///
/// Used for embedding lookup: extract row `token_id` from `[vocab_size × n_embd]`.
/// Only reads the bytes for that one row — does not touch the rest of the tensor.
pub fn extract_row(
    tensor_raw: &[u8],
    ggml_type: u32,
    n_cols: usize,
    row: usize,
    out: &mut [f32],
) {
    let bs = block_size(ggml_type);
    let bb = block_bytes(ggml_type);
    let blocks_per_row = if bs == 1 { n_cols } else { n_cols / bs };
    let row_bytes = blocks_per_row * bb;
    let row_start = row * row_bytes;
    let row_data = &tensor_raw[row_start..row_start + row_bytes];

    let mut buf = [0.0f32; 256];
    for blk in 0..blocks_per_row {
        let blk_start = blk * bb;
        let blk_data = &row_data[blk_start..blk_start + bb];
        let col_offset = blk * bs;

        dequantize_block(blk_data, ggml_type, &mut buf[..bs]);
        let copy_len = bs.min(n_cols - col_offset);
        out[col_offset..col_offset + copy_len].copy_from_slice(&buf[..copy_len]);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_f32_identity() {
        // 2×2 identity matrix in F32
        let mut weight = Vec::new();
        // Row 0: [1.0, 0.0]
        weight.extend_from_slice(&1.0f32.to_le_bytes());
        weight.extend_from_slice(&0.0f32.to_le_bytes());
        // Row 1: [0.0, 1.0]
        weight.extend_from_slice(&0.0f32.to_le_bytes());
        weight.extend_from_slice(&1.0f32.to_le_bytes());

        let x = [3.0f32, 7.0];
        let mut out = [0.0f32; 2];
        matvec(&weight, 0, 2, 2, &x, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_f32_3x2() {
        // [[1, 2], [3, 4], [5, 6]] × [1, 1] = [3, 7, 11]
        let mut weight = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            weight.extend_from_slice(&v.to_le_bytes());
        }
        let x = [1.0f32, 1.0];
        let mut out = [0.0f32; 3];
        matvec(&weight, 0, 3, 2, &x, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
        assert!((out[2] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_matvec_q8_0() {
        // 1 row × 32 cols in Q8_0
        // scale = 1.0, quants = [1, 1, 1, ..., 1]
        // dot with x = [1, 1, ..., 1] should give 32.0
        let scale = half::f16::from_f32(1.0);
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        for j in 0..32 {
            block[2 + j] = 1u8; // i8 = 1
        }

        let x = [1.0f32; 32];
        let mut out = [0.0f32; 1];
        matvec(&block, 8, 1, 32, &x, &mut out);
        assert!((out[0] - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_extract_row_f32() {
        // 3×2 matrix, extract row 1 = [3.0, 4.0]
        let mut tensor = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            tensor.extend_from_slice(&v.to_le_bytes());
        }
        let mut out = [0.0f32; 2];
        extract_row(&tensor, 0, 2, 1, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
    }
}
