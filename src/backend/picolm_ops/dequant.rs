//! Dequantization kernels for GGML quantized tensors.
//!
//! Converts raw GGUF tensor bytes into f32 values. Supports all common GGML
//! quantization types used in LLaMA-family models.
//!
//! Design: block-at-a-time dequantization. Never materializes full weight
//! matrices — peak buffer = 256 floats = 1 KB.

use half::f16;

// ── Block sizes ──────────────────────────────────────────────────────────────

/// Number of elements per quantization block.
pub fn block_size(ggml_type: u32) -> usize {
    match ggml_type {
        0 | 1 => 1,     // F32, F16: element-wise
        2 | 3 => 32,    // Q4_0, Q4_1
        6 | 7 => 32,    // Q5_0, Q5_1
        8 => 32,        // Q8_0
        10..=14 => 256, // Q2_K..Q6_K
        _ => 1,         // fallback
    }
}

/// Byte size of one quantization block.
pub fn block_bytes(ggml_type: u32) -> usize {
    match ggml_type {
        0 => 4,    // F32
        1 => 2,    // F16
        2 => 18,   // Q4_0: 2 (scale) + 16 (nibbles)
        3 => 20,   // Q4_1: 2 (scale) + 2 (min) + 16 (nibbles)
        6 => 22,   // Q5_0: 2 (scale) + 4 (high bits) + 16 (nibbles)
        7 => 24,   // Q5_1: 2 (scale) + 2 (min) + 4 (high bits) + 16 (nibbles)
        8 => 34,   // Q8_0: 2 (scale) + 32 (quants)
        10 => 84,  // Q2_K
        11 => 110, // Q3_K
        12 => 144, // Q4_K
        13 => 176, // Q5_K
        14 => 210, // Q6_K
        _ => 4,    // fallback: F32
    }
}

// ── Single-block dequantization ──────────────────────────────────────────────

/// Dequantize one block of raw bytes into f32.
/// `out` must have length >= `block_size(ggml_type)`.
pub fn dequantize_block(block: &[u8], ggml_type: u32, out: &mut [f32]) {
    match ggml_type {
        0 => dequant_f32(block, out),
        1 => dequant_f16(block, out),
        2 => dequant_q4_0(block, out),
        3 => dequant_q4_1(block, out),
        6 => dequant_q5_0(block, out),
        7 => dequant_q5_1(block, out),
        8 => dequant_q8_0(block, out),
        12 => dequant_q4_k(block, out),
        13 => dequant_q5_k(block, out),
        14 => dequant_q6_k(block, out),
        _ => {
            // Unknown type: zero-fill
            for v in out.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

// ── F32 (type 0) ─────────────────────────────────────────────────────────────

fn dequant_f32(block: &[u8], out: &mut [f32]) {
    let bytes: [u8; 4] = [block[0], block[1], block[2], block[3]];
    out[0] = f32::from_le_bytes(bytes);
}

// ── F16 (type 1) ─────────────────────────────────────────────────────────────

fn dequant_f16(block: &[u8], out: &mut [f32]) {
    out[0] = f16::from_le_bytes([block[0], block[1]]).to_f32();
}

// ── Q4_0 (type 2): 32 elements, 18 bytes ────────────────────────────────────
// Layout: [f16 scale (2B)] [16 bytes of 4-bit nibbles]
// value = (nibble - 8) * scale

fn dequant_q4_0(block: &[u8], out: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..16 {
        let byte = block[2 + j];
        out[j] = ((byte & 0x0F) as f32 - 8.0) * scale;
        out[j + 16] = ((byte >> 4) as f32 - 8.0) * scale;
    }
}

// ── Q4_1 (type 3): 32 elements, 20 bytes ────────────────────────────────────
// Layout: [f16 scale (2B)] [f16 min (2B)] [16 bytes of 4-bit nibbles]
// value = nibble * scale + min

fn dequant_q4_1(block: &[u8], out: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let min = f16::from_le_bytes([block[2], block[3]]).to_f32();
    for j in 0..16 {
        let byte = block[4 + j];
        out[j] = (byte & 0x0F) as f32 * scale + min;
        out[j + 16] = (byte >> 4) as f32 * scale + min;
    }
}

// ── Q5_0 (type 6): 32 elements, 22 bytes ────────────────────────────────────
// Layout: [f16 scale (2B)] [4 bytes high bits] [16 bytes lo nibbles]
// value = (nibble | (high_bit << 4) - 16) * scale

fn dequant_q5_0(block: &[u8], out: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for j in 0..16 {
        let byte = block[6 + j];
        let lo = byte & 0x0F;
        let hi_lo = ((qh >> j) & 1) as u8;
        let hi_hi = ((qh >> (j + 16)) & 1) as u8;
        out[j] = ((lo | (hi_lo << 4)) as f32 - 16.0) * scale;
        out[j + 16] = (((byte >> 4) | (hi_hi << 4)) as f32 - 16.0) * scale;
    }
}

// ── Q5_1 (type 7): 32 elements, 24 bytes ────────────────────────────────────
// Layout: [f16 scale (2B)] [f16 min (2B)] [4 bytes high bits] [16 bytes lo nibbles]

fn dequant_q5_1(block: &[u8], out: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let min = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for j in 0..16 {
        let byte = block[8 + j];
        let lo = byte & 0x0F;
        let hi_lo = ((qh >> j) & 1) as u8;
        let hi_hi = ((qh >> (j + 16)) & 1) as u8;
        out[j] = (lo | (hi_lo << 4)) as f32 * scale + min;
        out[j + 16] = ((byte >> 4) | (hi_hi << 4)) as f32 * scale + min;
    }
}

// ── Q8_0 (type 8): 32 elements, 34 bytes ────────────────────────────────────
// Layout: [f16 scale (2B)] [32 × i8 quants]
// value = quant * scale

fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..32 {
        out[j] = (block[2 + j] as i8) as f32 * scale;
    }
}

// ── Q4_K (type 12): 256 elements, 144 bytes ─────────────────────────────────
// Layout: [f16 d (2B)] [f16 dmin (2B)] [12B scales/mins] [128B nibbles]
//
// The 256 elements are produced in 4 chunks of 64. Each chunk uses 32 qs bytes:
//   - first 32 outputs: lo nibbles of qs[0..32], scaled by d*(scale[is+0])
//   - next  32 outputs: hi nibbles of qs[0..32], scaled by d*(scale[is+1])
// is advances by 2 per chunk (0,2,4,6).
//
// get_scale_min_k4(j, scales):
//   j < 4: sc = scales[j] & 63,          mn = scales[j+4] & 63
//   j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//           mn = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales_raw = &block[4..16]; // 12 bytes
    let qs = &block[16..144]; // 128 bytes of nibbles

    // 4 chunks of 64 elements; each chunk consumes 32 qs bytes and 2 scale indices.
    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut out_off = 0usize;
    for _ in 0..4 {
        let (sc1, mn1) = get_scale_min_k4(is, scales_raw);
        let (sc2, mn2) = get_scale_min_k4(is + 1, scales_raw);
        let d1 = d * sc1 as f32;
        let m1 = dmin * mn1 as f32;
        let d2 = d * sc2 as f32;
        let m2 = dmin * mn2 as f32;

        // lo nibbles → first 32 outputs
        for l in 0..32 {
            out[out_off + l] = (qs[q_off + l] & 0x0F) as f32 * d1 - m1;
        }
        // hi nibbles → next 32 outputs
        for l in 0..32 {
            out[out_off + 32 + l] = (qs[q_off + l] >> 4) as f32 * d2 - m2;
        }

        q_off += 32;
        out_off += 64;
        is += 2;
    }
}

// ── Q5_K (type 13): 256 elements, 176 bytes ─────────────────────────────────
// Layout: [f16 d (2B)] [f16 dmin (2B)] [12B scales] [32B qh] [128B qs lo nibbles]
//
// Same 4-chunk structure as Q4_K. Each chunk uses 32 qs bytes and the same 32
// qh bytes (with a rotating 2-bit mask u1/u2 that shifts left by 2 each chunk).
//   lo output: (ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)
//   hi output: (ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)
// u1 starts at 1, u2 starts at 2; both shift left by 2 each chunk.

fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

    let scales_raw = &block[4..16];
    let qh = &block[16..48]; // 32 bytes — shared across all 4 chunks
    let qs = &block[48..176]; // 128 bytes of lo nibbles

    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut out_off = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for _ in 0..4 {
        let (sc1, mn1) = get_scale_min_k4(is, scales_raw);
        let (sc2, mn2) = get_scale_min_k4(is + 1, scales_raw);
        let d1 = d * sc1 as f32;
        let m1 = dmin * mn1 as f32;
        let d2 = d * sc2 as f32;
        let m2 = dmin * mn2 as f32;

        // lo nibbles + high bit from u1 mask
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16.0f32 } else { 0.0 };
            out[out_off + l] = ((qs[q_off + l] & 0x0F) as f32 + hi) * d1 - m1;
        }
        // hi nibbles + high bit from u2 mask
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16.0f32 } else { 0.0 };
            out[out_off + 32 + l] = ((qs[q_off + l] >> 4) as f32 + hi) * d2 - m2;
        }

        q_off += 32;
        out_off += 64;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// ── Q6_K (type 14): 256 elements, 210 bytes ─────────────────────────────────
// Layout: [128B ql (low 4 bits)] [64B qh (high 2 bits)] [16B scales (i8)] [f16 d (2B)]
//
// The 256 elements are split into two halves of 128. Within each half,
// 32 ql bytes produce 4 groups of 32 elements via low/high nibble interleaving.
// The qh bytes provide the upper 2 bits for each element.
// Matches ggml's dequantize_row_q6_K exactly.

fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

    for (half_idx, &out_base) in [0usize, 128usize].iter().enumerate() {
        let ql_base = half_idx * 64; // 0 or 64
        let qh_base = half_idx * 32; // 0 or 32
        let sc_base = half_idx * 8; // 0 or 8

        for l in 0..32 {
            let ql_lo = ql[ql_base + l];
            let ql_hi = ql[ql_base + l + 32];
            let h = qh[qh_base + l];

            // Each group of 16 elements shares one i8 scale
            let sc0 = (sc[sc_base + l / 16] as i8) as f32 * d;
            let sc1 = (sc[sc_base + l / 16 + 2] as i8) as f32 * d;
            let sc2 = (sc[sc_base + l / 16 + 4] as i8) as f32 * d;
            let sc3 = (sc[sc_base + l / 16 + 6] as i8) as f32 * d;

            let q0 = ((ql_lo & 0xF) | (((h >> 0) & 3) << 4)) as i8 - 32;
            let q1 = ((ql_hi & 0xF) | (((h >> 2) & 3) << 4)) as i8 - 32;
            let q2 = ((ql_lo >> 4) | (((h >> 4) & 3) << 4)) as i8 - 32;
            let q3 = ((ql_hi >> 4) | (((h >> 6) & 3) << 4)) as i8 - 32;

            out[out_base + l] = q0 as f32 * sc0;
            out[out_base + l + 32] = q1 as f32 * sc1;
            out[out_base + l + 64] = q2 as f32 * sc2;
            out[out_base + l + 96] = q3 as f32 * sc3;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes() {
        assert_eq!(block_size(0), 1); // F32
        assert_eq!(block_size(1), 1); // F16
        assert_eq!(block_size(2), 32); // Q4_0
        assert_eq!(block_size(8), 32); // Q8_0
        assert_eq!(block_size(12), 256); // Q4_K
    }

    #[test]
    fn test_f32_roundtrip() {
        let val: f32 = 3.14;
        let bytes = val.to_le_bytes();
        let mut out = [0.0f32; 1];
        dequant_f32(&bytes, &mut out);
        assert!((out[0] - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_f16_roundtrip() {
        let val = f16::from_f32(2.5);
        let bytes = val.to_le_bytes();
        let mut out = [0.0f32; 1];
        dequant_f16(&bytes, &mut out);
        assert!((out[0] - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_q8_0_known_values() {
        // scale = 1.0 (f16), quants = [1, -1, 2, -2, ...]
        let scale = f16::from_f32(1.0);
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        block[2] = 1u8; // +1
        block[3] = (-1i8) as u8; // -1
        block[4] = 2u8; // +2
        block[5] = (-2i8) as u8; // -2

        let mut out = [0.0f32; 32];
        dequant_q8_0(&block, &mut out);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - (-1.0)).abs() < 0.01);
        assert!((out[2] - 2.0).abs() < 0.01);
        assert!((out[3] - (-2.0)).abs() < 0.01);
    }

    #[test]
    fn test_q4_0_symmetry() {
        // scale = 1.0, all nibbles = 8 → value should be 0
        let scale = f16::from_f32(1.0);
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        // nibble 8 = 0x88 per byte (lo=8, hi=8)
        for j in 0..16 {
            block[2 + j] = 0x88;
        }
        let mut out = [0.0f32; 32];
        dequant_q4_0(&block, &mut out);
        for v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_q4_0_range() {
        // scale = 2.0
        // byte[2] = 0x00: lo=0 → (0-8)*2=-16, hi=0 → (0-8)*2=-16
        // byte[3] = 0xFF: lo=15 → (15-8)*2=14, hi=15 → (15-8)*2=14
        let scale = f16::from_f32(2.0);
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        block[2] = 0x00;
        block[3] = 0xFF;

        let mut out = [0.0f32; 32];
        dequant_q4_0(&block, &mut out);
        // byte[2] lo → out[0], byte[3] lo → out[1]
        assert!((out[0] - (-16.0)).abs() < 0.1); // nibble 0: (0-8)*2
        assert!((out[1] - 14.0).abs() < 0.1); // nibble 15: (15-8)*2
                                              // byte[2] hi → out[16], byte[3] hi → out[17]
        assert!((out[16] - (-16.0)).abs() < 0.1);
        assert!((out[17] - 14.0).abs() < 0.1);
    }

    #[test]
    fn test_q4_k_element_layout() {
        // Verify that lo nibbles and hi nibbles of the same qs byte get DIFFERENT
        // scales (d1 vs d2), which is the key property of the corrected layout.
        //
        // Set up: d=1.0, dmin=0.0
        // scales_raw: chunk 0 uses is=0,1
        //   get_scale_min_k4(0): sc=scales[0]&63=2, mn=scales[4]&63=0
        //   get_scale_min_k4(1): sc=scales[1]&63=3, mn=scales[5]&63=0
        // So d1=2.0, d2=3.0 for the first 64 elements.
        // qs[0] = 0x21 → lo nibble=1, hi nibble=2
        // out[0]  = 1 * 2.0 = 2.0  (lo, scale d1)
        // out[32] = 2 * 3.0 = 6.0  (hi, scale d2)
        let mut block = [0u8; 144];
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        // dmin = 0 (already zero)
        // scales_raw at block[4..16]: set scales[0]=2, scales[1]=3
        block[4] = 2;
        block[5] = 3;
        // qs at block[16..144]: set qs[0] = 0x21 (lo=1, hi=2)
        block[16] = 0x21;

        let mut out = [0.0f32; 256];
        dequant_q4_k(&block, &mut out);
        assert!(
            (out[0] - 2.0).abs() < 0.01,
            "out[0] expected 2.0, got {}",
            out[0]
        );
        assert!(
            (out[32] - 6.0).abs() < 0.01,
            "out[32] expected 6.0, got {}",
            out[32]
        );
    }

    #[test]
    fn test_q4_k_all_elements_written() {
        let mut block = [0u8; 144];
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        // Give all sub-blocks scale=1, min=0
        for i in 0..4 {
            block[4 + i] = 1;
        } // scales[0..3] = 1
          // qs: all 0x11 so lo=1, hi=1
        for i in 16..144 {
            block[i] = 0x11;
        }

        let mut out = [f32::NAN; 256];
        dequant_q4_k(&block, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "Q4_K element {i} not written (NaN)");
        }
    }

    #[test]
    fn test_q5_k_high_bit() {
        // Verify the u1/u2 rotating mask correctly sets the 5th bit.
        // d=1.0, dmin=0.0, scales[0]=1 (d1=1.0), scales[1]=1 (d2=1.0)
        // chunk 0: u1=1, u2=2
        // qh[0] = 0x03 → bit0=1 (u1=1 matches), bit1=1 (u2=2 matches)
        // qs[0] = 0x00 → lo=0, hi=0
        // out[0]  = (0 + 16) * 1.0 = 16.0  (lo + high bit via u1)
        // out[32] = (0 + 16) * 1.0 = 16.0  (hi + high bit via u2)
        let mut block = [0u8; 176];
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[4] = 1; // scales[0] = 1 → d1=1.0
        block[5] = 1; // scales[1] = 1 → d2=1.0
        block[16] = 0x03; // qh[0]: bit0=1, bit1=1
                          // qs[0] = 0x00 (already zero)

        let mut out = [0.0f32; 256];
        dequant_q5_k(&block, &mut out);
        assert!(
            (out[0] - 16.0).abs() < 0.01,
            "out[0] expected 16.0, got {}",
            out[0]
        );
        assert!(
            (out[32] - 16.0).abs() < 0.01,
            "out[32] expected 16.0, got {}",
            out[32]
        );
    }

    #[test]
    fn test_q5_k_chunk1_mask() {
        // chunk 1: u1=4, u2=8 — verify the mask shifts correctly
        // d=1.0, dmin=0.0
        // scales for chunk1: is=2,3 → get_scale_min_k4(2)=(scales[2]&63, scales[6]&63)
        // set scales[2]=1, scales[3]=1 → d1=d2=1.0
        // qh[0] = 0x0C → bit2=1 (u1=4 matches), bit3=1 (u2=8 matches)
        // qs[32] = 0x00 (chunk1 starts at q_off=32)
        // out[64]  = (0 + 16) * 1.0 = 16.0
        // out[96]  = (0 + 16) * 1.0 = 16.0
        let mut block = [0u8; 176];
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[6] = 1; // scales[2] = 1
        block[7] = 1; // scales[3] = 1
        block[16] = 0x0C; // qh[0]: bit2=1, bit3=1

        let mut out = [0.0f32; 256];
        dequant_q5_k(&block, &mut out);
        assert!(
            (out[64] - 16.0).abs() < 0.01,
            "out[64] expected 16.0, got {}",
            out[64]
        );
        assert!(
            (out[96] - 16.0).abs() < 0.01,
            "out[96] expected 16.0, got {}",
            out[96]
        );
    }

    #[test]
    fn test_q6_k_zero_quants() {
        // All ql=0, qh=0, scales=1 (i8), d=1.0
        // q = (0 | (0 << 4)) as i8 - 32 = -32
        // value = -32 * 1.0 * 1.0 = -32.0
        let mut block = [0u8; 210];
        // scales: all 1 (as i8)
        for i in 192..208 {
            block[i] = 1;
        }
        // d = 1.0 as f16
        let d = f16::from_f32(1.0);
        block[208..210].copy_from_slice(&d.to_le_bytes());

        let mut out = [0.0f32; 256];
        dequant_q6_k(&block, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - (-32.0)).abs() < 0.1,
                "Q6_K element {i}: expected -32.0, got {v}"
            );
        }
    }

    #[test]
    fn test_q6_k_all_elements_written() {
        // Verify all 256 elements are written (no gaps)
        let mut block = [0u8; 210];
        // Set d=1.0, scales=1, and put distinct patterns in ql
        let d = f16::from_f32(1.0);
        block[208..210].copy_from_slice(&d.to_le_bytes());
        for i in 192..208 {
            block[i] = 1;
        }
        // Fill ql with 0x11 so lo=1, hi=1 → different from zero
        for i in 0..128 {
            block[i] = 0x11;
        }

        let mut out = [f32::NAN; 256];
        dequant_q6_k(&block, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Q6_K element {i} was not written (still NaN)"
            );
        }
    }
}
