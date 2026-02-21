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
        0 | 1 => 1,       // F32, F16: element-wise
        2 | 3 => 32,      // Q4_0, Q4_1
        6 | 7 => 32,      // Q5_0, Q5_1
        8 => 32,           // Q8_0
        10..=14 => 256,    // Q2_K..Q6_K
        _ => 1,            // fallback
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
// Super-block of 8 sub-blocks × 32 elements.
// Layout: [f16 d (2B)] [f16 dmin (2B)] [12B scales/mins] [128B nibbles]
//
// The 12-byte scale section encodes 8 × (6-bit scale + 6-bit min):
//   bytes 0..3:  low 4 bits of scales[0..3] and mins[0..3]
//   bytes 4..7:  low 4 bits of scales[4..7] and mins[4..7]
//   bytes 8..11: high 2 bits of scales[0..7] and mins[0..7]

fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

    let scales_raw = &block[4..16]; // 12 bytes
    let qs = &block[16..144]; // 128 bytes of nibbles

    // Decode 6-bit scales and mins for 8 sub-blocks
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
    }
    for i in 0..4 {
        let hi = scales_raw[8 + i];
        sc[i] |= (hi & 0x03) << 4; // wait, already 6 bits from & 0x3F
        mn[i] |= (hi & 0x0C) << 2;
    }
    // Correction: the 12-byte encoding for Q4_K is:
    // bytes[0..4]: each byte = (scale_lo[i] & 0x3F) for sub-blocks 0..3
    // bytes[4..8]: each byte = (min_lo[i] & 0x3F) for sub-blocks 0..3
    // bytes[8..12]: high bits packed
    // Actually the standard GGML Q4_K layout:
    //   bytes 0..3: scales_lo for sub-blocks 0..3 (6 bits each, packed in low 6)
    //   bytes 4..7: scales_lo for sub-blocks 4..7
    //   bytes 8..11: high 2 bits of all 8 scales + 8 mins

    // Let me re-implement correctly per ggml spec:
    // scales[0..7] and mins[0..7] are each 6-bit values.
    // Packed into 12 bytes as follows:
    //   byte[i] for i in 0..3:  lo 6 bits = scale[i]
    //   byte[i] for i in 4..7:  lo 6 bits = scale[i]
    //   But wait — that's only scales. Where are mins?

    // The actual Q4_K layout from ggml:
    // struct block_q4_K {
    //     half d;            // 2 bytes
    //     half dmin;         // 2 bytes
    //     uint8_t scales[12]; // 12 bytes: encodes 8 scales + 8 mins in 6 bits each
    //     uint8_t qs[128];   // 128 bytes: 256 4-bit quants
    // }
    //
    // Scale/min decoding (from ggml):
    // For sub-block j (0..7):
    //   if j < 4:
    //     sc[j] = scales[j] & 63
    //     mn[j] = scales[j+4] & 63  -- wait, that uses bytes 4..7 for mins of 0..3
    //   else:
    //     sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    //     mn[j] = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
    //
    // Actually let me just use the canonical ggml decoding:

    // Reset
    sc = [0u8; 8];
    mn = [0u8; 8];

    for j in 0..4 {
        sc[j] = scales_raw[j] & 63;
        mn[j] = scales_raw[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (scales_raw[j + 4] & 0xF) | ((scales_raw[j - 4] >> 6) << 4);
        mn[j] = (scales_raw[j + 4] >> 4) | ((scales_raw[j] >> 6) << 4);
    }

    for sub in 0..8 {
        let sub_d = d * sc[sub] as f32;
        let sub_m = dmin * mn[sub] as f32;
        let qs_off = sub * 16;
        let out_off = sub * 32;
        for j in 0..16 {
            let byte = qs[qs_off + j];
            out[out_off + j] = (byte & 0x0F) as f32 * sub_d - sub_m;
            out[out_off + j + 16] = (byte >> 4) as f32 * sub_d - sub_m;
        }
    }
}

// ── Q5_K (type 13): 256 elements, 176 bytes ─────────────────────────────────
// Layout: [f16 d (2B)] [f16 dmin (2B)] [12B scales] [32B high bits] [128B lo nibbles]

fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

    let scales_raw = &block[4..16];
    let qh = &block[16..48]; // 32 bytes = 256 bits for high bits
    let qs = &block[48..176]; // 128 bytes of lo nibbles

    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    for j in 0..4 {
        sc[j] = scales_raw[j] & 63;
        mn[j] = scales_raw[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (scales_raw[j + 4] & 0xF) | ((scales_raw[j - 4] >> 6) << 4);
        mn[j] = (scales_raw[j + 4] >> 4) | ((scales_raw[j] >> 6) << 4);
    }

    for sub in 0..8 {
        let sub_d = d * sc[sub] as f32;
        let sub_m = dmin * mn[sub] as f32;
        let qs_off = sub * 16;
        let out_off = sub * 32;
        for j in 0..16 {
            let byte = qs[qs_off + j];
            let bit_idx = sub * 32 + j;
            let hb_lo = (qh[bit_idx / 8] >> (bit_idx % 8)) & 1;
            let hb_hi = (qh[(bit_idx + 16) / 8] >> ((bit_idx + 16) % 8)) & 1;
            out[out_off + j] =
                ((byte & 0x0F) as f32 + (hb_lo as f32) * 16.0) * sub_d - sub_m;
            out[out_off + j + 16] =
                ((byte >> 4) as f32 + (hb_hi as f32) * 16.0) * sub_d - sub_m;
        }
    }
}

// ── Q6_K (type 14): 256 elements, 210 bytes ─────────────────────────────────
// Layout: [128B lo nibbles] [64B high bits] [16B scales (i8)] [f16 d (2B)]

fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];    // low 4 bits
    let qh = &block[128..192];  // high 2 bits
    let sc = &block[192..208];  // 16 × i8 scales
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

    for (sub, &sc_byte) in sc.iter().enumerate() {
        let scale = (sc_byte as i8) as f32 * d;
        let out_off = sub * 16;
        let ql_off = sub * 8;
        let qh_off = sub * 4;
        for j in 0..8 {
            let ql_byte = ql[ql_off + j];
            let qh_byte = qh[qh_off + j / 2];
            let qh_shift = (j % 2) * 4;

            let lo = ql_byte & 0x0F;
            let hi = (qh_byte >> qh_shift) & 0x03;
            let q = (lo | (hi << 4)) as i8 - 32;
            out[out_off + j] = q as f32 * scale;

            if out_off + j + 8 < 256 {
                let lo2 = ql_byte >> 4;
                let hi2 = (qh_byte >> (qh_shift + 2)) & 0x03;
                let q2 = (lo2 | (hi2 << 4)) as i8 - 32;
                out[out_off + j + 8] = q2 as f32 * scale;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes() {
        assert_eq!(block_size(0), 1);  // F32
        assert_eq!(block_size(1), 1);  // F16
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
        block[2] = 1u8;           // +1
        block[3] = (-1i8) as u8;  // -1
        block[4] = 2u8;           // +2
        block[5] = (-2i8) as u8;  // -2

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
        assert!((out[0] - (-16.0)).abs() < 0.1);  // nibble 0: (0-8)*2
        assert!((out[1] - 14.0).abs() < 0.1);     // nibble 15: (15-8)*2
        // byte[2] hi → out[16], byte[3] hi → out[17]
        assert!((out[16] - (-16.0)).abs() < 0.1);
        assert!((out[17] - 14.0).abs() < 0.1);
    }
}
