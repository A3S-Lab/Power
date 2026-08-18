#![cfg(feature = "picolm")]

use std::path::PathBuf;

use a3s_power::backend::gguf_stream::GgufFile;
use a3s_power::backend::picolm_ops::qwen35::{
    Qwen35LayerState, Qwen35ModelState, Qwen35MropeConfig, Qwen35TensorLayout,
};

/// Opt-in artifact test for a selected Qwen3.8-27B GGUF conversion.
///
/// Qwen3.8 intentionally retains the `qwen35` GGUF architecture identifier,
/// so product-version validation lives separately from the reusable executor.
#[test]
fn validates_real_qwen38_27b_layout() {
    let Some(path) = std::env::var_os("A3S_POWER_QWEN38_TEST_MODEL").map(PathBuf::from) else {
        eprintln!("A3S_POWER_QWEN38_TEST_MODEL is not set; skipping real Qwen3.8 artifact test");
        return;
    };

    let model = GgufFile::open(&path).expect("real Qwen3.8 GGUF must parse with bounded tensors");
    assert_eq!(model.meta.architecture.name(), "qwen35");
    assert_eq!(model.meta.n_embd, 5120);
    assert_eq!(model.meta.n_heads, 24);
    assert_eq!(model.meta.n_kv_heads, 4);
    assert_eq!(model.meta.n_ff, 17_408);
    assert_eq!(model.meta.context_length, 262_144);
    assert_eq!(model.meta.vocab_size, 248_320);

    let layout = Qwen35TensorLayout::validate(&model.meta)
        .expect("real Qwen3.8 GGUF must match the typed qwen35 layout");
    assert_eq!(layout.trunk_layers, 64);
    assert_eq!(layout.recurrent_layers, 48);
    assert_eq!(layout.full_attention_layers, 16);
    assert_eq!(layout.mtp_layers, 1);
    assert!(!layout.tied_output);

    let state = Qwen35ModelState::from_metadata(&model.meta, 32)
        .expect("real Qwen3.8 metadata must construct typed hybrid state");
    let recurrent = (0..state.len())
        .filter(|layer| matches!(state.layer(*layer), Some(Qwen35LayerState::Recurrent(_))))
        .count();
    let full_attention = (0..state.len())
        .filter(|layer| {
            matches!(
                state.layer(*layer),
                Some(Qwen35LayerState::FullAttention(_))
            )
        })
        .count();
    let mtp = (0..state.len())
        .filter(|layer| matches!(state.layer(*layer), Some(Qwen35LayerState::Mtp(_))))
        .count();
    assert_eq!(recurrent, 48);
    assert_eq!(full_attention, 16);
    assert_eq!(mtp, 1);

    let mrope = Qwen35MropeConfig::from_metadata(&model.meta)
        .expect("real Qwen3.8 metadata must construct IMRoPE configuration");
    assert_eq!(mrope.sections(), [11, 11, 10, 0]);
    assert_eq!(mrope.rope_dim(), 64);
}
