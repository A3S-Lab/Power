#![cfg(feature = "picolm")]

use std::path::PathBuf;

use a3s_power::backend::gguf_stream::GgufFile;
use a3s_power::backend::picolm_ops::qwen35::{
    Qwen35LayerState, Qwen35ModelState, Qwen35MropeConfig, Qwen35TensorLayout,
};

/// Opt-in artifact test. The model is intentionally supplied by environment
/// because even the smallest published Qwen3.5 GGUF is hundreds of megabytes.
#[test]
fn validates_real_qwen35_gguf_layout() {
    let Some(path) = std::env::var_os("A3S_POWER_QWEN35_TEST_MODEL").map(PathBuf::from) else {
        eprintln!("A3S_POWER_QWEN35_TEST_MODEL is not set; skipping real Qwen3.5 artifact test");
        return;
    };

    let model = GgufFile::open(&path).expect("real Qwen3.5 GGUF must parse with bounded tensors");
    assert_eq!(model.meta.architecture.name(), "qwen35");

    let layout = Qwen35TensorLayout::validate(&model.meta)
        .expect("real Qwen3.5 GGUF must match the typed native layout");
    assert_eq!(layout.trunk_layers, 24);
    assert_eq!(layout.recurrent_layers, 18);
    assert_eq!(layout.full_attention_layers, 6);
    assert_eq!(layout.mtp_layers, 0);
    assert!(layout.tied_output);

    let state = Qwen35ModelState::from_metadata(&model.meta, 32)
        .expect("real Qwen3.5 metadata must construct typed hybrid state");
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
    assert_eq!(recurrent, 18);
    assert_eq!(full_attention, 6);

    let mrope = Qwen35MropeConfig::from_metadata(&model.meta)
        .expect("real Qwen3.5 metadata must construct IMRoPE configuration");
    assert_eq!(mrope.sections(), [11, 11, 10, 0]);
    assert_eq!(mrope.rope_dim(), 64);
}
