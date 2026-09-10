//! Env-gated live evidence for llama.cpp opaque state + decode after restore.
//!
//! Skips unless `A3S_POWER_LLAMACPP_PHASE_STATE_MODEL` points at a loadable
//! GGUF. Does not run in default CI. When present, proves
//! `LlamaCppContextStateApi` capture → restore round-trip on a real context.
//! Ready decode still requires binding a live completion hook
//! (`LlamaCppLiveDecodeTokenPort`); this test only seals state API evidence.

#![cfg(feature = "llamacpp")]

use std::num::NonZeroU32;
use std::path::PathBuf;

use a3s_power::serving::{
    LlamaCppBackendPhaseStateOwnership, LlamaCppContextStateApi, LlamaCppLayoutFacts,
};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

fn model_path() -> Option<PathBuf> {
    std::env::var_os("A3S_POWER_LLAMACPP_PHASE_STATE_MODEL").map(PathBuf::from)
}

#[test]
fn live_llamacpp_context_state_api_capture_restore_round_trip() {
    let Some(path) = model_path() else {
        eprintln!(
            "A3S_POWER_LLAMACPP_PHASE_STATE_MODEL is not set; skipping live llama.cpp phase-state test"
        );
        return;
    };
    if !path.is_file() {
        panic!(
            "A3S_POWER_LLAMACPP_PHASE_STATE_MODEL does not point at a file: {}",
            path.display()
        );
    }

    let backend = LlamaBackend::init().expect("llama backend");
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &path, &model_params)
        .expect("load GGUF for live phase-state evidence");
    let n_ctx = NonZeroU32::new(64).expect("n_ctx");
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    let mut context = model
        .new_context(&backend, ctx_params)
        .expect("new context");

    let facts = LlamaCppLayoutFacts::from_model(&model, n_ctx.get());
    let ownership = LlamaCppBackendPhaseStateOwnership::from_layout_facts(
        facts,
        Some("1".repeat(64)),
        Some("2".repeat(64)),
        Some("3".repeat(64)),
    );

    let handle = {
        let port = LlamaCppContextStateApi::new(&mut context);
        ownership
            .capture_from_port(&port)
            .expect("capture live opaque state")
    };
    let exported = ownership
        .export_opaque_state(&handle)
        .expect("export live opaque state");
    assert!(
        !exported.is_empty(),
        "live llama.cpp state snapshot must be non-empty"
    );

    {
        let mut port = LlamaCppContextStateApi::new(&mut context);
        ownership
            .restore_into_port(&handle, &mut port)
            .expect("restore live opaque state");
    }
}
