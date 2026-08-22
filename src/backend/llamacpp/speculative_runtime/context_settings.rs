use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};

use super::super::nonzero_context_size;
use crate::speculative::minimum_mtp_batch;

#[derive(Debug, Clone, Copy)]
pub(in crate::backend::llamacpp) struct LlamaContextSettings {
    pub(in crate::backend::llamacpp) ctx_size: u32,
    pub(in crate::backend::llamacpp) num_batch: Option<u32>,
    pub(in crate::backend::llamacpp) num_thread: Option<u32>,
    pub(in crate::backend::llamacpp) num_thread_batch: Option<u32>,
    pub(in crate::backend::llamacpp) flash_attention: bool,
    pub(in crate::backend::llamacpp) mtp_fr_vocab_size: Option<u32>,
}

impl LlamaContextSettings {
    pub(in crate::backend::llamacpp) fn params(
        self,
        context_type: LlamaContextType,
        recurrent_snapshots: u32,
        minimum_batch: u32,
        output_rows_per_sequence: u32,
    ) -> LlamaContextParams {
        let mtp_fr_vocab = if matches!(context_type, LlamaContextType::Mtp) {
            self.mtp_fr_vocab_size.unwrap_or(0)
        } else {
            0
        };
        let mut params = LlamaContextParams::default()
            .with_n_ctx(Some(nonzero_context_size(self.ctx_size)))
            .with_context_type(context_type)
            .with_n_rs_seq(recurrent_snapshots);
        if let Some(batch) = self.num_batch {
            params = params.with_n_batch(batch.max(minimum_batch));
        } else if minimum_batch > params.n_batch() {
            params = params.with_n_batch(minimum_batch);
        }
        if let Some(threads) = self.num_thread {
            params = params.with_n_threads(threads as i32);
        }
        if let Some(threads_batch) = self.num_thread_batch {
            params = params.with_n_threads_batch(threads_batch as i32);
        }
        if self.flash_attention {
            params = params.with_flash_attention_policy(1);
        }
        let output_rows = output_rows_per_sequence.max(1).min(params.n_batch());
        with_llamacpp_context_extensions(params, output_rows, output_rows, mtp_fr_vocab)
    }
}

// The reviewed llama-cpp-2 revision wraps `llama_context_params` in a
// single-field Rust type, but does not yet expose Power's speculative context
// extensions. Keep this bridge next to context construction and fail
// compilation if that pinned representation changes. The direct sys package
// is pinned to the exact same git revision in Cargo.toml.
const _: () = {
    assert!(
        std::mem::size_of::<LlamaContextParams>()
            == std::mem::size_of::<llama_cpp_sys_2::llama_context_params>()
    );
    assert!(
        std::mem::align_of::<LlamaContextParams>()
            == std::mem::align_of::<llama_cpp_sys_2::llama_context_params>()
    );
};

fn with_llamacpp_context_extensions(
    mut params: LlamaContextParams,
    total: u32,
    per_sequence: u32,
    mtp_fr_vocab: u32,
) -> LlamaContextParams {
    // SAFETY: at the pinned llama-cpp-2 revision, `LlamaContextParams` contains
    // exactly one `llama_context_params` field. Equal size and alignment are
    // asserted above, so that sole non-zero-sized field starts at offset zero.
    let raw = unsafe {
        &mut *std::ptr::from_mut(&mut params).cast::<llama_cpp_sys_2::llama_context_params>()
    };
    raw.n_outputs_max = total;
    raw.n_outputs_max_per_seq = per_sequence;
    #[cfg(feature = "llamacpp-mtp-fr")]
    {
        raw.n_mtp_fr_vocab = mtp_fr_vocab;
    }
    #[cfg(not(feature = "llamacpp-mtp-fr"))]
    {
        debug_assert_eq!(mtp_fr_vocab, 0);
    }
    params
}

#[cfg(test)]
pub(super) fn llamacpp_context_output_limits(params: &LlamaContextParams) -> (u32, u32) {
    // SAFETY: same pinned single-field representation as the setter above.
    let raw =
        unsafe { &*std::ptr::from_ref(params).cast::<llama_cpp_sys_2::llama_context_params>() };
    (raw.n_outputs_max, raw.n_outputs_max_per_seq)
}

pub(super) fn external_target_context_params(
    context_settings: LlamaContextSettings,
    draft_max: u32,
    recurrent_snapshots: u32,
) -> LlamaContextParams {
    context_settings.params(
        LlamaContextType::Default,
        recurrent_snapshots.min(draft_max),
        minimum_mtp_batch(draft_max),
        draft_max.saturating_add(1),
    )
}

#[cfg(all(test, feature = "llamacpp-mtp-fr"))]
pub(super) fn llamacpp_context_mtp_fr_vocab(params: &LlamaContextParams) -> u32 {
    // SAFETY: same pinned single-field representation as the setter above.
    let raw =
        unsafe { &*std::ptr::from_ref(params).cast::<llama_cpp_sys_2::llama_context_params>() };
    raw.n_mtp_fr_vocab
}

#[derive(Debug, Clone)]
pub(in crate::backend::llamacpp) struct MtpCompletionSettings {
    pub(in crate::backend::llamacpp) max_tokens: usize,
    pub(in crate::backend::llamacpp) stop_sequences: Vec<String>,
    pub(in crate::backend::llamacpp) draft_max: u32,
    pub(in crate::backend::llamacpp) recurrent_snapshots: u32,
    pub(in crate::backend::llamacpp) recurrent_chain: bool,
    pub(in crate::backend::llamacpp) adaptive: bool,
    pub(in crate::backend::llamacpp) draft_min: u32,
    pub(in crate::backend::llamacpp) draft_p_min: f32,
}
