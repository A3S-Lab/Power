//! llama.cpp adapter for Power's model-neutral speculative runtime.
//!
//! This module contains backend mechanics only. Strategy selection, capability
//! negotiation, block verification, and metrics remain in `crate::speculative`.

use llama_cpp_2::context::{
    params::{LlamaContextParams, LlamaContextType},
    LlamaContext,
};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};

use super::{backend_ref, nonzero_context_size, send_completion_result};
use crate::backend::types::CompletionResponseChunk;
use crate::error::{PowerError, Result};
use crate::speculative::{
    bounded_draft_len, minimum_mtp_batch, verify_token_block_until, SpeculativeCapabilities,
    SpeculativeMetrics, SpeculativeStrategy,
};

fn metadata_entry_enables_mtp(key: &str, value: &str) -> bool {
    key.ends_with(".nextn_predict_layers")
        && value.trim().parse::<u32>().is_ok_and(|layers| layers > 0)
}

pub(super) fn ensure_mtp_fr_available(vocab_size: Option<u32>) -> Result<()> {
    #[cfg(not(feature = "llamacpp-mtp-fr"))]
    if vocab_size.is_some() {
        return Err(PowerError::Config(
            "spec_mtp_fr_vocab_size requires the experimental \
             llamacpp-mtp-fr crate feature and the reviewed llama.cpp patch"
                .to_string(),
        ));
    }

    let _ = vocab_size;
    Ok(())
}

pub(super) fn llamacpp_speculative_capabilities(model: &LlamaModel) -> SpeculativeCapabilities {
    let mut capabilities = SpeculativeCapabilities::none();
    let has_mtp = (0..model.meta_count()).any(|index| {
        let Ok(key) = model.meta_key_by_index(index) else {
            return false;
        };
        if !key.ends_with(".nextn_predict_layers") {
            return false;
        }
        model
            .meta_val_str_by_index(index)
            .is_ok_and(|value| metadata_entry_enables_mtp(&key, &value))
    });
    if has_mtp {
        capabilities = capabilities.with(SpeculativeStrategy::Mtp);
    }
    capabilities
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LlamaContextSettings {
    pub(super) ctx_size: u32,
    pub(super) num_batch: Option<u32>,
    pub(super) num_thread: Option<u32>,
    pub(super) num_thread_batch: Option<u32>,
    pub(super) flash_attention: bool,
    pub(super) mtp_fr_vocab_size: Option<u32>,
}

impl LlamaContextSettings {
    pub(super) fn params(
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
fn llamacpp_context_output_limits(params: &LlamaContextParams) -> (u32, u32) {
    // SAFETY: same pinned single-field representation as the setter above.
    let raw =
        unsafe { &*std::ptr::from_ref(params).cast::<llama_cpp_sys_2::llama_context_params>() };
    (raw.n_outputs_max, raw.n_outputs_max_per_seq)
}

#[cfg(all(test, feature = "llamacpp-mtp-fr"))]
fn llamacpp_context_mtp_fr_vocab(params: &LlamaContextParams) -> u32 {
    // SAFETY: same pinned single-field representation as the setter above.
    let raw =
        unsafe { &*std::ptr::from_ref(params).cast::<llama_cpp_sys_2::llama_context_params>() };
    raw.n_mtp_fr_vocab
}

#[derive(Debug, Clone)]
pub(super) struct LlamaSamplingSettings {
    pub(super) response_format: Option<serde_json::Value>,
    pub(super) repeat_penalty: Option<f32>,
    pub(super) frequency_penalty: Option<f32>,
    pub(super) presence_penalty: Option<f32>,
    pub(super) repeat_last_n: i32,
    pub(super) mirostat: Option<u32>,
    pub(super) mirostat_tau: Option<f32>,
    pub(super) mirostat_eta: Option<f32>,
    pub(super) temperature: f32,
    pub(super) top_k: Option<i32>,
    pub(super) typical_p: Option<f32>,
    pub(super) top_p: f32,
    pub(super) min_p: Option<f32>,
    pub(super) seed: u32,
}

fn use_greedy_fast_path(settings: &LlamaSamplingSettings) -> bool {
    settings.mirostat.is_none()
        && settings.temperature <= 0.0
        && settings.top_k.is_none()
        && settings.typical_p.is_none()
        && settings.top_p >= 1.0
        && settings.min_p.is_none()
}

pub(super) fn use_backend_greedy(settings: &LlamaSamplingSettings) -> bool {
    settings.response_format.is_none()
        && settings.repeat_penalty.is_none()
        && settings.frequency_penalty.is_none()
        && settings.presence_penalty.is_none()
        && use_greedy_fast_path(settings)
}

pub(super) fn sample_target_token(
    sampler: &mut LlamaSampler,
    context: &LlamaContext<'_>,
    index: i32,
    backend_greedy: bool,
) -> llama_cpp_2::token::LlamaToken {
    // `llama_sampler_sample` asks llama.cpp for the sampled token, sampled
    // probabilities, sampled logits, and candidate ids before it notices the
    // CUDA graph already selected a token. Stateless greedy sampling needs no
    // CPU-side accept step, so read that result directly. Retain the generic
    // sampler as a defensive fallback if backend sampling was unavailable for
    // a particular output row.
    if backend_greedy {
        if let Some(token) = context.sampled_token_ith(index) {
            return token;
        }
    }
    sampler.sample(context, index)
}

pub(super) fn build_llamacpp_sampler(
    model: &LlamaModel,
    settings: &LlamaSamplingSettings,
) -> Result<LlamaSampler> {
    let mut samplers = Vec::new();
    if let Some(ref format) = settings.response_format {
        match super::super::json_schema::format_to_gbnf(format) {
            Ok(Some(grammar)) => samplers.push(
                LlamaSampler::grammar(model, &grammar, "root").map_err(|error| {
                    PowerError::InferenceFailed(format!(
                        "Failed to create grammar sampler: {error}"
                    ))
                })?,
            ),
            Ok(None) => {}
            Err(error) => {
                return Err(PowerError::InvalidRequest(format!(
                    "unsupported response_format grammar: {error}"
                )));
            }
        }
    }

    if settings.repeat_penalty.is_some()
        || settings.frequency_penalty.is_some()
        || settings.presence_penalty.is_some()
    {
        samplers.push(LlamaSampler::penalties(
            model.n_vocab(),
            settings.repeat_last_n,
            settings.repeat_penalty.unwrap_or(1.0),
            settings.frequency_penalty.unwrap_or(0.0),
            settings.presence_penalty.unwrap_or(0.0),
        ));
    }

    match settings.mirostat {
        Some(1) => {
            samplers.push(LlamaSampler::temp(settings.temperature));
            samplers.push(LlamaSampler::mirostat(
                model.n_vocab(),
                settings.seed,
                settings.mirostat_tau.unwrap_or(5.0),
                settings.mirostat_eta.unwrap_or(0.1),
                100,
            ));
        }
        Some(2) => {
            samplers.push(LlamaSampler::temp(settings.temperature));
            samplers.push(LlamaSampler::mirostat_v2(
                settings.seed,
                settings.mirostat_tau.unwrap_or(5.0),
                settings.mirostat_eta.unwrap_or(0.1),
            ));
        }
        _ => {
            if use_greedy_fast_path(settings) {
                samplers.push(LlamaSampler::greedy());
                return Ok(LlamaSampler::chain_simple(samplers));
            }
            if let Some(top_k) = settings.top_k {
                samplers.push(LlamaSampler::top_k(top_k));
            }
            if let Some(typical_p) = settings.typical_p {
                samplers.push(LlamaSampler::typical(typical_p, 1));
            }
            samplers.push(LlamaSampler::top_p(settings.top_p, 1));
            if let Some(min_p) = settings.min_p {
                samplers.push(LlamaSampler::min_p(min_p, 1));
            }
            samplers.push(LlamaSampler::temp(settings.temperature));
            samplers.push(LlamaSampler::dist(settings.seed));
        }
    }

    Ok(LlamaSampler::chain_simple(samplers))
}

#[derive(Debug, Clone)]
pub(super) struct MtpCompletionSettings {
    pub(super) max_tokens: usize,
    pub(super) stop_sequences: Vec<String>,
    pub(super) draft_max: u32,
    pub(super) recurrent_snapshots: u32,
    pub(super) draft_min: u32,
    pub(super) draft_p_min: f32,
}

fn mtp_speculative_params(settings: &MtpCompletionSettings) -> Result<MtpSpeculativeParams> {
    if settings.draft_max == 0 || settings.draft_max > 64 {
        return Err(PowerError::Config(format!(
            "spec_draft_max must be between 1 and 64 for llama.cpp MTP, got {}",
            settings.draft_max
        )));
    }
    if settings.draft_min > settings.draft_max {
        return Err(PowerError::Config(format!(
            "spec_draft_min ({}) must not exceed llama.cpp MTP spec_draft_max ({})",
            settings.draft_min, settings.draft_max
        )));
    }
    if settings.recurrent_snapshots == 0 || settings.recurrent_snapshots > 64 {
        return Err(PowerError::Config(format!(
            "spec_mtp_recurrent_snapshots must be between 1 and 64 for llama.cpp MTP, got {}",
            settings.recurrent_snapshots
        )));
    }
    if !settings.draft_p_min.is_finite() || !(0.0..=1.0).contains(&settings.draft_p_min) {
        return Err(PowerError::Config(format!(
            "spec_draft_p_min must be finite and between 0 and 1 for llama.cpp MTP, got {}",
            settings.draft_p_min
        )));
    }

    Ok(MtpSpeculativeParams {
        n_max: settings.draft_max as i32,
        n_min: settings.draft_min as i32,
        p_min: settings.draft_p_min,
    })
}

#[derive(Debug, Clone, Copy)]
struct MtpPhaseTimings {
    draft_ns: u64,
    target_decode_ns: u64,
    accepted_prefix_sync_ns: u64,
    sampling_ns: u64,
    state_management_ns: u64,
    streaming_ns: u64,
    fallback_replays: u32,
    max_rejected_suffix: usize,
    accepted_prefix_histogram: [u32; 65],
}

impl Default for MtpPhaseTimings {
    fn default() -> Self {
        Self {
            draft_ns: 0,
            target_decode_ns: 0,
            accepted_prefix_sync_ns: 0,
            sampling_ns: 0,
            state_management_ns: 0,
            streaming_ns: 0,
            fallback_replays: 0,
            max_rejected_suffix: 0,
            accepted_prefix_histogram: [0; 65],
        }
    }
}

impl MtpPhaseTimings {
    fn instrumented_ns(self) -> u64 {
        self.draft_ns
            .saturating_add(self.target_decode_ns)
            .saturating_add(self.accepted_prefix_sync_ns)
            .saturating_add(self.sampling_ns)
            .saturating_add(self.state_management_ns)
            .saturating_add(self.streaming_ns)
    }
}

fn elapsed_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn log_metrics(
    metrics: SpeculativeMetrics,
    timings: MtpPhaseTimings,
    emitted_tokens: usize,
    decode_started: std::time::Instant,
) {
    let elapsed_duration_ns = elapsed_ns(decode_started);
    let elapsed = elapsed_duration_ns as f64 / 1_000_000_000.0;
    let tokens_per_second = if elapsed > 0.0 {
        emitted_tokens as f64 / elapsed
    } else {
        0.0
    };
    tracing::info!(
        strategy = "mtp",
        rounds = metrics.rounds,
        drafted_tokens = metrics.drafted_tokens,
        accepted_tokens = metrics.accepted_tokens,
        emitted_tokens,
        verified_emitted_tokens = metrics.emitted_tokens,
        acceptance_rate = metrics.acceptance_rate(),
        tokens_per_target_pass = metrics.tokens_per_target_pass(),
        tokens_per_second,
        draft_duration_ns = timings.draft_ns,
        target_decode_duration_ns = timings.target_decode_ns,
        accepted_prefix_sync_duration_ns = timings.accepted_prefix_sync_ns,
        sampling_duration_ns = timings.sampling_ns,
        state_management_duration_ns = timings.state_management_ns,
        streaming_duration_ns = timings.streaming_ns,
        runtime_overhead_ns = elapsed_duration_ns.saturating_sub(timings.instrumented_ns()),
        fallback_replays = timings.fallback_replays,
        max_rejected_suffix = timings.max_rejected_suffix,
        accepted_prefix_histogram = ?timings.accepted_prefix_histogram,
        "llama.cpp speculative completion finished"
    );
}

#[allow(clippy::too_many_arguments)]
fn stream_token(
    model: &LlamaModel,
    token: llama_cpp_2::token::LlamaToken,
    eos_token: llama_cpp_2::token::LlamaToken,
    generated_text: &mut String,
    generated_count: &mut usize,
    stop_sequences: &[String],
    prompt_token_count: u32,
    prompt_eval_duration_ns: u64,
    tx: &tokio::sync::mpsc::Sender<Result<CompletionResponseChunk>>,
) -> bool {
    if token == eos_token {
        send_completion_result(
            tx,
            Ok(CompletionResponseChunk {
                text: String::new(),
                done: true,
                prompt_tokens: Some(prompt_token_count),
                done_reason: Some("stop".to_string()),
                prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                token_id: None,
            }),
        );
        return false;
    }

    let text = token_piece(model, token);
    generated_text.push_str(&text);
    *generated_count += 1;
    let should_stop = stop_sequences
        .iter()
        .any(|stop| generated_text.ends_with(stop));
    send_completion_result(
        tx,
        Ok(CompletionResponseChunk {
            text,
            done: should_stop,
            prompt_tokens: should_stop.then_some(prompt_token_count),
            done_reason: should_stop.then(|| "stop".to_string()),
            prompt_eval_duration_ns: should_stop.then_some(prompt_eval_duration_ns),
            token_id: Some(token.0 as u32),
        }),
    ) && !should_stop
}

fn token_piece(model: &LlamaModel, token: llama_cpp_2::token::LlamaToken) -> String {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .unwrap_or_default()
}

fn replay_target_prefix(
    speculative: &mut MtpSpeculative<'_>,
    committed_tokens: &[llama_cpp_2::token::LlamaToken],
    sync_batch: &mut LlamaBatch<'_>,
) -> Result<()> {
    speculative.target_context_mut().clear_kv_cache();
    let batch_size = usize::try_from(speculative.target_context().n_batch())
        .unwrap_or(usize::MAX)
        .max(1);
    for (chunk_index, chunk) in committed_tokens.chunks(batch_size).enumerate() {
        let offset = chunk_index.saturating_mul(batch_size);
        let mut batch = LlamaBatch::new(chunk.len().max(1), 1);
        for (index, &token) in chunk.iter().enumerate() {
            let absolute = offset.saturating_add(index);
            let position = i32::try_from(absolute).map_err(|_| {
                PowerError::InferenceFailed(
                    "MTP target replay position exceeds llama.cpp limits".to_string(),
                )
            })?;
            batch.add(token, position, &[0], false).map_err(|_| {
                PowerError::InferenceFailed(
                    "Failed to add committed token to MTP target replay batch".to_string(),
                )
            })?;
        }
        speculative
            .target_context_mut()
            .decode(&mut batch)
            .map_err(|error| {
                PowerError::InferenceFailed(format!("MTP committed target replay failed: {error}"))
            })?;
    }
    speculative
        .target_context_mut()
        .decode(sync_batch)
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "MTP accepted-prefix target replay failed: {error}"
            ))
        })
}

pub(super) fn run_mtp_completion(
    model: &LlamaModel,
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    context_settings: LlamaContextSettings,
    sampling_settings: &LlamaSamplingSettings,
    settings: MtpCompletionSettings,
    tx: &tokio::sync::mpsc::Sender<Result<CompletionResponseChunk>>,
) -> Result<()> {
    let speculative_params = mtp_speculative_params(&settings)?;
    if tokens.is_empty() {
        return Err(PowerError::InferenceFailed(
            "llama.cpp MTP requires a non-empty tokenized prompt".to_string(),
        ));
    }

    // Verification outputs one anchor plus at most draft_max proposal rows.
    // llama.cpp's recurrent splitter additionally requires n_batch to be
    // strictly larger than its anchor-plus-snapshot tail, hence the staging
    // row in minimum_batch. Keep output storage at the exact verification-row
    // count even though the physical batch has one more slot.
    let minimum_batch = minimum_mtp_batch(settings.draft_max);
    let target_output_rows = settings.draft_max.saturating_add(1);
    let recurrent_snapshots = settings.draft_max.min(settings.recurrent_snapshots);
    let target_params = context_settings.params(
        LlamaContextType::Default,
        recurrent_snapshots,
        minimum_batch,
        target_output_rows,
    );
    // A fresh MTP context is request-scoped, so deterministic greedy sampling
    // can live in its CUDA graph. This avoids copying every 248k-vocabulary
    // verification row to the CPU. Keep a separate CPU sampler below so the
    // streaming verifier retains one model-neutral state machine.
    let backend_greedy = use_backend_greedy(sampling_settings);
    let target_context_result = if backend_greedy {
        let backend_sampler = build_llamacpp_sampler(model, sampling_settings)?;
        model.new_context_with_samplers(backend_ref(), target_params, [(0, backend_sampler)])
    } else {
        model.new_context(backend_ref(), target_params)
    };
    let target_context = target_context_result.map_err(|error| {
        PowerError::InferenceFailed(format!("Failed to create MTP target context: {error}"))
    })?;
    tracing::debug!(
        backend_greedy,
        recurrent_snapshots,
        configured_recurrent_snapshots = settings.recurrent_snapshots,
        "Configured MTP target sampling and rollback snapshots"
    );
    let draft_params = context_settings.params(LlamaContextType::Mtp, 0, minimum_batch, 1);
    let draft_context = model
        .new_context_with_ctx_other(backend_ref(), draft_params, &target_context)
        .map_err(|error| {
            PowerError::InferenceFailed(format!("Failed to create MTP draft context: {error}"))
        })?;

    let mut speculative = MtpSpeculative::new(target_context, draft_context, speculative_params)
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "Failed to initialize llama.cpp MTP speculation: {error}"
            ))
        })?;

    let prompt_eval_started = std::time::Instant::now();
    let prompt_batch_size = usize::try_from(speculative.target_context().n_batch())
        .unwrap_or(usize::MAX)
        .max(1);
    for (chunk_index, chunk) in tokens.chunks(prompt_batch_size).enumerate() {
        let offset = chunk_index.saturating_mul(prompt_batch_size);
        let mut batch = LlamaBatch::new(chunk.len().max(1), 1);
        for (index, &token) in chunk.iter().enumerate() {
            let absolute = offset.saturating_add(index);
            let position = i32::try_from(absolute).map_err(|_| {
                PowerError::InferenceFailed(
                    "MTP prompt position exceeds llama.cpp limits".to_string(),
                )
            })?;
            batch
                .add(token, position, &[0], absolute + 1 == tokens.len())
                .map_err(|_| {
                    PowerError::InferenceFailed(
                        "Failed to add MTP prompt token to batch".to_string(),
                    )
                })?;
        }
        speculative
            .target_context_mut()
            .decode(&mut batch)
            .map_err(|error| {
                PowerError::InferenceFailed(format!("MTP prompt decode failed: {error}"))
            })?;
        speculative.process(&batch).map_err(|error| {
            PowerError::InferenceFailed(format!("MTP prompt-state synchronization failed: {error}"))
        })?;
    }
    speculative.begin(&tokens).map_err(|error| {
        PowerError::InferenceFailed(format!("Failed to begin MTP generation: {error}"))
    })?;
    let prompt_eval_duration_ns = prompt_eval_started.elapsed().as_nanos() as u64;

    let mut sampler = build_llamacpp_sampler(model, sampling_settings)?;
    let eos_token = model.token_eos();
    let prompt_token_count = tokens.len() as u32;
    let mut committed_tokens = tokens;
    let mut generated_text = String::new();
    let mut generated_count = 0usize;
    let mut metrics = SpeculativeMetrics::default();
    let mut timings = MtpPhaseTimings::default();
    let decode_started = std::time::Instant::now();
    let target_context_size = speculative.target_context().n_ctx() as usize;
    if settings.max_tokens == 0 {
        log_metrics(metrics, timings, generated_count, decode_started);
        send_completion_result(
            tx,
            Ok(CompletionResponseChunk {
                text: String::new(),
                done: true,
                prompt_tokens: Some(prompt_token_count),
                done_reason: Some("length".to_string()),
                prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                token_id: None,
            }),
        );
        return Ok(());
    }
    let sampling_started = std::time::Instant::now();
    let mut anchor = sample_target_token(
        &mut sampler,
        speculative.target_context(),
        -1,
        backend_greedy,
    );
    timings.sampling_ns = timings
        .sampling_ns
        .saturating_add(elapsed_ns(sampling_started));

    // The first generated token is sampled from the final prompt row before
    // the first speculative target pass. It becomes the next round's anchor,
    // but must still be streamed and counted exactly once.
    let streaming_started = std::time::Instant::now();
    let keep_streaming = stream_token(
        model,
        anchor,
        eos_token,
        &mut generated_text,
        &mut generated_count,
        &settings.stop_sequences,
        prompt_token_count,
        prompt_eval_duration_ns,
        tx,
    );
    timings.streaming_ns = timings
        .streaming_ns
        .saturating_add(elapsed_ns(streaming_started));
    if !keep_streaming {
        log_metrics(metrics, timings, generated_count, decode_started);
        return Ok(());
    }

    while generated_count < settings.max_tokens {
        if committed_tokens.len() >= target_context_size {
            return Err(PowerError::InferenceFailed(
                "llama.cpp MTP context capacity was exhausted before generation completed"
                    .to_string(),
            ));
        }
        let n_past = i32::try_from(committed_tokens.len()).map_err(|_| {
            PowerError::InferenceFailed(
                "MTP sequence position exceeds llama.cpp limits".to_string(),
            )
        })?;
        let remaining_tokens = settings.max_tokens.saturating_sub(generated_count);
        let remaining_context =
            target_context_size.saturating_sub(committed_tokens.len().saturating_add(1));
        let draft_limit = bounded_draft_len(
            settings.draft_max as usize,
            remaining_tokens,
            remaining_context,
        );
        let draft_started = std::time::Instant::now();
        let mut drafts = if draft_limit >= settings.draft_min as usize && draft_limit > 0 {
            // This handle contains only llama.cpp's MTP implementation. At the
            // pinned revision MTP proposes from its synchronized recurrent
            // state plus `anchor`; unlike n-gram drafters it never reads the
            // history pointer. Avoid serializing and copying a growing token
            // history across Rust and C++ on every speculative round.
            speculative.draft(n_past, anchor, &[]).map_err(|error| {
                PowerError::InferenceFailed(format!("MTP drafting failed: {error}"))
            })?
        } else {
            Vec::new()
        };
        timings.draft_ns = timings.draft_ns.saturating_add(elapsed_ns(draft_started));
        let draft_pending = !drafts.is_empty();
        drafts.truncate(draft_limit);

        let state_management_started = std::time::Instant::now();
        let rollback_from = u32::try_from(committed_tokens.len()).map_err(|_| {
            PowerError::InferenceFailed(
                "MTP draft rollback position exceeds llama.cpp limits".to_string(),
            )
        })?;
        speculative
            .draft_context_mut()
            .kv_cache_seq_rm(0, Some(rollback_from), None)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "Failed to rewind MTP draft state before verification: {error}"
                ))
            })?;
        timings.state_management_ns = timings
            .state_management_ns
            .saturating_add(elapsed_ns(state_management_started));

        let mut verify_batch = LlamaBatch::new(drafts.len() + 1, 1);
        verify_batch.add(anchor, n_past, &[0], true).map_err(|_| {
            PowerError::InferenceFailed("Failed to add MTP anchor token to batch".to_string())
        })?;
        for (index, &draft) in drafts.iter().enumerate() {
            let position = n_past
                .checked_add(i32::try_from(index + 1).map_err(|_| {
                    PowerError::InferenceFailed(
                        "MTP draft position exceeds llama.cpp limits".to_string(),
                    )
                })?)
                .ok_or_else(|| {
                    PowerError::InferenceFailed(
                        "MTP draft position exceeds llama.cpp limits".to_string(),
                    )
                })?;
            verify_batch.add(draft, position, &[0], true).map_err(|_| {
                PowerError::InferenceFailed(
                    "Failed to add MTP draft token to verification batch".to_string(),
                )
            })?;
        }

        let target_decode_started = std::time::Instant::now();
        speculative
            .target_context_mut()
            .decode(&mut verify_batch)
            .map_err(|error| {
                PowerError::InferenceFailed(format!("MTP target verification failed: {error}"))
            })?;
        timings.target_decode_ns = timings
            .target_decode_ns
            .saturating_add(elapsed_ns(target_decode_started));

        let sampling_started = std::time::Instant::now();
        let mut preview_text =
            (!settings.stop_sequences.is_empty()).then(|| generated_text.clone());
        let verified = verify_token_block_until(
            &drafts,
            |row| {
                // `LlamaSampler::sample` also accepts the selected token,
                // advancing grammar, penalty, mirostat, and RNG state once.
                sample_target_token(
                    &mut sampler,
                    speculative.target_context(),
                    row as i32,
                    backend_greedy,
                )
            },
            |token| {
                if token == eos_token {
                    return true;
                }
                let Some(preview_text) = preview_text.as_mut() else {
                    return false;
                };
                preview_text.push_str(&token_piece(model, token));
                settings
                    .stop_sequences
                    .iter()
                    .any(|stop| preview_text.ends_with(stop))
            },
        );
        timings.sampling_ns = timings
            .sampling_ns
            .saturating_add(elapsed_ns(sampling_started));

        // Keep only a bounded number of recurrent rollback snapshots. Most
        // mismatches reject a short suffix and use llama.cpp's exact in-place
        // rollback. If a rejection exceeds that bound, rebuild the target from
        // the committed token history before synchronizing the accepted prefix.
        // This remains exact while decoupling peak state memory from draft_max.
        let accepted_prefix_sync_started = std::time::Instant::now();
        let mut sync_batch = LlamaBatch::new(verified.accepted.saturating_add(1), 1);
        sync_batch.add(anchor, n_past, &[0], true).map_err(|_| {
            PowerError::InferenceFailed(
                "Failed to add MTP anchor token to synchronization batch".to_string(),
            )
        })?;
        for (index, &draft) in drafts.iter().take(verified.accepted).enumerate() {
            let position = n_past
                .checked_add(i32::try_from(index + 1).map_err(|_| {
                    PowerError::InferenceFailed(
                        "MTP synchronization position exceeds llama.cpp limits".to_string(),
                    )
                })?)
                .ok_or_else(|| {
                    PowerError::InferenceFailed(
                        "MTP synchronization position exceeds llama.cpp limits".to_string(),
                    )
                })?;
            sync_batch.add(draft, position, &[0], true).map_err(|_| {
                PowerError::InferenceFailed(
                    "Failed to add accepted MTP token to synchronization batch".to_string(),
                )
            })?;
        }
        let rejected_suffix = drafts.len().saturating_sub(verified.accepted);
        timings.accepted_prefix_histogram[verified.accepted.min(64)] =
            timings.accepted_prefix_histogram[verified.accepted.min(64)].saturating_add(1);
        timings.max_rejected_suffix = timings.max_rejected_suffix.max(rejected_suffix);
        if rejected_suffix > recurrent_snapshots as usize {
            timings.fallback_replays = timings.fallback_replays.saturating_add(1);
            replay_target_prefix(&mut speculative, &committed_tokens, &mut sync_batch)?;
        }
        speculative.process(&sync_batch).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "MTP accepted-prefix synchronization failed: {error}"
            ))
        })?;
        timings.accepted_prefix_sync_ns = timings
            .accepted_prefix_sync_ns
            .saturating_add(elapsed_ns(accepted_prefix_sync_started));

        let state_management_started = std::time::Instant::now();
        if draft_pending {
            speculative
                .accept(verified.accepted as u16)
                .map_err(|error| {
                    PowerError::InferenceFailed(format!(
                        "Failed to commit MTP acceptance state: {error}"
                    ))
                })?;
        }

        let keep_position = committed_tokens
            .len()
            .checked_add(1)
            .and_then(|position| position.checked_add(verified.accepted))
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| {
                PowerError::InferenceFailed(
                    "MTP commit position exceeds llama.cpp limits".to_string(),
                )
            })?;
        speculative
            .target_context_mut()
            .kv_cache_seq_rm(0, Some(keep_position), None)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "Failed to rewind rejected MTP target state: {error}"
                ))
            })?;
        speculative
            .draft_context_mut()
            .kv_cache_seq_rm(0, Some(keep_position), None)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "Failed to rewind rejected MTP draft state: {error}"
                ))
            })?;

        committed_tokens.push(anchor);
        committed_tokens.extend(drafts.iter().take(verified.accepted).copied());
        timings.state_management_ns = timings
            .state_management_ns
            .saturating_add(elapsed_ns(state_management_started));

        let mut emitted_this_round = 0usize;
        let streaming_started = std::time::Instant::now();
        for token in verified.emitted.iter().copied() {
            if generated_count >= settings.max_tokens {
                break;
            }
            let count_before = generated_count;
            if !stream_token(
                model,
                token,
                eos_token,
                &mut generated_text,
                &mut generated_count,
                &settings.stop_sequences,
                prompt_token_count,
                prompt_eval_duration_ns,
                tx,
            ) {
                emitted_this_round += generated_count.saturating_sub(count_before);
                metrics.record_round(drafts.len(), verified.accepted, emitted_this_round);
                timings.streaming_ns = timings
                    .streaming_ns
                    .saturating_add(elapsed_ns(streaming_started));
                log_metrics(metrics, timings, generated_count, decode_started);
                return Ok(());
            }
            emitted_this_round += generated_count.saturating_sub(count_before);
        }
        timings.streaming_ns = timings
            .streaming_ns
            .saturating_add(elapsed_ns(streaming_started));

        metrics.record_round(drafts.len(), verified.accepted, emitted_this_round);
        anchor = *verified.emitted.last().ok_or_else(|| {
            PowerError::InferenceFailed("MTP verification emitted no continuation".to_string())
        })?;
    }

    log_metrics(metrics, timings, generated_count, decode_started);
    send_completion_result(
        tx,
        Ok(CompletionResponseChunk {
            text: String::new(),
            done: true,
            prompt_tokens: Some(prompt_token_count),
            done_reason: Some("length".to_string()),
            prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
            token_id: None,
        }),
    );
    Ok(())
}

#[cfg(test)]
mod tests;
