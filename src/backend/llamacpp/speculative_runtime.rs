//! llama.cpp adapter for Power's model-neutral speculative runtime.
//!
//! This module contains backend mechanics only. Strategy selection, capability
//! negotiation, block verification, and metrics remain in `crate::speculative`.

mod adapter;
mod context_settings;
mod metrics;
mod rollback_guard;
mod sampling;
mod stop_sequences;
mod streaming;

use llama_cpp_2::context::params::LlamaContextType;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::speculative::{
    ExternalDraftSpeculative, ExternalDraftSpeculativeKind, ExternalDraftSpeculativeParams,
    MtpSpeculative, MtpSpeculativeParams,
};

use super::{backend_ref, send_completion_result};
use crate::backend::types::CompletionResponseChunk;
use crate::error::{PowerError, Result};
use crate::model::external_draft::ExternalDraftKind;
use crate::speculative::{
    bounded_draft_len, minimum_mtp_batch, verify_token_block_until, AdaptiveSpeculationController,
    SpeculativeCapabilities, SpeculativeMetrics, SpeculativeStrategy,
};

use self::adapter::LlamaSpeculativeAdapter;
use self::context_settings::external_target_context_params;
#[cfg(all(test, feature = "llamacpp-mtp-fr"))]
use self::context_settings::llamacpp_context_mtp_fr_vocab;
#[cfg(test)]
use self::context_settings::llamacpp_context_output_limits;
pub(super) use self::context_settings::{LlamaContextSettings, MtpCompletionSettings};
pub(super) use self::metrics::SpeculativeTelemetry;
use self::metrics::{elapsed_ns, log_metrics, MtpPhaseTimings};
use self::rollback_guard::RollbackReplayGuard;
#[cfg(test)]
use self::sampling::use_greedy_fast_path;
pub(super) use self::sampling::{
    build_llamacpp_sampler, sample_target_token, use_backend_greedy, LlamaSamplingSettings,
};
use self::stop_sequences::StopSequenceTracker;
use self::streaming::{stream_token, token_piece};

fn metadata_entry_enables_mtp(key: &str, value: &str) -> bool {
    key.ends_with(".nextn_predict_layers")
        && value.trim().parse::<u32>().is_ok_and(|layers| layers > 0)
}

pub(super) fn ensure_mtp_fr_available(
    vocab_size: Option<u32>,
    model_architecture: Option<&str>,
) -> Result<()> {
    #[cfg(not(feature = "llamacpp-mtp-fr"))]
    if vocab_size.is_some() {
        return Err(PowerError::Config(
            "spec_mtp_fr_vocab_size requires the experimental \
             llamacpp-mtp-fr crate feature and the reviewed llama.cpp patch"
                .to_string(),
        ));
    }

    #[cfg(feature = "llamacpp-mtp-fr")]
    if vocab_size.is_some() && model_architecture != Some("qwen35") {
        return Err(PowerError::Config(format!(
            "spec_mtp_fr_vocab_size is not implemented for GGUF architecture '{}'; \
             use full-vocabulary MTP or a backend adapter that advertises reduced-vocabulary support",
            model_architecture.unwrap_or("unknown")
        )));
    }

    let _ = (vocab_size, model_architecture);
    Ok(())
}

pub(super) fn llamacpp_speculative_capabilities(
    model: &LlamaModel,
    mtp_weights_loaded: bool,
    external_draft: Option<ExternalDraftKind>,
) -> SpeculativeCapabilities {
    let mut capabilities = SpeculativeCapabilities::none();
    let has_mtp = mtp_weights_loaded
        && (0..model.meta_count()).any(|index| {
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
    if let Some(kind) = external_draft {
        capabilities = capabilities.with(match kind {
            ExternalDraftKind::Dflash => SpeculativeStrategy::Dflash,
            ExternalDraftKind::Dspark => SpeculativeStrategy::Dspark,
        });
    }
    capabilities
}

#[derive(Debug, Clone, Copy)]
struct MtpAdapterParams {
    upstream: MtpSpeculativeParams,
    draft_greedy: bool,
    recurrent_draft: bool,
}

// Field assignment is intentional: the default supplies experimental fields
// when the reviewed patch is present while the same source still compiles
// against the smaller upstream struct.
#[allow(clippy::field_reassign_with_default)]
fn mtp_speculative_params(
    settings: &MtpCompletionSettings,
    greedy_draft: bool,
) -> Result<MtpAdapterParams> {
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
    if settings.adaptive && settings.recurrent_snapshots < settings.draft_min.max(1) {
        return Err(PowerError::Config(format!(
            "spec_mtp_recurrent_snapshots ({}) must cover spec_draft_min ({}) when adaptive MTP is enabled",
            settings.recurrent_snapshots,
            settings.draft_min.max(1)
        )));
    }
    if !settings.draft_p_min.is_finite() || !(0.0..=1.0).contains(&settings.draft_p_min) {
        return Err(PowerError::Config(format!(
            "spec_draft_p_min must be finite and between 0 and 1 for llama.cpp MTP, got {}",
            settings.draft_p_min
        )));
    }
    let requested_greedy_draft = greedy_draft && settings.draft_p_min == 0.0;
    let draft_greedy = cfg!(feature = "llamacpp-mtp-fr") && requested_greedy_draft;
    let recurrent_draft = cfg!(feature = "llamacpp-mtp-fr") && settings.recurrent_chain;

    // Construct from the upstream default so an unpatched llama-cpp-rs remains
    // source-compatible. The experimental feature is the only place allowed to
    // reference fields added by Power's reviewed binding patch.
    let mut upstream = MtpSpeculativeParams::default();
    upstream.n_max = settings.draft_max as i32;
    upstream.n_min = settings.draft_min as i32;
    upstream.p_min = settings.draft_p_min;
    #[cfg(feature = "llamacpp-mtp-fr")]
    {
        upstream.greedy_draft = draft_greedy;
        upstream.recurrent_draft = recurrent_draft;
    }

    Ok(MtpAdapterParams {
        upstream,
        draft_greedy,
        recurrent_draft,
    })
}

fn replay_target_prefix<'target, 'draft, S: LlamaSpeculativeAdapter<'target, 'draft>>(
    speculative: &mut S,
    strategy_name: &str,
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
                PowerError::InferenceFailed(format!(
                    "{strategy_name} target replay position exceeds llama.cpp limits"
                ))
            })?;
            batch.add(token, position, &[0], false).map_err(|_| {
                PowerError::InferenceFailed(format!(
                    "Failed to add committed token to {strategy_name} target replay batch"
                ))
            })?;
        }
        speculative
            .target_context_mut()
            .decode(&mut batch)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "{strategy_name} committed target replay failed: {error}"
                ))
            })?;
    }
    speculative
        .target_context_mut()
        .decode(sync_batch)
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "{strategy_name} accepted-prefix target replay failed: {error}"
            ))
        })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_mtp_completion(
    model: &LlamaModel,
    model_name: &str,
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    context_settings: LlamaContextSettings,
    sampling_settings: &LlamaSamplingSettings,
    settings: MtpCompletionSettings,
    telemetry: &SpeculativeTelemetry,
    tx: &tokio::sync::mpsc::Sender<Result<CompletionResponseChunk>>,
) -> Result<()> {
    let backend_greedy = use_backend_greedy(sampling_settings);
    let adapter_params = mtp_speculative_params(&settings, backend_greedy)?;
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
    let target_context_result = if backend_greedy {
        let backend_sampler = build_llamacpp_sampler(model, sampling_settings)?;
        model.new_context_with_samplers(backend_ref(), target_params, [(0, backend_sampler)])
    } else {
        model.new_context(backend_ref(), target_params)
    };
    let target_context = target_context_result.map_err(|error| {
        PowerError::InferenceFailed(format!("Failed to create MTP target context: {error}"))
    })?;
    tracing::info!(
        backend_greedy,
        draft_greedy = adapter_params.draft_greedy,
        recurrent_draft = adapter_params.recurrent_draft,
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

    let mut speculative =
        MtpSpeculative::new(target_context, draft_context, adapter_params.upstream).map_err(
            |error| {
                PowerError::InferenceFailed(format!(
                    "Failed to initialize llama.cpp MTP speculation: {error}"
                ))
            },
        )?;

    run_speculative_completion(
        model,
        model_name,
        tokens,
        context_settings,
        sampling_settings,
        settings,
        SpeculativeStrategy::Mtp,
        recurrent_snapshots,
        &mut speculative,
        telemetry,
        tx,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_external_draft_completion(
    model: &LlamaModel,
    model_name: &str,
    draft_model: &LlamaModel,
    draft_kind: ExternalDraftKind,
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    context_settings: LlamaContextSettings,
    sampling_settings: &LlamaSamplingSettings,
    mut settings: MtpCompletionSettings,
    telemetry: &SpeculativeTelemetry,
    tx: &tokio::sync::mpsc::Sender<Result<CompletionResponseChunk>>,
) -> Result<()> {
    if tokens.is_empty() {
        return Err(PowerError::InferenceFailed(
            "llama.cpp external drafting requires a non-empty tokenized prompt".to_string(),
        ));
    }
    if settings.draft_max == 0 || settings.draft_max > 64 {
        return Err(PowerError::Config(format!(
            "spec_draft_max must be between 1 and 64 for llama.cpp external drafting, got {}",
            settings.draft_max
        )));
    }
    if settings.draft_min > settings.draft_max {
        return Err(PowerError::Config(format!(
            "spec_draft_min ({}) must not exceed spec_draft_max ({})",
            settings.draft_min, settings.draft_max
        )));
    }
    if !settings.draft_p_min.is_finite() || !(0.0..=1.0).contains(&settings.draft_p_min) {
        return Err(PowerError::Config(format!(
            "spec_draft_p_min must be finite and between 0 and 1, got {}",
            settings.draft_p_min
        )));
    }
    if settings.recurrent_snapshots == 0 || settings.recurrent_snapshots > 64 {
        return Err(PowerError::Config(format!(
            "spec_mtp_recurrent_snapshots must be between 1 and 64 for llama.cpp external drafting, got {}",
            settings.recurrent_snapshots
        )));
    }

    // DSpark already carries a confidence-aware trained head. Keep Power's
    // MTP-only target-disable controller out of this path until an independent
    // external-draft policy has been calibrated.
    settings.adaptive = false;
    let backend_greedy = use_backend_greedy(sampling_settings);
    let minimum_batch = minimum_mtp_batch(settings.draft_max);
    let rollback_window = settings.draft_max.min(settings.recurrent_snapshots);
    let target_params =
        external_target_context_params(context_settings, settings.draft_max, rollback_window);
    let target_context_result = if backend_greedy {
        let backend_sampler = build_llamacpp_sampler(model, sampling_settings)?;
        model.new_context_with_samplers(backend_ref(), target_params, [(0, backend_sampler)])
    } else {
        model.new_context(backend_ref(), target_params)
    };
    let target_context = target_context_result.map_err(|error| {
        PowerError::InferenceFailed(format!(
            "Failed to create external-draft target context: {error}"
        ))
    })?;
    let draft_params = context_settings.params(LlamaContextType::Default, 0, minimum_batch, 1);
    let draft_context = draft_model
        .new_context_with_ctx_other(backend_ref(), draft_params, &target_context)
        .map_err(|error| {
            PowerError::InferenceFailed(format!("Failed to create external-draft context: {error}"))
        })?;
    let strategy = match draft_kind {
        ExternalDraftKind::Dflash => SpeculativeStrategy::Dflash,
        ExternalDraftKind::Dspark => SpeculativeStrategy::Dspark,
    };
    let params = ExternalDraftSpeculativeParams {
        kind: match draft_kind {
            ExternalDraftKind::Dflash => ExternalDraftSpeculativeKind::Dflash,
            ExternalDraftKind::Dspark => ExternalDraftSpeculativeKind::Dspark,
        },
        n_max: settings.draft_max as i32,
        n_min: settings.draft_min as i32,
        p_min: settings.draft_p_min,
    };
    let mut speculative = ExternalDraftSpeculative::new(target_context, draft_context, params)
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "Failed to initialize llama.cpp {} speculation: {error}",
                strategy.as_str()
            ))
        })?;
    tracing::info!(
        strategy = strategy.as_str(),
        backend_greedy,
        draft_max = settings.draft_max,
        draft_min = settings.draft_min,
        draft_p_min = settings.draft_p_min,
        rollback_window,
        "Configured external-draft target sampling"
    );

    // DFlash-family draft state is transformer KV. The target may be hybrid
    // recurrent, so retain only the configured bounded rollback window; a
    // longer rejection is recovered by exact committed-prefix replay.
    run_speculative_completion(
        model,
        model_name,
        tokens,
        LlamaContextSettings {
            mtp_fr_vocab_size: None,
            ..context_settings
        },
        sampling_settings,
        settings,
        strategy,
        rollback_window,
        &mut speculative,
        telemetry,
        tx,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_speculative_completion<'target, 'draft, S: LlamaSpeculativeAdapter<'target, 'draft>>(
    model: &LlamaModel,
    model_name: &str,
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    context_settings: LlamaContextSettings,
    sampling_settings: &LlamaSamplingSettings,
    settings: MtpCompletionSettings,
    strategy: SpeculativeStrategy,
    recurrent_snapshots: u32,
    speculative: &mut S,
    telemetry: &SpeculativeTelemetry,
    tx: &tokio::sync::mpsc::Sender<Result<CompletionResponseChunk>>,
) -> Result<()> {
    let backend_greedy = use_backend_greedy(sampling_settings);
    let strategy_name = strategy.as_str();

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
                PowerError::InferenceFailed(format!(
                    "{strategy_name} prompt position exceeds llama.cpp limits"
                ))
            })?;
            batch
                .add(token, position, &[0], absolute + 1 == tokens.len())
                .map_err(|_| {
                    PowerError::InferenceFailed(format!(
                        "Failed to add {strategy_name} prompt token to batch"
                    ))
                })?;
        }
        speculative
            .target_context_mut()
            .decode(&mut batch)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "{strategy_name} prompt decode failed: {error}"
                ))
            })?;
        speculative.process(&batch)?;
    }
    speculative.begin(&tokens)?;
    let prompt_eval_duration_ns = prompt_eval_started.elapsed().as_nanos() as u64;

    let mut sampler = build_llamacpp_sampler(model, sampling_settings)?;
    let eos_token = model.token_eos();
    let prompt_token_count = tokens.len() as u32;
    let mut committed_tokens = tokens;
    let mut stop_tracker = StopSequenceTracker::new(&settings.stop_sequences);
    let mut generated_count = 0usize;
    let mut metrics = SpeculativeMetrics::default();
    let mut timings = MtpPhaseTimings::default();
    let mut adaptive = settings.adaptive.then(|| {
        AdaptiveSpeculationController::new(
            settings.draft_max as usize,
            settings.draft_min.max(1) as usize,
            settings.draft_max as usize,
            recurrent_snapshots as usize,
        )
    });
    let mut rollback_guard = (!settings.adaptive).then(|| {
        RollbackReplayGuard::new(settings.draft_max as usize, recurrent_snapshots as usize)
    });
    let decode_started = std::time::Instant::now();
    let target_context_size = speculative.target_context().n_ctx() as usize;
    let mtp_fr_vocab_size = context_settings.mtp_fr_vocab_size;
    if settings.max_tokens == 0 {
        log_metrics(
            model_name,
            strategy_name,
            telemetry,
            metrics,
            timings,
            generated_count,
            decode_started,
        );
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
        &mut stop_tracker,
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
        log_metrics(
            model_name,
            strategy_name,
            telemetry,
            metrics,
            timings,
            generated_count,
            decode_started,
        );
        return Ok(());
    }

    while generated_count < settings.max_tokens {
        if committed_tokens.len() >= target_context_size {
            return Err(PowerError::InferenceFailed(format!(
                "llama.cpp {strategy_name} context capacity was exhausted before generation completed"
            )));
        }
        let n_past = i32::try_from(committed_tokens.len()).map_err(|_| {
            PowerError::InferenceFailed(format!(
                "{strategy_name} sequence position exceeds llama.cpp limits"
            ))
        })?;
        if adaptive
            .as_ref()
            .is_some_and(|controller| controller.draft_limit().is_none())
        {
            let mut target_batch = LlamaBatch::new(1, 1);
            target_batch.add(anchor, n_past, &[0], true).map_err(|_| {
                PowerError::InferenceFailed(format!(
                    "Failed to add target-only {strategy_name} token to batch"
                ))
            })?;
            let target_decode_started = std::time::Instant::now();
            speculative
                .target_context_mut()
                .decode(&mut target_batch)
                .map_err(|error| {
                    PowerError::InferenceFailed(format!(
                        "{strategy_name} target-only decode failed: {error}"
                    ))
                })?;
            speculative.process(&target_batch)?;
            timings.target_only_decode_ns = timings
                .target_only_decode_ns
                .saturating_add(elapsed_ns(target_decode_started));

            let sampling_started = std::time::Instant::now();
            let next = sample_target_token(
                &mut sampler,
                speculative.target_context(),
                -1,
                backend_greedy,
            );
            timings.sampling_ns = timings
                .sampling_ns
                .saturating_add(elapsed_ns(sampling_started));
            committed_tokens.push(anchor);

            let streaming_started = std::time::Instant::now();
            let count_before = generated_count;
            let keep_streaming = stream_token(
                model,
                next,
                eos_token,
                &mut stop_tracker,
                &mut generated_count,
                &settings.stop_sequences,
                prompt_token_count,
                prompt_eval_duration_ns,
                tx,
            );
            timings.target_only_tokens = timings
                .target_only_tokens
                .saturating_add(generated_count.saturating_sub(count_before) as u64);
            timings.streaming_ns = timings
                .streaming_ns
                .saturating_add(elapsed_ns(streaming_started));
            if !keep_streaming {
                log_metrics(
                    model_name,
                    strategy_name,
                    telemetry,
                    metrics,
                    timings,
                    generated_count,
                    decode_started,
                );
                return Ok(());
            }
            anchor = next;
            continue;
        }
        let remaining_tokens = settings.max_tokens.saturating_sub(generated_count);
        let remaining_context =
            target_context_size.saturating_sub(committed_tokens.len().saturating_add(1));
        let requested_draft_limit = adaptive
            .as_ref()
            .and_then(AdaptiveSpeculationController::draft_limit)
            .or_else(|| {
                rollback_guard
                    .as_ref()
                    .map(RollbackReplayGuard::draft_limit)
            })
            .unwrap_or(settings.draft_max as usize);
        let draft_limit =
            bounded_draft_len(requested_draft_limit, remaining_tokens, remaining_context);
        timings.draft_limit_histogram[draft_limit.min(64)] =
            timings.draft_limit_histogram[draft_limit.min(64)].saturating_add(1);
        let draft_started = std::time::Instant::now();
        let mut recurrent_draft_steps = 0usize;
        let mut drafts = if draft_limit >= settings.draft_min as usize && draft_limit > 0 {
            // Model-backed adapters propose from synchronized state plus the
            // anchor and do not consume the full history on every round.
            let (drafts, steps) = speculative.draft_with_max(n_past, anchor, draft_limit)?;
            recurrent_draft_steps = steps;
            drafts
        } else {
            Vec::new()
        };
        if recurrent_draft_steps > 0 {
            timings.recurrent_draft_chains = timings.recurrent_draft_chains.saturating_add(1);
            timings.recurrent_draft_steps = timings
                .recurrent_draft_steps
                .saturating_add(u64::try_from(recurrent_draft_steps).unwrap_or(u64::MAX));
        }
        timings.draft_ns = timings.draft_ns.saturating_add(elapsed_ns(draft_started));
        let draft_pending = !drafts.is_empty();
        drafts.truncate(draft_limit);

        let state_management_started = std::time::Instant::now();
        let rollback_from = u32::try_from(committed_tokens.len()).map_err(|_| {
            PowerError::InferenceFailed(format!(
                "{strategy_name} draft rollback position exceeds llama.cpp limits"
            ))
        })?;
        speculative
            .draft_context_mut()
            .kv_cache_seq_rm(0, Some(rollback_from), None)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "Failed to rewind {strategy_name} draft state before verification: {error}"
                ))
            })?;
        timings.state_management_ns = timings
            .state_management_ns
            .saturating_add(elapsed_ns(state_management_started));

        let mut verify_batch = LlamaBatch::new(drafts.len() + 1, 1);
        verify_batch.add(anchor, n_past, &[0], true).map_err(|_| {
            PowerError::InferenceFailed(format!(
                "Failed to add {strategy_name} anchor token to batch"
            ))
        })?;
        for (index, &draft) in drafts.iter().enumerate() {
            let position = n_past
                .checked_add(i32::try_from(index + 1).map_err(|_| {
                    PowerError::InferenceFailed(format!(
                        "{strategy_name} draft position exceeds llama.cpp limits"
                    ))
                })?)
                .ok_or_else(|| {
                    PowerError::InferenceFailed(format!(
                        "{strategy_name} draft position exceeds llama.cpp limits"
                    ))
                })?;
            verify_batch.add(draft, position, &[0], true).map_err(|_| {
                PowerError::InferenceFailed(format!(
                    "Failed to add {strategy_name} draft token to verification batch"
                ))
            })?;
        }

        let target_decode_started = std::time::Instant::now();
        speculative
            .target_context_mut()
            .decode(&mut verify_batch)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "{strategy_name} target verification failed: {error}"
                ))
            })?;
        timings.target_decode_ns = timings
            .target_decode_ns
            .saturating_add(elapsed_ns(target_decode_started));

        let sampling_started = std::time::Instant::now();
        let mut preview_stop_tracker =
            (!settings.stop_sequences.is_empty()).then(|| stop_tracker.clone());
        let verified = verify_token_block_until(
            &drafts,
            |row| {
                // `LlamaSampler::sample` also accepts the selected token,
                // advancing grammar, penalty, mirostat, and RNG state once.
                let token = sample_target_token(
                    &mut sampler,
                    speculative.target_context(),
                    row as i32,
                    backend_greedy,
                );
                timings.fr.observe_target_sample(token.0, mtp_fr_vocab_size);
                token
            },
            |token| {
                if token == eos_token {
                    return true;
                }
                let Some(preview_stop_tracker) = preview_stop_tracker.as_mut() else {
                    return false;
                };
                preview_stop_tracker.push(&token_piece(model, token), &settings.stop_sequences)
            },
        );
        timings.sampling_ns = timings
            .sampling_ns
            .saturating_add(elapsed_ns(sampling_started));
        if verified.accepted < drafts.len() {
            let correction = verified.emitted.last().ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "{strategy_name} rejection did not emit a target correction"
                ))
            })?;
            timings
                .fr
                .observe_rejection(correction.0, mtp_fr_vocab_size);
        }

        // Keep only a bounded number of recurrent rollback snapshots. Most
        // mismatches reject a short suffix and use llama.cpp's exact in-place
        // rollback. If a rejection exceeds that bound, rebuild the target from
        // the committed token history before synchronizing the accepted prefix.
        // This remains exact while decoupling peak state memory from draft_max.
        let accepted_prefix_sync_started = std::time::Instant::now();
        let mut sync_batch = LlamaBatch::new(verified.accepted.saturating_add(1), 1);
        sync_batch.add(anchor, n_past, &[0], true).map_err(|_| {
            PowerError::InferenceFailed(format!(
                "Failed to add {strategy_name} anchor token to synchronization batch"
            ))
        })?;
        for (index, &draft) in drafts.iter().take(verified.accepted).enumerate() {
            let position = n_past
                .checked_add(i32::try_from(index + 1).map_err(|_| {
                    PowerError::InferenceFailed(format!(
                        "{strategy_name} synchronization position exceeds llama.cpp limits"
                    ))
                })?)
                .ok_or_else(|| {
                    PowerError::InferenceFailed(format!(
                        "{strategy_name} synchronization position exceeds llama.cpp limits"
                    ))
                })?;
            sync_batch.add(draft, position, &[0], true).map_err(|_| {
                PowerError::InferenceFailed(format!(
                    "Failed to add accepted {strategy_name} token to synchronization batch"
                ))
            })?;
        }
        let rejected_suffix = drafts.len().saturating_sub(verified.accepted);
        timings.accepted_prefix_histogram[verified.accepted.min(64)] =
            timings.accepted_prefix_histogram[verified.accepted.min(64)].saturating_add(1);
        timings.max_rejected_suffix = timings.max_rejected_suffix.max(rejected_suffix);
        if let Some(controller) = adaptive.as_mut() {
            let guard_before = controller.rollback_guard_after_round();
            controller.observe(verified.accepted, drafts.len());
            let guard_after = controller.rollback_guard_after_round();
            if guard_before.is_none() && guard_after.is_some() {
                timings.rollback_guard_activations =
                    timings.rollback_guard_activations.saturating_add(1);
                timings.rollback_guard_draft_limit = controller.effective_max();
                timings.rollback_guard_after_round = guard_after;
            }
            timings.target_only_after_round = controller.target_only_after_round();
        }
        if let Some(guard) = rollback_guard.as_mut() {
            if guard.observe_rejected_suffix(rejected_suffix, metrics.rounds.saturating_add(1)) {
                timings.rollback_guard_activations =
                    timings.rollback_guard_activations.saturating_add(1);
                timings.rollback_guard_draft_limit = guard.draft_limit();
                timings.rollback_guard_after_round = guard.activated_after_round();
            }
        }
        if rejected_suffix > recurrent_snapshots as usize {
            timings.fallback_replays = timings.fallback_replays.saturating_add(1);
            replay_target_prefix(
                speculative,
                strategy_name,
                &committed_tokens,
                &mut sync_batch,
            )?;
        }
        speculative.process(&sync_batch)?;
        timings.accepted_prefix_sync_ns = timings
            .accepted_prefix_sync_ns
            .saturating_add(elapsed_ns(accepted_prefix_sync_started));

        let state_management_started = std::time::Instant::now();
        if draft_pending {
            speculative.accept(verified.accepted as u16)?;
        }

        let keep_position = committed_tokens
            .len()
            .checked_add(1)
            .and_then(|position| position.checked_add(verified.accepted))
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "{strategy_name} commit position exceeds llama.cpp limits"
                ))
            })?;
        speculative
            .target_context_mut()
            .kv_cache_seq_rm(0, Some(keep_position), None)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "Failed to rewind rejected {strategy_name} target state: {error}"
                ))
            })?;
        speculative
            .draft_context_mut()
            .kv_cache_seq_rm(0, Some(keep_position), None)
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "Failed to rewind rejected {strategy_name} draft state: {error}"
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
                &mut stop_tracker,
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
                log_metrics(
                    model_name,
                    strategy_name,
                    telemetry,
                    metrics,
                    timings,
                    generated_count,
                    decode_started,
                );
                return Ok(());
            }
            emitted_this_round += generated_count.saturating_sub(count_before);
        }
        timings.streaming_ns = timings
            .streaming_ns
            .saturating_add(elapsed_ns(streaming_started));

        metrics.record_round(drafts.len(), verified.accepted, emitted_this_round);
        anchor = *verified.emitted.last().ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "{strategy_name} verification emitted no continuation"
            ))
        })?;
    }

    log_metrics(
        model_name,
        strategy_name,
        telemetry,
        metrics,
        timings,
        generated_count,
        decode_started,
    );
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
