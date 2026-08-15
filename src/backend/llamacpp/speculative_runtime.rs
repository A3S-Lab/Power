//! llama.cpp adapter for Power's model-neutral speculative runtime.
//!
//! This module contains backend mechanics only. Strategy selection, capability
//! negotiation, block verification, and metrics remain in `crate::speculative`.

use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};

use super::{backend_ref, nonzero_context_size, send_completion_result};
use crate::backend::types::CompletionResponseChunk;
use crate::error::{PowerError, Result};
use crate::speculative::{
    bounded_draft_len, verify_token_block_until, SpeculativeCapabilities, SpeculativeMetrics,
    SpeculativeStrategy,
};

fn metadata_entry_enables_mtp(key: &str, value: &str) -> bool {
    key.ends_with(".nextn_predict_layers")
        && value.trim().parse::<u32>().is_ok_and(|layers| layers > 0)
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
}

impl LlamaContextSettings {
    pub(super) fn params(
        self,
        context_type: LlamaContextType,
        recurrent_snapshots: u32,
        minimum_batch: u32,
    ) -> LlamaContextParams {
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
        params
    }
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

fn log_metrics(
    metrics: SpeculativeMetrics,
    emitted_tokens: usize,
    decode_started: std::time::Instant,
) {
    let elapsed = decode_started.elapsed().as_secs_f64();
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

    let minimum_batch = settings.draft_max.saturating_add(1);
    let target_params =
        context_settings.params(LlamaContextType::Default, settings.draft_max, minimum_batch);
    let target_context = model
        .new_context(backend_ref(), target_params)
        .map_err(|error| {
            PowerError::InferenceFailed(format!("Failed to create MTP target context: {error}"))
        })?;
    let draft_params = context_settings.params(LlamaContextType::Mtp, 0, minimum_batch);
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
    let decode_started = std::time::Instant::now();
    let target_context_size = speculative.target_context().n_ctx() as usize;
    if settings.max_tokens == 0 {
        log_metrics(metrics, generated_count, decode_started);
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
    let mut anchor = sampler.sample(speculative.target_context(), -1);

    // The first generated token is sampled from the final prompt row before
    // the first speculative target pass. It becomes the next round's anchor,
    // but must still be streamed and counted exactly once.
    if !stream_token(
        model,
        anchor,
        eos_token,
        &mut generated_text,
        &mut generated_count,
        &settings.stop_sequences,
        prompt_token_count,
        prompt_eval_duration_ns,
        tx,
    ) {
        log_metrics(metrics, generated_count, decode_started);
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
        let mut drafts = if draft_limit >= settings.draft_min as usize && draft_limit > 0 {
            speculative
                .draft(n_past, anchor, &committed_tokens)
                .map_err(|error| {
                    PowerError::InferenceFailed(format!("MTP drafting failed: {error}"))
                })?
        } else {
            Vec::new()
        };
        let draft_pending = !drafts.is_empty();
        drafts.truncate(draft_limit);

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

        speculative
            .target_context_mut()
            .decode(&mut verify_batch)
            .map_err(|error| {
                PowerError::InferenceFailed(format!("MTP target verification failed: {error}"))
            })?;
        speculative.process(&verify_batch).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "MTP verification-state synchronization failed: {error}"
            ))
        })?;

        let mut preview_text = generated_text.clone();
        let verified = verify_token_block_until(
            &drafts,
            |row| {
                // `LlamaSampler::sample` also accepts the selected token,
                // advancing grammar, penalty, mirostat, and RNG state once.
                sampler.sample(speculative.target_context(), row as i32)
            },
            |token| {
                if token == eos_token {
                    return true;
                }
                preview_text.push_str(&token_piece(model, token));
                settings
                    .stop_sequences
                    .iter()
                    .any(|stop| preview_text.ends_with(stop))
            },
        );
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

        let mut emitted_this_round = 0usize;
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
                log_metrics(metrics, generated_count, decode_started);
                return Ok(());
            }
            emitted_this_round += generated_count.saturating_sub(count_before);
        }

        metrics.record_round(drafts.len(), verified.accepted, emitted_this_round);
        anchor = *verified.emitted.last().ok_or_else(|| {
            PowerError::InferenceFailed("MTP verification emitted no continuation".to_string())
        })?;
    }

    log_metrics(metrics, generated_count, decode_started);
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
mod tests {
    use super::{metadata_entry_enables_mtp, mtp_speculative_params, MtpCompletionSettings};

    fn settings() -> MtpCompletionSettings {
        MtpCompletionSettings {
            max_tokens: 16,
            stop_sequences: Vec::new(),
            draft_max: 3,
            draft_min: 0,
            draft_p_min: 0.0,
        }
    }

    #[test]
    fn mtp_metadata_detection_is_architecture_neutral() {
        assert!(metadata_entry_enables_mtp(
            "qwen35.nextn_predict_layers",
            "1"
        ));
        assert!(metadata_entry_enables_mtp(
            "future_arch.nextn_predict_layers",
            "3"
        ));
        assert!(!metadata_entry_enables_mtp(
            "qwen35.nextn_predict_layers",
            "0"
        ));
        assert!(!metadata_entry_enables_mtp(
            "general.architecture",
            "qwen35"
        ));
        assert!(!metadata_entry_enables_mtp(
            "future_arch.nextn_predict_layers",
            "invalid"
        ));
    }

    #[test]
    fn mtp_parameters_accept_adapter_defaults() {
        let params = mtp_speculative_params(&settings()).unwrap();
        assert_eq!(params.n_max, 3);
        assert_eq!(params.n_min, 0);
        assert_eq!(params.p_min, 0.0);
    }

    #[test]
    fn mtp_parameters_reject_minimum_above_adapter_default() {
        let error = mtp_speculative_params(&MtpCompletionSettings {
            draft_min: 4,
            ..settings()
        })
        .unwrap_err();
        assert!(error.to_string().contains("must not exceed"));
    }

    #[test]
    fn mtp_parameters_reject_invalid_programmatic_values() {
        for draft_max in [0, 65] {
            let error = mtp_speculative_params(&MtpCompletionSettings {
                draft_max,
                ..settings()
            })
            .unwrap_err();
            assert!(error.to_string().contains("between 1 and 64"));
        }

        let error = mtp_speculative_params(&MtpCompletionSettings {
            draft_p_min: f32::NAN,
            ..settings()
        })
        .unwrap_err();
        assert!(error.to_string().contains("finite"));
    }
}
