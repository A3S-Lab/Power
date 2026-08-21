use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::{PowerError, Result};

#[derive(Debug, Clone)]
pub(in crate::backend::llamacpp) struct LlamaSamplingSettings {
    pub(in crate::backend::llamacpp) response_format: Option<serde_json::Value>,
    pub(in crate::backend::llamacpp) repeat_penalty: Option<f32>,
    pub(in crate::backend::llamacpp) frequency_penalty: Option<f32>,
    pub(in crate::backend::llamacpp) presence_penalty: Option<f32>,
    pub(in crate::backend::llamacpp) repeat_last_n: i32,
    pub(in crate::backend::llamacpp) mirostat: Option<u32>,
    pub(in crate::backend::llamacpp) mirostat_tau: Option<f32>,
    pub(in crate::backend::llamacpp) mirostat_eta: Option<f32>,
    pub(in crate::backend::llamacpp) temperature: f32,
    pub(in crate::backend::llamacpp) top_k: Option<i32>,
    pub(in crate::backend::llamacpp) typical_p: Option<f32>,
    pub(in crate::backend::llamacpp) top_p: f32,
    pub(in crate::backend::llamacpp) min_p: Option<f32>,
    pub(in crate::backend::llamacpp) seed: u32,
}

pub(in crate::backend::llamacpp) fn use_greedy_fast_path(settings: &LlamaSamplingSettings) -> bool {
    settings.mirostat.is_none()
        && settings.temperature <= 0.0
        && settings.top_k.is_none()
        && settings.typical_p.is_none()
        && settings.top_p >= 1.0
        && settings.min_p.is_none()
}

pub(in crate::backend::llamacpp) fn use_backend_greedy(settings: &LlamaSamplingSettings) -> bool {
    settings.response_format.is_none()
        && settings.repeat_penalty.is_none()
        && settings.frequency_penalty.is_none()
        && settings.presence_penalty.is_none()
        && use_greedy_fast_path(settings)
}

pub(in crate::backend::llamacpp) fn sample_target_token(
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

pub(in crate::backend::llamacpp) fn build_llamacpp_sampler(
    model: &LlamaModel,
    settings: &LlamaSamplingSettings,
) -> Result<LlamaSampler> {
    let mut samplers = Vec::new();
    if let Some(ref format) = settings.response_format {
        match super::super::super::json_schema::format_to_gbnf(format) {
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
