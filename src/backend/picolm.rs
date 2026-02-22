//! picolm inference backend — pure Rust layer-streaming GGUF inference.
//!
//! Streams transformer layers one at a time through a single working buffer.
//! Peak RAM stays at O(layer_size) rather than O(model_size), enabling 7B+
//! models inside a 512MB TEE EPC budget.
//!
//! Supported: GGUF v2/v3, LLaMA-architecture models, Q4_K_M / Q8_0 / F16 / F32.
//! Not supported: embeddings, vision, SafeTensors, HuggingFace format.
//!
//! # TEE supply-chain note
//!
//! This backend has zero C dependencies. The entire inference path is pure Rust
//! and can be audited without a C/C++ toolchain.

use std::pin::Pin;
use std::sync::Arc;

#[cfg(feature = "picolm")]
use std::collections::HashMap;
#[cfg(feature = "picolm")]
use std::sync::Mutex;

use async_trait::async_trait;
use futures::Stream;

#[cfg(feature = "picolm")]
use tokio::sync::mpsc;
#[cfg(feature = "picolm")]
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::types::{
    ChatMessage, ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk,
    EmbeddingRequest, EmbeddingResponse, MessageContent,
};
use crate::backend::Backend;
use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};
use crate::server::request_context::RequestContext;

#[cfg(feature = "picolm")]
use super::gguf_stream::GgufFile;
#[cfg(feature = "picolm")]
use super::picolm_ops::attention::ModelConfig;
#[cfg(feature = "picolm")]
use super::picolm_ops::buffers::ForwardBuffers;
#[cfg(feature = "picolm")]
use super::picolm_ops::ffn::FfnActivation;
#[cfg(feature = "picolm")]
use super::picolm_ops::kv_cache::KvCache;
#[cfg(feature = "picolm")]
use super::picolm_ops::rope::RopeTable;
#[cfg(feature = "picolm")]
use super::picolm_ops::tensor_cache::TensorCache;
#[cfg(feature = "picolm")]
use super::picolm_ops::tokenizer::BpeTokenizer;

// ── Loaded model state ────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
struct LoadedModel {
    gguf: Arc<GgufFile>,
    cfg: ModelConfig,
    tokenizer: Arc<BpeTokenizer>,
    activation: FfnActivation,
    /// Pre-computed RoPE cos/sin tables (eliminates powf/sin/cos from hot path).
    rope_table: Arc<RopeTable>,
    /// Pre-dequantized output norm weights only (`n_embd` floats).
    /// Per-layer norms are dequantized on-the-fly during the forward pass.
    output_norm: Arc<Vec<f32>>,
    /// Per-layer tensor pointer cache — eliminates HashMap lookups from the hot path.
    tensor_cache: Arc<TensorCache>,
    /// Jinja2 chat template from GGUF metadata (None = ChatML fallback).
    chat_template: Option<String>,
    /// Maximum sequence length for this model (from GGUF metadata, capped).
    max_seq_len: usize,
    /// Session-keyed KV caches for multi-turn reuse.
    sessions: HashMap<String, KvCache>,
}

// ── Sampler ───────────────────────────────────────────────────────────────────

/// Minimal top-p + temperature sampler operating on a logit vector.
#[cfg(feature = "picolm")]
fn sample_token(logits: &[f32], temperature: f32, top_p: f32, rng_state: &mut u64) -> usize {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) / temperature).exp())
        .collect();

    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
    sorted_indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumulative = 0.0f32;
    let mut nucleus: Vec<usize> = Vec::new();
    for &idx in &sorted_indices {
        nucleus.push(idx);
        cumulative += probs[idx];
        if cumulative >= top_p {
            break;
        }
    }

    // xorshift64 PRNG
    *rng_state ^= *rng_state << 13;
    *rng_state ^= *rng_state >> 7;
    *rng_state ^= *rng_state << 17;
    let r = (*rng_state as f32) / (u64::MAX as f32);

    let nucleus_sum: f32 = nucleus.iter().map(|&i| probs[i]).sum();
    let mut threshold = r * nucleus_sum;
    for &idx in &nucleus {
        threshold -= probs[idx];
        if threshold <= 0.0 {
            return idx;
        }
    }
    nucleus[0]
}

// ── Generation parameters ────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
struct GenerateParams<'a> {
    gguf: &'a GgufFile,
    tensor_cache: &'a TensorCache,
    tokenizer: &'a BpeTokenizer,
    cfg: &'a ModelConfig,
    activation: FfnActivation,
    kv_cache: &'a mut KvCache,
    rope_table: &'a RopeTable,
    /// Pre-dequantized output norm only (`n_embd` floats).
    output_norm: &'a [f32],
    input_ids: &'a [u32],
    max_new_tokens: u32,
    max_seq_len: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
    /// Stop sequences — generation stops when any is found in the output.
    stop: Vec<String>,
}

// ── Forward pass ─────────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
fn forward_pass_streaming(
    params: &mut GenerateParams<'_>,
    tx: &mpsc::Sender<Result<ChatResponseChunk>>,
) {
    use super::picolm_ops::{attention, ffn, matmul, norm};

    let cfg = params.cfg;
    let gguf = params.gguf;
    let tc = params.tensor_cache;
    let tokenizer = params.tokenizer;
    let activation = params.activation;
    let kv_cache = &mut params.kv_cache;
    let rope_table = params.rope_table;
    let output_norm_w = params.output_norm;
    let input_ids = params.input_ids;
    let n_embd = cfg.n_embd;

    // Single hidden-state buffer reused across all tokens.
    let mut hidden = vec![0.0f32; n_embd];

    // Pre-allocated working buffers — zero heap allocation in the hot path.
    let mut buf = ForwardBuffers::new(
        cfg.n_embd,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.n_ff,
        cfg.vocab_size,
        params.max_seq_len,
    );

    // Scratch buffers for on-the-fly norm dequantization (n_embd floats each).
    let mut attn_norm_buf = vec![0.0f32; n_embd];
    let mut ffn_norm_buf = vec![0.0f32; n_embd];

    let mut rng_state: u64 = if params.seed == 0 {
        0xDEAD_BEEF_CAFE_1234
    } else {
        params.seed
    };

    // Embedding tensor — looked up once, reused every token.
    let embd_raw = match gguf.tensor_bytes("token_embd.weight") {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.blocking_send(Err(e));
            return;
        }
    };
    let embd_type = match gguf.tensor_type("token_embd.weight") {
        Ok(t) => t,
        Err(e) => {
            let _ = tx.blocking_send(Err(e));
            return;
        }
    };

    // Output projection tensors — looked up once.
    let (out_raw, out_type) = match gguf.tensor_bytes("output.weight") {
        Ok(r) => (r, gguf.tensor_type("output.weight").unwrap_or(embd_type)),
        Err(_) => (embd_raw, embd_type), // weight tying
    };

    let start_pos = kv_cache.seq_len();

    /// Dequantize a single norm weight vector from the GGUF file into `out`.
    /// Norm tensors are tiny (n_embd floats) so this is microseconds.
    macro_rules! load_norm {
        ($name:expr, $buf:expr) => {
            match gguf.tensor_bytes($name) {
                Ok(raw) => {
                    let t = gguf.tensor_type($name).unwrap_or(0);
                    matmul::extract_row(raw, t, n_embd, 0, $buf);
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(e));
                    return;
                }
            }
        };
    }

    // Prefill: process all input tokens.
    let prefill_start = std::time::Instant::now();
    for (i, &token_id) in input_ids.iter().enumerate() {
        let pos = start_pos + i;
        matmul::extract_row(embd_raw, embd_type, n_embd, token_id as usize, &mut hidden);

        for layer in 0..cfg.n_layers {
            let attn_name = format!("blk.{layer}.attn_norm.weight");
            let ffn_name = format!("blk.{layer}.ffn_norm.weight");
            load_norm!(&attn_name, &mut attn_norm_buf);
            load_norm!(&ffn_name, &mut ffn_norm_buf);

            if let Err(e) = attention::attention_layer(
                &mut hidden,
                tc,
                layer,
                pos,
                kv_cache.layer_mut(layer),
                cfg,
                rope_table,
                &attn_norm_buf,
                &mut buf,
            ) {
                let _ = tx.blocking_send(Err(e));
                return;
            }
            if let Err(e) = ffn::ffn_layer(
                &mut hidden,
                tc,
                layer,
                cfg,
                activation,
                &ffn_norm_buf,
                &mut buf,
            ) {
                let _ = tx.blocking_send(Err(e));
                return;
            }

            // Release physical pages for this layer's weights + norms.
            let _ = tc.release_layer(gguf, layer);
            let _ = gguf.advise_dontneed(&attn_name);
            let _ = gguf.advise_dontneed(&ffn_name);
        }
    }
    let prefill_elapsed = prefill_start.elapsed();
    let prefill_tok_per_sec = if prefill_elapsed.as_secs_f64() > 0.0 {
        input_ids.len() as f64 / prefill_elapsed.as_secs_f64()
    } else {
        0.0
    };
    tracing::debug!(
        tokens = input_ids.len(),
        elapsed_ms = prefill_elapsed.as_millis() as u64,
        tok_per_sec = format!("{:.1}", prefill_tok_per_sec),
        "picolm: prefill complete"
    );

    // Generate tokens.
    let mut gen_pos = start_pos + input_ids.len();
    let decode_start = std::time::Instant::now();
    let mut decode_count = 0u32;
    let mut generated_text = String::new();

    for _step in 0..params.max_new_tokens {
        // Final norm — write into buf.normed_final, then copy to buf.normed_final.
        buf.normed_final[..n_embd].copy_from_slice(&hidden);
        norm::rms_norm_f32(&mut buf.normed_final[..n_embd], output_norm_w, cfg.norm_eps);

        // Logit projection into pre-allocated buf.logits.
        matmul::matvec(
            out_raw,
            out_type,
            cfg.vocab_size,
            n_embd,
            &buf.normed_final[..n_embd],
            &mut buf.logits,
        );

        // Sample.
        let next_token = sample_token(
            &buf.logits,
            params.temperature,
            params.top_p,
            &mut rng_state,
        ) as u32;

        // Decode and send.
        decode_count += 1;
        match tokenizer.decode(next_token) {
            None => {
                let decode_elapsed = decode_start.elapsed();
                let decode_tok_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
                    decode_count as f64 / decode_elapsed.as_secs_f64()
                } else {
                    0.0
                };
                tracing::debug!(
                    tokens = decode_count,
                    elapsed_ms = decode_elapsed.as_millis() as u64,
                    tok_per_sec = format!("{:.1}", decode_tok_per_sec),
                    "picolm: decode complete (eos)"
                );
                let _ = tx.blocking_send(Ok(ChatResponseChunk {
                    content: String::new(),
                    thinking_content: None,
                    done: true,
                    prompt_tokens: Some(input_ids.len() as u32),
                    done_reason: Some("stop".to_string()),
                    prompt_eval_duration_ns: None,
                    tool_calls: None,
                }));
                return;
            }
            Some(piece) => {
                generated_text.push_str(&piece);

                // Check stop sequences
                let mut hit_stop = false;
                for stop_seq in &params.stop {
                    if let Some(pos) = generated_text.find(stop_seq.as_str()) {
                        // Trim the piece to exclude the stop sequence
                        let overshoot = generated_text.len() - pos;
                        let trimmed = if overshoot <= piece.len() {
                            &piece[..piece.len() - overshoot]
                        } else {
                            ""
                        };
                        if !trimmed.is_empty() {
                            let _ = tx.blocking_send(Ok(ChatResponseChunk {
                                content: trimmed.to_string(),
                                thinking_content: None,
                                done: false,
                                prompt_tokens: None,
                                done_reason: None,
                                prompt_eval_duration_ns: None,
                                tool_calls: None,
                            }));
                        }
                        hit_stop = true;
                        break;
                    }
                }

                if hit_stop {
                    let decode_elapsed = decode_start.elapsed();
                    let decode_tok_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
                        decode_count as f64 / decode_elapsed.as_secs_f64()
                    } else {
                        0.0
                    };
                    tracing::debug!(
                        tokens = decode_count,
                        elapsed_ms = decode_elapsed.as_millis() as u64,
                        tok_per_sec = format!("{:.1}", decode_tok_per_sec),
                        "picolm: decode complete (stop sequence)"
                    );
                    let _ = tx.blocking_send(Ok(ChatResponseChunk {
                        content: String::new(),
                        thinking_content: None,
                        done: true,
                        prompt_tokens: Some(input_ids.len() as u32),
                        done_reason: Some("stop".to_string()),
                        prompt_eval_duration_ns: None,
                        tool_calls: None,
                    }));
                    return;
                }

                if tx
                    .blocking_send(Ok(ChatResponseChunk {
                        content: piece,
                        thinking_content: None,
                        done: false,
                        prompt_tokens: None,
                        done_reason: None,
                        prompt_eval_duration_ns: None,
                        tool_calls: None,
                    }))
                    .is_err()
                {
                    return;
                }
            }
        }

        // Forward pass for the new token.
        matmul::extract_row(
            embd_raw,
            embd_type,
            n_embd,
            next_token as usize,
            &mut hidden,
        );

        for layer in 0..cfg.n_layers {
            let attn_name = format!("blk.{layer}.attn_norm.weight");
            let ffn_name = format!("blk.{layer}.ffn_norm.weight");
            load_norm!(&attn_name, &mut attn_norm_buf);
            load_norm!(&ffn_name, &mut ffn_norm_buf);

            if let Err(e) = attention::attention_layer(
                &mut hidden,
                tc,
                layer,
                gen_pos,
                kv_cache.layer_mut(layer),
                cfg,
                rope_table,
                &attn_norm_buf,
                &mut buf,
            ) {
                let _ = tx.blocking_send(Err(e));
                return;
            }
            if let Err(e) = ffn::ffn_layer(
                &mut hidden,
                tc,
                layer,
                cfg,
                activation,
                &ffn_norm_buf,
                &mut buf,
            ) {
                let _ = tx.blocking_send(Err(e));
                return;
            }

            // Release physical pages for this layer's weights + norms.
            let _ = tc.release_layer(gguf, layer);
            let _ = gguf.advise_dontneed(&attn_name);
            let _ = gguf.advise_dontneed(&ffn_name);
        }

        gen_pos += 1;
    }

    // Max tokens reached.
    let decode_elapsed = decode_start.elapsed();
    let decode_tok_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
        decode_count as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };
    tracing::debug!(
        tokens = decode_count,
        elapsed_ms = decode_elapsed.as_millis() as u64,
        tok_per_sec = format!("{:.1}", decode_tok_per_sec),
        "picolm: decode complete (max tokens)"
    );
    let _ = tx.blocking_send(Ok(ChatResponseChunk {
        content: String::new(),
        thinking_content: None,
        done: true,
        prompt_tokens: Some(input_ids.len() as u32),
        done_reason: Some("length".to_string()),
        prompt_eval_duration_ns: None,
        tool_calls: None,
    }));
}

// ── Backend implementation ────────────────────────────────────────────────────

/// picolm inference backend — pure Rust, layer-streaming, zero C dependencies.
pub struct PicolmBackend {
    #[cfg(feature = "picolm")]
    loaded: Arc<Mutex<HashMap<String, LoadedModel>>>,
    #[cfg(feature = "picolm")]
    max_seq_len: usize,
}

impl PicolmBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        tracing::info!("picolm backend initialized — pure Rust layer-streaming inference");
        let _ = &config;
        Self {
            #[cfg(feature = "picolm")]
            loaded: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "picolm")]
            max_seq_len: 32768,
        }
    }
}

#[async_trait]
impl Backend for PicolmBackend {
    fn name(&self) -> &str {
        "picolm"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        #[cfg(not(feature = "picolm"))]
        {
            let _ = manifest;
            return Err(PowerError::BackendNotAvailable(
                "picolm feature not enabled — rebuild with --features picolm".to_string(),
            ));
        }

        #[cfg(feature = "picolm")]
        {
            let path = manifest.path.clone();
            let name = manifest.name.clone();
            let max_seq_cap = self.max_seq_len;

            let gguf = tokio::task::spawn_blocking(move || GgufFile::open(&path))
                .await
                .map_err(|e| PowerError::InferenceFailed(format!("picolm load task: {e}")))?
                .map_err(|e| {
                    PowerError::InferenceFailed(format!("picolm: failed to open GGUF: {e}"))
                })?;

            let meta = &gguf.meta;
            let arch = &meta.arch;
            let supported = ["llama", "mistral", "phi", "gemma", "qwen"];
            if !supported.iter().any(|a| arch.contains(a)) {
                return Err(PowerError::InvalidFormat(format!(
                    "picolm only supports LLaMA-compatible architectures, got '{arch}'."
                )));
            }

            let head_dim = meta.n_embd as usize / meta.n_heads as usize;
            let rope_dim = meta.rope_dim.map(|d| d as usize).unwrap_or(head_dim);

            let cfg = ModelConfig {
                n_embd: meta.n_embd as usize,
                n_heads: meta.n_heads as usize,
                n_kv_heads: meta.n_kv_heads as usize,
                head_dim,
                n_layers: meta.n_layers,
                n_ff: meta.n_ff as usize,
                vocab_size: meta.vocab_size as usize,
                norm_eps: meta.norm_eps,
                rope_theta: meta.rope_theta,
                rope_dim,
                context_length: meta.context_length as usize,
                bos_token_id: meta.bos_token_id as u32,
                eos_token_id: meta.eos_token_id as u32,
            };

            // Determine activation
            let activation = if arch.contains("gemma") {
                FfnActivation::Gelu
            } else {
                FfnActivation::Silu
            };

            // Build tokenizer from GGUF metadata
            let tokenizer = BpeTokenizer::from_gguf(
                &meta.vocab_tokens,
                &meta.vocab_scores,
                &meta.vocab_types,
                cfg.bos_token_id,
                cfg.eos_token_id,
            );

            // Use model's context length from GGUF metadata, capped by backend limit.
            // This avoids the hardcoded 2048 that silently truncated long-context models.
            let max_seq = (meta.context_length as usize).min(max_seq_cap);

            // Pre-compute RoPE cos/sin tables (eliminates powf/sin/cos from hot path)
            let rope_table = RopeTable::new(max_seq, head_dim, rope_dim, cfg.rope_theta);

            // Pre-dequantize only the output norm (used every token).
            // Per-layer norms are dequantized on-the-fly in the forward pass —
            // they are tiny (n_embd floats) and take microseconds each.
            let n_embd = cfg.n_embd;
            let out_norm_name = "output_norm.weight";
            let out_norm_raw = gguf.tensor_bytes(out_norm_name).map_err(|e| {
                PowerError::InferenceFailed(format!("picolm: missing {out_norm_name}: {e}"))
            })?;
            let out_norm_type = gguf.tensor_type(out_norm_name).map_err(|e| {
                PowerError::InferenceFailed(format!("picolm: missing {out_norm_name} type: {e}"))
            })?;
            let mut output_norm = vec![0.0f32; n_embd];
            super::picolm_ops::matmul::extract_row(
                out_norm_raw,
                out_norm_type,
                n_embd,
                0,
                &mut output_norm,
            );

            // Build per-layer tensor pointer cache (eliminates HashMap lookups from hot path).
            let tensor_cache = super::picolm_ops::tensor_cache::TensorCache::build(
                &gguf,
                meta.n_layers,
            )
            .map_err(|e| {
                PowerError::InferenceFailed(format!("picolm: tensor cache build failed: {e}"))
            })?;

            let kv_mem =
                (meta.n_layers as usize) * 2 * (meta.n_kv_heads as usize) * head_dim * max_seq * 2; // f16: 2 bytes

            // Clone metadata fields before moving gguf into Arc.
            let model_chat_template = meta.chat_template.clone();
            let log_arch = meta.arch.clone();
            let log_n_layers = meta.n_layers;
            let log_n_embd = meta.n_embd;
            let log_n_ff = meta.n_ff;
            let log_vocab_size = meta.vocab_size;

            tracing::info!(
                model = %name,
                arch = %log_arch,
                n_layers = log_n_layers,
                n_embd = log_n_embd,
                n_ff = log_n_ff,
                vocab_size = log_vocab_size,
                max_seq_len = max_seq,
                kv_cache_mb = kv_mem / (1024 * 1024),
                "picolm: model loaded (layer-streaming mode, optimized)"
            );

            let mut loaded = self.loaded.lock().map_err(|_| {
                PowerError::InferenceFailed("picolm: loaded models lock poisoned".to_string())
            })?;
            loaded.insert(
                name,
                LoadedModel {
                    gguf: Arc::new(gguf),
                    cfg,
                    tokenizer: Arc::new(tokenizer),
                    activation,
                    rope_table: Arc::new(rope_table),
                    output_norm: Arc::new(output_norm),
                    tensor_cache: Arc::new(tensor_cache),
                    chat_template: model_chat_template,
                    max_seq_len: max_seq,
                    sessions: HashMap::new(),
                },
            );
            Ok(())
        }
    }

    async fn unload(&self, model_name: &str) -> Result<()> {
        #[cfg(not(feature = "picolm"))]
        {
            let _ = model_name;
            return Ok(());
        }

        #[cfg(feature = "picolm")]
        {
            let mut loaded = self.loaded.lock().map_err(|_| {
                PowerError::InferenceFailed("picolm: loaded models lock poisoned".to_string())
            })?;
            if loaded.remove(model_name).is_some() {
                tracing::debug!(model = %model_name, "picolm: model unloaded");
            }
            Ok(())
        }
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        #[cfg(not(feature = "picolm"))]
        {
            let _ = (model_name, request);
            return Err(PowerError::BackendNotAvailable(
                "picolm feature not enabled".to_string(),
            ));
        }

        #[cfg(feature = "picolm")]
        {
            let (
                gguf,
                cfg,
                tokenizer,
                activation,
                kv_cache,
                rope_table,
                output_norm,
                tensor_cache,
                chat_template,
                max_seq_len,
            ) = {
                let mut loaded = self.loaded.lock().map_err(|_| {
                    PowerError::InferenceFailed("picolm: loaded models lock poisoned".to_string())
                })?;
                let model = loaded
                    .get_mut(model_name)
                    .ok_or_else(|| PowerError::ModelNotFound(model_name.to_string()))?;

                let model_max_seq = model.max_seq_len;

                // Get or create KV cache for this session
                let session_key = request.session_id.clone().unwrap_or_default();
                let kv = if session_key.is_empty() {
                    // Transient: new cache per request
                    KvCache::new(
                        model.cfg.n_layers,
                        model.cfg.n_kv_heads,
                        model.cfg.head_dim,
                        model_max_seq,
                    )
                } else {
                    model.sessions.remove(&session_key).unwrap_or_else(|| {
                        KvCache::new(
                            model.cfg.n_layers,
                            model.cfg.n_kv_heads,
                            model.cfg.head_dim,
                            model_max_seq,
                        )
                    })
                };

                (
                    Arc::clone(&model.gguf),
                    model.cfg.clone(),
                    Arc::clone(&model.tokenizer),
                    model.activation,
                    kv,
                    Arc::clone(&model.rope_table),
                    Arc::clone(&model.output_norm),
                    Arc::clone(&model.tensor_cache),
                    model.chat_template.clone(),
                    model_max_seq,
                )
            };

            let prompt = build_prompt(&request.messages, chat_template.as_deref());
            let input_ids = tokenizer.encode(&prompt);
            let temperature = request.temperature.unwrap_or(0.8);
            let top_p = request.top_p.unwrap_or(0.95);
            let max_new_tokens = request.max_tokens.unwrap_or(512);
            let seed = request.seed.map(|s| s as u64).unwrap_or(0);
            let stop = request.stop.clone().unwrap_or_default();

            let (tx, rx) = mpsc::channel::<Result<ChatResponseChunk>>(128);

            let session_key = request.session_id.clone().unwrap_or_default();
            let model_name_owned = model_name.to_string();

            // Shuttle the KV cache through the blocking task via a shared slot.
            let kv_slot = Arc::new(Mutex::new(Some(kv_cache)));
            let kv_return = Arc::clone(&kv_slot);

            let blocking_handle = tokio::task::spawn_blocking(move || {
                let mut kv = kv_return.lock().unwrap().take().unwrap();
                let mut params = GenerateParams {
                    gguf: &gguf,
                    tensor_cache: &tensor_cache,
                    tokenizer: &tokenizer,
                    cfg: &cfg,
                    activation,
                    kv_cache: &mut kv,
                    rope_table: &rope_table,
                    output_norm: &output_norm,
                    input_ids: &input_ids,
                    max_new_tokens,
                    max_seq_len,
                    temperature,
                    top_p,
                    seed,
                    stop,
                };
                forward_pass_streaming(&mut params, &tx);
                // Put KV cache back into the slot so the return task can pick it up.
                *kv_return.lock().unwrap() = Some(kv);
            });

            // Return the KV cache to the session map once generation finishes.
            if !session_key.is_empty() {
                let loaded_arc = Arc::clone(&self.loaded);
                tokio::spawn(async move {
                    let _ = blocking_handle.await;
                    if let Ok(Some(kv)) = kv_slot.lock().map(|mut g| g.take()) {
                        if let Ok(mut map) = loaded_arc.lock() {
                            if let Some(model) = map.get_mut(&model_name_owned) {
                                model.sessions.insert(session_key, kv);
                            }
                        }
                    }
                });
            }

            Ok(Box::pin(ReceiverStream::new(rx)))
        }
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        let chat_req = completion_to_chat(request);
        let chat_stream = self.chat(model_name, chat_req).await?;

        use futures::StreamExt;
        let stream = chat_stream.map(|r| {
            r.map(|chunk| CompletionResponseChunk {
                text: chunk.content,
                done: chunk.done,
                prompt_tokens: chunk.prompt_tokens,
                done_reason: chunk.done_reason,
                prompt_eval_duration_ns: chunk.prompt_eval_duration_ns,
                token_id: None,
            })
        });
        Ok(Box::pin(stream))
    }

    async fn embed(
        &self,
        _model_name: &str,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        Err(PowerError::BackendNotAvailable(
            "picolm does not support embeddings; use mistralrs with a HuggingFace embedding model"
                .to_string(),
        ))
    }

    async fn cleanup_request(&self, _model_name: &str, _ctx: &RequestContext) -> Result<()> {
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[cfg(any(feature = "picolm", test))]
fn build_prompt(messages: &[ChatMessage], chat_template: Option<&str>) -> String {
    if let Some(tmpl) = chat_template {
        if let Ok(rendered) = render_jinja_template(tmpl, messages) {
            return rendered;
        }
        // Fall through to ChatML on template error
    }
    // ChatML fallback
    let mut out = String::new();
    for msg in messages {
        let content = msg.content.text();
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Render a Jinja2 chat template with the given messages.
#[cfg(any(feature = "picolm", test))]
fn render_jinja_template(
    template_str: &str,
    messages: &[ChatMessage],
) -> std::result::Result<String, String> {
    let env = minijinja::Environment::new();
    let tmpl = env
        .template_from_str(template_str)
        .map_err(|e| format!("template parse error: {e}"))?;

    // Build messages array for the template context
    let msgs: Vec<minijinja::Value> = messages
        .iter()
        .map(|m| {
            let mut map = std::collections::BTreeMap::new();
            map.insert("role".to_string(), minijinja::Value::from(m.role.as_str()));
            map.insert(
                "content".to_string(),
                minijinja::Value::from(m.content.text()),
            );
            minijinja::Value::from_serialize(&map)
        })
        .collect();

    let ctx = minijinja::context! {
        messages => msgs,
        add_generation_prompt => true,
        bos_token => "<s>",
        eos_token => "</s>",
    };

    tmpl.render(ctx)
        .map_err(|e| format!("template render error: {e}"))
}

fn completion_to_chat(req: CompletionRequest) -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(req.prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }],
        session_id: req.session_id,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stop: req.stop,
        stream: req.stream,
        top_k: req.top_k,
        min_p: req.min_p,
        repeat_penalty: req.repeat_penalty,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        seed: req.seed,
        num_ctx: req.num_ctx,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format: req.response_format,
        tools: None,
        tool_choice: None,
        repeat_last_n: req.repeat_last_n,
        penalize_newline: req.penalize_newline,
        num_batch: req.num_batch,
        num_thread: req.num_thread,
        num_thread_batch: req.num_thread_batch,
        flash_attention: req.flash_attention,
        num_gpu: req.num_gpu,
        main_gpu: req.main_gpu,
        use_mmap: req.use_mmap,
        use_mlock: req.use_mlock,
        num_parallel: req.num_parallel,
        images: req.images,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Arc<PowerConfig> {
        Arc::new(PowerConfig::default())
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(PicolmBackend::new(test_config()).name(), "picolm");
    }

    #[test]
    fn test_supports_gguf_only() {
        let b = PicolmBackend::new(test_config());
        assert!(b.supports(&ModelFormat::Gguf));
        assert!(!b.supports(&ModelFormat::SafeTensors));
        assert!(!b.supports(&ModelFormat::HuggingFace));
        assert!(!b.supports(&ModelFormat::Vision));
    }

    #[test]
    fn test_build_prompt_ends_with_assistant() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }];
        let p = build_prompt(&msgs, None);
        assert!(p.ends_with("<|im_start|>assistant\n"));
        assert!(p.contains("Hello"));
    }

    #[tokio::test]
    async fn test_embed_returns_error() {
        let b = PicolmBackend::new(test_config());
        let r = b
            .embed(
                "m",
                EmbeddingRequest {
                    input: vec!["x".into()],
                },
            )
            .await;
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("embeddings"));
    }

    #[tokio::test]
    async fn test_unload_nonexistent_is_ok() {
        assert!(PicolmBackend::new(test_config())
            .unload("ghost")
            .await
            .is_ok());
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_sample_greedy_picks_max() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.2];
        let mut rng = 42u64;
        assert_eq!(sample_token(&logits, 0.0, 1.0, &mut rng), 1);
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_sample_returns_valid_index() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.2];
        let mut rng = 99999u64;
        let t = sample_token(&logits, 0.8, 0.95, &mut rng);
        assert!(t < logits.len());
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_kv_cache_session_insert_remove() {
        // Verify the HashMap insert/remove pattern used for session KV reuse.
        let mut sessions: HashMap<String, KvCache> = HashMap::new();
        let key = "session-abc".to_string();

        // First turn: no existing cache → create fresh.
        let kv = sessions
            .remove(&key)
            .unwrap_or_else(|| KvCache::new(2, 4, 64, 128));
        assert_eq!(kv.seq_len(), 0);

        // Simulate generation completing: put cache back.
        sessions.insert(key.clone(), kv);
        assert!(sessions.contains_key(&key));

        // Second turn: existing cache is retrieved and removed.
        let kv2 = sessions
            .remove(&key)
            .unwrap_or_else(|| KvCache::new(2, 4, 64, 128));
        // Cache was returned, so it should be gone from the map now.
        assert!(!sessions.contains_key(&key));
        drop(kv2);

        // Transient (empty session key): never inserted.
        let transient_key = String::new();
        assert!(transient_key.is_empty());
        let _kv = sessions
            .remove(&transient_key)
            .unwrap_or_else(|| KvCache::new(2, 4, 64, 128));
        // Empty key must never be stored back.
        assert!(!sessions.contains_key(&transient_key));
    }

    #[test]
    fn test_build_prompt_chatml_fallback() {
        // When no template is provided, should use ChatML format
        let msgs = vec![
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text("You are helpful.".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Hi".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
        ];
        let p = build_prompt(&msgs, None);
        assert!(p.contains("<|im_start|>system"));
        assert!(p.contains("<|im_start|>user"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_build_prompt_jinja_template() {
        // Llama 3 style template
        let template = "{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }];
        let p = build_prompt(&msgs, Some(template));
        assert!(p.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(p.contains("Hello"));
        assert!(p.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn test_build_prompt_invalid_template_falls_back() {
        // Invalid Jinja2 should fall back to ChatML
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("Hi".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }];
        let p = build_prompt(&msgs, Some("{% invalid jinja %}"));
        assert!(p.contains("<|im_start|>"));
    }
}
