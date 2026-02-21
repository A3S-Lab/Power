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
use super::picolm_ops::ffn::FfnActivation;
#[cfg(feature = "picolm")]
use super::picolm_ops::kv_cache::KvCache;
#[cfg(feature = "picolm")]
use super::picolm_ops::tokenizer::BpeTokenizer;

// ── Loaded model state ────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
struct LoadedModel {
    gguf: Arc<GgufFile>,
    cfg: ModelConfig,
    tokenizer: Arc<BpeTokenizer>,
    activation: FfnActivation,
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
    tokenizer: &'a BpeTokenizer,
    cfg: &'a ModelConfig,
    activation: FfnActivation,
    kv_cache: &'a mut KvCache,
    input_ids: &'a [u32],
    max_new_tokens: u32,
    temperature: f32,
    top_p: f32,
    seed: u64,
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
    let tokenizer = params.tokenizer;
    let activation = params.activation;
    let kv_cache = &mut params.kv_cache;
    let input_ids = params.input_ids;
    let n_embd = cfg.n_embd;
    let mut hidden = vec![0.0f32; n_embd];
    let mut rng_state: u64 = if params.seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { params.seed };

    // Get embedding tensor info
    let embd_raw = match gguf.tensor_bytes("token_embd.weight") {
        Ok(r) => r,
        Err(e) => { let _ = tx.blocking_send(Err(e)); return; }
    };
    let embd_type = match gguf.tensor_type("token_embd.weight") {
        Ok(t) => t,
        Err(e) => { let _ = tx.blocking_send(Err(e)); return; }
    };

    // Output norm + logit projection tensors
    let norm_raw = match gguf.tensor_bytes("output_norm.weight") {
        Ok(r) => r,
        Err(e) => { let _ = tx.blocking_send(Err(e)); return; }
    };
    let norm_type = match gguf.tensor_type("output_norm.weight") {
        Ok(t) => t,
        Err(e) => { let _ = tx.blocking_send(Err(e)); return; }
    };

    // output.weight may be tied to token_embd.weight
    let (out_raw, out_type) = match gguf.tensor_bytes("output.weight") {
        Ok(r) => (r, gguf.tensor_type("output.weight").unwrap_or(embd_type)),
        Err(_) => (embd_raw, embd_type), // weight tying
    };

    let start_pos = kv_cache.seq_len();

    // Prefill: process all input tokens (sequential, one at a time)
    for (i, &token_id) in input_ids.iter().enumerate() {
        let pos = start_pos + i;

        // Embedding lookup
        matmul::extract_row(embd_raw, embd_type, n_embd, token_id as usize, &mut hidden);

        // Layer-streaming: process each layer, then drop weight references
        for layer in 0..cfg.n_layers {
            if let Err(e) = attention::attention_layer(
                &mut hidden, gguf, layer, pos, kv_cache.layer_mut(layer), cfg,
            ) {
                let _ = tx.blocking_send(Err(e));
                return;
            }
            if let Err(e) = ffn::ffn_layer(&mut hidden, gguf, layer, cfg, activation) {
                let _ = tx.blocking_send(Err(e));
                return;
            }
        }
    }

    // Generate tokens
    let mut gen_pos = start_pos + input_ids.len();

    for _step in 0..params.max_new_tokens {
        // Final norm on hidden state from last prefill/generate step
        let mut normed = hidden.clone();
        norm::rms_norm(&mut normed, norm_raw, norm_type, cfg.norm_eps);

        // Logit projection
        let mut logits = vec![0.0f32; cfg.vocab_size];
        matmul::matvec(out_raw, out_type, cfg.vocab_size, n_embd, &normed, &mut logits);

        // Sample
        let next_token = sample_token(&logits, params.temperature, params.top_p, &mut rng_state) as u32;

        // Decode and send
        match tokenizer.decode(next_token) {
            None => {
                // EOS
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
                    return; // receiver dropped
                }
            }
        }

        // Forward pass for the new token
        matmul::extract_row(embd_raw, embd_type, n_embd, next_token as usize, &mut hidden);

        for layer in 0..cfg.n_layers {
            if let Err(e) = attention::attention_layer(
                &mut hidden, gguf, layer, gen_pos, kv_cache.layer_mut(layer), cfg,
            ) {
                let _ = tx.blocking_send(Err(e));
                return;
            }
            if let Err(e) = ffn::ffn_layer(&mut hidden, gguf, layer, cfg, activation) {
                let _ = tx.blocking_send(Err(e));
                return;
            }
        }

        gen_pos += 1;
    }

    // Max tokens reached
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
    loaded: Mutex<HashMap<String, LoadedModel>>,
    #[cfg(feature = "picolm")]
    max_seq_len: usize,
}

impl PicolmBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        tracing::info!("picolm backend initialized — pure Rust layer-streaming inference");
        let _ = &config;
        Self {
            #[cfg(feature = "picolm")]
            loaded: Mutex::new(HashMap::new()),
            #[cfg(feature = "picolm")]
            max_seq_len: 2048,
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
            let max_seq = self.max_seq_len;

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

            let kv_mem = (meta.n_layers as usize)
                * 2
                * (meta.n_kv_heads as usize)
                * head_dim
                * max_seq
                * 4;

            tracing::info!(
                model = %name,
                arch = %arch,
                n_layers = meta.n_layers,
                n_embd = meta.n_embd,
                n_ff = meta.n_ff,
                vocab_size = meta.vocab_size,
                max_seq_len = max_seq,
                kv_cache_mb = kv_mem / (1024 * 1024),
                "picolm: model loaded (layer-streaming mode)"
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
            let (gguf, cfg, tokenizer, activation, kv_cache) = {
                let mut loaded = self.loaded.lock().map_err(|_| {
                    PowerError::InferenceFailed(
                        "picolm: loaded models lock poisoned".to_string(),
                    )
                })?;
                let model = loaded
                    .get_mut(model_name)
                    .ok_or_else(|| PowerError::ModelNotFound(model_name.to_string()))?;

                // Get or create KV cache for this session
                let session_key = request
                    .session_id
                    .clone()
                    .unwrap_or_default();
                let kv = if session_key.is_empty() {
                    // Transient: new cache per request
                    KvCache::new(
                        model.cfg.n_layers,
                        model.cfg.n_kv_heads,
                        model.cfg.head_dim,
                        self.max_seq_len,
                    )
                } else {
                    model.sessions.remove(&session_key).unwrap_or_else(|| {
                        KvCache::new(
                            model.cfg.n_layers,
                            model.cfg.n_kv_heads,
                            model.cfg.head_dim,
                            self.max_seq_len,
                        )
                    })
                };

                (
                    Arc::clone(&model.gguf),
                    model.cfg.clone(),
                    Arc::clone(&model.tokenizer),
                    model.activation,
                    kv,
                )
            };

            let prompt = build_prompt(&request.messages);
            let input_ids = tokenizer.encode(&prompt);
            let temperature = request.temperature.unwrap_or(0.8);
            let top_p = request.top_p.unwrap_or(0.95);
            let max_new_tokens = request.max_tokens.unwrap_or(512);
            let seed = request.seed.map(|s| s as u64).unwrap_or(0);

            let (tx, rx) = mpsc::channel::<Result<ChatResponseChunk>>(128);

            // Clone what we need for the session return
            let _session_key = request.session_id.clone().unwrap_or_default();

            let loaded_mutex = {
                // We need a way to return the KV cache after generation.
                // Use a shared Arc<Mutex<Option<KvCache>>> pattern.
                Arc::new(Mutex::new(Some(kv_cache)))
            };
            let kv_return = Arc::clone(&loaded_mutex);

            tokio::task::spawn_blocking(move || {
                let mut kv = kv_return.lock().unwrap().take().unwrap();
                let mut params = GenerateParams {
                    gguf: &gguf,
                    tokenizer: &tokenizer,
                    cfg: &cfg,
                    activation,
                    kv_cache: &mut kv,
                    input_ids: &input_ids,
                    max_new_tokens,
                    temperature,
                    top_p,
                    seed,
                };
                forward_pass_streaming(&mut params, &tx);
                // Put KV cache back for session reuse
                *kv_return.lock().unwrap() = Some(kv);
            });

            // TODO: return KV cache to session map after stream completes
            // For now, session KV reuse requires the stream to complete fully.

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
fn build_prompt(messages: &[ChatMessage]) -> String {
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
        let p = build_prompt(&msgs);
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
}
