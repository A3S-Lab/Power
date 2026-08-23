use super::*;

pub(super) async fn complete(
    backend: &LlamaCppBackend,
    model_name: &str,
    request: CompletionRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;

    if request
        .use_mmap
        .is_some_and(|use_mmap| use_mmap != backend.config.use_mmap)
    {
        return Err(PowerError::InvalidRequest(format!(
            "llama.cpp use_mmap is fixed at model load ({}) and must be configured globally",
            backend.config.use_mmap
        )));
    }

    let (
        model_arc,
        session_cache,
        lora_adapter,
        model_n_ctx_train,
        mtmd_ctx,
        speculative_capabilities,
        external_draft,
    ) = {
        let models = backend.models.read().await;
        models
            .get(model_name)
            .map(|m| {
                (
                    m.model.clone(),
                    m.session_cache.clone(),
                    m.lora_adapter.clone(),
                    m.n_ctx_train,
                    m.mtmd_ctx.clone(),
                    m.speculative_capabilities,
                    m.external_draft
                        .as_ref()
                        .map(|draft| (draft.model.clone(), draft.identity.clone())),
                )
            })
            .ok_or_else(|| {
                PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
            })?
    };

    let max_tokens = request.max_tokens.unwrap_or(512) as usize;
    let temperature = request.temperature.unwrap_or(0.8);
    let top_p = request.top_p.unwrap_or(0.95);
    let top_k = request.top_k;
    let min_p = request.min_p;
    let repeat_penalty = request.repeat_penalty;
    let frequency_penalty = request.frequency_penalty;
    let presence_penalty = request.presence_penalty;
    let repeat_last_n = request.repeat_last_n.unwrap_or(64);
    let _penalize_newline = request.penalize_newline.unwrap_or(true);
    let seed = request.seed.unwrap_or(0).max(0) as u32;
    let ctx_size = match request.num_ctx {
        Some(requested) => {
            if requested > model_n_ctx_train {
                tracing::warn!(
                    requested = requested,
                    trained = model_n_ctx_train,
                    "Requested context size exceeds model's trained context length, quality may degrade"
                );
            }
            requested
        }
        None => {
            let effective = DEFAULT_CTX_SIZE.min(model_n_ctx_train);
            tracing::info!(
                default = effective,
                trained = model_n_ctx_train,
                "Using default context size (override with num_ctx or --num-ctx)"
            );
            effective
        }
    };
    let num_batch = validated_llamacpp_batch(request.num_batch, ctx_size)?;
    // Per-request num_thread overrides config default; fall back to config if not set
    let num_thread = request.num_thread.or(backend.config.num_thread);
    let num_thread_batch = request.num_thread_batch;
    // Per-request flash_attention overrides config default
    let flash_attention = request
        .flash_attention
        .unwrap_or(backend.config.flash_attention);
    let mirostat = request.mirostat;
    let mirostat_tau = request.mirostat_tau;
    let mirostat_eta = request.mirostat_eta;
    let _tfs_z = request.tfs_z; // tail_free sampling removed in llama-cpp-2 v0.1.133
    let typical_p = request.typical_p;
    let response_format = request.response_format.clone();
    let stop_sequences = request.stop.clone().unwrap_or_default();
    let has_images = request.images.as_ref().is_some_and(|v| !v.is_empty());
    ensure_llamacpp_images_supported(model_name, has_images, mtmd_ctx.is_some())?;

    let requested_strategy = crate::speculative::SpeculativeStrategy::parse(
        &backend.config.spec_mode,
    )
    .ok_or_else(|| {
        PowerError::Config(format!(
            "unsupported spec_mode '{}'",
            backend.config.spec_mode
        ))
    })?;
    let model_backed_request_compatible =
        !has_images && request.session_id.is_none() && lora_adapter.is_none();
    let external_default = external_draft
        .as_ref()
        .map(|(_, identity)| external_draft_strategy(identity.kind));
    let backend_default = if model_backed_request_compatible {
        external_default
            .filter(|strategy| speculative_capabilities.supports(*strategy))
            .or_else(|| {
                speculative_capabilities
                    .supports(crate::speculative::SpeculativeStrategy::Mtp)
                    .then_some(crate::speculative::SpeculativeStrategy::Mtp)
            })
            .unwrap_or(crate::speculative::SpeculativeStrategy::Off)
    } else {
        crate::speculative::SpeculativeStrategy::Off
    };
    let speculative_strategy = speculative_capabilities
        .resolve(requested_strategy, backend_default)
        .map_err(|error| PowerError::Config(format!("llama.cpp: {error}")))?;
    let model_architecture = model_arc.meta_val_str("general.architecture").ok();
    if matches!(
        speculative_strategy,
        crate::speculative::SpeculativeStrategy::Mtp
    ) {
        ensure_mtp_fr_available(
            backend.config.spec_mtp_fr_vocab_size,
            model_architecture.as_deref(),
        )?;
    }
    if matches!(
        speculative_strategy,
        crate::speculative::SpeculativeStrategy::Mtp
            | crate::speculative::SpeculativeStrategy::Dflash
            | crate::speculative::SpeculativeStrategy::Dflash2
            | crate::speculative::SpeculativeStrategy::Dspark
    ) && !model_backed_request_compatible
    {
        return Err(PowerError::InvalidRequest(format!(
            "llama.cpp {} currently requires text-only inference without session caching or LoRA",
            speculative_strategy.as_str()
        )));
    }
    if matches!(
        speculative_strategy,
        crate::speculative::SpeculativeStrategy::Mtp
            | crate::speculative::SpeculativeStrategy::Dflash
            | crate::speculative::SpeculativeStrategy::Dflash2
            | crate::speculative::SpeculativeStrategy::Dspark
    ) {
        let default_draft_max = if matches!(
            speculative_strategy,
            crate::speculative::SpeculativeStrategy::Mtp
        ) {
            3
        } else {
            4
        };
        let draft_max = backend.config.spec_draft_max.unwrap_or(default_draft_max);
        let minimum_batch = crate::speculative::minimum_mtp_batch(draft_max);
        if num_batch.is_some_and(|batch| batch < minimum_batch) {
            return Err(PowerError::InvalidRequest(format!(
                "llama.cpp {} num_batch must be at least draft_max + 2 ({minimum_batch})",
                speculative_strategy.as_str()
            )));
        }
    }

    let context_settings = LlamaContextSettings {
        ctx_size,
        num_batch,
        num_thread,
        num_thread_batch,
        flash_attention,
        mtp_fr_vocab_size: backend.config.spec_mtp_fr_vocab_size,
    };
    let sampling_settings = LlamaSamplingSettings {
        response_format,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        repeat_last_n,
        mirostat,
        mirostat_tau,
        mirostat_eta,
        temperature,
        top_k,
        typical_p,
        top_p,
        min_p,
        seed,
    };
    let default_draft_max = if matches!(
        speculative_strategy,
        crate::speculative::SpeculativeStrategy::Mtp
    ) {
        3
    } else {
        4
    };
    let speculative_settings = MtpCompletionSettings {
        max_tokens,
        stop_sequences: stop_sequences.clone(),
        // llama.cpp's native MTP adapter defaults to three draft tokens.
        // Other model-backed adapters can resolve a different default.
        draft_max: backend.config.spec_draft_max.unwrap_or(default_draft_max),
        recurrent_snapshots: backend.config.spec_mtp_recurrent_snapshots,
        recurrent_chain: backend.config.spec_mtp_recurrent_chain,
        adaptive: backend.config.spec_mtp_adaptive,
        draft_min: backend.config.spec_draft_min,
        draft_p_min: backend.config.spec_draft_p_min,
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<CompletionResponseChunk>>(32);

    let session_id = request.session_id.clone();
    // Anonymous greedy requests receive a fresh context, so their stateless
    // sampler can be owned by that context and executed in the CUDA graph.
    // Session contexts may outlive this request and therefore cannot retain
    // a request-specific backend sampler.
    let plain_backend_greedy = session_id.is_none() && use_backend_greedy(&sampling_settings);
    let session_cache_for_return = session_cache.clone();
    let prompt_cache_telemetry = backend.prompt_cache_telemetry.clone();
    let speculative_telemetry = backend.speculative_telemetry.clone();
    let speculative_model_name = model_name.to_string();

    // Run inference in a blocking task
    tokio::task::spawn_blocking(move || {
        let prompt_eval_start = std::time::Instant::now();

        // Determine whether to use the MTMD (multimodal) path.
        // Conditions: images present in request AND mtmd_ctx loaded for this model.
        let use_mtmd = has_images && mtmd_ctx.is_some();

        // ----------------------------------------------------------------
        // MTMD path: vision/multimodal inference
        // ----------------------------------------------------------------
        if use_mtmd {
            use llama_cpp_2::mtmd::{mtmd_default_marker, MtmdBitmap, MtmdInputText};

            let mtmd_guard = match mtmd_ctx.as_ref() {
                Some(ctx) => match lock_mtmd_context(ctx) {
                    Ok(guard) => guard,
                    Err(e) => {
                        send_completion_result(&tx, Err(e));
                        return;
                    }
                },
                None => {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(
                            "llama.cpp: MTMD context missing for multimodal request".to_string(),
                        )),
                    );
                    return;
                }
            };
            let mtmd = &mtmd_guard.0;

            // Build bitmaps from base64-encoded images.
            // Images are never logged — they pass through the privacy boundary here.
            let mut bitmaps: Vec<MtmdBitmap> = Vec::new();
            for b64 in request.images.as_deref().unwrap_or(&[]) {
                // Strip data URI prefix if present (e.g. "data:image/png;base64,...")
                let b64_data = b64.find(',').map(|i| &b64[i + 1..]).unwrap_or(b64.as_str());
                let raw = match base64::Engine::decode(
                    &base64::engine::general_purpose::STANDARD,
                    b64_data,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "Failed to decode base64 image: {e}"
                            ))),
                        );
                        return;
                    }
                };
                match MtmdBitmap::from_buffer(mtmd, &raw, false) {
                    Ok(bm) => bitmaps.push(bm),
                    Err(e) => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "Failed to create bitmap from image data: {e}"
                            ))),
                        );
                        return;
                    }
                }
            }

            // Insert media markers into the prompt — one per image.
            let marker = mtmd_default_marker();
            let markers: String = std::iter::repeat_n(marker, bitmaps.len())
                .collect::<Vec<_>>()
                .join("\n");
            let prompt_with_markers = format!("{markers}\n{}", request.prompt);

            let input_text = MtmdInputText {
                text: prompt_with_markers,
                add_special: true,
                parse_special: true,
            };

            let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
            let chunks = match mtmd.tokenize(input_text, &bitmap_refs) {
                Ok(c) => c,
                Err(e) => {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "MTMD tokenization failed: {e}"
                        ))),
                    );
                    return;
                }
            };

            let prompt_token_count = chunks.total_tokens() as u32;

            // Create a fresh context for multimodal inference (no KV cache reuse —
            // image embeddings are request-specific and must not leak across sessions).
            let ctx_params =
                LlamaContextParams::default().with_n_ctx(Some(nonzero_context_size(ctx_size)));
            let mut ctx = match model_arc.new_context(backend_ref(), ctx_params) {
                Ok(c) => {
                    let c: llama_cpp_2::context::LlamaContext<'static> =
                        unsafe { std::mem::transmute(c) };
                    c
                }
                Err(e) => {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "Failed to create MTMD context: {e}"
                        ))),
                    );
                    return;
                }
            };

            // Evaluate all chunks (text + image embeddings) via the MTMD helper.
            let n_batch = num_batch.unwrap_or(512) as i32;
            let n_past = match chunks.eval_chunks(mtmd, &ctx, 0, 0, n_batch, true) {
                Ok(n) => n,
                Err(e) => {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "MTMD eval_chunks failed: {e}"
                        ))),
                    );
                    return;
                }
            };

            let prompt_eval_duration_ns = prompt_eval_start.elapsed().as_nanos() as u64;

            // Build sampler and generate tokens (same as text path below)
            let mut samplers: Vec<llama_cpp_2::sampling::LlamaSampler> = Vec::new();
            if let Some(temp) = request.temperature {
                if temp > 0.0 {
                    samplers.push(llama_cpp_2::sampling::LlamaSampler::temp(temp));
                }
            }
            samplers.push(llama_cpp_2::sampling::LlamaSampler::greedy());
            let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain(samplers, false);

            let eos_token = model_arc.token_eos();
            let mut generated_text = String::new();

            for generated_count in 0..max_tokens {
                let new_token = sampler.sample(&ctx, -1);
                if new_token == eos_token {
                    send_completion_result(
                        &tx,
                        Ok(CompletionResponseChunk {
                            text: String::new(),
                            done: true,
                            prompt_tokens: Some(prompt_token_count),
                            done_reason: Some("stop".to_string()),
                            prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                            token_id: None,
                        }),
                    );
                    return;
                }

                let text = {
                    let mut decoder = encoding_rs::UTF_8.new_decoder();
                    model_arc
                        .token_to_piece(new_token, &mut decoder, true, None)
                        .unwrap_or_default()
                };
                generated_text.push_str(&text);

                let should_stop = stop_sequences.iter().any(|s| generated_text.ends_with(s));

                if !send_completion_result(
                    &tx,
                    Ok(CompletionResponseChunk {
                        text,
                        done: should_stop,
                        prompt_tokens: if should_stop {
                            Some(prompt_token_count)
                        } else {
                            None
                        },
                        done_reason: if should_stop {
                            Some("stop".to_string())
                        } else {
                            None
                        },
                        prompt_eval_duration_ns: if should_stop {
                            Some(prompt_eval_duration_ns)
                        } else {
                            None
                        },
                        token_id: Some(new_token.0 as u32),
                    }),
                ) || should_stop
                {
                    return;
                }

                let mut batch = LlamaBatch::new(1, 1);
                let n_cur = n_past + generated_count as i32;
                if batch.add(new_token, n_cur, &[0], true).is_err() {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(
                            "Failed to add generated token to MTMD batch".to_string(),
                        )),
                    );
                    return;
                }
                if let Err(e) = ctx.decode(&mut batch) {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "MTMD decode failed: {e}"
                        ))),
                    );
                    return;
                }
            }

            send_completion_result(
                &tx,
                Ok(CompletionResponseChunk {
                    text: String::new(),
                    done: true,
                    prompt_tokens: Some(prompt_token_count),
                    done_reason: Some("length".to_string()),
                    prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                    token_id: None,
                }),
            );
            return;
        }

        // ----------------------------------------------------------------
        // Text-only path (original implementation)
        // ----------------------------------------------------------------

        // Tokenize the prompt
        let tokens =
            match model_arc.str_to_token(&request.prompt, llama_cpp_2::model::AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "Tokenization failed: {e}"
                        ))),
                    );
                    return;
                }
            };

        let prompt_token_count = tokens.len() as u32;

        match speculative_strategy {
            crate::speculative::SpeculativeStrategy::Mtp => {
                if let Err(error) = run_mtp_completion(
                    &model_arc,
                    &speculative_model_name,
                    tokens,
                    context_settings,
                    &sampling_settings,
                    speculative_settings,
                    &speculative_telemetry,
                    &tx,
                ) {
                    send_completion_result(&tx, Err(error));
                }
                return;
            }
            crate::speculative::SpeculativeStrategy::Dflash
            | crate::speculative::SpeculativeStrategy::Dflash2
            | crate::speculative::SpeculativeStrategy::Dspark => {
                let Some((draft_model, identity)) = external_draft.as_ref() else {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "Verified {} draft disappeared after capability negotiation",
                            speculative_strategy.as_str()
                        ))),
                    );
                    return;
                };
                if external_draft_strategy(identity.kind) != speculative_strategy {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "External draft kind '{}' does not match negotiated strategy '{}'",
                            identity.kind.as_str(),
                            speculative_strategy.as_str()
                        ))),
                    );
                    return;
                }
                if let Err(error) = run_external_draft_completion(
                    &model_arc,
                    &speculative_model_name,
                    draft_model,
                    identity.kind,
                    tokens,
                    context_settings,
                    &sampling_settings,
                    speculative_settings,
                    &speculative_telemetry,
                    &tx,
                ) {
                    send_completion_result(&tx, Err(error));
                }
                return;
            }
            _ => {}
        }

        // Try to reuse cached context with KV cache prefix matching.
        // Only reuse if the request carries a session_id — anonymous requests
        // always get a fresh context to prevent cross-request cache leakage.
        let cached = match session_id.as_deref() {
            Some(sid) => match lock_session_cache(&session_cache) {
                Ok(mut cache) => cache.take(sid),
                Err(e) => {
                    send_completion_result(&tx, Err(e));
                    return;
                }
            },
            None => None,
        };
        let (mut ctx, skip_tokens) = match cached {
            Some(mut cached) if cached.ctx_size == ctx_size => {
                // Find common prefix between cached tokens and new tokens
                let (common_len, mut reusable_len) =
                    matched_and_reusable_prompt_prefix_len(&cached.evaluated_tokens, &tokens);

                if common_len > 0 && common_len <= tokens.len() {
                    // Remove KV cache entries after the common prefix
                    if reusable_len < cached.evaluated_tokens.len() {
                        let truncated =
                            cached
                                .ctx
                                .clear_kv_cache_seq(Some(0), Some(reusable_len as u32), None);
                        if !matches!(&truncated, Ok(true)) {
                            tracing::debug!(
                                result = ?truncated,
                                "llama.cpp cannot roll this cached state back partially; using an exact cache miss"
                            );
                            cached.ctx.clear_kv_cache();
                            reusable_len = 0;
                        }
                    }
                    tracing::debug!(
                        matched = common_len,
                        reused = reusable_len,
                        total = tokens.len(),
                        "Reusing KV cache prefix"
                    );
                    (cached.ctx, reusable_len)
                } else {
                    // No useful prefix — clear and reuse the context
                    cached.ctx.clear_kv_cache();
                    (cached.ctx, 0)
                }
            }
            _ => {
                // No cached context or size mismatch — create new
                let cache_rollback_snapshots = u32::from(session_id.is_some());
                let ctx_params = context_settings.params(
                    llama_cpp_2::context::params::LlamaContextType::Default,
                    cache_rollback_snapshots,
                    cache_rollback_snapshots.saturating_add(1),
                    1,
                );
                let context_result = if plain_backend_greedy {
                    let backend_sampler =
                        match build_llamacpp_sampler(&model_arc, &sampling_settings) {
                            Ok(sampler) => sampler,
                            Err(error) => {
                                send_completion_result(&tx, Err(error));
                                return;
                            }
                        };
                    model_arc.new_context_with_samplers(
                        backend_ref(),
                        ctx_params,
                        [(0, backend_sampler)],
                    )
                } else {
                    model_arc.new_context(backend_ref(), ctx_params)
                };
                match context_result {
                    Ok(c) => {
                        // Safety: model_arc is an Arc kept alive in LoadedModel for the
                        // entire duration the context exists. The context is returned to
                        // CachedContext (which stores LlamaContext<'static>) and is always
                        // dropped before the model.
                        let c: llama_cpp_2::context::LlamaContext<'static> =
                            unsafe { std::mem::transmute(c) };
                        (c, 0)
                    }
                    Err(e) => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "Failed to create context: {e}"
                            ))),
                        );
                        return;
                    }
                }
            }
        };
        if session_id.is_some() {
            prompt_cache_telemetry
                .record_lookup(skip_tokens, tokens.len().saturating_sub(skip_tokens));
        }

        // Only evaluate tokens not already in the KV cache
        let tokens_to_eval = &tokens[skip_tokens..];
        let prompt_eval_start = std::time::Instant::now();

        // Apply LoRA adapter to context if available
        if let Some(ref adapter_arc) = lora_adapter {
            let mut wrapper = match lock_lora_adapter(adapter_arc) {
                Ok(wrapper) => wrapper,
                Err(e) => {
                    send_completion_result(&tx, Err(e));
                    return;
                }
            };
            if let Err(e) = ctx.lora_adapter_set(&mut wrapper.0, 1.0) {
                send_completion_result(
                    &tx,
                    Err(PowerError::InferenceFailed(format!(
                        "Failed to apply LoRA adapter: {e}"
                    ))),
                );
                return;
            }
        }

        if !tokens_to_eval.is_empty() {
            // Respect the context's bounded decode batch during prompt
            // prefill. In particular, speculative A/B runs intentionally
            // use a small n_batch so llama.cpp allocates only the target
            // verification logits rows instead of its much larger default.
            let prompt_batch_size = usize::try_from(ctx.n_batch()).unwrap_or(usize::MAX).max(1);
            for (chunk_index, chunk) in tokens_to_eval.chunks(prompt_batch_size).enumerate() {
                let chunk_offset = chunk_index.saturating_mul(prompt_batch_size);
                let mut batch = LlamaBatch::new(chunk.len().max(1), 1);
                for (index, &token) in chunk.iter().enumerate() {
                    let absolute = skip_tokens
                        .saturating_add(chunk_offset)
                        .saturating_add(index);
                    let position = match i32::try_from(absolute) {
                        Ok(position) => position,
                        Err(_) => {
                            send_completion_result(
                                &tx,
                                Err(PowerError::InferenceFailed(
                                    "Prompt position exceeds llama.cpp limits".to_string(),
                                )),
                            );
                            return;
                        }
                    };
                    if batch
                        .add(token, position, &[0], absolute + 1 == tokens.len())
                        .is_err()
                    {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(
                                "Failed to add token to prompt batch".to_string(),
                            )),
                        );
                        return;
                    }
                }

                if let Err(error) = ctx.decode(&mut batch) {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "Prompt decode failed: {error}"
                        ))),
                    );
                    return;
                }
            }
        }
        let prompt_eval_duration_ns = prompt_eval_start.elapsed().as_nanos() as u64;

        // Preserve recurrent/SWA state at the prompt boundary on the same
        // device. Generation may advance the context; the boundary is
        // restored before the reusable context returns to the cache.
        let prompt_boundary_checkpoint = if session_id.is_some() {
            let flags = llama_cpp_2::LlamaStateSeqFlags::PARTIAL_ONLY
                | llama_cpp_2::LlamaStateSeqFlags::ON_DEVICE;
            match ctx.state_seq_get(0, flags) {
                Ok(checkpoint) => Some(checkpoint),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "llama.cpp: failed to capture prompt-boundary recurrent state"
                    );
                    None
                }
            }
        } else {
            None
        };

        let mut sampler = match build_llamacpp_sampler(&model_arc, &sampling_settings) {
            Ok(sampler) => sampler,
            Err(error) => {
                send_completion_result(&tx, Err(error));
                return;
            }
        };

        let eos_token = model_arc.token_eos();
        let mut generated_text = String::new();
        let prompt_tokens_for_cache = tokens.clone();

        // Generate tokens
        for generated_count in 0..max_tokens {
            let new_token = sample_target_token(&mut sampler, &ctx, -1, plain_backend_greedy);

            if new_token == eos_token {
                send_completion_result(
                    &tx,
                    Ok(CompletionResponseChunk {
                        text: String::new(),
                        done: true,
                        prompt_tokens: Some(prompt_token_count),
                        done_reason: Some("stop".to_string()),
                        prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                        token_id: None,
                    }),
                );
                // Return context to session cache (only when session_id is set).
                if let Some(ref sid) = session_id {
                    cache_prompt_boundary_context(
                        &session_cache_for_return,
                        sid,
                        ctx,
                        prompt_tokens_for_cache,
                        ctx_size,
                        prompt_boundary_checkpoint.as_ref(),
                    );
                }
                return;
            }

            let text = {
                let mut decoder = encoding_rs::UTF_8.new_decoder();
                model_arc
                    .token_to_piece(new_token, &mut decoder, true, None)
                    .unwrap_or_default()
            };

            generated_text.push_str(&text);

            // Check stop sequences
            let mut should_stop = false;
            for stop in &stop_sequences {
                if generated_text.ends_with(stop) {
                    should_stop = true;
                    break;
                }
            }

            if !send_completion_result(
                &tx,
                Ok(CompletionResponseChunk {
                    text,
                    done: should_stop,
                    prompt_tokens: if should_stop {
                        Some(prompt_token_count)
                    } else {
                        None
                    },
                    done_reason: if should_stop {
                        Some("stop".to_string())
                    } else {
                        None
                    },
                    prompt_eval_duration_ns: if should_stop {
                        Some(prompt_eval_duration_ns)
                    } else {
                        None
                    },
                    token_id: Some(new_token.0 as u32),
                }),
            ) {
                // Receiver dropped — cache context if session is set
                if let Some(ref sid) = session_id {
                    cache_prompt_boundary_context(
                        &session_cache_for_return,
                        sid,
                        ctx,
                        prompt_tokens_for_cache,
                        ctx_size,
                        prompt_boundary_checkpoint.as_ref(),
                    );
                }
                return;
            }

            if should_stop {
                // Return context to session cache
                if let Some(ref sid) = session_id {
                    cache_prompt_boundary_context(
                        &session_cache_for_return,
                        sid,
                        ctx,
                        prompt_tokens_for_cache,
                        ctx_size,
                        prompt_boundary_checkpoint.as_ref(),
                    );
                }
                return;
            }

            // Prepare next batch
            let mut batch = LlamaBatch::new(1, 1);
            let n_cur = tokens.len() + generated_count;
            if batch.add(new_token, n_cur as i32, &[0], true).is_err() {
                send_completion_result(
                    &tx,
                    Err(PowerError::InferenceFailed(
                        "Failed to add generated token to batch".to_string(),
                    )),
                );
                return;
            }

            if let Err(e) = ctx.decode(&mut batch) {
                send_completion_result(
                    &tx,
                    Err(PowerError::InferenceFailed(format!("Decode failed: {e}"))),
                );
                return;
            }
        }

        // Max tokens reached — cache context if session is set
        send_completion_result(
            &tx,
            Ok(CompletionResponseChunk {
                text: String::new(),
                done: true,
                prompt_tokens: Some(prompt_token_count),
                done_reason: Some("length".to_string()),
                prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                token_id: None,
            }),
        );
        if let Some(ref sid) = session_id {
            cache_prompt_boundary_context(
                &session_cache_for_return,
                sid,
                ctx,
                prompt_tokens_for_cache,
                ctx_size,
                prompt_boundary_checkpoint.as_ref(),
            );
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Box::pin(stream))
}
