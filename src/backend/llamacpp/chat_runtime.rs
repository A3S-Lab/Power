use super::*;

pub(super) async fn chat(
    backend: &LlamaCppBackend,
    model_name: &str,
    request: ChatRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
    // Look up the chat template and projector path for this model
    let (template, raw_template, projector_path, _model_n_ctx_train) = {
        let models = backend.models.read().await;
        let model = models.get(model_name).ok_or_else(|| {
            PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
        })?;
        (
            model.chat_template.clone(),
            model.raw_template.clone(),
            model.projector_path.clone(),
            model.n_ctx_train,
        )
    };

    // Render chat template in a blocking task to avoid blocking the async executor.
    // Some GGUF models carry complex Jinja2 templates that can be slow to render.
    let messages_clone = request.messages.clone();
    let raw_template_clone = raw_template.clone();
    let template_clone = template.clone();
    let prompt = tokio::task::spawn_blocking(move || {
        chat_template::format_chat_prompt(
            &messages_clone,
            &template_clone,
            raw_template_clone.as_deref(),
        )
    })
    .await
    .map_err(|e| PowerError::InferenceFailed(format!("Chat template rendering task failed: {e}")))?
    .map_err(PowerError::InferenceFailed)?;

    let has_images = request.has_image_inputs();
    ensure_llamacpp_images_supported(model_name, has_images, projector_path.is_some())?;
    if has_images {
        tracing::info!("Vision inference with multimodal projector");
    }

    let images = if has_images {
        collect_llamacpp_chat_images(&request)?
    } else {
        Vec::new()
    };

    let completion_req = CompletionRequest {
        prompt,
        session_id: request.session_id,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop: request.stop,
        stream: request.stream,
        top_k: request.top_k,
        min_p: request.min_p,
        repeat_penalty: request.repeat_penalty,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        seed: request.seed,
        num_ctx: request.num_ctx,
        mirostat: request.mirostat,
        mirostat_tau: request.mirostat_tau,
        mirostat_eta: request.mirostat_eta,
        tfs_z: request.tfs_z,
        typical_p: request.typical_p,
        response_format: request.response_format,
        stream_options: request.stream_options,
        images: if images.is_empty() {
            None
        } else {
            Some(images)
        },
        projector_path,
        repeat_last_n: request.repeat_last_n,
        penalize_newline: request.penalize_newline,
        num_batch: request.num_batch,
        num_thread: request.num_thread,
        num_thread_batch: request.num_thread_batch,
        flash_attention: request.flash_attention,
        num_gpu: request.num_gpu,
        main_gpu: request.main_gpu,
        use_mmap: request.use_mmap,
        use_mlock: request.use_mlock,
        num_parallel: request.num_parallel,
        suffix: None,
        context: None,
    };

    // Get completion stream from the underlying complete() method
    let stream = backend.complete(model_name, completion_req).await?;

    // Map CompletionResponseChunk -> ChatResponseChunk with tool call and think block detection
    use futures::StreamExt;
    let collected_text = Arc::new(Mutex::new(String::new()));
    let text_clone = collected_text.clone();
    let has_tools = request.tools.is_some();
    let mut think_parser = super::super::think_parser::ThinkBlockParser::new();
    let chat_stream = stream.map(move |chunk_result| {
        chunk_result.map(|chunk| {
            // Accumulate text for tool call detection
            if has_tools && !chunk.text.is_empty() {
                let mut text = lock_collected_text(text_clone.as_ref());
                text.push_str(&chunk.text);
            }

            // Parse think blocks from the token stream
            let (content, thinking) = if chunk.done {
                let (mut c, mut t) = think_parser.flush();
                // Prepend any remaining text from the final chunk
                if !chunk.text.is_empty() {
                    let (fc, ft) = think_parser.feed(&chunk.text);
                    c = fc + &c;
                    t = ft + &t;
                }
                (c, t)
            } else {
                think_parser.feed(&chunk.text)
            };

            let thinking_content = if thinking.is_empty() {
                None
            } else {
                Some(thinking)
            };

            // On the final chunk, try to parse tool calls from accumulated text
            let tool_calls = if chunk.done && has_tools {
                let full_text = lock_collected_text(text_clone.as_ref());
                super::super::tool_parser::parse_tool_calls(&full_text)
            } else {
                None
            };

            ChatResponseChunk {
                content,
                thinking_content,
                done: chunk.done,
                prompt_tokens: chunk.prompt_tokens,
                done_reason: if tool_calls.is_some() && chunk.done {
                    Some("tool_calls".to_string())
                } else {
                    chunk.done_reason
                },
                prompt_eval_duration_ns: chunk.prompt_eval_duration_ns,
                tool_calls,
            }
        })
    });

    Ok(Box::pin(chat_stream))
}

pub(super) async fn effective_chat_prompt_digest(
    backend: &LlamaCppBackend,
    model_name: &str,
    request: &ChatRequest,
) -> Result<Option<EffectivePromptDigest>> {
    if request.has_image_inputs() {
        return Ok(None);
    }

    let (template, raw_template) = {
        let models = backend.models.read().await;
        let model = models.get(model_name).ok_or_else(|| {
            PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
        })?;
        (model.chat_template.clone(), model.raw_template.clone())
    };

    let messages = request.messages.clone();
    let prompt = tokio::task::spawn_blocking(move || {
        chat_template::format_chat_prompt(&messages, &template, raw_template.as_deref())
    })
    .await
    .map_err(|e| PowerError::InferenceFailed(format!("Chat template rendering task failed: {e}")))?
    .map_err(PowerError::InferenceFailed)?;

    Ok(Some(EffectivePromptDigest::chat_rendered_prompt(
        "llama.cpp",
        &prompt,
    )))
}

pub(super) async fn effective_completion_prompt_digest(
    _backend: &LlamaCppBackend,
    _model_name: &str,
    request: &CompletionRequest,
) -> Result<Option<EffectivePromptDigest>> {
    if request
        .images
        .as_ref()
        .is_some_and(|images| !images.is_empty())
    {
        return Ok(None);
    }

    Ok(Some(EffectivePromptDigest::text_prompt(
        "llama.cpp",
        &request.prompt,
    )))
}
