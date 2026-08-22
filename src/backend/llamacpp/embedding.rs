use super::*;

pub(super) async fn embed(
    backend: &LlamaCppBackend,
    model_name: &str,
    request: EmbeddingRequest,
) -> Result<EmbeddingResponse> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    // Check if model needs to be reloaded with embedding mode
    let needs_reload = {
        let models = backend.models.read().await;
        match models.get(model_name) {
            Some(m) => m.load_mode != LoadMode::Embedding,
            None => {
                return Err(PowerError::InferenceFailed(format!(
                    "Model '{model_name}' not loaded"
                )));
            }
        }
    };

    if needs_reload {
        let (path, chat_template, raw_template, lora_adapter, projector_path) = {
            let models = backend.models.read().await;
            let m = models.get(model_name).ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "Model '{model_name}' was unloaded during embed reload"
                ))
            })?;
            (
                m.path.clone(),
                m.chat_template.clone(),
                m.raw_template.clone(),
                m.lora_adapter.clone(),
                m.projector_path.clone(),
            )
        };

        tracing::info!(model = model_name, "Reloading model with embedding mode");

        let gpu_layers = llamacpp_gpu_layers(backend.config.gpu.gpu_layers)?;
        let main_gpu = backend.config.gpu.main_gpu;
        let use_mmap = backend.config.use_mmap;
        let use_mlock = backend.config.use_mlock;
        let has_tensor_split = !backend.config.gpu.tensor_split.is_empty();
        let cpu_tensors = backend.config.gpu.cpu_tensors.clone();
        let gpu_tensors = backend.config.gpu.gpu_tensors.clone();

        let path_clone = path.clone();
        let model = tokio::task::spawn_blocking(move || {
            let mut params = llamacpp_model_params(
                gpu_layers,
                main_gpu,
                use_mmap,
                use_mlock,
                has_tensor_split,
                false,
            );
            let _tensor_overrides = if cpu_tensors.is_empty() && gpu_tensors.is_empty() {
                None
            } else {
                let overrides = LlamaTensorOverrides::new(&cpu_tensors, &gpu_tensors, main_gpu)?;
                overrides.apply(&mut params);
                Some(overrides)
            };
            load_llamacpp_model(&path_clone, params).map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "Failed to reload model for embedding: {error}"
                ))
            })
        })
        .await
        .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

        let model_arc = Arc::new(model);
        let n_ctx_train = model_arc.n_ctx_train();
        let speculative_capabilities = llamacpp_speculative_capabilities(&model_arc, false, None);
        let name = model_name.to_string();
        backend.models.write().await.insert(
            name.clone(),
            LoadedModel {
                name,
                path,
                model: model_arc,
                chat_template,
                raw_template,
                load_mode: LoadMode::Embedding,
                n_ctx_train,
                speculative_capabilities,
                external_draft: None,
                session_cache: backend.new_session_cache(),
                lora_adapter,
                projector_path,
                mtmd_ctx: None, // Embedding models don't use multimodal projectors
            },
        );
    }

    let model_arc = {
        let models = backend.models.read().await;
        models
            .get(model_name)
            .ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "Model '{model_name}' was unloaded during embed"
                ))
            })?
            .model
            .clone()
    };

    let input = request.input.clone();

    tokio::task::spawn_blocking(move || {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048))
            .with_embeddings(true);
        let mut ctx = model_arc
            .new_context(backend_ref(), ctx_params)
            .map_err(|e| PowerError::InferenceFailed(format!("Failed to create context: {e}")))?;

        let mut embeddings = Vec::with_capacity(input.len());

        for text in &input {
            let tokens = model_arc
                .str_to_token(text, llama_cpp_2::model::AddBos::Always)
                .map_err(|e| PowerError::InferenceFailed(format!("Tokenization failed: {e}")))?;

            let mut batch = LlamaBatch::new(2048, 1);
            for (i, &token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                batch.add(token, i as i32, &[0], is_last).map_err(|_| {
                    PowerError::InferenceFailed("Failed to add token to batch".to_string())
                })?;
            }

            ctx.decode(&mut batch)
                .map_err(|e| PowerError::InferenceFailed(format!("Decode failed: {e}")))?;

            let emb = ctx.embeddings_seq_ith(0).map_err(|e| {
                PowerError::InferenceFailed(format!("Failed to get embeddings: {e}"))
            })?;
            embeddings.push(emb.to_vec());

            ctx.clear_kv_cache();
        }

        Ok(EmbeddingResponse { embeddings })
    })
    .await
    .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))?
}
