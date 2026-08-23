use super::*;

pub(super) async fn load(backend: &LlamaCppBackend, manifest: &ModelManifest) -> Result<()> {
    use llama_cpp_2::llama_backend::LlamaBackend;

    tracing::info!(model = %manifest.name, path = %manifest.path.display(), "Loading model");

    // Ensure the backend is initialized (first call wins, subsequent calls are no-ops).
    // We pre-check initialization to avoid panicking inside get_or_init.
    if backend.llama_backend.get().is_none() {
        let initialized_backend = LlamaBackend::init().map_err(|e| {
            PowerError::InferenceFailed(format!("Failed to initialize llama.cpp backend: {e}"))
        })?;
        // Route native logs through Power's tracing filter. CUDA graph
        // reuse and backend-sampler DEBUG messages must not perform
        // synchronous stderr I/O in the token hot path of a release server.
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
        let _ = backend.llama_backend.set(initialized_backend); // Ignore if another thread won the race
    }

    let gpu_layers = llamacpp_gpu_layers(backend.config.gpu.gpu_layers)?;
    let main_gpu = backend.config.gpu.main_gpu;
    let use_mmap = backend.config.use_mmap;
    let use_mlock = backend.config.use_mlock;
    let has_tensor_split = !backend.config.gpu.tensor_split.is_empty();
    let cpu_tensors = backend.config.gpu.cpu_tensors.clone();
    let gpu_tensors = backend.config.gpu.gpu_tensors.clone();
    let external_artifact = selected_external_draft(manifest, &backend.config.spec_mode)?;
    if external_artifact.is_some() && backend.config.spec_mtp_fr_vocab_size.is_some() {
        return Err(PowerError::Config(
            "spec_mtp_fr_vocab_size applies only to MTP and cannot be combined with an external draft"
                .to_string(),
        ));
    }
    let load_mtp =
        llamacpp_loads_mtp_weights(&backend.config.spec_mode, external_artifact.is_some());

    let path = manifest.path.clone();
    let target_size = manifest.size;
    let target_sha256 = manifest.sha256.clone();
    let model_name = manifest.name.clone();

    // Load the model in a blocking task since it is CPU-intensive. The raw
    // parameter value contains pointers and is built inside the task rather
    // than crossing the async scheduler boundary.
    let (model, external_draft) = tokio::task::spawn_blocking(move || {
        let verified_external = external_artifact
            .as_ref()
            .map(|artifact| artifact.verify_for_target_file(&path, target_size, &target_sha256))
            .transpose()?;
        let mut p = llamacpp_model_params(
            gpu_layers,
            main_gpu,
            use_mmap,
            use_mlock,
            has_tensor_split,
            load_mtp,
        );
        let tensor_overrides = if cpu_tensors.is_empty() && gpu_tensors.is_empty() {
            None
        } else {
            let overrides = LlamaTensorOverrides::new(&cpu_tensors, &gpu_tensors, main_gpu)?;
            overrides.apply(&mut p);
            Some(overrides)
        };
        if has_tensor_split {
            tracing::info!("Multi-GPU layer splitting enabled");
        }
        tracing::info!(
            load_mtp,
            cpu_tensors = tensor_overrides.as_ref().map_or(0, |_| cpu_tensors.len()),
            gpu_tensors = tensor_overrides.as_ref().map_or(0, |_| gpu_tensors.len()),
            "Configured llama.cpp model tensor loading"
        );

        let model = load_llamacpp_model(&path, p)?;
        let external = if let Some(identity) = verified_external {
            // Tensor-name overrides describe the target model and must not
            // leak into a separately trained draft artifact.
            let draft_params = llamacpp_model_params(
                gpu_layers,
                main_gpu,
                use_mmap,
                use_mlock,
                has_tensor_split,
                false,
            );
            let draft_model = load_llamacpp_model(&identity.path, draft_params)?;
            Some((draft_model, identity))
        } else {
            None
        };
        Ok::<_, PowerError>((model, external))
    })
    .await
    .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

    let model_arc = Arc::new(model);
    let external_draft = external_draft.map(|(model, identity)| LoadedExternalDraft {
        model: Arc::new(model),
        identity,
    });

    // Detect chat template: prefer manifest.template_override (from Ollama registry),
    // then GGUF metadata, then fallback to Phi.
    let gguf_template = model_arc.meta_val_str("tokenizer.chat_template").ok();

    let raw_template_str = manifest.template_override.clone().or(gguf_template);

    let chat_template = raw_template_str
        .as_deref()
        .map(chat_template::detect)
        .unwrap_or(ChatTemplateKind::Phi);

    // Read trained context length from model metadata
    let n_ctx_train = model_arc.n_ctx_train();
    tracing::info!(model = %manifest.name, n_ctx_train = n_ctx_train, "Model context window detected");
    let speculative_capabilities = llamacpp_speculative_capabilities(
        &model_arc,
        load_mtp,
        external_draft.as_ref().map(|draft| draft.identity.kind),
    );
    tracing::info!(
        model = %manifest.name,
        mtp = speculative_capabilities
            .supports(crate::speculative::SpeculativeStrategy::Mtp),
        dflash = speculative_capabilities
            .supports(crate::speculative::SpeculativeStrategy::Dflash),
        dflash2 = speculative_capabilities
            .supports(crate::speculative::SpeculativeStrategy::Dflash2),
        dspark = speculative_capabilities
            .supports(crate::speculative::SpeculativeStrategy::Dspark),
        external_draft_sha256 = external_draft
            .as_ref()
            .map(|draft| draft.identity.sha256.as_str()),
        "Model speculative capabilities detected"
    );

    // Load LoRA adapter if specified in manifest
    let lora_adapter = if let Some(ref adapter_path) = manifest.adapter_path {
        let adapter_path_buf = std::path::PathBuf::from(adapter_path);
        if adapter_path_buf.exists() {
            let model_ref = model_arc.clone();
            let path = adapter_path_buf.clone();
            let adapter = tokio::task::spawn_blocking(move || {
                let adapter = model_ref.lora_adapter_init(&path).map_err(|e| {
                    PowerError::InferenceFailed(format!(
                        "Failed to load LoRA adapter from {}: {e}",
                        path.display()
                    ))
                })?;
                // Wrap immediately inside spawn_blocking so we never send
                // the raw LlamaLoraAdapter across threads.
                Ok::<_, PowerError>(SendableLoraAdapter(adapter))
            })
            .await
            .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

            tracing::info!(
                model = %manifest.name,
                adapter = %adapter_path,
                "LoRA adapter loaded"
            );
            Some(Arc::new(Mutex::new(adapter)))
        } else {
            tracing::warn!(
                model = %manifest.name,
                adapter = %adapter_path,
                "LoRA adapter file not found, skipping"
            );
            None
        }
    } else {
        None
    };

    // Initialize multimodal context if projector_path is set.
    // MtmdContext::init_from_file is blocking (loads the projector weights).
    let mtmd_ctx = if let Some(ref proj_path) = manifest.projector_path {
        let proj_path_str = proj_path.clone();
        let model_ref = model_arc.clone();
        let model_name_for_log = manifest.name.clone();
        match tokio::task::spawn_blocking(move || {
            use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};
            let params = MtmdContextParams::default();
            MtmdContext::init_from_file(&proj_path_str, &model_ref, &params)
                .map(SendableMtmdContext)
                .map_err(|e| {
                    PowerError::InferenceFailed(format!(
                        "Failed to initialize MTMD context from {proj_path_str}: {e}"
                    ))
                })
        })
        .await
        .map_err(|e| PowerError::InferenceFailed(format!("MTMD init task failed: {e}")))
        {
            Ok(Ok(ctx)) => {
                tracing::info!(
                    model = %model_name_for_log,
                    projector = %proj_path,
                    "Multimodal projector loaded"
                );
                Some(Arc::new(Mutex::new(ctx)))
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    model = %manifest.name,
                    projector = %proj_path,
                    error = %e,
                    "Failed to load multimodal projector, vision inference disabled"
                );
                None
            }
            Err(e) => {
                tracing::warn!(error = %e, "MTMD init task panicked");
                None
            }
        }
    } else {
        None
    };

    backend.models.write().await.insert(
        model_name.clone(),
        LoadedModel {
            name: model_name.clone(),
            path: manifest.path.clone(),
            model: model_arc,
            chat_template,
            raw_template: raw_template_str,
            load_mode: LoadMode::Inference,
            n_ctx_train,
            speculative_capabilities,
            external_draft,
            session_cache: backend.new_session_cache(),
            lora_adapter,
            projector_path: manifest.projector_path.clone(),
            mtmd_ctx,
        },
    );

    tracing::info!(model = %manifest.name, "Model loaded successfully");
    Ok(())
}

pub(super) async fn unload(backend: &LlamaCppBackend, model_name: &str) -> Result<()> {
    if backend.models.write().await.remove(model_name).is_some() {
        tracing::info!(model = model_name, "Model unloaded");
    }
    Ok(())
}
