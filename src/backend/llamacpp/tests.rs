use super::*;

use crate::backend::types::{ChatMessage, ContentPart, ImageUrl, MessageContent};
use crate::backend::Backend;
use crate::model::manifest::ModelFormat;

fn test_config() -> Arc<PowerConfig> {
    Arc::new(PowerConfig::default())
}

fn test_chat_request() -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("describe this".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }],
        temperature: None,
        top_p: None,
        max_tokens: None,
        stop: None,
        stream: false,
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        num_ctx: None,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format: None,
        stream_options: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        repeat_last_n: None,
        penalize_newline: None,
        num_batch: None,
        num_thread: None,
        num_thread_batch: None,
        flash_attention: None,
        num_gpu: None,
        main_gpu: None,
        use_mmap: None,
        use_mlock: None,
        num_parallel: None,
        images: None,
        session_id: None,
    }
}

#[test]
fn test_new_creates_backend() {
    let backend = LlamaCppBackend::new(test_config());
    assert_eq!(backend.name(), "llama.cpp");
}

#[test]
fn test_supports_gguf() {
    let backend = LlamaCppBackend::new(test_config());
    assert!(backend.supports(&ModelFormat::Gguf));
}

#[test]
fn test_does_not_support_safetensors() {
    let backend = LlamaCppBackend::new(test_config());
    assert!(!backend.supports(&ModelFormat::SafeTensors));
}

#[test]
fn test_backend_stores_config() {
    let mut config = PowerConfig::default();
    config.gpu.gpu_layers = -1;
    let config = Arc::new(config);
    let backend = LlamaCppBackend::new(config.clone());
    assert_eq!(backend.config.gpu.gpu_layers, -1);
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_llamacpp_gpu_layers_preserves_power_semantics() {
    use llama_cpp_2::model::params::LlamaModelParams;

    let cpu_only = LlamaModelParams::default()
        .with_n_gpu_layers(llamacpp_gpu_layers(0).expect("zero should select CPU only"));
    assert_eq!(cpu_only.n_gpu_layers(), 0);

    let all_layers = LlamaModelParams::default()
        .with_n_gpu_layers(llamacpp_gpu_layers(-1).expect("minus one should offload all layers"));
    assert_eq!(all_layers.n_gpu_layers(), i32::MAX);

    let partial = LlamaModelParams::default()
        .with_n_gpu_layers(llamacpp_gpu_layers(17).expect("positive values should be exact"));
    assert_eq!(partial.n_gpu_layers(), 17);

    let error = llamacpp_gpu_layers(-2).expect_err("values below minus one are invalid");
    assert!(error.to_string().contains("gpu_layers"));
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_llamacpp_batch_is_bounded_by_the_effective_context() {
    assert_eq!(validated_llamacpp_batch(None, 2048).unwrap(), None);
    assert_eq!(validated_llamacpp_batch(Some(24), 4096).unwrap(), Some(24));
    assert!(validated_llamacpp_batch(Some(0), 4096).is_err());
    assert!(validated_llamacpp_batch(Some(4097), 4096).is_err());
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_llamacpp_mtp_weights_follow_runtime_mode() {
    assert!(llamacpp_loads_mtp_weights("mtp", false));
    assert!(llamacpp_loads_mtp_weights("auto", false));
    assert!(!llamacpp_loads_mtp_weights("auto", true));
    assert!(!llamacpp_loads_mtp_weights("off", false));
    assert!(!llamacpp_loads_mtp_weights("prompt-lookup", false));
}

#[cfg(feature = "llamacpp-external-draft")]
#[test]
fn test_external_draft_selection_is_typed_and_fail_closed() {
    use crate::model::external_draft::{ExternalDraftArtifact, ExternalDraftKind};

    let mut manifest = ModelManifest::remote("target");
    manifest.sha256 = "a".repeat(64);
    manifest.external_draft = Some(ExternalDraftArtifact {
        kind: ExternalDraftKind::Dspark,
        path: std::path::PathBuf::from("draft.gguf"),
        size: 42,
        sha256: "b".repeat(64),
        target_sha256: manifest.sha256.clone(),
        source: None,
        revision: None,
        license: None,
    });

    assert_eq!(
        selected_external_draft(&manifest, "dspark")
            .unwrap()
            .unwrap()
            .kind,
        ExternalDraftKind::Dspark
    );
    assert_eq!(
        selected_external_draft(&manifest, "auto")
            .unwrap()
            .unwrap()
            .kind,
        ExternalDraftKind::Dspark
    );
    assert!(selected_external_draft(&manifest, "off").unwrap().is_none());
    let error = selected_external_draft(&manifest, "dflash").unwrap_err();
    assert!(error.to_string().contains("found 'dspark'"));

    manifest.external_draft.as_mut().unwrap().kind = ExternalDraftKind::Dflash2;
    for mode in ["dflash2", "auto"] {
        assert_eq!(
            selected_external_draft(&manifest, mode)
                .unwrap()
                .unwrap()
                .kind,
            ExternalDraftKind::Dflash2
        );
    }
}

#[cfg(all(feature = "llamacpp", not(feature = "llamacpp-external-draft")))]
#[test]
fn test_external_draft_selection_requires_reviewed_binding_feature() {
    use crate::model::external_draft::{ExternalDraftArtifact, ExternalDraftKind};

    let mut manifest = ModelManifest::remote("target");
    manifest.sha256 = "a".repeat(64);
    manifest.external_draft = Some(ExternalDraftArtifact {
        kind: ExternalDraftKind::Dspark,
        path: std::path::PathBuf::from("draft.gguf"),
        size: 42,
        sha256: "b".repeat(64),
        target_sha256: manifest.sha256.clone(),
        source: None,
        revision: None,
        license: None,
    });

    assert!(selected_external_draft(&manifest, "auto")
        .unwrap()
        .is_none());
    let error = selected_external_draft(&manifest, "dspark").unwrap_err();
    assert!(error.to_string().contains("llamacpp-external-draft"));
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_exact_tensor_pattern_escapes_regex_metacharacters() {
    assert_eq!(
        exact_tensor_pattern("blk.3.attn_q.weight"),
        r"^blk\.3\.attn_q\.weight$"
    );
    assert_eq!(
        exact_tensor_pattern(r"tensor[0]+(draft)\\path"),
        r"^tensor\[0\]\+\(draft\)\\\\path$"
    );
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_llamacpp_raw_model_params_preserve_load_controls() {
    let params = llamacpp_model_params(u32::MAX, 2, true, true, true, true);

    assert_eq!(params.n_gpu_layers, i32::MAX);
    assert_eq!(params.main_gpu, 2);
    assert_eq!(
        params.load_mode,
        llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP_MLOCK
    );
    assert_eq!(params.split_mode, llama_cpp_sys_2::LLAMA_SPLIT_MODE_LAYER);
    assert!(params.load_mtp);

    let copied = llamacpp_model_params(u32::MAX, 0, false, false, false, false);
    assert_eq!(copied.load_mode, llama_cpp_sys_2::LLAMA_LOAD_MODE_NONE);
}

#[test]
fn test_collect_llamacpp_openai_images_accepts_data_uri() {
    let parts = vec![
        ContentPart::Text {
            text: "describe this".to_string(),
            unsupported: Default::default(),
        },
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "data:image/png;base64,aGVsbG8=".to_string(),
                detail: None,
                unsupported: Default::default(),
            },
            unsupported: Default::default(),
        },
    ];

    let images = collect_llamacpp_openai_images(0, &parts).unwrap();

    assert_eq!(images, vec!["aGVsbG8=".to_string()]);
}

#[test]
fn test_collect_llamacpp_openai_images_accepts_base64_data() {
    let parts = vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: " aGVsbG8= ".to_string(),
            detail: None,
            unsupported: Default::default(),
        },
        unsupported: Default::default(),
    }];

    let images = collect_llamacpp_openai_images(1, &parts).unwrap();

    assert_eq!(images, vec!["aGVsbG8=".to_string()]);
}

#[test]
fn test_collect_llamacpp_openai_images_rejects_remote_urls() {
    let parts = vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "https://example.com/image.png".to_string(),
            detail: None,
            unsupported: Default::default(),
        },
        unsupported: Default::default(),
    }];

    let err = collect_llamacpp_openai_images(2, &parts).unwrap_err();

    let msg = err.to_string();
    assert!(msg.contains("message 2"), "error: {msg}");
    assert!(msg.contains("part 0"), "error: {msg}");
    assert!(msg.contains("remote image URLs"), "error: {msg}");
}

#[test]
fn test_collect_llamacpp_openai_images_rejects_empty_data() {
    let parts = vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "data:image/png;base64,".to_string(),
            detail: None,
            unsupported: Default::default(),
        },
        unsupported: Default::default(),
    }];

    let err = collect_llamacpp_openai_images(3, &parts).unwrap_err();

    let msg = err.to_string();
    assert!(msg.contains("message 3"), "error: {msg}");
    assert!(msg.contains("empty image data"), "error: {msg}");
}

#[test]
fn test_collect_llamacpp_chat_images_combines_supported_sources() {
    let mut request = test_chat_request();
    request.messages[0].images = Some(vec!["message-base64-image".to_string()]);
    request.messages[0].content = MessageContent::Parts(vec![
        ContentPart::Text {
            text: "describe this".to_string(),
            unsupported: Default::default(),
        },
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "data:image/png;base64,part-base64-image".to_string(),
                detail: None,
                unsupported: Default::default(),
            },
            unsupported: Default::default(),
        },
    ]);
    request.images = Some(vec!["request-base64-image".to_string()]);

    let images = collect_llamacpp_chat_images(&request).unwrap();

    assert_eq!(
        images,
        vec![
            "message-base64-image".to_string(),
            "part-base64-image".to_string(),
            "request-base64-image".to_string(),
        ]
    );
}

#[cfg(feature = "llamacpp")]
#[tokio::test]
async fn test_effective_prompt_digest_absent_for_llamacpp_images() {
    let backend = LlamaCppBackend::new(test_config());
    let mut request = test_chat_request();
    request.images = Some(vec!["request-base64-image".to_string()]);

    let digest = backend
        .effective_chat_prompt_digest("not-loaded", &request)
        .await
        .unwrap();

    assert!(digest.is_none());
}

#[test]
fn test_ensure_llamacpp_images_supported_allows_text_only_without_projector() {
    assert!(ensure_llamacpp_images_supported("llama3", false, false).is_ok());
}

#[test]
fn test_ensure_llamacpp_images_supported_allows_images_with_projector() {
    assert!(ensure_llamacpp_images_supported("llava", true, true).is_ok());
}

#[test]
fn test_ensure_llamacpp_images_supported_rejects_images_without_projector() {
    let err = ensure_llamacpp_images_supported("llama3", true, false).unwrap_err();

    let msg = err.to_string();
    assert!(msg.contains("llama3"), "error: {msg}");
    assert!(msg.contains("multimodal projector"), "error: {msg}");
    assert!(
        msg.contains("image inputs cannot be processed"),
        "error: {msg}"
    );
}

#[cfg(feature = "llamacpp")]
fn test_completion_chunk(done: bool) -> CompletionResponseChunk {
    CompletionResponseChunk {
        text: String::new(),
        done,
        prompt_tokens: None,
        done_reason: None,
        prompt_eval_duration_ns: None,
        token_id: None,
    }
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_send_completion_result_sends_when_receiver_open() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);

    assert!(send_completion_result(&tx, Ok(test_completion_chunk(true))));

    let sent = rx.blocking_recv().unwrap().unwrap();
    assert!(sent.done);
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_send_completion_result_reports_closed_receiver() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    drop(rx);

    assert!(!send_completion_result(
        &tx,
        Ok(test_completion_chunk(false))
    ));
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_session_cache_lock_poison_returns_error() {
    let cache: SessionCache = Arc::new(Mutex::new(BoundedPromptCache::new(
        1,
        std::time::Duration::from_secs(300),
        Arc::new(PromptCacheTelemetry::default()),
    )));
    let poison_cache = Arc::clone(&cache);
    let _ = std::panic::catch_unwind(move || {
        let _guard = poison_cache.lock().unwrap();
        panic!("poison session cache");
    });

    let err = match lock_session_cache(&cache) {
        Ok(_) => panic!("expected poisoned session cache error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("session cache lock poisoned"));
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_reusable_prompt_prefix_keeps_fresh_logits_boundary() {
    assert_eq!(
        matched_and_reusable_prompt_prefix_len(&[1, 2, 3, 9], &[1, 2, 4]),
        (2, 2)
    );
    assert_eq!(
        matched_and_reusable_prompt_prefix_len(&[1, 2, 3, 9], &[1, 2, 3]),
        (3, 2)
    );
    assert_eq!(matched_and_reusable_prompt_prefix_len(&[1], &[1]), (1, 0));
    assert_eq!(
        matched_and_reusable_prompt_prefix_len(&[1, 2], &[3, 4]),
        (0, 0)
    );
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_collected_text_lock_recovers_from_poison() {
    let text = Arc::new(Mutex::new(String::from("prefix")));
    let poison_text = Arc::clone(&text);
    let _ = std::panic::catch_unwind(move || {
        let mut guard = poison_text.lock().unwrap();
        guard.push_str("-poisoned");
        panic!("poison collected text");
    });

    {
        let mut guard = lock_collected_text(text.as_ref());
        guard.push_str("-recovered");
    }

    assert_eq!(
        lock_collected_text(text.as_ref()).as_str(),
        "prefix-poisoned-recovered"
    );
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_nonzero_context_size_preserves_valid_value() {
    assert_eq!(nonzero_context_size(4096).get(), 4096);
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_nonzero_context_size_falls_back_for_zero() {
    assert_eq!(nonzero_context_size(0).get(), DEFAULT_CTX_SIZE);
}

#[cfg(not(feature = "llamacpp"))]
#[tokio::test]
async fn test_stub_load_returns_error() {
    use crate::model::manifest::ModelManifest;
    use std::path::PathBuf;

    let backend = LlamaCppBackend::new(test_config());
    let manifest = ModelManifest {
        name: "test".to_string(),
        format: ModelFormat::Gguf,
        size: 0,
        sha256: "abc".to_string(),
        parameters: None,
        created_at: chrono::Utc::now(),
        path: PathBuf::from("/tmp/test"),
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        adapter_artifact: None,
        external_draft: None,
        projector_path: None,
        projector_artifact: None,
        messages: vec![],
        family: None,
        families: None,
    };
    let result = backend.load(&manifest).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("llamacpp"));
}

#[cfg(not(feature = "llamacpp"))]
#[tokio::test]
async fn test_stub_chat_returns_error() {
    let backend = LlamaCppBackend::new(test_config());
    let request = ChatRequest {
        messages: vec![],
        temperature: None,
        top_p: None,
        max_tokens: None,
        stop: None,
        stream: false,
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        num_ctx: None,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format: None,
        stream_options: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        repeat_last_n: None,
        penalize_newline: None,
        num_batch: None,
        num_thread: None,
        num_thread_batch: None,
        flash_attention: None,
        num_gpu: None,
        main_gpu: None,
        use_mmap: None,
        use_mlock: None,
        num_parallel: None,
        images: None,
        session_id: None,
    };
    let result = backend.chat("test", request).await;
    assert!(result.is_err());
}

#[cfg(not(feature = "llamacpp"))]
#[tokio::test]
async fn test_stub_complete_returns_error() {
    let backend = LlamaCppBackend::new(test_config());
    let request = CompletionRequest {
        prompt: "test".to_string(),
        temperature: None,
        top_p: None,
        max_tokens: None,
        stop: None,
        stream: false,
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        num_ctx: None,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format: None,
        stream_options: None,
        images: None,
        projector_path: None,
        repeat_last_n: None,
        penalize_newline: None,
        num_batch: None,
        num_thread: None,
        num_thread_batch: None,
        flash_attention: None,
        num_gpu: None,
        main_gpu: None,
        use_mmap: None,
        use_mlock: None,
        num_parallel: None,
        suffix: None,
        context: None,
        session_id: None,
    };
    let result = backend.complete("test", request).await;
    assert!(result.is_err());
}

#[cfg(not(feature = "llamacpp"))]
#[tokio::test]
async fn test_stub_unload_succeeds() {
    let backend = LlamaCppBackend::new(test_config());
    let result = backend.unload("test").await;
    assert!(result.is_ok());
}

#[cfg(not(feature = "llamacpp"))]
#[tokio::test]
async fn test_stub_embed_returns_error() {
    let backend = LlamaCppBackend::new(test_config());
    let request = EmbeddingRequest {
        input: vec!["test".to_string()],
    };
    let result = backend.embed("test", request).await;
    assert!(result.is_err());
}

#[test]
fn test_backend_name() {
    let backend = LlamaCppBackend::new(test_config());
    assert_eq!(backend.name(), "llama.cpp");
}

#[test]
fn test_backend_does_not_support_unknown_format() {
    let backend = LlamaCppBackend::new(test_config());
    assert!(!backend.supports(&ModelFormat::SafeTensors));
}

#[test]
fn test_backend_config_gpu_layers_default() {
    let config = PowerConfig::default();
    let backend = LlamaCppBackend::new(Arc::new(config));
    assert_eq!(backend.config.gpu.gpu_layers, 0);
}

#[test]
fn test_default_ctx_size_is_2048() {
    // Matches Ollama's default to prevent OOM on resource-constrained machines.
    assert_eq!(DEFAULT_CTX_SIZE, 2048);
}

#[test]
fn test_default_ctx_size_less_than_large_model_ctx() {
    // Models like llama3.2 have n_ctx_train = 131072 (128K).
    // DEFAULT_CTX_SIZE must be much smaller to avoid OOM.
    const { assert!(DEFAULT_CTX_SIZE < 131072) };
    const { assert!(DEFAULT_CTX_SIZE <= 8192) };
}
