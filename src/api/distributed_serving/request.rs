use crate::api::types::{ChatCompletionRequest, CompletionRequest as OpenAiCompletionRequest};
use crate::backend::types::{
    ChatMessage, ChatRequest, CompletionRequest as BackendCompletionRequest,
};
use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::serving::PhaseRequest;

use super::{PhaseRequestPayload, DISTRIBUTED_PROMPT_CACHE_KEY_PREFIX};

impl PhaseRequestPayload {
    /// Translate the public OpenAI shape at the presentation boundary. The
    /// serving domain receives only the existing process-local backend request.
    pub(crate) fn into_phase_request(self, config: &PowerConfig) -> Result<(String, PhaseRequest)> {
        match self {
            Self::ChatCompletions { body } => {
                let request: ChatCompletionRequest = decode_body(body, "chat completion")?;
                validate_chat(&request)?;
                let model = validate_model(&request.model)?;
                let effective_tools = request.effective_tools();
                let effective_tool_choice = request.effective_tool_choice();
                let effective_max_tokens = request.effective_max_tokens();
                let backend = ChatRequest {
                    messages: request
                        .messages
                        .into_iter()
                        .map(|message| ChatMessage {
                            role: message.role,
                            content: message.content,
                            name: message.name,
                            tool_calls: message.tool_calls,
                            tool_call_id: message.tool_call_id,
                            images: message.images,
                        })
                        .collect(),
                    session_id: request.prompt_cache_key,
                    temperature: request.temperature,
                    top_p: request.top_p,
                    max_tokens: effective_max_tokens,
                    stop: request.stop,
                    stream: request.stream.unwrap_or(false),
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
                    response_format: request
                        .response_format
                        .as_ref()
                        .map(crate::api::openai::openai_wire_response_format),
                    stream_options: request
                        .stream_options
                        .as_ref()
                        .map(|options| serde_json::json!(options)),
                    tools: effective_tools,
                    tool_choice: effective_tool_choice,
                    parallel_tool_calls: request.parallel_tool_calls,
                    repeat_last_n: request.repeat_last_n,
                    penalize_newline: request.penalize_newline,
                    num_batch: None,
                    num_thread: config.num_thread,
                    num_thread_batch: None,
                    flash_attention: config.flash_attention.then_some(true),
                    num_gpu: None,
                    main_gpu: None,
                    use_mmap: None,
                    use_mlock: config.use_mlock.then_some(true),
                    num_parallel: Some(config.num_parallel as u32),
                    images: None,
                };
                Ok((model, PhaseRequest::Chat(backend)))
            }
            Self::Completions { body } => {
                let request: OpenAiCompletionRequest = decode_body(body, "text completion")?;
                validate_completion(&request)?;
                let model = validate_model(&request.model)?;
                let backend = BackendCompletionRequest {
                    prompt: request.prompt,
                    session_id: request.prompt_cache_key,
                    temperature: request.temperature,
                    top_p: request.top_p,
                    max_tokens: request.max_tokens,
                    stop: request.stop,
                    stream: request.stream.unwrap_or(false),
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
                    response_format: request
                        .response_format
                        .as_ref()
                        .map(crate::api::openai::openai_wire_response_format),
                    stream_options: request
                        .stream_options
                        .as_ref()
                        .map(|options| serde_json::json!(options)),
                    images: None,
                    projector_path: None,
                    repeat_last_n: request.repeat_last_n,
                    penalize_newline: request.penalize_newline,
                    num_batch: request.num_batch,
                    num_thread: config.num_thread,
                    num_thread_batch: None,
                    flash_attention: config.flash_attention.then_some(true),
                    num_gpu: None,
                    main_gpu: None,
                    use_mmap: None,
                    use_mlock: config.use_mlock.then_some(true),
                    num_parallel: Some(config.num_parallel as u32),
                    suffix: None,
                    context: None,
                };
                Ok((model, PhaseRequest::Completion(backend)))
            }
        }
    }
}

fn decode_body<T: serde::de::DeserializeOwned>(
    body: serde_json::Value,
    endpoint: &str,
) -> Result<T> {
    serde_json::from_value(body).map_err(|_| {
        PowerError::InvalidRequest(format!(
            "distributed {endpoint} body does not match the closed OpenAI request contract"
        ))
    })
}

fn validate_model(model: &str) -> Result<String> {
    if model.is_empty()
        || model.len() > 256
        || model.trim() != model
        || model.chars().any(char::is_control)
    {
        return Err(PowerError::InvalidRequest(
            "distributed request model is invalid".to_string(),
        ));
    }
    Ok(model.to_string())
}

fn validate_chat(request: &ChatCompletionRequest) -> Result<()> {
    let invalid = request.unsupported_fields_message().is_some()
        || request.n.is_some_and(|choices| choices != 1)
        || request.has_conflicting_max_token_limits()
        || request.has_stream_options_without_stream()
        || request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.unsupported_fields_message().is_some())
        || request.logprobs.unwrap_or(false)
        || request.top_logprobs.is_some()
        || request.logit_bias.is_some()
        || request.has_unsupported_modalities()
        || request.audio.is_some()
        || request.prediction.is_some()
        || request.reasoning_effort.is_some()
        || request
            .response_format
            .as_ref()
            .is_some_and(|format| format.validation_error().is_some())
        || request.unsupported_message_fields_message().is_some()
        || request.has_thinking_inputs()
        || request.has_conflicting_tool_definitions()
        || request.has_conflicting_tool_choice()
        || request.unsupported_tool_fields_message().is_some()
        || request.unsupported_tool_choice_fields_message().is_some()
        || request.has_image_inputs()
        || request.keep_alive.is_some();
    validate_cache_key(request.prompt_cache_key.as_deref())?;
    if invalid {
        return Err(PowerError::InvalidRequest(
            "distributed chat request uses an unsupported field or profile".to_string(),
        ));
    }
    Ok(())
}

fn validate_completion(request: &OpenAiCompletionRequest) -> Result<()> {
    let invalid = request.unsupported_fields_message().is_some()
        || request.n.is_some_and(|choices| choices != 1)
        || request.best_of.is_some_and(|choices| choices != 1)
        || request.has_stream_options_without_stream()
        || request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.unsupported_fields_message().is_some())
        || request.suffix.is_some()
        || request.logprobs.is_some()
        || request.echo.unwrap_or(false)
        || request.logit_bias.is_some()
        || request
            .num_batch
            .is_some_and(|batch| batch == 0 || request.num_ctx.is_some_and(|ctx| batch > ctx))
        || request
            .response_format
            .as_ref()
            .is_some_and(|format| format.validation_error().is_some())
        || request.keep_alive.is_some();
    validate_cache_key(request.prompt_cache_key.as_deref())?;
    if invalid {
        return Err(PowerError::InvalidRequest(
            "distributed text request uses an unsupported field or profile".to_string(),
        ));
    }
    Ok(())
}

fn validate_cache_key(key: Option<&str>) -> Result<()> {
    crate::api::prompt_cache::validate_prompt_cache_key(key).map_err(|_| {
        PowerError::InvalidRequest("distributed prompt-cache key is invalid".to_string())
    })?;
    let Some(key) = key else {
        return Ok(());
    };
    let digest = key
        .strip_prefix(DISTRIBUTED_PROMPT_CACHE_KEY_PREFIX)
        .ok_or_else(|| {
            PowerError::InvalidRequest(
                "distributed prompt-cache key must be scoped by Gateway".to_string(),
            )
        })?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidRequest(
            "distributed prompt-cache key digest is invalid".to_string(),
        ));
    }
    Ok(())
}
