use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::num::NonZeroU32;

use magnitude_chat::request::{
    AssistantTurn, Conversation, Entry, GenerationControls, OutputFormat, PromptCache,
    SamplingControls, ToolCall as MagnitudeToolCall, ToolDefinition, ToolExchange, ToolResultPart,
    Tools, UserPart,
};
use magnitude_chat::schema::JsonSchema;
use magnitude_chat::{
    ChatInput, EndOfGeneration, GenerationRequest, ReasoningIntent,
    ToolChoice as MagnitudeToolChoice,
};
use serde_json::{Map, Value};

use super::super::types::{ChatMessage, ChatRequest, ContentPart, MessageContent, ToolChoice};
use crate::error::{PowerError, Result};

pub(super) fn generation_request(request: &ChatRequest) -> Result<GenerationRequest> {
    validate_request(request)?;
    Ok(GenerationRequest {
        input: ChatInput {
            conversation: conversation(&request.messages)?,
            tools: tools(request)?,
            reasoning: ReasoningIntent::ModelDefault,
            output: output_format(request.response_format.as_ref())?,
            template_arguments: Map::new(),
        },
        controls: GenerationControls {
            max_output_tokens: request
                .max_tokens
                .map(|value| {
                    NonZeroU32::new(value).ok_or_else(|| {
                        PowerError::InvalidRequest("max_tokens must be positive".into())
                    })
                })
                .transpose()?,
            sampling: sampling(request)?,
            stops: request.stop.clone().unwrap_or_default(),
            end_of_generation: EndOfGeneration::Stop,
            reasoning_budget: None,
            prompt_cache: PromptCache::Disabled,
        },
    })
}

fn validate_request(request: &ChatRequest) -> Result<()> {
    if request.session_id.is_some() {
        return unsupported("prompt_cache_key requires namespaced prefix caching");
    }
    if request.has_image_inputs() {
        return unsupported("image input is not implemented by the Magnitude adapter");
    }
    let unsupported_fields = [
        (request.num_ctx.is_some(), "num_ctx"),
        (request.mirostat.is_some(), "mirostat"),
        (request.mirostat_tau.is_some(), "mirostat_tau"),
        (request.mirostat_eta.is_some(), "mirostat_eta"),
        (request.tfs_z.is_some(), "tfs_z"),
        (request.typical_p.is_some(), "typical_p"),
        (request.repeat_last_n.is_some(), "repeat_last_n"),
        (request.penalize_newline.is_some(), "penalize_newline"),
        (request.num_batch.is_some(), "num_batch"),
        (request.num_thread.is_some(), "num_thread"),
        (request.num_thread_batch.is_some(), "num_thread_batch"),
        (request.num_gpu.is_some(), "num_gpu"),
        (request.main_gpu.is_some(), "main_gpu"),
        (request.use_mmap.is_some(), "use_mmap"),
        (request.use_mlock.is_some(), "use_mlock"),
    ];
    if let Some((_, field)) = unsupported_fields.into_iter().find(|(present, _)| *present) {
        return unsupported(&format!(
            "{field} cannot be represented by the Magnitude engine"
        ));
    }
    if request
        .stop
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(String::is_empty)
    {
        return Err(PowerError::InvalidRequest(
            "stop sequences must not be empty".into(),
        ));
    }
    Ok(())
}

fn sampling(request: &ChatRequest) -> Result<SamplingControls> {
    let temperature = request.temperature.unwrap_or(1.0);
    if !finite_in(temperature, 0.0..=2.0) {
        return invalid("temperature must be finite and between 0 and 2");
    }
    let top_p = request.top_p.unwrap_or(0.95);
    if !finite_in(top_p, 0.0..=1.0) || top_p == 0.0 {
        return invalid("top_p must be finite, greater than 0 and at most 1");
    }
    let top_k = request.top_k.unwrap_or(20);
    if !(0..=1 << 24).contains(&top_k) {
        return invalid("top_k must be between 0 and 16777216");
    }
    let min_p = request.min_p.unwrap_or(0.0);
    if !finite_in(min_p, 0.0..=1.0) {
        return invalid("min_p must be finite and between 0 and 1");
    }
    let repetition_penalty = request.repeat_penalty.unwrap_or(1.0);
    if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
        return invalid("repeat_penalty must be finite and positive");
    }
    let presence_penalty = request.presence_penalty.unwrap_or(0.0);
    let frequency_penalty = request.frequency_penalty.unwrap_or(0.0);
    if !finite_in(presence_penalty, -2.0..=2.0) || !finite_in(frequency_penalty, -2.0..=2.0) {
        return invalid("presence_penalty and frequency_penalty must be between -2 and 2");
    }
    let seed = match request.seed {
        Some(seed) if seed < 0 => return invalid("seed must not be negative"),
        Some(seed) => seed as u64,
        None => std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish(),
    };
    Ok(SamplingControls {
        temperature,
        top_p,
        top_k: top_k as u32,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        seed,
    })
}

fn finite_in(value: f32, range: std::ops::RangeInclusive<f32>) -> bool {
    value.is_finite() && range.contains(&value)
}

fn conversation(messages: &[ChatMessage]) -> Result<Conversation> {
    let mut messages = VecDeque::from(messages.to_vec());
    let mut instructions = Vec::new();
    while matches!(messages.front(), Some(message) if message.role == "system") {
        let message = messages.pop_front().expect("front was present");
        reject_message_name(&message)?;
        let content = text_content(&message.content)?.join("\n");
        if !content.is_empty() {
            instructions.push(content);
        }
    }
    let system = (!instructions.is_empty()).then(|| instructions.join("\n"));
    let mut entries = Vec::new();
    while let Some(message) = messages.pop_front() {
        match message.role.as_str() {
            "system" => {
                return invalid("system messages must precede conversation entries");
            }
            "user" => {
                reject_message_name(&message)?;
                let parts = text_content(&message.content)?
                    .into_iter()
                    .filter(|text| !text.is_empty())
                    .map(UserPart::Text)
                    .collect();
                entries.push(Entry::User(parts));
            }
            "assistant" => {
                reject_message_name(&message)?;
                let text = text_content(&message.content)?.join("\n");
                let text = (!text.is_empty()).then_some(text);
                let calls = message.tool_calls.unwrap_or_default();
                if text.is_none() && calls.is_empty() {
                    continue;
                }
                let mut results = BTreeMap::new();
                while matches!(messages.front(), Some(next) if next.role == "tool") {
                    let result = messages.pop_front().expect("front was present");
                    let id = result.tool_call_id.ok_or_else(|| {
                        PowerError::InvalidRequest("tool result requires tool_call_id".into())
                    })?;
                    if id.is_empty() {
                        return invalid("tool_call_id must not be empty");
                    }
                    let content = text_content(&result.content)?
                        .into_iter()
                        .filter(|text| !text.is_empty())
                        .map(ToolResultPart::Text)
                        .collect::<Vec<_>>();
                    if results.insert(id.clone(), content).is_some() {
                        return Err(PowerError::InvalidRequest(format!(
                            "duplicate tool result for call: {id}"
                        )));
                    }
                }
                let mut exchanges = Vec::with_capacity(calls.len());
                let mut ids = BTreeSet::new();
                for call in calls {
                    if call.tool_type != "function" {
                        return invalid("only function tool calls are supported");
                    }
                    if call.id.is_empty() || call.function.name.is_empty() {
                        return invalid("tool call id and function name must not be empty");
                    }
                    if !ids.insert(call.id.clone()) {
                        return Err(PowerError::InvalidRequest(format!(
                            "duplicate assistant tool-call id: {}",
                            call.id
                        )));
                    }
                    let arguments =
                        serde_json::from_str::<Map<String, Value>>(&call.function.arguments)
                            .map_err(|error| {
                                PowerError::InvalidRequest(format!(
                                    "assistant tool-call arguments must be a JSON object: {error}"
                                ))
                            })?;
                    let result = results.remove(&call.id).ok_or_else(|| {
                        PowerError::InvalidRequest(format!(
                            "assistant tool call {} has no immediately following result",
                            call.id
                        ))
                    })?;
                    exchanges.push(ToolExchange {
                        call: MagnitudeToolCall {
                            id: call.id,
                            name: call.function.name,
                            arguments,
                        },
                        result,
                    });
                }
                if let Some(unmatched) = results.keys().next() {
                    return Err(PowerError::InvalidRequest(format!(
                        "tool result {unmatched} does not match an assistant tool call"
                    )));
                }
                entries.push(Entry::Assistant(AssistantTurn {
                    reasoning: None,
                    text,
                    tool_calls: exchanges,
                }));
            }
            "tool" => {
                return invalid(
                    "tool results must immediately follow the assistant tool calls they complete",
                );
            }
            role => {
                return Err(PowerError::InvalidRequest(format!(
                    "unsupported chat message role: {role}"
                )));
            }
        }
    }
    Conversation::new(system, entries)
        .map_err(|error| PowerError::InvalidRequest(error.to_string()))
}

fn reject_message_name(message: &ChatMessage) -> Result<()> {
    if message.name.is_some() {
        return unsupported("chat message names are not represented by the Magnitude engine");
    }
    Ok(())
}

fn text_content(content: &MessageContent) -> Result<Vec<String>> {
    match content {
        MessageContent::Text(text) => Ok(vec![text.clone()]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text, .. } => Ok(text.clone()),
                ContentPart::ImageUrl { .. } => {
                    unsupported("image input is not implemented by the Magnitude adapter")
                }
            })
            .collect(),
    }
}

fn tools(request: &ChatRequest) -> Result<Tools> {
    let definitions = request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| {
            if tool.tool_type != "function" {
                return invalid("only function tools are supported");
            }
            if tool.function.strict.is_some() {
                return unsupported("tool function strict policy is not represented by Magnitude");
            }
            let Value::Object(parameters) = &tool.function.parameters else {
                return invalid("tool function parameters must be a JSON object");
            };
            Ok(ToolDefinition {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: JsonSchema::new(parameters.clone())
                    .map_err(|error| PowerError::InvalidRequest(error.to_string()))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let choice = match &request.tool_choice {
        None => MagnitudeToolChoice::Auto,
        Some(ToolChoice::String(choice)) => match choice.as_str() {
            "auto" => MagnitudeToolChoice::Auto,
            "none" => MagnitudeToolChoice::None,
            "required" => MagnitudeToolChoice::Required,
            _ => return invalid("tool_choice must be auto, none, required, or a function"),
        },
        Some(ToolChoice::Specific(choice)) => {
            if choice.tool_type != "function" {
                return invalid("only function tool choices are supported");
            }
            MagnitudeToolChoice::Named(choice.function.name.clone())
        }
    };
    Tools::new(
        definitions,
        choice,
        request.parallel_tool_calls.unwrap_or(true),
    )
    .map_err(|error| PowerError::InvalidRequest(error.to_string()))
}

fn output_format(format: Option<&Value>) -> Result<OutputFormat> {
    match format {
        None | Some(Value::Null) => Ok(OutputFormat::Text),
        Some(Value::String(value)) if value == "text" => Ok(OutputFormat::Text),
        Some(Value::String(value)) if value == "json" => Ok(OutputFormat::JsonObject),
        Some(Value::Object(value)) if value.get("type").and_then(Value::as_str) == Some("text") => {
            Ok(OutputFormat::Text)
        }
        Some(Value::Object(value))
            if value.get("type").and_then(Value::as_str) == Some("json_object") =>
        {
            Ok(OutputFormat::JsonObject)
        }
        Some(Value::Object(value)) => Ok(OutputFormat::JsonSchema {
            // Power's local backend format carries only the schema; Magnitude ignores this name.
            name: String::new(),
            schema: JsonSchema::new(value.clone())
                .map_err(|error| PowerError::InvalidRequest(error.to_string()))?,
        }),
        _ => invalid("response_format must be text, json, json_object, or a JSON Schema"),
    }
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(PowerError::InvalidRequest(message.into()))
}

fn unsupported<T>(message: &str) -> Result<T> {
    Err(PowerError::BackendNotAvailable(format!(
        "magnitude: {message}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::types::{ChatMessage, MessageContent};

    fn request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            messages,
            session_id: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: true,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: Some(7),
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
            num_parallel: Some(1),
            images: None,
        }
    }

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: MessageContent::Text(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }
    }

    #[test]
    fn converts_text_conversation_and_sampling() {
        let converted = generation_request(&request(vec![
            message("system", "be concise"),
            message("user", "hello"),
            message("assistant", "hi"),
            message("user", "why?"),
        ]))
        .unwrap();
        assert_eq!(converted.input.conversation.system(), Some("be concise"));
        assert_eq!(converted.input.conversation.entries().len(), 3);
        assert_eq!(converted.controls.sampling.seed, 7);
        assert_eq!(converted.controls.sampling.top_k, 20);
        assert_eq!(converted.controls.prompt_cache, PromptCache::Disabled);
    }

    #[test]
    fn rejects_cache_and_unrepresentable_sampling() {
        let mut cached = request(vec![message("user", "hello")]);
        cached.session_id = Some("tenant-key".into());
        assert!(generation_request(&cached).is_err());

        let mut sampled = request(vec![message("user", "hello")]);
        sampled.mirostat = Some(1);
        assert!(generation_request(&sampled).is_err());

        let mut negative_seed = request(vec![message("user", "hello")]);
        negative_seed.seed = Some(-1);
        assert!(generation_request(&negative_seed).is_err());
    }

    #[test]
    fn converts_json_schema_output() {
        let mut input = request(vec![message("user", "answer")]);
        input.response_format = Some(serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        }));
        let converted = generation_request(&input).unwrap();
        let OutputFormat::JsonSchema { schema, .. } = converted.input.output else {
            panic!("expected JSON schema output");
        };
        assert_eq!(schema.source()["type"], "object");
    }

    #[test]
    fn converts_tool_definitions_and_immediate_results() {
        let mut assistant = message("assistant", "");
        assistant.tool_calls = Some(vec![crate::backend::types::ToolCall {
            id: "call-1".into(),
            tool_type: "function".into(),
            function: crate::backend::types::FunctionCall {
                name: "weather".into(),
                arguments: "{\"city\":\"Paris\"}".into(),
            },
            index: Some(0),
        }]);
        let mut result = message("tool", "sunny");
        result.tool_call_id = Some("call-1".into());
        let mut input = request(vec![
            message("user", "weather?"),
            assistant,
            result,
            message("user", "thanks"),
        ]);
        input.tools = Some(vec![crate::backend::types::Tool {
            tool_type: "function".into(),
            function: crate::backend::types::FunctionDefinition {
                name: "weather".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
                unsupported: BTreeMap::new(),
            },
            unsupported: BTreeMap::new(),
        }]);
        input.tool_choice = Some(ToolChoice::String("required".into()));

        let converted = generation_request(&input).unwrap();
        assert_eq!(converted.input.tools.definitions().len(), 1);
        assert!(converted.input.tools.enabled());
        let Entry::Assistant(turn) = &converted.input.conversation.entries()[1] else {
            panic!("expected assistant tool exchange");
        };
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].call.id, "call-1");
        assert!(matches!(
            turn.tool_calls[0].result.as_slice(),
            [ToolResultPart::Text(text)] if text == "sunny"
        ));
    }
}
