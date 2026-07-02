//! Per-request attestation receipts for prompt and decoding policy.
//!
//! These receipts do not replace hardware attestation. They give verifiers a
//! stable digest of the prompt-bearing request fields and sampling/decoding
//! policy used for a specific inference request. When a backend can prove the
//! exact prompt representation it submits to the model, the receipt also
//! carries an `effective_prompt` digest.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::api::types::{ChatCompletionRequest, CompletionRequest};
use crate::backend::types::EffectivePromptDigest;
use crate::tee::attestation::RuntimePolicyClaim;

/// Request-level receipt returned with inference responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttestationReceipt {
    pub schema: String,
    pub request_type: ReceiptRequestType,
    pub model: String,
    pub input: ReceiptInputDigest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_policy: Option<RuntimePolicyClaim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_prompt: Option<EffectivePromptDigest>,
    pub decoding: ReceiptDecodingPolicy,
}

impl AttestationReceipt {
    pub const SCHEMA: &'static str = "a3s.power.inference-receipt.v2";
}

/// Kind of request covered by the receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReceiptRequestType {
    ChatCompletion,
    TextCompletion,
}

/// Hash of the prompt-bearing input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptInputDigest {
    pub kind: String,
    pub sha256: String,
}

/// Stable sampling and output-policy digest inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiptDecodingPolicy {
    pub parameters: BTreeMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_tokens_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice_sha256: Option<String>,
}

/// Build a chat-completion receipt and its SHA-256 digest.
pub fn chat_receipt(request: &ChatCompletionRequest) -> crate::error::Result<AttestationReceipt> {
    chat_receipt_with_runtime_policy(request, None)
}

/// Build a chat-completion receipt with model runtime policy claims.
pub fn chat_receipt_with_runtime_policy(
    request: &ChatCompletionRequest,
    runtime_policy: Option<RuntimePolicyClaim>,
) -> crate::error::Result<AttestationReceipt> {
    chat_receipt_with_runtime_policy_and_effective_prompt(request, runtime_policy, None)
}

/// Build a chat-completion receipt with model runtime policy and effective prompt claims.
pub fn chat_receipt_with_runtime_policy_and_effective_prompt(
    request: &ChatCompletionRequest,
    runtime_policy: Option<RuntimePolicyClaim>,
    effective_prompt: Option<EffectivePromptDigest>,
) -> crate::error::Result<AttestationReceipt> {
    let input = ReceiptInputDigest {
        kind: "chat.messages".to_string(),
        sha256: digest_json(&request.messages)?,
    };

    let mut parameters = BTreeMap::new();
    insert_optional_f32(&mut parameters, "temperature", request.temperature);
    insert_optional_f32(&mut parameters, "top_p", request.top_p);
    insert_optional_u32(&mut parameters, "max_tokens", request.max_tokens);
    insert_optional_i32(&mut parameters, "top_k", request.top_k);
    insert_optional_f32(&mut parameters, "min_p", request.min_p);
    insert_optional_f32(&mut parameters, "repeat_penalty", request.repeat_penalty);
    insert_optional_i32(&mut parameters, "repeat_last_n", request.repeat_last_n);
    insert_optional_bool(
        &mut parameters,
        "penalize_newline",
        request.penalize_newline,
    );
    insert_optional_u32(&mut parameters, "num_ctx", request.num_ctx);
    insert_optional_u32(&mut parameters, "mirostat", request.mirostat);
    insert_optional_f32(&mut parameters, "mirostat_tau", request.mirostat_tau);
    insert_optional_f32(&mut parameters, "mirostat_eta", request.mirostat_eta);
    insert_optional_f32(&mut parameters, "tfs_z", request.tfs_z);
    insert_optional_f32(&mut parameters, "typical_p", request.typical_p);
    insert_optional_i64(&mut parameters, "seed", request.seed);
    insert_optional_f32(
        &mut parameters,
        "presence_penalty",
        request.presence_penalty,
    );
    insert_optional_f32(
        &mut parameters,
        "frequency_penalty",
        request.frequency_penalty,
    );
    parameters.insert(
        "stream".to_string(),
        serde_json::Value::Bool(request.stream.unwrap_or(false)),
    );
    parameters.insert(
        "parallel_tool_calls".to_string(),
        request
            .parallel_tool_calls
            .map(serde_json::Value::Bool)
            .unwrap_or(serde_json::Value::Null),
    );

    Ok(AttestationReceipt {
        schema: AttestationReceipt::SCHEMA.to_string(),
        request_type: ReceiptRequestType::ChatCompletion,
        model: request.model.clone(),
        input,
        runtime_policy,
        effective_prompt,
        decoding: ReceiptDecodingPolicy {
            parameters,
            stream_options_sha256: digest_optional_json(request.stream_options.as_ref())?,
            stop_tokens_sha256: digest_optional_json(request.stop.as_ref())?,
            response_format_sha256: digest_optional_json(request.response_format.as_ref())?,
            tools_sha256: digest_optional_json(request.tools.as_ref())?,
            tool_choice_sha256: digest_optional_json(request.tool_choice.as_ref())?,
        },
    })
}

/// Build a text-completion receipt and its SHA-256 digest.
pub fn completion_receipt(request: &CompletionRequest) -> crate::error::Result<AttestationReceipt> {
    completion_receipt_with_runtime_policy(request, None)
}

/// Build a text-completion receipt with model runtime policy claims.
pub fn completion_receipt_with_runtime_policy(
    request: &CompletionRequest,
    runtime_policy: Option<RuntimePolicyClaim>,
) -> crate::error::Result<AttestationReceipt> {
    let input = ReceiptInputDigest {
        kind: "text.prompt".to_string(),
        sha256: digest_json(&request.prompt)?,
    };

    let mut parameters = BTreeMap::new();
    insert_optional_f32(&mut parameters, "temperature", request.temperature);
    insert_optional_f32(&mut parameters, "top_p", request.top_p);
    insert_optional_u32(&mut parameters, "max_tokens", request.max_tokens);
    insert_optional_i32(&mut parameters, "top_k", request.top_k);
    insert_optional_f32(&mut parameters, "min_p", request.min_p);
    insert_optional_f32(&mut parameters, "repeat_penalty", request.repeat_penalty);
    insert_optional_i32(&mut parameters, "repeat_last_n", request.repeat_last_n);
    insert_optional_bool(
        &mut parameters,
        "penalize_newline",
        request.penalize_newline,
    );
    insert_optional_u32(&mut parameters, "num_ctx", request.num_ctx);
    insert_optional_u32(&mut parameters, "mirostat", request.mirostat);
    insert_optional_f32(&mut parameters, "mirostat_tau", request.mirostat_tau);
    insert_optional_f32(&mut parameters, "mirostat_eta", request.mirostat_eta);
    insert_optional_f32(&mut parameters, "tfs_z", request.tfs_z);
    insert_optional_f32(&mut parameters, "typical_p", request.typical_p);
    insert_optional_i64(&mut parameters, "seed", request.seed);
    insert_optional_f32(
        &mut parameters,
        "presence_penalty",
        request.presence_penalty,
    );
    insert_optional_f32(
        &mut parameters,
        "frequency_penalty",
        request.frequency_penalty,
    );
    parameters.insert(
        "stream".to_string(),
        serde_json::Value::Bool(request.stream.unwrap_or(false)),
    );

    Ok(AttestationReceipt {
        schema: AttestationReceipt::SCHEMA.to_string(),
        request_type: ReceiptRequestType::TextCompletion,
        model: request.model.clone(),
        input,
        runtime_policy,
        effective_prompt: None,
        decoding: ReceiptDecodingPolicy {
            parameters,
            stream_options_sha256: digest_optional_json(request.stream_options.as_ref())?,
            stop_tokens_sha256: digest_optional_json(request.stop.as_ref())?,
            response_format_sha256: None,
            tools_sha256: None,
            tool_choice_sha256: None,
        },
    })
}

/// SHA-256 digest of the canonical receipt bytes as lowercase hex.
pub fn receipt_digest(receipt: &AttestationReceipt) -> crate::error::Result<String> {
    let bytes = canonical_json_bytes(receipt)?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

/// SHA-256 digest of the canonical request decoding-parameter map.
///
/// This covers sampling and request-visible decoding controls such as
/// temperature, top_p, max_tokens, seed, penalties, stream, and tool-call mode.
/// Output-shaping fields that are already digests in the receipt
/// (`stream_options_sha256`, `stop_tokens_sha256`, `response_format_sha256`,
/// `tools_sha256`, and `tool_choice_sha256`) are pinned separately by verifier
/// policy.
pub fn receipt_decoding_parameters_digest(
    receipt: &AttestationReceipt,
) -> crate::error::Result<String> {
    digest_json(&receipt.decoding.parameters)
}

fn insert_optional_f32(
    map: &mut BTreeMap<String, serde_json::Value>,
    key: &'static str,
    value: Option<f32>,
) {
    let value = value
        .and_then(|v| serde_json::Number::from_f64(v as f64))
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null);
    map.insert(key.to_string(), value);
}

fn insert_optional_u32(
    map: &mut BTreeMap<String, serde_json::Value>,
    key: &'static str,
    value: Option<u32>,
) {
    map.insert(
        key.to_string(),
        value
            .map(|v| serde_json::Value::Number(v.into()))
            .unwrap_or(serde_json::Value::Null),
    );
}

fn insert_optional_i32(
    map: &mut BTreeMap<String, serde_json::Value>,
    key: &'static str,
    value: Option<i32>,
) {
    map.insert(
        key.to_string(),
        value
            .map(|v| serde_json::Value::Number(v.into()))
            .unwrap_or(serde_json::Value::Null),
    );
}

fn insert_optional_i64(
    map: &mut BTreeMap<String, serde_json::Value>,
    key: &'static str,
    value: Option<i64>,
) {
    map.insert(
        key.to_string(),
        value
            .map(|v| serde_json::Value::Number(v.into()))
            .unwrap_or(serde_json::Value::Null),
    );
}

fn insert_optional_bool(
    map: &mut BTreeMap<String, serde_json::Value>,
    key: &'static str,
    value: Option<bool>,
) {
    map.insert(
        key.to_string(),
        value
            .map(serde_json::Value::Bool)
            .unwrap_or(serde_json::Value::Null),
    );
}

fn digest_optional_json<T: Serialize>(value: Option<&T>) -> crate::error::Result<Option<String>> {
    value.map(digest_json).transpose()
}

fn digest_json<T: Serialize>(value: &T) -> crate::error::Result<String> {
    let bytes = canonical_json_bytes(value)?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> crate::error::Result<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    serde_json::to_vec(&canonicalize_json_value(&value)).map_err(crate::error::PowerError::from)
}

fn canonicalize_json_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(canonicalize_json_value)
                .collect::<Vec<_>>(),
        ),
        serde_json::Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (key, value) in map {
                sorted.insert(key.clone(), canonicalize_json_value(value));
            }
            let mut canonical = serde_json::Map::new();
            for (key, value) in sorted {
                canonical.insert(key, value);
            }
            serde_json::Value::Object(canonical)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::ChatCompletionMessage;
    use crate::backend::types::{ContentPart, ImageUrl, MessageContent};

    fn chat_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "llama3".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
                thinking: None,
            }],
            temperature: Some(0.2),
            top_p: Some(0.9),
            max_tokens: Some(128),
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            stop: Some(vec!["</s>".to_string()]),
            stream: Some(false),
            stream_options: None,
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.0),
            seed: Some(7),
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            keep_alive: None,
        }
    }

    #[test]
    fn chat_receipt_digest_is_stable_for_same_request() {
        let receipt = chat_receipt(&chat_request()).unwrap();
        assert_eq!(receipt.schema, "a3s.power.inference-receipt.v2");
        assert_eq!(
            receipt_digest(&receipt).unwrap(),
            receipt_digest(&receipt).unwrap()
        );
    }

    #[test]
    fn chat_receipt_runtime_policy_changes_digest() {
        let request = chat_request();
        let base = chat_receipt(&request).unwrap();
        let runtime_policy =
            RuntimePolicyClaim::new().with_prompt(crate::tee::attestation::PromptPolicyClaim {
                chat_template_source: Some("manifest.template_override".to_string()),
                chat_template_sha256: Some(vec![0x11; 32]),
                system_prompt_sha256: None,
                messages_sha256: None,
            });
        let with_runtime =
            chat_receipt_with_runtime_policy(&request, Some(runtime_policy)).unwrap();

        assert!(with_runtime.runtime_policy.is_some());
        assert_ne!(
            receipt_digest(&base).unwrap(),
            receipt_digest(&with_runtime).unwrap()
        );
    }

    #[test]
    fn chat_receipt_effective_prompt_changes_digest() {
        let request = chat_request();
        let base = chat_receipt(&request).unwrap();
        let with_effective = chat_receipt_with_runtime_policy_and_effective_prompt(
            &request,
            None,
            Some(EffectivePromptDigest::chat_rendered_prompt(
                "test-backend",
                "rendered prompt",
            )),
        )
        .unwrap();

        assert!(with_effective.effective_prompt.is_some());
        assert_ne!(
            receipt_digest(&base).unwrap(),
            receipt_digest(&with_effective).unwrap()
        );
    }

    #[test]
    fn chat_receipt_input_digest_changes_when_message_images_change() {
        let mut first = chat_request();
        let mut second = chat_request();
        first.messages[0].images = Some(vec!["base64-image-a".to_string()]);
        second.messages[0].images = Some(vec!["base64-image-b".to_string()]);

        let first_receipt = chat_receipt(&first).unwrap();
        let second_receipt = chat_receipt(&second).unwrap();

        assert_eq!(first_receipt.input.kind, "chat.messages");
        assert_ne!(first_receipt.input.sha256, second_receipt.input.sha256);
        assert_ne!(
            receipt_digest(&first_receipt).unwrap(),
            receipt_digest(&second_receipt).unwrap()
        );
    }

    #[test]
    fn chat_receipt_input_digest_changes_when_image_url_part_changes() {
        let mut first = chat_request();
        let mut second = chat_request();
        first.messages[0].content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aW1hZ2UtYQ==".to_string(),
                    detail: Some("low".to_string()),
                },
            },
        ]);
        second.messages[0].content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aW1hZ2UtYg==".to_string(),
                    detail: Some("low".to_string()),
                },
            },
        ]);

        let first_receipt = chat_receipt(&first).unwrap();
        let second_receipt = chat_receipt(&second).unwrap();

        assert_ne!(first_receipt.input.sha256, second_receipt.input.sha256);
    }

    #[test]
    fn chat_receipt_digest_changes_when_sampling_changes() {
        let mut changed = chat_request();
        let first = receipt_digest(&chat_receipt(&chat_request()).unwrap()).unwrap();
        changed.temperature = Some(0.7);
        let second = receipt_digest(&chat_receipt(&changed).unwrap()).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn chat_receipt_covers_extended_sampling_controls() {
        let mut changed = chat_request();
        let first = receipt_digest(&chat_receipt(&changed).unwrap()).unwrap();

        changed.top_k = Some(40);
        changed.min_p = Some(0.5);
        changed.repeat_penalty = Some(1.25);
        changed.repeat_last_n = Some(64);
        changed.penalize_newline = Some(true);
        changed.num_ctx = Some(4096);
        changed.mirostat = Some(2);
        changed.mirostat_tau = Some(5.0);
        changed.mirostat_eta = Some(0.25);
        changed.tfs_z = Some(0.75);
        changed.typical_p = Some(0.5);

        let receipt = chat_receipt(&changed).unwrap();
        assert_eq!(receipt.decoding.parameters["top_k"], serde_json::json!(40));
        assert_eq!(receipt.decoding.parameters["min_p"], serde_json::json!(0.5));
        assert_eq!(
            receipt.decoding.parameters["repeat_penalty"],
            serde_json::json!(1.25)
        );
        assert_eq!(
            receipt.decoding.parameters["repeat_last_n"],
            serde_json::json!(64)
        );
        assert_eq!(
            receipt.decoding.parameters["penalize_newline"],
            serde_json::json!(true)
        );
        assert_eq!(
            receipt.decoding.parameters["num_ctx"],
            serde_json::json!(4096)
        );
        assert_ne!(first, receipt_digest(&receipt).unwrap());
    }

    #[test]
    fn chat_receipt_digest_changes_when_stop_tokens_change() {
        let mut changed = chat_request();
        let first = receipt_digest(&chat_receipt(&chat_request()).unwrap()).unwrap();
        changed.stop = Some(vec!["END".to_string()]);
        let second = receipt_digest(&chat_receipt(&changed).unwrap()).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn chat_receipt_covers_stream_options() {
        let first = receipt_digest(&chat_receipt(&chat_request()).unwrap()).unwrap();
        let mut changed = chat_request();
        changed.stream = Some(true);
        changed.stream_options = Some(crate::api::types::StreamOptions {
            include_usage: true,
        });

        let receipt = chat_receipt(&changed).unwrap();

        assert!(receipt.decoding.stream_options_sha256.is_some());
        assert_ne!(first, receipt_digest(&receipt).unwrap());
    }

    #[test]
    fn completion_receipt_hashes_prompt() {
        let request = CompletionRequest {
            model: "llama3".to_string(),
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            stop: None,
            stream: Some(false),
            stream_options: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            keep_alive: None,
        };

        let receipt = completion_receipt(&request).unwrap();
        assert_eq!(receipt.input.kind, "text.prompt");
        assert_eq!(receipt.input.sha256.len(), 64);
    }

    #[test]
    fn completion_receipt_covers_extended_sampling_controls() {
        let request = CompletionRequest {
            model: "llama3".to_string(),
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            top_k: Some(40),
            min_p: Some(0.5),
            repeat_penalty: Some(1.25),
            repeat_last_n: Some(64),
            penalize_newline: Some(false),
            num_ctx: Some(2048),
            mirostat: Some(1),
            mirostat_tau: Some(4.0),
            mirostat_eta: Some(0.25),
            tfs_z: Some(0.75),
            typical_p: Some(0.5),
            stop: None,
            stream: Some(false),
            stream_options: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            keep_alive: None,
        };

        let receipt = completion_receipt(&request).unwrap();

        assert_eq!(receipt.decoding.parameters["top_k"], serde_json::json!(40));
        assert_eq!(
            receipt.decoding.parameters["penalize_newline"],
            serde_json::json!(false)
        );
        assert_eq!(
            receipt.decoding.parameters["typical_p"],
            serde_json::json!(0.5)
        );
    }

    #[test]
    fn completion_receipt_covers_stream_options() {
        let request = CompletionRequest {
            model: "llama3".to_string(),
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            stop: None,
            stream: Some(true),
            stream_options: Some(crate::api::types::StreamOptions {
                include_usage: true,
            }),
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            keep_alive: None,
        };

        let receipt = completion_receipt(&request).unwrap();

        assert!(receipt.decoding.stream_options_sha256.is_some());
    }
}
