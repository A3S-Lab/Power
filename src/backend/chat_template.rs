use super::types::ChatMessage;

/// Recognized chat template formats for prompt construction.
///
/// Used as fallback when no raw Jinja2 template string is available.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplateKind {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>`
    ChatMl,
    /// Llama: `[INST] ... [/INST]`
    Llama,
    /// Phi: `<|system|>\ncontent<|end|>`
    Phi,
    /// Generic fallback: `role: content\n`
    Generic,
}

/// Detect the chat template kind from a raw template string (e.g. from GGUF metadata).
pub fn detect(template_str: &str) -> ChatTemplateKind {
    if template_str.contains("<|im_start|>") {
        ChatTemplateKind::ChatMl
    } else if template_str.contains("[INST]") {
        ChatTemplateKind::Llama
    } else if template_str.contains("<|system|>") && template_str.contains("<|end|>") {
        ChatTemplateKind::Phi
    } else {
        ChatTemplateKind::Generic
    }
}

/// Render a chat prompt using a raw Jinja2 template string.
///
/// The template receives `messages` (array of `{role, content}` objects) and
/// `add_generation_prompt` (bool). This matches the HuggingFace/Ollama convention.
///
/// Returns `None` if rendering fails, allowing the caller to fall back to
/// hardcoded template formatting.
pub fn render_jinja(
    template_str: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Option<String> {
    let env = minijinja::Environment::new();

    // Build messages as simple Value objects for the template
    let msg_values: Vec<minijinja::Value> = messages
        .iter()
        .map(|m| {
            let mut map = std::collections::BTreeMap::new();
            map.insert("role".to_string(), minijinja::Value::from(m.role.as_str()));
            map.insert(
                "content".to_string(),
                minijinja::Value::from(m.content.text()),
            );
            minijinja::Value::from_object(map)
        })
        .collect();

    let result = env.render_str(
        template_str,
        minijinja::context! {
            messages => msg_values,
            add_generation_prompt => add_generation_prompt,
            bos_token => "<s>",
            eos_token => "</s>",
        },
    );

    match result {
        Ok(rendered) => Some(rendered),
        Err(e) => {
            tracing::warn!(error = %e, "Jinja2 template rendering failed, falling back to hardcoded template");
            None
        }
    }
}

/// Format chat messages into a prompt string.
///
/// If a raw Jinja2 template string is provided, attempts to render it first.
/// Falls back to the hardcoded template kind if Jinja2 rendering fails or
/// no raw template is available.
pub fn format_chat_prompt(
    messages: &[ChatMessage],
    kind: &ChatTemplateKind,
    raw_template: Option<&str>,
) -> String {
    // Try Jinja2 rendering first if a raw template is available
    if let Some(template_str) = raw_template {
        if let Some(rendered) = render_jinja(template_str, messages, true) {
            return rendered;
        }
    }

    // Fallback to hardcoded templates
    match kind {
        ChatTemplateKind::ChatMl => format_chatml(messages),
        ChatTemplateKind::Llama => format_llama(messages),
        ChatTemplateKind::Phi => format_phi(messages),
        ChatTemplateKind::Generic => format_generic(messages),
    }
}

fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role,
            msg.content.text()
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn format_llama(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut system_text = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                system_text = msg.content.text();
            }
            "user" => {
                prompt.push_str("<s>[INST] ");
                if !system_text.is_empty() {
                    prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", system_text));
                    system_text.clear();
                }
                prompt.push_str(&format!("{} [/INST]", msg.content.text()));
            }
            "assistant" => {
                prompt.push_str(&format!(" {} </s>", msg.content.text()));
            }
            _ => {
                prompt.push_str(&format!("{}: {}\n", msg.role, msg.content.text()));
            }
        }
    }
    prompt
}

fn format_phi(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|{}|>\n{}<|end|>\n",
            msg.role,
            msg.content.text()
        ));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn format_generic(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content.text()));
    }
    prompt.push_str("assistant: ");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::types::MessageContent;

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
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
                content: MessageContent::Text("Hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
        ]
    }

    #[test]
    fn test_detect_chatml() {
        let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";
        assert_eq!(detect(template), ChatTemplateKind::ChatMl);
    }

    #[test]
    fn test_detect_llama() {
        let template = "{% if messages[0]['role'] == 'system' %}[INST] <<SYS>>{{ messages[0]['content'] }}<</SYS>>{% endif %}";
        assert_eq!(detect(template), ChatTemplateKind::Llama);
    }

    #[test]
    fn test_detect_phi() {
        let template = "<|system|>\n{{ system_message }}<|end|>\n<|user|>\n{{ user_message }}<|end|>\n<|assistant|>\n";
        assert_eq!(detect(template), ChatTemplateKind::Phi);
    }

    #[test]
    fn test_detect_generic_fallback() {
        let template = "some unknown template format";
        assert_eq!(detect(template), ChatTemplateKind::Generic);
    }

    #[test]
    fn test_format_chatml() {
        let msgs = sample_messages();
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl, None);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_llama() {
        let msgs = sample_messages();
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Llama, None);
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("Hello [/INST]"));
    }

    #[test]
    fn test_format_phi() {
        let msgs = sample_messages();
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Phi, None);
        assert!(prompt.contains("<|system|>\nYou are helpful.<|end|>"));
        assert!(prompt.contains("<|user|>\nHello<|end|>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_format_generic() {
        let msgs = sample_messages();
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Generic, None);
        assert!(prompt.contains("system: You are helpful."));
        assert!(prompt.contains("user: Hello"));
        assert!(prompt.ends_with("assistant: "));
    }

    #[test]
    fn test_format_empty_messages() {
        let msgs: Vec<ChatMessage> = vec![];
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl, None);
        assert_eq!(prompt, "<|im_start|>assistant\n");

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Phi, None);
        assert_eq!(prompt, "<|assistant|>\n");

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Generic, None);
        assert_eq!(prompt, "assistant: ");
    }

    #[test]
    fn test_format_multimodal_message_extracts_text() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Parts(vec![
                crate::backend::types::ContentPart::Text {
                    text: "Describe this image".to_string(),
                },
                crate::backend::types::ContentPart::ImageUrl {
                    image_url: crate::backend::types::ImageUrl {
                        url: "https://example.com/img.jpg".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }];

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl, None);
        assert!(prompt.contains("Describe this image"));
        assert!(!prompt.contains("example.com"));
    }

    #[test]
    fn test_format_tool_message_extracts_text() {
        let msgs = vec![
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("weather?".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                content: MessageContent::Text("72F sunny".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                images: None,
            },
        ];

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl, None);
        assert!(prompt.contains("weather?"));
        assert!(prompt.contains("72F sunny"));
    }

    #[test]
    fn test_format_message_with_name_field() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("hello".to_string()),
            name: Some("Alice".to_string()),
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }];

        // Name field doesn't affect template formatting, just content
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Phi, None);
        assert!(prompt.contains("hello"));
    }

    // ========================================================================
    // Jinja2 template rendering tests
    // ========================================================================

    #[test]
    fn test_render_jinja_chatml_template() {
        let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let msgs = sample_messages();
        let result = render_jinja(template, &msgs, true);
        assert!(result.is_some());
        let rendered = result.unwrap();
        assert!(rendered.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(rendered.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(rendered.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_render_jinja_phi_template() {
        let template = "{% for message in messages %}<|{{ message.role }}|>\n{{ message.content }}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";
        let msgs = sample_messages();
        let result = render_jinja(template, &msgs, true);
        assert!(result.is_some());
        let rendered = result.unwrap();
        assert!(rendered.contains("<|system|>\nYou are helpful.<|end|>"));
        assert!(rendered.contains("<|user|>\nHello<|end|>"));
        assert!(rendered.contains("<|assistant|>"));
    }

    #[test]
    fn test_render_jinja_with_bos_eos_tokens() {
        let template = "{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{{ eos_token }}";
        let msgs = sample_messages();
        let result = render_jinja(template, &msgs, false);
        assert!(result.is_some());
        let rendered = result.unwrap();
        assert!(rendered.starts_with("<s>"));
        assert!(rendered.ends_with("</s>"));
    }

    #[test]
    fn test_render_jinja_empty_messages() {
        let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
        let msgs: Vec<ChatMessage> = vec![];
        let result = render_jinja(template, &msgs, true);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "assistant: ");
    }

    #[test]
    fn test_render_jinja_invalid_template_returns_none() {
        let template = "{% invalid syntax %}";
        let msgs = sample_messages();
        let result = render_jinja(template, &msgs, true);
        assert!(result.is_none());
    }

    #[test]
    fn test_format_chat_prompt_prefers_jinja_over_fallback() {
        let template =
            "{% for message in messages %}[{{ message.role }}] {{ message.content }}\n{% endfor %}";
        let msgs = sample_messages();
        // Even though kind is ChatMl, the raw template should be used
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl, Some(template));
        assert!(prompt.contains("[system] You are helpful."));
        assert!(prompt.contains("[user] Hello"));
        // Should NOT contain ChatML markers
        assert!(!prompt.contains("<|im_start|>"));
    }

    #[test]
    fn test_format_chat_prompt_falls_back_on_bad_jinja() {
        let bad_template = "{% invalid %}";
        let msgs = sample_messages();
        // Should fall back to ChatMl hardcoded format
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl, Some(bad_template));
        assert!(prompt.contains("<|im_start|>"));
    }

    #[test]
    fn test_render_jinja_llama3_style_template() {
        // Llama 3 uses a different template style
        let template = "{% for message in messages %}{% if message.role == 'system' %}<|start_header_id|>system<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% elif message.role == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% elif message.role == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";
        let msgs = sample_messages();
        let result = render_jinja(template, &msgs, true);
        assert!(result.is_some());
        let rendered = result.unwrap();
        assert!(rendered.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(rendered.contains("You are helpful."));
        assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(rendered.contains("Hello"));
        assert!(rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_render_jinja_gemma_style_template() {
        let template = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";
        let msgs = sample_messages();
        let result = render_jinja(template, &msgs, true);
        assert!(result.is_some());
        let rendered = result.unwrap();
        assert!(rendered.contains("<start_of_turn>system\nYou are helpful.<end_of_turn>"));
        assert!(rendered.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(rendered.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_render_jinja_without_generation_prompt() {
        let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
        let msgs = sample_messages();
        let result = render_jinja(template, &msgs, false);
        assert!(result.is_some());
        let rendered = result.unwrap();
        assert!(!rendered.ends_with("assistant: "));
    }

    #[test]
    fn test_render_jinja_multimodal_extracts_text() {
        let template =
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}";
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Parts(vec![
                crate::backend::types::ContentPart::Text {
                    text: "Describe this".to_string(),
                },
                crate::backend::types::ContentPart::ImageUrl {
                    image_url: crate::backend::types::ImageUrl {
                        url: "https://example.com/img.jpg".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }];
        let result = render_jinja(template, &msgs, false);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Describe this"));
    }
}
