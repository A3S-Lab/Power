use super::types::ChatMessage;

/// Recognized chat template formats for prompt construction.
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

/// Format chat messages into a prompt string using the given template kind.
pub fn format_chat_prompt(messages: &[ChatMessage], kind: &ChatTemplateKind) -> String {
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
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
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
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_llama() {
        let msgs = sample_messages();
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Llama);
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("Hello [/INST]"));
    }

    #[test]
    fn test_format_phi() {
        let msgs = sample_messages();
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Phi);
        assert!(prompt.contains("<|system|>\nYou are helpful.<|end|>"));
        assert!(prompt.contains("<|user|>\nHello<|end|>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_format_generic() {
        let msgs = sample_messages();
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Generic);
        assert!(prompt.contains("system: You are helpful."));
        assert!(prompt.contains("user: Hello"));
        assert!(prompt.ends_with("assistant: "));
    }

    #[test]
    fn test_format_empty_messages() {
        let msgs: Vec<ChatMessage> = vec![];
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl);
        assert_eq!(prompt, "<|im_start|>assistant\n");

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Phi);
        assert_eq!(prompt, "<|assistant|>\n");

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Generic);
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
        }];

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl);
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
            },
            ChatMessage {
                role: "tool".to_string(),
                content: MessageContent::Text("72F sunny".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
            },
        ];

        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::ChatMl);
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
        }];

        // Name field doesn't affect template formatting, just content
        let prompt = format_chat_prompt(&msgs, &ChatTemplateKind::Phi);
        assert!(prompt.contains("hello"));
    }
}
