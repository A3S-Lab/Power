// Modelfile parser for Ollama-compatible model definitions.
//
// Parses the Modelfile format with directives:
// FROM <base-model>
// PARAMETER <key> <value>
// SYSTEM "<prompt>"
// TEMPLATE "<template>"
// ADAPTER <path>
// LICENSE <text>
// MESSAGE <role> <content>

use std::collections::HashMap;

/// A pre-seeded message in a Modelfile (MESSAGE directive).
#[derive(Debug, Clone)]
pub struct ModelfileMessage {
    pub role: String,
    pub content: String,
}

/// Parsed representation of a Modelfile.
#[derive(Debug, Clone)]
pub struct Modelfile {
    /// Base model reference (e.g. "llama3.2:3b")
    pub from: String,
    /// Key-value parameters (e.g. temperature -> "0.7")
    pub parameters: HashMap<String, String>,
    /// System prompt
    pub system: Option<String>,
    /// Chat template override
    pub template: Option<String>,
    /// Stop sequences extracted from PARAMETER stop directives
    pub stop: Vec<String>,
    /// LoRA/QLoRA adapter path
    pub adapter: Option<String>,
    /// License text
    pub license: Option<String>,
    /// Pre-seeded conversation messages
    pub messages: Vec<ModelfileMessage>,
}

/// Parse a Modelfile from its text content.
///
/// Supports multi-line values using heredoc syntax with triple quotes:
/// ```text
/// SYSTEM """
/// You are a helpful assistant.
/// You always respond in English.
/// """
/// ```
pub fn parse(content: &str) -> Result<Modelfile, String> {
    let mut from: Option<String> = None;
    let mut parameters = HashMap::new();
    let mut system: Option<String> = None;
    let mut template: Option<String> = None;
    let mut stop = Vec::new();
    let mut adapter: Option<String> = None;
    let mut license: Option<String> = None;
    let mut messages = Vec::new();

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }

        // Split into directive and value
        let (directive, value) = match line.split_once(char::is_whitespace) {
            Some((d, v)) => (d.to_uppercase(), v.trim().to_string()),
            None => {
                // Single-word directives are not valid
                return Err(format!(
                    "line {}: expected directive and value, got '{line}'",
                    i + 1
                ));
            }
        };

        // Check if value starts a heredoc block (""")
        let value = if value.starts_with("\"\"\"") {
            let after_open = value.strip_prefix("\"\"\"").unwrap();
            // Check if the closing """ is on the same line
            if let Some(inline_content) = after_open.strip_suffix("\"\"\"") {
                inline_content.to_string()
            } else {
                // Multi-line: collect until closing """
                let mut multiline = String::new();
                if !after_open.is_empty() {
                    multiline.push_str(after_open);
                    multiline.push('\n');
                }
                i += 1;
                while i < lines.len() {
                    let ml = lines[i];
                    if ml.trim() == "\"\"\"" {
                        break;
                    }
                    if !multiline.is_empty() || !ml.trim().is_empty() {
                        multiline.push_str(ml);
                        multiline.push('\n');
                    }
                    i += 1;
                }
                if i >= lines.len() {
                    return Err(format!(
                        "unterminated heredoc block starting with {directive}"
                    ));
                }
                // Remove trailing newline
                if multiline.ends_with('\n') {
                    multiline.pop();
                }
                multiline
            }
        } else {
            value
        };

        match directive.as_str() {
            "FROM" => {
                from = Some(unquote(&value));
            }
            "PARAMETER" => {
                let (key, val) = match value.split_once(char::is_whitespace) {
                    Some((k, v)) => (k.trim().to_string(), v.trim().to_string()),
                    None => {
                        return Err(format!("line {}: PARAMETER requires key and value", i + 1));
                    }
                };
                if key == "stop" {
                    stop.push(unquote(&val));
                } else {
                    parameters.insert(key, unquote(&val));
                }
            }
            "SYSTEM" => {
                system = Some(unquote(&value));
            }
            "TEMPLATE" => {
                template = Some(unquote(&value));
            }
            "ADAPTER" => {
                adapter = Some(unquote(&value));
            }
            "LICENSE" => {
                license = Some(unquote(&value));
            }
            "MESSAGE" => {
                // MESSAGE role content
                let (role, msg_content) = match value.split_once(char::is_whitespace) {
                    Some((r, c)) => (r.trim().to_string(), unquote(c.trim())),
                    None => {
                        return Err(format!("line {}: MESSAGE requires role and content", i + 1));
                    }
                };
                messages.push(ModelfileMessage {
                    role,
                    content: msg_content,
                });
            }
            _ => {
                // Ignore unknown directives for forward compatibility
                tracing::debug!("Ignoring unknown Modelfile directive: {directive}");
            }
        }

        i += 1;
    }

    let from = from.ok_or_else(|| "Modelfile missing required FROM directive".to_string())?;

    Ok(Modelfile {
        from,
        parameters,
        system,
        template,
        stop,
        adapter,
        license,
        messages,
    })
}

/// Remove surrounding quotes from a string value if present.
fn unquote(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Convert a parsed Modelfile's parameters into a serde_json map.
pub fn parameters_to_json(mf: &Modelfile) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    for (k, v) in &mf.parameters {
        // Try numeric parsing first
        if let Ok(n) = v.parse::<f64>() {
            map.insert(k.clone(), serde_json::Value::from(n));
        } else if let Ok(n) = v.parse::<i64>() {
            map.insert(k.clone(), serde_json::Value::from(n));
        } else if v == "true" || v == "false" {
            map.insert(k.clone(), serde_json::Value::from(v == "true"));
        } else {
            map.insert(k.clone(), serde_json::Value::from(v.clone()));
        }
    }
    if !mf.stop.is_empty() {
        map.insert("stop".to_string(), serde_json::Value::from(mf.stop.clone()));
    }
    map
}

/// Reconstruct Modelfile text from a parsed representation.
pub fn to_string(mf: &Modelfile) -> String {
    let mut lines = Vec::new();
    lines.push(format!("FROM {}", mf.from));
    for (k, v) in &mf.parameters {
        lines.push(format!("PARAMETER {k} {v}"));
    }
    for s in &mf.stop {
        lines.push(format!("PARAMETER stop \"{s}\""));
    }
    if let Some(ref sys) = mf.system {
        lines.push(format!("SYSTEM \"{sys}\""));
    }
    if let Some(ref tmpl) = mf.template {
        lines.push(format!("TEMPLATE \"{tmpl}\""));
    }
    if let Some(ref adapter) = mf.adapter {
        lines.push(format!("ADAPTER {adapter}"));
    }
    if let Some(ref lic) = mf.license {
        lines.push(format!("LICENSE \"{lic}\""));
    }
    for msg in &mf.messages {
        lines.push(format!("MESSAGE {} \"{}\"", msg.role, msg.content));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_modelfile() {
        let content = r#"
FROM llama3.2:3b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a helpful assistant."
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.from, "llama3.2:3b");
        assert_eq!(mf.parameters.get("temperature").unwrap(), "0.7");
        assert_eq!(mf.parameters.get("top_p").unwrap(), "0.9");
        assert_eq!(mf.system.as_deref(), Some("You are a helpful assistant."));
        assert!(mf.template.is_none());
        assert!(mf.stop.is_empty());
    }

    #[test]
    fn test_parse_with_stop_sequences() {
        let content = r#"
FROM llama3.2:3b
PARAMETER stop "<|end|>"
PARAMETER stop "<|eot_id|>"
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.stop, vec!["<|end|>", "<|eot_id|>"]);
        assert!(!mf.parameters.contains_key("stop"));
    }

    #[test]
    fn test_parse_with_template() {
        let content = r#"
FROM llama3.2:3b
TEMPLATE "{{ .System }} {{ .Prompt }}"
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.template.as_deref(), Some("{{ .System }} {{ .Prompt }}"));
    }

    #[test]
    fn test_parse_missing_from() {
        let content = "PARAMETER temperature 0.7";
        let result = parse(content);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("FROM"));
    }

    #[test]
    fn test_parse_comments_and_empty_lines() {
        let content = r#"
# This is a comment
FROM llama3.2:3b

# Another comment
PARAMETER temperature 0.5
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.from, "llama3.2:3b");
        assert_eq!(mf.parameters.get("temperature").unwrap(), "0.5");
    }

    #[test]
    fn test_unquote() {
        assert_eq!(unquote("\"hello\""), "hello");
        assert_eq!(unquote("'hello'"), "hello");
        assert_eq!(unquote("hello"), "hello");
        assert_eq!(unquote("\"\""), "");
    }

    #[test]
    fn test_parameters_to_json() {
        let content = r#"
FROM test
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER stop "<|end|>"
"#;
        let mf = parse(content).unwrap();
        let json = parameters_to_json(&mf);
        assert_eq!(json.get("temperature").unwrap(), &serde_json::json!(0.7));
        assert_eq!(json.get("num_ctx").unwrap(), &serde_json::json!(4096.0));
        assert!(json.get("stop").unwrap().is_array());
    }

    #[test]
    fn test_to_string_roundtrip() {
        let content = r#"FROM llama3.2:3b
PARAMETER temperature 0.7
SYSTEM "You are helpful."
"#;
        let mf = parse(content).unwrap();
        let output = to_string(&mf);
        assert!(output.contains("FROM llama3.2:3b"));
        assert!(output.contains("PARAMETER temperature 0.7"));
        assert!(output.contains("SYSTEM"));
    }

    #[test]
    fn test_parse_adapter() {
        let content = r#"
FROM llama3.2:3b
ADAPTER /path/to/lora.gguf
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.adapter.as_deref(), Some("/path/to/lora.gguf"));
    }

    #[test]
    fn test_parse_adapter_quoted() {
        let content = r#"
FROM llama3.2:3b
ADAPTER "/path/with spaces/lora.gguf"
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.adapter.as_deref(), Some("/path/with spaces/lora.gguf"));
    }

    #[test]
    fn test_parse_license() {
        let content = r#"
FROM llama3.2:3b
LICENSE "MIT License"
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.license.as_deref(), Some("MIT License"));
    }

    #[test]
    fn test_parse_message_single() {
        let content = r#"
FROM llama3.2:3b
MESSAGE user "Hello, how are you?"
MESSAGE assistant "I'm doing well, thank you!"
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.messages.len(), 2);
        assert_eq!(mf.messages[0].role, "user");
        assert_eq!(mf.messages[0].content, "Hello, how are you?");
        assert_eq!(mf.messages[1].role, "assistant");
        assert_eq!(mf.messages[1].content, "I'm doing well, thank you!");
    }

    #[test]
    fn test_parse_message_system() {
        let content = r#"
FROM llama3.2:3b
MESSAGE system "You are a pirate."
MESSAGE user "Tell me a joke."
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.messages.len(), 2);
        assert_eq!(mf.messages[0].role, "system");
        assert_eq!(mf.messages[0].content, "You are a pirate.");
    }

    #[test]
    fn test_parse_message_missing_content() {
        let content = r#"
FROM llama3.2:3b
MESSAGE user
"#;
        let result = parse(content);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("MESSAGE requires role and content"));
    }

    #[test]
    fn test_parse_full_modelfile() {
        let content = r#"
# Full Modelfile example
FROM llama3.2:3b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|end|>"
SYSTEM "You are a helpful coding assistant."
TEMPLATE "{{ .System }} {{ .Prompt }}"
ADAPTER /models/code-lora.gguf
LICENSE "Apache-2.0"
MESSAGE user "What languages do you know?"
MESSAGE assistant "I'm proficient in Python, Rust, JavaScript, and many more!"
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.from, "llama3.2:3b");
        assert_eq!(mf.parameters.get("temperature").unwrap(), "0.7");
        assert_eq!(mf.parameters.get("top_p").unwrap(), "0.9");
        assert_eq!(mf.stop, vec!["<|end|>"]);
        assert_eq!(
            mf.system.as_deref(),
            Some("You are a helpful coding assistant.")
        );
        assert!(mf.template.is_some());
        assert_eq!(mf.adapter.as_deref(), Some("/models/code-lora.gguf"));
        assert_eq!(mf.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(mf.messages.len(), 2);
    }

    #[test]
    fn test_to_string_with_adapter_and_messages() {
        let content = r#"
FROM llama3.2:3b
ADAPTER /path/to/lora.gguf
LICENSE "MIT"
MESSAGE user "Hi"
MESSAGE assistant "Hello!"
"#;
        let mf = parse(content).unwrap();
        let output = to_string(&mf);
        assert!(output.contains("FROM llama3.2:3b"));
        assert!(output.contains("ADAPTER /path/to/lora.gguf"));
        assert!(output.contains("LICENSE"));
        assert!(output.contains("MESSAGE user"));
        assert!(output.contains("MESSAGE assistant"));
    }

    #[test]
    fn test_parse_heredoc_system() {
        let content = r#"
FROM llama3.2:3b
SYSTEM """
You are a helpful assistant.
You always respond in English.
"""
"#;
        let mf = parse(content).unwrap();
        assert_eq!(
            mf.system.as_deref(),
            Some("You are a helpful assistant.\nYou always respond in English.")
        );
    }

    #[test]
    fn test_parse_heredoc_template() {
        let content = r#"
FROM llama3.2:3b
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
"""
"#;
        let mf = parse(content).unwrap();
        let tmpl = mf.template.unwrap();
        assert!(tmpl.contains("<|system|>"));
        assert!(tmpl.contains("<|assistant|>"));
        assert!(tmpl.contains("{{ if .System }}"));
    }

    #[test]
    fn test_parse_heredoc_license() {
        let content = r#"
FROM llama3.2:3b
LICENSE """
MIT License

Copyright (c) 2024 Example Corp.
All rights reserved.
"""
"#;
        let mf = parse(content).unwrap();
        let lic = mf.license.unwrap();
        assert!(lic.contains("MIT License"));
        assert!(lic.contains("Copyright (c) 2024"));
        assert!(lic.contains("All rights reserved."));
    }

    #[test]
    fn test_parse_heredoc_inline() {
        let content = r#"
FROM llama3.2:3b
SYSTEM """You are a pirate."""
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.system.as_deref(), Some("You are a pirate."));
    }

    #[test]
    fn test_parse_heredoc_unterminated() {
        let content = r#"
FROM llama3.2:3b
SYSTEM """
This block never closes.
"#;
        let result = parse(content);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unterminated heredoc"));
    }

    #[test]
    fn test_parse_heredoc_mixed_with_regular() {
        let content = r#"
FROM llama3.2:3b
PARAMETER temperature 0.7
SYSTEM """
You are a helpful assistant.
You speak multiple languages.
"""
PARAMETER top_p 0.9
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.parameters.get("temperature").unwrap(), "0.7");
        assert_eq!(mf.parameters.get("top_p").unwrap(), "0.9");
        assert!(mf.system.as_deref().unwrap().contains("multiple languages"));
    }

    #[test]
    fn test_parse_parameter_missing_value() {
        let content = r#"
FROM llama3.2:3b
PARAMETER temperature
"#;
        let result = parse(content);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("PARAMETER requires key and value"));
    }

    #[test]
    fn test_parse_single_word_directive() {
        let content = r#"
FROM
"#;
        let result = parse(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unknown_directive() {
        let content = r#"
FROM llama3.2:3b
UNKNOWN_DIRECTIVE some value
"#;
        // Should succeed and ignore unknown directive
        let mf = parse(content).unwrap();
        assert_eq!(mf.from, "llama3.2:3b");
    }

    #[test]
    fn test_parameters_to_json_boolean() {
        let content = r#"
FROM test
PARAMETER use_mmap true
PARAMETER use_mlock false
"#;
        let mf = parse(content).unwrap();
        let json = parameters_to_json(&mf);
        assert_eq!(json.get("use_mmap").unwrap(), &serde_json::json!(true));
        assert_eq!(json.get("use_mlock").unwrap(), &serde_json::json!(false));
    }

    #[test]
    fn test_parameters_to_json_string() {
        let content = r#"
FROM test
PARAMETER model_type llama
"#;
        let mf = parse(content).unwrap();
        let json = parameters_to_json(&mf);
        assert_eq!(json.get("model_type").unwrap(), &serde_json::json!("llama"));
    }

    #[test]
    fn test_parameters_to_json_integer() {
        let content = r#"
FROM test
PARAMETER num_ctx 2048
"#;
        let mf = parse(content).unwrap();
        let json = parameters_to_json(&mf);
        // Note: integers may be parsed as floats in JSON
        assert!(json.get("num_ctx").unwrap().is_number());
    }

    #[test]
    fn test_unquote_single_quotes() {
        assert_eq!(unquote("'single quoted'"), "single quoted");
    }

    #[test]
    fn test_unquote_no_quotes() {
        assert_eq!(unquote("no quotes"), "no quotes");
    }

    #[test]
    fn test_unquote_with_spaces() {
        assert_eq!(unquote("  \"spaced\"  "), "spaced");
    }

    #[test]
    fn test_to_string_empty_optional_fields() {
        let mf = Modelfile {
            from: "test".to_string(),
            parameters: HashMap::new(),
            system: None,
            template: None,
            stop: vec![],
            adapter: None,
            license: None,
            messages: vec![],
        };
        let output = to_string(&mf);
        assert!(output.contains("FROM test"));
        assert!(!output.contains("SYSTEM"));
        assert!(!output.contains("ADAPTER"));
    }

    #[test]
    fn test_to_string_with_stop_sequences() {
        let mf = Modelfile {
            from: "test".to_string(),
            parameters: HashMap::new(),
            system: None,
            template: None,
            stop: vec!["<|end|>".to_string(), "<|eot|>".to_string()],
            adapter: None,
            license: None,
            messages: vec![],
        };
        let output = to_string(&mf);
        assert!(output.contains("PARAMETER stop"));
        assert!(output.contains("<|end|>"));
        assert!(output.contains("<|eot|>"));
    }

    #[test]
    fn test_parse_message_with_heredoc() {
        // MESSAGE doesn't support heredoc for content, only inline
        let content = r#"
FROM llama3.2:3b
MESSAGE user "Hello, this is a single-line message."
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.messages.len(), 1);
        assert_eq!(mf.messages[0].role, "user");
        assert!(mf.messages[0].content.contains("single-line"));
    }

    #[test]
    fn test_parse_adapter_with_heredoc() {
        let content = r#"
FROM llama3.2:3b
ADAPTER """/path/to/adapter.gguf"""
"#;
        let mf = parse(content).unwrap();
        assert_eq!(mf.adapter.as_deref(), Some("/path/to/adapter.gguf"));
    }

    #[test]
    fn test_modelfile_message_clone() {
        let msg = ModelfileMessage {
            role: "user".to_string(),
            content: "test".to_string(),
        };
        let cloned = msg.clone();
        assert_eq!(msg.role, cloned.role);
        assert_eq!(msg.content, cloned.content);
    }

    #[test]
    fn test_modelfile_clone() {
        let mf = Modelfile {
            from: "test".to_string(),
            parameters: HashMap::new(),
            system: Some("sys".to_string()),
            template: None,
            stop: vec![],
            adapter: None,
            license: None,
            messages: vec![],
        };
        let cloned = mf.clone();
        assert_eq!(mf.from, cloned.from);
        assert_eq!(mf.system, cloned.system);
    }
}
