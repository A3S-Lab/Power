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
pub fn parse(content: &str) -> Result<Modelfile, String> {
    let mut from: Option<String> = None;
    let mut parameters = HashMap::new();
    let mut system: Option<String> = None;
    let mut template: Option<String> = None;
    let mut stop = Vec::new();
    let mut adapter: Option<String> = None;
    let mut license: Option<String> = None;
    let mut messages = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Split into directive and value
        let (directive, value) = match line.split_once(char::is_whitespace) {
            Some((d, v)) => (d.to_uppercase(), v.trim().to_string()),
            None => {
                // Single-word directives are not valid
                return Err(format!(
                    "line {}: expected directive and value, got '{line}'",
                    line_num + 1
                ));
            }
        };

        match directive.as_str() {
            "FROM" => {
                from = Some(value);
            }
            "PARAMETER" => {
                let (key, val) = match value.split_once(char::is_whitespace) {
                    Some((k, v)) => (k.trim().to_string(), v.trim().to_string()),
                    None => {
                        return Err(format!(
                            "line {}: PARAMETER requires key and value",
                            line_num + 1
                        ));
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
                let (role, content) = match value.split_once(char::is_whitespace) {
                    Some((r, c)) => (r.trim().to_string(), unquote(c.trim())),
                    None => {
                        return Err(format!(
                            "line {}: MESSAGE requires role and content",
                            line_num + 1
                        ));
                    }
                };
                messages.push(ModelfileMessage { role, content });
            }
            _ => {
                // Ignore unknown directives for forward compatibility
                tracing::debug!("Ignoring unknown Modelfile directive: {directive}");
            }
        }
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
        assert!(mf.parameters.get("stop").is_none());
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
}
