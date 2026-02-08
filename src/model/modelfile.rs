// Modelfile parser for Ollama-compatible model definitions.
//
// Parses the Modelfile format with directives:
// FROM <base-model>
// PARAMETER <key> <value>
// SYSTEM "<prompt>"
// TEMPLATE "<template>"

use std::collections::HashMap;

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
}

/// Parse a Modelfile from its text content.
pub fn parse(content: &str) -> Result<Modelfile, String> {
    let mut from: Option<String> = None;
    let mut parameters = HashMap::new();
    let mut system: Option<String> = None;
    let mut template: Option<String> = None;
    let mut stop = Vec::new();

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
}
