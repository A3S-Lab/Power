/// JSON Schema to GBNF grammar converter.
///
/// Converts a JSON Schema object into a GBNF (GGML BNF) grammar string
/// that constrains llama.cpp output to match the schema structure.
///
/// Supports: object, array, string, number, integer, boolean, null,
/// enum values, required fields, and nested schemas.
use serde_json::Value;

/// Convert a JSON Schema to a GBNF grammar string.
///
/// Returns a grammar where `root` matches the top-level schema.
pub fn schema_to_gbnf(schema: &Value) -> String {
    let mut rules = Vec::new();
    let mut ctx = GbnfContext::new();

    let root_rule = ctx.generate_rule("root", schema);
    rules.push(root_rule);
    rules.extend(ctx.extra_rules);

    // Add common primitives
    rules.push(WS_RULE.to_string());
    rules.push(STRING_RULE.to_string());
    rules.push(NUMBER_RULE.to_string());
    rules.push(INTEGER_RULE.to_string());
    rules.push(BOOLEAN_RULE.to_string());
    rules.push(NULL_RULE.to_string());
    rules.push(VALUE_RULE.to_string());

    rules.join("\n")
}

/// Fallback GBNF grammar for unconstrained JSON output.
pub const GENERIC_JSON_GRAMMAR: &str = r#"
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^\\"\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
"#;

const WS_RULE: &str = r#"ws ::= ([ \t\n] ws)?"#;

const STRING_RULE: &str = r#"string ::= "\"" ([^\\"\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws"#;

const NUMBER_RULE: &str = r#"number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)? ws"#;

const INTEGER_RULE: &str = r#"integer ::= ("-"? ([0-9] | [1-9] [0-9]*)) ws"#;

const BOOLEAN_RULE: &str = r#"boolean ::= ("true" | "false") ws"#;

const NULL_RULE: &str = r#"null ::= "null" ws"#;

const VALUE_RULE: &str = r#"value ::= object | array | string | number | boolean | null"#;

struct GbnfContext {
    counter: usize,
    extra_rules: Vec<String>,
}

impl GbnfContext {
    fn new() -> Self {
        Self {
            counter: 0,
            extra_rules: Vec::new(),
        }
    }

    fn next_name(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    fn generate_rule(&mut self, name: &str, schema: &Value) -> String {
        // Handle enum constraint
        if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
            let alternatives: Vec<String> = enum_values
                .iter()
                .map(|v| match v {
                    Value::String(s) => format!("\"\\\"{}\\\"\"", escape_gbnf(s)),
                    Value::Number(n) => format!("\"{}\"", n),
                    Value::Bool(b) => format!("\"{}\"", b),
                    Value::Null => "\"null\"".to_string(),
                    _ => "value".to_string(),
                })
                .collect();
            return format!("{name} ::= ({}) ws", alternatives.join(" | "));
        }

        let type_str = schema
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("object");

        match type_str {
            "object" => self.generate_object_rule(name, schema),
            "array" => self.generate_array_rule(name, schema),
            "string" => format!("{name} ::= string"),
            "number" => format!("{name} ::= number"),
            "integer" => format!("{name} ::= integer"),
            "boolean" => format!("{name} ::= boolean"),
            "null" => format!("{name} ::= null"),
            _ => format!("{name} ::= value"),
        }
    }

    fn generate_object_rule(&mut self, name: &str, schema: &Value) -> String {
        let properties = schema.get("properties").and_then(|v| v.as_object());
        let required: Vec<&str> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let properties = match properties {
            Some(props) if !props.is_empty() => props,
            _ => {
                // No properties defined — allow any JSON object
                return format!(
                    "{name} ::= \"{{\" ws (string \":\" ws value (\",\" ws string \":\" ws value)*)? \"}}\" ws"
                );
            }
        };

        // Generate rules for each property value
        let mut prop_parts = Vec::new();
        for (key, prop_schema) in properties {
            let prop_rule_name = self.next_name(&format!("{name}-"));
            let rule = self.generate_rule(&prop_rule_name, prop_schema);
            self.extra_rules.push(rule);
            prop_parts.push((key.clone(), prop_rule_name));
        }

        // Build the object rule with required/optional fields
        // For simplicity, emit all properties in order (required first, then optional)
        let mut field_exprs = Vec::new();
        let mut optional_exprs = Vec::new();

        for (key, rule_name) in &prop_parts {
            let field_expr = format!("\"\\\"{}\\\"\" \":\" ws {}", escape_gbnf(key), rule_name);
            if required.contains(&key.as_str()) {
                field_exprs.push(field_expr);
            } else {
                optional_exprs.push(field_expr);
            }
        }

        if field_exprs.is_empty() && optional_exprs.is_empty() {
            return format!("{name} ::= \"{{\" ws \"}}\" ws");
        }

        // Build: required fields separated by commas, then optional fields
        let mut parts = Vec::new();
        for (i, expr) in field_exprs.iter().enumerate() {
            if i == 0 {
                parts.push(expr.clone());
            } else {
                parts.push(format!("\",\" ws {expr}"));
            }
        }

        for expr in &optional_exprs {
            if parts.is_empty() {
                parts.push(format!("({expr})?"));
            } else {
                parts.push(format!("(\",\" ws {expr})?"));
            }
        }

        let body = parts.join(" ");
        format!("{name} ::= \"{{\" ws {body} \"}}\" ws")
    }

    fn generate_array_rule(&mut self, name: &str, schema: &Value) -> String {
        let items = schema.get("items");

        match items {
            Some(item_schema) => {
                let item_rule_name = self.next_name(&format!("{name}-item"));
                let rule = self.generate_rule(&item_rule_name, item_schema);
                self.extra_rules.push(rule);
                format!(
                    "{name} ::= \"[\" ws ({item_rule_name} (\",\" ws {item_rule_name})*)? \"]\" ws"
                )
            }
            None => {
                format!("{name} ::= \"[\" ws (value (\",\" ws value)*)? \"]\" ws")
            }
        }
    }
}

/// Escape special characters for GBNF string literals.
fn escape_gbnf(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Parse a `format` field value into a GBNF grammar string.
///
/// Accepts:
/// - `"json"` → generic JSON grammar
/// - `{"type": "object", ...}` → schema-specific grammar
/// - `{"type": "json_object"}` → generic JSON grammar (OpenAI compat)
///
/// Returns `None` if no grammar constraint should be applied.
pub fn format_to_gbnf(format: &Value) -> Option<String> {
    match format {
        Value::String(s) if s == "json" => Some(GENERIC_JSON_GRAMMAR.to_string()),
        Value::Object(obj) => {
            let type_val = obj.get("type").and_then(|v| v.as_str());
            match type_val {
                Some("json_object") => Some(GENERIC_JSON_GRAMMAR.to_string()),
                Some("object") | Some("array") | Some("string") | Some("number")
                | Some("integer") | Some("boolean") => Some(schema_to_gbnf(format)),
                _ => {
                    // If it has "properties", treat as object schema
                    if obj.contains_key("properties") {
                        Some(schema_to_gbnf(format))
                    } else {
                        None
                    }
                }
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_generic_json_grammar_is_valid() {
        let grammar = GENERIC_JSON_GRAMMAR;
        assert!(grammar.contains("root"));
        assert!(grammar.contains("object"));
        assert!(grammar.contains("string"));
        assert!(grammar.contains("number"));
    }

    #[test]
    fn test_schema_to_gbnf_simple_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("name"));
        assert!(grammar.contains("age"));
        assert!(grammar.contains("string"));
        assert!(grammar.contains("integer"));
    }

    #[test]
    fn test_schema_to_gbnf_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("address"));
        assert!(grammar.contains("street"));
        assert!(grammar.contains("city"));
    }

    #[test]
    fn test_schema_to_gbnf_array() {
        let schema = json!({
            "type": "array",
            "items": {"type": "string"}
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("["));
        assert!(grammar.contains("]"));
        assert!(grammar.contains("string"));
    }

    #[test]
    fn test_schema_to_gbnf_enum() {
        let schema = json!({
            "type": "string",
            "enum": ["red", "green", "blue"]
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("red"));
        assert!(grammar.contains("green"));
        assert!(grammar.contains("blue"));
    }

    #[test]
    fn test_schema_to_gbnf_boolean() {
        let schema = json!({"type": "boolean"});
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::= boolean"));
    }

    #[test]
    fn test_schema_to_gbnf_number() {
        let schema = json!({"type": "number"});
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::= number"));
    }

    #[test]
    fn test_schema_to_gbnf_null() {
        let schema = json!({"type": "null"});
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::= null"));
    }

    #[test]
    fn test_schema_to_gbnf_empty_object() {
        let schema = json!({"type": "object"});
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("{"));
    }

    #[test]
    fn test_schema_to_gbnf_array_without_items() {
        let schema = json!({"type": "array"});
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("value"));
    }

    #[test]
    fn test_format_to_gbnf_string_json() {
        let result = format_to_gbnf(&json!("json"));
        assert!(result.is_some());
        assert!(result.unwrap().contains("root"));
    }

    #[test]
    fn test_format_to_gbnf_json_object_type() {
        let result = format_to_gbnf(&json!({"type": "json_object"}));
        assert!(result.is_some());
    }

    #[test]
    fn test_format_to_gbnf_schema_object() {
        let result = format_to_gbnf(&json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }));
        assert!(result.is_some());
        assert!(result.unwrap().contains("name"));
    }

    #[test]
    fn test_format_to_gbnf_unknown_returns_none() {
        let result = format_to_gbnf(&json!("text"));
        assert!(result.is_none());
    }

    #[test]
    fn test_format_to_gbnf_null_returns_none() {
        let result = format_to_gbnf(&Value::Null);
        assert!(result.is_none());
    }

    #[test]
    fn test_format_to_gbnf_with_properties_no_type() {
        // If it has "properties" but no "type", treat as object schema
        let result = format_to_gbnf(&json!({
            "properties": {
                "x": {"type": "number"}
            }
        }));
        assert!(result.is_some());
    }

    #[test]
    fn test_schema_to_gbnf_required_fields_first() {
        let schema = json!({
            "type": "object",
            "properties": {
                "optional_field": {"type": "string"},
                "required_field": {"type": "integer"}
            },
            "required": ["required_field"]
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("required_field"));
        assert!(grammar.contains("optional_field"));
    }

    #[test]
    fn test_schema_to_gbnf_enum_with_numbers() {
        let schema = json!({
            "enum": [1, 2, 3]
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("\"1\""));
        assert!(grammar.contains("\"2\""));
        assert!(grammar.contains("\"3\""));
    }

    #[test]
    fn test_schema_to_gbnf_enum_with_booleans() {
        let schema = json!({
            "enum": [true, false]
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("\"true\""));
        assert!(grammar.contains("\"false\""));
    }

    #[test]
    fn test_escape_gbnf() {
        assert_eq!(escape_gbnf("hello"), "hello");
        assert_eq!(escape_gbnf("he\"llo"), "he\\\"llo");
        assert_eq!(escape_gbnf("he\\llo"), "he\\\\llo");
    }

    #[test]
    fn test_schema_to_gbnf_complex_nested() {
        let schema = json!({
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "active": {"type": "boolean"}
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["users"]
        });
        let grammar = schema_to_gbnf(&schema);
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("users"));
        assert!(grammar.contains("name"));
        assert!(grammar.contains("active"));
        assert!(grammar.contains("boolean"));
    }
}
