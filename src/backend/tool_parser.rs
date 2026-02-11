/// Tool call output parser.
///
/// Parses model-generated text to detect and extract structured tool calls.
/// Supports multiple common formats used by LLMs for function calling:
///
/// 1. **JSON object** with `name` and `arguments` fields
/// 2. **`<tool_call>` XML tags** (used by Hermes, Qwen, etc.)
/// 3. **`<functioncall>` tags** (used by some fine-tuned models)
/// 4. **`[TOOL_CALLS]` prefix** (used by Mistral-style models)
use super::types::{FunctionCall, ToolCall};

/// Attempt to parse tool calls from model-generated text.
///
/// Returns `Some(Vec<ToolCall>)` if tool calls were detected, `None` otherwise.
/// This is a best-effort parser — it tries multiple formats and returns the
/// first successful parse. Each tool call is assigned an index for OpenAI
/// streaming delta compatibility.
pub fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let trimmed = text.trim();

    // Try XML-style <tool_call> tags first (most common in fine-tuned models)
    if let Some(calls) = parse_xml_tool_calls(trimmed) {
        if !calls.is_empty() {
            return Some(assign_indices(calls));
        }
    }

    // Try [TOOL_CALLS] prefix (Mistral-style)
    if let Some(calls) = parse_mistral_tool_calls(trimmed) {
        if !calls.is_empty() {
            return Some(assign_indices(calls));
        }
    }

    // Try raw JSON object with name/arguments
    if let Some(mut call) = parse_json_tool_call(trimmed) {
        call.index = Some(0);
        return Some(vec![call]);
    }

    None
}

/// Assign sequential indices to tool calls for OpenAI streaming format.
fn assign_indices(mut calls: Vec<ToolCall>) -> Vec<ToolCall> {
    for (i, call) in calls.iter_mut().enumerate() {
        call.index = Some(i as u32);
    }
    calls
}

/// Parse `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` format.
///
/// Used by Hermes, Qwen, and many fine-tuned models.
fn parse_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    loop {
        let start_tag = "<tool_call>";
        let end_tag = "</tool_call>";

        let start = text[search_from..].find(start_tag)?;
        let content_start = search_from + start + start_tag.len();
        let end = text[content_start..].find(end_tag)?;
        let content = text[content_start..content_start + end].trim();

        if let Some(call) = parse_json_tool_call(content) {
            calls.push(call);
        }

        search_from = content_start + end + end_tag.len();
        if search_from >= text.len() {
            break;
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Parse `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]` format.
///
/// Used by Mistral-style models.
fn parse_mistral_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let prefix = "[TOOL_CALLS]";
    let idx = text.find(prefix)?;
    let after = text[idx + prefix.len()..].trim();

    // Expect a JSON array
    let parsed: Vec<serde_json::Value> = serde_json::from_str(after).ok()?;

    let calls: Vec<ToolCall> = parsed
        .into_iter()
        .filter_map(|v| {
            let name = v.get("name")?.as_str()?.to_string();
            let arguments = if let Some(args) = v.get("arguments") {
                if args.is_string() {
                    args.as_str().unwrap().to_string()
                } else {
                    serde_json::to_string(args).ok()?
                }
            } else {
                "{}".to_string()
            };
            Some(ToolCall {
                id: generate_call_id(),
                tool_type: "function".to_string(),
                function: FunctionCall { name, arguments },
                index: None,
            })
        })
        .collect();

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Parse a single JSON object as a tool call.
///
/// Expects `{"name": "function_name", "arguments": {...}}` or
/// `{"function": {"name": "...", "arguments": "..."}}`.
fn parse_json_tool_call(text: &str) -> Option<ToolCall> {
    // Find the first '{' and last '}' to extract JSON
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if start >= end {
        return None;
    }
    let json_str = &text[start..=end];

    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;

    // Format 1: {"name": "...", "arguments": {...}}
    if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
        let arguments = if let Some(args) = v.get("arguments") {
            if args.is_string() {
                args.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(args).ok()?
            }
        } else {
            "{}".to_string()
        };
        return Some(ToolCall {
            id: generate_call_id(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments,
            },
            index: None,
        });
    }

    // Format 2: {"function": {"name": "...", "arguments": "..."}}
    if let Some(func) = v.get("function") {
        let name = func.get("name")?.as_str()?.to_string();
        let arguments = if let Some(args) = func.get("arguments") {
            if args.is_string() {
                args.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(args).ok()?
            }
        } else {
            "{}".to_string()
        };
        return Some(ToolCall {
            id: generate_call_id(),
            tool_type: "function".to_string(),
            function: FunctionCall { name, arguments },
            index: None,
        });
    }

    None
}

/// Generate a unique call ID for tool calls.
fn generate_call_id() -> String {
    format!(
        "call_{}",
        &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xml_single_tool_call() {
        let text = r#"<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("San Francisco"));
        assert_eq!(calls[0].tool_type, "function");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_parse_xml_multiple_tool_calls() {
        let text = r#"<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>
<tool_call>{"name": "get_time", "arguments": {"timezone": "PST"}}</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn test_parse_xml_with_surrounding_text() {
        let text = r#"I'll check the weather for you.
<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>
Let me know if you need anything else."#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_mistral_tool_calls() {
        let text = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Paris"));
    }

    #[test]
    fn test_parse_mistral_multiple_calls() {
        let text = r#"[TOOL_CALLS] [{"name": "search", "arguments": {"query": "rust"}}, {"name": "calculate", "arguments": {"expr": "2+2"}}]"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[1].function.name, "calculate");
    }

    #[test]
    fn test_parse_raw_json_tool_call() {
        let text = r#"{"name": "get_weather", "arguments": {"location": "London"}}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("London"));
    }

    #[test]
    fn test_parse_json_with_function_wrapper() {
        let text = r#"{"function": {"name": "search", "arguments": "{\"query\": \"hello\"}"}}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_parse_json_arguments_as_string() {
        let text = r#"{"name": "calc", "arguments": "{\"x\": 1, \"y\": 2}"}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(calls[0].function.arguments, r#"{"x": 1, "y": 2}"#);
    }

    #[test]
    fn test_parse_json_no_arguments() {
        let text = r#"{"name": "get_time"}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.name, "get_time");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn test_parse_no_tool_calls_in_plain_text() {
        let text = "The weather in San Francisco is 72°F and sunny.";
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_no_tool_calls_in_empty_string() {
        let calls = parse_tool_calls("");
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_no_tool_calls_in_regular_json() {
        // JSON without name/arguments should not be parsed as tool call
        let text = r#"{"temperature": 72, "unit": "fahrenheit"}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_xml_with_whitespace() {
        let text = r#"<tool_call>
  {"name": "get_weather", "arguments": {"location": "Tokyo"}}
</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        assert_eq!(calls.unwrap()[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_json_embedded_in_text() {
        let text = r#"Sure, let me call the function: {"name": "search", "arguments": {"q": "test"}} for you."#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        assert_eq!(calls.unwrap()[0].function.name, "search");
    }

    #[test]
    fn test_call_id_format() {
        let id = generate_call_id();
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 29); // "call_" (5) + 24 hex chars
    }

    #[test]
    fn test_call_ids_are_unique() {
        let id1 = generate_call_id();
        let id2 = generate_call_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_parse_xml_invalid_json_inside_tags() {
        let text = r#"<tool_call>not valid json</tool_call>"#;
        let calls = parse_tool_calls(text);
        // Should return None because the JSON inside is invalid
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_mistral_invalid_json() {
        let text = r#"[TOOL_CALLS] not an array"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_xml_mixed_valid_invalid() {
        let text = r#"<tool_call>{"name": "valid", "arguments": {}}</tool_call>
<tool_call>invalid json</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        // Only the valid one should be parsed
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "valid");
    }

    #[test]
    fn test_parse_json_arguments_as_object() {
        let text = r#"{"name": "test", "arguments": {"key": "value"}}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.name, "test");
        assert!(calls[0].function.arguments.contains("key"));
    }

    #[test]
    fn test_parse_mistral_with_prefix_text() {
        let text = r#"Here are the tool calls: [TOOL_CALLS] [{"name": "func", "arguments": {}}]"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        assert_eq!(calls.unwrap()[0].function.name, "func");
    }

    #[test]
    fn test_parse_mistral_empty_array() {
        let text = r#"[TOOL_CALLS] []"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_mistral_arguments_as_object() {
        let text = r#"[TOOL_CALLS] [{"name": "test", "arguments": {"x": 1}}]"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.name, "test");
        assert!(calls[0].function.arguments.contains("x"));
    }

    #[test]
    fn test_parse_mistral_no_arguments() {
        let text = r#"[TOOL_CALLS] [{"name": "test"}]"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn test_parse_json_with_extra_fields() {
        let text = r#"{"name": "test", "arguments": {}, "extra": "ignored"}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        assert_eq!(calls.unwrap()[0].function.name, "test");
    }

    #[test]
    fn test_parse_json_function_wrapper_no_arguments() {
        let text = r#"{"function": {"name": "test"}}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.name, "test");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn test_parse_json_function_wrapper_arguments_as_object() {
        let text = r#"{"function": {"name": "test", "arguments": {"a": 1}}}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_some());
        let calls = calls.unwrap();
        assert!(calls[0].function.arguments.contains("a"));
    }

    #[test]
    fn test_parse_xml_no_closing_tag() {
        let text = r#"<tool_call>{"name": "test", "arguments": {}}"#;
        let calls = parse_tool_calls(text);
        // Should fall back to JSON parsing since XML is incomplete
        assert!(calls.is_some());
        assert_eq!(calls.unwrap()[0].function.name, "test");
    }

    #[test]
    fn test_parse_xml_no_opening_tag() {
        let text = r#"{"name": "test", "arguments": {}}</tool_call>"#;
        let calls = parse_tool_calls(text);
        // Should fall back to JSON parsing
        assert!(calls.is_some());
        assert_eq!(calls.unwrap()[0].function.name, "test");
    }

    #[test]
    fn test_parse_json_malformed_braces() {
        let text = r#"{"name": "test""#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_json_only_opening_brace() {
        let text = r#"{"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_json_only_closing_brace() {
        let text = r#"}"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_json_reversed_braces() {
        let text = r#"}{"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_mistral_missing_name() {
        let text = r#"[TOOL_CALLS] [{"arguments": {"x": 1}}]"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_xml_empty_content() {
        let text = r#"<tool_call></tool_call>"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_xml_whitespace_only() {
        let text = r#"<tool_call>   </tool_call>"#;
        let calls = parse_tool_calls(text);
        assert!(calls.is_none());
    }

    #[test]
    fn test_tool_calls_have_sequential_indices() {
        let text = r#"<tool_call>{"name": "a", "arguments": {}}</tool_call>
<tool_call>{"name": "b", "arguments": {}}</tool_call>
<tool_call>{"name": "c", "arguments": {}}</tool_call>"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[1].index, Some(1));
        assert_eq!(calls[2].index, Some(2));
    }

    #[test]
    fn test_single_tool_call_has_index_zero() {
        let text = r#"{"name": "test", "arguments": {}}"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].index, Some(0));
    }

    #[test]
    fn test_mistral_tool_calls_have_indices() {
        let text = r#"[TOOL_CALLS] [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[1].index, Some(1));
    }
}
