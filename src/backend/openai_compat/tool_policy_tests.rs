#![cfg(test)]
use super::*;
use crate::backend::types::FunctionCall;
use serde_json::json;
fn request(choice: serde_json::Value, parallel: bool) -> ChatRequest {
    serde_json::from_value(
        json!({"messages":[{"role":"user","content":"Read record 17"}],
        "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
        "tool_choice":choice,"parallel_tool_calls":parallel}),
    )
    .unwrap()
}
fn call(index: u32, id: &str, name: &str, args: &str) -> ToolCall {
    ToolCall {
        index: Some(index),
        id: id.into(),
        tool_type: if id.is_empty() {
            "".into()
        } else {
            "function".into()
        },
        function: FunctionCall {
            name: name.into(),
            arguments: args.into(),
        },
    }
}
#[test]
fn named_selection_accepts_fragmented_complete_offered_call() {
    let mut policy = ChatToolPolicy::from_request(&request(
        json!({"type":"function","function":{"name":"lookup"}}),
        false,
    ))
    .unwrap();
    policy
        .observe(&[call(0, "call-a", "look", "{\"id\":")])
        .unwrap();
    policy.observe(&[call(0, "", "up", "17}")]).unwrap();
    policy.finish("tool_calls").unwrap();
}
#[test]
fn unavailable_names_unknown_choices_and_duplicate_tools_refuse_before_startup() {
    for choice in [
        json!("unknown"),
        json!({"type":"function","function":{"name":"unoffered"}}),
    ] {
        assert!(ChatToolPolicy::from_request(&request(choice, false)).is_err());
    }
    let mut req = request(json!("required"), false);
    req.tools = None;
    assert!(ChatToolPolicy::from_request(&req).is_err());
    let mut req = request(json!("auto"), false);
    let duplicate = req.tools.as_ref().unwrap()[0].clone();
    req.tools.as_mut().unwrap().push(duplicate);
    assert!(ChatToolPolicy::from_request(&req).is_err());
}
#[test]
fn none_and_single_tool_policy_reject_calls_before_successful_finish() {
    let mut none = ChatToolPolicy::from_request(&request(json!("none"), false)).unwrap();
    assert!(none.observe(&[call(0, "call-a", "lookup", "{}")]).is_err());
    let mut one = ChatToolPolicy::from_request(&request(json!("auto"), false)).unwrap();
    one.observe(&[call(0, "call-a", "lookup", "{}")]).unwrap();
    assert!(one.observe(&[call(1, "call-b", "lookup", "{}")]).is_err());
}
#[test]
fn wrong_function_incomplete_json_and_missing_required_call_cannot_complete() {
    for (name, args, reason) in [
        ("unoffered", "{}", "tool_calls"),
        ("lookup", "{", "tool_calls"),
        ("lookup", "[]", "tool_calls"),
        ("lookup", "{}", "stop"),
    ] {
        let mut policy = ChatToolPolicy::from_request(&request(json!("auto"), true)).unwrap();
        policy.observe(&[call(0, "call-a", name, args)]).unwrap();
        assert!(policy.finish(reason).is_err());
    }
    let required = ChatToolPolicy::from_request(&request(json!("required"), true)).unwrap();
    assert!(required.finish("stop").is_err());
    let auto = ChatToolPolicy::from_request(&request(json!("auto"), true)).unwrap();
    auto.finish("stop").unwrap();
    assert!(auto.finish("tool_calls").is_err());
}
#[test]
fn identity_conflicts_and_unbounded_fragmented_arguments_refuse() {
    let mut policy = ChatToolPolicy::from_request(&request(json!("auto"), true)).unwrap();
    policy.observe(&[call(0, "call-a", "lookup", "{")]).unwrap();
    assert!(policy.observe(&[call(0, "changed", "", "}")]).is_err());
    let mut policy = ChatToolPolicy::from_request(&request(json!("auto"), true)).unwrap();
    assert!(policy
        .observe(&[call(0, "call-a", "lookup", &"x".repeat(MAX_TOOL_BYTES))])
        .is_err());
}
