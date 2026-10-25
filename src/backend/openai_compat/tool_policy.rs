//! Explicit tool selection and generated-call checks shared by proxy backends.
use crate::backend::types::{ChatRequest, ToolCall, ToolChoice};
use crate::error::{PowerError, Result};
use std::collections::{BTreeMap, BTreeSet};

const MAX_TOOL_BYTES: usize = 8 * 1024 * 1024;
#[derive(Debug)]
enum Selection {
    Auto,
    None,
    Required,
    Named(String),
}
#[derive(Default)]
struct Pending {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}
pub(crate) struct ChatToolPolicy {
    selection: Selection,
    offered: BTreeSet<String>,
    parallel: bool,
    calls: BTreeMap<u32, Pending>,
    bytes: usize,
}
impl ChatToolPolicy {
    pub(crate) fn from_request(request: &ChatRequest) -> Result<Self> {
        let mut offered = BTreeSet::new();
        for tool in request.tools.iter().flatten() {
            if tool.tool_type != "function"
                || tool.function.name.is_empty()
                || !offered.insert(tool.function.name.clone())
            {
                return Err(PowerError::InvalidRequest(
                    "tools require distinct nonempty function names".into(),
                ));
            }
        }
        let selection = match request.tool_choice.as_ref() {
            None => Selection::Auto,
            Some(ToolChoice::String(value)) => match value.as_str() {
                "auto" => Selection::Auto,
                "none" => Selection::None,
                "required" if !offered.is_empty() => Selection::Required,
                _ => {
                    return Err(PowerError::InvalidRequest(
                        "tool_choice must be auto, none, or required with offered tools".into(),
                    ))
                }
            },
            Some(ToolChoice::Specific(choice))
                if choice.tool_type == "function" && offered.contains(&choice.function.name) =>
            {
                Selection::Named(choice.function.name.clone())
            }
            _ => {
                return Err(PowerError::InvalidRequest(
                    "tool_choice must name an offered function".into(),
                ))
            }
        };
        Ok(Self {
            selection,
            offered,
            parallel: request.parallel_tool_calls != Some(false),
            calls: BTreeMap::new(),
            bytes: 0,
        })
    }
    pub(crate) fn observe(&mut self, calls: &[ToolCall]) -> Result<()> {
        if matches!(self.selection, Selection::None) && !calls.is_empty() {
            return Err(violation("tool_choice=none produced a call"));
        }
        for call in calls {
            let index = call
                .index
                .ok_or_else(|| violation("tool delta omitted its index"))?;
            if !self.calls.contains_key(&index) {
                if self.calls.len() >= 128 || (!self.parallel && !self.calls.is_empty()) {
                    return Err(violation(
                        "generated call count violates parallel_tool_calls policy",
                    ));
                }
                if !call.id.is_empty() && self.calls.values().any(|previous| previous.id == call.id)
                {
                    return Err(violation("tool id is shared by different indices"));
                }
            }
            let added = call
                .id
                .len()
                .saturating_add(call.tool_type.len())
                .saturating_add(call.function.name.len())
                .saturating_add(call.function.arguments.len());
            self.bytes = self.bytes.saturating_add(added);
            if self.bytes > MAX_TOOL_BYTES {
                return Err(violation("generated tool data exceeds 8 MiB"));
            }
            let pending = self.calls.entry(index).or_default();
            for (old, new) in [
                (&mut pending.id, &call.id),
                (&mut pending.kind, &call.tool_type),
            ] {
                if !new.is_empty() {
                    if !old.is_empty() && old != new {
                        return Err(violation("tool identity changed during generation"));
                    }
                    if old.is_empty() {
                        old.clone_from(new);
                    }
                }
            }
            pending.name.push_str(&call.function.name);
            pending.arguments.push_str(&call.function.arguments);
        }
        Ok(())
    }
    pub(crate) fn finish(&self, reason: &str) -> Result<()> {
        if self.calls.is_empty() {
            if matches!(self.selection, Selection::Required | Selection::Named(_))
                || matches!(reason, "tool_calls" | "function_call")
            {
                return Err(violation("required tool call was not completed"));
            }
            return Ok(());
        }
        if !matches!(reason, "tool_calls" | "function_call") {
            return Err(violation("generated tool calls lack a tool finish reason"));
        }
        let mut identities = BTreeSet::new();
        for (expected, (index, call)) in self.calls.iter().enumerate() {
            if *index as usize != expected
                || call.id.is_empty()
                || call.kind != "function"
                || !identities.insert(&call.id)
            {
                return Err(violation("incomplete or duplicate generated tool identity"));
            }
            if !self.offered.contains(&call.name)
                || matches!(&self.selection,Selection::Named(name) if name!=&call.name)
            {
                return Err(violation(
                    "generated tool does not match the offered selection",
                ));
            }
            if !serde_json::from_str::<serde_json::Value>(&call.arguments)
                .is_ok_and(|args| args.is_object())
            {
                return Err(violation(
                    "generated tool arguments are not a complete JSON object",
                ));
            }
        }
        Ok(())
    }
}
fn violation(message: &str) -> PowerError {
    PowerError::InferenceFailed(format!("OpenAI upstream tool protocol error: {message}"))
}

#[cfg(test)]
#[path = "tool_policy_tests.rs"]
mod tests;
