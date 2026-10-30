use std::collections::BTreeMap;

use magnitude_chat::output::{Completion, OutputEvent, Termination};
use magnitude_engine::chat::SessionEvent;

use super::super::types::{ChatResponseChunk, FunctionCall, ToolCall};
use crate::error::{PowerError, Result};

#[derive(Default)]
pub(super) struct StreamAssembler {
    open_calls: BTreeMap<usize, OpenCall>,
    terminal: bool,
}

struct OpenCall;

impl StreamAssembler {
    pub(super) fn apply(&mut self, event: SessionEvent) -> Result<Vec<ChatResponseChunk>> {
        if self.terminal {
            return Err(PowerError::InferenceFailed(
                "magnitude emitted an event after its terminal outcome".into(),
            ));
        }
        match event {
            SessionEvent::Progress(_) | SessionEvent::Admitted { .. } => Ok(Vec::new()),
            SessionEvent::Output { event, .. } => self.output(event),
            SessionEvent::Completed(completion) => {
                let chunk = self.complete(completion)?;
                self.terminal = true;
                Ok(vec![chunk])
            }
            SessionEvent::Failed(error) => {
                self.terminal = true;
                Err(PowerError::InferenceFailed(error.to_string()))
            }
        }
    }

    pub(super) fn finish(self) -> Result<()> {
        if self.terminal {
            Ok(())
        } else {
            Err(PowerError::InferenceFailed(
                "magnitude chat stream ended without a terminal outcome".into(),
            ))
        }
    }

    fn output(&mut self, event: OutputEvent) -> Result<Vec<ChatResponseChunk>> {
        match event {
            OutputEvent::Started => Ok(Vec::new()),
            OutputEvent::ReasoningDelta(text) => Ok(vec![ChatResponseChunk {
                thinking_content: Some(text),
                ..empty_chunk()
            }]),
            OutputEvent::TextDelta(text) => Ok(vec![ChatResponseChunk {
                content: text,
                ..empty_chunk()
            }]),
            OutputEvent::ToolCallStarted { index, id, name } => {
                let stream_index = u32::try_from(index).map_err(|_| {
                    PowerError::InferenceFailed("magnitude tool call index exceeds u32".into())
                })?;
                if self.open_calls.insert(index, OpenCall).is_some() {
                    return Err(PowerError::InferenceFailed(format!(
                        "magnitude started tool call index {index} twice"
                    )));
                }
                Ok(vec![ChatResponseChunk {
                    tool_calls: Some(vec![ToolCall {
                        id,
                        tool_type: "function".into(),
                        function: FunctionCall {
                            name,
                            arguments: String::new(),
                        },
                        index: Some(stream_index),
                    }]),
                    ..empty_chunk()
                }])
            }
            OutputEvent::ToolInputDelta { index, fragment } => {
                if !self.open_calls.contains_key(&index) {
                    return Err(PowerError::InferenceFailed(format!(
                        "magnitude emitted arguments for unopened tool call index {index}"
                    )));
                }
                let stream_index = u32::try_from(index).map_err(|_| {
                    PowerError::InferenceFailed("magnitude tool call index exceeds u32".into())
                })?;
                Ok(vec![ChatResponseChunk {
                    tool_calls: Some(vec![ToolCall {
                        id: String::new(),
                        tool_type: "function".into(),
                        function: FunctionCall {
                            name: String::new(),
                            arguments: fragment,
                        },
                        index: Some(stream_index),
                    }]),
                    ..empty_chunk()
                }])
            }
            OutputEvent::ToolCallFinished { index } => {
                self.open_calls.remove(&index).ok_or_else(|| {
                    PowerError::InferenceFailed(format!(
                        "magnitude finished unopened tool call index {index}"
                    ))
                })?;
                Ok(Vec::new())
            }
        }
    }

    fn complete(&mut self, completion: Completion) -> Result<ChatResponseChunk> {
        if completion.termination == Termination::ToolCalls && !self.open_calls.is_empty() {
            return Err(PowerError::InferenceFailed(
                "magnitude completed tool calls with an open call".into(),
            ));
        }
        self.open_calls.clear();
        let usage = completion.usage;
        let total_tokens = usage
            .input_tokens
            .checked_add(usage.output_tokens)
            .ok_or_else(|| PowerError::InferenceFailed("magnitude token usage overflow".into()))?;
        let prompt_tokens = u32::try_from(usage.input_tokens).map_err(|_| {
            PowerError::InferenceFailed("magnitude prompt token count exceeds u32".into())
        })?;
        let timings = completion.timings;
        Ok(ChatResponseChunk {
            done: true,
            prompt_tokens: Some(prompt_tokens),
            done_reason: Some(
                match completion.termination {
                    Termination::Natural | Termination::StopSequence(_) => "stop",
                    Termination::OutputLimit => "length",
                    Termination::ToolCalls => "tool_calls",
                }
                .into(),
            ),
            prompt_eval_duration_ns: Some(milliseconds_to_nanoseconds(
                timings.prompt_ms,
                "prompt",
            )?),
            upstream_timings: Some(serde_json::json!({
                "source": "magnitude",
                "usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": total_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": usage.cached_input_tokens,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": usage.reasoning_output_tokens,
                    },
                },
                "timings": {
                    "prompt_ms": timings.prompt_ms,
                    "decode_ms": timings.decode_ms,
                    "time_to_first_token_ms": timings.time_to_first_token_ms,
                    "sampler_ms": timings.sampler_ms,
                    "parser_ms": timings.parser_ms,
                    "draft_tokens": timings.draft_tokens,
                    "accepted_draft_tokens": timings.accepted_draft_tokens,
                },
            })),
            ..empty_chunk()
        })
    }
}

fn empty_chunk() -> ChatResponseChunk {
    ChatResponseChunk {
        content: String::new(),
        thinking_content: None,
        done: false,
        prompt_tokens: None,
        done_reason: None,
        prompt_eval_duration_ns: None,
        tool_calls: None,
        upstream_timings: None,
    }
}

fn milliseconds_to_nanoseconds(value: f64, name: &str) -> Result<u64> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 / 1e6 {
        return Err(PowerError::InferenceFailed(format!(
            "magnitude reported invalid {name} duration"
        )));
    }
    Ok((value * 1e6).round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_chat::output::{GenerationTimings, TokenUsage};

    fn completion(termination: Termination) -> Completion {
        Completion {
            usage: TokenUsage {
                input_tokens: 11,
                cached_input_tokens: 7,
                output_tokens: 3,
                reasoning_output_tokens: 2,
            },
            termination,
            timings: GenerationTimings {
                prompt_ms: 1.25,
                decode_ms: 4.0,
                time_to_first_token_ms: 2.0,
                sampler_ms: 0.5,
                parser_ms: 0.25,
                draft_tokens: 5,
                accepted_draft_tokens: 4,
            },
        }
    }

    #[test]
    fn maps_semantic_output_and_exact_usage() {
        let mut assembler = StreamAssembler::default();
        assert!(assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::Started,
                snapshot: None,
            })
            .unwrap()
            .is_empty());
        let reasoning = assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::ReasoningDelta("think".into()),
                snapshot: None,
            })
            .unwrap();
        assert_eq!(reasoning[0].thinking_content.as_deref(), Some("think"));
        let text = assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::TextDelta("answer".into()),
                snapshot: None,
            })
            .unwrap();
        assert_eq!(text[0].content, "answer");
        let done = assembler
            .apply(SessionEvent::Completed(completion(Termination::Natural)))
            .unwrap();
        assert!(done[0].done);
        assert_eq!(done[0].prompt_tokens, Some(11));
        assert_eq!(done[0].prompt_eval_duration_ns, Some(1_250_000));
        let usage = &done[0].upstream_timings.as_ref().unwrap()["usage"];
        assert_eq!(usage["total_tokens"], 14);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 7);
        assert!(assembler.finish().is_ok());
    }

    #[test]
    fn streams_tool_call_deltas_before_completion() {
        let mut assembler = StreamAssembler::default();
        let started = assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::ToolCallStarted {
                    index: 0,
                    id: "call-1".into(),
                    name: "lookup".into(),
                },
                snapshot: None,
            })
            .unwrap();
        let call = &started[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.id, "call-1");
        assert_eq!(call.function.name, "lookup");
        assert!(call.function.arguments.is_empty());

        let first = assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::ToolInputDelta {
                    index: 0,
                    fragment: "{\"city\":".into(),
                },
                snapshot: None,
            })
            .unwrap();
        let second = assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::ToolInputDelta {
                    index: 0,
                    fragment: "\"Paris\"}".into(),
                },
                snapshot: None,
            })
            .unwrap();
        let arguments = format!(
            "{}{}",
            first[0].tool_calls.as_ref().unwrap()[0].function.arguments,
            second[0].tool_calls.as_ref().unwrap()[0].function.arguments
        );
        assert_eq!(arguments, "{\"city\":\"Paris\"}");
        assert!(assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::ToolCallFinished { index: 0 },
                snapshot: None,
            })
            .unwrap()
            .is_empty());
        let done = assembler
            .apply(SessionEvent::Completed(completion(Termination::ToolCalls)))
            .unwrap();
        assert_eq!(done[0].done_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn rejects_missing_terminal_and_unopened_tool_input() {
        assert!(StreamAssembler::default().finish().is_err());
        let mut assembler = StreamAssembler::default();
        assert!(assembler
            .apply(SessionEvent::Output {
                event: OutputEvent::ToolInputDelta {
                    index: 0,
                    fragment: "{}".into(),
                },
                snapshot: None,
            })
            .is_err());
    }
}
