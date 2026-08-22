use llama_cpp_2::model::LlamaModel;

use super::stop_sequences::StopSequenceTracker;
use crate::backend::llamacpp::send_completion_result;
use crate::backend::types::CompletionResponseChunk;
use crate::error::Result;

#[allow(clippy::too_many_arguments)]
pub(super) fn stream_token(
    model: &LlamaModel,
    token: llama_cpp_2::token::LlamaToken,
    eos_token: llama_cpp_2::token::LlamaToken,
    stop_tracker: &mut StopSequenceTracker,
    generated_count: &mut usize,
    stop_sequences: &[String],
    prompt_token_count: u32,
    prompt_eval_duration_ns: u64,
    tx: &tokio::sync::mpsc::Sender<Result<CompletionResponseChunk>>,
) -> bool {
    if token == eos_token {
        send_completion_result(
            tx,
            Ok(CompletionResponseChunk {
                text: String::new(),
                done: true,
                prompt_tokens: Some(prompt_token_count),
                done_reason: Some("stop".to_string()),
                prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                token_id: None,
            }),
        );
        return false;
    }

    let text = token_piece(model, token);
    *generated_count += 1;
    let should_stop = stop_tracker.push(&text, stop_sequences);
    send_completion_result(
        tx,
        Ok(CompletionResponseChunk {
            text,
            done: should_stop,
            prompt_tokens: should_stop.then_some(prompt_token_count),
            done_reason: should_stop.then(|| "stop".to_string()),
            prompt_eval_duration_ns: should_stop.then_some(prompt_eval_duration_ns),
            token_id: Some(token.0 as u32),
        }),
    ) && !should_stop
}

pub(super) fn token_piece(model: &LlamaModel, token: llama_cpp_2::token::LlamaToken) -> String {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .unwrap_or_default()
}
