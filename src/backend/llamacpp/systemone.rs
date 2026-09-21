//! System One decision scoring via next-token option-label logits.

use std::collections::BTreeMap;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use super::*;
use crate::backend::systemone_scoring::{
    choice_label, distribution_confidence, score_label, subset_softmax,
};
use crate::backend::types::{
    ChatMessage, MessageContent, SystemOneAnswerSpec, SystemOneQuestionSpec, SystemOneRequest,
    SystemOneResponse,
};

pub(super) async fn systemone(
    backend: &LlamaCppBackend,
    model_name: &str,
    request: SystemOneRequest,
) -> Result<SystemOneResponse> {
    let (model_arc, template, raw_template, n_ctx_train) = {
        let models = backend.models.read().await;
        let model = models.get(model_name).ok_or_else(|| {
            PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
        })?;
        (
            model.model.clone(),
            model.chat_template.clone(),
            model.raw_template.clone(),
            model.n_ctx_train,
        )
    };

    let ctx_size = DEFAULT_CTX_SIZE.min(n_ctx_train).max(512);

    tokio::task::spawn_blocking(move || {
        score_all_questions(
            &model_arc,
            &template,
            raw_template.as_deref(),
            ctx_size,
            &request,
        )
    })
    .await
    .map_err(|e| PowerError::InferenceFailed(format!("System One scoring task failed: {e}")))?
}

fn score_all_questions(
    model: &LlamaModel,
    template: &ChatTemplateKind,
    raw_template: Option<&str>,
    ctx_size: u32,
    request: &SystemOneRequest,
) -> Result<SystemOneResponse> {
    let mut answers = BTreeMap::new();
    let mut total_input_tokens = 0_u32;

    for (name, question) in &request.questions {
        let scored = score_one_question(
            model,
            template,
            raw_template,
            ctx_size,
            &request.state,
            question,
        )
        .map_err(|e| {
            PowerError::InferenceFailed(format!("System One question '{name}' failed: {e}"))
        })?;
        total_input_tokens = total_input_tokens.saturating_add(scored.input_tokens);
        answers.insert(name.clone(), scored.answer);
    }

    Ok(SystemOneResponse {
        answers,
        input_tokens: total_input_tokens,
    })
}

struct ScoredQuestion {
    answer: SystemOneAnswerSpec,
    input_tokens: u32,
}

fn score_one_question(
    model: &LlamaModel,
    template: &ChatTemplateKind,
    raw_template: Option<&str>,
    ctx_size: u32,
    state: &str,
    question: &SystemOneQuestionSpec,
) -> Result<ScoredQuestion> {
    let (user_prompt, labels, keys) = build_question_prompt(state, question)?;
    let prompt = render_decision_prompt(template, raw_template, &user_prompt)?;
    let (probabilities, input_tokens) = score_labels(model, ctx_size, &prompt, &labels)?;

    let answer = match question {
        SystemOneQuestionSpec::Choice { .. } => {
            let mut map = BTreeMap::new();
            let mut best_key = keys[0].clone();
            let mut best_p = 0.0_f64;
            for (key, prob) in keys.iter().zip(probabilities.iter()) {
                map.insert(key.clone(), *prob);
                if *prob > best_p {
                    best_p = *prob;
                    best_key = key.clone();
                }
            }
            SystemOneAnswerSpec::Choice {
                choice: best_key,
                confidence: distribution_confidence(&probabilities),
                probabilities: map,
            }
        }
        SystemOneQuestionSpec::Noul { .. } => {
            // labels are ["yes", "no"] — noul is P(yes)
            let noul = probabilities.first().copied().unwrap_or(0.0);
            SystemOneAnswerSpec::Noul { noul }
        }
        SystemOneQuestionSpec::Score { criteria, .. } => {
            let mut legend = BTreeMap::new();
            let mut map = BTreeMap::new();
            let mut best_idx = 0_usize;
            let mut best_p = 0.0_f64;
            for (i, (level, prob)) in criteria.iter().zip(probabilities.iter()).enumerate() {
                let key = i.to_string();
                legend.insert(key.clone(), level.clone());
                map.insert(key, *prob);
                if *prob > best_p {
                    best_p = *prob;
                    best_idx = i;
                }
            }
            SystemOneAnswerSpec::Score {
                score: best_idx as f64,
                confidence: distribution_confidence(&probabilities),
                legend,
                probabilities: map,
            }
        }
    };

    Ok(ScoredQuestion {
        answer,
        input_tokens,
    })
}

fn build_question_prompt(
    state: &str,
    question: &SystemOneQuestionSpec,
) -> Result<(String, Vec<String>, Vec<String>)> {
    match question {
        SystemOneQuestionSpec::Choice {
            instructions,
            criteria,
        } => {
            let mut labels = Vec::new();
            let mut keys = Vec::new();
            let mut lines = Vec::new();
            for (i, (key, description)) in criteria.iter().enumerate() {
                let label = choice_label(i);
                lines.push(format!("{label}. {key} — {description}"));
                labels.push(label);
                keys.push(key.clone());
            }
            let prompt = format!(
                "State:\n{state}\n\nQuestion: {instructions}\n\nOptions:\n{}\n\nReply with the single option letter or number only.",
                lines.join("\n")
            );
            Ok((prompt, labels, keys))
        }
        SystemOneQuestionSpec::Noul {
            instructions,
            criteria,
        } => {
            let mut prompt = format!(
                "State:\n{state}\n\nQuestion: {instructions}\n\nReply with yes or no only."
            );
            if let Some(criteria) = criteria {
                if !criteria.is_empty() {
                    let mut lines = Vec::new();
                    for (key, description) in criteria {
                        lines.push(format!("- {key}: {description}"));
                    }
                    prompt.push_str("\n\nCriteria:\n");
                    prompt.push_str(&lines.join("\n"));
                }
            }
            Ok((
                prompt,
                vec!["yes".to_string(), "no".to_string()],
                vec!["yes".to_string(), "no".to_string()],
            ))
        }
        SystemOneQuestionSpec::Score {
            instructions,
            criteria,
        } => {
            let mut labels = Vec::new();
            let mut keys = Vec::new();
            let mut lines = Vec::new();
            for (i, description) in criteria.iter().enumerate() {
                let label = score_label(i);
                lines.push(format!("{label}. {description}"));
                labels.push(label.clone());
                keys.push(label);
            }
            let prompt = format!(
                "State:\n{state}\n\nQuestion: {instructions}\n\nLevels:\n{}\n\nReply with the single level number only.",
                lines.join("\n")
            );
            Ok((prompt, labels, keys))
        }
    }
}

fn render_decision_prompt(
    template: &ChatTemplateKind,
    raw_template: Option<&str>,
    user_prompt: &str,
) -> Result<String> {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContent::Text(user_prompt.to_string()),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        images: None,
    }];
    chat_template::format_chat_prompt(&messages, template, raw_template)
        .map_err(PowerError::InferenceFailed)
}

fn score_labels(
    model: &LlamaModel,
    ctx_size: u32,
    prompt: &str,
    labels: &[String],
) -> Result<(Vec<f64>, u32)> {
    let token_ids = resolve_label_tokens(model, labels)?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| PowerError::InferenceFailed(format!("Tokenization failed: {e}")))?;
    if tokens.is_empty() {
        return Err(PowerError::InferenceFailed(
            "System One prompt tokenized to zero tokens".to_string(),
        ));
    }
    if tokens.len() as u32 >= ctx_size {
        return Err(PowerError::InferenceFailed(format!(
            "System One prompt length {} exceeds context size {ctx_size}",
            tokens.len()
        )));
    }

    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(nonzero_context_size(ctx_size)));
    let mut ctx = model
        .new_context(backend_ref(), ctx_params)
        .map_err(|e| PowerError::InferenceFailed(format!("Failed to create context: {e}")))?;

    let batch_capacity = tokens.len().max(1);
    let mut batch = LlamaBatch::new(batch_capacity, 1);
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i + 1 == tokens.len();
        batch
            .add(token, i as i32, &[0], is_last)
            .map_err(|_| PowerError::InferenceFailed("Failed to add token to batch".to_string()))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| PowerError::InferenceFailed(format!("Decode failed: {e}")))?;

    let logits = ctx.get_logits();
    let mut subset = Vec::with_capacity(token_ids.len());
    for token_id in &token_ids {
        let idx = token_id.0 as usize;
        if idx >= logits.len() {
            return Err(PowerError::InferenceFailed(format!(
                "option token id {idx} out of vocab range {}",
                logits.len()
            )));
        }
        subset.push(logits[idx]);
    }

    let probabilities = subset_softmax(&subset);
    Ok((probabilities, tokens.len() as u32))
}

fn resolve_label_tokens(model: &LlamaModel, labels: &[String]) -> Result<Vec<LlamaToken>> {
    let mut out = Vec::with_capacity(labels.len());
    for label in labels {
        let token = resolve_single_label_token(model, label)?;
        out.push(token);
    }
    Ok(out)
}

fn resolve_single_label_token(model: &LlamaModel, label: &str) -> Result<LlamaToken> {
    // Prefer a bare label; fall back to a leading-space form common in BPE tokenizers.
    for candidate in [label.to_string(), format!(" {label}")] {
        let tokens = model.str_to_token(&candidate, AddBos::Never).map_err(|e| {
            PowerError::InferenceFailed(format!("failed to tokenize option label '{label}': {e}"))
        })?;
        if tokens.len() == 1 {
            return Ok(tokens[0]);
        }
    }
    Err(PowerError::InvalidRequest(format!(
        "option label '{label}' does not tokenize to exactly one token; choose a shorter key or fewer options"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_choice_prompt_uses_letter_labels() {
        let criteria = BTreeMap::from([
            ("billing".to_string(), "money".to_string()),
            ("bug".to_string(), "broken".to_string()),
        ]);
        let (prompt, labels, keys) = build_question_prompt(
            "charged twice",
            &SystemOneQuestionSpec::Choice {
                instructions: "What?".into(),
                criteria,
            },
        )
        .unwrap();
        assert!(prompt.contains("A."));
        assert!(prompt.contains("B."));
        assert_eq!(labels, vec!["A".to_string(), "B".to_string()]);
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn build_noul_prompt_yes_no() {
        let (prompt, labels, _) = build_question_prompt(
            "state",
            &SystemOneQuestionSpec::Noul {
                instructions: "Escalate?".into(),
                criteria: None,
            },
        )
        .unwrap();
        assert!(prompt.contains("yes or no"));
        assert_eq!(labels, vec!["yes".to_string(), "no".to_string()]);
    }
}
