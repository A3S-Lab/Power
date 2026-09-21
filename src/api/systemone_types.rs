//! Jev-compatible System One wire types.
//!
//! This surface is intentionally separate from OpenAI `logprobs`. Probabilities
//! are a softmax over declared option-label tokens only and are not calibrated
//! TypeSafe Jev confidence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::backend::systemone_scoring::{choice_label, distribution_confidence, subset_softmax};

/// Maximum number of choice options (Jev cardinality).
pub const MAX_CHOICE_OPTIONS: usize = 255;

/// Minimum / maximum ordered score levels.
pub const MIN_SCORE_LEVELS: usize = 2;
pub const MAX_SCORE_LEVELS: usize = 10;

/// System One request body (`POST /v1/systemone`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneHttpRequest {
    pub model: String,
    pub state: SystemOneState,
    pub questions: BTreeMap<String, SystemOneQuestion>,
    /// Optional keep-alive override (Power extension; ignored by TypeSafe clients).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

/// Unstructured decision context: string, JSON object, or text array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemOneState {
    Text(String),
    Array(Vec<String>),
    Object(serde_json::Value),
}

impl SystemOneState {
    /// Render state into a single prompt string.
    pub fn render(&self) -> Result<String, String> {
        match self {
            SystemOneState::Text(s) => Ok(s.clone()),
            SystemOneState::Array(parts) => Ok(parts.join("\n")),
            SystemOneState::Object(value) => serde_json::to_string_pretty(value)
                .map_err(|e| format!("failed to serialize state object: {e}")),
        }
    }
}

/// A typed System One question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneQuestion {
    Choice {
        instructions: String,
        criteria: BTreeMap<String, String>,
    },
    Noul {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<BTreeMap<String, String>>,
    },
    Score {
        instructions: String,
        /// Ordered level descriptions (2–10).
        criteria: Vec<String>,
    },
}

/// System One response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneHttpResponse {
    pub model: String,
    pub answers: BTreeMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
}

/// Per-question typed answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneAnswer {
    Choice {
        choice: String,
        confidence: f64,
        probabilities: BTreeMap<String, f64>,
    },
    Noul {
        noul: f64,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Validate a System One HTTP request before inference.
pub fn validate_systemone_request(request: &SystemOneHttpRequest) -> Result<(), String> {
    if request.model.trim().is_empty() {
        return Err("model must be a non-empty string".to_string());
    }
    if request.questions.is_empty() {
        return Err("questions must contain at least one question".to_string());
    }
    for (name, question) in &request.questions {
        if name.trim().is_empty() {
            return Err("question names must be non-empty".to_string());
        }
        validate_question(name, question)?;
    }
    Ok(())
}

fn validate_question(name: &str, question: &SystemOneQuestion) -> Result<(), String> {
    match question {
        SystemOneQuestion::Choice {
            instructions,
            criteria,
        } => {
            if instructions.trim().is_empty() {
                return Err(format!("question '{name}': instructions must be non-empty"));
            }
            if criteria.is_empty() {
                return Err(format!(
                    "question '{name}': choice criteria must be non-empty"
                ));
            }
            if criteria.len() > MAX_CHOICE_OPTIONS {
                return Err(format!(
                    "question '{name}': choice criteria exceed {MAX_CHOICE_OPTIONS} options"
                ));
            }
            for key in criteria.keys() {
                if key.trim().is_empty() {
                    return Err(format!(
                        "question '{name}': choice criteria keys must be non-empty"
                    ));
                }
            }
        }
        SystemOneQuestion::Noul { instructions, .. } => {
            if instructions.trim().is_empty() {
                return Err(format!("question '{name}': instructions must be non-empty"));
            }
        }
        SystemOneQuestion::Score {
            instructions,
            criteria,
        } => {
            if instructions.trim().is_empty() {
                return Err(format!("question '{name}': instructions must be non-empty"));
            }
            if criteria.len() < MIN_SCORE_LEVELS || criteria.len() > MAX_SCORE_LEVELS {
                return Err(format!(
                    "question '{name}': score criteria must have {MIN_SCORE_LEVELS}–{MAX_SCORE_LEVELS} levels"
                ));
            }
            if criteria.iter().any(|c| c.trim().is_empty()) {
                return Err(format!(
                    "question '{name}': score criteria entries must be non-empty"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_render_variants() {
        assert_eq!(SystemOneState::Text("hi".into()).render().unwrap(), "hi");
        assert_eq!(
            SystemOneState::Array(vec!["a".into(), "b".into()])
                .render()
                .unwrap(),
            "a\nb"
        );
        let obj = SystemOneState::Object(serde_json::json!({"k": 1}));
        assert!(obj.render().unwrap().contains("\"k\""));
    }

    #[test]
    fn validate_rejects_empty_choice_criteria() {
        let req = SystemOneHttpRequest {
            model: "m".into(),
            state: SystemOneState::Text("s".into()),
            questions: BTreeMap::from([(
                "topic".into(),
                SystemOneQuestion::Choice {
                    instructions: "pick".into(),
                    criteria: BTreeMap::new(),
                },
            )]),
            keep_alive: None,
        };
        let err = validate_systemone_request(&req).unwrap_err();
        assert!(err.contains("criteria must be non-empty"));
    }

    #[test]
    fn validate_rejects_too_many_choice_options() {
        let mut criteria = BTreeMap::new();
        for i in 0..=MAX_CHOICE_OPTIONS {
            criteria.insert(format!("k{i}"), "d".into());
        }
        let req = SystemOneHttpRequest {
            model: "m".into(),
            state: SystemOneState::Text("s".into()),
            questions: BTreeMap::from([(
                "topic".into(),
                SystemOneQuestion::Choice {
                    instructions: "pick".into(),
                    criteria,
                },
            )]),
            keep_alive: None,
        };
        let err = validate_systemone_request(&req).unwrap_err();
        assert!(err.contains("exceed"));
    }

    #[test]
    fn validate_rejects_bad_score_levels() {
        let req = SystemOneHttpRequest {
            model: "m".into(),
            state: SystemOneState::Text("s".into()),
            questions: BTreeMap::from([(
                "sev".into(),
                SystemOneQuestion::Score {
                    instructions: "how bad".into(),
                    criteria: vec!["only one".into()],
                },
            )]),
            keep_alive: None,
        };
        let err = validate_systemone_request(&req).unwrap_err();
        assert!(err.contains("score criteria"));
    }

    #[test]
    fn validate_accepts_minimal_choice() {
        let req = SystemOneHttpRequest {
            model: "m".into(),
            state: SystemOneState::Text("s".into()),
            questions: BTreeMap::from([(
                "topic".into(),
                SystemOneQuestion::Choice {
                    instructions: "pick".into(),
                    criteria: BTreeMap::from([
                        ("billing".into(), "money".into()),
                        ("bug".into(), "broken".into()),
                    ]),
                },
            )]),
            keep_alive: None,
        };
        validate_systemone_request(&req).unwrap();
    }

    #[test]
    fn serde_roundtrip_choice_answer() {
        let answer = SystemOneAnswer::Choice {
            choice: "billing".into(),
            confidence: 0.9,
            probabilities: BTreeMap::from([("billing".into(), 0.8), ("bug".into(), 0.2)]),
        };
        let json = serde_json::to_value(&answer).unwrap();
        assert_eq!(json["type"], "choice");
        assert_eq!(json["choice"], "billing");
        let back: SystemOneAnswer = serde_json::from_value(json).unwrap();
        match back {
            SystemOneAnswer::Choice { choice, .. } => assert_eq!(choice, "billing"),
            _ => panic!("expected choice"),
        }
    }

    #[test]
    fn helpers_available_from_scoring_module() {
        assert_eq!(choice_label(0), "A");
        assert!(distribution_confidence(&[1.0, 0.0]) > 0.9);
        assert_eq!(subset_softmax(&[0.0, 1.0]).len(), 2);
    }
}
