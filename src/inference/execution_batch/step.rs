use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::error::{PowerError, Result};

use super::super::ExecutionDigest;
use super::digest::{step_output_sha256, transcript_cancel, transcript_commit};
use super::lifecycle::{ensure_open, lock, remove_member, BatchInner};
use super::types::{
    validate_execution_digest, ExecutionBatchRow, ExecutionBatchRowDisposition,
    ExecutionBatchRowOutcome, ExecutionBatchStepEvidence,
};

pub(super) struct ProposedUpdate {
    pub(super) admission: u64,
    pub(super) member_id_sha256: String,
    pub(super) next_position: usize,
    pub(super) next_generated_items: usize,
    pub(super) next_state_bytes: u64,
    pub(super) disposition: ExecutionBatchRowDisposition,
    pub(super) output: ExecutionDigest,
}

/// Immutable row roster for one model-owned execution step.
pub struct ExecutionBatchStep {
    pub(super) inner: Arc<BatchInner>,
    pub(super) id: u64,
    pub(super) rows: Vec<ExecutionBatchRow>,
    pub(super) terminal: bool,
}

impl ExecutionBatchStep {
    pub fn rows(&self) -> &[ExecutionBatchRow] {
        &self.rows
    }

    /// Atomically commits every non-cancelled row. A member cancelled during
    /// arithmetic is discarded and therefore requires no outcome.
    pub fn commit(
        mut self,
        outcomes: Vec<ExecutionBatchRowOutcome>,
    ) -> Result<ExecutionBatchStepEvidence> {
        let result = self.commit_inner(outcomes);
        if result.is_ok() {
            self.terminal = true;
        }
        result
    }

    fn commit_inner(
        &self,
        outcomes: Vec<ExecutionBatchRowOutcome>,
    ) -> Result<ExecutionBatchStepEvidence> {
        let mut state = lock(&self.inner.state);
        ensure_open(&state)?;
        let active = state.active_step.as_ref().ok_or_else(|| {
            PowerError::InferenceFailed("execution batch step is no longer active".to_string())
        })?;
        if active.id != self.id {
            return Err(PowerError::InferenceFailed(
                "execution batch step identity changed before commit".to_string(),
            ));
        }
        let active_ids = active.member_ids.clone();
        let input_sha256 = active.input_sha256.clone();

        let mut outcome_by_id = BTreeMap::new();
        for outcome in outcomes {
            super::super::sealed_state::decode_sha256(
                &outcome.member_id_sha256,
                "execution batch outcome member",
            )?;
            if outcome_by_id
                .insert(outcome.member_id_sha256.clone(), outcome)
                .is_some()
            {
                return Err(PowerError::InvalidRequest(
                    "execution batch outcomes contain a duplicate member".to_string(),
                ));
            }
        }

        let limits = self.inner.runtime.limits();
        let mut cancelled = BTreeSet::new();
        for member_id in &active_ids {
            let admission = *state.member_index.get(member_id).ok_or_else(|| {
                PowerError::InferenceFailed("execution batch active step lost a member".to_string())
            })?;
            let member = state.members.get(&admission).ok_or_else(|| {
                PowerError::InferenceFailed(
                    "execution batch member index lost its state".to_string(),
                )
            })?;
            if member.cancel_requested || member.cancellation.is_cancelled() {
                cancelled.insert(member_id.clone());
            }
        }
        let expected_outcomes = active_ids.len().saturating_sub(cancelled.len());
        if outcome_by_id.len() != expected_outcomes
            || outcome_by_id
                .keys()
                .any(|member_id| !active_ids.contains(member_id) || cancelled.contains(member_id))
        {
            return Err(PowerError::InvalidRequest(
                "execution batch outcomes must cover every non-cancelled step member exactly once"
                    .to_string(),
            ));
        }

        let mut next_total_state = state.current_state_bytes;
        let mut proposed = Vec::with_capacity(expected_outcomes);
        for member_id in &active_ids {
            let admission = *state.member_index.get(member_id).ok_or_else(|| {
                PowerError::InferenceFailed(
                    "execution batch active member index disappeared".to_string(),
                )
            })?;
            let member = state.members.get(&admission).ok_or_else(|| {
                PowerError::InferenceFailed(
                    "execution batch active member state disappeared".to_string(),
                )
            })?;
            if cancelled.contains(member_id) {
                next_total_state = next_total_state.saturating_sub(member.state_bytes);
                continue;
            }
            let outcome = outcome_by_id.remove(member_id).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "execution batch outcome omitted a non-cancelled member".to_string(),
                )
            })?;
            validate_execution_digest(&outcome.output, limits, "execution batch row output")?;
            if outcome.next_position <= member.position
                || outcome.next_position > limits.max_context_tokens
            {
                return Err(PowerError::InvalidRequest(
                    "execution batch outcome position must advance within the context bound"
                        .to_string(),
                ));
            }
            if outcome.next_generated_items <= member.generated_items
                || outcome.next_generated_items > member.max_generated_items
                || outcome.next_generated_items > limits.max_generated_tokens
            {
                return Err(PowerError::InvalidRequest(
                    "execution batch outcome generation count must advance within member and runtime bounds"
                        .to_string(),
                ));
            }
            if outcome.disposition == ExecutionBatchRowDisposition::Continue
                && (outcome.next_generated_items == member.max_generated_items
                    || outcome.next_position == limits.max_context_tokens)
            {
                return Err(PowerError::InvalidRequest(
                    "execution batch member cannot continue after reaching a terminal bound"
                        .to_string(),
                ));
            }
            limits.checked_state_bytes(
                outcome.next_state_bytes,
                "execution batch next member state",
            )?;
            next_total_state = next_total_state
                .checked_sub(member.state_bytes)
                .and_then(|bytes| bytes.checked_add(outcome.next_state_bytes))
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "execution batch next aggregate state bytes overflowed".to_string(),
                    )
                })?;
            proposed.push(ProposedUpdate {
                admission,
                member_id_sha256: member_id.clone(),
                next_position: outcome.next_position,
                next_generated_items: outcome.next_generated_items,
                next_state_bytes: outcome.next_state_bytes,
                disposition: outcome.disposition,
                output: outcome.output,
            });
        }
        if next_total_state > limits.max_state_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "execution batch next state requires {next_total_state} bytes, exceeding the {} byte limit",
                limits.max_state_bytes
            )));
        }

        let output_sha256 = step_output_sha256(
            &self.inner.declaration_sha256,
            self.id,
            &active_ids,
            &cancelled,
            &proposed,
        )?;
        let continued_members = proposed
            .iter()
            .filter(|update| update.disposition == ExecutionBatchRowDisposition::Continue)
            .count();
        let completed_members = proposed.len().saturating_sub(continued_members);
        let completed_state_bytes = proposed
            .iter()
            .filter(|update| update.disposition == ExecutionBatchRowDisposition::Complete)
            .try_fold(0_u64, |bytes, update| {
                bytes.checked_add(update.next_state_bytes).ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "execution batch completed state byte count overflowed".to_string(),
                    )
                })
            })?;
        let next_active_state = next_total_state
            .checked_sub(completed_state_bytes)
            .ok_or_else(|| {
                PowerError::InferenceFailed(
                    "execution batch completed state accounting underflowed".to_string(),
                )
            })?;
        let next_step = state.next_step.checked_add(1).ok_or_else(|| {
            PowerError::InferenceFailed("execution batch step sequence overflowed".to_string())
        })?;

        for member_id in &cancelled {
            if let Some(admission) = state.member_index.get(member_id).copied() {
                let _ = remove_member(&mut state, admission);
            }
        }
        for update in &proposed {
            match update.disposition {
                ExecutionBatchRowDisposition::Continue => {
                    let member = state.members.get_mut(&update.admission).ok_or_else(|| {
                        PowerError::InferenceFailed(
                            "execution batch continuing member disappeared".to_string(),
                        )
                    })?;
                    member.position = update.next_position;
                    member.generated_items = update.next_generated_items;
                    member.state_bytes = update.next_state_bytes;
                }
                ExecutionBatchRowDisposition::Complete => {
                    let _ = remove_member(&mut state, update.admission);
                }
            }
        }
        state.current_state_bytes = next_active_state;
        state.peak_state_bytes = state.peak_state_bytes.max(next_total_state);
        state.active_step = None;
        state.next_step = next_step;
        state.committed_steps = state.committed_steps.saturating_add(1);
        state.processed_rows = state
            .processed_rows
            .saturating_add(u64::try_from(active_ids.len()).unwrap_or(u64::MAX));
        state.completed_members = state
            .completed_members
            .saturating_add(u64::try_from(completed_members).unwrap_or(u64::MAX));
        state.cancelled_members = state
            .cancelled_members
            .saturating_add(u64::try_from(cancelled.len()).unwrap_or(u64::MAX));
        transcript_commit(
            &mut state.transcript,
            self.id,
            &input_sha256,
            &output_sha256,
            active_ids.len(),
            continued_members,
            completed_members,
            cancelled.len(),
        );
        Ok(ExecutionBatchStepEvidence {
            schema: ExecutionBatchStepEvidence::SCHEMA.to_string(),
            declaration_sha256: self.inner.declaration_sha256.clone(),
            step: self.id,
            row_count: active_ids.len(),
            continued_members,
            completed_members,
            cancelled_members: cancelled.len(),
            active_members_after: state.members.len(),
            state_bytes_after: state.current_state_bytes,
            input_sha256,
            output_sha256,
        })
    }
}

impl Drop for ExecutionBatchStep {
    fn drop(&mut self) {
        if self.terminal {
            return;
        }
        let mut state = lock(&self.inner.state);
        let Some(active) = state.active_step.as_ref() else {
            return;
        };
        if active.id != self.id {
            return;
        }
        let member_ids = active.member_ids.clone();
        state.active_step = None;
        for member_id in member_ids {
            let Some(admission) = state.member_index.get(&member_id).copied() else {
                continue;
            };
            let cancelled = state.members.get(&admission).is_some_and(|member| {
                member.cancel_requested || member.cancellation.is_cancelled()
            });
            if cancelled {
                if let Some(member) = remove_member(&mut state, admission) {
                    transcript_cancel(&mut state.transcript, &member);
                    state.cancelled_members = state.cancelled_members.saturating_add(1);
                }
            }
        }
    }
}

impl std::fmt::Debug for ExecutionBatchStep {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBatchStep")
            .field("row_count", &self.rows.len())
            .field("terminal", &self.terminal)
            .finish()
    }
}
