use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::{EmbeddedRuntime, ExecutionPermit};
use super::digest::{declaration_sha256, step_input_sha256, transcript_admit, transcript_cancel};
use super::step::ExecutionBatchStep;
use super::types::{
    validate_execution_digest, ExecutionBatchBinding, ExecutionBatchLifecycleEvidence,
    ExecutionBatchMemberSnapshot, ExecutionBatchMemberSpec, ExecutionBatchRow,
    ExecutionBatchRowSpec,
};

/// Bounded, model-neutral membership and atomic-step lifecycle.
///
/// Every admitted member owns one permit from the runtime's existing admission
/// controller. A step contains every member present at its boundary exactly
/// once, preventing starvation inside this lifecycle. The model still owns
/// sequence scheduling, tensors, state topology, kernels, and arithmetic.
#[derive(Clone)]
pub struct ExecutionBatchLifecycle {
    inner: Arc<BatchInner>,
}

pub(super) struct BatchInner {
    pub(super) runtime: EmbeddedRuntime,
    pub(super) declaration_sha256: String,
    pub(super) state: Mutex<LifecycleState>,
}

pub(super) struct LifecycleState {
    pub(super) finalized: bool,
    pub(super) members: BTreeMap<u64, MemberState>,
    pub(super) member_index: BTreeMap<String, u64>,
    pub(super) state_ids: BTreeSet<String>,
    pub(super) active_step: Option<ActiveStep>,
    pub(super) next_admission: u64,
    pub(super) next_step: u64,
    pub(super) current_state_bytes: u64,
    pub(super) peak_state_bytes: u64,
    pub(super) admitted_members: u64,
    pub(super) completed_members: u64,
    pub(super) cancelled_members: u64,
    pub(super) committed_steps: u64,
    pub(super) processed_rows: u64,
    pub(super) max_active_members: usize,
    pub(super) transcript: Sha256,
}

pub(super) struct MemberState {
    pub(super) binding: super::types::ExecutionBatchMemberBinding,
    pub(super) position: usize,
    pub(super) generated_items: usize,
    pub(super) max_generated_items: usize,
    pub(super) state_bytes: u64,
    pub(super) permit: ExecutionPermit,
    pub(super) cancellation: CancellationToken,
    pub(super) cancel_requested: bool,
}

pub(super) struct ActiveStep {
    pub(super) id: u64,
    pub(super) member_ids: Vec<String>,
    pub(super) input_sha256: String,
}

impl EmbeddedRuntime {
    pub fn execution_batch(
        &self,
        binding: ExecutionBatchBinding,
    ) -> Result<ExecutionBatchLifecycle> {
        binding.validate()?;
        let declaration_sha256 = declaration_sha256(self, &binding)?;
        let mut transcript = Sha256::new();
        transcript.update(b"a3s-power-execution-batch-transcript-v1\0");
        transcript.update(declaration_sha256.as_bytes());
        Ok(ExecutionBatchLifecycle {
            inner: Arc::new(BatchInner {
                runtime: self.clone(),
                declaration_sha256,
                state: Mutex::new(LifecycleState {
                    finalized: false,
                    members: BTreeMap::new(),
                    member_index: BTreeMap::new(),
                    state_ids: BTreeSet::new(),
                    active_step: None,
                    next_admission: 0,
                    next_step: 0,
                    current_state_bytes: 0,
                    peak_state_bytes: 0,
                    admitted_members: 0,
                    completed_members: 0,
                    cancelled_members: 0,
                    committed_steps: 0,
                    processed_rows: 0,
                    max_active_members: 0,
                    transcript,
                }),
            }),
        })
    }
}

impl ExecutionBatchLifecycle {
    pub fn declaration_sha256(&self) -> &str {
        &self.inner.declaration_sha256
    }

    /// Waits through the runtime's bounded queue and admits one member.
    ///
    /// This is the cancellation-safe convenience path for continuously batched
    /// schedulers. It reuses the runtime permit and never creates a separate
    /// batch queue.
    pub async fn admit_wait(
        &self,
        spec: ExecutionBatchMemberSpec,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let permit = self.inner.runtime.begin_wait(&cancellation).await?;
        self.admit(spec, permit, cancellation)
    }

    /// Admits a member with a permit from this exact runtime.
    ///
    /// The permit is consumed and held until completion or cancellation. This
    /// reuses the existing runtime admission controller instead of creating a
    /// batch-specific semaphore or queue.
    pub fn admit(
        &self,
        spec: ExecutionBatchMemberSpec,
        permit: ExecutionPermit,
        cancellation: CancellationToken,
    ) -> Result<()> {
        self.validate_member_spec(&spec, &permit, &cancellation)?;
        let mut state = lock(&self.inner.state);
        ensure_open(&state)?;
        let limits = self.inner.runtime.limits();
        if state.members.len() >= limits.max_concurrent_requests {
            return Err(PowerError::InferenceFailed(format!(
                "execution batch already has {} active member(s)",
                limits.max_concurrent_requests
            )));
        }
        if state
            .member_index
            .contains_key(spec.binding.member_id_sha256())
        {
            return Err(PowerError::InvalidRequest(
                "execution batch member identity is already active".to_string(),
            ));
        }
        if state.state_ids.contains(spec.binding.state_id_sha256()) {
            return Err(PowerError::InvalidRequest(
                "execution batch members cannot alias one model-owned state identity".to_string(),
            ));
        }
        if state
            .members
            .values()
            .any(|member| member.permit.same_admission(&permit))
        {
            return Err(PowerError::InvalidRequest(
                "execution batch members require distinct runtime admission permits".to_string(),
            ));
        }
        let next_state_bytes = state
            .current_state_bytes
            .checked_add(spec.state_bytes)
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "execution batch aggregate state byte count overflowed".to_string(),
                )
            })?;
        if next_state_bytes > limits.max_state_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "execution batch requires {next_state_bytes} state bytes, exceeding the {} byte limit",
                limits.max_state_bytes
            )));
        }
        let admission = state.next_admission;
        state.next_admission = state.next_admission.checked_add(1).ok_or_else(|| {
            PowerError::InferenceFailed("execution batch admission sequence overflowed".to_string())
        })?;
        transcript_admit(&mut state.transcript, admission, &spec);
        state
            .member_index
            .insert(spec.binding.member_id_sha256().to_string(), admission);
        state
            .state_ids
            .insert(spec.binding.state_id_sha256().to_string());
        state.members.insert(
            admission,
            MemberState {
                binding: spec.binding,
                position: spec.position,
                generated_items: spec.generated_items,
                max_generated_items: spec.max_generated_items,
                state_bytes: spec.state_bytes,
                permit,
                cancellation,
                cancel_requested: false,
            },
        );
        state.current_state_bytes = next_state_bytes;
        state.peak_state_bytes = state.peak_state_bytes.max(next_state_bytes);
        state.admitted_members = state.admitted_members.saturating_add(1);
        state.max_active_members = state.max_active_members.max(state.members.len());
        Ok(())
    }

    /// Creates an immutable fair roster for one model-owned forward.
    ///
    /// Rows may have different shapes and positions, but every member present
    /// at this boundary must appear exactly once. Members admitted after this
    /// method returns join the next step.
    pub fn begin_step(&self, specs: Vec<ExecutionBatchRowSpec>) -> Result<ExecutionBatchStep> {
        let mut state = lock(&self.inner.state);
        ensure_open(&state)?;
        if state.active_step.is_some() {
            return Err(PowerError::InferenceFailed(
                "execution batch already has a step in flight".to_string(),
            ));
        }
        purge_cancelled(&mut state);
        if state.members.is_empty() {
            return Err(PowerError::InvalidRequest(
                "execution batch has no active members".to_string(),
            ));
        }
        if specs.len() != state.members.len() {
            return Err(PowerError::InvalidRequest(
                "execution batch step must contain every active member exactly once".to_string(),
            ));
        }
        let limits = self.inner.runtime.limits();
        if specs.len() > limits.max_graph_nodes {
            return Err(PowerError::InvalidRequest(format!(
                "execution batch step contains {} rows, exceeding the {} graph-node limit",
                specs.len(),
                limits.max_graph_nodes
            )));
        }

        let mut by_id = BTreeMap::new();
        for spec in specs {
            super::super::sealed_state::decode_sha256(
                &spec.member_id_sha256,
                "execution batch row member",
            )?;
            if by_id.insert(spec.member_id_sha256.clone(), spec).is_some() {
                return Err(PowerError::InvalidRequest(
                    "execution batch step contains a duplicate member".to_string(),
                ));
            }
        }

        let mut rows = Vec::with_capacity(state.members.len());
        let mut total_bytes = 0_usize;
        let mut total_elements = 0_usize;
        for (canonical_index, member) in state.members.values().enumerate() {
            let spec = by_id
                .remove(member.binding.member_id_sha256())
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "execution batch step omitted an active member".to_string(),
                    )
                })?;
            if spec.position != member.position {
                return Err(PowerError::InvalidRequest(
                    "execution batch row position does not match its member lifecycle".to_string(),
                ));
            }
            if spec.shape.len() > limits.max_graph_nodes {
                return Err(PowerError::InvalidRequest(format!(
                    "execution batch row rank exceeds the {} graph-node bound",
                    limits.max_graph_nodes
                )));
            }
            let elements = limits.checked_elements(&spec.shape, "execution batch row")?;
            validate_execution_digest(&spec.input, limits, "execution batch row input")?;
            if spec.input.item_count != elements {
                return Err(PowerError::InvalidRequest(
                    "execution batch row shape and input item count do not match".to_string(),
                ));
            }
            total_bytes = total_bytes
                .checked_add(spec.input.byte_length)
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "execution batch input byte count overflowed".to_string(),
                    )
                })?;
            total_elements = total_elements.checked_add(elements).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "execution batch input element count overflowed".to_string(),
                )
            })?;
            rows.push(ExecutionBatchRow {
                canonical_index,
                member_id_sha256: member.binding.member_id_sha256().to_string(),
                state_id_sha256: member.binding.state_id_sha256().to_string(),
                position: spec.position,
                shape: spec.shape,
                input: spec.input,
                permit: member.permit.clone(),
                cancellation: member.cancellation.clone(),
            });
        }
        if !by_id.is_empty() {
            return Err(PowerError::InvalidRequest(
                "execution batch step contains an unknown member".to_string(),
            ));
        }
        if total_bytes > limits.max_input_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "execution batch step contains {total_bytes} input bytes, exceeding the {} byte limit",
                limits.max_input_bytes
            )));
        }
        if total_elements > limits.max_tensor_elements {
            return Err(PowerError::InvalidRequest(format!(
                "execution batch step contains {total_elements} elements, exceeding the {} element limit",
                limits.max_tensor_elements
            )));
        }

        let step_id = state.next_step;
        let input_sha256 = step_input_sha256(&self.inner.declaration_sha256, step_id, &rows);
        let member_ids = rows
            .iter()
            .map(|row| row.member_id_sha256.clone())
            .collect::<Vec<_>>();
        state.active_step = Some(ActiveStep {
            id: step_id,
            member_ids,
            input_sha256,
        });
        Ok(ExecutionBatchStep {
            inner: Arc::clone(&self.inner),
            id: step_id,
            rows,
            terminal: false,
        })
    }

    /// Requests cancellation. An in-flight row becomes terminal at commit or
    /// abort; a member outside the current immutable roster is removed now.
    pub fn cancel(&self, member_id_sha256: &str) -> Result<()> {
        let mut state = lock(&self.inner.state);
        ensure_open(&state)?;
        let admission = *state.member_index.get(member_id_sha256).ok_or_else(|| {
            PowerError::InvalidRequest("execution batch member was not found".to_string())
        })?;
        let in_flight = state.active_step.as_ref().is_some_and(|step| {
            step.member_ids
                .iter()
                .any(|member| member == member_id_sha256)
        });
        if in_flight {
            let member = state.members.get_mut(&admission).ok_or_else(|| {
                PowerError::InferenceFailed(
                    "execution batch member index lost its active state".to_string(),
                )
            })?;
            member.cancel_requested = true;
            member.cancellation.cancel();
        } else if let Some(member) = remove_member(&mut state, admission) {
            transcript_cancel(&mut state.transcript, &member);
            state.cancelled_members = state.cancelled_members.saturating_add(1);
        }
        Ok(())
    }

    pub fn active_member_count(&self) -> usize {
        lock(&self.inner.state).members.len()
    }

    pub fn active_members(&self) -> Vec<ExecutionBatchMemberSnapshot> {
        lock(&self.inner.state)
            .members
            .values()
            .map(|member| {
                ExecutionBatchMemberSnapshot::new(
                    member.binding.member_id_sha256().to_string(),
                    member.position,
                    member.generated_items,
                    member.max_generated_items,
                    member.state_bytes,
                )
            })
            .collect()
    }

    /// Finalizes one empty lifecycle exactly once.
    pub fn finish(&self) -> Result<ExecutionBatchLifecycleEvidence> {
        let mut state = lock(&self.inner.state);
        if state.finalized {
            return Err(PowerError::InvalidRequest(
                "execution batch lifecycle was already finalized".to_string(),
            ));
        }
        if state.active_step.is_some() {
            return Err(PowerError::InferenceFailed(
                "execution batch cannot finish with a step in flight".to_string(),
            ));
        }
        purge_cancelled(&mut state);
        if !state.members.is_empty() {
            return Err(PowerError::InferenceFailed(
                "execution batch cannot finish with active members".to_string(),
            ));
        }
        state.transcript.update(b"finish\0");
        let admitted_members = state.admitted_members;
        let completed_members = state.completed_members;
        let cancelled_members = state.cancelled_members;
        state.transcript.update(admitted_members.to_le_bytes());
        state.transcript.update(completed_members.to_le_bytes());
        state.transcript.update(cancelled_members.to_le_bytes());
        let transcript_sha256 = format!("{:x}", state.transcript.clone().finalize());
        state.finalized = true;
        Ok(ExecutionBatchLifecycleEvidence {
            schema: ExecutionBatchLifecycleEvidence::SCHEMA.to_string(),
            declaration_sha256: self.inner.declaration_sha256.clone(),
            transcript_sha256,
            admitted_members: state.admitted_members,
            completed_members: state.completed_members,
            cancelled_members: state.cancelled_members,
            committed_steps: state.committed_steps,
            processed_rows: state.processed_rows,
            max_active_members: state.max_active_members,
            peak_state_bytes: state.peak_state_bytes,
        })
    }

    fn validate_member_spec(
        &self,
        spec: &ExecutionBatchMemberSpec,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        spec.binding.validate()?;
        if !permit.belongs_to(&self.inner.runtime) {
            return Err(PowerError::InvalidRequest(
                "execution batch permit belongs to a different embedded runtime".to_string(),
            ));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "execution batch member was cancelled before admission".to_string(),
            ));
        }
        let limits = self.inner.runtime.limits();
        if spec.position >= limits.max_context_tokens {
            return Err(PowerError::InvalidRequest(format!(
                "execution batch member position must be below {}",
                limits.max_context_tokens
            )));
        }
        if spec.max_generated_items == 0
            || spec.max_generated_items > limits.max_generated_tokens
            || spec.generated_items >= spec.max_generated_items
        {
            return Err(PowerError::InvalidRequest(format!(
                "execution batch generation bounds must describe unfinished work within the {} item limit",
                limits.max_generated_tokens
            )));
        }
        limits.checked_state_bytes(spec.state_bytes, "execution batch member state")?;
        Ok(())
    }
}

impl std::fmt::Debug for ExecutionBatchLifecycle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock(&self.inner.state);
        formatter
            .debug_struct("ExecutionBatchLifecycle")
            .field("declaration", &"sha256")
            .field("active_members", &state.members.len())
            .field("step_in_flight", &state.active_step.is_some())
            .field("finalized", &state.finalized)
            .finish()
    }
}

pub(super) fn ensure_open(state: &LifecycleState) -> Result<()> {
    if state.finalized {
        Err(PowerError::InvalidRequest(
            "execution batch lifecycle was already finalized".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn purge_cancelled(state: &mut LifecycleState) {
    let admissions = state
        .members
        .iter()
        .filter_map(|(admission, member)| {
            (member.cancel_requested || member.cancellation.is_cancelled()).then_some(*admission)
        })
        .collect::<Vec<_>>();
    for admission in admissions {
        if let Some(member) = remove_member(state, admission) {
            transcript_cancel(&mut state.transcript, &member);
            state.cancelled_members = state.cancelled_members.saturating_add(1);
        }
    }
}

pub(super) fn remove_member(state: &mut LifecycleState, admission: u64) -> Option<MemberState> {
    let member = state.members.remove(&admission)?;
    state.member_index.remove(member.binding.member_id_sha256());
    state.state_ids.remove(member.binding.state_id_sha256());
    state.current_state_bytes = state.current_state_bytes.saturating_sub(member.state_bytes);
    Some(member)
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
