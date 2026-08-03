use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::super::{
    EmbeddedRuntime, ExecutionDigest, ExecutionRepresentation, RuntimeDeviceIdentity,
};
use super::lifecycle::MemberState;
use super::step::ProposedUpdate;
use super::types::{
    ExecutionBatchBinding, ExecutionBatchMemberSpec, ExecutionBatchRow,
    ExecutionBatchRowDisposition,
};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Declaration<'a> {
    schema: &'static str,
    binding: &'a ExecutionBatchBinding,
    runtime_device: RuntimeDeviceIdentity,
    limits: DeclarationLimits,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DeclarationLimits {
    max_concurrent_requests: usize,
    max_state_bytes: u64,
    max_input_bytes: usize,
    max_tensor_elements: usize,
    max_graph_nodes: usize,
    max_context_tokens: usize,
    max_generated_tokens: usize,
}

pub(super) fn declaration_sha256(
    runtime: &EmbeddedRuntime,
    binding: &ExecutionBatchBinding,
) -> Result<String> {
    let limits = runtime.limits();
    let declaration = Declaration {
        schema: "a3s.power.execution-batch-declaration.v1",
        binding,
        runtime_device: runtime.device().identity(),
        limits: DeclarationLimits {
            max_concurrent_requests: limits.max_concurrent_requests,
            max_state_bytes: limits.max_state_bytes,
            max_input_bytes: limits.max_input_bytes,
            max_tensor_elements: limits.max_tensor_elements,
            max_graph_nodes: limits.max_graph_nodes,
            max_context_tokens: limits.max_context_tokens,
            max_generated_tokens: limits.max_generated_tokens,
        },
    };
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&declaration)?)
    ))
}

pub(super) fn step_input_sha256(
    declaration: &str,
    step: u64,
    rows: &[ExecutionBatchRow],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"a3s-power-execution-batch-step-input-v1\0");
    hasher.update(declaration.as_bytes());
    hasher.update(step.to_le_bytes());
    hasher.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        hasher.update(row.member_id_sha256.as_bytes());
        hasher.update(row.state_id_sha256.as_bytes());
        hasher.update((row.position as u64).to_le_bytes());
        hasher.update((row.shape.len() as u64).to_le_bytes());
        for dimension in &row.shape {
            hasher.update((*dimension as u64).to_le_bytes());
        }
        hash_execution_digest(&mut hasher, &row.input);
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn step_output_sha256(
    declaration: &str,
    step: u64,
    active_ids: &[String],
    cancelled: &BTreeSet<String>,
    proposed: &[ProposedUpdate],
) -> Result<String> {
    let by_id = proposed
        .iter()
        .map(|update| (update.member_id_sha256.as_str(), update))
        .collect::<BTreeMap<_, _>>();
    let mut hasher = Sha256::new();
    hasher.update(b"a3s-power-execution-batch-step-output-v1\0");
    hasher.update(declaration.as_bytes());
    hasher.update(step.to_le_bytes());
    hasher.update((active_ids.len() as u64).to_le_bytes());
    for member_id in active_ids {
        hasher.update(member_id.as_bytes());
        if cancelled.contains(member_id) {
            hasher.update(b"cancelled\0");
            continue;
        }
        let update = by_id.get(member_id.as_str()).ok_or_else(|| {
            PowerError::InferenceFailed(
                "validated execution batch output lost a member update".to_string(),
            )
        })?;
        hasher.update(match update.disposition {
            ExecutionBatchRowDisposition::Continue => b"continue\0".as_slice(),
            ExecutionBatchRowDisposition::Complete => b"complete\0".as_slice(),
        });
        hasher.update((update.next_position as u64).to_le_bytes());
        hasher.update((update.next_generated_items as u64).to_le_bytes());
        hasher.update(update.next_state_bytes.to_le_bytes());
        hash_execution_digest(&mut hasher, &update.output);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_execution_digest(hasher: &mut Sha256, digest: &ExecutionDigest) {
    let representation = match digest.representation {
        ExecutionRepresentation::F32Tensor => 0_u8,
        ExecutionRepresentation::ImageRequest => 1,
        ExecutionRepresentation::TokenIds => 2,
        ExecutionRepresentation::Utf8Text => 3,
    };
    hasher.update([representation]);
    hasher.update(digest.sha256.as_bytes());
    hasher.update((digest.byte_length as u64).to_le_bytes());
    hasher.update((digest.item_count as u64).to_le_bytes());
}

pub(super) fn transcript_admit(
    transcript: &mut Sha256,
    admission: u64,
    spec: &ExecutionBatchMemberSpec,
) {
    transcript.update(b"admit\0");
    transcript.update(admission.to_le_bytes());
    transcript.update(spec.binding.member_id_sha256().as_bytes());
    transcript.update(spec.binding.state_id_sha256().as_bytes());
    transcript.update((spec.position as u64).to_le_bytes());
    transcript.update((spec.generated_items as u64).to_le_bytes());
    transcript.update((spec.max_generated_items as u64).to_le_bytes());
    transcript.update(spec.state_bytes.to_le_bytes());
}

pub(super) fn transcript_cancel(transcript: &mut Sha256, member: &MemberState) {
    transcript.update(b"cancel\0");
    transcript.update(member.binding.member_id_sha256().as_bytes());
    transcript.update(member.binding.state_id_sha256().as_bytes());
    transcript.update((member.position as u64).to_le_bytes());
    transcript.update((member.generated_items as u64).to_le_bytes());
    transcript.update(member.state_bytes.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
pub(super) fn transcript_commit(
    transcript: &mut Sha256,
    step: u64,
    input_sha256: &str,
    output_sha256: &str,
    rows: usize,
    continued: usize,
    completed: usize,
    cancelled: usize,
) {
    transcript.update(b"commit\0");
    transcript.update(step.to_le_bytes());
    transcript.update(input_sha256.as_bytes());
    transcript.update(output_sha256.as_bytes());
    transcript.update((rows as u64).to_le_bytes());
    transcript.update((continued as u64).to_le_bytes());
    transcript.update((completed as u64).to_le_bytes());
    transcript.update((cancelled as u64).to_le_bytes());
}
