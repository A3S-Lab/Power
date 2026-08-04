use crate::error::{PowerError, Result};

use super::super::{
    EmbeddedRuntime, ExecutionBatchBinding, ExecutionDigest, ExecutionPermit, ExecutionReceipt,
    MicrobatchExecutionEvidence, ModelIdentity, PlannedMicrobatch,
};
use super::MicrobatchPlan;

/// Admitted guard for exactly one revalidated microbatch.
pub struct MicrobatchExecution {
    runtime: EmbeddedRuntime,
    permit: ExecutionPermit,
    binding: ExecutionBatchBinding,
    expected_model: Option<ModelIdentity>,
    session_declaration_sha256: Option<String>,
    plan_sha256: String,
    batch: PlannedMicrobatch,
    batch_count: usize,
}

impl MicrobatchExecution {
    pub(super) fn new(
        runtime: EmbeddedRuntime,
        permit: ExecutionPermit,
        plan: &MicrobatchPlan,
        batch: PlannedMicrobatch,
        expected_model: Option<ModelIdentity>,
    ) -> Result<Self> {
        plan.validate()?;
        if !permit.belongs_to(&runtime)
            || plan.runtime_device != runtime.device().identity()
            || batch.index >= plan.batches.len()
            || plan.batches[batch.index] != batch
        {
            return Err(PowerError::InvalidRequest(
                "microbatch execution does not match its runtime, permit, or plan".to_string(),
            ));
        }
        Ok(Self {
            runtime,
            permit,
            binding: plan.binding.clone(),
            expected_model,
            session_declaration_sha256: plan.session_declaration_sha256.clone(),
            plan_sha256: plan.declaration_sha256.clone(),
            batch,
            batch_count: plan.batches.len(),
        })
    }

    pub fn permit(&self) -> &ExecutionPermit {
        &self.permit
    }

    pub fn batch(&self) -> &PlannedMicrobatch {
        &self.batch
    }

    /// Builds a receipt bound to the exact admitted plan and pool session.
    pub fn receipt(
        &self,
        model: ModelIdentity,
        input: ExecutionDigest,
        output: ExecutionDigest,
    ) -> Result<ExecutionReceipt> {
        if self.binding.weights_sha256 != model.weights_sha256
            || self
                .expected_model
                .as_ref()
                .is_some_and(|expected| expected != &model)
        {
            return Err(PowerError::InvalidRequest(
                "microbatch receipt model does not match its execution plan or pool session"
                    .to_string(),
            ));
        }
        if input.byte_length != self.batch.input_bytes
            || input.item_count != self.batch.input_elements
        {
            return Err(PowerError::InvalidRequest(
                "microbatch receipt input does not match the planned aggregate input".to_string(),
            ));
        }
        let evidence = MicrobatchExecutionEvidence {
            schema: MicrobatchExecutionEvidence::SCHEMA.to_string(),
            session_declaration_sha256: self.session_declaration_sha256.clone(),
            plan_sha256: self.plan_sha256.clone(),
            batch_index: self.batch.index,
            batch_count: self.batch_count,
            slot_count: self.batch.slots.len(),
            model_admission_queued: self.permit.model_admission_was_queued(),
            device_admission_queued: self.permit.device_admission_was_queued(),
        };
        evidence.validate()?;
        let mut receipt = self.runtime.receipt(model, input, output);
        receipt.schema = ExecutionReceipt::MICROBATCH_SCHEMA.to_string();
        receipt.microbatch = Some(evidence);
        Ok(receipt)
    }
}

impl std::fmt::Debug for MicrobatchExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MicrobatchExecution")
            .field("plan", &"sha256")
            .field("batch_index", &self.batch.index)
            .field("slot_count", &self.batch.slots.len())
            .finish_non_exhaustive()
    }
}
