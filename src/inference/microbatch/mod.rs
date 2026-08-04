mod execution;
pub(super) mod planner;
mod types;

pub use execution::MicrobatchExecution;
pub use types::{
    MicrobatchCandidate, MicrobatchLimits, MicrobatchPlan, MicrobatchPolicy, PlannedMicrobatch,
    PlannedMicrobatchSlot,
};

use crate::error::Result;

use super::{EmbeddedRuntime, ExecutionBatchBinding, ModelSession, RuntimeDeviceIdentity};

impl EmbeddedRuntime {
    /// Creates a deterministic microbatch plan from a fresh memory snapshot.
    pub fn plan_microbatches(
        &self,
        binding: ExecutionBatchBinding,
        policy: MicrobatchPolicy,
        candidates: Vec<MicrobatchCandidate>,
    ) -> Result<MicrobatchPlan> {
        planner::plan(
            self.device().identity(),
            self.memory_snapshot()?,
            binding,
            None,
            self.limits().clone(),
            policy,
            candidates,
        )
    }

    /// Rechecks an existing plan against current memory pressure.
    pub fn revalidate_microbatch_plan(&self, plan: &MicrobatchPlan) -> Result<()> {
        plan.revalidate_for_runtime(
            self.device().identity(),
            self.limits(),
            &self.memory_snapshot()?,
        )
    }

    /// Revalidates current pressure and waits through bounded admission before
    /// exposing a permit for one standalone microbatch.
    pub async fn begin_microbatch(
        &self,
        plan: &MicrobatchPlan,
        batch_index: usize,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<MicrobatchExecution> {
        if plan.session_declaration_sha256.is_some() {
            return Err(crate::error::PowerError::InvalidRequest(
                "a pooled microbatch plan requires its exact model session".to_string(),
            ));
        }
        self.revalidate_microbatch_plan(plan)?;
        let batch = plan.batches.get(batch_index).cloned().ok_or_else(|| {
            crate::error::PowerError::InvalidRequest(
                "microbatch execution index is outside the plan".to_string(),
            )
        })?;
        let permit = self.begin_wait(cancellation).await?;
        MicrobatchExecution::new(self.clone(), permit, plan, batch, None)
    }
}

impl<T> ModelSession<T> {
    /// Plans microbatches bound to this exact pooled model session.
    pub fn plan_microbatches(
        &self,
        binding: ExecutionBatchBinding,
        policy: MicrobatchPolicy,
        candidates: Vec<MicrobatchCandidate>,
    ) -> Result<MicrobatchPlan> {
        planner::plan(
            self.runtime().device().identity(),
            self.runtime().memory_snapshot()?,
            binding,
            Some(self.declaration_sha256().to_string()),
            self.runtime().limits().clone(),
            policy,
            candidates,
        )
    }

    /// Revalidates and admits one microbatch against this exact pooled session.
    pub async fn begin_microbatch(
        &self,
        plan: &MicrobatchPlan,
        batch_index: usize,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<MicrobatchExecution> {
        if plan.session_declaration_sha256.as_deref() != Some(self.declaration_sha256()) {
            return Err(crate::error::PowerError::InvalidRequest(
                "microbatch plan belongs to a different model session".to_string(),
            ));
        }
        self.runtime().revalidate_microbatch_plan(plan)?;
        let batch = plan.batches.get(batch_index).cloned().ok_or_else(|| {
            crate::error::PowerError::InvalidRequest(
                "microbatch execution index is outside the plan".to_string(),
            )
        })?;
        let permit = self.runtime().begin_wait(cancellation).await?;
        MicrobatchExecution::new(
            self.runtime().clone(),
            permit,
            plan,
            batch,
            Some(self.binding().model.clone()),
        )
    }
}

pub(super) fn device_name(device: RuntimeDeviceIdentity) -> String {
    device.name()
}
