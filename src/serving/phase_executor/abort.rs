use uuid::Uuid;

use crate::error::Result;

use super::{validate_command_identity, PhaseExecutionHandle};

/// Cancellation command for either an in-progress preparation or a prepared
/// backend execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbortPhaseExecution {
    pub execution_id: Uuid,
    pub local_worker_epoch: Uuid,
    pub execution: Option<PhaseExecutionHandle>,
}

impl AbortPhaseExecution {
    pub fn preparing(execution_id: Uuid, local_worker_epoch: Uuid) -> Result<Self> {
        let command = Self {
            execution_id,
            local_worker_epoch,
            execution: None,
        };
        command.validate()?;
        Ok(command)
    }

    pub fn prepared(
        execution_id: Uuid,
        local_worker_epoch: Uuid,
        execution: PhaseExecutionHandle,
    ) -> Result<Self> {
        let command = Self {
            execution_id,
            local_worker_epoch,
            execution: Some(execution),
        };
        command.validate()?;
        Ok(command)
    }

    pub fn validate(&self) -> Result<()> {
        validate_command_identity(self.execution_id, self.local_worker_epoch)
    }

    pub fn execution(&self) -> Option<&PhaseExecutionHandle> {
        self.execution.as_ref()
    }
}
