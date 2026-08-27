use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use super::{axis_index, GraphExecutor, GraphNode, GraphOp, TensorInput, TensorOutput};
use crate::error::{PowerError, Result};
use crate::inference::ExecutionPermit;

struct TerminalClassifierBoundary<'a> {
    terminal: &'a GraphNode,
    matmul_index: usize,
    features_name: &'a str,
    weights: &'a Tensor,
    bias: &'a Tensor,
}

mod row_coalesced;

impl GraphExecutor {
    /// Executes a graph through the input of a terminal last-axis Softmax and
    /// applies a model-owned equivalent projection before host materialization.
    ///
    /// The reviewed graph must end in exactly one Softmax that publishes the
    /// graph output. Power validates that boundary and retains the same permit,
    /// cancellation, device-residency, and output bounds as ordinary graph
    /// execution. The model crate owns the projected Softmax arithmetic and
    /// must bind it into its execution identity.
    pub fn run_with_terminal_softmax_projection<F>(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<TensorOutput>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        if !permit.belongs_to(&self.runtime) {
            return Err(PowerError::InvalidRequest(
                "graph execution permit belongs to a different embedded runtime".to_string(),
            ));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }

        let terminal = self.plan.nodes.last().ok_or_else(|| {
            PowerError::InvalidFormat("static graph contains no terminal node".to_string())
        })?;
        let [logits_name] = terminal.inputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax must consume exactly one tensor".to_string(),
            ));
        };
        let [terminal_output] = terminal.outputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax must publish exactly one tensor".to_string(),
            ));
        };
        if terminal.op != GraphOp::Softmax || terminal_output != &self.plan.outputs[0].name {
            return Err(PowerError::InvalidFormat(
                "static graph does not end in the declared output Softmax".to_string(),
            ));
        }

        let (input, upload_guard) = input.into_candle(
            self.runtime.device().tensor_device(),
            self.runtime.limits(),
            permit.input_upload_pool(),
        )?;
        let logits =
            self.run_tensor_prefix(input, cancellation, self.plan.nodes.len() - 1, logits_name)?;
        let axis = axis_index(terminal.int("axis", -1)?, logits.rank(), terminal)?;
        if logits.rank() == 0 || axis != logits.rank() - 1 {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax projection requires the last tensor axis".to_string(),
            ));
        }
        let projected = projection(&logits)?;
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        if !projected.device().same_device(logits.device()) {
            return Err(PowerError::InferenceFailed(
                "terminal Softmax projection changed tensor devices".to_string(),
            ));
        }
        let output = TensorOutput::from_candle(&projected, self.runtime.limits());
        upload_guard.complete();
        output
    }

    /// Executes through the dynamic input of a private terminal
    /// `Add(initializer bias) -> Identity* -> Softmax` chain and applies a
    /// model-owned equivalent projection before host materialization.
    ///
    /// Every skipped intermediate must have one consumer, the Add must have
    /// exactly one initializer input, and Softmax must use the last axis and
    /// publish the graph output. These topology checks keep the optimization
    /// independent of model names while preventing removal of observable graph
    /// work.
    pub fn run_with_terminal_bias_softmax_projection<F>(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<TensorOutput>
    where
        F: FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
    {
        if !permit.belongs_to(&self.runtime) {
            return Err(PowerError::InvalidRequest(
                "graph execution permit belongs to a different embedded runtime".to_string(),
            ));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }

        let terminal_index = self.plan.nodes.len().checked_sub(1).ok_or_else(|| {
            PowerError::InvalidFormat("static graph contains no terminal node".to_string())
        })?;
        let terminal = &self.plan.nodes[terminal_index];
        let [softmax_input] = terminal.inputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax must consume exactly one tensor".to_string(),
            ));
        };
        let [terminal_output] = terminal.outputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax must publish exactly one tensor".to_string(),
            ));
        };
        if terminal.op != GraphOp::Softmax || terminal_output != &self.plan.outputs[0].name {
            return Err(PowerError::InvalidFormat(
                "static graph does not end in the declared output Softmax".to_string(),
            ));
        }

        let mut chained_input = softmax_input.as_str();
        let mut add_boundary = terminal_index;
        while let Some(identity) = add_boundary
            .checked_sub(1)
            .and_then(|index| self.plan.nodes.get(index))
            .filter(|node| node.op == GraphOp::Identity)
        {
            let [identity_input] = identity.inputs.as_slice() else {
                return Err(PowerError::InvalidFormat(
                    "terminal Identity must consume exactly one tensor".to_string(),
                ));
            };
            let [identity_output] = identity.outputs.as_slice() else {
                return Err(PowerError::InvalidFormat(
                    "terminal Identity must publish exactly one tensor".to_string(),
                ));
            };
            if identity_output != chained_input {
                break;
            }
            require_private_value(identity_output, &self.value_use_counts)?;
            chained_input = identity_input;
            add_boundary -= 1;
        }

        let add_index = add_boundary.checked_sub(1).ok_or_else(|| {
            PowerError::InvalidFormat(
                "terminal bias Softmax projection requires a preceding Add".to_string(),
            )
        })?;
        let add = &self.plan.nodes[add_index];
        let [add_output] = add.outputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal bias Add must publish exactly one tensor".to_string(),
            ));
        };
        let [left, right] = add.inputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal bias Add must consume exactly two tensors".to_string(),
            ));
        };
        if add.op != GraphOp::Add || add_output != chained_input {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax chain does not end in a private bias Add".to_string(),
            ));
        }
        require_private_value(add_output, &self.value_use_counts)?;

        let (dynamic_input, bias) = match (self.constants.get(left), self.constants.get(right)) {
            (None, Some(bias)) => (left.clone(), bias.tensor(&add.name)?.clone()),
            (Some(bias), None) => (right.clone(), bias.tensor(&add.name)?.clone()),
            _ => {
                return Err(PowerError::InvalidFormat(
                    "terminal bias Add must have exactly one initializer input".to_string(),
                ));
            }
        };

        let (input, upload_guard) = input.into_candle(
            self.runtime.device().tensor_device(),
            self.runtime.limits(),
            permit.input_upload_pool(),
        )?;
        let logits =
            self.run_tensor_prefix(input, cancellation, add_index, dynamic_input.as_str())?;
        let axis = axis_index(terminal.int("axis", -1)?, logits.rank(), terminal)?;
        if logits.rank() == 0 || axis != logits.rank() - 1 {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax projection requires the last tensor axis".to_string(),
            ));
        }
        if !bias.device().same_device(logits.device()) {
            return Err(PowerError::InferenceFailed(
                "terminal bias and logits use different tensor devices".to_string(),
            ));
        }
        let projected = projection(&logits, &bias)?;
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        if !projected.device().same_device(logits.device()) {
            return Err(PowerError::InferenceFailed(
                "terminal Softmax projection changed tensor devices".to_string(),
            ));
        }
        let output = TensorOutput::from_candle(&projected, self.runtime.limits());
        upload_guard.complete();
        output
    }

    /// Executes through the dynamic input of a private terminal
    /// `MatMul(initializer weights) -> Add(initializer bias) -> Identity* ->
    /// Softmax` chain and applies a model-owned bounded classifier projection.
    ///
    /// The topology, initializer ownership, private intermediate values,
    /// last-axis Softmax, permit, cancellation, residency, and output bounds
    /// are validated before any terminal graph work is skipped.
    pub fn run_with_terminal_matmul_bias_softmax_projection<F>(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<TensorOutput>
    where
        F: FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        self.validate_projection_execution(permit, cancellation)?;
        let boundary = self.terminal_classifier_boundary()?;
        let (input, upload_guard) = input.into_candle(
            self.runtime.device().tensor_device(),
            self.runtime.limits(),
            permit.input_upload_pool(),
        )?;
        let projected =
            self.enqueue_terminal_classifier(input, &boundary, cancellation, projection)?;
        let output = TensorOutput::from_candle(&projected, self.runtime.limits());
        upload_guard.complete();
        output
    }

    /// Executes an ordered, aggregate-resource-bounded input window through
    /// the same reviewed terminal classifier boundary.
    ///
    /// Inputs are uploaded before any graph is enqueued, projected outputs
    /// remain device-resident until the complete window has been submitted,
    /// and only then begin host materialization. This removes upload/output
    /// fences between independent graph calls without changing tensor shapes,
    /// arithmetic, execution order, permit ownership, or cancellation checks.
    pub fn run_many_with_terminal_matmul_bias_softmax_projection<F>(
        &self,
        inputs: Vec<TensorInput>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<Vec<TensorOutput>>
    where
        F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        self.validate_projection_execution(permit, cancellation)?;
        let boundary = self.terminal_classifier_boundary()?;
        let (inputs, upload_guard) = TensorInput::into_candle_many(
            inputs,
            self.runtime.device().tensor_device(),
            self.runtime.limits(),
            permit.input_upload_pool(),
        )?;
        let mut projected = Vec::with_capacity(inputs.len());
        for input in inputs {
            projected.push(self.enqueue_terminal_classifier(
                input,
                &boundary,
                cancellation,
                &projection,
            )?);
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        let outputs = TensorOutput::from_candle_many(projected, self.runtime.limits());
        upload_guard.complete();
        outputs
    }

    fn validate_projection_execution(
        &self,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        if !permit.belongs_to(&self.runtime) {
            return Err(PowerError::InvalidRequest(
                "graph execution permit belongs to a different embedded runtime".to_string(),
            ));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        Ok(())
    }

    fn terminal_classifier_boundary(&self) -> Result<TerminalClassifierBoundary<'_>> {
        let terminal_index = self.plan.nodes.len().checked_sub(1).ok_or_else(|| {
            PowerError::InvalidFormat("static graph contains no terminal node".to_string())
        })?;
        let terminal = &self.plan.nodes[terminal_index];
        let [softmax_input] = terminal.inputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax must consume exactly one tensor".to_string(),
            ));
        };
        let [terminal_output] = terminal.outputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax must publish exactly one tensor".to_string(),
            ));
        };
        if terminal.op != GraphOp::Softmax || terminal_output != &self.plan.outputs[0].name {
            return Err(PowerError::InvalidFormat(
                "static graph does not end in the declared output Softmax".to_string(),
            ));
        }

        let mut chained_input = softmax_input.as_str();
        let mut add_boundary = terminal_index;
        while let Some(identity) = add_boundary
            .checked_sub(1)
            .and_then(|index| self.plan.nodes.get(index))
            .filter(|node| node.op == GraphOp::Identity)
        {
            let [identity_input] = identity.inputs.as_slice() else {
                return Err(PowerError::InvalidFormat(
                    "terminal Identity must consume exactly one tensor".to_string(),
                ));
            };
            let [identity_output] = identity.outputs.as_slice() else {
                return Err(PowerError::InvalidFormat(
                    "terminal Identity must publish exactly one tensor".to_string(),
                ));
            };
            if identity_output != chained_input {
                break;
            }
            require_private_value(identity_output, &self.value_use_counts)?;
            chained_input = identity_input;
            add_boundary -= 1;
        }

        let add_index = add_boundary.checked_sub(1).ok_or_else(|| {
            PowerError::InvalidFormat(
                "terminal classifier projection requires a preceding bias Add".to_string(),
            )
        })?;
        let add = &self.plan.nodes[add_index];
        let [add_output] = add.outputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal bias Add must publish exactly one tensor".to_string(),
            ));
        };
        let [add_left, add_right] = add.inputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal bias Add must consume exactly two tensors".to_string(),
            ));
        };
        if add.op != GraphOp::Add || add_output != chained_input {
            return Err(PowerError::InvalidFormat(
                "terminal classifier chain does not end in a private bias Add".to_string(),
            ));
        }
        require_private_value(add_output, &self.value_use_counts)?;
        let (matmul_output, bias) =
            match (self.constants.get(add_left), self.constants.get(add_right)) {
                (None, Some(bias)) => (add_left, bias.tensor(&add.name)?),
                (Some(bias), None) => (add_right, bias.tensor(&add.name)?),
                _ => {
                    return Err(PowerError::InvalidFormat(
                        "terminal bias Add must have exactly one initializer input".to_string(),
                    ));
                }
            };

        let matmul_index = add_index.checked_sub(1).ok_or_else(|| {
            PowerError::InvalidFormat(
                "terminal classifier projection requires a preceding MatMul".to_string(),
            )
        })?;
        let matmul = &self.plan.nodes[matmul_index];
        let [matmul_published] = matmul.outputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal classifier MatMul must publish exactly one tensor".to_string(),
            ));
        };
        let [features_name, weights_name] = matmul.inputs.as_slice() else {
            return Err(PowerError::InvalidFormat(
                "terminal classifier MatMul must consume exactly two tensors".to_string(),
            ));
        };
        if matmul.op != GraphOp::MatMul || matmul_published != matmul_output {
            return Err(PowerError::InvalidFormat(
                "terminal bias Add is not preceded by its private classifier MatMul".to_string(),
            ));
        }
        require_private_value(matmul_published, &self.value_use_counts)?;
        if self.constants.contains_key(features_name) {
            return Err(PowerError::InvalidFormat(
                "terminal classifier features must be graph-derived".to_string(),
            ));
        }
        let weights = self
            .constants
            .get(weights_name)
            .ok_or_else(|| {
                PowerError::InvalidFormat(
                    "terminal classifier weights must be an initializer".to_string(),
                )
            })?
            .tensor(&matmul.name)?;

        Ok(TerminalClassifierBoundary {
            terminal,
            matmul_index,
            features_name,
            weights,
            bias,
        })
    }

    fn enqueue_terminal_classifier<F>(
        &self,
        input: Tensor,
        boundary: &TerminalClassifierBoundary<'_>,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<Tensor>
    where
        F: FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        let features = self.enqueue_terminal_classifier_features(input, boundary, cancellation)?;
        let projected = projection(&features, boundary.weights, boundary.bias)?;
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        if !projected.device().same_device(features.device()) {
            return Err(PowerError::InferenceFailed(
                "terminal classifier projection changed tensor devices".to_string(),
            ));
        }
        Ok(projected)
    }

    fn enqueue_terminal_classifier_features(
        &self,
        input: Tensor,
        boundary: &TerminalClassifierBoundary<'_>,
        cancellation: &CancellationToken,
    ) -> Result<Tensor> {
        let features = self.run_tensor_prefix(
            input,
            cancellation,
            boundary.matmul_index,
            boundary.features_name,
        )?;
        let axis = axis_index(
            boundary.terminal.int("axis", -1)?,
            features.rank(),
            boundary.terminal,
        )?;
        if features.rank() == 0 || axis != features.rank() - 1 {
            return Err(PowerError::InvalidFormat(
                "terminal Softmax projection requires the last tensor axis".to_string(),
            ));
        }
        if !boundary.weights.device().same_device(features.device())
            || !boundary.bias.device().same_device(features.device())
        {
            return Err(PowerError::InferenceFailed(
                "terminal classifier tensors use different devices".to_string(),
            ));
        }
        Ok(features)
    }
}

fn require_private_value(
    name: &str,
    value_use_counts: &std::collections::HashMap<String, usize>,
) -> Result<()> {
    if value_use_counts.get(name).copied() != Some(1) {
        return Err(PowerError::InvalidFormat(format!(
            "terminal Softmax intermediate '{name}' must have exactly one consumer"
        )));
    }
    Ok(())
}
