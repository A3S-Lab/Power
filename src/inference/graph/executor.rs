use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::{EmbeddedRuntime, ExecutionPermit, TensorInput, TensorOutput, WeightStore};
use super::plan::{GraphNode, GraphOp, GraphPlan};
use super::value::GraphValue;

mod batch_norm;
mod biased_activation;
mod biased_swish;
mod concatenation;
mod constant_reshape;
mod contiguous_mean;
mod contiguous_transpose;
mod convolution_post;
#[cfg(feature = "embedded-cpu-optimized")]
mod cpu_graph_segment;
#[cfg(feature = "embedded-cuda")]
mod cuda_fast_divisor;
#[cfg(all(test, feature = "embedded-cuda"))]
mod cuda_graph_tests;
#[cfg(feature = "embedded-cuda")]
mod cuda_reproducibility;
mod depthwise;
mod gated_hard_sigmoid;
mod gelu_erf;
mod identity;
mod layer_norm_affine;
mod liveness;
mod matmul_bias;
mod max_pool;
mod pointwise_convolution;
mod profiling;
mod scalar_affine;
mod scalar_affine_hard_swish;
mod sigmoid_product;
mod spatial;
mod spatial_convolution;
mod tensor_geometry;
mod terminal_softmax;

#[cfg(feature = "embedded-cuda")]
use spatial::conv;
use spatial::{conv_transpose, pool, resize};

#[cfg(test)]
use tensor_geometry::same_upper_padding;
use tensor_geometry::{
    axis_index, execution_error, nonnegative_usize, normalized_axes, resolve_reshape, slice_bounds,
    slice_tensor,
};

/// Validated single-input/single-output static graph executor.
pub struct GraphExecutor {
    plan: GraphPlan,
    constants: HashMap<String, GraphValue>,
    scalar_constants: HashMap<String, f32>,
    batch_norms: HashMap<String, batch_norm::PreparedBatchNorm>,
    #[cfg(feature = "embedded-cpu-optimized")]
    cpu_graph_segments: HashMap<usize, cpu_graph_segment::PreparedSegment>,
    value_use_counts: HashMap<String, usize>,
    runtime: EmbeddedRuntime,
}

impl GraphExecutor {
    pub fn new(
        plan: GraphPlan,
        weights: Arc<WeightStore>,
        runtime: EmbeddedRuntime,
    ) -> Result<Self> {
        let mut plan = plan
            .elide_private_identities()
            .fold_private_transposes()
            .elide_private_identities();
        let mut constants = HashMap::with_capacity(plan.initializers.len());
        let mut scalar_constants = HashMap::new();
        for initializer in &plan.initializers {
            let (value, scalar) =
                GraphValue::load(initializer, &weights, runtime.device().tensor_device())?;
            constants.insert(initializer.name.clone(), value);
            if let Some(scalar) = scalar {
                scalar_constants.insert(initializer.name.clone(), scalar);
            }
        }
        let retained_outputs = plan
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        constant_reshape::fold_private_constants(
            &mut plan.nodes,
            &retained_outputs,
            &mut constants,
            &mut scalar_constants,
            runtime.limits().max_tensor_elements,
        );
        let value_use_counts = liveness::value_use_counts(&plan);
        let output_name = plan
            .outputs
            .first()
            .ok_or_else(|| {
                PowerError::InvalidFormat(
                    "static graph must declare exactly one output".to_string(),
                )
            })?
            .name
            .clone();
        let batch_norms = batch_norm::prepare(
            &plan.nodes,
            &constants,
            &scalar_constants,
            &value_use_counts,
            &output_name,
        );
        #[cfg(feature = "embedded-cpu-optimized")]
        let cpu_graph_segments = if runtime.device().tensor_device().is_cpu() {
            cpu_graph_segment::prepare(
                &plan,
                &constants,
                &value_use_counts,
                &output_name,
                runtime.limits(),
            )?
        } else {
            HashMap::new()
        };
        Ok(Self {
            plan,
            constants,
            scalar_constants,
            batch_norms,
            #[cfg(feature = "embedded-cpu-optimized")]
            cpu_graph_segments,
            value_use_counts,
            runtime,
        })
    }

    /// Executes a graph under a permit from the same shared runtime.
    pub fn run(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<TensorOutput> {
        self.run_with_output_projection(input, permit, cancellation, |output| Ok(output.clone()))
    }

    /// Executes a graph and applies one model-owned projection before the
    /// bounded output is copied back to the host.
    ///
    /// The projection remains on the graph device and is useful when a model
    /// consumes a compact deterministic view of a much larger graph output.
    /// The model crate owns the projection arithmetic and must bind it into
    /// its execution identity. Power still enforces permit, cancellation,
    /// device-residency, tensor-element, dtype, and finite-output bounds.
    pub fn run_with_output_projection<F>(
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
        let (input, upload_guard) = input.into_candle(
            self.runtime.device().tensor_device(),
            self.runtime.limits(),
            permit.input_upload_pool(),
        )?;
        let output = self.run_tensor(input, cancellation)?;
        let projected = projection(&output)?;
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        if !projected.device().same_device(output.device()) {
            return Err(PowerError::InferenceFailed(
                "static graph output projection changed tensor devices".to_string(),
            ));
        }
        let output = TensorOutput::from_candle(&projected, self.runtime.limits());
        upload_guard.complete();
        output
    }

    fn run_tensor(&self, input: Tensor, cancellation: &CancellationToken) -> Result<Tensor> {
        let output_name = self
            .plan
            .outputs
            .first()
            .ok_or_else(|| {
                PowerError::InvalidFormat(
                    "static graph must declare exactly one output".to_string(),
                )
            })?
            .name
            .clone();
        self.run_tensor_prefix(input, cancellation, self.plan.nodes.len(), &output_name)
    }

    fn run_tensor_prefix(
        &self,
        input: Tensor,
        cancellation: &CancellationToken,
        node_count: usize,
        output_name: &str,
    ) -> Result<Tensor> {
        let input_name = self
            .plan
            .inputs
            .first()
            .ok_or_else(|| {
                PowerError::InvalidFormat("static graph must declare exactly one input".to_string())
            })?
            .name
            .clone();
        let mut profile = profiling::GraphExecutionProfile::from_environment(
            input.dims(),
            self.runtime.device().tensor_device(),
        )?;
        let mut values = self.constants.clone();
        let mut remaining_uses = self.value_use_counts.clone();
        values.insert(input_name, GraphValue::Tensor(input));
        let mut node_index = 0;
        while let Some(node) = self
            .plan
            .nodes
            .get(node_index)
            .filter(|_| node_index < node_count)
        {
            let node_started = profile
                .as_ref()
                .map(|profile| profile.start_node(node, &values))
                .transpose()?;
            if cancellation.is_cancelled() {
                return Err(PowerError::InferenceFailed(
                    "static graph execution was cancelled".to_string(),
                ));
            }
            if identity::try_commit(node, output_name, &mut remaining_uses, &mut values)? {
                profiling::record_node(&mut profile, node_started, 1)?;
                node_index += 1;
                continue;
            }
            #[cfg(feature = "embedded-cpu-optimized")]
            if let Some(segment) = self.cpu_graph_segments.get(&node_index) {
                if let Some(fused) =
                    segment.try_execute(&values, self.runtime.limits(), cancellation)?
                {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
            }
            if node.op == GraphOp::BatchNormalization {
                if let Some(prepared) = self.batch_norms.get(&node.name) {
                    if let Some(fused) = batch_norm::try_execute(node, &values, prepared)? {
                        let (consumed, terminal) = fused_node_window(
                            &self.plan.nodes,
                            node_index,
                            fused.consumed_nodes,
                            &node.name,
                        )?;
                        for fused_node in consumed {
                            liveness::release_consumed_values(
                                &fused_node.inputs,
                                output_name,
                                &mut remaining_uses,
                                &mut values,
                            );
                        }
                        commit_node_output(
                            terminal,
                            fused.value,
                            output_name,
                            self.runtime.limits().max_tensor_elements,
                            &mut remaining_uses,
                            &mut values,
                        )?;
                        profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                        node_index += fused.consumed_nodes;
                        continue;
                    }
                }
            }
            if node.op == GraphOp::Conv {
                if let Some(fused) = batch_norm::try_execute_convolution(
                    &self.plan.nodes[node_index..],
                    batch_norm::ConvolutionExecutionContext {
                        values: &values,
                        prepared: &self.batch_norms,
                        use_counts: &self.value_use_counts,
                        retained_output: output_name,
                        device: self.runtime.device().tensor_device(),
                        element_limit: self.runtime.limits().max_tensor_elements,
                        cancellation,
                    },
                )? {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
                if let Some(fused) = biased_activation::try_execute(
                    &self.plan.nodes[node_index..],
                    biased_activation::ExecutionContext {
                        values: &values,
                        scalar_constants: &self.scalar_constants,
                        use_counts: &self.value_use_counts,
                        retained_output: output_name,
                        device: self.runtime.device().tensor_device(),
                        element_limit: self.runtime.limits().max_tensor_elements,
                        cancellation,
                    },
                )? {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
            }
            if node.op == GraphOp::MatMul {
                if let Some(fused) = matmul_bias::try_execute(
                    &self.plan.nodes[node_index..],
                    &values,
                    &self.value_use_counts,
                    output_name,
                    self.runtime.limits().max_tensor_elements,
                    cancellation,
                )? {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
            }
            if node.op == GraphOp::Add {
                if let Some(fused) = biased_swish::try_execute(
                    &self.plan.nodes[node_index..],
                    &values,
                    &self.value_use_counts,
                    output_name,
                    self.runtime.limits().max_tensor_elements,
                    cancellation,
                )? {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
            }
            if node.op == GraphOp::ReduceMean {
                if let Some(window) = self.plan.nodes.get(node_index..node_index + 9) {
                    if let Some(output) = layer_norm_affine::try_execute_full(
                        window,
                        &values,
                        &self.scalar_constants,
                        &self.value_use_counts,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        cancellation,
                    )? {
                        for fused_node in &window[..8] {
                            liveness::release_consumed_values(
                                &fused_node.inputs,
                                output_name,
                                &mut remaining_uses,
                                &mut values,
                            );
                        }
                        commit_node_output(
                            &window[8],
                            output,
                            output_name,
                            self.runtime.limits().max_tensor_elements,
                            &mut remaining_uses,
                            &mut values,
                        )?;
                        profiling::record_node(&mut profile, node_started, 9)?;
                        node_index += 9;
                        continue;
                    }
                }
            }
            if node.op == GraphOp::Add {
                if let Some(window) = self.plan.nodes.get(node_index..node_index + 5) {
                    if let Some(output) = layer_norm_affine::try_execute(
                        window,
                        &values,
                        &self.scalar_constants,
                        &self.value_use_counts,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        cancellation,
                    )? {
                        for fused_node in &window[..4] {
                            liveness::release_consumed_values(
                                &fused_node.inputs,
                                output_name,
                                &mut remaining_uses,
                                &mut values,
                            );
                        }
                        commit_node_output(
                            &window[4],
                            output,
                            output_name,
                            self.runtime.limits().max_tensor_elements,
                            &mut remaining_uses,
                            &mut values,
                        )?;
                        profiling::record_node(&mut profile, node_started, 5)?;
                        node_index += 5;
                        continue;
                    }
                }
            }
            if node.op == GraphOp::Div {
                if let Some(window) = self.plan.nodes.get(node_index..node_index + 5) {
                    if let Some(output) = gelu_erf::try_execute(
                        window,
                        &values,
                        &self.scalar_constants,
                        &self.value_use_counts,
                        output_name,
                        cancellation,
                    )? {
                        for fused_node in &window[..4] {
                            liveness::release_consumed_values(
                                &fused_node.inputs,
                                output_name,
                                &mut remaining_uses,
                                &mut values,
                            );
                        }
                        commit_node_output(
                            &window[4],
                            output,
                            output_name,
                            self.runtime.limits().max_tensor_elements,
                            &mut remaining_uses,
                            &mut values,
                        )?;
                        profiling::record_node(&mut profile, node_started, 5)?;
                        node_index += 5;
                        continue;
                    }
                }
            }
            if node.op == GraphOp::Sigmoid {
                if let Some(fused) = sigmoid_product::try_execute(
                    &self.plan.nodes[node_index..],
                    &values,
                    &self.value_use_counts,
                    output_name,
                    self.runtime.limits().max_tensor_elements,
                    cancellation,
                )? {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
            }
            if node.op == GraphOp::HardSigmoid {
                if let Some(next) = self.plan.nodes.get(node_index + 1) {
                    if let Some(output) = gated_hard_sigmoid::try_mul(
                        node,
                        next,
                        &values,
                        &self.value_use_counts,
                        output_name,
                        cancellation,
                    )? {
                        liveness::release_consumed_values(
                            &node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                        commit_node_output(
                            next,
                            output,
                            output_name,
                            self.runtime.limits().max_tensor_elements,
                            &mut remaining_uses,
                            &mut values,
                        )?;
                        profiling::record_node(&mut profile, node_started, 2)?;
                        node_index += 2;
                        continue;
                    }
                }
            }
            if node.op == GraphOp::Mul {
                if let Some(fused) = scalar_affine_hard_swish::try_execute(
                    &self.plan.nodes[node_index..],
                    &values,
                    &self.scalar_constants,
                    &self.value_use_counts,
                    output_name,
                    cancellation,
                )? {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
                if let Some(fused) = scalar_affine::try_execute(
                    &self.plan.nodes[node_index..],
                    &values,
                    &self.scalar_constants,
                    &self.value_use_counts,
                    output_name,
                    cancellation,
                )? {
                    let (consumed, terminal) = fused_node_window(
                        &self.plan.nodes,
                        node_index,
                        fused.consumed_nodes,
                        &node.name,
                    )?;
                    for fused_node in consumed {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        terminal,
                        fused.value,
                        output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    profiling::record_node(&mut profile, node_started, fused.consumed_nodes)?;
                    node_index += fused.consumed_nodes;
                    continue;
                }
            }
            let output = execute(
                node,
                &values,
                &self.scalar_constants,
                self.runtime.device().tensor_device(),
            )?;
            commit_node_output(
                node,
                output,
                output_name,
                self.runtime.limits().max_tensor_elements,
                &mut remaining_uses,
                &mut values,
            )?;
            profiling::record_node(&mut profile, node_started, 1)?;
            node_index += 1;
        }
        let output = values
            .remove(output_name)
            .ok_or_else(|| {
                PowerError::InferenceFailed("static graph returned no output".to_string())
            })?
            .tensor("graph output")
            .cloned()?;
        if let Some(profile) = profile {
            profile.emit()?;
        }
        Ok(output)
    }
}

fn fused_node_window<'a>(
    nodes: &'a [GraphNode],
    start: usize,
    consumed_nodes: usize,
    fusion_name: &str,
) -> Result<(&'a [GraphNode], &'a GraphNode)> {
    let end = start.checked_add(consumed_nodes).ok_or_else(|| {
        PowerError::InferenceFailed(format!(
            "static graph fusion '{fusion_name}' node window overflowed"
        ))
    })?;
    let window = nodes.get(start..end).ok_or_else(|| {
        PowerError::InferenceFailed(format!(
            "static graph fusion '{fusion_name}' requested {consumed_nodes} nodes from index {start}, but the graph contains {} nodes",
            nodes.len()
        ))
    })?;
    window
        .split_last()
        .map(|(terminal, consumed)| (consumed, terminal))
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph fusion '{fusion_name}' requested an empty node window"
            ))
        })
}

fn commit_node_output(
    node: &GraphNode,
    output: GraphValue,
    retained_output: &str,
    element_limit: usize,
    remaining_uses: &mut HashMap<String, usize>,
    values: &mut HashMap<String, GraphValue>,
) -> Result<()> {
    #[cfg(test)]
    trace_non_finite(node, &output)?;
    let elements = output
        .shape()
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' tensor element count overflowed",
                node.name
            ))
        })?;
    if elements > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {elements} tensor elements, exceeding the {element_limit}-element limit",
            node.name,
        )));
    }
    liveness::release_consumed_values(&node.inputs, retained_output, remaining_uses, values);
    let output_name = node.outputs.first().ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "static graph node '{}' is missing output 0",
            node.name
        ))
    })?;
    values.insert(output_name.clone(), output);
    Ok(())
}

#[cfg(test)]
fn trace_non_finite(node: &GraphNode, value: &GraphValue) -> Result<()> {
    if std::env::var_os("A3S_POWER_TRACE_NONFINITE").is_none() {
        return Ok(());
    }
    let GraphValue::Tensor(tensor) = value else {
        return Ok(());
    };
    let values = tensor
        .to_dtype(candle_core::DType::F32)
        .and_then(|value| value.to_device(&Device::Cpu))
        .and_then(|value| value.flatten_all())
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(|error| execution_error(node, error))?;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(execution_error(
            node,
            format!("produced non-finite value {value} at flat index {index}"),
        ));
    }
    Ok(())
}

fn execute(
    node: &GraphNode,
    values: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    device: &Device,
) -> Result<GraphValue> {
    let inputs = node
        .inputs
        .iter()
        .map(|name| {
            values.get(name).ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "static graph node '{}' could not resolve input '{name}'",
                    node.name
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let value = match node.op {
        GraphOp::Add => binary(node, &inputs, Tensor::broadcast_add)?,
        GraphOp::Sub => binary(node, &inputs, Tensor::broadcast_sub)?,
        GraphOp::Mul => binary(node, &inputs, Tensor::broadcast_mul)?,
        GraphOp::Div => binary(node, &inputs, Tensor::broadcast_div)?,
        GraphOp::Pow => pow(node, &inputs, scalar_constants)?,
        GraphOp::Erf => unary_tensor(node, &inputs, Tensor::erf)?,
        GraphOp::Relu => unary_tensor(node, &inputs, Tensor::relu)?,
        GraphOp::Sqrt => unary_tensor(node, &inputs, Tensor::sqrt)?,
        GraphOp::Sigmoid => GraphValue::Tensor(
            candle_nn::ops::sigmoid(required_tensor(node, &inputs, 0)?)
                .map_err(|error| execution_error(node, error))?,
        ),
        GraphOp::HardSigmoid => hard_sigmoid(node, &inputs)?,
        GraphOp::Identity => required(node, &inputs, 0)?.clone(),
        GraphOp::Concat => concat(node, &inputs)?,
        GraphOp::ReduceMean => reduce_mean(node, &inputs)?,
        GraphOp::GlobalAveragePool => global_average_pool(node, &inputs)?,
        GraphOp::Conv => spatial::conv(node, &inputs, device)?,
        GraphOp::ConvTranspose => conv_transpose(node, &inputs)?,
        GraphOp::MaxPool => pool(node, &inputs, true)?,
        GraphOp::AveragePool => pool(node, &inputs, false)?,
        GraphOp::Resize => resize(node, &inputs)?,
        GraphOp::BatchNormalization => batch_norm_fallback(node, &inputs)?,
        GraphOp::MatMul => matmul(node, &inputs)?,
        GraphOp::Reshape => reshape(node, &inputs)?,
        GraphOp::Shape => GraphValue::Ints {
            values: required(node, &inputs, 0)?
                .shape()
                .iter()
                .map(|value| i64::try_from(*value))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| execution_error(node, "shape dimension exceeds i64"))?,
            shape: vec![required(node, &inputs, 0)?.shape().len()],
        },
        GraphOp::Slice => slice(node, &inputs, device)?,
        GraphOp::Squeeze => squeeze(node, &inputs)?,
        GraphOp::Unsqueeze => unsqueeze(node, &inputs)?,
        GraphOp::Transpose => transpose(node, &inputs)?,
        GraphOp::Softmax => softmax(node, &inputs)?,
    };
    Ok(value)
}

fn required<'a>(
    node: &GraphNode,
    inputs: &'a [&GraphValue],
    index: usize,
) -> Result<&'a GraphValue> {
    inputs.get(index).copied().ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "static graph node '{}' is missing input {index}",
            node.name
        ))
    })
}

fn required_tensor<'a>(
    node: &GraphNode,
    inputs: &'a [&GraphValue],
    index: usize,
) -> Result<&'a Tensor> {
    required(node, inputs, index)?.tensor(&node.name)
}

fn binary(
    node: &GraphNode,
    inputs: &[&GraphValue],
    operation: fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
) -> Result<GraphValue> {
    let left = required_tensor(node, inputs, 0)?;
    let right = required_tensor(node, inputs, 1)?;
    operation(left, right)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn matmul(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let left = required_tensor(node, inputs, 0)?;
    let right = required_tensor(node, inputs, 1)?;
    if let Some(output) = super::matrix_multiplication::try_cuda_transposed_lhs(left, right)
        .map_err(|error| execution_error(node, error))?
    {
        return Ok(GraphValue::Tensor(output));
    }

    // ONNX Transpose and Slice legitimately produce strided views. Candle's
    // general matmul path still requires dense operands, so materialize only
    // this operator boundary instead of rejecting a valid reviewed graph.
    let left =
        contiguous_transpose::materialize(left).map_err(|error| execution_error(node, error))?;
    let right =
        contiguous_transpose::materialize(right).map_err(|error| execution_error(node, error))?;
    super::matrix_multiplication::broadcast(&left, &right)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn unary_tensor(
    node: &GraphNode,
    inputs: &[&GraphValue],
    operation: fn(&Tensor) -> candle_core::Result<Tensor>,
) -> Result<GraphValue> {
    operation(required_tensor(node, inputs, 0)?)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn pow(
    node: &GraphNode,
    inputs: &[&GraphValue],
    scalar_constants: &HashMap<String, f32>,
) -> Result<GraphValue> {
    let base = required_tensor(node, inputs, 0)?;
    let cached_exponent = cached_scalar_pow_enabled()
        .then(|| {
            node.inputs
                .get(1)
                .and_then(|name| scalar_constants.get(name))
                .copied()
        })
        .flatten();
    let is_square = if let Some(exponent) = cached_exponent {
        exponent == 2.0
    } else {
        let exponent = required_tensor(node, inputs, 1)?;
        let exponent = exponent
            .to_dtype(candle_core::DType::F32)
            .and_then(|value| value.to_device(&Device::Cpu))
            .and_then(|value| value.flatten_all())
            .and_then(|value| value.to_vec1::<f32>())
            .map_err(|error| execution_error(node, error))?;
        exponent.as_slice() == [2.0]
    };
    if !is_square {
        return Err(execution_error(
            node,
            "the static graph executor only permits a scalar square exponent",
        ));
    }
    base.sqr()
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

#[cfg(not(test))]
fn cached_scalar_pow_enabled() -> bool {
    true
}

#[cfg(test)]
fn cached_scalar_pow_enabled() -> bool {
    std::env::var_os("A3S_POWER_TEST_DISABLE_CACHED_SCALAR_POW").is_none()
}

fn hard_sigmoid(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let alpha = node.float("alpha", 0.2)?;
    let beta = node.float("beta", 0.5)?;
    let value = (required_tensor(node, inputs, 0)? * alpha)
        .and_then(|value| value.affine(1.0, beta))
        .and_then(|value| value.clamp(0.0, 1.0))
        .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(value))
}

fn concat(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let axis = node.int("axis", 0)?;
    match required(node, inputs, 0)? {
        GraphValue::Tensor(_) => {
            let tensors = inputs
                .iter()
                .map(|value| value.tensor(&node.name))
                .collect::<Result<Vec<_>>>()?;
            let rank = tensors[0].rank();
            let axis = axis_index(axis, rank, node)?;
            if tensors.len() == 2
                && tensors[0].device().is_cpu()
                && tensors.iter().all(|tensor| {
                    tensor.dtype() == candle_core::DType::F32 && tensor.is_contiguous()
                })
            {
                return concatenation::concat_two(tensors[0], tensors[1], axis)
                    .map(GraphValue::Tensor)
                    .map_err(|error| execution_error(node, error));
            }
            Tensor::cat(&tensors, axis)
                .map(GraphValue::Tensor)
                .map_err(|error| execution_error(node, error))
        }
        GraphValue::Ints { .. } => {
            if axis != 0 {
                return Err(execution_error(
                    node,
                    "control concatenation axis must be zero",
                ));
            }
            let mut values = Vec::new();
            for value in inputs {
                values.extend_from_slice(value.ints(&node.name)?);
            }
            let length = values.len();
            Ok(GraphValue::Ints {
                values,
                shape: vec![length],
            })
        }
    }
}

fn reduce_mean(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let axes = node.ints("axes", &[])?;
    let axes = normalized_axes(&axes, input.rank(), node)?;
    let keep = node.int("keepdims", 1)? != 0;
    if let Some(output) = contiguous_mean::try_execute(input, &axes, keep)
        .map_err(|error| execution_error(node, error))?
    {
        return Ok(GraphValue::Tensor(output));
    }
    let output = if keep {
        input.mean_keepdim(axes.as_slice())
    } else {
        input.mean(axes.as_slice())
    }
    .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(output))
}

fn global_average_pool(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    if input.rank() < 3 {
        return Err(execution_error(
            node,
            "global average pool requires rank >= 3",
        ));
    }
    let axes = (2..input.rank()).collect::<Vec<_>>();
    input
        .mean_keepdim(axes.as_slice())
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn batch_norm_fallback(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    if input.rank() < 2 {
        return Err(execution_error(
            node,
            "BatchNormalization input must have rank >= 2",
        ));
    }
    let channels = input.dim(1).map_err(|error| execution_error(node, error))?;
    let mut parameter_shape = vec![1_usize; input.rank()];
    parameter_shape[1] = channels;
    let broadcast = |index| -> Result<Tensor> {
        required_tensor(node, inputs, index)?
            .reshape(parameter_shape.as_slice())
            .map_err(|error| execution_error(node, error))
    };
    let scale = broadcast(1)?;
    let bias = broadcast(2)?;
    let mean = broadcast(3)?;
    let variance = broadcast(4)?;
    let epsilon = node.float("epsilon", 1e-5)?;
    let output = input
        .broadcast_sub(&mean)
        .and_then(|value| {
            variance
                .affine(1.0, epsilon)
                .and_then(|variance| variance.sqrt())
                .and_then(|stddev| value.broadcast_div(&stddev))
        })
        .and_then(|value| value.broadcast_mul(&scale))
        .and_then(|value| value.broadcast_add(&bias))
        .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(output))
}

fn reshape(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let requested = required(node, inputs, 1)?.ints(&node.name)?;
    let shape = resolve_reshape(input.dims(), requested, node)?;
    input
        .reshape(shape.as_slice())
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn slice(node: &GraphNode, inputs: &[&GraphValue], device: &Device) -> Result<GraphValue> {
    let starts = required(node, inputs, 1)?.ints(&node.name)?;
    let ends = required(node, inputs, 2)?.ints(&node.name)?;
    let default_axes = (0..starts.len())
        .map(|value| value as i64)
        .collect::<Vec<_>>();
    let axes = inputs
        .get(3)
        .map(|value| value.ints(&node.name))
        .transpose()?
        .unwrap_or(default_axes.as_slice());
    let default_steps = vec![1_i64; starts.len()];
    let steps = inputs
        .get(4)
        .map(|value| value.ints(&node.name))
        .transpose()?
        .unwrap_or(default_steps.as_slice());
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(execution_error(
            node,
            "Slice controls have different lengths",
        ));
    }
    match required(node, inputs, 0)? {
        GraphValue::Tensor(input) => {
            let mut output = input.clone();
            for (((start, end), axis), step) in starts.iter().zip(ends).zip(axes).zip(steps) {
                let axis = axis_index(*axis, output.rank(), node)?;
                output = slice_tensor(&output, axis, *start, *end, *step, device, node)?;
            }
            Ok(GraphValue::Tensor(output))
        }
        GraphValue::Ints { values, shape } => {
            if shape.len() != 1 || axes != [0] || steps != [1] {
                return Err(execution_error(node, "unsupported control Slice layout"));
            }
            let (start, end) = slice_bounds(values.len(), starts[0], ends[0], node)?;
            let values = values[start..end].to_vec();
            let length = values.len();
            Ok(GraphValue::Ints {
                values,
                shape: vec![length],
            })
        }
    }
}

fn squeeze(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let axes = node.ints("axes", &[])?;
    match required(node, inputs, 0)? {
        GraphValue::Tensor(input) => {
            let mut axes = normalized_axes(&axes, input.rank(), node)?;
            axes.sort_unstable_by(|left, right| right.cmp(left));
            let mut output = input.clone();
            for axis in axes {
                output = output
                    .squeeze(axis)
                    .map_err(|error| execution_error(node, error))?;
            }
            Ok(GraphValue::Tensor(output))
        }
        GraphValue::Ints { values, shape } => {
            let mut shape = shape.clone();
            let mut axes = normalized_axes(&axes, shape.len(), node)?;
            axes.sort_unstable_by(|left, right| right.cmp(left));
            for axis in axes {
                if shape[axis] != 1 {
                    return Err(execution_error(node, "cannot squeeze a non-unit dimension"));
                }
                shape.remove(axis);
            }
            Ok(GraphValue::Ints {
                values: values.clone(),
                shape,
            })
        }
    }
}

fn unsqueeze(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let axes = node.ints("axes", &[])?;
    match required(node, inputs, 0)? {
        GraphValue::Tensor(input) => {
            let final_rank = input.rank() + axes.len();
            let mut axes = normalized_axes(&axes, final_rank, node)?;
            axes.sort_unstable();
            let mut output = input.clone();
            for axis in axes {
                output = output
                    .unsqueeze(axis)
                    .map_err(|error| execution_error(node, error))?;
            }
            Ok(GraphValue::Tensor(output))
        }
        GraphValue::Ints { values, shape } => {
            let final_rank = shape.len() + axes.len();
            let mut axes = normalized_axes(&axes, final_rank, node)?;
            axes.sort_unstable();
            let mut shape = shape.clone();
            for axis in axes {
                shape.insert(axis, 1);
            }
            Ok(GraphValue::Ints {
                values: values.clone(),
                shape,
            })
        }
    }
}

fn transpose(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let default = (0..input.rank())
        .rev()
        .map(|value| value as i64)
        .collect::<Vec<_>>();
    let permutation = node.ints("perm", &default)?;
    let permutation = permutation
        .into_iter()
        .map(|value| nonnegative_usize(value, "perm", node))
        .collect::<Result<Vec<_>>>()?;
    let mut reviewed = permutation.clone();
    reviewed.sort_unstable();
    if reviewed != (0..input.rank()).collect::<Vec<_>>() {
        return Err(execution_error(node, "perm must be a complete permutation"));
    }
    if permutation.iter().copied().eq(0..input.rank()) {
        return Ok(GraphValue::Tensor(input.clone()));
    }
    input
        .permute(permutation.as_slice())
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn softmax(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let axis = axis_index(node.int("axis", -1)?, input.rank(), node)?;
    let output = if axis + 1 == input.rank() {
        candle_nn::ops::softmax_last_dim(input)
    } else {
        candle_nn::ops::softmax(input, axis)
    };
    output
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

#[cfg(test)]
mod tests;
