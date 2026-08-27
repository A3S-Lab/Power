use std::collections::BTreeMap;
#[cfg(feature = "embedded-cuda")]
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "embedded-cuda")]
use candle_core::cuda_backend::cudarc::driver::sys::CUevent_flags;
#[cfg(feature = "embedded-cuda")]
use candle_core::cuda_backend::cudarc::driver::{CudaEvent, CudaStream};
use candle_core::Device;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

const TRACE_ENVIRONMENT_VARIABLE: &str = "A3S_POWER_TRACE_GRAPH_TIMINGS";
const TOTALS_TRACE_ENVIRONMENT_VARIABLE: &str = "A3S_POWER_TRACE_GRAPH_TOTALS";
const CUDA_TRACE_ENVIRONMENT_VARIABLE: &str = "A3S_POWER_TRACE_CUDA_GRAPH_TIMINGS";

pub(super) struct GraphExecutionProfile {
    input_shape: Vec<usize>,
    started: Instant,
    record_host_operations: bool,
    record_host_totals: bool,
    operations: BTreeMap<String, OperationTiming>,
    #[cfg(feature = "embedded-cuda")]
    cuda: Option<CudaExecutionProfile>,
}

pub(super) struct NodeTiming {
    operation: Option<String>,
    started: Option<Instant>,
    #[cfg(feature = "embedded-cuda")]
    cuda_started: Option<CudaEvent>,
}

#[derive(Default)]
struct OperationTiming {
    executions: usize,
    consumed_nodes: usize,
    elapsed: Duration,
}

#[cfg(feature = "embedded-cuda")]
struct CudaExecutionProfile {
    stream: Arc<CudaStream>,
    started: CudaEvent,
    nodes: Vec<CudaNodeTiming>,
}

#[cfg(feature = "embedded-cuda")]
struct CudaNodeTiming {
    operation: String,
    consumed_nodes: usize,
    started: CudaEvent,
    completed: CudaEvent,
}

#[cfg(feature = "embedded-cuda")]
#[derive(Default)]
struct CudaOperationTiming {
    executions: usize,
    consumed_nodes: usize,
    elapsed_ms: f64,
}

#[cfg(feature = "embedded-cuda")]
struct CudaProfileReport {
    total_ms: f64,
    attributed_ms: f64,
    operations: BTreeMap<String, CudaOperationTiming>,
}

impl GraphExecutionProfile {
    pub(super) fn from_environment(input_shape: &[usize], device: &Device) -> Result<Option<Self>> {
        Self::from_requests(
            input_shape,
            device,
            std::env::var_os(TRACE_ENVIRONMENT_VARIABLE).is_some(),
            std::env::var_os(TOTALS_TRACE_ENVIRONMENT_VARIABLE).is_some(),
            std::env::var_os(CUDA_TRACE_ENVIRONMENT_VARIABLE).is_some(),
        )
    }

    fn from_requests(
        input_shape: &[usize],
        device: &Device,
        record_host_operations: bool,
        record_host_totals: bool,
        record_cuda: bool,
    ) -> Result<Option<Self>> {
        if !record_host_operations && !record_host_totals && !record_cuda {
            return Ok(None);
        }
        let profile = if record_host_operations {
            Self::new(input_shape)
        } else if record_host_totals {
            Self::totals_only(input_shape)
        } else {
            Self::cuda_only(input_shape)
        };
        if record_cuda {
            #[cfg(feature = "embedded-cuda")]
            {
                return profile.with_cuda(device).map(Some);
            }
            #[cfg(not(feature = "embedded-cuda"))]
            {
                let _ = device;
                return Err(PowerError::BackendNotAvailable(
                    "CUDA graph timing requires the embedded-cuda feature".to_string(),
                ));
            }
        }
        Ok(Some(profile))
    }

    #[cfg(feature = "embedded-cuda")]
    fn with_cuda(mut self, device: &Device) -> Result<Self> {
        self.cuda = Some(CudaExecutionProfile::new(device)?);
        Ok(self)
    }

    fn new(input_shape: &[usize]) -> Self {
        Self {
            input_shape: input_shape.to_vec(),
            started: Instant::now(),
            record_host_operations: true,
            record_host_totals: false,
            operations: BTreeMap::new(),
            #[cfg(feature = "embedded-cuda")]
            cuda: None,
        }
    }

    fn totals_only(input_shape: &[usize]) -> Self {
        Self {
            input_shape: input_shape.to_vec(),
            started: Instant::now(),
            record_host_operations: false,
            record_host_totals: true,
            operations: BTreeMap::new(),
            #[cfg(feature = "embedded-cuda")]
            cuda: None,
        }
    }

    fn cuda_only(input_shape: &[usize]) -> Self {
        Self {
            input_shape: input_shape.to_vec(),
            started: Instant::now(),
            record_host_operations: false,
            record_host_totals: false,
            operations: BTreeMap::new(),
            #[cfg(feature = "embedded-cuda")]
            cuda: None,
        }
    }

    pub(super) fn start_node(
        &self,
        node: &GraphNode,
        values: &std::collections::HashMap<String, GraphValue>,
    ) -> Result<NodeTiming> {
        let operation = self
            .records_operation_keys()
            .then(|| operation_key(node, values));
        let started = self.record_host_operations.then(Instant::now);
        #[cfg(feature = "embedded-cuda")]
        let cuda_started = self
            .cuda
            .as_ref()
            .map(CudaExecutionProfile::record_event)
            .transpose()?;
        Ok(NodeTiming {
            operation,
            started,
            #[cfg(feature = "embedded-cuda")]
            cuda_started,
        })
    }

    fn records_operation_keys(&self) -> bool {
        self.record_host_operations || {
            #[cfg(feature = "embedded-cuda")]
            {
                self.cuda.is_some()
            }
            #[cfg(not(feature = "embedded-cuda"))]
            {
                false
            }
        }
    }

    fn record(&mut self, timing: NodeTiming, consumed_nodes: usize) -> Result<()> {
        let NodeTiming {
            operation,
            started,
            #[cfg(feature = "embedded-cuda")]
            cuda_started,
        } = timing;
        if let (Some(operation), Some(started)) = (operation.as_ref(), started) {
            self.record_elapsed(operation.clone(), consumed_nodes, started.elapsed());
        }
        #[cfg(feature = "embedded-cuda")]
        if let Some(started) = cuda_started {
            let operation = operation.ok_or_else(|| {
                PowerError::InferenceFailed(
                    "CUDA graph timing lost its content-free operation key".to_string(),
                )
            })?;
            let cuda = self.cuda.as_mut().ok_or_else(|| {
                PowerError::InferenceFailed(
                    "CUDA graph timing lost its execution stream".to_string(),
                )
            })?;
            let completed = cuda.record_event()?;
            cuda.nodes.push(CudaNodeTiming {
                operation,
                consumed_nodes,
                started,
                completed,
            });
        }
        Ok(())
    }

    fn record_elapsed(&mut self, operation: String, consumed_nodes: usize, elapsed: Duration) {
        let timing = self.operations.entry(operation).or_default();
        timing.executions = timing.executions.saturating_add(1);
        timing.consumed_nodes = timing.consumed_nodes.saturating_add(consumed_nodes);
        timing.elapsed = timing.elapsed.saturating_add(elapsed);
    }

    pub(super) fn emit(self) -> Result<()> {
        let host_submit_ms = self.started.elapsed().as_secs_f64() * 1_000.0;
        if self.record_host_operations {
            eprintln!(
                "A3S_POWER_GRAPH_TIMING input_shape={:?} total_ms={host_submit_ms:.3} operations={}",
                self.input_shape,
                self.render_operations(),
            );
        } else if self.record_host_totals {
            eprintln!(
                "A3S_POWER_GRAPH_TOTAL input_shape={:?} total_ms={host_submit_ms:.3}",
                self.input_shape,
            );
        }
        #[cfg(feature = "embedded-cuda")]
        if let Some(cuda) = self.cuda {
            let report = cuda.finish()?;
            let unattributed_ms = (report.total_ms - report.attributed_ms).max(0.0);
            eprintln!(
                "A3S_POWER_CUDA_GRAPH_TIMING input_shape={:?} host_submit_ms={host_submit_ms:.3} stream_ms={:.3} attributed_ms={:.3} unattributed_ms={unattributed_ms:.3} operations={}",
                self.input_shape,
                report.total_ms,
                report.attributed_ms,
                render_cuda_operations(&report.operations),
            );
        }
        Ok(())
    }

    fn render_operations(&self) -> String {
        self.operations
            .iter()
            .map(|(name, timing)| {
                format!(
                    "{name}:executions={}:nodes={}:ms={:.3}",
                    timing.executions,
                    timing.consumed_nodes,
                    timing.elapsed.as_secs_f64() * 1_000.0,
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

pub(super) fn record_node(
    profile: &mut Option<GraphExecutionProfile>,
    started: Option<NodeTiming>,
    consumed_nodes: usize,
) -> Result<()> {
    if let (Some(profile), Some(started)) = (profile, started) {
        profile.record(started, consumed_nodes)?;
    }
    Ok(())
}

#[cfg(feature = "embedded-cuda")]
impl CudaExecutionProfile {
    fn new(device: &Device) -> Result<Self> {
        let Device::Cuda(device) = device else {
            return Err(PowerError::InvalidRequest(
                "CUDA graph timing requires a CUDA execution device".to_string(),
            ));
        };
        let stream = device.cuda_stream();
        let started = record_cuda_event(&stream, "record graph start")?;
        Ok(Self {
            stream,
            started,
            nodes: Vec::new(),
        })
    }

    fn record_event(&self) -> Result<CudaEvent> {
        record_cuda_event(&self.stream, "record node boundary")
    }

    fn finish(self) -> Result<CudaProfileReport> {
        let completed = record_cuda_event(&self.stream, "record graph completion")?;
        let total_ms = cuda_elapsed_ms(&self.started, &completed, "measure graph interval")?;
        let mut attributed_ms = 0.0_f64;
        let mut operations = BTreeMap::<String, CudaOperationTiming>::new();
        for node in self.nodes {
            let elapsed_ms =
                cuda_elapsed_ms(&node.started, &node.completed, "measure node interval")?;
            attributed_ms += elapsed_ms;
            let timing = operations.entry(node.operation).or_default();
            timing.executions = timing.executions.saturating_add(1);
            timing.consumed_nodes = timing.consumed_nodes.saturating_add(node.consumed_nodes);
            timing.elapsed_ms += elapsed_ms;
        }
        Ok(CudaProfileReport {
            total_ms,
            attributed_ms,
            operations,
        })
    }
}

#[cfg(feature = "embedded-cuda")]
fn record_cuda_event(stream: &CudaStream, action: &str) -> Result<CudaEvent> {
    stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| cuda_profile_error(action, error))
}

#[cfg(feature = "embedded-cuda")]
fn cuda_elapsed_ms(started: &CudaEvent, completed: &CudaEvent, action: &str) -> Result<f64> {
    started
        .elapsed_ms(completed)
        .map(f64::from)
        .map_err(|error| cuda_profile_error(action, error))
}

#[cfg(feature = "embedded-cuda")]
fn cuda_profile_error(action: &str, error: impl std::fmt::Display) -> PowerError {
    PowerError::InferenceFailed(format!("failed to {action} for CUDA graph timing: {error}"))
}

#[cfg(feature = "embedded-cuda")]
fn render_cuda_operations(operations: &BTreeMap<String, CudaOperationTiming>) -> String {
    operations
        .iter()
        .map(|(name, timing)| {
            format!(
                "{name}:executions={}:nodes={}:ms={:.3}",
                timing.executions, timing.consumed_nodes, timing.elapsed_ms,
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn operation_key(
    node: &GraphNode,
    values: &std::collections::HashMap<String, GraphValue>,
) -> String {
    if matches!(
        node.op,
        GraphOp::Add | GraphOp::Sub | GraphOp::Mul | GraphOp::Div | GraphOp::Pow | GraphOp::MatMul
    ) {
        return binary_shape_key(node, values);
    }
    if node.op == GraphOp::HardSigmoid {
        return unary_shape_key(node, values);
    }
    if node.op != GraphOp::Conv {
        return operation_name(node.op).to_string();
    }
    let Some(input) = node
        .inputs
        .first()
        .and_then(|name| values.get(name))
        .and_then(|value| value.tensor(&node.name).ok())
    else {
        return "Conv[unresolved]".to_string();
    };
    let Some(kernel) = node
        .inputs
        .get(1)
        .and_then(|name| values.get(name))
        .and_then(|value| value.tensor(&node.name).ok())
    else {
        return "Conv[unresolved]".to_string();
    };
    let Ok((_, input_channels, _, _)) = input.dims4() else {
        return "Conv[unresolved]".to_string();
    };
    let Ok((output_channels, kernel_channels, kernel_height, kernel_width)) = kernel.dims4() else {
        return "Conv[unresolved]".to_string();
    };
    let groups = node.int("group", 1).unwrap_or(1).max(1) as usize;
    let strides = node.ints("strides", &[1, 1]).unwrap_or_else(|_| vec![1, 1]);
    let stride_height = strides.first().copied().unwrap_or(1).max(1);
    let stride_width = strides.get(1).copied().unwrap_or(stride_height).max(1);
    let kind = if kernel_height == 1 && kernel_width == 1 && groups == 1 {
        "pointwise"
    } else if groups == input_channels && output_channels == input_channels && kernel_channels == 1
    {
        "depthwise"
    } else {
        "spatial"
    };
    format!(
        "Conv[{kind};input={};out={output_channels};kernel={kernel_height}x{kernel_width};stride={stride_height}x{stride_width};groups={groups}]",
        render_shape(input.dims()),
    )
}

fn binary_shape_key(
    node: &GraphNode,
    values: &std::collections::HashMap<String, GraphValue>,
) -> String {
    let operation = operation_name(node.op);
    let Some(left) = node.inputs.first().and_then(|name| values.get(name)) else {
        return format!("{operation}[unresolved]");
    };
    let Some(right) = node.inputs.get(1).and_then(|name| values.get(name)) else {
        return format!("{operation}[unresolved]");
    };
    format!(
        "{operation}[left={};right={}]",
        render_shape(left.shape()),
        render_shape(right.shape())
    )
}

fn unary_shape_key(
    node: &GraphNode,
    values: &std::collections::HashMap<String, GraphValue>,
) -> String {
    let operation = operation_name(node.op);
    let Some(input) = node.inputs.first().and_then(|name| values.get(name)) else {
        return format!("{operation}[unresolved]");
    };
    format!("{operation}[input={}]", render_shape(input.shape()))
}

fn render_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn operation_name(operation: GraphOp) -> &'static str {
    match operation {
        GraphOp::Add => "Add",
        GraphOp::AveragePool => "AveragePool",
        GraphOp::BatchNormalization => "BatchNormalization",
        GraphOp::Concat => "Concat",
        GraphOp::Conv => "Conv",
        GraphOp::ConvTranspose => "ConvTranspose",
        GraphOp::Div => "Div",
        GraphOp::Erf => "Erf",
        GraphOp::GlobalAveragePool => "GlobalAveragePool",
        GraphOp::HardSigmoid => "HardSigmoid",
        GraphOp::Identity => "Identity",
        GraphOp::MatMul => "MatMul",
        GraphOp::MaxPool => "MaxPool",
        GraphOp::Mul => "Mul",
        GraphOp::Pow => "Pow",
        GraphOp::ReduceMean => "ReduceMean",
        GraphOp::Relu => "Relu",
        GraphOp::Reshape => "Reshape",
        GraphOp::Resize => "Resize",
        GraphOp::Shape => "Shape",
        GraphOp::Sigmoid => "Sigmoid",
        GraphOp::Slice => "Slice",
        GraphOp::Softmax => "Softmax",
        GraphOp::Sqrt => "Sqrt",
        GraphOp::Squeeze => "Squeeze",
        GraphOp::Sub => "Sub",
        GraphOp::Transpose => "Transpose",
        GraphOp::Unsqueeze => "Unsqueeze",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_aggregates_only_shape_and_operator_metrics() {
        let mut profile = GraphExecutionProfile::new(&[2, 3, 48, 320]);
        profile.record_elapsed("Conv".to_string(), 3, Duration::from_millis(7));
        profile.record_elapsed("Conv".to_string(), 2, Duration::from_millis(5));
        profile.record_elapsed("MatMul".to_string(), 1, Duration::from_millis(2));

        assert_eq!(profile.input_shape, [2, 3, 48, 320]);
        assert_eq!(
            profile.render_operations(),
            "Conv:executions=2:nodes=5:ms=12.000,MatMul:executions=1:nodes=1:ms=2.000"
        );
    }

    #[test]
    fn totals_only_profile_skips_operator_keys() {
        let profile = GraphExecutionProfile::totals_only(&[2, 3, 48, 320]);
        let node = GraphNode {
            name: "must-not-be-rendered".to_string(),
            op: GraphOp::Add,
            inputs: vec!["missing-left".to_string(), "missing-right".to_string()],
            outputs: vec!["output".to_string()],
            attributes: Default::default(),
        };

        let timing = profile
            .start_node(&node, &std::collections::HashMap::new())
            .unwrap();
        assert!(timing.operation.is_none());
        assert!(timing.started.is_none());
        assert!(profile.render_operations().is_empty());
    }

    #[test]
    fn profiler_is_absent_when_no_trace_is_requested() {
        assert!(GraphExecutionProfile::from_requests(
            &[2, 3, 48, 320],
            &Device::Cpu,
            false,
            false,
            false,
        )
        .unwrap()
        .is_none());
    }

    #[cfg(not(feature = "embedded-cuda"))]
    #[test]
    fn cuda_trace_fails_closed_without_cuda_support() {
        let error = GraphExecutionProfile::from_requests(
            &[2, 3, 48, 320],
            &Device::Cpu,
            false,
            false,
            true,
        )
        .err()
        .expect("a non-CUDA build must reject CUDA timing");

        assert!(error.to_string().contains("embedded-cuda"));
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    fn cuda_trace_fails_closed_on_a_cpu_execution_device() {
        let error = GraphExecutionProfile::from_requests(
            &[2, 3, 48, 320],
            &Device::Cpu,
            false,
            false,
            true,
        )
        .err()
        .expect("a CPU execution must reject CUDA timing");

        assert!(error.to_string().contains("CUDA execution device"));
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn cuda_trace_measures_the_owning_stream_without_content_keys() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        let left =
            candle_core::Tensor::ones((1, 1_048_576), candle_core::DType::F32, &device).unwrap();
        let right =
            candle_core::Tensor::ones((1, 1_048_576), candle_core::DType::F32, &device).unwrap();
        let mut values = std::collections::HashMap::new();
        values.insert("left".to_string(), GraphValue::Tensor(left.clone()));
        values.insert("right".to_string(), GraphValue::Tensor(right.clone()));
        let node = GraphNode {
            name: "content-derived-name-must-not-appear".to_string(),
            op: GraphOp::Add,
            inputs: vec!["left".to_string(), "right".to_string()],
            outputs: vec!["output".to_string()],
            attributes: Default::default(),
        };
        let mut profile =
            GraphExecutionProfile::from_requests(&[1, 1_048_576], &device, false, false, true)
                .unwrap()
                .unwrap();

        let timing = profile.start_node(&node, &values).unwrap();
        let _output = (&left + &right).unwrap();
        profile.record(timing, 1).unwrap();
        let report = profile.cuda.take().unwrap().finish().unwrap();

        assert!(report.total_ms > 0.0);
        assert!(report.attributed_ms > 0.0);
        assert!(report.attributed_ms <= report.total_ms + 0.01);
        assert_eq!(report.operations.len(), 1);
        assert!(report
            .operations
            .contains_key("Add[left=1x1048576;right=1x1048576]"));
        assert!(!render_cuda_operations(&report.operations)
            .contains("content-derived-name-must-not-appear"));
    }

    #[test]
    fn binary_profile_key_contains_only_operator_and_tensor_shapes() {
        let device = candle_core::Device::Cpu;
        let mut values = std::collections::HashMap::new();
        values.insert(
            "left".to_string(),
            GraphValue::Tensor(
                candle_core::Tensor::zeros((2, 3, 5), candle_core::DType::F32, &device).unwrap(),
            ),
        );
        values.insert(
            "right".to_string(),
            GraphValue::Tensor(
                candle_core::Tensor::zeros((3, 5), candle_core::DType::F32, &device).unwrap(),
            ),
        );
        let node = GraphNode {
            name: "content-derived-name-must-not-appear".to_string(),
            op: GraphOp::MatMul,
            inputs: vec!["left".to_string(), "right".to_string()],
            outputs: vec!["output".to_string()],
            attributes: Default::default(),
        };

        assert_eq!(
            operation_key(&node, &values),
            "MatMul[left=2x3x5;right=3x5]"
        );
    }

    #[test]
    fn unary_profile_key_contains_only_operator_and_tensor_shape() {
        let device = candle_core::Device::Cpu;
        let mut values = std::collections::HashMap::new();
        values.insert(
            "input".to_string(),
            GraphValue::Tensor(
                candle_core::Tensor::zeros((1, 8, 1, 1), candle_core::DType::F32, &device).unwrap(),
            ),
        );
        let node = GraphNode {
            name: "content-derived-name-must-not-appear".to_string(),
            op: GraphOp::HardSigmoid,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attributes: Default::default(),
        };

        assert_eq!(operation_key(&node, &values), "HardSigmoid[input=1x8x1x1]");
    }

    #[test]
    fn convolution_profile_key_contains_only_geometry_and_operator_attributes() {
        let device = candle_core::Device::Cpu;
        let mut values = std::collections::HashMap::new();
        values.insert(
            "input".to_string(),
            GraphValue::Tensor(
                candle_core::Tensor::zeros((7, 16, 23, 41), candle_core::DType::F32, &device)
                    .unwrap(),
            ),
        );
        values.insert(
            "kernel".to_string(),
            GraphValue::Tensor(
                candle_core::Tensor::zeros((16, 1, 5, 5), candle_core::DType::F32, &device)
                    .unwrap(),
            ),
        );
        let node = GraphNode {
            name: "content-derived-name-must-not-appear".to_string(),
            op: GraphOp::Conv,
            inputs: vec!["input".to_string(), "kernel".to_string()],
            outputs: vec!["output".to_string()],
            attributes: std::collections::BTreeMap::from([
                ("group".to_string(), serde_json::json!(16)),
                ("strides".to_string(), serde_json::json!([2, 1])),
            ]),
        };

        assert_eq!(
            operation_key(&node, &values),
            "Conv[depthwise;input=7x16x23x41;out=16;kernel=5x5;stride=2x1;groups=16]"
        );
    }
}
