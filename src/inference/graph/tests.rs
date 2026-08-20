use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, RuntimeMemoryReservations, TensorInput,
    TensorOutput, WeightStore,
};

const SOURCE_SHA256: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn plan_json() -> String {
    serde_json::json!({
        "schemaVersion": 1,
        "family": "test-model",
        "role": "encoder",
        "source": {
            "format": "onnx",
            "sha256": SOURCE_SHA256,
            "opset": 17
        },
        "inputs": [{"name": "input", "shape": [1, 2]}],
        "outputs": [{"name": "output", "shape": [1, 2]}],
        "initializers": [{"name": "bias", "dtype": "float32", "shape": [2]}],
        "nodes": [{
            "name": "add-bias",
            "op": "Add",
            "inputs": ["input", "bias"],
            "outputs": ["output"],
            "attributes": {}
        }]
    })
    .to_string()
}

#[test]
fn model_owned_reviewed_plan_executes_on_shared_runtime() {
    let directory = tempfile::tempdir().unwrap();
    let values = [1_f32, 2_f32];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![2], &bytes).unwrap();
    serialize_to_file(
        vec![("bias", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let weights_sha256 = store.sha256().to_string();
    let identity = GraphIdentity::new("test-model", "encoder", "onnx", SOURCE_SHA256, 17);
    let graph_sha256 = identity.binding_sha256().unwrap();
    let plan = GraphPlan::parse(&plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let profile_binding = graph
        .shape_profile_binding(RuntimeMemoryReservations::default(), "b".repeat(64))
        .unwrap();
    assert_eq!(profile_binding.weights_sha256, weights_sha256);
    assert_eq!(profile_binding.graph_sha256, graph_sha256);
    assert_ne!(profile_binding.graph_sha256, SOURCE_SHA256);
    assert_eq!(profile_binding.runtime_device, runtime.device().identity());
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let input = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits).unwrap();

    let output = graph.run(input, &permit, &cancellation).unwrap();

    assert_eq!(output.shape, [1, 2]);
    assert_eq!(output.values, [4.0, 6.0]);
}

fn gated_hard_sigmoid_plan_json() -> String {
    serde_json::json!({
        "schemaVersion": 1,
        "family": "test-model",
        "role": "encoder",
        "source": {
            "format": "onnx",
            "sha256": SOURCE_SHA256,
            "opset": 17
        },
        "inputs": [{"name": "input", "shape": [1, 2, 1, 1]}],
        "outputs": [{"name": "output", "shape": [1, 2, 2, 2]}],
        "initializers": [{"name": "features", "dtype": "float32", "shape": [1, 2, 2, 2]}],
        "nodes": [
            {
                "name": "bounded-gate",
                "op": "HardSigmoid",
                "inputs": ["input"],
                "outputs": ["gate"],
                "attributes": {"alpha": 0.2, "beta": 0.5}
            },
            {
                "name": "apply-gate",
                "op": "Mul",
                "inputs": ["features", "gate"],
                "outputs": ["output"],
                "attributes": {}
            }
        ]
    })
    .to_string()
}

fn gelu_erf_plan_json() -> String {
    serde_json::json!({
        "schemaVersion": 1,
        "family": "test-model",
        "role": "encoder",
        "source": {
            "format": "onnx",
            "sha256": SOURCE_SHA256,
            "opset": 17
        },
        "inputs": [{"name": "input", "shape": [1, 4]}],
        "outputs": [{"name": "output", "shape": [1, 4]}],
        "initializers": [
            {"name": "divisor", "dtype": "float32", "shape": [1]},
            {"name": "offset", "dtype": "float32", "shape": [1]},
            {"name": "scale", "dtype": "float32", "shape": [1]}
        ],
        "nodes": [
            {
                "name": "divide",
                "op": "Div",
                "inputs": ["input", "divisor"],
                "outputs": ["divided"],
                "attributes": {}
            },
            {
                "name": "erf",
                "op": "Erf",
                "inputs": ["divided"],
                "outputs": ["activated"],
                "attributes": {}
            },
            {
                "name": "add",
                "op": "Add",
                "inputs": ["activated", "offset"],
                "outputs": ["shifted"],
                "attributes": {}
            },
            {
                "name": "multiply-input",
                "op": "Mul",
                "inputs": ["input", "shifted"],
                "outputs": ["product"],
                "attributes": {}
            },
            {
                "name": "multiply-scale",
                "op": "Mul",
                "inputs": ["product", "scale"],
                "outputs": ["output"],
                "attributes": {}
            }
        ]
    })
    .to_string()
}

#[test]
fn model_owned_output_projection_runs_before_host_materialization() {
    let directory = tempfile::tempdir().unwrap();
    let values = [1_f32, 2_f32];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![2], &bytes).unwrap();
    serialize_to_file(
        vec![("bias", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let identity = GraphIdentity::new("test-model", "encoder", "onnx", SOURCE_SHA256, 17);
    let plan = GraphPlan::parse(&plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let input = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits).unwrap();

    let output = graph
        .run_with_output_projection(input, &permit, &cancellation, |tensor| {
            tensor
                .sum_keepdim(1)
                .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))
        })
        .unwrap();

    assert_eq!(output.shape, [1, 1]);
    assert_eq!(output.values, [10.0]);
}

fn run_gated_hard_sigmoid_plan(device: DevicePreference) -> TensorOutput {
    let directory = tempfile::tempdir().unwrap();
    let features = [1_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let bytes = features
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![1, 2, 2, 2], &bytes).unwrap();
    serialize_to_file(
        vec![("features", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let identity = GraphIdentity::new("test-model", "encoder", "onnx", SOURCE_SHA256, 17);
    let plan =
        GraphPlan::parse(&gated_hard_sigmoid_plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(device, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let input = TensorInput::new(vec![1, 2, 1, 1], vec![-9.0, 9.0], &limits).unwrap();

    graph.run(input, &permit, &cancellation).unwrap()
}

#[test]
fn gated_hard_sigmoid_plan_retains_the_cpu_fallback() {
    let output = run_gated_hard_sigmoid_plan(DevicePreference::Cpu);

    assert_eq!(output.shape, [1, 2, 2, 2]);
    assert_eq!(output.values, [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0]);
}

fn run_gelu_erf_plan(device: DevicePreference) -> TensorOutput {
    let directory = tempfile::tempdir().unwrap();
    let divisor = std::f32::consts::SQRT_2.to_le_bytes();
    let offset = 1.0_f32.to_le_bytes();
    let scale = 0.5_f32.to_le_bytes();
    let divisor_view = TensorView::new(Dtype::F32, vec![1], &divisor).unwrap();
    let offset_view = TensorView::new(Dtype::F32, vec![1], &offset).unwrap();
    let scale_view = TensorView::new(Dtype::F32, vec![1], &scale).unwrap();
    serialize_to_file(
        vec![
            ("divisor", divisor_view),
            ("offset", offset_view),
            ("scale", scale_view),
        ],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let identity = GraphIdentity::new("test-model", "encoder", "onnx", SOURCE_SHA256, 17);
    let plan = GraphPlan::parse(&gelu_erf_plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(device, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let input = TensorInput::new(vec![1, 4], vec![-1.0, -0.0, 1.0, 3.0], &limits).unwrap();

    graph.run(input, &permit, &cancellation).unwrap()
}

fn assert_gelu_erf_output(output: TensorOutput) {
    assert_eq!(output.shape, [1, 4]);
    let expected = [-0.158_655_26_f32, -0.0, 0.841_344_7, 2.995_950_2];
    assert!(output
        .values
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= 0.000_001));
}

#[test]
fn gelu_erf_plan_retains_the_cpu_fallback() {
    assert_gelu_erf_output(run_gelu_erf_plan(DevicePreference::Cpu));
}

#[cfg(feature = "embedded-cuda")]
#[test]
#[ignore = "requires an explicit CUDA device"]
fn gelu_erf_plan_executes_through_the_fused_cuda_step() {
    assert_gelu_erf_output(run_gelu_erf_plan(DevicePreference::Cuda { ordinal: 0 }));
}

#[cfg(feature = "embedded-cuda")]
#[test]
#[ignore = "requires an explicit CUDA device"]
fn gated_hard_sigmoid_plan_executes_through_the_fused_cuda_step() {
    let output = run_gated_hard_sigmoid_plan(DevicePreference::Cuda { ordinal: 0 });

    assert_eq!(output.shape, [1, 2, 2, 2]);
    assert_eq!(output.values, [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn graph_identity_mismatch_fails_before_execution() {
    let directory = tempfile::tempdir().unwrap();
    let view = TensorView::new(Dtype::F32, vec![2], &[0; 8]).unwrap();
    serialize_to_file(
        vec![("bias", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let limits = InferenceLimits::default();
    let store = WeightStore::open(directory.path(), &limits).unwrap();
    let wrong = GraphIdentity::new("other-model", "encoder", "onnx", SOURCE_SHA256, 17);

    assert!(GraphPlan::parse(&plan_json(), &wrong, &store, &limits).is_err());
}
