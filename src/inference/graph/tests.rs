use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, TensorInput, WeightStore,
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
    let identity = GraphIdentity::new("test-model", "encoder", "onnx", SOURCE_SHA256, 17);
    let plan = GraphPlan::parse(&plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let input = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits).unwrap();

    let output = graph.run(input, &permit, &cancellation).unwrap();

    assert_eq!(output.shape, [1, 2]);
    assert_eq!(output.values, [4.0, 6.0]);
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
