use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, TensorInput, TensorOutput, WeightStore,
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

fn terminal_softmax_plan_json() -> String {
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
        "nodes": [
            {
                "name": "add-bias",
                "op": "Add",
                "inputs": ["input", "bias"],
                "outputs": ["biased-logits"],
                "attributes": {}
            },
            {
                "name": "alias-logits",
                "op": "Identity",
                "inputs": ["biased-logits"],
                "outputs": ["logits"],
                "attributes": {}
            },
            {
                "name": "terminal-softmax",
                "op": "Softmax",
                "inputs": ["logits"],
                "outputs": ["output"],
                "attributes": {"axis": -1}
            }
        ]
    })
    .to_string()
}

fn terminal_classifier_plan_json() -> String {
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
        "outputs": [{"name": "output", "shape": [1, 3]}],
        "initializers": [
            {"name": "weights", "dtype": "float32", "shape": [2, 3]},
            {"name": "bias", "dtype": "float32", "shape": [3]}
        ],
        "nodes": [
            {
                "name": "classifier",
                "op": "MatMul",
                "inputs": ["input", "weights"],
                "outputs": ["unbiased-logits"],
                "attributes": {}
            },
            {
                "name": "add-classifier-bias",
                "op": "Add",
                "inputs": ["unbiased-logits", "bias"],
                "outputs": ["biased-logits"],
                "attributes": {}
            },
            {
                "name": "alias-classifier-logits",
                "op": "Identity",
                "inputs": ["biased-logits"],
                "outputs": ["logits"],
                "attributes": {}
            },
            {
                "name": "terminal-softmax",
                "op": "Softmax",
                "inputs": ["logits"],
                "outputs": ["output"],
                "attributes": {"axis": -1}
            }
        ]
    })
    .to_string()
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

#[test]
fn terminal_softmax_projection_receives_logits_under_the_graph_boundary() {
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
    let plan = GraphPlan::parse(&terminal_softmax_plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let direct_input = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits).unwrap();
    let direct = graph.run(direct_input, &permit, &cancellation).unwrap();
    let input = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits).unwrap();

    let output = graph
        .run_with_terminal_softmax_projection(input, &permit, &cancellation, |logits| {
            assert_eq!(logits.to_vec2::<f32>().unwrap(), [[4.0, 6.0]]);
            candle_nn::ops::softmax(logits, logits.rank() - 1)
                .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))
        })
        .unwrap();

    assert_eq!(direct.shape, [1, 2]);
    assert!((direct.values[0] - 0.119_202_92).abs() <= 1e-7);
    assert!((direct.values[1] - 0.880_797).abs() <= 1e-7);
    assert_eq!(output.shape, [1, 2]);
    assert!((output.values[0] - 0.119_202_92).abs() <= 1e-7);
    assert!((output.values[1] - 0.880_797).abs() <= 1e-7);
}

#[test]
fn terminal_bias_softmax_projection_receives_unbiased_logits_and_initializer() {
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
    let plan = GraphPlan::parse(&terminal_softmax_plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let input = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits).unwrap();

    let output = graph
        .run_with_terminal_bias_softmax_projection(input, &permit, &cancellation, |logits, bias| {
            assert_eq!(logits.to_vec2::<f32>().unwrap(), [[3.0, 4.0]]);
            assert_eq!(bias.to_vec1::<f32>().unwrap(), [1.0, 2.0]);
            let biased = logits
                .broadcast_add(bias)
                .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))?;
            candle_nn::ops::softmax(&biased, biased.rank() - 1)
                .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))
        })
        .unwrap();

    assert_eq!(output.shape, [1, 2]);
    assert!((output.values[0] - 0.119_202_92).abs() <= 1e-7);
    assert!((output.values[1] - 0.880_797).abs() <= 1e-7);
}

#[test]
fn terminal_classifier_projection_receives_features_weights_and_bias() {
    let directory = tempfile::tempdir().unwrap();
    let weights = [1_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = [0.5_f32, -0.5, 1.0];
    let weight_bytes = weights
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let bias_bytes = bias
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let weight_view = TensorView::new(Dtype::F32, vec![2, 3], &weight_bytes).unwrap();
    let bias_view = TensorView::new(Dtype::F32, vec![3], &bias_bytes).unwrap();
    serialize_to_file(
        vec![("weights", weight_view), ("bias", bias_view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let identity = GraphIdentity::new("test-model", "encoder", "onnx", SOURCE_SHA256, 17);
    let plan =
        GraphPlan::parse(&terminal_classifier_plan_json(), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let direct_input = TensorInput::new(vec![1, 2], vec![2.0, 3.0], &limits).unwrap();
    let direct = graph.run(direct_input, &permit, &cancellation).unwrap();
    let input = TensorInput::new(vec![1, 2], vec![2.0, 3.0], &limits).unwrap();

    let output = graph
        .run_with_terminal_matmul_bias_softmax_projection(
            input,
            &permit,
            &cancellation,
            |features, weights, bias| {
                assert_eq!(features.to_vec2::<f32>().unwrap(), [[2.0, 3.0]]);
                assert_eq!(
                    weights.to_vec2::<f32>().unwrap(),
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                );
                assert_eq!(bias.to_vec1::<f32>().unwrap(), [0.5, -0.5, 1.0]);
                let logits = features.broadcast_matmul(weights).map_err(|error| {
                    crate::error::PowerError::InferenceFailed(error.to_string())
                })?;
                let biased = logits.broadcast_add(bias).map_err(|error| {
                    crate::error::PowerError::InferenceFailed(error.to_string())
                })?;
                candle_nn::ops::softmax(&biased, biased.rank() - 1)
                    .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))
            },
        )
        .unwrap();

    let second_direct = graph
        .run(
            TensorInput::new(vec![1, 2], vec![1.0, 4.0], &limits).unwrap(),
            &permit,
            &cancellation,
        )
        .unwrap();
    let window = graph
        .run_many_with_terminal_matmul_bias_softmax_projection(
            vec![
                TensorInput::new(vec![1, 2], vec![2.0, 3.0], &limits).unwrap(),
                TensorInput::new(vec![1, 2], vec![1.0, 4.0], &limits).unwrap(),
            ],
            &permit,
            &cancellation,
            |features, weights, bias| {
                let logits = features.broadcast_matmul(weights).map_err(|error| {
                    crate::error::PowerError::InferenceFailed(error.to_string())
                })?;
                let biased = logits.broadcast_add(bias).map_err(|error| {
                    crate::error::PowerError::InferenceFailed(error.to_string())
                })?;
                candle_nn::ops::softmax(&biased, biased.rank() - 1)
                    .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))
            },
        )
        .unwrap();
    let ranked_direct = graph
        .run(
            TensorInput::new(vec![1, 2, 2], vec![1.0, 4.0, 3.0, 2.0], &limits).unwrap(),
            &permit,
            &cancellation,
        )
        .unwrap();
    let row_coalesced = graph
        .run_many_with_row_coalesced_terminal_matmul_bias_softmax_projection(
            vec![
                TensorInput::new(vec![1, 2], vec![2.0, 3.0], &limits).unwrap(),
                TensorInput::new(vec![1, 2, 2], vec![1.0, 4.0, 3.0, 2.0], &limits).unwrap(),
            ],
            &permit,
            &cancellation,
            |features, weights, bias| {
                assert_eq!(features.dims(), [3, 2]);
                let logits = features.broadcast_matmul(weights).map_err(|error| {
                    crate::error::PowerError::InferenceFailed(error.to_string())
                })?;
                let biased = logits.broadcast_add(bias).map_err(|error| {
                    crate::error::PowerError::InferenceFailed(error.to_string())
                })?;
                candle_nn::ops::softmax(&biased, biased.rank() - 1)
                    .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))
            },
        )
        .unwrap();
    let empty = graph.run_many_with_row_coalesced_terminal_matmul_bias_softmax_projection(
        Vec::new(),
        &permit,
        &cancellation,
        |features, _, _| Ok(features.clone()),
    );

    assert_eq!(output.shape, direct.shape);
    assert_eq!(output.values, direct.values);
    assert_eq!(window, [direct, second_direct]);
    assert_eq!(row_coalesced, [output, ranked_direct]);
    assert!(matches!(
        empty,
        Err(crate::error::PowerError::InvalidRequest(_))
    ));
}

#[cfg(feature = "embedded-cuda")]
#[test]
#[ignore = "requires an explicit CUDA device"]
fn cuda_terminal_classifier_window_upload_matches_individual_inputs_bit_for_bit() {
    let directory = tempfile::tempdir().unwrap();
    let weights = [1_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = [0.5_f32, -0.5, 1.0];
    let weight_bytes = weights
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let bias_bytes = bias
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let weight_view = TensorView::new(Dtype::F32, vec![2, 3], &weight_bytes).unwrap();
    let bias_view = TensorView::new(Dtype::F32, vec![3], &bias_bytes).unwrap();
    serialize_to_file(
        vec![("weights", weight_view), ("bias", bias_view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let identity = GraphIdentity::new("test-model", "encoder", "onnx", SOURCE_SHA256, 17);
    let plan =
        GraphPlan::parse(&terminal_classifier_plan_json(), &identity, &store, &limits).unwrap();
    let runtime =
        EmbeddedRuntime::new(DevicePreference::Cuda { ordinal: 0 }, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    let first_input = || TensorInput::new(vec![1, 2], vec![2.0, 3.0], &limits).unwrap();
    let second_input =
        || TensorInput::new(vec![1, 2, 2], vec![1.0, 4.0, 3.0, 2.0], &limits).unwrap();
    let project = |features: &candle_core::Tensor,
                   weights: &candle_core::Tensor,
                   bias: &candle_core::Tensor| {
        let logits = features
            .broadcast_matmul(weights)
            .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))?;
        let biased = logits
            .broadcast_add(bias)
            .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))?;
        candle_nn::ops::softmax(&biased, biased.rank() - 1)
            .map_err(|error| crate::error::PowerError::InferenceFailed(error.to_string()))
    };

    let first = graph
        .run_with_terminal_matmul_bias_softmax_projection(
            first_input(),
            &permit,
            &cancellation,
            project,
        )
        .unwrap();
    let second = graph
        .run_with_terminal_matmul_bias_softmax_projection(
            second_input(),
            &permit,
            &cancellation,
            project,
        )
        .unwrap();
    let window = graph
        .run_many_with_terminal_matmul_bias_softmax_projection(
            vec![first_input(), second_input()],
            &permit,
            &cancellation,
            project,
        )
        .unwrap();

    assert_eq!(window, [first, second]);
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
