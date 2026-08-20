use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::inference::{
    DevicePreference, EmbeddedRuntime, ExecutionDigest, InferenceLimits, RuntimeDeviceKind,
    TensorInput, WeightStore,
};

const SOURCE_SHA256: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

fn graph(
    runtime: EmbeddedRuntime,
    width: usize,
    input_shape: serde_json::Value,
    output_shape: serde_json::Value,
) -> (tempfile::TempDir, GraphExecutor) {
    let directory = tempfile::tempdir().unwrap();
    let bias = (0..width)
        .map(|index| (index + 1) as f32)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![width], &bias).unwrap();
    serialize_to_file(
        vec![("bias", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let identity = GraphIdentity::new(
        "generic-test-family",
        "generic-transform",
        "reviewed-json",
        SOURCE_SHA256,
        1,
    );
    let plan_source = serde_json::json!({
        "schemaVersion": 1,
        "family": "generic-test-family",
        "role": "generic-transform",
        "source": {
            "format": "reviewed-json",
            "sha256": SOURCE_SHA256,
            "opset": 1
        },
        "inputs": [{"name": "input", "shape": input_shape}],
        "outputs": [{"name": "output", "shape": output_shape}],
        "initializers": [{"name": "bias", "dtype": "float32", "shape": [width]}],
        "nodes": [{
            "name": "add-bias",
            "op": "Add",
            "inputs": ["input", "bias"],
            "outputs": ["output"],
            "attributes": {}
        }]
    })
    .to_string();
    let store = Arc::new(WeightStore::open(directory.path(), runtime.limits()).unwrap());
    let plan = GraphPlan::parse(&plan_source, &identity, &store, runtime.limits()).unwrap();
    let executor = GraphExecutor::new(plan, store, runtime).unwrap();
    (directory, executor)
}

fn input(values: [f32; 2], limits: &InferenceLimits) -> TensorInput {
    TensorInput::new(vec![1, 2], values.to_vec(), limits).unwrap()
}

#[test]
fn adjacent_graphs_keep_one_owned_boundary_and_canonical_digests() {
    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_first_directory, first) = graph(
        runtime.clone(),
        2,
        serde_json::json!(["batch", 2]),
        serde_json::json!(["batch", 2]),
    );
    let (_second_directory, second) = graph(
        runtime.clone(),
        2,
        serde_json::json!(["batch", 2]),
        serde_json::json!(["batch", 2]),
    );
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    let handle = first
        .run_to_resident(input([3.0, 4.0], &limits), &permit, &cancellation)
        .unwrap();
    assert_eq!(runtime.resident_tensor_snapshot().active_handles, 1);
    let handle = second.run_resident(handle, &permit, &cancellation).unwrap();
    let materialized = handle.materialize(&cancellation).unwrap();

    assert_eq!(materialized.output.shape, [1, 2]);
    assert_eq!(materialized.output.values, [5.0, 8.0]);
    assert_eq!(
        materialized.input_digest,
        ExecutionDigest::f32_tensor(&[1, 2], &[3.0, 4.0])
    );
    assert_eq!(
        materialized.output_digest,
        ExecutionDigest::f32_tensor(&[1, 2], &[5.0, 8.0])
    );
    assert_eq!(materialized.boundary.input_materializations, 1);
    assert_eq!(materialized.boundary.output_materializations, 1);
    assert_eq!(materialized.boundary.input_host_bytes, 8);
    assert_eq!(materialized.boundary.output_host_bytes, 8);
    assert_eq!(materialized.boundary.host_to_device_copy_operations, 0);
    assert_eq!(materialized.boundary.device_to_host_copy_operations, 0);
    let snapshot = runtime.resident_tensor_snapshot();
    assert_eq!(snapshot.active_handles, 0);
    assert_eq!(snapshot.resident_bytes, 0);
    assert_eq!(snapshot.peak_resident_bytes, 8);
}

#[test]
fn resident_handle_requires_the_same_request_permit() {
    let limits = InferenceLimits {
        max_concurrent_requests: 2,
        ..InferenceLimits::default()
    };
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_directory, graph) = graph(
        runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let cancellation = CancellationToken::new();
    let first_permit = runtime.begin(&cancellation).unwrap();
    let other_permit = runtime.begin(&cancellation).unwrap();
    let handle = graph
        .run_to_resident(input([1.0, 2.0], &limits), &first_permit, &cancellation)
        .unwrap();

    assert!(graph
        .run_resident(handle, &other_permit, &cancellation)
        .is_err());
    assert_eq!(runtime.resident_tensor_snapshot().active_handles, 0);
}

#[test]
fn resident_handle_rejects_a_different_runtime_and_shape() {
    let limits = InferenceLimits::default();
    let source_runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let target_runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_source_directory, source) = graph(
        source_runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let (_target_directory, target) = graph(
        target_runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let cancellation = CancellationToken::new();
    let source_permit = source_runtime.begin(&cancellation).unwrap();
    let target_permit = target_runtime.begin(&cancellation).unwrap();
    let handle = source
        .run_to_resident(input([1.0, 2.0], &limits), &source_permit, &cancellation)
        .unwrap();
    assert!(target
        .run_resident(handle, &target_permit, &cancellation)
        .is_err());

    let shared_runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_source_directory, source) = graph(
        shared_runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let (_target_directory, target) = graph(
        shared_runtime.clone(),
        3,
        serde_json::json!([1, 3]),
        serde_json::json!([1, 3]),
    );
    let permit = shared_runtime.begin(&cancellation).unwrap();
    let handle = source
        .run_to_resident(input([1.0, 2.0], &limits), &permit, &cancellation)
        .unwrap();
    assert!(target.run_resident(handle, &permit, &cancellation).is_err());
}

#[test]
fn explicit_owned_fallback_preserves_the_intermediate_digest() {
    let limits = InferenceLimits::default();
    let source_runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let target_runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_source_directory, source) = graph(
        source_runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let (_target_directory, target) = graph(
        target_runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let cancellation = CancellationToken::new();
    let source_permit = source_runtime.begin(&cancellation).unwrap();
    let first = source
        .run_to_resident(input([3.0, 4.0], &limits), &source_permit, &cancellation)
        .unwrap()
        .materialize(&cancellation)
        .unwrap();
    let intermediate_digest = first.output_digest.clone();
    let target_input = first.output.into_input(&limits).unwrap();

    let target_permit = target_runtime.begin(&cancellation).unwrap();
    let second = target
        .run_to_resident(target_input, &target_permit, &cancellation)
        .unwrap()
        .materialize(&cancellation)
        .unwrap();

    assert_eq!(second.input_digest, intermediate_digest);
    assert_eq!(second.output.values, [5.0, 8.0]);
}

#[test]
fn resident_budget_is_aggregate_and_releases_on_cancellation() {
    let limits = InferenceLimits {
        max_tensor_elements: 4,
        ..InferenceLimits::default()
    };
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_directory, graph) = graph(
        runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let first = graph
        .run_to_resident(input([1.0, 2.0], &limits), &permit, &cancellation)
        .unwrap();
    let second = graph
        .run_to_resident(input([3.0, 4.0], &limits), &permit, &cancellation)
        .unwrap();
    assert!(graph
        .run_to_resident(input([5.0, 6.0], &limits), &permit, &cancellation)
        .is_err());
    let snapshot = runtime.resident_tensor_snapshot();
    assert_eq!(snapshot.maximum_bytes, 16);
    assert_eq!(snapshot.active_handles, 2);
    assert_eq!(snapshot.resident_bytes, 16);
    assert_eq!(snapshot.rejected_reservations, 1);
    drop(first);
    drop(second);

    let handle = graph
        .run_to_resident(input([7.0, 8.0], &limits), &permit, &cancellation)
        .unwrap();
    cancellation.cancel();
    assert!(handle.materialize(&cancellation).is_err());
    let snapshot = runtime.resident_tensor_snapshot();
    assert_eq!(snapshot.active_handles, 0);
    assert_eq!(snapshot.resident_bytes, 0);
}

#[test]
fn descriptor_is_typed_and_debug_output_omits_values() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ResidentGraphTensor>();
    assert_send_sync::<ResidentGraphMaterialization>();

    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_directory, graph) = graph(
        runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let handle = graph
        .run_to_resident(input([123.5, 987.25], &limits), &permit, &cancellation)
        .unwrap();
    let descriptor = handle.descriptor();

    assert_eq!(descriptor.dtype, GraphTensorDType::F32);
    assert_eq!(descriptor.shape, [1, 2]);
    assert_eq!(descriptor.device.kind, RuntimeDeviceKind::Cpu);
    let debug = format!("{handle:?}");
    assert!(!debug.contains("123.5"));
    assert!(!debug.contains("987.25"));
    assert!(!debug.contains(SOURCE_SHA256));
}

#[test]
fn graph_output_must_match_its_reviewed_shape() {
    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_directory, graph) = graph(
        runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 3]),
    );
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    assert!(graph
        .run_to_resident(input([1.0, 2.0], &limits), &permit, &cancellation)
        .is_err());
}

#[test]
fn accelerator_identity_records_one_copy_at_each_owned_edge() {
    let limits = InferenceLimits::default();
    let runtime =
        EmbeddedRuntime::new_test_accelerator(RuntimeDeviceKind::Cuda, 0, limits.clone()).unwrap();
    let (_directory, graph) = graph(
        runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let handle = graph
        .run_to_resident(input([1.0, 2.0], &limits), &permit, &cancellation)
        .unwrap();

    assert_eq!(handle.descriptor().device.kind, RuntimeDeviceKind::Cuda);
    let materialized = handle.materialize(&cancellation).unwrap();
    assert_eq!(materialized.boundary.host_to_device_copy_operations, 1);
    assert_eq!(materialized.boundary.device_to_host_copy_operations, 1);
}

#[test]
fn ordinary_owned_execution_also_enforces_the_reviewed_shape() {
    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let (_directory, graph) = graph(
        runtime.clone(),
        2,
        serde_json::json!([1, 2]),
        serde_json::json!([1, 2]),
    );
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let transposed_shape = TensorInput::new(vec![2, 1], vec![1.0, 2.0], &limits).unwrap();

    assert!(graph.run(transposed_shape, &permit, &cancellation).is_err());
}

#[test]
fn resident_graph_output_must_remain_f32() {
    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let f16_values = [0x00_u8, 0x3c, 0x00, 0x40];
    let view = TensorView::new(Dtype::F16, vec![1, 2], &f16_values).unwrap();
    serialize_to_file(
        vec![("constant", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let identity = GraphIdentity::new(
        "generic-test-family",
        "generic-transform",
        "reviewed-json",
        SOURCE_SHA256,
        1,
    );
    let source = serde_json::json!({
        "schemaVersion": 1,
        "family": "generic-test-family",
        "role": "generic-transform",
        "source": {
            "format": "reviewed-json",
            "sha256": SOURCE_SHA256,
            "opset": 1
        },
        "inputs": [{"name": "input", "shape": [1, 2]}],
        "outputs": [{"name": "output", "shape": [1, 2]}],
        "initializers": [{"name": "constant", "dtype": "float16", "shape": [1, 2]}],
        "nodes": [{
            "name": "constant-output",
            "op": "Identity",
            "inputs": ["constant"],
            "outputs": ["output"],
            "attributes": {}
        }]
    })
    .to_string();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let plan = GraphPlan::parse(&source, &identity, &store, &limits).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    assert!(graph
        .run_to_resident(input([1.0, 2.0], &limits), &permit, &cancellation)
        .is_err());
    assert_eq!(runtime.resident_tensor_snapshot().active_handles, 0);
}
