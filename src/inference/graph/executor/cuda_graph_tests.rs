use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::sys::{
    CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
};
use candle_core::cuda_backend::cudarc::driver::{CudaGraph, DriverError};
use candle_core::{DType, Device, Tensor, Var};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::GraphExecutor;
use crate::inference::graph::{GraphIdentity, GraphPlan};
use crate::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, TensorInput, WeightStore,
};

const SOURCE_SHA256: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const EXTERNAL_ROOT_ENV: &str = "A3S_POWER_GRAPH_PROBE_ROOT";
const EXTERNAL_GRAPH_ENV: &str = "A3S_POWER_GRAPH_PROBE_GRAPH";
const EXTERNAL_SHAPE_ENV: &str = "A3S_POWER_GRAPH_PROBE_BENCH_SHAPE";
const EXTERNAL_NODE_COUNT_ENV: &str = "A3S_POWER_CUDA_GRAPH_TEST_NODE_COUNT";
const EXTERNAL_NO_AUTO_FREE_ENV: &str = "A3S_POWER_CUDA_GRAPH_TEST_NO_AUTO_FREE";
const EXTERNAL_STABLE_OUTPUT_ENV: &str = "A3S_POWER_CUDA_GRAPH_TEST_STABLE_OUTPUT";

#[test]
#[ignore = "requires an explicit CUDA device"]
fn safe_stream_capture_replays_static_graph_with_exact_output() {
    let directory = tempfile::tempdir().unwrap();
    write_bias(directory.path());

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let identity = GraphIdentity::new("capture-test", "encoder", "onnx", SOURCE_SHA256, 17);
    let plan = GraphPlan::parse(&plan_json(), &identity, &weights, &limits).unwrap();
    let runtime =
        EmbeddedRuntime::new(DevicePreference::Cuda { ordinal: 0 }, limits.clone()).unwrap();
    let Device::Cuda(cuda) = runtime.device().tensor_device() else {
        panic!("explicit CUDA runtime resolved another device");
    };
    // SAFETY: this test owns one device identity and one stream. Every tensor
    // remains on that stream until the final synchronization, matching the
    // production ModelSessionPool isolation contract.
    unsafe { cuda.disable_event_tracking() };

    let executor = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let (first, upload_guard) = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits)
        .unwrap()
        .into_candle(
            runtime.device().tensor_device(),
            &limits,
            permit.input_upload_pool(),
        )
        .unwrap();
    let expected_first = executor
        .run_tensor(first.clone(), &cancellation)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    let staging = Var::zeros(first.shape(), DType::F32, runtime.device().tensor_device()).unwrap();
    staging.set(&first).unwrap();
    let stream = cuda.cuda_stream();
    stream.synchronize().unwrap();
    stream
        .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let captured_output = executor
        .run_tensor(staging.as_tensor().clone(), &cancellation)
        .unwrap();
    let replay = stream
        .end_capture(CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .expect("the static graph must capture at least one CUDA operation");
    replay.upload().unwrap();
    replay.launch().unwrap();
    assert_eq!(captured_output.to_vec2::<f32>().unwrap(), expected_first);

    let second = Tensor::from_vec(
        vec![-5.0_f32, 7.5],
        (1, 2),
        runtime.device().tensor_device(),
    )
    .unwrap();
    let expected_second = executor
        .run_tensor(second.clone(), &cancellation)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    staging.set(&second).unwrap();
    replay.launch().unwrap();
    assert_eq!(captured_output.to_vec2::<f32>().unwrap(), expected_second);

    stream.synchronize().unwrap();
    drop(upload_guard);
    drop(replay);
    drop(captured_output);
    stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires an explicit reviewed graph bundle and CUDA device"]
fn safe_stream_capture_replays_external_graph_with_exact_output() {
    let root = std::env::var_os(EXTERNAL_ROOT_ENV)
        .map(std::path::PathBuf::from)
        .map(std::fs::canonicalize)
        .transpose()
        .unwrap()
        .unwrap_or_else(|| panic!("{EXTERNAL_ROOT_ENV} must be set"));
    let graph_path = std::env::var_os(EXTERNAL_GRAPH_ENV)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join("graph.json"));
    let graph_source = std::fs::read_to_string(graph_path).unwrap();
    let identity = external_identity(&graph_source);
    let shape_source = std::env::var(EXTERNAL_SHAPE_ENV)
        .unwrap_or_else(|_| panic!("{EXTERNAL_SHAPE_ENV} must be set"));
    let shape: Vec<usize> = serde_json::from_str(&shape_source).unwrap();
    assert!(!shape.is_empty() && shape.iter().all(|dimension| *dimension > 0));
    let elements = shape.iter().product::<usize>();

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime =
        EmbeddedRuntime::new(DevicePreference::Cuda { ordinal: 0 }, limits.clone()).unwrap();
    let Device::Cuda(cuda) = runtime.device().tensor_device() else {
        panic!("explicit CUDA runtime resolved another device");
    };
    // SAFETY: this test owns one device identity and stream, and does not
    // expose any tensor to another stream before final synchronization.
    unsafe { cuda.disable_event_tracking() };
    let executor = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let node_count = std::env::var(EXTERNAL_NODE_COUNT_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(executor.plan.nodes.len());
    assert!((1..=executor.plan.nodes.len()).contains(&node_count));
    let terminal_node = &executor.plan.nodes[node_count - 1];
    let output_name = terminal_node
        .outputs
        .first()
        .expect("the captured terminal node must publish an output")
        .clone();
    let cancellation = CancellationToken::new();
    let input = Tensor::from_vec(
        vec![0.0_f32; elements],
        shape.as_slice(),
        runtime.device().tensor_device(),
    )
    .unwrap();
    let expected_output = executor
        .run_tensor_prefix(input.clone(), &cancellation, node_count, &output_name)
        .unwrap();
    let output_shape = expected_output.dims().to_vec();
    let expected = expected_output
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    drop(expected_output);

    let staging = Var::zeros(
        shape.as_slice(),
        DType::F32,
        runtime.device().tensor_device(),
    )
    .unwrap();
    staging.set(&input).unwrap();
    let stable_output = std::env::var_os(EXTERNAL_STABLE_OUTPUT_ENV).is_some();
    let output_staging = stable_output
        .then(|| {
            Var::zeros(
                output_shape.as_slice(),
                DType::F32,
                runtime.device().tensor_device(),
            )
        })
        .transpose()
        .unwrap();
    let stream = cuda.cuda_stream();
    stream.synchronize().unwrap();
    stream
        .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let transient_output = executor
        .run_tensor_prefix(
            staging.as_tensor().clone(),
            &cancellation,
            node_count,
            &output_name,
        )
        .unwrap();
    let captured_output = if let Some(output_staging) = output_staging.as_ref() {
        output_staging.set(&transient_output).unwrap();
        drop(transient_output);
        output_staging.as_tensor().clone()
    } else {
        transient_output
    };
    let auto_free = stable_output || std::env::var_os(EXTERNAL_NO_AUTO_FREE_ENV).is_none();
    let instantiate_flags = if auto_free {
        CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
    } else {
        // cudarc's safe end_capture surface currently requires one enum
        // value and cannot express the ordinary zero-flags case. Node
        // priorities were copied from this same stream during capture, so
        // retaining them changes no scheduling authority while avoiding the
        // allocation auto-free behavior under diagnosis. UPLOAD cannot be
        // used here because it is valid only with cuGraphInstantiateWithParams,
        // whereas cudarc calls cuGraphInstantiateWithFlags.
        CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
    };
    let replay = stream
        .end_capture(instantiate_flags)
        .unwrap()
        .expect("the external graph must capture CUDA work");
    if std::env::var_os("A3S_POWER_CUDA_GRAPH_TEST_TRACE_MEMORY_NODES").is_some() {
        trace_graph_memory_nodes(&replay).unwrap();
    }
    replay.upload().unwrap();
    let mut launch_differences = Vec::new();
    let launches = if auto_free || stable_output { 2 } else { 1 };
    for launch_index in 0..launches {
        replay.launch().unwrap();
        let actual = captured_output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        launch_differences.push((launch_index, output_differences(&actual, &expected)));
    }
    if stable_output {
        drop(expected);
        let second = Tensor::from_vec(
            (0..elements)
                .map(|index| {
                    let value = (index % 257) * 17 % 257;
                    (value as f32 - 128.0) / 131.0
                })
                .collect::<Vec<_>>(),
            shape.as_slice(),
            runtime.device().tensor_device(),
        )
        .unwrap();
        let expected_second = executor
            .run_tensor_prefix(second.clone(), &cancellation, node_count, &output_name)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        staging.set(&second).unwrap();
        replay.launch().unwrap();
        let actual = captured_output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual.len(), expected_second.len());
        launch_differences.push((launches, output_differences(&actual, &expected_second)));
    }

    stream.synchronize().unwrap();
    drop(replay);
    drop(captured_output);
    stream.synchronize().unwrap();
    assert!(
        launch_differences
            .iter()
            .all(|(_, (differing_words, _, _))| *differing_words == 0),
        "capture through node_count={node_count} terminal={:?}/{:?} changed output: {launch_differences:?}",
        terminal_node.name,
        terminal_node.op,
    );
}

fn trace_graph_memory_nodes(graph: &CudaGraph) -> Result<(), DriverError> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::mem::MaybeUninit;

    use candle_core::cuda_backend::cudarc::driver::sys;

    let raw = graph.cu_graph();
    let mut node_count = 0_usize;
    unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut node_count) }.result()?;
    let mut nodes = vec![std::ptr::null_mut(); node_count];
    unsafe { sys::cuGraphGetNodes(raw, nodes.as_mut_ptr(), &mut node_count) }.result()?;
    nodes.truncate(node_count);

    let mut type_counts = BTreeMap::<u32, usize>::new();
    let mut allocation_bytes = 0_usize;
    let mut allocation_pointers = Vec::new();
    let mut free_pointers = Vec::new();
    for node in nodes {
        let mut node_type = MaybeUninit::<sys::CUgraphNodeType>::uninit();
        unsafe { sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()) }.result()?;
        let node_type = unsafe { node_type.assume_init() };
        *type_counts.entry(node_type as u32).or_default() += 1;
        match node_type {
            sys::CUgraphNodeType_enum::CU_GRAPH_NODE_TYPE_MEM_ALLOC => {
                let mut parameters = MaybeUninit::<sys::CUDA_MEM_ALLOC_NODE_PARAMS>::uninit();
                unsafe { sys::cuGraphMemAllocNodeGetParams(node, parameters.as_mut_ptr()) }
                    .result()?;
                let parameters = unsafe { parameters.assume_init() };
                allocation_bytes = allocation_bytes.saturating_add(parameters.bytesize);
                allocation_pointers.push(parameters.dptr);
            }
            sys::CUgraphNodeType_enum::CU_GRAPH_NODE_TYPE_MEM_FREE => {
                let mut pointer = MaybeUninit::<sys::CUdeviceptr>::uninit();
                unsafe { sys::cuGraphMemFreeNodeGetParams(node, pointer.as_mut_ptr()) }.result()?;
                free_pointers.push(unsafe { pointer.assume_init() });
            }
            _ => {}
        }
    }
    let unique_allocations = allocation_pointers.iter().copied().collect::<BTreeSet<_>>();
    let unique_frees = free_pointers.iter().copied().collect::<BTreeSet<_>>();
    eprintln!(
        "A3S_POWER_CUDA_GRAPH_MEMORY_NODES total={} types={type_counts:?} allocations={} unique_allocations={} allocation_bytes={} frees={} unique_frees={} alloc_free_pointer_sets_equal={}",
        type_counts.values().sum::<usize>(),
        allocation_pointers.len(),
        unique_allocations.len(),
        allocation_bytes,
        free_pointers.len(),
        unique_frees.len(),
        unique_allocations == unique_frees,
    );
    Ok(())
}

fn output_differences(actual: &[f32], expected: &[f32]) -> (usize, Option<(usize, f32, f32)>, f32) {
    let mut first_difference = None;
    let mut differing_words = 0_usize;
    let mut maximum_absolute = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            differing_words += 1;
            maximum_absolute = maximum_absolute.max((actual - expected).abs());
            first_difference.get_or_insert((index, actual, expected));
        }
    }
    (differing_words, first_difference, maximum_absolute)
}

fn external_identity(source: &str) -> GraphIdentity {
    let document: serde_json::Value = serde_json::from_str(source).unwrap();
    let source_identity = document
        .get("source")
        .and_then(serde_json::Value::as_object)
        .unwrap();
    GraphIdentity::new(
        document
            .get("family")
            .and_then(serde_json::Value::as_str)
            .unwrap(),
        document
            .get("role")
            .and_then(serde_json::Value::as_str)
            .unwrap(),
        source_identity
            .get("format")
            .and_then(serde_json::Value::as_str)
            .unwrap(),
        source_identity
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .unwrap(),
        source_identity
            .get("opset")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap(),
    )
}

fn write_bias(root: &std::path::Path) {
    let values = [1_f32, 2_f32];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![2], &bytes).unwrap();
    serialize_to_file(vec![("bias", view)], None, &root.join("model.safetensors")).unwrap();
}

fn plan_json() -> String {
    serde_json::json!({
        "schemaVersion": 1,
        "family": "capture-test",
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
