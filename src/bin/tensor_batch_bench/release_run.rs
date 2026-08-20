use a3s_power::error::Result;
use a3s_power::inference::{
    EmbeddedRuntime, ExecutionDigest, InferenceLimits, ReleaseCapture, ReleaseCaptureSecurity,
    TensorOutput,
};

use super::release_contract::{
    collect_contracts, parse_reservations, validate_reservations, DevicePoolObservation,
    ReleaseContractWorkload,
};
use super::{
    benchmark, build_reviewed_graph, read_bounded_regular, Arguments, CommonOptions,
    MAX_INPUT_DOCUMENT_BYTES,
};

pub(super) fn run(arguments: &mut Arguments, common: &CommonOptions) -> Result<ReleaseCapture> {
    let reservations = parse_reservations(arguments)?;
    let tee_policy_sha256 = arguments.required("--tee-policy-sha256")?;
    let profile_implementation_sha256 = arguments.required("--profile-implementation-sha256")?;
    let profile_shape_class_sha256 = arguments.required("--profile-shape-class-sha256")?;
    let fallback_implementation_sha256 = arguments.required("--fallback-implementation-sha256")?;
    let fallback_request_class_sha256 = arguments.required("--fallback-request-class-sha256")?;
    let reference_path = arguments.required_path("--reference-output")?;
    let reference_source = read_bounded_regular(
        &reference_path,
        MAX_INPUT_DOCUMENT_BYTES,
        "release reference output",
    )?;
    let reference = serde_json::from_slice::<TensorOutput>(&reference_source)?;
    reference.clone().into_input(&InferenceLimits::default())?;
    let reference_output = ExecutionDigest::f32_tensor(&reference.shape, &reference.values);

    let probe = EmbeddedRuntime::new(common.device, InferenceLimits::default())?;
    validate_reservations(&reservations, probe.device().identity())?;
    let mut device_memory = DevicePoolObservation::begin(&probe)?;
    drop(probe);

    let workload = build_reviewed_graph(arguments, common)?;
    let tensor_batch = benchmark(&workload.graph, &workload.inputs, common)?;
    let shape_binding = workload
        .graph
        .shape_profile_binding(reservations, tee_policy_sha256)?;
    let contract_workload = ReleaseContractWorkload {
        graph: &workload.graph,
        runtime: &workload.runtime,
        inputs: &workload.inputs,
        profile_implementation_sha256,
        profile_shape_class_sha256,
        fallback_implementation_sha256,
        fallback_request_class_sha256,
        reference_output,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;
    let contracts = runtime.block_on(collect_contracts(
        &contract_workload,
        &tensor_batch,
        &shape_binding,
        &mut device_memory,
    ))?;
    ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        shape_binding,
        tensor_batch,
        contracts,
    )
}
