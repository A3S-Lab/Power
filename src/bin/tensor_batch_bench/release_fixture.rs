use a3s_power::error::Result;
use a3s_power::inference::{
    EmbeddedRuntime, ExecutionDigest, InferenceLimits, ReleaseCapture, ReleaseCaptureSecurity,
};

use super::release_contract::{
    collect_contracts, domain_sha256, parse_reservations, validate_reservations,
    DevicePoolObservation, ReleaseContractWorkload,
};
use super::{benchmark, build_fixture, Arguments, CommonOptions};

const FIXTURE_BIAS: f32 = 0.25;

pub(super) fn run(arguments: &mut Arguments, common: &CommonOptions) -> Result<ReleaseCapture> {
    let reservations = parse_reservations(arguments)?;
    let tee_policy_sha256 = arguments.required("--tee-policy-sha256")?;
    let probe = EmbeddedRuntime::new(common.device, InferenceLimits::default())?;
    validate_reservations(&reservations, probe.device().identity())?;
    let mut device_memory = DevicePoolObservation::begin(&probe)?;
    drop(probe);

    let fixture = build_fixture(arguments, common)?;
    let tensor_batch = benchmark(&fixture.graph, &fixture.inputs, common)?;
    let shape_binding = fixture
        .graph
        .shape_profile_binding(reservations, tee_policy_sha256)?;
    let input = &fixture.inputs[0];
    let reference_values = input
        .values
        .iter()
        .map(|value| *value + FIXTURE_BIAS)
        .collect::<Vec<_>>();
    let contract_workload = ReleaseContractWorkload {
        graph: &fixture.graph,
        runtime: &fixture.runtime,
        inputs: &fixture.inputs,
        profile_implementation_sha256: domain_sha256(
            b"a3s-power-release-fixture-profile-implementation-v1\0",
        ),
        profile_shape_class_sha256: domain_sha256(b"a3s-power-release-fixture-profile-class-v1\0"),
        fallback_implementation_sha256: domain_sha256(
            b"a3s-power-release-fixture-fallback-implementation-v1\0",
        ),
        fallback_request_class_sha256: domain_sha256(
            b"a3s-power-release-fixture-dynamic-class-v1\0",
        ),
        reference_output: ExecutionDigest::f32_tensor(&input.shape, &reference_values),
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
