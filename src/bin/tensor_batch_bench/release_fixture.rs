use a3s_power::error::{PowerError, Result};
use a3s_power::inference::{
    AcceleratorFallbackMode, AcceleratorFusedBatchSpec, AcceleratorResidencyDeclaration,
    AcceleratorSecurityRequirement, DevicePreference, EmbeddedRuntime, ExecutionDigest,
    InferenceLimits, ReleaseCapture, ReleaseCaptureSecurity, ResidencyCandidate, ResidencyPolicy,
    RuntimeDeviceKind, RuntimeMemoryReservations, WeightHierarchy, WeightKey,
};
use tokio_util::sync::CancellationToken;

use super::release_contract::{
    collect_contracts, domain_sha256, parse_reservations, validate_reservations,
    DevicePoolObservation, ReleaseContractWorkload,
};
use super::{benchmark, build_fixture, Arguments, CommonOptions};

const FIXTURE_BIAS: f32 = 0.25;

pub(super) fn run(arguments: &mut Arguments, common: &CommonOptions) -> Result<ReleaseCapture> {
    let (reservations, tee_policy_sha256, mut device_memory) = prepare_capture(arguments, common)?;
    let fixture = build_fixture(arguments, common)?;
    collect_fixture_capture(
        &fixture,
        common,
        reservations,
        tee_policy_sha256,
        &mut device_memory,
    )
}

pub(super) fn run_confidential_source(
    arguments: &mut Arguments,
    common: &CommonOptions,
) -> Result<(ReleaseCapture, AcceleratorResidencyDeclaration)> {
    if !matches!(common.device, DevicePreference::Cuda { .. }) {
        return Err(PowerError::BackendNotAvailable(
            "confidential-GPU release source capture requires an explicitly selected CUDA device"
                .to_string(),
        ));
    }
    let (reservations, tee_policy_sha256, mut device_memory) = prepare_capture(arguments, common)?;
    let fixture = build_fixture(arguments, common)?;
    if !fixture
        .fixture_directory
        .as_ref()
        .is_some_and(|directory| directory.is_persistent())
    {
        return Err(PowerError::InvalidRequest(
            "confidential-GPU release source capture requires --fixture-weights so attestation can bind the same persistent collection"
                .to_string(),
        ));
    }
    let bias_bytes = fixture
        .weights
        .descriptor("bias")
        .ok_or_else(|| {
            PowerError::InvalidFormat(
                "release fixture weight collection is missing the bias tensor".to_string(),
            )
        })?
        .bytes;
    let hierarchy = WeightHierarchy::new(
        fixture.weights.clone(),
        fixture.runtime.clone(),
        ResidencyPolicy {
            device_cache_bytes: bias_bytes,
            ..ResidencyPolicy::default()
        },
    )?;
    let cancellation = CancellationToken::new();
    let permit = fixture.runtime.begin(&cancellation)?;
    let plan = hierarchy.plan_residency(&[ResidencyCandidate::new(
        "release-fixture",
        1,
        vec![WeightKey::new(0, "bias")],
    )])?;
    hierarchy.apply_residency_plan(&plan, &permit, &cancellation)?;
    let declaration = hierarchy.declare_accelerator_residency(
        &AcceleratorFusedBatchSpec::new(
            domain_sha256(b"a3s-power-release-fixture-fused-kernel-v1\0"),
            domain_sha256(b"a3s-power-release-fixture-exact-fallback-v1\0"),
            tee_policy_sha256.clone(),
            vec!["release-fixture".to_string()],
        )
        .with_fallback_mode(AcceleratorFallbackMode::AllowExact)
        .with_security(AcceleratorSecurityRequirement::ConfidentialGpu),
    )?;
    if declaration.runtime_device.kind != RuntimeDeviceKind::Cuda {
        return Err(PowerError::InferenceFailed(
            "confidential-GPU release declaration did not resolve to CUDA".to_string(),
        ));
    }
    // Plan pins live in the hierarchy. Release the single-request admission
    // permit before the benchmark acquires that same runtime gate.
    drop(permit);
    drop(cancellation);
    let capture = collect_fixture_capture(
        &fixture,
        common,
        reservations,
        tee_policy_sha256,
        &mut device_memory,
    )?;
    if capture.shape_binding.weights_sha256 != declaration.weights_sha256 {
        return Err(PowerError::InferenceFailed(
            "release capture and accelerator declaration bind different weights".to_string(),
        ));
    }
    Ok((capture, declaration))
}

fn prepare_capture(
    arguments: &mut Arguments,
    common: &CommonOptions,
) -> Result<(
    RuntimeMemoryReservations,
    String,
    Option<DevicePoolObservation>,
)> {
    let reservations = parse_reservations(arguments)?;
    let tee_policy_sha256 = arguments.required("--tee-policy-sha256")?;
    let probe = EmbeddedRuntime::new(common.device, InferenceLimits::default())?;
    validate_reservations(&reservations, probe.device().identity())?;
    let device_memory = DevicePoolObservation::begin(&probe)?;
    drop(probe);
    Ok((reservations, tee_policy_sha256, device_memory))
}

fn collect_fixture_capture(
    fixture: &super::BenchmarkWorkload,
    common: &CommonOptions,
    reservations: RuntimeMemoryReservations,
    tee_policy_sha256: String,
    device_memory: &mut Option<DevicePoolObservation>,
) -> Result<ReleaseCapture> {
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
        device_memory,
    ))?;
    ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        shape_binding,
        tensor_batch,
        contracts,
    )
}
