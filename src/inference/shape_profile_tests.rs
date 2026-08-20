use serde_json::json;
use sha2::{Digest, Sha256};

use super::{
    AcceleratorExecutionEvidence, AcceleratorExecutionPath, DevicePreference, DynamicShapeFallback,
    EmbeddedRuntime, ExecutionDigest, ExecutionReceipt, InferenceLimits,
    MicrobatchExecutionEvidence, ModelIdentity, RuntimeDeviceIdentity, RuntimeDeviceKind,
    RuntimeMemoryReservations, ShapeProfile, ShapeProfileBinding, ShapeProfileDeclaration,
    ShapeProfileExecutionPath, ShapeProfileFallbackReason, ShapeProfileRequest,
};

fn digest(byte: char) -> String {
    byte.to_string().repeat(64)
}

fn unique_digest(index: usize) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("shape-profile-{index}").as_bytes())
    )
}

fn cpu() -> RuntimeDeviceIdentity {
    RuntimeDeviceIdentity {
        kind: RuntimeDeviceKind::Cpu,
        ordinal: None,
    }
}

fn reservations() -> RuntimeMemoryReservations {
    RuntimeMemoryReservations::default()
        .with_host_fixed_bytes(32)
        .with_host_scratch_bytes(128)
        .with_device_fixed_bytes(64)
        .with_device_scratch_bytes(256)
}

fn binding() -> ShapeProfileBinding {
    ShapeProfileBinding::new(
        digest('1'),
        digest('2'),
        cpu(),
        digest('3'),
        reservations(),
        digest('4'),
    )
    .unwrap()
}

fn profile(class: char, implementation: char, batch: usize, elements: usize) -> ShapeProfile {
    ShapeProfile::new(
        digest(implementation),
        digest(class),
        batch,
        elements,
        64,
        128,
    )
    .unwrap()
}

fn request(
    input: &ExecutionDigest,
    class: char,
    batch: usize,
    elements: usize,
) -> ShapeProfileRequest {
    ShapeProfileRequest::new(&input.sha256, digest(class), batch, elements).unwrap()
}

#[test]
fn declaration_is_order_independent_and_selects_only_the_model_owned_class() {
    let current = binding();
    let first = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('b', '6', 8, 4_096), profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::allow(digest('9')).unwrap(),
    )
    .unwrap();
    let second = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048), profile('b', '6', 8, 4_096)],
        DynamicShapeFallback::allow(digest('9')).unwrap(),
    )
    .unwrap();
    assert_eq!(first.declaration_sha256(), second.declaration_sha256());
    assert_eq!(first.profiles()[0].shape_class_sha256(), digest('a'));

    let input = ExecutionDigest::token_ids(&[1, 2, 3]);
    let selection = first
        .select(&current, &request(&input, 'a', 4, 2_048))
        .unwrap();
    let ShapeProfileExecutionPath::Profile {
        profile_sha256,
        implementation_sha256,
    } = selection.path()
    else {
        panic!("an exact declared class should not use the dynamic fallback");
    };
    assert_eq!(profile_sha256, first.profiles()[0].profile_sha256());
    assert_eq!(implementation_sha256, &digest('5'));
    selection.evidence().validate().unwrap();

    let encoded = serde_json::to_string(selection.evidence()).unwrap();
    assert!(encoded.contains("implementationSha256"));
    assert!(!encoded.contains("implementation_sha256"));
    assert!(!encoded.contains("shapeClass"));
    assert!(!encoded.contains("batchSize"));
    assert!(!encoded.contains("tensorElements"));
    assert!(!encoded.contains(&digest('a')));

    let declaration_json = serde_json::to_string(&first).unwrap();
    assert!(declaration_json.contains("implementationSha256"));
    assert!(!declaration_json.contains("implementation_sha256"));
}

#[test]
fn selection_is_independent_of_model_family_and_input_representation() {
    let current = binding();
    let declaration = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::Deny,
    )
    .unwrap();
    let inputs = [
        ExecutionDigest::token_ids(&[1, 2]),
        ExecutionDigest::utf8_text("model-neutral"),
        ExecutionDigest::image_request(&[1, 2, 3, 4], 1),
        ExecutionDigest::f32_tensor(&[1, 2], &[0.25, 0.75]),
    ];

    for input in inputs {
        let selection = declaration
            .select(&current, &request(&input, 'a', 1, 2))
            .unwrap();
        assert!(matches!(
            selection.path(),
            ShapeProfileExecutionPath::Profile { .. }
        ));
        assert_eq!(selection.evidence().input_sha256, input.sha256);
    }
}

#[test]
fn dynamic_fallback_records_a_bounded_reason_and_exact_implementation() {
    let current = binding();
    let declaration = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::allow(digest('9')).unwrap(),
    )
    .unwrap();
    let input = ExecutionDigest::token_ids(&[1, 2, 3]);

    for (request, expected) in [
        (
            request(&input, 'b', 1, 1),
            ShapeProfileFallbackReason::ShapeClassUnavailable,
        ),
        (
            request(&input, 'a', 5, 1),
            ShapeProfileFallbackReason::BatchBoundExceeded,
        ),
        (
            request(&input, 'a', 4, 2_049),
            ShapeProfileFallbackReason::TensorElementBoundExceeded,
        ),
    ] {
        let selection = declaration.select(&current, &request).unwrap();
        assert_eq!(
            selection.path(),
            &ShapeProfileExecutionPath::DynamicFallback {
                reason: expected,
                implementation_sha256: digest('9'),
            }
        );
    }
}

#[test]
fn denied_dynamic_fallback_fails_closed() {
    let current = binding();
    let declaration = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::Deny,
    )
    .unwrap();
    let input = ExecutionDigest::token_ids(&[1]);

    let error = declaration
        .select(&current, &request(&input, 'b', 1, 1))
        .unwrap_err();
    assert!(error.to_string().contains("dynamic fallback is denied"));
}

#[test]
fn every_runtime_binding_dimension_rejects_a_stale_declaration() {
    let declared = binding();
    let declaration = ShapeProfileDeclaration::new(
        declared.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::Deny,
    )
    .unwrap();
    let input = ExecutionDigest::token_ids(&[1]);
    let request = request(&input, 'a', 1, 1);

    let mut mutations = Vec::new();
    let mut weights = declared.clone();
    weights.weights_sha256 = digest('a');
    mutations.push(weights);
    let mut graph = declared.clone();
    graph.graph_sha256 = digest('b');
    mutations.push(graph);
    let mut device = declared.clone();
    device.runtime_device = RuntimeDeviceIdentity {
        kind: RuntimeDeviceKind::Cuda,
        ordinal: Some(0),
    };
    mutations.push(device);
    let mut topology = declared.clone();
    topology.device_topology_sha256 = digest('c');
    mutations.push(topology);
    let mut host_scratch = declared.clone();
    host_scratch.runtime_reservations.host_scratch_bytes += 1;
    mutations.push(host_scratch);
    let mut device_scratch = declared.clone();
    device_scratch.runtime_reservations.device_scratch_bytes += 1;
    mutations.push(device_scratch);
    let mut tee = declared.clone();
    tee.tee_policy_sha256 = digest('d');
    mutations.push(tee);

    for current in mutations {
        let error = declaration.select(&current, &request).unwrap_err();
        assert!(error.to_string().contains("stale"));
    }
}

#[test]
fn single_device_binding_derives_topology_from_the_resolved_device() {
    let cpu = ShapeProfileBinding::for_single_device(
        digest('1'),
        digest('2'),
        cpu(),
        reservations(),
        digest('4'),
    )
    .unwrap();
    let cuda = ShapeProfileBinding::for_single_device(
        digest('1'),
        digest('2'),
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cuda,
            ordinal: Some(0),
        },
        reservations(),
        digest('4'),
    )
    .unwrap();
    assert_ne!(cpu.device_topology_sha256, cuda.device_topology_sha256);
    assert_ne!(
        cpu.binding_sha256().unwrap(),
        cuda.binding_sha256().unwrap()
    );
}

#[test]
fn malformed_duplicate_and_overcommitted_profiles_are_rejected() {
    let current = binding();
    assert!(ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048), profile('a', '6', 8, 4_096)],
        DynamicShapeFallback::Deny,
    )
    .is_err());

    let overcommitted = ShapeProfile::new(digest('5'), digest('a'), 4, 2_048, 129, 128).unwrap();
    assert!(ShapeProfileDeclaration::new(
        current.clone(),
        vec![overcommitted],
        DynamicShapeFallback::Deny,
    )
    .is_err());

    let too_many = (0..=256)
        .map(|index| {
            ShapeProfile::new(
                unique_digest(index + 1_000),
                unique_digest(index),
                1,
                1,
                0,
                0,
            )
            .unwrap()
        })
        .collect();
    assert!(ShapeProfileDeclaration::new(current, too_many, DynamicShapeFallback::Deny,).is_err());
}

#[test]
fn serialized_declaration_tampering_is_detected_before_selection() {
    let current = binding();
    let declaration = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::Deny,
    )
    .unwrap();
    let input = ExecutionDigest::token_ids(&[1]);
    let request = request(&input, 'a', 1, 1);

    for pointer in [
        "/binding/graphSha256",
        "/binding/deviceTopologySha256",
        "/binding/runtimeReservations/hostScratchBytes",
        "/profiles/0/maxBatchSize",
        "/profiles/0/profileSha256",
        "/declarationSha256",
    ] {
        let mut encoded = serde_json::to_value(&declaration).unwrap();
        let target = encoded.pointer_mut(pointer).unwrap();
        *target = if target.is_number() {
            json!(9_999)
        } else {
            json!(digest('e'))
        };
        let tampered: ShapeProfileDeclaration = serde_json::from_value(encoded).unwrap();
        assert!(tampered.select(&current, &request).is_err(), "{pointer}");
    }
}

#[test]
fn receipt_binds_the_selection_to_model_runtime_and_exact_input() {
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap();
    let current = binding();
    let declaration = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::Deny,
    )
    .unwrap();
    let input = ExecutionDigest::token_ids(&[1, 2, 3]);
    let output = ExecutionDigest::token_ids(&[4]);
    let selection = declaration
        .select(&current, &request(&input, 'a', 1, 3))
        .unwrap();
    let model = ModelIdentity::new("test-model", "revision-1", digest('1'));

    let receipt = runtime
        .receipt_with_shape_profile(model.clone(), input.clone(), output.clone(), &selection)
        .unwrap();
    assert_eq!(receipt.schema, ExecutionReceipt::SHAPE_PROFILE_SCHEMA);
    assert_eq!(receipt.shape_profile, Some(selection.evidence().clone()));
    assert!(runtime.attach_shape_profile(receipt, &selection).is_err());

    let attached = runtime
        .attach_shape_profile(
            runtime.receipt(model.clone(), input.clone(), output.clone()),
            &selection,
        )
        .unwrap();
    assert_eq!(attached.schema, ExecutionReceipt::SHAPE_PROFILE_SCHEMA);
    assert_eq!(attached.shape_profile, Some(selection.evidence().clone()));

    let wrong_model = ModelIdentity::new("test-model", "revision-1", digest('f'));
    assert!(runtime
        .receipt_with_shape_profile(wrong_model, input.clone(), output.clone(), &selection)
        .is_err());
    assert!(runtime
        .receipt_with_shape_profile(model, ExecutionDigest::token_ids(&[9]), output, &selection,)
        .is_err());
}

#[test]
fn receipt_composition_preserves_only_valid_existing_execution_evidence() {
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap();
    let current = binding();
    let declaration = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::Deny,
    )
    .unwrap();
    let input = ExecutionDigest::token_ids(&[1, 2, 3]);
    let output = ExecutionDigest::token_ids(&[4]);
    let selection = declaration
        .select(&current, &request(&input, 'a', 1, 3))
        .unwrap();
    let model = ModelIdentity::new("test-model", "revision-1", digest('1'));
    let mut microbatch_receipt = runtime.receipt(model.clone(), input.clone(), output.clone());
    microbatch_receipt.schema = ExecutionReceipt::MICROBATCH_SCHEMA.to_string();
    microbatch_receipt.microbatch = Some(MicrobatchExecutionEvidence {
        schema: MicrobatchExecutionEvidence::SCHEMA.to_string(),
        session_declaration_sha256: Some(digest('7')),
        plan_sha256: digest('8'),
        batch_index: 0,
        batch_count: 1,
        slot_count: 1,
        model_admission_queued: false,
        device_admission_queued: false,
    });

    let combined = runtime
        .attach_shape_profile(microbatch_receipt.clone(), &selection)
        .unwrap();
    assert_eq!(combined.schema, ExecutionReceipt::SHAPE_PROFILE_SCHEMA);
    assert_eq!(combined.microbatch, microbatch_receipt.microbatch);
    assert_eq!(combined.shape_profile, Some(selection.evidence().clone()));

    let accelerator = AcceleratorExecutionEvidence {
        schema: AcceleratorExecutionEvidence::SCHEMA.to_string(),
        declaration_sha256: digest('6'),
        weights_sha256: digest('1'),
        runtime_device: cpu(),
        execution_device: cpu(),
        path: AcceleratorExecutionPath::Accelerator,
        fallback_target: None,
        implementation_sha256: digest('9'),
        confidential_claims_sha256: None,
        device_mesh_sha256: None,
        execution_devices: Vec::new(),
        peer_transfers_sha256: None,
        input_sha256: input.sha256.clone(),
        output_sha256: output.sha256.clone(),
    };
    let accelerator_receipt = runtime
        .receipt_with_accelerator(model, input, output, accelerator)
        .unwrap();
    let combined = runtime
        .attach_shape_profile(accelerator_receipt, &selection)
        .unwrap();
    assert!(combined.accelerator.is_some());
    assert_eq!(combined.shape_profile, Some(selection.evidence().clone()));

    microbatch_receipt.microbatch.as_mut().unwrap().batch_count = 0;
    assert!(runtime
        .attach_shape_profile(microbatch_receipt, &selection)
        .is_err());
}

#[test]
fn public_types_are_send_sync_and_debug_output_is_geometry_free() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ShapeProfileBinding>();
    assert_send_sync::<ShapeProfile>();
    assert_send_sync::<ShapeProfileDeclaration>();
    assert_send_sync::<ShapeProfileRequest>();
    assert_send_sync::<super::ShapeProfileSelection>();
    assert_send_sync::<super::ShapeProfileExecutionEvidence>();

    let current = binding();
    let declaration = ShapeProfileDeclaration::new(
        current.clone(),
        vec![profile('a', '5', 4, 2_048)],
        DynamicShapeFallback::allow(digest('9')).unwrap(),
    )
    .unwrap();
    let input = ExecutionDigest::token_ids(&[1]);
    let request = request(&input, 'a', 1, 1);
    let selection = declaration.select(&current, &request).unwrap();
    let profile_sha256 = declaration.profiles()[0].profile_sha256().to_string();
    for debug in [
        format!("{current:?}"),
        format!("{:?}", declaration.profiles()[0]),
        format!("{declaration:?}"),
        format!("{request:?}"),
        format!("{selection:?}"),
        format!("{:?}", selection.path()),
        format!("{:?}", selection.evidence()),
    ] {
        assert!(!debug.contains(&digest('a')));
        assert!(!debug.contains(&digest('5')));
        assert!(!debug.contains(&profile_sha256));
        assert!(!debug.contains("2048"));
    }
}
