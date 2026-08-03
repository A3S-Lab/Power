use candle_core::Tensor;

use super::accelerator_mesh_tests::{hierarchy_with_limits, local_mesh, spec};
use super::*;

#[test]
fn mesh_reuses_existing_graph_input_and_state_limits() {
    let limits = InferenceLimits {
        max_graph_nodes: 1,
        ..InferenceLimits::default()
    };
    let (_directory, runtime, hierarchy, _permit, _cancellation) = hierarchy_with_limits(limits);
    assert!(hierarchy
        .declare_accelerator_mesh_residency(&spec(), &local_mesh(&runtime))
        .is_err());

    let limits = InferenceLimits {
        max_input_bytes: 4,
        ..InferenceLimits::default()
    };
    let (_directory, runtime, hierarchy, _permit, _cancellation) = hierarchy_with_limits(limits);
    assert!(hierarchy
        .declare_accelerator_mesh_residency(&spec(), &local_mesh(&runtime))
        .is_err());

    let limits = InferenceLimits {
        max_state_bytes: 8,
        ..InferenceLimits::default()
    };
    let (_directory, runtime, hierarchy, _permit, _cancellation) = hierarchy_with_limits(limits);
    assert!(hierarchy
        .declare_accelerator_mesh_residency(&spec(), &local_mesh(&runtime))
        .is_err());
}

#[test]
fn malformed_mesh_declarations_and_transfer_count_overruns_fail_closed() {
    let (_directory, runtime, hierarchy, permit, cancellation) =
        hierarchy_with_limits(InferenceLimits::default());
    let mesh = local_mesh(&runtime);
    let mut tampered = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    tampered.device_mesh.as_mut().unwrap().peer_transfers[0].max_transfer_bytes += 1;
    assert!(hierarchy
        .resolve_accelerator_mesh_batch(&tampered, &mesh, None, &permit, &cancellation,)
        .is_err());

    let declaration = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_mesh_batch(&declaration, &mesh, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("mesh should resolve");
    };
    let input = Tensor::new(&[2_f32, 3_f32], runtime.device().tensor_device()).unwrap();
    assert!(batch
        .execute_mesh_or_fallback(&input, &cancellation, |input, _groups, mesh, _| {
            let _ = mesh.transfer("home", "peer", input)?;
            let _ = mesh.transfer("home", "peer", input)?;
            Ok(AcceleratorKernelOutcome::Output(input.clone()))
        })
        .is_err());
}

#[test]
fn aggregate_transfer_budget_and_cancellation_fail_closed() {
    let (_directory, runtime, hierarchy, permit, cancellation) =
        hierarchy_with_limits(InferenceLimits::default());
    let mesh = AcceleratorDeviceMesh::new(
        "home",
        vec![
            AcceleratorMeshDevice::new("home", runtime.device().clone()),
            AcceleratorMeshDevice::new(
                "peer",
                RuntimeDevice::test_accelerator(RuntimeDeviceKind::Cuda, 1).unwrap(),
            ),
        ],
        vec![
            AcceleratorPeerTransferSpec::new("home", "peer", 8, 1),
            AcceleratorPeerTransferSpec::new("peer", "home", 8, 1),
        ],
        8,
    )
    .unwrap();
    let declaration = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_mesh_batch(&declaration, &mesh, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("mesh should resolve");
    };
    let input = Tensor::new(&[2_f32, 3_f32], runtime.device().tensor_device()).unwrap();
    assert!(batch
        .execute_mesh_or_fallback(&input, &cancellation, |input, _groups, mesh, _| {
            let AcceleratorPeerTransferOutcome::Transferred(peer) =
                mesh.transfer("home", "peer", input)?
            else {
                return Ok(AcceleratorKernelOutcome::Unavailable);
            };
            let _ = mesh.transfer("peer", "home", &peer)?;
            Ok(AcceleratorKernelOutcome::Output(input.clone()))
        })
        .is_err());

    let declaration = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_mesh_batch(&declaration, &mesh, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("mesh should resolve");
    };
    cancellation.cancel();
    assert!(batch
        .execute_mesh_or_fallback(&input, &cancellation, |input, _groups, _mesh, _| {
            Ok(AcceleratorKernelOutcome::Output(input.clone()))
        })
        .is_err());
}
