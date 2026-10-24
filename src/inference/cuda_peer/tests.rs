use super::*;
use crate::inference::DevicePreference;
use candle_core::{Device, Var};

#[test]
fn peer_scalars_are_closed_and_preserve_the_exact_container_dtype() {
    for (kind, dtype) in [
        (CudaPeerScalar::U8, DType::U8),
        (CudaPeerScalar::U32, DType::U32),
        (CudaPeerScalar::F16, DType::F16),
        (CudaPeerScalar::Bf16, DType::BF16),
        (CudaPeerScalar::F32, DType::F32),
    ] {
        assert_eq!(CudaPeerScalar::parse(kind.as_str()).unwrap(), kind);
        assert_eq!(CudaPeerScalar::from_dtype(dtype).unwrap(), kind);
    }
    for name in ["", "auto", "cpu", "f64", "FP32"] {
        assert!(CudaPeerScalar::parse(name).is_err());
    }
    assert!(CudaPeerScalar::from_dtype(DType::F64).is_err());
}

#[test]
fn narrow_transposed_view_admission_counts_only_the_private_snapshot_not_backing_storage() {
    let data: Vec<_> = (0..1024).map(|i| i as f32 / 8.).collect();
    let source = Var::from_vec(data.clone(), (32, 32), &Device::Cpu).unwrap();
    let view = source
        .as_tensor()
        .narrow(0, 7, 2)
        .unwrap()
        .narrow(1, 9, 3)
        .unwrap()
        .t()
        .unwrap();
    assert!(!view.is_contiguous());
    assert_eq!(
        admit_tensor(&InferenceLimits::default(), &view, 24).unwrap(),
        (CudaPeerScalar::F32, 24)
    );
    let snapshot = view.force_contiguous().unwrap().detach();
    assert!(snapshot.is_contiguous());
    assert_eq!(snapshot.layout().start_offset(), 0);
    let expected = vec![
        vec![data[7 * 32 + 9], data[8 * 32 + 9]],
        vec![data[7 * 32 + 10], data[8 * 32 + 10]],
        vec![data[7 * 32 + 11], data[8 * 32 + 11]],
    ];
    source
        .set(&Tensor::zeros((32, 32), DType::F32, &Device::Cpu).unwrap())
        .unwrap();
    assert_eq!(snapshot.to_vec2::<f32>().unwrap(), expected);
}

#[test]
fn view_byte_live_element_dtype_and_rank_bounds_fail_before_cuda() {
    let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
    assert!(admit_tensor(&InferenceLimits::default(), &tensor, 23).is_err());
    let limits = InferenceLimits {
        max_state_bytes: 71,
        ..InferenceLimits::default()
    };
    assert!(admit_tensor(&limits, &tensor, 24).is_err());
    let limits = InferenceLimits {
        max_tensor_elements: 5,
        ..InferenceLimits::default()
    };
    assert!(admit_tensor(&limits, &tensor, 24).is_err());
    assert!(admit_tensor(
        &InferenceLimits::default(),
        &tensor.to_dtype(DType::F64).unwrap(),
        48
    )
    .is_err());
    let rank = Tensor::zeros(vec![1; 9], DType::U8, &Device::Cpu).unwrap();
    assert!(admit_tensor(&InferenceLimits::default(), &rank, 64).is_err());
    let empty = Tensor::zeros((0, 288), DType::U8, &Device::Cpu).unwrap();
    assert_eq!(
        admit_tensor(&InferenceLimits::default(), &empty, 1).unwrap(),
        (CudaPeerScalar::U8, 0)
    );
    assert!(admit_tensor(&InferenceLimits::default(), &empty, 0).is_err());
}

#[test]
fn public_group_never_presents_cpu_storage_as_cuda_or_a_host_fallback() {
    let cpu = RuntimeDevice::resolve(DevicePreference::Cpu).unwrap();
    assert!(
        CudaPeerGroup::new(&[cpu.clone(), cpu], &[(0, 1)], InferenceLimits::default()).is_err()
    );
    assert!(CudaPeerGroup::new(&[], &[], InferenceLimits::default()).is_err());
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(matches!(
        check_cancel(&cancel),
        Err(PowerError::InferenceCancelled)
    ));
}
