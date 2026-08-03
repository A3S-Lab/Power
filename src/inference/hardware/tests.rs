use super::*;
use crate::inference::{CacheEvictionPolicy, InferenceLimits, ResidencyPolicy, TelemetryMode};

fn pool(
    total_bytes: u64,
    available_bytes: u64,
    source: MemoryDiscoverySource,
    unified_with_host: bool,
) -> MemoryPoolSnapshot {
    MemoryPoolSnapshot::new(total_bytes, available_bytes, source, unified_with_host).unwrap()
}

#[test]
fn memory_pool_rejects_impossible_availability() {
    assert!(
        MemoryPoolSnapshot::new(100, 101, MemoryDiscoverySource::LinuxProcMeminfo, false,).is_err()
    );
    assert!(
        MemoryPoolSnapshot::new(0, 0, MemoryDiscoverySource::LinuxProcMeminfo, false,).is_err()
    );
}

#[test]
fn snapshot_requires_a_device_pool_only_for_accelerators() {
    let host = pool(100, 50, MemoryDiscoverySource::LinuxProcMeminfo, false);
    let device = pool(100, 50, MemoryDiscoverySource::CudaDriver, false);

    assert!(HardwareMemorySnapshot::new("cpu", host.clone(), Some(device.clone())).is_err());
    assert!(HardwareMemorySnapshot::new("cuda:0", host.clone(), None).is_err());
    assert!(HardwareMemorySnapshot::new("cuda:0", host, Some(device)).is_ok());
}

#[test]
fn linux_meminfo_parser_is_strict_and_checked() {
    let valid = "MemTotal:       1024 kB\nMemAvailable:    512 kB\n";
    assert_eq!(parse_meminfo_kib(valid, "MemTotal").unwrap(), 1_048_576);
    assert!(parse_meminfo_kib("MemTotal: 1 MB\n", "MemTotal").is_err());
    assert!(parse_meminfo_kib("MemFree: 1 kB\n", "MemTotal").is_err());
    assert!(parse_meminfo_kib("MemTotal: nope kB\n", "MemTotal").is_err());
}

#[test]
fn cpu_budget_is_fractional_reserved_and_runtime_capped() {
    let snapshot = HardwareMemorySnapshot::new(
        "cpu",
        pool(1_000, 800, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    let policy = ResidencyBudgetPolicy::new(5_000, 5_000)
        .unwrap()
        .with_host_reserve_bytes(100);
    let limits = InferenceLimits {
        max_resident_weight_bytes: 300,
        ..InferenceLimits::default()
    };

    let plan = policy.plan(&snapshot, &limits).unwrap();

    assert_eq!(plan.host_cache_bytes, 300);
    assert_eq!(plan.device_cache_bytes, 0);
    assert_eq!(plan.total_cache_bytes, 300);
}

#[test]
fn fixed_state_and_peak_scratch_are_reserved_before_cache_bytes() {
    let snapshot = HardwareMemorySnapshot::new(
        "cuda:0",
        pool(2_000, 1_000, MemoryDiscoverySource::LinuxProcMeminfo, false),
        Some(pool(1_500, 800, MemoryDiscoverySource::CudaDriver, false)),
    )
    .unwrap();
    let reservations = RuntimeMemoryReservations::default()
        .with_host_fixed_bytes(200)
        .with_host_scratch_bytes(100)
        .with_device_fixed_bytes(100)
        .with_device_scratch_bytes(50);
    let policy = ResidencyBudgetPolicy::new(5_000, 5_000)
        .unwrap()
        .with_host_reserve_bytes(100)
        .with_device_reserve_bytes(50)
        .with_runtime_reservations(reservations)
        .unwrap();

    let plan = policy.plan(&snapshot, &InferenceLimits::default()).unwrap();

    assert_eq!(plan.host_cache_bytes, 150);
    assert_eq!(plan.device_cache_bytes, 225);
    assert_eq!(plan.total_cache_bytes, 375);
    assert!(plan.apply_to(&ResidencyPolicy::default()).is_err());
    let projected = plan
        .apply_to_revalidated(&ResidencyPolicy::default(), &snapshot)
        .unwrap();
    assert_eq!(projected.host_cache_bytes, 150);
    assert_eq!(projected.device_cache_bytes, 225);
}

#[test]
fn unified_memory_counts_both_runtime_reservations_once() {
    let snapshot = HardwareMemorySnapshot::new(
        "metal:0",
        pool(
            2_000,
            1_000,
            MemoryDiscoverySource::MachHostStatistics,
            false,
        ),
        Some(pool(
            1_500,
            900,
            MemoryDiscoverySource::MetalRecommendedWorkingSet,
            true,
        )),
    )
    .unwrap();
    let reservations = RuntimeMemoryReservations::default()
        .with_host_fixed_bytes(100)
        .with_host_scratch_bytes(50)
        .with_device_fixed_bytes(100)
        .with_device_scratch_bytes(50);
    let policy = ResidencyBudgetPolicy::new(10_000, 10_000)
        .unwrap()
        .with_host_reserve_bytes(50)
        .with_device_reserve_bytes(100)
        .with_runtime_reservations(reservations)
        .unwrap();

    let plan = policy.plan(&snapshot, &InferenceLimits::default()).unwrap();

    assert_eq!(plan.shared_available_bytes, Some(900));
    assert_eq!(plan.host_cache_bytes, 0);
    assert_eq!(plan.device_cache_bytes, 500);
    assert_eq!(plan.total_cache_bytes, 500);
    plan.revalidate_pressure(&snapshot).unwrap();

    let pressured = HardwareMemorySnapshot::new(
        "metal:0",
        pool(
            2_000,
            1_000,
            MemoryDiscoverySource::MachHostStatistics,
            false,
        ),
        Some(pool(
            1_500,
            899,
            MemoryDiscoverySource::MetalRecommendedWorkingSet,
            true,
        )),
    )
    .unwrap();
    assert!(plan.revalidate_pressure(&pressured).is_err());
}

#[test]
fn discrete_device_is_prioritized_without_exceeding_runtime_limit() {
    let snapshot = HardwareMemorySnapshot::new(
        "cuda:0",
        pool(2_000, 1_000, MemoryDiscoverySource::LinuxProcMeminfo, false),
        Some(pool(1_500, 900, MemoryDiscoverySource::CudaDriver, false)),
    )
    .unwrap();
    let policy = ResidencyBudgetPolicy::new(5_000, 5_000)
        .unwrap()
        .with_host_reserve_bytes(100)
        .with_device_reserve_bytes(100);
    let limits = InferenceLimits {
        max_resident_weight_bytes: 600,
        ..InferenceLimits::default()
    };

    let plan = policy.plan(&snapshot, &limits).unwrap();

    assert_eq!(plan.device_cache_bytes, 400);
    assert_eq!(plan.host_cache_bytes, 200);
    assert_eq!(plan.total_cache_bytes, 600);
    assert!(!plan.unified_memory);
}

#[test]
fn unified_memory_is_never_double_counted() {
    let snapshot = HardwareMemorySnapshot::new(
        "metal:0",
        pool(
            2_000,
            1_000,
            MemoryDiscoverySource::MachHostStatistics,
            false,
        ),
        Some(pool(
            1_500,
            900,
            MemoryDiscoverySource::MetalRecommendedWorkingSet,
            true,
        )),
    )
    .unwrap();
    let policy = ResidencyBudgetPolicy::new(5_000, 5_000)
        .unwrap()
        .with_host_reserve_bytes(100)
        .with_device_reserve_bytes(100);

    let plan = policy.plan(&snapshot, &InferenceLimits::default()).unwrap();

    assert_eq!(plan.device_cache_bytes, 400);
    assert_eq!(plan.host_cache_bytes, 400);
    assert_eq!(plan.total_cache_bytes, 800);
    assert!(plan.unified_memory);
    assert_eq!(plan.shared_available_bytes, Some(900));
}

#[test]
fn host_first_order_is_explicit_and_deterministic() {
    let snapshot = HardwareMemorySnapshot::new(
        "cuda:0",
        pool(2_000, 1_000, MemoryDiscoverySource::LinuxProcMeminfo, false),
        Some(pool(1_500, 900, MemoryDiscoverySource::CudaDriver, false)),
    )
    .unwrap();
    let policy = ResidencyBudgetPolicy::new(5_000, 5_000)
        .unwrap()
        .with_allocation_order(ResidencyAllocationOrder::HostFirst);
    let limits = InferenceLimits {
        max_resident_weight_bytes: 600,
        ..InferenceLimits::default()
    };

    let plan = policy.plan(&snapshot, &limits).unwrap();

    assert_eq!(plan.host_cache_bytes, 500);
    assert_eq!(plan.device_cache_bytes, 100);
}

#[test]
fn optional_caps_apply_before_the_global_limit() {
    let snapshot = HardwareMemorySnapshot::new(
        "cuda:0",
        pool(2_000, 1_000, MemoryDiscoverySource::LinuxProcMeminfo, false),
        Some(pool(2_000, 1_000, MemoryDiscoverySource::CudaDriver, false)),
    )
    .unwrap();
    let policy = ResidencyBudgetPolicy::new(10_000, 10_000)
        .unwrap()
        .with_max_host_cache_bytes(250)
        .unwrap()
        .with_max_device_cache_bytes(350)
        .unwrap();

    let plan = policy.plan(&snapshot, &InferenceLimits::default()).unwrap();

    assert_eq!(plan.host_cache_bytes, 250);
    assert_eq!(plan.device_cache_bytes, 350);
    assert_eq!(plan.total_cache_bytes, 600);
}

#[test]
fn budget_plan_projects_only_cache_bytes() {
    let snapshot = HardwareMemorySnapshot::new(
        "cpu",
        pool(1_000, 800, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    let plan = ResidencyBudgetPolicy::new(5_000, 0)
        .unwrap()
        .plan(&snapshot, &InferenceLimits::default())
        .unwrap();
    let base = ResidencyPolicy {
        host_cache_bytes: 1,
        device_cache_bytes: 2,
        max_entries_per_layer: 7,
        cache_eviction: CacheEvictionPolicy::Lru,
        max_prefetch_tasks: 2,
        telemetry: TelemetryMode::Aggregate,
        ..ResidencyPolicy::default()
    };

    let projected = plan.apply_to(&base).unwrap();

    assert_eq!(projected.host_cache_bytes, 400);
    assert_eq!(projected.device_cache_bytes, 0);
    assert_eq!(projected.max_entries_per_layer, 7);
    assert_eq!(projected.cache_eviction, CacheEvictionPolicy::Lru);
    assert_eq!(projected.max_prefetch_tasks, 2);
    assert_eq!(projected.telemetry, TelemetryMode::Aggregate);
}

#[test]
fn serialized_or_mutated_plans_are_revalidated_before_use() {
    let snapshot = HardwareMemorySnapshot::new(
        "cpu",
        pool(1_000, 800, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    let mut plan = ResidencyBudgetPolicy::new(5_000, 0)
        .unwrap()
        .plan(&snapshot, &InferenceLimits::default())
        .unwrap();
    plan.host_cache_bytes += 1;

    assert!(plan.apply_to(&ResidencyPolicy::default()).is_err());

    let mut reservation_mutated = ResidencyBudgetPolicy::new(5_000, 0)
        .unwrap()
        .plan(&snapshot, &InferenceLimits::default())
        .unwrap();
    reservation_mutated
        .policy
        .runtime_reservations
        .host_fixed_bytes = 1;
    assert!(reservation_mutated
        .apply_to(&ResidencyPolicy::default())
        .is_err());

    let invalid_policy = serde_json::from_str::<ResidencyBudgetPolicy>(
        r#"{"hostAvailableFractionBps":10001,"deviceAvailableFractionBps":0}"#,
    )
    .unwrap();
    assert!(invalid_policy
        .plan(&snapshot, &InferenceLimits::default())
        .is_err());
    assert!(serde_json::from_str::<ResidencyBudgetPolicy>(
        r#"{"hostAvailableFractionBps":1,"deviceAvailableFractionBps":0,"unknown":true}"#,
    )
    .is_err());
}

#[test]
fn pressure_revalidation_uses_current_availability_and_exact_topology() {
    let planned_snapshot = HardwareMemorySnapshot::new(
        "cpu",
        pool(2_000, 1_000, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    let reservations = RuntimeMemoryReservations::default()
        .with_host_fixed_bytes(200)
        .with_host_scratch_bytes(100);
    let policy = ResidencyBudgetPolicy::new(10_000, 0)
        .unwrap()
        .with_host_reserve_bytes(100)
        .with_runtime_reservations(reservations)
        .unwrap();
    let plan = policy
        .plan(&planned_snapshot, &InferenceLimits::default())
        .unwrap();
    assert_eq!(plan.host_cache_bytes, 600);

    plan.revalidate_pressure(&planned_snapshot).unwrap();
    let pressured = HardwareMemorySnapshot::new(
        "cpu",
        pool(2_000, 999, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    assert!(plan.revalidate_pressure(&pressured).is_err());
    assert!(plan
        .apply_to_revalidated(&ResidencyPolicy::default(), &pressured)
        .is_err());

    let changed_host = HardwareMemorySnapshot::new(
        "cpu",
        pool(2_001, 1_000, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    assert!(plan.revalidate_pressure(&changed_host).is_err());
}

#[test]
fn invalid_or_inapplicable_runtime_reservations_fail_closed() {
    let overflow = RuntimeMemoryReservations {
        host_fixed_bytes: u64::MAX,
        host_scratch_bytes: 1,
        ..RuntimeMemoryReservations::default()
    };
    assert!(ResidencyBudgetPolicy::new(1, 0)
        .unwrap()
        .with_runtime_reservations(overflow)
        .is_err());

    let cpu = HardwareMemorySnapshot::new(
        "cpu",
        pool(1_000, 800, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    let device_state = RuntimeMemoryReservations::default().with_device_fixed_bytes(1);
    assert!(ResidencyBudgetPolicy::new(1, 0)
        .unwrap()
        .with_runtime_reservations(device_state)
        .unwrap()
        .plan(&cpu, &InferenceLimits::default())
        .is_err());

    let oversized_host = RuntimeMemoryReservations::default().with_host_fixed_bytes(700);
    assert!(ResidencyBudgetPolicy::new(0, 1)
        .unwrap()
        .with_host_reserve_bytes(101)
        .with_runtime_reservations(oversized_host)
        .unwrap()
        .plan(&cpu, &InferenceLimits::default())
        .is_err());

    assert!(serde_json::from_str::<RuntimeMemoryReservations>(
        r#"{"hostFixedBytes":1,"unknown":true}"#
    )
    .is_err());
}

#[test]
fn runtime_reservations_have_a_backward_compatible_serde_default() {
    let policy = serde_json::from_str::<ResidencyBudgetPolicy>(
        r#"{"hostAvailableFractionBps":5000,"deviceAvailableFractionBps":0}"#,
    )
    .unwrap();
    assert_eq!(
        policy.runtime_reservations,
        RuntimeMemoryReservations::default()
    );

    let snapshot = HardwareMemorySnapshot::new(
        "cpu",
        pool(1_000, 800, MemoryDiscoverySource::LinuxProcMeminfo, false),
        None,
    )
    .unwrap();
    let plan = policy.plan(&snapshot, &InferenceLimits::default()).unwrap();
    let mut serialized = serde_json::to_value(&plan).unwrap();
    serialized["policy"]
        .as_object_mut()
        .unwrap()
        .remove("runtimeReservations");
    let restored: ResidencyBudgetPlan = serde_json::from_value(serialized).unwrap();
    restored.validate().unwrap();
    assert_eq!(restored, plan);
}

#[test]
fn invalid_budget_controls_fail_closed() {
    assert!(ResidencyBudgetPolicy::new(0, 0).is_err());
    assert!(ResidencyBudgetPolicy::new(10_001, 0).is_err());
    assert!(ResidencyBudgetPolicy::new(0, 10_001).is_err());
    assert!(ResidencyBudgetPolicy::new(1, 0)
        .unwrap()
        .with_max_host_cache_bytes(0)
        .is_err());
    assert!(ResidencyBudgetPolicy::new(0, 1)
        .unwrap()
        .with_max_device_cache_bytes(0)
        .is_err());
}

#[test]
fn live_cpu_snapshot_is_bounded_and_device_bound() {
    let snapshot =
        crate::inference::RuntimeDevice::resolve(crate::inference::DevicePreference::Cpu)
            .unwrap()
            .memory_snapshot()
            .unwrap();

    assert_eq!(snapshot.runtime_device, "cpu");
    assert!(snapshot.host.total_bytes > 0);
    assert!(snapshot.host.available_bytes <= snapshot.host.total_bytes);
    assert!(snapshot.device.is_none());
}

#[cfg(all(feature = "embedded-metal", target_os = "macos"))]
#[test]
fn live_metal_snapshot_uses_one_unified_pool() {
    let device =
        crate::inference::RuntimeDevice::resolve(crate::inference::DevicePreference::Metal {
            ordinal: 0,
        })
        .unwrap();
    let snapshot = device.memory_snapshot().unwrap();
    let metal = snapshot.device.as_ref().unwrap();

    assert_eq!(snapshot.runtime_device, "metal:0");
    assert_eq!(
        metal.source,
        MemoryDiscoverySource::MetalRecommendedWorkingSet
    );
    assert!(metal.unified_with_host);
    assert!(metal.total_bytes > 0);
    assert!(metal.available_bytes <= metal.total_bytes);

    let plan = ResidencyBudgetPolicy::new(5_000, 5_000)
        .unwrap()
        .plan(&snapshot, &InferenceLimits::default())
        .unwrap();
    assert!(plan.unified_memory);
    assert!(plan.total_cache_bytes <= snapshot.host.available_bytes);
    assert!(plan.total_cache_bytes <= metal.available_bytes);
}

#[test]
fn public_hardware_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HardwareMemorySnapshot>();
    assert_send_sync::<RuntimeMemoryReservations>();
    assert_send_sync::<ResidencyBudgetPolicy>();
    assert_send_sync::<ResidencyBudgetPlan>();
}
