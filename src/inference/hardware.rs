use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::{InferenceLimits, ResidencyPolicy, RuntimeDevice, RuntimeDeviceKind};

const BASIS_POINTS: u64 = 10_000;

/// Operating-system or accelerator API used for one memory observation.
///
/// Snapshots are returned only to the caller. Power never logs or persists
/// them automatically because exact capacity can be deployment-sensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryDiscoverySource {
    LinuxProcMeminfo,
    PosixSysconf,
    MachHostStatistics,
    WindowsGlobalMemoryStatus,
    CudaDriver,
    MetalRecommendedWorkingSet,
}

/// A bounded observation of one host or accelerator memory pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryPoolSnapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub source: MemoryDiscoverySource,
    /// True only when this accelerator allocation consumes the host pool too.
    pub unified_with_host: bool,
}

impl MemoryPoolSnapshot {
    pub fn new(
        total_bytes: u64,
        available_bytes: u64,
        source: MemoryDiscoverySource,
        unified_with_host: bool,
    ) -> Result<Self> {
        let snapshot = Self {
            total_bytes,
            available_bytes,
            source,
            unified_with_host,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    fn validate(&self) -> Result<()> {
        if self.total_bytes == 0 {
            return Err(PowerError::Config(
                "discovered memory total must be greater than zero".to_string(),
            ));
        }
        if self.available_bytes > self.total_bytes {
            return Err(PowerError::Config(format!(
                "discovered memory availability {} exceeds total {}",
                self.available_bytes, self.total_bytes
            )));
        }
        Ok(())
    }
}

/// Host plus optional device memory bound to one resolved runtime device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HardwareMemorySnapshot {
    pub runtime_device: String,
    pub host: MemoryPoolSnapshot,
    pub device: Option<MemoryPoolSnapshot>,
}

impl HardwareMemorySnapshot {
    pub fn new(
        runtime_device: impl Into<String>,
        host: MemoryPoolSnapshot,
        device: Option<MemoryPoolSnapshot>,
    ) -> Result<Self> {
        let snapshot = Self {
            runtime_device: runtime_device.into(),
            host,
            device,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    fn validate(&self) -> Result<()> {
        if self.runtime_device.is_empty()
            || self.runtime_device.len() > 64
            || self.runtime_device.chars().any(char::is_control)
        {
            return Err(PowerError::Config(
                "runtime device identity must contain 1 through 64 non-control characters"
                    .to_string(),
            ));
        }
        self.host.validate()?;
        if self.host.unified_with_host {
            return Err(PowerError::Config(
                "the host memory pool cannot be marked unified with itself".to_string(),
            ));
        }
        if let Some(device) = &self.device {
            device.validate()?;
        }
        if self.runtime_device == "cpu" && self.device.is_some() {
            return Err(PowerError::Config(
                "a CPU runtime cannot declare a device memory pool".to_string(),
            ));
        }
        if self.runtime_device != "cpu" && self.device.is_none() {
            return Err(PowerError::Config(
                "an accelerator runtime must declare its device memory pool".to_string(),
            ));
        }
        Ok(())
    }
}

/// Which tier receives scarce shared/global budget first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResidencyAllocationOrder {
    #[default]
    DeviceFirst,
    HostFirst,
}

/// Explicit policy for deriving cache budgets from a point-in-time snapshot.
///
/// Fractions use basis points. Automatic planning is opt-in; the default
/// [`ResidencyPolicy`] continues to cache zero bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidencyBudgetPolicy {
    pub host_available_fraction_bps: u16,
    pub device_available_fraction_bps: u16,
    #[serde(default)]
    pub host_reserve_bytes: u64,
    #[serde(default)]
    pub device_reserve_bytes: u64,
    #[serde(default)]
    pub max_host_cache_bytes: Option<u64>,
    #[serde(default)]
    pub max_device_cache_bytes: Option<u64>,
    #[serde(default)]
    pub allocation_order: ResidencyAllocationOrder,
}

impl ResidencyBudgetPolicy {
    pub fn new(
        host_available_fraction_bps: u16,
        device_available_fraction_bps: u16,
    ) -> Result<Self> {
        let policy = Self {
            host_available_fraction_bps,
            device_available_fraction_bps,
            host_reserve_bytes: 0,
            device_reserve_bytes: 0,
            max_host_cache_bytes: None,
            max_device_cache_bytes: None,
            allocation_order: ResidencyAllocationOrder::DeviceFirst,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn with_host_reserve_bytes(mut self, bytes: u64) -> Self {
        self.host_reserve_bytes = bytes;
        self
    }

    pub fn with_device_reserve_bytes(mut self, bytes: u64) -> Self {
        self.device_reserve_bytes = bytes;
        self
    }

    pub fn with_max_host_cache_bytes(mut self, bytes: u64) -> Result<Self> {
        if bytes == 0 {
            return Err(PowerError::Config(
                "maximum automatic host cache bytes must be greater than zero".to_string(),
            ));
        }
        self.max_host_cache_bytes = Some(bytes);
        Ok(self)
    }

    pub fn with_max_device_cache_bytes(mut self, bytes: u64) -> Result<Self> {
        if bytes == 0 {
            return Err(PowerError::Config(
                "maximum automatic device cache bytes must be greater than zero".to_string(),
            ));
        }
        self.max_device_cache_bytes = Some(bytes);
        Ok(self)
    }

    pub fn with_allocation_order(mut self, order: ResidencyAllocationOrder) -> Self {
        self.allocation_order = order;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.host_available_fraction_bps == 0 && self.device_available_fraction_bps == 0 {
            return Err(PowerError::Config(
                "at least one automatic residency fraction must be greater than zero".to_string(),
            ));
        }
        if u64::from(self.host_available_fraction_bps) > BASIS_POINTS
            || u64::from(self.device_available_fraction_bps) > BASIS_POINTS
        {
            return Err(PowerError::Config(
                "automatic residency fractions cannot exceed 10,000 basis points".to_string(),
            ));
        }
        if self.max_host_cache_bytes == Some(0) || self.max_device_cache_bytes == Some(0) {
            return Err(PowerError::Config(
                "automatic residency cache caps must be greater than zero when present".to_string(),
            ));
        }
        Ok(())
    }

    pub fn plan(
        &self,
        snapshot: &HardwareMemorySnapshot,
        limits: &InferenceLimits,
    ) -> Result<ResidencyBudgetPlan> {
        self.validate()?;
        snapshot.validate()?;
        limits.validate()?;
        ResidencyBudgetPlan::calculate(self, snapshot, limits.max_resident_weight_bytes)
    }
}

/// Reproducible automatic cache budget derived without loading model weights.
///
/// The snapshot can reveal host capacity, so Power returns it to the caller but
/// never includes it in placement telemetry or execution receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidencyBudgetPlan {
    pub schema: String,
    pub runtime_device: String,
    pub snapshot: HardwareMemorySnapshot,
    pub policy: ResidencyBudgetPolicy,
    pub runtime_limit_bytes: u64,
    pub host_cache_bytes: u64,
    pub device_cache_bytes: u64,
    pub total_cache_bytes: u64,
    pub unified_memory: bool,
    pub shared_available_bytes: Option<u64>,
}

impl ResidencyBudgetPlan {
    pub const SCHEMA: &'static str = "a3s.power.residency-budget-plan.v1";

    fn calculate(
        policy: &ResidencyBudgetPolicy,
        snapshot: &HardwareMemorySnapshot,
        runtime_limit_bytes: u64,
    ) -> Result<Self> {
        if runtime_limit_bytes == 0 {
            return Err(PowerError::Config(
                "runtime resident-weight limit must be greater than zero".to_string(),
            ));
        }
        let host_target = fractional_budget(
            snapshot.host.available_bytes,
            policy.host_reserve_bytes,
            policy.host_available_fraction_bps,
            policy.max_host_cache_bytes,
        );
        let device_target = snapshot.device.as_ref().map_or(0, |device| {
            fractional_budget(
                device.available_bytes,
                policy.device_reserve_bytes,
                policy.device_available_fraction_bps,
                policy.max_device_cache_bytes,
            )
        });
        let unified_memory = snapshot
            .device
            .as_ref()
            .is_some_and(|device| device.unified_with_host);
        let shared_available_bytes = snapshot
            .device
            .as_ref()
            .filter(|device| device.unified_with_host)
            .map(|device| snapshot.host.available_bytes.min(device.available_bytes));
        let shared_limit = shared_available_bytes.map(|available| {
            available.saturating_sub(policy.host_reserve_bytes.max(policy.device_reserve_bytes))
        });
        let total_limit = shared_limit.map_or(runtime_limit_bytes, |shared| {
            shared.min(runtime_limit_bytes)
        });
        let (host_cache_bytes, device_cache_bytes) = allocate_pair(
            host_target,
            device_target,
            total_limit,
            policy.allocation_order,
        );
        let total_cache_bytes = host_cache_bytes
            .checked_add(device_cache_bytes)
            .ok_or_else(|| {
                PowerError::Config("automatic residency budget overflowed".to_string())
            })?;

        Ok(Self {
            schema: Self::SCHEMA.to_string(),
            runtime_device: snapshot.runtime_device.clone(),
            snapshot: snapshot.clone(),
            policy: policy.clone(),
            runtime_limit_bytes,
            host_cache_bytes,
            device_cache_bytes,
            total_cache_bytes,
            unified_memory,
            shared_available_bytes,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA || self.runtime_device != self.snapshot.runtime_device {
            return Err(PowerError::Config(
                "automatic residency plan identity is invalid".to_string(),
            ));
        }
        self.policy.validate()?;
        self.snapshot.validate()?;
        let expected = Self::calculate(&self.policy, &self.snapshot, self.runtime_limit_bytes)?;
        if *self != expected {
            return Err(PowerError::Config(
                "automatic residency plan does not match its policy and memory snapshot"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Replaces only cache byte budgets, retaining all eviction, prefetch, and
    /// telemetry choices from the caller-owned base policy.
    pub fn apply_to(&self, base: &ResidencyPolicy) -> Result<ResidencyPolicy> {
        self.validate()?;
        let mut policy = base.clone();
        policy.host_cache_bytes = self.host_cache_bytes;
        policy.device_cache_bytes = self.device_cache_bytes;
        policy.validate()?;
        Ok(policy)
    }
}

impl RuntimeDevice {
    /// Discovers host and selected-device capacity without spawning a process.
    pub fn memory_snapshot(&self) -> Result<HardwareMemorySnapshot> {
        let host = discover_host_memory()?;
        let device = match self.kind() {
            RuntimeDeviceKind::Cpu => None,
            RuntimeDeviceKind::Cuda => Some(self.discover_cuda_memory()?),
            RuntimeDeviceKind::Metal => Some(self.discover_metal_memory()?),
        };
        HardwareMemorySnapshot::new(self.name(), host, device)
    }

    #[cfg(feature = "embedded-cuda")]
    fn discover_cuda_memory(&self) -> Result<MemoryPoolSnapshot> {
        let candle_core::Device::Cuda(device) = self.tensor_device() else {
            return Err(PowerError::BackendNotAvailable(
                "resolved CUDA runtime has no CUDA tensor device".to_string(),
            ));
        };
        let stream = device.cuda_stream();
        let (available, total) = stream.context().mem_get_info().map_err(|error| {
            PowerError::BackendNotAvailable(format!("failed to query CUDA device memory: {error}"))
        })?;
        MemoryPoolSnapshot::new(
            usize_to_u64(total, "CUDA total memory")?,
            usize_to_u64(available, "CUDA available memory")?,
            MemoryDiscoverySource::CudaDriver,
            false,
        )
    }

    #[cfg(not(feature = "embedded-cuda"))]
    fn discover_cuda_memory(&self) -> Result<MemoryPoolSnapshot> {
        Err(PowerError::BackendNotAvailable(
            "CUDA memory discovery requires the embedded-cuda feature".to_string(),
        ))
    }

    #[cfg(all(feature = "embedded-metal", target_os = "macos"))]
    fn discover_metal_memory(&self) -> Result<MemoryPoolSnapshot> {
        let candle_core::Device::Metal(device) = self.tensor_device() else {
            return Err(PowerError::BackendNotAvailable(
                "resolved Metal runtime has no Metal tensor device".to_string(),
            ));
        };
        let total = usize_to_u64(
            device.metal_device().recommended_max_working_set_size(),
            "Metal recommended working set",
        )?;
        let allocated = usize_to_u64(
            device.metal_device().current_allocated_size(),
            "Metal allocated memory",
        )?;
        MemoryPoolSnapshot::new(
            total,
            total.saturating_sub(allocated),
            MemoryDiscoverySource::MetalRecommendedWorkingSet,
            true,
        )
    }

    #[cfg(not(all(feature = "embedded-metal", target_os = "macos")))]
    fn discover_metal_memory(&self) -> Result<MemoryPoolSnapshot> {
        Err(PowerError::BackendNotAvailable(
            "Metal memory discovery requires a macOS embedded-metal build".to_string(),
        ))
    }
}

fn fractional_budget(
    available_bytes: u64,
    reserve_bytes: u64,
    fraction_bps: u16,
    cap_bytes: Option<u64>,
) -> u64 {
    let usable = available_bytes.saturating_sub(reserve_bytes);
    let target = (u128::from(usable) * u128::from(fraction_bps) / u128::from(BASIS_POINTS))
        .min(u128::from(u64::MAX)) as u64;
    cap_bytes.map_or(target, |cap| target.min(cap))
}

fn allocate_pair(
    host_target: u64,
    device_target: u64,
    total_limit: u64,
    order: ResidencyAllocationOrder,
) -> (u64, u64) {
    match order {
        ResidencyAllocationOrder::DeviceFirst => {
            let device = device_target.min(total_limit);
            let host = host_target.min(total_limit.saturating_sub(device));
            (host, device)
        }
        ResidencyAllocationOrder::HostFirst => {
            let host = host_target.min(total_limit);
            let device = device_target.min(total_limit.saturating_sub(host));
            (host, device)
        }
    }
}

#[cfg(any(
    feature = "embedded-cuda",
    all(feature = "embedded-metal", target_os = "macos")
))]
fn usize_to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| PowerError::BackendNotAvailable(format!("{label} does not fit in u64")))
}

#[cfg(target_os = "linux")]
fn discover_host_memory() -> Result<MemoryPoolSnapshot> {
    discover_linux_proc_memory().or_else(|_| discover_posix_sysconf_memory())
}

#[cfg(target_os = "linux")]
fn discover_linux_proc_memory() -> Result<MemoryPoolSnapshot> {
    use std::io::Read;

    const MAX_MEMINFO_BYTES: u64 = 64 * 1024;
    let file = std::fs::File::open("/proc/meminfo")?;
    let mut bytes = Vec::new();
    file.take(MAX_MEMINFO_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_MEMINFO_BYTES {
        return Err(PowerError::BackendNotAvailable(
            "/proc/meminfo exceeds the bounded parser limit".to_string(),
        ));
    }
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        PowerError::BackendNotAvailable(format!("/proc/meminfo is not UTF-8: {error}"))
    })?;
    let total = parse_meminfo_kib(text, "MemTotal")?;
    let available = parse_meminfo_kib(text, "MemAvailable")?;
    MemoryPoolSnapshot::new(
        total,
        available.min(total),
        MemoryDiscoverySource::LinuxProcMeminfo,
        false,
    )
}

#[cfg(any(target_os = "linux", test))]
fn parse_meminfo_kib(text: &str, field: &str) -> Result<u64> {
    let prefix = format!("{field}:");
    let line = text
        .lines()
        .find(|line| line.starts_with(&prefix))
        .ok_or_else(|| {
            PowerError::BackendNotAvailable(format!("/proc/meminfo is missing {field}"))
        })?;
    let mut values = line[prefix.len()..].split_ascii_whitespace();
    let kib = values
        .next()
        .ok_or_else(|| PowerError::BackendNotAvailable(format!("{field} has no value")))?
        .parse::<u64>()
        .map_err(|error| PowerError::BackendNotAvailable(format!("{field} is invalid: {error}")))?;
    if values.next() != Some("kB") || values.next().is_some() {
        return Err(PowerError::BackendNotAvailable(format!(
            "{field} must contain one kB value"
        )));
    }
    kib.checked_mul(1024)
        .ok_or_else(|| PowerError::BackendNotAvailable(format!("{field} byte count overflowed")))
}

#[cfg(target_os = "linux")]
fn discover_posix_sysconf_memory() -> Result<MemoryPoolSnapshot> {
    // SAFETY: sysconf has no pointer arguments and is called with supported
    // constants. Negative values are rejected before conversion.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    // SAFETY: same as above.
    let total_pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    // SAFETY: same as above.
    let available_pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
    if page_size <= 0 || total_pages <= 0 || available_pages < 0 {
        return Err(PowerError::BackendNotAvailable(
            "POSIX memory discovery returned invalid page counts".to_string(),
        ));
    }
    let page_size = u64::try_from(page_size).map_err(|_| {
        PowerError::BackendNotAvailable("POSIX page size does not fit in u64".to_string())
    })?;
    let total = u64::try_from(total_pages)
        .ok()
        .and_then(|pages| pages.checked_mul(page_size))
        .ok_or_else(|| {
            PowerError::BackendNotAvailable("POSIX total memory overflowed".to_string())
        })?;
    let available = u64::try_from(available_pages)
        .ok()
        .and_then(|pages| pages.checked_mul(page_size))
        .ok_or_else(|| {
            PowerError::BackendNotAvailable("POSIX available memory overflowed".to_string())
        })?;
    MemoryPoolSnapshot::new(
        total,
        available.min(total),
        MemoryDiscoverySource::PosixSysconf,
        false,
    )
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc keeps these stable Mach bindings in Power's existing TCB.
fn discover_host_memory() -> Result<MemoryPoolSnapshot> {
    use std::mem::MaybeUninit;

    let mut total = 0_u64;
    let mut total_size = std::mem::size_of::<u64>();
    // SAFETY: the NUL-terminated key is static, and the output pointer and
    // length point to a valid u64 allocation.
    let total_status = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut total_size,
            std::ptr::null_mut(),
            0,
        )
    };
    if total_status != 0 || total_size != std::mem::size_of::<u64>() || total == 0 {
        return Err(PowerError::BackendNotAvailable(format!(
            "failed to query macOS total memory: {}",
            std::io::Error::last_os_error()
        )));
    }

    let mut statistics = MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: the output buffer is a correctly sized vm_statistics64 value,
    // and count is initialized to the ABI-provided field count.
    let status = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            statistics.as_mut_ptr().cast(),
            &mut count,
        )
    };
    // SAFETY: sysconf has no pointer arguments and the result is checked.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let minimum_count = ((std::mem::offset_of!(libc::vm_statistics64, speculative_count)
        + std::mem::size_of::<libc::natural_t>())
        / std::mem::size_of::<libc::integer_t>())
        as libc::mach_msg_type_number_t;
    if status != libc::KERN_SUCCESS || count < minimum_count || page_size <= 0 {
        return Err(PowerError::BackendNotAvailable(format!(
            "failed to query macOS available memory (status {status}, fields {count}/{minimum_count}, page size {page_size})"
        )));
    }
    // SAFETY: host_statistics64 returned success for the full output count.
    let statistics = unsafe { statistics.assume_init() };
    let pages = u64::from(statistics.free_count)
        .saturating_add(u64::from(statistics.inactive_count))
        .saturating_add(u64::from(statistics.speculative_count))
        .saturating_add(u64::from(statistics.purgeable_count));
    let page_size = u64::try_from(page_size).map_err(|_| {
        PowerError::BackendNotAvailable("macOS page size does not fit in u64".to_string())
    })?;
    let available = pages.saturating_mul(page_size);
    MemoryPoolSnapshot::new(
        total,
        available.min(total),
        MemoryDiscoverySource::MachHostStatistics,
        false,
    )
}

#[cfg(target_os = "windows")]
fn discover_host_memory() -> Result<MemoryPoolSnapshot> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).map_err(|_| {
            PowerError::BackendNotAvailable(
                "Windows memory status size does not fit in u32".to_string(),
            )
        })?,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: status points to a valid MEMORYSTATUSEX with dwLength set to
    // the exact structure size required by GlobalMemoryStatusEx.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(PowerError::BackendNotAvailable(format!(
            "GlobalMemoryStatusEx failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    MemoryPoolSnapshot::new(
        status.ullTotalPhys,
        status.ullAvailPhys.min(status.ullTotalPhys),
        MemoryDiscoverySource::WindowsGlobalMemoryStatus,
        false,
    )
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn discover_host_memory() -> Result<MemoryPoolSnapshot> {
    Err(PowerError::BackendNotAvailable(
        "host memory discovery is supported on Linux, macOS, and Windows".to_string(),
    ))
}

#[cfg(test)]
mod tests;
