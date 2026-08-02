use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

use crate::error::{PowerError, Result};

use super::{run_sample, StorageBenchmarkConfig, StorageCachePreparation, WeightStore};

pub(super) fn prepare_cache_state(
    config: &StorageBenchmarkConfig,
    store: &Arc<WeightStore>,
    names: &[String],
) -> Result<&'static str> {
    match config.cache_preparation {
        StorageCachePreparation::WarmSequence => {
            let _ = run_sample(Arc::clone(store), names, config.concurrency)?;
            Ok(config.cache_preparation.procedure())
        }
        StorageCachePreparation::LinuxFadviseDontNeed => {
            prepare_linux_cold_cache(store, names)?;
            Ok(config.cache_preparation.procedure())
        }
    }
}

#[cfg(target_os = "linux")]
fn prepare_linux_cold_cache(store: &WeightStore, names: &[String]) -> Result<()> {
    let mut ranges_by_path = BTreeMap::new();
    for range in store.verified_cache_ranges(names)? {
        ranges_by_path
            .entry(range.path)
            .or_insert_with(Vec::new)
            .push((range.absolute_offset, range.bytes));
    }
    let mut files = Vec::with_capacity(ranges_by_path.len());
    for (path, ranges) in ranges_by_path {
        let file = File::open(path)?;
        file.sync_all()?;
        // SAFETY: `file` owns a valid read-only descriptor for the duration of
        // this call. The preceding sync makes all file-backed pages eligible
        // for discard; `posix_fadvise` does not access Rust-managed memory.
        let result =
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if result != 0 {
            return Err(PowerError::Io(std::io::Error::from_raw_os_error(result)));
        }
        files.push((file, ranges));
    }

    for (file, ranges) in &files {
        for (offset, bytes) in ranges {
            if *bytes != 0 {
                verify_linux_range_not_resident(file, *offset, *bytes)?;
            }
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn prepare_linux_cold_cache(_store: &WeightStore, _names: &[String]) -> Result<()> {
    Err(PowerError::BackendNotAvailable(
        "verified cold page-cache preparation is currently supported only on Linux".to_string(),
    ))
}

#[cfg(target_os = "linux")]
fn verify_linux_range_not_resident(file: &File, offset: u64, bytes: u64) -> Result<()> {
    // SAFETY: `sysconf` has no pointer arguments or Rust aliasing requirements.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(PowerError::BackendNotAvailable(
            "Linux did not report a usable page size for cold-cache verification".to_string(),
        ));
    }
    let page_size = u64::try_from(page_size).map_err(|_| {
        PowerError::BackendNotAvailable("Linux page size exceeds the supported range".to_string())
    })?;
    let page_size_usize = usize::try_from(page_size).map_err(|_| {
        PowerError::BackendNotAvailable(
            "Linux page size exceeds the host address space".to_string(),
        )
    })?;
    let end = offset.checked_add(bytes).ok_or_else(|| {
        PowerError::InvalidFormat("cold-cache tensor range overflowed".to_string())
    })?;
    let aligned_offset = offset / page_size * page_size;
    let mapped_bytes = end.checked_sub(aligned_offset).ok_or_else(|| {
        PowerError::InvalidFormat("cold-cache mapped range underflowed".to_string())
    })?;
    let mapped_len = usize::try_from(mapped_bytes).map_err(|_| {
        PowerError::BackendNotAvailable(
            "cold-cache verification range exceeds the host address space".to_string(),
        )
    })?;
    let aligned_offset = libc::off_t::try_from(aligned_offset).map_err(|_| {
        PowerError::BackendNotAvailable(
            "cold-cache verification offset exceeds the platform file range".to_string(),
        )
    })?;

    // SAFETY: the offset is page-aligned, the non-zero length is bounded by a
    // previously validated tensor range, and `file` remains open until unmap.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mapped_len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            aligned_offset,
        )
    };
    if mapping == libc::MAP_FAILED {
        return Err(PowerError::Io(std::io::Error::last_os_error()));
    }
    let page_count = mapped_len
        .checked_add(page_size_usize.saturating_sub(1))
        .map(|value| value / page_size_usize)
        .ok_or_else(|| PowerError::InvalidFormat("cold-cache page count overflowed".to_string()))?;
    let mut residency = vec![0_u8; page_count];
    // SAFETY: `mapping` is valid for `mapped_len`, and `residency` has one byte
    // for every page intersecting that mapping as required by `mincore`.
    let mincore_result = unsafe { libc::mincore(mapping, mapped_len, residency.as_mut_ptr()) };
    let mincore_error = (mincore_result != 0).then(std::io::Error::last_os_error);
    // SAFETY: this unmaps the exact region returned by `mmap` above once.
    let unmap_result = unsafe { libc::munmap(mapping, mapped_len) };
    if let Some(error) = mincore_error {
        return Err(PowerError::Io(error));
    }
    if unmap_result != 0 {
        return Err(PowerError::Io(std::io::Error::last_os_error()));
    }
    if residency.iter().any(|page| page & 1 != 0) {
        return Err(PowerError::BackendNotAvailable(
            "Linux retained at least one requested weight page after POSIX_FADV_DONTNEED; refusing to label the run cold"
                .to_string(),
        ));
    }
    Ok(())
}
