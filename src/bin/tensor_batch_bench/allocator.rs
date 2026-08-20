use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use a3s_power::inference::{HostAllocationCounter, HostAllocationSnapshot};

struct CountingAllocator;

static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static REALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation delegates to `System` with the original pointer and
// layout contract. Atomic bookkeeping neither dereferences nor changes the
// allocation, and counters are updated only after a successful allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated with the caller-provided valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            increase_live_bytes(layout.size() as u64);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated with the caller-provided valid layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            increase_live_bytes(layout.size() as u64);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: delegated with the same pointer/layout contract supplied by
        // the caller of this global allocator.
        unsafe { System.dealloc(pointer, layout) };
        decrease_live_bytes(layout.size() as u64);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: delegated with the caller-provided allocation and new size.
        let resized = unsafe { System.realloc(pointer, layout, new_size) };
        if !resized.is_null() {
            REALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            REALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            let old_size = layout.size() as u64;
            let new_size = new_size as u64;
            if new_size >= old_size {
                increase_live_bytes(new_size - old_size);
            } else {
                decrease_live_bytes(old_size - new_size);
            }
        }
        resized
    }
}

pub(super) struct ProcessAllocationCounter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ProcessLiveAllocationObservation {
    pub(super) baseline_bytes: u64,
    pub(super) peak_bytes: u64,
    pub(super) final_bytes: u64,
}

impl ProcessAllocationCounter {
    /// Starts one isolated-process live-byte observation.
    ///
    /// The benchmark runner is single-purpose, so resetting the peak does not
    /// hide another caller's observation. Allocations from backend worker
    /// threads remain visible because the allocator and counters are global.
    pub(super) fn begin_live_observation() -> u64 {
        let baseline = LIVE_BYTES.load(Ordering::Acquire);
        PEAK_LIVE_BYTES.store(baseline, Ordering::Release);
        baseline
    }

    pub(super) fn finish_live_observation(baseline_bytes: u64) -> ProcessLiveAllocationObservation {
        ProcessLiveAllocationObservation {
            baseline_bytes,
            peak_bytes: PEAK_LIVE_BYTES.load(Ordering::Acquire),
            final_bytes: LIVE_BYTES.load(Ordering::Acquire),
        }
    }
}

impl HostAllocationCounter for ProcessAllocationCounter {
    fn snapshot(&self) -> HostAllocationSnapshot {
        HostAllocationSnapshot {
            allocation_count: ALLOCATION_COUNT.load(Ordering::Relaxed),
            allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            reallocation_count: REALLOCATION_COUNT.load(Ordering::Relaxed),
            reallocated_bytes: REALLOCATED_BYTES.load(Ordering::Relaxed),
        }
    }
}

fn increase_live_bytes(bytes: u64) {
    if bytes == 0 {
        return;
    }
    let previous = LIVE_BYTES
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            Some(current.saturating_add(bytes))
        })
        .unwrap_or_else(|current| current);
    PEAK_LIVE_BYTES.fetch_max(previous.saturating_add(bytes), Ordering::Relaxed);
}

fn decrease_live_bytes(bytes: u64) {
    if bytes == 0 {
        return;
    }
    let _ = LIVE_BYTES.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(bytes))
    });
}
