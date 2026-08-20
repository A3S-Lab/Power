use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use a3s_power::inference::{HostAllocationCounter, HostAllocationSnapshot};

struct CountingAllocator;

static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static REALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

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
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated with the caller-provided valid layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: delegated with the same pointer/layout contract supplied by
        // the caller of this global allocator.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: delegated with the caller-provided allocation and new size.
        let resized = unsafe { System.realloc(pointer, layout, new_size) };
        if !resized.is_null() {
            REALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            REALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        resized
    }
}

pub(super) struct ProcessAllocationCounter;

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
