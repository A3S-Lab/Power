// Rust 1.98 deprecates Atomic::fetch_update, while the pinned 1.97 toolchain
// does not provide its replacement. Keep the established semantics on both.
#[macro_export]
macro_rules! atomic_fetch_update {
    ($atomic:expr, $set_order:expr, $fail_order:expr, $update:expr) => {{
        let mut current = $atomic.load($fail_order);
        loop {
            match $update(current) {
                Some(next) => {
                    match $atomic.compare_exchange_weak(current, next, $set_order, $fail_order) {
                        Ok(_) => break Ok(current),
                        Err(observed) => current = observed,
                    }
                }
                None => break Err(current),
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn atomic_update_commits_and_returns_the_previous_value() {
        let value = AtomicU64::new(4);
        assert_eq!(
            crate::atomic_fetch_update!(
                value,
                Ordering::AcqRel,
                Ordering::Acquire,
                |current: u64| { current.checked_add(3) }
            ),
            Ok(4)
        );
        assert_eq!(value.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn atomic_update_refuses_overflow_and_underflow_without_writing() {
        let value = AtomicU64::new(u64::MAX);
        assert_eq!(
            crate::atomic_fetch_update!(
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current: u64| { current.checked_add(1) }
            ),
            Err(u64::MAX)
        );
        assert_eq!(value.load(Ordering::Relaxed), u64::MAX);

        value.store(0, Ordering::Relaxed);
        assert_eq!(
            crate::atomic_fetch_update!(
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current: u64| { current.checked_sub(1) }
            ),
            Err(0)
        );
        assert_eq!(value.load(Ordering::Relaxed), 0);
    }
}
