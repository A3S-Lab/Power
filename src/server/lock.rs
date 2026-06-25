use std::sync::{Mutex, MutexGuard, RwLock};

/// Read from a potentially poisoned RwLock, recovering from poison.
///
/// Short-held std::sync::RwLock is correct here (no await across lock),
/// but we must handle poison to prevent cascade crashes.
pub(crate) fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| {
        tracing::warn!("RwLock was poisoned, recovering read guard");
        poisoned.into_inner()
    })
}

/// Write to a potentially poisoned RwLock, recovering from poison.
pub(crate) fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|poisoned| {
        tracing::warn!("RwLock was poisoned, recovering write guard");
        poisoned.into_inner()
    })
}

/// Lock a potentially poisoned Mutex, recovering from poison.
pub(crate) fn mutex_lock<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("Mutex was poisoned, recovering guard");
        poisoned.into_inner()
    })
}
