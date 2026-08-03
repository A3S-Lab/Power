use std::collections::{BTreeMap, VecDeque};

use crate::error::{PowerError, Result};

/// Shared item-and-byte admission for background weight workers.
///
/// Prefetch and current-layer staging use this same window so their parallel
/// reads cannot each enforce a different interpretation of the configured
/// in-flight bound.
pub(super) struct BackgroundLoadWindow {
    max_workers: usize,
    max_bytes: u64,
    workers: usize,
    bytes: u64,
    peak_workers: usize,
    peak_bytes: u64,
    outstanding_sizes: BTreeMap<u64, usize>,
}

impl BackgroundLoadWindow {
    pub(super) fn new(max_workers: usize, max_bytes: u64) -> Result<Self> {
        if max_workers == 0 || max_bytes == 0 {
            return Err(PowerError::Config(
                "background load worker and byte bounds must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            max_workers,
            max_bytes,
            workers: 0,
            bytes: 0,
            peak_workers: 0,
            peak_bytes: 0,
            outstanding_sizes: BTreeMap::new(),
        })
    }

    /// Takes the earliest pending item that fits the remaining byte window.
    ///
    /// Looking past a temporarily blocked large item keeps spare workers busy;
    /// its relative queue position is retained and it becomes eligible as
    /// earlier work releases bytes. Every individual item is required to fit
    /// the complete window, preventing an idle queue from deadlocking.
    pub(super) fn take_fitting<T, F>(
        &mut self,
        pending: &mut VecDeque<T>,
        bytes_of: F,
    ) -> Result<Option<T>>
    where
        F: Fn(&T) -> u64,
    {
        if self.workers >= self.max_workers || pending.is_empty() {
            return Ok(None);
        }
        let available = self.max_bytes.saturating_sub(self.bytes);
        let position = pending.iter().position(|item| bytes_of(item) <= available);
        let Some(position) = position else {
            if self.workers == 0 {
                return Err(PowerError::InvalidRequest(format!(
                    "one background weight load exceeds the {} byte in-flight limit",
                    self.max_bytes
                )));
            }
            return Ok(None);
        };
        let item = pending.remove(position).ok_or_else(|| {
            PowerError::InferenceFailed(
                "background load queue changed during bounded admission".to_string(),
            )
        })?;
        let bytes = bytes_of(&item);
        if bytes == 0 {
            return Err(PowerError::InvalidFormat(
                "background weight load has an empty canonical byte range".to_string(),
            ));
        }
        self.workers = self.workers.checked_add(1).ok_or_else(|| {
            PowerError::InferenceFailed("background worker count overflowed".to_string())
        })?;
        self.bytes = self.bytes.checked_add(bytes).ok_or_else(|| {
            PowerError::InferenceFailed("background in-flight byte count overflowed".to_string())
        })?;
        self.peak_workers = self.peak_workers.max(self.workers);
        self.peak_bytes = self.peak_bytes.max(self.bytes);
        let count = self.outstanding_sizes.entry(bytes).or_default();
        *count = count.checked_add(1).ok_or_else(|| {
            PowerError::InferenceFailed("background load size count overflowed".to_string())
        })?;
        Ok(Some(item))
    }

    pub(super) fn release(&mut self, bytes: u64) -> Result<()> {
        let Some(count) = self.outstanding_sizes.get_mut(&bytes) else {
            return Err(PowerError::InferenceFailed(
                "background load completion does not match its admitted window".to_string(),
            ));
        };
        *count -= 1;
        if *count == 0 {
            self.outstanding_sizes.remove(&bytes);
        }
        self.workers -= 1;
        self.bytes -= bytes;
        Ok(())
    }

    pub(super) fn is_idle(&self) -> bool {
        self.workers == 0
    }

    pub(super) fn peak_workers(&self) -> usize {
        self.peak_workers
    }

    pub(super) fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_uses_spare_capacity_without_losing_queue_order() {
        let mut window = BackgroundLoadWindow::new(2, 10).unwrap();
        let mut pending = VecDeque::from([8_u64, 7, 2]);

        assert_eq!(
            window.take_fitting(&mut pending, |bytes| *bytes).unwrap(),
            Some(8)
        );
        assert_eq!(
            window.take_fitting(&mut pending, |bytes| *bytes).unwrap(),
            Some(2)
        );
        assert_eq!(pending, VecDeque::from([7]));
        assert_eq!(window.peak_workers(), 2);
        assert_eq!(window.peak_bytes(), 10);

        window.release(2).unwrap();
        assert_eq!(
            window.take_fitting(&mut pending, |bytes| *bytes).unwrap(),
            None
        );
        window.release(8).unwrap();
        assert_eq!(
            window.take_fitting(&mut pending, |bytes| *bytes).unwrap(),
            Some(7)
        );
        window.release(7).unwrap();
        assert!(window.is_idle());
    }

    #[test]
    fn oversized_or_mismatched_work_fails_closed() {
        let mut window = BackgroundLoadWindow::new(1, 4).unwrap();
        let mut pending = VecDeque::from([5_u64]);
        assert!(window.take_fitting(&mut pending, |bytes| *bytes).is_err());

        let mut pending = VecDeque::from([4_u64]);
        assert_eq!(
            window.take_fitting(&mut pending, |bytes| *bytes).unwrap(),
            Some(4)
        );
        assert!(window.release(3).is_err());
        assert!(!window.is_idle());
        window.release(4).unwrap();
        assert!(window.is_idle());
    }
}
