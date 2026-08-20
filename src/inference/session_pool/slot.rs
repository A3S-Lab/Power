use std::sync::{Arc, Mutex};

use tokio::sync::OnceCell;

/// Replaceable initialization generation for one anonymous session slot.
///
/// The pool never exposes the slot or generation identity. A health decision
/// can replace the current cell only while its exclusive lease is held, and
/// the next lease initializes the replacement lazily.
pub(super) struct SessionSlot<T> {
    state: Mutex<SlotState<T>>,
}

struct SlotState<T> {
    cell: Arc<OnceCell<Arc<T>>>,
    reconstruction_pending: bool,
}

impl<T> SessionSlot<T> {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(SlotState {
                cell: Arc::new(OnceCell::new()),
                reconstruction_pending: false,
            }),
        }
    }

    pub(super) fn cell(&self) -> Arc<OnceCell<Arc<T>>> {
        Arc::clone(&lock(&self.state).cell)
    }

    pub(super) fn is_ready(&self) -> bool {
        lock(&self.state).cell.get().is_some()
    }

    pub(super) fn reconstruction_pending(&self) -> bool {
        lock(&self.state).reconstruction_pending
    }

    /// Replaces a ready generation and returns its detached state.
    pub(super) fn retire(&self) -> Option<Arc<OnceCell<Arc<T>>>> {
        let mut state = lock(&self.state);
        state.cell.get()?;
        let retired = std::mem::replace(&mut state.cell, Arc::new(OnceCell::new()));
        state.reconstruction_pending = true;
        Some(retired)
    }

    /// Completes the pending reconstruction only for the current generation.
    pub(super) fn finish_reconstruction(&self, cell: &Arc<OnceCell<Arc<T>>>) -> bool {
        let mut state = lock(&self.state);
        if !state.reconstruction_pending || !Arc::ptr_eq(&state.cell, cell) || cell.get().is_none()
        {
            return false;
        }
        state.reconstruction_pending = false;
        true
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
