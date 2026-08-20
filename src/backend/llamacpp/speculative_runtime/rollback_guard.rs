/// Request-local circuit breaker for fixed-width MTP configurations whose
/// proposal width is larger than the resident recurrent rollback window.
///
/// The first oversized rejection still uses the exact prefix replay fallback.
/// Afterwards the request stays within the resident window, bounding fallback
/// replay count without reducing the high-acceptance path's configured width.
#[derive(Debug, Clone, Copy)]
pub(super) struct RollbackReplayGuard {
    rollback_limit: usize,
    effective_max: usize,
    activated_after_round: Option<u64>,
}

impl RollbackReplayGuard {
    pub(super) fn new(configured_max: usize, rollback_limit: usize) -> Self {
        let configured_max = configured_max.max(1);
        Self {
            rollback_limit: rollback_limit.max(1).min(configured_max),
            effective_max: configured_max,
            activated_after_round: None,
        }
    }

    pub(super) fn draft_limit(&self) -> usize {
        self.effective_max
    }

    /// Observe the rejected suffix from one completed verification round.
    /// Returns `true` only when this observation closes the circuit.
    pub(super) fn observe_rejected_suffix(&mut self, rejected_suffix: usize, round: u64) -> bool {
        if rejected_suffix <= self.rollback_limit || self.effective_max <= self.rollback_limit {
            return false;
        }
        self.effective_max = self.rollback_limit;
        self.activated_after_round = Some(round);
        true
    }

    pub(super) fn activated_after_round(&self) -> Option<u64> {
        self.activated_after_round
    }
}

#[cfg(test)]
mod tests {
    use super::RollbackReplayGuard;

    #[test]
    fn keeps_configured_width_while_rejections_fit_the_rollback_window() {
        let mut guard = RollbackReplayGuard::new(7, 6);

        assert_eq!(guard.draft_limit(), 7);
        assert!(!guard.observe_rejected_suffix(6, 1));
        assert_eq!(guard.draft_limit(), 7);
        assert_eq!(guard.activated_after_round(), None);
    }

    #[test]
    fn permanently_clamps_after_the_first_replay_worthy_rejection() {
        let mut guard = RollbackReplayGuard::new(7, 6);

        assert!(guard.observe_rejected_suffix(7, 3));
        assert_eq!(guard.draft_limit(), 6);
        assert_eq!(guard.activated_after_round(), Some(3));

        assert!(!guard.observe_rejected_suffix(7, 4));
        assert_eq!(guard.draft_limit(), 6);
        assert_eq!(guard.activated_after_round(), Some(3));
    }

    #[test]
    fn complete_rollback_windows_never_narrow_the_configured_width() {
        let mut guard = RollbackReplayGuard::new(7, 7);

        assert!(!guard.observe_rejected_suffix(7, 1));
        assert_eq!(guard.draft_limit(), 7);
        assert_eq!(guard.activated_after_round(), None);
    }

    #[test]
    fn rollback_windows_wider_than_the_draft_do_not_expand_it() {
        let guard = RollbackReplayGuard::new(5, 7);

        assert_eq!(guard.draft_limit(), 5);
    }
}
