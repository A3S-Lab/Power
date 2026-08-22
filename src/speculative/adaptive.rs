/// Model-neutral adaptive draft-length controller.
///
/// Tracks an exponential moving average of the per-round acceptance ratio and
/// grows the draft length when speculation is paying off, shrinks it when
/// drafts are mostly rejected (so a bad streak costs at most `min` wasted
/// verify slots rather than a fixed large block).
#[derive(Debug, Clone)]
pub struct AdaptiveK {
    k: usize,
    min: usize,
    max: usize,
    ema: f32,
}

impl AdaptiveK {
    /// EMA smoothing factor (higher = more reactive).
    const ALPHA: f32 = 0.3;
    /// Grow K above this acceptance ratio, shrink below `1 - GROW`.
    const GROW: f32 = 0.6;

    pub fn new(initial: usize, min: usize, max: usize) -> Self {
        let min = min.max(1);
        let max = max.max(min);
        Self {
            k: initial.clamp(min, max),
            min,
            max,
            ema: 0.5,
        }
    }

    /// Current draft length to propose.
    pub fn current(&self) -> usize {
        self.k
    }

    /// Update after a verify round: `accepted` of `drafted` tokens kept.
    pub fn update(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        let ratio = accepted as f32 / drafted as f32;
        self.ema = Self::ALPHA * ratio + (1.0 - Self::ALPHA) * self.ema;
        if self.ema > Self::GROW && self.k < self.max {
            self.k += 1;
        } else if self.ema < 1.0 - Self::GROW && self.k > self.min {
            self.k -= 1;
        }
    }
}

/// Request-local control loop for model-backed speculative decoding.
///
/// The controller sizes the next proposal from the observed accepted-prefix
/// length and opens a one-way target-only circuit after a sustained low-yield
/// window. A configured width may initially exceed the resident rollback
/// window: high-acceptance requests keep that faster path, while the first
/// oversized rejection is replayed exactly by the backend and clamps later
/// proposals to the resident window. The circuit is deliberately request-local:
/// a poor prompt must not poison later requests whose draft distribution may
/// be very different.
#[derive(Debug, Clone)]
pub struct AdaptiveSpeculationController {
    current: usize,
    min: usize,
    max: usize,
    accepted_prefix_ema: Option<f32>,
    rounds: u64,
    window: [SpeculationRound; Self::OBSERVATION_WINDOW],
    window_count: usize,
    window_cursor: usize,
    target_only_after_round: Option<u64>,
    rollback_limit: usize,
    rollback_guard_after_round: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SpeculationRound {
    accepted: usize,
    drafted: usize,
}

impl AdaptiveSpeculationController {
    const EMA_ALPHA: f32 = 0.35;
    const OBSERVATION_WINDOW: usize = 12;
    const MIN_ACCEPTED_PER_ROUND: f64 = 0.75;

    /// Create a controller with a bounded resident rollback window.
    ///
    /// The configured proposal maximum remains available until an observed
    /// rejected suffix actually exceeds `rollback_limit`. This avoids paying a
    /// permanent throughput penalty for a fallback path that high-acceptance
    /// requests never exercise.
    pub fn new(initial: usize, min: usize, max: usize, rollback_limit: usize) -> Self {
        let max = max.max(1);
        let min = min.max(1).min(max);
        let rollback_limit = rollback_limit.max(min).min(max);
        Self {
            current: initial.clamp(min, max),
            min,
            max,
            accepted_prefix_ema: None,
            rounds: 0,
            window: [SpeculationRound::default(); Self::OBSERVATION_WINDOW],
            window_count: 0,
            window_cursor: 0,
            target_only_after_round: None,
            rollback_limit,
            rollback_guard_after_round: None,
        }
    }

    /// Proposal width for the next round, or `None` after the circuit opens.
    pub fn draft_limit(&self) -> Option<usize> {
        self.target_only_after_round
            .is_none()
            .then_some(self.current)
    }

    /// Round on which speculation was disabled for this request.
    pub fn target_only_after_round(&self) -> Option<u64> {
        self.target_only_after_round
    }

    /// Current maximum proposal width after any rollback guard activation.
    pub fn effective_max(&self) -> usize {
        self.max
    }

    /// Round on which an oversized rejection first narrowed the proposal cap.
    pub fn rollback_guard_after_round(&self) -> Option<u64> {
        self.rollback_guard_after_round
    }

    /// Observe one verified proposal and update the next decision.
    pub fn observe(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 || self.target_only_after_round.is_some() {
            return;
        }
        let accepted = accepted.min(drafted);
        self.rounds = self.rounds.saturating_add(1);
        let rejected_suffix = drafted.saturating_sub(accepted);
        if rejected_suffix > self.rollback_limit && self.max > self.rollback_limit {
            self.max = self.rollback_limit;
            self.current = self.current.min(self.max);
            self.rollback_guard_after_round = Some(self.rounds);
        }
        self.window[self.window_cursor] = SpeculationRound { accepted, drafted };
        self.window_cursor = (self.window_cursor + 1) % Self::OBSERVATION_WINDOW;
        self.window_count = self
            .window_count
            .saturating_add(1)
            .min(Self::OBSERVATION_WINDOW);

        let previous_ema = self.accepted_prefix_ema.unwrap_or(accepted as f32);
        let ema = Self::EMA_ALPHA * accepted as f32 + (1.0 - Self::EMA_ALPHA) * previous_ema;
        self.accepted_prefix_ema = Some(ema);

        if accepted == 0 {
            self.current = self.current.div_ceil(2).max(self.min);
        } else if accepted == drafted {
            self.current = self.current.saturating_add(1).min(self.max);
        } else if accepted.saturating_mul(2) >= drafted {
            // A target pass that commits at least half of the proposal is
            // already well amortized. Treat this as a healthy partial round
            // and preserve the current graph shape instead of reacting to one
            // rejected tail. Low-yield rounds below this boundary still use
            // the accepted-prefix EMA to narrow quickly.
        } else {
            // One exploratory slot beyond the expected accepted prefix lets K
            // recover when the local distribution improves without repeatedly
            // paying for a long rejected tail.
            let desired = (ema.ceil() as usize)
                .saturating_add(1)
                .clamp(self.min, self.max);
            self.current = desired;
        }

        if self.window_count == Self::OBSERVATION_WINDOW {
            let observed = &self.window[..self.window_count];
            let accepted_total = observed.iter().map(|sample| sample.accepted).sum::<usize>();
            let drafted_total = observed.iter().map(|sample| sample.drafted).sum::<usize>();
            let zero_rounds = observed
                .iter()
                .filter(|sample| sample.accepted == 0)
                .count();
            let accepted_per_round = accepted_total as f64 / self.window_count as f64;
            let mostly_zero = zero_rounds * 2 >= self.window_count;
            if drafted_total > 0 && accepted_per_round < Self::MIN_ACCEPTED_PER_ROUND && mostly_zero
            {
                self.target_only_after_round = Some(self.rounds);
            }
        }
    }
}
