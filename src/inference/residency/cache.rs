use std::collections::{HashMap, HashSet};

use candle_core::Tensor;

use super::{CacheEvictionPolicy, ResidencyPolicy, WeightKey, WeightTier};
use crate::inference::telemetry::Telemetry;

#[derive(Clone)]
pub(super) struct CacheState {
    clock: u64,
    host: TierCache,
    device: TierCache,
}

#[derive(Clone)]
struct TierCache {
    entries: HashMap<WeightKey, CacheEntry>,
    bytes: u64,
}

#[derive(Clone)]
struct CacheEntry {
    tensor: Tensor,
    bytes: u64,
    last_used: u64,
    heat: u64,
    pins: PinState,
    prefetch_pending: bool,
}

pub(super) struct CacheLookup {
    pub(super) tensor: Tensor,
    pub(super) bytes: u64,
    pub(super) prefetch_useful: bool,
}

pub(super) struct CacheInsert {
    pub(super) key: WeightKey,
    pub(super) tensor: Tensor,
    pub(super) bytes: u64,
    pub(super) pin: Option<PinReason>,
    pub(super) access: CacheAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CacheAccess {
    Demand,
    Prefetch,
    /// Intermediate promotion into a faster tier. It is neither demand heat
    /// for this tier nor a reusable prefetch result for usefulness accounting.
    Staging,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PinReason {
    Manual,
    Plan,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PinState {
    manual: bool,
    plan: bool,
}

impl PinState {
    fn is_pinned(self) -> bool {
        self.manual || self.plan
    }

    fn set(&mut self, reason: PinReason, value: bool) {
        match reason {
            PinReason::Manual => self.manual = value,
            PinReason::Plan => self.plan = value,
        }
    }
}

impl CacheState {
    pub(super) fn new() -> Self {
        Self {
            clock: 0,
            host: TierCache::new(),
            device: TierCache::new(),
        }
    }

    pub(super) fn get(
        &mut self,
        tier: WeightTier,
        key: &WeightKey,
        access: CacheAccess,
        policy: &ResidencyPolicy,
    ) -> Option<CacheLookup> {
        if !self.cache(tier)?.entries.contains_key(key) {
            return None;
        }
        let clock = match access {
            CacheAccess::Demand => self.advance_clock(policy),
            CacheAccess::Prefetch | CacheAccess::Staging => self.clock,
        };
        let entry = self.cache_mut(tier)?.entries.get_mut(key)?;
        let prefetch_useful = access == CacheAccess::Demand && entry.prefetch_pending;
        if access == CacheAccess::Demand {
            entry.last_used = clock;
            entry.heat = entry.heat.saturating_add(1);
            entry.prefetch_pending = false;
        }
        Some(CacheLookup {
            tensor: entry.tensor.clone(),
            bytes: entry.bytes,
            prefetch_useful,
        })
    }

    pub(super) fn peek(&self, tier: WeightTier, key: &WeightKey) -> Option<CacheLookup> {
        let entry = self.cache(tier)?.entries.get(key)?;
        Some(CacheLookup {
            tensor: entry.tensor.clone(),
            bytes: entry.bytes,
            prefetch_useful: false,
        })
    }

    pub(super) fn insert(
        &mut self,
        tier: WeightTier,
        incoming: CacheInsert,
        policy: &ResidencyPolicy,
        telemetry: &Telemetry,
    ) -> bool {
        let (budget, max_entries) = match tier {
            WeightTier::Host => (policy.host_cache_bytes, policy.max_entries_per_layer),
            WeightTier::Device => (policy.device_cache_bytes, policy.max_entries_per_layer),
            WeightTier::Storage => return false,
        };
        if budget == 0 || incoming.bytes > budget {
            return false;
        }

        let clock = match incoming.access {
            CacheAccess::Demand => self.advance_clock(policy),
            CacheAccess::Prefetch | CacheAccess::Staging => self.clock,
        };
        let Some(cache) = self.cache_mut(tier) else {
            return false;
        };
        if let Some(entry) = cache.entries.get_mut(&incoming.key) {
            if incoming.access == CacheAccess::Demand {
                entry.last_used = clock;
                entry.heat = entry.heat.saturating_add(1);
                entry.prefetch_pending = false;
            }
            if let Some(reason) = incoming.pin {
                entry.pins.set(reason, true);
            }
            return true;
        }

        let Some(evictions) = cache.evictions_for(
            &incoming.key,
            incoming.bytes,
            budget,
            max_entries,
            policy.cache_eviction,
            clock,
        ) else {
            return false;
        };
        for eviction in evictions {
            if let Some(removed) = cache.entries.remove(&eviction) {
                cache.bytes = cache.bytes.saturating_sub(removed.bytes);
                record_eviction(tier, &removed, telemetry);
            }
        }
        let mut pins = PinState::default();
        if let Some(reason) = incoming.pin {
            pins.set(reason, true);
        }
        cache.bytes = cache.bytes.saturating_add(incoming.bytes);
        cache.entries.insert(
            incoming.key,
            CacheEntry {
                tensor: incoming.tensor,
                bytes: incoming.bytes,
                last_used: clock,
                heat: u64::from(incoming.access == CacheAccess::Demand),
                pins,
                prefetch_pending: incoming.access == CacheAccess::Prefetch,
            },
        );
        true
    }

    pub(super) fn set_pin(
        &mut self,
        tier: WeightTier,
        key: &WeightKey,
        reason: PinReason,
        pinned: bool,
    ) -> bool {
        let Some(cache) = self.cache_mut(tier) else {
            return false;
        };
        let Some(entry) = cache.entries.get_mut(key) else {
            return false;
        };
        entry.pins.set(reason, pinned);
        true
    }

    pub(super) fn clear_plan_pins(&mut self) {
        self.host.clear_pin_reason(PinReason::Plan);
        self.device.clear_pin_reason(PinReason::Plan);
    }

    pub(super) fn clear_unpinned(&mut self, telemetry: &Telemetry) {
        self.host.clear_unpinned(WeightTier::Host, telemetry);
        self.device.clear_unpinned(WeightTier::Device, telemetry);
    }

    pub(super) fn resident_bytes(&self) -> (u64, u64) {
        (self.host.bytes, self.device.bytes)
    }

    fn advance_clock(&mut self, policy: &ResidencyPolicy) -> u64 {
        self.clock = self.clock.saturating_add(1);
        if policy.cache_eviction == CacheEvictionPolicy::Lfru
            && self.clock.is_multiple_of(policy.cache_heat_decay_interval)
        {
            self.host.decay_heat();
            self.device.decay_heat();
        }
        self.clock
    }

    fn cache_mut(&mut self, tier: WeightTier) -> Option<&mut TierCache> {
        match tier {
            WeightTier::Storage => None,
            WeightTier::Host => Some(&mut self.host),
            WeightTier::Device => Some(&mut self.device),
        }
    }

    fn cache(&self, tier: WeightTier) -> Option<&TierCache> {
        match tier {
            WeightTier::Storage => None,
            WeightTier::Host => Some(&self.host),
            WeightTier::Device => Some(&self.device),
        }
    }
}

impl TierCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
        }
    }

    fn evictions_for(
        &self,
        key: &WeightKey,
        incoming_bytes: u64,
        budget: u64,
        max_entries_per_layer: usize,
        policy: CacheEvictionPolicy,
        clock: u64,
    ) -> Option<Vec<WeightKey>> {
        let mut candidates = self
            .entries
            .iter()
            .filter(|(_, entry)| !entry.pins.is_pinned())
            .map(|(key, entry)| {
                (
                    key.clone(),
                    eviction_score(entry, policy, clock),
                    entry.bytes,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)));

        let layer_entries = self
            .entries
            .keys()
            .filter(|existing| existing.layer == key.layer)
            .count();
        let required_layer_evictions = layer_entries
            .saturating_add(1)
            .saturating_sub(max_entries_per_layer);
        let mut selected = Vec::new();
        let mut selected_keys = HashSet::new();
        for (candidate, _, _) in candidates
            .iter()
            .filter(|(candidate, _, _)| candidate.layer == key.layer)
            .take(required_layer_evictions)
        {
            selected.push(candidate.clone());
            selected_keys.insert(candidate.clone());
        }
        if selected.len() < required_layer_evictions {
            return None;
        }

        let mut remaining_bytes = self.bytes.saturating_add(incoming_bytes);
        for selected_key in &selected {
            remaining_bytes = remaining_bytes.saturating_sub(self.entries[selected_key].bytes);
        }
        if remaining_bytes > budget {
            for (candidate, _, bytes) in &candidates {
                if selected_keys.insert(candidate.clone()) {
                    selected.push(candidate.clone());
                    remaining_bytes = remaining_bytes.saturating_sub(*bytes);
                    if remaining_bytes <= budget {
                        break;
                    }
                }
            }
        }
        (remaining_bytes <= budget).then_some(selected)
    }

    fn clear_unpinned(&mut self, tier: WeightTier, telemetry: &Telemetry) {
        let evictions = self
            .entries
            .iter()
            .filter(|(_, entry)| !entry.pins.is_pinned())
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in evictions {
            if let Some(entry) = self.entries.remove(&key) {
                self.bytes = self.bytes.saturating_sub(entry.bytes);
                record_eviction(tier, &entry, telemetry);
            }
        }
    }

    fn clear_pin_reason(&mut self, reason: PinReason) {
        for entry in self.entries.values_mut() {
            entry.pins.set(reason, false);
        }
    }

    fn decay_heat(&mut self) {
        for entry in self.entries.values_mut() {
            entry.heat >>= 1;
        }
    }
}

fn eviction_score(entry: &CacheEntry, policy: CacheEvictionPolicy, clock: u64) -> u64 {
    match policy {
        CacheEvictionPolicy::Lru => entry.last_used,
        CacheEvictionPolicy::Lfru => {
            let age = clock.saturating_sub(entry.last_used);
            let recency = 255_u64.saturating_sub(age.min(255));
            entry.heat.saturating_mul(256).saturating_add(recency)
        }
    }
}

fn record_eviction(tier: WeightTier, entry: &CacheEntry, telemetry: &Telemetry) {
    match tier {
        WeightTier::Host => telemetry.host_eviction(),
        WeightTier::Device => telemetry.device_eviction(),
        WeightTier::Storage => {}
    }
    if entry.prefetch_pending {
        telemetry.prefetch_unused(entry.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::TelemetryMode;

    fn tensor() -> Tensor {
        Tensor::new(&[1_f32], &candle_core::Device::Cpu).unwrap()
    }

    fn key(layer: u32, name: &str) -> WeightKey {
        WeightKey::new(layer, name)
    }

    fn incoming(layer: u32, name: &str, bytes: u64, pinned: bool) -> CacheInsert {
        CacheInsert {
            key: key(layer, name),
            tensor: tensor(),
            bytes,
            pin: pinned.then_some(PinReason::Manual),
            access: CacheAccess::Demand,
        }
    }

    #[test]
    fn lru_is_layer_local_before_global_budget_eviction() {
        let policy = ResidencyPolicy {
            host_cache_bytes: 8,
            device_cache_bytes: 0,
            max_entries_per_layer: 1,
            cache_eviction: CacheEvictionPolicy::Lru,
            ..ResidencyPolicy::default()
        };
        let telemetry = Telemetry::new(TelemetryMode::Aggregate);
        let mut cache = CacheState::new();
        assert!(cache.insert(
            WeightTier::Host,
            incoming(0, "a", 4, false),
            &policy,
            &telemetry,
        ));
        assert!(cache.insert(
            WeightTier::Host,
            incoming(1, "b", 4, false),
            &policy,
            &telemetry,
        ));
        assert!(cache.insert(
            WeightTier::Host,
            incoming(0, "c", 4, false),
            &policy,
            &telemetry,
        ));
        assert!(cache
            .get(WeightTier::Host, &key(0, "a"), CacheAccess::Demand, &policy)
            .is_none());
        assert!(cache
            .get(WeightTier::Host, &key(0, "c"), CacheAccess::Demand, &policy)
            .is_some());
        assert!(cache
            .get(WeightTier::Host, &key(1, "b"), CacheAccess::Demand, &policy)
            .is_some());
    }

    #[test]
    fn pinned_entries_are_not_evicted_or_allowed_to_break_bounds() {
        let policy = ResidencyPolicy {
            host_cache_bytes: 4,
            device_cache_bytes: 0,
            max_entries_per_layer: 1,
            ..ResidencyPolicy::default()
        };
        let telemetry = Telemetry::new(TelemetryMode::Disabled);
        let mut cache = CacheState::new();
        assert!(cache.insert(
            WeightTier::Host,
            incoming(0, "pinned", 4, true),
            &policy,
            &telemetry,
        ));
        assert!(!cache.insert(
            WeightTier::Host,
            incoming(0, "new", 4, false),
            &policy,
            &telemetry,
        ));
        assert!(cache
            .get(
                WeightTier::Host,
                &key(0, "pinned"),
                CacheAccess::Demand,
                &policy,
            )
            .is_some());
    }

    #[test]
    fn lfru_retains_frequency_over_a_single_recent_access() {
        let policy = ResidencyPolicy {
            host_cache_bytes: 8,
            device_cache_bytes: 0,
            max_entries_per_layer: 2,
            cache_eviction: CacheEvictionPolicy::Lfru,
            ..ResidencyPolicy::default()
        };
        let telemetry = Telemetry::new(TelemetryMode::Aggregate);
        let mut cache = CacheState::new();
        assert!(cache.insert(
            WeightTier::Host,
            incoming(0, "hot", 4, false),
            &policy,
            &telemetry,
        ));
        assert!(cache.insert(
            WeightTier::Host,
            incoming(0, "recent", 4, false),
            &policy,
            &telemetry,
        ));
        for _ in 0..3 {
            cache
                .get(
                    WeightTier::Host,
                    &key(0, "hot"),
                    CacheAccess::Demand,
                    &policy,
                )
                .unwrap();
        }
        cache
            .get(
                WeightTier::Host,
                &key(0, "recent"),
                CacheAccess::Demand,
                &policy,
            )
            .unwrap();
        assert!(cache.insert(
            WeightTier::Host,
            incoming(0, "new", 4, false),
            &policy,
            &telemetry,
        ));

        assert!(cache
            .get(
                WeightTier::Host,
                &key(0, "hot"),
                CacheAccess::Demand,
                &policy,
            )
            .is_some());
        assert!(cache
            .get(
                WeightTier::Host,
                &key(0, "recent"),
                CacheAccess::Demand,
                &policy,
            )
            .is_none());
    }

    #[test]
    fn demand_marks_prefetched_entry_useful() {
        let policy = ResidencyPolicy {
            host_cache_bytes: 4,
            ..ResidencyPolicy::default()
        };
        let telemetry = Telemetry::new(TelemetryMode::Aggregate);
        let mut cache = CacheState::new();
        let mut prefetched = incoming(0, "prefetched", 4, false);
        prefetched.access = CacheAccess::Prefetch;
        assert!(cache.insert(WeightTier::Host, prefetched, &policy, &telemetry));

        let lookup = cache
            .get(
                WeightTier::Host,
                &key(0, "prefetched"),
                CacheAccess::Demand,
                &policy,
            )
            .unwrap();
        assert!(lookup.prefetch_useful);
    }

    #[test]
    fn unused_prefetch_is_measured_when_evicted() {
        let policy = ResidencyPolicy {
            host_cache_bytes: 4,
            max_entries_per_layer: 1,
            ..ResidencyPolicy::default()
        };
        let telemetry = Telemetry::new(TelemetryMode::Aggregate);
        let mut cache = CacheState::new();
        let mut prefetched = incoming(0, "prefetched", 4, false);
        prefetched.access = CacheAccess::Prefetch;
        assert!(cache.insert(WeightTier::Host, prefetched, &policy, &telemetry));
        assert!(cache.insert(
            WeightTier::Host,
            incoming(0, "demand", 4, false),
            &policy,
            &telemetry,
        ));

        let snapshot = telemetry.snapshot(4, 0);
        assert_eq!(snapshot.prefetch_unused_weights, 1);
        assert_eq!(snapshot.prefetch_unused_bytes, 4);
    }
}
