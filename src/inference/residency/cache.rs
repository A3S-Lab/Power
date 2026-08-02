use std::collections::{HashMap, HashSet};

use candle_core::Tensor;

use super::{ResidencyPolicy, WeightKey, WeightTier};
use crate::inference::telemetry::Telemetry;

pub(super) struct CacheState {
    clock: u64,
    host: TierCache,
    device: TierCache,
}

struct TierCache {
    entries: HashMap<WeightKey, CacheEntry>,
    bytes: u64,
}

struct CacheEntry {
    tensor: Tensor,
    bytes: u64,
    last_used: u64,
    pinned: bool,
}

pub(super) struct CacheLookup {
    pub(super) tensor: Tensor,
    pub(super) bytes: u64,
}

pub(super) struct CacheInsert {
    pub(super) key: WeightKey,
    pub(super) tensor: Tensor,
    pub(super) bytes: u64,
    pub(super) pinned: bool,
}

impl CacheState {
    pub(super) fn new() -> Self {
        Self {
            clock: 0,
            host: TierCache::new(),
            device: TierCache::new(),
        }
    }

    pub(super) fn get(&mut self, tier: WeightTier, key: &WeightKey) -> Option<CacheLookup> {
        self.clock = self.clock.saturating_add(1);
        let clock = self.clock;
        let entry = self.cache_mut(tier)?.entries.get_mut(key)?;
        entry.last_used = clock;
        Some(CacheLookup {
            tensor: entry.tensor.clone(),
            bytes: entry.bytes,
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

        self.clock = self.clock.saturating_add(1);
        let clock = self.clock;
        let Some(cache) = self.cache_mut(tier) else {
            return false;
        };
        if let Some(entry) = cache.entries.get_mut(&incoming.key) {
            entry.last_used = clock;
            entry.pinned |= incoming.pinned;
            return true;
        }

        let Some(evictions) =
            cache.evictions_for(&incoming.key, incoming.bytes, budget, max_entries)
        else {
            return false;
        };
        for eviction in evictions {
            if let Some(removed) = cache.entries.remove(&eviction) {
                cache.bytes = cache.bytes.saturating_sub(removed.bytes);
                match tier {
                    WeightTier::Host => telemetry.host_eviction(),
                    WeightTier::Device => telemetry.device_eviction(),
                    WeightTier::Storage => {}
                }
            }
        }
        cache.bytes = cache.bytes.saturating_add(incoming.bytes);
        cache.entries.insert(
            incoming.key,
            CacheEntry {
                tensor: incoming.tensor,
                bytes: incoming.bytes,
                last_used: clock,
                pinned: incoming.pinned,
            },
        );
        true
    }

    pub(super) fn set_pinned(&mut self, tier: WeightTier, key: &WeightKey, pinned: bool) -> bool {
        let Some(cache) = self.cache_mut(tier) else {
            return false;
        };
        let Some(entry) = cache.entries.get_mut(key) else {
            return false;
        };
        entry.pinned = pinned;
        true
    }

    pub(super) fn pin_state(&self, tier: WeightTier, key: &WeightKey) -> Option<bool> {
        self.cache(tier)?.entries.get(key).map(|entry| entry.pinned)
    }

    pub(super) fn restore_pin_state(
        &mut self,
        tier: WeightTier,
        key: &WeightKey,
        prior: Option<bool>,
        telemetry: &Telemetry,
    ) {
        let Some(cache) = self.cache_mut(tier) else {
            return;
        };
        match prior {
            Some(pinned) => {
                if let Some(entry) = cache.entries.get_mut(key) {
                    entry.pinned = pinned;
                }
            }
            None => {
                if let Some(entry) = cache.entries.remove(key) {
                    cache.bytes = cache.bytes.saturating_sub(entry.bytes);
                    match tier {
                        WeightTier::Host => telemetry.host_eviction(),
                        WeightTier::Device => telemetry.device_eviction(),
                        WeightTier::Storage => {}
                    }
                }
            }
        }
    }

    pub(super) fn clear_unpinned(&mut self, telemetry: &Telemetry) {
        self.host.clear_unpinned(WeightTier::Host, telemetry);
        self.device.clear_unpinned(WeightTier::Device, telemetry);
    }

    pub(super) fn resident_bytes(&self) -> (u64, u64) {
        (self.host.bytes, self.device.bytes)
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
    ) -> Option<Vec<WeightKey>> {
        let mut candidates = self
            .entries
            .iter()
            .filter(|(_, entry)| !entry.pinned)
            .map(|(key, entry)| (key.clone(), entry.last_used, entry.bytes))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(_, last_used, _)| *last_used);

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
            .filter(|(_, entry)| !entry.pinned)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in evictions {
            if let Some(entry) = self.entries.remove(&key) {
                self.bytes = self.bytes.saturating_sub(entry.bytes);
                match tier {
                    WeightTier::Host => telemetry.host_eviction(),
                    WeightTier::Device => telemetry.device_eviction(),
                    WeightTier::Storage => {}
                }
            }
        }
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
            pinned,
        }
    }

    #[test]
    fn lru_is_layer_local_before_global_budget_eviction() {
        let policy = ResidencyPolicy {
            host_cache_bytes: 8,
            device_cache_bytes: 0,
            max_entries_per_layer: 1,
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
        assert!(cache.get(WeightTier::Host, &key(0, "a")).is_none());
        assert!(cache.get(WeightTier::Host, &key(0, "c")).is_some());
        assert!(cache.get(WeightTier::Host, &key(1, "b")).is_some());
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
        assert!(cache.get(WeightTier::Host, &key(0, "pinned")).is_some());
    }
}
