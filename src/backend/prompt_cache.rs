use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Prefix-cache behavior implemented by an inference backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromptCacheSupport {
    /// The backend cannot guarantee prompt-prefix KV reuse.
    #[default]
    Unsupported,
    /// The backend reuses the longest matching token prefix for an opaque key.
    PrefixMatch,
}

impl PromptCacheSupport {
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// Cumulative, process-local prefix-cache telemetry for one backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheMetricsSnapshot {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub reused_tokens: u64,
    pub evaluated_tokens: u64,
    pub evictions: u64,
    pub entries: u64,
    /// Sum of configured entry bounds for currently live model caches.
    pub capacity: u64,
}

/// Lock-free counters shared by all loaded models of one backend.
#[derive(Debug, Default)]
#[cfg_attr(not(feature = "llamacpp"), allow(dead_code))]
pub(crate) struct PromptCacheTelemetry {
    requests: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    reused_tokens: AtomicU64,
    evaluated_tokens: AtomicU64,
    evictions: AtomicU64,
    entries: AtomicU64,
    capacity: AtomicU64,
}

#[cfg_attr(not(feature = "llamacpp"), allow(dead_code))]
impl PromptCacheTelemetry {
    pub(crate) fn record_lookup(&self, reused_tokens: usize, evaluated_tokens: usize) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        if reused_tokens > 0 {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        self.reused_tokens
            .fetch_add(reused_tokens as u64, Ordering::Relaxed);
        self.evaluated_tokens
            .fetch_add(evaluated_tokens as u64, Ordering::Relaxed);
    }

    fn add_entries(&self, count: usize) {
        self.entries.fetch_add(count as u64, Ordering::Relaxed);
    }

    fn remove_entries(&self, count: usize) {
        let count = count as u64;
        let _ = self
            .entries
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |entries| {
                Some(entries.saturating_sub(count))
            });
    }

    fn add_capacity(&self, count: usize) {
        self.capacity.fetch_add(count as u64, Ordering::Relaxed);
    }

    fn remove_capacity(&self, count: usize) {
        let count = count as u64;
        let _ = self
            .capacity
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |capacity| {
                Some(capacity.saturating_sub(count))
            });
    }

    fn record_evictions(&self, count: usize) {
        self.evictions.fetch_add(count as u64, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> PromptCacheMetricsSnapshot {
        PromptCacheMetricsSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            reused_tokens: self.reused_tokens.load(Ordering::Relaxed),
            evaluated_tokens: self.evaluated_tokens.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            entries: self.entries.load(Ordering::Relaxed),
            capacity: self.capacity.load(Ordering::Relaxed),
        }
    }
}

struct CacheEntry<T> {
    value: T,
    last_used: Instant,
}

/// TTL- and capacity-bounded storage for expensive backend KV contexts.
///
/// Removal transfers ownership to an in-flight request. Re-insertion refreshes
/// LRU age. The shared entry gauge therefore reflects resident reusable
/// contexts, not contexts currently executing.
pub(crate) struct BoundedPromptCache<T> {
    entries: HashMap<String, CacheEntry<T>>,
    max_entries: usize,
    ttl: Duration,
    telemetry: Arc<PromptCacheTelemetry>,
}

#[cfg_attr(not(feature = "llamacpp"), allow(dead_code))]
impl<T> BoundedPromptCache<T> {
    pub(crate) fn new(
        max_entries: usize,
        ttl: Duration,
        telemetry: Arc<PromptCacheTelemetry>,
    ) -> Self {
        telemetry.add_capacity(max_entries);
        Self {
            entries: HashMap::new(),
            max_entries,
            ttl,
            telemetry,
        }
    }

    pub(crate) fn take(&mut self, key: &str) -> Option<T> {
        self.take_at(key, Instant::now())
    }

    fn take_at(&mut self, key: &str, now: Instant) -> Option<T> {
        self.evict_expired_at(now);
        let entry = self.entries.remove(key)?;
        self.telemetry.remove_entries(1);
        Some(entry.value)
    }

    pub(crate) fn insert(&mut self, key: String, value: T) {
        self.insert_at(key, value, Instant::now());
    }

    fn insert_at(&mut self, key: String, value: T, now: Instant) {
        self.evict_expired_at(now);

        if let Some(entry) = self.entries.get_mut(&key) {
            *entry = CacheEntry {
                value,
                last_used: now,
            };
            return;
        }

        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            {
                self.entries.remove(&oldest);
                self.telemetry.remove_entries(1);
                self.telemetry.record_evictions(1);
            }
        }

        self.entries.insert(
            key,
            CacheEntry {
                value,
                last_used: now,
            },
        );
        self.telemetry.add_entries(1);
    }

    fn evict_expired_at(&mut self, now: Instant) {
        let before = self.entries.len();
        self.entries.retain(|_, entry| {
            now.checked_duration_since(entry.last_used)
                .is_none_or(|age| age < self.ttl)
        });
        let removed = before.saturating_sub(self.entries.len());
        if removed > 0 {
            self.telemetry.remove_entries(removed);
            self.telemetry.record_evictions(removed);
        }
    }
}

impl<T> Drop for BoundedPromptCache<T> {
    fn drop(&mut self) {
        self.telemetry.remove_entries(self.entries.len());
        self.telemetry.remove_capacity(self.max_entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_cache_evicts_oldest_entry_at_capacity() {
        let telemetry = Arc::new(PromptCacheTelemetry::default());
        let mut cache = BoundedPromptCache::new(2, Duration::from_secs(60), telemetry.clone());
        let now = Instant::now();
        cache.insert_at("a".into(), 1, now);
        cache.insert_at("b".into(), 2, now + Duration::from_millis(1));
        cache.insert_at("c".into(), 3, now + Duration::from_millis(2));

        assert_eq!(cache.take_at("a", now + Duration::from_millis(3)), None);
        assert_eq!(cache.take_at("b", now + Duration::from_millis(3)), Some(2));
        assert_eq!(cache.take_at("c", now + Duration::from_millis(3)), Some(3));
        assert_eq!(telemetry.snapshot().evictions, 1);
        assert_eq!(telemetry.snapshot().entries, 0);
    }

    #[test]
    fn bounded_cache_expires_idle_entries() {
        let telemetry = Arc::new(PromptCacheTelemetry::default());
        let mut cache = BoundedPromptCache::new(1, Duration::from_secs(5), telemetry.clone());
        let now = Instant::now();
        cache.insert_at("a".into(), 1, now);

        assert_eq!(cache.take_at("a", now + Duration::from_secs(5)), None);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.evictions, 1);
        assert_eq!(snapshot.entries, 0);
    }

    #[test]
    fn telemetry_distinguishes_prefix_hits_and_misses() {
        let telemetry = PromptCacheTelemetry::default();
        telemetry.record_lookup(128, 16);
        telemetry.record_lookup(0, 32);

        assert_eq!(
            telemetry.snapshot(),
            PromptCacheMetricsSnapshot {
                requests: 2,
                hits: 1,
                misses: 1,
                reused_tokens: 128,
                evaluated_tokens: 48,
                evictions: 0,
                entries: 0,
                capacity: 0,
            }
        );
    }

    #[test]
    fn telemetry_tracks_live_cache_capacity() {
        let telemetry = Arc::new(PromptCacheTelemetry::default());
        let first = BoundedPromptCache::<u8>::new(2, Duration::from_secs(60), telemetry.clone());
        assert_eq!(telemetry.snapshot().capacity, 2);
        {
            let _second =
                BoundedPromptCache::<u8>::new(3, Duration::from_secs(60), telemetry.clone());
            assert_eq!(telemetry.snapshot().capacity, 5);
        }
        assert_eq!(telemetry.snapshot().capacity, 2);
        drop(first);
        assert_eq!(telemetry.snapshot().capacity, 0);
    }
}
