use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::{
    lock, write, ExecutionPermit, PlacementPreference, WeightHierarchy, WeightKey, WeightRequest,
    WeightTier,
};
use crate::error::{PowerError, Result};
use crate::inference::RuntimeDeviceKind;

/// One indivisible placement unit supplied by a model crate.
///
/// A routed expert normally contributes one candidate containing all of its
/// projections. Power ranks the group by observed heat but never splits it,
/// substitutes it, or interprets the model-specific tensor names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidencyCandidate {
    pub id: String,
    pub heat: u64,
    pub weights: Vec<WeightKey>,
}

impl ResidencyCandidate {
    pub fn new(id: impl Into<String>, heat: u64, weights: Vec<WeightKey>) -> Self {
        Self {
            id: id.into(),
            heat,
            weights,
        }
    }
}

/// Deterministic placement for one atomic candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannedResidencyGroup {
    pub id: String,
    pub tier: WeightTier,
    pub bytes: u64,
    pub weights: Vec<WeightKey>,
}

/// Model- and policy-bound hot-weight placement plan.
///
/// Placement reveals workload characteristics. Power returns this value to the
/// caller but never logs or persists it automatically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidencyPlan {
    pub schema: String,
    pub weights_sha256: String,
    pub runtime_device: String,
    pub host_budget_bytes: u64,
    pub device_budget_bytes: u64,
    pub max_entries_per_layer: usize,
    pub host_planned_bytes: u64,
    pub device_planned_bytes: u64,
    pub streaming_planned_bytes: u64,
    pub groups: Vec<PlannedResidencyGroup>,
}

impl ResidencyPlan {
    pub const SCHEMA: &'static str = "a3s.power.weight-residency-plan.v1";

    /// Flattens the plan into explicit model-neutral weight requests.
    pub fn requests(&self) -> Vec<WeightRequest> {
        self.groups
            .iter()
            .flat_map(|group| {
                let placement = match group.tier {
                    WeightTier::Storage => PlacementPreference::Streaming,
                    WeightTier::Host => PlacementPreference::Host,
                    WeightTier::Device => PlacementPreference::Device,
                };
                group
                    .weights
                    .iter()
                    .cloned()
                    .map(move |key| WeightRequest::new(key, placement))
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidencyApplyReport {
    pub groups_pinned: usize,
    pub weights_pinned: usize,
    pub host_bytes: u64,
    pub device_bytes: u64,
}

struct RankedCandidate {
    id: String,
    heat: u64,
    bytes: u64,
    weights: Vec<WeightKey>,
}

#[derive(Default)]
struct TierLedger {
    bytes: u64,
    entries_by_layer: BTreeMap<u32, usize>,
}

impl TierLedger {
    fn can_fit(&self, candidate: &RankedCandidate, budget: u64, max_entries: usize) -> bool {
        if self.bytes.saturating_add(candidate.bytes) > budget {
            return false;
        }
        let mut additions = BTreeMap::<u32, usize>::new();
        for weight in &candidate.weights {
            *additions.entry(weight.layer).or_default() += 1;
        }
        additions.into_iter().all(|(layer, count)| {
            self.entries_by_layer
                .get(&layer)
                .copied()
                .unwrap_or_default()
                .saturating_add(count)
                <= max_entries
        })
    }

    fn add(&mut self, candidate: &RankedCandidate) {
        self.bytes = self.bytes.saturating_add(candidate.bytes);
        for weight in &candidate.weights {
            let count = self.entries_by_layer.entry(weight.layer).or_default();
            *count = count.saturating_add(1);
        }
    }
}

impl WeightHierarchy {
    /// Builds a deterministic hot-store plan for model-supplied atomic groups.
    ///
    /// Higher-heat groups are offered to the device tier first, then host RAM;
    /// groups that do not fit remain exact storage-streamed weights. Equal heat
    /// is resolved by stable candidate ID and weight keys.
    pub fn plan_residency(&self, candidates: &[ResidencyCandidate]) -> Result<ResidencyPlan> {
        let mut ids = BTreeSet::new();
        let mut tensor_names = BTreeSet::new();
        let mut ranked = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            validate_candidate_id(&candidate.id)?;
            if !ids.insert(candidate.id.clone()) {
                return Err(PowerError::InvalidRequest(format!(
                    "residency candidate '{}' is declared more than once",
                    candidate.id
                )));
            }
            if candidate.weights.is_empty() {
                return Err(PowerError::InvalidRequest(format!(
                    "residency candidate '{}' contains no weights",
                    candidate.id
                )));
            }
            let mut weights = candidate.weights.clone();
            weights.sort();
            let mut bytes = 0_u64;
            for key in &weights {
                let descriptor = self.validate_request(&WeightRequest::new(
                    key.clone(),
                    PlacementPreference::Streaming,
                ))?;
                if !tensor_names.insert(key.name.clone()) {
                    return Err(PowerError::InvalidRequest(format!(
                        "tensor '{}' appears in more than one residency candidate",
                        key.name
                    )));
                }
                bytes = bytes.checked_add(descriptor.bytes).ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "residency candidate byte length overflowed".to_string(),
                    )
                })?;
            }
            ranked.push(RankedCandidate {
                id: candidate.id.clone(),
                heat: candidate.heat,
                bytes,
                weights,
            });
        }
        ranked.sort_by(|left, right| {
            right
                .heat
                .cmp(&left.heat)
                .then_with(|| left.id.cmp(&right.id))
                .then_with(|| left.weights.cmp(&right.weights))
        });

        let policy = self.policy();
        let device_budget = if self.runtime().device().kind() == RuntimeDeviceKind::Cpu {
            0
        } else {
            policy.device_cache_bytes
        };
        let mut host = TierLedger::default();
        let mut device = TierLedger::default();
        let mut streaming_bytes = 0_u64;
        let mut groups = Vec::with_capacity(ranked.len());
        for candidate in ranked {
            let tier = if device_budget > 0
                && device.can_fit(&candidate, device_budget, policy.max_entries_per_layer)
            {
                device.add(&candidate);
                WeightTier::Device
            } else if policy.host_cache_bytes > 0
                && host.can_fit(
                    &candidate,
                    policy.host_cache_bytes,
                    policy.max_entries_per_layer,
                )
            {
                host.add(&candidate);
                WeightTier::Host
            } else {
                streaming_bytes = streaming_bytes.saturating_add(candidate.bytes);
                WeightTier::Storage
            };
            groups.push(PlannedResidencyGroup {
                id: candidate.id,
                tier,
                bytes: candidate.bytes,
                weights: candidate.weights,
            });
        }

        Ok(ResidencyPlan {
            schema: ResidencyPlan::SCHEMA.to_string(),
            weights_sha256: self.store().sha256().to_string(),
            runtime_device: self.runtime().device().name().to_string(),
            host_budget_bytes: policy.host_cache_bytes,
            device_budget_bytes: device_budget,
            max_entries_per_layer: policy.max_entries_per_layer,
            host_planned_bytes: host.bytes,
            device_planned_bytes: device.bytes,
            streaming_planned_bytes: streaming_bytes,
            groups,
        })
    }

    /// Pins every resident group in a validated plan.
    ///
    /// On failure, cache entries and pin flags touched by this operation are
    /// restored to their prior state. Storage groups are intentionally skipped.
    pub fn apply_residency_plan(
        &self,
        plan: &ResidencyPlan,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<ResidencyApplyReport> {
        self.validate_plan(plan)?;
        self.validate_permit(permit)?;
        self.check_cancelled(cancellation)?;
        let _operation = write(&self.inner.operations);
        self.check_cancelled(cancellation)?;

        let mut prior = BTreeMap::<(WeightTier, WeightKey), Option<bool>>::new();
        for group in &plan.groups {
            for key in &group.weights {
                match group.tier {
                    WeightTier::Storage => {}
                    WeightTier::Host => self.capture_pin_state(&mut prior, WeightTier::Host, key),
                    WeightTier::Device => {
                        // Device promotion can populate both tiers.
                        self.capture_pin_state(&mut prior, WeightTier::Host, key);
                        self.capture_pin_state(&mut prior, WeightTier::Device, key);
                    }
                }
            }
        }

        let mut report = ResidencyApplyReport {
            groups_pinned: 0,
            weights_pinned: 0,
            host_bytes: 0,
            device_bytes: 0,
        };
        let applied = (|| -> Result<()> {
            for group in &plan.groups {
                let placement = match group.tier {
                    WeightTier::Storage => continue,
                    WeightTier::Host => PlacementPreference::Host,
                    WeightTier::Device => PlacementPreference::Device,
                };
                for key in &group.weights {
                    self.load_internal(
                        &WeightRequest::new(key.clone(), placement),
                        permit,
                        cancellation,
                        true,
                    )?;
                    report.weights_pinned = report.weights_pinned.saturating_add(1);
                }
                report.groups_pinned = report.groups_pinned.saturating_add(1);
                match group.tier {
                    WeightTier::Host => {
                        report.host_bytes = report.host_bytes.saturating_add(group.bytes)
                    }
                    WeightTier::Device => {
                        report.device_bytes = report.device_bytes.saturating_add(group.bytes)
                    }
                    WeightTier::Storage => {}
                }
            }
            Ok(())
        })();
        if let Err(error) = applied {
            let mut cache = lock(&self.inner.cache);
            for ((tier, key), state) in prior.into_iter().rev() {
                cache.restore_pin_state(tier, &key, state, &self.inner.telemetry);
            }
            return Err(error);
        }
        Ok(report)
    }

    fn capture_pin_state(
        &self,
        states: &mut BTreeMap<(WeightTier, WeightKey), Option<bool>>,
        tier: WeightTier,
        key: &WeightKey,
    ) {
        states.entry((tier, key.clone())).or_insert_with(|| {
            let cache = lock(&self.inner.cache);
            cache.pin_state(tier, key)
        });
    }

    fn validate_plan(&self, plan: &ResidencyPlan) -> Result<()> {
        let policy = self.policy();
        let device_budget = if self.runtime().device().kind() == RuntimeDeviceKind::Cpu {
            0
        } else {
            policy.device_cache_bytes
        };
        if plan.schema != ResidencyPlan::SCHEMA
            || plan.weights_sha256 != self.store().sha256()
            || plan.runtime_device != self.runtime().device().name()
            || plan.host_budget_bytes != policy.host_cache_bytes
            || plan.device_budget_bytes != device_budget
            || plan.max_entries_per_layer != policy.max_entries_per_layer
        {
            return Err(PowerError::InvalidFormat(
                "residency plan does not match this weight hierarchy and policy".to_string(),
            ));
        }

        let mut names = BTreeSet::new();
        let mut ids = BTreeSet::new();
        let mut host = TierLedger::default();
        let mut device = TierLedger::default();
        let mut streaming_bytes = 0_u64;
        for group in &plan.groups {
            validate_candidate_id(&group.id)?;
            if !ids.insert(group.id.clone()) || group.weights.is_empty() {
                return Err(PowerError::InvalidFormat(
                    "residency plan contains a duplicate or empty group".to_string(),
                ));
            }
            let mut bytes = 0_u64;
            for key in &group.weights {
                let descriptor = self.validate_request(&WeightRequest::new(
                    key.clone(),
                    PlacementPreference::Streaming,
                ))?;
                if !names.insert(key.name.clone()) {
                    return Err(PowerError::InvalidFormat(
                        "residency plan references a tensor more than once".to_string(),
                    ));
                }
                bytes = bytes.checked_add(descriptor.bytes).ok_or_else(|| {
                    PowerError::InvalidFormat("residency plan byte length overflowed".to_string())
                })?;
            }
            if bytes != group.bytes {
                return Err(PowerError::InvalidFormat(format!(
                    "residency group '{}' has an invalid byte length",
                    group.id
                )));
            }
            let ranked = RankedCandidate {
                id: group.id.clone(),
                heat: 0,
                bytes,
                weights: group.weights.clone(),
            };
            match group.tier {
                WeightTier::Storage => {
                    streaming_bytes = streaming_bytes.saturating_add(bytes);
                }
                WeightTier::Host => {
                    if policy.host_cache_bytes == 0
                        || !host.can_fit(
                            &ranked,
                            policy.host_cache_bytes,
                            policy.max_entries_per_layer,
                        )
                    {
                        return Err(PowerError::InvalidFormat(
                            "residency plan exceeds host cache bounds".to_string(),
                        ));
                    }
                    host.add(&ranked);
                }
                WeightTier::Device => {
                    if device_budget == 0
                        || !device.can_fit(&ranked, device_budget, policy.max_entries_per_layer)
                    {
                        return Err(PowerError::InvalidFormat(
                            "residency plan exceeds device cache bounds".to_string(),
                        ));
                    }
                    device.add(&ranked);
                }
            }
        }
        if host.bytes != plan.host_planned_bytes
            || device.bytes != plan.device_planned_bytes
            || streaming_bytes != plan.streaming_planned_bytes
        {
            return Err(PowerError::InvalidFormat(
                "residency plan totals are inconsistent".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_candidate_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 1024 || id.chars().any(char::is_control) {
        return Err(PowerError::InvalidRequest(
            "residency candidate contains an invalid ID".to_string(),
        ));
    }
    Ok(())
}
