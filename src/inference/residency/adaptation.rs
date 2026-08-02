use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::planner::validate_candidate_id;
use super::{
    lock, read, write, ExecutionPermit, PlannedResidencyGroup, ResidencyApplyReport,
    ResidencyCandidate, ResidencyPlan, WeightHierarchy, WeightTier,
};
use crate::error::{PowerError, Result};

const MAX_ADAPTATION_REPLACEMENTS: usize = 4_096;

/// Bounded policy for lossless live hot-tier adaptation.
///
/// The defaults match Colibri's live re-pin guard: a challenger must beat the
/// incumbent by more than 25% plus four heat units, and one pass may replace
/// at most four atomic groups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResidencyAdaptationPolicy {
    pub max_replacements: usize,
    pub hysteresis_basis_points: u32,
    pub min_heat_gain: u64,
}

impl Default for ResidencyAdaptationPolicy {
    fn default() -> Self {
        Self {
            max_replacements: 4,
            hysteresis_basis_points: 2_500,
            min_heat_gain: 4,
        }
    }
}

impl ResidencyAdaptationPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.max_replacements == 0 || self.max_replacements > MAX_ADAPTATION_REPLACEMENTS {
            return Err(PowerError::Config(format!(
                "residency adaptation replacements must be within 1..={MAX_ADAPTATION_REPLACEMENTS}"
            )));
        }
        if self.hysteresis_basis_points > 10_000 {
            return Err(PowerError::Config(
                "residency adaptation hysteresis must not exceed 10,000 basis points".to_string(),
            ));
        }
        Ok(())
    }
}

/// One value-preserving exchange between compatible residency groups.
///
/// This contains workload-sensitive identities and is intentionally not
/// serializable. Power never logs or persists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyReplacement {
    pub demoted_id: String,
    pub promoted_id: String,
    pub demoted_from_tier: WeightTier,
    pub demoted_to_tier: WeightTier,
    pub promoted_from_tier: WeightTier,
    pub promoted_to_tier: WeightTier,
    pub incumbent_heat: u64,
    pub challenger_heat: u64,
    pub heat_gain: u64,
}

/// Ephemeral adaptation bound to the active plan from which it was derived.
///
/// The hidden base plan lets application reject stale work atomically. The
/// adaptation is intentionally not serializable because group identities can
/// reveal private routing behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyAdaptation {
    base_plan: ResidencyPlan,
    plan: ResidencyPlan,
    replacements: Vec<ResidencyReplacement>,
}

impl ResidencyAdaptation {
    pub fn plan(&self) -> &ResidencyPlan {
        &self.plan
    }

    pub fn replacements(&self) -> &[ResidencyReplacement] {
        &self.replacements
    }

    pub fn is_noop(&self) -> bool {
        self.replacements.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ResidencyFootprint {
    bytes: u64,
    entries_by_layer: Vec<(u32, usize)>,
}

impl ResidencyFootprint {
    fn from_group(group: &PlannedResidencyGroup) -> Self {
        let mut entries = BTreeMap::<u32, usize>::new();
        for weight in &group.weights {
            *entries.entry(weight.layer).or_default() += 1;
        }
        Self {
            bytes: group.bytes,
            entries_by_layer: entries.into_iter().collect(),
        }
    }
}

#[derive(Default)]
struct FootprintPool {
    hottest_storage: Option<usize>,
    hottest_host: Option<usize>,
    coldest_host: Option<usize>,
    coldest_device: Option<usize>,
}

#[derive(Clone, Copy)]
struct SwapChoice {
    demoted: usize,
    promoted: usize,
    gain: u64,
}

impl WeightHierarchy {
    /// Derives a bounded, stable hot-tier update from caller-owned live heat.
    ///
    /// Call this only at a model-defined safe request boundary. Candidate IDs
    /// and weights must exactly match the active plan; only their heat may
    /// change. Power exchanges groups only when their byte and per-layer entry
    /// footprints match, so every update preserves the existing tier ledgers.
    pub fn plan_residency_adaptation(
        &self,
        candidates: &[ResidencyCandidate],
        policy: &ResidencyAdaptationPolicy,
    ) -> Result<ResidencyAdaptation> {
        policy.validate()?;
        let _operation = read(&self.inner.operations);
        let base_plan = lock(&self.inner.active_plan).clone().ok_or_else(|| {
            PowerError::InvalidRequest(
                "residency adaptation requires an active residency plan".to_string(),
            )
        })?;
        self.validate_plan(&base_plan)?;
        let heat = validate_heat_update(&base_plan, candidates)?;
        let footprints = base_plan
            .groups
            .iter()
            .map(ResidencyFootprint::from_group)
            .collect::<Vec<_>>();
        let mut plan = base_plan.clone();
        let mut changed = BTreeSet::new();
        let mut replacements = Vec::new();

        for _ in 0..policy.max_replacements {
            let Some(choice) = pick_swap(&plan, &footprints, &heat, &changed, policy) else {
                break;
            };
            let demoted = &plan.groups[choice.demoted];
            let promoted = &plan.groups[choice.promoted];
            let demoted_id = demoted.id.clone();
            let promoted_id = promoted.id.clone();
            let demoted_from_tier = demoted.tier;
            let promoted_from_tier = promoted.tier;
            let incumbent_heat = heat[&demoted_id];
            let challenger_heat = heat[&promoted_id];

            plan.groups[choice.demoted].tier = promoted_from_tier;
            plan.groups[choice.promoted].tier = demoted_from_tier;
            changed.insert(demoted_id.clone());
            changed.insert(promoted_id.clone());
            replacements.push(ResidencyReplacement {
                demoted_id,
                promoted_id,
                demoted_from_tier,
                demoted_to_tier: promoted_from_tier,
                promoted_from_tier,
                promoted_to_tier: demoted_from_tier,
                incumbent_heat,
                challenger_heat,
                heat_gain: choice.gain,
            });
        }

        self.validate_plan(&plan)?;
        Ok(ResidencyAdaptation {
            base_plan,
            plan,
            replacements,
        })
    }

    /// Applies a live adaptation through the existing transactional cache and
    /// plan-pin path. A changed active plan causes a fail-closed stale error.
    pub fn apply_residency_adaptation(
        &self,
        adaptation: &ResidencyAdaptation,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<ResidencyApplyReport> {
        self.validate_plan(&adaptation.plan)?;
        self.validate_permit(permit)?;
        self.check_cancelled(cancellation)?;
        let _operation = write(&self.inner.operations);
        self.check_cancelled(cancellation)?;
        let active_matches = lock(&self.inner.active_plan).as_ref() == Some(&adaptation.base_plan);
        if !active_matches {
            return Err(PowerError::InvalidRequest(
                "residency adaptation was derived from a stale active plan".to_string(),
            ));
        }
        self.apply_residency_plan_locked(&adaptation.plan, permit, cancellation)
    }
}

fn validate_heat_update(
    base_plan: &ResidencyPlan,
    candidates: &[ResidencyCandidate],
) -> Result<BTreeMap<String, u64>> {
    if candidates.len() != base_plan.groups.len() {
        return Err(PowerError::InvalidRequest(
            "residency adaptation candidates must exactly match the active plan".to_string(),
        ));
    }
    let mut updates = BTreeMap::<String, (u64, Vec<super::WeightKey>)>::new();
    for candidate in candidates {
        validate_candidate_id(&candidate.id)?;
        let mut weights = candidate.weights.clone();
        weights.sort();
        if weights.is_empty()
            || updates
                .insert(candidate.id.clone(), (candidate.heat, weights))
                .is_some()
        {
            return Err(PowerError::InvalidRequest(
                "residency adaptation contains a duplicate or empty candidate".to_string(),
            ));
        }
    }

    let mut heat = BTreeMap::new();
    for group in &base_plan.groups {
        let Some((candidate_heat, candidate_weights)) = updates.remove(&group.id) else {
            return Err(PowerError::InvalidRequest(format!(
                "residency adaptation is missing group '{}'",
                group.id
            )));
        };
        let mut group_weights = group.weights.clone();
        group_weights.sort();
        if candidate_weights != group_weights {
            return Err(PowerError::InvalidRequest(format!(
                "residency adaptation changed the weights for group '{}'",
                group.id
            )));
        }
        heat.insert(group.id.clone(), candidate_heat);
    }
    if !updates.is_empty() {
        return Err(PowerError::InvalidRequest(
            "residency adaptation contains a group outside the active plan".to_string(),
        ));
    }
    Ok(heat)
}

fn pick_swap(
    plan: &ResidencyPlan,
    footprints: &[ResidencyFootprint],
    heat: &BTreeMap<String, u64>,
    changed: &BTreeSet<String>,
    policy: &ResidencyAdaptationPolicy,
) -> Option<SwapChoice> {
    let mut pools = BTreeMap::<&ResidencyFootprint, FootprintPool>::new();
    for (index, group) in plan.groups.iter().enumerate() {
        if changed.contains(&group.id) {
            continue;
        }
        let pool = pools.entry(&footprints[index]).or_default();
        match group.tier {
            WeightTier::Storage => update_hottest(&mut pool.hottest_storage, index, plan, heat),
            WeightTier::Host => {
                update_hottest(&mut pool.hottest_host, index, plan, heat);
                update_coldest(&mut pool.coldest_host, index, plan, heat);
            }
            WeightTier::Device => update_coldest(&mut pool.coldest_device, index, plan, heat),
        }
    }

    let mut best = None;
    for pool in pools.values() {
        if let (Some(demoted), Some(promoted)) = (pool.coldest_host, pool.hottest_storage) {
            consider_swap(&mut best, demoted, promoted, plan, heat, policy);
        }
        if let Some(demoted) = pool.coldest_device {
            let promoted = hottest_of(pool.hottest_host, pool.hottest_storage, plan, heat);
            if let Some(promoted) = promoted {
                consider_swap(&mut best, demoted, promoted, plan, heat, policy);
            }
        }
    }
    best
}

fn update_hottest(
    slot: &mut Option<usize>,
    candidate: usize,
    plan: &ResidencyPlan,
    heat: &BTreeMap<String, u64>,
) {
    if slot.is_none_or(|current| hotter(candidate, current, plan, heat)) {
        *slot = Some(candidate);
    }
}

fn update_coldest(
    slot: &mut Option<usize>,
    candidate: usize,
    plan: &ResidencyPlan,
    heat: &BTreeMap<String, u64>,
) {
    if slot.is_none_or(|current| colder(candidate, current, plan, heat)) {
        *slot = Some(candidate);
    }
}

fn hottest_of(
    left: Option<usize>,
    right: Option<usize>,
    plan: &ResidencyPlan,
    heat: &BTreeMap<String, u64>,
) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(if hotter(left, right, plan, heat) {
            left
        } else {
            right
        }),
        (Some(index), None) | (None, Some(index)) => Some(index),
        (None, None) => None,
    }
}

fn hotter(left: usize, right: usize, plan: &ResidencyPlan, heat: &BTreeMap<String, u64>) -> bool {
    compare_heat(left, right, plan, heat) == Ordering::Greater
}

fn colder(left: usize, right: usize, plan: &ResidencyPlan, heat: &BTreeMap<String, u64>) -> bool {
    compare_heat(left, right, plan, heat) == Ordering::Less
}

fn compare_heat(
    left: usize,
    right: usize,
    plan: &ResidencyPlan,
    heat: &BTreeMap<String, u64>,
) -> Ordering {
    let left_group = &plan.groups[left];
    let right_group = &plan.groups[right];
    heat[&left_group.id]
        .cmp(&heat[&right_group.id])
        .then_with(|| right_group.id.cmp(&left_group.id))
}

fn consider_swap(
    best: &mut Option<SwapChoice>,
    demoted: usize,
    promoted: usize,
    plan: &ResidencyPlan,
    heat: &BTreeMap<String, u64>,
    policy: &ResidencyAdaptationPolicy,
) {
    let incumbent_heat = heat[&plan.groups[demoted].id];
    let challenger_heat = heat[&plan.groups[promoted].id];
    let relative_gain = basis_points(incumbent_heat, policy.hysteresis_basis_points);
    let threshold = incumbent_heat
        .saturating_add(relative_gain)
        .saturating_add(policy.min_heat_gain);
    if challenger_heat <= threshold {
        return;
    }
    let candidate = SwapChoice {
        demoted,
        promoted,
        gain: challenger_heat - incumbent_heat,
    };
    if best.is_none_or(|current| swap_is_better(candidate, current, plan)) {
        *best = Some(candidate);
    }
}

fn basis_points(value: u64, points: u32) -> u64 {
    let points = u64::from(points);
    (value / 10_000)
        .saturating_mul(points)
        .saturating_add((value % 10_000).saturating_mul(points) / 10_000)
}

fn swap_is_better(left: SwapChoice, right: SwapChoice, plan: &ResidencyPlan) -> bool {
    left.gain
        .cmp(&right.gain)
        .then_with(|| {
            tier_rank(plan.groups[left.demoted].tier)
                .cmp(&tier_rank(plan.groups[right.demoted].tier))
        })
        .then_with(|| {
            plan.groups[right.promoted]
                .id
                .cmp(&plan.groups[left.promoted].id)
        })
        .then_with(|| {
            plan.groups[right.demoted]
                .id
                .cmp(&plan.groups[left.demoted].id)
        })
        == Ordering::Greater
}

fn tier_rank(tier: WeightTier) -> u8 {
    match tier {
        WeightTier::Storage => 0,
        WeightTier::Host => 1,
        WeightTier::Device => 2,
    }
}
