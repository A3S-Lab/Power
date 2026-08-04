use std::collections::BTreeSet;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::super::sealed_state::decode_sha256;
use super::super::{
    ExecutionBatchBinding, HardwareMemorySnapshot, InferenceLimits, RuntimeDeviceIdentity,
};
use super::device_name;
use super::types::{
    MicrobatchCandidate, MicrobatchLimits, MicrobatchPlan, MicrobatchPolicy, PlannedMicrobatch,
    PlannedMicrobatchSlot, BASIS_POINTS,
};

pub(in crate::inference) fn plan(
    runtime_device: RuntimeDeviceIdentity,
    snapshot: HardwareMemorySnapshot,
    binding: ExecutionBatchBinding,
    session_declaration_sha256: Option<String>,
    limits: InferenceLimits,
    policy: MicrobatchPolicy,
    candidates: Vec<MicrobatchCandidate>,
) -> Result<MicrobatchPlan> {
    limits.validate()?;
    calculate(
        runtime_device,
        snapshot,
        binding,
        session_declaration_sha256,
        MicrobatchLimits::from(&limits),
        policy,
        candidates,
    )
}

pub(super) fn validate_plan(plan: &MicrobatchPlan) -> Result<()> {
    if plan.schema != MicrobatchPlan::SCHEMA {
        return Err(PowerError::InvalidRequest(
            "microbatch plan schema is unsupported".to_string(),
        ));
    }
    let mut candidates = Vec::with_capacity(plan.slot_count);
    let mut expected_source = 0_usize;
    for (batch_index, batch) in plan.batches.iter().enumerate() {
        if batch.index != batch_index || batch.slots.is_empty() {
            return Err(PowerError::InvalidRequest(
                "microbatch plan indices or batch membership are invalid".to_string(),
            ));
        }
        for slot in &batch.slots {
            if slot.source_index != expected_source {
                return Err(PowerError::InvalidRequest(
                    "microbatch plan does not preserve contiguous source order".to_string(),
                ));
            }
            expected_source = expected_source.checked_add(1).ok_or_else(|| {
                PowerError::InvalidRequest("microbatch source index overflowed".to_string())
            })?;
            candidates.push(slot.candidate());
        }
    }
    if expected_source != plan.slot_count {
        return Err(PowerError::InvalidRequest(
            "microbatch plan slot count does not match its batches".to_string(),
        ));
    }
    let expected = calculate(
        plan.runtime_device,
        plan.snapshot.clone(),
        plan.binding.clone(),
        plan.session_declaration_sha256.clone(),
        plan.limits.clone(),
        plan.policy.clone(),
        candidates,
    )?;
    if expected != *plan {
        return Err(PowerError::InvalidRequest(
            "microbatch plan does not match its canonical derivation".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn revalidate_pressure(
    plan: &MicrobatchPlan,
    current: &HardwareMemorySnapshot,
) -> Result<()> {
    validate_plan(plan)?;
    current.validate()?;
    validate_topology(&plan.snapshot, current)?;
    let budget = memory_budget(current, &plan.policy)?;
    for batch in &plan.batches {
        let totals = Totals {
            items: batch.slots.len(),
            input_bytes: batch.input_bytes,
            input_elements: batch.input_elements,
            state_bytes: batch.state_bytes,
            host_peak_bytes: batch.host_peak_bytes,
            device_peak_bytes: batch.device_peak_bytes,
        };
        if !fits(&totals, &plan.limits, &plan.policy, &budget) {
            return Err(PowerError::InferenceFailed(
                "current memory pressure no longer supports the planned microbatch".to_string(),
            ));
        }
    }
    Ok(())
}

fn calculate(
    runtime_device: RuntimeDeviceIdentity,
    snapshot: HardwareMemorySnapshot,
    binding: ExecutionBatchBinding,
    session_declaration_sha256: Option<String>,
    limits: MicrobatchLimits,
    policy: MicrobatchPolicy,
    candidates: Vec<MicrobatchCandidate>,
) -> Result<MicrobatchPlan> {
    runtime_device.validate()?;
    snapshot.validate()?;
    binding.validate()?;
    policy.validate()?;
    validate_limits(&limits)?;
    if snapshot.runtime_device != device_name(runtime_device) {
        return Err(PowerError::InvalidRequest(
            "microbatch memory snapshot belongs to a different runtime device".to_string(),
        ));
    }
    if let Some(declaration) = &session_declaration_sha256 {
        decode_sha256(declaration, "microbatch model session declaration")?;
    }
    if candidates.is_empty() || candidates.len() > limits.max_graph_nodes {
        return Err(PowerError::InvalidRequest(format!(
            "microbatch planning requires 1 through {} candidates",
            limits.max_graph_nodes
        )));
    }
    let budget = memory_budget(&snapshot, &policy)?;
    let mut seen = BTreeSet::new();
    let mut batches = Vec::new();
    let mut current = BatchBuilder::new(0);
    for (source_index, candidate) in candidates.iter().enumerate() {
        validate_candidate(candidate, &limits, &snapshot, &budget)?;
        if !seen.insert(candidate.slot_sha256.as_str()) {
            return Err(PowerError::InvalidRequest(
                "microbatch slot identities must be unique".to_string(),
            ));
        }
        let proposed = current.totals.checked_add(candidate)?;
        if !current.slots.is_empty() && !fits(&proposed, &limits, &policy, &budget) {
            batches.push(current.finish());
            current = BatchBuilder::new(batches.len());
        }
        let proposed = current.totals.checked_add(candidate)?;
        if !fits(&proposed, &limits, &policy, &budget) {
            return Err(PowerError::InvalidRequest(
                "one microbatch candidate cannot fit the configured runtime and memory bounds"
                    .to_string(),
            ));
        }
        current.push(source_index, candidate.clone(), proposed);
    }
    if !current.slots.is_empty() {
        batches.push(current.finish());
    }

    let mut plan = MicrobatchPlan {
        schema: MicrobatchPlan::SCHEMA.to_string(),
        declaration_sha256: String::new(),
        binding,
        session_declaration_sha256,
        runtime_device,
        snapshot,
        limits,
        policy,
        host_budget_bytes: budget.host,
        device_budget_bytes: budget.device,
        shared_budget_bytes: budget.shared,
        slot_count: candidates.len(),
        batches,
    };
    plan.declaration_sha256 = declaration_sha256(&plan)?;
    Ok(plan)
}

fn validate_limits(limits: &MicrobatchLimits) -> Result<()> {
    if limits.max_concurrent_requests == 0
        || limits.max_input_bytes == 0
        || limits.max_tensor_elements == 0
        || limits.max_state_bytes == 0
        || limits.max_graph_nodes == 0
    {
        return Err(PowerError::InvalidRequest(
            "microbatch runtime limits must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_candidate(
    candidate: &MicrobatchCandidate,
    limits: &MicrobatchLimits,
    snapshot: &HardwareMemorySnapshot,
    budget: &MemoryBudget,
) -> Result<()> {
    candidate.validate_intrinsic()?;
    if candidate.input_bytes > limits.max_input_bytes
        || candidate.input_elements > limits.max_tensor_elements
        || candidate.state_bytes > limits.max_state_bytes
    {
        return Err(PowerError::InvalidRequest(
            "microbatch candidate exceeds a per-execution runtime limit".to_string(),
        ));
    }
    if snapshot.device.is_none() && candidate.device_peak_bytes != 0 {
        return Err(PowerError::InvalidRequest(
            "CPU microbatch candidates cannot declare device memory".to_string(),
        ));
    }
    let totals = Totals::default().checked_add(candidate)?;
    if totals.host_peak_bytes > budget.host
        || totals.device_peak_bytes > budget.device
        || budget.shared.is_some_and(|shared| {
            totals
                .host_peak_bytes
                .checked_add(totals.device_peak_bytes)
                .is_none_or(|total| total > shared)
        })
    {
        return Err(PowerError::InvalidRequest(
            "microbatch candidate exceeds the memory budget".to_string(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct MemoryBudget {
    host: u64,
    device: u64,
    shared: Option<u64>,
}

fn memory_budget(
    snapshot: &HardwareMemorySnapshot,
    policy: &MicrobatchPolicy,
) -> Result<MemoryBudget> {
    policy.validate()?;
    let host = incremental_budget(
        snapshot.host.available_bytes,
        policy.host_reserve_bytes,
        policy.base_host_bytes,
        policy.host_available_fraction_bps,
        "host",
    )?;
    let device_budget = match &snapshot.device {
        Some(device_pool) => incremental_budget(
            device_pool.available_bytes,
            policy.device_reserve_bytes,
            policy.base_device_bytes,
            policy.device_available_fraction_bps,
            "device",
        )?,
        None => {
            if policy.device_reserve_bytes != 0 || policy.base_device_bytes != 0 {
                return Err(PowerError::Config(
                    "device microbatch reservations require an accelerator memory pool".to_string(),
                ));
            }
            0
        }
    };
    let shared = snapshot
        .device
        .as_ref()
        .filter(|device_pool| device_pool.unified_with_host)
        .map(|device_pool| {
            let available = snapshot
                .host
                .available_bytes
                .min(device_pool.available_bytes);
            let reserved = policy
                .host_reserve_bytes
                .checked_add(policy.device_reserve_bytes)
                .and_then(|bytes| bytes.checked_add(policy.base_host_bytes))
                .and_then(|bytes| bytes.checked_add(policy.base_device_bytes))
                .ok_or_else(|| {
                    PowerError::Config("unified microbatch reservation overflowed".to_string())
                })?;
            if reserved > available {
                return Err(PowerError::InferenceFailed(
                    "unified microbatch reservations exceed current memory availability"
                        .to_string(),
                ));
            }
            Ok(available
                .saturating_sub(reserved)
                .min(host.saturating_add(device_budget)))
        })
        .transpose()?;
    Ok(MemoryBudget {
        host,
        device: device_budget,
        shared,
    })
}

fn incremental_budget(
    available: u64,
    reserve: u64,
    base: u64,
    fraction_bps: u16,
    label: &str,
) -> Result<u64> {
    let reserved = reserve
        .checked_add(base)
        .ok_or_else(|| PowerError::Config(format!("{label} microbatch reservation overflowed")))?;
    if reserved > available {
        return Err(PowerError::InferenceFailed(format!(
            "{label} microbatch reservations exceed current memory availability"
        )));
    }
    let fractional = (u128::from(available.saturating_sub(reserve)) * u128::from(fraction_bps)
        / u128::from(BASIS_POINTS))
    .min(u128::from(u64::MAX)) as u64;
    if base > fractional {
        return Err(PowerError::InferenceFailed(format!(
            "{label} microbatch base memory exceeds the configured available-memory fraction"
        )));
    }
    Ok(fractional.saturating_sub(base))
}

fn fits(
    totals: &Totals,
    limits: &MicrobatchLimits,
    policy: &MicrobatchPolicy,
    budget: &MemoryBudget,
) -> bool {
    totals.items <= policy.max_batch_items
        && totals.input_bytes <= limits.max_input_bytes
        && totals.input_elements <= limits.max_tensor_elements
        && totals.state_bytes <= limits.max_state_bytes
        && totals.host_peak_bytes <= budget.host
        && totals.device_peak_bytes <= budget.device
        && budget.shared.is_none_or(|shared| {
            totals
                .host_peak_bytes
                .checked_add(totals.device_peak_bytes)
                .is_some_and(|total| total <= shared)
        })
}

#[derive(Clone, Copy, Default)]
struct Totals {
    items: usize,
    input_bytes: usize,
    input_elements: usize,
    state_bytes: u64,
    host_peak_bytes: u64,
    device_peak_bytes: u64,
}

impl Totals {
    fn checked_add(self, candidate: &MicrobatchCandidate) -> Result<Self> {
        Ok(Self {
            items: self.items.checked_add(1).ok_or_else(overflow)?,
            input_bytes: self
                .input_bytes
                .checked_add(candidate.input_bytes)
                .ok_or_else(overflow)?,
            input_elements: self
                .input_elements
                .checked_add(candidate.input_elements)
                .ok_or_else(overflow)?,
            state_bytes: self
                .state_bytes
                .checked_add(candidate.state_bytes)
                .ok_or_else(overflow)?,
            host_peak_bytes: self
                .host_peak_bytes
                .checked_add(candidate.host_peak_bytes)
                .ok_or_else(overflow)?,
            device_peak_bytes: self
                .device_peak_bytes
                .checked_add(candidate.device_peak_bytes)
                .ok_or_else(overflow)?,
        })
    }
}

fn overflow() -> PowerError {
    PowerError::InvalidRequest("microbatch aggregate resource count overflowed".to_string())
}

struct BatchBuilder {
    index: usize,
    slots: Vec<PlannedMicrobatchSlot>,
    totals: Totals,
}

impl BatchBuilder {
    fn new(index: usize) -> Self {
        Self {
            index,
            slots: Vec::new(),
            totals: Totals::default(),
        }
    }

    fn push(&mut self, source_index: usize, candidate: MicrobatchCandidate, totals: Totals) {
        self.slots.push(PlannedMicrobatchSlot {
            source_index,
            slot_sha256: candidate.slot_sha256,
            input_bytes: candidate.input_bytes,
            input_elements: candidate.input_elements,
            state_bytes: candidate.state_bytes,
            host_peak_bytes: candidate.host_peak_bytes,
            device_peak_bytes: candidate.device_peak_bytes,
        });
        self.totals = totals;
    }

    fn finish(self) -> PlannedMicrobatch {
        PlannedMicrobatch {
            index: self.index,
            slots: self.slots,
            input_bytes: self.totals.input_bytes,
            input_elements: self.totals.input_elements,
            state_bytes: self.totals.state_bytes,
            host_peak_bytes: self.totals.host_peak_bytes,
            device_peak_bytes: self.totals.device_peak_bytes,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PlanDeclaration<'a> {
    schema: &'a str,
    binding: &'a ExecutionBatchBinding,
    session_declaration_sha256: &'a Option<String>,
    runtime_device: RuntimeDeviceIdentity,
    snapshot: &'a HardwareMemorySnapshot,
    limits: &'a MicrobatchLimits,
    policy: &'a MicrobatchPolicy,
    host_budget_bytes: u64,
    device_budget_bytes: u64,
    shared_budget_bytes: Option<u64>,
    slot_count: usize,
    batches: &'a [PlannedMicrobatch],
}

fn declaration_sha256(plan: &MicrobatchPlan) -> Result<String> {
    let declaration = PlanDeclaration {
        schema: &plan.schema,
        binding: &plan.binding,
        session_declaration_sha256: &plan.session_declaration_sha256,
        runtime_device: plan.runtime_device,
        snapshot: &plan.snapshot,
        limits: &plan.limits,
        policy: &plan.policy,
        host_budget_bytes: plan.host_budget_bytes,
        device_budget_bytes: plan.device_budget_bytes,
        shared_budget_bytes: plan.shared_budget_bytes,
        slot_count: plan.slot_count,
        batches: &plan.batches,
    };
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&declaration)?)
    ))
}

fn validate_topology(
    planned: &HardwareMemorySnapshot,
    current: &HardwareMemorySnapshot,
) -> Result<()> {
    let device_matches = match (&planned.device, &current.device) {
        (None, None) => true,
        (Some(planned), Some(current)) => {
            planned.total_bytes == current.total_bytes
                && planned.unified_with_host == current.unified_with_host
        }
        (None, Some(_)) | (Some(_), None) => false,
    };
    if planned.runtime_device != current.runtime_device
        || planned.host.total_bytes != current.host.total_bytes
        || !device_matches
    {
        return Err(PowerError::InvalidRequest(
            "current memory snapshot does not match the microbatch plan topology".to_string(),
        ));
    }
    Ok(())
}
