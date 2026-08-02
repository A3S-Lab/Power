use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

/// Stable identity for one routed expert without assuming a model naming
/// convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

/// Exact router selection for one token or batch position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RoutedExpert {
    pub expert: u32,
    pub weight: f32,
}

/// One use of an expert in the original router output.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExpertAssignment {
    pub position: usize,
    pub weight: f32,
}

/// Validated union of exact expert routes across a batch.
///
/// The union is an I/O and scheduling optimization only. It never substitutes,
/// reorders, or renormalizes the model's router selections.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutedExpertBatch {
    layer: u32,
    expert_count: u32,
    selections: Vec<Vec<RoutedExpert>>,
    union: Vec<u32>,
    assignments: BTreeMap<u32, Vec<ExpertAssignment>>,
}

impl RoutedExpertBatch {
    pub fn new(
        layer: u32,
        selections: Vec<Vec<RoutedExpert>>,
        expert_count: u32,
        max_experts_per_position: usize,
    ) -> Result<Self> {
        if expert_count == 0 || max_experts_per_position == 0 || selections.is_empty() {
            return Err(PowerError::InvalidRequest(
                "routed expert batch bounds and positions must be non-zero".to_string(),
            ));
        }

        let mut union = BTreeSet::new();
        let mut assignments = BTreeMap::<u32, Vec<ExpertAssignment>>::new();
        for (position, routed) in selections.iter().enumerate() {
            if routed.is_empty() || routed.len() > max_experts_per_position {
                return Err(PowerError::InvalidRequest(format!(
                    "route position {position} selected {} experts, outside the 1..={max_experts_per_position} bound",
                    routed.len()
                )));
            }
            let mut position_experts = HashSet::with_capacity(routed.len());
            for selection in routed {
                if selection.expert >= expert_count {
                    return Err(PowerError::InvalidRequest(format!(
                        "route position {position} selected expert {}, but the layer has {expert_count} experts",
                        selection.expert
                    )));
                }
                if !selection.weight.is_finite() {
                    return Err(PowerError::InvalidRequest(format!(
                        "route position {position} contains a non-finite expert weight"
                    )));
                }
                if !position_experts.insert(selection.expert) {
                    return Err(PowerError::InvalidRequest(format!(
                        "route position {position} selected expert {} more than once",
                        selection.expert
                    )));
                }
                union.insert(selection.expert);
                assignments
                    .entry(selection.expert)
                    .or_default()
                    .push(ExpertAssignment {
                        position,
                        weight: selection.weight,
                    });
            }
        }

        Ok(Self {
            layer,
            expert_count,
            selections,
            union: union.into_iter().collect(),
            assignments,
        })
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn expert_count(&self) -> u32 {
        self.expert_count
    }

    pub fn selections(&self) -> &[Vec<RoutedExpert>] {
        &self.selections
    }

    pub fn experts(&self) -> &[u32] {
        &self.union
    }

    pub fn assignments(&self, expert: u32) -> &[ExpertAssignment] {
        self.assignments
            .get(&expert)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_union_reads_each_expert_once_without_changing_routes() {
        let selections = vec![
            vec![
                RoutedExpert {
                    expert: 3,
                    weight: 0.7,
                },
                RoutedExpert {
                    expert: 1,
                    weight: 0.3,
                },
            ],
            vec![
                RoutedExpert {
                    expert: 3,
                    weight: 0.6,
                },
                RoutedExpert {
                    expert: 2,
                    weight: 0.4,
                },
            ],
        ];
        let batch = RoutedExpertBatch::new(7, selections.clone(), 8, 2).unwrap();

        assert_eq!(batch.selections(), selections);
        assert_eq!(batch.experts(), [1, 2, 3]);
        assert_eq!(batch.assignments(3).len(), 2);
        assert_eq!(batch.assignments(1)[0].position, 0);
    }

    #[test]
    fn invalid_or_duplicate_routes_fail_closed() {
        assert!(RoutedExpertBatch::new(
            0,
            vec![vec![
                RoutedExpert {
                    expert: 1,
                    weight: 0.5,
                },
                RoutedExpert {
                    expert: 1,
                    weight: 0.5,
                },
            ]],
            2,
            2,
        )
        .is_err());
        assert!(RoutedExpertBatch::new(
            0,
            vec![vec![RoutedExpert {
                expert: 2,
                weight: 1.0,
            }]],
            2,
            1,
        )
        .is_err());
    }
}
