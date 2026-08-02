use super::coupling::{RouteCouplingPolicy, RouteCouplingTracker, RoutePrefetchHint};
use super::{RoutedExpert, RoutedExpertBatch, TelemetryMode};
use crate::error::PowerError;

fn batch(layer: u32, routes: &[&[u32]], expert_count: u32) -> RoutedExpertBatch {
    RoutedExpertBatch::new(
        layer,
        routes
            .iter()
            .map(|position| {
                position
                    .iter()
                    .map(|expert| RoutedExpert {
                        expert: *expert,
                        weight: 1.0 / position.len() as f32,
                    })
                    .collect()
            })
            .collect(),
        expert_count,
        routes.iter().map(|routes| routes.len()).max().unwrap(),
    )
    .unwrap()
}

fn tracker(mode: TelemetryMode) -> RouteCouplingTracker {
    RouteCouplingTracker::new(mode, "weights-a", RouteCouplingPolicy::default())
}

#[test]
fn learns_per_position_scores_and_preserves_a_deterministic_union() {
    let tracker = tracker(TelemetryMode::Detailed);
    let source = batch(3, &[&[1, 2], &[1]], 8);
    let target = batch(4, &[&[4, 3], &[3]], 8);
    tracker.record_transition(&source, &target).unwrap();

    let hints = tracker.hints(&source, 4, 2).unwrap();
    assert_eq!(hints.source_layer(), 3);
    assert_eq!(hints.target_layer(), 4);
    assert_eq!(hints.experts(), &[3, 4]);
    assert_eq!(
        hints.selections()[0],
        [
            RoutePrefetchHint {
                expert: 3,
                score: 3
            },
            RoutePrefetchHint {
                expert: 4,
                score: 2
            }
        ]
    );
    assert_eq!(hints.selections()[1][0].expert, 3);

    let evaluation = tracker.evaluate(&hints, &target).unwrap();
    assert_eq!(evaluation.actual_selections, 3);
    assert_eq!(evaluation.matched_selections, 3);
    assert_eq!(evaluation.recall(), 1.0);
    let telemetry = tracker.telemetry().unwrap();
    assert_eq!(telemetry.evaluations, 1);
    assert_eq!(telemetry.recall(), 1.0);
}

#[test]
fn privacy_modes_below_detailed_reject_coupling_data() {
    let source = batch(0, &[&[0]], 2);
    let target = batch(1, &[&[1]], 2);
    for mode in [TelemetryMode::Disabled, TelemetryMode::Aggregate] {
        let tracker = tracker(mode);
        assert!(matches!(
            tracker.record_transition(&source, &target),
            Err(PowerError::PolicyViolation(_))
        ));
        assert!(matches!(
            tracker.hints(&source, 1, 1),
            Err(PowerError::PolicyViolation(_))
        ));
        assert!(matches!(
            tracker.history(),
            Err(PowerError::PolicyViolation(_))
        ));
    }
}

#[test]
fn history_is_digest_bound_and_restore_is_atomic() {
    let source = batch(0, &[&[0]], 2);
    let target = batch(1, &[&[1]], 2);
    let learned = tracker(TelemetryMode::Detailed);
    learned.record_transition(&source, &target).unwrap();
    let history = learned.history().unwrap();

    let other = RouteCouplingTracker::new(
        TelemetryMode::Detailed,
        "weights-b",
        RouteCouplingPolicy::default(),
    );
    assert!(other.restore(&history).is_err());
    assert!(other.history().unwrap().entries.is_empty());

    let restored = tracker(TelemetryMode::Detailed);
    let mut invalid = history.clone();
    invalid.entries.push(invalid.entries[0]);
    assert!(restored.restore(&invalid).is_err());
    assert!(restored.history().unwrap().entries.is_empty());
    restored.restore(&history).unwrap();
    assert_eq!(restored.history().unwrap().entries, history.entries);
}

#[test]
fn policy_bounds_distance_positions_entries_and_hints() {
    let policy = RouteCouplingPolicy {
        max_lookahead_layers: 1,
        max_positions_per_batch: 1,
        max_entries: 1,
        max_hints_per_position: 1,
    };
    let tracker = RouteCouplingTracker::new(TelemetryMode::Detailed, "weights-a", policy);
    let source = batch(0, &[&[0]], 2);
    let target = batch(1, &[&[1]], 2);
    tracker.record_transition(&source, &target).unwrap();
    assert!(tracker.hints(&source, 1, 2).is_err());
    assert!(tracker.hints(&source, 2, 1).is_err());
    assert!(tracker
        .record_transition(&batch(0, &[&[0], &[0]], 2), &batch(1, &[&[1], &[1]], 2),)
        .is_err());
    assert!(tracker
        .record_transition(&batch(0, &[&[0, 1]], 2), &target)
        .is_err());
}
