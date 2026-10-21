#![cfg(test)]
use super::*;
#[test]
fn admission_histogram_and_refusals_have_closed_bounded_series() {
    let metrics = Metrics::new();
    metrics.record_http_admission_wait(HttpAdmissionOutcome::Admitted, 0.025);
    metrics.record_http_admission_wait(HttpAdmissionOutcome::Timeout, 1.0);
    metrics.record_http_admission_wait(HttpAdmissionOutcome::QueueFull, 0.0);
    metrics.record_http_admission_wait(HttpAdmissionOutcome::Cancelled, 0.05);
    let output = metrics.render();
    assert!(output
        .contains("power_http_admission_wait_seconds_bucket{outcome=\"admitted\",le=\"0.05\"} 1"));
    assert!(output
        .contains("power_http_admission_wait_seconds_bucket{outcome=\"admitted\",le=\"0.01\"} 0"));
    assert!(
        output.contains("power_http_admission_wait_seconds_sum{outcome=\"timeout\"} 1.000000000")
    );
    assert!(output.contains("power_http_admission_refusals_total{reason=\"queue_full\"} 1"));
    assert!(output.contains("power_http_admission_refusals_total{reason=\"cancelled\"} 1"));
    for _ in 0..20_000 {
        metrics.record_http_admission_wait(HttpAdmissionOutcome::Admitted, 0.001);
    }
    let output = metrics.render();
    assert!(output.contains("power_http_admission_wait_seconds_count{outcome=\"admitted\"} 20001"));
    assert_eq!(
        output
            .lines()
            .filter(|line| line.starts_with("power_http_admission_wait_seconds_bucket"))
            .count(),
        5 * 12
    );
}
