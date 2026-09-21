//! Pure System One scoring helpers (no backend I/O).

/// Softmax over a subset of logits; returns probabilities in the same order.
pub fn subset_softmax(logits: &[f32]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&logit| ((logit - max) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        let n = logits.len() as f64;
        return vec![1.0 / n; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

/// Confidence from a probability distribution: `1 - H(p) / ln(n)`.
///
/// Returns 1.0 for a single option. Not TypeSafe Jev's internal formula.
pub fn distribution_confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    if n <= 1 {
        return 1.0;
    }
    let mut entropy = 0.0_f64;
    for &p in probabilities {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    let max_entropy = (n as f64).ln();
    if max_entropy <= 0.0 {
        return 1.0;
    }
    (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
}

/// Map a zero-based choice index to a single-token-friendly label.
pub fn choice_label(index: usize) -> String {
    if index < 26 {
        ((b'A' + index as u8) as char).to_string()
    } else {
        index.to_string()
    }
}

/// Map a score level index to its label string.
pub fn score_label(index: usize) -> String {
    index.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subset_softmax_uniform_when_equal() {
        let probs = subset_softmax(&[1.0, 1.0, 1.0]);
        assert_eq!(probs.len(), 3);
        for p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn subset_softmax_peaks_on_max() {
        let probs = subset_softmax(&[0.0, 10.0, 0.0]);
        assert!(probs[1] > 0.99);
        assert!(probs[0] < 0.01);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn subset_softmax_empty() {
        assert!(subset_softmax(&[]).is_empty());
    }

    #[test]
    fn confidence_one_hot_is_high() {
        let c = distribution_confidence(&[1.0, 0.0, 0.0]);
        assert!(c > 0.99);
    }

    #[test]
    fn confidence_uniform_is_low() {
        let c = distribution_confidence(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert!(c < 0.01);
    }

    #[test]
    fn confidence_single_option() {
        assert_eq!(distribution_confidence(&[1.0]), 1.0);
    }

    #[test]
    fn choice_labels_letters_then_numbers() {
        assert_eq!(choice_label(0), "A");
        assert_eq!(choice_label(25), "Z");
        assert_eq!(choice_label(26), "26");
    }
}
