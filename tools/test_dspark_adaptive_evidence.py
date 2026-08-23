from __future__ import annotations

import copy
import unittest
from pathlib import Path

import dspark_adaptive_evidence as evidence


EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "qwen3.8-27b-q6k-rtx4090"
    / "dspark"
    / "adaptive"
    / "evidence.json"
)


class AdaptiveEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = evidence.load_json(EVIDENCE_PATH)

    def test_checked_in_evidence_is_valid_but_not_production_default(self) -> None:
        result = evidence.verify_evidence(
            self.payload, require_production_default=False
        )

        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(
            result["peak_median_decode_tokens_per_second"],
            164.75590733921018,
        )
        self.assertAlmostEqual(
            result["peak_minimum_decode_tokens_per_second"],
            160.88066200562793,
        )
        self.assertEqual(result["quality_gains"], 5)
        self.assertEqual(result["quality_losses"], 3)
        self.assertEqual(result["fallback_replays"], 0)
        self.assertFalse(result["production_default_eligible"])

    def test_task_vector_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["quality"]["task_pairs"][0]["candidate_correct"] = not tampered[
            "quality"
        ]["task_pairs"][0]["candidate_correct"]

        with self.assertRaisesRegex(evidence.EvidenceError, "paired summary"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_missing_clock_attestation_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["host_controls"]["quality"]["gpu_clock"][
            "lock_applied"
        ] = False

        with self.assertRaisesRegex(evidence.EvidenceError, "clock lock"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_production_default_gate_rejects_the_diagnostic(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "55/100"):
            evidence.verify_evidence(
                self.payload, require_production_default=True
            )


if __name__ == "__main__":
    unittest.main()
