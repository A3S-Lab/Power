from __future__ import annotations

import copy
import unittest
from pathlib import Path

import dspark_quality_followup_evidence as evidence


EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "qwen3.8-27b-q6k-rtx4090"
    / "dspark"
    / "quality"
    / "followup-evidence.json"
)


class DsparkQualityFollowupEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = evidence.load_json(EVIDENCE_PATH)

    def test_checked_in_followup_is_internally_consistent(self) -> None:
        result = evidence.verify_evidence(
            self.payload,
            require_production_default=False,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["tokens_512_losses"], 0)
        self.assertEqual(result["tokens_1024_losses"], 0)
        self.assertEqual(result["tokens_1024_both_untruncated"], 5)
        self.assertEqual(result["tokens_1024_prediction_parity"], 5)
        self.assertAlmostEqual(result["tokens_512_workload_speedup"], 1.2224188595507897)
        self.assertFalse(result["production_default_eligible"])

    def test_task_vector_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["captures"]["tokens_1024"]["task_pairs"][0]["candidate"][
            "correct"
        ] = False

        with self.assertRaisesRegex(evidence.EvidenceError, "paired summary"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_raw_capture_hash_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["source"]["raw"]["tokens_512"]["aggregate_sha256"] = "0" * 64

        with self.assertRaisesRegex(evidence.EvidenceError, "raw tokens_512"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_exact_output_production_gate_remains_closed(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "0/5"):
            evidence.verify_evidence(
                self.payload,
                require_production_default=True,
            )


if __name__ == "__main__":
    unittest.main()
