from __future__ import annotations

import copy
import unittest
from pathlib import Path

import qwen38_quality_evidence as evidence


EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "qwen3.8-27b-q6k-rtx4090"
    / "dspark"
    / "quality"
    / "evidence.json"
)


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = evidence.load_json(EVIDENCE_PATH)

    def test_checked_in_evidence_is_valid_but_not_production_default(self) -> None:
        result = evidence.verify_evidence(
            self.payload, require_production_default=False
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["total_requests"], 600)
        self.assertEqual(result["exact_output_parity"], 54)
        self.assertFalse(result["production_default_eligible"])

    def test_task_hash_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["task_pairs"][0]["base"]["content_sha256"] = "0" * 64
        tampered["task_pairs"][0]["dspark"]["content_sha256"] = "0" * 64

        with self.assertRaisesRegex(evidence.EvidenceError, "content vector"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_production_default_gate_rejects_the_diagnostic(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "54/100"):
            evidence.verify_evidence(
                self.payload, require_production_default=True
            )


if __name__ == "__main__":
    unittest.main()
