from __future__ import annotations

import copy
import unittest
from pathlib import Path

import qwen38_q6_quality_evidence as evidence


EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "qwen3.8-27b-q6k-rtx4090"
    / "quality"
    / "pure-q6-rtx4090-3x.evidence.json"
)


class Q6QualityEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = evidence.load_json(EVIDENCE_PATH)

    def test_checked_in_evidence_is_q6_only_and_valid(self) -> None:
        result = evidence.verify_evidence(
            self.payload, require_lossless=False
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["total_requests"], 600)
        self.assertEqual(result["target_sha256"], evidence.Q6_SHA256)
        self.assertEqual(result["exact_output_parity"], 50)
        self.assertFalse(result["production_default_eligible"])

    def test_mixed_target_or_auxiliary_artifact_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["artifacts"]["auxiliary"] = [
            {"kind": "tbq4", "sha256": "0" * 64}
        ]

        with self.assertRaisesRegex(evidence.EvidenceError, "auxiliary"):
            evidence.verify_evidence(tampered, require_lossless=False)

    def test_gpu_interference_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["gpu_exclusivity"]["monitors"][0][
            "interference_detected"
        ] = True

        with self.assertRaisesRegex(evidence.EvidenceError, "interference"):
            evidence.verify_evidence(tampered, require_lossless=False)

    def test_unstructured_payload_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["captured_at"] = "2026-08-24T00:00:00Z"

        with self.assertRaisesRegex(evidence.EvidenceError, "payload differs"):
            evidence.verify_evidence(tampered, require_lossless=False)

    def test_lossless_gate_rejects_output_and_strict_score_drift(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "50/100"):
            evidence.verify_evidence(self.payload, require_lossless=True)


if __name__ == "__main__":
    unittest.main()
