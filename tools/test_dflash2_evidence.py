from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import dflash2_evidence as evidence


EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "qwen3.8-27b-q6k-rtx4090"
    / "dflash2"
    / "evidence.json"
)


class DFlash2EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = evidence.load_json(EVIDENCE_PATH)

    def test_checked_in_capture_is_q6_only_and_experimental(self) -> None:
        result = evidence.verify_evidence(
            self.payload, require_production_default=False
        )

        self.assertEqual(result["target"], "Qwen3.8-27B Q6_K")
        self.assertEqual(result["quality_score"], "9/12 in both modes")
        self.assertEqual(result["complete_output_parity"], "7/12")
        self.assertFalse(result["production_default_eligible"])

    def test_evidence_is_path_free_and_contains_no_gpu_uuid(self) -> None:
        encoded = json.dumps(self.payload, ensure_ascii=False)

        self.assertNotIn("D:\\", encoded)
        self.assertNotIn("GPU-", encoded)

    def test_performance_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["performance"]["candidate"]["samples"][1][
            "predicted_tokens_per_second"
        ] = 175.0

        with self.assertRaisesRegex(evidence.EvidenceError, "median"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_task_pair_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["quality"]["task_pairs"][0]["target"][
            "content_sha256"
        ] = "0" * 64

        with self.assertRaisesRegex(evidence.EvidenceError, "task-pair vector"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_raw_report_identity_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["source"]["performance_report_sha256"] = "0" * 64

        with self.assertRaisesRegex(evidence.EvidenceError, "source identity"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_host_control_tampering_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["controls"]["gpu"]["clock_lock_applied"] = False

        with self.assertRaisesRegex(evidence.EvidenceError, "controls evidence"):
            evidence.verify_evidence(tampered, require_production_default=False)

    def test_production_gate_rejects_standalone_output_divergence(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "7/12"):
            evidence.verify_evidence(
                self.payload, require_production_default=True
            )


if __name__ == "__main__":
    unittest.main()
