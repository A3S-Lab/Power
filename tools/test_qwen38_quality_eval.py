from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import qwen38_quality_eval as quality


def result_row(
    task_id: str,
    expected: str,
    prediction: str | None,
    *,
    strict_prediction: str | None = None,
    tokens: int = 12,
    max_tokens: int = 32,
    latency: float = 1.0,
    content_hash: str | None = None,
) -> dict[str, object]:
    return {
        "id": task_id,
        "benchmark": "mmlu",
        "subject": "test",
        "expected": expected,
        "answer_type": "choice",
        "max_tokens": max_tokens,
        "prediction": prediction,
        "strict_prediction": strict_prediction,
        "correct": prediction == expected,
        "strict_correct": strict_prediction == expected,
        "content_sha256": content_hash or task_id,
        "finish_reason": "length" if tokens >= max_tokens else "stop",
        "latency_seconds": latency,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": tokens,
            "total_tokens": 10 + tokens,
        },
        "receipt_sha256": None,
        "error": None,
    }


class PredictionTests(unittest.TestCase):
    def test_choice_extractors_distinguish_explicit_and_fallback(self) -> None:
        content = "I considered A and B. The better option is C."
        self.assertIsNone(quality.strict_prediction(content, "choice"))
        self.assertEqual(quality.lenient_prediction(content, "choice"), "C")
        self.assertEqual(
            quality.strict_prediction("最终答案：D", "choice"),
            "D",
        )

    def test_number_extractor_normalizes_commas(self) -> None:
        content = "Reasoning: 20 * 3,500.\nFINAL: 70,000"
        self.assertEqual(quality.strict_prediction(content, "number"), "70000")
        self.assertEqual(quality.lenient_prediction(content, "number"), "70000")


class HttpTests(unittest.TestCase):
    def test_completion_body_locks_logical_batch_size(self) -> None:
        body = quality.completion_body("model", "prompt", 32, 42, 14)

        self.assertEqual(body["num_batch"], 14)
        self.assertEqual(body["num_ctx"], 4096)

    def test_transient_url_error_is_retried(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"ok": true}'

        with (
            mock.patch.object(
                quality.urllib.request,
                "urlopen",
                side_effect=[urllib.error.URLError("transient"), Response()],
            ) as urlopen,
            mock.patch.object(quality.time, "sleep") as sleep,
        ):
            payload = quality.request_json("GET", "https://example.invalid", attempts=2)
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)


class TaskTests(unittest.TestCase):
    def test_task_selection_is_ordered_and_hash_locked(self) -> None:
        tasks = [
            {"id": "a", "prompt": "A"},
            {"id": "b", "prompt": "B"},
            {"id": "c", "prompt": "C"},
        ]
        selected = [tasks[2], tasks[0]]
        digest = quality.sha256_text(quality.canonical_json(selected))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "a3s.power.quality-eval.selection.v1",
                        "task_ids": ["c", "a"],
                        "expected_tasks_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            actual, actual_digest = quality.select_tasks(tasks, path)

        self.assertEqual(actual, selected)
        self.assertEqual(actual_digest, digest)

    def test_run_parser_defaults_to_reviewed_logical_batch_size(self) -> None:
        args = quality.parser().parse_args(
            [
                "run",
                "--mode-label",
                "test",
                "--repetition",
                "1",
                "--order-index",
                "1",
                "--model-sha256",
                "model",
                "--server-sha256",
                "server",
                "--power-commit",
                "commit",
                "--tasks",
                "tasks.json",
                "--manifest",
                "manifest.json",
                "--output",
                "output.json",
                "--server-log",
                "server.log",
            ]
        )

        self.assertEqual(args.num_batch, 14)

    def test_atomic_write_retries_a_transient_windows_reader_lock(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                quality.Path,
                "replace",
                side_effect=[PermissionError("locked"), None],
            ) as replace,
            mock.patch.object(quality.time, "sleep") as sleep,
        ):
            quality.atomic_write(Path(directory) / "report.json", {"ok": True})
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once_with(0.05)

    def test_task_cache_is_hash_locked(self) -> None:
        tasks = [
            {
                "id": "mmlu:1",
                "benchmark": "mmlu",
                "subject": "test",
                "prompt": "Question",
                "expected": "A",
                "answer_type": "choice",
                "max_tokens": 32,
            }
        ]
        digest = quality.sha256_text(quality.canonical_json(tasks))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            cache = root / "tasks.json"
            manifest.write_text(
                json.dumps({"expected_tasks_sha256": digest}), encoding="utf-8"
            )
            cache.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
            loaded, actual = quality.load_tasks(cache, manifest)
        self.assertEqual(loaded, tasks)
        self.assertEqual(actual, digest)


class SummaryTests(unittest.TestCase):
    def test_partial_report_keeps_empty_benchmarks(self) -> None:
        summary = quality.report_summary(
            [result_row("a", "A", "A", strict_prediction="A")]
        )
        self.assertEqual(summary["mmlu"]["total"], 1)
        self.assertEqual(summary["gsm8k"]["total"], 0)
        self.assertEqual(summary["ceval"]["accuracy"], 0.0)

    def test_summary_counts_strict_and_truncated_results(self) -> None:
        rows = [
            result_row("a", "A", "A", strict_prediction="A"),
            result_row("b", "B", "B", tokens=32, max_tokens=32),
            result_row("c", "C", "D", strict_prediction="D"),
        ]
        summary = quality.summarize_rows(rows)
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["strict_correct"], 1)
        self.assertEqual(summary["truncated"], 1)
        self.assertAlmostEqual(summary["aggregate_completion_tokens_per_second"], 56 / 3)

    def test_pair_metrics_preserve_untruncated_answer_parity(self) -> None:
        base = {
            "results": [
                result_row("a", "A", "A", strict_prediction="A"),
                result_row("b", "B", "A", tokens=32, max_tokens=32),
            ]
        }
        candidate = {
            "results": [
                result_row("a", "A", "A", strict_prediction="A", content_hash="changed"),
                result_row("b", "B", "B", strict_prediction="B"),
            ]
        }
        metrics = quality.pair_metrics(base, candidate)
        self.assertEqual(metrics["gains"], 1)
        self.assertEqual(metrics["losses"], 0)
        self.assertEqual(metrics["both_untruncated"], 1)
        self.assertEqual(metrics["both_untruncated_prediction_parity"], 1)
        self.assertEqual(metrics["content_sha256_parity"], 1)


class RuntimeLogTests(unittest.TestCase):
    def test_mtp_log_parser_excludes_warmup_record(self) -> None:
        def line(emitted: int, accepted: int) -> str:
            return (
                "speculative completion finished strategy=\"mtp\" rounds=2 "
                f"drafted_tokens=8 accepted_tokens={accepted} emitted_tokens={emitted} "
                f"verified_emitted_tokens={emitted - 1} tokens_per_second=40.0 "
                "fallback_replays=1"
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "\n".join([line(4, 1), line(10, 4), line(12, 6)]),
                encoding="utf-8",
            )
            metrics = quality.parse_mtp_log(path, 2)
        assert metrics is not None
        self.assertEqual(metrics["overall"]["accepted_tokens"], 10)
        self.assertEqual(metrics["overall"]["drafted_tokens"], 16)
        self.assertEqual(metrics["overall"]["fallback_replays"], 2)

    def test_mtp_log_parser_aggregates_adaptive_and_fr_telemetry(self) -> None:
        line = (
            'speculative completion finished strategy="mtp" rounds=12 '
            "drafted_tokens=24 accepted_tokens=6 emitted_tokens=18 "
            "verified_emitted_tokens=18 tokens_per_second=45.0 fallback_replays=0 "
            "rollback_guard_activations=1 rollback_guard_draft_limit=6 "
            "target_only_tokens=9 target_only_after_round=Some(12) "
            "fr_target_samples=30 fr_target_samples_in_token_id_prefix=24 "
            "fr_rejected_rounds=10 fr_corrections_outside_token_id_prefix=4 "
            "draft_limit_histogram=[0, 2, 4, 6]"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(line, encoding="utf-8")
            metrics = quality.parse_mtp_log(path, 1)

        assert metrics is not None
        overall = metrics["overall"]
        self.assertEqual(overall["target_only_requests"], 1)
        self.assertEqual(overall["target_only_tokens"], 9)
        self.assertEqual(overall["rollback_guard_requests"], 1)
        self.assertEqual(overall["rollback_guard_activations"], 1)
        self.assertAlmostEqual(overall["fr_target_token_id_prefix_fraction"], 0.8)
        self.assertAlmostEqual(
            overall["fr_correction_outside_token_id_prefix_fraction"], 0.4
        )
        self.assertEqual(overall["draft_limit_histogram"], [0, 2, 4, 6])

    def test_mtp_log_parser_maps_legacy_prefix_fields_to_honest_names(self) -> None:
        line = (
            'speculative completion finished strategy="mtp" rounds=2 '
            "drafted_tokens=4 accepted_tokens=2 emitted_tokens=3 "
            "verified_emitted_tokens=3 tokens_per_second=30.0 fallback_replays=0 "
            "fr_target_samples=8 fr_target_samples_in_vocab=6 "
            "fr_rejected_rounds=2 fr_corrections_outside_vocab=1"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(line, encoding="utf-8")
            metrics = quality.parse_mtp_log(path, 1)

        assert metrics is not None
        overall = metrics["overall"]
        self.assertAlmostEqual(overall["fr_target_token_id_prefix_fraction"], 0.75)
        self.assertAlmostEqual(
            overall["fr_correction_outside_token_id_prefix_fraction"], 0.5
        )


class AggregateTests(unittest.TestCase):
    def test_report_metadata_excludes_untrusted_model_content(self) -> None:
        rows = [result_row("a", "A", "A", strict_prediction="A")]
        rows[0]["content"] = "control:\b unicode:\ud800"
        report = {
            "schema": "a3s.power.quality-eval.report.v3",
            "mode_label": "q6-off",
            "repetition": 1,
            "order_index": 1,
            "model": "model",
            "model_sha256": "model-hash",
            "tasks_sha256": "tasks",
            "server_sha256": "server",
            "power_commit": "commit",
            "seed": 42,
            "request": {
                "num_ctx": 4096,
                "num_batch": 14,
                "warmup_requests": 1,
            },
            "results": rows,
            "summary": quality.report_summary(rows),
            "completed_at": "now",
            "speculative_runtime": None,
        }

        metadata = quality.report_metadata(report)

        self.assertEqual(metadata["result_count"], 1)
        self.assertEqual(metadata["completed"], 1)
        self.assertFalse(metadata["has_speculative_runtime"])
        self.assertNotIn("content", metadata)

    def test_arbitrary_sweep_modes_are_aggregated(self) -> None:
        reports = []
        for order, mode in enumerate(("fr8192-k7-b14-fixed", "fr32768-k7-b14-fixed"), 1):
            rows = [result_row("a", "A", "A", strict_prediction="A")]
            reports.append(
                {
                    "mode_label": mode,
                    "repetition": 1,
                    "order_index": order,
                    "model_sha256": "model",
                    "tasks_sha256": "tasks",
                    "server_sha256": "server",
                    "results": rows,
                    "summary": quality.report_summary(rows),
                    "speculative_runtime": {
                        "overall": {
                            "weighted_acceptance_rate": 0.5,
                            "verified_tokens_per_target_pass": 2.0,
                            "fallback_replays": 0,
                            "aggregate_reported_tokens_per_second": 40.0,
                            "target_only_requests": 0,
                            "target_only_tokens": 0,
                            "fr_target_token_id_prefix_fraction": 0.9,
                            "fr_correction_outside_token_id_prefix_fraction": 0.1,
                        }
                    },
                }
            )

        aggregate = quality.aggregate_sweep_reports(reports)

        self.assertEqual(aggregate["repetitions"], 1)
        self.assertEqual(set(aggregate["modes"]), {report["mode_label"] for report in reports})
        self.assertEqual(
            aggregate["modes"]["fr8192-k7-b14-fixed"]["speculative_runtime"]
            ["weighted_acceptance_rate"]["mean"],
            0.5,
        )

    def test_three_mode_reports_are_aggregated(self) -> None:
        reports = []
        for order, mode in enumerate(("q6-off", "tbq4-off", "tbq4-mtp-fr"), 1):
            rows = [
                result_row("a", "A", "A", strict_prediction="A"),
                result_row("b", "B", "B", strict_prediction="B"),
            ]
            reports.append(
                {
                    "mode_label": mode,
                    "repetition": 1,
                    "order_index": order,
                    "model_sha256": mode,
                    "tasks_sha256": "tasks",
                    "server_sha256": "server",
                    "results": rows,
                    "summary": quality.report_summary(rows),
                    "speculative_runtime": None,
                }
            )
        aggregate = quality.aggregate_reports(reports)
        self.assertEqual(aggregate["repetitions"], 1)
        self.assertEqual(aggregate["modes"]["tbq4-off"]["prediction_stable_tasks"], 2)
        self.assertEqual(
            aggregate["modes"]["tbq4-off"]["by_benchmark"]["mmlu"]["accuracy"]["mean"],
            1.0,
        )
        self.assertEqual(
            aggregate["paired_runs"]["tbq4-off -> tbq4-mtp-fr"][0]["gains"],
            0,
        )

    def test_explicit_full_vocab_comparisons_are_aggregated(self) -> None:
        reports = []
        modes = ("q6-off", "tbq4-off", "tbq4-mtp-full-vocab")
        for order, mode in enumerate(modes, 1):
            rows = [
                result_row("a", "A", "A", strict_prediction="A"),
                result_row("b", "B", "B", strict_prediction="B"),
            ]
            reports.append(
                {
                    "mode_label": mode,
                    "repetition": 1,
                    "order_index": order,
                    "model_sha256": mode,
                    "tasks_sha256": "tasks",
                    "server_sha256": "server",
                    "results": rows,
                    "summary": quality.report_summary(rows),
                    "speculative_runtime": None,
                }
            )

        comparisons = [
            ("q6-off", "tbq4-off"),
            ("tbq4-off", "tbq4-mtp-full-vocab"),
        ]
        aggregate = quality.aggregate_reports(reports, comparisons=comparisons)

        self.assertEqual(set(aggregate["modes"]), set(modes))
        self.assertEqual(
            set(aggregate["paired_runs"]),
            {
                "q6-off -> tbq4-off",
                "tbq4-off -> tbq4-mtp-full-vocab",
            },
        )
        markdown = quality.render_markdown(aggregate)
        self.assertIn("TBQ4 + full-vocabulary MTP", markdown)
        self.assertIn("2/2", markdown)


if __name__ == "__main__":
    unittest.main()
