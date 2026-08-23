from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import qwen38_quality_eval as quality
import qwen38_quality_report as quality_report


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
        body = quality.completion_body("model", "prompt", 32, 42, 512, 12)

        self.assertEqual(body["num_batch"], 12)
        self.assertEqual(body["num_ctx"], 512)

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
        self.assertEqual(args.num_ctx, 4096)

    def test_max_tokens_override_can_raise_the_reviewed_task_limit(self) -> None:
        self.assertEqual(
            quality.resolve_max_tokens(256, cap=None, override=512),
            512,
        )
        self.assertEqual(
            quality.resolve_max_tokens(384, cap=128, override=None),
            128,
        )

    def test_max_tokens_cap_and_override_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            quality.resolve_max_tokens(256, cap=128, override=512)

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
    def test_llamacpp_log_parser_excludes_warmup_and_aggregates_decode(self) -> None:
        def timing(tokens: int, milliseconds: float) -> str:
            return (
                "0.10 I slot print_timing: id 0 |        eval time = "
                f"{milliseconds:.2f} ms / {tokens:5d} tokens"
            )

        def acceptance(accepted: int, drafted: int, mean: float) -> str:
            return (
                "0.10 I slot print_timing: id 0 | draft acceptance = "
                f"{accepted / drafted:.5f} ( {accepted:4d} accepted / "
                f"{drafted:5d} generated), mean len = {mean:5.2f}"
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "\n".join(
                    [
                        timing(8, 100.0),
                        acceptance(4, 8, 3.0),
                        timing(12, 120.0),
                        acceptance(8, 12, 5.0),
                        timing(20, 100.0),
                        acceptance(18, 20, 7.0),
                    ]
                ),
                encoding="utf-8",
            )
            metrics = quality.parse_speculative_log(path, 2, "dflash2")

        assert metrics is not None
        overall = metrics["overall"]
        self.assertEqual(metrics["source"], "llama-server-timing-log")
        self.assertEqual(metrics["strategy"], "dflash2")
        self.assertEqual(overall["drafted_tokens"], 32)
        self.assertEqual(overall["accepted_tokens"], 26)
        self.assertAlmostEqual(overall["weighted_acceptance_rate"], 26 / 32)
        self.assertAlmostEqual(overall["verified_tokens_per_target_pass"], 6.0)
        self.assertAlmostEqual(
            overall["aggregate_reported_tokens_per_second"],
            1000 * 32 / 220,
        )

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
            metrics = quality.parse_speculative_log(path, 2)
        assert metrics is not None
        self.assertEqual(metrics["strategy"], "mtp")
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
            metrics = quality.parse_speculative_log(path, 1)

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
            metrics = quality.parse_speculative_log(path, 1)

        assert metrics is not None
        overall = metrics["overall"]
        self.assertAlmostEqual(overall["fr_target_token_id_prefix_fraction"], 0.75)
        self.assertAlmostEqual(
            overall["fr_correction_outside_token_id_prefix_fraction"], 0.5
        )

    def test_speculative_log_parser_accepts_dspark_and_rejects_mixed_modes(self) -> None:
        def line(strategy: str) -> str:
            return (
                f'speculative completion finished strategy="{strategy}" rounds=2 '
                "drafted_tokens=4 accepted_tokens=3 emitted_tokens=4 "
                "verified_emitted_tokens=4 tokens_per_second=80.0 fallback_replays=0"
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(line("dspark"), encoding="utf-8")
            metrics = quality.parse_speculative_log(path, 1)
            assert metrics is not None
            self.assertEqual(metrics["strategy"], "dspark")

            path.write_text("\n".join([line("mtp"), line("dspark")]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mix strategies"):
                quality.parse_speculative_log(path, 2)

    def test_speculative_log_parser_handles_an_immediate_stop(self) -> None:
        line = (
            'speculative completion finished strategy="dspark" rounds=0 '
            "drafted_tokens=0 accepted_tokens=0 emitted_tokens=0 "
            "verified_emitted_tokens=0 tokens_per_second=0 fallback_replays=0"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(line, encoding="utf-8")
            metrics = quality.parse_speculative_log(path, 1)

        assert metrics is not None
        self.assertEqual(metrics["overall"]["weighted_acceptance_rate"], 0.0)
        self.assertEqual(metrics["overall"]["verified_tokens_per_target_pass"], 0.0)
        self.assertEqual(
            metrics["overall"]["aggregate_reported_tokens_per_second"], 0.0
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
                "max_tokens_cap": 256,
                "max_tokens_override": None,
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
        self.assertIsNone(metadata["speculative_strategy"])
        self.assertEqual(metadata["max_tokens_cap"], 256)
        self.assertIsNone(metadata["max_tokens_override"])
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
                        "strategy": "mtp",
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
        self.assertEqual(
            aggregate["modes"]["fr8192-k7-b14-fixed"]["speculative_runtime"]
            ["strategy"],
            "mtp",
        )

    def test_dspark_strategy_and_exact_output_parity_are_aggregated(self) -> None:
        rows = [
            result_row("a", "A", "A", strict_prediction="A", content_hash="one"),
            result_row("b", "B", "B", strict_prediction="B", content_hash="two"),
        ]
        runtime = {
            "strategy": "dspark",
            "overall": {},
            "by_benchmark": {
                benchmark: {} for benchmark in ("mmlu", "gsm8k", "ceval")
            },
        }
        reports = []
        for order, mode in enumerate(("q6-off", "q6-dspark"), 1):
            reports.append(
                {
                    "mode_label": mode,
                    "repetition": 1,
                    "order_index": order,
                    "model_sha256": "same-q6-model",
                    "tasks_sha256": "tasks",
                    "server_sha256": "server",
                    "results": rows,
                    "summary": quality.report_summary(rows),
                    "speculative_runtime": runtime if mode == "q6-dspark" else None,
                }
            )

        aggregate = quality.aggregate_reports(
            reports, comparisons=[("q6-off", "q6-dspark")]
        )
        paired = aggregate["paired_runs"]["q6-off -> q6-dspark"][0]

        self.assertEqual(
            aggregate["modes"]["q6-dspark"]["speculative_runtime"]["strategy"],
            "dspark",
        )
        self.assertEqual(paired["task_count"], 2)
        self.assertEqual(paired["content_sha256_parity"], 2)
        self.assertIn("Untouched Q6_K + DSpark Q4", quality.render_markdown(aggregate))

    def test_q6_mtp_markdown_uses_the_q6_label_and_utf8_strict_marker(
        self,
    ) -> None:
        rows = [result_row("a", "A", "A", strict_prediction="A")]
        reports = []
        for order, mode in enumerate(("q6-off", "q6-mtp-full-vocab"), 1):
            reports.append(
                {
                    "mode_label": mode,
                    "repetition": 1,
                    "order_index": order,
                    "model_sha256": "same-q6-model",
                    "tasks_sha256": "tasks",
                    "server_sha256": "server",
                    "results": rows,
                    "summary": quality.report_summary(rows),
                    "speculative_runtime": None,
                }
            )

        aggregate = quality.aggregate_reports(
            reports, comparisons=[("q6-off", "q6-mtp-full-vocab")]
        )
        markdown = quality.render_markdown(aggregate)

        self.assertIn("Untouched Q6_K + full-vocabulary MTP", markdown)
        self.assertIn("\u6700\u7ec8\u7b54\u6848", markdown)

    def test_selected_task_runtime_without_benchmark_partitions_is_aggregated(
        self,
    ) -> None:
        rows = [result_row("a", "A", "A", strict_prediction="A")]
        reports = []
        for order, mode in enumerate(("q6-off", "q6-dspark"), 1):
            reports.append(
                {
                    "mode_label": mode,
                    "repetition": 1,
                    "order_index": order,
                    "model_sha256": "same-q6-model",
                    "tasks_sha256": "selected-tasks",
                    "server_sha256": "server",
                    "results": rows,
                    "summary": quality.report_summary(rows),
                    "speculative_runtime": (
                        {"strategy": "dspark", "overall": {}}
                        if mode == "q6-dspark"
                        else None
                    ),
                }
            )

        aggregate = quality.aggregate_reports(reports)
        runtime = aggregate["modes"]["q6-dspark"]["speculative_runtime"]

        self.assertEqual(runtime["strategy"], "dspark")
        self.assertNotIn("by_benchmark", runtime)

    def test_mixed_speculative_strategies_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mix speculative strategies"):
            quality_report.one_speculative_strategy(
                [{"strategy": "mtp"}, {"strategy": "dspark"}]
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
