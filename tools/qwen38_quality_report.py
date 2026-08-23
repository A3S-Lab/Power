"""Aggregation and Markdown rendering for Qwen3.8 quality reports."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_truncated(row: dict[str, Any]) -> bool:
    if row.get("finish_reason") == "length":
        return True
    usage = row.get("usage") or {}
    return int(usage.get("completion_tokens", 0)) >= int(row["max_tokens"])


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gains, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def pair_metrics(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_rows = {row["id"]: row for row in base["results"]}
    candidate_rows = {row["id"]: row for row in candidate["results"]}
    if base_rows.keys() != candidate_rows.keys():
        raise ValueError("paired reports contain different task IDs")
    pairs = [(base_rows[key], candidate_rows[key]) for key in base_rows]
    gains = sum(not left["correct"] and right["correct"] for left, right in pairs)
    losses = sum(left["correct"] and not right["correct"] for left, right in pairs)
    strict_gains = sum(
        not left["strict_correct"] and right["strict_correct"]
        for left, right in pairs
    )
    strict_losses = sum(
        left["strict_correct"] and not right["strict_correct"]
        for left, right in pairs
    )
    untruncated = [
        (left, right)
        for left, right in pairs
        if not is_truncated(left) and not is_truncated(right)
    ]
    return {
        "task_count": len(pairs),
        "base_correct": sum(row["correct"] for row, _ in pairs),
        "candidate_correct": sum(row["correct"] for _, row in pairs),
        "gains": gains,
        "losses": losses,
        "exact_mcnemar_p": exact_mcnemar_p(gains, losses),
        "strict_base_correct": sum(row["strict_correct"] for row, _ in pairs),
        "strict_candidate_correct": sum(
            row["strict_correct"] for _, row in pairs
        ),
        "strict_gains": strict_gains,
        "strict_losses": strict_losses,
        "strict_exact_mcnemar_p": exact_mcnemar_p(strict_gains, strict_losses),
        "prediction_parity": sum(
            left["prediction"] == right["prediction"] for left, right in pairs
        ),
        "content_sha256_parity": sum(
            left["content_sha256"] == right["content_sha256"]
            for left, right in pairs
        ),
        "both_untruncated": len(untruncated),
        "both_untruncated_prediction_parity": sum(
            left["prediction"] == right["prediction"]
            for left, right in untruncated
        ),
    }


def describe(values: Iterable[float | None]) -> dict[str, float | None]:
    selected = [value for value in values if value is not None]
    if not selected:
        return {
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "mean": statistics.mean(selected),
        "median": statistics.median(selected),
        "minimum": min(selected),
        "maximum": max(selected),
    }


def one_speculative_strategy(reports: Iterable[dict[str, Any]]) -> str | None:
    strategies = {
        report.get("strategy")
        for report in reports
        if report.get("strategy") is not None
    }
    if len(strategies) > 1:
        raise ValueError(
            f"runtime reports mix speculative strategies: {sorted(strategies)}"
        )
    return next(iter(strategies), None)


def aggregate_reports(
    reports: list[dict[str, Any]],
    comparisons: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("quality matrix contains no reports")
    task_hashes = {report["tasks_sha256"] for report in reports}
    server_hashes = {report["server_sha256"] for report in reports}
    if len(task_hashes) != 1 or len(server_hashes) != 1:
        raise ValueError("reports do not share one task set and server binary")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_repetition: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for report in reports:
        mode = report["mode_label"]
        repetition = int(report["repetition"])
        if mode in by_repetition[repetition]:
            raise ValueError(f"duplicate mode {mode} in repetition {repetition}")
        grouped[mode].append(report)
        by_repetition[repetition][mode] = report
    required = set(grouped)
    repetitions = {len(items) for items in grouped.values()}
    if len(repetitions) != 1:
        raise ValueError("mode repetition counts differ")
    if any(set(modes) != required for modes in by_repetition.values()):
        raise ValueError("quality matrix repetition is incomplete")

    selected_comparisons = list(comparisons) if comparisons is not None else [
        (base, candidate)
        for base, candidate in (
            ("q6-off", "tbq4-off"),
            ("tbq4-off", "tbq4-mtp-fr"),
        )
        if base in required and candidate in required
    ]
    for base, candidate in selected_comparisons:
        if base not in required or candidate not in required:
            raise ValueError(
                f"comparison {base} -> {candidate} references an absent mode"
            )
        if base == candidate:
            raise ValueError("comparison base and candidate must differ")

    modes: dict[str, Any] = {}
    for mode, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["repetition"])
        summaries = [item["summary"]["overall"] for item in items]
        by_benchmark = {}
        for benchmark in ("mmlu", "gsm8k", "ceval"):
            benchmark_summaries = [
                item["summary"][benchmark] for item in items
            ]
            by_benchmark[benchmark] = {
                "accuracy": describe(
                    summary["accuracy"] for summary in benchmark_summaries
                ),
                "strict_accuracy": describe(
                    summary["strict_accuracy"] for summary in benchmark_summaries
                ),
                "aggregate_completion_tokens_per_second": describe(
                    summary["aggregate_completion_tokens_per_second"]
                    for summary in benchmark_summaries
                ),
            }
        task_predictions: dict[str, list[Any]] = defaultdict(list)
        task_outputs: dict[str, list[str]] = defaultdict(list)
        for item in items:
            for row in item["results"]:
                task_predictions[row["id"]].append(row["prediction"])
                task_outputs[row["id"]].append(row["content_sha256"])
        modes[mode] = {
            "model_sha256": items[0]["model_sha256"],
            "runs": [
                {
                    "repetition": item["repetition"],
                    "order_index": item["order_index"],
                    "summary": item["summary"],
                    "speculative_runtime": item.get("speculative_runtime"),
                }
                for item in items
            ],
            "accuracy": describe(summary["accuracy"] for summary in summaries),
            "strict_accuracy": describe(
                summary["strict_accuracy"] for summary in summaries
            ),
            "truncated": describe(float(summary["truncated"]) for summary in summaries),
            "median_latency_seconds": describe(
                summary["median_latency_seconds"] for summary in summaries
            ),
            "aggregate_completion_tokens_per_second": describe(
                summary["aggregate_completion_tokens_per_second"]
                for summary in summaries
            ),
            "prediction_stable_tasks": sum(
                len(set(values)) == 1 for values in task_predictions.values()
            ),
            "content_stable_tasks": sum(
                len(set(values)) == 1 for values in task_outputs.values()
            ),
            "task_count": len(items[0]["results"]),
            "by_benchmark": by_benchmark,
        }
        runtime_reports = [
            item["speculative_runtime"]
            for item in items
            if item.get("speculative_runtime") is not None
        ]
        if runtime_reports:
            runtime_summary = {
                "strategy": one_speculative_strategy(runtime_reports),
                "overall": {
                    field: describe(
                        report["overall"].get(field, 0) for report in runtime_reports
                    )
                    for field in (
                        "weighted_acceptance_rate",
                        "verified_tokens_per_target_pass",
                        "fallback_replays",
                        "rollback_guard_requests",
                        "rollback_guard_activations",
                        "aggregate_reported_tokens_per_second",
                    )
                },
            }
            if all("by_benchmark" in report for report in runtime_reports):
                runtime_summary["by_benchmark"] = {
                    benchmark: {
                        field: describe(
                            report["by_benchmark"][benchmark].get(field, 0)
                            for report in runtime_reports
                        )
                        for field in (
                            "weighted_acceptance_rate",
                            "verified_tokens_per_target_pass",
                            "fallback_replays",
                            "rollback_guard_requests",
                            "rollback_guard_activations",
                            "aggregate_reported_tokens_per_second",
                        )
                    }
                    for benchmark in ("mmlu", "gsm8k", "ceval")
                }
            modes[mode]["speculative_runtime"] = runtime_summary

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for repetition, modes_for_run in sorted(by_repetition.items()):
        for base, candidate in selected_comparisons:
            pairs[f"{base} -> {candidate}"].append(
                pair_metrics(modes_for_run[base], modes_for_run[candidate])
            )
    return {
        "schema": "a3s.power.quality-eval.aggregate.v1",
        "created_at": utc_now(),
        "tasks_sha256": next(iter(task_hashes)),
        "server_sha256": next(iter(server_hashes)),
        "repetitions": next(iter(repetitions)),
        "modes": modes,
        "paired_runs": pairs,
    }


def aggregate_sweep_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("sweep contains no reports")
    task_hashes = {report["tasks_sha256"] for report in reports}
    server_hashes = {report["server_sha256"] for report in reports}
    if len(task_hashes) != 1 or len(server_hashes) != 1:
        raise ValueError("sweep reports do not share one task set and server binary")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_repetition: dict[int, set[str]] = defaultdict(set)
    for report in reports:
        grouped[report["mode_label"]].append(report)
        by_repetition[int(report["repetition"])].add(report["mode_label"])
    expected_modes = set(grouped)
    if any(modes != expected_modes for modes in by_repetition.values()):
        raise ValueError("sweep repetition is incomplete")
    repetition_counts = {len(items) for items in grouped.values()}
    if len(repetition_counts) != 1:
        raise ValueError("sweep mode repetition counts differ")

    runtime_fields = (
        "weighted_acceptance_rate",
        "verified_tokens_per_target_pass",
        "fallback_replays",
        "rollback_guard_requests",
        "rollback_guard_activations",
        "aggregate_reported_tokens_per_second",
        "target_only_requests",
        "target_only_tokens",
        "fr_target_token_id_prefix_fraction",
        "fr_correction_outside_token_id_prefix_fraction",
    )
    modes: dict[str, Any] = {}
    for mode, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["repetition"])
        summaries = [item["summary"]["overall"] for item in items]
        task_predictions: dict[str, list[Any]] = defaultdict(list)
        for item in items:
            for row in item["results"]:
                task_predictions[row["id"]].append(row["prediction"])
        mode_summary: dict[str, Any] = {
            "model_sha256": items[0]["model_sha256"],
            "request": items[0].get("request"),
            "health": items[0].get("health"),
            "runs": [
                {
                    "repetition": item["repetition"],
                    "order_index": item["order_index"],
                    "summary": item["summary"],
                    "speculative_runtime": item.get("speculative_runtime"),
                }
                for item in items
            ],
            "accuracy": describe(summary["accuracy"] for summary in summaries),
            "strict_accuracy": describe(
                summary["strict_accuracy"] for summary in summaries
            ),
            "aggregate_completion_tokens_per_second": describe(
                summary["aggregate_completion_tokens_per_second"]
                for summary in summaries
            ),
            "median_latency_seconds": describe(
                summary["median_latency_seconds"] for summary in summaries
            ),
            "prediction_stable_tasks": sum(
                len(set(values)) == 1 for values in task_predictions.values()
            ),
            "task_count": len(items[0]["results"]),
        }
        runtime_reports = [
            item["speculative_runtime"]
            for item in items
            if item.get("speculative_runtime") is not None
        ]
        if runtime_reports:
            runtime = [report["overall"] for report in runtime_reports]
            mode_summary["speculative_runtime"] = {
                "strategy": one_speculative_strategy(runtime_reports),
                **{
                    field: describe(report.get(field) for report in runtime)
                    for field in runtime_fields
                },
            }
        modes[mode] = mode_summary

    return {
        "schema": "a3s.power.quality-eval.sweep.v1",
        "created_at": utc_now(),
        "tasks_sha256": next(iter(task_hashes)),
        "server_sha256": next(iter(server_hashes)),
        "repetitions": next(iter(repetition_counts)),
        "modes": modes,
    }


def render_sweep_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.8-27B MTP calibration sweep",
        "",
        f"Task SHA-256: \x60{aggregate['tasks_sha256']}\x60. Repetitions: "
        f"{aggregate['repetitions']} per mode.",
        "",
        "| Mode | Workload token/s | Acceptance | Tokens / target pass | Target-only requests | Target token-ID prefix | Correction outside prefix | Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, summary in aggregate["modes"].items():
        runtime = summary.get("speculative_runtime", {})

        def mean(field: str) -> float | None:
            return runtime.get(field, {}).get("mean")

        def percent(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.1%}"

        def decimal(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        lines.append(
            f"| \x60{mode}\x60 | "
            f"{decimal(summary['aggregate_completion_tokens_per_second']['mean'])} | "
            f"{percent(mean('weighted_acceptance_rate'))} | "
            f"{decimal(mean('verified_tokens_per_target_pass'))} | "
            f"{decimal(mean('target_only_requests'))} | "
            f"{percent(mean('fr_target_token_id_prefix_fraction'))} | "
            f"{percent(mean('fr_correction_outside_token_id_prefix_fraction'))} | "
            f"{summary['accuracy']['mean']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Token-ID prefix fields are exact only for the legacy prefix shortlist. They are "
            "diagnostics, not ranked d2t vocabulary membership or FR-caused rejection rates.",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(aggregate: dict[str, Any]) -> str:
    labels = {
        "q6-off": "Untouched Q6_K, speculation off",
        "tbq4-off": "TBQ4 mixed artifact, speculation off",
        "tbq4-mtp-fr": "TBQ4 + MTP + FR",
        "tbq4-mtp-full-vocab": "TBQ4 + full-vocabulary MTP",
        "q6-dspark": "Untouched Q6_K + DSpark Q4",
    }
    lines = [
        "# Qwen3.8-27B quality and workload-throughput matrix",
        "",
        f"Task SHA-256: \x60{aggregate['tasks_sha256']}\x60. Repetitions: "
        f"{aggregate['repetitions']} per mode.",
        "",
        "| Mode | Mean accuracy | Mean strict accuracy | Median run latency | Mean workload throughput | Stable answers |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, metrics in aggregate["modes"].items():
        lines.append(
            f"| {labels.get(mode, mode)} | {metrics['accuracy']['mean']:.1%} | "
            f"{metrics['strict_accuracy']['mean']:.1%} | "
            f"{metrics['median_latency_seconds']['median']:.3f} s | "
            f"{metrics['aggregate_completion_tokens_per_second']['mean']:.3f} token/s | "
            f"{metrics['prediction_stable_tasks']}/{metrics['task_count']} |"
        )
    lines.extend(
        [
            "",
            "Accuracy uses the deterministic lenient extractor; strict accuracy requires an "
            "explicit \x60FINAL:\x60 or \x60最终答案:\x60 marker. Workload throughput is completion tokens "
            "divided by full request latency and is not the repetitive-prompt peak benchmark.",
            "",
        ]
    )
    if aggregate["paired_runs"]:
        lines.extend(
            [
                "## Paired comparisons",
                "",
                "| Pair | Repetition | Score gains / losses | Strict gains / losses | Exact output hashes |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair, runs in aggregate["paired_runs"].items():
            for repetition, metrics in enumerate(runs, start=1):
                lines.append(
                    f"| {pair} | {repetition} | {metrics['gains']} / "
                    f"{metrics['losses']} | {metrics['strict_gains']} / "
                    f"{metrics['strict_losses']} | "
                    f"{metrics['content_sha256_parity']}/{metrics['task_count']} |"
                )
        lines.append("")
    return "\n".join(lines)
