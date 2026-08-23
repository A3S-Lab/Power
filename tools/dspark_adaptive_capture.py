"""Raw-capture compaction for adaptive DSpark evidence."""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from dspark_adaptive_evidence_core import SCHEMA, load_json, require, sha256_file
from qwen38_quality_report import is_truncated, pair_metrics


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]["overall"]
    return {
        field: summary[field]
        for field in (
            "total",
            "completed",
            "errors",
            "correct",
            "accuracy",
            "strict_correct",
            "strict_accuracy",
            "truncated",
            "completion_tokens",
            "total_latency_seconds",
            "aggregate_completion_tokens_per_second",
        )
    }


def compact_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    overall = runtime["overall"]
    return {
        "strategy": runtime["strategy"],
        "overall": {
            field: overall[field]
            for field in (
                "requests",
                "drafted_tokens",
                "accepted_tokens",
                "weighted_acceptance_rate",
                "verified_tokens_per_target_pass",
                "fallback_replays",
                "rollback_guard_requests",
                "rollback_guard_activations",
                "target_only_requests",
                "target_only_tokens",
                "draft_limit_histogram",
                "median_reported_tokens_per_second",
                "aggregate_reported_tokens_per_second",
            )
        },
    }


def build_task_pairs(
    base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    require(
        [row["id"] for row in base_rows]
        == [row["id"] for row in candidate_rows],
        "paired task order differs",
    )
    return [
        {
            "id": base["id"],
            "benchmark": base["benchmark"],
            "base_prediction": base["prediction"],
            "candidate_prediction": candidate["prediction"],
            "base_strict_prediction": base["strict_prediction"],
            "candidate_strict_prediction": candidate["strict_prediction"],
            "base_correct": base["correct"],
            "candidate_correct": candidate["correct"],
            "base_strict_correct": base["strict_correct"],
            "candidate_strict_correct": candidate["strict_correct"],
            "base_truncated": is_truncated(base),
            "candidate_truncated": is_truncated(candidate),
            "content_match": base["content_sha256"]
            == candidate["content_sha256"],
        }
        for base, candidate in zip(base_rows, candidate_rows, strict=True)
    ]


def parse_runtime_value(line: str, name: str, value_type: type) -> Any:
    match = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", line)
    require(match is not None, f"peak runtime metric is missing: {name}")
    return value_type(match.group(1))


def parse_peak_runtime(path: Path, measured_runs: int) -> list[dict[str, Any]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if "speculative completion finished" in line
    ]
    require(
        len(lines) >= measured_runs,
        "peak server log contains too few runtime metric rows",
    )
    result = []
    for line in lines[-measured_runs:]:
        result.append(
            {
                name: parse_runtime_value(line, name, value_type)
                for name, value_type in (
                    ("rounds", int),
                    ("drafted_tokens", int),
                    ("accepted_tokens", int),
                    ("emitted_tokens", int),
                    ("verified_emitted_tokens", int),
                    ("acceptance_rate", float),
                    ("tokens_per_target_pass", float),
                    ("fallback_replays", int),
                    ("rollback_guard_activations", int),
                    ("target_only_tokens", int),
                    ("max_rejected_suffix", int),
                )
            }
        )
    return result


def compact_admissions(environment: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "run": item["run"],
            "elapsed_milliseconds": item["admission"]["elapsed_milliseconds"],
            "observed_sample_count": item["admission"]["observed_sample_count"],
            "accepted_samples": [
                {
                    "gpu_index": sample["gpu_index"],
                    "utilization_percent": sample["utilization_percent"],
                    "memory_free_mib": sample["memory_free_mib"],
                }
                for sample in item["admission"]["accepted_samples"]
            ],
        }
        for item in environment["gpu_admissions"]
    ]


def build_evidence(
    quality_root: Path,
    peak_report_path: Path,
    peak_preflight_path: Path,
    peak_environment_path: Path,
    peak_server_log_path: Path,
) -> dict[str, Any]:
    quality_environment_path = quality_root / "environment.json"
    quality_aggregate_path = quality_root / "quality-matrix.json"
    tasks_path = quality_root / "tasks-v1.json"
    quality_environment = load_json(quality_environment_path)
    quality_aggregate = load_json(quality_aggregate_path)
    peak_report = load_json(peak_report_path)
    peak_preflight = load_json(peak_preflight_path)
    peak_environment = load_json(peak_environment_path)

    require(not quality_environment["dirty_worktree"], "quality tree was dirty")
    require(not peak_environment["dirty_worktree"], "peak tree was dirty")
    require(peak_preflight["passed"], "peak preflight did not pass")
    require(not peak_preflight["dirty_worktree"], "peak preflight tree was dirty")
    require(quality_aggregate["repetitions"] == 1, "quality run count differs")
    require(
        set(quality_aggregate["modes"]) == {"q6-off", "q6-dspark"},
        "mode set differs",
    )

    reports: dict[str, tuple[dict[str, Any], str]] = {}
    for receipt in quality_aggregate["reports"]:
        path = quality_root / receipt["path"]
        digest = sha256_file(path)
        require(digest == receipt["sha256"], "quality report receipt differs")
        report = load_json(path)
        reports[report["mode_label"]] = (report, digest)
    require(set(reports) == {"q6-off", "q6-dspark"}, "quality reports differ")
    base, base_digest = reports["q6-off"]
    candidate, candidate_digest = reports["q6-dspark"]
    paired = pair_metrics(base, candidate)
    require(
        paired == quality_aggregate["paired_runs"]["q6-off -> q6-dspark"][0],
        "raw paired summary differs from aggregate",
    )
    task_pairs = build_task_pairs(base["results"], candidate["results"])

    commits = {
        quality_environment["power_commit"],
        peak_report["identity"]["power_commit"],
        peak_preflight["power_commit"],
        peak_environment["power_commit"],
        base["power_commit"],
        candidate["power_commit"],
    }
    require(len(commits) == 1, "source commits differ")
    server_hashes = {
        quality_environment["server"]["sha256"],
        peak_preflight["server"]["sha256"],
        peak_environment["server"]["sha256"],
        base["server_sha256"],
        candidate["server_sha256"],
    }
    require(len(server_hashes) == 1, "server hashes differ")
    require(
        base["request_sha256"] == candidate["request_sha256"],
        "quality requests differ",
    )
    require(base.get("speculative_runtime") is None, "base used speculation")
    require(
        candidate["speculative_runtime"]["strategy"] == "dspark",
        "candidate strategy differs",
    )

    samples = peak_report["samples"]
    require(len(samples) == 3, "peak sample count differs")
    rates = [float(sample["decode_tokens_per_second"]) for sample in samples]
    require(all(rate >= 160.0 for rate in rates), "peak all-sample gate failed")
    require(
        len({sample["output_sha256"] for sample in samples}) == 1,
        "peak output differs",
    )
    require(
        len({sample["receipt_sha256"] for sample in samples}) == 1,
        "peak receipt differs",
    )

    gpu_fields = [
        field.strip() for field in quality_environment["gpu"][0].split(",")
    ]
    require(len(gpu_fields) == 6, "quality GPU identity differs")
    target = quality_environment["q6_model"]
    draft = target["external_draft"]
    runtime_rows = parse_peak_runtime(peak_server_log_path, len(samples))
    source_hashes = {
        "peak_report_sha256": sha256_file(peak_report_path),
        "peak_preflight_sha256": sha256_file(peak_preflight_path),
        "peak_environment_sha256": sha256_file(peak_environment_path),
        "peak_server_log_sha256": sha256_file(peak_server_log_path),
        "quality_environment_sha256": sha256_file(quality_environment_path),
        "quality_aggregate_sha256": sha256_file(quality_aggregate_path),
        "quality_base_report_sha256": base_digest,
        "quality_candidate_report_sha256": candidate_digest,
    }
    return {
        "schema": SCHEMA,
        "captured_at": candidate["completed_at"],
        "source": {
            "power_commit": next(iter(commits)),
            "dirty_worktree": False,
            "server_sha256": next(iter(server_hashes)),
            "benchmark_client_sha256": peak_environment["benchmark_client"][
                "sha256"
            ],
            **source_hashes,
        },
        "artifacts": {
            "target": {
                "bytes": target["size"],
                "sha256": target["sha256"],
                "file_hash_verified": target["file_hash_verified"],
            },
            "draft": {
                "kind": draft["kind"],
                "bytes": draft["size"],
                "sha256": draft["sha256"],
                "target_sha256": draft["target_sha256"],
                "source": draft["source"],
                "revision": draft["revision"],
                "license": draft["license"],
                "file_hash_verified": draft["file_hash_verified"],
            },
        },
        "hardware": {
            "os": quality_environment["os"],
            "cpu": quality_environment["cpu"],
            "gpu": {
                "name": gpu_fields[0],
                "driver": gpu_fields[1],
                "memory_total_mib": int(gpu_fields[2]),
                "compute_capability": gpu_fields[3],
                "driver_model": "WDDM",
            },
        },
        "host_controls": {
            "peak": {
                "power_scheme": peak_environment["active_power_scheme"],
                "process_priority": peak_environment["process_priority"],
                "process_affinity": peak_environment["process_affinity"],
                "cuda_high_priority": peak_environment["gpu"][
                    "cuda_high_priority"
                ],
                "gpu_clock": {
                    "gpu_index": peak_environment["gpu"]["indices"][0],
                    "requested_mhz": peak_environment["gpu"][
                        "clock_lock_mhz"
                    ],
                    "lock_applied": bool(
                        peak_preflight["gpu"]["clock_lock_applied_indices"]
                    ),
                },
                "maximum_idle_utilization_percent": peak_environment["gpu"][
                    "maximum_idle_utilization_percent"
                ],
                "maximum_observed_idle_utilization_percent": peak_environment[
                    "gpu"
                ]["maximum_observed_idle_utilization_percent"],
            },
            "quality": {
                "power_scheme": quality_environment["active_power_scheme"],
                "process_priority": quality_environment["process_priority"],
                "process_affinity": quality_environment["process_affinity"],
                "cuda_high_priority": quality_environment["host_controls"][
                    "cuda_high_priority"
                ],
                "gpu_clock": quality_environment["host_controls"]["gpu_clock"],
                "gpu_admission": quality_environment["gpu_admission"],
                "admission_windows": compact_admissions(quality_environment),
            },
        },
        "inputs": {
            "config_sha256": quality_environment["config_sha256"],
            "prompt_sha256": peak_report["workload"]["prompt_sha256"],
            "tasks_sha256": quality_aggregate["tasks_sha256"],
            "tasks_file_sha256": sha256_file(tasks_path),
            "task_manifest_sha256": quality_environment["task_manifest_sha256"],
        },
        "peak": {
            "request_sha256": peak_report["workload"]["request_sha256"],
            "workload": peak_report["workload"],
            "warmup_runs": peak_report["warmup_runs"],
            "samples": samples,
            "median_decode_tokens_per_second": statistics.median(rates),
            "minimum_decode_tokens_per_second": min(rates),
            "required_minimum_tokens_per_second": 160.0,
            "all_samples_passed": all(rate >= 160.0 for rate in rates),
            "output_sha256": peak_report["output_sha256"],
            "receipt_sha256": samples[0]["receipt_sha256"],
            "runtime": runtime_rows,
        },
        "quality": {
            "request_sha256": base["request_sha256"],
            "request": base["request"],
            "base": compact_summary(base),
            "candidate": compact_summary(candidate),
            "candidate_runtime": compact_runtime(
                candidate["speculative_runtime"]
            ),
            "paired": paired,
            "task_pairs": task_pairs,
        },
        "claim": {
            "classification": "diagnostic-output-divergence",
            "peak_all_sample_gate_passed": True,
            "workload_throughput_speedup": candidate["summary"]["overall"][
                "aggregate_completion_tokens_per_second"
            ]
            / base["summary"]["overall"][
                "aggregate_completion_tokens_per_second"
            ],
            "accuracy_delta": candidate["summary"]["overall"]["accuracy"]
            - base["summary"]["overall"]["accuracy"],
            "strict_accuracy_delta": candidate["summary"]["overall"][
                "strict_accuracy"
            ]
            - base["summary"]["overall"]["strict_accuracy"],
            "production_default_eligible": False,
            "boundary": "Single-request peak and one deterministic 100-task paired diagnostic on one controlled RTX 4090 host",
        },
    }
