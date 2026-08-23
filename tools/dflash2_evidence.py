#!/usr/bin/env python3
"""Package and verify the pinned Q6_K-only DFlash2 benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dflash2_evidence_contract import (
    CANDIDATE_MODE,
    EXPECTED_DRAFT,
    EXPECTED_REPORT_HASHES,
    EXPECTED_RUNTIME_FILES,
    EXPECTED_SECTION_DIGESTS,
    EXPECTED_SOURCE,
    EXPECTED_TARGET,
    EXPECTED_TASK_PAIRS_SHA256,
    EXPECTED_TOOLS,
    SCHEMA,
    TARGET_MODE,
)
from qwen38_quality_report import is_truncated, pair_metrics


class EvidenceError(ValueError):
    """Raised when the pinned evidence contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def report_path(root: Path, name: str) -> Path:
    require(Path(name).name == name, f"report name is not path-free: {name}")
    path = (root / name).resolve()
    require(path.parent == root.resolve(), f"report escapes capture root: {name}")
    require(path.is_file(), f"report is missing: {name}")
    return path


def vector(rows: list[dict[str, Any]], *, content: bool) -> list[dict[str, Any]]:
    if content:
        return [
            {"id": row["id"], "content_sha256": row["content_sha256"]}
            for row in rows
        ]
    return [
        {
            "id": row["id"],
            "prediction": row["prediction"],
            "strict_prediction": row["strict_prediction"],
            "correct": row["correct"],
            "strict_correct": row["strict_correct"],
        }
        for row in rows
    ]


def compact_runtime(runtime: dict[str, Any] | None) -> dict[str, Any] | None:
    if runtime is None:
        return None
    return {
        "source": runtime["source"],
        "strategy": runtime["strategy"],
        "overall": {
            field: runtime["overall"].get(field)
            for field in (
                "requests",
                "drafted_tokens",
                "accepted_tokens",
                "weighted_acceptance_rate",
                "verified_tokens_per_target_pass",
                "fallback_replays",
                "median_reported_tokens_per_second",
                "aggregate_reported_tokens_per_second",
            )
        },
    }


def compact_run(name: str, digest: str, report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]["overall"]
    return {
        "file": name,
        "sha256": digest,
        "repetition": report["repetition"],
        "order_index": report["order_index"],
        "request_sha256": report["request_sha256"],
        "completed": summary["completed"],
        "errors": summary["errors"],
        "correct": summary["correct"],
        "strict_correct": summary["strict_correct"],
        "truncated": summary["truncated"],
        "median_latency_seconds": summary["median_latency_seconds"],
        "completion_tokens": summary["completion_tokens"],
        "workload_tokens_per_second": summary[
            "aggregate_completion_tokens_per_second"
        ],
        "content_vector_sha256": canonical_digest(
            vector(report["results"], content=True)
        ),
        "prediction_vector_sha256": canonical_digest(
            vector(report["results"], content=False)
        ),
        "speculative_runtime": compact_runtime(report.get("speculative_runtime")),
    }


def compact_task_pairs(
    base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = {row["id"]: row for row in candidate_rows}
    require(set(candidates) == {row["id"] for row in base_rows}, "task IDs differ")

    def side(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prediction": row["prediction"],
            "strict_prediction": row["strict_prediction"],
            "correct": row["correct"],
            "strict_correct": row["strict_correct"],
            "content_sha256": row["content_sha256"],
            "truncated": is_truncated(row),
        }

    return [
        {
            "id": row["id"],
            "benchmark": row["benchmark"],
            "expected": row["expected"],
            "target": side(row),
            "dflash2": side(candidates[row["id"]]),
        }
        for row in base_rows
    ]


def compact_admission(admission: dict[str, Any]) -> dict[str, Any]:
    return {
        "elapsed_milliseconds": admission["elapsed_milliseconds"],
        "observed_sample_count": admission["observed_sample_count"],
        "accepted_samples": [
            {
                "gpu_index": sample["gpu_index"],
                "utilization_percent": sample["utilization_percent"],
                "memory_free_mib": sample["memory_free_mib"],
            }
            for sample in admission["accepted_samples"]
        ],
    }


def compact_hardware(environment: dict[str, Any]) -> dict[str, Any]:
    fields = [field.strip() for field in environment["gpu"].split(",")]
    require(len(fields) == 8, "unexpected NVIDIA identity shape")
    return {
        "gpu": {
            "name": fields[0],
            "driver": fields[2],
            "memory_total_mib": int(fields[3]),
            "compute_capability": fields[4],
            "power_limit_watts": float(fields[6]),
            "maximum_graphics_clock_mhz": int(fields[7]),
        },
        "cpu": environment["cpu"],
        "os": environment["os"],
    }


def compact_performance_mode(mode: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": mode["name"],
        "status": mode["status"],
        "admission": compact_admission(mode["admission"]),
        "process": mode["process"],
        "memory": mode["memory"],
        "samples": mode["samples"],
        "summary": mode["summary"],
        "logs": mode["logs"],
    }


def compact_mode_aggregate(mode: dict[str, Any]) -> dict[str, Any]:
    compact = {
        field: mode[field]
        for field in (
            "accuracy",
            "strict_accuracy",
            "truncated",
            "median_latency_seconds",
            "aggregate_completion_tokens_per_second",
            "prediction_stable_tasks",
            "content_stable_tasks",
            "task_count",
            "by_benchmark",
        )
    }
    if "speculative_runtime" in mode:
        compact["speculative_runtime"] = mode["speculative_runtime"]
    return compact


def build_evidence(performance_path: Path, quality_root: Path) -> dict[str, Any]:
    performance = load_json(performance_path)
    environment_path = quality_root / "environment.json"
    aggregate_path = quality_root / "quality-matrix.json"
    environment = load_json(environment_path)
    aggregate = load_json(aggregate_path)

    require(
        sha256_file(performance_path)
        == EXPECTED_SOURCE["performance_report_sha256"],
        "performance report hash differs",
    )
    require(
        sha256_file(environment_path)
        == EXPECTED_SOURCE["quality_environment_sha256"],
        "quality environment hash differs",
    )
    require(
        sha256_file(aggregate_path)
        == EXPECTED_SOURCE["quality_aggregate_sha256"],
        "quality aggregate hash differs",
    )
    require(
        performance["schema"]
        == "a3s.power.llamacpp-external-draft-benchmark.v1",
        "performance schema differs",
    )
    require(
        environment["schema"]
        == "a3s.power.llamacpp-external-draft-quality.environment.v1",
        "environment schema differs",
    )
    require(
        aggregate["schema"] == "a3s.power.quality-eval.aggregate.v1",
        "aggregate schema differs",
    )
    require(
        not performance["identity"]["dirty_worktree"],
        "performance worktree was dirty",
    )
    require(
        not environment["identity"]["dirty_worktree"],
        "quality worktree was dirty",
    )
    performance_identity = performance["identity"]
    quality_identity = environment["identity"]
    require(
        performance_identity["power_commit"] == EXPECTED_SOURCE["power_commit"],
        "performance commit differs",
    )
    require(
        performance_identity["llama_cpp_commit"]
        == EXPECTED_SOURCE["llama_cpp_commit"],
        "llama.cpp commit differs",
    )
    require(
        performance_identity["llama_server_sha256"]
        == EXPECTED_SOURCE["llama_server_sha256"],
        "performance server hash differs",
    )
    target = performance_identity["target"]
    require(
        {
            "file": target["file"],
            "bytes": target["size"],
            "sha256": target["sha256"],
            "quantization": "Q6_K",
        }
        == EXPECTED_TARGET,
        "performance target is not the pinned Q6_K artifact",
    )
    draft = performance_identity["draft"]
    require(
        {
            "file": draft["file"],
            "bytes": draft["size"],
            "sha256": draft["sha256"],
            "strategy": draft["mode"],
            "backend_mode": draft["backend_mode"],
            "role": "auxiliary-proposer-only",
        }
        == EXPECTED_DRAFT,
        "performance proposer identity differs",
    )
    require(
        quality_identity["power_commit"] == EXPECTED_SOURCE["power_commit"],
        "quality commit differs",
    )
    require(
        quality_identity["llama_cpp_commit"]
        == EXPECTED_SOURCE["llama_cpp_commit"],
        "quality llama.cpp commit differs",
    )
    require(
        quality_identity["target"]["sha256"] == EXPECTED_TARGET["sha256"]
        and quality_identity["target"]["quantization"] == "Q6_K",
        "quality target identity differs",
    )
    require(
        quality_identity["draft"]["sha256"] == EXPECTED_DRAFT["sha256"]
        and quality_identity["draft"]["mode"] == "dflash2",
        "quality proposer identity differs",
    )
    runtime_hashes = {
        row["file"]: row["sha256"]
        for row in quality_identity["llama_runtime_files"]
    }
    require(
        runtime_hashes == EXPECTED_RUNTIME_FILES,
        "quality runtime identity differs",
    )
    require(
        aggregate["server_sha256"] == EXPECTED_SOURCE["llama_server_sha256"],
        "quality aggregate server hash differs",
    )

    reports: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for receipt in aggregate["reports"]:
        name = receipt["path"]
        path = report_path(quality_root, name)
        digest = sha256_file(path)
        require(digest == receipt["sha256"], f"aggregate report hash differs: {name}")
        reports[name] = load_json(path)
        hashes[name] = digest
    require(hashes == EXPECTED_REPORT_HASHES, "quality report set differs")

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for name, report in reports.items():
        require(
            report["schema"] == "a3s.power.quality-eval.report.v3",
            "quality report schema differs",
        )
        require(
            report["model_sha256"] == EXPECTED_TARGET["sha256"],
            "quality target is not the pinned Q6_K model",
        )
        require(
            report["power_commit"] == EXPECTED_SOURCE["power_commit"],
            "quality commit differs",
        )
        grouped[report["mode_label"]].append((name, report))
    require(set(grouped) == {TARGET_MODE, CANDIDATE_MODE}, "quality mode set differs")
    for items in grouped.values():
        items.sort(key=lambda item: item[1]["repetition"])
        require(len(items) == 3, "quality mode must have three repetitions")

    base = {report["repetition"]: report for _, report in grouped[TARGET_MODE]}
    candidate = {report["repetition"]: report for _, report in grouped[CANDIDATE_MODE]}
    paired = [pair_metrics(base[index], candidate[index]) for index in range(1, 4)]
    require(
        paired
        == aggregate["paired_runs"][f"{TARGET_MODE} -> {CANDIDATE_MODE}"],
        "paired metrics differ",
    )
    task_pairs = compact_task_pairs(base[1]["results"], candidate[1]["results"])

    mode_evidence = {}
    for mode in (TARGET_MODE, CANDIDATE_MODE):
        mode_evidence[mode] = {
            "aggregate": compact_mode_aggregate(aggregate["modes"][mode]),
            "runs": [
                compact_run(name, hashes[name], report)
                for name, report in grouped[mode]
            ],
        }

    target_rate = aggregate["modes"][TARGET_MODE][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    candidate_rate = aggregate["modes"][CANDIDATE_MODE][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    quality_environment = environment["environment"]
    runtime_files = environment["identity"]["llama_runtime_files"]
    return {
        "schema": SCHEMA,
        "captured_at": performance["created_at"],
        "policy": {
            "target_model_fixed": True,
            "target_quantization": "Q6_K",
            "draft_is_target": False,
            "draft_role": "auxiliary-proposer-only",
            "q4_target_results_included": False,
        },
        "source": {
            **EXPECTED_SOURCE,
            "dirty_worktree": False,
            "backend_source": "standalone-exact-upstream-llama.cpp-dflash2-pr",
            "native_power_backend_available": False,
            "runtime_files": runtime_files,
            "tools": EXPECTED_TOOLS,
            "quality_reports": aggregate["reports"],
        },
        "artifacts": {
            "target": EXPECTED_TARGET,
            "draft": EXPECTED_DRAFT,
            "prompt": {
                "file": performance["identity"]["prompt"]["file"],
                "bytes": performance["identity"]["prompt"]["size"],
                "sha256": performance["identity"]["prompt"]["sha256"],
            },
        },
        "hardware": compact_hardware(quality_environment),
        "controls": {
            "process_priority": quality_environment["process_priority"],
            "process_affinity": quality_environment["process_affinity"],
            "gpu": quality_environment["gpu_controls"],
            "cuda_runtime": quality_environment["cuda_runtime"],
            "power_scheme_guid": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
        },
        "performance": {
            "configuration": performance["configuration"],
            "baseline": compact_performance_mode(performance["baseline"]),
            "candidate": compact_performance_mode(performance["candidate"]),
            "comparison": performance["comparison"],
        },
        "quality": {
            "configuration": environment["configuration"],
            "workload": environment["workload"],
            "request": next(iter(reports.values()))["request"],
            "modes": mode_evidence,
            "paired_runs": paired,
            "task_pairs": task_pairs,
            "gpu_admissions": [
                {"run": item["run"], **compact_admission(item["admission"])}
                for item in environment["gpu_admissions"]
            ],
            "processes": environment["processes"],
        },
        "claim": {
            "classification": "experimental-output-divergence",
            "peak_prompt_speedup": performance["comparison"]["throughput_speedup"],
            "quality_workload_speedup": candidate_rate / target_rate,
            "accuracy_delta": 0.0,
            "strict_accuracy_delta": 0.0,
            "answer_parity": paired[0]["prediction_parity"],
            "complete_output_parity": paired[0]["content_sha256_parity"],
            "task_count": paired[0]["task_count"],
            "stable_175_tokens_per_second_demonstrated": False,
            "production_default_eligible": False,
            "boundary": (
                "One RTX 4090; three execution repetitions over one fixed "
                "12-task calibration are not independent quality samples."
            ),
        },
    }


def derived_pair_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "task_count": len(rows),
        "base_correct": sum(row["target"]["correct"] for row in rows),
        "candidate_correct": sum(row["dflash2"]["correct"] for row in rows),
        "gains": sum(not row["target"]["correct"] and row["dflash2"]["correct"] for row in rows),
        "losses": sum(row["target"]["correct"] and not row["dflash2"]["correct"] for row in rows),
        "strict_base_correct": sum(row["target"]["strict_correct"] for row in rows),
        "strict_candidate_correct": sum(row["dflash2"]["strict_correct"] for row in rows),
        "strict_gains": sum(
            not row["target"]["strict_correct"]
            and row["dflash2"]["strict_correct"]
            for row in rows
        ),
        "strict_losses": sum(
            row["target"]["strict_correct"]
            and not row["dflash2"]["strict_correct"]
            for row in rows
        ),
        "prediction_parity": sum(
            row["target"]["prediction"] == row["dflash2"]["prediction"]
            for row in rows
        ),
        "content_sha256_parity": sum(
            row["target"]["content_sha256"]
            == row["dflash2"]["content_sha256"]
            for row in rows
        ),
        "both_untruncated": sum(
            not row["target"]["truncated"]
            and not row["dflash2"]["truncated"]
            for row in rows
        ),
        "both_untruncated_prediction_parity": sum(
            not row["target"]["truncated"]
            and not row["dflash2"]["truncated"]
            and row["target"]["prediction"] == row["dflash2"]["prediction"]
            for row in rows
        ),
    }


def verify_evidence(evidence: dict[str, Any], require_production_default: bool) -> dict[str, Any]:
    require(evidence.get("schema") == SCHEMA, "evidence schema differs")
    encoded = json.dumps(evidence, ensure_ascii=False)
    require(
        re.search(r"[A-Za-z]:\\\\", encoded) is None,
        "evidence contains an absolute Windows path",
    )
    require("GPU-" not in encoded, "evidence contains a GPU UUID")
    require(evidence["policy"] == {
        "target_model_fixed": True,
        "target_quantization": "Q6_K",
        "draft_is_target": False,
        "draft_role": "auxiliary-proposer-only",
        "q4_target_results_included": False,
    }, "Q6_K-only policy differs")

    source = evidence["source"]
    require(
        all(source[key] == value for key, value in EXPECTED_SOURCE.items()),
        "pinned source identity differs",
    )
    require(not source["dirty_worktree"], "capture used a dirty worktree")
    require(
        not source["native_power_backend_available"],
        "standalone capture was mislabeled as native Power support",
    )
    require(source["tools"] == EXPECTED_TOOLS, "capture tool hashes differ")
    runtime_hashes = {row["file"]: row["sha256"] for row in source["runtime_files"]}
    require(runtime_hashes == EXPECTED_RUNTIME_FILES, "runtime file hashes differ")
    observed_reports = {row["path"]: row["sha256"] for row in source["quality_reports"]}
    require(observed_reports == EXPECTED_REPORT_HASHES, "quality report hashes differ")
    require(
        evidence["artifacts"]["target"] == EXPECTED_TARGET,
        "target is not the pinned Q6_K artifact",
    )
    require(evidence["artifacts"]["draft"] == EXPECTED_DRAFT, "DFlash2 proposer identity differs")

    performance = evidence["performance"]
    require(performance["configuration"] == {
        "samples": 3, "warmup_runs": 1, "max_tokens": 256,
        "context_size": 512, "batch_size": 12, "threads": 10,
        "draft_max": 7, "target_gpu_layers": "all", "draft_gpu_layers": "all",
        "flash_attention": True, "fit": False, "parallel_slots": 1,
        "seed": 42, "greedy": True, "server_verbosity": 3,
    }, "performance configuration differs")
    base_rates = [
        sample["predicted_tokens_per_second"]
        for sample in performance["baseline"]["samples"]
    ]
    candidate_rates = [
        sample["predicted_tokens_per_second"]
        for sample in performance["candidate"]["samples"]
    ]
    require(len(base_rates) == len(candidate_rates) == 3, "performance sample count differs")
    require(close(statistics.median(base_rates), 35.379557441971976), "baseline median differs")
    require(
        close(statistics.median(candidate_rates), 108.42887830963208),
        "DFlash2 median differs",
    )
    require(close(min(candidate_rates), 103.74182524587827), "DFlash2 minimum differs")
    require(
        performance["candidate"]["summary"]["accepted_draft_tokens"] == 666,
        "peak accepted-token count differs",
    )
    require(
        performance["candidate"]["summary"]["draft_tokens"] == 678,
        "peak drafted-token count differs",
    )
    require(performance["comparison"]["deterministic_output_parity"], "peak output parity differs")
    require(
        close(
            statistics.median(candidate_rates) / statistics.median(base_rates),
            performance["comparison"]["throughput_speedup"],
        ),
        "peak speedup differs",
    )

    quality = evidence["quality"]
    require(quality["workload"]["task_count"] == 12, "quality task count differs")
    require(quality["workload"]["repetitions"] == 3, "quality repetition count differs")
    require(quality["workload"]["total_requests"] == 72, "quality request count differs")
    pairs = quality["task_pairs"]
    require(len({row["id"] for row in pairs}) == 12, "quality task IDs are not unique")
    require(
        Counter(row["benchmark"] for row in pairs)
        == {"mmlu": 4, "gsm8k": 4, "ceval": 4},
        "quality benchmark mix differs",
    )
    require(
        canonical_digest(pairs) == EXPECTED_TASK_PAIRS_SHA256,
        "quality task-pair vector differs",
    )
    derived = derived_pair_metrics(pairs)
    expected_pair = {
        "task_count": 12, "base_correct": 9, "candidate_correct": 9,
        "gains": 0, "losses": 0, "strict_base_correct": 9,
        "strict_candidate_correct": 9, "strict_gains": 0, "strict_losses": 0,
        "prediction_parity": 12, "content_sha256_parity": 7,
        "both_untruncated": 9, "both_untruncated_prediction_parity": 9,
    }
    require(derived == expected_pair, "derived paired quality metrics differ")
    require(len(quality["paired_runs"]) == 3, "paired repetition count differs")
    for recorded in quality["paired_runs"]:
        require(
            all(recorded[key] == value for key, value in expected_pair.items()),
            "recorded paired metrics differ",
        )

    expected_rates = {
        TARGET_MODE: [28.56869930629077, 30.404722459260746, 30.13375516135641],
        CANDIDATE_MODE: [54.586265499597836, 33.68860665345892, 47.155213335427824],
    }
    for mode in (TARGET_MODE, CANDIDATE_MODE):
        runs = quality["modes"][mode]["runs"]
        require(len(runs) == 3, f"{mode} run count differs")
        require(
            [run["sha256"] for run in runs]
            == [EXPECTED_REPORT_HASHES[run["file"]] for run in runs],
            f"{mode} compact report hashes differ",
        )
        rates = [run["workload_tokens_per_second"] for run in runs]
        require(
            all(
                close(left, right)
                for left, right in zip(rates, expected_rates[mode])
            ),
            f"{mode} throughput samples differ",
        )
        require(
            all(
                run["completed"] == 12 and run["errors"] == 0
                for run in runs
            ),
            f"{mode} has incomplete runs",
        )
        require(
            all(
                run["correct"] == 9 and run["strict_correct"] == 9
                for run in runs
            ),
            f"{mode} quality score differs",
        )
        aggregate_rate = quality["modes"][mode]["aggregate"][
            "aggregate_completion_tokens_per_second"
        ]
        require(
            close(statistics.mean(rates), aggregate_rate["mean"]),
            f"{mode} mean throughput differs",
        )
        require(
            close(statistics.median(rates), aggregate_rate["median"]),
            f"{mode} median throughput differs",
        )
        side = "target" if mode == TARGET_MODE else "dflash2"
        expected_content = canonical_digest(
            [
                {
                    "id": row["id"],
                    "content_sha256": row[side]["content_sha256"],
                }
                for row in pairs
            ]
        )
        expected_predictions = canonical_digest(
            [
                {
                    "id": row["id"],
                    "prediction": row[side]["prediction"],
                    "strict_prediction": row[side]["strict_prediction"],
                    "correct": row[side]["correct"],
                    "strict_correct": row[side]["strict_correct"],
                }
                for row in pairs
            ]
        )
        require(
            all(
                run["content_vector_sha256"] == expected_content
                for run in runs
            ),
            f"{mode} content stability differs",
        )
        require(
            all(
                run["prediction_vector_sha256"] == expected_predictions
                for run in runs
            ),
            f"{mode} prediction stability differs",
        )
        if mode == TARGET_MODE:
            require(
                all(run["speculative_runtime"] is None for run in runs),
                "target-only telemetry contains speculation",
            )
        else:
            for run in runs:
                runtime = run["speculative_runtime"]
                require(runtime["strategy"] == "dflash2", "candidate runtime strategy differs")
                require(
                    runtime["overall"]["requests"] == 12,
                    "candidate runtime request count differs",
                )
                require(
                    runtime["overall"]["drafted_tokens"] == 3246,
                    "candidate drafted-token count differs",
                )
                require(
                    runtime["overall"]["accepted_tokens"] == 1754,
                    "candidate accepted-token count differs",
                )
                require(
                    close(
                        runtime["overall"]["weighted_acceptance_rate"],
                        0.5403573629081947,
                    ),
                    "candidate acceptance differs",
                )
                require(
                    runtime["overall"]["fallback_replays"] == 0,
                    "candidate fallback count differs",
                )

    target_mean = statistics.mean(expected_rates[TARGET_MODE])
    candidate_mean = statistics.mean(expected_rates[CANDIDATE_MODE])
    claim = evidence["claim"]
    require(
        claim["classification"] == "experimental-output-divergence",
        "claim classification differs",
    )
    require(
        close(
            claim["quality_workload_speedup"], candidate_mean / target_mean
        ),
        "quality speedup differs",
    )
    require(
        claim["answer_parity"] == 12
        and claim["complete_output_parity"] == 7,
        "claim parity differs",
    )
    require(
        not claim["stable_175_tokens_per_second_demonstrated"],
        "175 token/s was not demonstrated",
    )
    require(
        not claim["production_default_eligible"],
        "experimental result was mislabeled as production",
    )
    for section, digest in EXPECTED_SECTION_DIGESTS.items():
        require(
            canonical_digest(evidence[section]) == digest,
            f"{section} evidence digest differs",
        )
    if require_production_default:
        raise EvidenceError(
            "production-default gate failed: complete output parity is 7/12 "
            "and native Power DFlash2 execution is unavailable"
        )
    return {
        "status": "passed",
        "target": "Qwen3.8-27B Q6_K",
        "quality_score": "9/12 in both modes",
        "answer_parity": "12/12",
        "complete_output_parity": "7/12",
        "target_workload_tokens_per_second": target_mean,
        "dflash2_workload_tokens_per_second": candidate_mean,
        "workload_speedup": candidate_mean / target_mean,
        "peak_median_tokens_per_second": statistics.median(candidate_rates),
        "production_default_eligible": False,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Package the pinned raw captures")
    capture.add_argument("--performance-report", type=Path, required=True)
    capture.add_argument("--quality-root", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify the checked-in path-free evidence")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--require-production-default", action="store_true")
    verify.add_argument("--json", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "capture":
            evidence = build_evidence(
                args.performance_report.resolve(), args.quality_root.resolve()
            )
            verify_evidence(evidence, require_production_default=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"wrote {args.output} "
                f"({len(evidence['quality']['task_pairs'])} paired tasks)"
            )
            return 0
        result = verify_evidence(load_json(args.evidence), args.require_production_default)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                "DFlash2 Q6_K-only evidence: PASS "
                "(experimental; not production-default eligible)"
            )
        return 0
    except (EvidenceError, KeyError, TypeError, ValueError) as error:
        print(f"DFlash2 Q6_K-only evidence: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
