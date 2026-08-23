#!/usr/bin/env python3
"""Package and verify the pinned Q6_K-only Qwen3.8 quality capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen38_quality_report import aggregate_reports, exact_mcnemar_p, is_truncated


SCHEMA = "a3s.power.q6-quality-evidence.v1"
Q6_SHA256 = "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727"
Q6_BYTES = 22_884_408_288
TASKS_SHA256 = "5798257e18b81188749196d34359278dfadf7986776eb2bd66d629cbfc33813c"
MODES = ("q6-off", "q6-mtp-full-vocab")
PAIR = "q6-off -> q6-mtp-full-vocab"

EXPECTED_IDENTITY = {
    "power_commit": "64aef15ddff7232c6261385700c8a912d1ed0963",
    "server_sha256": "dd7bcac62d3b941e40a4928f855e742140cca5f04a2aa5336347035a6d683c9a",
    "tasks_file_sha256": "2fd18d52a2e011692c6ffac2547da41ec8b6594e2d35a8aa40fa8648df96b40a",
    "task_manifest_sha256": "e254af009fbee304bb574515aece7630cfb6ea91fd0fece6c0d9b9f02de44a9d",
    "config_sha256": "c2b3aca41323b6c3d0ae101b0b68764a99e0a9318df565d13612dd718eb11767",
    "raw_environment_sha256": "56383fc160b563a96215fc9d9da4de707b10bfbd1f0b841f8b95922286639d63",
    "raw_aggregate_sha256": "7af64077ed5c897abdf0cb53ab01b9f030fdc3f5ebcf9664ce98cc8879b07fbb",
}
EXPECTED_TOOL_HASHES = {
    "runner_sha256": "d0ea98731762e8bbc4df9344d4a4513023a27ba0bb0bdf2d2a07585753450fde",
    "evaluator_sha256": "27ee973617668f04eb359c708336b94d45920236b168a29c471155ff228827fc",
    "reporter_sha256": "5f676d00dfa16b25e1613de502d34ce379afadff28c05c583c714b964034006b",
    "profile_helper_sha256": "27d889bbfedd7bd1d162579eb09da5861f1e31a8c840f9967880261905061af6",
    "pmon_helper_sha256": "3239b7ff801f8998b30b35ec31091a855b9fc824439f04b5842fd255913933bb",
}
EXPECTED_REPORT_HASHES = {
    "r01-o1-q6-off.json": "9711ef984ebb54db898e31c5d89dbabbe14a73e373af8df1e9894bb8e7923eda",
    "r01-o2-q6-mtp-full-vocab.json": "a95d1ce64f0481fe5572533ecafd5c61dab8a1efa612bc9b9892cae5e777e0aa",
    "r02-o1-q6-mtp-full-vocab.json": "72a8e18071de6db03b056c657a863e2fdeca11cb1d318541364bacc386e1f8b5",
    "r02-o2-q6-off.json": "89770f8dbc88618ece461193a0165898d7da675c72fe17d8605c82d952340466",
    "r03-o1-q6-off.json": "5ec095be3439802b93727f7b1b3e4642da7729271833c0af107a0070cc16b1b5",
    "r03-o2-q6-mtp-full-vocab.json": "c0de33e7815b37cf920038db1c31928aad4c63802ddbb367a326801f5bd20e2a",
}
EXPECTED_PMON_HASHES = {
    "r01-o1-q6-off.nvidia-pmon.log": "8420350161d634ddc57b9173cfb8dd732320e8df4f8e63b047202471d20fa34b",
    "r01-o2-q6-mtp-full-vocab.nvidia-pmon.log": "1a2f2d5ec8ee68afaa448b460a9628838db843b707489e7c230ed7ecae8f2624",
    "r02-o1-q6-mtp-full-vocab.nvidia-pmon.log": "c1cd2597e33fe48d42a5b6cefb72d4629e82612ab8cb74a266f6bba5aede005c",
    "r02-o2-q6-off.nvidia-pmon.log": "2a16f2146375e3273781e62503a1a8d596c038d0bdb08d46f570ef332c7c760e",
    "r03-o1-q6-off.nvidia-pmon.log": "e6b1ddcff270792fe02ad2a39aa606affa653dbd60971400e712ba5214151d26",
    "r03-o2-q6-mtp-full-vocab.nvidia-pmon.log": "b568c02a23cb4f9aa3a9107c8acd43f76f2af19fcce839ac9bd6d5c84ffe74f0",
}
EXPECTED_EVIDENCE_SHA256 = (
    "e1d9da6b8476e2dea7789211c11e88c3f7d3b1296876b029043b86f2eac4ce32"
)


class EvidenceError(ValueError):
    """Raised when the evidence contract is not satisfied."""


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
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_vector(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": row["id"], "content_sha256": row["content_sha256"]}
        for row in rows
    ]


def prediction_vector(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def compact_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": row["prediction"],
        "strict_prediction": row["strict_prediction"],
        "correct": row["correct"],
        "strict_correct": row["strict_correct"],
        "content_sha256": row["content_sha256"],
        "truncated": is_truncated(row),
    }


def compact_runtime(runtime: dict[str, Any] | None) -> dict[str, Any] | None:
    if runtime is None:
        return None
    fields = (
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
        "aggregate_reported_tokens_per_second",
    )
    return {
        "strategy": runtime["strategy"],
        "overall": {field: runtime["overall"].get(field) for field in fields},
    }


def compact_run(
    name: str, digest: str, report: dict[str, Any]
) -> dict[str, Any]:
    summary = report["summary"]["overall"]
    return {
        "file": name,
        "report_sha256": digest,
        "repetition": report["repetition"],
        "order_index": report["order_index"],
        "model_sha256": report["model_sha256"],
        "request_sha256": report["request_sha256"],
        "completed": summary["completed"],
        "errors": summary["errors"],
        "accuracy": summary["accuracy"],
        "strict_accuracy": summary["strict_accuracy"],
        "truncated": summary["truncated"],
        "completion_tokens": summary["completion_tokens"],
        "total_latency_seconds": summary["total_latency_seconds"],
        "workload_tokens_per_second": summary[
            "aggregate_completion_tokens_per_second"
        ],
        "content_vector_sha256": canonical_digest(
            content_vector(report["results"])
        ),
        "prediction_vector_sha256": canonical_digest(
            prediction_vector(report["results"])
        ),
        "speculative_runtime": compact_runtime(report.get("speculative_runtime")),
    }


def compact_mode_aggregate(mode: dict[str, Any]) -> dict[str, Any]:
    fields = (
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
    selected = {field: mode[field] for field in fields}
    if "speculative_runtime" in mode:
        selected["speculative_runtime"] = mode["speculative_runtime"]
    return selected


def check_expected_hashes(actual: dict[str, str], expected: dict[str, str]) -> None:
    require(actual == expected, "raw report or monitor hashes differ")


def validate_capture(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    environment_path = root / "environment.json"
    aggregate_path = root / "quality-matrix.json"
    tasks_path = root / "tasks-v1.json"
    for path in (environment_path, aggregate_path, tasks_path):
        require(path.is_file(), f"capture input is missing: {path.name}")

    environment = load_json(environment_path)
    aggregate = load_json(aggregate_path)
    require(
        sha256_file(environment_path)
        == EXPECTED_IDENTITY["raw_environment_sha256"],
        "raw environment hash differs",
    )
    require(
        sha256_file(aggregate_path) == EXPECTED_IDENTITY["raw_aggregate_sha256"],
        "raw aggregate hash differs",
    )
    require(
        sha256_file(tasks_path) == EXPECTED_IDENTITY["tasks_file_sha256"],
        "task file hash differs",
    )
    require(not environment["dirty_worktree"], "capture used a dirty worktree")
    require(environment["git_status"] == [], "capture git status is not empty")
    require(
        environment["power_commit"] == EXPECTED_IDENTITY["power_commit"],
        "commit differs",
    )
    require(environment["profile"] == "pure-q6", "profile is not pure-q6")
    require(environment["tbq4_model"] is None, "TBQ4 target is present")
    target = environment["q6_model"]
    require(target["sha256"] == Q6_SHA256, "Q6_K target hash differs")
    require(target["size"] == Q6_BYTES, "Q6_K target byte length differs")
    require(target["external_draft"] is None, "Q6_K external draft is present")
    require(target["file_hash_verified"], "Q6_K file hash was not verified")
    require(
        environment["benchmark_tools"] == EXPECTED_TOOL_HASHES,
        "benchmark tool hashes differ",
    )
    require(
        environment["task_manifest_sha256"]
        == EXPECTED_IDENTITY["task_manifest_sha256"],
        "task manifest differs",
    )
    require(
        environment["config_sha256"] == EXPECTED_IDENTITY["config_sha256"],
        "runtime config differs",
    )
    require(environment["repetitions"] == 3, "capture does not contain three repetitions")
    require(
        [mode["label"] for mode in environment["modes"]] == list(MODES),
        "environment mode set or order differs",
    )
    require(
        all(mode["model_sha256"] == Q6_SHA256 for mode in environment["modes"]),
        "environment mixes target models",
    )
    require(
        all(mode["external_draft_sha256"] is None for mode in environment["modes"]),
        "environment contains an auxiliary draft",
    )

    reports: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for receipt in aggregate["reports"]:
        name = receipt["path"]
        require(Path(name).name == name, "report path is not portable")
        path = root / name
        require(path.is_file(), f"report is missing: {name}")
        digest = sha256_file(path)
        require(
            digest == receipt["sha256"],
            f"aggregate report hash differs: {name}",
        )
        report = load_json(path)
        require(
            report["schema"] == "a3s.power.quality-eval.report.v3",
            "report schema differs",
        )
        require(report["mode_label"] in MODES, "report mode is not Q6_K-only")
        require(report["model_sha256"] == Q6_SHA256, "report target hash differs")
        require(
            report["server_sha256"] == EXPECTED_IDENTITY["server_sha256"],
            "report server hash differs",
        )
        require(
            report["power_commit"] == EXPECTED_IDENTITY["power_commit"],
            "report commit differs",
        )
        require(report["tasks_sha256"] == TASKS_SHA256, "report task hash differs")
        require(
            report["health"]["speculative"]["loaded_artifacts"] == [],
            "report loaded an auxiliary artifact",
        )
        summary = report["summary"]["overall"]
        require(
            summary["completed"] == 100 and summary["errors"] == 0,
            "report is incomplete",
        )
        if report["mode_label"] == "q6-off":
            require(
                report["health"]["speculative"]["mode"] == "off",
                "Q6_K control enabled speculation",
            )
            require(
                report["speculative_runtime"] is None,
                "Q6_K control has speculative metrics",
            )
        else:
            require(
                report["health"]["speculative"]["mode"] == "mtp",
                "MTP mode was not active",
            )
            require(
                report["speculative_runtime"]["strategy"] == "mtp",
                "MTP runtime strategy differs",
            )
            require(
                report["speculative_runtime"]["overall"]["fallback_replays"]
                == 0,
                "MTP used fallback replay",
            )
        reports[name] = report
        hashes[name] = digest
    check_expected_hashes(hashes, EXPECTED_REPORT_HASHES)

    recalculated = aggregate_reports(
        list(reports.values()), comparisons=[(MODES[0], MODES[1])]
    )
    for key in (
        "tasks_sha256",
        "server_sha256",
        "repetitions",
        "modes",
        "paired_runs",
    ):
        require(recalculated[key] == aggregate[key], f"aggregate {key} differs")
    require(aggregate["tasks_sha256"] == TASKS_SHA256, "aggregate task hash differs")
    require(
        aggregate["server_sha256"] == EXPECTED_IDENTITY["server_sha256"],
        "aggregate server hash differs",
    )
    require(set(aggregate["modes"]) == set(MODES), "aggregate mixes target modes")

    monitor_hashes: dict[str, str] = {}
    monitor_runs = set()
    for monitor in environment["gpu_monitors"]:
        monitor_runs.add(monitor["run"])
        require(monitor["failure"] is None, "GPU monitor failure is present")
        summary = monitor["summary"]
        require(not summary["interference_detected"], "GPU interference was detected")
        require(
            summary["foreign_processes"] == [],
            "foreign GPU processes are present",
        )
        require(summary["parsed_samples"] > 0, "GPU monitor has no samples")
        log_name = monitor["log"]
        require(Path(log_name).name == log_name, "GPU monitor path is not portable")
        log_path = root / log_name
        require(log_path.is_file(), f"GPU monitor log is missing: {log_name}")
        monitor_hashes[log_name] = sha256_file(log_path)
    check_expected_hashes(monitor_hashes, EXPECTED_PMON_HASHES)
    require(
        monitor_runs == {name.removesuffix(".json") for name in reports},
        "GPU monitor run set differs",
    )
    require(
        environment["gpu_exclusivity"]["required"],
        "continuous GPU exclusivity was disabled",
    )
    require(
        environment["gpu_exclusivity"][
            "maximum_foreign_sm_utilization_percent"
        ]
        == 2,
        "foreign GPU threshold differs",
    )
    require(len(environment["gpu_admissions"]) == 6, "GPU admission count differs")
    return environment, aggregate, reports


def build_evidence(capture_root: Path) -> dict[str, Any]:
    root = capture_root.resolve()
    environment, aggregate, reports = validate_capture(root)
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for name, report in reports.items():
        grouped[report["mode_label"]].append((name, report))
    for items in grouped.values():
        items.sort(key=lambda item: item[1]["repetition"])
        require(len(items) == 3, "mode does not have three reports")

    base = grouped[MODES[0]][0][1]
    candidate = grouped[MODES[1]][0][1]
    require(
        [row["id"] for row in base["results"]]
        == [row["id"] for row in candidate["results"]],
        "paired task order differs",
    )
    task_pairs = [
        {
            "id": left["id"],
            "benchmark": left["benchmark"],
            "expected": left["expected"],
            "base": compact_result(left),
            "mtp": compact_result(right),
        }
        for left, right in zip(base["results"], candidate["results"], strict=True)
    ]
    mode_evidence = {}
    for mode in MODES:
        items = grouped[mode]
        mode_evidence[mode] = {
            "runtime_config": {
                "speculative": items[0][1]["health"]["speculative"],
                "inference": items[0][1]["health"]["inference"],
            },
            "aggregate": compact_mode_aggregate(aggregate["modes"][mode]),
            "runs": [
                compact_run(name, EXPECTED_REPORT_HASHES[name], report)
                for name, report in items
            ],
        }

    monitor_by_run = {item["run"]: item for item in environment["gpu_monitors"]}
    monitors = []
    for log_name, digest in EXPECTED_PMON_HASHES.items():
        run = log_name.removesuffix(".nvidia-pmon.log")
        monitor = monitor_by_run[run]
        monitors.append(
            {
                "run": run,
                "log_sha256": digest,
                "parsed_samples": monitor["summary"]["parsed_samples"],
                "interference_detected": False,
                "foreign_processes": [],
                "failure": None,
            }
        )

    base_rate = aggregate["modes"][MODES[0]][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    mtp_rate = aggregate["modes"][MODES[1]][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    paired = aggregate["paired_runs"][PAIR]
    gpu = [field.strip() for field in environment["gpu"][0].split(",")]
    require(len(gpu) == 6, "GPU identity shape differs")
    return {
        "schema": SCHEMA,
        "captured_at": aggregate["created_at"],
        "source": {
            "power_commit": environment["power_commit"],
            "dirty_worktree": environment["dirty_worktree"],
            "server_sha256": aggregate["server_sha256"],
            "benchmark_tools": environment["benchmark_tools"],
            "raw_environment_sha256": EXPECTED_IDENTITY["raw_environment_sha256"],
            "raw_aggregate_sha256": EXPECTED_IDENTITY["raw_aggregate_sha256"],
            "raw_reports": [
                {"file": name, "sha256": digest}
                for name, digest in EXPECTED_REPORT_HASHES.items()
            ],
        },
        "hardware": {
            "os": "Windows 11",
            "os_version": environment["os"]["Version"],
            "os_build": environment["os"]["BuildNumber"],
            "cpu": environment["cpu"],
            "gpu": {
                "name": gpu[0],
                "driver": gpu[1],
                "memory_total_mib": int(gpu[2]),
                "compute_capability": gpu[3],
                "power_limit_watts": float(gpu[5]),
            },
        },
        "artifacts": {
            "target": {
                "format": "GGUF",
                "quantization": "Q6_K",
                "bytes": Q6_BYTES,
                "sha256": Q6_SHA256,
                "file_hash_verified": True,
            },
            "auxiliary": [],
        },
        "workload": {
            "model": base["model"],
            "profile": "pure-q6",
            "tasks_sha256": TASKS_SHA256,
            "tasks_file_sha256": EXPECTED_IDENTITY["tasks_file_sha256"],
            "task_manifest_sha256": EXPECTED_IDENTITY["task_manifest_sha256"],
            "config_sha256": EXPECTED_IDENTITY["config_sha256"],
            "repetitions": 3,
            "tasks_per_mode_per_repetition": 100,
            "total_requests": 600,
            "request_sha256": base["request_sha256"],
            "request": base["request"],
        },
        "gpu_exclusivity": {
            "provider": environment["gpu_exclusivity"]["provider"],
            "sample_interval_seconds": environment["gpu_exclusivity"][
                "sample_interval_seconds"
            ],
            "maximum_foreign_sm_utilization_percent": 2,
            "monitors": monitors,
        },
        "modes": mode_evidence,
        "paired_runs": paired,
        "task_pairs": task_pairs,
        "claim": {
            "classification": "q6-only-quality-diagnostic",
            "workload_throughput_speedup": mtp_rate / base_rate,
            "accuracy_delta": aggregate["modes"][MODES[1]]["accuracy"]["mean"]
            - aggregate["modes"][MODES[0]]["accuracy"]["mean"],
            "strict_accuracy_delta": aggregate["modes"][MODES[1]][
                "strict_accuracy"
            ]["mean"]
            - aggregate["modes"][MODES[0]]["strict_accuracy"]["mean"],
            "prediction_parity": paired[0]["prediction_parity"],
            "exact_output_parity": paired[0]["content_sha256_parity"],
            "both_untruncated": paired[0]["both_untruncated"],
            "both_untruncated_prediction_parity": paired[0][
                "both_untruncated_prediction_parity"
            ],
            "task_count": 100,
            "production_default_eligible": False,
            "boundary": (
                "Fixed 100-task diagnostic repeated three times on one RTX "
                "4090; repetitions measure execution stability, not "
                "independent task samples"
            ),
        },
    }


def derived_pair_metrics(task_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(task_pairs) == 100, "task-pair count differs")
    require(
        len({row["id"] for row in task_pairs}) == 100,
        "task IDs are not unique",
    )
    gains = sum(
        not row["base"]["correct"] and row["mtp"]["correct"]
        for row in task_pairs
    )
    losses = sum(
        row["base"]["correct"] and not row["mtp"]["correct"]
        for row in task_pairs
    )
    strict_gains = sum(
        not row["base"]["strict_correct"] and row["mtp"]["strict_correct"]
        for row in task_pairs
    )
    strict_losses = sum(
        row["base"]["strict_correct"] and not row["mtp"]["strict_correct"]
        for row in task_pairs
    )
    untruncated = [
        row
        for row in task_pairs
        if not row["base"]["truncated"] and not row["mtp"]["truncated"]
    ]
    return {
        "task_count": 100,
        "base_correct": sum(row["base"]["correct"] for row in task_pairs),
        "candidate_correct": sum(row["mtp"]["correct"] for row in task_pairs),
        "gains": gains,
        "losses": losses,
        "exact_mcnemar_p": exact_mcnemar_p(gains, losses),
        "strict_base_correct": sum(
            row["base"]["strict_correct"] for row in task_pairs
        ),
        "strict_candidate_correct": sum(
            row["mtp"]["strict_correct"] for row in task_pairs
        ),
        "strict_gains": strict_gains,
        "strict_losses": strict_losses,
        "strict_exact_mcnemar_p": exact_mcnemar_p(strict_gains, strict_losses),
        "prediction_parity": sum(
            row["base"]["prediction"] == row["mtp"]["prediction"]
            for row in task_pairs
        ),
        "content_sha256_parity": sum(
            row["base"]["content_sha256"] == row["mtp"]["content_sha256"]
            for row in task_pairs
        ),
        "both_untruncated": len(untruncated),
        "both_untruncated_prediction_parity": sum(
            row["base"]["prediction"] == row["mtp"]["prediction"]
            for row in untruncated
        ),
    }


def assert_path_free(value: Any, location: str = "evidence") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_path_free(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_path_free(item, f"{location}[{index}]")
    elif isinstance(value, str):
        require(
            re.match(r"^[A-Za-z]:[\\/]", value) is None
            and not value.startswith(("/", "\\")),
            f"absolute path found at {location}",
        )


def verify_evidence(
    evidence: dict[str, Any], require_lossless: bool
) -> dict[str, Any]:
    require(evidence["schema"] == SCHEMA, "evidence schema differs")
    assert_path_free(evidence)
    source = evidence["source"]
    for field in (
        "power_commit",
        "server_sha256",
        "raw_environment_sha256",
        "raw_aggregate_sha256",
    ):
        require(source[field] == EXPECTED_IDENTITY[field], f"source {field} differs")
    require(not source["dirty_worktree"], "source worktree was dirty")
    require(
        source["benchmark_tools"] == EXPECTED_TOOL_HASHES,
        "benchmark tool hashes differ",
    )
    check_expected_hashes(
        {row["file"]: row["sha256"] for row in source["raw_reports"]},
        EXPECTED_REPORT_HASHES,
    )
    artifacts = evidence["artifacts"]
    require(artifacts["target"]["sha256"] == Q6_SHA256, "target hash differs")
    require(artifacts["target"]["bytes"] == Q6_BYTES, "target byte length differs")
    require(artifacts["target"]["quantization"] == "Q6_K", "target is not Q6_K")
    require(
        artifacts["target"]["file_hash_verified"],
        "target hash was not verified",
    )
    require(artifacts["auxiliary"] == [], "auxiliary artifacts are not permitted")

    workload = evidence["workload"]
    require(workload["profile"] == "pure-q6", "workload profile differs")
    require(workload["tasks_sha256"] == TASKS_SHA256, "workload task hash differs")
    require(
        workload["tasks_file_sha256"]
        == EXPECTED_IDENTITY["tasks_file_sha256"],
        "task file hash differs",
    )
    require(
        workload["task_manifest_sha256"]
        == EXPECTED_IDENTITY["task_manifest_sha256"],
        "task manifest hash differs",
    )
    require(
        workload["config_sha256"] == EXPECTED_IDENTITY["config_sha256"],
        "config hash differs",
    )
    require(
        workload["repetitions"] == 3 and workload["total_requests"] == 600,
        "workload size differs",
    )

    monitors = evidence["gpu_exclusivity"]["monitors"]
    check_expected_hashes(
        {
            f"{row['run']}.nvidia-pmon.log": row["log_sha256"]
            for row in monitors
        },
        EXPECTED_PMON_HASHES,
    )
    require(
        evidence["gpu_exclusivity"][
            "maximum_foreign_sm_utilization_percent"
        ]
        == 2,
        "GPU interference threshold differs",
    )
    for monitor in monitors:
        require(monitor["parsed_samples"] > 0, "GPU monitor has no samples")
        require(not monitor["interference_detected"], "GPU interference is present")
        require(
            monitor["foreign_processes"] == [],
            "foreign GPU processes are present",
        )
        require(monitor["failure"] is None, "GPU monitor failure is present")

    require(set(evidence["modes"]) == set(MODES), "evidence mode set differs")
    task_pairs = evidence["task_pairs"]
    paired = derived_pair_metrics(task_pairs)
    require(
        evidence["paired_runs"] == [paired, paired, paired],
        "paired metrics differ",
    )
    side_names = {MODES[0]: "base", MODES[1]: "mtp"}
    for mode, side in side_names.items():
        mode_data = evidence["modes"][mode]
        runs = mode_data["runs"]
        require(len(runs) == 3, f"{mode} run count differs")
        require(
            {run["repetition"] for run in runs} == {1, 2, 3},
            f"{mode} repetitions differ",
        )
        require(
            all(run["model_sha256"] == Q6_SHA256 for run in runs),
            f"{mode} mixes target models",
        )
        expected_reports = {
            name: digest
            for name, digest in EXPECTED_REPORT_HASHES.items()
            if f"-{mode}.json" in name
        }
        check_expected_hashes(
            {run["file"]: run["report_sha256"] for run in runs},
            expected_reports,
        )
        rows = [
            {
                "id": row["id"],
                "content_sha256": row[side]["content_sha256"],
            }
            for row in task_pairs
        ]
        predictions = [
            {
                "id": row["id"],
                "prediction": row[side]["prediction"],
                "strict_prediction": row[side]["strict_prediction"],
                "correct": row[side]["correct"],
                "strict_correct": row[side]["strict_correct"],
            }
            for row in task_pairs
        ]
        require(
            all(
                run["content_vector_sha256"] == canonical_digest(rows)
                for run in runs
            ),
            f"{mode} content vector differs",
        )
        require(
            all(
                run["prediction_vector_sha256"]
                == canonical_digest(predictions)
                for run in runs
            ),
            f"{mode} prediction vector differs",
        )
        aggregate = mode_data["aggregate"]
        require(aggregate["task_count"] == 100, f"{mode} task count differs")
        require(
            aggregate["prediction_stable_tasks"] == 100,
            f"{mode} answers are unstable",
        )
        require(
            aggregate["content_stable_tasks"] == 100,
            f"{mode} outputs are unstable",
        )
        accuracy = sum(row[side]["correct"] for row in task_pairs) / 100
        strict_accuracy = sum(row[side]["strict_correct"] for row in task_pairs) / 100
        require(
            math.isclose(aggregate["accuracy"]["mean"], accuracy),
            f"{mode} accuracy differs",
        )
        require(
            math.isclose(aggregate["strict_accuracy"]["mean"], strict_accuracy),
            f"{mode} strict accuracy differs",
        )
        mean_rate = statistics.mean(
            run["workload_tokens_per_second"] for run in runs
        )
        require(
            math.isclose(
                aggregate["aggregate_completion_tokens_per_second"]["mean"],
                mean_rate,
            ),
            f"{mode} throughput differs",
        )
        if mode == MODES[0]:
            require(
                mode_data["runtime_config"]["speculative"]["mode"] == "off",
                "control speculation mode differs",
            )
            require(
                all(run["speculative_runtime"] is None for run in runs),
                "control has speculative metrics",
            )
        else:
            require(
                mode_data["runtime_config"]["speculative"]["mode"] == "mtp",
                "candidate speculation mode differs",
            )
            require(
                all(
                    run["speculative_runtime"]["strategy"] == "mtp"
                    for run in runs
                ),
                "candidate runtime strategy differs",
            )
            require(
                all(
                    run["speculative_runtime"]["overall"]["fallback_replays"]
                    == 0
                    for run in runs
                ),
                "candidate used fallback replay",
            )

    base_rate = evidence["modes"][MODES[0]]["aggregate"][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    mtp_rate = evidence["modes"][MODES[1]]["aggregate"][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    eligible = (
        paired["content_sha256_parity"] == 100
        and paired["prediction_parity"] == 100
        and paired["losses"] == 0
        and paired["strict_losses"] == 0
    )
    claim = evidence["claim"]
    require(
        math.isclose(
            claim["workload_throughput_speedup"], mtp_rate / base_rate
        ),
        "claim speedup differs",
    )
    require(
        math.isclose(
            claim["accuracy_delta"],
            (paired["candidate_correct"] - paired["base_correct"]) / 100,
        ),
        "claim accuracy delta differs",
    )
    require(
        math.isclose(
            claim["strict_accuracy_delta"],
            (
                paired["strict_candidate_correct"]
                - paired["strict_base_correct"]
            )
            / 100,
        ),
        "claim strict accuracy delta differs",
    )
    require(
        claim["exact_output_parity"] == paired["content_sha256_parity"],
        "claim output parity differs",
    )
    require(
        claim["production_default_eligible"] == eligible,
        "production-default eligibility differs",
    )
    require(
        canonical_digest(evidence) == EXPECTED_EVIDENCE_SHA256,
        "pinned compact evidence payload differs",
    )
    if require_lossless:
        require(
            eligible,
            (
                "MTP is not lossless: "
                f"{paired['content_sha256_parity']}/100 exact outputs and "
                f"{paired['strict_losses']} strict losses"
            ),
        )
    return {
        "schema": SCHEMA,
        "status": "passed",
        "target_sha256": Q6_SHA256,
        "total_requests": 600,
        "base_workload_tokens_per_second": base_rate,
        "mtp_workload_tokens_per_second": mtp_rate,
        "speedup": mtp_rate / base_rate,
        "accuracy_delta": claim["accuracy_delta"],
        "strict_accuracy_delta": claim["strict_accuracy_delta"],
        "exact_output_parity": paired["content_sha256_parity"],
        "production_default_eligible": eligible,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Package the pinned raw capture")
    capture.add_argument("--capture-root", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="Verify compact checked-in evidence")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--require-lossless", action="store_true")
    verify.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "capture":
            payload = build_evidence(args.capture_root)
            verify_evidence(payload, require_lossless=False)
            write_json(args.output, payload)
            print(f"wrote Q6_K-only quality evidence: {args.output}")
        else:
            result = verify_evidence(load_json(args.evidence), args.require_lossless)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("Q6_K-only quality evidence: PASS")
        return 0
    except (EvidenceError, KeyError, TypeError, ValueError) as error:
        print(f"Q6_K-only quality evidence: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
