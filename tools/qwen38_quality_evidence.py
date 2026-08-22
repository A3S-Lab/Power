#!/usr/bin/env python3
"""Package and verify path-free Qwen3.8 DSpark quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen38_quality_report import is_truncated, pair_metrics


SCHEMA = "a3s.power.dspark-quality-evidence.v1"
EXPECTED_IDENTITY = {
    "power_commit": "2c815288f637f94d94b6f9fe3841a5be21f18ae5",
    "server_sha256": "ed8ef327be7245e7c3129fa3298f12b0088b8b1f68463d858bfb047f8bc5859e",
    "tasks_sha256": "5798257e18b81188749196d34359278dfadf7986776eb2bd66d629cbfc33813c",
    "tasks_file_sha256": "2fd18d52a2e011692c6ffac2547da41ec8b6594e2d35a8aa40fa8648df96b40a",
    "task_manifest_sha256": "e254af009fbee304bb574515aece7630cfb6ea91fd0fece6c0d9b9f02de44a9d",
    "config_sha256": "20123c5ecf1afa6dd3d57deb058969c23721517c98f2abc054cf2a5a4bde71a0",
    "target_sha256": "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    "draft_sha256": "12003c7f2642e2e87e979729e16947a913e2213d82136cb5024a36ec4871fef2",
    "raw_environment_sha256": "b716c3951ca2d8d0f05c7dbff14817c2a93d4699410fb34825dd807a154c0c76",
    "raw_aggregate_sha256": "b8b2cd23c60ed17bd21fbc070d6ffd6ecdf36e7c6bf760ae202ddc7bed15fb58",
}
EXPECTED_REPORT_HASHES = {
    "r01-o1-q6-off.json": "dc72e8981d4494a84a339dece9a03977d80ed44452ee641727c85670c7519277",
    "r01-o2-q6-dspark.json": "9d56d105e6f0395d4e0365908c796a0e28e84a23284f15697b5c118a39b7157e",
    "r02-o1-q6-dspark.json": "80564a45cc888a1818f1414d55d5e9e2b3a38fada89a8960f89e34374f15756d",
    "r02-o2-q6-off.json": "b810e23ea4a374a8d10db9b44052308898a0afb6587feca06d53bf98ac2280cd",
    "r03-o1-q6-off.json": "8bbe3cc3d1e24fd6b8ed3dcf83ebc553d997a9cd3ccdee190cdd67706693ef0f",
    "r03-o2-q6-dspark.json": "c0f8728c3a03c3de127f5687fe290355d0871af955fbb23dbe3cb21651aba0f9",
}


class EvidenceError(ValueError):
    """Raised when an evidence contract is not satisfied."""


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


def relative_report_path(capture_root: Path, name: str) -> Path:
    require(Path(name).name == name, f"report path must be a plain file name: {name}")
    path = (capture_root / name).resolve()
    require(path.parent == capture_root.resolve(), f"report escapes capture root: {name}")
    require(path.is_file(), f"report is missing: {name}")
    return path


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
        "median_reported_tokens_per_second",
        "aggregate_reported_tokens_per_second",
        "phase_duration_ns",
    )
    return {
        "strategy": runtime["strategy"],
        "overall": {field: runtime["overall"].get(field) for field in fields},
        "by_benchmark": {
            benchmark: {
                field: values.get(field)
                for field in (
                    "requests",
                    "weighted_acceptance_rate",
                    "verified_tokens_per_target_pass",
                    "fallback_replays",
                    "aggregate_reported_tokens_per_second",
                )
            }
            for benchmark, values in runtime.get("by_benchmark", {}).items()
        },
    }


def compact_run(name: str, digest: str, report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]["overall"]
    rows = report["results"]
    return {
        "file": name,
        "report_sha256": digest,
        "repetition": report["repetition"],
        "order_index": report["order_index"],
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
        "content_vector_sha256": canonical_digest(content_vector(rows)),
        "prediction_vector_sha256": canonical_digest(prediction_vector(rows)),
        "speculative_runtime": compact_runtime(report.get("speculative_runtime")),
    }


def compact_mode_aggregate(mode: dict[str, Any]) -> dict[str, Any]:
    selected = {
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
        selected["speculative_runtime"] = mode["speculative_runtime"]
    return selected


def compact_hardware(environment: dict[str, Any]) -> dict[str, Any]:
    gpu_fields = [field.strip() for field in environment["gpu"][0].split(",")]
    require(len(gpu_fields) == 6, "unexpected NVIDIA identity shape")
    os_data = environment["os"]
    return {
        "os": "Windows 11",
        "os_version": os_data["Version"],
        "os_build": os_data["BuildNumber"],
        "visible_memory_kib": os_data["TotalVisibleMemorySize"],
        "cpu": environment["cpu"],
        "gpu": {
            "name": gpu_fields[0],
            "driver": gpu_fields[1],
            "memory_total_mib": int(gpu_fields[2]),
            "compute_capability": gpu_fields[3],
            "pstate_at_capture": gpu_fields[4],
            "power_limit_watts": float(gpu_fields[5]),
        },
        "power_scheme_guid": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
    }


def compact_admissions(environment: dict[str, Any]) -> dict[str, Any]:
    windows = []
    for item in environment["gpu_admissions"]:
        admission = item["admission"]
        windows.append(
            {
                "run": item["run"],
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
        )
    return {"requirements": environment["gpu_admission"], "windows": windows}


def build_task_pairs(
    base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    require(
        [row["id"] for row in base_rows] == [row["id"] for row in candidate_rows],
        "paired task order differs",
    )

    def result(row: dict[str, Any]) -> dict[str, Any]:
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
            "id": base["id"],
            "benchmark": base["benchmark"],
            "expected": base["expected"],
            "base": result(base),
            "dspark": result(candidate),
        }
        for base, candidate in zip(base_rows, candidate_rows, strict=True)
    ]


def build_evidence(capture_root: Path) -> dict[str, Any]:
    environment_path = capture_root / "environment.json"
    aggregate_path = capture_root / "quality-matrix.json"
    tasks_path = capture_root / "tasks-v1.json"
    for path in (environment_path, aggregate_path, tasks_path):
        require(path.is_file(), f"capture input is missing: {path.name}")

    environment = load_json(environment_path)
    aggregate = load_json(aggregate_path)
    require(not environment["dirty_worktree"], "capture used a dirty worktree")
    require(aggregate["repetitions"] == 3, "capture must contain three repetitions")
    require(set(aggregate["modes"]) == {"q6-off", "q6-dspark"}, "mode set differs")

    reports: dict[str, dict[str, Any]] = {}
    report_hashes: dict[str, str] = {}
    for receipt in aggregate["reports"]:
        name = receipt["path"]
        path = relative_report_path(capture_root, name)
        digest = sha256_file(path)
        require(digest == receipt["sha256"], f"aggregate report hash differs: {name}")
        report = load_json(path)
        require(report["schema"] == "a3s.power.quality-eval.report.v3", "report schema differs")
        require(report["server_sha256"] == aggregate["server_sha256"], "server hash differs")
        require(report["tasks_sha256"] == aggregate["tasks_sha256"], "task hash differs")
        require(report["power_commit"] == environment["power_commit"], "commit differs")
        reports[name] = report
        report_hashes[name] = digest

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for name, report in reports.items():
        grouped[report["mode_label"]].append((name, report))
    for items in grouped.values():
        items.sort(key=lambda item: item[1]["repetition"])
    require(all(len(items) == 3 for items in grouped.values()), "mode run count differs")

    for mode, items in grouped.items():
        expected_content = canonical_digest(content_vector(items[0][1]["results"]))
        expected_predictions = canonical_digest(prediction_vector(items[0][1]["results"]))
        for _, report in items:
            require(
                canonical_digest(content_vector(report["results"])) == expected_content,
                f"{mode} content is not stable across repetitions",
            )
            require(
                canonical_digest(prediction_vector(report["results"]))
                == expected_predictions,
                f"{mode} predictions are not stable across repetitions",
            )
            require(report["summary"]["overall"]["errors"] == 0, f"{mode} has errors")

    base_by_repetition = {
        report["repetition"]: report for _, report in grouped["q6-off"]
    }
    dspark_by_repetition = {
        report["repetition"]: report for _, report in grouped["q6-dspark"]
    }
    paired = [
        pair_metrics(base_by_repetition[index], dspark_by_repetition[index])
        for index in range(1, 4)
    ]
    require(
        paired == aggregate["paired_runs"]["q6-off -> q6-dspark"],
        "paired metrics differ from aggregate",
    )
    task_pairs = build_task_pairs(
        base_by_repetition[1]["results"], dspark_by_repetition[1]["results"]
    )

    request_digests = {
        canonical_digest(report["request"]) for report in reports.values()
    }
    request_hashes = {report["request_sha256"] for report in reports.values()}
    require(len(request_digests) == 1, "request settings differ across reports")
    require(len(request_hashes) == 1, "request hashes differ across reports")
    request = next(iter(reports.values()))["request"]
    target = environment["q6_model"]
    draft = target["external_draft"]

    mode_evidence: dict[str, Any] = {}
    for mode in ("q6-off", "q6-dspark"):
        items = grouped[mode]
        mode_evidence[mode] = {
            "runtime_config": {
                "speculative": items[0][1]["health"]["speculative"],
                "inference": items[0][1]["health"]["inference"],
            },
            "aggregate": compact_mode_aggregate(aggregate["modes"][mode]),
            "runs": [
                compact_run(name, report_hashes[name], report) for name, report in items
            ],
        }

    base_rate = aggregate["modes"]["q6-off"][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    dspark_rate = aggregate["modes"]["q6-dspark"][
        "aggregate_completion_tokens_per_second"
    ]["mean"]
    exact_parity = paired[0]["content_sha256_parity"]
    task_count = paired[0]["task_count"]
    return {
        "schema": SCHEMA,
        "captured_at": aggregate["created_at"],
        "source": {
            "power_commit": environment["power_commit"],
            "dirty_worktree": environment["dirty_worktree"],
            "server_sha256": aggregate["server_sha256"],
            "raw_environment_sha256": sha256_file(environment_path),
            "raw_aggregate_sha256": sha256_file(aggregate_path),
            "raw_reports": aggregate["reports"],
        },
        "hardware": compact_hardware(environment),
        "artifacts": {
            "target": {
                "bytes": target["size"],
                "sha256": target["sha256"],
                "file_hash_verified": target["file_hash_verified"],
            },
            "draft": {
                key: draft[key]
                for key in (
                    "kind",
                    "size",
                    "sha256",
                    "target_sha256",
                    "source",
                    "revision",
                    "license",
                    "file_hash_verified",
                )
            },
        },
        "workload": {
            "model": next(iter(reports.values()))["model"],
            "tasks_sha256": aggregate["tasks_sha256"],
            "tasks_file_sha256": sha256_file(tasks_path),
            "task_manifest_sha256": environment["task_manifest_sha256"],
            "config_sha256": environment["config_sha256"],
            "profile": environment["profile"],
            "repetitions": aggregate["repetitions"],
            "tasks_per_mode_per_repetition": task_count,
            "total_requests": task_count * 2 * aggregate["repetitions"],
            "request_sha256": next(iter(request_hashes)),
            "request": request,
        },
        "gpu_admission": compact_admissions(environment),
        "modes": mode_evidence,
        "paired_runs": paired,
        "task_pairs": task_pairs,
        "claim": {
            "classification": "diagnostic-output-divergence",
            "workload_throughput_speedup": dspark_rate / base_rate,
            "accuracy_delta": aggregate["modes"]["q6-dspark"]["accuracy"]["mean"]
            - aggregate["modes"]["q6-off"]["accuracy"]["mean"],
            "strict_accuracy_delta": aggregate["modes"]["q6-dspark"][
                "strict_accuracy"
            ]["mean"]
            - aggregate["modes"]["q6-off"]["strict_accuracy"]["mean"],
            "exact_output_parity": exact_parity,
            "task_count": task_count,
            "production_default_eligible": exact_parity == task_count,
            "boundary": (
                "Repeated deterministic multi-domain diagnostic on one RTX 4090; "
                "repetitions measure execution stability, not independent task samples"
            ),
        },
    }


def derived_pair_metrics(task_pairs: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "task_count": len(task_pairs),
        "base_correct": sum(row["base"]["correct"] for row in task_pairs),
        "candidate_correct": sum(row["dspark"]["correct"] for row in task_pairs),
        "gains": sum(
            not row["base"]["correct"] and row["dspark"]["correct"]
            for row in task_pairs
        ),
        "losses": sum(
            row["base"]["correct"] and not row["dspark"]["correct"]
            for row in task_pairs
        ),
        "strict_base_correct": sum(
            row["base"]["strict_correct"] for row in task_pairs
        ),
        "strict_candidate_correct": sum(
            row["dspark"]["strict_correct"] for row in task_pairs
        ),
        "strict_gains": sum(
            not row["base"]["strict_correct"] and row["dspark"]["strict_correct"]
            for row in task_pairs
        ),
        "strict_losses": sum(
            row["base"]["strict_correct"] and not row["dspark"]["strict_correct"]
            for row in task_pairs
        ),
        "prediction_parity": sum(
            row["base"]["prediction"] == row["dspark"]["prediction"]
            for row in task_pairs
        ),
        "content_sha256_parity": sum(
            row["base"]["content_sha256"] == row["dspark"]["content_sha256"]
            for row in task_pairs
        ),
    }


def verify_evidence(evidence: dict[str, Any], require_production_default: bool) -> dict[str, Any]:
    require(evidence.get("schema") == SCHEMA, "evidence schema differs")
    source = evidence["source"]
    workload = evidence["workload"]
    artifacts = evidence["artifacts"]
    observed_identity = {
        "power_commit": source["power_commit"],
        "server_sha256": source["server_sha256"],
        "tasks_sha256": workload["tasks_sha256"],
        "tasks_file_sha256": workload["tasks_file_sha256"],
        "task_manifest_sha256": workload["task_manifest_sha256"],
        "config_sha256": workload["config_sha256"],
        "target_sha256": artifacts["target"]["sha256"],
        "draft_sha256": artifacts["draft"]["sha256"],
        "raw_environment_sha256": source["raw_environment_sha256"],
        "raw_aggregate_sha256": source["raw_aggregate_sha256"],
    }
    require(observed_identity == EXPECTED_IDENTITY, "pinned capture identity differs")
    require(not source["dirty_worktree"], "capture used a dirty worktree")
    observed_reports = {
        report["path"]: report["sha256"] for report in source["raw_reports"]
    }
    require(observed_reports == EXPECTED_REPORT_HASHES, "pinned report hashes differ")
    require(artifacts["target"]["bytes"] == 22_884_408_288, "target size differs")
    require(artifacts["draft"]["size"] == 1_104_594_816, "draft size differs")
    require(artifacts["draft"]["kind"] == "dspark", "draft kind differs")
    require(
        artifacts["draft"]["target_sha256"] == artifacts["target"]["sha256"],
        "draft target binding differs",
    )
    require(
        artifacts["target"]["file_hash_verified"]
        and artifacts["draft"]["file_hash_verified"],
        "model file hash verification was disabled",
    )
    require(workload["repetitions"] == 3, "repetition count differs")
    require(workload["total_requests"] == 600, "request count differs")
    require(
        workload["request"]
        == {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_ctx": 1024,
            "num_batch": 12,
            "seed": 42,
            "warmup_requests": 1,
            "max_tokens_cap": 256,
            "template": "manual-qwen-chatml-v1",
        },
        "request settings differ",
    )

    task_pairs = evidence["task_pairs"]
    require(len({row["id"] for row in task_pairs}) == 100, "task IDs are not unique")
    require(
        Counter(row["benchmark"] for row in task_pairs)
        == {"mmlu": 50, "gsm8k": 20, "ceval": 30},
        "benchmark task counts differ",
    )
    derived = derived_pair_metrics(task_pairs)
    expected_pair = {
        "task_count": 100,
        "base_correct": 67,
        "candidate_correct": 73,
        "gains": 6,
        "losses": 0,
        "strict_base_correct": 58,
        "strict_candidate_correct": 59,
        "strict_gains": 2,
        "strict_losses": 1,
        "prediction_parity": 91,
        "content_sha256_parity": 54,
    }
    require(derived == expected_pair, "derived paired quality metrics differ")
    require(len(evidence["paired_runs"]) == 3, "paired repetition count differs")
    for pair in evidence["paired_runs"]:
        require(
            all(pair[key] == value for key, value in expected_pair.items()),
            "recorded paired metrics differ",
        )

    all_run_names = set(EXPECTED_REPORT_HASHES)
    for mode, expected_accuracy, expected_strict in (
        ("q6-off", 0.67, 0.58),
        ("q6-dspark", 0.73, 0.59),
    ):
        mode_data = evidence["modes"][mode]
        runs = mode_data["runs"]
        require(len(runs) == 3, f"{mode} run count differs")
        require(all(run["completed"] == 100 for run in runs), f"{mode} is incomplete")
        require(all(run["errors"] == 0 for run in runs), f"{mode} has errors")
        require(
            len({run["content_vector_sha256"] for run in runs}) == 1,
            f"{mode} content is unstable",
        )
        require(
            len({run["prediction_vector_sha256"] for run in runs}) == 1,
            f"{mode} predictions are unstable",
        )
        require(all(run["accuracy"] == expected_accuracy for run in runs), f"{mode} score differs")
        require(
            all(run["strict_accuracy"] == expected_strict for run in runs),
            f"{mode} strict score differs",
        )
        require(
            all(run["file"] in all_run_names for run in runs),
            f"{mode} references an unknown report",
        )
        require(
            all(
                run["report_sha256"] == EXPECTED_REPORT_HASHES[run["file"]]
                for run in runs
            ),
            f"{mode} compact report hashes differ",
        )
        pair_key = "base" if mode == "q6-off" else "dspark"
        expected_content_vector = canonical_digest(
            [
                {
                    "id": row["id"],
                    "content_sha256": row[pair_key]["content_sha256"],
                }
                for row in task_pairs
            ]
        )
        expected_prediction_vector = canonical_digest(
            [
                {
                    "id": row["id"],
                    "prediction": row[pair_key]["prediction"],
                    "strict_prediction": row[pair_key]["strict_prediction"],
                    "correct": row[pair_key]["correct"],
                    "strict_correct": row[pair_key]["strict_correct"],
                }
                for row in task_pairs
            ]
        )
        require(
            all(
                run["content_vector_sha256"] == expected_content_vector
                for run in runs
            ),
            f"{mode} task pairs do not match the captured content vector",
        )
        require(
            all(
                run["prediction_vector_sha256"] == expected_prediction_vector
                for run in runs
            ),
            f"{mode} task pairs do not match the captured prediction vector",
        )
        mean_rate = statistics.mean(
            run["workload_tokens_per_second"] for run in runs
        )
        require(
            abs(
                mean_rate
                - mode_data["aggregate"][
                    "aggregate_completion_tokens_per_second"
                ]["mean"]
            )
            <= 1e-12,
            f"{mode} aggregate throughput differs",
        )
        if mode == "q6-off":
            require(
                all(run["speculative_runtime"] is None for run in runs),
                "target-only run contains speculative telemetry",
            )
        else:
            for run in runs:
                runtime = run["speculative_runtime"]
                require(runtime["strategy"] == "dspark", "runtime strategy differs")
                require(runtime["overall"]["requests"] == 100, "runtime request count differs")
                require(runtime["overall"]["fallback_replays"] == 100, "fallback count differs")
                require(
                    runtime["overall"]["rollback_guard_requests"] == 100,
                    "rollback guard request count differs",
                )
                require(
                    runtime["overall"]["weighted_acceptance_rate"]
                    == 0.4472636664742168,
                    "acceptance rate differs",
                )

    admission = evidence["gpu_admission"]
    requirements = admission["requirements"]
    require(requirements["maximum_idle_utilization_percent"] == 20, "GPU utilization gate differs")
    require(requirements["minimum_idle_memory_free_mib"] == 23_000, "GPU memory gate differs")
    require(len(admission["windows"]) == 6, "GPU admission window count differs")
    require(
        {window["run"] for window in admission["windows"]}
        == {Path(name).stem for name in all_run_names},
        "GPU admission run identities differ",
    )
    for window in admission["windows"]:
        samples = window["accepted_samples"]
        require(len(samples) == 3, "GPU admission sample count differs")
        require(
            all(
                sample["utilization_percent"] <= 20
                and sample["memory_free_mib"] >= 23_000
                for sample in samples
            ),
            "GPU admission threshold was not satisfied",
        )

    base_rates = [
        run["workload_tokens_per_second"] for run in evidence["modes"]["q6-off"]["runs"]
    ]
    dspark_rates = [
        run["workload_tokens_per_second"]
        for run in evidence["modes"]["q6-dspark"]["runs"]
    ]
    speedup = statistics.mean(dspark_rates) / statistics.mean(base_rates)
    claim = evidence["claim"]
    require(
        abs(speedup - claim["workload_throughput_speedup"]) <= 1e-12,
        "workload speedup differs",
    )
    require(claim["exact_output_parity"] == 54, "claim output parity differs")
    require(not claim["production_default_eligible"], "diagnostic was mislabeled as production")
    require(
        claim["classification"] == "diagnostic-output-divergence",
        "claim classification differs",
    )
    if require_production_default:
        raise EvidenceError(
            "production-default gate failed: exact target/DSpark output parity is 54/100"
        )
    return {
        "status": "passed",
        "classification": claim["classification"],
        "total_requests": workload["total_requests"],
        "target_accuracy": 0.67,
        "dspark_accuracy": 0.73,
        "target_workload_tokens_per_second": statistics.mean(base_rates),
        "dspark_workload_tokens_per_second": statistics.mean(dspark_rates),
        "workload_speedup": speedup,
        "exact_output_parity": 54,
        "task_count": 100,
        "production_default_eligible": False,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Package raw capture files")
    capture.add_argument("--capture-root", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify checked-in compact evidence")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--require-production-default", action="store_true")
    verify.add_argument("--json", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "capture":
            evidence = build_evidence(args.capture_root.resolve())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.output} ({len(evidence['task_pairs'])} paired tasks)")
            return 0
        evidence = load_json(args.evidence)
        result = verify_evidence(evidence, args.require_production_default)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                "DSpark quality evidence: PASS "
                f"({result['exact_output_parity']}/{result['task_count']} exact outputs; "
                "not production-default eligible)"
            )
        return 0
    except (EvidenceError, KeyError, TypeError, ValueError) as error:
        print(f"DSpark quality evidence: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
