#!/usr/bin/env python3
"""Package and verify the pinned adaptive-DSpark quality follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen38_quality_report import exact_mcnemar_p, is_truncated


SCHEMA = "a3s.power.dspark-quality-followup-evidence.v1"
EXPECTED_TASK_IDS = [
    "mmlu:8400",
    "mmlu:11207",
    "ceval:teacher_qualification:1",
    "ceval:college_physics:2",
    "ceval:high_school_history:0",
]
EXPECTED_ANSWERS = ["B", "C", "B", "C", "B"]
EXPECTED_IDENTITY = {
    "power_commit": "7bdeb960f5a38ea7515c67a12636a29198fd95f6",
    "server_sha256": "92e532aca885d4babdd1bf15bd94ece07e013082a87007543514749b96d0f373",
    "runner_sha256": "ceca65bc8da1aff5550c446213af1174602af30fd7120db619db6368e722060a",
    "evaluator_sha256": "9244feb9f00c04e6b17783361172d1a00bf09beca37df10f5a172d570a233644",
    "reporter_sha256": "5f676d00dfa16b25e1613de502d34ce379afadff28c05c583c714b964034006b",
    "target_sha256": "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    "draft_sha256": "12003c7f2642e2e87e979729e16947a913e2213d82136cb5024a36ec4871fef2",
    "task_source_sha256": "5798257e18b81188749196d34359278dfadf7986776eb2bd66d629cbfc33813c",
    "tasks_file_sha256": "2fd18d52a2e011692c6ffac2547da41ec8b6594e2d35a8aa40fa8648df96b40a",
    "tasks_sha256": "881a115e276b2c2a80fcae009682b6098c045a0a7180b956189a93dd1fbb5a69",
    "task_selection_sha256": "f7e6ad9a3018c8de4341e427b6812f30ecde5840f46ac16401ab0a21b4316df3",
    "task_manifest_sha256": "e254af009fbee304bb574515aece7630cfb6ea91fd0fece6c0d9b9f02de44a9d",
    "config_sha256": "a8fb13a27d5b30d77bf7d3047b90c4fcae601e6d08b6d2b05dbc46b733eeecb9",
}
EXPECTED_RAW = {
    "tokens_512": {
        "environment_sha256": "2451b69134894890e5ff6d6c212d4dd2d7fc63d2f2b81fa7bb71fbf50150a380",
        "aggregate_sha256": "c715dc540dff3db1da13fd2a63553f74cd49ff8922bd6f4e9e6c35bba37af8da",
        "reports": {
            "r01-o1-q6-off.json": "22cf6853d47cd5215042dae091753c63a2d5205c4157f668bc89985a6e44e14e",
            "r01-o2-q6-dspark.json": "bc540d98bcf0a4b8822909deaa8972250beda8cd9b33758a4a1fe3c9fe39b839",
            "r02-o1-q6-dspark.json": "6cafeb8f4f62820d39a0ffda5dfdd67662774b55f7b85d07371c5956d68f9e2c",
            "r02-o2-q6-off.json": "ad2384ac2a78d89a937d4510b50e1dd8b33cfd6fb808324e12cc5c504fcc3871",
            "r03-o1-q6-off.json": "37081fe270d3b816db15c47790f5261e9acf1ef0941638d89cbb11337e646351",
            "r03-o2-q6-dspark.json": "0d6d3f5ee567b0af181b35f3cae18cb822055e7724a145398dc8097ec3cfd2c7",
        },
    },
    "tokens_1024": {
        "environment_sha256": "678b8d9c1f5154936665ad1a098913c5f2f7ba3dfa991862ed6c9e3586a548b8",
        "aggregate_sha256": "f17824d7271f6b42fc36344bddfb9962eb6daa2a0536b72281fca68822ce466a",
        "reports": {
            "r01-o1-q6-off.json": "95e7f820d50e8c5046392f0101d9839723a3e3cae37dafab329d7b8ecac7bd3c",
            "r01-o2-q6-dspark.json": "ff81d50d56e8535219dbf9004a114ee477e6c276e4be5b9c5a205372ab317b39",
        },
    },
}
CAPTURE_SPECS = {
    "tokens_512": {
        "max_tokens_override": 512,
        "num_ctx": 1024,
        "repetitions": 3,
        "base_tps": 24.96735192815211,
        "candidate_tps": 30.520561870014912,
        "truncated": 1,
        "quality_only": False,
    },
    "tokens_1024": {
        "max_tokens_override": 1024,
        "num_ctx": 2048,
        "repetitions": 1,
        "base_tps": 25.089060390762047,
        "candidate_tps": 22.30411188536747,
        "truncated": 0,
        "quality_only": True,
    },
}


class EvidenceError(ValueError):
    """Raised when the follow-up evidence contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def near(actual: float, expected: float, label: str) -> None:
    require(abs(actual - expected) <= 1e-9, f"{label} differs")


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


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = sum(not row["base"]["correct"] and row["candidate"]["correct"] for row in rows)
    losses = sum(row["base"]["correct"] and not row["candidate"]["correct"] for row in rows)
    strict_gains = sum(
        not row["base"]["strict_correct"] and row["candidate"]["strict_correct"]
        for row in rows
    )
    strict_losses = sum(
        row["base"]["strict_correct"] and not row["candidate"]["strict_correct"]
        for row in rows
    )
    untruncated = [
        row
        for row in rows
        if not row["base"]["truncated"] and not row["candidate"]["truncated"]
    ]
    return {
        "task_count": len(rows),
        "base_correct": sum(row["base"]["correct"] for row in rows),
        "candidate_correct": sum(row["candidate"]["correct"] for row in rows),
        "gains": gains,
        "losses": losses,
        "exact_mcnemar_p": exact_mcnemar_p(gains, losses),
        "strict_base_correct": sum(row["base"]["strict_correct"] for row in rows),
        "strict_candidate_correct": sum(
            row["candidate"]["strict_correct"] for row in rows
        ),
        "strict_gains": strict_gains,
        "strict_losses": strict_losses,
        "strict_exact_mcnemar_p": exact_mcnemar_p(strict_gains, strict_losses),
        "prediction_parity": sum(
            row["base"]["prediction"] == row["candidate"]["prediction"]
            for row in rows
        ),
        "content_sha256_parity": sum(
            row["base"]["content_sha256"] == row["candidate"]["content_sha256"]
            for row in rows
        ),
        "both_untruncated": len(untruncated),
        "both_untruncated_prediction_parity": sum(
            row["base"]["prediction"] == row["candidate"]["prediction"]
            for row in untruncated
        ),
    }


def compact_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": row["prediction"],
        "strict_prediction": row["strict_prediction"],
        "correct": row["correct"],
        "strict_correct": row["strict_correct"],
        "truncated": is_truncated(row),
        "finish_reason": row["finish_reason"],
        "max_tokens": row["max_tokens"],
        "completion_tokens": row["usage"]["completion_tokens"],
        "content_sha256": row["content_sha256"],
    }


def result_signature(row: dict[str, Any]) -> str:
    return json.dumps(compact_result(row), sort_keys=True, separators=(",", ":"))


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


def capture_one(root: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_raw = EXPECTED_RAW[label]
    environment_path = root / "environment.json"
    aggregate_path = root / "quality-matrix.json"
    require(sha256_file(environment_path) == expected_raw["environment_sha256"], f"raw {label} environment differs")
    require(sha256_file(aggregate_path) == expected_raw["aggregate_sha256"], f"raw {label} aggregate differs")
    environment = load_json(environment_path)
    aggregate = load_json(aggregate_path)

    reports: list[dict[str, Any]] = []
    raw_reports: list[dict[str, str]] = []
    for name, expected_hash in expected_raw["reports"].items():
        path = root / name
        require(path.is_file(), f"raw {label} report is missing: {name}")
        digest = sha256_file(path)
        require(digest == expected_hash, f"raw {label} report differs: {name}")
        reports.append(load_json(path))
        raw_reports.append({"file": name, "sha256": digest})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        require(report["schema"] == "a3s.power.quality-eval.report.v3", "report schema differs")
        require(report["tasks_sha256"] == EXPECTED_IDENTITY["tasks_sha256"], "selected task digest differs")
        require(report["task_source_sha256"] == EXPECTED_IDENTITY["task_source_sha256"], "task source digest differs")
        require(report["task_selection_sha256"] == EXPECTED_IDENTITY["task_selection_sha256"], "selection digest differs")
        grouped[report["mode_label"]].append(report)
    require(set(grouped) == {"q6-off", "q6-dspark"}, "capture modes differ")

    spec = CAPTURE_SPECS[label]
    for mode_reports in grouped.values():
        mode_reports.sort(key=lambda report: int(report["repetition"]))
        require(len(mode_reports) == spec["repetitions"], "capture repetition count differs")

    task_pairs: list[dict[str, Any]] = []
    base_rows = {row["id"]: row for row in grouped["q6-off"][0]["results"]}
    candidate_rows = {
        row["id"]: row for row in grouped["q6-dspark"][0]["results"]
    }
    require(list(base_rows) == EXPECTED_TASK_IDS, "base task order differs")
    require(list(candidate_rows) == EXPECTED_TASK_IDS, "candidate task order differs")
    for task_id in EXPECTED_TASK_IDS:
        for mode, first in (("q6-off", base_rows[task_id]), ("q6-dspark", candidate_rows[task_id])):
            signature = result_signature(first)
            require(
                all(
                    result_signature(
                        next(row for row in report["results"] if row["id"] == task_id)
                    )
                    == signature
                    for report in grouped[mode]
                ),
                f"{label} {mode} output is not stable for {task_id}",
            )
        task_pairs.append(
            {
                "id": task_id,
                "expected": base_rows[task_id]["expected"],
                "base": compact_result(base_rows[task_id]),
                "candidate": compact_result(candidate_rows[task_id]),
            }
        )

    derived = paired_summary(task_pairs)
    published_pairs = aggregate["paired_runs"]["q6-off -> q6-dspark"]
    require(
        all(pair == derived for pair in published_pairs),
        f"{label} aggregate paired summary differs",
    )
    modes = {
        mode: {
            "accuracy": aggregate["modes"][mode]["accuracy"],
            "strict_accuracy": aggregate["modes"][mode]["strict_accuracy"],
            "truncated": aggregate["modes"][mode]["truncated"],
            "aggregate_completion_tokens_per_second": aggregate["modes"][mode][
                "aggregate_completion_tokens_per_second"
            ],
            "prediction_stable_tasks": aggregate["modes"][mode][
                "prediction_stable_tasks"
            ],
            "content_stable_tasks": aggregate["modes"][mode][
                "content_stable_tasks"
            ],
        }
        for mode in ("q6-off", "q6-dspark")
    }
    runtime = aggregate["modes"]["q6-dspark"]["speculative_runtime"]
    modes["q6-dspark"]["speculative_runtime"] = {
        "strategy": runtime["strategy"],
        "overall": runtime["overall"],
    }
    capture = {
        "request": {
            "max_tokens_override": environment["max_tokens_override"],
            "num_ctx": environment["num_ctx"],
            "num_batch": environment["num_batch"],
            "repetitions": environment["repetitions"],
            "seed": 42,
            "quality_only": spec["quality_only"],
        },
        "modes": modes,
        "paired_runs": published_pairs,
        "task_pairs": task_pairs,
        "host_controls": {
            "active_power_scheme": environment["active_power_scheme"],
            "process_priority": environment["process_priority"],
            "process_affinity": environment["process_affinity"],
            "cuda_high_priority": environment["host_controls"]["cuda_high_priority"],
            "gpu_clock": environment["host_controls"]["gpu_clock"],
            "gpu_admission": environment["gpu_admission"],
            "admission_windows": compact_admissions(environment),
        },
    }
    raw = {
        "environment_sha256": sha256_file(environment_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "reports": raw_reports,
    }
    return capture, {"environment": environment, "raw": raw}


def build_evidence(tokens_512_root: Path, tokens_1024_root: Path) -> dict[str, Any]:
    capture_512, source_512 = capture_one(tokens_512_root, "tokens_512")
    capture_1024, source_1024 = capture_one(tokens_1024_root, "tokens_1024")
    environment = source_512["environment"]
    require(not environment["dirty_worktree"], "capture worktree was dirty")
    require(not source_1024["environment"]["dirty_worktree"], "capture worktree was dirty")
    return {
        "schema": SCHEMA,
        "source": {
            "power_commit": environment["power_commit"],
            "dirty_worktree": False,
            "server_sha256": environment["server"]["sha256"],
            "benchmark_tools": environment["benchmark_tools"],
            "raw": {
                "tokens_512": source_512["raw"],
                "tokens_1024": source_1024["raw"],
            },
        },
        "artifacts": {
            "target": {
                "bytes": environment["q6_model"]["size"],
                "sha256": environment["q6_model"]["sha256"],
                "file_hash_verified": environment["q6_model"]["file_hash_verified"],
            },
            "draft": {
                "kind": environment["q6_model"]["external_draft"]["kind"],
                "bytes": environment["q6_model"]["external_draft"]["size"],
                "sha256": environment["q6_model"]["external_draft"]["sha256"],
                "target_sha256": environment["q6_model"]["external_draft"]["target_sha256"],
                "file_hash_verified": environment["q6_model"]["external_draft"]["file_hash_verified"],
            },
        },
        "inputs": {
            "task_source_sha256": EXPECTED_IDENTITY["task_source_sha256"],
            "tasks_file_sha256": sha256_file(tokens_512_root / "tasks-v1.json"),
            "tasks_sha256": EXPECTED_IDENTITY["tasks_sha256"],
            "task_selection_sha256": environment["task_selection"]["sha256"],
            "task_manifest_sha256": environment["task_manifest_sha256"],
            "config_sha256": environment["config_sha256"],
            "task_ids": EXPECTED_TASK_IDS,
        },
        "hardware": {
            "gpu": environment["gpu"],
            "cpu": environment["cpu"],
            "os": environment["os"],
        },
        "captures": {
            "tokens_512": capture_512,
            "tokens_1024": capture_1024,
        },
        "claim": {
            "classification": "diagnostic-answer-parity-output-divergence",
            "selected_answer_non_regression": True,
            "tokens_512_workload_speedup": (
                capture_512["modes"]["q6-dspark"]["aggregate_completion_tokens_per_second"]["mean"]
                / capture_512["modes"]["q6-off"]["aggregate_completion_tokens_per_second"]["mean"]
            ),
            "tokens_1024_quality_only": True,
            "production_default_eligible": False,
        },
    }


def verify_evidence(
    evidence: dict[str, Any], require_production_default: bool
) -> dict[str, Any]:
    require(evidence.get("schema") == SCHEMA, "evidence schema differs")
    source = evidence["source"]
    require(source["power_commit"] == EXPECTED_IDENTITY["power_commit"], "published power commit differs")
    require(not source["dirty_worktree"], "published worktree was dirty")
    require(source["server_sha256"] == EXPECTED_IDENTITY["server_sha256"], "published server differs")
    for field in ("runner_sha256", "evaluator_sha256", "reporter_sha256"):
        require(source["benchmark_tools"][field] == EXPECTED_IDENTITY[field], f"published {field} differs")
    for label, expected in EXPECTED_RAW.items():
        actual = source["raw"][label]
        require(actual["environment_sha256"] == expected["environment_sha256"], f"raw {label} environment differs")
        require(actual["aggregate_sha256"] == expected["aggregate_sha256"], f"raw {label} aggregate differs")
        require(
            {item["file"]: item["sha256"] for item in actual["reports"]}
            == expected["reports"],
            f"raw {label} reports differ",
        )

    require(evidence["artifacts"]["target"]["sha256"] == EXPECTED_IDENTITY["target_sha256"], "target identity differs")
    require(evidence["artifacts"]["draft"]["sha256"] == EXPECTED_IDENTITY["draft_sha256"], "draft identity differs")
    require(evidence["artifacts"]["target"]["bytes"] == 22_884_408_288, "target size differs")
    require(evidence["artifacts"]["draft"]["bytes"] == 1_104_594_816, "draft size differs")
    require(evidence["artifacts"]["draft"]["kind"] == "dspark", "draft kind differs")
    require(
        evidence["artifacts"]["draft"]["target_sha256"]
        == EXPECTED_IDENTITY["target_sha256"],
        "draft target binding differs",
    )
    require(evidence["artifacts"]["target"]["file_hash_verified"], "target file hash was not verified")
    require(evidence["artifacts"]["draft"]["file_hash_verified"], "draft file hash was not verified")
    for field in (
        "task_source_sha256",
        "tasks_file_sha256",
        "tasks_sha256",
        "task_selection_sha256",
        "task_manifest_sha256",
        "config_sha256",
    ):
        require(evidence["inputs"][field] == EXPECTED_IDENTITY[field], f"published {field} differs")
    require(evidence["inputs"]["task_ids"] == EXPECTED_TASK_IDS, "published task IDs differ")

    summaries: dict[str, dict[str, Any]] = {}
    for label, spec in CAPTURE_SPECS.items():
        capture = evidence["captures"][label]
        request = capture["request"]
        require(request["max_tokens_override"] == spec["max_tokens_override"], f"{label} token override differs")
        require(request["num_ctx"] == spec["num_ctx"], f"{label} context differs")
        require(request["num_batch"] == 12, f"{label} batch differs")
        require(request["repetitions"] == spec["repetitions"], f"{label} repetitions differ")
        require(request["quality_only"] == spec["quality_only"], f"{label} claim scope differs")
        controls = capture["host_controls"]
        require(controls["process_priority"] == "High", f"{label} process priority differs")
        require(controls["process_affinity"]["requested_mask"] == "0x55555", f"{label} affinity differs")
        require(controls["cuda_high_priority"], f"{label} CUDA priority differs")
        require(controls["gpu_clock"]["requested_mhz"] == 2745, f"{label} GPU clock differs")
        require(controls["gpu_clock"]["lock_applied"], f"{label} GPU clock was not locked")
        require(
            len(controls["admission_windows"]) == spec["repetitions"] * 2,
            f"{label} admission count differs",
        )
        for admission in controls["admission_windows"]:
            require(
                len(admission["accepted_samples"]) == 3,
                f"{label} admission sample count differs",
            )
            require(
                all(
                    sample["utilization_percent"] <= 15
                    and sample["memory_free_mib"] >= 23_000
                    for sample in admission["accepted_samples"]
                ),
                f"{label} admission threshold differs",
            )

        rows = capture["task_pairs"]
        require([row["id"] for row in rows] == EXPECTED_TASK_IDS, f"{label} task vector differs")
        require(
            [row["expected"] for row in rows] == EXPECTED_ANSWERS,
            f"{label} expected answers differ",
        )
        for row in rows:
            for mode in ("base", "candidate"):
                result = row[mode]
                require(result["max_tokens"] == spec["max_tokens_override"], f"{label} row token budget differs")
                require(
                    result["truncated"]
                    == (result["finish_reason"] == "length" or result["completion_tokens"] >= result["max_tokens"]),
                    f"{label} truncation declaration differs",
                )
        summary = paired_summary(rows)
        require(
            len(capture["paired_runs"]) == spec["repetitions"],
            f"{label} paired repetition count differs",
        )
        require(all(pair == summary for pair in capture["paired_runs"]), f"{label} paired summary differs")
        require(summary["gains"] == 0 and summary["losses"] == 0, f"{label} lenient loss vector differs")
        require(summary["strict_gains"] == 0 and summary["strict_losses"] == 0, f"{label} strict loss vector differs")
        require(summary["prediction_parity"] == 5, f"{label} prediction parity differs")
        require(summary["content_sha256_parity"] == 0, f"{label} output parity differs")
        require(summary["both_untruncated"] == 5 - spec["truncated"], f"{label} untruncated count differs")
        require(summary["both_untruncated_prediction_parity"] == summary["both_untruncated"], f"{label} untruncated parity differs")
        for mode in ("q6-off", "q6-dspark"):
            values = capture["modes"][mode]
            require(values["prediction_stable_tasks"] == 5, f"{label} prediction stability differs")
            require(values["content_stable_tasks"] == 5, f"{label} content stability differs")
            near(values["accuracy"]["mean"], 0.8, f"{label} accuracy")
            near(values["strict_accuracy"]["mean"], 0.8, f"{label} strict accuracy")
            require(values["truncated"]["mean"] == spec["truncated"], f"{label} truncation aggregate differs")
        near(capture["modes"]["q6-off"]["aggregate_completion_tokens_per_second"]["mean"], spec["base_tps"], f"{label} base throughput")
        near(capture["modes"]["q6-dspark"]["aggregate_completion_tokens_per_second"]["mean"], spec["candidate_tps"], f"{label} candidate throughput")
        runtime = capture["modes"]["q6-dspark"]["speculative_runtime"]
        require(runtime["strategy"] == "dspark", f"{label} strategy differs")
        near(runtime["overall"]["weighted_acceptance_rate"]["mean"], 0.5444887118193891, f"{label} acceptance")
        near(runtime["overall"]["verified_tokens_per_target_pass"]["mean"], 2.7035490605427976, f"{label} tokens per pass")
        require(runtime["overall"]["fallback_replays"]["mean"] == 0, f"{label} replay differs")
        require(runtime["overall"]["rollback_guard_activations"]["mean"] == 0, f"{label} guard differs")
        summaries[label] = summary

    speedup = CAPTURE_SPECS["tokens_512"]["candidate_tps"] / CAPTURE_SPECS["tokens_512"]["base_tps"]
    claim = evidence["claim"]
    near(claim["tokens_512_workload_speedup"], speedup, "published 512-token speedup")
    require(claim["tokens_1024_quality_only"], "1024-token scope differs")
    selected_non_regression = all(
        summary["gains"] == 0 and summary["losses"] == 0
        for summary in summaries.values()
    )
    require(claim["selected_answer_non_regression"] == selected_non_regression, "selected non-regression declaration differs")
    production_eligible = summaries["tokens_1024"]["content_sha256_parity"] == 5
    require(claim["production_default_eligible"] == production_eligible, "production-default declaration differs")
    require(
        claim["classification"]
        == "diagnostic-answer-parity-output-divergence",
        "claim classification differs",
    )
    if require_production_default:
        require(
            production_eligible,
            "production-default gate failed: exact output parity is "
            f"{summaries['tokens_1024']['content_sha256_parity']}/5",
        )
    return {
        "status": "passed",
        "power_commit": source["power_commit"],
        "tokens_512_losses": summaries["tokens_512"]["losses"],
        "tokens_1024_losses": summaries["tokens_1024"]["losses"],
        "tokens_1024_both_untruncated": summaries["tokens_1024"]["both_untruncated"],
        "tokens_1024_prediction_parity": summaries["tokens_1024"]["prediction_parity"],
        "tokens_512_workload_speedup": speedup,
        "production_default_eligible": production_eligible,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="build path-free follow-up evidence")
    capture.add_argument("--tokens-512-root", type=Path, required=True)
    capture.add_argument("--tokens-1024-root", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify checked-in follow-up evidence")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--require-production-default", action="store_true")
    verify.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "capture":
            evidence = build_evidence(
                args.tokens_512_root.resolve(),
                args.tokens_1024_root.resolve(),
            )
            result = verify_evidence(evidence, require_production_default=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, indent=2))
            return 0
        evidence = load_json(args.evidence)
        result = verify_evidence(evidence, args.require_production_default)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Adaptive DSpark quality follow-up: PASS")
            print(f"  512-token workload speedup: {result['tokens_512_workload_speedup']:.3f}x")
            print("  1024-token paired answers: 5/5")
            print("  production default: no")
        return 0
    except (EvidenceError, KeyError, StopIteration, TypeError, ValueError) as error:
        print(f"adaptive DSpark quality follow-up failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
