"""Pinned adaptive DSpark evidence contract and offline verifier."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from qwen38_quality_report import exact_mcnemar_p


SCHEMA = "a3s.power.dspark-adaptive-evidence.v1"
EXPECTED_IDENTITY = {
    "power_commit": "cbdb3f673446b3532c9683dabc816a149ae27b1f",
    "server_sha256": "92e532aca885d4babdd1bf15bd94ece07e013082a87007543514749b96d0f373",
    "benchmark_client_sha256": "8425d4b3d1fb3166fdca0d6c2690150d7b3421c0f38e68081c528611ec1b3c96",
    "target_sha256": "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    "draft_sha256": "12003c7f2642e2e87e979729e16947a913e2213d82136cb5024a36ec4871fef2",
    "tasks_sha256": "5798257e18b81188749196d34359278dfadf7986776eb2bd66d629cbfc33813c",
    "tasks_file_sha256": "2fd18d52a2e011692c6ffac2547da41ec8b6594e2d35a8aa40fa8648df96b40a",
    "task_manifest_sha256": "e254af009fbee304bb574515aece7630cfb6ea91fd0fece6c0d9b9f02de44a9d",
    "config_sha256": "a8fb13a27d5b30d77bf7d3047b90c4fcae601e6d08b6d2b05dbc46b733eeecb9",
    "prompt_sha256": "d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89",
}
EXPECTED_SOURCE_HASHES = {
    "peak_report_sha256": "418b48d299de33a002e327154775f93f752a9ae50d66ee8af060d7c1b6b83cc5",
    "peak_preflight_sha256": "45c889cecd4623cf5076a08e31600fd65836361d3acf0411c6b62bccd4cb80d4",
    "peak_environment_sha256": "df9140fe78da1842a66af443f40b4874208f1ea2a6ac88dc32175885c7e178b5",
    "peak_server_log_sha256": "7003124e0a7213066ebdd78b8875700116cbbbc6ba94653ada6360833970bab3",
    "quality_environment_sha256": "79ab8e417249f6cd8a1d4b19744849d53d3156468ee1b4a3d84cede2848cadf0",
    "quality_aggregate_sha256": "2cd28b93a892c0942772880088369971ac88713f99caaa6048ac12ea3c647d2d",
    "quality_base_report_sha256": "a239b5a4da6217d0ae37e0261b0923d3d7577f8ae4426e1b9d54c52d15c347b8",
    "quality_candidate_report_sha256": "3db1ce802f9793616b3bcbb3b2689250780ae905fe71a1bc36a9303a1c89ca61",
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


def near(actual: float, expected: float, label: str) -> None:
    require(abs(actual - expected) <= 1e-9, f"{label} differs")


def summarize_task_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = sum(
        not row["base_correct"] and row["candidate_correct"] for row in rows
    )
    losses = sum(
        row["base_correct"] and not row["candidate_correct"] for row in rows
    )
    strict_gains = sum(
        not row["base_strict_correct"] and row["candidate_strict_correct"]
        for row in rows
    )
    strict_losses = sum(
        row["base_strict_correct"] and not row["candidate_strict_correct"]
        for row in rows
    )
    untruncated = [
        row
        for row in rows
        if not row["base_truncated"] and not row["candidate_truncated"]
    ]
    return {
        "task_count": len(rows),
        "base_correct": sum(row["base_correct"] for row in rows),
        "candidate_correct": sum(row["candidate_correct"] for row in rows),
        "gains": gains,
        "losses": losses,
        "exact_mcnemar_p": exact_mcnemar_p(gains, losses),
        "strict_base_correct": sum(row["base_strict_correct"] for row in rows),
        "strict_candidate_correct": sum(
            row["candidate_strict_correct"] for row in rows
        ),
        "strict_gains": strict_gains,
        "strict_losses": strict_losses,
        "strict_exact_mcnemar_p": exact_mcnemar_p(strict_gains, strict_losses),
        "prediction_parity": sum(
            row["base_prediction"] == row["candidate_prediction"]
            for row in rows
        ),
        "content_sha256_parity": sum(row["content_match"] for row in rows),
        "both_untruncated": len(untruncated),
        "both_untruncated_prediction_parity": sum(
            row["base_prediction"] == row["candidate_prediction"]
            for row in untruncated
        ),
    }


def verify_evidence(
    evidence: dict[str, Any], require_production_default: bool
) -> dict[str, Any]:
    require(evidence.get("schema") == SCHEMA, "evidence schema differs")
    source = evidence["source"]
    for field, expected in EXPECTED_IDENTITY.items():
        actual = (
            evidence["inputs"][field]
            if field in evidence["inputs"]
            else evidence["artifacts"]["target"]["sha256"]
            if field == "target_sha256"
            else evidence["artifacts"]["draft"]["sha256"]
            if field == "draft_sha256"
            else source[field]
        )
        require(actual == expected, f"published {field} differs")
    for field, expected in EXPECTED_SOURCE_HASHES.items():
        require(source[field] == expected, f"published {field} differs")
    require(not source["dirty_worktree"], "published tree was dirty")

    for name in ("peak", "quality"):
        controls = evidence["host_controls"][name]
        require(controls["cuda_high_priority"], f"{name} CUDA priority differs")
        require(
            controls["process_priority"] == "High",
            f"{name} process priority differs",
        )
        require(
            controls["gpu_clock"]["requested_mhz"] == 2745,
            f"{name} clock request differs",
        )
        require(
            controls["gpu_clock"]["lock_applied"],
            f"{name} clock lock was not attested",
        )
        require(
            controls["process_affinity"]["requested_mask"] == "0x55555",
            f"{name} affinity differs",
        )
    require(
        evidence["host_controls"]["peak"]["process_affinity"]["effective_mask"]
        == "0x55555",
        "peak effective affinity differs",
    )

    peak = evidence["peak"]
    samples = peak["samples"]
    require(len(samples) == 3, "peak sample count differs")
    rates = [float(sample["decode_tokens_per_second"]) for sample in samples]
    near(
        statistics.median(rates),
        peak["median_decode_tokens_per_second"],
        "peak median",
    )
    near(min(rates), peak["minimum_decode_tokens_per_second"], "peak minimum")
    require(all(rate >= 160.0 for rate in rates), "peak all-sample gate failed")
    require(peak["all_samples_passed"], "peak gate declaration differs")
    require(
        all(sample["output_sha256"] == peak["output_sha256"] for sample in samples),
        "peak output identity differs",
    )
    require(
        all(
            sample["receipt_sha256"] == peak["receipt_sha256"]
            for sample in samples
        ),
        "peak receipt identity differs",
    )
    require(len(peak["runtime"]) == 3, "peak runtime row count differs")
    for runtime in peak["runtime"]:
        require(runtime["rounds"] == 26, "peak target rounds differ")
        require(runtime["drafted_tokens"] == 247, "peak drafted tokens differ")
        require(runtime["accepted_tokens"] == 229, "peak accepted tokens differ")
        require(runtime["fallback_replays"] == 0, "peak replay differs")
        require(
            runtime["rollback_guard_activations"] == 0,
            "peak guard differs",
        )
        near(
            runtime["tokens_per_target_pass"],
            9.807692307692308,
            "peak tokens per pass",
        )

    quality = evidence["quality"]
    computed_pair = summarize_task_pairs(quality["task_pairs"])
    require(computed_pair == quality["paired"], "paired summary differs")
    require(
        quality["base"]["correct"] == computed_pair["base_correct"],
        "base score differs",
    )
    require(
        quality["candidate"]["correct"] == computed_pair["candidate_correct"],
        "candidate score differs",
    )
    require(
        quality["base"]["strict_correct"]
        == computed_pair["strict_base_correct"],
        "base strict score differs",
    )
    require(
        quality["candidate"]["strict_correct"]
        == computed_pair["strict_candidate_correct"],
        "candidate strict score differs",
    )
    require(
        quality["base"]["truncated"]
        == sum(row["base_truncated"] for row in quality["task_pairs"]),
        "base truncation differs",
    )
    require(
        quality["candidate"]["truncated"]
        == sum(row["candidate_truncated"] for row in quality["task_pairs"]),
        "candidate truncation differs",
    )
    runtime = quality["candidate_runtime"]["overall"]
    require(
        quality["candidate_runtime"]["strategy"] == "dspark",
        "quality strategy differs",
    )
    require(runtime["requests"] == 100, "quality runtime request count differs")
    require(runtime["fallback_replays"] == 0, "quality replay differs")
    require(runtime["rollback_guard_activations"] == 0, "quality guard differs")
    near(
        runtime["weighted_acceptance_rate"],
        12375 / 19681,
        "quality acceptance",
    )
    near(
        runtime["verified_tokens_per_target_pass"],
        3.37263604785797,
        "quality tokens per pass",
    )

    base_rate = quality["base"]["aggregate_completion_tokens_per_second"]
    candidate_rate = quality["candidate"][
        "aggregate_completion_tokens_per_second"
    ]
    near(
        candidate_rate / base_rate,
        evidence["claim"]["workload_throughput_speedup"],
        "quality speedup",
    )
    production_eligible = (
        computed_pair["prediction_parity"] == computed_pair["task_count"]
        and computed_pair["content_sha256_parity"] == computed_pair["task_count"]
    )
    require(
        evidence["claim"]["production_default_eligible"]
        == production_eligible,
        "production-default declaration differs",
    )
    expected_classification = (
        "production-default"
        if production_eligible
        else "diagnostic-output-divergence"
    )
    require(
        evidence["claim"]["classification"] == expected_classification,
        "claim classification differs",
    )
    if require_production_default:
        require(
            production_eligible,
            "production-default gate failed: exact output parity is "
            f"{computed_pair['content_sha256_parity']}/{computed_pair['task_count']}",
        )
    return {
        "status": "passed",
        "power_commit": source["power_commit"],
        "peak_median_decode_tokens_per_second": peak[
            "median_decode_tokens_per_second"
        ],
        "peak_minimum_decode_tokens_per_second": peak[
            "minimum_decode_tokens_per_second"
        ],
        "base_workload_tokens_per_second": base_rate,
        "candidate_workload_tokens_per_second": candidate_rate,
        "workload_speedup": candidate_rate / base_rate,
        "quality_gains": computed_pair["gains"],
        "quality_losses": computed_pair["losses"],
        "exact_output_parity": computed_pair["content_sha256_parity"],
        "fallback_replays": runtime["fallback_replays"],
        "production_default_eligible": production_eligible,
    }
