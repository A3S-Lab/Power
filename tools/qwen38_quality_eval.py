#!/usr/bin/env python3
"""Reproducible Qwen3.8 quality and workload-throughput evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qwen38_quality_report import (
    aggregate_reports,
    aggregate_sweep_reports,
    describe,
    exact_mcnemar_p,
    is_truncated,
    pair_metrics,
    render_markdown,
    render_sweep_markdown,
    utc_now,
)


DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
CHOICE_LETTERS = "ABCD"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EXPLICIT_CHOICE = re.compile(
    r"(?i)(?:FINAL(?:\s+ANSWER)?|最终答案)\s*[:：]\s*[\[(]?([ABCD])"
)
EXPLICIT_NUMBER = re.compile(
    r"(?i)(?:FINAL(?:\s+ANSWER)?|最终答案)\s*[:：]\s*\$?\s*(-?[\d,]+(?:\.\d+)?)"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise AssertionError("atomic report replacement did not return or raise")


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int = 120,
    attempts: int = 1,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "a3s-power-qwen38-quality-eval/1",
        },
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("request retry loop did not return or raise")


def fetch_rows(
    dataset: str, config: str, split: str, offset: int, length: int
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    response = request_json("GET", f"{DATASET_ROWS_URL}?{query}", attempts=4)
    rows = response.get("rows")
    if not isinstance(rows, list) or len(rows) != length:
        raise ValueError(
            f"expected {length} rows from {dataset}/{config}/{split}@{offset}"
        )
    return rows


def normalize_number(value: str) -> str:
    cleaned = re.sub(r"\s+", "", value.strip().replace(",", "").replace("$", ""))
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def choice_prompt(question: str, choices: list[str], language: str) -> str:
    rendered = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)
    )
    if language == "zh":
        return (
            "请选择下面单项选择题的最佳答案。只输出一行“最终答案: X”，"
            "其中 X 只能是 A、B、C 或 D；不要输出解释。\n\n"
            f"题目：{question}\n{rendered}\n\n/no_think"
        )
    return (
        "Choose the best answer to the following multiple-choice question. "
        "Return exactly one line in the form `FINAL: X`, where X is A, B, C, "
        "or D. Do not include an explanation.\n\n"
        f"Question: {question}\n{rendered}\n\n/no_think"
    )


def build_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        benchmark = source["benchmark"]
        if benchmark == "mmlu":
            for window in source["windows"]:
                for item in fetch_rows(
                    source["dataset"],
                    source["config"],
                    source["split"],
                    window["offset"],
                    window["length"],
                ):
                    row = item["row"]
                    tasks.append(
                        {
                            "id": f"mmlu:{item['row_idx']}",
                            "benchmark": benchmark,
                            "subject": row["subject"],
                            "prompt": choice_prompt(
                                row["question"], row["choices"], "en"
                            ),
                            "expected": CHOICE_LETTERS[int(row["answer"])],
                            "answer_type": "choice",
                            "max_tokens": 256,
                        }
                    )
        elif benchmark == "gsm8k":
            window = source["windows"][0]
            for item in fetch_rows(
                source["dataset"],
                source["config"],
                source["split"],
                window["offset"],
                window["length"],
            ):
                row = item["row"]
                match = re.search(r"####\s*([^\n]+)", row["answer"])
                if match is None:
                    raise ValueError(f"GSM8K row {item['row_idx']} has no answer")
                tasks.append(
                    {
                        "id": f"gsm8k:{item['row_idx']}",
                        "benchmark": benchmark,
                        "subject": "grade_school_math",
                        "prompt": (
                            "Solve the following problem carefully. You may show concise "
                            "reasoning, but end with a separate line in the exact form "
                            f"`FINAL: number`.\n\n{row['question']}"
                        ),
                        "expected": normalize_number(match.group(1)),
                        "answer_type": "number",
                        "max_tokens": 384,
                    }
                )
        elif benchmark == "ceval":
            window = source["windows"][0]
            for config in source["configs"]:
                for item in fetch_rows(
                    source["dataset"],
                    config,
                    source["split"],
                    window["offset"],
                    window["length"],
                ):
                    row = item["row"]
                    tasks.append(
                        {
                            "id": f"ceval:{config}:{item['row_idx']}",
                            "benchmark": benchmark,
                            "subject": config,
                            "prompt": choice_prompt(
                                row["question"],
                                [row[letter] for letter in CHOICE_LETTERS],
                                "zh",
                            ),
                            "expected": row["answer"].strip().upper(),
                            "answer_type": "choice",
                            "max_tokens": 256,
                        }
                    )
        else:
            raise ValueError(f"unsupported benchmark in task manifest: {benchmark}")
    if len(tasks) != 100 or len({task["id"] for task in tasks}) != 100:
        raise ValueError(f"expected 100 unique tasks, got {len(tasks)}")
    return tasks


def load_tasks(path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    digest = sha256_text(canonical_json(tasks))
    expected = manifest["expected_tasks_sha256"]
    if digest != expected:
        raise ValueError(f"task digest mismatch: expected {expected}, got {digest}")
    return tasks, digest


def select_tasks(
    tasks: list[dict[str, Any]], selection_path: Path
) -> tuple[list[dict[str, Any]], str]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "a3s.power.quality-eval.selection.v1":
        raise ValueError("unsupported task-selection schema")
    task_ids = selection.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task selection must contain unique task_ids")
    by_id = {task["id"]: task for task in tasks}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"task selection references unknown IDs: {missing}")
    selected = [by_id[task_id] for task_id in task_ids]
    digest = sha256_text(canonical_json(selected))
    expected = selection.get("expected_tasks_sha256")
    if digest != expected:
        raise ValueError(f"task selection digest mismatch: expected {expected}, got {digest}")
    return selected, digest


def prepare_tasks(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_path.exists():
        tasks, digest = load_tasks(output_path, manifest_path)
        print(f"reused {len(tasks)} tasks with SHA-256 {digest}")
        return
    tasks = build_tasks(manifest)
    digest = sha256_text(canonical_json(tasks))
    if digest != manifest["expected_tasks_sha256"]:
        raise ValueError(
            "fetched dataset rows do not match the reviewed task manifest: "
            f"expected {manifest['expected_tasks_sha256']}, got {digest}"
        )
    atomic_write(
        output_path,
        {
            "schema": "a3s.power.quality-eval.tasks.v1",
            "created_at": utc_now(),
            "manifest_sha256": sha256_file(manifest_path),
            "tasks_sha256": digest,
            "tasks": tasks,
        },
    )
    print(f"prepared {len(tasks)} tasks with SHA-256 {digest}")


def strict_prediction(content: str, answer_type: str) -> str | None:
    matches = (
        EXPLICIT_CHOICE.findall(content)
        if answer_type == "choice"
        else EXPLICIT_NUMBER.findall(content)
    )
    if not matches:
        return None
    value = matches[-1]
    return value.upper() if answer_type == "choice" else normalize_number(value)


def lenient_prediction(content: str, answer_type: str) -> str | None:
    explicit = strict_prediction(content, answer_type)
    if explicit is not None:
        return explicit
    if answer_type == "choice":
        matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", content.upper())
        return matches[-1] if matches else None
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", content)
    return normalize_number(matches[-1]) if matches else None


def render_qwen_chatml(user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a careful and concise benchmark solver.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total": 0,
            "completed": 0,
            "errors": 0,
            "correct": 0,
            "accuracy": 0.0,
            "strict_correct": 0,
            "strict_accuracy": 0.0,
            "truncated": 0,
            "prediction_missing": 0,
            "explicit_prediction_missing": 0,
            "median_latency_seconds": None,
            "mean_latency_seconds": None,
            "total_latency_seconds": 0.0,
            "completion_tokens": 0,
            "aggregate_completion_tokens_per_second": None,
        }
    completed = [row for row in rows if row.get("error") is None]
    latencies = [float(row["latency_seconds"]) for row in completed]
    completion_tokens = [int(row["usage"]["completion_tokens"]) for row in completed]
    return {
        "total": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "strict_correct": sum(bool(row["strict_correct"]) for row in rows),
        "strict_accuracy": sum(bool(row["strict_correct"]) for row in rows) / len(rows),
        "truncated": sum(is_truncated(row) for row in completed),
        "prediction_missing": sum(row.get("prediction") is None for row in rows),
        "explicit_prediction_missing": sum(
            row.get("strict_prediction") is None for row in rows
        ),
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "total_latency_seconds": sum(latencies),
        "completion_tokens": sum(completion_tokens),
        "aggregate_completion_tokens_per_second": (
            sum(completion_tokens) / sum(latencies) if latencies else None
        ),
    }


def report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        benchmark: summarize_rows(
            [row for row in rows if row["benchmark"] == benchmark]
        )
        for benchmark in ("mmlu", "gsm8k", "ceval")
    }
    summary["overall"] = summarize_rows(rows)
    return summary


def parse_mtp_log(path: Path, task_count: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    fields = (
        "rounds",
        "drafted_tokens",
        "accepted_tokens",
        "emitted_tokens",
        "verified_emitted_tokens",
        "tokens_per_second",
        "fallback_replays",
    )
    optional_fields = (
        "rollback_guard_activations",
        "rollback_guard_draft_limit",
        "target_only_tokens",
        "fr_target_samples",
        "fr_target_samples_in_token_id_prefix",
        "fr_corrections_outside_token_id_prefix",
        # Legacy prefix-vocabulary log keys remain readable so historical
        # captures can be regenerated without relabeling them as ranked d2t
        # membership coverage.
        "fr_target_samples_in_vocab",
        "fr_rejected_rounds",
        "fr_corrections_outside_vocab",
        "draft_duration_ns",
        "target_decode_duration_ns",
        "target_only_decode_duration_ns",
        "accepted_prefix_sync_duration_ns",
        "sampling_duration_ns",
    )
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        if "speculative completion finished" not in line:
            continue
        record: dict[str, Any] = {}
        for field in fields:
            match = re.search(rf"\b{field}=([0-9]+(?:\.[0-9]+)?)", line)
            if match is None:
                raise ValueError(f"missing {field} in speculative completion log")
            record[field] = float(match.group(1))
        for field in optional_fields:
            match = re.search(rf"\b{field}=([0-9]+(?:\.[0-9]+)?)", line)
            record[field] = float(match.group(1)) if match else 0.0
        circuit = re.search(r"\btarget_only_after_round=(?:Some\(([0-9]+)\)|None)", line)
        record["target_only_after_round"] = (
            int(circuit.group(1)) if circuit and circuit.group(1) else None
        )
        histogram = re.search(r"\bdraft_limit_histogram=\[([^]]*)\]", line)
        record["draft_limit_histogram"] = (
            [int(value.strip()) for value in histogram.group(1).split(",") if value.strip()]
            if histogram
            else []
        )
        records.append(record)
    if not records:
        return None
    if len(records) < task_count:
        raise ValueError(f"expected at least {task_count} MTP records, got {len(records)}")
    records = records[-task_count:]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        drafted = sum(row["drafted_tokens"] for row in selected)
        accepted = sum(row["accepted_tokens"] for row in selected)
        emitted = sum(row["emitted_tokens"] for row in selected)
        measured = sum(
            row["emitted_tokens"] / row["tokens_per_second"] for row in selected
        )
        fr_target_samples = sum(row["fr_target_samples"] for row in selected)
        fr_target_samples_in_token_id_prefix = sum(
            row["fr_target_samples_in_token_id_prefix"]
            + row["fr_target_samples_in_vocab"]
            for row in selected
        )
        fr_rejected_rounds = sum(row["fr_rejected_rounds"] for row in selected)
        fr_corrections_outside_token_id_prefix = sum(
            row["fr_corrections_outside_token_id_prefix"]
            + row["fr_corrections_outside_vocab"]
            for row in selected
        )
        histogram_length = max(
            (len(row["draft_limit_histogram"]) for row in selected), default=0
        )
        draft_limit_histogram = [
            sum(
                row["draft_limit_histogram"][index]
                if index < len(row["draft_limit_histogram"])
                else 0
                for row in selected
            )
            for index in range(histogram_length)
        ]
        return {
            "requests": len(selected),
            "drafted_tokens": int(drafted),
            "accepted_tokens": int(accepted),
            "weighted_acceptance_rate": accepted / drafted,
            "verified_tokens_per_target_pass": sum(
                row["verified_emitted_tokens"] for row in selected
            )
            / sum(row["rounds"] for row in selected),
            "fallback_replays": int(sum(row["fallback_replays"] for row in selected)),
            "rollback_guard_requests": sum(
                row["rollback_guard_activations"] > 0 for row in selected
            ),
            "rollback_guard_activations": int(
                sum(row["rollback_guard_activations"] for row in selected)
            ),
            "target_only_requests": sum(
                row["target_only_after_round"] is not None for row in selected
            ),
            "target_only_tokens": int(
                sum(row["target_only_tokens"] for row in selected)
            ),
            "fr_target_samples": int(fr_target_samples),
            "fr_target_samples_in_token_id_prefix": int(
                fr_target_samples_in_token_id_prefix
            ),
            "fr_target_token_id_prefix_fraction": (
                fr_target_samples_in_token_id_prefix / fr_target_samples
                if fr_target_samples
                else None
            ),
            "fr_rejected_rounds": int(fr_rejected_rounds),
            "fr_corrections_outside_token_id_prefix": int(
                fr_corrections_outside_token_id_prefix
            ),
            "fr_correction_outside_token_id_prefix_fraction": (
                fr_corrections_outside_token_id_prefix / fr_rejected_rounds
                if fr_rejected_rounds
                else None
            ),
            "draft_limit_histogram": draft_limit_histogram,
            "phase_duration_ns": {
                field.removesuffix("_duration_ns"): int(
                    sum(row[field] for row in selected)
                )
                for field in optional_fields
                if field.endswith("_duration_ns")
            },
            "median_reported_tokens_per_second": statistics.median(
                row["tokens_per_second"] for row in selected
            ),
            "aggregate_reported_tokens_per_second": emitted / measured,
        }

    result = {"overall": summarize(records)}
    if task_count == 100:
        result["by_benchmark"] = {
            "mmlu": summarize(records[:50]),
            "gsm8k": summarize(records[50:70]),
            "ceval": summarize(records[70:]),
        }
    return result


def completion_body(
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    num_batch: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": render_qwen_chatml(prompt),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "num_ctx": 4096,
        "num_batch": num_batch,
        "seed": seed,
        "stream": False,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }


def run_evaluation(args: argparse.Namespace) -> None:
    tasks, task_source_digest = load_tasks(args.tasks, args.manifest)
    selection_digest = None
    tasks_digest = task_source_digest
    if args.task_selection is not None:
        tasks, tasks_digest = select_tasks(tasks, args.task_selection)
        selection_digest = sha256_file(args.task_selection)
    health = request_json("GET", f"{args.url}/health", timeout_seconds=30)
    request_settings = {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_ctx": 4096,
        "num_batch": args.num_batch,
        "seed": args.seed,
        "warmup_requests": args.warmup_requests,
        "max_tokens_cap": args.max_tokens_cap,
        "template": "manual-qwen-chatml-v1",
    }
    for _ in range(args.warmup_requests):
        request_json(
            "POST",
            f"{args.url}/v1/completions",
            completion_body(
                args.model,
                "Return exactly `FINAL: B`. What is 17 * 6? A. 92 B. 102 C. 112 D. 122",
                64,
                args.seed,
                args.num_batch,
            ),
            timeout_seconds=args.timeout_seconds,
        )

    identity = {
        "mode_label": args.mode_label,
        "repetition": args.repetition,
        "order_index": args.order_index,
        "model": args.model,
        "model_sha256": args.model_sha256,
        "tasks_sha256": tasks_digest,
        "task_source_sha256": task_source_digest,
        "task_selection_sha256": selection_digest,
        "seed": args.seed,
        "server_sha256": args.server_sha256,
        "power_commit": args.power_commit,
        "request_sha256": sha256_text(canonical_json(request_settings)),
    }
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if report.get(key) != value:
                raise ValueError(f"existing report identity differs for {key}")
    else:
        report = {
            "schema": "a3s.power.quality-eval.report.v3",
            **identity,
            "started_at": utc_now(),
            "request": request_settings,
            "health": health,
            "results": [],
        }

    completed_ids = {row["id"] for row in report["results"]}
    for index, task in enumerate(tasks, start=1):
        if task["id"] in completed_ids:
            continue
        started = time.perf_counter()
        max_tokens = int(task["max_tokens"])
        if args.max_tokens_cap is not None:
            max_tokens = min(max_tokens, args.max_tokens_cap)
        try:
            response = request_json(
                "POST",
                f"{args.url}/v1/completions",
                completion_body(
                    args.model,
                    task["prompt"],
                    max_tokens,
                    args.seed,
                    args.num_batch,
                ),
                timeout_seconds=args.timeout_seconds,
            )
            choice = response["choices"][0]
            content = choice.get("text") or ""
            prediction = lenient_prediction(content, task["answer_type"])
            explicit = strict_prediction(content, task["answer_type"])
            row = {
                "id": task["id"],
                "benchmark": task["benchmark"],
                "subject": task["subject"],
                "expected": task["expected"],
                "answer_type": task["answer_type"],
                "max_tokens": max_tokens,
                "prediction": prediction,
                "strict_prediction": explicit,
                "correct": prediction == task["expected"],
                "strict_correct": explicit == task["expected"],
                "content_sha256": sha256_text(content),
                "finish_reason": choice.get("finish_reason"),
                "latency_seconds": time.perf_counter() - started,
                "usage": response["usage"],
                "receipt_sha256": response.get("attestation_receipt_sha256"),
                "error": None,
            }
            if args.include_content:
                row["content"] = content
        except Exception as error:
            row = {
                "id": task["id"],
                "benchmark": task["benchmark"],
                "subject": task["subject"],
                "expected": task["expected"],
                "answer_type": task["answer_type"],
                "max_tokens": max_tokens,
                "prediction": None,
                "strict_prediction": None,
                "correct": False,
                "strict_correct": False,
                "content_sha256": sha256_text(""),
                "finish_reason": None,
                "latency_seconds": time.perf_counter() - started,
                "usage": None,
                "receipt_sha256": None,
                "error": f"{type(error).__name__}: {error}",
            }
        report["results"].append(row)
        report["summary"] = report_summary(report["results"])
        atomic_write(args.output, report)
        print(
            f"[{args.mode_label} r{args.repetition}] {index:03d}/{len(tasks)} "
            f"{task['id']} expected={task['expected']} predicted={row['prediction']} "
            f"correct={row['correct']} latency={row['latency_seconds']:.3f}s",
            flush=True,
        )

    report["completed_at"] = utc_now()
    report["summary"] = report_summary(report["results"])
    report["speculative_runtime"] = parse_mtp_log(args.server_log, len(tasks))
    atomic_write(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def report_metadata(report: dict[str, Any]) -> dict[str, Any]:
    """Return control-plane fields without carrying model-generated content."""
    request = report.get("request") or {}
    overall = (report.get("summary") or {}).get("overall") or {}
    return {
        "schema": report.get("schema"),
        "mode_label": report.get("mode_label"),
        "repetition": report.get("repetition"),
        "order_index": report.get("order_index"),
        "model": report.get("model"),
        "model_sha256": report.get("model_sha256"),
        "tasks_sha256": report.get("tasks_sha256"),
        "server_sha256": report.get("server_sha256"),
        "power_commit": report.get("power_commit"),
        "seed": report.get("seed"),
        "num_ctx": request.get("num_ctx"),
        "num_batch": request.get("num_batch"),
        "warmup_requests": request.get("warmup_requests"),
        "result_count": len(report.get("results") or []),
        "completed_at": report.get("completed_at"),
        "completed": overall.get("completed"),
        "errors": overall.get("errors"),
        "has_speculative_runtime": report.get("speculative_runtime") is not None,
    }


def inspect_report_command(args: argparse.Namespace) -> None:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(json.dumps(report_metadata(report), ensure_ascii=True, sort_keys=True))


def aggregate_command(args: argparse.Namespace) -> None:
    report_paths = sorted(Path(path) for path in args.reports)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    aggregate = aggregate_reports(reports, comparisons=args.pair)
    aggregate["reports"] = [
        {"path": path.name, "sha256": sha256_file(path)} for path in report_paths
    ]
    atomic_write(args.output_json, aggregate)
    args.output_markdown.write_text(render_markdown(aggregate), encoding="utf-8")
    console_summary = {
        "repetitions": aggregate["repetitions"],
        "modes": {
            mode: {
                "accuracy": metrics["accuracy"]["mean"],
                "strict_accuracy": metrics["strict_accuracy"]["mean"],
                "workload_tokens_per_second": metrics[
                    "aggregate_completion_tokens_per_second"
                ]["mean"],
            }
            for mode, metrics in aggregate["modes"].items()
        },
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


def aggregate_sweep_command(args: argparse.Namespace) -> None:
    report_paths = sorted(Path(path) for path in args.reports)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    aggregate = aggregate_sweep_reports(reports)
    aggregate["reports"] = [
        {"path": path.name, "sha256": sha256_file(path)} for path in report_paths
    ]
    atomic_write(args.output_json, aggregate)
    args.output_markdown.write_text(
        render_sweep_markdown(aggregate), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                mode: {
                    "workload_tokens_per_second": summary[
                        "aggregate_completion_tokens_per_second"
                    ]["mean"],
                    "acceptance": summary.get("speculative_runtime", {})
                    .get("weighted_acceptance_rate", {})
                    .get("mean"),
                }
                for mode, summary in aggregate["modes"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="fetch and hash-lock the task cache")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="run one mode and repetition")
    run.add_argument("--url", default="http://127.0.0.1:11436")
    run.add_argument("--model", default="qwen3.8-27b-q6-k")
    run.add_argument("--mode-label", required=True)
    run.add_argument("--repetition", type=int, required=True)
    run.add_argument("--order-index", type=int, required=True)
    run.add_argument("--model-sha256", required=True)
    run.add_argument("--server-sha256", required=True)
    run.add_argument("--power-commit", required=True)
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-selection", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--server-log", type=Path, required=True)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--num-batch", type=int, default=14)
    run.add_argument("--max-tokens-cap", type=int)
    run.add_argument("--warmup-requests", type=int, default=1)
    run.add_argument("--timeout-seconds", type=int, default=900)
    run.add_argument("--include-content", action="store_true")

    inspect_report = commands.add_parser(
        "inspect-report",
        help="emit safe control-plane metadata for one report",
    )
    inspect_report.add_argument("--report", type=Path, required=True)

    aggregate = commands.add_parser("aggregate", help="aggregate complete reports")
    aggregate.add_argument("--reports", nargs="+", required=True)
    aggregate.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("BASE", "CANDIDATE"),
        help="add a paired per-repetition quality comparison",
    )
    aggregate.add_argument("--output-json", type=Path, required=True)
    aggregate.add_argument("--output-markdown", type=Path, required=True)

    sweep = commands.add_parser(
        "aggregate-sweep", help="aggregate an arbitrary MTP calibration grid"
    )
    sweep.add_argument("--reports", nargs="+", required=True)
    sweep.add_argument("--output-json", type=Path, required=True)
    sweep.add_argument("--output-markdown", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare_tasks(args.manifest, args.output)
    elif args.command == "run":
        run_evaluation(args)
    elif args.command == "inspect-report":
        inspect_report_command(args)
    elif args.command == "aggregate":
        aggregate_command(args)
    elif args.command == "aggregate-sweep":
        aggregate_sweep_command(args)
    else:
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
