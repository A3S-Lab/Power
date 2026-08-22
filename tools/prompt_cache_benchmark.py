#!/usr/bin/env python3
"""Capture reproducible cold/warm prompt-prefix cache evidence from Power."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "a3s.power.prompt-cache-benchmark.v1"
METRIC_NAMES = (
    "power_prompt_cache_requests_total",
    "power_prompt_cache_hits_total",
    "power_prompt_cache_misses_total",
    "power_prompt_cache_reused_tokens_total",
    "power_prompt_cache_evaluated_tokens_total",
    "power_prompt_cache_evictions_total",
    "power_prompt_cache_entries",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40,64}$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
LABEL_PATTERN = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


class BenchmarkError(RuntimeError):
    """Raised when a capture cannot meet the evidence contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _decode_prometheus_label(value: str) -> str:
    return value.replace(r"\n", "\n").replace(r'\"', '"').replace(r"\\", "\\")


def parse_prompt_cache_metrics(text: str, backend: str) -> dict[str, int]:
    """Extract the complete cache snapshot for one backend label."""

    values: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            identity, raw_value = line.rsplit(None, 1)
        except ValueError:
            continue
        if "{" not in identity or not identity.endswith("}"):
            continue
        name, raw_labels = identity.split("{", 1)
        if name not in METRIC_NAMES:
            continue
        labels = {
            key: _decode_prometheus_label(value)
            for key, value in LABEL_PATTERN.findall(raw_labels[:-1])
        }
        if labels.get("backend") != backend:
            continue
        try:
            parsed = float(raw_value)
        except ValueError as error:
            raise BenchmarkError(f"metric {name} is not numeric: {raw_value}") from error
        if not parsed.is_integer() or parsed < 0:
            raise BenchmarkError(f"metric {name} must be a non-negative integer")
        values[name] = int(parsed)

    missing = [name for name in METRIC_NAMES if name not in values]
    if missing:
        raise BenchmarkError(
            f"backend {backend!r} omitted prompt-cache metrics: {', '.join(missing)}"
        )
    return values


def metric_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for name in METRIC_NAMES:
        change = after[name] - before[name]
        if name != "power_prompt_cache_entries" and change < 0:
            raise BenchmarkError(f"counter {name} moved backwards")
        delta[name] = change
    return delta


def validate_call_delta(
    delta: Mapping[str, int], *, expect_hit: bool, minimum_reused_tokens: int
) -> None:
    expected_hits = 1 if expect_hit else 0
    expected_misses = 0 if expect_hit else 1
    exact = {
        "power_prompt_cache_requests_total": 1,
        "power_prompt_cache_hits_total": expected_hits,
        "power_prompt_cache_misses_total": expected_misses,
    }
    for name, expected in exact.items():
        if delta[name] != expected:
            raise BenchmarkError(
                f"isolated cache delta {name} was {delta[name]}, expected {expected}"
            )

    reused = delta["power_prompt_cache_reused_tokens_total"]
    if expect_hit and reused < minimum_reused_tokens:
        raise BenchmarkError(
            f"warm request reused {reused} tokens, below {minimum_reused_tokens}"
        )
    if not expect_hit and reused != 0:
        raise BenchmarkError(f"fresh cache key unexpectedly reused {reused} tokens")
    if delta["power_prompt_cache_evaluated_tokens_total"] <= 0:
        raise BenchmarkError("request reported no evaluated prompt tokens")


def _request_headers(api_key: str | None, content_type: bool = False) -> dict[str, str]:
    headers = {"user-agent": "a3s-power-prompt-cache-benchmark/1"}
    if content_type:
        headers["content-type"] = "application/json"
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def http_bytes(
    base_url: str,
    path: str,
    *,
    api_key: str | None,
    timeout_seconds: float,
    payload: Mapping[str, Any] | None = None,
) -> bytes:
    body = None
    method = "GET"
    if payload is not None:
        method = "POST"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers=_request_headers(api_key, payload is not None),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        details = error.read(4096).decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {error.code} for {path}: {details}") from error
    except urllib.error.URLError as error:
        raise BenchmarkError(f"request failed for {path}: {error.reason}") from error


def get_json(
    base_url: str, path: str, *, api_key: str | None, timeout_seconds: float
) -> dict[str, Any]:
    raw = http_bytes(
        base_url, path, api_key=api_key, timeout_seconds=timeout_seconds
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{path} did not return JSON") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"{path} did not return a JSON object")
    return value


def get_metrics(
    base_url: str, backend: str, *, api_key: str | None, timeout_seconds: float
) -> dict[str, int]:
    raw = http_bytes(
        base_url, "/metrics", api_key=api_key, timeout_seconds=timeout_seconds
    )
    return parse_prompt_cache_metrics(raw.decode("utf-8"), backend)


def run_streaming_completion(
    base_url: str,
    model: str,
    prompt: str,
    *,
    cache_key: str | None,
    api_key: str | None,
    timeout_seconds: float,
    max_tokens: int,
    num_ctx: int,
    num_batch: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "num_ctx": num_ctx,
        "num_batch": num_batch,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cache_key is not None:
        payload["prompt_cache_key"] = cache_key

    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ),
        headers=_request_headers(api_key, True),
        method="POST",
    )
    output_parts: list[str] = []
    final_event: dict[str, Any] | None = None
    started_ns = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise BenchmarkError("completion stream emitted invalid JSON") from error
                if not isinstance(event, dict):
                    raise BenchmarkError("completion stream emitted a non-object event")
                if "error" in event:
                    raise BenchmarkError(f"completion stream failed: {event['error']}")
                for choice in event.get("choices", []):
                    text = choice.get("text")
                    if isinstance(text, str):
                        output_parts.append(text)
                if "a3s_performance" in event:
                    final_event = event
    except urllib.error.HTTPError as error:
        details = error.read(4096).decode("utf-8", errors="replace")
        raise BenchmarkError(
            f"HTTP {error.code} for /v1/completions: {details}"
        ) from error
    except urllib.error.URLError as error:
        raise BenchmarkError(f"completion request failed: {error.reason}") from error
    wall_duration_ns = time.perf_counter_ns() - started_ns

    if final_event is None:
        raise BenchmarkError("completion stream omitted a3s_performance")
    performance = final_event.get("a3s_performance")
    usage = final_event.get("usage")
    receipt = final_event.get("attestation_receipt")
    if not isinstance(performance, dict) or not isinstance(usage, dict):
        raise BenchmarkError("final completion event omitted timing or usage evidence")
    prompt_eval_ns = performance.get("prompt_eval_duration_ns")
    if not isinstance(prompt_eval_ns, int) or prompt_eval_ns < 0:
        raise BenchmarkError("backend omitted prompt_eval_duration_ns")
    ttft_ns = performance.get("time_to_first_token_ns")
    if not isinstance(ttft_ns, int) or ttft_ns < 0:
        raise BenchmarkError("server omitted time_to_first_token_ns")
    if cache_key is not None:
        try:
            bound_key_hash = receipt["decoding"]["parameters"][
                "prompt_cache_key_sha256"
            ]
        except (KeyError, TypeError) as error:
            raise BenchmarkError("receipt omitted prompt_cache_key_sha256") from error
        if bound_key_hash != sha256_text(cache_key):
            raise BenchmarkError("receipt cache-key digest does not match the request")

    output = "".join(output_parts)
    return {
        "wall_duration_ns": wall_duration_ns,
        "time_to_first_token_ns": ttft_ns,
        "prompt_eval_duration_ns": prompt_eval_ns,
        "usage": usage,
        "output_sha256": sha256_text(output),
        "output_bytes": len(output.encode("utf-8")),
        "attestation_receipt_sha256": final_event.get(
            "attestation_receipt_sha256"
        ),
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cold_ttft = [sample["cold"]["time_to_first_token_ns"] for sample in samples]
    warm_ttft = [sample["warm"]["time_to_first_token_ns"] for sample in samples]
    cold_eval = [sample["cold"]["prompt_eval_duration_ns"] for sample in samples]
    warm_eval = [sample["warm"]["prompt_eval_duration_ns"] for sample in samples]
    cold_eval_median = float(statistics.median(cold_eval))
    warm_eval_median = float(statistics.median(warm_eval))
    cold_ttft_median = float(statistics.median(cold_ttft))
    warm_ttft_median = float(statistics.median(warm_ttft))
    reused_tokens = sum(
        sample["warm"]["metrics_delta"][
            "power_prompt_cache_reused_tokens_total"
        ]
        for sample in samples
    )
    cold_evaluated = sum(
        sample["cold"]["metrics_delta"][
            "power_prompt_cache_evaluated_tokens_total"
        ]
        for sample in samples
    )
    warm_evaluated = sum(
        sample["warm"]["metrics_delta"][
            "power_prompt_cache_evaluated_tokens_total"
        ]
        for sample in samples
    )
    return {
        "pairs": len(samples),
        "median_cold_prompt_eval_ns": int(cold_eval_median),
        "median_warm_prompt_eval_ns": int(warm_eval_median),
        "prompt_eval_speedup": cold_eval_median / max(warm_eval_median, 1.0),
        "median_cold_ttft_ns": int(cold_ttft_median),
        "median_warm_ttft_ns": int(warm_ttft_median),
        "ttft_speedup": cold_ttft_median / max(warm_ttft_median, 1.0),
        "reused_tokens": reused_tokens,
        "cold_evaluated_tokens": cold_evaluated,
        "warm_evaluated_tokens": warm_evaluated,
        "evaluated_token_reduction": 1.0
        - (warm_evaluated / max(cold_evaluated, 1)),
    }


def _validated_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    parsed_url = urllib.parse.urlparse(args.base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        parser.error("--base-url must be an absolute HTTP(S) URL")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        parser.error("--run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}")
    if not REVISION_PATTERN.fullmatch(args.server_revision):
        parser.error("--server-revision must be a 40-64 character hexadecimal revision")
    if not SHA256_PATTERN.fullmatch(args.expected_model_sha256):
        parser.error("--expected-model-sha256 must be a hexadecimal SHA-256 digest")
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--server-revision", required=True)
    parser.add_argument("--prefix-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-backend", default="llama.cpp")
    parser.add_argument("--api-key-env", default="A3S_POWER_API_KEY")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-batch", type=int, default=512)
    parser.add_argument("--minimum-prefix-bytes", type=int, default=4096)
    parser.add_argument("--minimum-reused-tokens", type=int, default=256)
    parser.add_argument("--minimum-prompt-eval-speedup", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser


def capture(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "pairs",
        "max_tokens",
        "num_ctx",
        "num_batch",
        "minimum_prefix_bytes",
        "minimum_reused_tokens",
    ):
        if getattr(args, name) <= 0:
            raise BenchmarkError(f"--{name.replace('_', '-')} must be greater than zero")
    if args.minimum_prompt_eval_speedup < 0 or args.timeout_seconds <= 0:
        raise BenchmarkError("speedup and timeout bounds must be non-negative")
    try:
        prefix_bytes = args.prefix_file.read_bytes()
    except OSError as error:
        raise BenchmarkError(f"failed to read prefix file: {error}") from error
    if len(prefix_bytes) < args.minimum_prefix_bytes:
        raise BenchmarkError(
            f"prefix has {len(prefix_bytes)} bytes, below {args.minimum_prefix_bytes}"
        )
    try:
        prefix = prefix_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BenchmarkError("prefix file must be UTF-8") from error
    shared_prefix = prefix + "\n\n[A3S prompt-cache benchmark boundary]\n"
    suffixes = (
        "Branch A: answer with the single word alpha.\n",
        "Branch B: answer with the single word beta.\n",
    )
    api_key = os.environ.get(args.api_key_env) or None

    health = get_json(
        args.base_url,
        "/health",
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
    )
    cache_status = health.get("prompt_cache")
    if not isinstance(cache_status, dict) or not cache_status.get("enabled"):
        raise BenchmarkError("server does not advertise prompt-cache support")
    if args.expected_backend not in cache_status.get("supported_backends", []):
        raise BenchmarkError(
            f"server does not advertise backend {args.expected_backend!r} for prompt caching"
        )
    inference = health.get("inference", {})
    if inference.get("suppress_token_metrics"):
        raise BenchmarkError("token metric suppression must be disabled")
    if inference.get("timing_padding_ms") is not None:
        raise BenchmarkError("timing padding must be disabled for a timing capture")
    if inference.get("num_parallel") != 1:
        raise BenchmarkError("num_parallel must equal one for an isolated capture")
    speculative = health.get("speculative", {})
    if speculative.get("mode") not in {"off", "auto"}:
        raise BenchmarkError("speculative mode must be off or auto for cache isolation")

    encoded_model = urllib.parse.quote(args.model, safe="")
    model_info = get_json(
        args.base_url,
        f"/v1/models/{encoded_model}",
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
    )
    actual_model_hash = str(model_info.get("sha256", "")).lower()
    if actual_model_hash != args.expected_model_sha256.lower():
        raise BenchmarkError(
            f"model SHA-256 was {actual_model_hash or '<missing>'}, expected {args.expected_model_sha256.lower()}"
        )

    run_streaming_completion(
        args.base_url,
        args.model,
        "A3S prompt-cache benchmark model-load warmup. Reply OK.",
        cache_key=None,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        num_ctx=args.num_ctx,
        num_batch=args.num_batch,
    )
    initial_metrics = get_metrics(
        args.base_url,
        args.expected_backend,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
    )

    samples: list[dict[str, Any]] = []
    previous_metrics = initial_metrics
    for index in range(args.pairs):
        cold_suffix = index % 2
        warm_suffix = 1 - cold_suffix
        cache_key = f"a3s-pcache-bench-{args.run_id}-{index}"
        cold_prompt = shared_prefix + suffixes[cold_suffix]
        warm_prompt = cold_prompt + "\n" + suffixes[warm_suffix]

        cold = run_streaming_completion(
            args.base_url,
            args.model,
            cold_prompt,
            cache_key=cache_key,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            num_ctx=args.num_ctx,
            num_batch=args.num_batch,
        )
        after_cold = get_metrics(
            args.base_url,
            args.expected_backend,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
        )
        cold_delta = metric_delta(previous_metrics, after_cold)
        validate_call_delta(
            cold_delta,
            expect_hit=False,
            minimum_reused_tokens=args.minimum_reused_tokens,
        )
        cold["metrics_delta"] = cold_delta

        warm = run_streaming_completion(
            args.base_url,
            args.model,
            warm_prompt,
            cache_key=cache_key,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            num_ctx=args.num_ctx,
            num_batch=args.num_batch,
        )
        after_warm = get_metrics(
            args.base_url,
            args.expected_backend,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
        )
        warm_delta = metric_delta(after_cold, after_warm)
        validate_call_delta(
            warm_delta,
            expect_hit=True,
            minimum_reused_tokens=args.minimum_reused_tokens,
        )
        warm["metrics_delta"] = warm_delta
        samples.append(
            {
                "index": index,
                "suffix_order": [cold_suffix, warm_suffix],
                "caller_key_sha256": sha256_text(cache_key),
                "cold": cold,
                "warm": warm,
            }
        )
        previous_metrics = after_warm

    summary = summarize_samples(samples)
    accepted = summary["prompt_eval_speedup"] >= args.minimum_prompt_eval_speedup
    if not accepted:
        raise BenchmarkError(
            "median prompt-evaluation speedup "
            f"{summary['prompt_eval_speedup']:.3f}x is below "
            f"{args.minimum_prompt_eval_speedup:.3f}x"
        )

    return {
        "schema": SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "accepted": True,
        "server_revision": args.server_revision.lower(),
        "server": {
            "base_url": args.base_url,
            "version": health.get("version"),
            "authenticated": api_key is not None,
            "speculative": speculative,
            "inference": inference,
            "prompt_cache": cache_status,
        },
        "model": model_info,
        "backend": args.expected_backend,
        "prefix": {
            "sha256": sha256_bytes(prefix_bytes),
            "bytes": len(prefix_bytes),
            "shared_bytes": len(shared_prefix.encode("utf-8")),
        },
        "request": {
            "pairs": args.pairs,
            "max_tokens": args.max_tokens,
            "num_ctx": args.num_ctx,
            "num_batch": args.num_batch,
            "temperature": 0,
            "seed": 0,
        },
        "thresholds": {
            "minimum_reused_tokens": args.minimum_reused_tokens,
            "minimum_prompt_eval_speedup": args.minimum_prompt_eval_speedup,
        },
        "metrics_before": initial_metrics,
        "metrics_after": previous_metrics,
        "summary": summary,
        "samples": samples,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = build_parser()
    args = _validated_args(parser)
    try:
        report = capture(args)
        write_report(args.output, report)
    except BenchmarkError as error:
        print(f"prompt-cache benchmark failed: {error}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(
        "prompt-cache benchmark accepted: "
        f"prefill {summary['prompt_eval_speedup']:.3f}x, "
        f"TTFT {summary['ttft_speedup']:.3f}x, "
        f"reused {summary['reused_tokens']} tokens"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
