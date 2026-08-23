#!/usr/bin/env python3
"""Package and verify the pinned adaptive DSpark benchmark evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dspark_adaptive_capture import build_evidence
from dspark_adaptive_evidence_core import (
    EvidenceError,
    load_json,
    verify_evidence,
)

__all__ = ["EvidenceError", "load_json", "verify_evidence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="build path-free evidence")
    capture.add_argument("--quality-root", type=Path, required=True)
    capture.add_argument("--peak-report", type=Path, required=True)
    capture.add_argument("--peak-preflight", type=Path, required=True)
    capture.add_argument("--peak-environment", type=Path, required=True)
    capture.add_argument("--peak-server-log", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify checked-in evidence")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--require-production-default", action="store_true")
    verify.add_argument("--json", action="store_true")
    return parser.parse_args()


def print_result(result: dict[str, object]) -> None:
    print("Adaptive DSpark evidence: PASS")
    print(
        "  peak median/minimum: "
        f"{result['peak_median_decode_tokens_per_second']:.3f} / "
        f"{result['peak_minimum_decode_tokens_per_second']:.3f} token/s"
    )
    print(f"  workload speedup:    {result['workload_speedup']:.3f}x")
    print(
        "  paired gains/losses: "
        f"{result['quality_gains']} / {result['quality_losses']}"
    )
    print("  production default: no")


def main() -> int:
    args = parse_args()
    try:
        if args.command == "capture":
            evidence = build_evidence(
                args.quality_root.resolve(),
                args.peak_report.resolve(),
                args.peak_preflight.resolve(),
                args.peak_environment.resolve(),
                args.peak_server_log.resolve(),
            )
            result = verify_evidence(evidence, require_production_default=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, indent=2))
            return 0

        evidence = load_json(args.evidence)
        result = verify_evidence(evidence, args.require_production_default)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_result(result)
        return 0
    except (EvidenceError, KeyError, TypeError, ValueError) as error:
        print(f"adaptive DSpark evidence failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
