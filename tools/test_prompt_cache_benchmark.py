#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompt_cache_benchmark as benchmark


class PromptCacheBenchmarkTests(unittest.TestCase):
    def test_parse_prompt_cache_metrics_selects_backend(self) -> None:
        lines = []
        for index, name in enumerate(benchmark.METRIC_NAMES):
            lines.append(f'{name}{{backend="other"}} 99')
            lines.append(f'{name}{{backend="llama.cpp"}} {index}')

        parsed = benchmark.parse_prompt_cache_metrics("\n".join(lines), "llama.cpp")

        self.assertEqual(parsed["power_prompt_cache_requests_total"], 0)
        self.assertEqual(parsed["power_prompt_cache_entries"], 6)

    def test_metric_delta_and_validation_distinguish_miss_and_hit(self) -> None:
        before = {name: 0 for name in benchmark.METRIC_NAMES}
        cold_after = dict(before)
        cold_after.update(
            {
                "power_prompt_cache_requests_total": 1,
                "power_prompt_cache_misses_total": 1,
                "power_prompt_cache_evaluated_tokens_total": 1024,
                "power_prompt_cache_entries": 1,
            }
        )
        cold = benchmark.metric_delta(before, cold_after)
        benchmark.validate_call_delta(cold, expect_hit=False, minimum_reused_tokens=512)

        warm_after = dict(cold_after)
        warm_after.update(
            {
                "power_prompt_cache_requests_total": 2,
                "power_prompt_cache_hits_total": 1,
                "power_prompt_cache_reused_tokens_total": 1000,
                "power_prompt_cache_evaluated_tokens_total": 1048,
            }
        )
        warm = benchmark.metric_delta(cold_after, warm_after)
        benchmark.validate_call_delta(warm, expect_hit=True, minimum_reused_tokens=512)

    def test_summary_uses_paired_server_timings_and_token_deltas(self) -> None:
        samples = [
            {
                "cold": {
                    "time_to_first_token_ns": 1000,
                    "prompt_eval_duration_ns": 800,
                    "metrics_delta": {
                        "power_prompt_cache_evaluated_tokens_total": 100,
                    },
                },
                "warm": {
                    "time_to_first_token_ns": 200,
                    "prompt_eval_duration_ns": 100,
                    "metrics_delta": {
                        "power_prompt_cache_reused_tokens_total": 90,
                        "power_prompt_cache_evaluated_tokens_total": 10,
                    },
                },
            },
            {
                "cold": {
                    "time_to_first_token_ns": 1200,
                    "prompt_eval_duration_ns": 1000,
                    "metrics_delta": {
                        "power_prompt_cache_evaluated_tokens_total": 120,
                    },
                },
                "warm": {
                    "time_to_first_token_ns": 300,
                    "prompt_eval_duration_ns": 100,
                    "metrics_delta": {
                        "power_prompt_cache_reused_tokens_total": 110,
                        "power_prompt_cache_evaluated_tokens_total": 10,
                    },
                },
            },
        ]

        summary = benchmark.summarize_samples(samples)

        self.assertEqual(summary["median_cold_prompt_eval_ns"], 900)
        self.assertEqual(summary["median_warm_prompt_eval_ns"], 100)
        self.assertEqual(summary["prompt_eval_speedup"], 9.0)
        self.assertEqual(summary["reused_tokens"], 200)
        self.assertAlmostEqual(summary["evaluated_token_reduction"], 1 - 20 / 220)


if __name__ == "__main__":
    unittest.main()
