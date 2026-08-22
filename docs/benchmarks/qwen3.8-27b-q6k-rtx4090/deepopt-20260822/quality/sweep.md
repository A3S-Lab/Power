# Qwen3.8-27B MTP calibration sweep

Task SHA-256: `bf8e3c044fe21863361ca2d8be6cf47483290042bc93d53663a7a59cefb2d8c7`. Repetitions: 1 per mode.

| Mode | Workload token/s | Acceptance | Tokens / target pass | Target-only requests | Target token-ID prefix | Correction outside prefix | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fr8192-k6-s6-b8-fixed` | 46.923 | 26.8% | 2.591 | 0.000 | 71.9% | 77.7% | 75.0% |
| `off-b8` | 28.713 | n/a | n/a | n/a | n/a | n/a | 75.0% |

Token-ID prefix fields are exact only for the legacy prefix shortlist. They are diagnostics, not ranked d2t vocabulary membership or FR-caused rejection rates.
