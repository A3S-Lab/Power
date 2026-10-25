# DeepSeek single-4090D native deployment

The managed Xolotl backend runs one DeepSeek instance on GPU6 through a3s-power. ACL declares the native executable, weights, FP4 KV, 1M context, DSpark depth 4, confidence 0.6000, CPU eviction 4096, indexer tile 2^26, and process/admission bounds. The shared OpenAI adapter handles WorkBuddy tool calls and message annotations for native backends.

Concurrent readiness checks previously used try_lock while another readiness probe held the lifecycle lock over HTTP, producing a false 503 on a loaded, unchanged process. Readiness now waits within a two-second deadline; lifecycle failures and unavailable upstreams still refuse readiness.

## Commands and counts

Executed from the initialized Aster root with Rust 1.97.1 and the root lockfile:

- cargo test --locked -p a3s-power --no-default-features --features server --lib: 2030 passed, 0 failed.
- Focused backend::xolotl tests: 45 passed, including concurrent readiness and the lifecycle deadline.
- cargo build --release --locked -p a3s-power --no-default-features --features server --bin a3s-power: passed. The compiler artifact was captured before shared-target reuse.
- Scoped rustfmt --check (edition 2021): passed.
- Live four-pair concurrent readiness reproduction: 4 of 8 probes incorrectly returned 503 before the fix; 32 of 32 concurrent probes returned 200 after deployment, with an unchanged Power PID.
- Native-engine profile measurements retained in the parent repository: warm 128-token decode median 34.8 tokens/s versus 26.0 without DSpark; cold 1,000,123-token prefill 2865.0067 seconds versus 4015.662 seconds with the prior exact profile. All five quality probes and ten WorkBuddy requests passed in that campaign.

The deployed Power binary SHA256 is 899084c8e07d7da2a902169dd21d6dc4e294352caf7b110c4eded34a78e59d28. Native engine, model weights and selected engine arguments are unchanged from the measured campaign. Parent-repository deployment evidence retains the raw logs, lifecycle receipts, exact profile, comparison inputs and preserved-service checks.

## Explicitly not claimed

The full cold 1M experiment was not repeated for this readiness-only wrapper rebuild. Results do not establish a hardware maximum, arbitrary-task quality equivalence, or online draft-weight mutation. No Box shim, image runtime, second DeepSeek instance, or speculative context truncation is introduced. Other sessions' Cloud and API changes are excluded.
