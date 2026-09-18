#!/usr/bin/env bash
# Capture an exact-parent Metal release fixture on real Apple Silicon.
#
# Usage (cwd = clean detached freeze-parent worktree; script may live on main):
#   git worktree add ../power-freeze 514031dc74edd72da7c3bfee40144a38d2d91434
#   git show main:docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json \
#     > /outside/local-execution-policy.json
#   cd ../power-freeze
#   POLICY_PATH=/outside/local-execution-policy.json \
#     bash /path/to/main/tools/release-capture/capture-macos-metal.sh /path/to/output-root
#
# Rejects dirty trees and refuses to run when HEAD is not a 40-char SHA.
# By default HEAD must equal the v1.0.0 freeze parent
# (514031dc74edd72da7c3bfee40144a38d2d91434). Override only when intentionally
# recutting a new evidence parent:
#   A3S_POWER_RELEASE_SOURCE_PARENT=<40-hex> bash ...
# Emulated / paravirtual Metal devices are not production evidence.

set -euo pipefail

output_root="${1:?usage: capture-macos-metal.sh <output-root>}"
policy_path="${POLICY_PATH:-docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json}"
expected_parent="${A3S_POWER_RELEASE_SOURCE_PARENT:-514031dc74edd72da7c3bfee40144a38d2d91434}"

test -z "$(git status --porcelain)" || {
  echo "capture requires a clean git worktree" >&2
  exit 1
}

power_commit="$(git rev-parse HEAD)"
test "${#power_commit}" -eq 40
case "$power_commit" in
  *[!0-9a-f]*)
    echo "HEAD must be a lowercase 40-hex SHA (got ${power_commit})" >&2
    exit 1
    ;;
esac

test "${#expected_parent}" -eq 40 || {
  echo "A3S_POWER_RELEASE_SOURCE_PARENT must be a 40-hex SHA" >&2
  exit 1
}
test "$power_commit" = "$expected_parent" || {
  echo "HEAD ${power_commit} is not release source parent ${expected_parent}" >&2
  echo "Detach to that commit before Metal capture, or set A3S_POWER_RELEASE_SOURCE_PARENT only when recutting." >&2
  exit 1
}

test -f "$policy_path" || {
  echo "missing policy file: $policy_path" >&2
  echo "The freeze parent does not contain this file. Export it outside the worktree:" >&2
  echo "  git show main:docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json > /outside/local-execution-policy.json" >&2
  echo "  POLICY_PATH=/outside/local-execution-policy.json bash <script-from-main> <output-root>" >&2
  exit 1
}
policy_hash="$(shasum -a 256 "$policy_path" | awk '{print $1}')"

uname_s="$(uname -s)"
uname_m="$(uname -m)"
test "$uname_s" = "Darwin" || {
  echo "Metal capture requires macOS" >&2
  exit 1
}
case "$uname_m" in
  arm64|aarch64) ;;
  *)
    echo "Metal production capture requires Apple Silicon (got $uname_m)" >&2
    exit 1
    ;;
esac

ram_bytes="$(sysctl -n hw.memsize)"
cpu_model="$(sysctl -n machdep.cpu.brand_string)"
mkdir -p "$output_root"

cargo run --locked --release --no-default-features \
  --features embedded-metal \
  --bin a3s-power-tensor-batch-bench -- release-fixture \
  --output "${output_root}/metal.json" \
  --device metal:0 \
  --power-commit "$power_commit" \
  --filesystem-class apfs \
  --device-class "Apple Silicon Metal GPU" \
  --cpu-model "$cpu_model" \
  --ram-bytes "$ram_bytes" \
  --tee-policy-sha256 "$policy_hash" \
  --host-fixed-bytes 67108864 \
  --host-scratch-bytes 67108864 \
  --device-fixed-bytes 67108864 \
  --device-scratch-bytes 67108864 \
  --items 8 --width 4096 \
  --warmup-rounds 2 --measured-rounds 9

test -z "$(git status --porcelain)" || {
  echo "capture worktree became dirty" >&2
  exit 1
}

sw_vers >"${output_root}/macos.txt"
uname -a >"${output_root}/uname.txt"
system_profiler SPHardwareDataType SPDisplaysDataType >"${output_root}/apple-hardware.txt"
rustc -Vv >"${output_root}/rustc.txt"
cargo -V >"${output_root}/cargo.txt"
shasum -a 256 "${output_root}/metal.json" Cargo.lock >"${output_root}/metal-inputs.sha256"

echo "Wrote Metal capture under ${output_root}"
echo "Authenticated parent: ${power_commit}"
echo "Authenticate apple-hardware.txt through the release trust root before assembly."
