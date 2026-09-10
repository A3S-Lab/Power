#!/usr/bin/env bash
# Assemble release/v<version>/ from four verified captures on a clean checkout
# of the frozen source parent S. Fails closed if any capture is missing or
# verify-release-capture rejects the binding.
#
# Usage:
#   bash tools/release-capture/assemble-evidence-child.sh \
#     --cpu /path/cpu.json \
#     --cuda /path/cuda.json \
#     --metal /path/metal.json \
#     --confidential-gpu /path/confidential-gpu.json
#
# After success: review, git add the two release files, commit the evidence
# child, push main, then create the annotated tag pointing at that child.

set -euo pipefail

cpu=""
cuda=""
metal=""
confidential=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) cpu="${2:?}"; shift 2 ;;
    --cuda) cuda="${2:?}"; shift 2 ;;
    --metal) metal="${2:?}"; shift 2 ;;
    --confidential-gpu) confidential="${2:?}"; shift 2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

test -n "$cpu" && test -n "$cuda" && test -n "$metal" && test -n "$confidential" || {
  echo "usage: assemble-evidence-child.sh --cpu ... --cuda ... --metal ... --confidential-gpu ..." >&2
  exit 1
}

test -z "$(git status --porcelain)" || {
  echo "assemble requires a clean git worktree at the frozen source parent" >&2
  exit 1
}

power_commit="$(git rev-parse HEAD)"
test "${#power_commit}" -eq 40
power_version="$(cargo metadata --locked --no-deps --format-version 1 \
  | python3 -c 'import json,sys; pkgs=json.load(sys.stdin)["packages"];
print(next(p["version"] for p in pkgs if p["name"]=="a3s-power"))')"

for platform_capture in \
  "cpu:${cpu}" \
  "cuda:${cuda}" \
  "metal:${metal}" \
  "confidential-gpu:${confidential}"
do
  platform="${platform_capture%%:*}"
  capture="${platform_capture#*:}"
  test -f "$capture" || {
    echo "missing $platform capture: $capture" >&2
    exit 1
  }
  features=embedded-inference
  case "$platform" in
    cuda|confidential-gpu) features=embedded-cuda ;;
    metal) features=embedded-metal ;;
  esac
  cargo run --locked --release --no-default-features \
    --features "$features" \
    --bin a3s-power-tensor-batch-bench -- \
    verify-release-capture \
    --capture "$capture" \
    --platform "$platform" \
    --power-version "$power_version" \
    --power-commit "$power_commit"
done

version_dir="release/v${power_version}"
test ! -e "$version_dir" || {
  echo "$version_dir already exists on the source parent; aborting" >&2
  exit 1
}
mkdir -p "$version_dir"

cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  build-release-bundle \
  --cpu-capture "$cpu" \
  --cuda-capture "$cuda" \
  --metal-capture "$metal" \
  --confidential-gpu-capture "$confidential" \
  --power-version "$power_version" \
  --power-commit "$power_commit" \
  --output "${version_dir}/release-evidence.json" \
  --sha256-output "${version_dir}/release-evidence.sha256"

echo "Assembled ${version_dir}/release-evidence.{json,sha256}"
echo "Next: git add those two files only, commit the evidence child, push, tag."
