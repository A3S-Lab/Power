#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
ci_workflow="${repo_root}/.github/workflows/ci.yml"
release_workflow="${repo_root}/.github/workflows/release.yml"

fail() {
  echo "release workflow contract failed: $1" >&2
  exit 1
}

cross_revision() {
  local workflow="$1"
  local declarations
  declarations="$(grep -Ec '^  CROSS_GIT_REV: "[0-9a-f]{40}"$' "$workflow")"
  [ "$declarations" -eq 1 ] || fail "$workflow must declare one full CROSS_GIT_REV"
  sed -n 's/^  CROSS_GIT_REV: "\([0-9a-f]\{40\}\)"$/\1/p' "$workflow"
}

ci_cross_revision="$(cross_revision "$ci_workflow")"
release_cross_revision="$(cross_revision "$release_workflow")"
[ "$ci_cross_revision" = "$release_cross_revision" ] ||
  fail "CI and release must pin the same cross revision"

install_prefix='cargo install cross --git https://github.com/cross-rs/cross'
for workflow in "$ci_workflow" "$release_workflow"; do
  [ "$(grep -Fc "$install_prefix" "$workflow")" -eq 1 ] ||
    fail "$workflow must contain one cross installation"
  grep -F -- '--rev "${CROSS_GIT_REV}" --locked' "$workflow" >/dev/null ||
    fail "$workflow must install cross from the pinned revision and lockfile"
  [ "$(grep -Fc 'bash tools/test-release-candidate.sh' "$workflow")" -eq 1 ] ||
    fail "$workflow must run the release candidate preflight contract test"
done

[ "$(grep -Ec '^  contents: read$' "$release_workflow")" -eq 1 ] ||
  fail "release workflow must default to read-only repository contents"
if grep -Eq '^  contents: write$' "$release_workflow"; then
  fail "release workflow must not grant write permission globally"
fi
[ "$(grep -Ec '^      contents: write$' "$release_workflow")" -eq 1 ] ||
  fail "exactly one release job must receive repository write permission"
grep -F '          save-if: false' "$release_workflow" >/dev/null ||
  fail "release package validation cache must be restore-only"
[ "$(grep -Fc 'bash tools/verify-release-candidate.sh' "$release_workflow")" -eq 1 ] ||
  fail "release workflow must invoke one centralized candidate preflight"
grep -F -- '--evidence-ref "${TAG_COMMIT}"' "$release_workflow" >/dev/null ||
  fail "release candidate preflight must verify the signed tag target"
grep -F -- '--main-ref refs/remotes/origin/main' "$release_workflow" >/dev/null ||
  fail "release candidate preflight must require remote-main containment"
if grep -F 'verify-release-bundle' "$release_workflow" >/dev/null; then
  fail "release workflow must not duplicate semantic bundle verification"
fi

release_source="$(cat -- "$release_workflow")"
case "$release_source" in
  *"LLM inference"*) fail "release metadata must not narrow Power to LLM inference" ;;
esac
grep -F 'desc \"Model-neutral verifiable inference for confidential systems\"' \
  "$release_workflow" >/dev/null || fail "release formula must use model-neutral metadata"

echo "Release workflow contract: PASS (${ci_cross_revision})"
