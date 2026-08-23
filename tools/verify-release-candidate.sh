#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

usage() {
  cat >&2 <<EOF
usage: $0 [--evidence-ref <commit>] --main-ref <commit-containing-candidate>

Verifies one clean, checked-out non-0.x production release candidate before a
signed tag is created. The command prints the frozen source commit on stdout.
EOF
}

fail() {
  echo "production release candidate rejected: $1" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [ -z "$value" ] || [[ "$value" == --* ]]; then
    echo "release candidate option $option requires a value" >&2
    exit 2
  fi
}

evidence_ref=HEAD
main_ref=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --evidence-ref)
      require_value "$1" "${2:-}"
      evidence_ref="$2"
      shift 2
      ;;
    --main-ref)
      require_value "$1" "${2:-}"
      main_ref="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown release candidate option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$main_ref" ]; then
  echo "release candidate verification requires --main-ref" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
layout_verifier="${script_dir}/verify-release-evidence-commit.sh"
cargo_bin="${CARGO:-cargo}"
cd -- "$repo_root"

evidence_commit="$(git rev-parse --verify --end-of-options "${evidence_ref}^{commit}")" ||
  fail "the evidence ref does not identify a commit"
head_commit="$(git rev-parse --verify HEAD)"
if [ "$evidence_commit" != "$head_commit" ]; then
  fail "the evidence ref must be the currently checked-out HEAD"
fi

if [ -n "$(git status --porcelain=v1 --untracked-files=all)" ]; then
  fail "the candidate checkout contains tracked or untracked changes"
fi

package_id="$("$cargo_bin" pkgid --locked -p a3s-power)" ||
  fail "Cargo could not resolve the locked a3s-power package identity"
package_fragment="${package_id##*#}"
case "$package_fragment" in
  *@*) power_version="${package_fragment##*@}" ;;
  *:*) power_version="${package_fragment##*:}" ;;
  [0-9]*) power_version="$package_fragment" ;;
  *) fail "Cargo returned an unrecognized a3s-power package identity" ;;
esac
if [[ ! "$power_version" =~ ^[0-9][0-9A-Za-z.+-]{0,63}$ ]]; then
  fail "the Cargo package version is not a bounded path-safe version"
fi
major_version="${power_version%%.*}"
if [[ ! "$major_version" =~ ^[0-9]+$ ]] || [ "$major_version" -eq 0 ]; then
  fail "the production candidate gate applies only to non-0.x versions"
fi

source_commit="$(bash "$layout_verifier" "$power_version" "$evidence_commit")" ||
  fail "the source-parent/evidence-child layout is invalid"

main_commit="$(git rev-parse --verify --end-of-options "${main_ref}^{commit}" 2>/dev/null)" ||
  fail "the requested main containment ref does not identify a commit"
if ! git merge-base --is-ancestor "$evidence_commit" "$main_commit"; then
  fail "the evidence commit is not reachable from the requested main containment ref"
fi

changelog="$(git show "${source_commit}:CHANGELOG.md")" ||
  fail "the frozen source commit has no readable CHANGELOG.md"
if ! printf '%s\n' "$changelog" | awk -v version="$power_version" '
  BEGIN { prefix = "## [" version "] - " }
  index($0, prefix) == 1 {
    date = substr($0, length(prefix) + 1)
    if (date ~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$/) {
      found = 1
    }
  }
  END { exit found ? 0 : 1 }
'; then
  fail "CHANGELOG.md has no dated entry for v${power_version}"
fi

unreleased_payload="$(printf '%s\n' "$changelog" | awk '
  $0 == "## [Unreleased]" {
    seen = 1
    active = 1
    next
  }
  active && /^## / { active = 0 }
  active && $0 !~ /^[[:space:]]*$/ { print }
  END { if (!seen) exit 2 }
')" || fail "CHANGELOG.md has no Unreleased section"
if [ -n "$unreleased_payload" ]; then
  fail "CHANGELOG.md still contains unreleased entries"
fi

bundle_path="release/v${power_version}/release-evidence.json"
pin_path="release/v${power_version}/release-evidence.sha256"
verification_receipt="$(
  "$cargo_bin" run --locked --release --no-default-features \
    --features embedded-inference \
    --bin a3s-power-tensor-batch-bench -- \
    verify-release-bundle \
    --bundle "$bundle_path" \
    --expected-sha256-file "$pin_path" \
    --power-version "$power_version" \
    --power-commit "$source_commit"
)" || fail "strict four-platform bundle verification failed"

printf '%s\n' "$verification_receipt" >&2
printf 'Production release candidate v%s: PASS (source=%s evidence=%s reachable_from=%s)\n' \
  "$power_version" "$source_commit" "$evidence_commit" "$main_ref" >&2
printf '%s\n' "$source_commit"
