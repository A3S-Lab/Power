#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <crate-version> [evidence-commit]" >&2
  exit 2
fi

version="$1"
evidence_ref="${2:-HEAD}"
if [[ ! "$version" =~ ^[0-9][0-9A-Za-z.+-]{0,63}$ ]]; then
  echo "release evidence version must be a bounded path-safe version" >&2
  exit 2
fi

evidence_commit="$(git rev-parse --verify --end-of-options "${evidence_ref}^{commit}")"
read -r -a commit_and_parents <<< "$(git rev-list --parents -n 1 "$evidence_commit")"
if [ "${#commit_and_parents[@]}" -ne 2 ]; then
  echo "production evidence commit must have exactly one source parent" >&2
  exit 1
fi
source_commit="${commit_and_parents[1]}"

bundle_path="release/v${version}/release-evidence.json"
pin_path="release/v${version}/release-evidence.sha256"
evidence_dir="release/v${version}"

if git cat-file -e "${source_commit}:${evidence_dir}" 2>/dev/null; then
  echo "production evidence directory must be absent from the source commit: $evidence_dir" >&2
  exit 1
fi

for path in "$bundle_path" "$pin_path"; do
  entry="$(git ls-tree "$evidence_commit" -- "$path")"
  if [[ ! "$entry" =~ ^100644[[:space:]]blob[[:space:]]([0-9a-f]{40}|[0-9a-f]{64})[[:space:]] ]]; then
    echo "production evidence commit must contain a regular 100644 blob at $path" >&2
    exit 1
  fi
done

expected_changes="$(printf 'A\t%s\nA\t%s\n' "$bundle_path" "$pin_path" | sort)"
actual_changes="$(git diff-tree \
  --no-commit-id \
  --name-status \
  --no-renames \
  -r \
  "$evidence_commit" | sort)"
if [ "$actual_changes" != "$expected_changes" ]; then
  echo "production evidence commit may add only $bundle_path and $pin_path" >&2
  if [ -n "$actual_changes" ]; then
    printf '%s\n' "$actual_changes" >&2
  fi
  exit 1
fi

printf '%s\n' "$source_commit"
