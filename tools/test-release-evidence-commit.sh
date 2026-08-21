#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
verifier="${script_dir}/verify-release-evidence-commit.sh"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/a3s-power-release-layout.XXXXXX")"

cleanup() {
  case "$test_root" in
    "${TMPDIR:-/tmp}"/a3s-power-release-layout.*)
      rm -rf -- "$test_root"
      ;;
    *)
      echo "refusing to remove unexpected test directory: $test_root" >&2
      return 1
      ;;
  esac
}
trap cleanup EXIT

init_source_repo() {
  local repo="$1"
  mkdir -p -- "$repo"
  git -C "$repo" init --quiet
  git -C "$repo" config user.name "A3S Power Test"
  git -C "$repo" config user.email "power-test@example.invalid"
  git -C "$repo" config commit.gpgSign false
  git -C "$repo" config tag.gpgSign false
  git -C "$repo" config core.autocrlf false
  printf '%s\n' 'source' > "$repo/source.txt"
  git -C "$repo" add source.txt
  git -C "$repo" commit --quiet -m "source"
}

write_evidence_pair() {
  local repo="$1"
  mkdir -p -- "$repo/release/v1.0.0"
  printf '%s\n' '{}' > "$repo/release/v1.0.0/release-evidence.json"
  printf '%064d\n' 0 > "$repo/release/v1.0.0/release-evidence.sha256"
}

expect_rejection() {
  local repo="$1"
  local reason="$2"
  if (cd -- "$repo"; bash "$verifier" 1.0.0 HEAD) >/dev/null 2>&1; then
    echo "release evidence layout accepted $reason" >&2
    exit 1
  fi
}

valid_repo="$test_root/valid"
init_source_repo "$valid_repo"
source_commit="$(git -C "$valid_repo" rev-parse HEAD)"
write_evidence_pair "$valid_repo"
git -C "$valid_repo" add release/v1.0.0
git -C "$valid_repo" commit --quiet -m "evidence"
evidence_commit="$(git -C "$valid_repo" rev-parse HEAD)"

actual_source="$(cd -- "$valid_repo"; bash "$verifier" 1.0.0 "$evidence_commit")"
if [ "$actual_source" != "$source_commit" ]; then
  echo "release evidence layout returned the wrong source commit" >&2
  exit 1
fi

extra_repo="$test_root/extra"
init_source_repo "$extra_repo"
write_evidence_pair "$extra_repo"
printf '%s\n' 'not allowed' > "$extra_repo/extra.txt"
git -C "$extra_repo" add release/v1.0.0 extra.txt
git -C "$extra_repo" commit --quiet -m "evidence plus source change"
expect_rejection "$extra_repo" "an extra source change"

preexisting_repo="$test_root/preexisting"
init_source_repo "$preexisting_repo"
mkdir -p -- "$preexisting_repo/release/v1.0.0"
printf '%s\n' 'pre-existing' > "$preexisting_repo/release/v1.0.0/README.md"
git -C "$preexisting_repo" add release/v1.0.0/README.md
git -C "$preexisting_repo" commit --quiet -m "pre-existing evidence directory"
write_evidence_pair "$preexisting_repo"
git -C "$preexisting_repo" add release/v1.0.0
git -C "$preexisting_repo" commit --quiet -m "evidence"
expect_rejection "$preexisting_repo" "a pre-existing evidence directory"

mode_repo="$test_root/mode"
init_source_repo "$mode_repo"
write_evidence_pair "$mode_repo"
git -C "$mode_repo" add release/v1.0.0
git -C "$mode_repo" update-index --chmod=+x release/v1.0.0/release-evidence.json
git -C "$mode_repo" commit --quiet -m "executable evidence"
expect_rejection "$mode_repo" "an executable evidence blob mode"

echo "Release evidence commit layout: PASS"
