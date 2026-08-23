#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
candidate_source="${script_dir}/verify-release-candidate.sh"
layout_source="${script_dir}/verify-release-evidence-commit.sh"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/a3s-power-release-candidate.XXXXXX")"
fake_cargo="${test_root}/fake-cargo"

cleanup() {
  case "$test_root" in
    "${TMPDIR:-/tmp}"/a3s-power-release-candidate.*)
      rm -rf -- "$test_root"
      ;;
    *)
      echo "refusing to remove unexpected test directory: $test_root" >&2
      return 1
      ;;
  esac
}
trap cleanup EXIT

fail() {
  echo "release candidate test failed: $1" >&2
  exit 1
}

cat > "$fake_cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  pkgid)
    printf 'path+file:///fixture#a3s-power@%s\n' \
      "${A3S_POWER_TEST_CRATE_VERSION:-1.0.0}"
    ;;
  run)
    printf '%s\n' "$*" > "${A3S_POWER_TEST_CARGO_LOG:?missing cargo log path}"
    if [ "${A3S_POWER_TEST_FAIL_SEMANTIC:-0}" = "1" ]; then
      echo "synthetic semantic rejection" >&2
      exit 19
    fi
    printf '{"schema":"test.release-verification.v1","verified":true}\n'
    ;;
  *)
    echo "unexpected fake cargo command: $*" >&2
    exit 20
    ;;
esac
EOF
chmod +x "$fake_cargo"

init_source_repo() {
  local repo="$1"
  local version="$2"
  local unreleased_text="${3:-}"
  local changelog_version="${4:-$version}"

  mkdir -p -- "$repo/tools"
  git -C "$repo" init --quiet
  git -C "$repo" branch -M main
  git -C "$repo" config user.name "A3S Power Test"
  git -C "$repo" config user.email "power-test@example.invalid"
  git -C "$repo" config commit.gpgSign false
  git -C "$repo" config tag.gpgSign false
  git -C "$repo" config core.autocrlf false

  cat > "$repo/Cargo.toml" <<EOF
[package]
name = "a3s-power"
version = "${version}"
edition = "2021"
EOF
  cat > "$repo/CHANGELOG.md" <<EOF
# Changelog

## [Unreleased]

${unreleased_text}
## [${changelog_version}] - 2026-08-24

### Added

- Frozen release source.
EOF
  cp -- "$candidate_source" "$repo/tools/verify-release-candidate.sh"
  cp -- "$layout_source" "$repo/tools/verify-release-evidence-commit.sh"
  git -C "$repo" add Cargo.toml CHANGELOG.md tools
  git -C "$repo" commit --quiet -m "source"
}

add_evidence_commit() {
  local repo="$1"
  local version="$2"
  mkdir -p -- "$repo/release/v${version}"
  printf '%s\n' '{}' > "$repo/release/v${version}/release-evidence.json"
  printf '%064d\n' 0 > "$repo/release/v${version}/release-evidence.sha256"
  git -C "$repo" add "release/v${version}"
  git -C "$repo" commit --quiet -m "evidence"
}

run_candidate() {
  local repo="$1"
  local version="$2"
  local main_ref="$3"
  local log_path="$test_root/$(basename -- "$repo").cargo.log"

  (
    cd -- "$repo"
    CARGO="$fake_cargo" \
      A3S_POWER_TEST_CRATE_VERSION="$version" \
      A3S_POWER_TEST_CARGO_LOG="$log_path" \
      bash tools/verify-release-candidate.sh --main-ref "$main_ref"
  )
}

expect_rejection() {
  local repo="$1"
  local version="$2"
  local main_ref="$3"
  local reason="$4"
  if run_candidate "$repo" "$version" "$main_ref" >/dev/null 2>&1; then
    fail "candidate verifier accepted $reason"
  fi
}

valid_repo="$test_root/valid"
init_source_repo "$valid_repo" 1.0.0
source_commit="$(git -C "$valid_repo" rev-parse HEAD)"
add_evidence_commit "$valid_repo" 1.0.0
evidence_commit="$(git -C "$valid_repo" rev-parse HEAD)"
actual_source="$(run_candidate "$valid_repo" 1.0.0 main 2>"$test_root/valid.stderr")"
[ "$actual_source" = "$source_commit" ] || fail "valid candidate returned the wrong source"
grep -F -- 'verify-release-bundle' "$test_root/valid.cargo.log" >/dev/null ||
  fail "valid candidate did not invoke semantic bundle verification"
grep -F -- '--bundle release/v1.0.0/release-evidence.json' \
  "$test_root/valid.cargo.log" >/dev/null || fail "bundle path was not canonical"
grep -F -- '--expected-sha256-file release/v1.0.0/release-evidence.sha256' \
  "$test_root/valid.cargo.log" >/dev/null || fail "pin path was not canonical"
grep -F -- "--power-commit $source_commit" "$test_root/valid.cargo.log" >/dev/null ||
  fail "semantic verification did not bind the source parent"
grep -F -- "Production release candidate v1.0.0: PASS" \
  "$test_root/valid.stderr" >/dev/null || fail "valid candidate emitted no PASS receipt"

printf '%s\n' dirty > "$valid_repo/untracked.txt"
expect_rejection "$valid_repo" 1.0.0 main "an unclean worktree"
rm -- "$valid_repo/untracked.txt"

if (
  cd -- "$valid_repo"
  CARGO="$fake_cargo" \
    A3S_POWER_TEST_CRATE_VERSION=1.0.0 \
    A3S_POWER_TEST_CARGO_LOG="$test_root/not-head.cargo.log" \
    bash tools/verify-release-candidate.sh \
      --evidence-ref "$source_commit" \
      --main-ref main
) >/dev/null 2>&1; then
  fail "candidate verifier accepted a non-HEAD evidence ref"
fi

detached_repo="$test_root/not-on-main"
init_source_repo "$detached_repo" 1.0.0
git -C "$detached_repo" switch --quiet -c release-candidate
add_evidence_commit "$detached_repo" 1.0.0
expect_rejection "$detached_repo" 1.0.0 main "an evidence child outside main"

unreleased_repo="$test_root/unreleased"
init_source_repo "$unreleased_repo" 1.0.0 $'- Pending release work.\n\n'
add_evidence_commit "$unreleased_repo" 1.0.0
expect_rejection "$unreleased_repo" 1.0.0 main "a non-empty Unreleased changelog"

changelog_repo="$test_root/mismatched-changelog"
init_source_repo "$changelog_repo" 1.0.0 '' 0.9.0
add_evidence_commit "$changelog_repo" 1.0.0
expect_rejection "$changelog_repo" 1.0.0 main "a missing versioned changelog entry"

pre_v1_repo="$test_root/pre-v1"
init_source_repo "$pre_v1_repo" 0.9.0
add_evidence_commit "$pre_v1_repo" 0.9.0
expect_rejection "$pre_v1_repo" 0.9.0 main "a pre-v1 package"

semantic_repo="$test_root/semantic-failure"
init_source_repo "$semantic_repo" 1.0.0
add_evidence_commit "$semantic_repo" 1.0.0
if (
  cd -- "$semantic_repo"
  CARGO="$fake_cargo" \
    A3S_POWER_TEST_CRATE_VERSION=1.0.0 \
    A3S_POWER_TEST_CARGO_LOG="$test_root/semantic-failure.cargo.log" \
    A3S_POWER_TEST_FAIL_SEMANTIC=1 \
    bash tools/verify-release-candidate.sh --main-ref main
) >/dev/null 2>&1; then
  fail "candidate verifier ignored semantic bundle rejection"
fi

missing_ref_repo="$test_root/missing-main-ref"
init_source_repo "$missing_ref_repo" 1.0.0
add_evidence_commit "$missing_ref_repo" 1.0.0
if (
  cd -- "$missing_ref_repo"
  CARGO="$fake_cargo" \
    A3S_POWER_TEST_CRATE_VERSION=1.0.0 \
    A3S_POWER_TEST_CARGO_LOG="$test_root/missing-main-ref.cargo.log" \
    bash tools/verify-release-candidate.sh
) >/dev/null 2>&1; then
  fail "candidate verifier accepted a missing main containment ref"
fi

[ "$evidence_commit" = "$(git -C "$valid_repo" rev-parse HEAD)" ] ||
  fail "test fixture unexpectedly changed the valid evidence commit"

echo "Release candidate preflight: PASS"
