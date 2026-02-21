#!/bin/bash
# Setup dev-dependency stubs for standalone CI builds.
#
# The Power repo uses a3s-box-sdk as a dev-dependency via local path
# (path = "../box/src/sdk"). In CI, the monorepo parent isn't available,
# so we create a minimal stub crate to satisfy cargo's dependency resolution.
#
# This only affects dev-dependencies — the stub is never compiled into
# release binaries.

set -euo pipefail

SDK_DIR="../box/src/sdk"

if [ -d "$SDK_DIR" ] && [ -f "$SDK_DIR/Cargo.toml" ]; then
  echo "a3s-box-sdk already exists at $SDK_DIR — skipping stub."
  exit 0
fi

echo "Creating a3s-box-sdk stub at $SDK_DIR..."
mkdir -p "$SDK_DIR/src"

cat > "$SDK_DIR/Cargo.toml" << 'EOF'
[package]
name = "a3s-box-sdk"
version = "0.6.0"
edition = "2021"
EOF

printf '' > "$SDK_DIR/src/lib.rs"

echo "Stub created."
