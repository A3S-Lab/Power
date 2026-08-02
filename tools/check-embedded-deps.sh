#!/usr/bin/env bash
set -euo pipefail

tree="$(cargo tree --no-default-features --features embedded-inference -e normal --prefix none)"
forbidden="$(printf '%s\n' "$tree" | grep -E '^(axum|axum-core|axum-extra|axum-server|chromiumoxide|http|http-body|http-body-util|hyper|hyper-util|onnxruntime|ort|reqwest|tokio-vsock|tower|tower-http|ureq) v' || true)"

if [[ -n "$forbidden" ]]; then
    echo "embedded-inference contains forbidden Web, external-service, or ONNX dependencies:" >&2
    printf '%s\n' "$forbidden" >&2
    exit 1
fi

features="$(cargo tree --no-default-features --features embedded-inference -e normal,features)"

if printf '%s\n' "$features" \
    | grep -Eq 'a3s-power feature "server"'; then
    echo "embedded-inference unexpectedly enables the server feature" >&2
    exit 1
fi

if printf '%s\n' "$features" | grep -Eq 'tokio feature "(full|net|process|signal)"'; then
    echo "embedded-inference unexpectedly enables Tokio networking or process features" >&2
    exit 1
fi

echo "embedded-inference dependency boundary verified"
