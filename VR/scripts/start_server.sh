#!/usr/bin/env bash
set -euo pipefail

CERT="$(dirname "$0")/../certs/dev-cert.pem"
KEY="$(dirname "$0")/../certs/dev-key.pem"

if [[ -f "$CERT" && -f "$KEY" ]]; then
  echo "Starting runtime server with TLS (wss://)"
  uv run --no-sync python -m virtual_field.server.app \
    --host 0.0.0.0 \
    --port 8765 \
    --ssl-cert "$CERT" \
    --ssl-key "$KEY"
else
  echo "TLS cert/key not found; starting runtime server without TLS (ws://)" >&2
  uv run --no-sync python -m virtual_field.server.app \
    --host 0.0.0.0 \
    --port 8765
fi
