#!/usr/bin/env bash
set -euo pipefail

CERT="$(dirname "$0")/../certs/dev-cert.pem"
KEY="$(dirname "$0")/../certs/dev-key.pem"

if [[ -f "$CERT" && -f "$KEY" ]]; then
  uv run --no-sync python -m virtual_field.server.app \
    --host 0.0.0.0 \
    --port 8765 \
    --ssl-cert "$CERT" \
    --ssl-key "$KEY"
else
  echo "ERROR: Certificate or key not found. Both must exist." >&2
  exit 1
fi
