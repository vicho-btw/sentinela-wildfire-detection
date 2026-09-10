#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-$(pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/condor/Sentinela-ModelS}"
KEY="${KEY:-$HOME/.ssh/vast_ed25519}"

if [[ -z "${VAST_HOST:-}" || -z "${VAST_PORT:-}" ]]; then
  echo "Set VAST_HOST and VAST_PORT first." >&2
  exit 2
fi

chmod 600 "$KEY"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p "$VAST_PORT" "root@$VAST_HOST")

"${SSH[@]}" "mkdir -p '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/src/sentinela_models' '$REMOTE_ROOT/src/sentinela_models/datasets' '$REMOTE_ROOT/vast'"

echo "Uploading Sentinela scripts/src/vast only..."
tar -C "$LOCAL_ROOT" -czf - \
  scripts \
  src \
  vast \
  requirements.txt \
  README.md \
  configs \
  data_contracts | "${SSH[@]}" "tar -xzf - -C '$REMOTE_ROOT'"

echo "Uploaded script/source subset to $REMOTE_ROOT"
