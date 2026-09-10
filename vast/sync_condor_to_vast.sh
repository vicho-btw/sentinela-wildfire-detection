#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$REPO_ROOT}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/condor/Sentinela-ModelS}"
KEY="${KEY:-$HOME/.ssh/vast_ed25519}"

if [[ -z "${VAST_HOST:-}" || -z "${VAST_PORT:-}" ]]; then
  echo "Set VAST_HOST and VAST_PORT first." >&2
  echo 'Example: export VAST_HOST="1.2.3.4"; export VAST_PORT="40000"' >&2
  exit 2
fi

if [[ ! -f "$KEY" ]]; then
  echo "SSH key not found: $KEY" >&2
  echo "Run: bash ${LOCAL_ROOT:-$(pwd)}/vast/ensure_key.sh" >&2
  exit 2
fi

chmod 600 "$KEY"

RSYNC_SSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $VAST_PORT"

echo "Checking SSH..."
ssh -i "$KEY" \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=10 \
  -p "$VAST_PORT" \
  "root@$VAST_HOST" \
  "echo ok && mkdir -p '$REMOTE_ROOT' /workspace/data/goes_fire"

echo "Syncing $LOCAL_ROOT -> root@$VAST_HOST:$REMOTE_ROOT"
rsync -avP --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'node_modules/' \
  --exclude '.next/' \
  --exclude '.vercel/' \
  --exclude '.wrangler/' \
  --exclude 'tsconfig.tsbuildinfo' \
  --exclude 'data/goes_fire/' \
  --exclude 'local_results/' \
  --exclude 'runs/' \
  --exclude 'models/' \
  -e "$RSYNC_SSH" \
  "$LOCAL_ROOT/" "root@$VAST_HOST:$REMOTE_ROOT/"

echo "Remote size:"
ssh -i "$KEY" -o IdentitiesOnly=yes -p "$VAST_PORT" "root@$VAST_HOST" "du -sh '$REMOTE_ROOT' || true"

echo "Done. Next run on Vast:"
echo "cd $REMOTE_ROOT && source .venv/bin/activate"
