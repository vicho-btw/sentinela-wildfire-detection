#!/usr/bin/env bash
set -euo pipefail

KEY="${KEY:-$HOME/.ssh/vast_ed25519}"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ ! -f "$KEY" ]]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "vast-ai-sentinela"
fi

chmod 600 "$KEY"

echo "Private key: $KEY"
echo "Public key:"
cat "${KEY}.pub"

