#!/usr/bin/env bash
set -euo pipefail

echo "HOST=${PUBLIC_IPADDR:-} PORT=${VAST_TCP_PORT_22:-}"
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"

