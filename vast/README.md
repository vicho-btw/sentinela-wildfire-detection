# Vast.ai Sync

Use these scripts to upload the Condor repo and prepare a Vast.ai instance for Sentinela-ModelS work.

## 1. In Vast Web Terminal

```bash
curl -fsSL https://raw.githubusercontent.com/placeholder/none/main/none 2>/dev/null || true
echo "HOST=$PUBLIC_IPADDR PORT=$VAST_TCP_PORT_22"
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

Copy the printed host and port.

## 2. Locally: Create/Upload Key

```bash
cd /Volumes/PnotP/condor/Sentinela-ModelS
bash vast/ensure_key.sh
cat ~/.ssh/vast_ed25519.pub
```

Paste the public key into the Vast instance `~/.ssh/authorized_keys`:

```bash
cat >> ~/.ssh/authorized_keys
# paste key, then Ctrl-D
chmod 600 ~/.ssh/authorized_keys
```

## 3. Locally: Sync Repo

```bash
export VAST_HOST="<host from Vast>"
export VAST_PORT="<port from Vast>"
export KEY="$HOME/.ssh/vast_ed25519"

bash vast/sync_condor_to_vast.sh
```

If remote `rsync` is blocked or not executable, use the tar-stream fallback:

```bash
bash vast/sync_condor_to_vast_tar.sh
```

## 4. On Vast: Prepare Environment

```bash
bash /workspace/condor/Sentinela-ModelS/vast/remote_setup.sh
```
