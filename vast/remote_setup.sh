#!/usr/bin/env bash
set -euo pipefail

cd /workspace/condor/Sentinela-ModelS

python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade || true
if ! python -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python /tmp/get-pip.py
fi
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python scripts/smoke_test_model.py

mkdir -p /workspace/datasets/sentinela /workspace/condor/Sentinela-ModelS/runs

echo "Sentinela remote setup complete."
echo "Repo: /workspace/condor"
echo "Venv: /workspace/condor/Sentinela-ModelS/.venv"
echo "Datasets: /workspace/datasets/sentinela"
