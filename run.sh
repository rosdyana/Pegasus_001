#!/usr/bin/env bash
# Bootstrap: find Python, create .venv, install base requirements, hand off to the CLI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
TORCH_INDEX="${MAILPIPE_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

find_python() {
  for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" >/dev/null 2>&1 &&
      "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

if [ ! -x "$VENV/bin/python" ]; then
  PY="$(find_python)" || {
    echo "FAIL  Python >= 3.10 not found. Install it with: sudo apt install python3.12 python3.12-venv" >&2
    exit 1
  }
  echo "Creating virtualenv in $VENV ($("$PY" --version))"
  "$PY" -m venv "$VENV" || {
    echo "FAIL  venv creation failed. Install it with: sudo apt install python3-venv" >&2
    exit 1
  }
  "$VENV/bin/python" -m pip install -q --upgrade pip
fi

PIP=("$VENV/bin/python" -m pip install -q)
STAMP="$VENV/.requirements.sha256"
REQ_HASH="$(cat "$HERE/requirements.txt" | sha256sum 2>/dev/null || shasum -a 256 "$HERE/requirements.txt")"
REQ_HASH="${REQ_HASH%% *}"

if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$REQ_HASH" ]; then
  # torch first and separately, so the CUDA build is the one that ends up installed
  if ! "$VENV/bin/python" -c 'import torch' 2>/dev/null; then
    if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
      echo "Installing torch (CUDA 12.8 build)"
      "${PIP[@]}" torch --index-url "$TORCH_INDEX"
    else
      echo "Installing torch (CPU build: no NVIDIA GPU detected)"
      "${PIP[@]}" torch
    fi
  fi
  echo "Installing requirements"
  "${PIP[@]}" -r "$HERE/requirements.txt"
  echo "$REQ_HASH" >"$STAMP"
fi

# keep the caller's cwd so relative --input/--output paths work
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/python" -m mailpipe "$@"
