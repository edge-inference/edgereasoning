#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/token2metrics"
REPO="https://github.com/bkubwimana/token2metrics.git"
PIN="${PIN:-main}"
RUN_T2M_SETUP="${RUN_T2M_SETUP:-1}"

mkdir -p "$(dirname "$DEST")"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch --all --tags --prune
else
  git clone "$REPO" "$DEST"
fi
git -C "$DEST" checkout --quiet "$PIN"
echo "[ok] token2metrics @ $(git -C "$DEST" rev-parse --short HEAD)"

if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
  python -m pip install -e "$DEST"
fi

if [ "$RUN_T2M_SETUP" = "1" ]; then
  echo "[token2metrics] Running setup.py to install requirements..."
  (cd "$DEST" && python setup.py) || {
    echo "[warn] token2metrics/setup.py failed";
    exit 1;
  }
fi
