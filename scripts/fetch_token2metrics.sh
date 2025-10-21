#!/usr/bin/env bash

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/token2metrics"
REPO="https://github.com/edge-inference/token2metrics.git"
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
  echo "[token2metrics] Installing package..."
  if ! python -m pip install -e "$DEST" 2>/dev/null; then
    echo "[warn] Editable install failed, trying regular install..."
    python -m pip install "$DEST"
  fi
fi

if [ "$RUN_T2M_SETUP" = "1" ]; then
  echo "[token2metrics] Installing requirements..."
  if [ -f "$DEST/requirements.txt" ]; then
    python -m pip install -r "$DEST/requirements.txt" || {
      echo "[warn] token2metrics requirements installation failed";
      exit 1;
    }
  else
    echo "[info] No requirements.txt found, skipping requirements installation";
  fi
fi
