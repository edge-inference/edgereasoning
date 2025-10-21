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

set -e

IMAGE="dustynv/vllm:0.8.6-r36.4-cu128-24.04"

CID=$(docker ps --filter "ancestor=${IMAGE}" --format "{{.ID}}" | head -n1)

if [ -z "$CID" ]; then
  echo "ERROR: No running container found for image '${IMAGE}'"
  exit 1
fi

echo "* Connecting to container $CID (image: $IMAGE)..."

if [ "$1" != "1" ]; then
  echo "* Installing packages..."

  APT_PACKAGES=(
    screen
    vim
    # add more apt packages here
  )

  PIP_PACKAGES=(
    datasets
    nvtx
    openpyxl
    matplotlib
    seaborn
    numpy
    pandas
  )

  docker exec -it "$CID" bash -c "
    apt update && \
    apt install -y ${APT_PACKAGES[*]} && \
    pip install --index-url https://pypi.org/simple ${PIP_PACKAGES[*]}
  "
else
  echo "* Skipping installation (argument '1' provided)"
fi

exec docker exec -it "$CID" bash -c "cd /workspace/edgereasoning && exec bash"

