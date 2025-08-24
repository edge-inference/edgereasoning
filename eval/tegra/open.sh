#!/usr/bin/env bash
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

