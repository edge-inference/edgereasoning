#!/bin/bash

# Mounts the entire edgereasoning repository for evaluations on tegra

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
echo "Repository root: $REPO_ROOT"

# Container image
CONTAINER_IMAGE="dustynv/vllm:0.8.6-r36.4-cu128-24.04"

DOCKER_CMD="jetson-containers run --cap-add=SYS_ADMIN --privileged=true --gpus=all -d"

echo "Mounting entire edgereasoning repository..."
DOCKER_CMD+=" -v $REPO_ROOT:/workspace/edgereasoning"

if [ -d "/home/ubuntu/modfi/vllm" ]; then
  echo "Mounting existing VLLM directory..."
  DOCKER_CMD+=" -v /home/ubuntu/modfi/vllm:/home/vllm"
fi

if [ -f "/home/ubuntu/modfi/vllm/docker_setup.sh" ]; then
  DOCKER_CMD+=" -v /home/ubuntu/modfi/vllm/docker_setup.sh:/docker_setup.sh"
fi

if [ -f "/usr/bin/tegrastats" ]; then
  echo "Mounting tegrastats..."
  DOCKER_CMD+=" -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro"
else
  echo "Warning: tegrastats not found at /usr/bin/tegrastats"
fi

echo "Mounting system monitoring files..."
DOCKER_CMD+=" -v /sys/class/hwmon:/sys/class/hwmon:ro"
DOCKER_CMD+=" -v /sys/devices/system/cpu:/sys/devices/system/cpu:ro"
DOCKER_CMD+=" -v /proc/stat:/proc/stat:ro"
DOCKER_CMD+=" -v /proc/meminfo:/proc/meminfo:ro"
DOCKER_CMD+=" -v /proc/cpuinfo:/proc/cpuinfo:ro"

if [ -f "/usr/bin/jetson_clocks" ]; then
  echo "Mounting jetson_clocks..."
  DOCKER_CMD+=" -v /usr/bin/jetson_clocks:/usr/bin/jetson_clocks:ro"
fi

if [ -d "/sys/kernel/debug/bpmp/debug/clk" ]; then
  echo "Mounting power management debug files..."
  DOCKER_CMD+=" -v /sys/kernel/debug:/sys/kernel/debug:ro"
fi

DOCKER_CMD+=" -v /proc/device-tree:/proc/device-tree:ro"

if [ -d "/opt/nvidia/nsight-systems" ]; then
  echo "Mounting full Nsight-Systems installation..."
  DOCKER_CMD+=" -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro"
else
  echo "Warning: /opt/nvidia/nsight-systems not found"
fi

if [ -f "/usr/local/cuda/bin/ncu" ]; then
  echo "Mounting ncu (Nsight Compute)..."
  DOCKER_CMD+=" -v /usr/local/cuda/bin/ncu:/usr/local/cuda/bin/ncu:ro"
else
  echo "Warning: ncu not found at /usr/local/cuda/bin/ncu"
fi

DOCKER_CMD+=" -w /workspace/edgereasoning"

DOCKER_CMD+=" $CONTAINER_IMAGE bash"

echo "Starting container..."
echo "Repository: $REPO_ROOT"
echo "Running Docker container: $CONTAINER_IMAGE"

eval $DOCKER_CMD

CONTAINER_ID=$(docker ps -q --filter ancestor=$CONTAINER_IMAGE | head -1)

if [ -n "$CONTAINER_ID" ] && [ -d "/opt/nvidia/nsight-systems" ]; then
  NSYS_PATH=$(find /opt/nvidia/nsight-systems -name nsys -path "*/target-linux-tegra-armv8/*" | head -1)
  if [ -n "$NSYS_PATH" ]; then
    echo "Creating nsys symlink inside container..."
    docker exec "$CONTAINER_ID" ln -sf "$NSYS_PATH" /usr/local/bin/nsys
  fi
fi

if [ ! -z "$CONTAINER_ID" ]; then
  echo ""
  echo "Container ready: $CONTAINER_ID"
  echo "Repository mounted at: /workspace/edgereasoning"
  echo ""
  echo "To enter: docker exec -it $CONTAINER_ID bash"
  echo "To setup: python setup.py --platform tegra"
  echo "To run evals: cd eval/tegra/mmlu && ./launch.sh [base|budget|scaling]"
fi
