#!/bin/bash

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

# Directory containing the .deb package
HG_CACHE="/mnt/nvme/cache/huggingface"

# Check if the package directory exists and mount it
if [ -d $HG_CACHE ]; then
  echo "Mounting $HG_CACHE to container."
else
  echo "Warning: $HG_CACHE not found."
  exit 1
fi

# Build the docker run command to mount the package directory
DOCKER_CMD="jetson-containers run --cap-add=SYS_ADMIN --privileged=true --gpus=all -d"

# Mount the package directory into the container
DOCKER_CMD+=" -v ${HG_CACHE}:${HG_CACHE}"

# Mount the vllm directory
DOCKER_CMD+=" -v /home/ubuntu/modfi/vllm:/home/vllm"

# Mount the setup script for custom aliases and vim
DOCKER_CMD+=" -v /home/ubuntu/modfi/vllm/docker_setup.sh:/docker_setup.sh"

# Mount tegrastats for system monitoring
if [ -f "/usr/bin/tegrastats" ]; then
  echo "Mounting tegrastats..."
  DOCKER_CMD+=" -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro"
else
  echo "Warning: tegrastats not found at /usr/bin/tegrastats"
fi

# Mount system monitoring directories and files
echo "Mounting system monitoring files..."
DOCKER_CMD+=" -v /sys/class/hwmon:/sys/class/hwmon:ro"
DOCKER_CMD+=" -v /sys/devices/system/cpu:/sys/devices/system/cpu:ro"
DOCKER_CMD+=" -v /proc/stat:/proc/stat:ro"
DOCKER_CMD+=" -v /proc/meminfo:/proc/meminfo:ro"
DOCKER_CMD+=" -v /proc/cpuinfo:/proc/cpuinfo:ro"

# Mount jetson_clocks if available
if [ -f "/usr/bin/jetson_clocks" ]; then
  echo "Mounting jetson_clocks..."
  DOCKER_CMD+=" -v /usr/bin/jetson_clocks:/usr/bin/jetson_clocks:ro"
fi

# Mount power management files
if [ -d "/sys/kernel/debug/bpmp/debug/clk" ]; then
  echo "Mounting power management debug files..."
  DOCKER_CMD+=" -v /sys/kernel/debug:/sys/kernel/debug:ro"
fi

# Mount device tree and hardware info
DOCKER_CMD+=" -v /proc/device-tree:/proc/device-tree:ro"

# Set working directory to the vllm volume
DOCKER_CMD+=" -w /home/vllm"

# Specify the container image and start an interactive bash shell
DOCKER_CMD+=" dustynv/vllm:0.8.6-r36.4-cu128-24.04 bash"

# Print the full command for debugging
echo "Running Docker container with command:"
echo "$DOCKER_CMD"
echo ""

# Run the docker container with all mounts
eval $DOCKER_CMD

# Get container ID for reference (FIXED: using correct image name)
CONTAINER_ID=$(docker ps -q --filter ancestor=dustynv/tensorrt_llm:0.12-r36.4.0 | head -1)
if [ ! -z "$CONTAINER_ID" ]; then
  echo ""
  echo "Container started with ID: $CONTAINER_ID"
  echo "To attach to the container, run: docker exec -it $CONTAINER_ID bash"
  echo ""
  echo "To set up your custom environment (vim + aliases), run inside container:"
  echo "  /docker_setup.sh"
  echo ""
  echo "To install nsys from mounted package, run inside container:"
  echo "  sudo dpkg -i /data/models/tensorrt_llm/research/nsight-systems-cli-DVS-1_tegra_igpu_arm64.deb"
  echo ""
  echo "Available monitoring tools in container:"
  echo "  - tegrastats (system monitoring)"
  echo "  - nsys (profiling)"
  echo "  - jetson_clocks (performance tuning)"
fi
