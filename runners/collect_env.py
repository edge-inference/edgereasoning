#!/usr/bin/env python3

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

import json, platform, subprocess, shutil, sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.bootstrap import HardwareDetector

def get_env_info(output_format="json"):
    """Get comprehensive environment information."""
    detector = HardwareDetector()
    
    env = {
        "detected_platform": detector.detect_platform(),
        "python_version": platform.python_version(),
        "system_platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": subprocess.getoutput("git rev-parse --short HEAD || echo NA"),
        "git_branch": subprocess.getoutput("git branch --show-current || echo NA"),
        "nvidia_smi": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo NA"),
        "device_info": detector.get_device_info()
    }
    
    if output_format == "platform":
        return env["detected_platform"]
    elif output_format == "json":
        return json.dumps(env, indent=2)
    else:
        return env

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Collect environment information')
    parser.add_argument('--format', choices=['json', 'platform'], default='json', 
                       help='Output format')
    parser.add_argument('--output', help='Output file path')
    args = parser.parse_args()
    
    if args.format == "platform":
        print(get_env_info("platform"))
    else:
        env_json = get_env_info("json")
        if args.output:
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
            with open(args.output, 'w') as f:
                f.write(env_json)
            print(f"wrote {args.output}")
        else:
            print(env_json)

if __name__ == "__main__":
    main()
