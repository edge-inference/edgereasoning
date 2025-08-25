#!/usr/bin/env python3
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
