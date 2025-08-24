#!/usr/bin/env python3
"""
Environment Setup
Automatically detects hardware platform and sets up appropriate environment
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

class HardwareDetector:
    """Advanced hardware detection for edge reasoning evaluation."""
    
    @staticmethod
    def detect_platform() -> str:
        """Detect hardware platform"""
        if HardwareDetector._is_tegra():
            return "tegra"
        
        if HardwareDetector._has_nvidia_gpu():
            return "server"
            
        if HardwareDetector._is_container():
            if Path("/proc/device-tree").exists():
                return "tegra"
            elif Path("/usr/bin/nvidia-smi").exists():
                return "server"
        
        return "unknown"
    
    @staticmethod
    def _is_tegra() -> bool:
        """Check if running on NVIDIA Tegra device."""
        tegra_indicators = [
            "/etc/nv_tegra_release",
            "/sys/firmware/devicetree/base/compatible",
            "/sys/firmware/devicetree/base/model"
        ]
        
        for indicator in tegra_indicators:
            try:
                if Path(indicator).exists():
                    content = Path(indicator).read_text().lower()
                    if "tegra" in content or "jetson" in content:
                        return True
            except:
                pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                return "orin" in output or "nvgpu" in output or "tegra" in output
        except:
            pass
        
        return False
    
    @staticmethod
    def _has_nvidia_gpu() -> bool:
        """Check for NVIDIA GPU availability."""
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except:
            pass
        
        try:
            result = subprocess.run(
                ["lspci"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return "nvidia" in result.stdout.lower()
        except:
            return False
    
    @staticmethod
    def _is_container() -> bool:
        """Check if running inside a container."""
        return (
            Path("/.dockerenv").exists() or 
            "docker" in Path("/proc/1/cgroup").read_text() if Path("/proc/1/cgroup").exists() else False
        )
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get device information."""
        info = {
            "platform": HardwareDetector.detect_platform(),
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "in_container": HardwareDetector._is_container(),
        }
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        name, memory = line.strip().split(',')
                        gpus.append({"name": name.strip(), "memory": memory.strip()})
                info["gpus"] = gpus
        except:
            info["gpus"] = []
        
        return info


class EnvironmentSetup:
    """Setup evaluation environment based on detected hardware."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.detector = HardwareDetector()
        
    def setup_common_env(self):
        """Set up common environment variables."""
        print("Setting up common environment variables...")
        env_vars = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", 
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": f"{self.repo_root}:{os.environ.get('PYTHONPATH', '')}"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"  {key}={value}")
    
    def setup_tegra_container(self):
        """Setup or connect to Tegra container."""
        print("Setting up Tegra container environment...")
        container_script = self.repo_root / "eval/tegra/tools/container.sh"
        
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "ancestor=dustynv/vllm:0.8.6-r36.4-cu128-24.04", "--format", "{{.ID}}"],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                print("* Container already running")
                print("Use: cd eval/tegra && ./open.sh 1  # to connect")
                return True
        except:
            pass
        
        if container_script.exists():
            print("Starting new container...")
            subprocess.run([str(container_script)], cwd=container_script.parent)
            return True
        else:
            print("Container script not found!")
            return False
    
    def install_dependencies(self, platform: str):
        """Install platform-specific dependencies."""
        print(f"Installing dependencies for {platform}...")
        
        requirements_file = self.repo_root / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        
        packages = {
            "tegra": ["vllm", "transformers", "datasets", "pyyaml", "tqdm", "psutil"],
            "server": ["vllm", "transformers", "datasets", "pyyaml", "tqdm", "psutil", "matplotlib", "pandas"]
        }
        
        if platform in packages:
            subprocess.run([sys.executable, "-m", "pip", "install"] + packages[platform])
    
    def verify_setup(self):
        """Verify installation and environment."""
        print("Verifying setup...")
        
        try:
            import vllm, transformers, datasets, yaml, tqdm
            print("* Core packages available")
        except ImportError as e:
            print(f"! Missing package: {e}")
            return False
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"* CUDA available: {cuda_available}")
            if cuda_available:
                print(f"  GPU count: {torch.cuda.device_count()}")
        except ImportError:
            print("! PyTorch not available")
        
        required_dirs = ["eval/server", "eval/tegra", "benchmarks"]
        for dir_name in required_dirs:
            dir_path = self.repo_root / dir_name
            if dir_path.exists():
                print(f"* Directory {dir_name}: OK")
            else:
                print(f"! Directory {dir_name}: Not found")
        
        return True
    
    @staticmethod
    def _is_in_venv() -> bool:
        """Check if running inside a virtual environment."""
        return (hasattr(sys, 'real_prefix') or 
                (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


def show_usage():
    print("=" * 60)
    print("Edge Reasoning Evaluation Framework")
    print("=" * 60)
    print("Usage:")
    print("  python setup.py [--platform {auto,server,tegra}] [--info-only]")
    print()
    print("Options:")
    print("  --platform    Force specific platform (default: auto-detect)")
    print("  --info-only   Only show device information, don't setup")
    print()
    print("After setup:")
    print("  make server   - Run server evaluations")
    print("  make jetson   - Run Tegra evaluations") 
    print()
    print("Manual commands:")
    print("  Server:  cd eval/server/[benchmark] && ./run.sh [mode]")
    print("  Tegra:   cd eval/tegra/[benchmark] && ./launch.sh [mode]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Environment Bootstrap")
    parser.add_argument("--platform", choices=["auto", "server", "tegra"], default="auto",
                      help="Target platform (default: auto-detect)")
    parser.add_argument("--info-only", action="store_true",
                      help="Only show device information")
    parser.add_argument("--yes", "-y", action="store_true",
                      help="Skip interactive prompts (for CI/automation)")
    
    args = parser.parse_args()
    
    # __file__ is scripts/bootstrap.py → repo_root should be the parent of 'scripts'
    repo_root = Path(__file__).resolve().parents[1]
    
    setup = EnvironmentSetup(repo_root)
    detector = HardwareDetector()
    
    print("=" * 60)
    print("Environment Setup")
    print("=" * 60)
    
    # Detect platform
    if args.platform == "auto":
        platform = detector.detect_platform()
        print(f"Auto-detected platform: {platform}")
    else:
        platform = args.platform
        print(f"Using specified platform: {platform}")
    
    # Show device information
    device_info = detector.get_device_info()
    print(f"\nDevice Information:")
    for key, value in device_info.items():
        if key == "gpus" and value:
            print(f"  {key}:")
            for i, gpu in enumerate(value):
                print(f"    GPU {i}: {gpu['name']} ({gpu['memory']})")
        else:
            print(f"  {key}: {value}")
    
    if args.info_only:
        return
    
    # Validate platform
    if platform == "unknown":
        print("\n! Unable to detect hardware platform.")
        print("Please specify manually with --platform [server|tegra]")
        sys.exit(1)
    
    print(f"\nSetting up environment for: {platform}")
    
    if device_info["in_container"]:
        print("Running inside container environment")
    else:
        print("Running on host system")
    
    # Setup environment
    try:
        setup.setup_common_env()
        
        if platform == "tegra":
            setup.setup_tegra_container()
        else:
            if not setup._is_in_venv():
                print("")
                print("WARNING: Not in a virtual environment!")
                print("Consider creating one first:")
                print("  make venv")
                print("  source .venv/bin/activate")
                print("  python scripts/bootstrap.py")
                print("")
                if not args.yes:
                    response = input("Continue anyway? (y/N): ").lower()
                    if response != 'y':
                        print("Setup cancelled. Create venv first!")
                        sys.exit(1)
                else:
                    print("Continuing without venv (--yes flag provided)")
                    print("Note: This may cause package conflicts!")
            
            setup.install_dependencies(platform)
            setup.verify_setup()
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"Platform: {platform}")
        print()
        if platform == "server":
            print("Next: make server")
        elif platform == "tegra":
            print("Next: cd eval/tegra && ./open.sh && ./launch.sh base")
        print("=" * 60)
            
    except Exception as e:
        print(f"! Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
