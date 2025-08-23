#!/usr/bin/env python3
import re
import csv
import os
from typing import List, Dict
from datetime import datetime

class TelemetryProcessor:
    def __init__(self, log_file: str, output_dir: str, config_suffix: str = ""):
        self.log_file = log_file
        self.output_dir = output_dir
        self.config_suffix = config_suffix
        self.energy_csv = os.path.join(output_dir, f"energy_{config_suffix}.csv")
        
    def parse_tegrastats_line(self, line: str) -> Dict:
        """Parse a single tegrastats log line with comprehensive metric extraction"""
        data = {}
        
        if not line or not line.strip():
            return data
            
        try:
            original_line = line.strip()
            
            # Extract timestamp if present
            timestamp_match = re.match(r'^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                data['timestamp'] = timestamp_match.group(1)
                line = line[len(timestamp_match.group(1)):].strip()
            
            # RAM usage - handle variations in format
            ram_match = re.search(r'RAM (\d+)/(\d+)MB(?:\s+\(lfb (\d+)x(\d+)MB\))?', line)
            if ram_match:
                data['ram_used_mb'] = int(ram_match.group(1))
                data['ram_total_mb'] = int(ram_match.group(2))
                if ram_match.group(3) and ram_match.group(4):
                    data['ram_lfb_blocks'] = int(ram_match.group(3))
                    data['ram_lfb_size_mb'] = int(ram_match.group(4))
            
            # SWAP usage
            swap_match = re.search(r'SWAP (\d+)/(\d+)MB(?:\s+\(cached (\d+)MB\))?', line)
            if swap_match:
                data['swap_used_mb'] = int(swap_match.group(1))
                data['swap_total_mb'] = int(swap_match.group(2))
                if swap_match.group(3):
                    data['swap_cached_mb'] = int(swap_match.group(3))
            
            # IRAM usage
            iram_match = re.search(r'IRAM (\d+)/(\d+)kB(?:\s*\(lfb (\d+)kB\))?', line)
            if iram_match:
                data['iram_used_kb'] = int(iram_match.group(1))
                data['iram_total_kb'] = int(iram_match.group(2))
                if iram_match.group(3):
                    data['iram_lfb_kb'] = int(iram_match.group(3))
            
            # CPU cores - handle variable frequencies per core
            cpu_match = re.search(r'CPU \[([^\]]+)\]', line)
            if cpu_match:
                cpu_data = cpu_match.group(1)
                cores = cpu_data.split(',')
                for i, core_info in enumerate(cores):
                    core_info = core_info.strip()
                    if '%@' in core_info:
                        parts = core_info.split('%@')
                        if len(parts) == 2:
                            usage = int(parts[0])
                            freq = int(parts[1])
                            data[f'cpu{i}_usage_pct'] = usage
                            data[f'cpu{i}_freq_mhz'] = freq
                    elif core_info.endswith('%'):
                        usage = int(core_info[:-1])
                        data[f'cpu{i}_usage_pct'] = usage
            
            # EMC (External Memory Controller)
            emc_match = re.search(r'EMC (\d+)%@(\d+)', line)
            if emc_match:
                data['emc_usage_pct'] = int(emc_match.group(1))
                data['emc_freq_mhz'] = int(emc_match.group(2))
            
            # GPU (GR3D_FREQ) - handle both single and dual GPC formats
            gpu_dual_match = re.search(r'GR3D_FREQ (\d+)%@\[(\d+),(\d+)\]', line)
            gpu_single_match = re.search(r'GR3D_FREQ (\d+)%@?(\d+)?', line)
            
            if gpu_dual_match:
                data['gpu_usage_pct'] = int(gpu_dual_match.group(1))
                data['gpu_gpc0_freq_mhz'] = int(gpu_dual_match.group(2))
                data['gpu_gpc1_freq_mhz'] = int(gpu_dual_match.group(3))
            elif gpu_single_match:
                data['gpu_usage_pct'] = int(gpu_single_match.group(1))
                if gpu_single_match.group(2):
                    data['gpu_freq_mhz'] = int(gpu_single_match.group(2))
            
            # VIC (Video Image Compositor)
            vic_match = re.search(r'VIC_FREQ (\d+)%@(\d+)', line)
            if vic_match:
                data['vic_usage_pct'] = int(vic_match.group(1))
                data['vic_freq_mhz'] = int(vic_match.group(2))
            
            # APE (Audio Processing Engine)
            ape_match = re.search(r'APE (\d+)', line)
            if ape_match:
                data['ape_freq_mhz'] = int(ape_match.group(1))
            
            # MTS (foreground/background tasks)
            mts_match = re.search(r'MTS fg (\d+)% bg (\d+)%', line)
            if mts_match:
                data['mts_fg_pct'] = int(mts_match.group(1))
                data['mts_bg_pct'] = int(mts_match.group(2))
            
            # NVENC (Video Encoder)
            nvenc_match = re.search(r'NVENC (\d+)', line)
            if nvenc_match:
                data['nvenc_freq_mhz'] = int(nvenc_match.group(1))
            
            # NVDEC (Video Decoder)
            nvdec_match = re.search(r'NVDEC (\d+)', line)
            if nvdec_match:
                data['nvdec_freq_mhz'] = int(nvdec_match.group(1))
            
            # NVDLA (Deep Learning Accelerator)
            for nvdla_match in re.finditer(r'NVDLA(\d+) (\d+)%@(\d+)|NVDLA(\d+) (\d+)', line):
                if nvdla_match.group(1):  # Format: NVDLA0 X%@Y
                    nvdla_id = nvdla_match.group(1)
                    usage = nvdla_match.group(2)
                    freq = nvdla_match.group(3)
                    data[f'nvdla{nvdla_id}_usage_pct'] = int(usage)
                    data[f'nvdla{nvdla_id}_freq_mhz'] = int(freq)
                elif nvdla_match.group(4):  # Format: NVDLA0 Y
                    nvdla_id = nvdla_match.group(4)
                    freq = nvdla_match.group(5)
                    data[f'nvdla{nvdla_id}_freq_mhz'] = int(freq)
            
            # GR3D_PCI (DGPU)
            gr3d_pci_match = re.search(r'GR3D_PCI (\d+)%@(\d+)|GR3D_PCI (\d+)%|GR3D_PCI (\d+)', line)
            if gr3d_pci_match:
                if gr3d_pci_match.group(1):
                    data['dgpu_usage_pct'] = int(gr3d_pci_match.group(1))
                    data['dgpu_freq_mhz'] = int(gr3d_pci_match.group(2))
                elif gr3d_pci_match.group(3):
                    data['dgpu_usage_pct'] = int(gr3d_pci_match.group(3))
                elif gr3d_pci_match.group(4):
                    data['dgpu_freq_mhz'] = int(gr3d_pci_match.group(4))
            
            # Temperature sensors
            for temp_match in re.finditer(r'([a-zA-Z0-9_]+)@([0-9.]+)C', line):
                sensor_name = temp_match.group(1).lower()
                temp_c = float(temp_match.group(2))
                data[f'temp_{sensor_name}_c'] = temp_c
            
            # Power rails - comprehensive pattern matching
            for power_match in re.finditer(r'([A-Z0-9_]+)\s+(\d+)mW/(\d+)mW', line):
                rail_name = power_match.group(1).lower()
                current_mw = int(power_match.group(2))
                avg_mw = int(power_match.group(3))
                data[f'{rail_name}_current_mw'] = current_mw
                data[f'{rail_name}_avg_mw'] = avg_mw
            
            # Legacy power format (POM_5V_IN)
            power_legacy_match = re.search(r'POM_5V_IN (\d+)/(\d+)', line)
            if power_legacy_match:
                data['pom_5v_in_current_mw'] = int(power_legacy_match.group(1))
                data['pom_5v_in_avg_mw'] = int(power_legacy_match.group(2))
            
            return data
            
        except ValueError as e:
            print(f"ValueError parsing tegrastats line: {e}")
            print(f"Line: {original_line[:100]}...")
            return {}
        except Exception as e:
            print(f"Unexpected error parsing tegrastats line: {e}")
            print(f"Line: {original_line[:100]}...")
            return {}
    
    def process_log(self) -> bool:
        """Process the tegrastats log file and generate energy CSV"""
        if not os.path.exists(self.log_file):
            print(f"Log file not found: {self.log_file}")
            return False
            
        try:
            parsed_data = []
            
            with open(self.log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = self.parse_tegrastats_line(line)
                    if data:
                        data['line_number'] = line_num
                        parsed_data.append(data)
            
            if not parsed_data:
                print("No valid data found in log file")
                return False
            
            # Get all possible fieldnames from all rows
            all_fieldnames = set()
            for data in parsed_data:
                all_fieldnames.update(data.keys())
            
            # Sort fieldnames for consistent output
            fieldnames = sorted(all_fieldnames)
                
            # Write to CSV
            with open(self.energy_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for data in parsed_data:
                    writer.writerow(data)
            
            print(f"Energy data saved to: {self.energy_csv}")
            print(f"Parsed {len(parsed_data)} lines with {len(fieldnames)} different metrics")
            
            # Calculate and print summary
            self._print_summary(parsed_data)
            return True
            
        except Exception as e:
            print(f"Error processing log file: {e}")
            return False
    
    def _print_summary(self, data: List[Dict]):
        """Print summary statistics"""
        if not data:
            return
        
        print(f"\nTelemetry Summary:")
        print(f"Total samples: {len(data)}")
        
        # Look for power data in various formats
        power_keys = [k for k in data[0].keys() if 'current_mw' in k or 'avg_mw' in k]
        gpu_util_keys = [k for k in data[0].keys() if 'gpu_usage_pct' in k]
        cpu_util_keys = [k for k in data[0].keys() if 'cpu' in k and 'usage_pct' in k]
        temp_keys = [k for k in data[0].keys() if k.startswith('temp_') and k.endswith('_c')]
        
        # Power summary with energy calculation
        if power_keys:
            print(f"\nPower Metrics ({len(power_keys)} rails):")
            duration_s = len(data)  # tegrastats --interval 1000 outputs every 1 second
            
            for power_key in sorted(power_keys):
                powers = [d.get(power_key, 0) for d in data if d.get(power_key, 0) > 0]
                if powers:
                    avg_power = sum(powers) / len(powers)
                    max_power = max(powers)
                    min_power = min(powers)
                    total_energy_j = (avg_power / 1000) * duration_s  # Convert mW to W, then to Joules
                    print(f"  {power_key}: avg={avg_power:.1f}mW, min={min_power}mW, max={max_power}mW, energy={total_energy_j:.2f}J")
            
            # Calculate total system energy if we have main power rail
            main_power_keys = [k for k in power_keys if 'sys5v' in k.lower() or 'pom_5v_in' in k.lower()]
            if main_power_keys:
                main_key = main_power_keys[0]
                main_powers = [d.get(main_key, 0) for d in data if d.get(main_key, 0) > 0]
                if main_powers:
                    avg_system_power = sum(main_powers) / len(main_powers)
                    total_system_energy = (avg_system_power / 1000) * duration_s
                    print(f"\nSystem Energy Summary:")
                    print(f"  Duration: {duration_s}s")
                    print(f"  Average System Power: {avg_system_power:.1f}mW")
                    print(f"  Total System Energy: {total_system_energy:.2f}J")
        
        # GPU utilization
        if gpu_util_keys:
            print(f"\nGPU Utilization:")
            for gpu_key in sorted(gpu_util_keys):
                gpu_utils = [d.get(gpu_key, 0) for d in data if gpu_key in d]
                if gpu_utils:
                    avg_util = sum(gpu_utils) / len(gpu_utils)
                    max_util = max(gpu_utils)
                    print(f"  {gpu_key}: avg={avg_util:.1f}%, max={max_util}%")
        
        # CPU utilization
        if cpu_util_keys:
            print(f"\nCPU Utilization ({len(cpu_util_keys)} cores):")
            for cpu_key in sorted(cpu_util_keys):
                cpu_utils = [d.get(cpu_key, 0) for d in data if cpu_key in d]
                if cpu_utils:
                    avg_util = sum(cpu_utils) / len(cpu_utils)
                    max_util = max(cpu_utils)
                    print(f"  {cpu_key}: avg={avg_util:.1f}%, max={max_util}%")
        
        # Temperature summary
        if temp_keys:
            print(f"\nTemperature Sensors ({len(temp_keys)} sensors):")
            for temp_key in sorted(temp_keys):
                temps = [d.get(temp_key, 0) for d in data if temp_key in d]
                if temps:
                    avg_temp = sum(temps) / len(temps)
                    max_temp = max(temps)
                    print(f"  {temp_key}: avg={avg_temp:.1f}°C, max={max_temp:.1f}°C")

def process_telemetry_log(log_file: str, output_dir: str, config_suffix: str = ""):
    """Convenience function to process telemetry log"""
    processor = TelemetryProcessor(log_file, output_dir, config_suffix)
    return processor.process_log()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process tegrastats log file')
    parser.add_argument('--log_file', default='tegrastats.log', help='Input log file')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    parser.add_argument('--config_suffix', default='', help='Config suffix for output files')
    
    args = parser.parse_args()
    process_telemetry_log(args.log_file, args.output_dir, args.config_suffix)