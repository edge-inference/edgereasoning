"""
NVML-based telemetry handler for DGX H100 systems
Replaces tegrastats with pynvml for GPU monitoring
"""
import os
import time
import csv
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available. Install with: pip install pynvml")

@dataclass
class GPUMetrics:
    timestamp: float
    gpu_id: int
    gpu_name: str
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    memory_used_mb: int
    memory_total_mb: int
    memory_util_pct: float
    gpu_util_pct: float
    sm_clock_mhz: int
    mem_clock_mhz: int
    pcie_gen: int
    pcie_width: int
    pcie_throughput_rx_mbps: float
    pcie_throughput_tx_mbps: float
    nvlink_throughput_rx_mbps: Optional[float] = None
    nvlink_throughput_tx_mbps: Optional[float] = None
    fan_speed_pct: Optional[int] = None
    performance_state: Optional[str] = None

class NVMLTelemetryHandler:
    """
    NVML-based telemetry handler for comprehensive GPU monitoring on DGX systems
    """
    
    def __init__(self, output_dir: str, config_suffix: str = "", sampling_interval: float = 1.0):
        if not NVML_AVAILABLE:
            raise ImportError("pynvml is required for NVML telemetry")
            
        self.output_dir = output_dir
        self.config_suffix = config_suffix
        self.sampling_interval = sampling_interval
        self.csv_file = os.path.join(output_dir, f"gpu_telemetry_{config_suffix}.csv")
        
        self._running = False
        self._thread = None
        self._metrics_buffer = []
        self._lock = threading.Lock()
        
        self._ensure_output_dir()
        self._initialize_nvml()
        
    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _initialize_nvml(self):
        """Initialize NVML and get GPU information"""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.devices = []
            # Only monitor GPUs in CUDA_VISIBLE_DEVICES
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible:
                try:
                    vis_list = [int(x) for x in visible.split(',') if x.strip().isdigit()]
                except:
                    vis_list = []
            else:
                vis_list = list(range(self.device_count))
            for i in range(self.device_count):
                if i not in vis_list:
                    continue
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else raw_name
                self.devices.append({
                    'handle': handle,
                    'name': name,
                    'index': i
                })
                
            print(f"Initialized NVML monitoring for {self.device_count} GPUs")
            for dev in self.devices:
                print(f"  GPU {dev['index']}: {dev['name']}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")
    
    def _collect_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect comprehensive GPU metrics from all devices"""
        metrics = []
        timestamp = time.time()
        
        for device in self.devices:
            handle = device['handle']
            gpu_id = device['index']
            gpu_name = device['name']
            
            try:
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0.0
                
                # Power
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power_draw = power_limit = 0.0
                
                # Memory
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_mb = mem_info.used // (1024 * 1024)
                    memory_total_mb = mem_info.total // (1024 * 1024)
                    memory_util_pct = (mem_info.used / mem_info.total) * 100
                except:
                    memory_used_mb = memory_total_mb = 0
                    memory_util_pct = 0.0
                
                # GPU Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu
                except:
                    gpu_util_pct = 0
                
                # Clock speeds
                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    sm_clock = mem_clock = 0
                
                # PCIe info
                try:
                    pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                    pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                except:
                    pcie_gen = pcie_width = 0
                
                # PCIe throughput
                try:
                    pcie_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_COUNT)
                    pcie_rx = pcie_throughput[pynvml.NVML_PCIE_UTIL_RX_BYTES] / (1024 * 1024)  # MB/s
                    pcie_tx = pcie_throughput[pynvml.NVML_PCIE_UTIL_TX_BYTES] / (1024 * 1024)  # MB/s
                except:
                    pcie_rx = pcie_tx = 0.0
                
                # Fan speed (if available)
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan_speed = None
                
                # Performance state
                try:
                    perf_state = pynvml.nvmlDeviceGetPerformanceState(handle)
                    perf_state_str = f"P{perf_state}"
                except:
                    perf_state_str = None
                
                # NVLink throughput (H100 specific)
                nvlink_rx = nvlink_tx = None
                try:
                    # Try to get NVLink utilization for each link
                    nvlink_rx_total = nvlink_tx_total = 0
                    link_count = 0
                    for link_id in range(18):  # H100 can have up to 18 NVLink connections
                        try:
                            nvlink_util = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, link_id, 0)  # RX
                            if nvlink_util[0] > 0:  # If link is active
                                nvlink_rx_total += nvlink_util[0]
                                link_count += 1
                            nvlink_util = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, link_id, 1)  # TX
                            if nvlink_util[0] > 0:
                                nvlink_tx_total += nvlink_util[0]
                        except:
                            continue
                    if link_count > 0:
                        nvlink_rx = nvlink_rx_total / (1024 * 1024)  # Convert to MB/s
                        nvlink_tx = nvlink_tx_total / (1024 * 1024)
                except:
                    pass
                
                gpu_metrics = GPUMetrics(
                    timestamp=timestamp,
                    gpu_id=gpu_id,
                    gpu_name=gpu_name,
                    temperature_c=temp,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                    memory_util_pct=memory_util_pct,
                    gpu_util_pct=gpu_util_pct,
                    sm_clock_mhz=sm_clock,
                    mem_clock_mhz=mem_clock,
                    pcie_gen=pcie_gen,
                    pcie_width=pcie_width,
                    pcie_throughput_rx_mbps=pcie_rx,
                    pcie_throughput_tx_mbps=pcie_tx,
                    nvlink_throughput_rx_mbps=nvlink_rx,
                    nvlink_throughput_tx_mbps=nvlink_tx,
                    fan_speed_pct=fan_speed,
                    performance_state=perf_state_str
                )
                
                metrics.append(gpu_metrics)
                
            except Exception as e:
                logging.warning(f"Failed to collect metrics for GPU {gpu_id}: {e}")
                
        return metrics
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                metrics = self._collect_gpu_metrics()
                with self._lock:
                    self._metrics_buffer.extend(metrics)
                    
                time.sleep(self.sampling_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                if self._running:  # Only sleep if still running
                    time.sleep(self.sampling_interval)
    
    def start_logging(self):
        """Start GPU telemetry logging"""
        if self._running:
            print("GPU telemetry already running")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        
        print(f"Started GPU telemetry logging to: {self.csv_file}")
        print(f"Monitoring {self.device_count} GPUs at {self.sampling_interval}s intervals")
    
    def stop_logging(self):
        """Stop GPU telemetry logging and save data"""
        if not self._running:
            return
            
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Save buffered metrics to CSV
        self._save_metrics_to_csv()
        
        print(f"Stopped GPU telemetry logging")
        if os.path.exists(self.csv_file):
            file_size = os.path.getsize(self.csv_file)
            print(f"GPU telemetry saved: {self.csv_file} ({file_size} bytes)")
    
    def _save_metrics_to_csv(self):
        """Save all buffered metrics to CSV file"""
        with self._lock:
            if not self._metrics_buffer:
                print("No GPU metrics to save")
                return
                
            metrics_to_save = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
        
        # Write to CSV
        with open(self.csv_file, 'w', newline='') as f:
            if not metrics_to_save:
                return
                
            writer = csv.writer(f)
            
            # Header
            header = [
                'timestamp', 'datetime', 'gpu_id', 'gpu_name',
                'temperature_c', 'power_draw_w', 'power_limit_w',
                'memory_used_mb', 'memory_total_mb', 'memory_util_pct',
                'gpu_util_pct', 'sm_clock_mhz', 'mem_clock_mhz',
                'pcie_gen', 'pcie_width', 'pcie_throughput_rx_mbps', 'pcie_throughput_tx_mbps',
                'nvlink_throughput_rx_mbps', 'nvlink_throughput_tx_mbps',
                'fan_speed_pct', 'performance_state'
            ]
            writer.writerow(header)
            
            # Data
            for metric in metrics_to_save:
                dt_str = datetime.fromtimestamp(metric.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                row = [
                    metric.timestamp, dt_str, metric.gpu_id, metric.gpu_name,
                    metric.temperature_c, metric.power_draw_w, metric.power_limit_w,
                    metric.memory_used_mb, metric.memory_total_mb, metric.memory_util_pct,
                    metric.gpu_util_pct, metric.sm_clock_mhz, metric.mem_clock_mhz,
                    metric.pcie_gen, metric.pcie_width, 
                    metric.pcie_throughput_rx_mbps, metric.pcie_throughput_tx_mbps,
                    metric.nvlink_throughput_rx_mbps, metric.nvlink_throughput_tx_mbps,
                    metric.fan_speed_pct, metric.performance_state
                ]
                writer.writerow(row)
        
        print(f"Saved {len(metrics_to_save)} GPU metric samples")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for GPU telemetry"""
        with self._lock:
            if not self._metrics_buffer:
                return {}
            
            metrics_by_gpu = {}
            for metric in self._metrics_buffer:
                if metric.gpu_id not in metrics_by_gpu:
                    metrics_by_gpu[metric.gpu_id] = []
                metrics_by_gpu[metric.gpu_id].append(metric)
        
        summary = {}
        for gpu_id, metrics in metrics_by_gpu.items():
            if not metrics:
                continue
                
            temps = [m.temperature_c for m in metrics if m.temperature_c > 0]
            powers = [m.power_draw_w for m in metrics if m.power_draw_w > 0]
            mem_utils = [m.memory_util_pct for m in metrics if m.memory_util_pct > 0]
            gpu_utils = [m.gpu_util_pct for m in metrics if m.gpu_util_pct > 0]
            
            summary[f'gpu_{gpu_id}'] = {
                'samples': len(metrics),
                'avg_temp_c': sum(temps) / len(temps) if temps else 0,
                'max_temp_c': max(temps) if temps else 0,
                'avg_power_w': sum(powers) / len(powers) if powers else 0,
                'max_power_w': max(powers) if powers else 0,
                'avg_memory_util_pct': sum(mem_utils) / len(mem_utils) if mem_utils else 0,
                'avg_gpu_util_pct': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                'total_energy_j': sum(powers) * self.sampling_interval if powers else 0
            }
        
        return summary
    
    def get_log_file(self) -> str:
        """Get the path to the CSV log file"""
        return self.csv_file
    
    def __enter__(self):
        self.start_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_logging()
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    
    def start(self):
        """Start background GPU metrics collection."""
        if self._running:
            return
        self._running = True
        # Initialize CSV with header
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","gpu_id","gpu_name","temperature_c","power_draw_w","power_limit_w","memory_used_mb","memory_total_mb","memory_util_pct","gpu_util_pct","sm_clock_mhz","mem_clock_mhz","pcie_gen","pcie_width","pcie_rx_mbps","pcie_tx_mbps","nvlink_rx_mbps","nvlink_tx_mbps","fan_speed_pct","performance_state"])
        # Start collection thread
        self._thread = threading.Thread(target=self._run_collection, daemon=True)
        self._thread.start()

    def _run_collection(self):
        """Thread target for collecting GPU metrics at intervals."""
        while self._running:
            metrics = self._collect_gpu_metrics()
            with self._lock:
                self._metrics_buffer.extend(metrics)
            # Append metrics to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for m in metrics:
                    writer.writerow([
                        m.timestamp, m.gpu_id, m.gpu_name, m.temperature_c,
                        m.power_draw_w, m.power_limit_w, m.memory_used_mb,
                        m.memory_total_mb, m.memory_util_pct, m.gpu_util_pct,
                        m.sm_clock_mhz, m.mem_clock_mhz, m.pcie_gen,
                        m.pcie_width, m.pcie_throughput_rx_mbps,
                        m.pcie_throughput_tx_mbps, m.nvlink_throughput_rx_mbps,
                        m.nvlink_throughput_tx_mbps, m.fan_speed_pct,
                        m.performance_state
                    ])
            time.sleep(self.sampling_interval)

    def stop(self) -> Dict[int, Dict[str, float]]:
        """Stop data collection, compute and return summary stats per GPU."""
        if not self._running:
            return {}
        self._running = False
        if self._thread:
            self._thread.join()
        # Compute summary from buffer
        stats: Dict[int, List[GPUMetrics]] = {}
        with self._lock:
            for m in self._metrics_buffer:
                stats.setdefault(m.gpu_id, []).append(m)
        summary: Dict[int, Dict[str, float]] = {}
        for gpu_id, records in stats.items():
            temps = [r.temperature_c for r in records]
            powers = [r.power_draw_w for r in records]
            utils = [r.gpu_util_pct for r in records]
            mem_utils = [r.memory_util_pct for r in records]
            # Estimate energy by trapezoidal integration
            energy_j = 0.0
            for i in range(1, len(records)):
                dt = records[i].timestamp - records[i-1].timestamp
                energy_j += ((powers[i] + powers[i-1]) / 2) * dt
            summary[gpu_id] = {
                'avg_temp_c': sum(temps)/len(temps) if temps else 0.0,
                'max_temp_c': max(temps) if temps else 0.0,
                'avg_power_w': sum(powers)/len(powers) if powers else 0.0,
                'max_power_w': max(powers) if powers else 0.0,
                'total_energy_j': energy_j,
                'avg_gpu_util_pct': sum(utils)/len(utils) if utils else 0.0,
                'avg_memory_util_pct': sum(mem_utils)/len(mem_utils) if mem_utils else 0.0
            }
        return summary