#!/usr/bin/env python3
"""Memory monitoring utilities for GPU (VRAM) and system RAM tracking."""

import threading
import time
import psutil
import pynvml


class MemoryMonitor:
    """Monitor GPU VRAM and system RAM usage during execution.

    Continuously samples memory usage in a background thread to capture
    peak usage that might be missed by single-point measurements.
    """

    def __init__(self, gpu_id=0, sample_interval=0.1):
        """Initialize memory monitor.

        Args:
            gpu_id: GPU device ID to monitor (default: 0)
            sample_interval: Sampling interval in seconds (default: 0.1s = 100ms)
        """
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        self.peak_vram_bytes = 0
        self.start_ram_bytes = 0
        self.peak_ram_bytes = 0
        self.monitoring = False
        self.thread = None

        # Initialize NVIDIA Management Library
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.process = psutil.Process()

    def _monitor_loop(self):
        """Background thread that continuously samples GPU and RAM memory."""
        while self.monitoring:
            try:
                # Sample GPU VRAM
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.peak_vram_bytes = max(self.peak_vram_bytes, mem_info.used)

                # Sample system RAM
                current_ram = self.process.memory_info().rss
                self.peak_ram_bytes = max(self.peak_ram_bytes, current_ram)
            except Exception:
                pass  # Ignore sampling errors

            time.sleep(self.sample_interval)

    def start(self):
        """Start monitoring memory usage in background thread."""
        self.start_ram_bytes = self.process.memory_info().rss
        self.peak_ram_bytes = self.start_ram_bytes
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return peak memory usage.

        Returns:
            tuple: (peak_vram_gb, peak_ram_gb) - Peak VRAM and RAM usage in GB
        """
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

        peak_vram_gb = self.peak_vram_bytes / (1024 ** 3)
        peak_ram_gb = (self.peak_ram_bytes - self.start_ram_bytes) / (1024 ** 3)

        return peak_vram_gb, peak_ram_gb

    def shutdown(self):
        """Clean up NVML resources."""
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
