"""Synchronized online wall time and per-device allocated CUDA memory."""
import os
import platform
from importlib.metadata import version
from time import perf_counter

import numpy as np
import torch


class OnlineMeasurement:
    def __init__(self, model):
        self.devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
        if not self.devices:
            raise ValueError("Performance measurement requires a CUDA model.")

    def synchronize(self):
        for device in self.devices:
            torch.cuda.synchronize(device)

    def prepare(self):
        self.synchronize()
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)

    def start(self):
        self.synchronize()
        self.prefill_peak = {d: torch.cuda.max_memory_allocated(d) for d in self.devices}
        self.before = {d: torch.cuda.memory_allocated(d) for d in self.devices}
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)
        self.started = perf_counter()

    def stop(self):
        self.synchronize()
        elapsed = perf_counter() - self.started
        memory = {}
        for device in self.devices:
            peak = torch.cuda.max_memory_allocated(device)
            memory[str(device)] = dict(allocated_before_bytes=self.before[device],
                                      online_peak_allocated_bytes=peak,
                                      additional_peak_bytes=peak - self.before[device],
                                      sample_peak_allocated_bytes=max(peak, self.prefill_peak[device]))
        self.stats = dict(online_time_s=elapsed, gpu_memory=memory)

    def environment(self):
        return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    transformers=version("transformers"), peft=version("peft"),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    devices={str(d): dict(name=torch.cuda.get_device_name(d),
                                         total_bytes=torch.cuda.get_device_properties(d).total_memory)
                             for d in self.devices})


def summarize_performance(records):
    times = [row["online_time_s"] for row in records]
    return dict(online_time_s=dict(total=sum(times), mean=float(np.mean(times)),
                                  p50=float(np.percentile(times, 50)), p95=float(np.percentile(times, 95))),
                mean_generated_tokens=float(np.mean([row["generated_tokens"] for row in records])),
                gpu_memory={device: {key: max(row["gpu_memory"][device][key] for row in records)
                                     for key in records[0]["gpu_memory"][device]}
                            for device in records[0]["gpu_memory"]})
