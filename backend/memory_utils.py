import gc
import os

import psutil
import torch


def get_memory_stats() -> dict:
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    stats = {
        "rss_mb": process.memory_info().rss / 1024 / 1024,
        "vms_mb": process.memory_info().vms / 1024 / 1024,
    }

    if torch.backends.mps.is_available():
        stats["mps_allocated_mb"] = torch.mps.current_allocated_memory() / 1024 / 1024
        stats["mps_driver_mb"] = torch.mps.driver_allocated_memory() / 1024 / 1024

    return stats


def log_memory(label: str):
    """Log memory usage with a label."""
    stats = get_memory_stats()
    print(f"[MEMORY] {label}: RSS={stats['rss_mb']:.1f}MB", end="")
    if "mps_allocated_mb" in stats:
        print(f", MPS={stats['mps_allocated_mb']:.1f}MB", end="")
    print()


def clear_memory_cache():
    """Clear garbage and device memory cache."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
