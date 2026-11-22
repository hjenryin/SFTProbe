"""
GPU Detection and Availability Utilities

This module provides utilities for detecting available GPUs based on memory utilization.
Useful for distributed training and multi-GPU job scheduling.

Usage:
    from gpu_utils import get_available_gpus, get_num_gpus
    
    available = get_available_gpus(threshold=10.0)
    print(f"Available GPUs: {available}")
"""

import os
import subprocess
import torch

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount, nvmlShutdown
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def get_allowed_gpus() -> list[int] | None:
    """Returns a list of physical GPU indices allowed by CUDA_VISIBLE_DEVICES, or None if not set."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    return None


def resolve_physical_gpu(logical_idx: int) -> int:
    """Translate a logical GPU index (relative to the current process's CUDA_VISIBLE_DEVICES)
    into its absolute physical GPU index.

    When spawning a subprocess with `env["CUDA_VISIBLE_DEVICES"] = "<x>"`, the child
    resolves `<x>` against the *physical* GPUs on the machine, NOT against the parent's
    CUDA_VISIBLE_DEVICES list. So if you pick GPU `gpu_idx` from `get_available_gpus()`
    (which returns logical indices), you MUST translate it back to a physical index
    before passing it to the child, or the child will land on the wrong physical GPU.

    Examples:
        CUDA_VISIBLE_DEVICES="2,3" in parent:
            resolve_physical_gpu(0) -> 2
            resolve_physical_gpu(1) -> 3
        CUDA_VISIBLE_DEVICES unset:
            resolve_physical_gpu(0) -> 0  (logical == physical)
    """
    allowed = get_allowed_gpus()
    if allowed is None:
        return logical_idx
    if logical_idx < 0 or logical_idx >= len(allowed):
        raise ValueError(
            f"Logical GPU {logical_idx} is out of range for "
            f"CUDA_VISIBLE_DEVICES={allowed} (len={len(allowed)})"
        )
    return allowed[logical_idx]



def get_available_gpus(threshold: float = 10.0, max_gpu: int = None) -> list[int]:
    """Get list of GPU indices with memory utilization below threshold.
    
    Uses NVML to check each GPU's memory utilization and returns only those
    with utilization below the specified threshold. This is useful for
    scheduling jobs on GPUs that are not currently busy.
    
    Args:
        threshold: Memory utilization percentage threshold. GPUs with utilization
                   above this value are considered busy and excluded. Default: 10.0
        max_gpu: Maximum number of GPUs to return. If None, returns all available.
    
    Returns:
        List of available GPU indices sorted by utilization (lowest first).
    
    Example:
        >>> available = get_available_gpus(threshold=15.0)
        >>> print(f"Available GPUs: {available}")
        GPU 0 has memory utilization 5.2%, included
        GPU 1 has memory utilization 85.0%, excluded
        GPU 2 has memory utilization 3.1%, included
        Available GPUs: [2, 0]
    """
    if not NVML_AVAILABLE:
        print("Warning: pynvml not available, falling back to torch.cuda.device_count()")
        gpus = list(range(torch.cuda.device_count()))
        return gpus[:max_gpu] if max_gpu else gpus
    
    try:
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        gpu_utils = []

        allowed_gpus = get_allowed_gpus()
        target_gpus = allowed_gpus if allowed_gpus is not None else list(range(gpu_count))

        for logical_idx, physical_idx in enumerate(target_gpus):
            try:
                handle = nvmlDeviceGetHandleByIndex(physical_idx)
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                memory_util = (memory_info.used / memory_info.total) * 100

                if memory_util < threshold:
                    gpu_utils.append((logical_idx, memory_util))
                    print(f"GPU {physical_idx} (logical {logical_idx}) has memory utilization {memory_util:.1f}%, included")
                else:
                    print(f"GPU {physical_idx} (logical {logical_idx}) has memory utilization {memory_util:.1f}%, excluded")
            except Exception as e:
                print(f"Warning: Could not query GPU {physical_idx}: {e}")
        
        nvmlShutdown()
        
        # Sort by utilization (lowest first) for optimal scheduling
        gpu_utils.sort(key=lambda x: x[1])
        available = [gpu_id for gpu_id, _ in gpu_utils]
        
        if max_gpu:
            available = available[:max_gpu]
        
        return available
        
    except Exception as e:
        print(f"Warning: Could not check GPU availability via NVML: {e}")
        print("Falling back to torch.cuda.device_count()")
        gpus = list(range(torch.cuda.device_count()))
        return gpus[:max_gpu] if max_gpu else gpus


def get_num_gpus() -> int:
    """Get the number of GPUs visible to this process.
    
    Respects CUDA_VISIBLE_DEVICES environment variable.
    
    Returns:
        Number of visible GPUs.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return len([x for x in cuda_visible.split(",") if x.strip()])
    
    if NVML_AVAILABLE:
        try:
            nvmlInit()
            count = nvmlDeviceGetCount()
            nvmlShutdown()
            return count
        except Exception:
            pass
    
    return torch.cuda.device_count()


def get_physical_num_gpus() -> int:
    """Get the physical number of GPUs on the system.
    
    Ignores CUDA_VISIBLE_DEVICES environment variable.
    
    Returns:
        Number of physical GPUs on the system.
    """
    if NVML_AVAILABLE:
        try:
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            nvmlInit()
            count = nvmlDeviceGetCount()
            nvmlShutdown()
            return count
        except Exception:
            pass
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return get_num_gpus()


def get_gpu_memory_info(gpu_id: int = 0) -> dict:
    """Get memory information for a specific GPU.
    
    Args:
        gpu_id: Logical GPU index to query.
    
    Returns:
        Dictionary with 'total', 'used', 'free' memory in bytes,
        and 'utilization' as a percentage.
    """
    if not NVML_AVAILABLE:
        raise RuntimeError("pynvml is required for get_gpu_memory_info")
    
    allowed_gpus = get_allowed_gpus()
    if allowed_gpus is not None:
        if gpu_id >= len(allowed_gpus):
            raise ValueError(f"Logical GPU {gpu_id} out of bounds")
        physical_id = allowed_gpus[gpu_id]
    else:
        physical_id = gpu_id
        
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(physical_id)
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    
    return {
        'total': memory_info.total,
        'used': memory_info.used,
        'free': memory_info.free,
        'utilization': (memory_info.used / memory_info.total) * 100
    }


def wait_for_gpu(threshold: float = 10.0, gpu_ids: list[int] = None, 
                 poll_interval: float = 30.0, timeout: float = None) -> int:
    """Wait until a GPU becomes available.
    
    Polls GPU memory utilization until a GPU falls below the threshold.
    
    Args:
        threshold: Memory utilization threshold percentage.
        gpu_ids: List of GPU IDs to consider. If None, checks all GPUs.
        poll_interval: Seconds to wait between checks.
        timeout: Maximum seconds to wait. None means wait indefinitely.
    
    Returns:
        Index of the first available GPU.
    
    Raises:
        TimeoutError: If timeout is reached before a GPU becomes available.
    """
    import time
    
    start_time = time.time()
    
    while True:
        available = get_available_gpus(threshold=threshold)
        
        if gpu_ids:
            available = [g for g in available if g in gpu_ids]
        
        if available:
            return available[0]
        
        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError(f"No GPU became available within {timeout} seconds")
        
        print(f"No GPUs available (threshold: {threshold}%). Waiting {poll_interval}s...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("GPU Detection Demo")
    print("=" * 60)
    
    print(f"\nNVML Available: {NVML_AVAILABLE}")
    print(f"Number of GPUs (get_num_gpus): {get_num_gpus()}")
    print(f"Physical GPUs (get_physical_num_gpus): {get_physical_num_gpus()}")
    
    print("\n" + "-" * 60)
    print("Available GPUs (threshold=10%):")
    print("-" * 60)
    available = get_available_gpus(threshold=10.0)
    print(f"Result: {available}")
    
    if available and NVML_AVAILABLE:
        print("\n" + "-" * 60)
        print(f"Memory info for GPU {available[0]}:")
        print("-" * 60)
        info = get_gpu_memory_info(available[0])
        for k, v in info.items():
            if k == 'utilization':
                print(f"  {k}: {v:.1f}%")
            else:
                print(f"  {k}: {v / (1024**3):.2f} GB")
