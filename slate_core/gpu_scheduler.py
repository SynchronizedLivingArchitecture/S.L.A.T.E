#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: gpu_scheduler [python]
# Author: Claude | Created: 2026-02-07T06:00:00Z
# Purpose: GPU resource management for dual-GPU SLATE setup
# ═══════════════════════════════════════════════════════════════════════════════
"""
GPU Scheduler Module
====================
Manages GPU resources for SLATE's dual-GPU setup (RTX 5070 Ti x2).

Features:
- Load-based GPU selection
- Memory tracking
- Task queuing with GPU affinity
- Ollama model placement

Usage:
    from slate_core.gpu_scheduler import GPUScheduler, get_available_gpu

    # Get best available GPU
    gpu_id = get_available_gpu()

    # Full scheduler
    scheduler = GPUScheduler()
    gpu = scheduler.allocate(memory_required=4096)  # 4GB
    scheduler.release(gpu)
"""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent


@dataclass
class GPUInfo:
    """GPU device information."""
    id: int
    name: str
    memory_total: int  # MB
    memory_used: int  # MB
    memory_free: int  # MB
    utilization: int  # percent
    temperature: int  # celsius
    compute_cap: str
    architecture: str


@dataclass
class GPUAllocation:
    """Represents a GPU allocation."""
    gpu_id: int
    task_id: str
    memory_requested: int
    allocated_at: float = field(default_factory=time.time)


class GPUScheduler:
    """
    GPU resource scheduler for SLATE system.

    Manages GPU allocation for tasks, balancing load across available GPUs.
    """

    def __init__(self, prefer_gpu: Optional[int] = None):
        """
        Initialize GPU scheduler.

        Args:
            prefer_gpu: Preferred GPU ID (0 or 1), or None for auto-select.
        """
        self.prefer_gpu = prefer_gpu
        self.allocations: List[GPUAllocation] = []
        self._lock = threading.Lock()
        self._state_file = WORKSPACE_ROOT / ".slate_gpu_state.json"

    def detect_gpus(self) -> List[GPUInfo]:
        """Detect available NVIDIA GPUs."""
        gpus = []

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                    "utilization.gpu,temperature.gpu,compute_cap",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return []

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    cc = parts[7]
                    arch = self._get_architecture(cc)

                    gpus.append(GPUInfo(
                        id=int(parts[0]),
                        name=parts[1],
                        memory_total=int(parts[2]),
                        memory_used=int(parts[3]),
                        memory_free=int(parts[4]),
                        utilization=int(parts[5]),
                        temperature=int(parts[6]),
                        compute_cap=cc,
                        architecture=arch
                    ))

        except Exception:
            pass

        return gpus

    def _get_architecture(self, compute_cap: str) -> str:
        """Map compute capability to architecture name."""
        cc = float(compute_cap) if compute_cap else 0
        if cc >= 12.0:
            return "Blackwell"
        elif cc >= 8.9:
            return "Ada Lovelace"
        elif cc >= 8.0:
            return "Ampere"
        elif cc >= 7.5:
            return "Turing"
        elif cc >= 7.0:
            return "Volta"
        elif cc >= 6.0:
            return "Pascal"
        return "Unknown"

    def get_best_gpu(self, memory_required: int = 0) -> Optional[int]:
        """
        Select the best GPU for a new task.

        Args:
            memory_required: Required GPU memory in MB.

        Returns:
            GPU ID (0 or 1) or None if no suitable GPU available.
        """
        gpus = self.detect_gpus()
        if not gpus:
            return None

        # Filter GPUs with enough free memory
        suitable = [g for g in gpus if g.memory_free >= memory_required]
        if not suitable:
            return None

        # If preferred GPU is specified and suitable, use it
        if self.prefer_gpu is not None:
            for g in suitable:
                if g.id == self.prefer_gpu:
                    return g.id

        # Score GPUs: prefer lower utilization and higher free memory
        def score(gpu: GPUInfo) -> float:
            util_score = (100 - gpu.utilization) / 100  # 0-1
            mem_score = gpu.memory_free / gpu.memory_total  # 0-1
            return util_score * 0.6 + mem_score * 0.4

        best = max(suitable, key=score)
        return best.id

    def allocate(
        self,
        task_id: str,
        memory_required: int = 0,
        gpu_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Allocate a GPU for a task.

        Args:
            task_id: Unique task identifier.
            memory_required: Required GPU memory in MB.
            gpu_id: Specific GPU to use, or None for auto-select.

        Returns:
            Allocated GPU ID or None if allocation failed.
        """
        with self._lock:
            # Determine GPU to use
            if gpu_id is None:
                gpu_id = self.get_best_gpu(memory_required)

            if gpu_id is None:
                return None

            # Record allocation
            allocation = GPUAllocation(
                gpu_id=gpu_id,
                task_id=task_id,
                memory_requested=memory_required
            )
            self.allocations.append(allocation)

            # Save state
            self._save_state()

            return gpu_id

    def release(self, task_id: str) -> bool:
        """
        Release GPU allocation for a task.

        Args:
            task_id: Task identifier to release.

        Returns:
            True if allocation was found and released.
        """
        with self._lock:
            for i, alloc in enumerate(self.allocations):
                if alloc.task_id == task_id:
                    del self.allocations[i]
                    self._save_state()
                    return True
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get GPU scheduler status."""
        gpus = self.detect_gpus()

        return {
            "gpus": [
                {
                    "id": g.id,
                    "name": g.name,
                    "memory_total_mb": g.memory_total,
                    "memory_used_mb": g.memory_used,
                    "memory_free_mb": g.memory_free,
                    "utilization_pct": g.utilization,
                    "temperature_c": g.temperature,
                    "compute_cap": g.compute_cap,
                    "architecture": g.architecture
                }
                for g in gpus
            ],
            "allocations": [
                {
                    "gpu_id": a.gpu_id,
                    "task_id": a.task_id,
                    "memory_mb": a.memory_requested,
                    "age_seconds": time.time() - a.allocated_at
                }
                for a in self.allocations
            ],
            "gpu_count": len(gpus),
            "total_memory_mb": sum(g.memory_total for g in gpus),
            "free_memory_mb": sum(g.memory_free for g in gpus)
        }

    def _save_state(self) -> None:
        """Save scheduler state to file."""
        state = {
            "allocations": [
                {
                    "gpu_id": a.gpu_id,
                    "task_id": a.task_id,
                    "memory_requested": a.memory_requested,
                    "allocated_at": a.allocated_at
                }
                for a in self.allocations
            ],
            "updated_at": time.time()
        }
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        """Load scheduler state from file."""
        if not self._state_file.exists():
            return

        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))
            self.allocations = [
                GPUAllocation(**a) for a in state.get("allocations", [])
            ]
        except Exception:
            pass


def get_available_gpu(memory_required: int = 0) -> Optional[int]:
    """
    Get the best available GPU ID.

    Args:
        memory_required: Required GPU memory in MB.

    Returns:
        GPU ID (0 or 1) or None if no GPU available.
    """
    scheduler = GPUScheduler()
    return scheduler.get_best_gpu(memory_required)


def get_gpu_status() -> Dict[str, Any]:
    """Get status of all GPUs."""
    scheduler = GPUScheduler()
    return scheduler.get_status()


def set_cuda_device(gpu_id: Optional[int] = None) -> None:
    """
    Set CUDA_VISIBLE_DEVICES environment variable.

    Args:
        gpu_id: GPU ID to use, or None for all GPUs.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Scheduler")
    parser.add_argument("--status", action="store_true", help="Show GPU status")
    parser.add_argument("--best", action="store_true", help="Get best available GPU")
    args = parser.parse_args()

    scheduler = GPUScheduler()

    if args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))
    elif args.best:
        gpu = scheduler.get_best_gpu()
        print(f"Best GPU: {gpu}")
    else:
        status = scheduler.get_status()
        print(f"GPUs: {status['gpu_count']}")
        print(f"Total Memory: {status['total_memory_mb']}MB")
        print(f"Free Memory: {status['free_memory_mb']}MB")
