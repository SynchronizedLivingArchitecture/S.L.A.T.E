#!/usr/bin/env python3
# Modified: 2026-02-08T07:00:00Z | Author: COPILOT | Change: Create stability module — retry/backoff, health monitoring, circuit breaker, resource guards
# Modified: 2026-02-08T15:00:00Z | Author: COPILOT | Change: Pass 3 — Add SORT-AI drift mitigation, adaptive orchestration, service watchdog, availability tracker, planning context deduplication
"""
SLATE Stability Module
======================
Production-grade resilience primitives for local AI orchestration.

Features:
    - Retry with exponential backoff (configurable)
    - Circuit breaker pattern (prevents cascading failures)
    - Health monitor (CPU/GPU/memory/disk thresholds)
    - Resource guard (blocks operations when resources are critical)
    - Service watchdog (restarts failed services)
    - SORT-AI drift mitigation (detects contradictory/redundant decisions)
    - Adaptive orchestration (reinforcement-based agent priority)
    - Availability tracker (99.9% target with rolling window)
    - Planning context deduplication (prevents redundant info in extended contexts)

Usage:
    from slate.stability import retry_with_backoff, CircuitBreaker, HealthMonitor
    from slate.stability import DriftMitigator, AdaptiveOrchestrator, AvailabilityTracker

    # Retry any function with backoff
    result = retry_with_backoff(flaky_api_call, max_attempts=5)

    # Circuit breaker for service calls
    cb = CircuitBreaker("ollama", failure_threshold=3)
    result = cb.call(ollama_request)

    # Health monitoring
    monitor = HealthMonitor()
    status = monitor.check_all()

    # SORT-AI drift mitigation
    drift = DriftMitigator()
    drift.record_decision("route_to_gpu", {"model": "slate-coder", "gpu": 0})
    conflicts = drift.check_drift()

    # Adaptive orchestration
    orchestrator = AdaptiveOrchestrator()
    orchestrator.record_outcome("ALPHA", success=True, latency=1.2)
    best_agent = orchestrator.select_agent(["ALPHA", "BETA", "GAMMA"])

    # Availability tracking
    tracker = AvailabilityTracker(target=0.999)
    tracker.record_check(healthy=True)
    print(tracker.availability)  # 0.9995
"""

import collections
import functools
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

WORKSPACE = Path(__file__).parent.parent

# ─── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("slate.stability")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Retry with Exponential Backoff
# ═══════════════════════════════════════════════════════════════════════════════

def retry_with_backoff(
    func: Callable[..., T],
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a function with exponential backoff retry.

    Args:
        func: Function to call.
        max_attempts: Maximum number of attempts (default 5).
        base_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay cap in seconds (default 60.0).
        backoff_factor: Multiplier per retry (default 2.0 = 1s, 2s, 4s, 8s, 16s).
        retryable_exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback(attempt, exception) called before each retry.
        *args, **kwargs: Passed to func.

    Returns:
        Result of func(*args, **kwargs).

    Raises:
        Last exception if all attempts exhausted.
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_attempts:
                log.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                raise

            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            log.warning(
                f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            if on_retry:
                on_retry(attempt, e)

            time.sleep(delay)

    raise last_exception  # type: ignore[misc]


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator version of retry_with_backoff.

    Usage:
        @retry(max_attempts=3)
        def fetch_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff(
                func, max_attempts=max_attempts, base_delay=base_delay,
                retryable_exceptions=retryable_exceptions, *args, **kwargs,
            )
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    When a service fails `failure_threshold` times consecutively,
    the circuit opens and rejects calls for `recovery_timeout` seconds.
    After timeout, it enters half-open state and allows one test call.

    Usage:
        cb = CircuitBreaker("ollama", failure_threshold=3, recovery_timeout=30)
        try:
            result = cb.call(my_function, arg1, arg2)
        except CircuitBreakerOpen:
            # Service is down, use fallback
            ...
    """
    name: str
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    log.info(f"Circuit '{self.name}' → HALF_OPEN (testing recovery)")
            return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute func through the circuit breaker."""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit '{self.name}' is OPEN — {self.name} appears down. "
                f"Recovery in {self.recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                log.info(f"Circuit '{self.name}' → CLOSED (recovered)")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def _on_failure(self, error: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                log.error(
                    f"Circuit '{self.name}' → OPEN after {self._failure_count} failures. "
                    f"Last error: {error}"
                )

    def reset(self) -> None:
        """Manually reset the circuit."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            log.info(f"Circuit '{self.name}' manually reset → CLOSED")

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class CircuitBreakerOpen(Exception):
    """Raised when a circuit breaker is open (service unavailable)."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Health Monitor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceThresholds:
    """Thresholds for resource alerts."""
    cpu_percent: float = 90.0       # CPU usage %
    memory_percent: float = 85.0    # Memory usage %
    disk_percent: float = 90.0      # Disk usage %
    gpu_memory_percent: float = 90.0  # GPU memory usage %
    gpu_temp_celsius: float = 85.0  # GPU temperature


class HealthMonitor:
    """Monitor system resources and service health.

    Usage:
        monitor = HealthMonitor()
        status = monitor.check_all()
        if not status["healthy"]:
            print("System degraded:", status["warnings"])
    """

    def __init__(self, thresholds: Optional[ResourceThresholds] = None):
        self.thresholds = thresholds or ResourceThresholds()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def register_service(self, name: str, **kwargs: Any) -> CircuitBreaker:
        """Register a service with a circuit breaker."""
        cb = CircuitBreaker(name, **kwargs)
        self._circuit_breakers[name] = cb
        return cb

    def check_cpu(self) -> dict:
        """Check CPU usage."""
        try:
            import psutil
            usage = psutil.cpu_percent(interval=0.5)
            ok = usage < self.thresholds.cpu_percent
            return {
                "healthy": ok,
                "usage_percent": usage,
                "threshold": self.thresholds.cpu_percent,
                "warning": None if ok else f"CPU at {usage:.1f}% (threshold: {self.thresholds.cpu_percent}%)",
            }
        except ImportError:
            # Fallback without psutil
            return {"healthy": True, "usage_percent": 0, "warning": "psutil not available"}

    def check_memory(self) -> dict:
        """Check memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            ok = mem.percent < self.thresholds.memory_percent
            return {
                "healthy": ok,
                "usage_percent": mem.percent,
                "available_gb": round(mem.available / (1024**3), 1),
                "total_gb": round(mem.total / (1024**3), 1),
                "threshold": self.thresholds.memory_percent,
                "warning": None if ok else f"Memory at {mem.percent:.1f}% (threshold: {self.thresholds.memory_percent}%)",
            }
        except ImportError:
            return {"healthy": True, "usage_percent": 0, "warning": "psutil not available"}

    def check_disk(self) -> dict:
        """Check disk usage."""
        try:
            import psutil
            disk = psutil.disk_usage(str(WORKSPACE))
            ok = disk.percent < self.thresholds.disk_percent
            return {
                "healthy": ok,
                "usage_percent": disk.percent,
                "free_gb": round(disk.free / (1024**3), 1),
                "total_gb": round(disk.total / (1024**3), 1),
                "threshold": self.thresholds.disk_percent,
                "warning": None if ok else f"Disk at {disk.percent:.1f}% (threshold: {self.thresholds.disk_percent}%)",
            }
        except ImportError:
            return {"healthy": True, "usage_percent": 0, "warning": "psutil not available"}

    def check_gpu(self) -> dict:
        """Check GPU health using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return {"healthy": False, "gpus": [], "warning": "nvidia-smi failed"}

            gpus = []
            all_ok = True
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                idx, name, mem_used, mem_total, temp, util = parts
                mem_pct = (float(mem_used) / float(mem_total)) * 100 if float(mem_total) > 0 else 0
                temp_f = float(temp)
                warnings = []
                if mem_pct > self.thresholds.gpu_memory_percent:
                    warnings.append(f"GPU {idx} memory at {mem_pct:.0f}%")
                    all_ok = False
                if temp_f > self.thresholds.gpu_temp_celsius:
                    warnings.append(f"GPU {idx} temp at {temp_f:.0f}°C")
                    all_ok = False

                gpus.append({
                    "index": int(idx),
                    "name": name,
                    "memory_used_mb": int(float(mem_used)),
                    "memory_total_mb": int(float(mem_total)),
                    "memory_percent": round(mem_pct, 1),
                    "temperature_c": int(temp_f),
                    "utilization_percent": int(float(util)),
                    "warnings": warnings,
                })

            return {
                "healthy": all_ok,
                "gpus": gpus,
                "gpu_count": len(gpus),
                "warning": "; ".join(w for g in gpus for w in g["warnings"]) or None,
            }
        except Exception as e:
            return {"healthy": False, "gpus": [], "warning": str(e)}

    def check_ollama(self) -> dict:
        """Check Ollama service health."""
        try:
            import urllib.request
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                return {
                    "healthy": True,
                    "models": len(models),
                    "running": True,
                }
        except Exception as e:
            return {"healthy": False, "running": False, "warning": str(e)}

    def check_services(self) -> dict:
        """Check all registered circuit breakers."""
        return {
            name: cb.status()
            for name, cb in self._circuit_breakers.items()
        }

    def check_all(self) -> dict:
        """Run all health checks and return aggregate status."""
        checks = {
            "cpu": self.check_cpu(),
            "memory": self.check_memory(),
            "disk": self.check_disk(),
            "gpu": self.check_gpu(),
            "ollama": self.check_ollama(),
        }

        if self._circuit_breakers:
            checks["services"] = self.check_services()

        warnings = [
            c.get("warning") for c in checks.values()
            if isinstance(c, dict) and c.get("warning")
        ]

        healthy = all(
            c.get("healthy", True) for c in checks.values()
            if isinstance(c, dict)
        )

        return {
            "healthy": healthy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "warnings": warnings,
        }

    def print_status(self) -> None:
        """Print health status to console."""
        status = self.check_all()

        print()
        print("=" * 60)
        print("  SLATE Health Monitor")
        print("=" * 60)

        icon = "✓" if status["healthy"] else "✗"
        print(f"\n  Overall: [{icon}] {'Healthy' if status['healthy'] else 'DEGRADED'}")

        for name, check in status["checks"].items():
            if not isinstance(check, dict):
                continue
            icon = "✓" if check.get("healthy", True) else "✗"
            detail = ""
            if name == "cpu":
                detail = f"{check.get('usage_percent', 0):.0f}%"
            elif name == "memory":
                detail = f"{check.get('usage_percent', 0):.0f}% ({check.get('available_gb', 0)} GB free)"
            elif name == "disk":
                detail = f"{check.get('usage_percent', 0):.0f}% ({check.get('free_gb', 0)} GB free)"
            elif name == "gpu":
                detail = f"{check.get('gpu_count', 0)} GPU(s)"
            elif name == "ollama":
                detail = f"{'running' if check.get('running') else 'offline'}"
            print(f"  [{icon}] {name:>10}: {detail}")
            if check.get("warning"):
                print(f"      ⚠ {check['warning']}")

        if status["warnings"]:
            print(f"\n  Warnings: {len(status['warnings'])}")
            for w in status["warnings"]:
                print(f"    ⚠ {w}")

        print()
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Resource Guard
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceGuard:
    """Block operations when system resources are critically low.

    Usage:
        guard = ResourceGuard()
        if guard.can_proceed():
            run_inference()
        else:
            print("Resources too low:", guard.last_check)
    """

    def __init__(self, thresholds: Optional[ResourceThresholds] = None):
        self.monitor = HealthMonitor(thresholds)
        self.last_check: dict = {}
        self._last_check_time: float = 0
        self._cache_ttl: float = 10.0  # Cache health check for 10s

    def can_proceed(self, require_gpu: bool = False) -> bool:
        """Check if it's safe to proceed with a resource-intensive operation."""
        now = time.time()
        if now - self._last_check_time > self._cache_ttl:
            self.last_check = self.monitor.check_all()
            self._last_check_time = now

        if not self.last_check.get("healthy", True):
            return False

        if require_gpu:
            gpu = self.last_check.get("checks", {}).get("gpu", {})
            if not gpu.get("healthy", False):
                return False

        return True

    def require(self, require_gpu: bool = False) -> None:
        """Raise if resources are insufficient."""
        if not self.can_proceed(require_gpu=require_gpu):
            raise ResourcesExhausted(
                f"System resources critically low. Warnings: {self.last_check.get('warnings', [])}"
            )


class ResourcesExhausted(Exception):
    """Raised when system resources are too low for operation."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Service Watchdog
# ═══════════════════════════════════════════════════════════════════════════════
# Modified: 2026-02-08T15:00:00Z | Author: COPILOT | Change: Add service watchdog for auto-restart of failed services

class ServiceWatchdog:
    """Periodically checks registered services and restarts them if unhealthy.

    Usage:
        watchdog = ServiceWatchdog()
        watchdog.register("ollama", check_fn=check_ollama, restart_fn=restart_ollama)
        watchdog.start(interval=30)  # Check every 30s
    """

    def __init__(self):
        self._services: dict[str, dict] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        check_fn: Callable[[], bool],
        restart_fn: Callable[[], bool],
        max_restarts: int = 3,
        cooldown: float = 60.0,
    ) -> None:
        """Register a service for monitoring."""
        self._services[name] = {
            "check_fn": check_fn,
            "restart_fn": restart_fn,
            "max_restarts": max_restarts,
            "cooldown": cooldown,
            "restart_count": 0,
            "last_restart": 0.0,
            "healthy": True,
        }
        log.info(f"Watchdog registered service: {name}")

    def check_service(self, name: str) -> bool:
        """Check a single service and restart if needed."""
        svc = self._services.get(name)
        if not svc:
            return False

        try:
            healthy = svc["check_fn"]()
            svc["healthy"] = healthy
            if healthy:
                svc["restart_count"] = 0
                return True
        except Exception as e:
            log.warning(f"Watchdog check failed for {name}: {e}")
            svc["healthy"] = False

        # Service unhealthy — attempt restart
        now = time.time()
        if svc["restart_count"] >= svc["max_restarts"]:
            log.error(f"Watchdog: {name} exceeded max restarts ({svc['max_restarts']})")
            return False
        if now - svc["last_restart"] < svc["cooldown"]:
            log.warning(f"Watchdog: {name} in cooldown, skipping restart")
            return False

        log.warning(f"Watchdog restarting {name} (attempt {svc['restart_count'] + 1}/{svc['max_restarts']})")
        try:
            success = svc["restart_fn"]()
            svc["restart_count"] += 1
            svc["last_restart"] = now
            svc["healthy"] = success
            return success
        except Exception as e:
            log.error(f"Watchdog restart failed for {name}: {e}")
            svc["restart_count"] += 1
            svc["last_restart"] = now
            return False

    def check_all(self) -> dict[str, bool]:
        """Check all registered services."""
        results = {}
        for name in self._services:
            results[name] = self.check_service(name)
        return results

    def start(self, interval: float = 30.0) -> None:
        """Start background watchdog thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, args=(interval,), daemon=True, name="slate-watchdog"
        )
        self._thread.start()
        log.info(f"Watchdog started (interval={interval}s)")

    def stop(self) -> None:
        """Stop background watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Watchdog stopped")

    def _run_loop(self, interval: float) -> None:
        while self._running:
            self.check_all()
            time.sleep(interval)

    def status(self) -> dict:
        return {
            "running": self._running,
            "services": {
                name: {
                    "healthy": svc["healthy"],
                    "restart_count": svc["restart_count"],
                    "max_restarts": svc["max_restarts"],
                }
                for name, svc in self._services.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SORT-AI Drift Mitigation
# ═══════════════════════════════════════════════════════════════════════════════
# Modified: 2026-02-08T15:00:00Z | Author: COPILOT | Change: Add SORT-AI drift mitigation for planning consistency

class DriftMitigator:
    """Detect and mitigate planning drift in extended AI orchestration contexts.

    Inspired by SORT-AI: prevents contradictory decisions, redundant information,
    and context degradation in long-running autonomous loops.

    Usage:
        drift = DriftMitigator(window_size=100)
        drift.record_decision("assign_gpu", {"model": "coder", "gpu": 0})
        drift.record_decision("assign_gpu", {"model": "coder", "gpu": 1})  # Contradiction!
        conflicts = drift.check_drift()
        # [{"type": "contradiction", "key": "assign_gpu", ...}]
    """

    def __init__(self, window_size: int = 100, similarity_threshold: float = 0.85):
        self._decisions: list[dict] = []
        self._window_size = window_size
        self._similarity_threshold = similarity_threshold
        self._lock = threading.Lock()

    def record_decision(self, key: str, params: dict, rationale: str = "") -> None:
        """Record a decision made by the orchestration system."""
        entry = {
            "key": key,
            "params": params,
            "rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hash": self._hash_params(key, params),
        }
        with self._lock:
            self._decisions.append(entry)
            # Trim to window
            if len(self._decisions) > self._window_size:
                self._decisions = self._decisions[-self._window_size:]

    def check_drift(self) -> list[dict]:
        """Analyze recent decisions for contradictions and redundancies.

        Returns list of drift issues found:
            - contradiction: Same key, different params (conflicting decisions)
            - redundancy: Exact duplicate decisions within window
            - oscillation: Key flip-flopping between values (>3 reversals)
        """
        issues = []
        with self._lock:
            decisions = list(self._decisions)

        # Group by key
        by_key: dict[str, list[dict]] = collections.defaultdict(list)
        for d in decisions:
            by_key[d["key"]].append(d)

        for key, entries in by_key.items():
            if len(entries) < 2:
                continue

            # Check for redundancy (same hash appearing multiple times)
            hashes = [e["hash"] for e in entries]
            hash_counts = collections.Counter(hashes)
            for h, count in hash_counts.items():
                if count > 2:
                    issues.append({
                        "type": "redundancy",
                        "key": key,
                        "occurrences": count,
                        "message": f"Decision '{key}' repeated {count} times with same params",
                    })

            # Check for contradictions (recent entries differ from each other)
            recent = entries[-5:]  # Last 5 entries for this key
            unique_hashes = set(e["hash"] for e in recent)
            if len(unique_hashes) > 1 and len(recent) >= 2:
                issues.append({
                    "type": "contradiction",
                    "key": key,
                    "values": [e["params"] for e in recent],
                    "message": f"Decision '{key}' has conflicting param values in recent history",
                })

            # Check for oscillation (values flip-flopping)
            if len(entries) >= 4:
                reversals = 0
                for i in range(2, len(entries)):
                    if entries[i]["hash"] == entries[i - 2]["hash"] and entries[i]["hash"] != entries[i - 1]["hash"]:
                        reversals += 1
                if reversals >= 3:
                    issues.append({
                        "type": "oscillation",
                        "key": key,
                        "reversals": reversals,
                        "message": f"Decision '{key}' is oscillating between values ({reversals} reversals)",
                    })

        return issues

    def get_stable_decision(self, key: str) -> Optional[dict]:
        """Return the most frequently chosen params for a key (majority vote)."""
        with self._lock:
            entries = [d for d in self._decisions if d["key"] == key]
        if not entries:
            return None

        hash_counts = collections.Counter(e["hash"] for e in entries)
        most_common_hash = hash_counts.most_common(1)[0][0]
        for e in reversed(entries):
            if e["hash"] == most_common_hash:
                return e["params"]
        return entries[-1]["params"]

    def clear(self) -> None:
        """Clear decision history."""
        with self._lock:
            self._decisions.clear()

    @staticmethod
    def _hash_params(key: str, params: dict) -> str:
        """Create stable hash for key+params."""
        data = json.dumps({"key": key, "params": params}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def status(self) -> dict:
        with self._lock:
            return {
                "decisions_recorded": len(self._decisions),
                "window_size": self._window_size,
                "unique_keys": len(set(d["key"] for d in self._decisions)),
                "drift_issues": self.check_drift(),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Orchestrator (RL-based Agent Priority)
# ═══════════════════════════════════════════════════════════════════════════════
# Modified: 2026-02-08T15:00:00Z | Author: COPILOT | Change: Add adaptive orchestration with bandit-style agent selection

class AdaptiveOrchestrator:
    """Reinforcement learning-inspired agent selection and task prioritization.

    Uses Upper Confidence Bound (UCB1) algorithm to balance exploration vs exploitation
    when selecting agents for tasks. Agents that perform well get more tasks;
    underperformers get fewer but still occasionally get chances to improve.

    Usage:
        orch = AdaptiveOrchestrator()
        orch.record_outcome("ALPHA", success=True, latency=1.2)
        orch.record_outcome("BETA", success=False, latency=10.0)
        best = orch.select_agent(["ALPHA", "BETA", "GAMMA"])  # Likely "ALPHA"
    """

    def __init__(self, exploration_weight: float = 1.41):
        self._stats: dict[str, dict] = {}  # agent -> {successes, attempts, total_latency}
        self._exploration_weight = exploration_weight
        self._lock = threading.Lock()
        self._total_selections = 0

    def record_outcome(
        self, agent: str, success: bool, latency: float = 0.0, reward: float = 0.0
    ) -> None:
        """Record the outcome of an agent's task execution."""
        with self._lock:
            if agent not in self._stats:
                self._stats[agent] = {
                    "successes": 0, "attempts": 0, "total_latency": 0.0,
                    "total_reward": 0.0, "history": [],
                }
            stats = self._stats[agent]
            stats["attempts"] += 1
            stats["total_latency"] += latency
            if success:
                stats["successes"] += 1
            computed_reward = reward if reward else (1.0 if success else 0.0) / max(latency, 0.1)
            stats["total_reward"] += computed_reward
            stats["history"].append({
                "success": success, "latency": latency, "reward": computed_reward,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Keep history bounded
            if len(stats["history"]) > 50:
                stats["history"] = stats["history"][-50:]

    def select_agent(self, candidates: list[str]) -> str:
        """Select the best agent using UCB1 algorithm.

        Balances exploitation (pick best performer) with exploration (try less-used agents).
        """
        import math

        with self._lock:
            self._total_selections += 1
            total = self._total_selections

            best_agent = candidates[0]
            best_score = -float("inf")

            for agent in candidates:
                stats = self._stats.get(agent)
                if not stats or stats["attempts"] == 0:
                    # Unvisited agent — always explore first
                    return agent

                avg_reward = stats["total_reward"] / stats["attempts"]
                exploration_bonus = self._exploration_weight * math.sqrt(
                    math.log(total) / stats["attempts"]
                )
                ucb_score = avg_reward + exploration_bonus

                if ucb_score > best_score:
                    best_score = ucb_score
                    best_agent = agent

            return best_agent

    def get_rankings(self) -> list[dict]:
        """Return agents ranked by success rate."""
        with self._lock:
            rankings = []
            for agent, stats in self._stats.items():
                rate = stats["successes"] / max(stats["attempts"], 1)
                avg_latency = stats["total_latency"] / max(stats["attempts"], 1)
                rankings.append({
                    "agent": agent,
                    "success_rate": round(rate, 3),
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "avg_latency": round(avg_latency, 2),
                })
            rankings.sort(key=lambda x: x["success_rate"], reverse=True)
            return rankings

    def status(self) -> dict:
        return {
            "total_selections": self._total_selections,
            "agents": len(self._stats),
            "rankings": self.get_rankings(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Availability Tracker
# ═══════════════════════════════════════════════════════════════════════════════
# Modified: 2026-02-08T15:00:00Z | Author: COPILOT | Change: Add availability tracker — 99.9% target with rolling window

class AvailabilityTracker:
    """Track system availability with a rolling window.

    Measures: availability = 1 - (downtime_checks / total_checks)
    Target: 99.9% (configurable)

    Usage:
        tracker = AvailabilityTracker(target=0.999, window_minutes=1440)
        tracker.record_check(healthy=True)
        tracker.record_check(healthy=False)
        print(tracker.availability)  # 0.5
        print(tracker.meets_target)  # False
    """

    def __init__(self, target: float = 0.999, window_minutes: int = 1440):
        self._target = target
        self._window_seconds = window_minutes * 60
        self._checks: list[tuple[float, bool]] = []  # (timestamp, healthy)
        self._lock = threading.Lock()
        self._state_file = WORKSPACE / ".slate_availability.json"

    @property
    def availability(self) -> float:
        """Calculate current availability within the rolling window."""
        self._trim_window()
        with self._lock:
            if not self._checks:
                return 1.0
            healthy = sum(1 for _, h in self._checks if h)
            return healthy / len(self._checks)

    @property
    def meets_target(self) -> bool:
        return self.availability >= self._target

    def record_check(self, healthy: bool) -> None:
        """Record a health check result."""
        with self._lock:
            self._checks.append((time.time(), healthy))

    def _trim_window(self) -> None:
        """Remove checks older than the rolling window."""
        cutoff = time.time() - self._window_seconds
        with self._lock:
            self._checks = [(t, h) for t, h in self._checks if t >= cutoff]

    def get_downtime_minutes(self) -> float:
        """Estimate downtime in minutes based on failed checks."""
        self._trim_window()
        with self._lock:
            if not self._checks:
                return 0.0
            failed = sum(1 for _, h in self._checks if not h)
            total = len(self._checks)
            if total < 2:
                return 0.0
            # Estimate interval between checks
            intervals = [
                self._checks[i][0] - self._checks[i - 1][0]
                for i in range(1, len(self._checks))
            ]
            avg_interval = sum(intervals) / len(intervals) if intervals else 60.0
            return (failed * avg_interval) / 60.0

    def save(self) -> None:
        """Persist availability data to disk."""
        try:
            data = {
                "target": self._target,
                "window_seconds": self._window_seconds,
                "checks": self._checks[-500:],  # Last 500 checks
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self._state_file.write_text(json.dumps(data, default=str), encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to save availability data: {e}")

    def load(self) -> None:
        """Load persisted availability data."""
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                self._checks = [(c[0], c[1]) for c in data.get("checks", [])]
                self._trim_window()
        except Exception as e:
            log.warning(f"Failed to load availability data: {e}")

    def status(self) -> dict:
        self._trim_window()
        with self._lock:
            total = len(self._checks)
            healthy = sum(1 for _, h in self._checks if h)
            failed = total - healthy
        return {
            "availability": round(self.availability, 6),
            "target": self._target,
            "meets_target": self.meets_target,
            "total_checks": total,
            "healthy_checks": healthy,
            "failed_checks": failed,
            "downtime_minutes": round(self.get_downtime_minutes(), 1),
            "window_minutes": self._window_seconds // 60,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Planning Context Deduplicator
# ═══════════════════════════════════════════════════════════════════════════════
# Modified: 2026-02-08T15:00:00Z | Author: COPILOT | Change: Add context dedup to prevent extended context bloat

class PlanningContextDeduplicator:
    """Prevents redundant information accumulation in extended AI planning contexts.

    As autonomous loops run, they accumulate context that can become contradictory
    or redundant. This deduplicator tracks facts and decisions, merging duplicates
    and flagging stale entries.

    Usage:
        dedup = PlanningContextDeduplicator()
        dedup.add_fact("gpu_0_model", "slate-coder")
        dedup.add_fact("gpu_0_model", "slate-coder")  # Deduped
        dedup.add_fact("gpu_0_model", "slate-planner")  # Supersedes
        context = dedup.get_context()  # Only latest facts
    """

    def __init__(self, max_facts: int = 500, stale_minutes: int = 240):
        self._facts: dict[str, dict] = {}  # key -> {value, timestamp, count}
        self._max_facts = max_facts
        self._stale_seconds = stale_minutes * 60
        self._lock = threading.Lock()

    def add_fact(self, key: str, value: Any, source: str = "") -> bool:
        """Add or update a fact. Returns True if new, False if deduped."""
        with self._lock:
            existing = self._facts.get(key)
            now = time.time()

            if existing and existing["value"] == value:
                # Exact duplicate — increment count, update timestamp
                existing["count"] += 1
                existing["last_seen"] = now
                return False

            # New or superseded fact
            self._facts[key] = {
                "value": value,
                "source": source,
                "first_seen": now if not existing else existing.get("first_seen", now),
                "last_seen": now,
                "count": 1,
                "superseded": existing["value"] if existing else None,
            }

            # Trim if over limit
            if len(self._facts) > self._max_facts:
                self._evict_oldest()

            return True

    def get_fact(self, key: str) -> Optional[Any]:
        """Get current value for a fact key."""
        entry = self._facts.get(key)
        return entry["value"] if entry else None

    def get_context(self, include_stale: bool = False) -> dict[str, Any]:
        """Get deduplicated context as a clean dict."""
        cutoff = time.time() - self._stale_seconds
        with self._lock:
            if include_stale:
                return {k: v["value"] for k, v in self._facts.items()}
            return {
                k: v["value"]
                for k, v in self._facts.items()
                if v["last_seen"] >= cutoff
            }

    def get_stale_keys(self) -> list[str]:
        """Return keys that haven't been updated within the stale window."""
        cutoff = time.time() - self._stale_seconds
        with self._lock:
            return [k for k, v in self._facts.items() if v["last_seen"] < cutoff]

    def clean_stale(self) -> int:
        """Remove stale facts. Returns count removed."""
        cutoff = time.time() - self._stale_seconds
        with self._lock:
            stale = [k for k, v in self._facts.items() if v["last_seen"] < cutoff]
            for k in stale:
                del self._facts[k]
            return len(stale)

    def _evict_oldest(self) -> None:
        """Evict oldest facts when over capacity."""
        sorted_keys = sorted(
            self._facts.keys(), key=lambda k: self._facts[k]["last_seen"]
        )
        to_remove = len(self._facts) - self._max_facts
        for k in sorted_keys[:to_remove]:
            del self._facts[k]

    def status(self) -> dict:
        with self._lock:
            return {
                "total_facts": len(self._facts),
                "max_facts": self._max_facts,
                "stale_count": len(self.get_stale_keys()),
                "stale_window_minutes": self._stale_seconds // 60,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="SLATE Stability & Health Monitor")
    parser.add_argument("--status", action="store_true", help="Show health status")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--guard", action="store_true", help="Check resource guard")
    parser.add_argument("--availability", action="store_true", help="Show availability stats")
    parser.add_argument("--drift", action="store_true", help="Show drift mitigation status")
    parser.add_argument("--agents", action="store_true", help="Show adaptive agent rankings")
    parser.add_argument("--full", action="store_true", help="Full stability report")
    args = parser.parse_args()

    monitor = HealthMonitor()

    if args.json:
        report = {
            "health": monitor.check_all(),
            "availability": AvailabilityTracker().status(),
            "drift": DriftMitigator().status(),
            "orchestrator": AdaptiveOrchestrator().status(),
        }
        print(json.dumps(report, indent=2, default=str))
    elif args.guard:
        guard = ResourceGuard()
        ok = guard.can_proceed(require_gpu=True)
        print(f"Resource guard: {'PASS' if ok else 'BLOCKED'}")
        if not ok:
            for w in guard.last_check.get("warnings", []):
                print(f"  ⚠ {w}")
    elif args.availability:
        tracker = AvailabilityTracker()
        tracker.load()
        tracker.record_check(healthy=True)
        tracker.save()
        status = tracker.status()
        print()
        print("=" * 60)
        print("  SLATE Availability Tracker")
        print("=" * 60)
        icon = "✓" if status["meets_target"] else "✗"
        print(f"  [{icon}] Availability: {status['availability'] * 100:.3f}%")
        print(f"      Target:       {status['target'] * 100:.1f}%")
        print(f"      Total checks: {status['total_checks']}")
        print(f"      Failed:       {status['failed_checks']}")
        print(f"      Downtime:     {status['downtime_minutes']:.1f} min")
        print(f"      Window:       {status['window_minutes']} min")
        print("=" * 60)
    elif args.drift:
        drift = DriftMitigator()
        status = drift.status()
        print(json.dumps(status, indent=2, default=str))
    elif args.agents:
        orch = AdaptiveOrchestrator()
        print(json.dumps(orch.status(), indent=2, default=str))
    elif args.full:
        # Full stability report
        monitor.print_status()
        tracker = AvailabilityTracker()
        tracker.load()
        status = tracker.status()
        print(f"\n  Availability: {status['availability'] * 100:.3f}% (target: {status['target'] * 100:.1f}%)")
        print(f"  Downtime:     {status['downtime_minutes']:.1f} min in last {status['window_minutes']} min")
    else:
        monitor.print_status()


if __name__ == "__main__":
    main()
