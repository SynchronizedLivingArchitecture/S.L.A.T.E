#!/usr/bin/env python3
# Modified: 2026-02-08T07:25:00Z | Author: COPILOT | Change: Create stability module tests — retry, circuit breaker, health monitor, resource guard
"""Tests for slate.stability module."""

import time
import pytest
from unittest.mock import patch, MagicMock
from slate.stability import (
    retry_with_backoff,
    retry,
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    HealthMonitor,
    ResourceGuard,
    ResourceThresholds,
    ResourcesExhausted,
)


# ═══════════════════════════════════════════════════════════════════════════════
# retry_with_backoff
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    def test_success_first_try(self):
        """Function succeeds on first call — no retries."""
        result = retry_with_backoff(lambda: 42, max_attempts=3, base_delay=0.01)
        assert result == 42

    def test_success_after_retries(self):
        """Function succeeds after initial failures."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        result = retry_with_backoff(flaky, max_attempts=5, base_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_all_attempts_exhausted(self):
        """All retries fail — last exception raised."""
        def always_fails():
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            retry_with_backoff(always_fails, max_attempts=3, base_delay=0.01)

    def test_non_retryable_exception(self):
        """Non-retryable exceptions propagate immediately."""
        call_count = 0

        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            retry_with_backoff(
                raises_type_error, max_attempts=5, base_delay=0.01,
                retryable_exceptions=(ValueError,),
            )
        assert call_count == 1  # Only one attempt — no retry

    def test_on_retry_callback(self):
        """on_retry callback is called before each retry."""
        retries = []

        def on_retry(attempt, error):
            retries.append((attempt, str(error)))

        call_count = 0

        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"fail {call_count}")
            return "done"

        retry_with_backoff(fail_twice, max_attempts=5, base_delay=0.01, on_retry=on_retry)
        assert len(retries) == 2
        assert retries[0][0] == 1
        assert retries[1][0] == 2

    def test_backoff_delay(self):
        """Verify exponential backoff delay."""
        start = time.time()
        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retry")
            return "ok"

        retry_with_backoff(fail_then_succeed, max_attempts=3, base_delay=0.05, backoff_factor=2.0)
        elapsed = time.time() - start
        # 0.05s (1st retry) + 0.1s (2nd retry) ≈ 0.15s minimum
        assert elapsed >= 0.1


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_decorator_usage(self):
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("not ready")
            return "done"

        result = sometimes_fails()
        assert result == "done"
        assert call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# CircuitBreaker
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_success_keeps_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=10)

        def fail():
            raise RuntimeError("down")

        # First failure
        with pytest.raises(RuntimeError):
            cb.call(fail)
        assert cb.state == CircuitState.CLOSED

        # Second failure → opens
        with pytest.raises(RuntimeError):
            cb.call(fail)
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=100)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "should not run")

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_recovers_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        time.sleep(0.15)
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_manual_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=100)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_status_dict(self):
        cb = CircuitBreaker("ollama", failure_threshold=3, recovery_timeout=30)
        status = cb.status()
        assert status["name"] == "ollama"
        assert status["state"] == "closed"
        assert status["threshold"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# HealthMonitor
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_check_all_returns_structure(self):
        monitor = HealthMonitor()
        status = monitor.check_all()
        assert "healthy" in status
        assert "timestamp" in status
        assert "checks" in status
        assert "warnings" in status
        assert isinstance(status["checks"], dict)

    def test_custom_thresholds(self):
        thresholds = ResourceThresholds(cpu_percent=50.0, memory_percent=50.0)
        monitor = HealthMonitor(thresholds=thresholds)
        assert monitor.thresholds.cpu_percent == 50.0
        assert monitor.thresholds.memory_percent == 50.0

    def test_register_service(self):
        monitor = HealthMonitor()
        cb = monitor.register_service("ollama", failure_threshold=5)
        assert isinstance(cb, CircuitBreaker)
        assert cb.name == "ollama"

    @patch("slate.stability.subprocess.run")
    def test_check_gpu_healthy(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA RTX 5070 Ti, 2048, 16384, 45, 10\n1, NVIDIA RTX 5070 Ti, 1024, 16384, 42, 5\n",
        )
        monitor = HealthMonitor()
        result = monitor.check_gpu()
        assert result["healthy"] is True
        assert result["gpu_count"] == 2
        assert result["gpus"][0]["name"] == "NVIDIA RTX 5070 Ti"

    @patch("slate.stability.subprocess.run")
    def test_check_gpu_high_temp(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA RTX 5070 Ti, 2048, 16384, 90, 80\n",
        )
        monitor = HealthMonitor()
        result = monitor.check_gpu()
        assert result["healthy"] is False
        assert "temp" in result["warning"].lower()

    def test_check_ollama_offline(self):
        """Ollama check returns unhealthy when service is unreachable."""
        monitor = HealthMonitor()
        # This will likely fail unless Ollama is running on port 11434
        result = monitor.check_ollama()
        # Just verify the structure
        assert "healthy" in result
        assert "running" in result


# ═══════════════════════════════════════════════════════════════════════════════
# ResourceGuard
# ═══════════════════════════════════════════════════════════════════════════════

class TestResourceGuard:
    """Tests for ResourceGuard class."""

    def test_can_proceed_default(self):
        guard = ResourceGuard()
        # Should generally pass unless system is actually overloaded
        result = guard.can_proceed()
        assert isinstance(result, bool)

    def test_require_raises_on_exhaust(self):
        guard = ResourceGuard()
        # Force a failed check
        guard.last_check = {"healthy": False, "warnings": ["test failure"]}
        guard._last_check_time = time.time()  # Mark as fresh

        with pytest.raises(ResourcesExhausted):
            guard.require()

    def test_cache_ttl(self):
        guard = ResourceGuard()
        guard._cache_ttl = 5.0  # Long TTL to ensure cache hit

        # First check
        guard.can_proceed()
        first_check_time = guard._last_check_time
        assert first_check_time > 0

        # Immediate second check — should use cache (TTL not expired)
        guard.can_proceed()
        assert guard._last_check_time == first_check_time

        # Force cache expiry by backdating timestamp
        guard._last_check_time = time.time() - 10.0
        old_time = guard._last_check_time
        guard.can_proceed()
        assert guard._last_check_time > old_time


# ═══════════════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestStabilityIntegration:
    """Integration tests combining stability primitives."""

    def test_retry_with_circuit_breaker(self):
        """Retry wrapping a circuit-broken service."""
        cb = CircuitBreaker("test-svc", failure_threshold=5, recovery_timeout=0.1)
        call_count = 0

        def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "success"

        result = retry_with_backoff(
            lambda: cb.call(flaky_service),
            max_attempts=5,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_monitor_with_registered_services(self):
        """Monitor tracks registered circuit breakers."""
        monitor = HealthMonitor()
        cb1 = monitor.register_service("ollama", failure_threshold=3)
        cb2 = monitor.register_service("chromadb", failure_threshold=3)

        services = monitor.check_services()
        assert "ollama" in services
        assert "chromadb" in services
        assert services["ollama"]["state"] == "closed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
