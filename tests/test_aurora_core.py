#!/usr/bin/env python3
# Modified: 2026-02-06T11:20:00Z | Author: COPILOT | Change: Added test for AI toolkit integration
"""Tests for aurora_core package."""

import pytest
import sys

def test_import_aurora_core():
    import aurora_core
    assert aurora_core.__version__ == "2.4.0"

def test_version_format():
    import aurora_core
    parts = aurora_core.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)

def test_slate_status_import():
    from aurora_core import slate_status
    assert hasattr(slate_status, 'get_status')

def test_slate_benchmark_import():
    from aurora_core import slate_benchmark
    assert hasattr(slate_benchmark, 'run_benchmarks')

def test_slate_hardware_optimizer_import():
    from aurora_core import slate_hardware_optimizer
    assert hasattr(slate_hardware_optimizer, 'get_gpu_list')

def test_slate_runtime_import():
    from aurora_core import slate_runtime
    assert hasattr(slate_runtime, 'check_all')

def test_slate_terminal_monitor_import():
    from aurora_core import slate_terminal_monitor
    assert hasattr(slate_terminal_monitor, 'is_blocked')

def test_slate_protocols_import():
    from aurora_core import slate_protocols
    assert hasattr(slate_protocols, 'main')

def test_slate_ai_toolkit_integration_import():
    from aurora_core import slate_ai_toolkit_integration
    assert hasattr(slate_ai_toolkit_integration, 'main')

def test_blocked_commands():
    from aurora_core.slate_terminal_monitor import is_blocked
    assert is_blocked("curl.exe https://example.com")[0] == True
    assert is_blocked("Start-Sleep -Seconds 10")[0] == True
    assert is_blocked("python --version")[0] == False
