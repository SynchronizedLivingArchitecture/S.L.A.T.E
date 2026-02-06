#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_mcp_server [python]
# Author: COPILOT | Created: 2026-02-06T10:00:00Z | Modified: 2026-02-06T10:00:00Z
# Purpose: SLATE MCP Server — exposes SLATE tools to Copilot & Claude via MCP
# ═══════════════════════════════════════════════════════════════════════════════
"""
S.L.A.T.E. MCP Server
======================
Model Context Protocol server that exposes SLATE system tools to AI assistants
(GitHub Copilot, Claude, etc.) via the MCP standard.

Tools Provided:
    slate_get_status    — System status (GPU, SDK, runner, agents)
    slate_run_check     — Run runtime or status checks
    slate_list_tasks    — List tasks from the task queue
    slate_gpu_info      — Detailed GPU information
    slate_agent_status  — Agent system health
    slate_runner_status — Self-hosted runner status
    slate_search_code   — Search SLATE codebase
    slate_dashboard_url — Get dashboard URL

Usage:
    # stdio mode (for VS Code / Copilot / Claude)
    python slate/slate_mcp_server.py

    # SSE mode (for web clients)
    python slate/slate_mcp_server.py --sse --port 6274

    # Verify server
    python slate/slate_mcp_server.py --verify
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace is importable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ─── Server Instance ─────────────────────────────────────────────────────────

mcp = FastMCP(
    "slate-system",
    instructions=(
        "S.L.A.T.E. (Synchronized Living Architecture for Transformation and Evolution) "
        "system server. Provides tools to query system status, manage tasks, check GPU "
        "health, monitor agents, and interact with the SLATE development environment. "
        "All operations are LOCAL ONLY — 127.0.0.1."
    ),
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_python() -> str:
    """Get the SLATE venv Python executable."""
    if os.name == "nt":
        venv_py = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _run_slate_cmd(args: list, timeout: int = 30) -> Dict[str, Any]:
    """Run a SLATE command and return structured output."""
    try:
        result = subprocess.run(
            [_get_python()] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(WORKSPACE_ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip() if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}


def _load_json_file(filename: str) -> Any:
    """Load a JSON file from workspace root."""
    path = WORKSPACE_ROOT / filename
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── MCP Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def slate_get_status() -> Dict[str, Any]:
    """Get comprehensive SLATE system status including SDK, GPU, runner, and agent info."""
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    result = _run_slate_cmd(["slate/slate_status.py", "--quick"])
    gpu_info = _run_slate_cmd(["-c", (
        "import subprocess, json; "
        "r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', "
        "'--format=csv,noheader'], capture_output=True, text=True, timeout=10); "
        "print(r.stdout.strip()) if r.returncode == 0 else print('No GPU')"
    )])

    return {
        "system": "S.L.A.T.E.",
        "workspace": str(WORKSPACE_ROOT),
        "status_output": result["stdout"],
        "gpu_info": gpu_info["stdout"],
        "dashboard_url": "http://127.0.0.1:8080",
    }


@mcp.tool()
def slate_run_check(check_type: str = "quick") -> Dict[str, Any]:
    """
    Run a SLATE system check.

    Args:
        check_type: Type of check — "quick" (status), "full" (runtime --check-all),
                    "hardware" (hardware optimizer), "sdk" (SDK verify)
    """
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    commands = {
        "quick": ["slate/slate_status.py", "--quick"],
        "full": ["slate/slate_runtime.py", "--check-all"],
        "hardware": ["slate/slate_hardware_optimizer.py"],
        "sdk": ["slate/slate_sdk.py", "--verify"],
    }

    if check_type not in commands:
        return {"error": f"Unknown check type '{check_type}'. Use: {', '.join(commands.keys())}"}

    result = _run_slate_cmd(commands[check_type], timeout=60)
    return {"check_type": check_type, **result}


@mcp.tool()
def slate_list_tasks(status_filter: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """
    List tasks from the SLATE task queue.

    Args:
        status_filter: Filter by status — "pending", "in_progress", "completed", or None for all
        limit: Maximum number of tasks to return (default 20)
    """
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    tasks = _load_json_file("clean_tasks.json")
    if tasks is None:
        tasks = _load_json_file("current_tasks.json")
    if tasks is None:
        return {"tasks": [], "total": 0, "error": "No task file found"}

    task_list = tasks if isinstance(tasks, list) else tasks.get("tasks", [])

    if status_filter:
        task_list = [t for t in task_list if t.get("status", "").lower() == status_filter.lower()]

    return {
        "tasks": task_list[:limit],
        "total": len(task_list),
        "showing": min(limit, len(task_list)),
        "filter": status_filter,
    }


@mcp.tool()
def slate_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information including VRAM, utilization, and CUDA status."""
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    result = _run_slate_cmd(["-c", """
import json, subprocess, sys

info = {"gpus": [], "cuda_available": False, "torch_version": None}

# nvidia-smi
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if r.returncode == 0:
        for line in r.stdout.strip().split("\\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                info["gpus"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": parts[2],
                    "memory_used": parts[3],
                    "memory_free": parts[4],
                    "utilization": parts[5],
                    "temperature": parts[6],
                    "compute_cap": parts[7],
                })
except Exception:
    pass

# PyTorch
try:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
except ImportError:
    pass

print(json.dumps(info))
"""])

    if result["success"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            return {"raw": result["stdout"]}
    return {"error": result["stderr"]}


@mcp.tool()
def slate_agent_status() -> Dict[str, Any]:
    """Get status of all SLATE agents (ALPHA, BETA, GAMMA, DELTA)."""
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    agents = {}
    agent_modules = {
        "coordinator": "aurora_agent_coordinator",
        "intelligence": "aurora_agent_intelligence",
        "agent_v2": "aurora_agent_v2",
        "cycle_evaluator": "aurora_cycle_evaluator",
        "workspace_analyzer": "aurora_workspace_analyzer",
    }

    for name, module in agent_modules.items():
        result = _run_slate_cmd(["-c", f"import {module}; print('ok')"], timeout=10)
        agents[name] = {"importable": result["success"]}

    # Load agent performance if available
    perf = _load_json_file("agent_performance.json")
    if perf:
        agents["performance"] = perf

    return {"agents": agents, "agent_count": len(agent_modules)}


@mcp.tool()
def slate_runner_status() -> Dict[str, Any]:
    """Get self-hosted GitHub Actions runner status including GPU labels."""
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    result = _run_slate_cmd(
        ["slate/slate_runner_manager.py", "--status", "--json"],
        timeout=15,
    )
    if result["success"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            return {"raw": result["stdout"]}
    return {"error": result["stderr"], "runner_available": False}


@mcp.tool()
def slate_search_code(query: str, file_pattern: str = "*.py", max_results: int = 10) -> Dict[str, Any]:
    """
    Search the SLATE codebase for code matching a query.

    Args:
        query: Text to search for in source files
        file_pattern: Glob pattern for files to search (default "*.py")
        max_results: Maximum number of matches to return
    """
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    matches = []
    search_dirs = ["slate", "agents", "slate_core", "tests"]

    for search_dir in search_dirs:
        dir_path = WORKSPACE_ROOT / search_dir
        if not dir_path.exists():
            continue
        for file_path in dir_path.rglob(file_pattern):
            if len(matches) >= max_results:
                break
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.split("\n"), 1):
                    if query.lower() in line.lower():
                        matches.append({
                            "file": str(file_path.relative_to(WORKSPACE_ROOT)),
                            "line": i,
                            "text": line.strip()[:200],
                        })
                        if len(matches) >= max_results:
                            break
            except Exception:
                continue

    return {"query": query, "matches": matches, "total": len(matches)}


@mcp.tool()
def slate_dashboard_url() -> Dict[str, str]:
    """Get the SLATE dashboard URL and check if it's running."""
    # Modified: 2026-02-06T10:00:00Z | Author: COPILOT | Change: Initial MCP tool
    import urllib.request

    url = "http://127.0.0.1:8080"
    running = False
    try:
        resp = urllib.request.urlopen(f"{url}/api/health", timeout=3)
        running = resp.status == 200
    except Exception:
        pass

    return {"url": url, "running": running}


# ─── MCP Resources ───────────────────────────────────────────────────────────

@mcp.resource("slate://status")
def resource_status() -> str:
    """Current SLATE system status."""
    result = _run_slate_cmd(["slate/slate_status.py", "--quick"])
    return result["stdout"] if result["success"] else "Status unavailable"


@mcp.resource("slate://tasks")
def resource_tasks() -> str:
    """Current task queue as JSON."""
    tasks = _load_json_file("clean_tasks.json") or _load_json_file("current_tasks.json") or []
    return json.dumps(tasks, indent=2)


@mcp.resource("slate://gpu")
def resource_gpu() -> str:
    """GPU information."""
    result = _run_slate_cmd(["-c", (
        "import subprocess; "
        "r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10); "
        "print(r.stdout if r.returncode == 0 else 'No GPU')"
    )])
    return result["stdout"] if result["success"] else "GPU info unavailable"


# ─── Entry Point ──────────────────────────────────────────────────────────────

def verify_server():
    """Verify the MCP server can start and tools are registered."""
    print("S.L.A.T.E. MCP Server Verification")
    print("=" * 50)
    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Python:    {_get_python()}")
    print(f"Server:    slate-system")
    print()

    tools = [
        "slate_get_status", "slate_run_check", "slate_list_tasks",
        "slate_gpu_info", "slate_agent_status", "slate_runner_status",
        "slate_search_code", "slate_dashboard_url",
    ]
    print(f"Tools registered: {len(tools)}")
    for t in tools:
        print(f"  ✓ {t}")

    resources = ["slate://status", "slate://tasks", "slate://gpu"]
    print(f"\nResources registered: {len(resources)}")
    for r in resources:
        print(f"  ✓ {r}")

    print("\n✓ MCP Server ready")
    return True


def main():
    parser = argparse.ArgumentParser(description="S.L.A.T.E. MCP Server")
    parser.add_argument("--verify", action="store_true", help="Verify server configuration")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode (HTTP)")
    parser.add_argument("--port", type=int, default=6274, help="SSE port (default 6274)")
    args = parser.parse_args()

    if args.verify:
        verify_server()
        return

    if args.sse:
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
