#!/usr/bin/env python3
# Modified: 2026-02-07T00:30:00Z | Author: COPILOT | Change: Add hardware, benchmark, runtime tools; fix AI tool
"""
SLATE MCP Server - Model Context Protocol server for Claude Code / Copilot integration.

Provides tools for:
- System status checking
- Runtime integration checks
- Hardware & GPU optimization
- Workflow management
- Orchestrator control
- Runner management
- Benchmark execution
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# MCP Protocol imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    # Fallback for when mcp package not installed
    print("MCP package not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)


# Initialize MCP server
server = Server("slate-mcp")

# Python executable in venv
PYTHON = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = WORKSPACE_ROOT / ".venv" / "bin" / "python"


def run_slate_command(module: str, *args: str) -> dict[str, Any]:
    """Run a SLATE command and return the result."""
    cmd = [str(PYTHON), f"slate/{module}"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Command timed out after 60 seconds",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available SLATE tools."""
    return [
        Tool(
            name="slate_status",
            description="Check the status of all SLATE services and system components (GPU, services, workflows)",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["quick", "json", "full"],
                        "description": "Output format: quick (human readable), json (machine readable), full (detailed)",
                        "default": "quick"
                    }
                }
            }
        ),
        Tool(
            name="slate_workflow",
            description="Manage the SLATE task workflow queue - view status, cleanup stale tasks, enforce rules",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "cleanup", "enforce"],
                        "description": "Action to perform: status (view queue), cleanup (fix stale tasks), enforce (check rules)",
                        "default": "status"
                    }
                }
            }
        ),
        Tool(
            name="slate_orchestrator",
            description="Control the SLATE orchestrator - start, stop, or check status of all services",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "status"],
                        "description": "Action to perform on the orchestrator",
                        "default": "status"
                    }
                },
                "required": ["action"]
            }
        ),
        Tool(
            name="slate_runner",
            description="Manage the GitHub Actions self-hosted runner - check status, setup, or dispatch workflows",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "setup", "dispatch"],
                        "description": "Action to perform on the runner",
                        "default": "status"
                    },
                    "workflow": {
                        "type": "string",
                        "description": "Workflow file to dispatch (only used with dispatch action)",
                        "default": "ci.yml"
                    }
                }
            }
        ),
        Tool(
            name="slate_ai",
            description="Execute AI tasks using SLATE's unified backend (routes to free local LLMs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The AI task to execute"
                    },
                    "check_status": {
                        "type": "boolean",
                        "description": "Check AI backend status instead of executing task",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="slate_runtime",
            description="Check all SLATE runtime integrations and dependencies (Python, GPU, PyTorch, Ollama, venv)",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["text", "json"],
                        "description": "Output format",
                        "default": "text"
                    }
                }
            }
        ),
        Tool(
            name="slate_hardware",
            description="Detect GPUs and optimize hardware configuration for PyTorch/CUDA",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["detect", "optimize", "install-pytorch"],
                        "description": "detect: show GPU info, optimize: apply settings, install-pytorch: install correct PyTorch for GPU arch",
                        "default": "detect"
                    }
                }
            }
        ),
        Tool(
            name="slate_benchmark",
            description="Run SLATE performance benchmarks (GPU, inference, system throughput)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a SLATE tool."""

    if name == "slate_status":
        fmt = arguments.get("format", "quick")
        flag = f"--{fmt}"
        result = run_slate_command("slate_status.py", flag)

    elif name == "slate_workflow":
        action = arguments.get("action", "status")
        flag = f"--{action}"
        result = run_slate_command("slate_workflow_manager.py", flag)

    elif name == "slate_orchestrator":
        action = arguments.get("action", "status")
        result = run_slate_command("slate_orchestrator.py", action)

    elif name == "slate_runner":
        action = arguments.get("action", "status")
        if action == "dispatch":
            workflow = arguments.get("workflow", "ci.yml")
            result = run_slate_command("slate_runner_manager.py", "--dispatch", workflow)
        else:
            flag = f"--{action}"
            result = run_slate_command("slate_runner_manager.py", flag)

    elif name == "slate_ai":
        if arguments.get("check_status", False):
            result = run_slate_command("unified_ai_backend.py", "--status")
        else:
            task = arguments.get("task", "")
            if not task:
                result = {
                    "success": False,
                    "stdout": "",
                    "stderr": "No task provided",
                    "returncode": 1
                }
            else:
                result = run_slate_command("unified_ai_backend.py", "--task", task)

    elif name == "slate_runtime":
        fmt = arguments.get("format", "text")
        if fmt == "json":
            result = run_slate_command("slate_runtime.py", "--check-all", "--json")
        else:
            result = run_slate_command("slate_runtime.py", "--check-all")

    elif name == "slate_hardware":
        action = arguments.get("action", "detect")
        if action == "optimize":
            result = run_slate_command("slate_hardware_optimizer.py", "--optimize")
        elif action == "install-pytorch":
            result = run_slate_command("slate_hardware_optimizer.py", "--install-pytorch")
        else:
            result = run_slate_command("slate_hardware_optimizer.py")

    elif name == "slate_benchmark":
        result = run_slate_command("slate_benchmark.py")

    else:
        result = {
            "success": False,
            "stdout": "",
            "stderr": f"Unknown tool: {name}",
            "returncode": 1
        }

    # Format output
    output = result["stdout"] if result["success"] else f"Error: {result['stderr']}"
    if not output.strip():
        output = "Command completed with no output" if result["success"] else "Command failed with no output"

    return [TextContent(type="text", text=output)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
