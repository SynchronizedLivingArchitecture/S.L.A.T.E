#!/usr/bin/env python3
# Modified: 2026-02-08T14:00:00Z | Author: COPILOT | Change: Create copilot-sdk bridge for SLATE agent integration
# Modified: 2026-02-09T03:30:00Z | Author: COPILOT | Change: Add functional SDK integration — tools, sessions, MCP, event bridge
"""
SLATE Copilot SDK Bridge
=========================
Integrates the GitHub Copilot SDK (vendor/copilot-sdk) into the SLATE agent
orchestration framework. Provides:

1. SDK version tracking and compatibility checks
2. Python SDK import bridge (adds vendor path to sys.path)
3. Agent-to-SDK tool mapping (SLATE agents ↔ Copilot SDK tools)
4. Upstream sync status monitoring
5. BYOK configuration for local-first operation
6. Functional tool registration (14 SLATE tools via define_tool)
7. Session management (create, resume, destroy with SLATE context)
8. MCP server configuration for SLATE local services
9. Custom agent definitions (Alpha/Beta/Gamma/Delta → SDK agents)
10. Event bridge between SDK sessions and SLATE event system

Architecture:
    SLATE Agents  →  CopilotSDKBridge  →  vendor/copilot-sdk/python
                                       →  copilot_sdk_tools.py (14 tools)
                                       →  copilot_sdk_session.py (session manager)
                                       →  vendor/copilot-sdk/nodejs (for extension)

Security:
    - LOCAL ONLY (127.0.0.1) — no external API calls without BYOK config
    - SDK Source Guard approved (GitHub is a trusted publisher)
    - ActionGuard integration — permission handler blocks dangerous operations
    - No eval/exec — uses importlib only
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Modified: 2026-02-08T14:00:00Z | Author: COPILOT | Change: Initial SDK bridge implementation
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SDK_SUBMODULE_PATH = WORKSPACE_ROOT / "vendor" / "copilot-sdk"
SDK_PYTHON_PATH = SDK_SUBMODULE_PATH / "python"
SDK_NODEJS_PATH = SDK_SUBMODULE_PATH / "nodejs"
SDK_PROTOCOL_VERSION_FILE = SDK_SUBMODULE_PATH / "sdk-protocol-version.json"

# SLATE-compatible SDK version constraints
SUPPORTED_PROTOCOL_VERSIONS = [2]  # Protocol versions we support
MIN_PYTHON_SDK_VERSION = "0.1.0"
MIN_NODEJS_SDK_VERSION = "0.1.0"


class CopilotSDKBridge:
    """
    Bridge between SLATE agent system and the GitHub Copilot SDK.
    
    Manages SDK lifecycle, version tracking, and provides import paths
    for both Python and Node.js SDK components.
    """

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or WORKSPACE_ROOT
        self.sdk_path = self.workspace_root / "vendor" / "copilot-sdk"
        self.python_sdk_path = self.sdk_path / "python"
        self.nodejs_sdk_path = self.sdk_path / "nodejs"
        self._sdk_available = None
        self._protocol_version = None
        self._python_version = None
        self._nodejs_version = None

    # ─── SDK Availability ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the copilot-sdk submodule is present and initialized."""
        if self._sdk_available is not None:
            return self._sdk_available
        self._sdk_available = (
            self.sdk_path.exists()
            and (self.sdk_path / "README.md").exists()
            and self.python_sdk_path.exists()
        )
        return self._sdk_available

    def get_protocol_version(self) -> Optional[int]:
        """Get the SDK protocol version from sdk-protocol-version.json."""
        if self._protocol_version is not None:
            return self._protocol_version
        version_file = self.sdk_path / "sdk-protocol-version.json"
        if not version_file.exists():
            return None
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._protocol_version = data.get("version")
            return self._protocol_version
        except (json.JSONDecodeError, OSError):
            return None

    def get_python_sdk_version(self) -> Optional[str]:
        """Get the Python SDK version from __init__.py."""
        if self._python_version is not None:
            return self._python_version
        init_file = self.python_sdk_path / "copilot" / "__init__.py"
        if not init_file.exists():
            return None
        try:
            with open(init_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        self._python_version = line.split('"')[1]
                        return self._python_version
        except (OSError, IndexError):
            pass
        return None

    def get_nodejs_sdk_version(self) -> Optional[str]:
        """Get the Node.js SDK version from package.json."""
        if self._nodejs_version is not None:
            return self._nodejs_version
        pkg_file = self.nodejs_sdk_path / "package.json"
        if not pkg_file.exists():
            return None
        try:
            with open(pkg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._nodejs_version = data.get("version")
            return self._nodejs_version
        except (json.JSONDecodeError, OSError):
            return None

    # ─── Compatibility Checks ─────────────────────────────────────────────

    def check_compatibility(self) -> dict[str, Any]:
        """Check if the SDK version is compatible with SLATE."""
        result: dict[str, Any] = {
            "compatible": False,
            "sdk_available": self.is_available(),
            "protocol_version": None,
            "python_version": None,
            "nodejs_version": None,
            "issues": [],
        }

        if not result["sdk_available"]:
            result["issues"].append("Copilot SDK submodule not found at vendor/copilot-sdk")
            return result

        # Protocol version check
        proto = self.get_protocol_version()
        result["protocol_version"] = proto
        if proto is None:
            result["issues"].append("Cannot read sdk-protocol-version.json")
        elif proto not in SUPPORTED_PROTOCOL_VERSIONS:
            result["issues"].append(
                f"Protocol version {proto} not supported (supported: {SUPPORTED_PROTOCOL_VERSIONS})"
            )

        # Python SDK
        py_ver = self.get_python_sdk_version()
        result["python_version"] = py_ver
        if py_ver is None:
            result["issues"].append("Cannot determine Python SDK version")

        # Node.js SDK
        node_ver = self.get_nodejs_sdk_version()
        result["nodejs_version"] = node_ver
        if node_ver is None:
            result["issues"].append("Cannot determine Node.js SDK version")

        result["compatible"] = len(result["issues"]) == 0
        return result

    # ─── Python SDK Import Bridge ─────────────────────────────────────────

    def ensure_python_path(self) -> bool:
        """Add the Python SDK to sys.path for import access."""
        if not self.is_available():
            return False
        sdk_str = str(self.python_sdk_path)
        if sdk_str not in sys.path:
            sys.path.insert(0, sdk_str)
        return True

    def import_copilot_client(self):
        """Import and return the CopilotClient class."""
        self.ensure_python_path()
        try:
            from copilot import CopilotClient
            return CopilotClient
        except ImportError as e:
            print(f"  ✗ Cannot import CopilotClient: {e}", file=sys.stderr)
            return None

    def import_copilot_tools(self):
        """Import and return the define_tool decorator."""
        self.ensure_python_path()
        try:
            from copilot import define_tool
            return define_tool
        except ImportError as e:
            print(f"  ✗ Cannot import define_tool: {e}", file=sys.stderr)
            return None

    def import_session_types(self):
        """Import and return key SDK session types."""
        self.ensure_python_path()
        try:
            from copilot.types import (
                SessionConfig, Tool, ToolResult, ToolInvocation,
                MCPLocalServerConfig, MCPServerConfig, CustomAgentConfig,
                ProviderConfig, PermissionRequest, PermissionRequestResult,
                SessionHooks, MessageOptions, ModelInfo,
            )
            return {
                "SessionConfig": SessionConfig,
                "Tool": Tool,
                "ToolResult": ToolResult,
                "ToolInvocation": ToolInvocation,
                "MCPLocalServerConfig": MCPLocalServerConfig,
                "MCPServerConfig": MCPServerConfig,
                "CustomAgentConfig": CustomAgentConfig,
                "ProviderConfig": ProviderConfig,
                "PermissionRequest": PermissionRequest,
                "PermissionRequestResult": PermissionRequestResult,
                "SessionHooks": SessionHooks,
                "MessageOptions": MessageOptions,
                "ModelInfo": ModelInfo,
            }
        except ImportError as e:
            print(f"  ✗ Cannot import session types: {e}", file=sys.stderr)
            return None

    # ─── Functional Integration (SLATE tools + sessions) ──────────────────

    # Modified: 2026-02-09T03:30:00Z | Author: COPILOT | Change: Add functional SDK integration methods

    def get_slate_sdk_tools(self) -> list:
        """
        Get all SLATE tools as Copilot SDK Tool objects (functional, with handlers).
        These are ready to pass to SessionConfig.tools for full SDK integration.
        
        Returns:
            List of Tool objects with Pydantic-validated handlers.
        """
        if not self.is_available():
            return []
        try:
            from slate.copilot_sdk_tools import get_all_slate_tools
            return get_all_slate_tools()
        except ImportError as e:
            print(f"  ✗ Cannot import SLATE SDK tools: {e}", file=sys.stderr)
            return []

    def get_tool_manifest(self) -> list[dict]:
        """Get JSON-serializable manifest of all registered SDK tools."""
        if not self.is_available():
            return []
        try:
            from slate.copilot_sdk_tools import get_tool_manifest
            return get_tool_manifest()
        except ImportError:
            return []

    def get_session_manager(self):
        """
        Get the SLATE Session Manager instance (singleton).
        Returns SLATESessionManager for creating/managing SDK sessions.
        """
        if not self.is_available():
            return None
        try:
            from slate.copilot_sdk_session import SLATESessionManager
            return SLATESessionManager(workspace_root=self.workspace_root)
        except ImportError as e:
            print(f"  ✗ Cannot import SLATESessionManager: {e}", file=sys.stderr)
            return None

    def get_mcp_configs(self) -> dict:
        """Get SLATE MCP server configurations for SDK sessions."""
        if not self.is_available():
            return {}
        try:
            from slate.copilot_sdk_session import SLATE_MCP_SERVERS
            return SLATE_MCP_SERVERS
        except ImportError:
            return {}

    def get_custom_agents(self) -> list[dict]:
        """Get SLATE custom agent configurations for SDK sessions."""
        if not self.is_available():
            return []
        try:
            from slate.copilot_sdk_session import SLATE_AGENT_CONFIGS
            return SLATE_AGENT_CONFIGS
        except ImportError:
            return []

    def get_byok_providers(self) -> dict:
        """Get available BYOK provider preset functions."""
        if not self.is_available():
            return {}
        try:
            from slate.copilot_sdk_session import (
                ollama_provider, azure_provider, anthropic_provider
            )
            return {
                "ollama": ollama_provider,
                "azure": azure_provider,
                "anthropic": anthropic_provider,
            }
        except ImportError:
            return {}

    def verify_full_integration(self) -> dict[str, Any]:
        """
        Verify the complete SDK integration stack:
        1. SDK submodule present
        2. Protocol version compatible
        3. Python SDK importable
        4. SLATE tools loadable
        5. Session manager importable
        6. MCP configs valid
        7. Custom agents defined
        8. BYOK providers available
        """
        checks = {
            "sdk_submodule": False,
            "protocol_version": False,
            "python_imports": False,
            "slate_tools": False,
            "session_manager": False,
            "mcp_configs": False,
            "custom_agents": False,
            "byok_providers": False,
            "issues": [],
        }

        # 1. Submodule
        checks["sdk_submodule"] = self.is_available()
        if not checks["sdk_submodule"]:
            checks["issues"].append("SDK submodule not found")
            return checks

        # 2. Protocol
        proto = self.get_protocol_version()
        checks["protocol_version"] = proto in SUPPORTED_PROTOCOL_VERSIONS
        if not checks["protocol_version"]:
            checks["issues"].append(f"Protocol v{proto} not supported")

        # 3. Python imports
        client = self.import_copilot_client()
        tools = self.import_copilot_tools()
        types = self.import_session_types()
        checks["python_imports"] = all([client, tools, types])
        if not checks["python_imports"]:
            checks["issues"].append("Python SDK imports failed")

        # 4. SLATE tools
        slate_tools = self.get_slate_sdk_tools()
        checks["slate_tools"] = len(slate_tools) > 0
        if not checks["slate_tools"]:
            checks["issues"].append("No SLATE SDK tools loaded")

        # 5. Session manager
        mgr = self.get_session_manager()
        checks["session_manager"] = mgr is not None
        if not checks["session_manager"]:
            checks["issues"].append("Session manager not available")

        # 6. MCP configs
        mcp = self.get_mcp_configs()
        checks["mcp_configs"] = len(mcp) > 0
        if not checks["mcp_configs"]:
            checks["issues"].append("No MCP server configs")

        # 7. Custom agents
        agents = self.get_custom_agents()
        checks["custom_agents"] = len(agents) > 0
        if not checks["custom_agents"]:
            checks["issues"].append("No custom agent configs")

        # 8. BYOK providers
        providers = self.get_byok_providers()
        checks["byok_providers"] = len(providers) > 0
        if not checks["byok_providers"]:
            checks["issues"].append("No BYOK provider presets")

        checks["all_passed"] = all(
            checks[k] for k in checks
            if k not in ("issues", "all_passed")
        )

        return checks

    # ─── Upstream Sync Status ─────────────────────────────────────────────

    def get_sync_status(self) -> dict[str, Any]:
        """Check how far behind the fork is from upstream."""
        result: dict[str, Any] = {
            "synced": False,
            "behind_count": None,
            "ahead_count": None,
            "last_sync": None,
            "current_commit": None,
            "upstream_remote": False,
            "error": None,
        }

        if not self.is_available():
            result["error"] = "SDK submodule not available"
            return result

        try:
            # Check if upstream remote exists
            remotes = subprocess.run(
                ["git", "remote"],
                capture_output=True, text=True,
                cwd=str(self.sdk_path)
            )
            has_upstream = "upstream" in remotes.stdout.split()
            result["upstream_remote"] = has_upstream

            # Get current commit
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True,
                cwd=str(self.sdk_path)
            )
            result["current_commit"] = head.stdout.strip()[:12]

            # Get last commit date
            last_date = subprocess.run(
                ["git", "log", "-1", "--format=%ci"],
                capture_output=True, text=True,
                cwd=str(self.sdk_path)
            )
            result["last_sync"] = last_date.stdout.strip()

            if has_upstream:
                # Fetch upstream (quick, no merge)
                subprocess.run(
                    ["git", "fetch", "upstream", "main", "--quiet"],
                    capture_output=True, text=True,
                    cwd=str(self.sdk_path),
                    timeout=30
                )

                # Count commits behind
                behind = subprocess.run(
                    ["git", "rev-list", "--count", "HEAD..upstream/main"],
                    capture_output=True, text=True,
                    cwd=str(self.sdk_path)
                )
                result["behind_count"] = int(behind.stdout.strip())

                # Count commits ahead (our local changes)
                ahead = subprocess.run(
                    ["git", "rev-list", "--count", "upstream/main..HEAD"],
                    capture_output=True, text=True,
                    cwd=str(self.sdk_path)
                )
                result["ahead_count"] = int(ahead.stdout.strip())

                result["synced"] = result["behind_count"] == 0

        except (subprocess.SubprocessError, ValueError, OSError) as e:
            result["error"] = str(e)

        return result

    # ─── SLATE Agent Tool Mapping ─────────────────────────────────────────

    def get_slate_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Generate SLATE-compatible tool definitions from the Copilot SDK.
        Maps SDK capabilities to SLATE agent patterns.
        
        NOTE: For functional SDK Tool objects (with handlers), use
        get_slate_sdk_tools() instead. This method returns metadata only.
        """
        tools = []

        if not self.is_available():
            return tools

        # Get functional tool manifest if available
        manifest = self.get_tool_manifest()
        if manifest:
            for t in manifest:
                tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "has_handler": t.get("has_handler", False),
                    "parameters": t.get("parameters"),
                    "sdk_type": "define_tool",
                })
            return tools

        # Fallback: basic definitions
        tools.extend([
            {
                "name": "copilot_sdk_session",
                "description": "Create a Copilot SDK session for agentic code operations",
                "agent": "COPILOT",
                "sdk_component": "CopilotClient.create_session",
                "requires_auth": True,
            },
            {
                "name": "copilot_sdk_define_tool",
                "description": "Define a custom tool using Copilot SDK's tool framework",
                "agent": "DELTA",
                "sdk_component": "copilot.define_tool",
                "requires_auth": False,
            },
            {
                "name": "copilot_sdk_models",
                "description": "List available models via Copilot SDK",
                "agent": "GAMMA",
                "sdk_component": "CopilotClient.list_models",
                "requires_auth": True,
            },
        ])

        return tools

    # ─── Full Status Report ───────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Get comprehensive SDK bridge status."""
        compat = self.check_compatibility()
        sync = self.get_sync_status()

        # Functional integration status
        tools = self.get_slate_sdk_tools()
        mcp = self.get_mcp_configs()
        agents = self.get_custom_agents()
        providers = self.get_byok_providers()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sdk_bridge": "copilot-sdk",
            "version": "2.0.0",
            "available": compat["sdk_available"],
            "compatible": compat["compatible"],
            "protocol_version": compat["protocol_version"],
            "python_sdk_version": compat["python_version"],
            "nodejs_sdk_version": compat["nodejs_version"],
            "submodule_path": str(self.sdk_path),
            "sync": sync,
            "issues": compat["issues"],
            "upstream_repo": "github/copilot-sdk",
            "fork_repo": "SynchronizedLivingArchitecture/copilot-sdk",
            "integration": {
                "sdk_tools": len(tools),
                "mcp_servers": list(mcp.keys()) if mcp else [],
                "custom_agents": [a.get("name", "?") for a in agents] if agents else [],
                "byok_providers": list(providers.keys()) if providers else [],
                "hooks_available": True,
                "permission_handler": "ActionGuard",
            },
        }

    def print_status(self):
        """Print human-readable status report."""
        s = self.status()

        print("=" * 60)
        print("  SLATE Copilot SDK Bridge v2.0")
        print("=" * 60)
        print()

        avail = "✓" if s["available"] else "✗"
        compat = "✓" if s["compatible"] else "✗"
        print(f"  SDK Available:      [{avail}]")
        print(f"  SDK Compatible:     [{compat}]")
        print(f"  Protocol Version:   {s['protocol_version'] or 'unknown'}")
        print(f"  Python SDK:         {s['python_sdk_version'] or 'unknown'}")
        print(f"  Node.js SDK:        {s['nodejs_sdk_version'] or 'unknown'}")
        print(f"  Submodule Path:     {s['submodule_path']}")
        print()

        sync = s["sync"]
        if sync.get("error"):
            print(f"  Sync Status:        ✗ {sync['error']}")
        else:
            synced = "✓ Up to date" if sync["synced"] else f"✗ {sync['behind_count']} commits behind"
            print(f"  Sync Status:        {synced}")
            if sync.get("ahead_count", 0) > 0:
                print(f"  Local Ahead:        {sync['ahead_count']} commits")
            print(f"  Current Commit:     {sync['current_commit'] or 'unknown'}")
            print(f"  Last Sync:          {sync['last_sync'] or 'unknown'}")
            print(f"  Upstream Remote:    {'✓' if sync['upstream_remote'] else '✗ Not configured'}")

        print()
        print(f"  Upstream:           {s['upstream_repo']}")
        print(f"  Fork:               {s['fork_repo']}")

        # Functional integration section
        integ = s.get("integration", {})
        print()
        print("  ─── Functional Integration ───")
        print(f"  SDK Tools:          {integ.get('sdk_tools', 0)} registered")
        mcp_servers = integ.get('mcp_servers', [])
        print(f"  MCP Servers:        {', '.join(mcp_servers) if mcp_servers else 'none'}")
        custom_agents = integ.get('custom_agents', [])
        print(f"  Custom Agents:      {', '.join(custom_agents) if custom_agents else 'none'}")
        byok = integ.get('byok_providers', [])
        print(f"  BYOK Providers:     {', '.join(byok) if byok else 'none'}")
        print(f"  Hooks Available:    {'✓' if integ.get('hooks_available') else '✗'}")
        print(f"  Permission Handler: {integ.get('permission_handler', 'none')}")

        if s["issues"]:
            print()
            print("  Issues:")
            for issue in s["issues"]:
                print(f"    ⚠ {issue}")

        print()
        print("=" * 60)


def main():
    """CLI entry point for the Copilot SDK bridge."""
    parser = argparse.ArgumentParser(
        description="SLATE Copilot SDK Bridge — manage GitHub Copilot SDK integration"
    )
    parser.add_argument("--status", action="store_true", help="Show SDK bridge status")
    parser.add_argument("--check", action="store_true", help="Run compatibility check")
    parser.add_argument("--sync-status", action="store_true", help="Check upstream sync status")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verify-import", action="store_true", help="Verify Python SDK can be imported")
    parser.add_argument("--verify-integration", action="store_true", help="Verify full SDK integration stack")
    parser.add_argument("--tools", action="store_true", help="List registered SDK tools")

    args = parser.parse_args()
    bridge = CopilotSDKBridge()

    if args.verify_integration:
        result = bridge.verify_full_integration()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n  SLATE Copilot SDK Integration Verification")
            print("  " + "─" * 50)
            checks = [
                ("SDK Submodule", result["sdk_submodule"]),
                ("Protocol Version", result["protocol_version"]),
                ("Python Imports", result["python_imports"]),
                ("SLATE Tools", result["slate_tools"]),
                ("Session Manager", result["session_manager"]),
                ("MCP Configs", result["mcp_configs"]),
                ("Custom Agents", result["custom_agents"]),
                ("BYOK Providers", result["byok_providers"]),
            ]
            for name, passed in checks:
                icon = "✓" if passed else "✗"
                print(f"  [{icon}] {name}")
            total = sum(1 for _, p in checks if p)
            print(f"\n  Result: {total}/{len(checks)} checks passed")
            if result.get("issues"):
                print("  Issues:")
                for issue in result["issues"]:
                    print(f"    ⚠ {issue}")
        return

    if args.tools:
        tools = bridge.get_slate_sdk_tools()
        if args.json:
            manifest = bridge.get_tool_manifest()
            print(json.dumps(manifest, indent=2))
        else:
            print(f"\n  SLATE Copilot SDK Tools: {len(tools)}")
            print("  " + "─" * 50)
            for t in tools:
                print(f"  • {t.name}: {t.description[:60]}...")
        return

    if args.verify_import:
        if bridge.ensure_python_path():
            try:
                # Attempt to import core SDK modules
                from copilot import CopilotClient, CopilotSession, define_tool  # noqa: F401
                from copilot.types import Tool, ToolInvocation, ToolResult  # noqa: F401
                print("  ✓ CopilotClient imported successfully")
                print("  ✓ CopilotSession imported successfully")
                print("  ✓ define_tool imported successfully")
                print("  ✓ Tool types imported successfully")
                print("\n  All Python SDK imports verified.")
            except ImportError as e:
                print(f"  ✗ Import failed: {e}")
                print("  Try: pip install pydantic python-dateutil")
                sys.exit(1)
        else:
            print("  ✗ SDK submodule not available")
            sys.exit(1)
        return

    if args.check:
        result = bridge.check_compatibility()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["compatible"]:
                print("  ✓ Copilot SDK is compatible with SLATE")
                print(f"    Protocol: v{result['protocol_version']}")
                print(f"    Python:   v{result['python_version']}")
                print(f"    Node.js:  v{result['nodejs_version']}")
            else:
                print("  ✗ Copilot SDK compatibility issues:")
                for issue in result["issues"]:
                    print(f"    ⚠ {issue}")
        return

    if args.sync_status:
        result = bridge.get_sync_status()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result.get("error"):
                print(f"  ✗ Sync error: {result['error']}")
            elif result["synced"]:
                print(f"  ✓ Fork is up to date with upstream")
                print(f"    Commit: {result['current_commit']}")
            else:
                print(f"  ⚠ Fork is {result['behind_count']} commits behind upstream")
                print(f"    Current: {result['current_commit']}")
                print(f"    Run: sync-copilot-sdk.yml workflow to update")
        return

    # Default: full status
    if args.json:
        print(json.dumps(bridge.status(), indent=2))
    else:
        bridge.print_status()


if __name__ == "__main__":
    main()
