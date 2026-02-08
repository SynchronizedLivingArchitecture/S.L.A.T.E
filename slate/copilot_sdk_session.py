#!/usr/bin/env python3
# Modified: 2026-02-09T03:15:00Z | Author: COPILOT | Change: Create Copilot SDK session manager for SLATE agent orchestration
"""
SLATE Copilot SDK Session Manager
===================================
Manages Copilot SDK sessions with full SLATE integration:

1. Session lifecycle (create, resume, destroy, list)
2. SLATE tool registration (14 tools via copilot_sdk_tools)
3. MCP server configuration for SLATE local services
4. Custom agent definitions mapping SLATE agents to SDK agents
5. Permission handling with ActionGuard integration
6. Event bridge between SDK sessions and SLATE event system
7. Hook system for pre/post tool use interception
8. BYOK (Bring Your Own Key) configuration for local-first operation

Architecture:
    SLATE Agent Loop  →  SLATESessionManager  →  CopilotClient
                                               →  CopilotSession
                                               →  SLATE SDK Tools
                                               →  MCP Servers (local)
                                               →  Custom Agents

Security:
    - LOCAL ONLY (127.0.0.1) — all sessions bind locally
    - ActionGuard enforced — permission handler intercepts dangerous operations
    - No external API calls without explicit BYOK configuration
    - SDK Source Guard approved — GitHub is a trusted publisher
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Ensure vendor SDK is on path
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SDK_PYTHON_PATH = WORKSPACE_ROOT / "vendor" / "copilot-sdk" / "python"
if str(SDK_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PYTHON_PATH))

from copilot import CopilotClient, CopilotSession, define_tool
from copilot.types import (
    ConnectionState,
    CopilotClientOptions,
    CustomAgentConfig,
    MCPLocalServerConfig,
    MCPRemoteServerConfig,
    MCPServerConfig,
    MessageOptions,
    PermissionRequest,
    PermissionRequestResult,
    ProviderConfig,
    SessionConfig,
    SessionHooks,
    Tool,
    ToolResult,
)

# Modified: 2026-02-09T03:15:00Z | Author: COPILOT | Change: Import SLATE SDK tools
from slate.copilot_sdk_tools import get_all_slate_tools, get_tool_manifest

logger = logging.getLogger("slate.copilot_sdk_session")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SLATE_SESSION_CONFIG_FILE = WORKSPACE_ROOT / ".slate_sdk_sessions.json"
SLATE_EVENT_LOG_FILE = WORKSPACE_ROOT / "slate_logs" / "copilot_sdk_events.log"

# SLATE agent → SDK custom agent mapping
SLATE_AGENT_CONFIGS: list[CustomAgentConfig] = [
    {
        "name": "slate-alpha",
        "display_name": "SLATE Alpha (Coder)",
        "description": "Coding agent — implements features, fixes bugs, builds components. GPU-accelerated.",
        "prompt": (
            "You are ALPHA, the SLATE coding agent. You implement code changes, fix bugs, "
            "and build new features for the S.L.A.T.E. framework. Use the slate_* tools to "
            "check system health before making changes. Always follow SLATE code edit rules: "
            "include timestamp+author comments. Local-only operations (127.0.0.1)."
        ),
        "tools": ["slate_system_status", "slate_runtime_check", "slate_workflow",
                   "slate_agent_registry", "slate_security_audit"],
    },
    {
        "name": "slate-beta",
        "display_name": "SLATE Beta (Tester)",
        "description": "Testing agent — validates, verifies, runs coverage analysis. GPU-accelerated.",
        "prompt": (
            "You are BETA, the SLATE testing agent. You run tests, validate code changes, "
            "verify integrations, and analyze test coverage. Use slate_benchmark for performance "
            "testing. Ensure all 7 integrations pass before approving changes."
        ),
        "tools": ["slate_system_status", "slate_runtime_check", "slate_benchmark",
                   "slate_security_audit"],
    },
    {
        "name": "slate-gamma",
        "display_name": "SLATE Gamma (Planner)",
        "description": "Planning agent — analyzes, plans, researches, documents. No GPU required.",
        "prompt": (
            "You are GAMMA, the SLATE planning agent. You analyze codebases, create plans, "
            "research solutions, and produce documentation. Use slate_workflow to manage tasks "
            "and slate_chromadb for semantic code search."
        ),
        "tools": ["slate_system_status", "slate_workflow", "slate_chromadb",
                   "slate_agent_registry"],
    },
    {
        "name": "slate-delta",
        "display_name": "SLATE Delta (Integrator)",
        "description": "External bridge agent — manages SDK integrations, MCP, Claude. No GPU required.",
        "prompt": (
            "You are DELTA, the SLATE integration agent. You manage SDK integrations, "
            "MCP server configurations, and external tool bridges. Ensure all connections "
            "bind to 127.0.0.1 only. Validate SDK sources against the approved publisher list."
        ),
        "tools": ["slate_system_status", "slate_runtime_check", "slate_security_audit"],
    },
]

# SLATE services as MCP servers
SLATE_MCP_SERVERS: dict[str, MCPServerConfig] = {
    "slate-dashboard": MCPLocalServerConfig(
        type="stdio",
        command=str(WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"),
        args=[str(WORKSPACE_ROOT / "agents" / "slate_dashboard_server.py")],
        env={"SLATE_WORKSPACE": str(WORKSPACE_ROOT)},
        cwd=str(WORKSPACE_ROOT),
        tools=["*"],
        timeout=30000,
    ),
    "slate-mcp": MCPLocalServerConfig(
        type="stdio",
        command=str(WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"),
        args=[str(WORKSPACE_ROOT / "slate" / "mcp_server.py")],
        env={"SLATE_WORKSPACE": str(WORKSPACE_ROOT)},
        cwd=str(WORKSPACE_ROOT),
        tools=["*"],
        timeout=30000,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Permission Handler — ActionGuard integration
# ═══════════════════════════════════════════════════════════════════════════════

# Modified: 2026-02-09T03:15:00Z | Author: COPILOT | Change: ActionGuard-integrated permission handler
BLOCKED_PATTERNS = [
    "rm -rf", "del /s", "format", "0.0.0.0",
    "eval(", "exec(os", "base64.b64decode",
]


async def slate_permission_handler(
    request: PermissionRequest,
    context: dict[str, str]
) -> PermissionRequestResult:
    """
    SLATE permission handler that integrates with ActionGuard.
    Blocks dangerous operations before they reach the SDK.
    """
    kind = request.get("kind", "")
    tool_call_id = request.get("toolCallId", "")

    # Shell commands need ActionGuard validation
    if kind == "shell":
        command = str(request.get("command", ""))
        for pattern in BLOCKED_PATTERNS:
            if pattern in command.lower():
                logger.warning(
                    f"ActionGuard BLOCKED shell command (pattern: {pattern}): {command[:80]}"
                )
                return PermissionRequestResult(
                    kind="denied-by-rules",
                    rules=[{"pattern": pattern, "source": "SLATE ActionGuard"}],
                )

    # Write operations — ensure no writes to protected paths
    if kind == "write":
        path = str(request.get("path", ""))
        protected_paths = [".github/workflows/", "CODEOWNERS", "action_guard.py"]
        for pp in protected_paths:
            if pp in path:
                logger.warning(f"ActionGuard BLOCKED write to protected path: {path}")
                return PermissionRequestResult(
                    kind="denied-by-rules",
                    rules=[{"path": pp, "source": "SLATE protected files"}],
                )

    # URL access — block external unless explicitly allowed
    if kind == "url":
        url = str(request.get("url", ""))
        allowed_hosts = ["127.0.0.1", "localhost", "github.com", "api.github.com"]
        if not any(h in url for h in allowed_hosts):
            logger.warning(f"ActionGuard BLOCKED external URL: {url}")
            return PermissionRequestResult(
                kind="denied-by-rules",
                rules=[{"url": url, "source": "SLATE local-only policy"}],
            )

    # Allow everything else
    return PermissionRequestResult(kind="approved", rules=[])


# ═══════════════════════════════════════════════════════════════════════════════
# Hook System — Pre/Post tool use interception
# ═══════════════════════════════════════════════════════════════════════════════

# Modified: 2026-02-09T03:15:00Z | Author: COPILOT | Change: SLATE session hooks for telemetry and safety

async def slate_pre_tool_use(hook_input: dict, context: dict) -> Optional[dict]:
    """Pre-tool-use hook — log and optionally intercept tool invocations."""
    tool_name = hook_input.get("toolName", "unknown")
    logger.debug(f"SDK tool invocation: {tool_name}")

    # Intercept dangerous tool args
    tool_args = hook_input.get("toolArgs", {})
    args_str = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
    for pattern in BLOCKED_PATTERNS:
        if pattern in args_str.lower():
            logger.warning(f"Pre-tool hook BLOCKED: {tool_name} (pattern: {pattern})")
            return {
                "permissionDecision": "deny",
                "permissionDecisionReason": f"ActionGuard blocked pattern: {pattern}",
            }

    return None  # Allow


async def slate_post_tool_use(hook_input: dict, context: dict) -> Optional[dict]:
    """Post-tool-use hook — log results for telemetry."""
    tool_name = hook_input.get("toolName", "unknown")
    logger.debug(f"SDK tool completed: {tool_name}")
    return None  # Pass through


async def slate_session_start(hook_input: dict, context: dict) -> Optional[dict]:
    """Session-start hook — inject SLATE context."""
    source = hook_input.get("source", "unknown")
    logger.info(f"SDK session started (source: {source})")
    return {
        "additionalContext": (
            "This is a SLATE (Synchronized Living Architecture for Transformation and Evolution) "
            "session. All operations are LOCAL ONLY (127.0.0.1). "
            f"Workspace: {WORKSPACE_ROOT}. "
            "Follow SLATE code edit rules: include # Modified: timestamp comments."
        ),
    }


async def slate_error_occurred(hook_input: dict, context: dict) -> Optional[dict]:
    """Error hook — log and determine recovery strategy."""
    error = hook_input.get("error", "unknown")
    error_context = hook_input.get("errorContext", "unknown")
    recoverable = hook_input.get("recoverable", False)

    logger.error(f"SDK session error ({error_context}): {error}")

    if recoverable:
        return {"errorHandling": "retry", "retryCount": 2}
    return {"errorHandling": "skip"}


SLATE_HOOKS: SessionHooks = {
    "on_pre_tool_use": slate_pre_tool_use,
    "on_post_tool_use": slate_post_tool_use,
    "on_session_start": slate_session_start,
    "on_error_occurred": slate_error_occurred,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Session Manager
# ═══════════════════════════════════════════════════════════════════════════════

class SLATESessionManager:
    """
    Manages Copilot SDK sessions with full SLATE integration.
    
    Provides session lifecycle management, tool registration, MCP server
    configuration, custom agent definitions, and event bridging.
    
    Usage:
        manager = SLATESessionManager()
        session = await manager.create_session()
        response = await session.send_and_wait({"prompt": "Check system status"})
        await manager.shutdown()
    """

    def __init__(
        self,
        client_options: Optional[CopilotClientOptions] = None,
        github_token: Optional[str] = None,
        workspace_root: Optional[Path] = None,
    ):
        self.workspace_root = workspace_root or WORKSPACE_ROOT
        self._client: Optional[CopilotClient] = None
        self._sessions: dict[str, CopilotSession] = {}
        self._event_handlers: list[Callable] = []
        self._lock = threading.Lock()

        # Build client options
        opts: CopilotClientOptions = client_options or {}
        opts.setdefault("cwd", str(self.workspace_root))
        opts.setdefault("log_level", "info")
        opts.setdefault("auto_start", True)
        opts.setdefault("auto_restart", True)

        if github_token:
            opts["github_token"] = github_token

        self._client_options = opts
        self._initialized = False

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Initialize the Copilot SDK client connection."""
        if self._initialized:
            return True

        try:
            self._client = CopilotClient(self._client_options)
            await self._client.start()
            self._initialized = True
            logger.info("Copilot SDK client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Copilot SDK client: {e}")
            self._client = None
            return False

    async def shutdown(self) -> list[str]:
        """Shut down the SDK client and all sessions."""
        errors: list[str] = []

        if self._client:
            try:
                stop_errors = await self._client.stop()
                errors.extend([e.message for e in stop_errors])
            except Exception as e:
                errors.append(f"Shutdown error: {e}")
            finally:
                self._client = None
                self._initialized = False

        with self._lock:
            self._sessions.clear()

        logger.info(f"SDK client shutdown complete ({len(errors)} errors)")
        return errors

    def get_state(self) -> ConnectionState:
        """Get the current SDK client connection state."""
        if self._client:
            return self._client.get_state()
        return "disconnected"

    # ─── Session Management ───────────────────────────────────────────────

    async def create_session(
        self,
        model: Optional[str] = None,
        include_tools: bool = True,
        include_mcp: bool = False,
        include_agents: bool = False,
        include_hooks: bool = True,
        streaming: bool = True,
        provider: Optional[ProviderConfig] = None,
        extra_tools: Optional[list[Tool]] = None,
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> CopilotSession:
        """
        Create a new Copilot SDK session with SLATE integration.
        
        Args:
            model: Model to use (e.g., 'gpt-4', 'claude-sonnet-4'). None = default.
            include_tools: Register all 14 SLATE SDK tools with the session.
            include_mcp: Include SLATE local MCP server configurations.
            include_agents: Include SLATE custom agent definitions.
            include_hooks: Enable SLATE session hooks (pre/post tool use, etc).
            streaming: Enable streaming responses.
            provider: Custom provider config (BYOK for local Ollama, etc).
            extra_tools: Additional tools to register alongside SLATE tools.
            session_id: Custom session ID. Auto-generated if None.
            system_message: Custom system message to append to default.
            
        Returns:
            CopilotSession ready for message exchange.
        """
        if not self._initialized:
            await self.initialize()

        if not self._client:
            raise RuntimeError("SDK client not initialized")

        # Build session config
        config: SessionConfig = {
            "working_directory": str(self.workspace_root),
            "streaming": streaming,
        }

        if model:
            config["model"] = model

        if session_id:
            config["session_id"] = session_id

        # Register SLATE tools
        if include_tools:
            tools = get_all_slate_tools()
            if extra_tools:
                tools.extend(extra_tools)
            config["tools"] = tools

        # System message
        if system_message:
            config["system_message"] = {
                "mode": "append",
                "content": system_message,
            }
        else:
            config["system_message"] = {
                "mode": "append",
                "content": (
                    "\n\n[SLATE Context]\n"
                    "This session is managed by SLATE v2.4.0. "
                    "All operations are LOCAL ONLY (127.0.0.1). "
                    f"Workspace: {self.workspace_root}. "
                    "Available SLATE tools: slate_system_status, slate_runtime_check, "
                    "slate_workflow, slate_hardware_info, slate_runner_status, "
                    "slate_orchestrator, slate_benchmark, slate_ml_orchestrator, "
                    "slate_agent_registry, slate_gpu_manager, slate_security_audit, "
                    "slate_autonomous, slate_chromadb, slate_model_trainer."
                ),
            }

        # Permission handler (ActionGuard)
        config["on_permission_request"] = slate_permission_handler

        # Hooks
        if include_hooks:
            config["hooks"] = SLATE_HOOKS

        # MCP servers
        if include_mcp:
            config["mcp_servers"] = SLATE_MCP_SERVERS

        # Custom agents
        if include_agents:
            config["custom_agents"] = SLATE_AGENT_CONFIGS

        # Provider (BYOK)
        if provider:
            config["provider"] = provider

        # Create session
        session = await self._client.create_session(config)

        with self._lock:
            self._sessions[session.session_id] = session

        logger.info(f"Created SDK session: {session.session_id}")
        return session

    async def create_local_session(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "slate-coder",
        **kwargs,
    ) -> CopilotSession:
        """
        Create a session using local Ollama as the provider (BYOK mode).
        
        This enables fully local-first operation without GitHub Copilot API access.
        
        Args:
            ollama_url: Ollama API endpoint (default: local)
            model: Ollama model name (default: slate-coder)
            **kwargs: Additional args passed to create_session
        """
        provider: ProviderConfig = {
            "type": "openai",
            "base_url": f"{ollama_url}/v1",
            "api_key": "ollama",  # Ollama accepts any key
            "wire_api": "completions",
        }
        return await self.create_session(
            model=model,
            provider=provider,
            **kwargs,
        )

    async def resume_session(self, session_id: str) -> CopilotSession:
        """Resume an existing SDK session by ID."""
        if not self._client:
            raise RuntimeError("SDK client not initialized")

        session = await self._client.resume_session(session_id, {
            "tools": get_all_slate_tools(),
            "on_permission_request": slate_permission_handler,
            "hooks": SLATE_HOOKS,
            "working_directory": str(self.workspace_root),
        })

        with self._lock:
            self._sessions[session.session_id] = session

        logger.info(f"Resumed SDK session: {session.session_id}")
        return session

    async def destroy_session(self, session_id: str) -> bool:
        """Destroy a specific session."""
        if not self._client:
            return False

        try:
            await self._client.delete_session(session_id)
            with self._lock:
                self._sessions.pop(session_id, None)
            logger.info(f"Destroyed SDK session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy session {session_id}: {e}")
            return False

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all known sessions with metadata."""
        if not self._client:
            return []

        try:
            sessions = await self._client.list_sessions()
            return [s.to_dict() for s in sessions]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    # ─── Message Exchange ─────────────────────────────────────────────────

    async def send_message(
        self,
        session_id: str,
        prompt: str,
        timeout: float = 60.0,
    ) -> Optional[dict[str, Any]]:
        """
        Send a message to a session and wait for the response.
        
        Args:
            session_id: Target session ID
            prompt: Message prompt
            timeout: Wait timeout in seconds
            
        Returns:
            Response event data dict, or None on timeout/error
        """
        with self._lock:
            session = self._sessions.get(session_id)

        if not session:
            logger.error(f"Session not found: {session_id}")
            return None

        try:
            response = await session.send_and_wait(
                {"prompt": prompt},
                timeout=timeout,
            )
            if response:
                return {
                    "type": str(response.type),
                    "data": response.data if hasattr(response, "data") else None,
                    "session_id": session_id,
                }
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Message timeout ({timeout}s) for session {session_id}")
            return None
        except Exception as e:
            logger.error(f"Message error in session {session_id}: {e}")
            return None

    # ─── Model Discovery ─────────────────────────────────────────────────

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from the SDK."""
        if not self._client:
            return []

        try:
            models = await self._client.list_models()
            return [m.to_dict() for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    # ─── Connectivity Check ──────────────────────────────────────────────

    async def ping(self) -> Optional[dict[str, Any]]:
        """Ping the SDK server to verify connectivity."""
        if not self._client:
            return None

        try:
            response = await self._client.ping("SLATE health check")
            return response.to_dict()
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return None

    async def get_auth_status(self) -> Optional[dict[str, Any]]:
        """Get current authentication status."""
        if not self._client:
            return None

        try:
            auth = await self._client.get_auth_status()
            return auth.to_dict()
        except Exception as e:
            logger.error(f"Auth status check failed: {e}")
            return None

    # ─── Event Bridge ─────────────────────────────────────────────────────

    def subscribe_events(
        self,
        session_id: str,
        handler: Callable,
    ) -> Optional[Callable[[], None]]:
        """
        Subscribe to events from an SDK session.
        Returns an unsubscribe function, or None if session not found.
        """
        with self._lock:
            session = self._sessions.get(session_id)

        if not session:
            return None

        return session.on(handler)

    # ─── Status Report ────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Get comprehensive session manager status."""
        with self._lock:
            session_ids = list(self._sessions.keys())

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initialized": self._initialized,
            "connection_state": self.get_state(),
            "active_sessions": len(session_ids),
            "session_ids": session_ids,
            "tools_registered": len(get_all_slate_tools()),
            "mcp_servers": list(SLATE_MCP_SERVERS.keys()),
            "custom_agents": [a["name"] for a in SLATE_AGENT_CONFIGS],
            "hooks_enabled": True,
            "permission_handler": "ActionGuard",
            "workspace": str(self.workspace_root),
        }

    def print_status(self):
        """Print human-readable status report."""
        s = self.status()

        print("=" * 60)
        print("  SLATE Copilot SDK Session Manager")
        print("=" * 60)
        print()

        init_icon = "✓" if s["initialized"] else "✗"
        print(f"  Initialized:        [{init_icon}]")
        print(f"  Connection State:   {s['connection_state']}")
        print(f"  Active Sessions:    {s['active_sessions']}")
        if s["session_ids"]:
            for sid in s["session_ids"]:
                print(f"    • {sid}")
        print()
        print(f"  Tools Registered:   {s['tools_registered']}")
        print(f"  MCP Servers:        {', '.join(s['mcp_servers'])}")
        print(f"  Custom Agents:      {', '.join(s['custom_agents'])}")
        print(f"  Hooks Enabled:      {s['hooks_enabled']}")
        print(f"  Permission Handler: {s['permission_handler']}")
        print(f"  Workspace:          {s['workspace']}")
        print()
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# BYOK Provider Presets
# ═══════════════════════════════════════════════════════════════════════════════

def ollama_provider(
    base_url: str = "http://127.0.0.1:11434",
) -> ProviderConfig:
    """Create a BYOK provider config for local Ollama."""
    return {
        "type": "openai",
        "base_url": f"{base_url}/v1",
        "api_key": "ollama",
        "wire_api": "completions",
    }


def azure_provider(
    base_url: str,
    api_key: str,
    api_version: str = "2024-10-21",
) -> ProviderConfig:
    """Create a BYOK provider config for Azure OpenAI."""
    return {
        "type": "azure",
        "base_url": base_url,
        "api_key": api_key,
        "azure": {"api_version": api_version},
    }


def anthropic_provider(
    api_key: str,
    base_url: str = "https://api.anthropic.com",
) -> ProviderConfig:
    """Create a BYOK provider config for Anthropic."""
    return {
        "type": "anthropic",
        "base_url": base_url,
        "api_key": api_key,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLATE Copilot SDK Session Manager"
    )
    parser.add_argument("--status", action="store_true", help="Show session manager status")
    parser.add_argument("--tools", action="store_true", help="List registered tools")
    parser.add_argument("--agents", action="store_true", help="List custom agent configs")
    parser.add_argument("--mcp", action="store_true", help="List MCP server configs")
    parser.add_argument("--providers", action="store_true", help="List BYOK provider presets")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    manager = SLATESessionManager()

    if args.tools:
        manifest = get_tool_manifest()
        if args.json:
            print(json.dumps(manifest, indent=2))
        else:
            print(f"\n  SLATE SDK Tools: {len(manifest)}")
            print("  " + "─" * 50)
            for t in manifest:
                print(f"  • {t['name']}: {t['description'][:60]}...")
        return

    if args.agents:
        if args.json:
            print(json.dumps(SLATE_AGENT_CONFIGS, indent=2))
        else:
            print(f"\n  Custom Agent Configs: {len(SLATE_AGENT_CONFIGS)}")
            print("  " + "─" * 50)
            for a in SLATE_AGENT_CONFIGS:
                print(f"  • {a['name']}: {a.get('display_name', '')}")
                print(f"    {a.get('description', '')[:70]}...")
        return

    if args.mcp:
        mcp_info = {
            k: {"type": v.get("type", "stdio"), "command": v.get("command", "")}
            for k, v in SLATE_MCP_SERVERS.items()
        }
        if args.json:
            print(json.dumps(mcp_info, indent=2))
        else:
            print(f"\n  MCP Servers: {len(SLATE_MCP_SERVERS)}")
            print("  " + "─" * 50)
            for name, info in mcp_info.items():
                print(f"  • {name}: {info['type']} → {Path(info['command']).name}")
        return

    if args.providers:
        providers = {
            "ollama (local)": {"type": "openai", "base_url": "http://127.0.0.1:11434/v1"},
            "azure": {"type": "azure", "requires": "api_key + base_url"},
            "anthropic": {"type": "anthropic", "requires": "api_key"},
        }
        if args.json:
            print(json.dumps(providers, indent=2))
        else:
            print("\n  BYOK Provider Presets:")
            print("  " + "─" * 50)
            for name, info in providers.items():
                print(f"  • {name}: {info}")
        return

    # Default: status
    if args.json:
        print(json.dumps(manager.status(), indent=2))
    else:
        manager.print_status()


if __name__ == "__main__":
    main()
