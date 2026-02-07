# Modified: 2026-02-07T03:00:00Z | Author: COPILOT | Change: Create ActionGuard security module
"""
ActionGuard - Security enforcement for SLATE agent actions.

Validates all agent actions before execution, blocks dangerous operations,
enforces network binding rules, and logs security decisions.

Security rules:
- ALL network bindings must use 127.0.0.1 (never 0.0.0.0)
- Blocked patterns: eval(, exec(os, rm -rf /, base64.b64decode
- Blocked external API domains (local-first enforcement)
- Rate limiting on API calls
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("slate.action_guard")

# ── Security Configuration ──────────────────────────────────────────────

BLOCKED_PATTERNS = [
    r"eval\(",
    r"exec\(os",
    r"rm\s+-rf\s+/",
    r"base64\.b64decode",
    r"0\.0\.0\.0",
    r"subprocess\.call.*shell\s*=\s*True",
    r"__import__\(",
    r"os\.system\(",
]

BLOCKED_DOMAINS = [
    "api.openai.com",
    "api.anthropic.com",
    "api.cohere.com",
    "generativelanguage.googleapis.com",
]

ALLOWED_HOSTS = [
    "127.0.0.1",
    "localhost",
    "::1",
]

ALLOWED_GITHUB_DOMAINS = [
    "api.github.com",
    "github.com",
    "raw.githubusercontent.com",
]

# Rate limiting: max calls per minute per action type
RATE_LIMITS = {
    "api_call": 60,
    "file_write": 120,
    "command_exec": 30,
    "network_request": 30,
}


# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Result of an action guard validation."""
    allowed: bool
    action: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        status = "ALLOWED" if self.allowed else "BLOCKED"
        return f"[{status}] {self.action}: {self.reason}"


@dataclass
class RateTracker:
    """Tracks action rates for rate limiting."""
    calls: list = field(default_factory=list)

    def add(self) -> None:
        self.calls.append(time.time())

    def count_recent(self, window: float = 60.0) -> int:
        cutoff = time.time() - window
        self.calls = [t for t in self.calls if t > cutoff]
        return len(self.calls)


# ── ActionGuard Class ───────────────────────────────────────────────────

class ActionGuard:
    """
    Validates agent actions against SLATE security policies.

    Usage:
        guard = ActionGuard()
        result = guard.validate_action("command_exec", "python slate/slate_status.py --quick")
        if result.allowed:
            # proceed
        else:
            logger.warning(f"Action blocked: {result}")
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self._rate_trackers: dict[str, RateTracker] = {}
        self._audit_log: list[ActionResult] = []
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]

    def validate_action(self, action_type: str, content: str) -> ActionResult:
        """
        Validate an action against security policies.

        Args:
            action_type: Type of action (command_exec, api_call, file_write, network_request)
            content: The content/command to validate

        Returns:
            ActionResult with allowed status and reason
        """
        # Check blocked patterns
        for pattern in self._compiled_patterns:
            if pattern.search(content):
                result = ActionResult(
                    allowed=False,
                    action=action_type,
                    reason=f"Blocked pattern: {pattern.pattern}",
                )
                self._audit(result)
                return result

        # Check network binding
        if "0.0.0.0" in content:
            result = ActionResult(
                allowed=False,
                action=action_type,
                reason="Network binding violation: must use 127.0.0.1",
            )
            self._audit(result)
            return result

        # Check blocked domains
        for domain in BLOCKED_DOMAINS:
            if domain in content:
                result = ActionResult(
                    allowed=False,
                    action=action_type,
                    reason=f"Blocked external domain: {domain}",
                )
                self._audit(result)
                return result

        # Rate limiting
        if action_type in RATE_LIMITS:
            tracker = self._rate_trackers.setdefault(action_type, RateTracker())
            if tracker.count_recent() >= RATE_LIMITS[action_type]:
                result = ActionResult(
                    allowed=False,
                    action=action_type,
                    reason=f"Rate limit exceeded: {RATE_LIMITS[action_type]}/min",
                )
                self._audit(result)
                return result
            tracker.add()

        result = ActionResult(
            allowed=True,
            action=action_type,
            reason="Passed all security checks",
        )
        self._audit(result)
        return result

    def validate_host(self, host: str) -> ActionResult:
        """Validate a network host binding."""
        if host in ALLOWED_HOSTS:
            return ActionResult(allowed=True, action="host_bind", reason=f"Allowed host: {host}")
        if host in ALLOWED_GITHUB_DOMAINS:
            return ActionResult(allowed=True, action="host_bind", reason=f"Allowed GitHub domain: {host}")
        return ActionResult(
            allowed=False,
            action="host_bind",
            reason=f"Blocked host: {host}. Only 127.0.0.1/localhost allowed.",
        )

    def validate_command(self, command: str) -> ActionResult:
        """Shorthand for validating a command execution."""
        return self.validate_action("command_exec", command)

    def validate_file_path(self, path: str) -> ActionResult:
        """Validate a file path for safety."""
        dangerous_paths = ["/etc/passwd", "/etc/shadow", "C:\\Windows\\System32"]
        for dp in dangerous_paths:
            if dp.lower() in path.lower():
                return ActionResult(
                    allowed=False,
                    action="file_access",
                    reason=f"Dangerous path: {dp}",
                )
        return ActionResult(allowed=True, action="file_access", reason="Path OK")

    def get_audit_log(self) -> list[ActionResult]:
        """Return the audit log of all validated actions."""
        return self._audit_log.copy()

    def get_blocked_count(self) -> int:
        """Return count of blocked actions."""
        return sum(1 for r in self._audit_log if not r.allowed)

    def _audit(self, result: ActionResult) -> None:
        """Record action in audit log."""
        self._audit_log.append(result)
        if not result.allowed:
            logger.warning(str(result))
        else:
            logger.debug(str(result))


# ── Module-Level Functions ──────────────────────────────────────────────

_default_guard: Optional[ActionGuard] = None


def get_guard() -> ActionGuard:
    """Get the default ActionGuard singleton."""
    global _default_guard
    if _default_guard is None:
        _default_guard = ActionGuard()
    return _default_guard


def validate_action(action_type: str, content: str) -> ActionResult:
    """Validate an action using the default guard."""
    return get_guard().validate_action(action_type, content)


def validate_command(command: str) -> ActionResult:
    """Validate a command using the default guard."""
    return get_guard().validate_command(command)


def is_safe(action_type: str, content: str) -> bool:
    """Quick check — returns True if action is allowed."""
    return get_guard().validate_action(action_type, content).allowed


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    guard = ActionGuard()

    # Self-test
    tests = [
        ("command_exec", "python slate/slate_status.py --quick", True),
        ("command_exec", 'eval("dangerous")', False),
        ("command_exec", "rm -rf /", False),
        ("network_request", "http://127.0.0.1:8080/api", True),
        ("network_request", "http://0.0.0.0:8080", False),
        ("network_request", "https://api.openai.com/v1/chat", False),
        ("network_request", "https://api.github.com/repos", True),
        ("command_exec", 'exec(os.system("whoami"))', False),
        ("command_exec", "base64.b64decode(payload)", False),
    ]

    print("=" * 50)
    print("  ActionGuard Self-Test")
    print("=" * 50)

    passed = 0
    failed = 0
    for action_type, content, expected in tests:
        result = guard.validate_action(action_type, content)
        ok = result.allowed == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {action_type} | {content[:50]} | expected={expected} got={result.allowed}")

    print(f"\n  Results: {passed} passed, {failed} failed")
    print(f"  Blocked: {guard.get_blocked_count()} actions")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
