# Modified: 2026-02-07T03:00:00Z | Author: COPILOT | Change: Create feature flags module
"""
Feature Flags - Runtime feature toggle system for SLATE.

Provides a simple feature flag mechanism for enabling/disabling
SLATE capabilities at runtime without code changes.
"""

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("slate.feature_flags")

# ── Default Flags ───────────────────────────────────────────────────────

DEFAULT_FLAGS = {
    # Core features
    "gpu_acceleration": True,
    "multi_gpu": True,
    "ollama_integration": True,
    "transformers_pipeline": True,

    # Agent features
    "agent_routing": True,
    "agent_memory": True,
    "task_queue": True,

    # Security features
    "action_guard": True,
    "sdk_source_guard": True,
    "pii_scanner": True,
    "network_isolation": True,

    # Dashboard
    "dashboard_api": True,
    "dashboard_ui": False,  # Not yet implemented

    # Experimental
    "cuda_graphs": False,
    "flash_attention": False,
    "quantization": False,

    # CI/CD
    "auto_dispatch": False,
    "nightly_benchmarks": True,
}

FLAGS_FILE = ".slate_flags.json"


# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class FeatureFlag:
    """A single feature flag with metadata."""
    name: str
    enabled: bool
    description: str = ""
    source: str = "default"  # default, file, override


# ── FeatureFlags Class ──────────────────────────────────────────────────

class FeatureFlags:
    """
    Runtime feature flag manager for SLATE.

    Usage:
        flags = FeatureFlags()
        if flags.is_enabled("gpu_acceleration"):
            # use GPU
        flags.set("experimental_feature", True)
    """

    def __init__(self, workspace: Optional[str] = None):
        self._flags: dict[str, bool] = dict(DEFAULT_FLAGS)
        self._overrides: dict[str, bool] = {}
        self._workspace = pathlib.Path(workspace or ".")
        self._load_from_file()

    def is_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled."""
        if flag in self._overrides:
            return self._overrides[flag]
        return self._flags.get(flag, False)

    def set(self, flag: str, enabled: bool) -> None:
        """Set a feature flag override (runtime only)."""
        self._overrides[flag] = enabled
        logger.info(f"Flag override: {flag} = {enabled}")

    def reset(self, flag: str) -> None:
        """Remove a runtime override, reverting to default/file value."""
        self._overrides.pop(flag, None)

    def get_all(self) -> dict[str, bool]:
        """Get all flags with overrides applied."""
        result = dict(self._flags)
        result.update(self._overrides)
        return result

    def get_enabled(self) -> list[str]:
        """Get list of all enabled flags."""
        all_flags = self.get_all()
        return sorted(k for k, v in all_flags.items() if v)

    def get_disabled(self) -> list[str]:
        """Get list of all disabled flags."""
        all_flags = self.get_all()
        return sorted(k for k, v in all_flags.items() if not v)

    def save(self) -> None:
        """Save current flags to file."""
        flags_path = self._workspace / FLAGS_FILE
        all_flags = self.get_all()
        flags_path.write_text(
            json.dumps(all_flags, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info(f"Flags saved to {flags_path}")

    def _load_from_file(self) -> None:
        """Load flags from workspace file if it exists."""
        flags_path = self._workspace / FLAGS_FILE
        if flags_path.exists():
            try:
                data = json.loads(flags_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, bool):
                            self._flags[k] = v
                    logger.debug(f"Loaded {len(data)} flags from {flags_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load flags from {flags_path}: {e}")


# ── Module-Level Functions ──────────────────────────────────────────────

_default_flags: Optional[FeatureFlags] = None


def get_flags(workspace: Optional[str] = None) -> FeatureFlags:
    """Get the default FeatureFlags singleton."""
    global _default_flags
    if _default_flags is None:
        _default_flags = FeatureFlags(workspace)
    return _default_flags


def is_enabled(flag: str) -> bool:
    """Quick check if a flag is enabled."""
    return get_flags().is_enabled(flag)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flags = FeatureFlags()

    print("=" * 50)
    print("  SLATE Feature Flags")
    print("=" * 50)

    enabled = flags.get_enabled()
    disabled = flags.get_disabled()

    print(f"\n  Enabled ({len(enabled)}):")
    for f in enabled:
        print(f"    ✓ {f}")

    print(f"\n  Disabled ({len(disabled)}):")
    for f in disabled:
        print(f"    ✗ {f}")

    print(f"\n  Total: {len(enabled) + len(disabled)} flags")
    print("=" * 50)
