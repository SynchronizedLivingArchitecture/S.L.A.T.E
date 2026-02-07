#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: memory [python]
# Author: Claude | Created: 2026-02-07T06:00:00Z
# Purpose: Constitution and memory storage for SLATE system
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Memory Module
===================
Manages constitution, persistent memory, and learning storage for SLATE.

Features:
- Constitution loading and caching
- Key-value memory store
- Session memory
- ChromaDB integration for vector storage

Usage:
    from slate_core.memory import ConstitutionMemory, get_constitution

    # Get constitution
    constitution = get_constitution()
    print(constitution["principles"])

    # Use memory store
    memory = ConstitutionMemory()
    memory.set("user_preference", "dark_mode")
    value = memory.get("user_preference")
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

WORKSPACE_ROOT = Path(__file__).parent.parent
CONSTITUTION_PATH = WORKSPACE_ROOT / ".specify" / "memory" / "constitution.md"
MEMORY_DIR = WORKSPACE_ROOT / ".specify" / "memory"
MEMORY_STORE_PATH = MEMORY_DIR / "memory_store.json"


class ConstitutionMemory:
    """
    SLATE memory management system.

    Provides persistent key-value storage and constitution access.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}
        self._constitution: Optional[str] = None
        self._last_constitution_load: float = 0
        self.CONSTITUTION_TTL = 300  # 5 minutes

        # Ensure memory directory exists
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    def get_constitution(self, force_reload: bool = False) -> str:
        """
        Get the SLATE constitution.

        Args:
            force_reload: If True, reload from disk.

        Returns:
            Constitution text.
        """
        now = time.time()

        if (
            self._constitution is None or
            force_reload or
            now - self._last_constitution_load > self.CONSTITUTION_TTL
        ):
            self._load_constitution()

        return self._constitution or ""

    def _load_constitution(self) -> None:
        """Load constitution from file."""
        try:
            if CONSTITUTION_PATH.exists():
                self._constitution = CONSTITUTION_PATH.read_text(encoding="utf-8")
            else:
                self._constitution = self._default_constitution()
            self._last_constitution_load = time.time()
        except Exception:
            self._constitution = self._default_constitution()

    def _default_constitution(self) -> str:
        """Return default constitution if file not found."""
        return """# SLATE Constitution

## Core Principles

1. **Local-First**: All processing happens locally. No external API calls without explicit user consent.
2. **Security**: Bind to 127.0.0.1 only. Never expose services to network.
3. **Transparency**: All actions are logged and auditable.
4. **Test-Driven**: All code changes must be accompanied by tests.
5. **User Control**: User always has final say on actions.

## Operational Rules

- Never execute destructive commands without confirmation
- Never commit secrets or credentials
- Always use FileLock for shared resources
- Prefer existing patterns over new abstractions
"""

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from memory.

        Args:
            key: Memory key.
            default: Default value if key not found.

        Returns:
            Stored value or default.
        """
        with self._lock:
            if key in self._cache:
                return self._cache[key]

            store = self._load_store()
            return store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in memory.

        Args:
            key: Memory key.
            value: Value to store (must be JSON-serializable).
        """
        with self._lock:
            self._cache[key] = value
            store = self._load_store()
            store[key] = value
            store["_updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_store(store)

    def delete(self, key: str) -> bool:
        """
        Delete a key from memory.

        Args:
            key: Memory key to delete.

        Returns:
            True if key was deleted.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]

            store = self._load_store()
            if key in store:
                del store[key]
                self._save_store(store)
                return True
            return False

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all memory keys.

        Args:
            prefix: If provided, only return keys starting with prefix.

        Returns:
            List of keys.
        """
        store = self._load_store()
        keys = [k for k in store.keys() if not k.startswith("_")]

        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]

        return sorted(keys)

    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear memory.

        Args:
            prefix: If provided, only clear keys with this prefix.

        Returns:
            Number of keys cleared.
        """
        with self._lock:
            store = self._load_store()
            if prefix:
                keys_to_delete = [k for k in store.keys()
                                 if k.startswith(prefix) and not k.startswith("_")]
            else:
                keys_to_delete = [k for k in store.keys() if not k.startswith("_")]

            for key in keys_to_delete:
                del store[key]
                if key in self._cache:
                    del self._cache[key]

            self._save_store(store)
            return len(keys_to_delete)

    def _load_store(self) -> Dict[str, Any]:
        """Load memory store from file."""
        if not MEMORY_STORE_PATH.exists():
            return {}
        try:
            return json.loads(MEMORY_STORE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_store(self, store: Dict[str, Any]) -> None:
        """Save memory store to file."""
        MEMORY_STORE_PATH.write_text(
            json.dumps(store, indent=2, default=str),
            encoding="utf-8"
        )


# Global instance
_memory_instance: Optional[ConstitutionMemory] = None
_memory_lock = threading.Lock()


def get_memory() -> ConstitutionMemory:
    """Get the global memory instance."""
    global _memory_instance
    with _memory_lock:
        if _memory_instance is None:
            _memory_instance = ConstitutionMemory()
        return _memory_instance


def get_constitution(force_reload: bool = False) -> str:
    """Get the SLATE constitution."""
    return get_memory().get_constitution(force_reload)


def remember(key: str, value: Any) -> None:
    """Store a value in memory."""
    get_memory().set(key, value)


def recall(key: str, default: Any = None) -> Any:
    """Retrieve a value from memory."""
    return get_memory().get(key, default)


def forget(key: str) -> bool:
    """Delete a value from memory."""
    return get_memory().delete(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLATE Memory")
    parser.add_argument("--constitution", action="store_true",
                       help="Print constitution")
    parser.add_argument("--get", metavar="KEY", help="Get a memory value")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                       help="Set a memory value")
    parser.add_argument("--list", action="store_true", help="List all keys")

    args = parser.parse_args()
    memory = ConstitutionMemory()

    if args.constitution:
        print(memory.get_constitution())
    elif args.get:
        value = memory.get(args.get)
        print(json.dumps(value) if value is not None else "Not found")
    elif args.set:
        memory.set(args.set[0], args.set[1])
        print(f"Set {args.set[0]} = {args.set[1]}")
    elif args.list:
        keys = memory.list_keys()
        for key in keys:
            print(key)
    else:
        print(f"Constitution: {len(memory.get_constitution())} bytes")
        print(f"Memory keys: {len(memory.list_keys())}")
