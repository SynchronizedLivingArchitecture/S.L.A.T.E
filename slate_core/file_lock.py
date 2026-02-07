#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: file_lock [python]
# Author: Claude | Created: 2026-02-07T06:00:00Z
# Purpose: Thread-safe file locking for atomic access to current_tasks.json
# ═══════════════════════════════════════════════════════════════════════════════
"""
File Lock Module
================
Provides thread-safe and process-safe file locking for SLATE task management.

The primary use case is atomic access to current_tasks.json to prevent race
conditions when multiple processes (dashboard, runner, workflow manager) are
reading and writing simultaneously.

Usage:
    from slate_core.file_lock import FileLock, file_lock

    # Context manager style
    with file_lock("current_tasks.json") as f:
        data = json.load(f)
        data["tasks"].append(new_task)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    # Class-based style
    lock = FileLock("current_tasks.json")
    with lock.acquire():
        # Critical section
        pass
"""

import fcntl
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, TextIO, Union

# Windows compatibility
if os.name == "nt":
    import msvcrt

    def lock_file(f: TextIO, exclusive: bool = True) -> None:
        """Lock a file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)

    def unlock_file(f: TextIO) -> None:
        """Unlock a file on Windows."""
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
else:
    def lock_file(f: TextIO, exclusive: bool = True) -> None:
        """Lock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def unlock_file(f: TextIO) -> None:
        """Unlock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class FileLock:
    """
    Thread-safe and process-safe file lock.

    Attributes:
        path: Path to the file being locked
        timeout: Maximum time to wait for lock (seconds)
        retry_interval: Time between lock attempts (seconds)
    """

    def __init__(
        self,
        path: Union[str, Path],
        timeout: float = 10.0,
        retry_interval: float = 0.1
    ):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.timeout = timeout
        self.retry_interval = retry_interval
        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None

    @contextmanager
    def acquire(self, exclusive: bool = True) -> Generator[None, None, None]:
        """
        Acquire the file lock.

        Args:
            exclusive: If True, acquire exclusive (write) lock.
                      If False, acquire shared (read) lock.

        Yields:
            None when lock is acquired.

        Raises:
            TimeoutError: If lock cannot be acquired within timeout.
        """
        with self._lock:
            start_time = time.time()
            while True:
                try:
                    # Create lock file
                    self._file = open(self.lock_path, "w")
                    lock_file(self._file, exclusive)
                    break
                except (IOError, OSError) as e:
                    if self._file:
                        self._file.close()
                        self._file = None

                    elapsed = time.time() - start_time
                    if elapsed >= self.timeout:
                        raise TimeoutError(
                            f"Could not acquire lock on {self.path} "
                            f"within {self.timeout}s"
                        ) from e

                    time.sleep(self.retry_interval)

        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        """Release the file lock."""
        if self._file:
            try:
                unlock_file(self._file)
                self._file.close()
            except Exception:
                pass
            finally:
                self._file = None

            # Remove lock file
            try:
                self.lock_path.unlink()
            except Exception:
                pass

    def __enter__(self) -> "FileLock":
        """Enter context manager."""
        self._context = self.acquire()
        self._context.__enter__()
        return self

    def __exit__(self, *args) -> None:
        """Exit context manager."""
        self._context.__exit__(*args)


@contextmanager
def file_lock(
    path: Union[str, Path],
    mode: str = "r+",
    timeout: float = 10.0,
    create: bool = True
) -> Generator[TextIO, None, None]:
    """
    Context manager for locked file access.

    This is the recommended way to access shared files like current_tasks.json.

    Args:
        path: Path to the file
        mode: File open mode (r+, w, a, etc.)
        timeout: Maximum time to wait for lock
        create: If True, create file if it doesn't exist

    Yields:
        File handle with lock held.

    Example:
        with file_lock("current_tasks.json", "r+") as f:
            tasks = json.load(f)
            tasks.append({"id": "new-task"})
            f.seek(0)
            json.dump(tasks, f, indent=2)
            f.truncate()
    """
    path = Path(path)

    # Create file if needed
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "")

    lock = FileLock(path, timeout=timeout)
    with lock.acquire(exclusive="w" in mode or "+" in mode or "a" in mode):
        with open(path, mode, encoding="utf-8") as f:
            yield f


def atomic_json_update(
    path: Union[str, Path],
    update_fn: callable,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Atomically update a JSON file.

    Args:
        path: Path to JSON file
        update_fn: Function that takes current data and returns updated data
        timeout: Maximum time to wait for lock

    Returns:
        The updated data.

    Example:
        def add_task(data):
            data["tasks"].append({"id": "new-task"})
            return data

        updated = atomic_json_update("current_tasks.json", add_task)
    """
    with file_lock(path, "r+", timeout=timeout) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}

        updated_data = update_fn(data)

        f.seek(0)
        json.dump(updated_data, f, indent=2)
        f.truncate()

        return updated_data


def atomic_json_read(
    path: Union[str, Path],
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Atomically read a JSON file.

    Args:
        path: Path to JSON file
        timeout: Maximum time to wait for lock

    Returns:
        The file contents as a dict.
    """
    path = Path(path)
    if not path.exists():
        return {}

    lock = FileLock(path, timeout=timeout)
    with lock.acquire(exclusive=False):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
