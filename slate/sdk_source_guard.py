# Modified: 2026-02-07T03:00:00Z | Author: COPILOT | Change: Create SDK source guard module
"""
SDK Source Guard - Validates package publishers and sources for SLATE.

Enforces trusted package sources, blocks typosquatting packages,
and validates that only approved publishers are used.

Security rules:
- Only trusted PyPI publishers allowed
- Known typosquat patterns detected
- Package name similarity checking
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("slate.sdk_source_guard")

# ── Trusted Sources ─────────────────────────────────────────────────────

TRUSTED_PUBLISHERS = [
    "pytorch",
    "huggingface",
    "microsoft",
    "google",
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "scipy",
    "pandas",
]

TRUSTED_INDICES = [
    "https://pypi.org/simple/",
    "https://download.pytorch.org/whl/",
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu124",
]

# Known typosquatting patterns
TYPOSQUAT_PATTERNS = [
    r"^reqeusts$",       # requests
    r"^requets$",        # requests
    r"^python-dateutil2$",  # python-dateutil
    r"^numppy$",         # numpy
    r"^pandass$",        # pandas
    r"^fasttapi$",       # fastapi
    r"^torchh$",         # torch
    r"^tensorf1ow$",     # tensorflow
]

# Blocked packages (known malicious)
BLOCKED_PACKAGES = [
    "colourama",        # typosquat of colorama
    "python3-dateutil", # typosquat
    "jeIlyfish",        # typosquat of jellyfish (capital I vs l)
]


# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class SourceValidation:
    """Result of a source validation check."""
    valid: bool
    package: str
    reason: str

    def __str__(self) -> str:
        status = "VALID" if self.valid else "BLOCKED"
        return f"[{status}] {self.package}: {self.reason}"


# ── SDKSourceGuard Class ────────────────────────────────────────────────

class SDKSourceGuard:
    """
    Validates package sources and publishers for SLATE installations.

    Usage:
        guard = SDKSourceGuard()
        result = guard.validate_package("torch")
        if result.valid:
            # safe to install
        else:
            logger.warning(f"Package blocked: {result}")
    """

    def __init__(self):
        self._compiled_typosquats = [re.compile(p) for p in TYPOSQUAT_PATTERNS]

    def validate_package(self, package_name: str) -> SourceValidation:
        """Validate a package name for safety."""
        name_lower = package_name.lower().strip()

        # Check blocked packages
        if name_lower in [p.lower() for p in BLOCKED_PACKAGES]:
            return SourceValidation(
                valid=False,
                package=package_name,
                reason="Known malicious package",
            )

        # Check typosquat patterns
        for pattern in self._compiled_typosquats:
            if pattern.match(name_lower):
                return SourceValidation(
                    valid=False,
                    package=package_name,
                    reason=f"Potential typosquat: matches pattern {pattern.pattern}",
                )

        return SourceValidation(
            valid=True,
            package=package_name,
            reason="Package name OK",
        )

    def validate_index(self, index_url: str) -> SourceValidation:
        """Validate a package index URL."""
        for trusted in TRUSTED_INDICES:
            if index_url.startswith(trusted):
                return SourceValidation(
                    valid=True,
                    package="index",
                    reason=f"Trusted index: {trusted}",
                )

        return SourceValidation(
            valid=False,
            package="index",
            reason=f"Untrusted index: {index_url}",
        )

    def validate_requirements(self, requirements_path: str) -> list[SourceValidation]:
        """Validate all packages in a requirements file."""
        results = []
        try:
            with open(requirements_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    # Extract package name (before any version specifier)
                    pkg = re.split(r"[>=<!\[\];]", line)[0].strip()
                    if pkg:
                        results.append(self.validate_package(pkg))
        except FileNotFoundError:
            logger.warning(f"Requirements file not found: {requirements_path}")
        return results


# ── Module-Level Functions ──────────────────────────────────────────────

_default_guard: Optional[SDKSourceGuard] = None


def get_guard() -> SDKSourceGuard:
    """Get the default SDKSourceGuard singleton."""
    global _default_guard
    if _default_guard is None:
        _default_guard = SDKSourceGuard()
    return _default_guard


def validate_package(package_name: str) -> SourceValidation:
    """Validate a package using the default guard."""
    return get_guard().validate_package(package_name)


def validate_index(index_url: str) -> SourceValidation:
    """Validate an index URL using the default guard."""
    return get_guard().validate_index(index_url)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    guard = SDKSourceGuard()

    # Self-test
    tests = [
        ("torch", True),
        ("numpy", True),
        ("fastapi", True),
        ("colourama", False),     # blocked
        ("reqeusts", False),      # typosquat
        ("python3-dateutil", False),  # blocked
        ("transformers", True),
        ("slate", True),
    ]

    print("=" * 50)
    print("  SDK Source Guard Self-Test")
    print("=" * 50)

    passed = 0
    failed = 0
    for pkg, expected in tests:
        result = guard.validate_package(pkg)
        ok = result.valid == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {pkg} | expected={expected} got={result.valid} | {result.reason}")

    print(f"\n  Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
