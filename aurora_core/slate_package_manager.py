#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Modified: 2026-02-06T18:00:00Z | Author: COPILOT | Change: Initial creation
# Purpose: SLATE Package & Release Manager — CLI for managing packages,
#          releases, versioning, changelog, and GitHub integration
# ═══════════════════════════════════════════════════════════════════════════════
"""
S.L.A.T.E. Package & Release Manager
=====================================

Manages the full lifecycle of SLATE packages and releases:
  - Version bumping (major/minor/patch) across all version files
  - Changelog generation and validation
  - Package building (sdist + wheel)
  - Release preparation and tagging
  - GitHub Packages integration status
  - Dependency auditing

Usage:
    slate-package --status          Show package/release status
    slate-package --bump patch      Bump version (major|minor|patch)
    slate-package --build           Build sdist + wheel
    slate-package --changelog       Show current changelog entry
    slate-package --validate        Validate package readiness
    slate-package --release         Prepare a release (tag + changelog)
    slate-package --deps            Audit dependencies
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_FILE = ROOT / "aurora_core" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"
DIST_DIR = ROOT / "dist"
VERSION_FILES = [PYPROJECT, INIT_FILE]

GITHUB_ORG = "SynchronizedLivingArchitecture"
GITHUB_REPO = "S.L.A.T.E."
GITHUB_URL = f"https://github.com/{GITHUB_ORG}/{GITHUB_REPO}"


# ═══════════════════════════════════════════════════════════════════════════════
# Version Management
# ═══════════════════════════════════════════════════════════════════════════════

def get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml."""
    if not PYPROJECT.exists():
        return "unknown"
    content = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else "unknown"


def get_version_from_init() -> str:
    """Extract version from aurora_core/__init__.py."""
    if not INIT_FILE.exists():
        return "unknown"
    content = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else "unknown"


def versions_consistent() -> tuple[bool, dict[str, str]]:
    """Check if all version files have the same version."""
    versions = {
        "pyproject.toml": get_version_from_pyproject(),
        "aurora_core/__init__.py": get_version_from_init(),
    }
    unique = set(v for v in versions.values() if v != "unknown")
    return len(unique) <= 1, versions


def bump_version(current: str, part: str) -> str:
    """Bump a semver version string."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", current)
    if not match:
        raise ValueError(f"Invalid version: {current}")
    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))

    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump part: {part}. Use major|minor|patch")


def apply_version(new_version: str) -> list[str]:
    """Apply new version to all version files."""
    updated = []

    # pyproject.toml
    if PYPROJECT.exists():
        content = PYPROJECT.read_text(encoding="utf-8")
        new_content = re.sub(
            r'(version\s*=\s*)"[^"]+"',
            f'\\1"{new_version}"',
            content,
        )
        if new_content != content:
            PYPROJECT.write_text(new_content, encoding="utf-8")
            updated.append("pyproject.toml")

    # aurora_core/__init__.py
    if INIT_FILE.exists():
        content = INIT_FILE.read_text(encoding="utf-8")
        new_content = re.sub(
            r'(__version__\s*=\s*)["\'][^"\']+["\']',
            f'\\1"{new_version}"',
            content,
        )
        if new_content != content:
            INIT_FILE.write_text(new_content, encoding="utf-8")
            updated.append("aurora_core/__init__.py")

    return updated


# ═══════════════════════════════════════════════════════════════════════════════
# Changelog Management
# ═══════════════════════════════════════════════════════════════════════════════

def get_changelog_entry(version: str) -> str | None:
    """Extract changelog entry for a specific version."""
    if not CHANGELOG.exists():
        return None
    content = CHANGELOG.read_text(encoding="utf-8")
    pattern = rf"## \[{re.escape(version)}\].*?\n(.*?)(?=\n## \[|$)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None


def get_unreleased_changes() -> str | None:
    """Extract unreleased changelog section."""
    if not CHANGELOG.exists():
        return None
    content = CHANGELOG.read_text(encoding="utf-8")
    pattern = r"## \[Unreleased\].*?\n(.*?)(?=\n## \[|$)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None


def stamp_changelog(version: str) -> bool:
    """Move unreleased changes into a versioned section."""
    if not CHANGELOG.exists():
        return False

    content = CHANGELOG.read_text(encoding="utf-8")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Insert new version header after [Unreleased]
    unreleased_pattern = r"(## \[Unreleased\].*?\n)(.*?)(\n## \[)"
    match = re.search(unreleased_pattern, content, re.DOTALL)
    if not match:
        return False

    unreleased_header = match.group(1)
    unreleased_body = match.group(2)
    next_section = match.group(3)

    new_section = f"{unreleased_header}\n## [{version}] - {today}\n{unreleased_body}{next_section}"
    new_content = content[: match.start()] + new_section + content[match.end() :]

    # Update comparison links at bottom
    old_link = f"[Unreleased]: {GITHUB_URL}/compare/v"
    if old_link in new_content:
        new_content = re.sub(
            r"\[Unreleased\]: .+?\.\.\.HEAD",
            f"[Unreleased]: {GITHUB_URL}/compare/v{version}...HEAD",
            new_content,
        )

    CHANGELOG.write_text(new_content, encoding="utf-8")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Package Building
# ═══════════════════════════════════════════════════════════════════════════════

def build_package() -> dict:
    """Build sdist and wheel distributions."""
    result = {"success": False, "files": [], "errors": []}

    # Clean dist directory
    if DIST_DIR.exists():
        for f in DIST_DIR.iterdir():
            f.unlink()

    # Build
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "build"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            result["errors"].append(proc.stderr)
            return result
    except FileNotFoundError:
        result["errors"].append("'build' module not installed. Run: pip install build")
        return result
    except subprocess.TimeoutExpired:
        result["errors"].append("Build timed out after 120s")
        return result

    # Collect built files
    if DIST_DIR.exists():
        for f in DIST_DIR.iterdir():
            result["files"].append({
                "name": f.name,
                "size": f.stat().st_size,
                "size_human": f"{f.stat().st_size / 1024:.1f} KB",
            })

    # Validate with twine
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "twine", "check", "dist/*"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            shell=True,
        )
        if proc.returncode != 0:
            result["errors"].append(f"twine check failed: {proc.stderr}")
    except FileNotFoundError:
        result["errors"].append("'twine' not installed (optional). Run: pip install twine")

    result["success"] = len(result["files"]) > 0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency Audit
# ═══════════════════════════════════════════════════════════════════════════════

def audit_dependencies() -> dict:
    """Audit installed dependencies against requirements."""
    result = {
        "core": [],
        "optional_ai": [],
        "optional_dev": [],
        "missing": [],
        "outdated": [],
    }

    # Parse pyproject.toml dependencies
    if not PYPROJECT.exists():
        return result

    content = PYPROJECT.read_text(encoding="utf-8")

    # Get installed packages
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        installed = {
            pkg["name"].lower(): pkg["version"]
            for pkg in json.loads(proc.stdout)
        } if proc.returncode == 0 else {}
    except Exception:
        installed = {}

    # Core dependencies
    dep_pattern = r'\[project\].*?dependencies\s*=\s*\[(.*?)\]'
    match = re.search(dep_pattern, content, re.DOTALL)
    if match:
        deps_text = match.group(1)
        for dep_match in re.finditer(r'"([^"]+)"', deps_text):
            dep_spec = dep_match.group(1)
            dep_name = re.split(r'[><=!~]', dep_spec)[0].strip().lower()
            installed_ver = installed.get(dep_name, installed.get(dep_name.replace("-", "_")))
            status = "installed" if installed_ver else "missing"
            entry = {
                "name": dep_name,
                "required": dep_spec,
                "installed": installed_ver or "—",
                "status": status,
            }
            result["core"].append(entry)
            if status == "missing":
                result["missing"].append(dep_name)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Git / Release Integration
# ═══════════════════════════════════════════════════════════════════════════════

def get_git_tags() -> list[str]:
    """Get existing git tags."""
    try:
        proc = subprocess.run(
            ["git", "tag", "-l", "v*", "--sort=-v:refname"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip().split("\n") if proc.stdout.strip() else []
    except Exception:
        return []


def get_latest_tag() -> str | None:
    """Get the latest version tag."""
    tags = get_git_tags()
    return tags[0] if tags else None


def get_commits_since_tag(tag: str | None) -> list[dict]:
    """Get commits since a tag."""
    try:
        cmd = ["git", "log", "--oneline", "--no-merges"]
        if tag:
            cmd.append(f"{tag}..HEAD")
        else:
            cmd.extend(["-20"])
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        commits = []
        for line in proc.stdout.strip().split("\n"):
            if line.strip():
                sha, _, msg = line.partition(" ")
                commits.append({"sha": sha, "message": msg})
        return commits
    except Exception:
        return []


def prepare_release(version: str, dry_run: bool = True) -> dict:
    """Prepare a release: validate, bump, changelog, tag."""
    result = {
        "version": version,
        "dry_run": dry_run,
        "steps": [],
        "success": True,
    }

    # 1. Version consistency
    consistent, versions = versions_consistent()
    result["steps"].append({
        "name": "Version Check",
        "status": "ok" if consistent else "warning",
        "detail": versions,
    })

    # 2. Changelog entry
    entry = get_changelog_entry(version)
    unreleased = get_unreleased_changes()
    has_changelog = entry is not None or (unreleased and unreleased.strip())
    result["steps"].append({
        "name": "Changelog",
        "status": "ok" if has_changelog else "missing",
        "detail": entry or unreleased or "No changelog entry found",
    })

    # 3. Git state
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        clean = not proc.stdout.strip()
        result["steps"].append({
            "name": "Git Clean",
            "status": "ok" if clean else "warning",
            "detail": "Working tree clean" if clean else f"{len(proc.stdout.strip().splitlines())} uncommitted changes",
        })
    except Exception:
        result["steps"].append({
            "name": "Git Clean",
            "status": "error",
            "detail": "Could not check git status",
        })

    # 4. Existing tag check
    tags = get_git_tags()
    tag_exists = f"v{version}" in tags
    result["steps"].append({
        "name": "Tag Check",
        "status": "error" if tag_exists else "ok",
        "detail": f"v{version} already exists!" if tag_exists else f"v{version} is available",
    })
    if tag_exists:
        result["success"] = False

    if not dry_run and result["success"]:
        # Apply version
        updated = apply_version(version)
        result["steps"].append({
            "name": "Version Bump",
            "status": "ok",
            "detail": f"Updated: {', '.join(updated)}",
        })

        # Stamp changelog
        if unreleased:
            stamped = stamp_changelog(version)
            result["steps"].append({
                "name": "Changelog Stamp",
                "status": "ok" if stamped else "skipped",
                "detail": f"Stamped [{version}] in CHANGELOG.md" if stamped else "No unreleased changes to stamp",
            })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_package() -> dict:
    """Full package validation for release readiness."""
    checks = {}

    # 1. pyproject.toml exists
    checks["pyproject.toml"] = {
        "status": "pass" if PYPROJECT.exists() else "fail",
        "detail": "Found" if PYPROJECT.exists() else "Missing",
    }

    # 2. Version consistency
    consistent, versions = versions_consistent()
    checks["version_consistency"] = {
        "status": "pass" if consistent else "fail",
        "detail": versions,
    }

    # 3. CHANGELOG.md
    checks["changelog"] = {
        "status": "pass" if CHANGELOG.exists() else "fail",
        "detail": "Found" if CHANGELOG.exists() else "Missing",
    }

    # 4. README.md
    readme = ROOT / "README.md"
    checks["readme"] = {
        "status": "pass" if readme.exists() else "fail",
        "detail": "Found" if readme.exists() else "Missing",
    }

    # 5. LICENSE
    license_file = ROOT / "LICENSE"
    checks["license"] = {
        "status": "pass" if license_file.exists() else "warn",
        "detail": "Found" if license_file.exists() else "Missing (recommended)",
    }

    # 6. Build system
    if PYPROJECT.exists():
        content = PYPROJECT.read_text(encoding="utf-8")
        has_build = "[build-system]" in content
        checks["build_system"] = {
            "status": "pass" if has_build else "fail",
            "detail": "setuptools backend configured" if has_build else "No [build-system] found",
        }

    # 7. Package importable
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import aurora_core; print(aurora_core.__version__)"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        importable = proc.returncode == 0
        checks["importable"] = {
            "status": "pass" if importable else "warn",
            "detail": f"v{proc.stdout.strip()}" if importable else proc.stderr.strip()[:100],
        }
    except Exception as e:
        checks["importable"] = {"status": "warn", "detail": str(e)[:100]}

    # 8. GitHub workflows
    wf_dir = ROOT / ".github" / "workflows"
    release_wf = wf_dir / "release.yml"
    publish_wf = wf_dir / "publish-package.yml"
    checks["release_workflow"] = {
        "status": "pass" if release_wf.exists() else "fail",
        "detail": "release.yml present" if release_wf.exists() else "Missing release.yml",
    }
    checks["publish_workflow"] = {
        "status": "pass" if publish_wf.exists() else "fail",
        "detail": "publish-package.yml present" if publish_wf.exists() else "Missing publish-package.yml",
    }

    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

def print_status():
    """Print comprehensive package/release status."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = get_version_from_pyproject()
    consistent, versions = versions_consistent()
    latest_tag = get_latest_tag()
    commits = get_commits_since_tag(latest_tag)

    print("═" * 70)
    print("  S.L.A.T.E. Package & Release Manager")
    print("═" * 70)
    print(f"  Timestamp     : {now}")
    print(f"  Version       : {version}")
    print(f"  Consistent    : {'✅ Yes' if consistent else '❌ No'}")
    for fname, ver in versions.items():
        print(f"    {fname}: {ver}")
    print(f"  Latest Tag    : {latest_tag or 'None (no releases yet)'}")
    print(f"  Commits Since : {len(commits)}")
    print(f"  Changelog     : {'✅ Found' if CHANGELOG.exists() else '❌ Missing'}")
    print(f"  Dist Dir      : {'✅ Exists' if DIST_DIR.exists() else '— Not built'}")

    if DIST_DIR.exists():
        files = list(DIST_DIR.iterdir())
        if files:
            print(f"  Built Packages:")
            for f in files:
                print(f"    📦 {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Unreleased changes
    unreleased = get_unreleased_changes()
    if unreleased:
        print(f"\n  Unreleased Changes:")
        for line in unreleased.split("\n")[:10]:
            print(f"    {line}")
        remaining = len(unreleased.split("\n")) - 10
        if remaining > 0:
            print(f"    ... and {remaining} more lines")

    # Recent commits
    if commits:
        print(f"\n  Recent Commits (since {latest_tag or 'beginning'}):")
        for c in commits[:8]:
            print(f"    {c['sha']} {c['message']}")
        if len(commits) > 8:
            print(f"    ... and {len(commits) - 8} more")

    print("═" * 70)


def print_validation():
    """Print validation results."""
    checks = validate_package()
    icons = {"pass": "✅", "fail": "❌", "warn": "⚠️"}

    print("═" * 70)
    print("  Package Validation")
    print("═" * 70)

    all_pass = True
    for name, result in checks.items():
        icon = icons.get(result["status"], "?")
        print(f"  {icon} {name}: {result['detail']}")
        if result["status"] == "fail":
            all_pass = False

    print("─" * 70)
    if all_pass:
        print("  ✅ Package is ready for release!")
    else:
        print("  ❌ Fix the above issues before releasing.")
    print("═" * 70)


def print_deps():
    """Print dependency audit."""
    deps = audit_dependencies()

    print("═" * 70)
    print("  Dependency Audit")
    print("═" * 70)

    if deps["core"]:
        print("\n  Core Dependencies:")
        for d in deps["core"]:
            icon = "✅" if d["status"] == "installed" else "❌"
            print(f"    {icon} {d['name']}: {d['installed']} (requires {d['required']})")

    if deps["missing"]:
        print(f"\n  ❌ Missing: {', '.join(deps['missing'])}")
        print(f"     Install: pip install {' '.join(deps['missing'])}")

    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="S.L.A.T.E. Package & Release Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  slate-package --status                Show package/release status
  slate-package --bump patch            Bump version 2.4.0 → 2.4.1
  slate-package --bump minor            Bump version 2.4.0 → 2.5.0
  slate-package --build                 Build sdist + wheel
  slate-package --validate              Validate release readiness
  slate-package --release 2.5.0         Prepare release (dry run)
  slate-package --release 2.5.0 --go    Prepare release (apply changes)
  slate-package --deps                  Audit dependencies
  slate-package --changelog             Show current changelog entry
        """,
    )
    parser.add_argument("--status", action="store_true", help="Show package status")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], help="Bump version")
    parser.add_argument("--build", action="store_true", help="Build package distributions")
    parser.add_argument("--validate", action="store_true", help="Validate package readiness")
    parser.add_argument("--release", metavar="VERSION", help="Prepare a release")
    parser.add_argument("--go", action="store_true", help="Apply release changes (not dry run)")
    parser.add_argument("--changelog", action="store_true", help="Show changelog entry")
    parser.add_argument("--deps", action="store_true", help="Audit dependencies")

    args = parser.parse_args()

    if args.status or not any([args.bump, args.build, args.validate, args.release, args.changelog, args.deps]):
        print_status()
        return

    if args.bump:
        current = get_version_from_pyproject()
        new_ver = bump_version(current, args.bump)
        print(f"Bumping version: {current} → {new_ver}")
        updated = apply_version(new_ver)
        for f in updated:
            print(f"  ✅ Updated {f}")
        print(f"\nVersion is now {new_ver}")
        print("Don't forget to update CHANGELOG.md!")

    if args.build:
        print("Building package...")
        result = build_package()
        if result["success"]:
            print("✅ Build successful!")
            for f in result["files"]:
                print(f"  📦 {f['name']} ({f['size_human']})")
        else:
            print("❌ Build failed:")
            for err in result["errors"]:
                print(f"  {err}")

    if args.validate:
        print_validation()

    if args.release:
        dry_run = not args.go
        print(f"{'[DRY RUN] ' if dry_run else ''}Preparing release v{args.release}...")
        result = prepare_release(args.release, dry_run=dry_run)
        for step in result["steps"]:
            icon = {"ok": "✅", "warning": "⚠️", "error": "❌", "missing": "❌", "skipped": "⏭️"}.get(
                step["status"], "?"
            )
            print(f"  {icon} {step['name']}: {step['detail']}")
        if dry_run:
            print(f"\n  Run with --go to apply changes")

    if args.changelog:
        version = get_version_from_pyproject()
        entry = get_changelog_entry(version)
        if entry:
            print(f"Changelog for v{version}:\n")
            print(entry)
        else:
            unreleased = get_unreleased_changes()
            if unreleased:
                print(f"No entry for v{version}. Unreleased changes:\n")
                print(unreleased)
            else:
                print("No changelog entries found.")

    if args.deps:
        print_deps()


if __name__ == "__main__":
    main()
