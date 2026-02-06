#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Modified: 2026-02-07T05:00:00Z | Author: COPILOT | Change: Initial creation
# Purpose: Local release & package runner — creates GitHub releases and publishes
#          packages directly from the self-hosted runner machine via GitHub API
# ═══════════════════════════════════════════════════════════════════════════════
"""
S.L.A.T.E. Release Runner
==========================

Runs the full release + publish pipeline locally on the self-hosted runner,
bypassing the need for GitHub-hosted ubuntu-latest runners.

Features:
  - Extracts GitHub credentials from Windows Credential Manager (via git)
  - Creates GitHub releases with auto-generated release notes
  - Uploads wheel + sdist assets to releases
  - Publishes packages to GitHub Packages (PyPI)
  - Full release orchestration: validate → build → tag → release → publish

Usage:
    python slate/slate_release_runner.py --status
    python slate/slate_release_runner.py --build
    python slate/slate_release_runner.py --create-release 2.4.0
    python slate/slate_release_runner.py --publish
    python slate/slate_release_runner.py --full-release 2.4.0
    python slate/slate_release_runner.py --full-release 2.4.0 --prerelease
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = WORKSPACE_ROOT / "dist"
PYPROJECT = WORKSPACE_ROOT / "pyproject.toml"
INIT_FILE = WORKSPACE_ROOT / "slate" / "__init__.py"
CHANGELOG = WORKSPACE_ROOT / "CHANGELOG.md"

GITHUB_ORG = "SynchronizedLivingArchitecture"
GITHUB_REPO = "S.L.A.T.E."
GITHUB_API = "https://api.github.com"
GITHUB_UPLOAD = "https://uploads.github.com"
GITHUB_PACKAGES_URL = "https://upload.pypi.org/legacy/"

logger = logging.getLogger("slate.release_runner")


# ═══════════════════════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════════════════════

def get_github_token() -> Optional[str]:
    """
    Get GitHub token from multiple sources (in priority order):
    1. GITHUB_TOKEN / GITHUB_PAT / GH_TOKEN env vars
    2. Windows Credential Manager via git credential fill
    3. .slate/credentials file
    """
    # 1. Environment variables
    for var in ("GITHUB_TOKEN", "GITHUB_PAT", "GH_TOKEN"):
        token = os.environ.get(var)
        if token:
            logger.debug("Token found from env: %s", var)
            return token

    # 2. Git credential manager (works with Windows Credential Manager)
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(WORKSPACE_ROOT),
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                if line.startswith("password="):
                    token = line.split("=", 1)[1].strip()
                    if token:
                        logger.debug("Token found from git credential manager")
                        return token
    except Exception as e:
        logger.debug("Git credential fill failed: %s", e)

    # 3. .slate/credentials file
    creds_file = Path.home() / ".slate" / "credentials"
    if creds_file.exists():
        try:
            for line in creds_file.read_text(encoding="utf-8").strip().split("\n"):
                if line.startswith("GITHUB_PAT="):
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass

    return None


def verify_token(token: str) -> Dict[str, Any]:
    """Verify a GitHub token and return user info + scopes."""
    try:
        req = urllib.request.Request(
            f"{GITHUB_API}/user",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "SLATE-Release-Runner/2.4",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            scopes = resp.headers.get("X-OAuth-Scopes", "")
            data = json.loads(resp.read())
            return {
                "authenticated": True,
                "username": data.get("login", "unknown"),
                "scopes": [s.strip() for s in scopes.split(",") if s.strip()],
                "has_repo": "repo" in scopes,
                "has_packages": "write:packages" in scopes or "packages" in scopes,
            }
    except Exception as e:
        return {"authenticated": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# GitHub API Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def github_api(
    method: str,
    endpoint: str,
    token: str,
    data: Optional[Dict] = None,
    base_url: Optional[str] = None,
    content_type: str = "application/json",
    raw_data: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Make a GitHub API request."""
    url = f"{base_url or GITHUB_API}{endpoint}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SLATE-Release-Runner/2.4",
    }

    body = None
    if raw_data is not None:
        body = raw_data
        headers["Content-Type"] = content_type
    elif data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = resp.read()
            return {
                "status": resp.status,
                "data": json.loads(resp_data) if resp_data else {},
                "success": True,
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return {
            "status": e.code,
            "error": error_body,
            "success": False,
        }
    except Exception as e:
        return {"status": 0, "error": str(e), "success": False}


# ═══════════════════════════════════════════════════════════════════════════════
# Version Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_version() -> str:
    """Get current version from pyproject.toml."""
    if PYPROJECT.exists():
        content = PYPROJECT.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return "unknown"


def get_changelog_entry(version: str) -> Optional[str]:
    """Extract changelog entry for a version."""
    if not CHANGELOG.exists():
        return None
    content = CHANGELOG.read_text(encoding="utf-8")
    pattern = rf"## \[{re.escape(version)}\].*?\n(.*?)(?=\n## \[|$)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None


def tag_exists(tag: str) -> bool:
    """Check if a git tag already exists."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", f"refs/tags/{tag}"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Package Building
# ═══════════════════════════════════════════════════════════════════════════════

def build_package() -> Dict[str, Any]:
    """Build sdist and wheel distributions."""
    result: Dict[str, Any] = {"success": False, "files": [], "errors": []}

    # Clean dist/
    if DIST_DIR.exists():
        for f in DIST_DIR.iterdir():
            if f.suffix in (".whl", ".gz"):
                f.unlink()

    # Build
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "build"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            result["errors"].append(f"Build failed:\n{proc.stderr}")
            return result
    except FileNotFoundError:
        result["errors"].append("'build' not installed. Run: pip install build")
        return result

    # Validate with twine
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "twine", "check"] + [
                str(f) for f in DIST_DIR.iterdir() if f.suffix in (".whl", ".gz")
            ],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            result["errors"].append(f"twine check warning: {proc.stdout}")
    except FileNotFoundError:
        pass  # twine is optional

    # Collect results
    if DIST_DIR.exists():
        for f in DIST_DIR.iterdir():
            if f.suffix in (".whl", ".gz"):
                result["files"].append({
                    "path": str(f),
                    "name": f.name,
                    "size": f.stat().st_size,
                    "size_human": f"{f.stat().st_size / 1024:.1f} KB",
                })

    result["success"] = len(result["files"]) > 0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GitHub Release Management
# ═══════════════════════════════════════════════════════════════════════════════

def list_releases(token: str, per_page: int = 5) -> List[Dict]:
    """List existing releases."""
    resp = github_api(
        "GET",
        f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases?per_page={per_page}",
        token,
    )
    if resp["success"]:
        return resp["data"]
    return []


def create_release(
    token: str,
    version: str,
    body: str = "",
    prerelease: bool = False,
    draft: bool = False,
    target: str = "main",
) -> Dict[str, Any]:
    """Create a GitHub release."""
    tag_name = f"v{version}" if not version.startswith("v") else version

    # Create tag locally first
    if not tag_exists(tag_name):
        print(f"  Creating git tag {tag_name}...")
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            timeout=10,
        )
        # Push the tag
        push_result = subprocess.run(
            ["git", "push", "origin", tag_name],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if push_result.returncode != 0:
            return {"success": False, "error": f"Failed to push tag: {push_result.stderr}"}
    else:
        print(f"  Tag {tag_name} already exists")

    # Create the release via API
    release_data = {
        "tag_name": tag_name,
        "target_commitish": target,
        "name": f"S.L.A.T.E. {tag_name}",
        "body": body or f"Release {tag_name}\n\nSee [CHANGELOG.md](https://github.com/{GITHUB_ORG}/{GITHUB_REPO}/blob/main/CHANGELOG.md) for details.",
        "draft": draft,
        "prerelease": prerelease,
        "generate_release_notes": True,
    }

    resp = github_api(
        "POST",
        f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases",
        token,
        data=release_data,
    )

    if resp["success"]:
        release = resp["data"]
        return {
            "success": True,
            "id": release["id"],
            "url": release["html_url"],
            "upload_url": release["upload_url"],
            "tag": tag_name,
        }
    else:
        return {"success": False, "error": resp.get("error", "Unknown error")}


def upload_release_asset(
    token: str,
    release_id: int,
    upload_url: str,
    file_path: Path,
) -> Dict[str, Any]:
    """Upload a file as a release asset."""
    # Parse the upload URL template
    base_url = upload_url.split("{")[0]
    url = f"{base_url}?name={urllib.parse.quote(file_path.name)}"

    # Determine content type
    if file_path.suffix == ".whl":
        content_type = "application/zip"
    elif file_path.name.endswith(".tar.gz"):
        content_type = "application/gzip"
    else:
        content_type = "application/octet-stream"

    file_data = file_path.read_bytes()

    headers = {
        "Authorization": f"token {token}",
        "Content-Type": content_type,
        "Content-Length": str(len(file_data)),
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SLATE-Release-Runner/2.4",
    }

    req = urllib.request.Request(url, data=file_data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return {
                "success": True,
                "name": data.get("name"),
                "size": data.get("size"),
                "download_url": data.get("browser_download_url"),
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return {"success": False, "error": f"HTTP {e.code}: {error_body[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_release(token: str, release_id: int) -> bool:
    """Delete a release (for cleanup/retry)."""
    resp = github_api(
        "DELETE",
        f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases/{release_id}",
        token,
    )
    return resp.get("success", False) or resp.get("status") == 204


# ═══════════════════════════════════════════════════════════════════════════════
# Package Publishing
# ═══════════════════════════════════════════════════════════════════════════════

def publish_to_github_packages(token: str) -> Dict[str, Any]:
    """Publish built packages to GitHub Packages (PyPI registry)."""
    result: Dict[str, Any] = {"success": False, "uploaded": [], "errors": []}

    if not DIST_DIR.exists():
        result["errors"].append("No dist/ directory. Run --build first.")
        return result

    dist_files = [f for f in DIST_DIR.iterdir() if f.suffix in (".whl", ".gz")]
    if not dist_files:
        result["errors"].append("No packages found in dist/. Run --build first.")
        return result

    # Use twine to upload to GitHub Packages
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "twine", "upload",
                "--repository-url", GITHUB_PACKAGES_URL,
                "--username", "__token__",
                "--password", token,
                "--skip-existing",
            ] + [str(f) for f in dist_files],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode == 0:
            result["success"] = True
            for f in dist_files:
                result["uploaded"].append(f.name)
        else:
            # Check if it's a "skip existing" message (which is OK)
            if "already exists" in proc.stdout.lower() or "skipping" in proc.stdout.lower():
                result["success"] = True
                result["uploaded"] = [f.name for f in dist_files]
                result["errors"].append("Some packages already existed (skipped)")
            else:
                result["errors"].append(f"twine upload failed:\n{proc.stderr}\n{proc.stdout}")
    except FileNotFoundError:
        result["errors"].append("'twine' not installed. Run: pip install twine")
    except subprocess.TimeoutExpired:
        result["errors"].append("Upload timed out after 120s")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Full Release Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def full_release(
    version: str,
    prerelease: bool = False,
    skip_build: bool = False,
    skip_publish: bool = False,
    target_branch: str = "main",
) -> Dict[str, Any]:
    """
    Execute a full release pipeline locally:
    1. Validate version + token
    2. Build package (sdist + wheel)
    3. Create git tag + push
    4. Create GitHub release
    5. Upload assets to release
    6. Publish to GitHub Packages
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result: Dict[str, Any] = {
        "version": version,
        "timestamp": now,
        "steps": [],
        "success": True,
    }

    def step(name: str, status: str, detail: str) -> None:
        icon = {"ok": "OK", "fail": "FAIL", "skip": "SKIP", "warn": "WARN"}[status]
        result["steps"].append({"name": name, "status": status, "detail": detail})
        print(f"  [{icon}] {name}: {detail}")
        if status == "fail":
            result["success"] = False

    print("=" * 70)
    print(f"  S.L.A.T.E. Release Runner — v{version}")
    print(f"  {now}")
    print("=" * 70)

    # Step 1: Get token
    token = get_github_token()
    if not token:
        step("Authentication", "fail", "No GitHub token found. Set GITHUB_TOKEN or configure git credential manager")
        return result
    auth_info = verify_token(token)
    if not auth_info.get("authenticated"):
        step("Authentication", "fail", f"Token invalid: {auth_info.get('error', 'unknown')}")
        return result
    step("Authentication", "ok", f"Authenticated as {auth_info['username']} (scopes: {', '.join(auth_info.get('scopes', [])[:5])})")

    # Step 2: Validate version
    current = get_version()
    tag_name = f"v{version}"
    if tag_exists(tag_name):
        step("Version Check", "fail", f"Tag {tag_name} already exists. Use a new version.")
        return result
    step("Version Check", "ok", f"Current: {current}, Release: {version}, Tag: {tag_name}")

    # Step 3: Build package
    if not skip_build:
        print("\n  Building package...")
        build_result = build_package()
        if build_result["success"]:
            files_str = ", ".join(f["name"] for f in build_result["files"])
            step("Build Package", "ok", f"Built: {files_str}")
        else:
            step("Build Package", "fail", "; ".join(build_result["errors"]))
            return result
    else:
        # Check existing dist/
        dist_files = list(DIST_DIR.iterdir()) if DIST_DIR.exists() else []
        if dist_files:
            step("Build Package", "skip", f"Using existing: {', '.join(f.name for f in dist_files)}")
        else:
            step("Build Package", "fail", "No existing packages and --skip-build specified")
            return result

    # Step 4: Get changelog
    changelog_body = get_changelog_entry(version)
    if not changelog_body:
        changelog_body = get_changelog_entry("Unreleased")
    release_body = f"""## S.L.A.T.E. {tag_name}

{changelog_body or 'See CHANGELOG.md for details.'}

---

### Installation

**From source:**
```bash
git clone https://github.com/{GITHUB_ORG}/{GITHUB_REPO}.git
cd {GITHUB_REPO}
pip install -e .
```

**Quick install:**
```bash
python install_slate.py
```

### System Requirements
- Python 3.11+
- Windows 10/11 (primary), Linux (secondary)
- NVIDIA GPU recommended (CUDA 12.x)
"""
    step("Changelog", "ok" if changelog_body else "warn",
         f"Found entry for {version}" if changelog_body else "No changelog entry, using default")

    # Step 5: Create release
    print("\n  Creating GitHub release...")
    release_resp = create_release(
        token=token,
        version=version,
        body=release_body,
        prerelease=prerelease,
        target=target_branch,
    )
    if release_resp["success"]:
        step("Create Release", "ok", f"{release_resp['url']}")
    else:
        step("Create Release", "fail", release_resp.get("error", "Unknown error"))
        return result

    # Step 6: Upload assets
    dist_files = [f for f in DIST_DIR.iterdir() if f.suffix in (".whl", ".gz")]
    upload_count = 0
    for dist_file in dist_files:
        print(f"  Uploading {dist_file.name}...")
        upload_resp = upload_release_asset(
            token=token,
            release_id=release_resp["id"],
            upload_url=release_resp["upload_url"],
            file_path=dist_file,
        )
        if upload_resp["success"]:
            upload_count += 1
        else:
            print(f"    Warning: Upload failed for {dist_file.name}: {upload_resp.get('error')}")

    if upload_count > 0:
        step("Upload Assets", "ok", f"{upload_count}/{len(dist_files)} files uploaded")
    elif dist_files:
        step("Upload Assets", "warn", "No assets uploaded successfully")
    else:
        step("Upload Assets", "skip", "No dist files to upload")

    # Step 7: Publish to GitHub Packages
    if not skip_publish:
        print("\n  Publishing to GitHub Packages...")
        pub_result = publish_to_github_packages(token)
        if pub_result["success"]:
            step("Publish Package", "ok", f"Published: {', '.join(pub_result['uploaded'])}")
        else:
            step("Publish Package", "warn", "; ".join(pub_result["errors"]))
    else:
        step("Publish Package", "skip", "Skipped (--skip-publish)")

    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for s in result["steps"] if s["status"] in ("ok", "skip"))
    total = len(result["steps"])
    print(f"  Release {'COMPLETE' if result['success'] else 'FAILED'}: {passed}/{total} steps passed")
    if result["success"] and release_resp.get("url"):
        print(f"  Release URL: {release_resp['url']}")
    print("=" * 70)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Status Display
# ═══════════════════════════════════════════════════════════════════════════════

def print_status() -> None:
    """Print release runner status."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = get_version()

    print("=" * 70)
    print("  S.L.A.T.E. Release Runner Status")
    print("=" * 70)
    print(f"  Timestamp  : {now}")
    print(f"  Version    : {version}")
    print(f"  Workspace  : {WORKSPACE_ROOT}")

    # Token check
    token = get_github_token()
    if token:
        auth = verify_token(token)
        if auth.get("authenticated"):
            print(f"  Auth       : [OK] {auth['username']}")
            print(f"  Scopes     : {', '.join(auth.get('scopes', []))}")
            has_repo = auth.get("has_repo", False)
            has_pkg = auth.get("has_packages", False)
            print(f"  Repo Scope : {'[OK]' if has_repo else '[--]'}")
            print(f"  Pkg Scope  : {'[OK]' if has_pkg else '[--]'}")
        else:
            print(f"  Auth       : [FAIL] {auth.get('error', 'unknown')}")
    else:
        print("  Auth       : [--] No token found")

    # Build tools
    tools = {}
    for mod in ("build", "twine", "setuptools", "wheel"):
        try:
            __import__(mod)
            tools[mod] = True
        except ImportError:
            tools[mod] = False
    tools_str = ", ".join(f"{k}: {'OK' if v else 'MISSING'}" for k, v in tools.items())
    print(f"  Build Tools: {tools_str}")

    # Dist files
    if DIST_DIR.exists():
        dist_files = [f for f in DIST_DIR.iterdir() if f.suffix in (".whl", ".gz")]
        if dist_files:
            print(f"  Packages   : {len(dist_files)} built")
            for f in dist_files:
                print(f"    {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        else:
            print("  Packages   : None built")
    else:
        print("  Packages   : No dist/ directory")

    # Git tags
    try:
        proc = subprocess.run(
            ["git", "tag", "-l", "v*", "--sort=-v:refname"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        tags = [t for t in proc.stdout.strip().split("\n") if t.strip()]
        if tags:
            print(f"  Tags       : {len(tags)} ({tags[0]} latest)")
        else:
            print("  Tags       : None")
    except Exception:
        print("  Tags       : [error reading]")

    # Remote releases (if token available)
    if token:
        releases = list_releases(token, per_page=3)
        if releases:
            print(f"  Releases   : {len(releases)} (showing latest)")
            for r in releases[:3]:
                pre = " [pre]" if r.get("prerelease") else ""
                draft = " [draft]" if r.get("draft") else ""
                print(f"    {r['tag_name']}{pre}{draft} — {r.get('published_at', 'N/A')[:10]}")
        else:
            print("  Releases   : None published")

    # Runner info
    runner_dir = WORKSPACE_ROOT / "actions-runner"
    runner_cfg = runner_dir / ".runner"
    if runner_cfg.exists():
        try:
            cfg = json.loads(runner_cfg.read_text(encoding="utf-8"))
            print(f"  Runner     : {cfg.get('agentName', 'unknown')}")
            print(f"  Runner Dir : {runner_dir}")
        except Exception:
            print(f"  Runner     : Configured at {runner_dir}")
    else:
        print("  Runner     : Not found at workspace")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S.L.A.T.E. Release Runner — local release & package management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --status                       Show release runner status
  %(prog)s --build                        Build sdist + wheel
  %(prog)s --create-release 2.4.0         Create release + upload assets
  %(prog)s --publish                      Publish packages to GitHub Packages
  %(prog)s --full-release 2.4.0           Full pipeline: build → release → publish
  %(prog)s --full-release 2.5.0-beta.1 --prerelease
  %(prog)s --list-releases                List existing releases
        """,
    )
    parser.add_argument("--status", action="store_true", help="Show release runner status")
    parser.add_argument("--build", action="store_true", help="Build sdist + wheel")
    parser.add_argument("--create-release", metavar="VERSION", help="Create a GitHub release")
    parser.add_argument("--publish", action="store_true", help="Publish packages to GitHub Packages")
    parser.add_argument("--full-release", metavar="VERSION", help="Full release pipeline")
    parser.add_argument("--list-releases", action="store_true", help="List existing releases")
    parser.add_argument("--prerelease", action="store_true", help="Mark as pre-release")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step (use existing dist/)")
    parser.add_argument("--skip-publish", action="store_true", help="Skip package publishing")
    parser.add_argument("--target", default="main", help="Target branch for release (default: main)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.status or not any([
        args.build, args.create_release, args.publish,
        args.full_release, args.list_releases,
    ]):
        print_status()
        return

    if args.list_releases:
        token = get_github_token()
        if not token:
            print("[FAIL] No GitHub token available")
            sys.exit(1)
        releases = list_releases(token, per_page=10)
        if not releases:
            print("No releases found.")
        else:
            print(f"\n{'Tag':<20} {'Published':<12} {'Pre':<5} {'Assets'}")
            print("-" * 55)
            for r in releases:
                pre = "Yes" if r.get("prerelease") else "No"
                pub = r.get("published_at", "N/A")[:10]
                assets = len(r.get("assets", []))
                print(f"  {r['tag_name']:<18} {pub:<12} {pre:<5} {assets}")
        return

    if args.build:
        print("Building package...")
        result = build_package()
        if result["success"]:
            print("  Build successful!")
            for f in result["files"]:
                print(f"  {f['name']} ({f['size_human']})")
        else:
            print("  Build failed:")
            for err in result["errors"]:
                print(f"    {err}")
            sys.exit(1)
        return

    if args.create_release:
        token = get_github_token()
        if not token:
            print("[FAIL] No GitHub token available")
            sys.exit(1)
        version = args.create_release
        changelog = get_changelog_entry(version) or ""
        print(f"Creating release v{version}...")
        resp = create_release(
            token=token,
            version=version,
            body=changelog,
            prerelease=args.prerelease,
            target=args.target,
        )
        if resp["success"]:
            print(f"  Release created: {resp['url']}")
            # Upload any existing dist files
            if DIST_DIR.exists():
                dist_files = [f for f in DIST_DIR.iterdir() if f.suffix in (".whl", ".gz")]
                for df in dist_files:
                    print(f"  Uploading {df.name}...")
                    up = upload_release_asset(token, resp["id"], resp["upload_url"], df)
                    if up["success"]:
                        print(f"    Uploaded: {up['download_url']}")
                    else:
                        print(f"    Failed: {up.get('error')}")
        else:
            print(f"  Failed: {resp.get('error')}")
            sys.exit(1)
        return

    if args.publish:
        token = get_github_token()
        if not token:
            print("[FAIL] No GitHub token available")
            sys.exit(1)
        print("Publishing to GitHub Packages...")
        result = publish_to_github_packages(token)
        if result["success"]:
            print("  Published successfully!")
            for f in result["uploaded"]:
                print(f"    {f}")
        else:
            print("  Publish failed:")
            for err in result["errors"]:
                print(f"    {err}")
            sys.exit(1)
        return

    if args.full_release:
        result = full_release(
            version=args.full_release,
            prerelease=args.prerelease,
            skip_build=args.skip_build,
            skip_publish=args.skip_publish,
            target_branch=args.target,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
