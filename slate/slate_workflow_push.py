#!/usr/bin/env python3
"""
SLATE Workflow Push Helper

Handles pushing workflow files to GitHub with proper OAuth scope.
GitHub requires the 'workflow' scope to push changes to .github/workflows/.
This script detects when workflow files need pushing and handles authentication.

Usage:
    python slate/slate_workflow_push.py          # Auto-detect and push
    python slate/slate_workflow_push.py --check  # Check if workflow scope available
    python slate/slate_workflow_push.py --auth   # Re-authenticate with workflow scope
    python slate/slate_workflow_push.py --clear  # Clear cached credentials
"""

import subprocess
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def run_command(cmd: list[str] | str, capture: bool = True, shell: bool = False) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        if isinstance(cmd, str):
            shell = True
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            shell=shell
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def get_gh_path() -> str | None:
    """Find the gh CLI executable."""
    if sys.platform == "win32":
        # Common Windows paths for gh
        common_paths = [
            Path.home() / "AppData/Local/Programs/GitHub CLI/gh.exe",
            Path("C:/Program Files/GitHub CLI/gh.exe"),
            Path("C:/Program Files (x86)/GitHub CLI/gh.exe"),
        ]
        for p in common_paths:
            if p.exists():
                return str(p)

    # Try PATH
    code, out, _ = run_command(["where" if sys.platform == "win32" else "which", "gh"])
    if code == 0 and out.strip():
        return out.strip().split("\n")[0]

    return None


def check_workflow_scope() -> bool:
    """Check if current GitHub auth has workflow scope."""
    gh = get_gh_path()
    if not gh:
        print("❌ GitHub CLI (gh) not found. Install from: https://cli.github.com/")
        return False

    code, out, err = run_command([gh, "auth", "status"])
    if code != 0:
        print(f"❌ Not authenticated with GitHub CLI: {err}")
        return False

    # Check scopes
    code, out, _ = run_command([gh, "api", "user", "-H", "Accept: application/vnd.github+json"])
    if code != 0:
        # Check the X-OAuth-Scopes header
        code, out, _ = run_command([gh, "api", "-i", "user"])
        if "workflow" in out.lower():
            return True
        return False

    return True


def authenticate_with_workflow_scope() -> bool:
    """Re-authenticate with workflow scope."""
    gh = get_gh_path()
    if not gh:
        print("❌ GitHub CLI (gh) not found.")
        print("   Install from: https://cli.github.com/")
        return False

    print("🔐 Authenticating with GitHub (workflow scope required)...")
    print("   This will open a browser for authentication.")
    print()

    # Run interactive auth with workflow scope
    cmd = [gh, "auth", "login", "-s", "workflow", "-w"]
    code, _, err = run_command(cmd)

    if code != 0:
        print(f"❌ Authentication failed: {err}")
        return False

    print("✅ Authentication successful with workflow scope!")
    return True


def get_staged_workflow_files() -> list[str]:
    """Get list of staged workflow files."""
    code, out, _ = run_command(["git", "diff", "--cached", "--name-only"])
    if code != 0:
        return []

    return [f for f in out.strip().split("\n") if f.startswith(".github/workflows/")]


def get_modified_workflow_files() -> list[str]:
    """Get list of modified (unstaged) workflow files."""
    code, out, _ = run_command(["git", "status", "--porcelain"])
    if code != 0:
        return []

    files = []
    for line in out.strip().split("\n"):
        if not line:
            continue
        status = line[:2]
        filepath = line[3:]
        if filepath.startswith(".github/workflows/"):
            files.append(filepath)
    return files


def clear_cached_credentials() -> bool:
    """Clear cached GitHub credentials from Windows Credential Manager."""
    if sys.platform != "win32":
        print("ℹ️  Credential clearing only needed on Windows")
        return True

    print("🔐 Clearing cached GitHub credentials...")

    # Clear git credential for github.com
    code, _, _ = run_command("cmdkey /delete:git:https://github.com", shell=True)

    # Also try the generic github.com entry
    run_command("cmdkey /delete:github.com", shell=True)

    # Clear git credential cache
    run_command(["git", "credential", "reject"], shell=False)

    print("✅ Cached credentials cleared")
    print()
    print("Next push will prompt for new credentials.")
    print("When prompted, use a Personal Access Token (PAT) with 'workflow' scope:")
    print("  1. Go to: https://github.com/settings/tokens")
    print("  2. Generate new token (classic)")
    print("  3. Select scopes: repo, workflow")
    print("  4. Use the token as your password when git prompts")
    print()
    return True


def get_stored_token() -> str | None:
    """Get stored GitHub PAT from environment or .slate config."""
    # Check environment variables
    token = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token

    # Check .slate/credentials file
    creds_file = Path.home() / ".slate" / "credentials"
    if creds_file.exists():
        try:
            content = creds_file.read_text().strip()
            for line in content.split("\n"):
                if line.startswith("GITHUB_PAT="):
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass

    return None


def push_with_token(token: str) -> bool:
    """Push using a specific token by modifying the remote URL temporarily."""
    # Get current remote URL
    code, url, _ = run_command(["git", "remote", "get-url", "origin"])
    if code != 0:
        return False

    url = url.strip()
    if not url.startswith("https://"):
        print("⚠️  Token auth only works with HTTPS remotes")
        return False

    # Create authenticated URL
    # https://github.com/... -> https://x-access-token:TOKEN@github.com/...
    auth_url = url.replace("https://", f"https://x-access-token:{token}@")

    # Push with authenticated URL
    code, _, err = run_command(["git", "push", auth_url, "HEAD"])
    if code == 0:
        return True

    print(f"❌ Push with token failed: {err}")
    return False


def push_with_retry() -> bool:
    """Attempt to push, handling workflow scope errors."""
    print("📤 Pushing to remote...")

    code, out, err = run_command(["git", "push"])

    if code == 0:
        print("✅ Push successful!")
        return True

    if "workflow" in err.lower() and "scope" in err.lower():
        print("⚠️  Push failed - workflow scope required for workflow files")
        print()

        # Try stored token first
        token = get_stored_token()
        if token:
            print("🔐 Using stored GitHub PAT...")
            if push_with_token(token):
                print("✅ Push successful!")
                return True
            print("⚠️  Stored token didn't work, trying alternatives...")
            print()

        print("The cached credentials don't have 'workflow' scope.")
        print()

        # Check if gh is available
        gh = get_gh_path()
        if gh:
            response = input("🔐 Re-authenticate with GitHub CLI? [Y/n]: ").strip().lower()
            if response in ("", "y", "yes"):
                if authenticate_with_workflow_scope():
                    print()
                    print("📤 Retrying push...")
                    code, _, err = run_command(["git", "push"])
                    if code == 0:
                        print("✅ Push successful!")
                        return True
                    else:
                        print(f"❌ Push still failed: {err}")
        else:
            print("GitHub CLI not found. Options:")
            print()
            print("  Option 1 - Set GITHUB_PAT environment variable:")
            print("    1. Create PAT at: https://github.com/settings/tokens")
            print("       - Select scopes: repo, workflow")
            print("    2. Set: $env:GITHUB_PAT = 'your_token'")
            print("    3. Retry: python slate/slate_workflow_push.py")
            print()
            print("  Option 2 - Store token in ~/.slate/credentials:")
            print("    1. Create file: ~/.slate/credentials")
            print("    2. Add line: GITHUB_PAT=your_token")
            print("    3. Retry: python slate/slate_workflow_push.py")
            print()
            print("  Option 3 - Install GitHub CLI:")
            print("    1. Install from: https://cli.github.com/")
            print("    2. Run: gh auth login -s workflow -w")
            print("    3. Retry: git push")
            print()

            response = input("🔐 Clear cached credentials now? [Y/n]: ").strip().lower()
            if response in ("", "y", "yes"):
                clear_cached_credentials()
                print("📤 Retrying push (will prompt for credentials)...")
                code, _, err = run_command(["git", "push"])
                if code == 0:
                    print("✅ Push successful!")
                    return True

        return False

    print(f"❌ Push failed: {err}")
    return False


def save_token(token: str) -> bool:
    """Save GitHub PAT to ~/.slate/credentials."""
    creds_dir = Path.home() / ".slate"
    creds_file = creds_dir / "credentials"

    try:
        creds_dir.mkdir(exist_ok=True)
        creds_file.write_text(f"GITHUB_PAT={token}\n")
        print(f"✅ Token saved to {creds_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to save token: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SLATE Workflow Push Helper")
    parser.add_argument("--check", action="store_true", help="Check workflow scope status")
    parser.add_argument("--auth", action="store_true", help="Re-authenticate with workflow scope")
    parser.add_argument("--clear", action="store_true", help="Clear cached credentials")
    parser.add_argument("--push", action="store_true", help="Push changes (handles workflow scope)")
    parser.add_argument("--save-token", metavar="TOKEN", help="Save a GitHub PAT for workflow pushes")
    args = parser.parse_args()

    if args.check:
        gh = get_gh_path()
        if gh:
            print(f"✅ GitHub CLI found: {gh}")
            if check_workflow_scope():
                print("✅ Workflow scope available")
            else:
                print("⚠️  Workflow scope may not be available")
                print("   Run: python slate/slate_workflow_push.py --auth")
        else:
            print("❌ GitHub CLI not found")
            print("   Install from: https://cli.github.com/")
            print()
            print("Alternative: Use Personal Access Token (PAT)")
            print("   1. Run: python slate/slate_workflow_push.py --clear")
            print("   2. Create PAT at: https://github.com/settings/tokens")
            print("   3. Select scopes: repo, workflow")
            print("   4. Use PAT as password when git prompts")
        return

    if args.auth:
        gh = get_gh_path()
        if gh:
            authenticate_with_workflow_scope()
        else:
            print("❌ GitHub CLI not found")
            print()
            print("Use Personal Access Token instead:")
            clear_cached_credentials()
        return

    if args.clear:
        clear_cached_credentials()
        return

    if args.save_token:
        save_token(args.save_token)
        return

    # Default: check for workflow files and push
    workflow_files = get_modified_workflow_files()
    if workflow_files:
        print(f"📁 Found {len(workflow_files)} modified workflow files:")
        for f in workflow_files[:5]:
            print(f"   - {f}")
        if len(workflow_files) > 5:
            print(f"   ... and {len(workflow_files) - 5} more")
        print()

    push_with_retry()


if __name__ == "__main__":
    main()
