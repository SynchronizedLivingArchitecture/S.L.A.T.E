"""
SLATE Runner Fallback Manager

Manages intelligent fallback between self-hosted and GitHub-hosted runners.
Priority: self-hosted (free) -> GitHub-hosted (paid, fallback only)

Usage:
    python slate/runner_fallback.py --status     # Check runner availability
    python slate/runner_fallback.py --config     # Generate workflow config
    python slate/runner_fallback.py --enforce    # Enforce self-hosted only mode
    python slate/runner_fallback.py --allow-fallback  # Enable fallback mode
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
CONFIG_FILE = WORKSPACE / ".slate_runner_config.json"

DEFAULT_CONFIG = {
    "mode": "self-hosted-only",  # or "fallback-enabled"
    "self_hosted_labels": ["self-hosted", "slate"],
    "fallback_runner": "ubuntu-latest",
    "fallback_timeout_minutes": 5,
    "cost_tracking": {
        "enabled": True,
        "monthly_budget_usd": 0,  # 0 = no hosted usage allowed by default
        "current_month_usage_usd": 0,
    },
    "last_updated": None,
}


def load_config() -> dict:
    """Load runner configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Save runner configuration."""
    config["last_updated"] = datetime.now().isoformat()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {CONFIG_FILE}")


def get_gh_path() -> str:
    """Get path to gh CLI."""
    gh_paths = [
        r"C:\Program Files\GitHub CLI\gh.exe",
        r"C:\Program Files (x86)\GitHub CLI\gh.exe",
        "gh",
    ]
    for path in gh_paths:
        if Path(path).exists() or path == "gh":
            return path
    return "gh"


def check_self_hosted_runner() -> dict:
    """Check if self-hosted runner is available."""
    gh = get_gh_path()
    try:
        result = subprocess.run(
            [gh, "api", "repos/SynchronizedLivingArchitecture/S.L.A.T.E/actions/runners"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            runners = data.get("runners", [])

            online_runners = [r for r in runners if r.get("status") == "online"]
            busy_runners = [r for r in runners if r.get("busy", False)]

            return {
                "available": len(online_runners) > 0,
                "total": len(runners),
                "online": len(online_runners),
                "busy": len(busy_runners),
                "runners": [
                    {
                        "name": r.get("name"),
                        "status": r.get("status"),
                        "busy": r.get("busy", False),
                        "labels": [l.get("name") for l in r.get("labels", [])],
                    }
                    for r in runners
                ],
            }
    except Exception as e:
        return {"available": False, "error": str(e), "runners": []}

    return {"available": False, "runners": []}


def get_runner_status() -> None:
    """Display comprehensive runner status."""
    config = load_config()
    runner_info = check_self_hosted_runner()

    print("=" * 60)
    print("SLATE Runner Fallback Status")
    print("=" * 60)
    print()

    # Mode
    mode = config.get("mode", "self-hosted-only")
    mode_icon = "[LOCKED]" if mode == "self-hosted-only" else "[FALLBACK]"
    print(f"Mode: {mode_icon} {mode}")
    print()

    # Self-hosted runners
    print("Self-Hosted Runners:")
    if runner_info.get("available"):
        for runner in runner_info.get("runners", []):
            status_icon = "[ONLINE]" if runner["status"] == "online" else "[OFFLINE]"
            busy_icon = " (busy)" if runner["busy"] else ""
            labels = ", ".join(runner["labels"])
            print(f"  {status_icon} {runner['name']}{busy_icon}")
            print(f"      Labels: {labels}")
    else:
        print("  [OFFLINE] No self-hosted runners available")
        if runner_info.get("error"):
            print(f"      Error: {runner_info['error']}")
    print()

    # Fallback configuration
    print("Fallback Configuration:")
    if mode == "fallback-enabled":
        print(f"  Fallback runner: {config.get('fallback_runner', 'ubuntu-latest')}")
        print(f"  Timeout before fallback: {config.get('fallback_timeout_minutes', 5)} minutes")
        budget = config.get("cost_tracking", {}).get("monthly_budget_usd", 0)
        usage = config.get("cost_tracking", {}).get("current_month_usage_usd", 0)
        print(f"  Monthly budget: ${budget}")
        print(f"  Current usage: ${usage}")
    else:
        print("  Fallback DISABLED - self-hosted only")
        print("  Run with --allow-fallback to enable")
    print()

    # Recommendation
    print("Recommendation:")
    if runner_info.get("available"):
        print("  [OK] Self-hosted runner is online - no action needed")
    else:
        if mode == "self-hosted-only":
            print("  [WARN] Self-hosted runner is offline!")
            print("     Workflows will queue until runner is back online.")
            print("     To enable fallback: python slate/runner_fallback.py --allow-fallback")
        else:
            print("  [WARN] Self-hosted runner is offline - fallback will be used")
            print("     Note: Fallback incurs GitHub-hosted runner costs")
    print()


def generate_workflow_config() -> None:
    """Generate workflow runner configuration snippet."""
    config = load_config()
    mode = config.get("mode", "self-hosted-only")

    print("# Add this to your workflow files:")
    print()

    if mode == "self-hosted-only":
        print("# Self-hosted only (no fallback)")
        print("jobs:")
        print("  your-job:")
        print("    runs-on: [self-hosted, slate]")
    else:
        print("# With fallback support")
        print("jobs:")
        print("  check-runner:")
        print("    runs-on: ubuntu-latest")
        print("    outputs:")
        print("      runner: ${{ steps.check.outputs.runner }}")
        print("    steps:")
        print("      - name: Check self-hosted availability")
        print("        id: check")
        print("        env:")
        print("          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}")
        print("        run: |")
        print('          RUNNERS=$(gh api repos/${{ github.repository }}/actions/runners --jq \'.runners[] | select(.status == "online") | .name\' 2>/dev/null || echo "")')
        print('          if [ -n "$RUNNERS" ]; then')
        print('            echo "runner=[self-hosted, slate]" >> $GITHUB_OUTPUT')
        print("          else")
        print(f'            echo "runner={config.get("fallback_runner", "ubuntu-latest")}" >> $GITHUB_OUTPUT')
        print("          fi")
        print()
        print("  your-job:")
        print("    needs: check-runner")
        print("    runs-on: ${{ fromJSON(needs.check-runner.outputs.runner) }}")


def set_mode(mode: str) -> None:
    """Set the runner mode."""
    config = load_config()

    if mode == "enforce":
        config["mode"] = "self-hosted-only"
        config["cost_tracking"]["monthly_budget_usd"] = 0
        print("[LOCKED] Mode set to: self-hosted-only")
        print("   No GitHub-hosted runners will be used.")
        print("   Workflows will queue if self-hosted runner is offline.")
    elif mode == "fallback":
        config["mode"] = "fallback-enabled"
        config["cost_tracking"]["monthly_budget_usd"] = 50  # Default budget
        print("[FALLBACK] Mode set to: fallback-enabled")
        print("   Will fall back to GitHub-hosted runners if self-hosted is unavailable.")
        print(f"   Monthly budget: ${config['cost_tracking']['monthly_budget_usd']}")

    save_config(config)


def set_budget(budget: float) -> None:
    """Set monthly budget for fallback usage."""
    config = load_config()
    config["cost_tracking"]["monthly_budget_usd"] = budget
    save_config(config)
    print(f"Monthly fallback budget set to: ${budget}")


def main():
    parser = argparse.ArgumentParser(description="SLATE Runner Fallback Manager")
    parser.add_argument("--status", action="store_true", help="Show runner status")
    parser.add_argument("--config", action="store_true", help="Generate workflow config")
    parser.add_argument("--enforce", action="store_true", help="Enforce self-hosted only mode")
    parser.add_argument("--allow-fallback", action="store_true", help="Enable fallback mode")
    parser.add_argument("--budget", type=float, help="Set monthly budget for fallback (USD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.enforce:
        set_mode("enforce")
    elif args.allow_fallback:
        set_mode("fallback")
    elif args.budget is not None:
        set_budget(args.budget)
    elif args.config:
        generate_workflow_config()
    elif args.json:
        config = load_config()
        runner_info = check_self_hosted_runner()
        print(json.dumps({"config": config, "runners": runner_info}, indent=2))
    else:
        get_runner_status()

    return 0


if __name__ == "__main__":
    sys.exit(main())
