"""
SLATE Runner Cost Tracker

Tracks objective cost data for GitHub Actions runner usage.
Maintains historical records and provides cost analytics.

Usage:
    python slate/runner_cost_tracker.py --report           # Show cost report
    python slate/runner_cost_tracker.py --update           # Update from GitHub API
    python slate/runner_cost_tracker.py --export           # Export to CSV
    python slate/runner_cost_tracker.py --savings          # Show savings from self-hosted
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
COST_DATA_FILE = WORKSPACE / ".slate_runner_costs.json"

# GitHub Actions pricing (USD per minute)
RUNNER_COSTS = {
    "ubuntu-latest": 0.008,
    "ubuntu-22.04": 0.008,
    "ubuntu-20.04": 0.008,
    "windows-latest": 0.016,
    "windows-2022": 0.016,
    "windows-2019": 0.016,
    "macos-latest": 0.08,
    "macos-14": 0.08,
    "macos-13": 0.08,
    "macos-12": 0.08,
    "self-hosted": 0.0,  # Free!
}

DEFAULT_DATA = {
    "last_updated": None,
    "runs": [],
    "monthly_summaries": {},
    "total_self_hosted_minutes": 0,
    "total_hosted_minutes": 0,
    "total_saved_usd": 0,
    "total_spent_usd": 0,
}


def load_cost_data() -> dict:
    """Load cost tracking data."""
    if COST_DATA_FILE.exists():
        with open(COST_DATA_FILE) as f:
            return json.load(f)
    return DEFAULT_DATA.copy()


def save_cost_data(data: dict) -> None:
    """Save cost tracking data."""
    data["last_updated"] = datetime.now().isoformat()
    with open(COST_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


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


def get_workflow_runs(limit: int = 100) -> list:
    """Fetch workflow runs from GitHub API."""
    gh = get_gh_path()
    try:
        result = subprocess.run(
            [
                gh, "run", "list",
                "--repo", "SynchronizedLivingArchitecture/S.L.A.T.E",
                "--limit", str(limit),
                "--json", "databaseId,workflowName,status,conclusion,createdAt,updatedAt"
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"Error fetching runs: {e}")
    return []


def get_run_jobs(run_id: int) -> list:
    """Fetch jobs for a specific run."""
    gh = get_gh_path()
    try:
        result = subprocess.run(
            [
                gh, "api",
                f"repos/SynchronizedLivingArchitecture/S.L.A.T.E/actions/runs/{run_id}/jobs",
                "--jq", ".jobs"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def calculate_job_cost(job: dict) -> dict:
    """Calculate cost for a single job."""
    # Determine runner type from labels
    labels = job.get("labels", [])
    is_self_hosted = "self-hosted" in labels or "slate" in labels

    # Calculate duration
    started = job.get("started_at")
    completed = job.get("completed_at")

    if started and completed:
        start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        duration_minutes = (end_time - start_time).total_seconds() / 60
    else:
        duration_minutes = 0

    # Determine cost rate
    if is_self_hosted:
        cost_rate = 0.0
        runner_type = "self-hosted"
    else:
        # Try to identify runner type
        runner_type = "ubuntu-latest"  # Default assumption
        for label in labels:
            if label in RUNNER_COSTS:
                runner_type = label
                break
        cost_rate = RUNNER_COSTS.get(runner_type, 0.008)

    actual_cost = duration_minutes * cost_rate
    would_cost = duration_minutes * 0.008  # What it would cost on ubuntu-latest

    return {
        "job_name": job.get("name"),
        "runner_type": runner_type,
        "is_self_hosted": is_self_hosted,
        "duration_minutes": round(duration_minutes, 2),
        "cost_rate": cost_rate,
        "actual_cost": round(actual_cost, 4),
        "saved": round(would_cost, 4) if is_self_hosted else 0,
    }


def update_cost_data(verbose: bool = False) -> dict:
    """Update cost data from GitHub API."""
    data = load_cost_data()
    runs = get_workflow_runs(200)

    total_self_hosted = 0
    total_hosted = 0
    total_saved = 0
    total_spent = 0

    processed_runs = []

    for run in runs:
        if run.get("status") != "completed":
            continue

        run_id = run.get("databaseId")
        workflow = run.get("workflowName")
        created = run.get("createdAt")

        if verbose:
            print(f"Processing: {workflow} (#{run_id})")

        jobs = get_run_jobs(run_id)

        run_data = {
            "id": run_id,
            "workflow": workflow,
            "created": created,
            "conclusion": run.get("conclusion"),
            "jobs": [],
            "total_minutes": 0,
            "total_cost": 0,
            "total_saved": 0,
        }

        for job in jobs:
            job_cost = calculate_job_cost(job)
            run_data["jobs"].append(job_cost)
            run_data["total_minutes"] += job_cost["duration_minutes"]
            run_data["total_cost"] += job_cost["actual_cost"]
            run_data["total_saved"] += job_cost["saved"]

            if job_cost["is_self_hosted"]:
                total_self_hosted += job_cost["duration_minutes"]
            else:
                total_hosted += job_cost["duration_minutes"]

            total_saved += job_cost["saved"]
            total_spent += job_cost["actual_cost"]

        processed_runs.append(run_data)

    data["runs"] = processed_runs[:100]  # Keep last 100 runs
    data["total_self_hosted_minutes"] = round(total_self_hosted, 2)
    data["total_hosted_minutes"] = round(total_hosted, 2)
    data["total_saved_usd"] = round(total_saved, 2)
    data["total_spent_usd"] = round(total_spent, 2)

    # Monthly summary
    current_month = datetime.now().strftime("%Y-%m")
    if current_month not in data["monthly_summaries"]:
        data["monthly_summaries"][current_month] = {
            "self_hosted_minutes": 0,
            "hosted_minutes": 0,
            "saved_usd": 0,
            "spent_usd": 0,
        }

    data["monthly_summaries"][current_month] = {
        "self_hosted_minutes": round(total_self_hosted, 2),
        "hosted_minutes": round(total_hosted, 2),
        "saved_usd": round(total_saved, 2),
        "spent_usd": round(total_spent, 2),
    }

    save_cost_data(data)
    return data


def show_report() -> None:
    """Display cost report."""
    data = load_cost_data()

    print("=" * 70)
    print("SLATE Runner Cost Report")
    print("=" * 70)
    print()

    last_updated = data.get("last_updated", "Never")
    print(f"Last Updated: {last_updated}")
    print()

    # Overall stats
    print("Overall Statistics:")
    print("-" * 40)
    self_hosted_mins = data.get("total_self_hosted_minutes", 0)
    hosted_mins = data.get("total_hosted_minutes", 0)
    total_mins = self_hosted_mins + hosted_mins

    print(f"  Total run time:      {total_mins:,.1f} minutes")
    print(f"  Self-hosted time:    {self_hosted_mins:,.1f} minutes ({self_hosted_mins/max(total_mins,1)*100:.1f}%)")
    print(f"  GitHub-hosted time:  {hosted_mins:,.1f} minutes ({hosted_mins/max(total_mins,1)*100:.1f}%)")
    print()

    # Cost stats
    print("Cost Analysis:")
    print("-" * 40)
    saved = data.get("total_saved_usd", 0)
    spent = data.get("total_spent_usd", 0)
    would_have_cost = saved + spent

    print(f"  Would have cost:     ${would_have_cost:,.2f}")
    print(f"  Actually spent:      ${spent:,.2f}")
    print(f"  ** SAVED:            ${saved:,.2f}")
    print()

    if would_have_cost > 0:
        savings_pct = (saved / would_have_cost) * 100
        print(f"  Savings rate:        {savings_pct:.1f}%")
    print()

    # Monthly breakdown
    print("Monthly Breakdown:")
    print("-" * 40)
    print(f"  {'Month':<10} {'Self-Hosted':<15} {'Hosted':<10} {'Saved':<10} {'Spent':<10}")
    print(f"  {'-'*10} {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    for month, summary in sorted(data.get("monthly_summaries", {}).items(), reverse=True)[:6]:
        sh_mins = summary.get("self_hosted_minutes", 0)
        h_mins = summary.get("hosted_minutes", 0)
        saved = summary.get("saved_usd", 0)
        spent = summary.get("spent_usd", 0)
        print(f"  {month:<10} {sh_mins:>10.1f} min  {h_mins:>6.1f} min  ${saved:>7.2f}  ${spent:>7.2f}")
    print()

    # Recent runs
    print("Recent Workflow Runs:")
    print("-" * 40)
    for run in data.get("runs", [])[:10]:
        workflow = run.get("workflow", "Unknown")[:30]
        mins = run.get("total_minutes", 0)
        cost = run.get("total_cost", 0)
        saved = run.get("total_saved", 0)
        icon = "[FREE]" if cost == 0 else "[PAID]"
        print(f"  {icon} {workflow:<30} {mins:>5.1f}m  cost: ${cost:.3f}  saved: ${saved:.3f}")
    print()


def show_savings() -> None:
    """Show savings summary."""
    data = load_cost_data()

    saved = data.get("total_saved_usd", 0)
    spent = data.get("total_spent_usd", 0)
    self_hosted_mins = data.get("total_self_hosted_minutes", 0)

    print()
    print("+" + "=" * 50 + "+")
    print("|" + " SLATE Self-Hosted Runner Savings ".center(50) + "|")
    print("+" + "=" * 50 + "+")
    print(f"|  Self-hosted run time:  {self_hosted_mins:>10,.1f} minutes     |")
    print(f"|  If using GitHub-hosted: ${saved + spent:>10,.2f}            |")
    print(f"|  Actually spent:         ${spent:>10,.2f}            |")
    print("+" + "=" * 50 + "+")
    print(f"|  ** TOTAL SAVED:         ${saved:>10,.2f}            |")
    print("+" + "=" * 50 + "+")
    print()

    # Annual projection
    if self_hosted_mins > 0:
        days_tracked = 7  # Approximate
        annual_projection = (saved / days_tracked) * 365
        print(f"  Projected annual savings: ${annual_projection:,.2f}")
    print()


def export_csv() -> None:
    """Export cost data to CSV."""
    data = load_cost_data()
    csv_file = WORKSPACE / "runner_costs.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Run ID", "Workflow", "Created", "Job", "Runner Type",
            "Is Self-Hosted", "Duration (min)", "Cost Rate", "Actual Cost", "Saved"
        ])

        for run in data.get("runs", []):
            for job in run.get("jobs", []):
                writer.writerow([
                    run.get("id"),
                    run.get("workflow"),
                    run.get("created"),
                    job.get("job_name"),
                    job.get("runner_type"),
                    job.get("is_self_hosted"),
                    job.get("duration_minutes"),
                    job.get("cost_rate"),
                    job.get("actual_cost"),
                    job.get("saved"),
                ])

    print(f"Exported to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="SLATE Runner Cost Tracker")
    parser.add_argument("--report", action="store_true", help="Show cost report")
    parser.add_argument("--update", action="store_true", help="Update from GitHub API")
    parser.add_argument("--export", action="store_true", help="Export to CSV")
    parser.add_argument("--savings", action="store_true", help="Show savings summary")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.update:
        print("Updating cost data from GitHub...")
        data = update_cost_data(verbose=args.verbose)
        print(f"Updated! Processed {len(data.get('runs', []))} runs.")
        if not args.json:
            show_report()
    elif args.export:
        export_csv()
    elif args.savings:
        show_savings()
    elif args.json:
        data = load_cost_data()
        print(json.dumps(data, indent=2))
    else:
        show_report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
