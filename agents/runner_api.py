#!/usr/bin/env python3
# Modified: 2026-02-07T00:30:00Z | Author: COPILOT | Change: Add GitHub API status, run monitoring, workflow dispatch
"""
SLATE Runner API
================
Provides an API interface for the GitHub Actions self-hosted runner,
exposing runner status, job history, workflow dispatch, and run monitoring
for the SLATE dashboard and agent integration.
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

RUNNER_DIR = WORKSPACE_ROOT / "actions-runner"
REPO_API = "https://api.github.com/repos/SynchronizedLivingArchitecture/S.L.A.T.E"


class RunnerAPI:
    """API wrapper for the SLATE self-hosted GitHub Actions runner."""

    def __init__(self, runner_dir: Optional[Path] = None):
        self.runner_dir = Path(runner_dir) if runner_dir else RUNNER_DIR
        self._config: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def config(self) -> Dict[str, Any]:
        """Load and cache the runner .runner config."""
        if self._config is None:
            config_path = self.runner_dir / ".runner"
            if config_path.exists():
                self._config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            else:
                self._config = {}
        return self._config

    @property
    def agent_name(self) -> str:
        return self.config.get("agentName", "unknown")

    @property
    def work_folder(self) -> str:
        return self.config.get("workFolder", "_work")

    @property
    def github_url(self) -> str:
        return self.config.get("gitHubUrl", "")

    @property
    def work_dir(self) -> Path:
        return self.runner_dir / self.work_folder

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Check if the runner is configured."""
        return (self.runner_dir / ".runner").exists() and (
            self.runner_dir / ".credentials"
        ).exists()

    def is_running(self) -> bool:
        """Check if the runner process is currently active."""
        try:
            import psutil  # type: ignore

            for proc in psutil.process_iter(["name"]):
                if "Runner.Listener" in (proc.info.get("name") or ""):
                    return True
        except ImportError:
            # Fallback: check diag logs for recent activity
            log_dir = self.runner_dir / "_diag"
            if log_dir.exists():
                logs = sorted(log_dir.glob("Runner_*.log"), key=lambda p: p.stat().st_mtime)
                if logs:
                    last_modified = datetime.fromtimestamp(logs[-1].stat().st_mtime)
                    delta = (datetime.now() - last_modified).total_seconds()
                    return delta < 120  # Active within last 2 minutes
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return a status summary of the runner."""
        return {
            "configured": self.is_configured(),
            "running": self.is_running(),
            "agent_name": self.agent_name,
            "work_folder": self.work_folder,
            "github_url": self.github_url,
            "work_dir_exists": self.work_dir.exists(),
            "runner_dir": str(self.runner_dir),
        }

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def get_env(self) -> Dict[str, str]:
        """Load the runner .env file."""
        env_path = self.runner_dir / ".env"
        env_vars: Dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()
        return env_vars

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def get_recent_logs(self, lines: int = 50) -> List[str]:
        """Return the last N lines from the most recent runner log."""
        log_dir = self.runner_dir / "_diag"
        if not log_dir.exists():
            return []
        logs = sorted(log_dir.glob("Runner_*.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return []
        text = logs[-1].read_text(encoding="utf-8", errors="replace")
        return text.splitlines()[-lines:]

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print runner status to stdout."""
        status = self.get_status()
        print("=== SLATE Runner Status ===")
        for key, value in status.items():
            print(f"  {key}: {value}")

    # ------------------------------------------------------------------
    # GitHub API
    # ------------------------------------------------------------------

    @staticmethod
    def _get_github_token() -> Optional[str]:
        """Get GitHub token from git credential manager."""
        try:
            result = subprocess.run(
                ["git", "credential", "fill"],
                input="protocol=https\nhost=github.com\n",
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(WORKSPACE_ROOT),
            )
            for line in result.stdout.splitlines():
                if line.startswith("password="):
                    return line.split("=", 1)[1]
        except Exception:
            pass
        return None

    def _api(self, path: str, method: str = "GET", data: Optional[dict] = None) -> Optional[dict]:
        """Make a GitHub API request."""
        token = self._get_github_token()
        if not token:
            return None
        url = f"{REPO_API}/{path}" if not path.startswith("http") else path
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        req = urllib.request.Request(url, headers=headers, method=method)
        if data:
            req.data = json.dumps(data).encode()
            req.add_header("Content-Type", "application/json")
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            body = resp.read().decode()
            return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            return {"error": e.code, "message": e.read().decode()[:200]}
        except Exception as e:
            return {"error": str(e)}

    def get_github_runner_status(self) -> Optional[Dict[str, Any]]:
        """Get runner status from GitHub API."""
        resp = self._api("actions/runners")
        if not resp or "runners" not in resp:
            return None
        for runner in resp["runners"]:
            if runner.get("name") == self.agent_name:
                return {
                    "id": runner["id"],
                    "name": runner["name"],
                    "status": runner["status"],
                    "busy": runner.get("busy", False),
                    "labels": [l["name"] for l in runner.get("labels", [])],
                    "os": runner.get("os", "unknown"),
                }
        return None

    def get_active_runs(self) -> List[Dict[str, Any]]:
        """Get all queued and in-progress workflow runs."""
        runs = []
        for status in ["queued", "in_progress"]:
            resp = self._api(f"actions/runs?status={status}&per_page=20")
            if resp and "workflow_runs" in resp:
                for r in resp["workflow_runs"]:
                    runs.append({
                        "id": r["id"],
                        "run_number": r["run_number"],
                        "name": r["name"],
                        "status": r["status"],
                        "conclusion": r.get("conclusion"),
                        "event": r["event"],
                        "created_at": r["created_at"],
                    })
        return runs

    def get_recent_runs(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow runs."""
        resp = self._api(f"actions/runs?per_page={count}")
        if not resp or "workflow_runs" not in resp:
            return []
        return [
            {
                "id": r["id"],
                "run_number": r["run_number"],
                "name": r["name"],
                "status": r["status"],
                "conclusion": r.get("conclusion"),
                "event": r["event"],
                "created_at": r["created_at"],
            }
            for r in resp["workflow_runs"]
        ]

    def get_run_jobs(self, run_id: int) -> List[Dict[str, Any]]:
        """Get jobs for a specific workflow run."""
        resp = self._api(f"actions/runs/{run_id}/jobs")
        if not resp or "jobs" not in resp:
            return []
        jobs = []
        for j in resp["jobs"]:
            failed_steps = [
                s["name"] for s in j.get("steps", [])
                if s.get("conclusion") == "failure"
            ]
            jobs.append({
                "name": j["name"],
                "status": j["status"],
                "conclusion": j.get("conclusion"),
                "runner_name": j.get("runner_name"),
                "failed_steps": failed_steps,
            })
        return jobs

    def dispatch_workflow(self, workflow_file: str = "ci.yml", ref: str = "main") -> Dict[str, Any]:
        """Dispatch a workflow run."""
        resp = self._api(
            f"actions/workflows/{workflow_file}/dispatches",
            method="POST",
            data={"ref": ref},
        )
        if resp is None or resp == {}:
            return {"success": True, "workflow": workflow_file}
        return {"success": False, "error": resp}

    def cancel_run(self, run_id: int) -> bool:
        """Cancel a workflow run."""
        resp = self._api(f"actions/runs/{run_id}/cancel", method="POST")
        return resp is not None and "error" not in resp

    def cancel_all_active(self) -> List[int]:
        """Cancel all queued and in-progress runs. Returns list of cancelled run IDs."""
        cancelled = []
        active = self.get_active_runs()
        for r in active:
            if self.cancel_run(r["id"]):
                cancelled.append(r["run_number"])
        return cancelled

    def print_full_status(self) -> None:
        """Print comprehensive runner status including GitHub API data."""
        # Local status
        status = self.get_status()
        print("=" * 55)
        print("  S.L.A.T.E. Runner Status")
        print("=" * 55)
        print()
        for key, value in status.items():
            print(f"  {key}: {value}")

        # GitHub API status
        gh = self.get_github_runner_status()
        if gh:
            print()
            print("  --- GitHub API ---")
            print(f"  online: {gh['status']}")
            print(f"  busy: {gh['busy']}")
            print(f"  labels: {', '.join(gh['labels'])}")
        else:
            print("\n  --- GitHub API: unavailable ---")

        # Active runs
        active = self.get_active_runs()
        if active:
            print()
            print(f"  --- Active Runs ({len(active)}) ---")
            for r in active:
                print(f"  #{r['run_number']} {r['name']} - {r['status']} [{r['event']}]")
        else:
            print("\n  --- No active runs ---")
        print()


if __name__ == "__main__":
    api = RunnerAPI()
    api.print_full_status()
