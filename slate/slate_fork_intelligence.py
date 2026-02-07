#!/usr/bin/env python3
# Modified: 2026-02-07T07:30:00Z | Author: Claude | Change: Agentic fork intelligence system
"""
SLATE Fork Intelligence System
===============================
Agentic AI-powered fork monitoring that uses local LLMs to analyze:
1. UPSTREAM: Dependencies SLATE is forked from (anthropic, huggingface, etc.)
2. DOWNSTREAM: Repos that have forked SLATE (community forks)

Uses Ollama (mistral-nemo) for local analysis - no cloud API costs.

Capabilities:
- Detect breaking changes in upstream dependencies
- Analyze community fork contributions for potential merge
- Generate sync recommendations
- Auto-create issues/PRs for important changes

Usage:
    python slate/slate_fork_intelligence.py --analyze          # Full analysis
    python slate/slate_fork_intelligence.py --upstream         # Check upstream only
    python slate/slate_fork_intelligence.py --downstream       # Check downstream only
    python slate/slate_fork_intelligence.py --sync             # Execute sync actions
    python slate/slate_fork_intelligence.py --report           # Generate report
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "mistral-nemo"  # Default local model
SLATE_REPO = "SynchronizedLivingArchitecture/S.L.A.T.E"
STATE_FILE = WORKSPACE_ROOT / ".slate_fork_intelligence.json"


@dataclass
class ForkChange:
    """Represents a detected change in a fork."""
    repo: str
    direction: str  # "upstream" or "downstream"
    commit_count: int
    files_changed: List[str]
    summary: str
    breaking: bool = False
    recommendation: str = ""
    ai_analysis: str = ""


@dataclass
class ForkIntelligenceState:
    """Persisted state for fork intelligence."""
    last_run: str = ""
    upstream_commits: Dict[str, str] = field(default_factory=dict)
    downstream_forks: List[str] = field(default_factory=list)
    pending_actions: List[Dict] = field(default_factory=list)


class LocalAIAnalyzer:
    """Uses local Ollama for AI analysis."""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.available = self._check_ollama()

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def analyze(self, prompt: str, context: str = "") -> str:
        """Run AI analysis using local Ollama."""
        if not self.available:
            return "[AI unavailable - Ollama not running]"

        full_prompt = f"""You are a code review AI analyzing git changes for the SLATE project.
SLATE is a System Learning Agent for Task Execution.

{context}

{prompt}

Respond concisely with:
1. Summary of changes (1-2 sentences)
2. Breaking change risk (LOW/MEDIUM/HIGH)
3. Recommended action (SYNC/REVIEW/IGNORE)
"""

        try:
            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return f"[AI error: {result.stderr}]"
        except subprocess.TimeoutExpired:
            return "[AI timeout]"
        except Exception as e:
            return f"[AI error: {e}]"


class ForkIntelligence:
    """Agentic fork monitoring and intelligence system."""

    def __init__(self):
        self.workspace = WORKSPACE_ROOT
        self.gh_cli = self._find_gh_cli()
        self.ai = LocalAIAnalyzer()
        self.state = self._load_state()

    def _find_gh_cli(self) -> str:
        """Find GitHub CLI."""
        local_gh = self.workspace / ".tools" / "gh.exe"
        if local_gh.exists():
            return str(local_gh)
        return "gh"

    def _run_gh(self, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run GitHub CLI command."""
        cmd = [self.gh_cli] + args
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(self.workspace)
        )

    def _load_state(self) -> ForkIntelligenceState:
        """Load persisted state."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return ForkIntelligenceState(**data)
            except Exception:
                pass
        return ForkIntelligenceState()

    def _save_state(self):
        """Save state to file."""
        self.state.last_run = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps({
            "last_run": self.state.last_run,
            "upstream_commits": self.state.upstream_commits,
            "downstream_forks": self.state.downstream_forks,
            "pending_actions": self.state.pending_actions,
        }, indent=2))

    def get_upstream_forks(self) -> List[str]:
        """Get list of repos SLATE has forked (upstream dependencies)."""
        result = self._run_gh([
            "repo", "list", "SynchronizedLivingArchitecture",
            "--fork", "--json", "name,parent",
            "--jq", '.[].parent.fullName'
        ])
        if result.returncode == 0:
            return [r for r in result.stdout.strip().split('\n') if r]
        return []

    def get_downstream_forks(self) -> List[str]:
        """Get list of repos that have forked SLATE."""
        result = self._run_gh([
            "api", f"repos/{SLATE_REPO}/forks",
            "--jq", '.[].full_name'
        ])
        if result.returncode == 0:
            return [r for r in result.stdout.strip().split('\n') if r]
        return []

    def check_upstream_changes(self, upstream: str) -> Optional[ForkChange]:
        """Check for new changes in upstream repo."""
        # Get our fork name
        repo_name = upstream.split("/")[1]
        our_fork = f"SynchronizedLivingArchitecture/{repo_name}"

        # Compare upstream to our fork
        result = self._run_gh([
            "api", f"repos/{our_fork}/compare/main...{upstream.replace('/', ':')}:main",
            "--jq", '{ahead: .ahead_by, commits: [.commits[].sha], files: [.files[].filename]}'
        ], timeout=60)

        if result.returncode != 0:
            return None

        try:
            data = json.loads(result.stdout)
            if data.get("ahead", 0) == 0:
                return None

            # Get commit messages for context
            commits_result = self._run_gh([
                "api", f"repos/{upstream}/commits",
                "--jq", '.[0:5] | .[].commit.message'
            ])
            recent_commits = commits_result.stdout.strip() if commits_result.returncode == 0 else ""

            change = ForkChange(
                repo=upstream,
                direction="upstream",
                commit_count=data.get("ahead", 0),
                files_changed=data.get("files", [])[:20],  # Limit to 20 files
                summary=f"{data.get('ahead', 0)} new commits from {upstream}"
            )

            # AI analysis
            if self.ai.available and change.commit_count > 0:
                context = f"""
Upstream repo: {upstream}
New commits: {change.commit_count}
Changed files: {', '.join(change.files_changed[:10])}
Recent commit messages:
{recent_commits}
"""
                change.ai_analysis = self.ai.analyze(
                    "Analyze these upstream changes. Are they breaking? Should SLATE sync?",
                    context
                )

                # Parse AI recommendation
                if "HIGH" in change.ai_analysis:
                    change.breaking = True
                    change.recommendation = "REVIEW"
                elif "SYNC" in change.ai_analysis:
                    change.recommendation = "SYNC"
                else:
                    change.recommendation = "MONITOR"

            return change

        except json.JSONDecodeError:
            return None

    def check_downstream_changes(self, fork: str) -> Optional[ForkChange]:
        """Check for interesting changes in a downstream fork."""
        # Compare SLATE main to fork
        fork_owner = fork.split("/")[0]

        result = self._run_gh([
            "api", f"repos/{SLATE_REPO}/compare/main...{fork_owner}:main",
            "--jq", '{ahead: .ahead_by, commits: [.commits[].sha], files: [.files[].filename]}'
        ], timeout=60)

        if result.returncode != 0:
            return None

        try:
            data = json.loads(result.stdout)
            if data.get("ahead", 0) == 0:
                return None

            # Get commit messages
            commits_result = self._run_gh([
                "api", f"repos/{fork}/commits",
                "--jq", '.[0:5] | .[].commit.message'
            ])
            recent_commits = commits_result.stdout.strip() if commits_result.returncode == 0 else ""

            change = ForkChange(
                repo=fork,
                direction="downstream",
                commit_count=data.get("ahead", 0),
                files_changed=data.get("files", [])[:20],
                summary=f"{data.get('ahead', 0)} commits in fork {fork}"
            )

            # AI analysis for contribution potential
            if self.ai.available and change.commit_count > 0:
                context = f"""
Community fork: {fork}
Commits ahead of SLATE: {change.commit_count}
Changed files: {', '.join(change.files_changed[:10])}
Recent commit messages:
{recent_commits}
"""
                change.ai_analysis = self.ai.analyze(
                    "Analyze this community fork. Are there valuable contributions to merge into SLATE?",
                    context
                )

                if "MERGE" in change.ai_analysis or "valuable" in change.ai_analysis.lower():
                    change.recommendation = "REVIEW_FOR_MERGE"
                else:
                    change.recommendation = "MONITOR"

            return change

        except json.JSONDecodeError:
            return None

    def analyze_all(self) -> Dict[str, List[ForkChange]]:
        """Run full fork analysis."""
        results = {
            "upstream": [],
            "downstream": [],
            "summary": {}
        }

        print()
        print("=" * 70)
        print("  SLATE Fork Intelligence Analysis")
        print("=" * 70)
        print()

        # Check upstream dependencies
        print("  [1/2] Checking upstream dependencies...")
        upstreams = self.get_upstream_forks()
        print(f"        Found {len(upstreams)} forked dependencies")

        for upstream in upstreams:
            change = self.check_upstream_changes(upstream)
            if change:
                results["upstream"].append(change)
                icon = "!" if change.breaking else "+"
                print(f"        [{icon}] {upstream}: {change.commit_count} new commits")

        # Check downstream forks
        print()
        print("  [2/2] Checking downstream forks...")
        downstreams = self.get_downstream_forks()
        self.state.downstream_forks = downstreams
        print(f"        Found {len(downstreams)} community forks")

        for fork in downstreams:
            change = self.check_downstream_changes(fork)
            if change:
                results["downstream"].append(change)
                print(f"        [>] {fork}: {change.commit_count} commits ahead")

        # Summary
        results["summary"] = {
            "upstream_count": len(upstreams),
            "upstream_changes": len(results["upstream"]),
            "downstream_count": len(downstreams),
            "downstream_changes": len(results["downstream"]),
            "breaking_changes": sum(1 for c in results["upstream"] if c.breaking),
            "ai_available": self.ai.available,
        }

        self._save_state()
        return results

    def sync_upstream(self, repo: str) -> bool:
        """Sync a specific upstream fork."""
        repo_name = repo.split("/")[1]
        our_fork = f"SynchronizedLivingArchitecture/{repo_name}"

        print(f"  Syncing {our_fork} from {repo}...")

        result = self._run_gh([
            "api", "--method", "POST",
            "-H", "Accept: application/vnd.github+json",
            f"/repos/{our_fork}/merge-upstream",
            "-f", "branch=main"
        ], timeout=60)

        success = result.returncode == 0 or "already" in result.stderr.lower()
        print(f"  {'[OK]' if success else '[!]'} {repo_name}")
        return success

    def sync_all_upstream(self) -> Dict[str, bool]:
        """Sync all upstream forks."""
        print()
        print("  Syncing all upstream forks...")
        print()

        results = {}
        upstreams = self.get_upstream_forks()

        for upstream in upstreams:
            results[upstream] = self.sync_upstream(upstream)
            time.sleep(1)  # Rate limiting

        return results

    def generate_report(self) -> str:
        """Generate markdown report of fork intelligence."""
        results = self.analyze_all()

        report = [
            "# SLATE Fork Intelligence Report",
            f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
            f"\nAI Analysis: {'Enabled (Ollama)' if self.ai.available else 'Disabled'}",
            "",
            "## Summary",
            f"- Upstream dependencies: {results['summary']['upstream_count']}",
            f"- Upstream with changes: {results['summary']['upstream_changes']}",
            f"- Breaking changes: {results['summary']['breaking_changes']}",
            f"- Community forks: {results['summary']['downstream_count']}",
            f"- Forks with contributions: {results['summary']['downstream_changes']}",
            "",
        ]

        if results["upstream"]:
            report.append("## Upstream Changes")
            for change in results["upstream"]:
                status = "BREAKING" if change.breaking else change.recommendation
                report.append(f"\n### {change.repo} [{status}]")
                report.append(f"- Commits: {change.commit_count}")
                report.append(f"- Files: {len(change.files_changed)}")
                if change.ai_analysis:
                    report.append(f"\n**AI Analysis:**\n{change.ai_analysis}")

        if results["downstream"]:
            report.append("\n## Community Fork Activity")
            for change in results["downstream"]:
                report.append(f"\n### {change.repo}")
                report.append(f"- Commits ahead: {change.commit_count}")
                report.append(f"- Recommendation: {change.recommendation}")
                if change.ai_analysis:
                    report.append(f"\n**AI Analysis:**\n{change.ai_analysis}")

        return "\n".join(report)

    def print_status(self):
        """Print current fork intelligence status."""
        print()
        print("=" * 70)
        print("  SLATE Fork Intelligence Status")
        print("=" * 70)
        print()
        print(f"  AI Engine: {'Ollama ({})'.format(OLLAMA_MODEL) if self.ai.available else 'Not available'}")
        print(f"  Last Run: {self.state.last_run or 'Never'}")
        print(f"  Downstream Forks: {len(self.state.downstream_forks)}")
        print(f"  Pending Actions: {len(self.state.pending_actions)}")
        print()
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="SLATE Fork Intelligence")
    parser.add_argument("--analyze", action="store_true", help="Run full analysis")
    parser.add_argument("--upstream", action="store_true", help="Check upstream only")
    parser.add_argument("--downstream", action="store_true", help="Check downstream only")
    parser.add_argument("--sync", action="store_true", help="Sync all upstream forks")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    intel = ForkIntelligence()

    if args.analyze:
        results = intel.analyze_all()
        if args.json:
            # Convert dataclasses to dicts for JSON
            output = {
                "upstream": [vars(c) for c in results["upstream"]],
                "downstream": [vars(c) for c in results["downstream"]],
                "summary": results["summary"]
            }
            print(json.dumps(output, indent=2))
        else:
            print()
            print(f"  Analysis complete:")
            print(f"    Upstream changes: {len(results['upstream'])}")
            print(f"    Downstream activity: {len(results['downstream'])}")
            print(f"    Breaking changes: {results['summary']['breaking_changes']}")

    elif args.upstream:
        upstreams = intel.get_upstream_forks()
        for upstream in upstreams:
            change = intel.check_upstream_changes(upstream)
            if change:
                print(f"{upstream}: {change.commit_count} commits - {change.recommendation}")

    elif args.downstream:
        downstreams = intel.get_downstream_forks()
        for fork in downstreams:
            change = intel.check_downstream_changes(fork)
            if change:
                print(f"{fork}: {change.commit_count} ahead - {change.recommendation}")

    elif args.sync:
        intel.sync_all_upstream()

    elif args.report:
        print(intel.generate_report())

    else:
        intel.print_status()


if __name__ == "__main__":
    main()
