#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: install_slate [python]
# Author: COPILOT | Created: 2026-02-06T00:30:00Z | Modified: 2026-02-06T10:15:00Z
# Purpose: SLATE public installation script with dashboard-first tracking + runner + AI agents
# ═══════════════════════════════════════════════════════════════════════════════
"""
S.L.A.T.E. Installation Script
===============================
Installs and configures SLATE with real-time dashboard progress tracking.
The dashboard is the FIRST system installed — every subsequent step is
visible in the browser at http://127.0.0.1:8080.

12 Installation Steps:
    0. Dashboard boot      6. SDK validate       
    1. Python check         7. Dirs create        
    2. Venv setup           8. Git sync           
    3. Deps install         9. Benchmark          
    4. GPU detect          10. Runner setup (opt) 
    5. AI agents           11. Runtime check

Architecture:
    install_slate.py → InstallTracker → install_state.json ← Dashboard reads
                                      → SSE broadcast     ← Dashboard listens

Usage:
    python install_slate.py                     # Full install with dashboard
    python install_slate.py --no-dashboard      # CLI-only (no browser)
    python install_slate.py --skip-gpu          # Skip GPU detection
    python install_slate.py --dev               # Developer mode (editable install)
    python install_slate.py --resume            # Resume a failed install
    python install_slate.py --runner             # Include self-hosted runner setup
    python install_slate.py --runner --runner-token TOK  # Runner + auto-configure
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent
DASHBOARD_PORT = 8080
DASHBOARD_URL = f"http://127.0.0.1:{DASHBOARD_PORT}"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_python_exe() -> Path:
    """Get venv python executable path."""
    if os.name == "nt":
        return WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
    return WORKSPACE_ROOT / ".venv" / "bin" / "python"


def _get_pip_exe() -> Path:
    """Get venv pip executable path."""
    if os.name == "nt":
        return WORKSPACE_ROOT / ".venv" / "Scripts" / "pip.exe"
    return WORKSPACE_ROOT / ".venv" / "bin" / "pip"


def _run_cmd(cmd: list, timeout: int = 120, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with defaults."""
    return subprocess.run(
        [str(c) for c in cmd],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(WORKSPACE_ROOT), **kwargs,
    )


def _gpu_arch(compute_cap: str) -> str:
    """Map CUDA compute capability to architecture name."""
    if compute_cap.startswith("12."):
        return "Blackwell"
    elif compute_cap == "8.9":
        return "Ada Lovelace"
    elif compute_cap.startswith("8."):
        return "Ampere"
    elif compute_cap == "7.5":
        return "Turing"
    elif compute_cap == "7.0":
        return "Volta"
    elif compute_cap.startswith("6."):
        return "Pascal"
    return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  INSTALLATION STEPS — each maps to an InstallTracker canonical step
# ═══════════════════════════════════════════════════════════════════════════════

def step_dashboard_boot(tracker, args):
    """Step 0: Boot the install dashboard (first thing visible)."""
    tracker.start_step("dashboard_boot")

    if args.no_dashboard:
        tracker.skip_step("dashboard_boot", "Dashboard disabled (--no-dashboard)")
        return True

    try:
        # The install_api module can run a standalone server
        sys.path.insert(0, str(WORKSPACE_ROOT))
        from agents.install_api import run_standalone_server
        tracker.update_progress("dashboard_boot", 50, "Starting dashboard server")
        server_thread = run_standalone_server(port=DASHBOARD_PORT)

        if server_thread and server_thread.is_alive():
            tracker.update_progress("dashboard_boot", 80, "Opening browser")
            time.sleep(0.5)
            webbrowser.open(DASHBOARD_URL)
            tracker.complete_step("dashboard_boot", success=True,
                                  details=f"Dashboard live at {DASHBOARD_URL}")
            return True
        else:
            tracker.complete_step("dashboard_boot", success=True, warning=True,
                                  details="Dashboard failed to start — continuing CLI-only")
            return True
    except ImportError:
        tracker.complete_step("dashboard_boot", success=True, warning=True,
                              details="FastAPI not yet installed — dashboard available after deps")
        return True
    except Exception as e:
        tracker.complete_step("dashboard_boot", success=True, warning=True,
                              details=f"Dashboard skipped: {e}")
        return True


def step_python_check(tracker, args):
    """Step 1: Verify Python version."""
    tracker.start_step("python_check")

    version = sys.version_info
    py_str = f"{version.major}.{version.minor}.{version.micro}"
    tracker.update_progress("python_check", 50, f"Found Python {py_str}")

    if version.major < 3 or (version.major == 3 and version.minor < 11):
        tracker.complete_step("python_check", success=False,
                              error=f"Python 3.11+ required, found {py_str}")
        return False

    details = f"Python {py_str} ({sys.executable})"
    tracker.complete_step("python_check", success=True, details=details)
    return True


def step_venv_setup(tracker, args):
    """Step 2: Create or verify virtual environment."""
    tracker.start_step("venv_setup")
    venv_path = WORKSPACE_ROOT / ".venv"

    if venv_path.exists() and _get_python_exe().exists():
        tracker.update_progress("venv_setup", 80, "Virtual environment exists")
        # Verify it works
        try:
            result = _run_cmd([_get_python_exe(), "-c", "import sys; print(sys.version)"], timeout=10)
            if result.returncode == 0:
                tracker.complete_step("venv_setup", success=True,
                                      details=f"Existing venv OK ({result.stdout.strip().split()[0]})")
                return True
        except Exception:
            pass
        # Exists but broken — recreate
        tracker.update_progress("venv_setup", 30, "Venv broken, recreating...")

    tracker.update_progress("venv_setup", 20, "Creating virtual environment")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)],
                       check=True, timeout=60)
        tracker.complete_step("venv_setup", success=True, details=f"Created at {venv_path}")
        return True
    except subprocess.CalledProcessError as e:
        tracker.complete_step("venv_setup", success=False,
                              error=f"venv creation failed: {e}")
        return False
    except subprocess.TimeoutExpired:
        tracker.complete_step("venv_setup", success=False,
                              error="venv creation timed out after 60s")
        return False


def step_deps_install(tracker, args):
    """Step 3: Install core dependencies from requirements.txt."""
    tracker.start_step("deps_install")
    pip = _get_pip_exe()

    if not pip.exists():
        tracker.complete_step("deps_install", success=False,
                              error="pip not found in venv — venv may be corrupt")
        return False

    # Upgrade pip first
    tracker.update_progress("deps_install", 10, "Upgrading pip")
    try:
        _run_cmd([pip, "install", "--upgrade", "pip", "--quiet"], timeout=60)
    except Exception:
        pass  # Non-fatal

    req_file = WORKSPACE_ROOT / "requirements.txt"
    if not req_file.exists():
        tracker.complete_step("deps_install", success=False,
                              error="requirements.txt not found")
        return False

    # Count packages for progress estimation
    pkg_count = sum(1 for line in req_file.read_text().splitlines()
                    if line.strip() and not line.strip().startswith("#"))
    tracker.update_progress("deps_install", 20,
                            f"Installing {pkg_count} packages from requirements.txt")

    try:
        install_args = [str(pip), "install", "-r", str(req_file)]
        if not args.dev:
            install_args.append("--quiet")
        result = subprocess.run(install_args, capture_output=True, text=True,
                                timeout=600, cwd=str(WORKSPACE_ROOT))

        if result.returncode != 0:
            # Try to extract useful error
            err_lines = [l for l in (result.stderr or "").splitlines() if "ERROR" in l]
            error_msg = err_lines[-1] if err_lines else "pip install failed"
            tracker.complete_step("deps_install", success=False, error=error_msg)
            return False

        tracker.update_progress("deps_install", 90, "Verifying core imports")
        # Quick verification — can we import key packages?
        verify = _run_cmd([_get_python_exe(), "-c",
                           "import fastapi; import uvicorn; print('core OK')"], timeout=15)
        if verify.returncode == 0:
            tracker.complete_step("deps_install", success=True,
                                  details=f"{pkg_count} packages installed, core imports OK")
        else:
            tracker.complete_step("deps_install", success=True, warning=True,
                                  details=f"{pkg_count} packages installed (some optional imports missing)")
        return True

    except subprocess.TimeoutExpired:
        tracker.complete_step("deps_install", success=False,
                              error="Package installation timed out after 10 minutes")
        return False
    except Exception as e:
        tracker.complete_step("deps_install", success=False, error=str(e))
        return False


def step_gpu_detect(tracker, args):
    """Step 4: Detect NVIDIA GPUs and compute capability."""
    tracker.start_step("gpu_detect")

    if args.skip_gpu:
        tracker.skip_step("gpu_detect", "GPU detection skipped (--skip-gpu)")
        return True

    tracker.update_progress("gpu_detect", 30, "Querying nvidia-smi")
    try:
        result = _run_cmd(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"],
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().split("\n")
            gpu_info = []
            for gpu_line in gpus:
                parts = [p.strip() for p in gpu_line.split(",")]
                if len(parts) >= 3:
                    name, cc, mem = parts[0], parts[1], parts[2]
                    gpu_info.append(f"{name} ({_gpu_arch(cc)}, CC {cc}, {mem})")

            details = f"{len(gpus)} GPU(s): " + "; ".join(gpu_info)
            tracker.complete_step("gpu_detect", success=True, details=details)
        else:
            tracker.complete_step("gpu_detect", success=True, warning=True,
                                  details="No NVIDIA GPU detected — running in CPU mode")
        return True

    except FileNotFoundError:
        tracker.complete_step("gpu_detect", success=True, warning=True,
                              details="nvidia-smi not found — running in CPU mode")
        return True
    except subprocess.TimeoutExpired:
        tracker.complete_step("gpu_detect", success=True, warning=True,
                              details="GPU detection timed out — skipping")
        return True
    except Exception as e:
        tracker.complete_step("gpu_detect", success=True, warning=True,
                              details=f"GPU detection error: {e}")
        return True


def step_sdk_validate(tracker, args):
    """Step 5: Validate slate SDK imports and version."""
    tracker.start_step("sdk_validate")
    tracker.update_progress("sdk_validate", 30, "Importing slate SDK")

    try:
        result = _run_cmd([
            _get_python_exe(), "-c",
            "import slate; print(getattr(slate, '__version__', 'unknown'))"
        ], timeout=15)

        if result.returncode == 0:
            version = result.stdout.strip()
            tracker.update_progress("sdk_validate", 70, f"SDK v{version} found")
            # Check critical modules
            check = _run_cmd([
                _get_python_exe(), "-c",
                "from slate import slate_status; "
                "from slate import slate_runtime; "
                "print('all_ok')"
            ], timeout=15)
            if check.returncode == 0 and "all_ok" in check.stdout:
                tracker.complete_step("sdk_validate", success=True,
                                      details=f"slate v{version} — all modules OK")
            else:
                tracker.complete_step("sdk_validate", success=True, warning=True,
                                      details=f"slate v{version} — some modules missing")
        else:
            tracker.complete_step("sdk_validate", success=True, warning=True,
                                  details="slate not importable (first install)")
        return True

    except Exception as e:
        tracker.complete_step("sdk_validate", success=True, warning=True,
                              details=f"SDK validation skipped: {e}")
        return True


def step_dirs_create(tracker, args):
    """Step 6: Create workspace directories and init files."""
    tracker.start_step("dirs_create")

    dirs = [
        WORKSPACE_ROOT / "slate",
        WORKSPACE_ROOT / "agents",
        WORKSPACE_ROOT / "slate_web",
        WORKSPACE_ROOT / "tests",
        WORKSPACE_ROOT / ".github",
        WORKSPACE_ROOT / ".slate_install",
        WORKSPACE_ROOT / ".slate_fork",
        WORKSPACE_ROOT / "logs",
        WORKSPACE_ROOT / "data",
    ]

    created = 0
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created += 1

    # Ensure __init__.py files exist for Python packages
    init_files = [
        WORKSPACE_ROOT / "slate" / "__init__.py",
        WORKSPACE_ROOT / "agents" / "__init__.py",
        WORKSPACE_ROOT / "slate_web" / "__init__.py",
        WORKSPACE_ROOT / "tests" / "__init__.py",
    ]
    for f in init_files:
        f.touch(exist_ok=True)

    tracker.complete_step("dirs_create", success=True,
                          details=f"{len(dirs)} directories checked, {created} created")
    return True


def step_git_sync(tracker, args):
    """Step 7: Sync with GitHub repository state."""
    tracker.start_step("git_sync")
    tracker.update_progress("git_sync", 20, "Checking git state")

    # Verify git is available
    try:
        result = _run_cmd(["git", "--version"], timeout=10)
        if result.returncode != 0:
            tracker.complete_step("git_sync", success=True, warning=True,
                                  details="git not found — skipping sync")
            return True
    except FileNotFoundError:
        tracker.complete_step("git_sync", success=True, warning=True,
                              details="git not installed — skipping sync")
        return True

    # Check if we're in a git repo
    in_repo = _run_cmd(["git", "rev-parse", "--is-inside-work-tree"], timeout=10)
    if in_repo.returncode != 0:
        tracker.update_progress("git_sync", 40, "Initializing git repository")
        _run_cmd(["git", "init"], timeout=15)

    # Verify remotes
    tracker.update_progress("git_sync", 60, "Verifying remotes")
    remote = _run_cmd(["git", "remote", "-v"], timeout=10)
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
    commit = _run_cmd(["git", "rev-parse", "--short", "HEAD"], timeout=10)

    branch_name = branch.stdout.strip() if branch.returncode == 0 else "unknown"
    commit_hash = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    has_remote = "origin" in (remote.stdout or "")

    details = f"Branch: {branch_name} @ {commit_hash}"
    if has_remote:
        details += " (origin configured)"
    tracker.complete_step("git_sync", success=True, details=details)
    return True


def step_benchmark(tracker, args):
    """Step 8: Run system benchmarks."""
    tracker.start_step("benchmark")

    benchmark_script = WORKSPACE_ROOT / "slate" / "slate_benchmark.py"
    if not benchmark_script.exists():
        tracker.skip_step("benchmark", "Benchmark script not found")
        return True

    python_exe = _get_python_exe()
    if not python_exe.exists():
        tracker.skip_step("benchmark", "Python venv not available")
        return True

    tracker.update_progress("benchmark", 20, "Running CPU + memory benchmarks")
    try:
        result = _run_cmd([python_exe, str(benchmark_script), "--json"], timeout=120)
        if result.returncode == 0:
            # Try to parse benchmark results
            try:
                bench_data = json.loads(result.stdout)
                score = bench_data.get("overall_score", "N/A")
                details = f"Benchmark complete — Overall score: {score}"
            except (json.JSONDecodeError, ValueError):
                details = "Benchmark complete"
            tracker.complete_step("benchmark", success=True, details=details)
        else:
            tracker.complete_step("benchmark", success=True, warning=True,
                                  details="Benchmark ran with warnings")
        return True
    except subprocess.TimeoutExpired:
        tracker.complete_step("benchmark", success=True, warning=True,
                              details="Benchmark timed out after 120s — skipping")
        return True
    except Exception as e:
        tracker.complete_step("benchmark", success=True, warning=True,
                              details=f"Benchmark skipped: {e}")
        return True


def step_runner_setup(tracker, args):
    """Step 9: Set up GitHub Actions self-hosted runner (optional)."""
    # Modified: 2026-02-09T06:30:00Z | Author: COPILOT | Change: Add runner step to install
    tracker.start_step("runner_setup")

    if not args.runner:
        tracker.skip_step("runner_setup", "Self-hosted runner not requested (use --runner)")
        return True

    tracker.update_progress("runner_setup", 10, "Importing runner manager")
    try:
        from slate.slate_runner_manager import SlateRunnerManager
    except ImportError:
        tracker.complete_step("runner_setup", success=True, warning=True,
                              details="Runner manager not available — skipping")
        return True

    manager = SlateRunnerManager()

    # Step 1: Check current runner status
    tracker.update_progress("runner_setup", 15, "Checking runner status")
    status = manager.get_status()

    # Step 2: Download runner if not installed
    if not status["installed"]:
        tracker.update_progress("runner_setup", 20, "Downloading GitHub Actions runner")
        if not manager.download_runner():
            tracker.complete_step("runner_setup", success=True, warning=True,
                                  details="Runner download failed — setup manually later")
            return True
        tracker.update_progress("runner_setup", 40, "Runner downloaded")
    else:
        tracker.update_progress("runner_setup", 40, "Runner already installed")

    # Step 3: Configure runner if token provided and not already configured
    if args.runner_token and not status["configured"]:
        tracker.update_progress("runner_setup", 50, "Configuring runner")
        repo_url = "https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E."
        result = manager.configure_runner(
            repo_url=repo_url,
            token=args.runner_token,
        )
        if result["success"]:
            tracker.update_progress("runner_setup", 60, "Runner configured")
        else:
            errors = "; ".join(result.get("errors", ["Unknown error"]))
            tracker.complete_step("runner_setup", success=True, warning=True,
                                  details=f"Runner config failed: {errors}")
            return True
    elif not status["configured"]:
        tracker.update_progress("runner_setup", 60,
                                "Runner downloaded but needs --runner-token to configure")

    # Step 4: Provision SLATE environment on runner
    tracker.update_progress("runner_setup", 65, "Provisioning SLATE environment")
    provision_result = manager.provision_slate_environment(WORKSPACE_ROOT)
    if provision_result["success"]:
        tracker.update_progress("runner_setup", 80, "SLATE environment provisioned")
    else:
        errors = "; ".join(provision_result.get("errors", []))
        tracker.complete_step("runner_setup", success=True, warning=True,
                              details=f"Provisioning partial: {errors}")
        return True

    # Step 5: Create startup and auto-start scripts
    tracker.update_progress("runner_setup", 85, "Creating startup scripts")
    try:
        startup_script = manager.create_startup_script()
        if os.name == "nt":
            manager.create_windows_service_config()
        tracker.update_progress("runner_setup", 95, f"Startup script: {startup_script}")
    except Exception as e:
        tracker.update_progress("runner_setup", 95, f"Startup script warning: {e}")

    # Build summary
    final_status = manager.get_status()
    parts = []
    parts.append(f"Runner {'installed' if final_status['installed'] else 'pending'}")
    parts.append(f"{'configured' if final_status['configured'] else 'needs token'}")
    parts.append(f"{'provisioned' if final_status['provisioned'] else 'not provisioned'}")
    parts.append(f"{final_status['gpu']['gpu_count']} GPU(s)")
    parts.append(f"Labels: {', '.join(final_status['labels'][:5])}")

    tracker.complete_step("runner_setup", success=True,
                          details=" | ".join(parts))
    return True


def step_ai_agents(tracker, args):
    """Step 10: Configure AI agent integrations (Copilot + Claude + VS Code)."""
    # Modified: 2026-02-06T23:30:00Z | Author: COPILOT | Change: Add VS Code extension validation
    tracker.start_step("ai_agents")
    tracker.update_progress("ai_agents", 5, "Checking AI agent configurations")

    configured = []
    warnings = []

    # 1. Verify MCP server exists
    mcp_server = WORKSPACE_ROOT / "slate" / "slate_mcp_server.py"
    if mcp_server.exists():
        configured.append("MCP server")
    else:
        warnings.append("MCP server not found (slate/slate_mcp_server.py)")

    tracker.update_progress("ai_agents", 25, "Checking Copilot agent")

    # 2. Verify Copilot instructions
    copilot_instructions = WORKSPACE_ROOT / ".github" / "copilot-instructions.md"
    if copilot_instructions.exists():
        content = copilot_instructions.read_text(encoding="utf-8", errors="replace")
        if "S.L.A.T.E." in content and "MCP Server" in content:
            configured.append("Copilot agent")
        else:
            warnings.append("Copilot instructions exist but may be incomplete")
    else:
        warnings.append("Copilot instructions missing (.github/copilot-instructions.md)")

    # 3. Verify VS Code MCP config
    mcp_config = WORKSPACE_ROOT / ".vscode" / "mcp.json"
    if mcp_config.exists():
        configured.append("VS Code MCP config")
    else:
        warnings.append("VS Code MCP config missing (.vscode/mcp.json)")

    tracker.update_progress("ai_agents", 50, "Checking Claude plugin")

    # 4. Verify CLAUDE.md
    claude_md = WORKSPACE_ROOT / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8", errors="replace")
        if "MCP" in content and "slate_mcp_server" in content:
            configured.append("CLAUDE.md")
        else:
            warnings.append("CLAUDE.md exists but missing MCP references")
    else:
        warnings.append("CLAUDE.md not found")

    # 5. Verify Claude skills
    skills_dir = WORKSPACE_ROOT / ".claude" / "skills"
    if skills_dir.exists():
        skill_count = sum(1 for d in skills_dir.iterdir() if d.is_dir())
        if skill_count > 0:
            configured.append(f"Claude skills ({skill_count})")
        else:
            warnings.append("Claude skills directory empty (.claude/skills/)")
    else:
        warnings.append("Claude skills not found (.claude/skills/)")

    tracker.update_progress("ai_agents", 75, "Checking MCP dependency")

    # 6. Verify MCP SDK installed
    python_exe = _get_python_exe()
    if python_exe.exists():
        result = _run_cmd([python_exe, "-c", "import mcp; print(mcp.__version__)"], timeout=10)
        if result.returncode == 0:
            configured.append(f"MCP SDK v{result.stdout.strip()}")
        else:
            # Try to install it
            tracker.update_progress("ai_agents", 80, "Installing MCP SDK")
            pip = _get_pip_exe()
            if pip.exists():
                install_result = _run_cmd([pip, "install", "mcp", "--quiet"], timeout=60)
                if install_result.returncode == 0:
                    configured.append("MCP SDK (just installed)")
                else:
                    warnings.append("MCP SDK install failed — run: pip install mcp")

    # 7. Verify VS Code integration
    tracker.update_progress("ai_agents", 85, "Checking VS Code extensions")

    # 7a. Check VS Code CLI is available
    code_exe = shutil.which("code")
    if code_exe:
        configured.append("VS Code CLI")

        # 7b. Check installed extensions
        required_extensions = [
            "github.copilot",
            "github.copilot-chat",
            "github.vscode-github-actions",
            "github.vscode-pull-request-github",
            "ms-python.python",
            "ms-python.vscode-pylance",
        ]
        try:
            ext_result = _run_cmd([code_exe, "--list-extensions"], timeout=15)
            if ext_result.returncode == 0:
                installed_exts = ext_result.stdout.strip().lower().splitlines()
                missing_exts = [
                    ext for ext in required_extensions
                    if ext.lower() not in installed_exts
                ]
                if not missing_exts:
                    configured.append(f"VS Code extensions ({len(required_extensions)} required)")
                else:
                    warnings.append(
                        f"Missing VS Code extensions: {', '.join(missing_exts)}"
                    )
        except Exception:
            warnings.append("Could not check VS Code extensions")
    else:
        warnings.append("VS Code CLI not found (install VS Code and add to PATH)")

    # 7c. Check VS Code extensions.json
    extensions_json = WORKSPACE_ROOT / ".vscode" / "extensions.json"
    if extensions_json.exists():
        configured.append("VS Code extensions.json")
    else:
        warnings.append(".vscode/extensions.json missing")

    # 7d. Check VS Code tasks.json
    tasks_json = WORKSPACE_ROOT / ".vscode" / "tasks.json"
    if tasks_json.exists():
        configured.append("VS Code tasks.json")

    # 8. Verify GitHub integrations config
    tracker.update_progress("ai_agents", 90, "Verifying GitHub integrations")
    github_files = {
        "Actions workflows": WORKSPACE_ROOT / ".github" / "workflows",
        "Issue templates": WORKSPACE_ROOT / ".github" / "ISSUE_TEMPLATE",
        "PR template": WORKSPACE_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
        "CODEOWNERS": WORKSPACE_ROOT / ".github" / "CODEOWNERS",
        "Dependabot": WORKSPACE_ROOT / ".github" / "dependabot.yml",
        "Labels": WORKSPACE_ROOT / ".github" / "labels.yml",
        "Security": WORKSPACE_ROOT / ".github" / "SECURITY.md",
        "Funding": WORKSPACE_ROOT / ".github" / "FUNDING.yml",
    }
    for name, path in github_files.items():
        if path.exists():
            configured.append(name)
        else:
            warnings.append(f"{name} missing ({path.relative_to(WORKSPACE_ROOT)})")

    # Build result
    details = f"{len(configured)} integrations active"
    if warnings:
        details += f", {len(warnings)} warnings"

    if warnings:
        tracker.complete_step("ai_agents", success=True, warning=True,
                              details=f"{details}: {'; '.join(warnings[:3])}")
    else:
        tracker.complete_step("ai_agents", success=True, details=details)

    return True


def step_runtime_check(tracker, args):
    """Step 11: Final runtime verification."""
    tracker.start_step("runtime_check")
    tracker.update_progress("runtime_check", 20, "Running runtime checks")

    checks_passed = 0
    checks_total = 0
    issues = []

    # Check 1: slate_status.py exists and runs
    checks_total += 1
    status_script = WORKSPACE_ROOT / "slate" / "slate_status.py"
    if status_script.exists():
        result = _run_cmd([_get_python_exe(), str(status_script), "--quick"], timeout=30)
        if result.returncode == 0:
            checks_passed += 1
        else:
            issues.append("slate_status.py failed")
    else:
        issues.append("slate_status.py not found")

    tracker.update_progress("runtime_check", 50, f"{checks_passed}/{checks_total} checks passed")

    # Check 2: slate_runtime.py exists and runs
    checks_total += 1
    runtime_script = WORKSPACE_ROOT / "slate" / "slate_runtime.py"
    if runtime_script.exists():
        result = _run_cmd([_get_python_exe(), str(runtime_script), "--check-all"], timeout=30)
        if result.returncode == 0:
            checks_passed += 1
        else:
            issues.append("slate_runtime.py returned errors")
    else:
        issues.append("slate_runtime.py not found")

    # Check 3: Dashboard server importable
    checks_total += 1
    result = _run_cmd([
        _get_python_exe(), "-c",
        "from agents.slate_dashboard_server import app; print('ok')"
    ], timeout=15)
    if result.returncode == 0:
        checks_passed += 1
    else:
        issues.append("Dashboard server not importable")

    tracker.update_progress("runtime_check", 80,
                            f"{checks_passed}/{checks_total} runtime checks passed")

    if checks_passed == checks_total:
        tracker.complete_step("runtime_check", success=True,
                              details=f"All {checks_total} runtime checks passed")
    elif checks_passed > 0:
        tracker.complete_step("runtime_check", success=True, warning=True,
                              details=f"{checks_passed}/{checks_total} passed — {'; '.join(issues)}")
    else:
        tracker.complete_step("runtime_check", success=False,
                              error=f"All checks failed: {'; '.join(issues)}")
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  BANNER & COMPLETION
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print installation banner."""
    print()
    print("═" * 70)
    print("  S.L.A.T.E. Installation")
    print("  System Learning Agent for Task Execution")
    print("═" * 70)
    print()


def print_completion(success: bool, tracker=None):
    """Print completion message."""
    print()
    print("═" * 70)
    if success:
        print("  ✓ S.L.A.T.E. Installation Complete!")
    else:
        print("  ✗ S.L.A.T.E. Installation Failed")
    print("═" * 70)
    print()

    if success:
        print("  Next steps:")
        print()
        if os.name == "nt":
            print("    1. Activate:  .\\.venv\\Scripts\\activate")
        else:
            print("    1. Activate:  source .venv/bin/activate")
        print("    2. Status:    python slate/slate_status.py --quick")
        print("    3. Runtime:   python slate/slate_runtime.py --check-all")
        print("    4. Dashboard: python agents/slate_dashboard_server.py")
        print("    5. Hardware:  python slate/slate_hardware_optimizer.py")
        print()
        print("  For GPU support (optional):")
        print("    python slate/slate_hardware_optimizer.py --install-pytorch")
        print()
        print("  Self-hosted runner (optional):")
        print("    python install_slate.py --runner --runner-token TOKEN")
        print("    python slate/slate_runner_manager.py --status")
        print()
        print("  AI Agent integration:")
        print("    Copilot:  .github/copilot-instructions.md (auto-loaded)")
        print("    MCP:      .vscode/mcp.json (Copilot Chat MCP tools)")
        print("    Claude:   CLAUDE.md + .claude/skills/ (5 skills)")
        print("    Verify:   python slate/slate_mcp_server.py --verify")
        print()
    else:
        print("  Troubleshooting:")
        print()
        print("    • Check install log:  cat .slate_install/install.log")
        print("    • Check install state: cat .slate_install/install_state.json")
        print("    • Resume install:     python install_slate.py --resume")
        print("    • Skip GPU:           python install_slate.py --skip-gpu")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="S.L.A.T.E. Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_slate.py                   Full install with live dashboard
  python install_slate.py --no-dashboard    CLI-only installation
  python install_slate.py --skip-gpu        Skip GPU detection step
  python install_slate.py --runner          Set up GitHub Actions runner
  python install_slate.py --resume          Resume a previously failed install
  python install_slate.py --dev             Developer mode (verbose + editable)
        """,
    )
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable live dashboard (CLI output only)")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip NVIDIA GPU detection step")
    parser.add_argument("--dev", action="store_true",
                        help="Developer mode — verbose output, editable install")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously failed installation")
    parser.add_argument("--runner", action="store_true",
                        help="Set up self-hosted GitHub Actions runner")
    parser.add_argument("--runner-token", type=str, default=None,
                        help="GitHub Actions runner registration token")
    return parser.parse_args()


def main():
    """Main installation entry point."""
    args = parse_args()
    print_banner()

    # Initialize the install tracker
    sys.path.insert(0, str(WORKSPACE_ROOT))
    try:
        from slate.install_tracker import InstallTracker
        tracker = InstallTracker()
    except ImportError:
        # InstallTracker not available yet — create a minimal shim
        class _MinimalTracker:
            """Shim that prints to console when InstallTracker isn't available."""
            def begin_install(self): pass
            def finish_install(self, success=True): pass
            def start_step(self, sid, substep=None):
                print(f"  → Starting: {sid}")
            def update_progress(self, sid, pct, detail=None):
                if detail:
                    print(f"    ... {detail}")
            def complete_step(self, sid, success=True, details=None,
                              error=None, warning=False):
                icon = "✓" if success else ("⚠" if warning else "✗")
                msg = details or error or ("Done" if success else "Failed")
                print(f"  {icon} {sid}: {msg}")
            def skip_step(self, sid, reason="Skipped"):
                print(f"  ○ {sid}: {reason}")
            def get_state(self): return {}
        tracker = _MinimalTracker()

    # Resume support — skip already-completed steps
    resume_completed = set()
    if args.resume:
        try:
            from slate.install_tracker import InstallTracker as IT
            state = IT.load_state()
            if state and state.get("steps"):
                resume_completed = {
                    s["id"] for s in state["steps"]
                    if s.get("status") in ("success", "skipped", "warning")
                }
                print(f"  ℹ Resuming — {len(resume_completed)} steps already complete\n")
        except Exception:
            pass

    # Define all installation steps in canonical order
    install_steps = [
        ("dashboard_boot", step_dashboard_boot),
        ("python_check",   step_python_check),
        ("venv_setup",     step_venv_setup),
        ("deps_install",   step_deps_install),
        ("gpu_detect",     step_gpu_detect),
        ("sdk_validate",   step_sdk_validate),
        ("dirs_create",    step_dirs_create),
        ("git_sync",       step_git_sync),
        ("benchmark",      step_benchmark),
        ("runner_setup",   step_runner_setup),
        ("ai_agents",      step_ai_agents),
        ("runtime_check",  step_runtime_check),
    ]

    tracker.begin_install()
    all_ok = True

    for step_id, step_fn in install_steps:
        # Skip already-completed steps on resume
        if step_id in resume_completed:
            tracker.skip_step(step_id, "Already completed (resumed)")
            continue

        try:
            success = step_fn(tracker, args)
            if not success:
                all_ok = False
                # Fatal steps stop the install
                if step_id in ("python_check", "venv_setup", "deps_install"):
                    print(f"\n  ✗ Fatal step '{step_id}' failed — cannot continue")
                    break
        except KeyboardInterrupt:
            tracker.complete_step(step_id, success=False, error="Cancelled by user")
            all_ok = False
            break
        except Exception as e:
            tracker.complete_step(step_id, success=False, error=str(e))
            all_ok = False
            if step_id in ("python_check", "venv_setup", "deps_install"):
                break

    tracker.finish_install(success=all_ok)
    print_completion(all_ok, tracker)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
