#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: install_slate [python]
# Author: COPILOT | Created: 2026-02-06T00:30:00Z | Modified: 2026-02-06T22:30:00Z
# Purpose: SLATE public installation script — delegates to slate_installer for full ecosystem
# ═══════════════════════════════════════════════════════════════════════════════
"""
S.L.A.T.E. Installation Script
===============================
Installs and configures the complete SLATE ecosystem with real-time dashboard
progress tracking. The dashboard is the FIRST system installed — every
subsequent step is visible in the browser at http://127.0.0.1:8080.

Full ecosystem setup includes:
    - Git repository clone/verify
    - Python virtual environment
    - pip dependencies (requirements.txt)
    - PyTorch (GPU-aware — auto-detects CUDA compute capability)
    - Ollama (local LLM inference — auto-installs via winget)
    - Docker (container deployment — detection + guidance)
    - VS Code extension (@slate chat participant)
    - SLATE custom models (slate-coder, slate-fast, slate-planner)
    - Workspace configuration + .env

Architecture:
    install_slate.py → SlateInstaller → InstallTracker → install_state.json
                                      → SSE broadcast  → Dashboard listens

Usage:
    python install_slate.py                     # Full ecosystem install with dashboard
    python install_slate.py --no-dashboard      # CLI-only (no browser)
    python install_slate.py --skip-gpu          # Skip GPU detection
    python install_slate.py --beta              # Init from S.L.A.T.E.-BETA fork
    python install_slate.py --dev               # Developer mode (editable install)
    python install_slate.py --resume            # Resume a failed install
    python install_slate.py --update            # Update mode — pull latest + re-validate
    python install_slate.py --check             # Check ecosystem dependencies only
"""

import argparse
import json
import os
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
            err_lines = [line for line in (result.stderr or "").splitlines() if "ERROR" in line]
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


# Modified: 2026-02-06T22:30:00Z | Author: COPILOT | Change: New ecosystem setup steps

def step_pytorch_setup(tracker, args):
    """Step 4b: Install PyTorch with GPU support if available."""
    tracker.start_step("pytorch_setup")

    if args.skip_gpu:
        tracker.skip_step("pytorch_setup", "Skipped (--skip-gpu)")
        return True

    # Check if already installed
    tracker.update_progress("pytorch_setup", 20, "Checking PyTorch installation")
    try:
        check = _run_cmd([_get_python_exe(), "-c",
                          "import torch; print(torch.__version__); "
                          "print(torch.cuda.is_available()); "
                          "print(torch.version.cuda or 'none')"], timeout=30)
        if check.returncode == 0:
            lines = check.stdout.strip().splitlines()
            if len(lines) >= 3:
                version = lines[0]
                cuda_ok = lines[1].strip().lower() == "true"
                cuda_ver = lines[2] if lines[2] != "none" else "N/A"
                if cuda_ok:
                    tracker.complete_step("pytorch_setup", success=True,
                                          details=f"PyTorch {version} with CUDA {cuda_ver}")
                    return True
                else:
                    tracker.update_progress("pytorch_setup", 30,
                                            f"PyTorch {version} found but no CUDA — upgrading")
    except Exception:
        pass

    # Detect GPU compute capability for correct CUDA wheel
    tracker.update_progress("pytorch_setup", 40, "Detecting GPU for PyTorch CUDA version")
    pip = _get_pip_exe()
    cuda_tag = "cpu"

    try:
        gpu_result = _run_cmd(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            timeout=15,
        )
        if gpu_result.returncode == 0 and gpu_result.stdout.strip():
            ccs = gpu_result.stdout.strip().splitlines()
            max_cc = max(float(cc.strip()) for cc in ccs if cc.strip())
            if max_cc >= 8.9:  # Ada Lovelace / Blackwell
                cuda_tag = "cu124"
            elif max_cc >= 8.0:  # Ampere
                cuda_tag = "cu121"
            else:
                cuda_tag = "cu118"
            tracker.update_progress("pytorch_setup", 50,
                                    f"GPU compute {max_cc} → installing with {cuda_tag}")
    except Exception:
        tracker.update_progress("pytorch_setup", 50, "No GPU detected — installing CPU version")

    # Install PyTorch
    tracker.update_progress("pytorch_setup", 60, f"Installing PyTorch ({cuda_tag})...")
    try:
        install_cmd = [str(pip), "install", "torch", "torchvision", "torchaudio"]
        if cuda_tag != "cpu":
            install_cmd += ["--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"]
        else:
            install_cmd += ["--index-url", "https://download.pytorch.org/whl/cpu"]

        result = _run_cmd(install_cmd, timeout=600)
        if result.returncode == 0:
            # Verify
            verify = _run_cmd([_get_python_exe(), "-c",
                               "import torch; print(torch.__version__, "
                               "'CUDA' if torch.cuda.is_available() else 'CPU')"], timeout=15)
            details = verify.stdout.strip() if verify.returncode == 0 else "installed"
            tracker.complete_step("pytorch_setup", success=True,
                                  details=f"PyTorch {details}")
        else:
            tracker.complete_step("pytorch_setup", success=True, warning=True,
                                  details="PyTorch install failed (non-fatal)")
        return True
    except subprocess.TimeoutExpired:
        tracker.complete_step("pytorch_setup", success=True, warning=True,
                              details="PyTorch install timed out (non-fatal)")
        return True
    except Exception as e:
        tracker.complete_step("pytorch_setup", success=True, warning=True,
                              details=f"PyTorch setup error: {e}")
        return True


def step_ollama_setup(tracker, args):
    """Step 4c: Check/install Ollama for local LLM inference."""
    tracker.start_step("ollama_setup")
    tracker.update_progress("ollama_setup", 20, "Checking Ollama installation")

    try:
        result = _run_cmd(["ollama", "--version"], timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            # Check if running
            list_result = _run_cmd(["ollama", "list"], timeout=10)
            if list_result.returncode == 0:
                models = [l.split()[0] for l in list_result.stdout.strip().splitlines()[1:] if l.strip()]
                tracker.complete_step("ollama_setup", success=True,
                                      details=f"Ollama {version} running, {len(models)} model(s)")
            else:
                tracker.complete_step("ollama_setup", success=True, warning=True,
                                      details=f"Ollama {version} installed but not running — start with: ollama serve")
            return True
    except FileNotFoundError:
        pass

    # Not installed — try automatic install
    tracker.update_progress("ollama_setup", 40, "Ollama not found — attempting install")

    if os.name == "nt":
        # Try winget
        try:
            winget_check = _run_cmd(["winget", "--version"], timeout=10)
            if winget_check.returncode == 0:
                tracker.update_progress("ollama_setup", 50, "Installing Ollama via winget...")
                result = _run_cmd(
                    ["winget", "install", "Ollama.Ollama",
                     "--accept-package-agreements", "--accept-source-agreements"],
                    timeout=300,
                )
                if result.returncode == 0:
                    tracker.complete_step("ollama_setup", success=True,
                                          details="Ollama installed via winget — restart terminal and run: ollama serve")
                    return True
        except FileNotFoundError:
            pass

    tracker.complete_step("ollama_setup", success=True, warning=True,
                          details="Ollama not installed. Install from: https://ollama.com/download")
    return True


def step_docker_check(tracker, args):
    """Step 4d: Check Docker installation."""
    tracker.start_step("docker_check")
    tracker.update_progress("docker_check", 30, "Checking Docker")

    try:
        result = _run_cmd(["docker", "--version"], timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            # Check daemon
            ps_result = _run_cmd(["docker", "ps"], timeout=10)
            running = ps_result.returncode == 0
            # Check compose
            compose = _run_cmd(["docker", "compose", "version"], timeout=10)
            compose_ok = compose.returncode == 0

            status_parts = [version]
            if running:
                status_parts.append("daemon running")
            else:
                status_parts.append("daemon not running")
            if compose_ok:
                status_parts.append("compose available")

            tracker.complete_step("docker_check", success=True,
                                  details=" | ".join(status_parts))
            return True
    except FileNotFoundError:
        pass

    tracker.complete_step("docker_check", success=True, warning=True,
                          details="Docker not installed (optional). Install: https://docs.docker.com/get-docker/")
    return True


def step_vscode_extension(tracker, args):
    """Step 7b: Install SLATE VS Code extension."""
    tracker.start_step("vscode_ext")
    tracker.update_progress("vscode_ext", 20, "Checking VS Code")

    try:
        code_check = _run_cmd(["code", "--version"], timeout=15)
        if code_check.returncode != 0:
            tracker.complete_step("vscode_ext", success=True, warning=True,
                                  details="VS Code 'code' command not on PATH")
            return True
    except FileNotFoundError:
        tracker.complete_step("vscode_ext", success=True, warning=True,
                              details="VS Code not detected — install code command in PATH")
        return True

    # Check if extension already installed
    tracker.update_progress("vscode_ext", 40, "Checking for SLATE extension")
    ext_result = _run_cmd(["code", "--list-extensions"], timeout=15)
    if ext_result.returncode == 0:
        extensions = ext_result.stdout.strip().splitlines()
        if any("slate" in ext.lower() for ext in extensions):
            tracker.complete_step("vscode_ext", success=True,
                                  details="SLATE extension already installed")
            return True

    # Build and install from source
    ext_dir = WORKSPACE_ROOT / "plugins" / "slate-copilot"
    if not ext_dir.exists():
        tracker.complete_step("vscode_ext", success=True, warning=True,
                              details="Extension source not found at plugins/slate-copilot/")
        return True

    tracker.update_progress("vscode_ext", 50, "Building SLATE extension")

    # Check npm
    try:
        npm_check = _run_cmd(["npm", "--version"], timeout=10)
        if npm_check.returncode != 0:
            tracker.complete_step("vscode_ext", success=True, warning=True,
                                  details="npm not found — install Node.js to build extension")
            return True
    except FileNotFoundError:
        tracker.complete_step("vscode_ext", success=True, warning=True,
                              details="npm not found — install Node.js first")
        return True

    # npm install + compile
    tracker.update_progress("vscode_ext", 60, "Installing extension dependencies")
    npm_install = _run_cmd(["npm", "install"], timeout=120)
    if npm_install.returncode != 0:
        tracker.complete_step("vscode_ext", success=True, warning=True,
                              details="npm install failed for extension")
        return True

    tracker.update_progress("vscode_ext", 70, "Compiling extension")
    compile_result = subprocess.run(
        ["npm", "run", "compile"], capture_output=True, text=True,
        timeout=60, cwd=str(ext_dir),
    )
    if compile_result.returncode != 0:
        tracker.complete_step("vscode_ext", success=True, warning=True,
                              details="Extension TypeScript compilation failed")
        return True

    # Try vsce package + install
    tracker.update_progress("vscode_ext", 80, "Packaging extension")
    try:
        # Install vsce if needed
        _run_cmd(["npm", "install", "-g", "@vscode/vsce"], timeout=60)
        pkg = subprocess.run(
            ["vsce", "package", "--no-dependencies"],
            capture_output=True, text=True, timeout=60, cwd=str(ext_dir),
        )
        if pkg.returncode == 0:
            vsix_files = list(ext_dir.glob("*.vsix"))
            if vsix_files:
                vsix = vsix_files[-1]
                tracker.update_progress("vscode_ext", 90, f"Installing {vsix.name}")
                inst = _run_cmd(["code", "--install-extension", str(vsix)], timeout=60)
                if inst.returncode == 0:
                    tracker.complete_step("vscode_ext", success=True,
                                          details="SLATE extension installed — reload VS Code")
                    return True
    except Exception:
        pass

    # Fallback: dev mode symlink
    tracker.update_progress("vscode_ext", 90, "Using development mode link")
    # Modified: 2026-02-07T06:00:00Z | Author: COPILOT | Change: Sync extension dev-link version to 2.6.1
    extension_version = "2.6.1"
    if os.name == "nt":
        ext_target = Path(os.environ.get("USERPROFILE", "~")) / ".vscode" / "extensions" / f"slate.slate-copilot-{extension_version}"
    else:
        ext_target = Path.home() / ".vscode" / "extensions" / f"slate.slate-copilot-{extension_version}"

    try:
        ext_target.parent.mkdir(parents=True, exist_ok=True)
        if ext_target.exists():
            import shutil
            shutil.rmtree(ext_target, ignore_errors=True)
        if os.name == "nt":
            _run_cmd(["cmd", "/c", "mklink", "/J", str(ext_target), str(ext_dir)], timeout=10)
        else:
            ext_target.symlink_to(ext_dir)
        tracker.complete_step("vscode_ext", success=True,
                              details="SLATE extension linked (dev mode) — reload VS Code")
    except Exception as e:
        tracker.complete_step("vscode_ext", success=True, warning=True,
                              details=f"Extension link failed: {e}")
    return True


def step_sdk_validate(tracker, args):
    """Step 5: Validate SLATE SDK imports and version."""
    tracker.start_step("sdk_validate")
    tracker.update_progress("sdk_validate", 30, "Importing SLATE SDK")

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
                                  details="slate SDK not importable (first install)")
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

    # Configure remotes based on --beta flag
    if args.beta:
        tracker.update_progress("git_sync", 50, "Configuring beta fork remote")
        try:
            from slate.slate_fork_manager import SlateForkManager
            manager = SlateForkManager(str(WORKSPACE_ROOT))
            manager.configure_beta_remote()
            tracker.complete_step("git_sync", success=True,
                                  details="Git synced with S.L.A.T.E.-BETA fork")
            return True
        except ImportError:
            _run_cmd(["git", "remote", "add", "beta",
                       "https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.-BETA.git"],
                      timeout=10)
            tracker.complete_step("git_sync", success=True,
                                  details="Beta remote added (fork manager not available)")
            return True
    else:
        # Standard sync — just verify remote
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
                details = f"Benchmark complete -- Overall score: {score}"
            except (json.JSONDecodeError, ValueError):
                details = "Benchmark complete"
            tracker.complete_step("benchmark", success=True, details=details)
        else:
            tracker.complete_step("benchmark", success=True, warning=True,
                                  details="Benchmark ran with warnings")
        return True
    except subprocess.TimeoutExpired:
        tracker.complete_step("benchmark", success=True, warning=True,
                              details="Benchmark timed out after 120s -- skipping")
        return True
    except Exception as e:
        tracker.complete_step("benchmark", success=True, warning=True,
                              details=f"Benchmark skipped: {e}")
        return True


# Modified: 2026-02-07T06:00:00Z | Author: COPILOT | Change: Add ecosystem steps for models, skills, ChromaDB, GPU, runner
def step_slate_models(tracker, args):
    """Step: Build SLATE custom Ollama models (slate-coder, slate-fast, slate-planner)."""
    tracker.start_step("slate_models")
    tracker.update_progress("slate_models", 10, "Checking Ollama availability")

    python_exe = _get_python_exe()
    if not python_exe.exists():
        tracker.skip_step("slate_models", "Python venv not available")
        return True

    # Check if Ollama is running
    try:
        ollama_check = _run_cmd([python_exe, "-c",
            "import subprocess; r = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10); "
            "print('OK' if r.returncode == 0 else 'FAIL')"], timeout=15)
        if "OK" not in (ollama_check.stdout or ""):
            tracker.complete_step("slate_models", success=True, warning=True,
                                  details="Ollama not running -- models deferred to later")
            return True
    except Exception:
        tracker.complete_step("slate_models", success=True, warning=True,
                              details="Ollama not available -- models deferred")
        return True

    models_dir = WORKSPACE_ROOT / "models"
    modelfiles = list(models_dir.glob("Modelfile.*")) if models_dir.exists() else []
    if not modelfiles:
        tracker.complete_step("slate_models", success=True, warning=True,
                              details="No Modelfile definitions found in models/")
        return True

    tracker.update_progress("slate_models", 30, f"Found {len(modelfiles)} model definitions")

    # Check which models need building
    try:
        list_result = _run_cmd(["ollama", "list"], timeout=15)
        existing = list_result.stdout if list_result.returncode == 0 else ""
    except Exception:
        existing = ""

    needed = []
    for mf in modelfiles:
        model_name = mf.name.replace("Modelfile.", "")
        if model_name not in existing:
            needed.append((model_name, mf))

    if not needed:
        tracker.complete_step("slate_models", success=True,
                              details=f"All {len(modelfiles)} SLATE models already built")
        return True

    tracker.update_progress("slate_models", 40, f"Building {len(needed)} model(s)")
    built = []
    for i, (name, mf) in enumerate(needed):
        pct = 40 + int(50 * (i + 1) / len(needed))
        tracker.update_progress("slate_models", pct, f"Building {name}...")
        try:
            result = _run_cmd(["ollama", "create", name, "-f", str(mf)], timeout=600)
            if result.returncode == 0:
                built.append(name)
        except subprocess.TimeoutExpired:
            pass

    tracker.complete_step("slate_models", success=True,
                          details=f"Built {len(built)}/{len(needed)} models: {', '.join(built) or 'none'}")
    return True


def step_skills_validate(tracker, args):
    """Step: Validate Copilot Chat skills are present."""
    tracker.start_step("skills_validate")
    tracker.update_progress("skills_validate", 20, "Checking skill definitions")

    skills_dir = WORKSPACE_ROOT / "skills"
    expected_skills = ["slate-status", "slate-runner", "slate-orchestrator",
                       "slate-workflow", "slate-help"]
    found = []
    missing = []

    for skill in expected_skills:
        skill_file = skills_dir / skill / "SKILL.md"
        if skill_file.exists():
            found.append(skill)
        else:
            missing.append(skill)

    if missing:
        tracker.complete_step("skills_validate", success=True, warning=True,
                              details=f"Found {len(found)}/{len(expected_skills)} skills, missing: {', '.join(missing)}")
    else:
        tracker.complete_step("skills_validate", success=True,
                              details=f"All {len(expected_skills)} Copilot Chat skills present")
    return True


def step_chromadb_check(tracker, args):
    """Step: Verify ChromaDB vector store integration."""
    tracker.start_step("chromadb_check")
    tracker.update_progress("chromadb_check", 20, "Checking ChromaDB")

    python_exe = _get_python_exe()
    if not python_exe.exists():
        tracker.skip_step("chromadb_check", "Python venv not available")
        return True

    try:
        result = _run_cmd([python_exe, "-c",
            "import chromadb; print(f'chromadb {chromadb.__version__}')"], timeout=15)
        if result.returncode == 0 and "chromadb" in result.stdout:
            version_info = result.stdout.strip()
            tracker.complete_step("chromadb_check", success=True,
                                  details=f"{version_info} available for vector store")
        else:
            tracker.complete_step("chromadb_check", success=True, warning=True,
                                  details="ChromaDB not importable -- pip install chromadb")
    except Exception as e:
        tracker.complete_step("chromadb_check", success=True, warning=True,
                              details=f"ChromaDB check failed: {e}")
    return True


def step_gpu_manager(tracker, args):
    """Step: Configure dual-GPU load balancing."""
    tracker.start_step("gpu_manager")

    if getattr(args, 'skip_gpu', False):
        tracker.skip_step("gpu_manager", "GPU setup skipped (--skip-gpu)")
        return True

    python_exe = _get_python_exe()
    gpu_script = WORKSPACE_ROOT / "slate" / "slate_gpu_manager.py"
    if not python_exe.exists() or not gpu_script.exists():
        tracker.skip_step("gpu_manager", "GPU manager not available")
        return True

    tracker.update_progress("gpu_manager", 30, "Configuring GPU load balancing")
    try:
        result = _run_cmd([python_exe, str(gpu_script), "--status"], timeout=30)
        if result.returncode == 0:
            tracker.complete_step("gpu_manager", success=True,
                                  details="Dual-GPU manager configured")
        else:
            tracker.complete_step("gpu_manager", success=True, warning=True,
                                  details="GPU manager check returned warnings")
    except Exception as e:
        tracker.complete_step("gpu_manager", success=True, warning=True,
                              details=f"GPU manager: {e}")
    return True


def step_runner_check(tracker, args):
    """Step: Verify GitHub Actions self-hosted runner."""
    tracker.start_step("runner_check")
    tracker.update_progress("runner_check", 20, "Checking runner installation")

    runner_dir = WORKSPACE_ROOT / "actions-runner"
    if not runner_dir.exists():
        tracker.complete_step("runner_check", success=True, warning=True,
                              details="Runner directory not found at actions-runner/")
        return True

    config_file = runner_dir / ".runner"
    if config_file.exists():
        tracker.update_progress("runner_check", 60, "Runner configured, checking process")
        # Check if runner process is active
        python_exe = _get_python_exe()
        runner_script = WORKSPACE_ROOT / "slate" / "slate_runner_manager.py"
        if python_exe.exists() and runner_script.exists():
            try:
                result = _run_cmd([python_exe, str(runner_script), "--detect"], timeout=15)
                if result.returncode == 0:
                    tracker.complete_step("runner_check", success=True,
                                          details="Runner configured and detected")
                else:
                    tracker.complete_step("runner_check", success=True, warning=True,
                                          details="Runner configured but not currently active")
            except Exception:
                tracker.complete_step("runner_check", success=True, warning=True,
                                      details="Runner configured, status check timed out")
        else:
            tracker.complete_step("runner_check", success=True,
                                  details="Runner directory and config present")
    else:
        tracker.complete_step("runner_check", success=True, warning=True,
                              details="Runner exists but not configured -- run config.cmd")
    return True


def step_watchdog_setup(tracker, args):
    """Step: Setup and start the service watchdog for auto-restart."""
    tracker.start_step("watchdog_setup")
    tracker.update_progress("watchdog_setup", 20, "Checking watchdog availability")

    python_exe = _get_python_exe()
    watchdog_script = WORKSPACE_ROOT / "slate" / "slate_service_watchdog.py"

    if not python_exe.exists():
        tracker.skip_step("watchdog_setup", "Python venv not available")
        return True

    if not watchdog_script.exists():
        tracker.complete_step("watchdog_setup", success=True, warning=True,
                              details="Watchdog script not found")
        return True

    # Check if watchdog is already running
    tracker.update_progress("watchdog_setup", 40, "Checking if watchdog is already running")
    try:
        pid_file = WORKSPACE_ROOT / ".slate_watchdog.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            # Check if process exists
            if os.name == "nt":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, text=True
                )
                if str(pid) in result.stdout:
                    tracker.complete_step("watchdog_setup", success=True,
                                          details=f"Watchdog already running (PID {pid})")
                    return True
    except Exception:
        pass

    # Start the watchdog in background
    tracker.update_progress("watchdog_setup", 60, "Starting service watchdog")
    try:
        if os.name == "nt":
            # Windows: start detached
            process = subprocess.Popen(
                [str(python_exe), str(watchdog_script), "start"],
                cwd=str(WORKSPACE_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
        else:
            process = subprocess.Popen(
                [str(python_exe), str(watchdog_script), "start"],
                cwd=str(WORKSPACE_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

        # Give it a moment to start
        time.sleep(2)

        # Verify it started
        pid_file = WORKSPACE_ROOT / ".slate_watchdog.pid"
        if pid_file.exists():
            pid = pid_file.read_text().strip()
            tracker.complete_step("watchdog_setup", success=True,
                                  details=f"Watchdog started (PID {pid}) - auto-restart enabled")
        else:
            tracker.complete_step("watchdog_setup", success=True, warning=True,
                                  details="Watchdog started but PID file not found")
        return True

    except Exception as e:
        tracker.complete_step("watchdog_setup", success=True, warning=True,
                              details=f"Watchdog start failed: {e}")
        return True


def step_runtime_check(tracker, args):
    # Modified: 2025-07-12T21:30:00Z | Author: COPILOT | Change: Expand to 8 ecosystem checks
    """Step 9: Final runtime verification — full ecosystem validation."""
    tracker.start_step("runtime_check")
    tracker.update_progress("runtime_check", 10, "Running ecosystem runtime checks")

    checks_passed = 0
    checks_total = 0
    issues = []

    def _check(label, cmd, timeout_s=30):
        nonlocal checks_passed, checks_total
        checks_total += 1
        result = _run_cmd(cmd, timeout=timeout_s)
        if result.returncode == 0:
            checks_passed += 1
            return True
        issues.append(label)
        return False

    def _check_file(label, path):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if path.exists():
            checks_passed += 1
            return True
        issues.append(f"{label} not found")
        return False

    py = _get_python_exe()

    # 1. System health (slate_status.py)
    status_script = WORKSPACE_ROOT / "slate" / "slate_status.py"
    if status_script.exists():
        _check("slate_status.py failed", [py, str(status_script), "--quick"])
    else:
        checks_total += 1
        issues.append("slate_status.py not found")

    tracker.update_progress("runtime_check", 20, f"{checks_passed}/{checks_total} checks")

    # 2. Runtime integrations (slate_runtime.py)
    runtime_script = WORKSPACE_ROOT / "slate" / "slate_runtime.py"
    if runtime_script.exists():
        _check("slate_runtime.py returned errors", [py, str(runtime_script), "--check-all"])
    else:
        checks_total += 1
        issues.append("slate_runtime.py not found")

    # 3. Dashboard server importable
    _check("Dashboard server not importable", [
        py, "-c", "from agents.slate_dashboard_server import app; print('ok')"
    ], timeout_s=15)

    tracker.update_progress("runtime_check", 40, f"{checks_passed}/{checks_total} checks")

    # 4. VS Code extension built
    vsix_pattern = list((WORKSPACE_ROOT / "plugins" / "slate-copilot").glob("*.vsix"))
    checks_total += 1
    if vsix_pattern:
        checks_passed += 1
    else:
        issues.append("VS Code extension VSIX not built")

    # 5. ChromaDB importable
    _check("ChromaDB not importable", [
        py, "-c", "import chromadb; print('ok')"
    ], timeout_s=10)

    tracker.update_progress("runtime_check", 60, f"{checks_passed}/{checks_total} checks")

    # 6. Copilot Chat skills present (all 5)
    skills_dir = WORKSPACE_ROOT / "skills"
    expected_skills = ["slate-status", "slate-runner", "slate-orchestrator",
                       "slate-workflow", "slate-help"]
    checks_total += 1
    missing_skills = [s for s in expected_skills
                      if not (skills_dir / s).is_dir()]
    if not missing_skills:
        checks_passed += 1
    else:
        issues.append(f"Missing skills: {', '.join(missing_skills)}")

    # 7. SLATE core modules importable
    _check("SLATE SDK not importable", [
        py, "-c", "import slate; print('ok')"
    ], timeout_s=10)

    tracker.update_progress("runtime_check", 80, f"{checks_passed}/{checks_total} checks")

    # 8. Ollama models directory present
    _check_file("Ollama Modelfiles", WORKSPACE_ROOT / "models" / "Modelfile.slate-coder")

    tracker.update_progress("runtime_check", 95,
                            f"{checks_passed}/{checks_total} ecosystem checks complete")

    if checks_passed == checks_total:
        tracker.complete_step("runtime_check", success=True,
                              details=f"All {checks_total} ecosystem checks passed")
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
    # Modified: 2025-07-12T21:30:00Z | Author: COPILOT | Change: Comprehensive ecosystem next steps
    """Print completion message with full ecosystem guidance."""
    print()
    print("═" * 70)
    if success:
        print("  ✓ S.L.A.T.E. Installation Complete!")
    else:
        print("  ✗ S.L.A.T.E. Installation Failed")
    print("═" * 70)
    print()

    if success:
        print("  ───── Quick Start ─────")
        print()
        if os.name == "nt":
            print("    1. Activate venv:   .\\.venv\\Scripts\\activate")
        else:
            print("    1. Activate venv:   source .venv/bin/activate")
        print("    2. System health:   python slate/slate_status.py --quick")
        print("    3. Full runtime:    python slate/slate_runtime.py --check-all")
        print("    4. Start services:  python slate/slate_orchestrator.py start")
        print("    5. Open dashboard:  http://127.0.0.1:8080")
        print()
        print("  ───── VS Code Integration ─────")
        print()
        print("    • Reload VS Code to activate the @slate extension")
        print("    • Click the SLATE icon in the Activity Bar for the dashboard")
        print("    • Use @slate /status in Copilot Chat for system health")
        print("    • Use @slate /help for all available commands")
        print()
        print("  ───── GPU & AI (optional) ─────")
        print()
        print("    • PyTorch:     python slate/slate_hardware_optimizer.py --install-pytorch")
        print("    • GPU detect:  python slate/slate_hardware_optimizer.py --optimize")
        print("    • Ollama:      ollama serve  (then: ollama create slate-coder -f models/Modelfile.slate-coder)")
        print("    • Benchmarks:  python slate/slate_benchmark.py")
        print()
        print("  ───── Ecosystem Validation ─────")
        print()
        print("    • Full check:  python install_slate.py --check")
        print("    • Update:      python install_slate.py --update")
        print("    • Installer:   python slate/slate_installer.py --check")
        print()
    else:
        print("  ───── Troubleshooting ─────")
        print()
        print("    • Install log:    cat .slate_install/install.log")
        print("    • Install state:  cat .slate_install/install_state.json")
        print("    • Resume:         python install_slate.py --resume")
        print("    • Skip GPU:       python install_slate.py --skip-gpu")
        print("    • Full ecosystem: python slate/slate_installer.py --install")
        print("    • Manual venv:    python -m venv .venv && pip install -r requirements.txt")
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
  python install_slate.py --beta            Initialize from S.L.A.T.E.-BETA fork
  python install_slate.py --resume          Resume a previously failed install
  python install_slate.py --dev             Developer mode (verbose + editable)
  python install_slate.py --update          Update from git + re-validate ecosystem
  python install_slate.py --check           Check ecosystem dependencies only
  python install_slate.py --full            Full ecosystem install (PyTorch, Ollama, Docker, VS Code ext)
        """,
    )
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable live dashboard (CLI output only)")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip NVIDIA GPU detection step")
    parser.add_argument("--beta", action="store_true",
                        help="Initialize from S.L.A.T.E.-BETA fork instead of upstream")
    parser.add_argument("--dev", action="store_true",
                        help="Developer mode — verbose output, editable install")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously failed installation")
    parser.add_argument("--update", action="store_true",
                        help="Update mode — pull latest and re-validate ecosystem")
    parser.add_argument("--check", action="store_true",
                        help="Check ecosystem dependencies only")
    parser.add_argument("--full", action="store_true",
                        help="Full ecosystem install (PyTorch, Ollama, Docker, VS Code extension)")
    return parser.parse_args()


def main():
    """Main installation entry point."""
    args = parse_args()

    # ── Route to ecosystem installer for --update, --check, --full ────
    if args.update or args.check or args.full:
        sys.path.insert(0, str(WORKSPACE_ROOT))
        try:
            from slate.slate_installer import SlateInstaller
            installer = SlateInstaller(workspace=WORKSPACE_ROOT)

            if args.check:
                installer.run_check()
                return 0
            elif args.update:
                result = installer.run_update()
                return 0 if result["success"] else 1
            elif args.full:
                result = installer.run_install(beta=args.beta)
                return 0 if result["success"] else 1
        except ImportError as e:
            print(f"  ✗ Could not load ecosystem installer: {e}")
            print("    Run the base install first: python install_slate.py")
            return 1

    # ── Standard install flow (dashboard-first) ──────────────────────
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
    # Modified: 2026-02-07T06:00:00Z | Author: COPILOT | Change: Full ecosystem coverage with all SLATE subsystems
    install_steps = [
        ("dashboard_boot", step_dashboard_boot),
        ("python_check",   step_python_check),
        ("venv_setup",     step_venv_setup),
        ("deps_install",   step_deps_install),
        ("gpu_detect",     step_gpu_detect),
        ("pytorch_setup",  step_pytorch_setup),
        ("ollama_setup",   step_ollama_setup),
        ("docker_check",   step_docker_check),
        ("sdk_validate",   step_sdk_validate),
        ("dirs_create",    step_dirs_create),
        ("git_sync",       step_git_sync),
        ("vscode_ext",     step_vscode_extension),
        ("slate_models",   step_slate_models),
        ("skills_validate", step_skills_validate),
        ("chromadb_check", step_chromadb_check),
        ("gpu_manager",    step_gpu_manager),
        ("runner_check",   step_runner_check),
        ("watchdog_setup", step_watchdog_setup),
        ("benchmark",      step_benchmark),
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
