#!/usr/bin/env python3
# Modified: 2026-02-07T20:00:00Z | Author: COPILOT | Change: Add fork upstream + GitHub Pages setup for contributor installs
"""
S.L.A.T.E. Ecosystem Installer
================================
Comprehensive installer that sets up the entire SLATE ecosystem on a new system.
Handles: git clone, venv, pip deps, PyTorch (GPU-aware), Ollama, Docker, VS Code
extension, personalization, and system validation.

Triggered by:
    - @slate /install   → Full fresh install from git
    - @slate /update    → Pull latest + re-validate ecosystem
    - python slate/slate_installer.py --install   → CLI fresh install
    - python slate/slate_installer.py --update    → CLI update
    - python slate/slate_installer.py --install --personalize → Full install with personalization

Architecture:
    slate_installer.py → InstallTracker → install_state.json ← Dashboard reads
                       → slateRunner.ts  → VS Code extension ← /install /update
                       → slate_personalization.py → Custom logo/theme generation
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Fix Windows console encoding for Unicode characters (═, ✓, ✗, etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Modified: 2026-02-07T06:00:00Z | Author: COPILOT | Change: Sync extension version to 2.6.1
SLATE_REPO = "https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git"
SLATE_BETA_REPO = "https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.-BETA.git"
SLATE_VERSION = "2.4.0"
EXTENSION_ID = "slate.slate-copilot"
EXTENSION_VERSION = "2.6.1"
OLLAMA_DOWNLOAD_WIN = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_DOWNLOAD_LINUX = "https://ollama.ai/install.sh"
DOCKER_DOWNLOAD_WIN = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"


def _is_windows() -> bool:
    return os.name == "nt"


def _run(cmd: list, timeout: int = 120, cwd: str = None, env: dict = None) -> subprocess.CompletedProcess:
    """Run a subprocess with sensible defaults."""
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [str(c) for c in cmd],
        capture_output=True, text=True, timeout=timeout,
        cwd=cwd, env=merged_env,
    )


def _cmd_exists(name: str) -> bool:
    """Check if a command exists on PATH."""
    return shutil.which(name) is not None


def _print_step(icon: str, msg: str):
    """Print a formatted step message."""
    print(f"  {icon} {msg}")


def _print_header(title: str):
    print()
    print(f"  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Modified: 2026-02-06T22:30:00Z | Author: COPILOT | Change: Dependency detection functions

def detect_python() -> dict:
    """Detect Python installation and version."""
    info = {
        "installed": True,
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "executable": sys.executable,
        "meets_requirement": sys.version_info >= (3, 11),
        "platform": platform.platform(),
    }
    return info


def detect_git() -> dict:
    """Detect Git installation."""
    info = {"installed": False, "version": None, "path": None}
    try:
        result = _run(["git", "--version"], timeout=10)
        if result.returncode == 0:
            info["installed"] = True
            info["version"] = result.stdout.strip().replace("git version ", "")
            info["path"] = shutil.which("git")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def detect_nvidia_gpu() -> dict:
    """Detect NVIDIA GPU(s) via nvidia-smi."""
    info = {"installed": False, "gpus": [], "cuda_version": None, "driver_version": None}
    try:
        result = _run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"],
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["installed"] = True
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    info["gpus"].append({
                        "name": parts[0],
                        "compute_cap": parts[1],
                        "memory": parts[2],
                        "driver": parts[3],
                    })
                    info["driver_version"] = parts[3]

        # Get CUDA version
        cuda_result = _run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"], timeout=10)
        if cuda_result.returncode == 0:
            # Also check nvcc
            nvcc = _run(["nvcc", "--version"], timeout=10)
            if nvcc.returncode == 0:
                for line in nvcc.stdout.splitlines():
                    if "release" in line.lower():
                        parts = line.split("release")
                        if len(parts) > 1:
                            info["cuda_version"] = parts[1].strip().split(",")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def detect_pytorch() -> dict:
    """Detect PyTorch installation and CUDA support."""
    info = {"installed": False, "version": None, "cuda_available": False, "cuda_version": None}
    try:
        result = _run([sys.executable, "-c",
                       "import torch; print(torch.__version__); "
                       "print(torch.cuda.is_available()); "
                       "print(torch.version.cuda or 'none')"], timeout=30)
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 3:
                info["installed"] = True
                info["version"] = lines[0]
                info["cuda_available"] = lines[1].strip().lower() == "true"
                info["cuda_version"] = lines[2] if lines[2] != "none" else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def detect_ollama() -> dict:
    """Detect Ollama installation and running state."""
    info = {"installed": False, "version": None, "running": False, "models": []}
    try:
        result = _run(["ollama", "--version"], timeout=10)
        if result.returncode == 0:
            info["installed"] = True
            info["version"] = result.stdout.strip()

        # Check if running
        list_result = _run(["ollama", "list"], timeout=10)
        if list_result.returncode == 0:
            info["running"] = True
            for line in list_result.stdout.strip().splitlines()[1:]:  # Skip header
                parts = line.split()
                if parts:
                    info["models"].append(parts[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def detect_docker() -> dict:
    """Detect Docker installation."""
    info = {"installed": False, "version": None, "running": False, "compose": False}
    try:
        result = _run(["docker", "--version"], timeout=10)
        if result.returncode == 0:
            info["installed"] = True
            info["version"] = result.stdout.strip()

        # Check if daemon is running
        ps_result = _run(["docker", "ps"], timeout=10)
        info["running"] = ps_result.returncode == 0

        # Check docker compose
        compose = _run(["docker", "compose", "version"], timeout=10)
        info["compose"] = compose.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def detect_vscode() -> dict:
    """Detect VS Code installation and installed extensions."""
    info = {"installed": False, "version": None, "slate_extension": False, "extensions": []}
    try:
        result = _run(["code", "--version"], timeout=15)
        if result.returncode == 0:
            info["installed"] = True
            info["version"] = result.stdout.strip().splitlines()[0]

        # Check for slate extension
        ext_result = _run(["code", "--list-extensions"], timeout=15)
        if ext_result.returncode == 0:
            extensions = ext_result.stdout.strip().splitlines()
            info["extensions"] = extensions
            info["slate_extension"] = any(
                "slate" in ext.lower() for ext in extensions
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def detect_all() -> dict:
    """Run all dependency detection and return summary."""
    # Modified: 2026-02-06T22:30:00Z | Author: COPILOT | Change: Full dependency scan
    return {
        "python": detect_python(),
        "git": detect_git(),
        "gpu": detect_nvidia_gpu(),
        "pytorch": detect_pytorch(),
        "ollama": detect_ollama(),
        "docker": detect_docker(),
        "vscode": detect_vscode(),
        "platform": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  INSTALLATION STEPS
# ═══════════════════════════════════════════════════════════════════════════════

# Modified: 2026-02-06T22:30:00Z | Author: COPILOT | Change: Full ecosystem install steps

class SlateInstaller:
    """Full SLATE ecosystem installer."""

    def __init__(self, workspace: Path = None, tracker=None):
        self.workspace = workspace or Path.cwd()
        self.tracker = tracker
        self.results = {}
        self.errors = []
        self._venv_python = None
        self._venv_pip = None
        self._scan_report = None  # Stores dependency scan results

    def _python(self) -> Path:
        if self._venv_python:
            return self._venv_python
        if _is_windows():
            p = self.workspace / ".venv" / "Scripts" / "python.exe"
        else:
            p = self.workspace / ".venv" / "bin" / "python"
        self._venv_python = p
        return p

    def _pip(self) -> Path:
        if self._venv_pip:
            return self._venv_pip
        if _is_windows():
            p = self.workspace / ".venv" / "Scripts" / "pip.exe"
        else:
            p = self.workspace / ".venv" / "bin" / "pip"
        self._venv_pip = p
        return p

    def _log(self, icon: str, msg: str):
        _print_step(icon, msg)
        if self.tracker:
            try:
                self.tracker.add_log(msg)
            except Exception:
                pass

    # ── Step 0: Dependency Scan (SLATE Installation Ethos) ───────────────

    def step_dependency_scan(self) -> bool:
        """
        Scan the system for existing dependencies before installing.

        SLATE Installation Ethos:
            1. SCAN: Detect if dependencies exist outside the environment
            2. LINK: If found, create symlinks/junctions to reuse them
            3. INSTALL: Only install directly to workspace if not found
        """
        _print_header("Step 0: Dependency Scan")

        try:
            from slate.slate_dependency_scanner import DependencyScanner

            scanner = DependencyScanner(workspace=self.workspace, verbose=True)
            self._log("->", "Scanning system for existing dependencies...")

            # Run full scan
            report = scanner.scan_all()
            self._scan_report = report

            # Report findings
            found_count = sum(1 for r in report.results.values() if r.found)
            total_count = len(report.results)
            link_count = sum(1 for r in report.results.values() if r.action == "link")

            self._log("i", f"Scanned {len(report.drives_scanned)} drives")
            self._log("i", f"Found {found_count}/{total_count} dependencies already installed")

            # Show what was found
            for name, result in report.results.items():
                if result.found:
                    action_str = {"link": "-> can reuse", "skip": "-> in workspace", "install": "-> need fresh"}.get(result.action, "")
                    loc_str = f" at {result.best_location.path}" if result.best_location else ""
                    self._log("+", f"{name}: FOUND{loc_str} {action_str}")
                else:
                    self._log("!", f"{name}: NOT FOUND -> will install fresh")

            # Create links for reusable dependencies
            if link_count > 0:
                self._log("->", f"Creating {link_count} links to existing dependencies...")
                links_created = scanner.create_all_links()
                self._log("+", f"Created {links_created} symlinks/junctions")

            self.results["dependency_scan"] = {
                "status": "complete",
                "found": found_count,
                "total": total_count,
                "links_created": len(report.links_created),
                "needs_install": report.needs_install,
            }
            return True

        except ImportError as e:
            self._log("!", f"Dependency scanner not available: {e}")
            self._log("i", "Will perform fresh install of all dependencies")
            self.results["dependency_scan"] = {"status": "skipped", "reason": "scanner_unavailable"}
            return True  # Non-fatal

        except Exception as e:
            self._log("!", f"Dependency scan failed: {e}")
            self._log("i", "Will perform fresh install of all dependencies")
            self.results["dependency_scan"] = {"status": "error", "error": str(e)}
            return True  # Non-fatal

    # ── Step 1: Git Clone / Init ─────────────────────────────────────────

    def step_git_setup(self, target_dir: str = None, beta: bool = False) -> bool:
        """Clone SLATE repository or verify existing git setup."""
        _print_header("Step 1: Git Repository Setup")

        target = Path(target_dir) if target_dir else self.workspace
        repo_url = SLATE_BETA_REPO if beta else SLATE_REPO

        # Check if already in a SLATE repo
        git_dir = target / ".git"
        if git_dir.exists():
            self._log("✓", f"Git repository exists at {target}")
            # Verify remote
            result = _run(["git", "remote", "-v"], cwd=str(target), timeout=10)
            if result.returncode == 0:
                self._log("ℹ", f"Remotes:\n{result.stdout.strip()}")

            # Ensure origin points to SLATE
            if "SynchronizedLivingArchitecture" not in (result.stdout or ""):
                self._log("→", "Adding SLATE upstream remote")
                _run(["git", "remote", "add", "upstream", repo_url],
                     cwd=str(target), timeout=10)
                self._log("✓", "Upstream remote added")

            self.results["git_setup"] = {"status": "exists", "path": str(target)}
            return True

        # Clone fresh
        if not _cmd_exists("git"):
            self._log("✗", "Git is not installed! Please install git first.")
            self._log("ℹ", "  Windows: https://git-scm.com/download/win")
            self._log("ℹ", "  Linux:   sudo apt install git")
            self._log("ℹ", "  macOS:   xcode-select --install")
            self.errors.append("Git not installed")
            return False

        self._log("→", f"Cloning SLATE from {repo_url}")
        self._log("ℹ", f"  Target: {target}")

        try:
            result = _run(
                ["git", "clone", repo_url, str(target)],
                timeout=300,
            )
            if result.returncode == 0:
                self._log("✓", "Repository cloned successfully")
                self.workspace = target
                self.results["git_setup"] = {"status": "cloned", "path": str(target)}
                return True
            else:
                self._log("✗", f"Clone failed: {result.stderr.strip()}")
                self.errors.append(f"Git clone failed: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            self._log("✗", "Git clone timed out (5 minutes)")
            self.errors.append("Git clone timeout")
            return False

    # ── Step 2: Virtual Environment ──────────────────────────────────────

    def step_venv_setup(self) -> bool:
        """Create or verify Python virtual environment."""
        _print_header("Step 2: Python Virtual Environment")

        py_info = detect_python()
        if not py_info["meets_requirement"]:
            self._log("✗", f"Python 3.11+ required, found {py_info['version']}")
            self._log("ℹ", "  Download: https://www.python.org/downloads/")
            self.errors.append(f"Python {py_info['version']} < 3.11")
            return False

        self._log("✓", f"Python {py_info['version']} ({py_info['executable']})")

        venv_path = self.workspace / ".venv"

        # Check if scan found a compatible venv we can link to (SLATE Installation Ethos)
        if self._scan_report and hasattr(self._scan_report, 'results'):
            venv_result = self._scan_report.results.get('venv')
            if venv_result and venv_result.action == "link" and venv_result.best_location:
                source_venv = venv_result.best_location.path
                if not venv_path.exists():
                    self._log("→", f"Found compatible venv at {source_venv}")
                    self._log("→", "Creating link instead of fresh venv (SLATE ethos)...")
                    try:
                        if _is_windows():
                            result = _run(["cmd", "/c", "mklink", "/J", str(venv_path), str(source_venv)], timeout=10)
                            if result.returncode == 0:
                                self._log("✓", f"Linked to existing venv at {source_venv}")
                                self.results["venv_setup"] = {"status": "linked", "source": str(source_venv)}
                                return True
                        else:
                            venv_path.symlink_to(source_venv)
                            self._log("✓", f"Linked to existing venv at {source_venv}")
                            self.results["venv_setup"] = {"status": "linked", "source": str(source_venv)}
                            return True
                    except Exception as e:
                        self._log("⚠", f"Link failed: {e}, will create fresh venv")

        if venv_path.exists() and self._python().exists():
            # Verify it works
            result = _run([str(self._python()), "-c", "import sys; print(sys.version)"], timeout=10)
            if result.returncode == 0:
                self._log("✓", f"Virtual environment OK ({result.stdout.strip().split()[0]})")
                self.results["venv_setup"] = {"status": "exists"}
                return True
            self._log("⚠", "Existing venv is broken, recreating...")
            shutil.rmtree(venv_path, ignore_errors=True)

        self._log("→", "Creating virtual environment...")
        try:
            result = _run([sys.executable, "-m", "venv", str(venv_path)], timeout=60)
            if result.returncode == 0:
                self._log("✓", f"Virtual environment created at {venv_path}")
                self.results["venv_setup"] = {"status": "created"}
                return True
            else:
                self._log("✗", f"venv creation failed: {result.stderr.strip()}")
                self.errors.append("venv creation failed")
                return False
        except subprocess.TimeoutExpired:
            self._log("✗", "venv creation timed out")
            self.errors.append("venv timeout")
            return False

    # ── Step 3: Core Dependencies ────────────────────────────────────────

    def step_core_deps(self) -> bool:
        """Install core pip dependencies from requirements.txt."""
        _print_header("Step 3: Core Dependencies")

        pip = self._pip()
        if not pip.exists():
            self._log("✗", "pip not found in venv")
            self.errors.append("pip missing")
            return False

        # Upgrade pip
        self._log("→", "Upgrading pip...")
        _run([str(pip), "install", "--upgrade", "pip", "--quiet"], timeout=60)

        req_file = self.workspace / "requirements.txt"
        if not req_file.exists():
            self._log("⚠", "requirements.txt not found — skipping core deps")
            self.results["core_deps"] = {"status": "skipped"}
            return True

        pkg_count = sum(1 for line in req_file.read_text(encoding='utf-8').splitlines()
                        if line.strip() and not line.strip().startswith("#"))
        self._log("→", f"Installing {pkg_count} packages from requirements.txt...")

        try:
            result = _run(
                [str(pip), "install", "-r", str(req_file), "--quiet"],
                timeout=600, cwd=str(self.workspace),
            )
            if result.returncode == 0:
                self._log("✓", f"{pkg_count} packages installed")
                self.results["core_deps"] = {"status": "installed", "count": pkg_count}
                return True
            else:
                err = result.stderr.strip().splitlines()[-1] if result.stderr else "Unknown error"
                self._log("✗", f"pip install failed: {err}")
                self.errors.append(f"pip install failed: {err}")
                return False
        except subprocess.TimeoutExpired:
            self._log("✗", "pip install timed out (10 min)")
            self.errors.append("pip install timeout")
            return False

    # ── Step 4: PyTorch (GPU-aware) ──────────────────────────────────────

    def step_pytorch_setup(self) -> bool:
        """Install PyTorch with CUDA support if GPU detected."""
        _print_header("Step 4: PyTorch Setup")

        gpu = detect_nvidia_gpu()
        pytorch = detect_pytorch()

        if pytorch["installed"] and pytorch["cuda_available"]:
            self._log("✓", f"PyTorch {pytorch['version']} already installed with CUDA {pytorch['cuda_version']}")
            self.results["pytorch"] = {"status": "exists", **pytorch}
            return True

        if pytorch["installed"] and not gpu["installed"]:
            self._log("✓", f"PyTorch {pytorch['version']} installed (CPU mode — no GPU detected)")
            self.results["pytorch"] = {"status": "cpu_mode", **pytorch}
            return True

        pip = self._pip()
        if not pip.exists():
            self._log("⚠", "pip not available — skipping PyTorch")
            self.results["pytorch"] = {"status": "skipped"}
            return True

        if gpu["installed"]:
            self._log("→", f"GPU detected: {gpu['gpus'][0]['name'] if gpu['gpus'] else 'Unknown'}")

            # Determine CUDA version for PyTorch
            compute_caps = [g.get("compute_cap", "0") for g in gpu["gpus"]]
            max_cc = max(compute_caps) if compute_caps else "0"

            # Use CUDA 12.4 for Blackwell (cc 12.x) and Ada (8.9+)
            if max_cc.startswith("12.") or float(max_cc) >= 8.9:
                cuda_tag = "cu124"
            elif float(max_cc) >= 8.0:
                cuda_tag = "cu121"
            else:
                cuda_tag = "cu118"

            self._log("→", f"Installing PyTorch with CUDA ({cuda_tag}) for compute {max_cc}...")

            try:
                result = _run(
                    [str(pip), "install",
                     "torch", "torchvision", "torchaudio",
                     "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"],
                    timeout=600, cwd=str(self.workspace),
                )
                if result.returncode == 0:
                    # Verify
                    verify = detect_pytorch()
                    if verify["cuda_available"]:
                        self._log("✓", f"PyTorch {verify['version']} installed with CUDA {verify['cuda_version']}")
                    else:
                        self._log("⚠", f"PyTorch {verify.get('version', '?')} installed but CUDA not available")
                    self.results["pytorch"] = {"status": "installed", **verify}
                    return True
                else:
                    self._log("⚠", "PyTorch GPU install failed, trying CPU fallback...")
            except subprocess.TimeoutExpired:
                self._log("⚠", "PyTorch GPU install timed out, trying CPU...")

        # CPU fallback or no GPU
        self._log("→", "Installing PyTorch (CPU)...")
        try:
            result = _run(
                [str(pip), "install", "torch", "torchvision", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cpu"],
                timeout=600, cwd=str(self.workspace),
            )
            if result.returncode == 0:
                self._log("✓", "PyTorch (CPU) installed")
                self.results["pytorch"] = {"status": "cpu_installed"}
                return True
            else:
                self._log("⚠", "PyTorch installation failed (non-fatal)")
                self.results["pytorch"] = {"status": "failed"}
                return True  # Non-fatal
        except subprocess.TimeoutExpired:
            self._log("⚠", "PyTorch install timed out (non-fatal)")
            self.results["pytorch"] = {"status": "timeout"}
            return True

    # ── Step 5: Ollama ───────────────────────────────────────────────────

    def step_ollama_setup(self) -> bool:
        """Check/install Ollama for local LLM inference."""
        _print_header("Step 5: Ollama Setup")

        ollama = detect_ollama()

        if ollama["installed"]:
            self._log("✓", f"Ollama installed ({ollama['version']})")
            if ollama["running"]:
                self._log("✓", f"Ollama running — {len(ollama['models'])} model(s) loaded")
                if ollama["models"]:
                    for m in ollama["models"][:5]:
                        self._log("  ", f"  • {m}")
            else:
                self._log("ℹ", "Ollama installed but not running. Start with: ollama serve")
            self.results["ollama"] = {"status": "installed", **ollama}
            return True

        # Not installed — guide user
        self._log("⚠", "Ollama not detected")
        if _is_windows():
            self._log("ℹ", "  Install Ollama for Windows:")
            self._log("ℹ", f"  → Download: {OLLAMA_DOWNLOAD_WIN}")
            self._log("ℹ", "  → Or run: winget install Ollama.Ollama")

            # Try winget install
            if _cmd_exists("winget"):
                self._log("→", "Attempting automatic install via winget...")
                try:
                    result = _run(
                        ["winget", "install", "Ollama.Ollama", "--accept-package-agreements",
                         "--accept-source-agreements"],
                        timeout=300,
                    )
                    if result.returncode == 0:
                        self._log("✓", "Ollama installed via winget!")
                        self._log("ℹ", "  You may need to restart your terminal and run: ollama serve")
                        self.results["ollama"] = {"status": "auto_installed"}
                        return True
                    else:
                        self._log("ℹ", "  Automatic install failed — please install manually")
                except subprocess.TimeoutExpired:
                    self._log("ℹ", "  Install timed out — please install manually")
        else:
            self._log("ℹ", "  Install Ollama:")
            self._log("ℹ", "  → curl -fsSL https://ollama.ai/install.sh | sh")

        self.results["ollama"] = {"status": "not_installed"}
        return True  # Non-fatal

    # ── Step 6: Docker ───────────────────────────────────────────────────

    def step_docker_setup(self) -> bool:
        """Check Docker installation."""
        _print_header("Step 6: Docker Setup")

        docker = detect_docker()

        if docker["installed"]:
            self._log("✓", f"Docker installed ({docker['version']})")
            if docker["running"]:
                self._log("✓", "Docker daemon running")
            else:
                self._log("ℹ", "Docker installed but daemon not running")
                self._log("ℹ", "  Start Docker Desktop or run: dockerd")
            if docker["compose"]:
                self._log("✓", "Docker Compose available")
            self.results["docker"] = {"status": "installed", **docker}
            return True

        self._log("ℹ", "Docker not detected (optional for containerized deployment)")
        if _is_windows():
            self._log("ℹ", "  Install: https://docs.docker.com/desktop/install/windows-install/")
            self._log("ℹ", "  Or run: winget install Docker.DockerDesktop")
        elif platform.system() == "Linux":
            self._log("ℹ", "  Install: curl -fsSL https://get.docker.com | sh")
        else:
            self._log("ℹ", "  Install: https://docs.docker.com/desktop/install/mac-install/")

        self.results["docker"] = {"status": "not_installed"}
        return True  # Non-fatal

    # ── Step 7: SLATE SDK Editable Install ───────────────────────────────

    def step_slate_sdk(self) -> bool:
        """Install SLATE as editable package (pip install -e .)."""
        _print_header("Step 7: SLATE SDK Install")

        pip = self._pip()
        pyproject = self.workspace / "pyproject.toml"

        if not pyproject.exists():
            self._log("⚠", "pyproject.toml not found — skipping SDK install")
            self.results["slate_sdk"] = {"status": "skipped"}
            return True

        self._log("→", "Installing SLATE SDK (editable mode)...")
        try:
            result = _run(
                [str(pip), "install", "-e", str(self.workspace), "--quiet"],
                timeout=120, cwd=str(self.workspace),
            )
            if result.returncode == 0:
                # Verify import
                verify = _run(
                    [str(self._python()), "-c",
                     "import slate; print(slate.__version__)"],
                    timeout=15, cwd=str(self.workspace),
                )
                if verify.returncode == 0:
                    version = verify.stdout.strip()
                    self._log("✓", f"SLATE SDK v{version} installed")
                    self.results["slate_sdk"] = {"status": "installed", "version": version}
                else:
                    self._log("⚠", "SLATE SDK installed but import failed")
                    self.results["slate_sdk"] = {"status": "installed_no_import"}
                return True
            else:
                self._log("⚠", f"SDK install failed: {result.stderr.strip()[:200]}")
                self.results["slate_sdk"] = {"status": "failed"}
                return True  # Non-fatal
        except subprocess.TimeoutExpired:
            self._log("⚠", "SDK install timed out")
            self.results["slate_sdk"] = {"status": "timeout"}
            return True

    # ── Step 8: VS Code Extension ────────────────────────────────────────

    def step_vscode_extension(self) -> bool:
        """Install/update the SLATE VS Code extension."""
        _print_header("Step 8: VS Code Extension Setup")

        vsc = detect_vscode()

        if not vsc["installed"]:
            self._log("ℹ", "VS Code not detected on PATH")
            self._log("ℹ", "  The 'code' command must be available in terminal")
            self._log("ℹ", "  VS Code → Cmd+Shift+P → 'Shell Command: Install code in PATH'")
            self.results["vscode_extension"] = {"status": "vscode_not_found"}
            return True  # Non-fatal

        self._log("✓", f"VS Code {vsc['version']}")

        # Check if extension is already installed
        if vsc["slate_extension"]:
            self._log("✓", "SLATE extension already installed")
            self.results["vscode_extension"] = {"status": "exists"}
            return True

        # Build and install from source
        ext_dir = self.workspace / "plugins" / "slate-copilot"
        if ext_dir.exists():
            self._log("→", "Building SLATE extension from source...")

            # Check for npm
            if not _cmd_exists("npm"):
                self._log("⚠", "npm not found — cannot build extension")
                self._log("ℹ", "  Install Node.js: https://nodejs.org/")
                self.results["vscode_extension"] = {"status": "npm_missing"}
                return True

            # npm install
            self._log("→", "Installing extension dependencies...")
            npm_install = _run(["npm", "install"], timeout=120, cwd=str(ext_dir))
            if npm_install.returncode != 0:
                self._log("⚠", "npm install failed")
                self.results["vscode_extension"] = {"status": "npm_install_failed"}
                return True

            # Compile TypeScript
            self._log("→", "Compiling extension...")
            compile_result = _run(["npm", "run", "compile"], timeout=60, cwd=str(ext_dir))
            if compile_result.returncode != 0:
                self._log("⚠", "Extension compilation failed")
                self.results["vscode_extension"] = {"status": "compile_failed"}
                return True

            # Check for vsce
            if not _cmd_exists("vsce"):
                self._log("→", "Installing vsce (VS Code Extension manager)...")
                _run(["npm", "install", "-g", "@vscode/vsce"], timeout=60)

            if _cmd_exists("vsce"):
                # Package as VSIX
                self._log("→", "Packaging extension...")
                pkg_result = _run(
                    ["vsce", "package", "--no-dependencies"],
                    timeout=60, cwd=str(ext_dir),
                )
                if pkg_result.returncode == 0:
                    # Find the .vsix file
                    vsix_files = list(ext_dir.glob("*.vsix"))
                    if vsix_files:
                        vsix = vsix_files[-1]  # Latest
                        self._log("→", f"Installing {vsix.name}...")
                        install_result = _run(
                            ["code", "--install-extension", str(vsix)],
                            timeout=60,
                        )
                        if install_result.returncode == 0:
                            self._log("✓", "SLATE extension installed in VS Code!")
                            self._log("ℹ", "  Reload VS Code to activate: Ctrl+Shift+P → 'Reload Window'")
                            self.results["vscode_extension"] = {"status": "installed"}
                            return True
                        else:
                            self._log("⚠", "Extension install command failed")

            # Fallback: symlink dev extension
            self._log("→", "Falling back to development mode (symlink)...")
            if _is_windows():
                ext_target = Path(os.environ.get("USERPROFILE", "~")) / ".vscode" / "extensions" / f"slate.slate-copilot-{EXTENSION_VERSION}"
            else:
                ext_target = Path.home() / ".vscode" / "extensions" / f"slate.slate-copilot-{EXTENSION_VERSION}"

            try:
                ext_target.parent.mkdir(parents=True, exist_ok=True)
                if ext_target.exists():
                    if ext_target.is_symlink():
                        ext_target.unlink()
                    else:
                        shutil.rmtree(ext_target, ignore_errors=True)

                if _is_windows():
                    # Windows: use junction or copy
                    result = _run(["cmd", "/c", "mklink", "/J", str(ext_target), str(ext_dir)], timeout=10)
                    if result.returncode != 0:
                        shutil.copytree(ext_dir, ext_target, dirs_exist_ok=True)
                else:
                    ext_target.symlink_to(ext_dir)

                self._log("✓", "SLATE extension linked in dev mode")
                self._log("ℹ", "  Reload VS Code to activate")
                self.results["vscode_extension"] = {"status": "dev_linked"}
                return True
            except Exception as e:
                self._log("⚠", f"Extension link failed: {e}")
                self.results["vscode_extension"] = {"status": "link_failed"}
                return True
        else:
            self._log("ℹ", "Extension source not found at plugins/slate-copilot/")
            self.results["vscode_extension"] = {"status": "source_missing"}
            return True

    # ── Step 9: SLATE Models (Ollama) ────────────────────────────────────

    def step_slate_models(self) -> bool:
        """Build SLATE custom Ollama models if Ollama is available."""
        _print_header("Step 9: SLATE Custom Models")

        ollama = detect_ollama()
        if not ollama["installed"] or not ollama["running"]:
            self._log("ℹ", "Ollama not available — skipping model setup")
            self._log("ℹ", "  Run this later: python slate/slate_model_trainer.py --build-all")
            self.results["slate_models"] = {"status": "skipped"}
            return True

        models_dir = self.workspace / "models"
        modelfiles = list(models_dir.glob("Modelfile.*")) if models_dir.exists() else []

        if not modelfiles:
            self._log("ℹ", "No Modelfile definitions found — skipping")
            self.results["slate_models"] = {"status": "no_modelfiles"}
            return True

        # Check which models already exist
        existing = set(ollama.get("models", []))
        needed = []
        for mf in modelfiles:
            model_name = mf.name.replace("Modelfile.", "")
            if model_name not in existing and f"{model_name}:latest" not in existing:
                needed.append((model_name, mf))

        if not needed:
            self._log("✓", f"All {len(modelfiles)} SLATE models already built")
            self.results["slate_models"] = {"status": "exists", "models": [m.name.replace("Modelfile.", "") for m in modelfiles]}
            return True

        self._log("→", f"Building {len(needed)} SLATE model(s)...")
        built = []
        for name, mf in needed:
            self._log("→", f"  Building {name}...")
            try:
                result = _run(
                    ["ollama", "create", name, "-f", str(mf)],
                    timeout=600, cwd=str(self.workspace),
                )
                if result.returncode == 0:
                    self._log("✓", f"  {name} built")
                    built.append(name)
                else:
                    self._log("⚠", f"  {name} failed: {result.stderr.strip()[:100]}")
            except subprocess.TimeoutExpired:
                self._log("⚠", f"  {name} build timed out")

        self._log("✓" if built else "⚠", f"{len(built)}/{len(needed)} models built")
        self.results["slate_models"] = {"status": "built", "built": built, "total": len(needed)}
        return True

    # ── Step 10: Personalization (Fork Identity) ─────────────────────────

    def step_personalization(self, interactive: bool = True, fork_name: str = None) -> bool:
        """Set up personalized fork identity with custom name, logo, and theme."""
        _print_header("Step 10: Personalization - Create Your SLATE Identity")

        try:
            from slate.slate_personalization import (
                PersonalizationManager,
                validate_fork_name,
                get_name_suggestions,
                COLOR_PALETTES,
            )
        except ImportError as e:
            self._log("⚠", f"Personalization module not available: {e}")
            self.results["personalization"] = {"status": "skipped", "reason": "module_missing"}
            return True  # Non-fatal

        manager = PersonalizationManager(workspace_dir=self.workspace)

        # Check if already configured
        if manager.is_configured():
            config = manager.get_config()
            self._log("✓", f"Already personalized as: {config.fork_name}")
            self.results["personalization"] = {"status": "exists", "name": config.fork_name}
            return True

        if not interactive:
            # Non-interactive mode - use provided name or default
            if fork_name:
                valid, error = validate_fork_name(fork_name)
                if not valid:
                    self._log("⚠", f"Invalid fork name '{fork_name}': {error}")
                    fork_name = "SLATE"
            else:
                fork_name = "SLATE"

            try:
                config = manager.quick_setup(fork_name=fork_name)
                self._log("✓", f"Created identity: {config.fork_name}")
                self.results["personalization"] = {"status": "created", "name": config.fork_name}
                return True
            except Exception as e:
                self._log("⚠", f"Quick personalization failed: {e}")
                self.results["personalization"] = {"status": "failed", "error": str(e)}
                return True  # Non-fatal

        # Interactive mode
        print()
        print("  Would you like to give your SLATE fork a unique identity?")
        print("  This includes:")
        print("    • A custom name (e.g., PHOENIX, NOVA, ATLAS)")
        print("    • A personalized logo with your name")
        print("    • Custom color theme and UI preferences")
        print()

        response = input("  Set up personalization now? (y/n) [y]: ").strip().lower()

        if response == 'n':
            self._log("ℹ", "Skipping personalization (run later with: python slate/slate_personalization.py --setup)")
            self.results["personalization"] = {"status": "skipped", "reason": "user_declined"}
            return True

        try:
            config = manager.run_interactive_setup()
            self._log("✓", f"Created identity: {config.fork_name}")
            self.results["personalization"] = {
                "status": "created",
                "name": config.fork_name,
                "palette": config.color_palette,
                "theme": config.ui_theme,
            }
            return True
        except KeyboardInterrupt:
            self._log("ℹ", "Personalization cancelled")
            self.results["personalization"] = {"status": "cancelled"}
            return True
        except Exception as e:
            self._log("⚠", f"Personalization failed: {e}")
            self.results["personalization"] = {"status": "failed", "error": str(e)}
            return True  # Non-fatal

    # ── Step 11: Workspace Directories & Config ──────────────────────────

    def step_workspace_setup(self) -> bool:
        """Create workspace directories and configuration."""
        _print_header("Step 11: Workspace Configuration")

        dirs = [
            "slate", "agents", "tests", "slate_web", "slate_logs",
            "slate_memory", "models", "plugins", "skills",
            ".github", ".slate_install", ".slate_identity", "logs", "data",
        ]
        created = 0
        for d in dirs:
            dp = self.workspace / d
            if not dp.exists():
                dp.mkdir(parents=True, exist_ok=True)
                created += 1

        # Ensure __init__.py files
        for pkg in ["slate", "agents", "tests"]:
            init = self.workspace / pkg / "__init__.py"
            init.touch(exist_ok=True)

        self._log("✓", f"Workspace configured — {created} directories created")

        # Create .env if it doesn't exist
        env_file = self.workspace / ".env"
        if not env_file.exists():
            # Get fork identity name if personalization was run
            fork_name = "SLATE"
            if "personalization" in self.results:
                fork_name = self.results["personalization"].get("name", "SLATE")

            env_content = (
                f"# {fork_name} Environment Configuration\n"
                f"# Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                f"SLATE_FORK_NAME={fork_name}\n"
                "SLATE_VERSION=2.4.0\n"
                f"SLATE_WORKSPACE={self.workspace}\n"
                "SLATE_HOST=127.0.0.1\n"
                "SLATE_DASHBOARD_PORT=8080\n"
                "CUDA_VISIBLE_DEVICES=0,1\n"
                "PYTHONIOENCODING=utf-8\n"
            )
            env_file.write_text(env_content, encoding='utf-8')
            self._log("✓", ".env configuration created")
        else:
            self._log("✓", ".env already exists")

        self.results["workspace"] = {"status": "configured", "created_dirs": created}
        return True

    # ── Step 11b: Generative UI Setup (Logo + Design Tokens) ─────────────

    def step_generative_ui(self) -> bool:
        """Generate logos and design tokens for the SLATE UI."""
        _print_header("Step 11b: Generative UI Assets")

        identity_dir = self.workspace / ".slate_identity"
        identity_dir.mkdir(parents=True, exist_ok=True)
        logos_dir = identity_dir / "logos"
        logos_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate design tokens
        try:
            result = _run([
                str(self._python()), "-c",
                "from slate.design_tokens import DesignTokens; from pathlib import Path; "
                f"t = DesignTokens(); t.save_css(Path('{identity_dir}/design-tokens.css')); "
                f"t.save_json(Path('{identity_dir}/design-tokens.json')); print('Tokens generated')"
            ], timeout=30, cwd=str(self.workspace))
            if result.returncode == 0:
                self._log("✓", "Design tokens generated")
            else:
                self._log("⚠", "Design tokens generation failed (non-critical)")
        except Exception as e:
            self._log("⚠", f"Design tokens: {e}")

        # 2. Generate starburst logo
        try:
            result = _run([
                str(self._python()), "-c",
                "from slate.logo_generator.starburst import StarburstLogo, StarburstConfig; "
                "from pathlib import Path; "
                f"logos_dir = Path('{logos_dir}'); "
                "logo = StarburstLogo(); logo.save(logos_dir / 'slate-logo.svg'); "
                "dark = StarburstLogo(StarburstConfig(ray_color='#D4785A', center_fill='#D4785A', "
                "letter_color='#1A1816', background='#1A1816')); "
                "dark.save(logos_dir / 'slate-logo-dark.svg'); "
                "animated = StarburstLogo(StarburstConfig(animate_pulse=True)); "
                "animated.save(logos_dir / 'slate-logo-animated.svg'); "
                "print('Logos generated')"
            ], timeout=30, cwd=str(self.workspace))
            if result.returncode == 0:
                self._log("✓", "Starburst logos generated (default, dark, animated)")
            else:
                self._log("⚠", f"Logo generation failed: {result.stderr.strip()[:100]}")
        except Exception as e:
            self._log("⚠", f"Logo generation: {e}")

        # 3. Initialize theme value
        theme_file = identity_dir / "theme_value.json"
        if not theme_file.exists():
            theme_file.write_text('{"value": 0.15}', encoding='utf-8')
            self._log("✓", "Theme preference initialized (dark mode)")
        else:
            self._log("✓", "Theme preference exists")

        self.results["generative_ui"] = {"status": "configured", "logos_dir": str(logos_dir)}
        return True

    # ── Step 12: Guided Mode Setup ────────────────────────────────────────

    def step_guided_mode(self, auto_launch: bool = False) -> bool:
        """
        Configure guided mode for new users.

        Saves guided mode state and optionally offers to launch
        the interactive guided setup experience.
        """
        _print_header("Step 12: Guided Mode Setup")

        state_dir = self.workspace / ".slate_identity"
        state_file = state_dir / "guided_mode_state.json"

        # Check if guided mode was already completed
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                if state.get("completed"):
                    self._log("✓", "Guided mode previously completed")
                    self.results["guided_mode"] = {"status": "already_completed"}
                    return True
            except Exception:
                pass

        # Initialize guided mode state
        state_dir.mkdir(parents=True, exist_ok=True)
        initial_state = {
            "completed": False,
            "installer_version": SLATE_VERSION,
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "steps_completed": [],
            "auto_launch_offered": auto_launch,
        }

        try:
            state_file.write_text(json.dumps(initial_state, indent=2))
            self._log("✓", "Guided mode state initialized")
        except Exception as e:
            self._log("⚠", f"Could not save guided mode state: {e}")

        # Check if guided_mode.py exists
        guided_py = self.workspace / "slate" / "guided_mode.py"
        if guided_py.exists():
            self._log("✓", "Guided mode module available")
            self._log("→", "Run 'python slate/guided_mode.py' for interactive setup")
        else:
            self._log("⚠", "Guided mode module not found")

        self.results["guided_mode"] = {
            "status": "configured",
            "state_file": str(state_file),
            "ready": guided_py.exists(),
        }
        return True

    # ── Step 13: Final Validation ────────────────────────────────────────

    # ── Step 12b: Fork Upstream Configuration ──────────────────────────

    # Modified: 2026-02-07T20:00:00Z | Author: COPILOT | Change: Add fork upstream + Pages setup for contributor installs
    def step_fork_upstream(self) -> bool:
        """Configure fork's upstream remote back to SynchronizedLivingArchitecture/S.L.A.T.E main."""
        _print_header("Step 12b: Fork Upstream Configuration")

        if not _cmd_exists("git"):
            self._log("⚠", "Git not available — skipping upstream setup")
            self.results["fork_upstream"] = {"status": "skipped", "reason": "no_git"}
            return True

        # Check current remotes
        result = _run(["git", "remote", "-v"], cwd=str(self.workspace), timeout=10)
        remotes = result.stdout if result.returncode == 0 else ""

        has_origin = "origin" in remotes
        has_upstream = "upstream" in remotes
        is_fork = has_origin and "SynchronizedLivingArchitecture/S.L.A.T.E" not in remotes.split("origin")[1].split("\n")[0] if has_origin and "origin" in remotes else False

        if has_upstream:
            self._log("✓", "Upstream remote already configured")
            self.results["fork_upstream"] = {"status": "exists"}
            return True

        # If origin IS the main repo (not a fork), no upstream needed
        if has_origin and "SynchronizedLivingArchitecture/S.L.A.T.E" in remotes:
            # Check if this is a fork by looking at origin URL
            origin_line = [l for l in remotes.splitlines() if l.startswith("origin") and "(fetch)" in l]
            if origin_line:
                origin_url = origin_line[0].split()[1]
                if "SynchronizedLivingArchitecture/S.L.A.T.E" in origin_url:
                    self._log("✓", "Origin points to upstream (direct clone, not a fork)")
                    self.results["fork_upstream"] = {"status": "direct_clone"}
                    return True

        # This is a fork — add upstream
        upstream_url = "https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git"
        self._log("→", "Adding upstream remote to SynchronizedLivingArchitecture/S.L.A.T.E")
        add_result = _run(["git", "remote", "add", "upstream", upstream_url],
                          cwd=str(self.workspace), timeout=10)

        if add_result.returncode == 0:
            # Fetch upstream to sync
            self._log("→", "Fetching upstream branches")
            _run(["git", "fetch", "upstream", "--no-tags"], cwd=str(self.workspace), timeout=60)

            # Set default branch to track upstream/main
            _run(["git", "branch", "--set-upstream-to=upstream/main", "main"],
                 cwd=str(self.workspace), timeout=10)

            self._log("✓", "Upstream configured — fork syncs with SynchronizedLivingArchitecture/S.L.A.T.E main")
            self.results["fork_upstream"] = {"status": "configured", "upstream": upstream_url}
        else:
            self._log("⚠", f"Could not add upstream: {add_result.stderr.strip()}")
            self.results["fork_upstream"] = {"status": "failed", "error": add_result.stderr.strip()}

        return True

    # ── Step 12c: GitHub Pages Setup ─────────────────────────────────────

    def step_github_pages(self) -> bool:
        """Enable GitHub Pages on the fork for the SLATE feature/docs site."""
        _print_header("Step 12c: GitHub Pages Setup")

        # Check if gh CLI is available
        if not _cmd_exists("gh"):
            self._log("⚠", "GitHub CLI (gh) not installed — Pages setup requires manual config")
            self._log("ℹ", "  Install from: https://cli.github.com")
            self._log("ℹ", "  Or manually enable Pages in your repo Settings → Pages")
            self._log("ℹ", "  Source: GitHub Actions, Workflow: pages.yml")
            self.results["github_pages"] = {"status": "skipped", "reason": "no_gh_cli"}
            return True

        # Check gh auth
        auth_result = _run(["gh", "auth", "status"], timeout=10)
        if auth_result.returncode != 0:
            self._log("⚠", "GitHub CLI not authenticated — run: gh auth login")
            self._log("ℹ", "  After auth, re-run install or manually enable Pages in Settings")
            self.results["github_pages"] = {"status": "skipped", "reason": "not_authenticated"}
            return True

        # Get the repo name from git remote
        remote_result = _run(["git", "remote", "get-url", "origin"], cwd=str(self.workspace), timeout=10)
        if remote_result.returncode != 0:
            self._log("⚠", "No origin remote — cannot configure Pages")
            self.results["github_pages"] = {"status": "skipped", "reason": "no_origin"}
            return True

        origin_url = remote_result.stdout.strip()
        # Extract owner/repo from URL
        repo_slug = None
        if "github.com" in origin_url:
            # Handle https://github.com/owner/repo.git or git@github.com:owner/repo.git
            if origin_url.startswith("https://"):
                parts = origin_url.replace("https://github.com/", "").replace(".git", "").strip("/")
                repo_slug = parts
            elif ":" in origin_url:
                parts = origin_url.split(":")[-1].replace(".git", "").strip("/")
                repo_slug = parts

        if not repo_slug:
            self._log("⚠", f"Could not parse repo from: {origin_url}")
            self.results["github_pages"] = {"status": "skipped", "reason": "unparseable_remote"}
            return True

        self._log("→", f"Configuring GitHub Pages for {repo_slug}")

        # Modified: 2026-02-07T22:00:00Z | Author: COPILOT | Change: Use /docs folder source for Pages (simpler for forks)
        # Enable Pages via GitHub API (gh api)
        # Use /docs folder on main branch as the source (contains Watchmaker dashboard, components, etc.)
        enable_result = _run([
            "gh", "api", f"repos/{repo_slug}/pages",
            "--method", "POST",
            "--raw-field", 'source={"branch":"main","path":"/docs"}',
            "--silent",
        ], timeout=30)

        if enable_result.returncode == 0:
            self._log("✓", f"GitHub Pages enabled for {repo_slug}")
            pages_url = f"https://{repo_slug.split('/')[0]}.github.io/{repo_slug.split('/')[1]}/"
            self._log("ℹ", f"  Pages URL: {pages_url}")
            self._log("ℹ", "  Deploy trigger: push to main (docs/pages/**)")
            self.results["github_pages"] = {"status": "enabled", "url": pages_url}
        elif "409" in (enable_result.stderr or "") or "already" in (enable_result.stderr or "").lower():
            self._log("✓", "GitHub Pages already enabled")
            pages_url = f"https://{repo_slug.split('/')[0]}.github.io/{repo_slug.split('/')[1]}/"
            self._log("ℹ", f"  Pages URL: {pages_url}")
            self.results["github_pages"] = {"status": "already_enabled", "url": pages_url}
        else:
            self._log("⚠", f"Pages setup returned: {enable_result.stderr.strip()}")
            self._log("ℹ", "  You can manually enable Pages: repo Settings → Pages → Source: GitHub Actions")
            self.results["github_pages"] = {"status": "manual_required", "error": enable_result.stderr.strip()}

        return True

    def step_validate(self) -> bool:
        """Run final validation of the SLATE ecosystem."""
        _print_header("Step 13: Final Validation")

        checks = []

        # 1. Python venv works
        result = _run([str(self._python()), "-c", "import sys; print(sys.version)"], timeout=10)
        checks.append(("Python venv", result.returncode == 0))

        # 2. SLATE SDK importable
        result = _run([str(self._python()), "-c", "import slate; print(slate.__version__)"],
                      timeout=15, cwd=str(self.workspace))
        checks.append(("SLATE SDK import", result.returncode == 0))

        # 3. slate_status.py runs
        status_py = self.workspace / "slate" / "slate_status.py"
        if status_py.exists():
            result = _run([str(self._python()), str(status_py), "--quick"],
                          timeout=30, cwd=str(self.workspace))
            checks.append(("slate_status.py", result.returncode == 0))
        else:
            checks.append(("slate_status.py", False))

        # 4. slate_runtime.py runs
        runtime_py = self.workspace / "slate" / "slate_runtime.py"
        if runtime_py.exists():
            result = _run([str(self._python()), str(runtime_py), "--check-all"],
                          timeout=30, cwd=str(self.workspace))
            checks.append(("slate_runtime.py", result.returncode == 0))
        else:
            checks.append(("slate_runtime.py", False))

        # 5. Dashboard importable
        result = _run([str(self._python()), "-c",
                       "from agents.slate_dashboard_server import app; print('ok')"],
                      timeout=15, cwd=str(self.workspace))
        checks.append(("Dashboard server", result.returncode == 0))

        # Report
        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        for name, ok in checks:
            self._log("✓" if ok else "✗", name)

        self._log("" if passed == total else "⚠",
                  f"Validation: {passed}/{total} checks passed")

        self.results["validation"] = {
            "passed": passed,
            "total": total,
            "checks": {name: ok for name, ok in checks},
        }
        return passed > 0

    # ── Orchestrators ────────────────────────────────────────────────────

    def run_install(
        self,
        target_dir: str = None,
        beta: bool = False,
        personalize: bool = True,
        fork_name: str = None,
        interactive: bool = True,
    ) -> dict:
        """
        Run full SLATE ecosystem installation.

        Args:
            target_dir: Target directory for installation
            beta: Use BETA fork as source
            personalize: Run personalization setup (custom name, logo, theme)
            fork_name: Pre-set fork name for non-interactive personalization
            interactive: Allow interactive prompts
        """
        # Modified: 2026-02-07T20:00:00Z | Author: COPILOT | Change: Add fork upstream + Pages setup steps
        print()
        print("═" * 64)
        print("  S.L.A.T.E. Ecosystem Installer v2.4.0")
        print("  Synchronized Living Architecture for Transformation & Evolution")
        print("═" * 64)

        start_time = time.time()
        steps = [
            ("Dependency Scan", self.step_dependency_scan),  # Scan first - SLATE ethos
            ("Git Repository", lambda: self.step_git_setup(target_dir, beta)),
            ("Virtual Environment", self.step_venv_setup),
            ("Core Dependencies", self.step_core_deps),
            ("PyTorch Setup", self.step_pytorch_setup),
            ("Ollama Setup", self.step_ollama_setup),
            ("Docker Setup", self.step_docker_setup),
            ("SLATE SDK", self.step_slate_sdk),
            ("VS Code Extension", self.step_vscode_extension),
            ("SLATE Models", self.step_slate_models),
            ("Personalization", lambda: self.step_personalization(interactive=interactive, fork_name=fork_name) if personalize else True),
            ("Workspace Config", self.step_workspace_setup),
            ("Fork Upstream", self.step_fork_upstream),
            ("GitHub Pages", self.step_github_pages),
            ("Generative UI", self.step_generative_ui),
            ("Guided Mode", self.step_guided_mode),
            ("Final Validation", self.step_validate),
        ]

        completed = 0
        failed = 0
        for name, fn in steps:
            try:
                ok = fn()
                if ok:
                    completed += 1
                else:
                    failed += 1
                    # Fatal steps
                    if name in ("Git Repository", "Virtual Environment"):
                        self._log("✗", f"Fatal: {name} failed — cannot continue")
                        break
            except KeyboardInterrupt:
                self._log("✗", "Installation cancelled by user")
                break
            except Exception as e:
                self._log("✗", f"{name} error: {e}")
                failed += 1

        elapsed = time.time() - start_time

        # Get personalization info for display
        fork_display = "S.L.A.T.E."
        if "personalization" in self.results:
            p_result = self.results["personalization"]
            if p_result.get("status") == "created" or p_result.get("status") == "exists":
                fork_display = p_result.get("name", "S.L.A.T.E.")

        print()
        print("═" * 64)
        if failed == 0:
            print(f"  ✓ {fork_display} Installation Complete! ({elapsed:.1f}s)")
        else:
            print(f"  ⚠ {fork_display} installed with {failed} warning(s) ({elapsed:.1f}s)")
        print("═" * 64)
        print()
        # Modified: 2026-02-09T07:10:00Z | Author: COPILOT | Change: Test install team - add AI hobbyist disclaimer and knowledge pointers
        if failed == 0:
            print("  ⚠ EXPERIMENTAL AI HOBBYIST PROJECT")
            print("    Entirely created and managed by AI via vibe coding.")
            print("    Production-ready aim - not for critical systems.")
            print()
        print("  Next steps:")
        if _is_windows():
            print("    1. Activate:  .\\.venv\\Scripts\\activate")
        else:
            print("    1. Activate:  source .venv/bin/activate")
        print("    2. Guided setup: python slate/guided_mode.py")
        print("    3. In VS Code: Open workspace and use @slate /status")
        print("    4. System check: python slate/slate_status.py --quick")
        print("    5. Start services: python slate/slate_orchestrator.py start")
        print()
        print("  Knowledge & Docs:")
        print("    • README:     https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E#readme")
        print("    • Wiki:       https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/wiki")
        print("    • Components: https://synchronizedlivingarchitecture.github.io/S.L.A.T.E/components.html")

        if "personalization" in self.results and self.results["personalization"].get("status") == "skipped":
            print()
            print("  Personalization:")
            print("    To customize your SLATE later, run:")
            print("    python slate/slate_personalization.py --setup")

        print()

        return {
            "success": failed == 0,
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": round(elapsed, 1),
            "errors": self.errors,
            "results": self.results,
            "fork_name": fork_display,
        }

    def run_update(self) -> dict:
        """Update SLATE from git and re-validate ecosystem."""
        # Modified: 2026-02-06T22:30:00Z | Author: COPILOT | Change: Update orchestration
        print()
        print("═" * 64)
        print("  S.L.A.T.E. Update")
        print("═" * 64)

        start_time = time.time()
        steps_done = []

        # Step 1: Git pull
        _print_header("Git Pull")
        if (self.workspace / ".git").exists():
            # Stash any local changes
            _run(["git", "stash"], cwd=str(self.workspace), timeout=15)
            self._log("→", "Pulling latest from origin...")

            result = _run(["git", "pull", "--rebase", "origin", "main"],
                          cwd=str(self.workspace), timeout=120)
            if result.returncode == 0:
                self._log("✓", f"Updated: {result.stdout.strip()}")
                steps_done.append("git_pull")
            else:
                # Try without rebase
                result = _run(["git", "pull", "origin", "main"],
                              cwd=str(self.workspace), timeout=120)
                if result.returncode == 0:
                    self._log("✓", f"Updated (merge): {result.stdout.strip()}")
                    steps_done.append("git_pull")
                else:
                    self._log("⚠", f"Git pull failed: {result.stderr.strip()[:200]}")
                    self.errors.append("git pull failed")

            # Pop stash
            _run(["git", "stash", "pop"], cwd=str(self.workspace), timeout=15)
        else:
            self._log("⚠", "Not a git repository — skipping pull")

        # Step 2: Update dependencies
        _print_header("Dependencies Update")
        pip = self._pip()
        if pip.exists():
            req_file = self.workspace / "requirements.txt"
            if req_file.exists():
                self._log("→", "Updating pip dependencies...")
                result = _run(
                    [str(pip), "install", "-r", str(req_file), "--upgrade", "--quiet"],
                    timeout=600, cwd=str(self.workspace),
                )
                if result.returncode == 0:
                    self._log("✓", "Dependencies updated")
                    steps_done.append("deps_update")
                else:
                    self._log("⚠", "Dependency update had errors")

        # Step 3: Re-install SLATE SDK
        _print_header("SLATE SDK Update")
        if (self.workspace / "pyproject.toml").exists():
            result = _run(
                [str(pip), "install", "-e", str(self.workspace), "--quiet"],
                timeout=120, cwd=str(self.workspace),
            )
            if result.returncode == 0:
                self._log("✓", "SLATE SDK updated")
                steps_done.append("sdk_update")

        # Step 4: Update VS Code extension
        _print_header("VS Code Extension Update")
        ext_dir = self.workspace / "plugins" / "slate-copilot"
        if ext_dir.exists() and _cmd_exists("npm"):
            _run(["npm", "install"], timeout=120, cwd=str(ext_dir))
            compile_result = _run(["npm", "run", "compile"], timeout=60, cwd=str(ext_dir))
            if compile_result.returncode == 0:
                self._log("✓", "Extension compiled")

                # Re-package and install if vsce available
                if _cmd_exists("vsce"):
                    _run(["vsce", "package", "--no-dependencies"], timeout=60, cwd=str(ext_dir))
                    vsix_files = list(ext_dir.glob("*.vsix"))
                    if vsix_files:
                        _run(["code", "--install-extension", str(vsix_files[-1])], timeout=60)
                        self._log("✓", "Extension updated in VS Code")
                        steps_done.append("extension_update")
            else:
                self._log("⚠", "Extension compile failed")

        # Step 5: Check PyTorch
        _print_header("PyTorch Check")
        pytorch = detect_pytorch()
        if pytorch["installed"]:
            cuda_str = f"CUDA {pytorch['cuda_version']}" if pytorch["cuda_available"] else "CPU"
            self._log("✓", f"PyTorch {pytorch['version']} ({cuda_str})")
        else:
            self._log("ℹ", "PyTorch not installed — run: @slate /install for full setup")

        # Step 6: Check Ollama
        _print_header("Ollama Check")
        ollama = detect_ollama()
        if ollama["installed"]:
            self._log("✓", f"Ollama {ollama['version']}")
        else:
            self._log("ℹ", "Ollama not installed")

        # Step 7: Final validation
        self.step_validate()

        elapsed = time.time() - start_time
        print()
        print("═" * 64)
        print(f"  ✓ S.L.A.T.E. Update Complete ({elapsed:.1f}s)")
        print(f"  Steps completed: {', '.join(steps_done) if steps_done else 'none'}")
        print("═" * 64)
        print()

        return {
            "success": len(self.errors) == 0,
            "steps": steps_done,
            "elapsed_seconds": round(elapsed, 1),
            "errors": self.errors,
        }

    def run_check(self) -> dict:
        """Run dependency check only (no install)."""
        # Modified: 2026-02-06T22:30:00Z | Author: COPILOT | Change: Dependency check mode
        print()
        print("═" * 64)
        print("  S.L.A.T.E. Ecosystem Check")
        print("═" * 64)

        deps = detect_all()

        def _status(ok: bool) -> str:
            return "✓" if ok else "✗"

        print()
        py = deps["python"]
        _print_step(_status(py["meets_requirement"]),
                    f"Python {py['version']} (need 3.11+) — {py['executable']}")

        git = deps["git"]
        _print_step(_status(git["installed"]),
                    f"Git {'v' + git['version'] if git['installed'] else 'not found'}")

        gpu = deps["gpu"]
        if gpu["installed"]:
            for g in gpu["gpus"]:
                _print_step("✓", f"GPU: {g['name']} (CC {g['compute_cap']}, {g['memory']})")
        else:
            _print_step("ℹ", "No NVIDIA GPU detected (CPU mode)")

        pt = deps["pytorch"]
        if pt["installed"]:
            cuda_str = f"CUDA {pt['cuda_version']}" if pt["cuda_available"] else "CPU only"
            _print_step("✓", f"PyTorch {pt['version']} ({cuda_str})")
        else:
            _print_step("✗", "PyTorch not installed")

        ol = deps["ollama"]
        if ol["installed"]:
            run_str = "running" if ol["running"] else "not running"
            _print_step("✓", f"Ollama {ol['version']} ({run_str})")
        else:
            _print_step("✗", "Ollama not installed")

        dk = deps["docker"]
        if dk["installed"]:
            run_str = "running" if dk["running"] else "not running"
            _print_step("✓", f"Docker {dk['version']} ({run_str})")
        else:
            _print_step("ℹ", "Docker not installed (optional)")

        vs = deps["vscode"]
        if vs["installed"]:
            ext_str = "extension installed" if vs["slate_extension"] else "no SLATE extension"
            _print_step("✓" if vs["slate_extension"] else "⚠",
                        f"VS Code {vs['version']} ({ext_str})")
        else:
            _print_step("ℹ", "VS Code not detected on PATH")

        print()

        # Summary
        required_ok = py["meets_requirement"] and git["installed"]
        optional_count = sum([
            pt["installed"], ol["installed"], dk["installed"],
            vs.get("slate_extension", False),
        ])
        _print_step("✓" if required_ok else "✗",
                    f"Required: {'OK' if required_ok else 'MISSING'}")
        _print_step("ℹ", f"Optional: {optional_count}/4 (PyTorch, Ollama, Docker, Extension)")
        print()

        return deps


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point for SLATE installer."""
    # Modified: 2026-02-07T12:00:00Z | Author: COPILOT | Change: Add personalization CLI options
    parser = argparse.ArgumentParser(
        description="S.L.A.T.E. Ecosystem Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slate/slate_installer.py --install                    # Full install with personalization
  python slate/slate_installer.py --install --target ~/slate   # Install to specific directory
  python slate/slate_installer.py --install --fork-name PHOENIX  # Install with preset fork name
  python slate/slate_installer.py --install --no-personalize   # Skip personalization
  python slate/slate_installer.py --update                     # Update existing installation
  python slate/slate_installer.py --check                      # Check dependencies only
  python slate/slate_installer.py --install --beta             # Install from BETA fork
        """,
    )
    parser.add_argument("--install", action="store_true", help="Full ecosystem install")
    parser.add_argument("--update", action="store_true", help="Update existing installation")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    parser.add_argument("--target", type=str, help="Target directory for fresh install")
    parser.add_argument("--beta", action="store_true", help="Use BETA fork")
    parser.add_argument("--json", action="store_true", help="JSON output")

    # Personalization options
    parser.add_argument("--fork-name", type=str, dest="fork_name",
                        help="Custom name for your SLATE fork (e.g., PHOENIX, NOVA)")
    parser.add_argument("--no-personalize", action="store_true", dest="no_personalize",
                        help="Skip personalization setup")
    parser.add_argument("--non-interactive", action="store_true", dest="non_interactive",
                        help="Run without interactive prompts")

    args = parser.parse_args()

    workspace = Path(args.target) if args.target else Path(__file__).parent.parent
    installer = SlateInstaller(workspace=workspace)

    if args.check:
        result = installer.run_check()
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        return 0

    if args.update:
        result = installer.run_update()
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        return 0 if result["success"] else 1

    if args.install:
        result = installer.run_install(
            target_dir=args.target,
            beta=args.beta,
            personalize=not args.no_personalize,
            fork_name=args.fork_name,
            interactive=not args.non_interactive,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        return 0 if result["success"] else 1

    # Default: show check
    installer.run_check()
    print("  Use --install for full setup or --update to pull latest")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
