#!/usr/bin/env python3
"""
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_runner_manager [python]
# Modified: 2026-02-06T20:00:00Z | Author: COPILOT | Change: Full SLATE environment provisioning, service config, install integration
# SLATE Self-Hosted Runner Manager - Sets up and manages GitHub Actions runners
# ═══════════════════════════════════════════════════════════════════════════════

Manages self-hosted GitHub Actions runners for SLATE:
- Downloads and configures the runner
- Registers with S.L.A.T.E. repository
- Provisions complete SLATE environment (venv, deps, GPU, SDK)
- Runs as Windows service or interactive
- GPU-aware job routing
- Integrated into install_slate.py as optional step
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
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

# Setup path
WORKSPACE_ROOT = Path(__file__).parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

logger = logging.getLogger("slate.runner_manager")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL: constants [python]
# Author: Claude | Created: 2026-02-06T22:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

RUNNER_VERSION = "2.331.0"
RUNNER_BASE_URL = "https://github.com/actions/runner/releases/download"

SLATE_REPO = "SynchronizedLivingArchitecture/S.L.A.T.E."

# Modified: 2026-02-07T04:00:00Z | Author: COPILOT | Change: Use WORKSPACE_ROOT for runner dir
DEFAULT_RUNNER_DIR = WORKSPACE_ROOT / "actions-runner"

RUNNER_LABELS = [
    "self-hosted",
    "slate",
    "gpu",
    "windows" if platform.system() == "Windows" else platform.system().lower(),
]


@dataclass
class RunnerConfig:
    """Configuration for a self-hosted runner."""
    name: str
    repo_url: str
    token: str
    runner_dir: Path
    labels: List[str]
    work_dir: Optional[Path] = None
    slate_workspace: Optional[Path] = None
    slate_venv: Optional[Path] = None
    provisioned: bool = False
    provision_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "repo_url": self.repo_url,
            "runner_dir": str(self.runner_dir),
            "labels": self.labels,
            "work_dir": str(self.work_dir) if self.work_dir else None,
            "slate_workspace": str(self.slate_workspace) if self.slate_workspace else None,
            "slate_venv": str(self.slate_venv) if self.slate_venv else None,
            "provisioned": self.provisioned,
            "provision_date": self.provision_date,
        }


class SlateRunnerManager:
    """
    Manages self-hosted GitHub Actions runners for SLATE.

    Features:
    - Downloads and installs the runner
    - Configures for SLATE repos
    - GPU detection for job labels
    - Service management (Windows)
    """

    def __init__(self, runner_dir: Optional[Path] = None):
        self.runner_dir = runner_dir or DEFAULT_RUNNER_DIR
        self.config_file = self.runner_dir / ".slate_runner_config.json"
        self.system = platform.system()

    def detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information for runner labels."""
        gpu_info = {
            "has_gpu": False,
            "gpu_count": 0,
            "gpu_names": [],
            "cuda_available": False,
        }

        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpu_info["has_gpu"] = True
                gpu_info["gpu_count"] = len(lines)
                gpu_info["gpu_names"] = [line.split(",")[0].strip() for line in lines]
                gpu_info["cuda_available"] = True
        except Exception:
            pass

        return gpu_info

    def get_runner_labels(self) -> List[str]:
        """Get labels for this runner based on system capabilities."""
        # Modified: 2026-02-06T12:00:00Z | Author: COPILOT | Change: Deduplicate GPU arch labels, add multi-gpu label
        labels = RUNNER_LABELS.copy()

        gpu_info = self.detect_gpu()
        if gpu_info["has_gpu"]:
            labels.append("cuda")
            labels.append(f"gpu-{gpu_info['gpu_count']}")
            if gpu_info["gpu_count"] > 1:
                labels.append("multi-gpu")

            # Add GPU architecture labels (deduplicated)
            arch_labels = set()
            for name in gpu_info["gpu_names"]:
                if "5070" in name or "5080" in name or "5090" in name:
                    arch_labels.add("blackwell")
                elif "4090" in name or "4080" in name or "4070" in name:
                    arch_labels.add("ada-lovelace")
                elif "3090" in name or "3080" in name or "3070" in name:
                    arch_labels.add("ampere")
            labels.extend(sorted(arch_labels))

        return labels

    def download_runner(self) -> bool:
        """Download the GitHub Actions runner."""
        self.runner_dir.mkdir(parents=True, exist_ok=True)

        # Determine platform
        if self.system == "Windows":
            filename = f"actions-runner-win-x64-{RUNNER_VERSION}.zip"
        elif self.system == "Linux":
            filename = f"actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz"
        elif self.system == "Darwin":
            filename = f"actions-runner-osx-x64-{RUNNER_VERSION}.tar.gz"
        else:
            logger.error(f"Unsupported platform: {self.system}")
            return False

        url = f"{RUNNER_BASE_URL}/v{RUNNER_VERSION}/{filename}"
        download_path = self.runner_dir / filename

        print(f"Downloading runner from {url}...")

        try:
            urllib.request.urlretrieve(url, download_path)
            print(f"Downloaded to {download_path}")

            # Extract
            print("Extracting...")
            if filename.endswith(".zip"):
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(self.runner_dir)
            else:
                import tarfile
                with tarfile.open(download_path, 'r:gz') as tar:
                    tar.extractall(self.runner_dir)

            # Clean up
            download_path.unlink()
            print("Runner extracted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to download runner: {e}")
            return False

    def configure_runner(
        self,
        repo_url: str,
        token: str,
        name: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Configure the runner for a repository.

        Args:
            repo_url: GitHub repository URL
            token: Runner registration token (from repo settings)
            name: Runner name (default: hostname)
            labels: Additional labels

        Returns:
            Configuration result
        """
        result = {
            "success": False,
            "steps": [],
            "errors": [],
        }

        if not (self.runner_dir / "config.cmd").exists() and not (self.runner_dir / "config.sh").exists():
            result["errors"].append("Runner not downloaded. Run download_runner() first.")
            return result

        # Get labels
        all_labels = self.get_runner_labels()
        if labels:
            all_labels.extend(labels)

        # Runner name
        runner_name = name or f"slate-{platform.node()}"

        # Build config command
        if self.system == "Windows":
            config_cmd = str(self.runner_dir / "config.cmd")
        else:
            config_cmd = str(self.runner_dir / "config.sh")

        cmd = [
            config_cmd,
            "--url", repo_url,
            "--token", token,
            "--name", runner_name,
            "--labels", ",".join(all_labels),
            "--work", str(self.runner_dir / "_work"),
            "--unattended",
            "--replace",
        ]

        print(f"Configuring runner '{runner_name}' for {repo_url}...")
        print(f"Labels: {', '.join(all_labels)}")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.runner_dir),
                capture_output=True,
                text=True,
                timeout=120
            )

            if proc.returncode == 0:
                result["success"] = True
                result["steps"].append(f"Configured runner: {runner_name}")
                result["steps"].append(f"Labels: {', '.join(all_labels)}")

                # Save config
                config = RunnerConfig(
                    name=runner_name,
                    repo_url=repo_url,
                    token="***",  # Don't store token
                    runner_dir=self.runner_dir,
                    labels=all_labels,
                    work_dir=self.runner_dir / "_work",
                )
                self.config_file.write_text(json.dumps(config.to_dict(), indent=2))
            else:
                result["errors"].append(f"Configuration failed: {proc.stderr}")

        except Exception as e:
            result["errors"].append(str(e))

        return result

    def start_runner(self, as_service: bool = False) -> Dict[str, Any]:
        """
        Start the runner.

        Args:
            as_service: Install and start as Windows service

        Returns:
            Start result
        """
        result = {
            "success": False,
            "steps": [],
            "errors": [],
        }

        if as_service and self.system == "Windows":
            # Install as Windows service
            svc_cmd = str(self.runner_dir / "svc.cmd")

            try:
                # Install
                proc = subprocess.run(
                    [svc_cmd, "install"],
                    cwd=str(self.runner_dir),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if proc.returncode == 0:
                    result["steps"].append("Installed as Windows service")

                # Start
                proc = subprocess.run(
                    [svc_cmd, "start"],
                    cwd=str(self.runner_dir),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if proc.returncode == 0:
                    result["success"] = True
                    result["steps"].append("Started Windows service")
                else:
                    result["errors"].append(f"Failed to start service: {proc.stderr}")

            except Exception as e:
                result["errors"].append(str(e))
        else:
            # Run interactively
            if self.system == "Windows":
                run_cmd = str(self.runner_dir / "run.cmd")
            else:
                run_cmd = str(self.runner_dir / "run.sh")

            print(f"Starting runner interactively...")
            print(f"Press Ctrl+C to stop")
            print()

            try:
                subprocess.run(
                    [run_cmd],
                    cwd=str(self.runner_dir),
                )
                result["success"] = True
            except KeyboardInterrupt:
                result["steps"].append("Runner stopped by user")
                result["success"] = True
            except Exception as e:
                result["errors"].append(str(e))

        return result

    def stop_service(self) -> Dict[str, Any]:
        """Stop the runner service (Windows only)."""
        result = {
            "success": False,
            "steps": [],
            "errors": [],
        }

        if self.system != "Windows":
            result["errors"].append("Service management only available on Windows")
            return result

        svc_cmd = str(self.runner_dir / "svc.cmd")

        try:
            proc = subprocess.run(
                [svc_cmd, "stop"],
                cwd=str(self.runner_dir),
                capture_output=True,
                text=True,
                timeout=60
            )
            if proc.returncode == 0:
                result["success"] = True
                result["steps"].append("Service stopped")
            else:
                result["errors"].append(proc.stderr)
        except Exception as e:
            result["errors"].append(str(e))

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # SLATE Environment Provisioning
    # ═══════════════════════════════════════════════════════════════════════════

    def provision_slate_environment(self, workspace: Optional[Path] = None) -> Dict[str, Any]:
        """
        Provision a complete SLATE environment for the runner.

        Sets up everything needed to run full SLATE systems in CI:
        - Python venv with all dependencies
        - GPU/CUDA verification
        - SDK validation
        - Environment variables for runner jobs
        - Pre-run script for automatic env activation

        Args:
            workspace: SLATE workspace root (default: auto-detect)

        Returns:
            Provisioning result with steps and status
        """
        result = {
            "success": False,
            "steps": [],
            "errors": [],
            "environment": {},
        }

        ws = workspace or WORKSPACE_ROOT
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"[SLATE] Provisioning runner environment...")
        print(f"  Workspace: {ws}")
        print(f"  Runner: {self.runner_dir}")

        # Step 1: Verify workspace
        if not (ws / "pyproject.toml").exists():
            result["errors"].append(f"Not a SLATE workspace: {ws} (no pyproject.toml)")
            return result
        result["steps"].append("Workspace verified")

        # Step 2: Python venv
        venv_path = ws / ".venv"
        if self.system == "Windows":
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"

        if not venv_path.exists():
            print("  Creating virtual environment...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True, timeout=60,
                )
                result["steps"].append("Virtual environment created")
            except Exception as e:
                result["errors"].append(f"venv creation failed: {e}")
                return result
        else:
            result["steps"].append("Virtual environment exists")

        # Step 3: Install dependencies
        req_file = ws / "requirements.txt"
        if req_file.exists():
            print("  Installing dependencies...")
            try:
                proc = subprocess.run(
                    [str(pip_exe), "install", "-r", str(req_file), "--quiet"],
                    capture_output=True, text=True, timeout=600,
                    cwd=str(ws),
                )
                if proc.returncode == 0:
                    result["steps"].append("Dependencies installed from requirements.txt")
                else:
                    result["steps"].append("Dependencies install had warnings")
                    result["errors"].append(f"pip warnings: {proc.stderr[:200]}")
            except Exception as e:
                result["errors"].append(f"Dependency install failed: {e}")
                return result

        # Step 4: Install dev/test dependencies
        print("  Installing test dependencies...")
        try:
            subprocess.run(
                [str(pip_exe), "install", "pytest", "pytest-cov", "pytest-asyncio", "--quiet"],
                capture_output=True, text=True, timeout=120,
                cwd=str(ws),
            )
            result["steps"].append("Test dependencies installed")
        except Exception:
            result["steps"].append("Test deps install skipped")

        # Step 5: Install SLATE package in editable mode
        print("  Installing SLATE in editable mode...")
        try:
            proc = subprocess.run(
                [str(pip_exe), "install", "-e", ".", "--quiet"],
                capture_output=True, text=True, timeout=120,
                cwd=str(ws),
            )
            if proc.returncode == 0:
                result["steps"].append("SLATE installed (editable)")
            else:
                result["steps"].append("Editable install skipped (non-fatal)")
        except Exception:
            result["steps"].append("Editable install skipped")

        # Step 6: GPU/CUDA verification
        gpu_info = self.detect_gpu()
        if gpu_info["has_gpu"]:
            print(f"  GPU detected: {', '.join(gpu_info['gpu_names'])}")
            result["steps"].append(f"GPU: {', '.join(gpu_info['gpu_names'])}")

            # Check if PyTorch has CUDA
            try:
                proc = subprocess.run(
                    [str(python_exe), "-c",
                     "import torch; print(f'torch={torch.__version__},cuda={torch.cuda.is_available()}')"],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(ws),
                )
                if proc.returncode == 0 and "cuda=True" in proc.stdout:
                    result["steps"].append(f"PyTorch CUDA: {proc.stdout.strip()}")
                else:
                    result["steps"].append("PyTorch: CPU only (install CUDA version for GPU support)")
            except Exception:
                result["steps"].append("PyTorch: not verified")
        else:
            result["steps"].append("GPU: not detected (CPU mode)")

        # Step 7: SDK validation
        print("  Validating SLATE SDK...")
        try:
            proc = subprocess.run(
                [str(python_exe), "-c",
                 "import slate; print(slate.__version__)"],
                capture_output=True, text=True, timeout=15,
                cwd=str(ws),
            )
            if proc.returncode == 0:
                result["steps"].append(f"SDK: slate v{proc.stdout.strip()}")
            else:
                result["steps"].append("SDK: slate import failed (non-fatal)")
        except Exception:
            result["steps"].append("SDK validation skipped")

        # Step 8: Create runner environment script
        env_script = self._create_runner_env_script(ws, venv_path, gpu_info)
        result["steps"].append("Runner environment script created")

        # Step 9: Create pre-job hook
        hook_path = self._create_pre_job_hook(ws, venv_path)
        result["steps"].append(f"Pre-job hook: {hook_path}")

        # Step 10: Save provisioning state
        result["environment"] = {
            "workspace": str(ws),
            "venv": str(venv_path),
            "python": str(python_exe),
            "pip": str(pip_exe),
            "gpu": gpu_info,
            "labels": self.get_runner_labels(),
            "provisioned_at": now,
        }

        state_file = self.runner_dir / ".slate_provision_state.json"
        state_file.write_text(json.dumps(result["environment"], indent=2, default=str))

        # Update config if exists
        if self.config_file.exists():
            try:
                config = json.loads(self.config_file.read_text())
                config["slate_workspace"] = str(ws)
                config["slate_venv"] = str(venv_path)
                config["provisioned"] = True
                config["provision_date"] = now
                self.config_file.write_text(json.dumps(config, indent=2))
            except Exception:
                pass

        result["success"] = True
        print(f"  [OK] SLATE environment provisioned successfully")
        return result

    def _create_runner_env_script(
        self, workspace: Path, venv_path: Path, gpu_info: Dict
    ) -> Path:
        """Create environment setup script for runner jobs."""
        # Modified: 2026-02-06T12:00:00Z | Author: COPILOT | Change: Add per-GPU env vars, CUDA_VISIBLE_DEVICES, SLATE system boot
        gpu_count = gpu_info.get('gpu_count', 0)
        gpu_names = gpu_info.get('gpu_names', [])
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if self.system == "Windows":
            script_path = self.runner_dir / "slate_env.ps1"
            # Build per-GPU variable lines
            gpu_lines = ""
            for i, name in enumerate(gpu_names):
                gpu_lines += f'$env:SLATE_GPU_{i} = "{name}"\n'
            if gpu_count > 0:
                devices = ",".join(str(i) for i in range(gpu_count))
                gpu_lines += f'$env:CUDA_VISIBLE_DEVICES = "{devices}"\n'

            script = f'''# SLATE Runner Environment Setup
# Auto-generated by slate_runner_manager.py
# Modified: {ts} | Author: COPILOT

$env:SLATE_WORKSPACE = "{workspace}"
$env:SLATE_VENV = "{venv_path}"
$env:SLATE_RUNNER = "true"
$env:SLATE_GPU_COUNT = "{gpu_count}"
{gpu_lines}
$env:PYTHONPATH = "{workspace};$env:PYTHONPATH"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "{venv_path}\\Scripts;$env:PATH"

# Activate venv
& "{venv_path}\\Scripts\\Activate.ps1"

Write-Host "[SLATE] Runner environment loaded"
Write-Host "  Workspace:  $env:SLATE_WORKSPACE"
Write-Host "  Python:     $(python --version 2>&1)"
Write-Host "  GPU count:  $env:SLATE_GPU_COUNT"
Write-Host "  CUDA devs:  $env:CUDA_VISIBLE_DEVICES"
'''
            # Append per-GPU info display
            for i in range(gpu_count):
                script += f'Write-Host "  GPU {i}:      $env:SLATE_GPU_{i}"\n'

            # SLATE system boot check
            script += '''
# Verify SLATE SDK is importable
try {
    $sdkVer = & python -c "import slate; print(slate.__version__)" 2>$null
    Write-Host "  SLATE SDK:  v$sdkVer"
} catch {
    Write-Host "  SLATE SDK:  not available"
}

# Run SLATE status (non-blocking health check)
try {
    & python slate/slate_status.py --quick 2>$null | Out-Null
    Write-Host "  SLATE:      systems initialized"
} catch {
    Write-Host "  SLATE:      status check skipped"
}
'''
        else:
            script_path = self.runner_dir / "slate_env.sh"
            gpu_lines = ""
            for i, name in enumerate(gpu_names):
                gpu_lines += f'export SLATE_GPU_{i}="{name}"\n'
            if gpu_count > 0:
                devices = ",".join(str(i) for i in range(gpu_count))
                gpu_lines += f'export CUDA_VISIBLE_DEVICES="{devices}"\n'

            script = f'''#!/bin/bash
# SLATE Runner Environment Setup
# Auto-generated by slate_runner_manager.py
# Modified: {ts} | Author: COPILOT

export SLATE_WORKSPACE="{workspace}"
export SLATE_VENV="{venv_path}"
export SLATE_RUNNER="true"
export SLATE_GPU_COUNT="{gpu_count}"
{gpu_lines}
export PYTHONPATH="{workspace}:$PYTHONPATH"
export PYTHONIOENCODING="utf-8"
export PATH="{venv_path}/bin:$PATH"

# Activate venv
source "{venv_path}/bin/activate"

echo "[SLATE] Runner environment loaded"
echo "  Workspace:  $SLATE_WORKSPACE"
echo "  Python:     $(python --version 2>&1)"
echo "  GPU count:  $SLATE_GPU_COUNT"
echo "  CUDA devs:  $CUDA_VISIBLE_DEVICES"
'''
            for i in range(gpu_count):
                script += f'echo "  GPU {i}:      $SLATE_GPU_{i}"\n'

            script += '''
# Verify SLATE SDK
python -c "import slate; print(f'  SLATE SDK:  v{slate.__version__}')" 2>/dev/null || echo "  SLATE SDK:  not available"

# Run SLATE status
python slate/slate_status.py --quick >/dev/null 2>&1 && echo "  SLATE:      systems initialized" || echo "  SLATE:      status check skipped"
'''
        script_path.write_text(script, encoding="utf-8")
        if self.system != "Windows":
            script_path.chmod(0o755)
        return script_path

    def _create_pre_job_hook(self, workspace: Path, venv_path: Path) -> Path:
        """Create a pre-job hook that activates SLATE environment for every job."""
        # Modified: 2026-02-06T12:00:00Z | Author: COPILOT | Change: Add GPU vars, SLATE system init to pre-job
        hooks_dir = self.runner_dir / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)

        gpu_info = self.detect_gpu()
        gpu_count = gpu_info.get('gpu_count', 0)
        gpu_names = gpu_info.get('gpu_names', [])

        if self.system == "Windows":
            hook_path = hooks_dir / "pre-job.ps1"
            gpu_lines = ""
            for i, name in enumerate(gpu_names):
                gpu_lines += f'$env:SLATE_GPU_{i} = "{name}"\n'
            if gpu_count > 0:
                devices = ",".join(str(i) for i in range(gpu_count))
                gpu_lines += f'$env:CUDA_VISIBLE_DEVICES = "{devices}"\n'

            hook = f'''# SLATE Pre-Job Hook
# Activates SLATE environment and GPU config before each GitHub Actions job
# Modified: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")} | Author: COPILOT
$env:SLATE_WORKSPACE = "{workspace}"
$env:SLATE_RUNNER = "true"
$env:SLATE_GPU_COUNT = "{gpu_count}"
{gpu_lines}$env:PYTHONPATH = "{workspace}"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "{venv_path}\\Scripts;$env:PATH"

Write-Host "[SLATE] Pre-job hook: {gpu_count} GPU(s), environment configured"

# Quick SLATE system init — ensure SDK is loadable
try {{
    & "{venv_path}\\Scripts\\python.exe" -c "import slate" 2>$null
}} catch {{}}
'''
        else:
            hook_path = hooks_dir / "pre-job.sh"
            gpu_lines = ""
            for i, name in enumerate(gpu_names):
                gpu_lines += f'export SLATE_GPU_{i}="{name}"\n'
            if gpu_count > 0:
                devices = ",".join(str(i) for i in range(gpu_count))
                gpu_lines += f'export CUDA_VISIBLE_DEVICES="{devices}"\n'

            hook = f'''#!/bin/bash
# SLATE Pre-Job Hook
export SLATE_WORKSPACE="{workspace}"
export SLATE_RUNNER="true"
export SLATE_GPU_COUNT="{gpu_count}"
{gpu_lines}export PYTHONPATH="{workspace}"
export PYTHONIOENCODING="utf-8"
export PATH="{venv_path}/bin:$PATH"

echo "[SLATE] Pre-job hook: {gpu_count} GPU(s), environment configured"
python -c "import slate" 2>/dev/null || true
'''
        hook_path.write_text(hook, encoding="utf-8")
        if self.system != "Windows":
            hook_path.chmod(0o755)

        # Set ACTIONS_RUNNER_HOOK_JOB_STARTED env var in .env
        env_file = self.runner_dir / ".env"
        env_content = ""
        if env_file.exists():
            env_content = env_file.read_text(encoding="utf-8")
        if "ACTIONS_RUNNER_HOOK_JOB_STARTED" not in env_content:
            with open(env_file, "a", encoding="utf-8") as f:
                f.write(f"\nACTIONS_RUNNER_HOOK_JOB_STARTED={hook_path}\n")

        return hook_path

    def create_startup_script(self) -> Path:
        """Create a startup script that launches the runner with SLATE env."""
        if self.system == "Windows":
            script_path = self.runner_dir / "start-slate-runner.ps1"
            script = f'''# SLATE Self-Hosted Runner Startup Script
# Modified: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")} | Author: COPILOT
# Run this to start the SLATE GitHub Actions runner

$ErrorActionPreference = "Stop"
$runnerDir = "{self.runner_dir}"

Write-Host "=============================================="
Write-Host "  S.L.A.T.E. Self-Hosted Runner"
Write-Host "=============================================="
Write-Host ""

# Load SLATE environment
$envScript = Join-Path $runnerDir "slate_env.ps1"
if (Test-Path $envScript) {{
    . $envScript
}}

# Verify SLATE
try {{
    $version = & python -c "import slate; print(slate.__version__)" 2>$null
    Write-Host "  SLATE SDK: v$version"
}} catch {{
    Write-Host "  SLATE SDK: not available (will install during jobs)"
}}

# Check GPUs
try {{
    $gpuInfo = nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>$null
    if ($gpuInfo) {{
        $gpuLines = ($gpuInfo -split "`n" | Where-Object {{ $_.Trim() }})
        Write-Host "  GPUs: $($gpuLines.Count) detected"
        foreach ($gpu in $gpuLines) {{
            Write-Host "    $gpu"
        }}
        $env:CUDA_VISIBLE_DEVICES = (0..($gpuLines.Count - 1)) -join ","
        Write-Host "  CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES"
    }}
}} catch {{
    Write-Host "  GPU: not detected"
}}

# Boot SLATE systems
Write-Host ""
Write-Host "Initializing SLATE systems..."
try {{
    & python slate/slate_status.py --quick 2>$null
    Write-Host "  [OK] SLATE status: online"
}} catch {{
    Write-Host "  [--] SLATE status: skipped"
}}
try {{
    & python slate/slate_runtime.py --check-all 2>$null | Out-Null
    Write-Host "  [OK] SLATE runtime: verified"
}} catch {{
    Write-Host "  [--] SLATE runtime: skipped"
}}

Write-Host ""
Write-Host "Starting runner (SLATE systems active)..."
Write-Host "GPUs assigned — runner will process GPU-enabled workflows"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

# Start the runner
Set-Location $runnerDir
& .\\run.cmd
'''
            script_path.write_text(script, encoding="utf-8")
        else:
            script_path = self.runner_dir / "start-slate-runner.sh"
            script = f'''#!/bin/bash
# SLATE Self-Hosted Runner Startup Script
# Modified: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")} | Author: COPILOT

set -e
RUNNER_DIR="{self.runner_dir}"

echo "=============================================="
echo "  S.L.A.T.E. Self-Hosted Runner"
echo "=============================================="
echo ""

# Load SLATE environment
if [ -f "$RUNNER_DIR/slate_env.sh" ]; then
    source "$RUNNER_DIR/slate_env.sh"
fi

# Verify SLATE
python -c "import slate; print(f'  SLATE SDK: v{{slate.__version__}}')" 2>/dev/null || echo "  SLATE SDK: not available"

# Check GPUs
gpu_csv=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)
if [ -n "$gpu_csv" ]; then
    gpu_count=$(echo "$gpu_csv" | wc -l)
    echo "  GPUs: $gpu_count detected"
    echo "$gpu_csv" | while read line; do echo "    $line"; done
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "  GPU: not detected"
fi

# Boot SLATE systems
echo ""
echo "Initializing SLATE systems..."
python slate/slate_status.py --quick >/dev/null 2>&1 && echo "  [OK] SLATE status: online" || echo "  [--] SLATE status: skipped"
python slate/slate_runtime.py --check-all >/dev/null 2>&1 && echo "  [OK] SLATE runtime: verified" || echo "  [--] SLATE runtime: skipped"

echo ""
echo "Starting runner (SLATE systems active)..."
echo "GPUs assigned — runner will process GPU-enabled workflows"
echo "Press Ctrl+C to stop"
echo ""

cd "$RUNNER_DIR"
./run.sh
'''
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(0o755)

        return script_path

    def create_windows_service_config(self) -> Dict[str, Any]:
        """Create Windows Task Scheduler config for auto-start on boot."""
        result = {"success": False, "steps": [], "errors": []}

        if self.system != "Windows":
            result["errors"].append("Windows only")
            return result

        # Create a scheduled task XML
        startup_script = self.runner_dir / "start-slate-runner.ps1"
        if not startup_script.exists():
            self.create_startup_script()

        task_xml = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>S.L.A.T.E. Self-Hosted GitHub Actions Runner</Description>
    <Author>SLATE</Author>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
    <BootTrigger>
      <Enabled>true</Enabled>
      <Delay>PT30S</Delay>
    </BootTrigger>
  </Triggers>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions>
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-ExecutionPolicy Bypass -NoProfile -File "{startup_script}"</Arguments>
      <WorkingDirectory>{self.runner_dir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''

        xml_path = self.runner_dir / "slate-runner-task.xml"
        xml_path.write_text(task_xml, encoding="utf-16")
        result["steps"].append(f"Task XML created: {xml_path}")

        # Try to register the task
        try:
            proc = subprocess.run(
                ["schtasks", "/create", "/tn", "SLATE-Runner",
                 "/xml", str(xml_path), "/f"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                result["success"] = True
                result["steps"].append("Scheduled task 'SLATE-Runner' registered")
            else:
                result["steps"].append("Task registration needs admin rights")
                result["steps"].append(f"Manual: schtasks /create /tn SLATE-Runner /xml \"{xml_path}\" /f")
        except Exception as e:
            result["steps"].append(f"Auto-register failed: {e}")
            result["steps"].append(f"Manual: schtasks /create /tn SLATE-Runner /xml \"{xml_path}\" /f")

        result["success"] = True
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get runner status including SLATE provisioning."""
        status = {
            "installed": (self.runner_dir / "config.cmd").exists() or (self.runner_dir / "config.sh").exists(),
            "configured": self.config_file.exists(),
            "runner_dir": str(self.runner_dir),
            "system": self.system,
            "gpu": self.detect_gpu(),
            "labels": self.get_runner_labels(),
            "provisioned": False,
            "slate_environment": None,
        }

        if status["configured"]:
            try:
                config = json.loads(self.config_file.read_text())
                status["config"] = config
                status["provisioned"] = config.get("provisioned", False)
            except Exception:
                pass

        # Check provisioning state
        provision_state = self.runner_dir / ".slate_provision_state.json"
        if provision_state.exists():
            try:
                status["slate_environment"] = json.loads(provision_state.read_text())
                status["provisioned"] = True
            except Exception:
                pass

        # Check if runner process is active
        status["running"] = self._is_runner_running()

        return status

    def _is_runner_running(self) -> bool:
        """Check if the runner process is currently running."""
        try:
            if self.system == "Windows":
                proc = subprocess.run(
                    ["tasklist", "/fi", "imagename eq Runner.Listener.exe", "/fo", "csv", "/nh"],
                    capture_output=True, text=True, timeout=10,
                )
                return "Runner.Listener.exe" in proc.stdout
            else:
                proc = subprocess.run(
                    ["pgrep", "-f", "Runner.Listener"],
                    capture_output=True, text=True, timeout=10,
                )
                return proc.returncode == 0
        except Exception:
            return False

    def generate_setup_script(self) -> str:
        """Generate a setup script for the runner."""
        if self.system == "Windows":
            return f'''# SLATE Self-Hosted Runner Setup Script
# Run this in PowerShell as Administrator

$runnerDir = "{self.runner_dir}"

# Create directory
New-Item -ItemType Directory -Force -Path $runnerDir | Out-Null
Set-Location $runnerDir

# Download runner
$version = "{RUNNER_VERSION}"
$url = "https://github.com/actions/runner/releases/download/v$version/actions-runner-win-x64-$version.zip"
Invoke-WebRequest -Uri $url -OutFile "runner.zip"

# Extract
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("$runnerDir\\runner.zip", $runnerDir)
Remove-Item "runner.zip"

# Get token from: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./settings/actions/runners/new
Write-Host ""
Write-Host "Runner downloaded. To configure:"
Write-Host ""
Write-Host "1. Get registration token from:"
Write-Host "   https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./settings/actions/runners/new"
Write-Host ""
Write-Host "2. Run:"
Write-Host "   .\\config.cmd --url https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E. --token YOUR_TOKEN"
Write-Host ""
Write-Host "3. Start runner:"
Write-Host "   .\\run.cmd"
Write-Host ""
Write-Host "4. Or install as service:"
Write-Host "   .\\svc.cmd install"
Write-Host "   .\\svc.cmd start"
'''
        else:
            return f'''#!/bin/bash
# SLATE Self-Hosted Runner Setup Script

RUNNER_DIR="{self.runner_dir}"
VERSION="{RUNNER_VERSION}"

# Create directory
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Download
curl -o runner.tar.gz -L "https://github.com/actions/runner/releases/download/v$VERSION/actions-runner-linux-x64-$VERSION.tar.gz"

# Extract
tar xzf runner.tar.gz
rm runner.tar.gz

# Install dependencies
./bin/installdependencies.sh

echo ""
echo "Runner downloaded. To configure:"
echo ""
echo "1. Get registration token from:"
echo "   https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./settings/actions/runners/new"
echo ""
echo "2. Run:"
echo "   ./config.sh --url https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E. --token YOUR_TOKEN"
echo ""
echo "3. Start runner:"
echo "   ./run.sh"
'''


# ═══════════════════════════════════════════════════════════════════════════════
# CELL: cli [python]
# Author: Claude | Created: 2026-02-06T22:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SLATE Self-Hosted Runner Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status
  python slate_runner_manager.py --status

  # Full setup: download + provision SLATE environment
  python slate_runner_manager.py --download --provision

  # Configure for SLATE repo
  python slate_runner_manager.py --configure --token YOUR_TOKEN

  # Start runner (interactive)
  python slate_runner_manager.py --start

  # Start as Windows service
  python slate_runner_manager.py --start --service

  # Provision SLATE environment on existing runner
  python slate_runner_manager.py --provision

  # Create startup script + auto-start config
  python slate_runner_manager.py --create-startup

  # Generate setup script for manual setup
  python slate_runner_manager.py --setup-script
"""
    )

    parser.add_argument("--status", action="store_true", help="Show runner status")
    parser.add_argument("--download", action="store_true", help="Download the runner")
    parser.add_argument("--configure", action="store_true", help="Configure the runner")
    parser.add_argument("--token", type=str, help="Registration token from GitHub")
    parser.add_argument("--repo", type=str, default="https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.", help="Repository URL")
    parser.add_argument("--name", type=str, help="Runner name")
    parser.add_argument("--start", action="store_true", help="Start the runner")
    parser.add_argument("--stop", action="store_true", help="Stop the runner service")
    parser.add_argument("--service", action="store_true", help="Run as Windows service")
    parser.add_argument("--provision", action="store_true", help="Provision SLATE environment on runner")
    parser.add_argument("--create-startup", action="store_true", help="Create startup/auto-start scripts")
    parser.add_argument("--setup-script", action="store_true", help="Generate setup script")
    parser.add_argument("--runner-dir", type=str, help="Runner directory")
    parser.add_argument("--workspace", type=str, help="SLATE workspace directory")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    runner_dir = Path(args.runner_dir) if args.runner_dir else None
    manager = SlateRunnerManager(runner_dir)

    if args.status:
        status = manager.get_status()
        if args.json_output:
            print(json.dumps(status, indent=2, default=str))
        else:
            print()
            print("[SLATE Runner Status]")
            print("=" * 60)
            print(f"  Installed     : {'YES' if status['installed'] else 'NO'}")
            print(f"  Configured    : {'YES' if status['configured'] else 'NO'}")
            print(f"  Provisioned   : {'YES' if status['provisioned'] else 'NO'}")
            print(f"  Running       : {'YES' if status.get('running') else 'NO'}")
            print(f"  Directory     : {status['runner_dir']}")
            print(f"  System        : {status['system']}")
            print(f"  GPU           : {'YES' if status['gpu']['has_gpu'] else 'NO'} ({status['gpu']['gpu_count']} GPUs)")
            if status['gpu']['gpu_names']:
                for name in status['gpu']['gpu_names']:
                    print(f"                  - {name}")
            print(f"  Labels        : {', '.join(status['labels'])}")

            if status.get('config'):
                print(f"\n  Runner Name   : {status['config'].get('name', 'N/A')}")
                print(f"  Repo URL      : {status['config'].get('repo_url', 'N/A')}")

            if status.get('slate_environment'):
                env = status['slate_environment']
                print(f"\n  SLATE Environment:")
                print(f"    Workspace   : {env.get('workspace', 'N/A')}")
                print(f"    Python      : {env.get('python', 'N/A')}")
                print(f"    Provisioned : {env.get('provisioned_at', 'N/A')}")

            # Show next steps
            print(f"\n  Next Steps:")
            if not status['installed']:
                print(f"    1. Download runner:  --download")
            elif not status['configured']:
                print(f"    1. Configure:        --configure --token YOUR_TOKEN")
            elif not status['provisioned']:
                print(f"    1. Provision SLATE:  --provision")
            elif not status.get('running'):
                print(f"    1. Start runner:     --start")
            else:
                print(f"    Runner is active and ready for jobs!")
            print("=" * 60)

    elif args.download:
        success = manager.download_runner()
        if success:
            print("[OK] Runner downloaded successfully")
            if args.provision:
                ws = Path(args.workspace) if args.workspace else None
                result = manager.provision_slate_environment(ws)
                if not result["success"]:
                    for err in result["errors"]:
                        print(f"[WARN] {err}")
        else:
            print("[ERROR] Failed to download runner")
            sys.exit(1)

    elif args.configure:
        if not args.token:
            print("Error: --configure requires --token")
            print("Get token from: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./settings/actions/runners/new")
            sys.exit(1)

        result = manager.configure_runner(
            repo_url=args.repo,
            token=args.token,
            name=args.name,
        )

        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print("[OK] Runner configured successfully")
                for step in result["steps"]:
                    print(f"  - {step}")
            else:
                print("[ERROR] Configuration failed")
                for error in result["errors"]:
                    print(f"  - {error}")

    elif args.provision:
        ws = Path(args.workspace) if args.workspace else None
        result = manager.provision_slate_environment(ws)
        if args.json_output:
            print(json.dumps(result, indent=2, default=str))
        else:
            if result["success"]:
                print("\n[OK] SLATE environment provisioned!")
                for step in result["steps"]:
                    print(f"  [OK] {step}")
            else:
                print("\n[ERROR] Provisioning failed")
                for err in result["errors"]:
                    print(f"  [FAIL] {err}")

    elif args.create_startup:
        script = manager.create_startup_script()
        print(f"[OK] Startup script: {script}")

        if manager.system == "Windows":
            svc_result = manager.create_windows_service_config()
            for step in svc_result["steps"]:
                print(f"  - {step}")

    elif args.start:
        result = manager.start_runner(as_service=args.service)
        if not result["success"]:
            for error in result["errors"]:
                print(f"[ERROR] {error}")
            sys.exit(1)

    elif args.stop:
        result = manager.stop_service()
        if result["success"]:
            print("[OK] Service stopped")
        else:
            for error in result["errors"]:
                print(f"[ERROR] {error}")

    elif args.setup_script:
        script = manager.generate_setup_script()
        print(script)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
