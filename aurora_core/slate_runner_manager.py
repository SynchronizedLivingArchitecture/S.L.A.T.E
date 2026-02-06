#!/usr/bin/env python3
"""
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_runner_manager [python]
# Author: Claude | Created: 2026-02-06T22:00:00Z
# SLATE Self-Hosted Runner Manager - Sets up and manages GitHub Actions runners
# ═══════════════════════════════════════════════════════════════════════════════

Manages self-hosted GitHub Actions runners for SLATE:
- Downloads and configures the runner
- Registers with SLATE repos (main and BETA)
- Runs as Windows service or interactive
- GPU-aware job routing
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup path
WORKSPACE_ROOT = Path(__file__).parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

logger = logging.getLogger("aurora.runner_manager")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL: constants [python]
# Author: Claude | Created: 2026-02-06T22:00:00Z
# ═══════════════════════════════════════════════════════════════════════════════

RUNNER_VERSION = "2.321.0"
RUNNER_BASE_URL = "https://github.com/actions/runner/releases/download"

SLATE_REPOS = [
    "SynchronizedLivingArchitecture/S.L.A.T.E.",
    "SynchronizedLivingArchitecture/S.L.A.T.E.-BETA",
]

DEFAULT_RUNNER_DIR = Path("C:/actions-runner") if platform.system() == "Windows" else Path.home() / "actions-runner"

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "repo_url": self.repo_url,
            "runner_dir": str(self.runner_dir),
            "labels": self.labels,
            "work_dir": str(self.work_dir) if self.work_dir else None,
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
        labels = RUNNER_LABELS.copy()

        gpu_info = self.detect_gpu()
        if gpu_info["has_gpu"]:
            labels.append("cuda")
            labels.append(f"gpu-{gpu_info['gpu_count']}")

            # Add GPU architecture labels
            for name in gpu_info["gpu_names"]:
                name_lower = name.lower()
                if "5070" in name or "5080" in name or "5090" in name:
                    labels.append("blackwell")
                elif "4090" in name or "4080" in name or "4070" in name:
                    labels.append("ada-lovelace")
                elif "3090" in name or "3080" in name or "3070" in name:
                    labels.append("ampere")

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

    def get_status(self) -> Dict[str, Any]:
        """Get runner status."""
        status = {
            "installed": (self.runner_dir / "config.cmd").exists() or (self.runner_dir / "config.sh").exists(),
            "configured": self.config_file.exists(),
            "runner_dir": str(self.runner_dir),
            "system": self.system,
            "gpu": self.detect_gpu(),
            "labels": self.get_runner_labels(),
        }

        if status["configured"]:
            try:
                config = json.loads(self.config_file.read_text())
                status["config"] = config
            except Exception:
                pass

        return status

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

  # Download runner
  python slate_runner_manager.py --download

  # Configure for SLATE repo
  python slate_runner_manager.py --configure --token YOUR_TOKEN

  # Start runner
  python slate_runner_manager.py --start

  # Start as Windows service
  python slate_runner_manager.py --start --service

  # Generate setup script
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
    parser.add_argument("--setup-script", action="store_true", help="Generate setup script")
    parser.add_argument("--runner-dir", type=str, help="Runner directory")
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
            print(json.dumps(status, indent=2))
        else:
            print("\n📊 SLATE Runner Status")
            print("=" * 50)
            print(f"  Installed:    {'✅' if status['installed'] else '❌'}")
            print(f"  Configured:   {'✅' if status['configured'] else '❌'}")
            print(f"  Directory:    {status['runner_dir']}")
            print(f"  System:       {status['system']}")
            print(f"  GPU:          {'✅' if status['gpu']['has_gpu'] else '❌'} ({status['gpu']['gpu_count']} GPUs)")
            if status['gpu']['gpu_names']:
                for name in status['gpu']['gpu_names']:
                    print(f"                - {name}")
            print(f"  Labels:       {', '.join(status['labels'])}")

            if status.get('config'):
                print(f"\n  Runner Name:  {status['config'].get('name', 'N/A')}")
                print(f"  Repo URL:     {status['config'].get('repo_url', 'N/A')}")

    elif args.download:
        success = manager.download_runner()
        if success:
            print("✅ Runner downloaded successfully")
        else:
            print("❌ Failed to download runner")
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
                print("✅ Runner configured successfully")
                for step in result["steps"]:
                    print(f"  • {step}")
            else:
                print("❌ Configuration failed")
                for error in result["errors"]:
                    print(f"  • {error}")

    elif args.start:
        result = manager.start_runner(as_service=args.service)
        if not result["success"]:
            for error in result["errors"]:
                print(f"❌ {error}")
            sys.exit(1)

    elif args.stop:
        result = manager.stop_service()
        if result["success"]:
            print("✅ Service stopped")
        else:
            for error in result["errors"]:
                print(f"❌ {error}")

    elif args.setup_script:
        script = manager.generate_setup_script()
        print(script)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
