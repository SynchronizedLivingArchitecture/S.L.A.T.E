# Self-Hosted GitHub Actions Runners

SLATE supports self-hosted GitHub Actions runners for GPU-intensive CI/CD tasks. This guide covers setup, configuration, and management.

## Overview

Self-hosted runners allow SLATE to:
- Run GPU-accelerated tests (CUDA, PyTorch)
- Execute Polecat agent workloads locally
- Avoid GitHub-hosted runner time limits
- Use your local hardware (RTX 5070 Ti, etc.)

## Prerequisites

- Windows 10/11 or Linux
- NVIDIA GPU with CUDA support
- Python 3.11+
- GitHub account with repo admin access

## Quick Setup

### 1. Get Registration Token

1. Go to your repository on GitHub
2. Navigate to **Settings > Actions > Runners**
3. Click **New self-hosted runner**
4. Copy the registration token

### 2. Run SLATE Runner Manager

```powershell
# Check current status
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --status

# Download the runner
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --download

# Configure with your token
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --configure --token YOUR_TOKEN

# Start the runner
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --start
```

### 3. Verify Registration

Check GitHub Settings > Actions > Runners to see your runner listed as "Idle".

## Runner Labels

SLATE automatically detects and applies these labels:

| Label | Description |
|-------|-------------|
| `self-hosted` | Standard self-hosted label |
| `slate` | SLATE system runner |
| `gpu` | Has NVIDIA GPU |
| `cuda` | CUDA available |
| `windows` / `linux` | Operating system |
| `gpu-2` | Number of GPUs |
| `blackwell` | RTX 50 series |
| `ada-lovelace` | RTX 40 series |
| `ampere` | RTX 30 series |

## Running as Windows Service

For persistent background operation:

```powershell
# Install and start as service (requires Admin)
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --start --service

# Stop the service
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --stop
```

## Manual Setup (Alternative)

If you prefer manual setup:

```powershell
# Generate setup script
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --setup-script > setup_runner.ps1

# Run the script
.\setup_runner.ps1
```

## Workflow Configuration

Workflows that use self-hosted runners specify labels:

```yaml
jobs:
  gpu-tests:
    runs-on: [self-hosted, slate, gpu, windows]
    steps:
      - uses: actions/checkout@v4
      # ... GPU-enabled steps
```

## Security Considerations

- Self-hosted runners execute code from PRs
- Only enable for **private repos** or **trusted contributors**
- Runners have access to your local machine
- Use dedicated runner accounts when possible

### Recommended Security Settings

1. **Repository Settings > Actions > General**
   - "Require approval for all outside collaborators"

2. **Limit runner scope**
   - Register per-repository, not organization-wide

3. **Network isolation**
   - Runner binds to localhost only
   - ActionGuard blocks external API calls

## Troubleshooting

### Runner Not Appearing

```powershell
# Check configuration
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --status

# Re-configure with new token
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --configure --token NEW_TOKEN
```

### GPU Not Detected

```powershell
# Verify NVIDIA driver
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Service Won't Start

```powershell
# Check Windows Event Viewer for errors
# Or run interactively to see output:
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --start
```

## Dual-Repository Setup

For SLATE + SLATE-BETA, register runners for both repos:

```powershell
# Register for main SLATE
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --configure \
  --repo https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E. \
  --token SLATE_TOKEN \
  --name slate-runner-main

# Register for BETA (separate runner or use organization runner)
.\.venv\Scripts\python.exe aurora_core/slate_runner_manager.py --configure \
  --repo https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.-BETA \
  --token BETA_TOKEN \
  --name slate-runner-beta
```

## Related Documentation

- [Configuration.md](Configuration.md) - Environment setup
- [Development.md](Development.md) - Contributing guide
- [Troubleshooting.md](Troubleshooting.md) - Common issues
