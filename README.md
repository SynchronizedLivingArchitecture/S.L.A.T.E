<p align="center">
  <img src="docs/assets/slate-logo-banner.svg" alt="S.L.A.T.E. — Synchronized Living Architecture for Transformation and Evolution" width="800">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./actions"><img src="https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./wiki"><img src="https://img.shields.io/badge/docs-wiki-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./releases"><img src="https://img.shields.io/github/v/release/SynchronizedLivingArchitecture/S.L.A.T.E.?include_prereleases&label=release&color=7c3aed" alt="Release"></a>
</p>

<p align="center">
  A <strong>GitHub-centric</strong> development framework with GPU-accelerated<br>
  self-hosted runners for automated CI/CD workflows.
</p>

---

> **Note**: This project is experimental. Not suitable for production use.

## Overview

SLATE provides GPU-accelerated CI/CD through GitHub Actions with self-hosted runners:

```
GitHub Repository (S.L.A.T.E.)
         |
    +----+----+
    v         v
  Cloud     Self-Hosted
  Runner     Runner
             (2x RTX 5070 Ti)
         |
         v
   GPU Workloads
   (tests, builds)
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E..git
cd S.L.A.T.E.
python install_slate.py
```

### 2. Verify Installation

```bash
python slate/slate_sdk.py --verify
```

### 3. With Self-Hosted Runner

```bash
python install_slate.py --runner --runner-token YOUR_TOKEN
```

Get your runner token from: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./settings/actions/runners

## Self-Hosted Runner

| Property | Value |
|----------|-------|
| **Name** | `slate-DESKTOP-R3UD82D` |
| **Labels** | `self-hosted`, `slate`, `gpu`, `windows`, `cuda`, `gpu-2`, `multi-gpu`, `blackwell` |
| **Hardware** | 2x RTX 5070 Ti (Blackwell) |

### Runner Commands

```bash
# Check status
python slate/slate_runner_manager.py --status

# Start runner
python slate/slate_runner_manager.py --start

# Stop runner
python slate/slate_runner_manager.py --stop

# Install as Windows service
python slate/slate_runner_manager.py --install-service
```

## Project Structure

```
slate/                  # Core SLATE SDK modules
  slate_sdk.py          # Unified SDK (setup, status, verify)
  slate_status.py       # System status checker
  slate_runner_manager.py     # GitHub runner management
  slate_github_integration.py # GitHub API integration
agents/                 # Dashboard and API servers
tests/                  # Test suite
.github/
  workflows/            # GitHub Actions workflows
```

## CLI Reference

```bash
# SDK Status
python slate/slate_sdk.py --status

# System Status
python slate/slate_status.py --quick

# Run Tests
python -m pytest tests/ -v

# Start Dashboard
python agents/slate_dashboard_server.py
```

## System Requirements

- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.11+
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA RTX series (for self-hosted runner)

## AI Agent Integration

SLATE provides first-class integration with AI coding assistants via the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

### GitHub Copilot

Copilot automatically loads SLATE context when the workspace is open:

- **Agent instructions**: `.github/copilot-instructions.md` — project conventions, architecture, security
- **MCP tools**: `.vscode/mcp.json` — 8 SLATE tools available in Copilot Chat

### Claude (Desktop / Code)

- **CLAUDE.md** — full project context, commands, integration checklist
- **`.claude/skills/`** — 5 domain skills (status, tasks, agents, benchmark, traces)
- **MCP server** — same server works with Claude Desktop via `claude_desktop_config.json`

### VS Code Integration

SLATE ships with full VS Code workspace configuration:

- **Extensions** (`.vscode/extensions.json`): Copilot, Copilot Chat, GitHub Actions, PR & Issues, Python, Pylance, Debugpy
- **MCP Tools** (`.vscode/mcp.json`): 8 SLATE tools available in Copilot Chat
- **Tasks** (`.vscode/tasks.json`): Build, run, and test tasks

Verify your setup: `python slate/slate_sdk.py --integrate-vscode`

### MCP Server

```bash
# Verify tools are registered
python slate/slate_mcp_server.py --verify

# stdio mode (Copilot / Claude)
python slate/slate_mcp_server.py

# SSE mode (web clients)
python slate/slate_mcp_server.py --sse --port 6274
```

Available tools: `slate_get_status`, `slate_run_check`, `slate_list_tasks`,
`slate_gpu_info`, `slate_agent_status`, `slate_runner_status`, `slate_search_code`,
`slate_dashboard_url`.

## GitHub Integrations

| Integration | Config |
|-------------|--------|
| Actions (17 workflows) | `.github/workflows/` |
| Self-Hosted Runner (2x GPU) | `slate/slate_runner_manager.py` |
| Projects (3 templates) | `.github/projects.json` |
| Issue Templates (3) | `.github/ISSUE_TEMPLATE/` |
| PR Template | `.github/PULL_REQUEST_TEMPLATE.md` |
| Dependabot | `.github/dependabot.yml` |
| CodeQL | `.github/workflows/codeql.yml` |
| CODEOWNERS | `.github/CODEOWNERS` |
| Releases & Packages | `.github/workflows/release.yml` |
| Label Sync | `.github/labels.yml` |
| Security Advisories | `.github/SECURITY.md` |
| Funding | `.github/FUNDING.yml` |
| Wiki (14 pages) | `docs/wiki/` |
| Copilot Agent | `.github/copilot-instructions.md` |
| Claude Plugin | `CLAUDE.md` + `.claude/skills/` |
| MCP Server | `slate/slate_mcp_server.py` |
| VS Code Extensions | `.vscode/extensions.json` |
| VS Code MCP Tools | `.vscode/mcp.json` |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Repository](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.)
- [Wiki](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./wiki)
- [Issues](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./issues)

---

<p align="center">
  <img src="docs/assets/slate-logo.svg" alt="S.L.A.T.E." width="80">
  <br>
  <strong>S.L.A.T.E.</strong> — Synchronized Living Architecture for Transformation and Evolution
</p>
