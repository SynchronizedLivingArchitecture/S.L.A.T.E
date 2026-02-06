# Changelog

All notable changes to S.L.A.T.E. (System Learning Agent for Task Execution) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GPU tests workflow for self-hosted runners
- Self-hosted GitHub Actions runner manager (`aurora_core/slate_runner_manager.py`)
- Self-Hosted-Runners wiki documentation

## [2.4.0] - 2026-02-06

### Added
- **GitHub Packages Integration**: Automated package publishing via `publish-package.yml`
- **SLATE Package Manager**: CLI tool for managing packages, releases, and versioning (`aurora_core/slate_package_manager.py`)
- **CHANGELOG.md**: Structured changelog following Keep a Changelog format
- **Release Automation**: Enhanced release workflow with changelog generation and package assets
- SVG logo assets (orbital agent diagram and banner)
- Wiki pages: Packages, Projects, Self-Hosted-Runners
- GitHub Issue Templates (bug report, feature request, task)
- Dependabot configuration (pip + GitHub Actions weekly scans)
- FUNDING.yml for sponsorship configuration
- Beta fork configuration (`.slate_fork/config.json`)
- Fork manager with beta remote support and credential bypass
- `install_slate.py` 10-step canonical installer with InstallTracker
- Install progress API with SSE endpoints
- Install dashboard UI (dark glass theme)

### Changed
- Rebuilt SVG logos for GitHub compatibility (no defs/gradients/animations)
- Updated README with comprehensive badges and documentation links
- Wiki sidebar updated with Packages & Projects sections

### Fixed
- SVG rendering on GitHub (removed unsupported SVG features)
- Git push for workflow files (credential bypass for OAuth scope)

## [2.3.0] - 2026-01-28

### Added
- 14 GitHub Actions workflows (CI, CD, PR, nightly, CodeQL, release, fork-validation, label-sync, docs, dependabot-auto-merge, stale, contributor-pr, contribute-to-main)
- PR template with review checklist
- Labels configuration and auto-sync workflow
- CODEOWNERS for review routing
- SECURITY.md security policy
- `.github/slate.config.yaml` centralized configuration

### Changed
- Repository renamed from `11132025` to `S.L.A.T.E.-BETA`

## [2.2.0] - 2026-01-20

### Added
- Dashboard server framework (`agents/aurora_dashboard_server.py`)
- Data visualization dashboard branch (`001-data-viz-dashboard`)
- Subagent visual monitoring system
- ML orchestrator for model training pipelines
- Unified autonomous runner

## [2.1.0] - 2026-01-15

### Added
- Multi-agent system (ALPHA, BETA, GAMMA, DELTA)
- Task queue and autonomous execution loop
- Integrated autonomous loop with tech tree task generation
- Copilot autonomous runner integration

## [2.0.0] - 2026-01-10

### Added
- Complete rewrite with `aurora_core` SDK
- Hardware detection and GPU optimization (Blackwell/Ada/Ampere/Turing)
- PyTorch 2.7+ CUDA integration
- CLI tools: `slate-status`, `slate-runtime`, `slate-benchmark`, `slate-hardware`
- Modern `pyproject.toml` packaging
- Virtual environment management
- OpenTelemetry tracing foundation

### Changed
- Migrated from flat scripts to modular package architecture
- Python 3.11+ required

## [1.0.0] - 2025-11-13

### Added
- Initial S.L.A.T.E. prototype
- Basic agent execution framework
- Aurora workflow engine
- Task management system

[Unreleased]: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./compare/v2.4.0...HEAD
[2.4.0]: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./releases/tag/v1.0.0
