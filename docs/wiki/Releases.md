# Releases

This page documents S.L.A.T.E.'s release process, versioning strategy, and changelog management.

## Release Strategy

SLATE follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

| Version Part | When to Bump | Example |
|-------------|-------------|---------|
| **Major** | Breaking API changes, architecture rewrites | 2.0.0 → 3.0.0 |
| **Minor** | New features, modules, or capabilities | 2.4.0 → 2.5.0 |
| **Patch** | Bug fixes, documentation, minor tweaks | 2.4.0 → 2.4.1 |

Pre-release versions use suffixes: `2.5.0-beta.1`, `2.5.0-rc.1`

## Release Process

### 1. Prepare Release Locally

```bash
# Check current status
slate-package --status

# Validate package readiness
slate-package --validate

# Bump version (updates pyproject.toml + slate/__init__.py)
slate-package --bump minor

# Update CHANGELOG.md with release notes
# (edit manually or stamp unreleased section)
slate-package --changelog

# Build and verify package
slate-package --build

# Dry-run release check
slate-package --release 2.5.0

# Apply release (version bump + changelog stamp)
slate-package --release 2.5.0 --go
```

### 2. Commit and Push

```bash
git add -A
git commit -m "release: v2.5.0"
git push origin main
```

### 3. Trigger GitHub Release

Navigate to **Actions → Release** workflow on GitHub:
1. Click **Run workflow**
2. Enter version: `2.5.0`
3. Toggle pre-release if applicable
4. Toggle package publishing

The workflow will:
- ✅ Validate version format and consistency
- ✅ Extract changelog entry for the version
- ✅ Run test suite
- ✅ Build sdist + wheel
- ✅ Create git tag `v2.5.0`
- ✅ Create GitHub Release with notes and artifacts
- ✅ Publish to GitHub Packages (if enabled)

### 4. Verify Release

```bash
# Check the release was created
slate-package --status

# Verify package is installable
pip install slate --index-url https://pypi.pkg.github.com/SynchronizedLivingArchitecture/simple
```

## Automated Workflows

### `release.yml` — Full Release Pipeline

| Stage | Description |
|-------|-------------|
| **Validate** | Version format, consistency, tag uniqueness, changelog extraction |
| **Build Package** | Build sdist + wheel via `python -m build` |
| **Create Release** | Tag repo, create GitHub Release with notes + artifacts |
| **Publish Package** | Upload to GitHub Packages registry |

### `publish-package.yml` — Package Publishing

| Stage | Description |
|-------|-------------|
| **Build** | Build + verify with `twine check`, smoke test import |
| **Publish GitHub** | Upload to GitHub Packages, attach to release |
| **Publish PyPI** | Upload to PyPI (manual trigger only) |

### `cd.yml` — Continuous Deployment

| Stage | Description |
|-------|-------------|
| **Version** | Extract version from pyproject.toml or tag |
| **Build EXE** | Build Windows SLATEPI.exe via PyInstaller |
| **Release** | Create release with EXE artifact (tag pushes only) |

## Changelog

SLATE maintains a structured [CHANGELOG.md](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E./blob/main/CHANGELOG.md) following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

### Changelog Categories

| Category | Description |
|----------|-------------|
| **Added** | New features, modules, capabilities |
| **Changed** | Changes to existing functionality |
| **Deprecated** | Features that will be removed |
| **Removed** | Removed features or code |
| **Fixed** | Bug fixes |
| **Security** | Security patches |

### Version History

| Version | Date | Highlights |
|---------|------|------------|
| **2.4.0** | 2026-02-06 | GitHub Packages, release automation, package manager CLI |
| **2.3.0** | 2026-01-28 | 14 GitHub Actions workflows, PR templates, labels |
| **2.2.0** | 2026-01-20 | Dashboard framework, data visualization branch |
| **2.1.0** | 2026-01-15 | Multi-agent system (ALPHA/BETA/GAMMA/DELTA) |
| **2.0.0** | 2026-01-10 | Complete rewrite with slate SDK |
| **1.0.0** | 2025-11-13 | Initial prototype |

## GitHub Packages

SLATE packages are available on [GitHub Packages](https://github.com/orgs/SynchronizedLivingArchitecture/packages):

### Install

```bash
pip install slate --index-url https://pypi.pkg.github.com/SynchronizedLivingArchitecture/simple
```

### Package Contents

```
slate-2.4.0-py3-none-any.whl
├── slate/         # Core SDK modules
├── agents/              # Dashboard + API servers
└── slate_web/        # Templates + static assets
```

### Authentication

For private packages, configure pip with a GitHub token:

```bash
pip install slate \
  --index-url https://__token__:${GITHUB_TOKEN}@pypi.pkg.github.com/SynchronizedLivingArchitecture/simple
```

## Release Artifacts

Each release includes:

| Artifact | Type | Availability |
|----------|------|-------------|
| `slate-{ver}.tar.gz` | Source dist | GitHub Release + Packages |
| `slate-{ver}-py3-none-any.whl` | Wheel | GitHub Release + Packages |
| `SLATEPI-{ver}-win64.zip` | Windows EXE | GitHub Release (CD pipeline) |
| Release Notes | Markdown | GitHub Release page |
| Changelog Entry | Markdown | CHANGELOG.md |
