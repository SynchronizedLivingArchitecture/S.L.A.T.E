# Contributing to SLATE

Thank you for your interest in contributing to S.L.A.T.E. (Synchronized Living Architecture for Transformation and Evolution)!

## Quick Start

1. **Fork** the repository: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.
2. **Clone** your fork locally
3. **Initialize** your SLATE workspace:
   ```bash
   python slate/slate_fork_manager.py --init --name "Your Name" --email "you@example.com"
   ```
4. **Validate** your changes before submitting:
   ```bash
   python slate/slate_fork_manager.py --validate
   ```

## Contribution Workflow

```
Your Fork → Feature Branch → Validate → PR → Review → Merge
```

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the existing code style
- Add tests for new features
- Update documentation if needed

### 3. Validate

```bash
# Run tests
python -m pytest tests/ -v

# Validate SLATE prerequisites
python slate/slate_fork_manager.py --validate
```

### 4. Submit PR

Push your branch and create a Pull Request against the `main` branch.

## Requirements

All contributions must:

- [ ] Pass all tests
- [ ] Pass validation checks
- [ ] Not modify protected files (see below)
- [ ] Bind only to `127.0.0.1` (never `0.0.0.0`)
- [ ] Keep ActionGuard intact

## Protected Files

These files cannot be modified by external contributors:

- `.github/workflows/*` - CI/CD automation
- `.github/CODEOWNERS` - Access control
- `slate/action_guard.py` - Security enforcement
- `slate/sdk_source_guard.py` - Package validation

## Code Style

- **Python 3.11+** required
- **Type hints** for all functions
- **Google-style docstrings**
- **Ruff** for linting and formatting

```bash
# Lint
ruff check slate/ agents/

# Format
ruff format slate/ agents/
```

## Security

SLATE is a **local-only** system. All contributions must:

1. Bind servers to `127.0.0.1` only
2. Avoid `eval()`, `exec()` with dynamic content
3. Never include credentials or API keys
4. Not make external network calls without explicit user consent

## Getting Help

- Check the [wiki](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/wiki)
- Open an [issue](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/issues)
- Read `CLAUDE.md` for project guidelines

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*Thank you for helping make SLATE better!*
