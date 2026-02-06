# Contributor Guide

How to contribute to SLATE as an external user.

## Repository Structure

SLATE uses a dual-repository model:

```
SLATE-BETA (Development)     SLATE (Public)           User Forks
         │                         │                       │
         │    sync-to-public       │                       │
         ├────────────────────────>│                       │
         │                         │      fork             │
         │                         ├──────────────────────>│
         │                         │                       │
         │                         │      contribute       │
         │                         │<──────────────────────┤
         │                         │                       │
```

| Repository | Purpose | Access |
|------------|---------|--------|
| **S.L.A.T.E.-BETA** | Development repo | Maintainers only |
| **S.L.A.T.E.** | Public installer | Everyone |
| **Your Fork** | Your personal SLATE | You |

## Getting Started

### 1. Fork the Public Repository

Fork from: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/S.L.A.T.E.git
cd S.L.A.T.E
```

### 3. Set Up Remotes

```bash
# Add upstream (public SLATE)
git remote add upstream https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git

# Verify remotes
git remote -v
# origin    https://github.com/YOUR-USERNAME/S.L.A.T.E.git (fetch)
# origin    https://github.com/YOUR-USERNAME/S.L.A.T.E.git (push)
# upstream  https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E.git (fetch)
```

### 4. Initialize Your SLATE

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -e ".[dev]"

# Initialize fork manager
python slate/slate_fork_manager.py --init --name "Your Name" --email "your@email.com"

# Set up your fork URL
python slate/slate_fork_manager.py --setup-fork https://github.com/YOUR-USERNAME/S.L.A.T.E.git
```

## Making Contributions

### 1. Sync with Upstream

Always sync before starting work:

```bash
# Fetch latest from upstream
git fetch upstream

# Merge into your branch
git merge upstream/main

# Or use the fork manager
python slate/slate_fork_manager.py --sync
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Or use the fork manager:

```bash
python slate/slate_fork_manager.py --contribute "your-feature" --title "Add feature X"
```

### 3. Make Your Changes

- Follow the [Development Guide](Development)
- Run tests: `python -m pytest tests/ -v`
- Validate: `python slate/slate_fork_manager.py --validate`

### 4. Validate Before Pushing

```bash
python slate/slate_fork_manager.py --validate
```

This checks:
- Required files exist
- Core modules import correctly
- No security violations
- pyproject.toml is valid

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub from your fork to the public SLATE repo.

## Contribution Requirements

### Must Have

- [ ] All tests pass
- [ ] Validation passes (`--validate`)
- [ ] No modifications to protected files
- [ ] Code binds to `127.0.0.1` only
- [ ] ActionGuard remains intact

### Protected Files

These files cannot be modified by contributors:

| File | Reason |
|------|--------|
| `.github/workflows/*` | Security-critical automation |
| `.github/CODEOWNERS` | Access control |
| `slate/action_guard.py` | Security enforcement |
| `slate/sdk_source_guard.py` | Package validation |

### Security Requirements

1. **Network binding**: Only `127.0.0.1` allowed (never `0.0.0.0`)
2. **No eval/exec**: Avoid dynamic code execution
3. **No external calls**: SLATE is local-only
4. **No credentials**: Never commit secrets

## What Happens After PR

1. **Automated Validation**
   - Fork validation workflow runs
   - Security scans check for dangerous patterns
   - SLATE prerequisites verified

2. **Bot Comment**
   - First-time contributors get a welcome message
   - Validation results posted as comment

3. **Labels Applied**
   - `external-contributor` - PR from fork
   - `validation-passed` - All checks passed
   - `needs-fixes` - Issues found

4. **Maintainer Review**
   - Once validation passes, maintainers review
   - May request changes
   - Approved PRs get merged

## Your SLATE as a Personal Development Environment

Your fork isn't just for contributing - it's your personal SLATE installation:

```
Your Fork (S.L.A.T.E.)
├── Your customizations
├── Your agents
├── Your tasks
└── Syncs with upstream for updates
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

### Personal Customizations

You can customize your SLATE without contributing back:

- Add personal agents to `agents/`
- Customize `CLAUDE.md` for your workflow
- Add project-specific specs to `specs/`

Just keep these on branches that you don't PR upstream.

## Feedback Loop Vision

The ultimate goal is a feedback loop:

```
Upstream SLATE
     │
     ├─> User A's SLATE ─> Improvements ─┐
     │                                    │
     ├─> User B's SLATE ─> Bug fixes ────>├──> Upstream SLATE
     │                                    │
     └─> User C's SLATE ─> Features ─────┘
```

Each SLATE user:
1. Installs from the public repo
2. Customizes for their needs
3. Discovers improvements/fixes
4. Contributes back to benefit everyone

## Troubleshooting

### "Fork validation failed"

Run locally:
```bash
python slate/slate_fork_manager.py --validate
```

Fix any errors shown.

### "Protected file modified"

You cannot modify security-critical files. Revert those changes:
```bash
git checkout upstream/main -- .github/workflows/
```

### "Merge conflicts"

```bash
# Fetch latest
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Resolve conflicts, then continue
git rebase --continue
```

## Next Steps

- [Development Guide](Development) - Code style and testing
- [Architecture](Architecture) - System design
- [API Reference](API-Reference) - Module documentation
