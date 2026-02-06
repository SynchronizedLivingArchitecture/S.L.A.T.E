# Development Guide

Contributing to SLATE - guidelines, workflows, and best practices.

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- Ollama (for local LLM)
- NVIDIA GPU (optional but recommended)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-org/slate.git
cd slate

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Check system status
python slate/slatepi_status.py --quick
```

## Code Style

### Python Guidelines

- **Type hints required** for all function signatures
- **Google-style docstrings** for public functions
- **Use `Annotated`** for tool parameters
- **Maximum line length**: 100 characters

```python
from typing import Annotated

def process_task(
    task_id: str,
    priority: Annotated[int, "Priority level 1-5"] = 3,
) -> dict:
    """Process a task with the given priority.

    Args:
        task_id: Unique task identifier.
        priority: Priority level (1=lowest, 5=highest).

    Returns:
        Dictionary containing task results.

    Raises:
        ValueError: If task_id is invalid.
    """
    ...
```

### Import Organization

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import httpx
from fastapi import FastAPI

# Local imports - add workspace root to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from slate import get_status
```

### File Locking

Always use `FileLock` when accessing `current_tasks.json`:

```python
from slate_core.file_lock import FileLock

with FileLock("current_tasks.json"):
    with open("current_tasks.json", "r") as f:
        tasks = json.load(f)
    # Modify tasks...
    with open("current_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
```

## Testing

### Test-Driven Development

SLATE follows strict TDD (mandated by constitution):

```
1. WRITE TEST → Define expected behavior
2. RUN TEST   → Verify it fails (red)
3. IMPLEMENT  → Minimum code to pass
4. RUN TEST   → Verify it passes (green)
5. REFACTOR   → Clean up, keep green
```

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_greeting.py -v

# With coverage
python -m pytest tests/ --cov=slate -q

# Coverage report
python -m pytest tests/ --cov=slate --cov-report=html
```

### Test Structure

```
tests/
  unit/           # Unit tests
  integration/    # Integration tests
  contract/       # Contract tests
  test_*.py       # Top-level tests
```

### Writing Tests

```python
import pytest
from slate.greeting import greet

class TestGreeting:
    """Tests for greeting module."""

    def test_greet_default(self):
        """Test greeting with default name."""
        result = greet()
        assert result == "Hello, World!"

    def test_greet_custom_name(self):
        """Test greeting with custom name."""
        result = greet("SLATE")
        assert result == "Hello, SLATE!"

    @pytest.mark.parametrize("name,expected", [
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
    ])
    def test_greet_parametrized(self, name, expected):
        """Test greeting with various names."""
        assert greet(name) == expected
```

### Coverage Requirements

- Target: 50%+ coverage for `slate/` and `slate_core/`
- All new code must include tests
- Critical paths require higher coverage

## Git Workflow

### Branch Naming

```
feature/short-description
bugfix/issue-number-description
refactor/component-name
docs/what-is-being-documented
```

### Commit Messages

Follow conventional commits:

```
feat: Add new task routing algorithm
fix: Resolve GPU memory leak in batch processing
docs: Update CLI reference documentation
refactor: Simplify agent dispatcher logic
test: Add unit tests for greeting module
chore: Update dependencies
```

### Pull Request Process

1. Create feature branch from `main`
2. Make changes with tests
3. Run full test suite
4. Update documentation if needed
5. Create PR with description
6. Address review feedback
7. Squash and merge

## Project Structure

```
slate/          # Core modules
  __init__.py         # Package exports
  action_guard.py     # Security enforcement
  ollama_client.py    # Ollama integration
  foundry_local.py    # Foundry Local client
  unified_ai_backend.py # AI routing
  slatepi_*.py        # CLI tools

agents/               # Agent implementations
  slate_dashboard_server.py

slate_core/           # Shared infrastructure
  file_lock.py        # File locking

tests/                # Test suite
  unit/
  integration/

docs/                 # Documentation
  wiki/

specs/                # Feature specifications (numbered for ordering)
  NNN-feature-name/   # e.g., 001-user-auth/, 002-dashboard/
```

## Adding New Features

### 1. Create Specification

Create a new spec folder:

```
specs/
  00X-feature-name/
    spec.md       # Feature specification
    plan.md       # Implementation plan
    tasks.md      # Task breakdown
```

### 2. Write Tests First

```python
# tests/test_new_feature.py
import pytest

def test_new_feature_basic():
    """Test basic functionality."""
    # This will fail until implemented
    from slate.new_feature import do_thing
    assert do_thing() == "expected result"
```

### 3. Implement Feature

```python
# slate/new_feature.py
"""New feature module."""

def do_thing() -> str:
    """Do the thing.

    Returns:
        The expected result.
    """
    return "expected result"
```

### 4. Export from Package

```python
# slate/__init__.py
from .new_feature import do_thing

__all__ = [
    # ... existing exports
    "do_thing",
]
```

### 5. Update Documentation

- Add to CLAUDE.md if significant
- Update relevant wiki pages
- Add CLI documentation if applicable

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System State

```bash
# Full status
python slate/slatepi_status.py

# Check specific integration
python slate/slatepi_runtime.py --check ollama
```

### Common Debug Points

- `current_tasks.json` - Task queue state
- `.slate_errors/` - Error logs
- `slate_cache/` - LLM response cache

## Security Guidelines

### Never Do

- Bind servers to `0.0.0.0`
- Call external paid APIs
- Store secrets in code
- Disable ActionGuard

### Always Do

- Use localhost binding
- Validate all inputs
- Use file locks for shared state
- Follow CSP guidelines

## Getting Help

- Check [Troubleshooting](Troubleshooting) for common issues
- Review existing specs in `specs/`
- Read the constitution at `.specify/memory/constitution.md`

## Next Steps

- [Troubleshooting](Troubleshooting) - Common issues and solutions
- [API Reference](API-Reference) - Module documentation
- [Architecture](Architecture) - System design
