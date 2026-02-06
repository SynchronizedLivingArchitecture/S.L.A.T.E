# /slate-help

Display help for all SLATE commands and skills.

## Description
Quick reference for all SLATE plugin commands.

## Available Commands

### Core Commands
| Command | Description |
|---------|-------------|
| `/slate [start\|stop\|status]` | Manage SLATE orchestrator |
| `/slate-status` | Check system and service status |
| `/slate-workflow` | Manage task workflow queue |
| `/slate-runner` | Manage GitHub Actions runner |

### Quick Actions
```powershell
# Start all services
/slate start

# Check everything
/slate-status --quick

# View task queue
/slate-workflow --status

# Clean up stuck tasks
/slate-workflow --cleanup
```

### PowerShell Equivalents
```powershell
# Direct Python commands (if skills not working)
.\.venv\Scripts\python.exe slate/slate_orchestrator.py start
.\.venv\Scripts\python.exe slate/slate_status.py --quick
.\.venv\Scripts\python.exe slate/slate_workflow_manager.py --status
.\.venv\Scripts\python.exe slate/slate_runner_manager.py --status
```

## Getting Help
- GitHub Issues: https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/issues
- Documentation: See CLAUDE.md in project root
