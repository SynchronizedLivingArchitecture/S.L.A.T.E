# Modified: 2026-02-07T07:10:00Z | Author: COPILOT | Change: Create diagnostic script for workflow AI access issues
"""
SLATE Workflow AI Access Diagnostic
====================================
Checks all prerequisites for workflows to access local AI services:
- Ollama status & model availability
- Runner configuration & hooks
- Environment variables
- GPU accessibility
- Recent workflow failure analysis

Usage:
    python fix_workflow_ai_access.py              # Full diagnostic
    python fix_workflow_ai_access.py --fix        # Diagnose + auto-fix
    python fix_workflow_ai_access.py --json       # JSON output
"""

import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

WORKSPACE = pathlib.Path(__file__).parent.resolve()
RUNNER_DIR = WORKSPACE / 'actions-runner'
HOOKS_DIR = RUNNER_DIR / 'hooks'
PRE_JOB_HOOK = HOOKS_DIR / 'pre-job.ps1'
VENV_PYTHON = WORKSPACE / '.venv' / 'Scripts' / 'python.exe'
OLLAMA_URL = 'http://127.0.0.1:11434'


def check_ollama():
    """Check Ollama service status and models."""
    result = {'name': 'Ollama Service', 'status': 'unknown', 'details': {}}
    try:
        req = urllib.request.Request(f'{OLLAMA_URL}/api/tags', method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = data.get('models', [])
        model_names = [m.get('name', '?') for m in models]
        result['status'] = 'ok'
        result['details'] = {
            'url': OLLAMA_URL,
            'model_count': len(models),
            'models': model_names,
            'slate_models': [m for m in model_names if m.startswith('slate-')],
        }
    except Exception as e:
        result['status'] = 'error'
        result['details'] = {'error': str(e), 'url': OLLAMA_URL}
    return result


def check_runner():
    """Check GitHub Actions runner configuration."""
    result = {'name': 'Runner Configuration', 'status': 'unknown', 'details': {}}
    details = {}

    # Runner directory
    details['runner_dir'] = str(RUNNER_DIR)
    details['runner_exists'] = RUNNER_DIR.exists()

    if not RUNNER_DIR.exists():
        result['status'] = 'error'
        result['details'] = details
        return result

    # Runner config
    runner_cfg = RUNNER_DIR / '.runner'
    if runner_cfg.exists():
        try:
            cfg = json.loads(runner_cfg.read_text(encoding='utf-8'))
            details['agent_name'] = cfg.get('agentName', 'unknown')
            details['server_url'] = cfg.get('gitHubUrl', 'unknown')
        except Exception:
            details['agent_name'] = 'parse_error'

    # Hooks
    details['hooks_dir'] = str(HOOKS_DIR)
    details['hooks_exists'] = HOOKS_DIR.exists()
    details['pre_job_hook'] = str(PRE_JOB_HOOK)
    details['pre_job_exists'] = PRE_JOB_HOOK.exists()

    if PRE_JOB_HOOK.exists():
        hook_content = PRE_JOB_HOOK.read_text(encoding='utf-8')
        details['hook_has_ollama_host'] = 'OLLAMA_HOST' in hook_content
        details['hook_has_slate_ollama_url'] = 'SLATE_OLLAMA_URL' in hook_content
        details['hook_has_cuda_devices'] = 'CUDA_VISIBLE_DEVICES' in hook_content
        details['hook_has_python_path'] = 'PYTHONPATH' in hook_content
        details['hook_has_venv'] = '.venv' in hook_content
    else:
        details['hook_has_ollama_host'] = False

    # Check if runner process is running
    try:
        ps = subprocess.run(
            ['powershell', '-Command',
             'Get-Process -Name Runner.Listener -ErrorAction SilentlyContinue | Select-Object Id, StartTime'],
            capture_output=True, text=True, timeout=10
        )
        details['runner_process'] = ps.stdout.strip() if ps.stdout.strip() else 'not running'
    except Exception:
        details['runner_process'] = 'check_failed'

    # hooks.json
    hooks_json = RUNNER_DIR / 'hooks' / 'hooks.json'
    if hooks_json.exists():
        try:
            hooks_cfg = json.loads(hooks_json.read_text(encoding='utf-8'))
            details['hooks_json_pre_job'] = hooks_cfg.get('pre_job', {}).get('path', 'not set')
        except Exception:
            details['hooks_json_pre_job'] = 'parse_error'
    else:
        details['hooks_json_pre_job'] = 'missing'

    all_ok = (
        details['runner_exists']
        and details['pre_job_exists']
        and details.get('hook_has_ollama_host', False)
        and details.get('hook_has_cuda_devices', False)
    )
    result['status'] = 'ok' if all_ok else 'warning'
    result['details'] = details
    return result


def check_environment():
    """Check environment variables relevant to AI access."""
    result = {'name': 'Environment Variables', 'status': 'unknown', 'details': {}}
    env_vars = {
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
        'SLATE_WORKSPACE': os.environ.get('SLATE_WORKSPACE', 'not set'),
        'SLATE_RUNNER': os.environ.get('SLATE_RUNNER', 'not set'),
        'OLLAMA_HOST': os.environ.get('OLLAMA_HOST', 'not set'),
        'SLATE_OLLAMA_URL': os.environ.get('SLATE_OLLAMA_URL', 'not set'),
        'PYTHONPATH': os.environ.get('PYTHONPATH', 'not set'),
        'SLATE_GPU_COUNT': os.environ.get('SLATE_GPU_COUNT', 'not set'),
    }
    result['details'] = env_vars
    missing = [k for k, v in env_vars.items() if v == 'not set']
    result['status'] = 'ok' if not missing else 'warning'
    result['details']['missing'] = missing
    return result


def check_gpu():
    """Check GPU accessibility."""
    result = {'name': 'GPU Access', 'status': 'unknown', 'details': {}}
    try:
        ps = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if ps.returncode == 0 and ps.stdout.strip():
            gpus = []
            for line in ps.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpus.append({
                        'index': parts[0],
                        'name': parts[1],
                        'memory_used': parts[2],
                        'memory_total': parts[3],
                        'utilization': parts[4],
                    })
            result['status'] = 'ok'
            result['details'] = {'gpu_count': len(gpus), 'gpus': gpus}
        else:
            result['status'] = 'error'
            result['details'] = {'error': 'nvidia-smi failed', 'stderr': ps.stderr}
    except FileNotFoundError:
        result['status'] = 'error'
        result['details'] = {'error': 'nvidia-smi not found'}
    except Exception as e:
        result['status'] = 'error'
        result['details'] = {'error': str(e)}
    return result


def check_pytorch():
    """Check PyTorch CUDA availability."""
    result = {'name': 'PyTorch CUDA', 'status': 'unknown', 'details': {}}
    try:
        ps = subprocess.run(
            [str(VENV_PYTHON), '-c',
             'import torch; import json; print(json.dumps({"version": torch.__version__, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))'],
            capture_output=True, text=True, timeout=15
        )
        if ps.returncode == 0:
            data = json.loads(ps.stdout.strip())
            result['status'] = 'ok' if data['cuda'] else 'error'
            result['details'] = data
        else:
            result['status'] = 'error'
            result['details'] = {'error': ps.stderr.strip()[:200]}
    except Exception as e:
        result['status'] = 'error'
        result['details'] = {'error': str(e)}
    return result


def check_recent_failures():
    """Check recent workflow failures via GitHub API."""
    result = {'name': 'Recent Workflow Failures', 'status': 'unknown', 'details': {}}
    try:
        cred_result = subprocess.run(
            ['git', 'credential', 'fill'],
            input='protocol=https\nhost=github.com\n',
            capture_output=True, text=True, timeout=10
        )
        token = None
        for line in cred_result.stdout.splitlines():
            if line.startswith('password='):
                token = line.split('=', 1)[1]
                break

        if not token:
            result['status'] = 'warning'
            result['details'] = {'error': 'No GitHub token available'}
            return result

        base = 'https://api.github.com/repos/SynchronizedLivingArchitecture/S.L.A.T.E'
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
        }

        # Failed runs
        req = urllib.request.Request(
            f'{base}/actions/runs?status=failure&per_page=10',
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        total_failed = data.get('total_count', 0)
        recent = []
        for r in data.get('workflow_runs', [])[:10]:
            recent.append({
                'run_number': r['run_number'],
                'name': r['name'],
                'conclusion': r['conclusion'],
                'created_at': r['created_at'][:16],
            })

        # Categorize failures by workflow
        failure_by_workflow = {}
        for r in data.get('workflow_runs', []):
            name = r['name']
            failure_by_workflow[name] = failure_by_workflow.get(name, 0) + 1

        result['status'] = 'warning' if total_failed > 50 else 'ok'
        result['details'] = {
            'total_failed': total_failed,
            'recent_failures': recent,
            'failures_by_workflow': failure_by_workflow,
        }
    except Exception as e:
        result['status'] = 'error'
        result['details'] = {'error': str(e)}
    return result


def check_ollama_inference():
    """Test actual inference capability."""
    result = {'name': 'Inference Test', 'status': 'unknown', 'details': {}}
    try:
        payload = json.dumps({
            'model': 'llama3.2:3b',
            'prompt': 'Say hello in one word.',
            'stream': False,
            'options': {'num_predict': 10}
        }).encode()
        req = urllib.request.Request(
            f'{OLLAMA_URL}/api/generate',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start

        response_text = data.get('response', '')
        eval_count = data.get('eval_count', 0)
        eval_duration = data.get('eval_duration', 1)
        tps = eval_count / max(eval_duration / 1e9, 0.001) if eval_count else 0

        result['status'] = 'ok'
        result['details'] = {
            'model': 'llama3.2:3b',
            'response': response_text[:100],
            'tokens': eval_count,
            'tok_per_sec': round(tps, 1),
            'elapsed_sec': round(elapsed, 2),
        }
    except Exception as e:
        result['status'] = 'error'
        result['details'] = {'error': str(e)}
    return result


def apply_fixes():
    """Auto-fix common issues."""
    fixes_applied = []

    # Fix 1: Ensure pre-job hook has OLLAMA_HOST
    if PRE_JOB_HOOK.exists():
        content = PRE_JOB_HOOK.read_text(encoding='utf-8')
        if 'OLLAMA_HOST' not in content:
            # Add Ollama env vars before the log section
            addition = '\n# Inference / Agentic AI support\n$env:OLLAMA_HOST = "127.0.0.1:11434"\n$env:SLATE_OLLAMA_URL = "http://127.0.0.1:11434"\n'
            if '# Log job start' in content:
                content = content.replace('# Log job start', addition + '\n# Log job start')
            else:
                content += addition
            PRE_JOB_HOOK.write_text(content, encoding='utf-8')
            fixes_applied.append('Added OLLAMA_HOST to pre-job hook')

    # Fix 2: Ensure hooks.json points to pre-job hook
    hooks_json = HOOKS_DIR / 'hooks.json'
    if HOOKS_DIR.exists():
        expected = {
            'pre_job': {
                'path': str(PRE_JOB_HOOK),
                'args': []
            }
        }
        if hooks_json.exists():
            try:
                current = json.loads(hooks_json.read_text(encoding='utf-8'))
                if current.get('pre_job', {}).get('path') != str(PRE_JOB_HOOK):
                    hooks_json.write_text(json.dumps(expected, indent=2), encoding='utf-8')
                    fixes_applied.append('Fixed hooks.json pre_job path')
            except Exception:
                hooks_json.write_text(json.dumps(expected, indent=2), encoding='utf-8')
                fixes_applied.append('Recreated hooks.json')
        else:
            hooks_json.write_text(json.dumps(expected, indent=2), encoding='utf-8')
            fixes_applied.append('Created hooks.json')

    # Fix 3: Create .env file for runner
    env_file = RUNNER_DIR / '.env'
    env_content = (
        'OLLAMA_HOST=127.0.0.1:11434\n'
        'SLATE_OLLAMA_URL=http://127.0.0.1:11434\n'
        'CUDA_VISIBLE_DEVICES=0,1\n'
        'SLATE_GPU_COUNT=2\n'
        f'SLATE_WORKSPACE={WORKSPACE}\n'
        f'PYTHONPATH={WORKSPACE}\n'
        'PYTHONIOENCODING=utf-8\n'
        'SLATE_RUNNER=true\n'
    )
    if not env_file.exists() or 'OLLAMA_HOST' not in env_file.read_text(encoding='utf-8'):
        env_file.write_text(env_content, encoding='utf-8')
        fixes_applied.append('Created/updated .env with AI access variables')

    # Fix 4: Verify Ollama is running, start if not
    try:
        urllib.request.urlopen(f'{OLLAMA_URL}/api/tags', timeout=3)
    except Exception:
        try:
            subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x00000008  # DETACHED_PROCESS on Windows
            )
            time.sleep(3)
            fixes_applied.append('Started Ollama service')
        except Exception as e:
            fixes_applied.append(f'Failed to start Ollama: {e}')

    return fixes_applied


def main():
    as_json = '--json' in sys.argv
    do_fix = '--fix' in sys.argv

    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    checks = [
        check_ollama(),
        check_runner(),
        check_environment(),
        check_gpu(),
        check_pytorch(),
        check_recent_failures(),
    ]

    # Only run inference test if Ollama is up
    if checks[0]['status'] == 'ok':
        checks.append(check_ollama_inference())

    if as_json:
        output = {
            'timestamp': timestamp,
            'checks': checks,
        }
        if do_fix:
            output['fixes'] = apply_fixes()
        print(json.dumps(output, indent=2))
        return

    # Pretty print
    print('=' * 60)
    print('  SLATE Workflow AI Access Diagnostic')
    print(f'  {timestamp}')
    print('=' * 60)

    all_ok = True
    for check in checks:
        icon = 'OK' if check['status'] == 'ok' else ('WARN' if check['status'] == 'warning' else 'FAIL')
        if check['status'] != 'ok':
            all_ok = False
        print(f'\n  [{icon:>4}] {check["name"]}')
        details = check.get('details', {})
        for k, v in details.items():
            if isinstance(v, list) and len(v) > 5:
                print(f'         {k}: {len(v)} items')
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    print(f'         {k}.{dk}: {dv}')
            else:
                print(f'         {k}: {v}')

    if do_fix:
        print('\n' + '-' * 60)
        print('  Applying fixes...')
        fixes = apply_fixes()
        for f in fixes:
            print(f'    + {f}')
        if not fixes:
            print('    No fixes needed')

    print('\n' + '=' * 60)
    if all_ok:
        print('  Result: ALL CHECKS PASSED')
    else:
        print('  Result: ISSUES DETECTED - run with --fix to auto-repair')
    print('=' * 60)


if __name__ == '__main__':
    main()
