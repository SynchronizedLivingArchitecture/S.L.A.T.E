#!/usr/bin/env python3
# Modified: 2026-02-06T21:00:00Z | Author: Claude | Change: SLATE Dashboard Server with runner integration
"""
SLATE Dashboard Server
======================
Unified dashboard server for S.L.A.T.E. system monitoring and control.

Features:
- System status overview
- Self-hosted runner management
- GPU monitoring
- Installation tracking
- Task queue visualization

Endpoints:
    GET  /                      → Dashboard HTML
    GET  /api/status            → Full SLATE status
    GET  /api/runner/*          → Runner control endpoints
    GET  /api/install/*         → Install tracking endpoints
    GET  /api/gpu               → GPU status
    GET  /api/tasks             → Task queue status
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Run: pip install fastapi uvicorn")
    sys.exit(1)


# ─── Create FastAPI App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="S.L.A.T.E. Dashboard",
    description="Synchronized Living Architecture for Transformation and Evolution",
    version="2.4.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Mount API endpoints ────────────────────────────────────────────────────────

# Runner API
try:
    from agents.runner_api import add_runner_endpoints
    add_runner_endpoints(app)
except ImportError:
    print("[WARN] Runner API not available")

# Install API
try:
    from agents.install_api import add_install_endpoints
    add_install_endpoints(app)
except ImportError:
    print("[WARN] Install API not available")

# GitHub Models API
try:
    from agents.github_models_api import add_github_models_endpoints
    add_github_models_endpoints(app)
except ImportError:
    print("[WARN] GitHub Models API not available")


# ─── Core API Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/status")
async def slate_status():
    """Return comprehensive SLATE system status."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "version": "2.4.0",
        "components": {},
    }

    # SLATE SDK status
    try:
        from slate.slate_sdk import SlateSDK
        sdk = SlateSDK()
        status["components"]["sdk"] = sdk.get_status()
    except Exception as e:
        status["components"]["sdk"] = {"error": str(e)}

    # Runner status
    try:
        from slate.slate_runner_manager import SlateRunnerManager
        manager = SlateRunnerManager()
        status["components"]["runner"] = manager.get_status()
    except Exception as e:
        status["components"]["runner"] = {"error": str(e)}

    # GPU status
    try:
        from slate.slate_runner_manager import SlateRunnerManager
        manager = SlateRunnerManager()
        status["components"]["gpu"] = manager.detect_gpu()
    except Exception as e:
        status["components"]["gpu"] = {"error": str(e)}

    # Task queue
    try:
        tasks_file = WORKSPACE_ROOT / "current_tasks.json"
        if tasks_file.exists():
            with open(tasks_file) as f:
                tasks = json.load(f)
            pending = len([t for t in tasks if t.get("status") == "pending"])
            in_progress = len([t for t in tasks if t.get("status") == "in-progress"])
            completed = len([t for t in tasks if t.get("status") == "completed"])
            status["components"]["tasks"] = {
                "total": len(tasks),
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
            }
        else:
            status["components"]["tasks"] = {"total": 0}
    except Exception as e:
        status["components"]["tasks"] = {"error": str(e)}

    return JSONResponse(content=status)


@app.get("/api/gpu")
async def gpu_status():
    """Return GPU status."""
    try:
        from slate.slate_runner_manager import SlateRunnerManager
        manager = SlateRunnerManager()
        return JSONResponse(content=manager.detect_gpu())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/tasks")
async def task_status():
    """Return task queue status."""
    try:
        tasks_file = WORKSPACE_ROOT / "current_tasks.json"
        if tasks_file.exists():
            with open(tasks_file) as f:
                tasks = json.load(f)
            return JSONResponse(content={"tasks": tasks, "count": len(tasks)})
        return JSONResponse(content={"tasks": [], "count": 0})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── Dashboard HTML ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S.L.A.T.E. Dashboard</title>
<style>
:root {
  --bg-primary: #08090b;
  --bg-secondary: #111114;
  --bg-tertiary: #1a1a1f;
  --text-primary: #e4e4e7;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  --accent: #a78bfa;
  --accent-dim: #a78bfa40;
  --success: #22c55e;
  --warning: #eab308;
  --error: #ef4444;
  --border: #27272a;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  min-height: 100vh;
  line-height: 1.5;
}

.header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header h1 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--accent);
}

.header .version {
  color: var(--text-muted);
  font-size: 0.875rem;
}

.main {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 1.5rem;
}

.panel {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

.panel-header {
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-header h2 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.panel-body {
  padding: 1.25rem;
}

.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
}

.status-row:last-child { border-bottom: none; }

.status-label {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.status-value {
  font-weight: 500;
  font-size: 0.875rem;
}

.status-value.online { color: var(--success); }
.status-value.offline { color: var(--error); }
.status-value.warning { color: var(--warning); }

.badge {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
}

.badge.success { background: var(--success); color: #000; }
.badge.warning { background: var(--warning); color: #000; }
.badge.error { background: var(--error); color: #fff; }
.badge.neutral { background: var(--bg-tertiary); color: var(--text-secondary); }

.gpu-card {
  background: var(--bg-tertiary);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.75rem;
}

.gpu-card:last-child { margin-bottom: 0; }

.gpu-name {
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.gpu-details {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.label-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.label {
  background: var(--accent-dim);
  color: var(--accent);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-family: 'Consolas', monospace;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: var(--accent);
  color: #000;
}

.btn-primary:hover { filter: brightness(1.1); }

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border);
}

.btn-secondary:hover { border-color: var(--accent); }

.btn-danger {
  background: var(--error);
  color: #fff;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-group {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
}

.loading {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.log-area {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  font-family: 'Consolas', monospace;
  font-size: 0.75rem;
  max-height: 200px;
  overflow-y: auto;
}

.log-line {
  padding: 0.25rem 0;
  color: var(--text-muted);
}

.log-line.success { color: var(--success); }
.log-line.error { color: var(--error); }

@media (max-width: 900px) {
  .main {
    grid-template-columns: 1fr;
    padding: 1rem;
  }
}
</style>
</head>
<body>

<header class="header">
  <h1>S.L.A.T.E. Dashboard</h1>
  <span class="version" id="version">v2.4.0</span>
</header>

<main class="main">
  <!-- System Status Panel -->
  <div class="panel">
    <div class="panel-header">
      <h2>System Status</h2>
      <span id="status-indicator" class="badge neutral">Loading...</span>
    </div>
    <div class="panel-body">
      <div class="status-row">
        <span class="status-label">SDK</span>
        <span class="status-value" id="sdk-status">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">Runner</span>
        <span class="status-value" id="runner-status">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">GPU</span>
        <span class="status-value" id="gpu-status">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">Tasks</span>
        <span class="status-value" id="task-status">--</span>
      </div>
    </div>
  </div>

  <!-- Runner Control Panel -->
  <div class="panel">
    <div class="panel-header">
      <h2>Self-Hosted Runner</h2>
      <span id="runner-indicator" class="badge neutral">--</span>
    </div>
    <div class="panel-body">
      <div class="status-row">
        <span class="status-label">Installed</span>
        <span class="status-value" id="runner-installed">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">Configured</span>
        <span class="status-value" id="runner-configured">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">Provisioned</span>
        <span class="status-value" id="runner-provisioned">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">Running</span>
        <span class="status-value" id="runner-running">--</span>
      </div>
      <div class="btn-group">
        <button class="btn btn-primary" id="btn-start" onclick="startRunner()">Start Runner</button>
        <button class="btn btn-danger" id="btn-stop" onclick="stopRunner()">Stop</button>
        <button class="btn btn-secondary" id="btn-provision" onclick="provisionRunner()">Provision</button>
      </div>
    </div>
  </div>

  <!-- GPU Panel -->
  <div class="panel">
    <div class="panel-header">
      <h2>GPU Status</h2>
      <span id="gpu-count" class="badge neutral">--</span>
    </div>
    <div class="panel-body" id="gpu-list">
      <div class="gpu-card">
        <div class="gpu-name">Loading GPU info...</div>
      </div>
    </div>
  </div>

  <!-- GitHub Models Panel -->
  <div class="panel">
    <div class="panel-header">
      <h2>GitHub Models</h2>
      <span id="models-indicator" class="badge neutral">--</span>
    </div>
    <div class="panel-body">
      <div class="status-row">
        <span class="status-label">Token Set</span>
        <span class="status-value" id="models-token">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">SDK Installed</span>
        <span class="status-value" id="models-sdk">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">Connection</span>
        <span class="status-value" id="models-connection">--</span>
      </div>
      <div class="status-row">
        <span class="status-label">Models Available</span>
        <span class="status-value" id="models-count">--</span>
      </div>
      <div style="margin-top: 1rem;">
        <select id="model-select" class="btn btn-secondary" style="width: 100%; padding: 0.5rem;">
          <option value="gpt-4o-mini">gpt-4o-mini (OpenAI)</option>
          <option value="gpt-4o">gpt-4o (OpenAI)</option>
          <option value="mistral-large">mistral-large (Mistral AI)</option>
          <option value="meta-llama-3.1-70b-instruct">llama-3.1-70b (Meta)</option>
        </select>
      </div>
      <div class="btn-group">
        <button class="btn btn-primary" onclick="testGitHubModels()">Test Connection</button>
        <button class="btn btn-secondary" onclick="openModelsMarketplace()">View Models</button>
      </div>
    </div>
  </div>

  <!-- Labels Panel -->
  <div class="panel">
    <div class="panel-header">
      <h2>Runner Labels</h2>
    </div>
    <div class="panel-body">
      <div class="label-list" id="label-list">
        <span class="label">loading...</span>
      </div>
    </div>
  </div>

  <!-- Runner Log Panel -->
  <div class="panel" style="grid-column: 1 / -1;">
    <div class="panel-header">
      <h2>Runner Activity</h2>
      <button class="btn btn-secondary" onclick="clearLog()">Clear</button>
    </div>
    <div class="panel-body">
      <div class="log-area" id="runner-log">
        <div class="log-line">Waiting for runner events...</div>
      </div>
    </div>
  </div>
</main>

<script>
let eventSource = null;

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    // Update status indicator
    document.getElementById('status-indicator').className = 'badge success';
    document.getElementById('status-indicator').textContent = 'Online';

    // SDK status
    if (data.components?.sdk) {
      document.getElementById('sdk-status').textContent = data.components.sdk.version || 'OK';
      document.getElementById('sdk-status').className = 'status-value online';
    }

    // Runner status
    if (data.components?.runner) {
      const r = data.components.runner;
      document.getElementById('runner-status').textContent = r.running ? 'Running' : 'Stopped';
      document.getElementById('runner-status').className = 'status-value ' + (r.running ? 'online' : 'offline');
    }

    // GPU status
    if (data.components?.gpu) {
      const g = data.components.gpu;
      document.getElementById('gpu-status').textContent = g.has_gpu ? `${g.gpu_count} GPU(s)` : 'None';
      document.getElementById('gpu-status').className = 'status-value ' + (g.has_gpu ? 'online' : 'warning');
    }

    // Task status
    if (data.components?.tasks) {
      const t = data.components.tasks;
      document.getElementById('task-status').textContent = `${t.pending || 0} pending / ${t.total || 0} total`;
    }
  } catch (e) {
    document.getElementById('status-indicator').className = 'badge error';
    document.getElementById('status-indicator').textContent = 'Error';
  }
}

async function loadRunner() {
  try {
    const res = await fetch('/api/runner/status');
    const data = await res.json();

    // Update runner panel
    document.getElementById('runner-installed').textContent = data.installed ? 'Yes' : 'No';
    document.getElementById('runner-installed').className = 'status-value ' + (data.installed ? 'online' : 'offline');

    document.getElementById('runner-configured').textContent = data.configured ? 'Yes' : 'No';
    document.getElementById('runner-configured').className = 'status-value ' + (data.configured ? 'online' : 'offline');

    document.getElementById('runner-provisioned').textContent = data.provisioned ? 'Yes' : 'No';
    document.getElementById('runner-provisioned').className = 'status-value ' + (data.provisioned ? 'online' : 'warning');

    document.getElementById('runner-running').textContent = data.running ? 'Active' : 'Stopped';
    document.getElementById('runner-running').className = 'status-value ' + (data.running ? 'online' : 'offline');

    // Update indicator
    if (data.running) {
      document.getElementById('runner-indicator').className = 'badge success';
      document.getElementById('runner-indicator').textContent = 'Active';
    } else if (data.configured) {
      document.getElementById('runner-indicator').className = 'badge warning';
      document.getElementById('runner-indicator').textContent = 'Ready';
    } else {
      document.getElementById('runner-indicator').className = 'badge neutral';
      document.getElementById('runner-indicator').textContent = 'Not Configured';
    }

    // Update buttons
    document.getElementById('btn-start').disabled = data.running || !data.configured;
    document.getElementById('btn-stop').disabled = !data.running;
    document.getElementById('btn-provision').disabled = !data.configured;

    // Update labels
    if (data.labels) {
      document.getElementById('label-list').innerHTML = data.labels.map(l =>
        `<span class="label">${l}</span>`
      ).join('');
    }
  } catch (e) {
    console.error('Error loading runner status:', e);
  }
}

async function loadGPU() {
  try {
    const res = await fetch('/api/runner/gpu');
    const data = await res.json();

    document.getElementById('gpu-count').textContent = data.has_gpu ? `${data.gpu_count} GPU(s)` : 'No GPU';
    document.getElementById('gpu-count').className = 'badge ' + (data.has_gpu ? 'success' : 'neutral');

    if (data.has_gpu && data.gpu_names.length > 0) {
      document.getElementById('gpu-list').innerHTML = data.gpu_names.map((name, i) => `
        <div class="gpu-card">
          <div class="gpu-name">GPU ${i}: ${name}</div>
          <div class="gpu-details">CUDA Compute Capable</div>
        </div>
      `).join('');
    } else {
      document.getElementById('gpu-list').innerHTML = `
        <div class="gpu-card">
          <div class="gpu-name">No GPU Detected</div>
          <div class="gpu-details">Runner will use CPU only</div>
        </div>
      `;
    }
  } catch (e) {
    console.error('Error loading GPU info:', e);
  }
}

async function startRunner() {
  addLog('Starting runner...', 'info');
  document.getElementById('btn-start').disabled = true;
  try {
    const res = await fetch('/api/runner/start', { method: 'POST' });
    const data = await res.json();
    addLog(data.message || 'Start command sent', data.success ? 'success' : 'error');
    setTimeout(loadRunner, 2000);
  } catch (e) {
    addLog('Failed to start runner: ' + e.message, 'error');
  }
}

async function stopRunner() {
  addLog('Stopping runner...', 'info');
  document.getElementById('btn-stop').disabled = true;
  try {
    const res = await fetch('/api/runner/stop', { method: 'POST' });
    const data = await res.json();
    addLog(data.success ? 'Runner stopped' : 'Stop failed', data.success ? 'success' : 'error');
    setTimeout(loadRunner, 1000);
  } catch (e) {
    addLog('Failed to stop runner: ' + e.message, 'error');
  }
}

async function provisionRunner() {
  addLog('Provisioning SLATE environment...', 'info');
  document.getElementById('btn-provision').disabled = true;
  try {
    const res = await fetch('/api/runner/provision', { method: 'POST' });
    const data = await res.json();
    addLog(data.message || 'Provision started', 'success');
  } catch (e) {
    addLog('Failed to provision: ' + e.message, 'error');
  }
}

function addLog(message, type = 'info') {
  const log = document.getElementById('runner-log');
  const time = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = 'log-line ' + type;
  line.textContent = `[${time}] ${message}`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

function clearLog() {
  document.getElementById('runner-log').innerHTML = '<div class="log-line">Log cleared</div>';
}

function connectSSE() {
  if (eventSource) eventSource.close();

  eventSource = new EventSource('/api/runner/events');

  eventSource.addEventListener('init', (e) => {
    console.log('SSE connected');
  });

  eventSource.addEventListener('update', (e) => {
    const data = JSON.parse(e.data);
    if (data.event === 'runner_started') {
      addLog('Runner started successfully', 'success');
      loadRunner();
    } else if (data.event === 'runner_stopped') {
      addLog('Runner stopped', 'info');
      loadRunner();
    } else if (data.event === 'provision_complete') {
      addLog('Provisioning complete!', 'success');
      loadRunner();
    } else if (data.event === 'provision_error') {
      addLog('Provisioning failed', 'error');
      loadRunner();
    }
  });

  eventSource.onerror = () => {
    console.log('SSE reconnecting...');
    setTimeout(connectSSE, 5000);
  };
}

// Initialize
loadStatus();
loadRunner();
loadGPU();
connectSSE();

// Refresh every 30 seconds
setInterval(() => {
  loadStatus();
  loadRunner();
}, 30000);
</script>
</body>
</html>"""


@app.get("/")
async def dashboard():
    """Serve the dashboard HTML."""
    return HTMLResponse(DASHBOARD_HTML)


# ─── Static files (if available) ────────────────────────────────────────────────

static_dir = WORKSPACE_ROOT / "slate_web"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Main entry point ───────────────────────────────────────────────────────────

def main():
    """Start the dashboard server."""
    import argparse

    parser = argparse.ArgumentParser(description="SLATE Dashboard Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  S.L.A.T.E. Dashboard Server")
    print("=" * 60)
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  API: http://{args.host}:{args.port}/api/status")
    print("=" * 60)
    print()

    uvicorn.run(
        "agents.slate_dashboard_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
