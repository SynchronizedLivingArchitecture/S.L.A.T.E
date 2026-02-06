#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# CELL: slate_dashboard_server [python]
# Author: Claude | Created: 2026-02-06T23:45:00Z | Modified: 2026-02-07T00:00:00Z
# Purpose: SLATE Dashboard Server - Robust FastAPI server for agentic workflow management
# ═══════════════════════════════════════════════════════════════════════════════
"""
SLATE Dashboard Server
======================
Production-grade FastAPI server for SLATE system monitoring and agentic workflow management.

Features:
- Real-time WebSocket updates for live status
- GitHub Actions integration (workflows, runners, PRs)
- Task queue management with CRUD operations
- System metrics (GPU, CPU, memory)
- Glassmorphism UI with SLATE design system

Design System:
- Theme: Glassmorphism (75% opacity, blur effects)
- Colors: Muted pastels (slate blues, soft greens, warm grays)
- Fonts: System fonts (Segoe UI, Consolas)
- Security: 127.0.0.1 binding only

Endpoints:
    GET  /                    -> Dashboard UI
    GET  /health              -> Health check
    WS   /ws                  -> WebSocket for real-time updates

    GET  /api/status          -> Full system status
    GET  /api/orchestrator    -> Orchestrator status
    GET  /api/runner          -> GitHub runner status
    GET  /api/workflows       -> Recent workflow runs
    GET  /api/tasks           -> Task queue
    POST /api/tasks           -> Create task
    PUT  /api/tasks/{id}      -> Update task
    DELETE /api/tasks/{id}    -> Delete task
    POST /api/dispatch/{name} -> Dispatch workflow

Usage:
    python agents/slate_dashboard_server.py
    # Opens http://127.0.0.1:8080
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# ─── Dependencies ─────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("[!] Missing dependencies. Run: pip install fastapi uvicorn websockets")
    sys.exit(1)

# ─── App Configuration ────────────────────────────────────────────────────────

app = FastAPI(
    title="SLATE Dashboard",
    description="Agentic Workflow Management System",
    version="2.4.0"
)

# CORS for local development and VSCode webviews
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for VSCode webview compatibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Custom headers middleware for VSCode webview compatibility
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class VSCodeCompatMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        # Add headers for VSCode webview compatibility
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.add_middleware(VSCodeCompatMiddleware)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        with self._lock:
            connections = list(self.active_connections)
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_gh_cli() -> str:
    """Get GitHub CLI path."""
    gh_path = WORKSPACE_ROOT / ".tools" / "gh.exe"
    if gh_path.exists():
        return str(gh_path)
    return "gh"

def load_tasks() -> List[Dict[str, Any]]:
    """Load tasks from current_tasks.json."""
    task_file = WORKSPACE_ROOT / "current_tasks.json"
    if task_file.exists():
        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return data.get("tasks", [])
        except Exception:
            pass
    return []

def save_tasks(tasks: List[Dict[str, Any]]):
    """Save tasks to current_tasks.json."""
    task_file = WORKSPACE_ROOT / "current_tasks.json"
    task_file.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

# ─── Health & Status Endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "service": "slate-dashboard"}

@app.get("/api/status")
async def api_status():
    """Get comprehensive system status."""
    try:
        from slate.slate_status import get_status
        status = get_status()
        return JSONResponse(content=status)
    except Exception as e:
        return JSONResponse(content={"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()})

@app.get("/api/orchestrator")
async def api_orchestrator():
    """Get orchestrator status."""
    try:
        from slate.slate_orchestrator import SlateOrchestrator
        orch = SlateOrchestrator()
        return JSONResponse(content=orch.status())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/runner")
async def api_runner():
    """Get GitHub runner status with detailed info."""
    try:
        from slate.slate_runner_manager import SlateRunnerManager
        mgr = SlateRunnerManager()
        detection = mgr.detect()

        # Add GitHub API runner status
        try:
            gh_cli = get_gh_cli()
            result = subprocess.run(
                [gh_cli, "api", "repos/SynchronizedLivingArchitecture/S.L.A.T.E/actions/runners",
                 "--jq", ".runners[0]"],
                capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE_ROOT)
            )
            if result.returncode == 0:
                detection["github_runner"] = json.loads(result.stdout)
        except Exception:
            pass

        return JSONResponse(content=detection)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/workflows")
async def api_workflows():
    """Get recent GitHub workflow runs."""
    try:
        gh_cli = get_gh_cli()
        result = subprocess.run(
            [gh_cli, "run", "list", "--limit", "15", "--json",
             "name,status,conclusion,createdAt,updatedAt,databaseId,headBranch,event"],
            capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE_ROOT)
        )
        if result.returncode == 0:
            runs = json.loads(result.stdout)
            return JSONResponse(content={"runs": runs, "count": len(runs)})
        return JSONResponse(content={"error": result.stderr, "runs": []})
    except Exception as e:
        return JSONResponse(content={"error": str(e), "runs": []})

@app.get("/api/workflow/{run_id}")
async def api_workflow_detail(run_id: int):
    """Get detailed workflow run info."""
    try:
        gh_cli = get_gh_cli()
        result = subprocess.run(
            [gh_cli, "run", "view", str(run_id), "--json",
             "name,status,conclusion,jobs,createdAt,updatedAt"],
            capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE_ROOT)
        )
        if result.returncode == 0:
            return JSONResponse(content=json.loads(result.stdout))
        return JSONResponse(content={"error": result.stderr}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ─── Task Management Endpoints ────────────────────────────────────────────────

@app.get("/api/tasks")
async def api_tasks():
    """Get all tasks."""
    tasks = load_tasks()

    # Calculate stats
    stats = {"total": len(tasks), "pending": 0, "in_progress": 0, "completed": 0}
    for t in tasks:
        status = t.get("status", "pending")
        if status in stats:
            stats[status] += 1

    return JSONResponse(content={"tasks": tasks, "stats": stats})

@app.post("/api/tasks")
async def create_task(request: Request):
    """Create a new task."""
    try:
        data = await request.json()
        tasks = load_tasks()

        new_task = {
            "id": str(uuid.uuid4())[:8],
            "title": data.get("title", "Untitled Task"),
            "description": data.get("description", ""),
            "status": "pending",
            "priority": data.get("priority", 3),
            "assigned_to": data.get("assigned_to", "auto"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "dashboard"
        }

        tasks.append(new_task)
        save_tasks(tasks)

        # Broadcast update
        await manager.broadcast({"type": "task_created", "task": new_task})

        return JSONResponse(content=new_task, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/tasks/{task_id}")
async def update_task(task_id: str, request: Request):
    """Update a task."""
    try:
        data = await request.json()
        tasks = load_tasks()

        for task in tasks:
            if task.get("id") == task_id:
                task.update(data)
                task["updated_at"] = datetime.now(timezone.utc).isoformat()
                save_tasks(tasks)
                await manager.broadcast({"type": "task_updated", "task": task})
                return JSONResponse(content=task)

        raise HTTPException(status_code=404, detail="Task not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    tasks = load_tasks()
    original_len = len(tasks)
    tasks = [t for t in tasks if t.get("id") != task_id]

    if len(tasks) == original_len:
        raise HTTPException(status_code=404, detail="Task not found")

    save_tasks(tasks)
    await manager.broadcast({"type": "task_deleted", "task_id": task_id})
    return JSONResponse(content={"deleted": task_id})

# ─── Workflow Dispatch ────────────────────────────────────────────────────────

@app.post("/api/dispatch/{workflow_name}")
async def dispatch_workflow(workflow_name: str, request: Request):
    """Dispatch a GitHub workflow."""
    try:
        gh_cli = get_gh_cli()

        # Get optional inputs
        try:
            inputs = await request.json()
        except Exception:
            inputs = {}

        cmd = [gh_cli, "workflow", "run", workflow_name]
        for key, value in inputs.items():
            cmd.extend(["-f", f"{key}={value}"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(WORKSPACE_ROOT))

        if result.returncode == 0:
            await manager.broadcast({"type": "workflow_dispatched", "workflow": workflow_name})
            return JSONResponse(content={"success": True, "workflow": workflow_name})
        return JSONResponse(content={"success": False, "error": result.stderr}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await manager.connect(websocket)
    try:
        # Send initial status
        try:
            from slate.slate_orchestrator import SlateOrchestrator
            orch = SlateOrchestrator()
            await websocket.send_json({"type": "status", "data": orch.status()})
        except Exception:
            pass

        while True:
            # Keep connection alive and handle client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                msg = json.loads(data)

                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("type") == "refresh":
                    try:
                        from slate.slate_orchestrator import SlateOrchestrator
                        orch = SlateOrchestrator()
                        await websocket.send_json({"type": "status", "data": orch.status()})
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                # Send periodic status update
                try:
                    from slate.slate_orchestrator import SlateOrchestrator
                    orch = SlateOrchestrator()
                    await websocket.send_json({"type": "status", "data": orch.status()})
                except Exception:
                    await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ─── Dashboard HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'self' http://127.0.0.1:* ws://127.0.0.1:*; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:*; img-src 'self' data:;">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>S.L.A.T.E. Dashboard</title>
    <style>
        :root {
            --bg-dark: #0f1419;
            --bg-card: rgba(30, 41, 59, 0.75);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border: rgba(148, 163, 184, 0.1);
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #60a5fa;
            --accent-green: #4ade80;
            --accent-yellow: #fbbf24;
            --accent-red: #f87171;
            --accent-purple: #a78bfa;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            background-image:
                radial-gradient(ellipse at 20% 30%, rgba(96, 165, 250, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 70%, rgba(167, 139, 250, 0.06) 0%, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            font-family: Consolas, monospace;
        }

        .logo-text h1 {
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .logo-text span {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .header-status {
            display: flex;
            align-items: center;
            gap: 24px;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.875rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.online { background: var(--accent-green); }
        .status-dot.offline { background: var(--accent-red); }
        .status-dot.pending { background: var(--accent-yellow); }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Grid Layout */
        .grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 20px;
        }

        .col-3 { grid-column: span 3; }
        .col-4 { grid-column: span 4; }
        .col-6 { grid-column: span 6; }
        .col-8 { grid-column: span 8; }
        .col-12 { grid-column: span 12; }

        @media (max-width: 1024px) {
            .col-3, .col-4, .col-6, .col-8 { grid-column: span 12; }
        }

        /* Cards */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.2s ease;
        }

        .card:hover {
            background: var(--bg-card-hover);
            border-color: rgba(148, 163, 184, 0.2);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .card-action {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .card-action:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }

        /* Stat Cards */
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            font-family: Consolas, monospace;
            margin-bottom: 4px;
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        .stat-value.green { color: var(--accent-green); }
        .stat-value.blue { color: var(--accent-blue); }
        .stat-value.yellow { color: var(--accent-yellow); }
        .stat-value.red { color: var(--accent-red); }

        /* Service Status */
        .service-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .service-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }

        .service-name {
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 500;
        }

        .service-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            background: rgba(255, 255, 255, 0.05);
        }

        .badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge.online { background: rgba(74, 222, 128, 0.15); color: var(--accent-green); }
        .badge.offline { background: rgba(248, 113, 113, 0.15); color: var(--accent-red); }
        .badge.pending { background: rgba(251, 191, 36, 0.15); color: var(--accent-yellow); }
        .badge.busy { background: rgba(96, 165, 250, 0.15); color: var(--accent-blue); }

        /* Task List */
        .task-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 400px;
            overflow-y: auto;
        }

        .task-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            border-left: 3px solid transparent;
        }

        .task-item.pending { border-left-color: var(--accent-yellow); }
        .task-item.in-progress { border-left-color: var(--accent-blue); }
        .task-item.completed { border-left-color: var(--accent-green); }

        .task-content { flex: 1; }

        .task-title {
            font-weight: 500;
            margin-bottom: 2px;
        }

        .task-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* Workflow List */
        .workflow-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            margin-bottom: 8px;
        }

        .workflow-name {
            font-weight: 500;
            font-size: 0.875rem;
        }

        .workflow-time {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* GPU Info */
        .gpu-card {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin-bottom: 8px;
        }

        .gpu-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #76b900, #4a7c00);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .gpu-info h4 {
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 2px;
        }

        .gpu-info span {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }

        .btn-primary {
            background: var(--accent-blue);
            color: white;
        }

        .btn-primary:hover {
            background: #3b82f6;
        }

        .btn-ghost {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
        }

        .btn-ghost:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
        }

        /* Connection Status */
        #connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            z-index: 1000;
        }

        #connection-status.connected {
            background: rgba(74, 222, 128, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(74, 222, 128, 0.3);
        }

        #connection-status.disconnected {
            background: rgba(248, 113, 113, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(248, 113, 113, 0.3);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="logo">
                <div class="logo-icon">S</div>
                <div class="logo-text">
                    <h1>S.L.A.T.E.</h1>
                    <span>Synchronized Living Architecture</span>
                </div>
            </div>
            <div class="header-status">
                <div class="status-indicator">
                    <span class="status-dot" id="runner-dot"></span>
                    <span id="runner-status-text">Runner: Checking...</span>
                </div>
                <button class="btn btn-ghost" onclick="refreshAll()">Refresh</button>
            </div>
        </header>

        <!-- Main Grid -->
        <div class="grid">
            <!-- Stats Row -->
            <div class="card col-3">
                <div class="stat-value green" id="stat-online">-</div>
                <div class="stat-label">Services Online</div>
            </div>
            <div class="card col-3">
                <div class="stat-value blue" id="stat-tasks">-</div>
                <div class="stat-label">Active Tasks</div>
            </div>
            <div class="card col-3">
                <div class="stat-value yellow" id="stat-pending">-</div>
                <div class="stat-label">Pending</div>
            </div>
            <div class="card col-3">
                <div class="stat-value" id="stat-workflows">-</div>
                <div class="stat-label">Workflows Today</div>
            </div>

            <!-- Services & GPU -->
            <div class="card col-4">
                <div class="card-header">
                    <span class="card-title">Services</span>
                </div>
                <div class="service-list" id="services-list">
                    <div class="service-item">
                        <div class="service-name">
                            <div class="service-icon">O</div>
                            <span>Orchestrator</span>
                        </div>
                        <span class="badge pending" id="orch-badge">Checking</span>
                    </div>
                    <div class="service-item">
                        <div class="service-name">
                            <div class="service-icon">R</div>
                            <span>GitHub Runner</span>
                        </div>
                        <span class="badge pending" id="runner-badge">Checking</span>
                    </div>
                    <div class="service-item">
                        <div class="service-name">
                            <div class="service-icon">D</div>
                            <span>Dashboard</span>
                        </div>
                        <span class="badge online">Online</span>
                    </div>
                </div>
            </div>

            <!-- GPU Status -->
            <div class="card col-4">
                <div class="card-header">
                    <span class="card-title">GPU Configuration</span>
                </div>
                <div id="gpu-list">
                    <div class="empty-state">Loading GPU info...</div>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="card col-4">
                <div class="card-header">
                    <span class="card-title">Quick Actions</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <button class="btn btn-primary" onclick="dispatchWorkflow('ci.yml')">Run CI Pipeline</button>
                    <button class="btn btn-ghost" onclick="dispatchWorkflow('slate.yml')">Run SLATE Checks</button>
                    <button class="btn btn-ghost" onclick="dispatchWorkflow('nightly.yml')">Run Nightly Suite</button>
                </div>
            </div>

            <!-- Task Queue -->
            <div class="card col-8">
                <div class="card-header">
                    <span class="card-title">Task Queue</span>
                    <button class="card-action" onclick="refreshTasks()">Refresh</button>
                </div>
                <div class="task-list" id="task-list">
                    <div class="empty-state">Loading tasks...</div>
                </div>
            </div>

            <!-- Recent Workflows -->
            <div class="card col-4">
                <div class="card-header">
                    <span class="card-title">Recent Workflows</span>
                </div>
                <div id="workflow-list">
                    <div class="empty-state">Loading workflows...</div>
                </div>
            </div>
        </div>
    </div>

    <div id="connection-status" class="disconnected">Connecting...</div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;

        // WebSocket Connection
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connection-status').className = 'connected';
                document.getElementById('connection-status').textContent = 'Live';
                reconnectAttempts = 0;
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'status') {
                    updateStatus(msg.data);
                } else if (msg.type === 'task_created' || msg.type === 'task_updated' || msg.type === 'task_deleted') {
                    refreshTasks();
                } else if (msg.type === 'workflow_dispatched') {
                    refreshWorkflows();
                }
            };

            ws.onclose = () => {
                document.getElementById('connection-status').className = 'disconnected';
                document.getElementById('connection-status').textContent = 'Reconnecting...';
                reconnectAttempts++;
                setTimeout(connectWebSocket, Math.min(1000 * reconnectAttempts, 10000));
            };

            ws.onerror = () => ws.close();
        }

        // Update Functions
        function updateStatus(data) {
            // Orchestrator
            const orchBadge = document.getElementById('orch-badge');
            if (data.orchestrator?.running) {
                orchBadge.className = 'badge online';
                orchBadge.textContent = 'Running';
            } else {
                orchBadge.className = 'badge offline';
                orchBadge.textContent = 'Stopped';
            }

            // Runner
            const runnerBadge = document.getElementById('runner-badge');
            const runnerDot = document.getElementById('runner-dot');
            const runnerText = document.getElementById('runner-status-text');

            if (data.runner?.running) {
                runnerBadge.className = data.runner.busy ? 'badge busy' : 'badge online';
                runnerBadge.textContent = data.runner.busy ? 'Busy' : 'Online';
                runnerDot.className = 'status-dot online';
                runnerText.textContent = 'Runner: Online';
            } else {
                runnerBadge.className = 'badge offline';
                runnerBadge.textContent = data.runner?.status || 'Offline';
                runnerDot.className = 'status-dot offline';
                runnerText.textContent = 'Runner: Offline';
            }

            // Stats
            let online = 1; // Dashboard always online
            if (data.orchestrator?.running) online++;
            if (data.runner?.running) online++;
            document.getElementById('stat-online').textContent = online + '/3';
            document.getElementById('stat-tasks').textContent = data.workflow?.task_count || 0;
            document.getElementById('stat-pending').textContent = data.workflow?.in_progress || 0;
        }

        async function refreshTasks() {
            try {
                const res = await fetch('/api/tasks');
                const data = await res.json();
                const list = document.getElementById('task-list');

                if (data.tasks && data.tasks.length > 0) {
                    list.innerHTML = data.tasks.slice(0, 10).map(t => `
                        <div class="task-item ${t.status || 'pending'}">
                            <div class="task-content">
                                <div class="task-title">${t.title || t.name || 'Untitled'}</div>
                                <div class="task-meta">${t.assigned_to || 'auto'} | ${t.status || 'pending'}</div>
                            </div>
                            <span class="badge ${t.status === 'completed' ? 'online' : t.status === 'in-progress' ? 'busy' : 'pending'}">
                                ${t.status || 'pending'}
                            </span>
                        </div>
                    `).join('');
                } else {
                    list.innerHTML = '<div class="empty-state">No tasks in queue</div>';
                }

                document.getElementById('stat-tasks').textContent = data.stats?.total || 0;
                document.getElementById('stat-pending').textContent = data.stats?.in_progress || 0;
            } catch (e) {
                console.error('Failed to fetch tasks:', e);
            }
        }

        async function refreshWorkflows() {
            try {
                const res = await fetch('/api/workflows');
                const data = await res.json();
                const list = document.getElementById('workflow-list');

                if (data.runs && data.runs.length > 0) {
                    list.innerHTML = data.runs.slice(0, 8).map(w => {
                        const status = w.conclusion || w.status;
                        const badgeClass = status === 'success' ? 'online' :
                                          status === 'in_progress' ? 'busy' :
                                          status === 'failure' ? 'offline' : 'pending';
                        const time = new Date(w.createdAt).toLocaleTimeString();
                        return `
                            <div class="workflow-item">
                                <div>
                                    <div class="workflow-name">${w.name}</div>
                                    <div class="workflow-time">${time}</div>
                                </div>
                                <span class="badge ${badgeClass}">${status}</span>
                            </div>
                        `;
                    }).join('');

                    document.getElementById('stat-workflows').textContent = data.runs.length;
                } else {
                    list.innerHTML = '<div class="empty-state">No recent workflows</div>';
                }
            } catch (e) {
                console.error('Failed to fetch workflows:', e);
            }
        }

        async function refreshRunner() {
            try {
                const res = await fetch('/api/runner');
                const data = await res.json();
                const list = document.getElementById('gpu-list');

                if (data.gpu_info?.gpus && data.gpu_info.gpus.length > 0) {
                    list.innerHTML = data.gpu_info.gpus.map(gpu => `
                        <div class="gpu-card">
                            <div class="gpu-icon">G</div>
                            <div class="gpu-info">
                                <h4>${gpu.name}</h4>
                                <span>${gpu.architecture} | ${gpu.memory}</span>
                            </div>
                        </div>
                    `).join('');
                } else {
                    list.innerHTML = '<div class="empty-state">No GPUs detected</div>';
                }
            } catch (e) {
                console.error('Failed to fetch runner:', e);
            }
        }

        async function dispatchWorkflow(name) {
            try {
                const res = await fetch(`/api/dispatch/${name}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    alert(`Workflow ${name} dispatched successfully!`);
                    setTimeout(refreshWorkflows, 2000);
                } else {
                    alert(`Failed to dispatch: ${data.error}`);
                }
            } catch (e) {
                alert(`Error: ${e.message}`);
            }
        }

        function refreshAll() {
            refreshTasks();
            refreshWorkflows();
            refreshRunner();
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'refresh' }));
            }
        }

        // Initial status fetch (works even without WebSocket)
        async function fetchInitialStatus() {
            try {
                const res = await fetch('/api/orchestrator');
                const data = await res.json();
                updateStatus(data);
            } catch (e) {
                console.log('Initial status fetch failed, will retry');
            }
        }

        // Initialize - fetch data immediately, then connect WebSocket
        fetchInitialStatus();
        refreshAll();

        // Try WebSocket, but polling will keep working regardless
        try {
            connectWebSocket();
        } catch (e) {
            console.log('WebSocket failed, using polling only');
        }

        // More frequent polling for VSCode webview compatibility (every 10s)
        setInterval(refreshAll, 10000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the SLATE dashboard."""
    return DASHBOARD_HTML

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run the dashboard server."""
    print()
    print("=" * 60)
    print("  S.L.A.T.E. Dashboard Server")
    print("=" * 60)
    print()
    print("  URL:      http://127.0.0.1:8080")
    print("  WebSocket: ws://127.0.0.1:8080/ws")
    print()
    print("  Press Ctrl+C to stop")
    print()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
        log_level="warning",
        access_log=False
    )


if __name__ == "__main__":
    main()
