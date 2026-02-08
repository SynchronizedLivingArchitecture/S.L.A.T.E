#!/usr/bin/env python3
# Modified: 2026-02-08T07:20:00Z | Author: COPILOT | Change: Create SLATE K8s local deployment manager
"""
SLATE K8s Manager — Local Container Orchestration
===================================================
Manages SLATE deployment on local Kubernetes (Docker Desktop, minikube, k3s).

Commands:
    python slate/slate_k8s.py --status     # Show K8s deployment status
    python slate/slate_k8s.py --deploy     # Deploy SLATE to local K8s
    python slate/slate_k8s.py --teardown   # Remove SLATE from K8s
    python slate/slate_k8s.py --dashboard  # Port-forward dashboard to localhost:8080
    python slate/slate_k8s.py --logs       # Tail logs from all SLATE pods
    python slate/slate_k8s.py --detect     # Detect local K8s provider
    python slate/slate_k8s.py --build      # Build local Docker images for K8s
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

WORKSPACE = Path(__file__).parent.parent
K8S_DIR = WORKSPACE / "k8s"
K8S_LOCAL = K8S_DIR / "overlays" / "local"
HELM_DIR = WORKSPACE / "helm" / "slate"
NAMESPACE = "slate"


def run(cmd: list[str], check: bool = True, capture: bool = False, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture, text=True, timeout=timeout,
            encoding="utf-8", cwd=str(WORKSPACE),
        )
        if check and result.returncode != 0:
            stderr = result.stderr if capture else ""
            print(f"  [!] Command failed: {' '.join(cmd)}")
            if stderr:
                print(f"      {stderr.strip()}")
        return result
    except FileNotFoundError:
        print(f"  [!] Command not found: {cmd[0]}")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=f"{cmd[0]} not found")
    except subprocess.TimeoutExpired:
        print(f"  [!] Command timed out: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="timeout")


def detect_provider() -> dict:
    """Detect which local K8s provider is available."""
    providers = {}

    # Check kubectl
    kubectl = run(["kubectl", "version", "--client", "--output=json"], check=False, capture=True)
    if kubectl.returncode == 0:
        try:
            ver = json.loads(kubectl.stdout)
            providers["kubectl"] = ver.get("clientVersion", {}).get("gitVersion", "unknown")
        except json.JSONDecodeError:
            providers["kubectl"] = "available"
    else:
        print("  [!] kubectl not found — install it first")
        return {"available": False}

    # Check cluster connectivity
    cluster = run(["kubectl", "cluster-info", "--request-timeout=5s"], check=False, capture=True)
    if cluster.returncode == 0:
        providers["cluster_connected"] = True
        # Detect provider from context
        ctx = run(["kubectl", "config", "current-context"], check=False, capture=True)
        if ctx.returncode == 0:
            context = ctx.stdout.strip()
            providers["context"] = context
            if "docker-desktop" in context.lower():
                providers["provider"] = "Docker Desktop"
            elif "minikube" in context.lower():
                providers["provider"] = "minikube"
            elif "k3s" in context.lower() or "k3d" in context.lower():
                providers["provider"] = "k3s/k3d"
            elif "rancher" in context.lower():
                providers["provider"] = "Rancher Desktop"
            else:
                providers["provider"] = "Unknown"
    else:
        providers["cluster_connected"] = False
        providers["provider"] = "None (cluster not running)"

    # Check Helm
    helm = run(["helm", "version", "--short"], check=False, capture=True)
    if helm.returncode == 0:
        providers["helm"] = helm.stdout.strip()
    else:
        providers["helm"] = None

    # Check Docker
    docker = run(["docker", "version", "--format", "{{.Server.Version}}"], check=False, capture=True)
    if docker.returncode == 0:
        providers["docker"] = docker.stdout.strip()
    else:
        providers["docker"] = None

    # Check NVIDIA runtime
    nvidia = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=False, capture=True)
    if nvidia.returncode == 0:
        gpus = [g.strip() for g in nvidia.stdout.strip().split("\n") if g.strip()]
        providers["gpus"] = gpus
        providers["gpu_available"] = True
    else:
        providers["gpus"] = []
        providers["gpu_available"] = False

    providers["available"] = True
    return providers


def build_images():
    """Build SLATE Docker images for local K8s."""
    print("\n  Building SLATE images for local K8s...")
    print("  ─────────────────────────────────────")

    # Build GPU image
    print("\n  [1/3] Building slate:local (GPU)...")
    r = run(["docker", "build", "-t", "slate:local", "-f", "Dockerfile", "."], check=False)
    if r.returncode == 0:
        print("    ✓ slate:local built")
    else:
        print("    ✗ GPU image build failed, trying CPU...")

    # Build CPU image
    print("\n  [2/3] Building slate:local-cpu...")
    r = run(["docker", "build", "-t", "slate:local-cpu", "-f", "Dockerfile.cpu", "."], check=False)
    if r.returncode == 0:
        print("    ✓ slate:local-cpu built")
    else:
        print("    ✗ CPU image build failed")

    # Build dev image
    dev_dockerfile = WORKSPACE / "Dockerfile.dev"
    if dev_dockerfile.exists():
        print("\n  [3/3] Building slate:local-dev...")
        r = run(["docker", "build", "-t", "slate:local-dev", "-f", "Dockerfile.dev", "."], check=False)
        if r.returncode == 0:
            print("    ✓ slate:local-dev built")
        else:
            print("    ✗ Dev image build failed")
    else:
        print("\n  [3/3] No Dockerfile.dev found, skipping dev image")

    print("\n  ✓ Image build complete")


def deploy(method: str = "kustomize"):
    """Deploy SLATE to local K8s."""
    provider = detect_provider()
    if not provider.get("cluster_connected"):
        print("  [!] No K8s cluster connected. Start Docker Desktop K8s or minikube first.")
        return

    print(f"\n  Deploying SLATE to {provider.get('provider', 'local K8s')}...")
    print(f"  Method: {method}")
    print("  ─────────────────────────────────────")

    if method == "helm":
        if not provider.get("helm"):
            print("  [!] Helm not found. Install Helm or use --method kustomize")
            return
        print("\n  Installing SLATE via Helm...")
        r = run(["helm", "upgrade", "--install", "slate", str(HELM_DIR),
                 "--namespace", NAMESPACE, "--create-namespace"], check=False)
        if r.returncode == 0:
            print("  ✓ Helm install complete")
        else:
            print("  ✗ Helm install failed")
            return
    else:
        # Kustomize
        if K8S_LOCAL.exists():
            print("\n  Applying local Kustomize overlay...")
            r = run(["kubectl", "apply", "-k", str(K8S_LOCAL)], check=False)
        else:
            print("\n  Applying base K8s manifests...")
            r = run(["kubectl", "apply", "-k", str(K8S_DIR)], check=False)

        if r.returncode == 0:
            print("  ✓ Manifests applied")
        else:
            print("  ✗ Apply failed")
            return

    # Wait for pods
    print("\n  Waiting for pods to be ready...")
    for i in range(30):
        result = run(["kubectl", "get", "pods", "-n", NAMESPACE,
                      "-o", "jsonpath={.items[*].status.phase}"], check=False, capture=True)
        if result.returncode == 0:
            phases = result.stdout.strip().split()
            if phases and all(p == "Running" for p in phases):
                print(f"  ✓ All {len(phases)} pods running")
                break
        time.sleep(2)
        if i % 5 == 0 and i > 0:
            print(f"    ... waiting ({i * 2}s)")
    else:
        print("  ⚠ Pods not all ready after 60s — check with: kubectl get pods -n slate")

    print("\n  ✓ SLATE deployed to local K8s")
    print("  ─────────────────────────────────────")
    print(f"  Dashboard: kubectl port-forward -n {NAMESPACE} svc/slate-dashboard-svc 8080:8080")
    print(f"  Ollama:    kubectl port-forward -n {NAMESPACE} svc/ollama-svc 11434:11434")
    print(f"  ChromaDB:  kubectl port-forward -n {NAMESPACE} svc/chromadb-svc 8000:8000")


def teardown(method: str = "kustomize"):
    """Remove SLATE from K8s."""
    print("\n  Tearing down SLATE from K8s...")
    print("  ─────────────────────────────────────")

    if method == "helm":
        run(["helm", "uninstall", "slate", "--namespace", NAMESPACE], check=False)
    else:
        if K8S_LOCAL.exists():
            run(["kubectl", "delete", "-k", str(K8S_LOCAL), "--ignore-not-found"], check=False)
        else:
            run(["kubectl", "delete", "-k", str(K8S_DIR), "--ignore-not-found"], check=False)

    print("  ✓ SLATE removed from K8s")
    print("  Note: PersistentVolumes are retained (Retain policy). Delete manually if needed.")


def port_forward(service: str = "dashboard"):
    """Port-forward a SLATE service."""
    services = {
        "dashboard": ("slate-dashboard-svc", 8080),
        "ollama": ("ollama-svc", 11434),
        "chromadb": ("chromadb-svc", 8000),
    }
    svc_name, port = services.get(service, ("slate-dashboard-svc", 8080))
    print(f"\n  Port-forwarding {svc_name} → 127.0.0.1:{port}")
    print(f"  Access: http://127.0.0.1:{port}")
    print("  Press Ctrl+C to stop\n")
    try:
        subprocess.run(
            ["kubectl", "port-forward", "-n", NAMESPACE,
             f"svc/{svc_name}", f"{port}:{port}", "--address=127.0.0.1"],
            cwd=str(WORKSPACE),
        )
    except KeyboardInterrupt:
        print("\n  Port-forward stopped")


def show_logs(follow: bool = False, lines: int = 50):
    """Show logs from SLATE pods."""
    cmd = ["kubectl", "logs", "-n", NAMESPACE, "-l", "app.kubernetes.io/part-of=slate-system",
           "--all-containers", f"--tail={lines}"]
    if follow:
        cmd.append("-f")
    run(cmd, check=False)


def show_status():
    """Show SLATE K8s deployment status."""
    provider = detect_provider()

    print()
    print("=" * 60)
    print("  SLATE K8s — Local Deployment Status")
    print("=" * 60)

    # Provider
    print(f"\n  Provider:   {provider.get('provider', 'N/A')}")
    print(f"  Context:    {provider.get('context', 'N/A')}")
    print(f"  kubectl:    {provider.get('kubectl', 'N/A')}")
    print(f"  Helm:       {provider.get('helm', 'not installed')}")
    print(f"  Docker:     {provider.get('docker', 'not installed')}")
    print(f"  GPUs:       {', '.join(provider.get('gpus', [])) or 'none detected'}")

    if not provider.get("cluster_connected"):
        print("\n  ✗ No K8s cluster connected")
        print("    Start Docker Desktop K8s, minikube, or k3s")
        print()
        print("=" * 60)
        return

    # Namespace check
    ns_check = run(["kubectl", "get", "namespace", NAMESPACE], check=False, capture=True)
    if ns_check.returncode != 0:
        print(f"\n  ✗ Namespace '{NAMESPACE}' does not exist — SLATE not deployed")
        print(f"    Deploy with: python slate/slate_k8s.py --deploy")
        print()
        print("=" * 60)
        return

    # Pods
    print(f"\n  Pods ({NAMESPACE}):")
    run(["kubectl", "get", "pods", "-n", NAMESPACE, "-o", "wide"], check=False)

    # Services
    print(f"\n  Services ({NAMESPACE}):")
    run(["kubectl", "get", "svc", "-n", NAMESPACE], check=False)

    # PVCs
    print(f"\n  Storage ({NAMESPACE}):")
    run(["kubectl", "get", "pvc", "-n", NAMESPACE], check=False)

    # Resource usage
    print(f"\n  Resource Usage:")
    run(["kubectl", "top", "pods", "-n", NAMESPACE], check=False, capture=False)

    print()
    print("=" * 60)
    print(f"  Dashboard: kubectl port-forward -n {NAMESPACE} svc/slate-dashboard-svc 8080:8080")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SLATE K8s Manager — Local Container Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--deploy", action="store_true", help="Deploy SLATE to local K8s")
    parser.add_argument("--teardown", action="store_true", help="Remove SLATE from K8s")
    parser.add_argument("--dashboard", action="store_true", help="Port-forward dashboard")
    parser.add_argument("--logs", action="store_true", help="Show pod logs")
    parser.add_argument("--detect", action="store_true", help="Detect local K8s provider")
    parser.add_argument("--build", action="store_true", help="Build local Docker images")
    parser.add_argument("--method", choices=["kustomize", "helm"], default="kustomize",
                        help="Deployment method (default: kustomize)")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    parser.add_argument("--lines", type=int, default=50, help="Number of log lines")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if args.detect:
        info = detect_provider()
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"\n  K8s Provider: {info.get('provider', 'N/A')}")
            print(f"  Context:      {info.get('context', 'N/A')}")
            print(f"  kubectl:      {info.get('kubectl', 'N/A')}")
            print(f"  Helm:         {info.get('helm', 'not installed')}")
            print(f"  Docker:       {info.get('docker', 'not installed')}")
            print(f"  GPUs:         {', '.join(info.get('gpus', [])) or 'none'}")
    elif args.build:
        build_images()
    elif args.deploy:
        deploy(method=args.method)
    elif args.teardown:
        teardown(method=args.method)
    elif args.dashboard:
        port_forward("dashboard")
    elif args.logs:
        show_logs(follow=args.follow, lines=args.lines)
    else:
        show_status()


if __name__ == "__main__":
    main()
