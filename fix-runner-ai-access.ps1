# Modified: 2026-02-07T07:10:00Z | Author: COPILOT | Change: Create PowerShell fix script for runner AI access
# fix-runner-ai-access.ps1
# SLATE Runner AI Access Fix Script
# Ensures the self-hosted runner can access local AI services (Ollama, GPUs)
#
# Usage:
#   .\fix-runner-ai-access.ps1           # Full fix + verify
#   .\fix-runner-ai-access.ps1 -Verify   # Verify only (no changes)

param(
    [switch]$Verify
)

$ErrorActionPreference = 'Continue'
$WORKSPACE = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RUNNER_DIR = Join-Path $WORKSPACE 'actions-runner'
$HOOKS_DIR = Join-Path $RUNNER_DIR 'hooks'
$PRE_JOB = Join-Path $HOOKS_DIR 'pre-job.ps1'
$OLLAMA_URL = 'http://127.0.0.1:11434'

Write-Host '============================================================'
Write-Host '  SLATE Runner AI Access Fix'
Write-Host "  $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')"
Write-Host '============================================================'
Write-Host ''

# ---- Step 1: Check Ollama ----
Write-Host '[1/6] Checking Ollama service...'
$ollamaOk = $false
try {
    $response = Invoke-RestMethod -Uri "$OLLAMA_URL/api/tags" -TimeoutSec 5
    $modelCount = $response.models.Count
    Write-Host "  OK: Ollama running with $modelCount models"
    $ollamaOk = $true
} catch {
    Write-Host '  FAIL: Ollama not responding'
    if (-not $Verify) {
        Write-Host '  Attempting to start Ollama...'
        try {
            $ollamaPath = (Get-Command ollama -ErrorAction SilentlyContinue).Source
            if ($ollamaPath) {
                Start-Process -FilePath $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden
                Start-Sleep -Seconds 5
                try {
                    $response = Invoke-RestMethod -Uri "$OLLAMA_URL/api/tags" -TimeoutSec 5
                    Write-Host "  FIXED: Ollama started successfully ($($response.models.Count) models)"
                    $ollamaOk = $true
                } catch {
                    Write-Host '  WARN: Ollama started but not yet responding'
                }
            } else {
                Write-Host '  ERROR: ollama.exe not found in PATH'
            }
        } catch {
            Write-Host "  ERROR: Failed to start Ollama - $_"
        }
    }
}

# ---- Step 2: Check/Create .env ----
Write-Host ''
Write-Host '[2/6] Checking runner .env file...'
$envFile = Join-Path $RUNNER_DIR '.env'
$envContent = @"
OLLAMA_HOST=127.0.0.1:11434
SLATE_OLLAMA_URL=http://127.0.0.1:11434
CUDA_VISIBLE_DEVICES=0,1
SLATE_GPU_COUNT=2
SLATE_WORKSPACE=$WORKSPACE
PYTHONPATH=$WORKSPACE
PYTHONIOENCODING=utf-8
SLATE_RUNNER=true
"@

if (Test-Path $envFile) {
    $existing = Get-Content $envFile -Raw
    if ($existing -match 'OLLAMA_HOST') {
        Write-Host '  OK: .env file has OLLAMA_HOST'
    } else {
        if (-not $Verify) {
            Set-Content -Path $envFile -Value $envContent -Encoding UTF8
            Write-Host '  FIXED: Updated .env with OLLAMA_HOST'
        } else {
            Write-Host '  WARN: .env missing OLLAMA_HOST'
        }
    }
} else {
    if (-not $Verify) {
        Set-Content -Path $envFile -Value $envContent -Encoding UTF8
        Write-Host '  FIXED: Created .env file'
    } else {
        Write-Host '  WARN: .env file missing'
    }
}

# ---- Step 3: Check hooks.json ----
Write-Host ''
Write-Host '[3/6] Checking hooks.json...'
$hooksJson = Join-Path $HOOKS_DIR 'hooks.json'
if (Test-Path $hooksJson) {
    $hooks = Get-Content $hooksJson -Raw | ConvertFrom-Json
    if ($hooks.pre_job.path) {
        Write-Host "  OK: hooks.json -> $($hooks.pre_job.path)"
    } else {
        Write-Host '  WARN: hooks.json missing pre_job path'
    }
} else {
    Write-Host '  WARN: hooks.json not found'
}

# ---- Step 4: Check pre-job hook ----
Write-Host ''
Write-Host '[4/6] Checking pre-job hook...'
if (Test-Path $PRE_JOB) {
    $hookContent = Get-Content $PRE_JOB -Raw
    $checks = @(
        @{ Name = 'OLLAMA_HOST'; Pattern = 'OLLAMA_HOST' },
        @{ Name = 'SLATE_OLLAMA_URL'; Pattern = 'SLATE_OLLAMA_URL' },
        @{ Name = 'CUDA_VISIBLE_DEVICES'; Pattern = 'CUDA_VISIBLE_DEVICES' },
        @{ Name = 'PYTHONPATH'; Pattern = 'PYTHONPATH' },
        @{ Name = 'Python PATH'; Pattern = '\.venv\\Scripts' }
    )
    foreach ($c in $checks) {
        if ($hookContent -match [regex]::Escape($c.Pattern)) {
            Write-Host "  OK: $($c.Name) configured"
        } else {
            Write-Host "  WARN: $($c.Name) missing from pre-job hook"
        }
    }
} else {
    Write-Host '  FAIL: pre-job.ps1 not found!'
}

# ---- Step 5: Check GPUs ----
Write-Host ''
Write-Host '[5/6] Checking GPU access...'
try {
    $gpuInfo = nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>$null
    if ($gpuInfo) {
        $gpuLines = ($gpuInfo -split "`n" | Where-Object { $_.Trim() })
        Write-Host "  OK: $($gpuLines.Count) GPUs detected"
        foreach ($gpu in $gpuLines) {
            Write-Host "    $gpu"
        }
    } else {
        Write-Host '  FAIL: nvidia-smi returned no data'
    }
} catch {
    Write-Host '  FAIL: nvidia-smi not available'
}

# ---- Step 6: Test connectivity ----
Write-Host ''
Write-Host '[6/6] Testing AI inference connectivity...'
if ($ollamaOk) {
    try {
        $body = @{
            model = 'llama3.2:3b'
            prompt = 'Reply with just the word OK'
            stream = $false
            options = @{ num_predict = 5 }
        } | ConvertTo-Json
        $start = Get-Date
        $result = Invoke-RestMethod -Uri "$OLLAMA_URL/api/generate" -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 30
        $elapsed = ((Get-Date) - $start).TotalSeconds
        $tokens = $result.eval_count
        $tps = if ($result.eval_duration -gt 0) { [math]::Round($tokens / ($result.eval_duration / 1e9), 1) } else { 0 }
        Write-Host "  OK: Inference working ($tokens tokens @ ${tps} tok/s in ${elapsed}s)"
    } catch {
        Write-Host "  FAIL: Inference test failed - $_"
    }
} else {
    Write-Host '  SKIP: Ollama not available'
}

# ---- Summary ----
Write-Host ''
Write-Host '============================================================'
Write-Host '  Fix Complete - Summary'
Write-Host '============================================================'
Write-Host "  Ollama:      $(if ($ollamaOk) { 'Running' } else { 'NOT RUNNING' })"
Write-Host "  Runner .env: $(if (Test-Path $envFile) { 'Present' } else { 'Missing' })"
Write-Host "  Pre-job:     $(if (Test-Path $PRE_JOB) { 'Present' } else { 'Missing' })"
Write-Host ''
if (-not $Verify) {
    Write-Host '  Next: Restart the runner service to apply changes:'
    Write-Host "    cd $RUNNER_DIR"
    Write-Host '    .\svc.cmd stop'
    Write-Host '    .\svc.cmd start'
    Write-Host ''
    Write-Host '  Or run interactively:'
    Write-Host '    .\run.cmd'
}
Write-Host '============================================================'
