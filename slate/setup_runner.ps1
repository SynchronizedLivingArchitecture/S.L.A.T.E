# SLATE Self-Hosted Runner Setup Script
# Configures the GitHub Actions runner to use E:\11132025 as workspace

$ErrorActionPreference = "Stop"

$RUNNER_DIR = "E:\actions-runner"
$WORK_DIR = "E:\11132025"
$REPO_URL = "https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E."

Write-Host "=== SLATE Self-Hosted Runner Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if runner directory exists
if (-not (Test-Path $RUNNER_DIR)) {
    Write-Host "Creating runner directory at $RUNNER_DIR..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $RUNNER_DIR -Force | Out-Null

    Write-Host "Downloading GitHub Actions runner..." -ForegroundColor Yellow
    $runnerUrl = "https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-win-x64-2.321.0.zip"
    $runnerZip = "$RUNNER_DIR\actions-runner.zip"
    Invoke-WebRequest -Uri $runnerUrl -OutFile $runnerZip

    Write-Host "Extracting runner..." -ForegroundColor Yellow
    Expand-Archive -Path $runnerZip -DestinationPath $RUNNER_DIR -Force
    Remove-Item $runnerZip
}

# Check if runner is already configured
if (Test-Path "$RUNNER_DIR\.runner") {
    Write-Host "Runner already configured. Removing old configuration..." -ForegroundColor Yellow
    Push-Location $RUNNER_DIR
    try {
        & ".\config.cmd" remove --token (Read-Host "Enter runner removal token")
    } catch {
        Write-Host "Could not remove old config. Continuing anyway..." -ForegroundColor Yellow
    }
    Pop-Location
}

Write-Host ""
Write-Host "=== Runner Configuration ===" -ForegroundColor Cyan
Write-Host "Repository: $REPO_URL"
Write-Host "Work Directory: $WORK_DIR"
Write-Host ""
Write-Host "Get your runner token from:"
Write-Host "$REPO_URL/settings/actions/runners/new?arch=x64&os=win" -ForegroundColor Blue
Write-Host ""

$token = Read-Host "Enter runner registration token"

if ([string]::IsNullOrEmpty($token)) {
    Write-Host "No token provided. Exiting." -ForegroundColor Red
    exit 1
}

# Configure the runner
Push-Location $RUNNER_DIR
try {
    & ".\config.cmd" --url $REPO_URL `
        --token $token `
        --name "slate-$env:COMPUTERNAME" `
        --work $WORK_DIR `
        --labels "self-hosted,slate,gpu,windows,cuda,gpu-2,multi-gpu,blackwell" `
        --replace

    Write-Host ""
    Write-Host "=== Runner Configured Successfully ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the runner:"
    Write-Host "  cd $RUNNER_DIR && .\run.cmd"
    Write-Host ""
    Write-Host "To install as Windows service:"
    Write-Host "  cd $RUNNER_DIR && .\svc.cmd install"
    Write-Host "  cd $RUNNER_DIR && .\svc.cmd start"
    Write-Host ""

} catch {
    Write-Host "Configuration failed: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
