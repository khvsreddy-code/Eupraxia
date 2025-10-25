<#
start-dev.ps1

Runs the backend (Python) and frontend (Vite) locally on Windows without Docker.
Opens two terminals and starts each service. Requires Node.js and Python to be installed.

Usage: Run from the repository root (double-click or from an elevated or normal PowerShell):
  .\start-dev.ps1

#>

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Starting Eupraxia-Pleroma dev environment (robust) ..." -ForegroundColor Cyan

# Use a temporary ExecutionPolicy bypass for this process
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Helper: start backend and wait for /health
function Start-Backend {
    $backendPath = Join-Path $repoRoot 'backend'
    if (-not (Test-Path $backendPath)) { Write-Host "Backend folder not found: $backendPath" -ForegroundColor Red; return }

    Write-Host "Preparing backend venv and dependencies..." -ForegroundColor Green
    $cmd = @(
        'python -m venv .venv',
        '.\\.venv\\Scripts\\Activate.ps1',
        'python -m pip install --upgrade pip setuptools wheel',
        'pip install -r requirements.txt'
    ) -join '; '

    # Start a persistent PowerShell window for backend
    $backendArgs = "-NoExit","-Command","Set-Location -LiteralPath '$backendPath'; $cmd; Write-Host 'Starting Uvicorn...'; python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
    Start-Process -FilePath powershell -ArgumentList $backendArgs -WorkingDirectory $backendPath -WindowStyle Normal

    # Wait for health endpoint
    Write-Host "Waiting for backend to report healthy (http://localhost:8000/health)..." -ForegroundColor Cyan
    for ($i=0; $i -lt 30; $i++) {
        try {
            $r = Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { Write-Host "Backend healthy." -ForegroundColor Green; return }
        } catch { Start-Sleep -Seconds 1 }
    }
    Write-Host "Backend did not become healthy within timeout. Check backend logs in the opened window." -ForegroundColor Yellow
}

function Start-Frontend {
    $frontendPath = Join-Path $repoRoot 'frontend'
    if (-not (Test-Path $frontendPath)) { Write-Host "Frontend folder not found: $frontendPath" -ForegroundColor Red; return }

    Write-Host "Starting frontend (Vite) in new window..." -ForegroundColor Green
    $frontendArgs = "-NoExit","-Command","Set-Location -LiteralPath '$frontendPath'; npm install; npm run dev -- --host 0.0.0.0"
    Start-Process -FilePath powershell -ArgumentList $frontendArgs -WorkingDirectory $frontendPath -WindowStyle Normal
}

Start-Backend
Start-Frontend

Write-Host "Started. Backend -> http://localhost:8000  Frontend -> http://localhost:3000" -ForegroundColor Cyan
