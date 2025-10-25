<#
Wrapper start-dev script at repository root.
This simply calls the real script in the `eupraxia-pleroma` folder so you can run it from the current working directory.
Usage: from the workspace folder run:
  .\start-dev.ps1
#>

$scriptPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'eupraxia-pleroma\start-dev.ps1'

if (-not (Test-Path $scriptPath)) {
    Write-Error "Cannot find inner script at: $scriptPath`nMake sure the repository contains 'eupraxia-pleroma\start-dev.ps1'"
    exit 1
}

Write-Host "Launching dev script: $scriptPath" -ForegroundColor Cyan
& $scriptPath
