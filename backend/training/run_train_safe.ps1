<#
Run training safely. Usage examples:

.\run_train_safe.ps1 -Model distilgpt2 -OutputDir .\runs\distilgpt2-smoke -Epochs 1

For DeepSpeed/ZeRO offload use:
.\run_train_safe.ps1 -Model THE/HF-MODEL -OutputDir .\runs\my-finetune -UseDeepSpeed -DeepSpeedConfig .\training\deepspeed_zero2_offload.json

#>

param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [int]$Epochs = 1,
    [switch]$UseDeepSpeed,
    [string]$DeepSpeedConfig = "",
    [switch]$Use8bit,
    [switch]$UseLoRA
)

# Activate virtualenv (assumes backend .venv)
Push-Location -Path (Split-Path -Path $MyInvocation.MyCommand.Definition -Parent)
Set-Location -Path ..\
. .venv\Scripts\Activate.ps1

if ($UseDeepSpeed) {
    if (-not (Test-Path $DeepSpeedConfig)) {
        Write-Error "DeepSpeed config not found: $DeepSpeedConfig"
        exit 1
    }
    Write-Host "Launching with accelerate + deepspeed (ZeRO offload). Monitor with nvidia-smi and Task Manager."
    $cmd = "training\train_safe.py --model $Model --output_dir $OutputDir --epochs $Epochs --deepspeed_config $DeepSpeedConfig"
    if ($Use8bit) { $cmd += " --use_8bit" }
    if ($UseLoRA) { $cmd += " --use_lora" }
    # Use accelerate launch
    Write-Host "Command: accelerate launch $cmd"
    accelerate launch $cmd
} else {
    Write-Host "Running trainer directly (no accelerate / deepspeed)."
    $cmd = "training\train_safe.py --model $Model --output_dir $OutputDir --epochs $Epochs"
    if ($Use8bit) { $cmd += " --use_8bit" }
    if ($UseLoRA) { $cmd += " --use_lora" }
    python $cmd
}

Pop-Location
