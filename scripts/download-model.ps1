<#
.SYNOPSIS
  Downloads a GGUF model tested with GPULlama3.java into .\models (resumable).

.EXAMPLE
  .\scripts\download-model.ps1                   # Llama 3.2 1B Instruct FP16 (2.5 GB, default)
  .\scripts\download-model.ps1 llama-3b          # Llama 3.2 3B Instruct FP16 (6.4 GB)
  .\scripts\download-model.ps1 deepseek-r1-1.5b  # DeepSeek-R1-Distill-Qwen-1.5B Q8_0 (1.9 GB)
  .\scripts\download-model.ps1 phi3-mini         # Phi-3 mini 4k Instruct Q4 (2.4 GB)
#>
param(
    [ValidateSet('llama-1b', 'llama-3b', 'deepseek-r1-1.5b', 'phi3-mini')]
    [string]$Model = 'llama-1b'
)

$ErrorActionPreference = 'Stop'

$catalog = @{
    'llama-1b'         = 'https://huggingface.co/beehive-lab/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-FP16.gguf'
    'llama-3b'         = 'https://huggingface.co/beehive-lab/Llama-3.2-3B-Instruct-GGUF-FP16/resolve/main/beehive-llama-3.2-3b-instruct-fp16.gguf'
    'deepseek-r1-1.5b' = 'https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf'
    'phi3-mini'        = 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf'
}

$url = $catalog[$Model]
$dir = Join-Path (Split-Path $PSScriptRoot -Parent) 'models'
New-Item -ItemType Directory -Force $dir | Out-Null
$target = Join-Path $dir ($url.Split('/')[-1])

Write-Host "Downloading $Model -> $target"
# curl.exe ships with Windows 10+; -C - resumes a partial download.
& curl.exe -L --fail -C - -o $target $url
if ($LASTEXITCODE -ne 0) { throw "Download failed (curl exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Done. Run with:"
Write-Host "  `$env:LLM_MODEL_PATH = '$target'"
Write-Host "  .\run.ps1 gpu"
