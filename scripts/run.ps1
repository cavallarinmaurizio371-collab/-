$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$env:TORCH_HOME = Join-Path $ProjectRoot 'models\torch'
$env:HF_HOME = Join-Path $ProjectRoot 'models\huggingface'
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot 'cache'
$env:TEMP = Join-Path $ProjectRoot 'cache\temp'
$env:TMP = $env:TEMP
Set-Location -LiteralPath $ProjectRoot
& (Join-Path $ProjectRoot '.venv\Scripts\python.exe') app.py @args

