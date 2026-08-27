$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$VenvPath = Join-Path $ProjectRoot '.venv'
$BasePython = 'C:\Users\33233\miniconda3\python.exe'
if (-not (Test-Path -LiteralPath $BasePython)) { $BasePython = (Get-Command python).Source }
if (-not (Test-Path -LiteralPath $VenvPath)) { & $BasePython -m venv --system-site-packages $VenvPath }
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot 'cache\pip'
$env:TEMP = Join-Path $ProjectRoot 'cache\temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR,$env:TEMP | Out-Null
& (Join-Path $VenvPath 'Scripts\python.exe') -m pip install --disable-pip-version-check --timeout 120 --retries 5 "numpy==1.26.4" "opencv-python==4.11.0.86" "opencv-contrib-python==4.11.0.86" "jax==0.4.38" "jaxlib==0.4.38" "mediapipe==0.10.21" pytest "transformers==4.57.6" Pillow
if ($LASTEXITCODE -ne 0) { throw "Project-local dependency installation failed with exit code $LASTEXITCODE" }
$CudaReady = & (Join-Path $VenvPath 'Scripts\python.exe') -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($CudaReady -ne '1') {
  & (Join-Path $VenvPath 'Scripts\python.exe') -m pip install --disable-pip-version-check --timeout 180 --retries 5 --force-reinstall --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
  if ($LASTEXITCODE -ne 0) { throw "Project-local CUDA PyTorch installation failed with exit code $LASTEXITCODE" }
}
Write-Host 'Setup complete. Run scripts\run.ps1'
