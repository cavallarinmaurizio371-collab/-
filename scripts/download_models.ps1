$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$ModelDir = Join-Path $ProjectRoot 'models\mediapipe'
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$HandModel = Join-Path $ModelDir 'hand_landmarker.task'
if (-not (Test-Path -LiteralPath $HandModel)) {
  Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile $HandModel
}
& (Join-Path $ProjectRoot '.venv\Scripts\python.exe') (Join-Path $ProjectRoot 'scripts\download_models.py')
if ($LASTEXITCODE -ne 0) { throw "Model download/validation failed with exit code $LASTEXITCODE" }
