$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\\Scripts\\python.exe"
$specPath = Join-Path $projectRoot "RegimeMapApp.spec"

if (-not (Test-Path $pythonExe)) {
    throw "Python interpreter not found: $pythonExe"
}

if (-not (Test-Path $specPath)) {
    throw "Spec file not found: $specPath"
}

& $pythonExe -m PyInstaller --noconfirm --clean $specPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Build completed: $(Join-Path $projectRoot 'dist\\RegimeMapApp.exe')"
