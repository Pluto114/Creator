[CmdletBinding()]
param([string]$ProxyUrl = '', [switch]$SkipModels)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-CreatorEnvironment.ps1') -ProxyUrl $ProxyUrl
$CreatorPython = Join-Path $CreatorRoot '.venv-tools\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $CreatorPython)) {
    & py -3.11 -m venv (Join-Path $CreatorRoot '.venv-tools')
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 first. Blender Python is not a substitute.' }
}
& $CreatorPython -m pip install 'uv==0.12.13' 'build==1.6.1'
if ($LASTEXITCODE -ne 0) { throw 'Tool installation failed.' }
$CreatorUv = Join-Path $CreatorRoot '.venv-tools\Scripts\uv.exe'
foreach ($CreatorProject in @('reconstruction', 'experiments', 'backends\da3')) {
    $CreatorArguments = @('sync', '--project', (Join-Path $CreatorRoot $CreatorProject), '--locked', '--python', $CreatorPython)
    if ($CreatorProject -eq 'backends\da3') { $CreatorArguments += @('--extra', 'inference') }
    & $CreatorUv @CreatorArguments
    if ($LASTEXITCODE -ne 0) { throw "Environment sync failed: $CreatorProject" }
    $CreatorProjectPython = Join-Path $CreatorRoot ($CreatorProject + '\.venv\Scripts\python.exe')
    & $CreatorUv pip check --python $CreatorProjectPython
    if ($LASTEXITCODE -ne 0) { throw "Dependency metadata mismatch: $CreatorProject" }
}
& $CreatorPython (Join-Path $PSScriptRoot 'check_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'Environment checks failed; inspect .runtime/environment.json.' }
if (-not $SkipModels) {
    & (Join-Path $PSScriptRoot 'Download-CreatorModels.ps1') -ProxyUrl $ProxyUrl
}
Write-Host 'Environments ready. See docs/environment.md for model smoke tests and Blender launch.'