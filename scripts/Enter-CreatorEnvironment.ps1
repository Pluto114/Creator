# Dot-source this file: . .\scripts\Enter-CreatorEnvironment.ps1
[CmdletBinding()]
param([string]$ProxyUrl = '')
$CreatorRoot = Split-Path -Parent $PSScriptRoot
$CreatorPaths = @{
    UV_CACHE_DIR = '.local\uv-cache'
    UV_PYTHON_INSTALL_DIR = '.local\python'
    PIP_CACHE_DIR = '.local\pip-cache'
    TEMP = '.local\tmp'
    TMP = '.local\tmp'
    HF_HOME = 'models\huggingface'
    HF_HUB_CACHE = 'models\huggingface\hub'
    TORCH_HOME = 'models\torch'
    TORCH_EXTENSIONS_DIR = '.local\torch-extensions'
    CUDA_CACHE_PATH = '.local\cuda-cache'
    TRITON_CACHE_DIR = '.local\triton-cache'
    XDG_CACHE_HOME = '.local\cache'
    MPLCONFIGDIR = '.local\matplotlib'
    IMAGEIO_USERDIR = '.local\imageio'
    IPYTHONDIR = '.local\ipython'
    JUPYTER_CONFIG_DIR = '.local\jupyter'
    PRE_COMMIT_HOME = '.local\pre-commit'
    NUMBA_CACHE_DIR = '.local\numba-cache'
}
foreach ($CreatorEntry in $CreatorPaths.GetEnumerator()) {
    $CreatorPath = Join-Path $CreatorRoot $CreatorEntry.Value
    New-Item -ItemType Directory -Path $CreatorPath -Force -ErrorAction Stop | Out-Null
    if (-not (Test-Path -LiteralPath $CreatorPath -PathType Container)) { throw "Cache directory unavailable: $CreatorPath" }
    [Environment]::SetEnvironmentVariable($CreatorEntry.Key, $CreatorPath, 'Process')
}
$env:UV_HTTP_TIMEOUT = '600'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:UV_LINK_MODE = 'hardlink'
$env:UV_PYTHON_DOWNLOADS = 'never'
$env:PYTHONNOUSERSITE = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$CreatorTools = Join-Path $CreatorRoot '.venv-tools\Scripts'
if ((Test-Path -LiteralPath $CreatorTools) -and (($env:PATH -split ';') -notcontains $CreatorTools)) {
    $env:PATH = "$CreatorTools;$env:PATH"
}
if ($ProxyUrl) {
    $env:HTTPS_PROXY = $ProxyUrl
    $env:HTTP_PROXY = $ProxyUrl
}
Write-Host "Creator session ready: $CreatorRoot (process-local settings; caches and temporary files stay here)."