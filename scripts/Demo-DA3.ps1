[CmdletBinding()]
param([string[]]$Images = @(), [ValidateSet('base','large')][string]$Model = 'base')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-CreatorEnvironment.ps1')
$CreatorArguments = @((Join-Path $PSScriptRoot 'demo_da3.py'), '--model', $Model)
if ($Images.Count) { $CreatorArguments += @('--images') + $Images }
& (Join-Path $CreatorRoot 'backends\da3\.venv\Scripts\python.exe') @CreatorArguments
if ($LASTEXITCODE -ne 0) { throw 'DA3 demo failed; inspect the newest .runtime/demo report.' }