[CmdletBinding()]
param([string]$BlenderPath = 'D:\CloudMusic\steam\steamapps\common\Blender\blender.exe', [switch]$Check)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-CreatorEnvironment.ps1')
if (-not (Test-Path -LiteralPath $BlenderPath -PathType Leaf)) { throw "Blender not found: $BlenderPath" }
$CreatorProfile = Join-Path $CreatorRoot '.local\blender-profile'
$env:BLENDER_USER_CONFIG = Join-Path $CreatorProfile 'config'
$env:BLENDER_USER_SCRIPTS = Join-Path $CreatorProfile 'scripts'
$env:BLENDER_USER_EXTENSIONS = Join-Path $CreatorProfile 'extensions'
$env:BLENDER_USER_DATAFILES = Join-Path $CreatorProfile 'datafiles'
foreach ($CreatorDirectory in @($env:BLENDER_USER_CONFIG, $env:BLENDER_USER_SCRIPTS, $env:BLENDER_USER_EXTENSIONS, $env:BLENDER_USER_DATAFILES)) {
    New-Item -ItemType Directory -Path $CreatorDirectory -Force | Out-Null
}
$CreatorAddonDirectory = Join-Path $env:BLENDER_USER_SCRIPTS 'addons'
New-Item -ItemType Directory -Path $CreatorAddonDirectory -Force | Out-Null
$CreatorAddonTarget = [IO.Path]::GetFullPath((Join-Path $CreatorAddonDirectory 'creator_recon'))
$CreatorExpectedTarget = [IO.Path]::GetFullPath((Join-Path $CreatorRoot '.local\blender-profile\scripts\addons\creator_recon'))
if ($CreatorAddonTarget -ne $CreatorExpectedTarget) { throw 'Unexpected add-on copy destination.' }
if (Test-Path -LiteralPath $CreatorAddonTarget) {
    $CreatorExisting = Get-Item -LiteralPath $CreatorAddonTarget -Force
    if ($CreatorExisting.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing to replace an add-on junction or link.' }
    if ($CreatorExisting.FullName -ne $CreatorExpectedTarget) { throw 'Unexpected resolved add-on directory.' }
    Remove-Item -LiteralPath $CreatorAddonTarget -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $CreatorRoot 'blender_addon\creator_recon') -Destination $CreatorAddonDirectory -Recurse -Force
$CreatorBlenderArguments = @('--python-exit-code', '1', '--python', (Join-Path $PSScriptRoot 'configure_blender.py'))
if ($Check) { $CreatorBlenderArguments = @('--background') + $CreatorBlenderArguments }
& $BlenderPath @CreatorBlenderArguments
if ($LASTEXITCODE -ne 0) { throw 'Creator Blender startup/configuration failed.' }