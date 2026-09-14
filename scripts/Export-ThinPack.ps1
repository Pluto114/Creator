[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BundleId,
    [string[]]$Cases = @(),
    [int[]]$Angles = @(),
    [string]$BlenderPath = 'D:\CloudMusic\steam\steamapps\common\Blender\blender.exe'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-CreatorEnvironment.ps1')
foreach ($CreatorSetting in @('CONFIG', 'SCRIPTS', 'EXTENSIONS', 'DATAFILES')) {
    $CreatorProfilePath = Join-Path $CreatorRoot ('.local\blender-profile\' + $CreatorSetting.ToLower())
    [Environment]::SetEnvironmentVariable(('BLENDER_USER_' + $CreatorSetting), $CreatorProfilePath, 'Process')
}
$env:OPENCV_IO_ENABLE_OPENEXR = '1'
$CreatorArgs = @((Join-Path $PSScriptRoot 'export_thin_pack.py'), '--bundle-id', $BundleId, '--blender', $BlenderPath)
if ($Cases.Count) { $CreatorArgs += @('--cases') + $Cases }
if ($Angles.Count) { $CreatorArgs += @('--angles') + @($Angles | ForEach-Object { [string]$_ }) }
& (Join-Path $CreatorRoot 'backends\da3\.venv\Scripts\python.exe') @CreatorArgs
if ($LASTEXITCODE -ne 0) { throw 'Paired export failed; partial outputs and logs were preserved.' }
