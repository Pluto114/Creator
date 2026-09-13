[CmdletBinding()]
param([ValidateSet('base', 'large')][string[]]$Models = @('base', 'large'), [string]$ProxyUrl = '')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-CreatorEnvironment.ps1') -ProxyUrl $ProxyUrl
$CreatorManifest = Get-Content -LiteralPath (Join-Path $CreatorRoot 'configs\models.lock.json') -Raw | ConvertFrom-Json
foreach ($CreatorModel in $CreatorManifest.models | Where-Object { $_.name -in $Models }) {
    $CreatorDestination = Join-Path $CreatorRoot ('models\da3-' + $CreatorModel.name + '\' + $CreatorModel.revision)
    New-Item -ItemType Directory -Path $CreatorDestination -Force | Out-Null
    foreach ($CreatorFile in $CreatorModel.files) {
        $CreatorTarget = Join-Path $CreatorDestination $CreatorFile.name
        $CreatorValid = $false
        if (Test-Path -LiteralPath $CreatorTarget) {
            $CreatorValid = (Get-Item -LiteralPath $CreatorTarget).Length -eq $CreatorFile.bytes
            if ($CreatorValid -and $CreatorFile.sha256) {
                $CreatorValid = (Get-FileHash -LiteralPath $CreatorTarget -Algorithm SHA256).Hash -eq $CreatorFile.sha256
            }
        }
        if ($CreatorValid) { Write-Host "Verified: $CreatorTarget"; continue }
        $CreatorPartial = $CreatorTarget + '.part'
        # A completed partial may remain if the previous process stopped before publication.
        if (Test-Path -LiteralPath $CreatorPartial) {
            $CreatorComplete = (Get-Item -LiteralPath $CreatorPartial).Length -eq $CreatorFile.bytes
            if ($CreatorComplete -and $CreatorFile.sha256) {
                $CreatorComplete = (Get-FileHash -LiteralPath $CreatorPartial -Algorithm SHA256).Hash -eq $CreatorFile.sha256
            }
            if ($CreatorComplete) {
                Move-Item -LiteralPath $CreatorPartial -Destination $CreatorTarget -Force
                Write-Host "Verified: $CreatorTarget"
                continue
            }
        }
        $CreatorUrl = 'https://huggingface.co/' + $CreatorModel.repo_id + '/resolve/' + $CreatorModel.revision + '/' + $CreatorFile.name
        $CreatorCurlArgs = @('--location', '--fail', '--show-error', '--silent', '--retry', '4', '--retry-delay', '3', '--connect-timeout', '30', '--max-time', '3600', '--speed-time', '60', '--speed-limit', '16384', '--continue-at', '-', '--output', $CreatorPartial, $CreatorUrl)
        if ($env:HTTPS_PROXY) { $CreatorCurlArgs += @('--proxy', $env:HTTPS_PROXY) }
        Write-Host "Downloading $($CreatorModel.name)/$($CreatorFile.name): $($CreatorFile.bytes) bytes to D/project storage"
        & curl.exe @CreatorCurlArgs
        if ($LASTEXITCODE -ne 0) { throw "Download failed. Partial file is retained for resume: $CreatorPartial" }
        if ((Get-Item -LiteralPath $CreatorPartial).Length -ne $CreatorFile.bytes) { throw "File size mismatch: $CreatorPartial" }
        if ($CreatorFile.sha256 -and ((Get-FileHash -LiteralPath $CreatorPartial -Algorithm SHA256).Hash -ne $CreatorFile.sha256)) {
            throw "SHA256 mismatch: $CreatorPartial. Inspect and remove this partial file before retrying."
        }
        Move-Item -LiteralPath $CreatorPartial -Destination $CreatorTarget -Force
        Write-Host "Verified: $CreatorTarget"
    }
}