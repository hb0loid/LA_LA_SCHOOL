param(
  [string]$Root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
)

$ErrorActionPreference = "Stop"
$tokenPath = Join-Path $Root ".secrets\Worker-Api-Token.txt"
$tokenDirectory = Split-Path -Parent $tokenPath
New-Item -ItemType Directory -Force -Path $tokenDirectory | Out-Null
$token = ([guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N"))
$temporaryPath = "$tokenPath.tmp"
Set-Content -LiteralPath $temporaryPath -Value $token -Encoding ASCII -NoNewline
Move-Item -LiteralPath $temporaryPath -Destination $tokenPath -Force
Write-Host "Worker API token rotated. Restart the release bot so workers can synchronize it."
