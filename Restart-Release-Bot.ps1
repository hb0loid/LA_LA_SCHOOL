$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

& (Join-Path $Root "Stop-Release-Bot.ps1")
Start-Sleep -Seconds 2
& (Join-Path $Root "Start-Release-Bot.ps1")
