$ErrorActionPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Root "Stop-Release-Bot.ps1")
