param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("on", "off", "status")]
  [string]$Mode
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$FlagPath = Join-Path $Root "runs\\bot-release\\maintenance.flag"

switch ($Mode) {
  "on" {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $FlagPath) | Out-Null
    $state = @{
      enabled = $true
      enabled_at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
      enabled_by = "local"
    } | ConvertTo-Json
    Set-Content -LiteralPath $FlagPath -Value $state -Encoding UTF8
    Write-Host "Maintenance mode is ON. Regular users are blocked."
  }
  "off" {
    Remove-Item -LiteralPath $FlagPath -Force -ErrorAction SilentlyContinue
    Write-Host "Maintenance mode is OFF. Regular users can use the bot."
  }
  "status" {
    if (Test-Path -LiteralPath $FlagPath) {
      Write-Host "Maintenance mode is ON."
    } else {
      Write-Host "Maintenance mode is OFF."
    }
  }
}
