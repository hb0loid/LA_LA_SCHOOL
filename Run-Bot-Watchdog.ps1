param(
  [Parameter(Mandatory = $true)]
  [string]$Root,

  [Parameter(Mandatory = $true)]
  [string]$Instance,

  [Parameter(Mandatory = $true)]
  [string]$OutLog,

  [Parameter(Mandatory = $true)]
  [string]$ErrLog,

  [string]$Python = "",

  [int]$RestartDelaySeconds = 8
)

$ErrorActionPreference = "Continue"

Set-Location -LiteralPath $Root
if (-not (Test-Path -LiteralPath $OutLog)) { New-Item -ItemType File -Path $OutLog | Out-Null }
if (-not (Test-Path -LiteralPath $ErrLog)) { New-Item -ItemType File -Path $ErrLog | Out-Null }

$python = if ($Python) {
  (Resolve-Path -LiteralPath $Python -ErrorAction Stop).Path
} else {
  (Get-Command python -ErrorAction Stop).Source
}
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Watchdog started instance=$Instance watchdog_pid=$PID" -ErrorAction SilentlyContinue

while ($true) {
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Watchdog launching instance=$Instance" -ErrorAction SilentlyContinue
  $command = "`"$python`" -m laladub.bot --instance $Instance 1>> `"$OutLog`" 2>> `"$ErrLog`""
  & cmd.exe /d /c $command
  $exitCode = $LASTEXITCODE
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Bot exited instance=$Instance code=$exitCode; restarting in ${RestartDelaySeconds}s" -ErrorAction SilentlyContinue
  Start-Sleep -Seconds $RestartDelaySeconds
}
