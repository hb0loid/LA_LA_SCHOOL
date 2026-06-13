$ErrorActionPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutLog = Join-Path $Root "bot.release.out.log"
$PidFile = Join-Path $Root "bot.release.pid"

Set-Location -LiteralPath $Root
if (-not (Test-Path -LiteralPath $OutLog)) { New-Item -ItemType File -Path $OutLog | Out-Null }

$stopped = $false
if (Test-Path -LiteralPath $PidFile) {
  $pidText = Get-Content -LiteralPath $PidFile | Select-Object -First 1
  if ($pidText) {
    $process = Get-Process -Id ([int]$pidText)
    if ($process) {
      Stop-Process -Id $process.Id -Force
      Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped release watchdog pid=$pidText"
      $stopped = $true
    }
  }
  Remove-Item -LiteralPath $PidFile -Force
}

$watchdogs = Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like "*Run-Bot-Watchdog.ps1*" -and
    $_.CommandLine -like "*-Instance release*" -and
    $_.CommandLine -like "*$Root*"
  }

foreach ($watchdog in $watchdogs) {
  Stop-Process -Id $watchdog.ProcessId -Force
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped extra release watchdog pid=$($watchdog.ProcessId)"
  $stopped = $true
}

$matches = Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like "*laladub.bot*" -and
    $_.CommandLine -like "*--instance release*"
  }

foreach ($match in $matches) {
  Stop-Process -Id $match.ProcessId -Force
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped extra release pid=$($match.ProcessId)"
  $stopped = $true
}

if (-not $stopped) {
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Release bot was not running."
}
