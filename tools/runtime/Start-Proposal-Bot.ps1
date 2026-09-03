param(
  [ValidateRange(0, 300)]
  [int]$StartupDelaySeconds = 0
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$RuntimeDir = Join-Path $Root "work\runtime"
$LogDir = Join-Path $Root "logs"
$OutLog = Join-Path $LogDir "proposal-bot.out.log"
$ErrLog = Join-Path $LogDir "proposal-bot.err.log"
$PidFile = Join-Path $RuntimeDir "proposal-bot.pid"
$WatchdogScript = Join-Path $PSScriptRoot "Watchdog.ps1"
$TokenFile = Join-Path $Root ".secrets\Proposal-Bot-Token.ps1"
$HoldFile = Join-Path $RuntimeDir "proposal-bot.hold"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $RuntimeDir,$LogDir,(Join-Path $Root "runs\proposal") | Out-Null
# Starting on purpose clears the "stopped by operator" hold left by the stop
# script, so the autostart health check resumes keeping the bot alive.
Remove-Item -LiteralPath $HoldFile -Force -ErrorAction SilentlyContinue
foreach ($logPath in @($OutLog, $ErrLog)) {
  if (-not (Test-Path -LiteralPath $logPath)) { New-Item -ItemType File -Path $logPath | Out-Null }
  $item = Get-Item -LiteralPath $logPath -ErrorAction SilentlyContinue
  # The watchdog keeps both logs open for redirection. A health check must not
  # report the already-running bot as failed merely because Windows has the log
  # file locked at this instant.
  if ($item -and $item.Length -gt 10MB) {
    Clear-Content -LiteralPath $logPath -ErrorAction SilentlyContinue
  }
}

$mutex = [Threading.Mutex]::new($false, "Local\LaLaDubProposalStart")
$hasMutex = $false
try {
  $hasMutex = $mutex.WaitOne([TimeSpan]::FromSeconds(15))
  if (-not $hasMutex) { throw "Another proposal bot start is still running." }
  if ($StartupDelaySeconds -gt 0) { Start-Sleep -Seconds $StartupDelaySeconds }

  $existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -like "*tools\runtime\Watchdog.ps1*" -and
      $_.CommandLine -like "*-Instance proposal*" -and
      $_.CommandLine -like "*$Root*"
    } |
    Select-Object -First 1
  if ($existing) {
    Set-Content -LiteralPath $PidFile -Value $existing.ProcessId
    Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Proposal watchdog already running, pid=$($existing.ProcessId)" -ErrorAction SilentlyContinue
    exit 0
  }
  Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue

  $ProposalBotToken = ""
  if (Test-Path -LiteralPath $TokenFile) {
    . $TokenFile
  }
  if (-not $ProposalBotToken) {
    $ProposalBotToken = [Environment]::GetEnvironmentVariable("LALADUB_PROPOSAL_BOT_TOKEN", "User")
  }
  if (-not $ProposalBotToken) {
    # Compatibility with the token name used by the old version of @ghienmigoobot.
    $ProposalBotToken = [Environment]::GetEnvironmentVariable("LALADUB_BOT_TOKEN", "User")
  }
  if (-not $ProposalBotToken) {
    Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Proposal bot token is not configured."
    exit 1
  }

  $python = Join-Path $Root ".venv-f5tts\Scripts\python.exe"
  if (-not (Test-Path -LiteralPath $python)) {
    Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Python is missing: $python"
    exit 1
  }
  $python = (Resolve-Path -LiteralPath $python).Path

  $env:LALADUB_PROPOSAL_BOT_TOKEN = $ProposalBotToken
  $env:LALADUB_PROPOSAL_DB = (Join-Path $Root "runs\proposal\proposals.sqlite3")
  $env:LALADUB_PROPOSAL_MODERATORS = "631551040"
  $env:LALADUB_PROPOSAL_MAIN_CHANNEL = "@elevenlabss"
  $env:LALADUB_PROPOSAL_SHAME_CHANNEL = "@ghienmigo"
  $env:LALADUB_PROPOSAL_KARMA_CHAT = "@lalaschoo"
  $env:PYTHONPATH = (Join-Path $Root "src")
  $env:PYTHONIOENCODING = "utf-8"

  $arguments = (
    "-NoProfile " +
    "-ExecutionPolicy Bypass " +
    "-File `"$WatchdogScript`" " +
    "-Root `"$Root`" " +
    "-Instance proposal " +
    "-OutLog `"$OutLog`" " +
    "-ErrLog `"$ErrLog`" " +
    "-Python `"$python`" " +
    "-Module laladub.proposal_bot " +
    "-RestartDelaySeconds 8"
  )
  $process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $arguments `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru
  Set-Content -LiteralPath $PidFile -Value $process.Id
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Started proposal watchdog pid=$($process.Id)"
} finally {
  if ($hasMutex) { $mutex.ReleaseMutex() }
  $mutex.Dispose()
}
