$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutLog = Join-Path $Root "bot.release.out.log"
$PidFile = Join-Path $Root "bot.release.pid"
$StartScript = Join-Path $Root "Start-Release-Bot.ps1"

Set-Location -LiteralPath $Root
if (-not (Test-Path -LiteralPath $OutLog)) { New-Item -ItemType File -Path $OutLog | Out-Null }

function Write-RestartLog {
  param([string]$Message)
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Restart shortcut: $Message" -ErrorAction SilentlyContinue
}

function Get-ReleaseWatchdog {
  $pidText = $null
  if (Test-Path -LiteralPath $PidFile) {
    $pidText = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  }
  if ($pidText) {
    $byPid = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
    if (
      $byPid -and
      $byPid.CommandLine -like "*Run-Bot-Watchdog.ps1*" -and
      $byPid.CommandLine -like "*-Instance release*" -and
      $byPid.CommandLine -like "*$Root*"
    ) {
      return $byPid
    }
  }

  Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -like "*Run-Bot-Watchdog.ps1*" -and
      $_.CommandLine -like "*-Instance release*" -and
      $_.CommandLine -like "*$Root*"
    } |
    Select-Object -First 1
}

function Get-DescendantProcesses {
  param([int]$ParentPid)

  $all = Get-CimInstance Win32_Process
  $result = New-Object System.Collections.Generic.List[object]
  $queue = New-Object System.Collections.Generic.Queue[int]
  $queue.Enqueue($ParentPid)

  while ($queue.Count -gt 0) {
    $current = $queue.Dequeue()
    $children = $all | Where-Object { $_.ParentProcessId -eq $current }
    foreach ($child in $children) {
      $result.Add($child)
      $queue.Enqueue([int]$child.ProcessId)
    }
  }

  $result
}

function Wait-ReleaseBotChild {
  param(
    [int]$WatchdogPid,
    [int[]]$OldPids
  )

  $deadline = (Get-Date).AddSeconds(45)
  do {
    Start-Sleep -Seconds 1
    $children = @(Get-DescendantProcesses -ParentPid $WatchdogPid)
    $bot = $children |
      Where-Object {
        $_.CommandLine -like "*laladub.bot*" -and
        $_.CommandLine -like "*--instance release*" -and
        ($OldPids -notcontains [int]$_.ProcessId)
      } |
      Select-Object -First 1
    if ($bot) {
      return $bot
    }
  } while ((Get-Date) -lt $deadline)

  return $null
}

$watchdog = Get-ReleaseWatchdog
if (-not $watchdog) {
  Write-RestartLog "watchdog is not running; starting release bot"
  & $StartScript
  exit $LASTEXITCODE
}

Set-Content -LiteralPath $PidFile -Value $watchdog.ProcessId -ErrorAction SilentlyContinue

$children = @(Get-DescendantProcesses -ParentPid ([int]$watchdog.ProcessId))
$botChildren = @(
  $children |
    Where-Object {
      $_.CommandLine -like "*laladub.bot*" -or
      $_.CommandLine -like "*cmd.exe*laladub.bot*"
    }
)

if (-not $botChildren -or $botChildren.Count -eq 0) {
  Write-RestartLog "watchdog pid=$($watchdog.ProcessId) has no bot child; waiting for watchdog"
  $started = Wait-ReleaseBotChild -WatchdogPid ([int]$watchdog.ProcessId) -OldPids @()
  if ($started) {
    Write-RestartLog "bot is running pid=$($started.ProcessId)"
    exit 0
  }
  Write-RestartLog "bot child did not appear in time"
  exit 1
}

$oldPids = @($children | ForEach-Object { [int]$_.ProcessId })
$stopOrder = $children | Sort-Object ProcessId -Descending
foreach ($process in $stopOrder) {
  try {
    Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction Stop
    Write-RestartLog "stopped child pid=$($process.ProcessId) name=$($process.Name)"
  } catch {
    Write-RestartLog "could not stop child pid=$($process.ProcessId): $($_.Exception.Message)"
  }
}

$startedBot = Wait-ReleaseBotChild -WatchdogPid ([int]$watchdog.ProcessId) -OldPids $oldPids
if ($startedBot) {
  Write-RestartLog "restarted bot under watchdog pid=$($watchdog.ProcessId), bot_pid=$($startedBot.ProcessId)"
  exit 0
}

Write-RestartLog "restart requested, but new bot child was not detected in time; starting a fresh watchdog"
try {
  $watchdogStillRunning = Get-CimInstance Win32_Process -Filter "ProcessId = $($watchdog.ProcessId)" -ErrorAction SilentlyContinue
  if ($watchdogStillRunning) {
    Stop-Process -Id ([int]$watchdog.ProcessId) -Force -ErrorAction Stop
    Write-RestartLog "stopped stale watchdog pid=$($watchdog.ProcessId)"
    Start-Sleep -Seconds 2
  }
} catch {
  Write-RestartLog "could not stop stale watchdog pid=$($watchdog.ProcessId): $($_.Exception.Message)"
}

& $StartScript
exit $LASTEXITCODE
