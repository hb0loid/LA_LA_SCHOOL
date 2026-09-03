$ErrorActionPreference = "SilentlyContinue"

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$RuntimeDir = Join-Path $Root "work\runtime"
$LogDir = Join-Path $Root "logs"
$OutLog = Join-Path $LogDir "bot.out.log"
$PidFile = Join-Path $RuntimeDir "bot.pid"
# Stopping by hand means "stay down". The autostart health check runs every 5
# minutes and would otherwise bring the bot straight back up, so it looks for
# this marker and skips the launcher while it exists. Start-Bot removes it.
$HoldFile = Join-Path $RuntimeDir "bot.hold"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $RuntimeDir,$LogDir | Out-Null
if (-not (Test-Path -LiteralPath $OutLog)) { New-Item -ItemType File -Path $OutLog | Out-Null }

$operationMutex = [Threading.Mutex]::new($false, "Local\LaLaDubReleaseStart")
$hasOperationMutex = $false
try {
  $hasOperationMutex = $operationMutex.WaitOne([TimeSpan]::FromSeconds(15))
  if (-not $hasOperationMutex) { throw "Another bot start or stop operation is still running." }

$allProcesses = @(Get-CimInstance Win32_Process)
$rootProcesses = @($allProcesses | Where-Object {
  (
    ($_.CommandLine -like "*tools\runtime\Watchdog.ps1*" -and $_.CommandLine -like "*-Instance release*") -or
    ($_.CommandLine -like "*Run-Bot-Watchdog.ps1*" -and $_.CommandLine -like "*-Instance release*") -or
    ($_.CommandLine -like "*laladub.bot*" -and $_.CommandLine -like "*--instance release*")
  ) -and $_.CommandLine -like "*$Root*"
})

$targetIds = [Collections.Generic.HashSet[int]]::new()
foreach ($process in $rootProcesses) { [void]$targetIds.Add([int]$process.ProcessId) }
do {
  $added = $false
  foreach ($process in $allProcesses) {
    if ($targetIds.Contains([int]$process.ParentProcessId) -and $targetIds.Add([int]$process.ProcessId)) {
      $added = $true
    }
  }
} while ($added)

$stopped = $targetIds.Count -gt 0
$remaining = [Collections.Generic.HashSet[int]]::new($targetIds)
while ($remaining.Count -gt 0) {
  $leaves = @($remaining | Where-Object { $candidate = $_; -not ($allProcesses | Where-Object { $remaining.Contains([int]$_.ProcessId) -and [int]$_.ParentProcessId -eq $candidate }) })
  if (-not $leaves) { $leaves = @($remaining) }
  foreach ($processId in $leaves) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    [void]$remaining.Remove($processId)
  }
}
if ($stopped) {
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped release process tree: $($targetIds.Count) process(es)"
}
Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue

$deadline = (Get-Date).AddSeconds(10)
do {
  $listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
  if (-not $listener) { break }
  Start-Sleep -Milliseconds 250
} while ((Get-Date) -lt $deadline)

if (-not $stopped) {
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Release bot was not running."
}

Set-Content -LiteralPath $HoldFile -Value "$(Get-Date -Format s) stopped by operator" -Encoding utf8
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Autostart hold set: the health check will not restart the bot until Start-Bot runs."
} finally {
  if ($hasOperationMutex) { $operationMutex.ReleaseMutex() }
  $operationMutex.Dispose()
}
