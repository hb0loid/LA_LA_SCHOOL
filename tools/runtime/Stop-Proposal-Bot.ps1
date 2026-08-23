$ErrorActionPreference = "SilentlyContinue"

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$PidFile = Join-Path $Root "work\runtime\proposal-bot.pid"
$OutLog = Join-Path $Root "logs\proposal-bot.out.log"
# See Stop-Bot.ps1: this marker keeps the 5-minute autostart health check from
# reviving a bot that was stopped on purpose. Start-Proposal-Bot removes it.
$HoldFile = Join-Path $Root "work\runtime\proposal-bot.hold"
$allProcesses = @(Get-CimInstance Win32_Process)
$roots = @($allProcesses | Where-Object {
  (
    ($_.CommandLine -like "*tools\runtime\Watchdog.ps1*" -and $_.CommandLine -like "*-Instance proposal*") -or
    $_.CommandLine -like "*laladub.proposal_bot*"
  ) -and $_.CommandLine -like "*$Root*"
})

$targetIds = [Collections.Generic.HashSet[int]]::new()
foreach ($process in $roots) { [void]$targetIds.Add([int]$process.ProcessId) }
do {
  $added = $false
  foreach ($process in $allProcesses) {
    if ($targetIds.Contains([int]$process.ParentProcessId) -and $targetIds.Add([int]$process.ProcessId)) {
      $added = $true
    }
  }
} while ($added)

$remaining = [Collections.Generic.HashSet[int]]::new($targetIds)
while ($remaining.Count -gt 0) {
  $leaves = @($remaining | Where-Object {
    $candidate = $_
    -not ($allProcesses | Where-Object {
      $remaining.Contains([int]$_.ProcessId) -and [int]$_.ParentProcessId -eq $candidate
    })
  })
  if (-not $leaves) { $leaves = @($remaining) }
  foreach ($processId in $leaves) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    [void]$remaining.Remove($processId)
  }
}
Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
if ($targetIds.Count -gt 0) {
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped proposal process tree: $($targetIds.Count) process(es)"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $HoldFile) | Out-Null
Set-Content -LiteralPath $HoldFile -Value "$(Get-Date -Format s) stopped by operator" -Encoding utf8
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Autostart hold set: the health check will not restart the bot until Start-Proposal-Bot runs."
