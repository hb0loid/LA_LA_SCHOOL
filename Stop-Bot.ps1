$ErrorActionPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutLog = Join-Path $Root "bot.out.log"
$PidFile = Join-Path $Root "bot.pid"

Set-Location -LiteralPath $Root
New-Item -ItemType File -Force -Path $OutLog | Out-Null

$stopped = $false
if (Test-Path -LiteralPath $PidFile) {
  $pidText = Get-Content -LiteralPath $PidFile | Select-Object -First 1
  if ($pidText) {
    $process = Get-Process -Id ([int]$pidText)
    if ($process) {
      Stop-Process -Id $process.Id -Force
      Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped pid=$pidText"
      $stopped = $true
    }
  }
  Remove-Item -LiteralPath $PidFile -Force
}

$matches = Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like "*laladub.bot*" -and
    $_.CommandLine -like "*$Root*" -and
    $_.CommandLine -like "*--instance test*"
  }

foreach ($match in $matches) {
  Stop-Process -Id $match.ProcessId -Force
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped extra pid=$($match.ProcessId)"
  $stopped = $true
}

if (-not $stopped) {
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Bot was not running."
}
