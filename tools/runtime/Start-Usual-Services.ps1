param(
  [ValidateRange(0, 300)]
  [int]$StartupDelaySeconds = 0
)

$ErrorActionPreference = "Continue"

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir "autostart-services.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-AutostartLog {
  param([string]$Message)
  Add-Content -LiteralPath $LogFile -Value "$(Get-Date -Format s) $Message" -ErrorAction SilentlyContinue
}

function Invoke-ServiceLauncher {
  param(
    [string]$Name,
    [string]$Script,
    [string[]]$Arguments = @()
  )

  if (-not (Test-Path -LiteralPath $Script)) {
    Write-AutostartLog "Skipped ${Name}: launcher is missing: $Script"
    return
  }

  try {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $Script @Arguments
    if ($LASTEXITCODE -eq 0) {
      Write-AutostartLog "Checked ${Name}: running."
    } else {
      Write-AutostartLog "Failed ${Name}: launcher exit code $LASTEXITCODE."
    }
  } catch {
    Write-AutostartLog "Failed ${Name}: $($_.Exception.Message)"
  }
}

if ($StartupDelaySeconds -gt 0) {
  Start-Sleep -Seconds $StartupDelaySeconds
}

Write-AutostartLog "Autostart health check began as $([Security.Principal.WindowsIdentity]::GetCurrent().Name)."

Invoke-ServiceLauncher `
  -Name "La La School release bot" `
  -Script (Join-Path $Root "tools\runtime\Start-Bot.ps1")

Invoke-ServiceLauncher `
  -Name "La La School proposal bot" `
  -Script (Join-Path $Root "tools\runtime\Start-Proposal-Bot.ps1")

$tgAllRoot = "C:\Users\HBoloid\Documents\TG ALL BOT"
if (
  (Test-Path -LiteralPath (Join-Path $tgAllRoot ".env")) -and
  (Test-Path -LiteralPath (Join-Path $tgAllRoot ".venv\Scripts\python.exe"))
) {
  Invoke-ServiceLauncher `
    -Name "TG ALL BOT" `
    -Script (Join-Path $tgAllRoot "Start-Bot.ps1") `
    -Arguments @("-NoPopup")
} else {
  Write-AutostartLog "Skipped TG ALL BOT: .env or Python environment is missing."
}

$discordRoot = "C:\Users\HBoloid\Documents\бот дс"
$discordConfigured = (
  (Test-Path -LiteralPath (Join-Path $discordRoot ".env")) -and
  (Test-Path -LiteralPath (Join-Path $discordRoot "node_modules")) -and
  (Test-Path -LiteralPath (Join-Path $discordRoot "src\index.js"))
)
if ($discordConfigured) {
  $discordRunning = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -eq "node.exe" -and ([string]$_.CommandLine).Contains($discordRoot)
  }
  if (-not $discordRunning) {
    try {
      Start-Process `
        -FilePath "C:\Program Files\nodejs\node.exe" `
        -ArgumentList "src\index.js" `
        -WorkingDirectory $discordRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $discordRoot "bot.out.log") `
        -RedirectStandardError (Join-Path $discordRoot "bot.err.log")
      Write-AutostartLog "Started Discord cinema bot."
    } catch {
      Write-AutostartLog "Failed Discord cinema bot: $($_.Exception.Message)"
    }
  } else {
    Write-AutostartLog "Checked Discord cinema bot: running."
  }
} else {
  Write-AutostartLog "Skipped Discord cinema bot: .env or runtime files are missing."
}

# This site copy is started only when it is actually configured on this PC.
$furdleRoot = "C:\Users\HBoloid\Documents\Codex\backups\furdle_2026-07-02_09-11-01"
$furdleLauncher = Join-Path $furdleRoot "Start-Public-Beta.ps1"
$furdleReady = (
  (Test-Path -LiteralPath (Join-Path $furdleRoot ".env")) -and
  (Test-Path -LiteralPath (Join-Path $furdleRoot "node_modules")) -and
  (Test-Path -LiteralPath (Join-Path $furdleRoot "dist\index.html"))
)
$furdleListening = Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue
if ($furdleReady -and -not $furdleListening -and (Test-Path -LiteralPath $furdleLauncher)) {
  try {
    Start-Process `
      -FilePath "powershell.exe" `
      -ArgumentList "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$furdleLauncher`"" `
      -WorkingDirectory $furdleRoot `
      -WindowStyle Hidden
    Write-AutostartLog "Started configured Furdle site."
  } catch {
    Write-AutostartLog "Failed Furdle site: $($_.Exception.Message)"
  }
} elseif (-not $furdleReady) {
  Write-AutostartLog "Skipped Furdle site: this PC has only an unconfigured backup copy."
}

Write-AutostartLog "Autostart health check finished."
