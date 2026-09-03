$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$ServicesLauncher = Join-Path $PSScriptRoot "Start-Usual-Services.ps1"
$TaskName = "LaLaSchool Release Bot Autostart"

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ServicesLauncher`"" `
  -WorkingDirectory $Root

$startupTrigger = New-ScheduledTaskTrigger -AtStartup
$startupTrigger.Delay = "PT1M"

$healthTrigger = New-ScheduledTaskTrigger `
  -Once `
  -At ((Get-Date).AddMinutes(1)) `
  -RepetitionInterval (New-TimeSpan -Minutes 5) `
  -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet `
  -MultipleInstances IgnoreNew `
  -StartWhenAvailable `
  -DontStopIfGoingOnBatteries `
  -AllowStartIfOnBatteries `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
$settings.Hidden = $true
$settings.RunOnlyIfNetworkAvailable = $true

Register-ScheduledTask `
  -TaskName $TaskName `
  -Description "Starts the usual local bots and configured sites after Windows boot, before user logon, and checks them every five minutes." `
  -Action $action `
  -Trigger @($startupTrigger, $healthTrigger) `
  -Settings $settings `
  -User "SYSTEM" `
  -RunLevel Highest `
  -Force | Out-Null

Write-Host "Autostart task updated: $TaskName"
