$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$HiddenLauncher = Join-Path $Root "Start-Worker-Hidden.vbs"
if (-not (Test-Path -LiteralPath $HiddenLauncher)) { exit 0 }

$TaskName = "LaLaDub Worker Autostart"
$User = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$HiddenLauncher`"" -WorkingDirectory $Root
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$logonTrigger.Delay = "PT30S"
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
  -ExecutionTimeLimit ([TimeSpan]::Zero)
$settings.Hidden = $true

Register-ScheduledTask `
  -TaskName $TaskName `
  -Description "Starts the LaLaDub laptop worker and checks it every five minutes." `
  -Action $action `
  -Trigger @($logonTrigger, $healthTrigger) `
  -Settings $settings `
  -User $User `
  -Force | Out-Null
