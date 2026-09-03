@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Set-Maintenance-Mode.ps1" -Mode off
pause
