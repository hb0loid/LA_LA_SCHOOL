@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\runtime\Stop-Bot.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\runtime\Stop-Proposal-Bot.ps1"
if errorlevel 1 pause
