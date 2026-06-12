@echo off
cd /d "%~dp0"
if not exist bot.out.log type nul > bot.out.log
if not exist bot.err.log type nul > bot.err.log
start "" notepad.exe bot.out.log
start "" notepad.exe bot.err.log
