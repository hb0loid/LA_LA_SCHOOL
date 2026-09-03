Option Explicit

Dim shell, fileSystem, scriptDirectory, startScript, proposalScript, command
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

scriptDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
startScript = fileSystem.BuildPath(scriptDirectory, "Start-Bot.ps1")
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & startScript & """"

shell.Run command, 0, False

proposalScript = fileSystem.BuildPath(scriptDirectory, "Start-Proposal-Bot.ps1")
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & proposalScript & """"
shell.Run command, 0, False
