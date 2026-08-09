Option Explicit

Dim shell, fileSystem, scriptDirectory, startScript, command
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

scriptDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
startScript = fileSystem.BuildPath(scriptDirectory, "Start-Worker.ps1")
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & startScript & """"

shell.Run command, 0, False
