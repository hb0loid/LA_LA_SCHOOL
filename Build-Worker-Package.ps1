param(
  [ValidateSet("Both", "Portable", "UpdateOnly")]
  [string]$Mode = "Both",
  [string]$Server = "http://MAIN_PC_IP:8765",
  [string]$WorkerToken = "",
  [string]$BuildId = "",
  [switch]$SkipRuntime,
  [switch]$SkipModels,
  [switch]$NoPortableZip
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistDir = Join-Path $Root "dist"
$PortableDir = Join-Path $DistDir "LaLaDubWorker"
$PortableZip = Join-Path $DistDir "LaLaDubWorker.zip"
$UpdateDir = Join-Path $DistDir "LaLaDubWorker-update"
$UpdateZip = Join-Path $DistDir "LaLaDubWorker-update.zip"
$UpdateManifest = Join-Path $DistDir "LaLaDubWorker-update.manifest.json"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

if (-not $BuildId) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $git = ""
  try {
    $git = (& git rev-parse --short HEAD 2>$null).Trim()
  } catch {
    $git = ""
  }
  $BuildId = if ($git) { "$stamp-$git" } else { $stamp }
}

$Version = "0.1.0"
try {
  $pyproject = Get-Content -LiteralPath (Join-Path $Root "pyproject.toml") -Raw -Encoding UTF8
  if ($pyproject -match '(?m)^version\s*=\s*"([^"]+)"') {
    $Version = $Matches[1]
  }
} catch {
  $Version = "0.1.0"
}

if (-not $WorkerToken) {
  $tokenPath = Join-Path $Root ".secrets\Worker-Api-Token.txt"
  if (Test-Path -LiteralPath $tokenPath) {
    $WorkerToken = (Get-Content -LiteralPath $tokenPath -Raw -Encoding UTF8).Trim()
  }
}

if ($Server -eq "http://MAIN_PC_IP:8765") {
  try {
    $addresses = Get-NetIPAddress -AddressFamily IPv4 |
      Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" }
    $preferredWifi = $addresses |
      Where-Object {
        $_.InterfaceAlias -match "Wi-?Fi|Wireless|Беспровод" -and
        $_.IPAddress -match '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)'
      } |
      Select-Object -First 1 -ExpandProperty IPAddress
    $privateIps = $addresses |
      Select-Object -ExpandProperty IPAddress |
      Where-Object { $_ -match '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' } |
      Select-Object -First 1
    $ip = if ($preferredWifi) { $preferredWifi } else { $privateIps }
    if (-not $ip) {
      $ip = $addresses | Select-Object -First 1 -ExpandProperty IPAddress
    }
    if ($ip) {
      $Server = "http://${ip}:8765"
    }
  } catch {
    $Server = "http://MAIN_PC_IP:8765"
  }
}

function Reset-Dir($Path) {
  if (Test-Path -LiteralPath $Path) {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Copy-CoreFiles($StageDir, [bool]$IncludeExampleConfig) {
  $files = @(
    "Start-Worker.cmd",
    "Start-Worker.ps1",
    "README.md",
    "pyproject.toml"
  )
  foreach ($file in $files) {
    $source = Join-Path $Root $file
    if (Test-Path -LiteralPath $source) {
      Copy-Item -LiteralPath $source -Destination (Join-Path $StageDir $file) -Force
    }
  }

  $dirs = @("src", "tools", "assets")
  foreach ($dir in $dirs) {
    $source = Join-Path $Root $dir
    if (Test-Path -LiteralPath $source) {
      Copy-Item -LiteralPath $source -Destination (Join-Path $StageDir $dir) -Recurse -Force
    }
  }

  @{
    version = $Version
    build_id = $BuildId
    built_at = (Get-Date).ToString("o")
  } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $StageDir "worker_version.json") -Encoding UTF8

  if ($IncludeExampleConfig) {
    $config = @{
      server = $Server
      token = $(if ($WorkerToken) { $WorkerToken } else { "PASTE_WORKER_TOKEN_HERE" })
      worker_id = "worker-pc"
    }
    $config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $StageDir "worker_config.example.json") -Encoding UTF8
    if ($WorkerToken) {
      $config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $StageDir "worker_config.json") -Encoding UTF8
    }
  }
}

function Compress-Stage($StageDir, $ZipPath) {
  if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
  }
  Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ZipPath -Force
}

function Get-VenvPythonHome($VenvDir) {
  $cfgPath = Join-Path $VenvDir "pyvenv.cfg"
  if (-not (Test-Path -LiteralPath $cfgPath)) {
    return ""
  }
  try {
    $cfg = Get-Content -LiteralPath $cfgPath -Raw -Encoding UTF8
    if ($cfg -match '(?m)^home\s*=\s*(.+?)\s*$') {
      return $Matches[1].Trim()
    }
  } catch {
    return ""
  }
  return ""
}

function Copy-PortablePythonRuntime($StageDir, $VenvDir) {
  $pythonHome = Get-VenvPythonHome $VenvDir
  if (-not $pythonHome -or -not (Test-Path -LiteralPath $pythonHome)) {
    Write-Warning "Could not find base Python from .venv-f5tts. Worker may require Python 3.11 on target PC."
    return
  }
  $destParent = Join-Path $StageDir "runtime"
  $dest = Join-Path $destParent "python311"
  New-Item -ItemType Directory -Force -Path $destParent | Out-Null
  if (Test-Path -LiteralPath $dest) {
    Remove-Item -LiteralPath $dest -Recurse -Force
  }
  Write-Host "Including base Python runtime: $pythonHome"
  Copy-Item -LiteralPath $pythonHome -Destination $dest -Recurse -Force
}

function Copy-ModelCaches($StageDir) {
  $cacheRoot = Join-Path $StageDir "models\cache"
  $whisperDest = Join-Path $cacheRoot "whisper"
  $hfDest = Join-Path $cacheRoot "huggingface\hub"
  New-Item -ItemType Directory -Force -Path $whisperDest | Out-Null
  New-Item -ItemType Directory -Force -Path $hfDest | Out-Null

  $userCache = Join-Path $env:USERPROFILE ".cache"
  $whisperCache = Join-Path $userCache "whisper"
  $openAiModels = @("large-v3.pt")
  foreach ($model in $openAiModels) {
    $source = Join-Path $whisperCache $model
    if (Test-Path -LiteralPath $source) {
      Write-Host "Including OpenAI Whisper cache: $model"
      Copy-Item -LiteralPath $source -Destination (Join-Path $whisperDest $model) -Force
    } else {
      Write-Warning "OpenAI Whisper cache not found: $source"
    }
  }

  $hfHub = Join-Path $userCache "huggingface\hub"
  $hfModels = @("models--Systran--faster-whisper-small")
  foreach ($modelDir in $hfModels) {
    $source = Join-Path $hfHub $modelDir
    if (Test-Path -LiteralPath $source) {
      Write-Host "Including HuggingFace cache: $modelDir"
      Copy-Item -LiteralPath $source -Destination (Join-Path $hfDest $modelDir) -Recurse -Force
    } else {
      Write-Warning "HuggingFace cache not found: $source"
    }
  }
}

function Build-PortablePackage {
  Reset-Dir $PortableDir
  Copy-CoreFiles $PortableDir $true

  if (-not $SkipRuntime) {
    $runtime = Join-Path $Root ".venv-f5tts"
    if (Test-Path -LiteralPath $runtime) {
      Write-Host "Including .venv-f5tts. This can make the package large."
      Copy-Item -LiteralPath $runtime -Destination (Join-Path $PortableDir ".venv-f5tts") -Recurse -Force
      Copy-PortablePythonRuntime $PortableDir $runtime
    } else {
      Write-Warning ".venv-f5tts not found. Portable package will require Python/dependencies on target PC."
    }
  }

  if (-not $SkipModels) {
    $models = Join-Path $Root "models"
    if (Test-Path -LiteralPath $models) {
      Write-Host "Including models. This can make the package very large."
      Copy-Item -LiteralPath $models -Destination (Join-Path $PortableDir "models") -Recurse -Force
    } else {
      Write-Warning "models directory not found. Worker will download or fail depending on model settings."
    }
    Copy-ModelCaches $PortableDir
  }

  if (-not $NoPortableZip) {
    Compress-Stage $PortableDir $PortableZip
  } elseif (Test-Path -LiteralPath $PortableZip) {
    Remove-Item -LiteralPath $PortableZip -Force
  }
  Write-Host "Portable worker folder: $PortableDir"
  if (-not $NoPortableZip) {
    Write-Host "Portable worker zip: $PortableZip"
  }
}

function Build-UpdatePackage {
  Reset-Dir $UpdateDir
  Copy-CoreFiles $UpdateDir $false
  Compress-Stage $UpdateDir $UpdateZip
  $hash = (Get-FileHash -LiteralPath $UpdateZip -Algorithm SHA256).Hash.ToLowerInvariant()
  $size = (Get-Item -LiteralPath $UpdateZip).Length
  @{
    version = $Version
    build_id = $BuildId
    built_at = (Get-Date).ToString("o")
    filename = (Split-Path -Leaf $UpdateZip)
    sha256 = $hash
    size = $size
  } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $UpdateManifest -Encoding UTF8
  Write-Host "Update worker folder: $UpdateDir"
  Write-Host "Update worker zip: $UpdateZip"
  Write-Host "Update manifest: $UpdateManifest"
}

if ($Mode -in @("Both", "Portable")) {
  Build-PortablePackage
}
if ($Mode -in @("Both", "UpdateOnly")) {
  Build-UpdatePackage
}
