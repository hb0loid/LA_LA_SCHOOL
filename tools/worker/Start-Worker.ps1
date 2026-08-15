$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $Root "worker_config.json"
$VersionPath = Join-Path $Root "worker_version.json"
$StatePath = Join-Path $Root ".worker_state.json"
$UpdatesDir = Join-Path $Root "updates"
$WorkDir = Join-Path $Root "runs\worker"
$LockPath = Join-Path $Root ".worker.lock"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

try {
  $WorkerLock = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch [IO.IOException] {
  exit 0
}

function Install-WorkerAutostart {
  $hiddenLauncher = Join-Path $Root "Start-Worker-Hidden.vbs"
  if (-not (Test-Path -LiteralPath $hiddenLauncher)) { return }
  try {
    $taskName = "LaLaDub Worker Autostart"
    $user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$hiddenLauncher`"" -WorkingDirectory $Root
    $logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $user
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
      -TaskName $taskName `
      -Description "Starts the LaLaDub laptop worker and checks it every five minutes." `
      -Action $action `
      -Trigger @($logonTrigger, $healthTrigger) `
      -Settings $settings `
      -User $user `
      -Force | Out-Null
  } catch {
    Write-Warning "Could not install worker autostart: $($_.Exception.Message)"
  }
}

Install-WorkerAutostart

function Read-WorkerConfig {
  if (-not (Test-Path -LiteralPath $ConfigPath)) {
    return @{}
  }
  try {
    $json = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8
    if (-not $json.Trim()) { return @{} }
    $object = ConvertFrom-Json -InputObject $json
    $result = @{}
    $object.PSObject.Properties | ForEach-Object { $result[$_.Name] = $_.Value }
    return $result
  } catch {
    Write-Warning "Could not read worker_config.json: $($_.Exception.Message)"
    return @{}
  }
}

function Write-WorkerConfig($Config) {
  $Config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
}

$config = Read-WorkerConfig
$server = $env:LALADUB_WORKER_SERVER
if (-not $server -and $config.ContainsKey("server")) { $server = [string]$config["server"] }
if (-not $server) { $server = Read-Host "Coordinator URL (example: http://127.0.0.1:8765)" }
if (-not $server) { $server = "http://127.0.0.1:8765" }

$token = $env:LALADUB_WORKER_TOKEN
if (-not $token -and $config.ContainsKey("token")) { $token = [string]$config["token"] }
if (-not $token) { $token = Read-Host "Worker token" }
if (-not $token) { throw "Worker token is required." }

$workerId = $env:LALADUB_WORKER_ID
if (-not $workerId -and $config.ContainsKey("worker_id")) { $workerId = [string]$config["worker_id"] }
if (-not $workerId) { $workerId = "$env:COMPUTERNAME-$env:USERNAME" }

if (-not (Test-Path -LiteralPath $ConfigPath)) {
  Write-WorkerConfig @{
    server = $server
    token = $token
    worker_id = $workerId
  }
}

function Read-JsonFile($Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return @{} }
  try {
    $json = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    if (-not $json.Trim()) { return @{} }
    $object = ConvertFrom-Json -InputObject $json
    $result = @{}
    $object.PSObject.Properties | ForEach-Object { $result[$_.Name] = $_.Value }
    return $result
  } catch {
    return @{}
  }
}

function Get-LocalBuildId {
  $version = Read-JsonFile $VersionPath
  if ($version.ContainsKey("build_id")) { return [string]$version["build_id"] }
  if ($version.ContainsKey("sha256")) { return [string]$version["sha256"] }
  return ""
}

function Invoke-WorkerUpdate {
  param(
    [string]$ServerUrl,
    [string]$WorkerToken
  )
  if ($env:LALADUB_WORKER_AUTO_UPDATE -eq "0") {
    return
  }
  $base = $ServerUrl.TrimEnd("/")
  $headers = @{ Authorization = "Bearer $WorkerToken" }
  try {
    $manifest = Invoke-RestMethod -Method Get -Uri "$base/api/v1/worker/manifest" -Headers $headers -TimeoutSec 20
  } catch {
    Write-Warning "Update check skipped: $($_.Exception.Message)"
    return
  }
  if (-not $manifest.available) {
    return
  }
  $remoteBuild = [string]$manifest.build_id
  if (-not $remoteBuild) { $remoteBuild = [string]$manifest.sha256 }
  if (-not $remoteBuild) { return }
  $localBuild = Get-LocalBuildId
  if ($localBuild -eq $remoteBuild) {
    return
  }

  Write-Host "Worker update found: $localBuild -> $remoteBuild"
  New-Item -ItemType Directory -Force -Path $UpdatesDir | Out-Null
  $zipPath = Join-Path $UpdatesDir ("worker-update-" + $remoteBuild + ".zip")
  $unpackPath = Join-Path $UpdatesDir ("unpack-" + $remoteBuild)
  if (Test-Path -LiteralPath $unpackPath) {
    Remove-Item -LiteralPath $unpackPath -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $unpackPath | Out-Null

  Invoke-WebRequest -Uri "$base/api/v1/worker/package" -Headers $headers -OutFile $zipPath -TimeoutSec 600
  if ($manifest.sha256) {
    $actualHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne ([string]$manifest.sha256).ToLowerInvariant()) {
      throw "Downloaded worker package hash mismatch."
    }
  }
  Expand-Archive -LiteralPath $zipPath -DestinationPath $unpackPath -Force
  Get-ChildItem -LiteralPath $unpackPath -Force | ForEach-Object {
    $name = $_.Name
    if ($name -in @("worker_config.json", "runs", "updates", ".worker_state.json")) {
      return
    }
    $destination = Join-Path $Root $name
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $destinationFull = [IO.Path]::GetFullPath($destination)
    if (-not $destinationFull.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
      throw "Refusing to install worker update outside worker root: $destinationFull"
    }
    if ($_.PSIsContainer -and (Test-Path -LiteralPath $destination)) {
      Remove-Item -LiteralPath $destination -Recurse -Force
    }
    Copy-Item -LiteralPath $_.FullName -Destination $destination -Recurse -Force
  }
  @{
    build_id = $remoteBuild
    sha256 = [string]$manifest.sha256
    updated_at = (Get-Date).ToString("o")
  } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatePath -Encoding UTF8
  Write-Host "Worker updated to $remoteBuild"
}

function Repair-PortableVenv {
  $venvDir = Join-Path $Root ".venv-f5tts"
  $venvCfg = Join-Path $venvDir "pyvenv.cfg"
  $portablePythonHome = Join-Path $Root "runtime\python311"
  $portablePython = Join-Path $portablePythonHome "python.exe"
  if (
    -not (Test-Path -LiteralPath $venvCfg) -or
    -not (Test-Path -LiteralPath $portablePython)
  ) {
    return
  }

  $version = "3.11.0"
  try {
    $cfg = Get-Content -LiteralPath $venvCfg -Raw -Encoding UTF8
    if ($cfg -match '(?m)^version\s*=\s*(.+?)\s*$') {
      $version = $Matches[1].Trim()
    }
  } catch {
    $version = "3.11.0"
  }

  @(
    "home = $portablePythonHome",
    "include-system-site-packages = false",
    "version = $version",
    "executable = $portablePython",
    "command = $portablePython -m venv $venvDir"
  ) | Set-Content -LiteralPath $venvCfg -Encoding UTF8

  $env:PATH = "$portablePythonHome;$env:PATH"
}

function Configure-ModelCaches {
  $portableCacheRoot = Join-Path $Root "models\cache"
  $userCacheRoot = Join-Path $env:USERPROFILE ".cache"
  $portableWhisperCache = Join-Path $portableCacheRoot "whisper"
  $userWhisperCache = Join-Path $userCacheRoot "whisper"
  $portableHfHome = Join-Path $portableCacheRoot "huggingface"
  $userHfHome = Join-Path $userCacheRoot "huggingface"

  if (-not $env:XDG_CACHE_HOME) {
    if (Test-Path -LiteralPath $portableWhisperCache) {
      $env:XDG_CACHE_HOME = $portableCacheRoot
    } elseif (Test-Path -LiteralPath $userWhisperCache) {
      $env:XDG_CACHE_HOME = $userCacheRoot
    } else {
      New-Item -ItemType Directory -Force -Path $portableCacheRoot | Out-Null
      $env:XDG_CACHE_HOME = $portableCacheRoot
    }
  }

  if (-not $env:HF_HOME) {
    if (Test-Path -LiteralPath $portableHfHome) {
      $env:HF_HOME = $portableHfHome
    } elseif (Test-Path -LiteralPath $userHfHome) {
      $env:HF_HOME = $userHfHome
    } else {
      New-Item -ItemType Directory -Force -Path $portableHfHome | Out-Null
      $env:HF_HOME = $portableHfHome
    }
  }

  if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
  }
}

if (-not $env:LALADUB_TRANSLATOR) { $env:LALADUB_TRANSLATOR = "hybrid" }
if (-not $env:LALADUB_TTS) { $env:LALADUB_TTS = "cosyvoice" }
if (-not $env:LALADUB_F5_PYTHON) { $env:LALADUB_F5_PYTHON = (Join-Path $Root ".venv-f5tts\Scripts\python.exe") }
if (-not $env:LALADUB_F5_MODEL) { $env:LALADUB_F5_MODEL = "F5TTS_v1_Base" }
if (-not $env:LALADUB_F5_HF_REPO) { $env:LALADUB_F5_HF_REPO = "Misha24-10/F5-TTS_RUSSIAN" }
if (-not $env:LALADUB_F5_HF_CKPT_PATH) { $env:LALADUB_F5_HF_CKPT_PATH = "F5TTS_v1_Base_v2/model_last_inference.safetensors" }
if (-not $env:LALADUB_F5_HF_VOCAB_PATH) { $env:LALADUB_F5_HF_VOCAB_PATH = "F5TTS_v1_Base/vocab.txt" }
if (-not $env:LALADUB_F5_CACHE_DIR) { $env:LALADUB_F5_CACHE_DIR = (Join-Path $Root "models\f5tts") }
if (-not $env:LALADUB_F5_DEVICE) { $env:LALADUB_F5_DEVICE = "auto" }
if (-not $env:LALADUB_F5_SPEED) { $env:LALADUB_F5_SPEED = "1.0" }
if (-not $env:LALADUB_F5_NFE_STEP) { $env:LALADUB_F5_NFE_STEP = "32" }
if (-not $env:LALADUB_F5_CFG_STRENGTH) { $env:LALADUB_F5_CFG_STRENGTH = "2.0" }
if (-not $env:LALADUB_F5_TARGET_RMS) { $env:LALADUB_F5_TARGET_RMS = "0.1" }
if (-not $env:LALADUB_F5_CROSS_FADE_DURATION) { $env:LALADUB_F5_CROSS_FADE_DURATION = "0.15" }
if (-not $env:LALADUB_F5_REMOVE_SILENCE) { $env:LALADUB_F5_REMOVE_SILENCE = "0" }
if (-not $env:LALADUB_F5_TIMEOUT_SECONDS) { $env:LALADUB_F5_TIMEOUT_SECONDS = "1800" }
if (-not $env:LALADUB_MULTI_SPEAKER) { $env:LALADUB_MULTI_SPEAKER = "1" }
if (-not $env:LALADUB_SPEAKER_REFERENCE_SECONDS) { $env:LALADUB_SPEAKER_REFERENCE_SECONDS = "5.0" }
if (-not $env:LALADUB_SPEAKER_CLUSTERING) { $env:LALADUB_SPEAKER_CLUSTERING = "1" }
if (-not $env:LALADUB_MAX_SPEAKER_CLUSTERS) { $env:LALADUB_MAX_SPEAKER_CLUSTERS = "9" }
if (-not $env:LALADUB_SPEAKER_CLUSTER_THRESHOLD) { $env:LALADUB_SPEAKER_CLUSTER_THRESHOLD = "0.08" }
if (-not $env:LALADUB_DIARIZATION_PYTHON) { $env:LALADUB_DIARIZATION_PYTHON = (Join-Path $Root ".venv-diarization\Scripts\python.exe") }
if (-not $env:LALADUB_DIARIZATION_MODEL) { $env:LALADUB_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1" }
if (-not $env:LALADUB_DIARIZATION_DEVICE) { $env:LALADUB_DIARIZATION_DEVICE = "auto" }
if (-not $env:LALADUB_DIARIZATION_CACHE_DIR) { $env:LALADUB_DIARIZATION_CACHE_DIR = (Join-Path $Root "models\diarization") }
if (-not $env:LALADUB_DIARIZATION_TOKEN_FILE) { $env:LALADUB_DIARIZATION_TOKEN_FILE = (Join-Path $Root ".secrets\HuggingFace-Token.txt") }
if (-not $env:LALADUB_DIARIZATION_TIMEOUT_SECONDS) { $env:LALADUB_DIARIZATION_TIMEOUT_SECONDS = "1800" }
if (-not $env:LALADUB_SEPARATION) { $env:LALADUB_SEPARATION = "demucs" }
if (-not $env:LALADUB_SEPARATION_DEVICE) { $env:LALADUB_SEPARATION_DEVICE = "cpu" }
if (-not $env:LALADUB_AUDIO_BED) { $env:LALADUB_AUDIO_BED = "instrumental" }
if (-not $env:LALADUB_ORIGINAL_VOLUME) { $env:LALADUB_ORIGINAL_VOLUME = "1.0" }
if (-not $env:LALADUB_COLLAPSE_REPETITIONS) { $env:LALADUB_COLLAPSE_REPETITIONS = "1" }
if (-not $env:LALADUB_MAX_PHRASE_REPEATS) { $env:LALADUB_MAX_PHRASE_REPEATS = "2" }
if (-not $env:LALADUB_MAX_WORD_REPEATS) { $env:LALADUB_MAX_WORD_REPEATS = "3" }
if (-not $env:LALADUB_INJECT_ARTIFACTS) { $env:LALADUB_INJECT_ARTIFACTS = "1" }
if (-not $env:LALADUB_ARTIFACT_MAX_SEGMENTS) { $env:LALADUB_ARTIFACT_MAX_SEGMENTS = "14" }
if (-not $env:LALADUB_ARTIFACT_RATIO) { $env:LALADUB_ARTIFACT_RATIO = "0.20" }
if (-not $env:LALADUB_ARTIFACT_MIN_SOURCE_SEGMENTS) { $env:LALADUB_ARTIFACT_MIN_SOURCE_SEGMENTS = "5" }
if (-not $env:LALADUB_ARTIFACT_MIN_GAP_SECONDS) { $env:LALADUB_ARTIFACT_MIN_GAP_SECONDS = "0.5" }
if (-not $env:LALADUB_DISTORT_TRANSLATION) { $env:LALADUB_DISTORT_TRANSLATION = "1" }
if (-not $env:LALADUB_TRANSLATION_PIVOTS) { $env:LALADUB_TRANSLATION_PIVOTS = "input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar|input,en,de|input,ja,ko,en|input,tr,ar,en|en,ms,he,en" }
if (-not $env:LALADUB_TRANSLATION_SECOND_PASS_RATIO) { $env:LALADUB_TRANSLATION_SECOND_PASS_RATIO = "0.45" }
if (-not $env:LALADUB_ASR_BACKEND) { $env:LALADUB_ASR_BACKEND = "faster-whisper" }
if (-not $env:LALADUB_DEFAULT_ASR_METHOD) { $env:LALADUB_DEFAULT_ASR_METHOD = "ow-large-v3-chaos-backbone" }
if (-not $env:LALADUB_WHISPER_DEVICE) { $env:LALADUB_WHISPER_DEVICE = "cpu" }
if (-not $env:LALADUB_WHISPER_COMPUTE_TYPE) { $env:LALADUB_WHISPER_COMPUTE_TYPE = "int8" }
if (-not $env:LALADUB_WHISPER_ONLY_MODEL) { $env:LALADUB_WHISPER_ONLY_MODEL = "turbo" }
if (-not $env:LALADUB_WHISPER_ONLY_DEVICE) { $env:LALADUB_WHISPER_ONLY_DEVICE = "cuda" }
if (-not $env:LALADUB_ARTIFACT_WHISPER_DEVICE) { $env:LALADUB_ARTIFACT_WHISPER_DEVICE = "cuda" }
if (-not $env:LALADUB_SUPPRESS_PLAIN_ASCII_TOKENS) { $env:LALADUB_SUPPRESS_PLAIN_ASCII_TOKENS = "0" }
if (-not $env:LALADUB_WATERMARK_IMAGE) { $env:LALADUB_WATERMARK_IMAGE = (Join-Path $Root "assets\watermark.png") }
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = (Join-Path $Root "src") }
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }

Repair-PortableVenv
Configure-ModelCaches

$python = Join-Path $Root ".venv-f5tts\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  $python = "python"
}

Write-Host "Starting worker $workerId -> $server"
while ($true) {
  Invoke-WorkerUpdate -ServerUrl $server -WorkerToken $token
  & $python -m laladub.worker --server $server --token $token --worker-id $workerId --workdir $WorkDir
  $exitCode = $LASTEXITCODE
  if ($exitCode -eq 42) {
    Write-Host "Worker requested update restart."
    Start-Sleep -Seconds 2
    continue
  }
  exit $exitCode
}
