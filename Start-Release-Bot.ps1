$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutLog = Join-Path $Root "bot.release.out.log"
$ErrLog = Join-Path $Root "bot.release.err.log"
$PidFile = Join-Path $Root "bot.release.pid"
$WatchdogScript = Join-Path $Root "Run-Bot-Watchdog.ps1"
$TokenFile = Join-Path $Root ".secrets\Release-Bot-Token.ps1"
$WorkerTokenFile = Join-Path $Root ".secrets\Worker-Api-Token.txt"
$WorkDir = Join-Path $Root "runs\bot-release"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
if (-not (Test-Path -LiteralPath $OutLog)) { New-Item -ItemType File -Path $OutLog | Out-Null }
if (-not (Test-Path -LiteralPath $ErrLog)) { New-Item -ItemType File -Path $ErrLog | Out-Null }

if (Test-Path -LiteralPath $PidFile) {
  $existingPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existingProcess) {
      if (
        $existingProcess.CommandLine -like "*Run-Bot-Watchdog.ps1*" -and
        $existingProcess.CommandLine -like "*-Instance release*" -and
        $existingProcess.CommandLine -like "*$Root*"
      ) {
        Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Release watchdog is already running, pid=$existingPid" -ErrorAction SilentlyContinue
        exit 0
      }
      if (
        $existingProcess.CommandLine -like "*laladub.bot*" -and
        $existingProcess.CommandLine -like "*--instance release*"
      ) {
        Stop-Process -Id ([int]$existingPid) -Force
        Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Stopped legacy direct release bot pid=$existingPid before starting watchdog" -ErrorAction SilentlyContinue
      } else {
        Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Ignoring stale release pid=$existingPid before starting watchdog" -ErrorAction SilentlyContinue
      }
    }
  }
  Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

$existingWatchdog = Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like "*Run-Bot-Watchdog.ps1*" -and
    $_.CommandLine -like "*-Instance release*" -and
    $_.CommandLine -like "*$Root*"
  } |
  Select-Object -First 1
if ($existingWatchdog) {
  Set-Content -LiteralPath $PidFile -Value $existingWatchdog.ProcessId
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Release watchdog is already running, pid=$($existingWatchdog.ProcessId)" -ErrorAction SilentlyContinue
  exit 0
}

if (-not (Test-Path -LiteralPath $TokenFile)) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Release token file is missing: $TokenFile" -ErrorAction SilentlyContinue
  exit 1
}

. $TokenFile
if (-not $ReleaseBotToken) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Release bot token is empty." -ErrorAction SilentlyContinue
  exit 1
}

if (-not (Test-Path -LiteralPath $WorkerTokenFile)) {
  $bytes = [byte[]]::new(32)
  $rng = [Security.Cryptography.RNGCryptoServiceProvider]::new()
  try {
    $rng.GetBytes($bytes)
  } finally {
    $rng.Dispose()
  }
  $workerToken = ([BitConverter]::ToString($bytes) -replace "-", "").ToLowerInvariant()
  Set-Content -LiteralPath $WorkerTokenFile -Value $workerToken -Encoding UTF8
} else {
  $workerToken = (Get-Content -LiteralPath $WorkerTokenFile -Raw -Encoding UTF8).Trim()
}
if (-not $workerToken) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Worker API token is empty: $WorkerTokenFile" -ErrorAction SilentlyContinue
  exit 1
}

function Set-LaLaDubModelCaches {
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

Set-LaLaDubModelCaches

$env:LALADUB_BOT_TOKEN = $ReleaseBotToken
$paidUsers = [Environment]::GetEnvironmentVariable("LALADUB_PAID_USERS", "User")
if ($paidUsers) { $env:LALADUB_PAID_USERS = $paidUsers }
$env:LALADUB_BOT_WORKDIR = $WorkDir
$env:LALADUB_EXECUTOR_MODE = "hybrid"
$env:LALADUB_MAX_LOCAL_JOBS = "1"
$env:LALADUB_WORKER_API_HOST = "0.0.0.0"
$env:LALADUB_WORKER_API_PORT = "8765"
$workerApiPort = $env:LALADUB_WORKER_API_PORT
try {
  $firewallRuleName = "LaLaDub Worker API 8765"
  $existingFirewallRule = Get-NetFirewallRule -DisplayName $firewallRuleName -ErrorAction SilentlyContinue
  if ($existingFirewallRule) {
    Set-NetFirewallRule -DisplayName $firewallRuleName -Enabled True -Direction Inbound -Action Allow -Profile Any -ErrorAction Stop | Out-Null
  } else {
    New-NetFirewallRule `
      -DisplayName $firewallRuleName `
      -Direction Inbound `
      -Action Allow `
      -Protocol TCP `
      -LocalPort $workerApiPort `
      -Profile Any `
      -ErrorAction Stop | Out-Null
  }
} catch {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Could not ensure worker API firewall rule: $($_.Exception.Message)" -ErrorAction SilentlyContinue
}
$env:LALADUB_WORKER_API_TOKEN = $workerToken
$env:LALADUB_WORKER_PACKAGE_PATH = (Join-Path $Root "dist\LaLaDubWorker-update.zip")
$env:LALADUB_WORKER_PACKAGE_MANIFEST = (Join-Path $Root "dist\LaLaDubWorker-update.manifest.json")
$env:LALADUB_JOB_RETENTION_SECONDS = "86400"
$env:LALADUB_CLEANUP_INTERVAL_SECONDS = "3600"
$env:LALADUB_TTS = "f5"
$env:LALADUB_TRANSLATOR = "hybrid"
$env:LALADUB_MAX_ACTIVE_JOBS = "2"
$env:LALADUB_MAX_ACTIVE_JOBS_PER_USER = "1"
$env:LALADUB_FREE_MAX_DURATION_SECONDS = "180"
$env:LALADUB_PAID_MAX_DURATION_SECONDS = "0"
$env:LALADUB_WATERMARK_IMAGE = (Join-Path $Root "assets\watermark.png")
$env:LALADUB_VOICE = "Microsoft Irina Desktop"
$env:LALADUB_XTTS_DEVICE = "cpu"
$env:LALADUB_F5_PYTHON = (Join-Path $Root ".venv-f5tts\Scripts\python.exe")
$env:LALADUB_F5_MODEL = "F5TTS_v1_Base"
$env:LALADUB_F5_HF_REPO = "Misha24-10/F5-TTS_RUSSIAN"
$env:LALADUB_F5_HF_CKPT_PATH = "F5TTS_v1_Base_v2/model_last_inference.safetensors"
$env:LALADUB_F5_HF_VOCAB_PATH = "F5TTS_v1_Base/vocab.txt"
$env:LALADUB_F5_CACHE_DIR = (Join-Path $Root "models\f5tts")
$env:LALADUB_F5_DEVICE = "auto"
$env:LALADUB_F5_SPEED = "1.0"
$env:LALADUB_F5_NFE_STEP = "32"
$env:LALADUB_F5_CFG_STRENGTH = "2.0"
$env:LALADUB_F5_TARGET_RMS = "0.1"
$env:LALADUB_F5_CROSS_FADE_DURATION = "0.15"
$env:LALADUB_F5_REMOVE_SILENCE = "0"
$env:LALADUB_F5_TIMEOUT_SECONDS = "1800"
$env:LALADUB_MULTI_SPEAKER = "1"
$env:LALADUB_SPEAKER_REFERENCE_SECONDS = "5.0"
$env:LALADUB_SPEAKER_CLUSTERING = "0"
$env:LALADUB_MAX_SPEAKER_CLUSTERS = "6"
$env:LALADUB_SPEAKER_CLUSTER_THRESHOLD = "0.08"
$env:LALADUB_SEPARATION = "demucs"
$env:LALADUB_SEPARATION_DEVICE = "cpu"
$env:LALADUB_AUDIO_BED = "instrumental"
$env:LALADUB_ORIGINAL_VOLUME = "1.0"
$env:LALADUB_COLLAPSE_REPETITIONS = "1"
$env:LALADUB_MAX_PHRASE_REPEATS = "2"
$env:LALADUB_MAX_WORD_REPEATS = "3"
$env:LALADUB_INJECT_ARTIFACTS = "1"
$env:LALADUB_ARTIFACT_MAX_SEGMENTS = "12"
$env:LALADUB_ARTIFACT_MIN_GAP_SECONDS = "0.5"
$env:LALADUB_DISTORT_TRANSLATION = "1"
$env:LALADUB_TRANSLATION_PIVOTS = "input,en|en,de|en,fr|en,es|input,en,de|en,es,de"
$env:LALADUB_ASR_BACKEND = "faster-whisper"
$env:LALADUB_DEFAULT_ASR_METHOD = "ow-large-v3-chaos-backbone"
$env:LALADUB_WHISPER_DEVICE = "cpu"
$env:LALADUB_WHISPER_COMPUTE_TYPE = "int8"
$env:LALADUB_WHISPER_ONLY_MODEL = "large-v3"
$env:LALADUB_WHISPER_ONLY_DEVICE = "cuda"
$env:LALADUB_SUPPRESS_PLAIN_ASCII_TOKENS = "0"
$env:PYTHONPATH = (Join-Path $Root "src")
$env:PYTHONIOENCODING = "utf-8"

$python = Join-Path $Root ".venv-f5tts\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) F5/CUDA Python is missing: $python" -ErrorAction SilentlyContinue
  exit 1
}
$python = (Resolve-Path -LiteralPath $python).Path
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Starting La La Dub Release Watchdog..." -ErrorAction SilentlyContinue
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Python: $python" -ErrorAction SilentlyContinue
$watchdogArguments = (
  "-NoProfile " +
  "-ExecutionPolicy Bypass " +
  "-File `"$WatchdogScript`" " +
  "-Root `"$Root`" " +
  "-Instance release " +
  "-OutLog `"$OutLog`" " +
  "-ErrLog `"$ErrLog`" " +
  "-Python `"$python`" " +
  "-RestartDelaySeconds 8"
)
$process = Start-Process `
  -FilePath "powershell.exe" `
  -ArgumentList $watchdogArguments `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -PassThru

Set-Content -LiteralPath $PidFile -Value $process.Id
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Started release watchdog pid=$($process.Id)" -ErrorAction SilentlyContinue
