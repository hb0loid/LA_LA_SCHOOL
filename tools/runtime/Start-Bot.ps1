param(
  [ValidateRange(0, 300)]
  [int]$StartupDelaySeconds = 0
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$RuntimeDir = Join-Path $Root "work\runtime"
$LogDir = Join-Path $Root "logs"
$OutLog = Join-Path $LogDir "bot.out.log"
$ErrLog = Join-Path $LogDir "bot.err.log"
$PidFile = Join-Path $RuntimeDir "bot.pid"
$WatchdogScript = Join-Path $PSScriptRoot "Watchdog.ps1"
$TokenFile = Join-Path $Root ".secrets\Release-Bot-Token.ps1"
$WorkerTokenFile = Join-Path $Root ".secrets\Worker-Api-Token.txt"
$WorkDir = Join-Path $Root "runs\bot-release"
$HoldFile = Join-Path $RuntimeDir "bot.hold"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $WorkDir,$RuntimeDir,$LogDir | Out-Null
# Starting on purpose clears the "stopped by operator" hold that Stop-Bot left,
# so the autostart health check resumes keeping the bot alive.
Remove-Item -LiteralPath $HoldFile -Force -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $OutLog)) { New-Item -ItemType File -Path $OutLog | Out-Null }
if (-not (Test-Path -LiteralPath $ErrLog)) { New-Item -ItemType File -Path $ErrLog | Out-Null }

$startMutex = [Threading.Mutex]::new($false, "Local\LaLaDubReleaseStart")
$hasStartMutex = $false
try {
  $hasStartMutex = $startMutex.WaitOne([TimeSpan]::FromSeconds(15))
  if (-not $hasStartMutex) { throw "Another bot start or stop operation is still running." }

  if ($StartupDelaySeconds -gt 0) {
    Start-Sleep -Seconds $StartupDelaySeconds
  }

if (Test-Path -LiteralPath $PidFile) {
  $existingPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existingProcess) {
      if (
        $existingProcess.CommandLine -like "*tools\runtime\Watchdog.ps1*" -and
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
    $_.CommandLine -like "*tools\runtime\Watchdog.ps1*" -and
    $_.CommandLine -like "*-Instance release*" -and
    $_.CommandLine -like "*$Root*"
  } |
  Select-Object -First 1
if ($existingWatchdog) {
  Set-Content -LiteralPath $PidFile -Value $existingWatchdog.ProcessId
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Release watchdog is already running, pid=$($existingWatchdog.ProcessId)" -ErrorAction SilentlyContinue
  exit 0
}

foreach ($logPath in @($OutLog, $ErrLog)) {
  $logItem = Get-Item -LiteralPath $logPath -ErrorAction SilentlyContinue
  if ($logItem -and $logItem.Length -gt 25MB) {
    Clear-Content -LiteralPath $logPath
  }
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
$env:LALADUB_PAID_USERS_FILE = (Join-Path $Root "paid_users.txt")
$env:LALADUB_ADMIN_USERS_FILE = (Join-Path $Root "admins.txt")
$env:LALADUB_BOT_WORKDIR = $WorkDir
$env:LALADUB_PROPOSAL_ENABLED = "1"
$env:LALADUB_PROPOSAL_DB = (Join-Path $Root "runs\proposal\proposals.sqlite3")
$env:LALADUB_PAYSUPPORT_CONTACT = "пиши @hboloid"
$env:LALADUB_EXECUTOR_MODE = "hybrid"
$env:LALADUB_MAX_LOCAL_JOBS = "1"
$env:LALADUB_WORKER_API_HOST = "0.0.0.0"
$env:LALADUB_WORKER_API_PORT = "8765"
$env:LALADUB_WORKER_API_TRUSTED_IPS = "192.168.1.67,192.168.1.180"
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
$workerApiTokenPath = Join-Path $Root ".secrets\Worker-Api-Token.txt"
if (-not (Test-Path -LiteralPath $workerApiTokenPath)) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $workerApiTokenPath) | Out-Null
  $generatedWorkerToken = ([guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N"))
  Set-Content -LiteralPath $workerApiTokenPath -Value $generatedWorkerToken -Encoding ASCII -NoNewline
}
$env:LALADUB_WORKER_API_TOKEN = (Get-Content -LiteralPath $workerApiTokenPath -Raw).Trim()
if (-not $env:LALADUB_WORKER_API_TOKEN) { throw "Worker API token file is empty: $workerApiTokenPath" }
$env:LALADUB_WORKER_PACKAGE_PATH = (Join-Path $Root "dist\LaLaDubWorker-update.zip")
$env:LALADUB_WORKER_PACKAGE_MANIFEST = (Join-Path $Root "dist\LaLaDubWorker-update.manifest.json")
$env:LALADUB_DOWNLOAD_CACHE_DIR = (Join-Path $Root "runs\cache\downloads")
$projectUserHome = Split-Path -Parent (Split-Path -Parent $Root)
$firefoxProfilesRoot = Join-Path $projectUserHome "AppData\Roaming\Mozilla\Firefox\Profiles"
$firefoxProfile = Get-ChildItem -LiteralPath $firefoxProfilesRoot -Directory -ErrorAction SilentlyContinue |
  Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "cookies.sqlite") } |
  Sort-Object `
    @{ Expression = { if ($_.Name -like "*.default-release") { 0 } else { 1 } } },
    @{ Expression = { $_.LastWriteTime }; Descending = $true } |
  Select-Object -First 1
if ($firefoxProfile) {
  $env:LALADUB_YTDLP_BROWSER_COOKIES = "firefox"
  $env:LALADUB_YTDLP_BROWSER_PROFILE = $firefoxProfile.FullName
  Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) yt-dlp Firefox profile: $($firefoxProfile.FullName)" -ErrorAction SilentlyContinue
} else {
  $env:LALADUB_YTDLP_BROWSER_COOKIES = ""
  $env:LALADUB_YTDLP_BROWSER_PROFILE = ""
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Firefox cookies.sqlite was not found under: $firefoxProfilesRoot" -ErrorAction SilentlyContinue
}
$env:LALADUB_MEDIA_CACHE_DIR = (Join-Path $Root "runs\cache\media")
$env:LALADUB_JOB_RETENTION_SECONDS = "2592000"
$env:LALADUB_CLEANUP_INTERVAL_SECONDS = "3600"
$env:LALADUB_TTS = "moss"
$env:LALADUB_TRANSLATOR = "hybrid"
# Argos language models live on F: they are several gigabytes and C: is the
# small SSD that gets cleaned out regularly. Only the first read of a model
# after a reboot touches the disk at all - after that Windows keeps it cached -
# so the slower drive costs a fraction of a second per language, once.
$env:ARGOS_PACKAGES_DIR = "F:\LaLaSchoolData\argos-translate\packages"
$env:LALADUB_MAX_ACTIVE_JOBS = "2"
$env:LALADUB_MAX_ACTIVE_JOBS_PER_USER = "1"
$env:LALADUB_WATERMARK_IMAGE = (Join-Path $Root "assets")
$trustedVisualSource = "F:\FFOutput\La La School\video_files"
if (-not (Test-Path -LiteralPath $trustedVisualSource -PathType Container)) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Trusted visual source is missing: $trustedVisualSource" -ErrorAction SilentlyContinue
  exit 1
}
$env:LALADUB_AUDIO_VISUAL_SOURCE_DIR = $trustedVisualSource
$env:LALADUB_AUDIO_VISUAL_SAFETY_ENABLED = "0"
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
$qwenRoot = $Root
$qwenPython = Join-Path $qwenRoot ".venv-qwen3tts\Scripts\python.exe"
$env:LALADUB_QWEN3_PYTHON = $qwenPython
$env:LALADUB_QWEN3_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
$env:LALADUB_QWEN3_CACHE_DIR = (Join-Path $qwenRoot "models\qwen3tts")
$env:LALADUB_QWEN3_TIMEOUT_SECONDS = "1800"
$env:LALADUB_COSYVOICE_PYTHON = (Join-Path $Root ".venv-cosyvoice\Scripts\python.exe")
$env:LALADUB_COSYVOICE_REPO_DIR = (Join-Path $Root "models\cosyvoice\CosyVoice")
$env:LALADUB_COSYVOICE_MODEL_DIR = (Join-Path $Root "models\cosyvoice\pretrained_models\Fun-CosyVoice3-0.5B")
$env:LALADUB_COSYVOICE_MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
$env:LALADUB_COSYVOICE_MODE = "cross_lingual"
$env:LALADUB_COSYVOICE_INSTRUCTION = "You are a helpful assistant.<|endofprompt|>"
$env:LALADUB_COSYVOICE_DEVICE = "auto"
$env:LALADUB_COSYVOICE_SPEED = "1.0"
$env:LALADUB_COSYVOICE_TIMEOUT_SECONDS = "1800"
$env:LALADUB_MOSS_PYTHON = "F:\LaLaSchoolData\tts-lab\MOSS-TTS\.venv\Scripts\python.exe"
$env:LALADUB_MOSS_MODEL_DIR = "F:\LaLaSchoolData\tts-lab\models\MOSS-TTS-Local-Transformer-v1.5"
$env:LALADUB_MOSS_CODEC_DIR = "F:\LaLaSchoolData\tts-lab\models\MOSS-Audio-Tokenizer-v2"
$env:LALADUB_MOSS_DEVICE = "auto"
$env:LALADUB_MOSS_TIMEOUT_SECONDS = "1800"
$env:LALADUB_MULTI_SPEAKER = "1"
$env:LALADUB_SPEAKER_REFERENCE_SECONDS = "5.0"
$env:LALADUB_SPEAKER_CLUSTERING = "1"
$env:LALADUB_MAX_SPEAKER_CLUSTERS = "9"
$env:LALADUB_SPEAKER_CLUSTER_THRESHOLD = "0.08"
$env:LALADUB_DIARIZATION_PYTHON = (Join-Path $Root ".venv-diarization\Scripts\python.exe")
$env:LALADUB_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
$env:LALADUB_DIARIZATION_DEVICE = "auto"
$env:LALADUB_DIARIZATION_CACHE_DIR = (Join-Path $Root "models\diarization")
$env:LALADUB_DIARIZATION_TOKEN_FILE = (Join-Path $Root ".secrets\HuggingFace-Token.txt")
$env:LALADUB_DIARIZATION_TIMEOUT_SECONDS = "1800"
$env:LALADUB_SEPARATION = "bsroformer"
$env:LALADUB_SEPARATION_DEVICE = "cuda"
$env:LALADUB_BSROFORMER_PYTHON = (Join-Path $Root ".venv-bsroformer\Scripts\python.exe")
$env:LALADUB_BSROFORMER_MODEL_DIR = (Join-Path $Root "models\audio-separator")
$env:LALADUB_BSROFORMER_MODEL_FILE = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
$env:LALADUB_BSROFORMER_TIMEOUT_SECONDS = "600"
$env:LALADUB_AUDIO_BED = "instrumental"
$env:LALADUB_ORIGINAL_VOLUME = "0.35"
$env:LALADUB_TRIM_TTS_SILENCE = "1"
$env:LALADUB_TTS_MAX_PAUSE_SECONDS = "0.3"
$env:LALADUB_COLLAPSE_REPETITIONS = "1"
$env:LALADUB_MAX_PHRASE_REPEATS = "2"
$env:LALADUB_MAX_WORD_REPEATS = "3"
$env:LALADUB_INJECT_ARTIFACTS = "1"
# Artifacts now come from assets/hallucinations instead of a second Whisper
# pass over the whole audio. That pass cost a median of 10s, 141s at the
# 90th percentile and up to 20 minutes on long input. Set to "whisper" to
# bring the hunt back.
$env:LALADUB_ARTIFACT_SOURCE = "catalog"
# How often a phrase is pulled from a language other than the decoy one.
$env:LALADUB_ARTIFACT_CROSS_LANGUAGE_SHARE = "0.15"
$env:LALADUB_ARTIFACT_MAX_SEGMENTS = "14"
$env:LALADUB_ARTIFACT_RATIO = "0.20"
$env:LALADUB_ARTIFACT_MIN_SOURCE_SEGMENTS = "5"
$env:LALADUB_ARTIFACT_MIN_GAP_SECONDS = "0.5"
$env:LALADUB_DISTORT_TRANSLATION = "1"
$env:LALADUB_TRANSLATION_PIVOTS = "input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar|input,en,de|input,ja,ko,en|input,tr,ar,en|en,th,he,en|en,ms,he,en"
# Each hop is one call to a free translation API; longer chains are what got the
# bot rate-limited. Chains above this length are trimmed, not dropped.
$env:LALADUB_MAX_TRANSLATION_HOPS = "3"
$env:LALADUB_TRANSLATION_SECOND_PASS_RATIO = "0.45"
$env:LALADUB_ASR_BACKEND = "faster-whisper"
$env:LALADUB_DEFAULT_ASR_METHOD = "ow-large-v3-chaos-backbone"
$env:LALADUB_WHISPER_DEVICE = "cuda"
$env:LALADUB_WHISPER_COMPUTE_TYPE = "int8"
$env:LALADUB_WHISPER_ONLY_MODEL = "turbo"
$env:LALADUB_WHISPER_ONLY_DEVICE = "cuda"
$env:LALADUB_ARTIFACT_WHISPER_DEVICE = "cuda"
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
} finally {
  if ($hasStartMutex) { $startMutex.ReleaseMutex() }
  $startMutex.Dispose()
}
