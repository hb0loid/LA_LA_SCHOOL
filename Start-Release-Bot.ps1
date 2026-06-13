$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutLog = Join-Path $Root "bot.release.out.log"
$ErrLog = Join-Path $Root "bot.release.err.log"
$PidFile = Join-Path $Root "bot.release.pid"
$WatchdogScript = Join-Path $Root "Run-Bot-Watchdog.ps1"
$TokenFile = Join-Path $Root ".secrets\Release-Bot-Token.ps1"
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

$env:LALADUB_BOT_TOKEN = $ReleaseBotToken
$paidUsers = [Environment]::GetEnvironmentVariable("LALADUB_PAID_USERS", "User")
if ($paidUsers) { $env:LALADUB_PAID_USERS = $paidUsers }
$env:LALADUB_BOT_WORKDIR = $WorkDir
$env:LALADUB_TTS = "f5"
$env:LALADUB_TRANSLATOR = "hybrid"
$env:LALADUB_MAX_ACTIVE_JOBS = "1"
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
$env:LALADUB_WHISPER_ONLY_DEVICE = "cpu"
$env:LALADUB_SUPPRESS_PLAIN_ASCII_TOKENS = "0"
$env:PYTHONPATH = (Join-Path $Root "src")
$env:PYTHONIOENCODING = "utf-8"

$python = (Get-Command python -ErrorAction Stop).Source
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
