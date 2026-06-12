$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutLog = Join-Path $Root "bot.release.out.log"
$ErrLog = Join-Path $Root "bot.release.err.log"
$PidFile = Join-Path $Root "bot.release.pid"
$TokenFile = Join-Path $Root ".secrets\Release-Bot-Token.ps1"
$WorkDir = Join-Path $Root "runs\bot-release"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType File -Force -Path $OutLog | Out-Null
New-Item -ItemType File -Force -Path $ErrLog | Out-Null

if (Test-Path -LiteralPath $PidFile) {
  $existingPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existingProcess = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
    if ($existingProcess) {
      Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Release bot is already running, pid=$existingPid"
      exit 0
    }
  }
}

if (-not (Test-Path -LiteralPath $TokenFile)) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Release token file is missing: $TokenFile"
  exit 1
}

. $TokenFile
if (-not $ReleaseBotToken) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) Release bot token is empty."
  exit 1
}

$env:LALADUB_BOT_TOKEN = $ReleaseBotToken
$paidUsers = [Environment]::GetEnvironmentVariable("LALADUB_PAID_USERS", "User")
if ($paidUsers) { $env:LALADUB_PAID_USERS = $paidUsers }
$env:LALADUB_BOT_WORKDIR = $WorkDir
$env:LALADUB_TTS = "xtts"
$env:LALADUB_TRANSLATOR = "hybrid"
$env:LALADUB_MAX_ACTIVE_JOBS = "2"
$env:LALADUB_MAX_ACTIVE_JOBS_PER_USER = "1"
$env:LALADUB_FREE_MAX_DURATION_SECONDS = "180"
$env:LALADUB_PAID_MAX_DURATION_SECONDS = "0"
$env:LALADUB_WATERMARK_IMAGE = (Join-Path $Root "assets\watermark.png")
$env:LALADUB_VOICE = "Microsoft Irina Desktop"
$env:LALADUB_XTTS_DEVICE = "cpu"
$env:LALADUB_MULTI_SPEAKER = "1"
$env:LALADUB_SPEAKER_REFERENCE_SECONDS = "5.0"
$env:LALADUB_SPEAKER_CLUSTERING = "1"
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
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Starting La La Dub Release Bot..."
$process = Start-Process `
  -FilePath $python `
  -ArgumentList @("-m", "laladub.bot", "--instance", "release") `
  -WorkingDirectory $Root `
  -RedirectStandardOutput $OutLog `
  -RedirectStandardError $ErrLog `
  -WindowStyle Hidden `
  -PassThru

Set-Content -LiteralPath $PidFile -Value $process.Id
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Started release pid=$($process.Id)"
