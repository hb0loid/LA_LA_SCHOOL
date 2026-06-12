$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutLog = Join-Path $Root "bot.out.log"
$ErrLog = Join-Path $Root "bot.err.log"
$PidFile = Join-Path $Root "bot.pid"

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path (Join-Path $Root "runs\bot") | Out-Null
New-Item -ItemType File -Force -Path $OutLog | Out-Null
New-Item -ItemType File -Force -Path $ErrLog | Out-Null

if (Test-Path -LiteralPath $PidFile) {
  $existingPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existingProcess = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
    if ($existingProcess) {
      Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Bot is already running, pid=$existingPid"
      exit 0
    }
  }
}

$token = [Environment]::GetEnvironmentVariable("LALADUB_BOT_TOKEN", "User")
if (-not $token) {
  $token = [Environment]::GetEnvironmentVariable("LALADUB_BOT_TOKEN", "Process")
}
if (-not $token) {
  Add-Content -LiteralPath $ErrLog -Value "$(Get-Date -Format s) LALADUB_BOT_TOKEN is not set."
  exit 1
}

$env:LALADUB_BOT_TOKEN = $token
$paidUsers = [Environment]::GetEnvironmentVariable("LALADUB_PAID_USERS", "User")
if (-not $env:LALADUB_PAID_USERS -and $paidUsers) { $env:LALADUB_PAID_USERS = $paidUsers }
if (-not $env:LALADUB_TRANSLATOR) { $env:LALADUB_TRANSLATOR = "hybrid" }
if (-not $env:LALADUB_TTS) { $env:LALADUB_TTS = "xtts" }
if (-not $env:LALADUB_MAX_ACTIVE_JOBS) { $env:LALADUB_MAX_ACTIVE_JOBS = "2" }
if (-not $env:LALADUB_MAX_ACTIVE_JOBS_PER_USER) { $env:LALADUB_MAX_ACTIVE_JOBS_PER_USER = "1" }
if (-not $env:LALADUB_FREE_MAX_DURATION_SECONDS) { $env:LALADUB_FREE_MAX_DURATION_SECONDS = "180" }
if (-not $env:LALADUB_PAID_MAX_DURATION_SECONDS) { $env:LALADUB_PAID_MAX_DURATION_SECONDS = "0" }
if (-not $env:LALADUB_VOICE) { $env:LALADUB_VOICE = "Microsoft Irina Desktop" }
if (-not $env:LALADUB_XTTS_DEVICE) { $env:LALADUB_XTTS_DEVICE = "cpu" }
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
if (-not $env:LALADUB_MAX_SPEAKER_CLUSTERS) { $env:LALADUB_MAX_SPEAKER_CLUSTERS = "6" }
if (-not $env:LALADUB_SPEAKER_CLUSTER_THRESHOLD) { $env:LALADUB_SPEAKER_CLUSTER_THRESHOLD = "0.08" }
if (-not $env:LALADUB_SEPARATION) { $env:LALADUB_SEPARATION = "demucs" }
if (-not $env:LALADUB_SEPARATION_DEVICE) { $env:LALADUB_SEPARATION_DEVICE = "cpu" }
if (-not $env:LALADUB_AUDIO_BED) { $env:LALADUB_AUDIO_BED = "instrumental" }
if (-not $env:LALADUB_ORIGINAL_VOLUME) { $env:LALADUB_ORIGINAL_VOLUME = "1.0" }
if (-not $env:LALADUB_COLLAPSE_REPETITIONS) { $env:LALADUB_COLLAPSE_REPETITIONS = "1" }
if (-not $env:LALADUB_MAX_PHRASE_REPEATS) { $env:LALADUB_MAX_PHRASE_REPEATS = "2" }
if (-not $env:LALADUB_MAX_WORD_REPEATS) { $env:LALADUB_MAX_WORD_REPEATS = "3" }
if (-not $env:LALADUB_INJECT_ARTIFACTS) { $env:LALADUB_INJECT_ARTIFACTS = "1" }
if (-not $env:LALADUB_ARTIFACT_MAX_SEGMENTS) { $env:LALADUB_ARTIFACT_MAX_SEGMENTS = "12" }
if (-not $env:LALADUB_ARTIFACT_MIN_GAP_SECONDS) { $env:LALADUB_ARTIFACT_MIN_GAP_SECONDS = "0.5" }
if (-not $env:LALADUB_DISTORT_TRANSLATION) { $env:LALADUB_DISTORT_TRANSLATION = "1" }
if (-not $env:LALADUB_TRANSLATION_PIVOTS) { $env:LALADUB_TRANSLATION_PIVOTS = "input,en|en,de|en,fr|en,es|input,en,de|en,es,de" }
if (-not $env:LALADUB_WHISPER_DEVICE) { $env:LALADUB_WHISPER_DEVICE = "cpu" }
if (-not $env:LALADUB_WHISPER_COMPUTE_TYPE) { $env:LALADUB_WHISPER_COMPUTE_TYPE = "int8" }
if (-not $env:LALADUB_SUPPRESS_PLAIN_ASCII_TOKENS) { $env:LALADUB_SUPPRESS_PLAIN_ASCII_TOKENS = "0" }
if (-not $env:LALADUB_ASR_BACKEND) { $env:LALADUB_ASR_BACKEND = "faster-whisper" }
if (-not $env:LALADUB_DEFAULT_ASR_METHOD) { $env:LALADUB_DEFAULT_ASR_METHOD = "ow-large-v3-chaos-backbone" }
if (-not $env:LALADUB_WHISPER_ONLY_MODEL) { $env:LALADUB_WHISPER_ONLY_MODEL = "large-v3" }
if (-not $env:LALADUB_WHISPER_ONLY_DEVICE) { $env:LALADUB_WHISPER_ONLY_DEVICE = "cpu" }
if (-not $env:LALADUB_BOT_WORKDIR) { $env:LALADUB_BOT_WORKDIR = (Join-Path $Root "runs\bot") }
if (-not $env:LALADUB_WATERMARK_IMAGE) { $env:LALADUB_WATERMARK_IMAGE = (Join-Path $Root "assets\watermark.png") }
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = (Join-Path $Root "src") }
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }

$python = (Get-Command python -ErrorAction Stop).Source
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Starting La La Dub Bot..."
$process = Start-Process `
  -FilePath $python `
  -ArgumentList @("-m", "laladub.bot", "--instance", "test") `
  -WorkingDirectory $Root `
  -RedirectStandardOutput $OutLog `
  -RedirectStandardError $ErrLog `
  -WindowStyle Hidden `
  -PassThru

Set-Content -LiteralPath $PidFile -Value $process.Id
Add-Content -LiteralPath $OutLog -Value "$(Get-Date -Format s) Started pid=$($process.Id)"
