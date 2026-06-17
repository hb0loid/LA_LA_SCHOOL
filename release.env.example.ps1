# Copy this file to .secrets\Release-Bot-Token.ps1 or set variables manually.
# Do not commit real tokens.

$env:LALADUB_BOT_TOKEN = "PASTE_TELEGRAM_BOT_TOKEN_HERE"
$env:LALADUB_PAID_USERS = "123456789,987654321"

# Runtime defaults used by the release launcher.
$env:LALADUB_TRANSLATOR = "hybrid"
$env:LALADUB_TTS = "f5"
$env:LALADUB_SEPARATION = "demucs"
$env:LALADUB_AUDIO_BED = "instrumental"
$env:LALADUB_FREE_MAX_DURATION_SECONDS = "180"
$env:LALADUB_PAID_MAX_DURATION_SECONDS = "0"
$env:LALADUB_BOT_WORKDIR = "runs/bot-release"

# Worker API. Generate a strong random token for real usage.
$env:LALADUB_WORKER_API_HOST = "0.0.0.0"
$env:LALADUB_WORKER_API_PORT = "8765"
$env:LALADUB_WORKER_API_TOKEN = "PASTE_WORKER_API_TOKEN_HERE"
