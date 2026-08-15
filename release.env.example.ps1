# Copy this file to .secrets\Release-Bot-Token.ps1 or set variables manually.
# Do not commit real tokens.

$env:LALADUB_BOT_TOKEN = "PASTE_TELEGRAM_BOT_TOKEN_HERE"
$env:LALADUB_PAID_USERS = "123456789,987654321"

# Runtime defaults used by the release launcher.
$env:LALADUB_TRANSLATOR = "hybrid"
$env:LALADUB_TTS = "f5"
$env:LALADUB_SEPARATION = "demucs"
$env:LALADUB_AUDIO_BED = "instrumental"
$env:LALADUB_ORIGINAL_VOLUME = "0.35"
$env:LALADUB_BOT_WORKDIR = "runs/bot-release"
$env:LALADUB_DOWNLOAD_CACHE_DIR = "runs/cache/downloads"
$env:LALADUB_MEDIA_CACHE_DIR = "runs/cache/media"
$env:LALADUB_ARTIFACT_TRIGGERS_FILE = "assets/artifact_triggers.txt"
$env:LALADUB_ARTIFACT_PROMPT_SEEDS_RU_FILE = "assets/artifact_prompt_seeds_ru.txt"

# Worker API. Generate a strong random token for real usage.
$env:LALADUB_WORKER_API_HOST = "0.0.0.0"
$env:LALADUB_WORKER_API_PORT = "8765"
$env:LALADUB_WORKER_API_TOKEN = "PASTE_WORKER_API_TOKEN_HERE"
$env:LALADUB_ARTIFACT_MAX_SEGMENTS = "14"
$env:LALADUB_DISTORT_TRANSLATION = "1"
$env:LALADUB_TRANSLATION_PIVOTS = "input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar|input,en,de|input,ja,ko,en|input,tr,ar,en|en,ms,he,en"
$env:LALADUB_TRANSLATION_SECOND_PASS_RATIO = "0.45"
