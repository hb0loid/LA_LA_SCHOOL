from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .models import DubConfig, Segment

# Argos splits text into sentences before translating, and its default splitter
# wants a Stanza model per language. There is none for Malay or Azerbaijani, so
# reading either raised "No processors to load for language ms" and killed the
# job whenever the online translator was rate-limited - the single most common
# failure people reported. MiniSBD needs no per-language model and, checked
# side by side on Vietnamese, English and Japanese, returns exactly the same
# translations. Set before argostranslate is imported anywhere: it reads this
# once, at import.
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")


class TranslationError(RuntimeError):
    pass


_TRANSLATION_CACHE_READY: set[str] = set()


def _translation_cache_path(config: DubConfig) -> Path | None:
    cache_root = getattr(config, "media_cache_dir", None)
    if not cache_root:
        return None
    return Path(cache_root).parent / "translations.sqlite3"


def _translation_cache_connect(config: DubConfig) -> sqlite3.Connection | None:
    path = _translation_cache_path(config)
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10.0)
        key = str(path)
        if key not in _TRANSLATION_CACHE_READY:
            # Pipeline stages run as separate spawned processes, so WAL is what
            # lets them share this file without blocking each other.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 10000")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS translations ("
                "key TEXT PRIMARY KEY, translated TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            connection.commit()
            _TRANSLATION_CACHE_READY.add(key)
        return connection
    except Exception:
        return None


def _translation_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    payload = f"{source_lang}\x00{target_lang}\x00{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _translation_cache_get(text: str, source_lang: str, target_lang: str, config: DubConfig) -> str | None:
    connection = _translation_cache_connect(config)
    if connection is None:
        return None
    try:
        row = connection.execute(
            "SELECT translated FROM translations WHERE key = ?",
            (_translation_cache_key(text, source_lang, target_lang),),
        ).fetchone()
        return str(row[0]) if row is not None else None
    except Exception:
        return None
    finally:
        connection.close()


def _translation_cache_put(
    text: str, source_lang: str, target_lang: str, translated: str, config: DubConfig
) -> None:
    # Never cache a failure: a rate-limited or over-long request comes back as
    # the translator's own error text, and storing it would make one bad moment
    # permanent for that phrase.
    if not translated.strip() or _looks_bad_machine_translation(translated):
        return
    connection = _translation_cache_connect(config)
    if connection is None:
        return
    try:
        connection.execute(
            "INSERT OR REPLACE INTO translations (key, translated, created_at) VALUES (?, ?, ?)",
            (_translation_cache_key(text, source_lang, target_lang), translated, time.time()),
        )
        connection.commit()
    except Exception:
        pass
    finally:
        connection.close()


_ARGOS_INDEX_UPDATED = False


def translate_segments(segments: list[Segment], config: DubConfig) -> list[Segment]:
    provider = config.translator.lower()
    if provider == "identity":
        for segment in segments:
            segment.translated_text = segment.text
        return segments

    if provider == "hybrid":
        return _translate_hybrid(segments, config)

    if provider == "googleweb":
        return _translate_googleweb(segments, config)

    if provider == "mymemory":
        return _translate_mymemory(segments, config)

    if provider == "argos":
        return _translate_argos(segments, config)

    if provider == "libretranslate":
        return _translate_libretranslate(segments, config)

    if provider == "llm":
        return _translate_llm(segments, config)

    raise TranslationError(f"Unknown translator provider: {config.translator}")


def translate_text(text: str, source_lang: str, target_lang: str, config: DubConfig) -> str:
    provider = config.translator.lower()
    if provider == "identity" or source_lang == target_lang or not text.strip():
        return text

    known = _translate_known_meta_text(text, source_lang, target_lang)
    if known is not None:
        return known

    if provider == "hybrid":
        return _translate_hybrid_text(text, source_lang, target_lang, config)

    if provider == "googleweb":
        return _translate_googleweb_text(text, source_lang, target_lang)

    if provider == "mymemory":
        return _translate_mymemory_text(text, source_lang, target_lang)

    if provider == "argos":
        return _translate_argos_provider_text(text, source_lang, target_lang)

    if provider == "libretranslate":
        return _translate_libretranslate_text(text, source_lang, target_lang, config)

    if provider == "llm":
        return _translate_llm_texts([text], target_lang, config)[0]

    raise TranslationError(f"Unknown translator provider: {config.translator}")


def translate_text_chain(text: str, languages: list[str], config: DubConfig) -> str:
    current = text
    for source_lang, target_lang in zip(languages, languages[1:]):
        current = translate_text(current, source_lang, target_lang, config)
    return _postprocess_translated_text(current, languages[-1] if languages else config.target_lang)


def _translate_hybrid(segments: list[Segment], config: DubConfig) -> list[Segment]:
    if not config.source_lang:
        raise TranslationError("Hybrid translator needs --source-lang, for example: --source-lang vi")

    for segment in segments:
        text = segment.text.strip()
        segment.translated_text = _postprocess_translated_text(
            _translate_hybrid_text(text, config.source_lang, config.target_lang, config),
            config.target_lang,
        )
    return segments


def _translate_hybrid_text(text: str, source_lang: str, target_lang: str, config: DubConfig) -> str:
    known = _translate_known_meta_text(text, source_lang, target_lang)
    if known is not None:
        return known

    # Distortion runs every segment through 15 chains of 3-7 languages, so the
    # same short phrase is translated over and over, both inside one video and
    # across videos. Reusing earlier answers is what keeps the free translation
    # APIs from answering 429 halfway through a job.
    cached = _translation_cache_get(text, source_lang, target_lang, config)
    if cached is not None:
        return cached

    result = _translate_hybrid_text_uncached(text, source_lang, target_lang, config)
    _translation_cache_put(text, source_lang, target_lang, result, config)
    return result


def _translate_hybrid_text_uncached(text: str, source_lang: str, target_lang: str, config: DubConfig) -> str:
    online_errors: list[Exception] = []
    try:
        translated = _translate_googleweb_text(text, source_lang, target_lang)
        if not _looks_bad_machine_translation(translated):
            return _postprocess_translated_text(translated, target_lang)
    except Exception as exc:
        online_errors.append(exc)

    try:
        translated = _translate_mymemory_text(text, source_lang, target_lang)
        if not _looks_bad_machine_translation(translated):
            return _postprocess_translated_text(translated, target_lang)
    except Exception as exc:
        online_errors.append(exc)

    try:
        return _postprocess_translated_text(_translate_argos_provider_text(text, source_lang, target_lang), target_lang)
    except Exception as exc:
        if online_errors:
            online_details = "; ".join(f"{type(item).__name__}: {item}" for item in online_errors)
            raise TranslationError(
                "Hybrid translator failed: "
                f"online={online_details}; "
                f"argos={type(exc).__name__}: {exc}"
            ) from exc
        raise


def _translate_googleweb(segments: list[Segment], config: DubConfig) -> list[Segment]:
    if not config.source_lang:
        raise TranslationError("GoogleWeb translator needs --source-lang, for example: --source-lang vi")

    for segment in segments:
        text = segment.text.strip()
        known = _translate_known_meta_text(text, config.source_lang, config.target_lang)
        segment.translated_text = _postprocess_translated_text(
            known
            if known is not None
            else _translate_googleweb_text(
                text,
                config.source_lang,
                config.target_lang,
            ),
            config.target_lang,
        )
    return segments


def _translate_googleweb_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text or source_lang == target_lang:
        return text

    query = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
    )
    request = urllib.request.Request(
        f"https://translate.googleapis.com/translate_a/single?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    try:
        translated = "".join(part[0] for part in data[0] if part and part[0])
    except Exception as exc:
        raise TranslationError(f"Unexpected GoogleWeb response: {data!r}") from exc
    if not translated.strip():
        raise TranslationError("GoogleWeb returned an empty translation.")
    return _postprocess_translated_text(translated.strip(), target_lang)


def _translate_mymemory(segments: list[Segment], config: DubConfig) -> list[Segment]:
    if not config.source_lang:
        raise TranslationError("MyMemory translator needs --source-lang, for example: --source-lang vi")

    for segment in segments:
        text = segment.text.strip()
        known = _translate_known_meta_text(text, config.source_lang, config.target_lang)
        segment.translated_text = _postprocess_translated_text(
            known
            if known is not None
            else _translate_mymemory_text(
                text,
                config.source_lang,
                config.target_lang,
            ),
            config.target_lang,
        )
    return segments


def _translate_mymemory_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text or source_lang == target_lang:
        return text

    query = urllib.parse.urlencode(
        {
            "q": text,
            "langpair": f"{source_lang}|{target_lang}",
        }
    )
    request = urllib.request.Request(
        f"https://api.mymemory.translated.net/get?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    translated = str(data.get("responseData", {}).get("translatedText") or "").strip()
    if not translated:
        raise TranslationError(f"Unexpected MyMemory response: {data!r}")
    return _postprocess_translated_text(translated, target_lang)


def _translate_argos_provider_text(text: str, source_lang: str, target_lang: str) -> str:
    try:
        import argostranslate.translate
    except ImportError as exc:
        raise TranslationError(
            "argostranslate is not installed. Run: python -m pip install -e .[translate]"
        ) from exc

    _ensure_argos_route(source_lang, target_lang, argostranslate.translate)
    return _postprocess_translated_text(
        _translate_argos_text(text, source_lang, target_lang, argostranslate.translate),
        target_lang,
    )


def _translate_argos(segments: list[Segment], config: DubConfig) -> list[Segment]:
    if not config.source_lang:
        raise TranslationError("Argos needs --source-lang, for example: --source-lang vi")
    try:
        import argostranslate.translate
    except ImportError as exc:
        raise TranslationError(
            "argostranslate is not installed. Run: python -m pip install -e .[translate]"
        ) from exc

    _ensure_argos_route(config.source_lang, config.target_lang, argostranslate.translate)

    for segment in segments:
        text = segment.text.strip()
        known = _translate_known_meta_text(text, config.source_lang, config.target_lang)
        segment.translated_text = _postprocess_translated_text(
            known
            if known is not None
            else _translate_argos_text(
                text,
                config.source_lang,
                config.target_lang,
                argostranslate.translate,
            ),
            config.target_lang,
        )
    return segments


def _translate_argos_text(text: str, source_lang: str, target_lang: str, translate_module: object) -> str:
    if not text or source_lang == target_lang:
        return text

    direct = _get_argos_translation(translate_module, source_lang, target_lang)
    if direct is not None:
        return _postprocess_translated_text(direct.translate(text), target_lang)

    pivot_lang = "en"
    if source_lang != pivot_lang and target_lang != pivot_lang:
        first = _get_argos_translation(translate_module, source_lang, pivot_lang)
        second = _get_argos_translation(translate_module, pivot_lang, target_lang)
        if first is not None and second is not None:
            return _postprocess_translated_text(second.translate(first.translate(text)), target_lang)

    raise TranslationError(
        f"Argos package is missing for {source_lang}->{target_lang}. "
        f"Install a direct package or pivot packages {source_lang}->en and en->{target_lang}."
    )


def _ensure_argos_route(source_lang: str, target_lang: str, translate_module: object) -> None:
    if not source_lang or source_lang == target_lang:
        return
    if _has_argos_route(source_lang, target_lang, translate_module):
        return

    print(f"      Missing Argos route {source_lang}->{target_lang}; trying auto-install")
    direct_installed = _install_argos_package(source_lang, target_lang)
    if direct_installed and _has_argos_route(source_lang, target_lang, translate_module):
        return

    pivot_lang = "en"
    if source_lang != pivot_lang:
        _install_argos_package(source_lang, pivot_lang)
    if target_lang != pivot_lang:
        _install_argos_package(pivot_lang, target_lang)
    if _has_argos_route(source_lang, target_lang, translate_module):
        return

    raise TranslationError(
        f"Argos package is missing for {source_lang}->{target_lang}. "
        f"Auto-install could not create a direct or {source_lang}->en->"
        f"{target_lang} route."
    )


def _has_argos_route(source_lang: str, target_lang: str, translate_module: object) -> bool:
    if _get_argos_translation(translate_module, source_lang, target_lang) is not None:
        return True
    pivot_lang = "en"
    if source_lang == pivot_lang or target_lang == pivot_lang:
        return False
    return (
        _get_argos_translation(translate_module, source_lang, pivot_lang) is not None
        and _get_argos_translation(translate_module, pivot_lang, target_lang) is not None
    )


def _install_argos_package(source_lang: str, target_lang: str) -> bool:
    if source_lang == target_lang:
        return True

    try:
        import argostranslate.package
    except ImportError as exc:
        raise TranslationError(
            "argostranslate package manager is not available. Run: python -m pip install -e .[translate]"
        ) from exc

    installed = argostranslate.package.get_installed_packages()
    if any(item.from_code == source_lang and item.to_code == target_lang for item in installed):
        return True

    global _ARGOS_INDEX_UPDATED
    if not _ARGOS_INDEX_UPDATED:
        argostranslate.package.update_package_index()
        _ARGOS_INDEX_UPDATED = True

    available = argostranslate.package.get_available_packages()
    match = next(
        (item for item in available if item.from_code == source_lang and item.to_code == target_lang),
        None,
    )
    if match is None:
        print(f"      No Argos package available for {source_lang}->{target_lang}")
        return False

    print(f"      Installing Argos package {source_lang}->{target_lang}")
    package_path = match.download()
    argostranslate.package.install_from_path(package_path)
    return True


# Whisper and Argos spell a few languages differently. Norwegian is the one
# that matters: Whisper reports "no" (and "nn" for Nynorsk, a frequent false
# positive on music), while the package is filed under "nb" - so without this
# the language was detected fine and then had no translator at all.
ARGOS_LANG_ALIASES = {
    "no": "nb",
    "nn": "nb",
    "iw": "he",
    "in": "id",
    "jw": "jv",
}


def _argos_lang(code: str) -> str:
    return ARGOS_LANG_ALIASES.get((code or "").strip().casefold(), code)


def _get_argos_translation(translate_module: object, source_lang: str, target_lang: str) -> object | None:
    try:
        return translate_module.get_translation_from_codes(
            _argos_lang(source_lang), _argos_lang(target_lang)
        )
    except Exception:
        return None


def _translate_libretranslate(segments: list[Segment], config: DubConfig) -> list[Segment]:
    if not config.source_lang:
        raise TranslationError("LibreTranslate needs --source-lang, for example: --source-lang vi")

    for segment in segments:
        text = segment.text.strip()
        if not text:
            segment.translated_text = ""
            continue

        segment.translated_text = _postprocess_translated_text(
            _translate_libretranslate_text(
                text,
                config.source_lang,
                config.target_lang,
                config,
            ),
            config.target_lang,
        )

    return segments


def _translate_libretranslate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    config: DubConfig,
) -> str:
    if not text or source_lang == target_lang:
        return text

    payload = {
        "q": text,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }
    if config.libretranslate_api_key:
        payload["api_key"] = config.libretranslate_api_key

    request = urllib.request.Request(
        config.libretranslate_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    try:
        return _postprocess_translated_text(data["translatedText"], target_lang)
    except KeyError as exc:
        raise TranslationError(f"Unexpected LibreTranslate response: {data}") from exc


def _translate_llm(segments: list[Segment], config: DubConfig) -> list[Segment]:
    texts = [segment.text.strip() for segment in segments]
    translated = _translate_llm_texts(texts, config.target_lang, config)
    for segment, text in zip(segments, translated):
        segment.translated_text = _postprocess_translated_text(text, config.target_lang)
    return segments


def _translate_llm_texts(texts: list[str], target_lang: str, config: DubConfig) -> list[str]:
    if not texts:
        return []

    numbered = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(texts))
    system_prompt = (
        "You are dubbing a video into "
        f"{target_lang}. You will receive a numbered list of rough, possibly mis-heard or "
        "nonsensical speech-to-text transcript lines. Rewrite them as one natural, fluent "
        f"{target_lang} narration. If a line is unclear or meaningless on its own, improvise "
        "something that flows naturally with the lines around it rather than leaving it "
        "broken, blank, or translated word-for-word. Some lines may be boilerplate "
        "hallucinated by imperfect speech-to-text rather than real speech - subtitle credits, "
        "requests to subscribe to a channel, 'thanks for watching', channel/website names. "
        "Treat these exactly like any other unclear line: rewrite them naturally in your own "
        "words as part of the flow instead of preserving them literally, dropping them, or "
        "using a fixed template.\n"
        "Every input line, even ones that look identical or near-identical to another line "
        "(e.g. a repeated jingle or catchphrase), corresponds to a separate moment in the "
        "video and MUST get its own output line - never merge, skip, or collapse repeated "
        "lines into fewer lines. You may (and should) still vary the wording between repeats "
        "so they don't sound robotic.\n"
        "Reply with strict JSON only, no commentary, no markdown fences, one object per input "
        'line, tagging each with its original line number: {"lines": [{"i": 1, "text": '
        '"..."}, {"i": 2, "text": "..."}]}'
    )
    payload = {
        "model": config.llm_model,
        "temperature": max(0.0, min(2.0, config.llm_temperature)),
        "max_tokens": min(16000, max(2000, len(texts) * 120)),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": numbered},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if config.llm_api_key:
        headers["Authorization"] = f"Bearer {config.llm_api_key}"

    request = urllib.request.Request(
        f"{config.llm_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.llm_timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError(f"Unexpected LLM response: {data!r}") from exc

    by_index = _extract_llm_indexed_lines(content)
    result: list[str] = []
    missing = 0
    for position, original in enumerate(texts, start=1):
        text = by_index.get(position)
        if text:
            result.append(text)
        else:
            missing += 1
            result.append(original)
    if missing:
        print(f"      LLM translation: {missing}/{len(texts)} lines missing from response, kept original text")
    return result


def _extract_llm_indexed_lines(content: str) -> dict[int, str]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    blob = match.group(0)

    try:
        parsed = json.loads(blob)
        items = parsed.get("lines")
        if isinstance(items, list) and items:
            return _index_llm_line_items(items)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Small local models sometimes emit near-JSON with a stray ":" instead of
    # "," between object separators on long outputs. Salvage by pulling every
    # {"i": N, "text": "..."} shaped object out with a regex instead of
    # giving up on the whole response.
    key_match = re.search(r'"lines"\s*:\s*\[', blob)
    search_area = blob[key_match.end():] if key_match else blob
    pairs = re.findall(
        r'"i"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        search_area,
    )
    return {
        int(index): _unescape_json_fragment(text)
        for index, text in pairs
        if _unescape_json_fragment(text)
    }


def _index_llm_line_items(items: list) -> dict[int, str]:
    result: dict[int, str] = {}
    for position, item in enumerate(items, start=1):
        if isinstance(item, dict):
            index = item.get("i")
            text = str(item.get("text", "")).strip()
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = position
        else:
            index = position
            text = str(item).strip()
        if text:
            result[index] = text
    return result


def _unescape_json_fragment(value: str) -> str:
    return value.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ").replace("\\\\", "\\").strip()


_VI_SUBSCRIBE_RE = re.compile(
    r"(?:^|\b)(?:các\s+bạn\s+)?hãy\s+"
    r"(?:subscribe|đăng\s*k[íy])\s+cho\s+kênh\s+"
    r"(?P<channel>.+?)"
    r"(?:\s+để\s+không\s+bỏ\s+lỡ\b|$)",
    re.IGNORECASE,
)

_TURKISH_SUBTITLES_CREDIT_RE = re.compile(
    r"^\s*altyaz[ıi]\s+m\.?\s*k\.?\s*$",
    re.IGNORECASE,
)

_RU_BAD_SUBTITLE_CREDIT_RE = re.compile(
    r"\b[Пп]одзаголов(?:ок|ки)\s+(?:M|М)\.?\s*(?:K|К)\.?",
    re.IGNORECASE,
)


# Cyrillic that survived a CP1251 -> Latin-1 misread comes back as runs of
# accented Latin letters; real words give several long Cyrillic runs once the
# bytes are re-decoded, while an ordinary accented word yields at most one or
# two stray letters. That difference is what makes the repair safe to apply.
_CYRILLIC_RUN_RE = re.compile(r"[А-Яа-яЁё]{4,}")


def _repair_mojibake(text: str) -> str:
    """Undoes a CP1251 payload that was decoded as Latin-1.

    MyMemory returns translation-memory hits exactly as whoever uploaded them
    stored them, so Russian entries can arrive as "Óñòàíîâêà Windows" instead
    of "Установка Windows". The text is already broken when it reaches us."""
    if not text or any("Ѐ" <= character <= "ӿ" for character in text):
        return text
    try:
        candidate = text.encode("latin-1").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    if len(_CYRILLIC_RUN_RE.findall(candidate)) >= 2:
        return candidate
    return text


def _postprocess_translated_text(text: str, target_lang: str) -> str:
    if not text:
        return text
    text = _repair_mojibake(text)
    if target_lang != "ru":
        return text
    return _RU_BAD_SUBTITLE_CREDIT_RE.sub("Субтитры М.К.", text)


def _translate_known_meta_text(text: str, source_lang: str, target_lang: str) -> str | None:
    if source_lang == "tr" and target_lang == "ru":
        compact = re.sub(r"\s+", " ", text).strip()
        if _TURKISH_SUBTITLES_CREDIT_RE.match(compact):
            return "Субтитры М.К."

    if source_lang != "vi" or target_lang != "ru":
        return None

    compact = re.sub(r"\s+", " ", text).strip()
    match = _VI_SUBSCRIBE_RE.search(compact)
    if not match:
        return None

    channel = match.group("channel").strip(" .,:;!?-")
    channel = re.sub(r"\s+", " ", channel)
    if not channel:
        channel = "канал"
    if len(channel) > 80:
        channel = channel[:80].rsplit(" ", 1)[0].strip() or channel[:80].strip()
    return f"Подпишитесь на канал {channel}, чтобы не пропустить новые видео."


def _looks_bad_machine_translation(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    if "�" in compact:
        return True
    if _looks_like_translator_error(compact):
        return True
    question_marks = compact.count("?")
    return question_marks >= 3 and question_marks / max(1, len(compact)) > 0.03


# MyMemory answers over-long or rate-limited requests with HTTP 200 and puts its
# own error message where the translation belongs, so it reads as a successful
# result. Left unchecked it becomes a subtitle line - and, worse, gets cached.
_TRANSLATOR_ERROR_RE = re.compile(
    r"query length limit exceeded|максимально допустимое количество запросов"
    r"|превышен лимит длины запроса|mymemory warning|please contact us",
    re.IGNORECASE,
)


def _looks_like_translator_error(text: str) -> bool:
    return bool(_TRANSLATOR_ERROR_RE.search(text))
