from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from .models import DubConfig, Segment


class TranslationError(RuntimeError):
    pass


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


def _get_argos_translation(translate_module: object, source_lang: str, target_lang: str) -> object | None:
    try:
        return translate_module.get_translation_from_codes(source_lang, target_lang)
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


def _postprocess_translated_text(text: str, target_lang: str) -> str:
    if target_lang != "ru" or not text:
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
    question_marks = compact.count("?")
    return question_marks >= 3 and question_marks / max(1, len(compact)) > 0.03
