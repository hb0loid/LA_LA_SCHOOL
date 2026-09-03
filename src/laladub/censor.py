from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Segment


@dataclass(frozen=True, slots=True)
class CensorResult:
    segments: list[Segment]
    changed_segments: int
    replacements: int


# Experimental intentionally broad bank. It is used only for parody/glitch
# replacement after translation; it is not a safety classifier.
_BANNED_TERMS = [
    # English profanity / insults
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "mf", "shit", "bullshit",
    "bitch", "bitches", "bastard", "asshole", "dick", "cock", "cocksucker", "pussy",
    "cunt", "twat", "whore", "slut", "hoe", "damn", "goddamn", "crap", "jerk",
    "prick", "wanker", "retard", "retarded", "moron", "idiot", "dumbass",
    # English racial / identity slurs and hateful insults
    "nigger", "nigga", "niggah", "niglet", "coon", "spic", "wetback", "chink",
    "gook", "kike", "yid", "raghead", "sandnigger", "towelhead", "beaner",
    "cracker", "honky", "gyppo", "paki", "kaffir", "jap", "zipperhead",
    "faggot", "fag", "tranny", "shemale", "dyke",
    # Russian profanity / insults
    "блядь", "блять", "сука", "суки", "хуй", "хуя", "хуе", "хуи", "хер", "пизда",
    "пиздец", "пизду", "пизды", "ебать", "ебал", "ебаный", "ёбаный", "ебанутый",
    "ёбнутый", "еблан", "уебок", "уёбок", "мудак", "мразь", "гандон", "долбоеб",
    "долбоёб", "пидор", "пидорас", "шлюха", "сучка", "жопа", "дерьмо",
    "педик", "педераст", "гомик", "проститутка", "шалава", "потаскуха", "давалка",
    # Russian hateful slurs
    "ниггер", "нига", "чурка", "чурки", "хач", "хачи", "жид", "жиды", "москаль",
    "кацап", "хохол", "укроп", "пендос", "пиндос", "пиндосы", "черножопый",
    "черномазый", "узкоглазый",
    # Common translit / obfuscated forms
    "blyat", "blyad", "suka", "cyka", "nahui", "nahuy", "huy", "hui", "pizda",
    "pizdec", "pidor", "pidoras", "mudak", "uebok", "ebat", "yebat", "yobany",
    "pedik", "pederast", "shluha", "shlyuha", "shlyukha", "shalava",
]

_BANNED_PHRASES = [
    "kill yourself",
    "go kill yourself",
    "go back to your country",
    "white power",
    "sieg heil",
    "heil hitler",
    "gas the jews",
    "убей себя",
    "иди убейся",
    "вали в свою страну",
]

_BANNED_REGEXES = [
    # Slur stems / inflected forms. These are deliberately aggressive for the
    # experimental comedy-censor mode, including forms that may be contextually
    # neutral in normal language.
    r"\bнигг[еэ]р\w*\b",
    r"\bн[еэ]гр\w*\b",
    r"\bnigg(?:er|a|ah|uh|let)?s?\b",
    r"\bchink\w*\b",
    r"\bgook\w*\b",
    r"\bspic\w*\b",
    r"\bkike\w*\b",
    r"\bfagg?ot\w*\b",
    r"\btrann(?:y|ies)\b",
    r"\bшлюх\w*\b",
    r"\bпроститут\w*\b",
    r"\bшалав\w*\b",
    r"\bпотаскух\w*\b",
    r"\bпедик\w*\b",
    r"\bпедераст\w*\b",
    r"\bпид(?:ор|ар|ерас|орас)\w*\b",
    r"\bгомик\w*\b",
    # Hate/violence constructions.
    r"\b(?:i\s+hate|hate)\s+(?:those\s+|all\s+)?(?:nigg\w+|black\s+people|jews|gays|fags|trann\w+)\b",
    r"\bя\s+ненавижу\s+(?:этих\s+|всех\s+)?[^\s,.!?;:]{2,24}\b",
    r"\b(?:убей|убить|застрелить|расстрелять)\s+(?:этого|эту|этих|всех)?\s*[^\s,.!?;:]{2,24}\b",
]

_REPLACEMENTS = [
    "[censored]",
    "[redacted]",
    "[непереводимый фрагмент]",
    "[бот отказался это произносить]",
    "[фраза удалена модерацией]",
    "[я не могу помочь с этим запросом]",
    "[не могу перевести эту фразу]",
    "[тут было что-то запрещённое]",
    "[текст скрыт из соображений безопасности]",
    "[предупреждение: токсичный фрагмент]",
    "[неподходящая лексика удалена]",
    "[системное предупреждение вместо слова]",
    "[контент заблокирован]",
    "[оскорбление заменено]",
    "[нецензурный шум]",
    "[эта реплика не прошла фильтр]",
    "[переводчик делает вид, что ничего не слышал]",
    "[фраза растворилась в модерации]",
    "[здесь бот молчит]",
    "[тут должен был быть плохой текст]",
    "[сообщение скрыто]",
    "[недопустимая лексика]",
    "[не могу озвучить это]",
    "[запрос отклонён]",
    "[цензура сработала]",
    "[это слово отправлено в теневой бан]",
    "[моральный фильтр упал сверху]",
    "[слишком остро для перевода]",
    "[автоматическая замена]",
    "[нейросеть отказалась продолжать]",
    "[не прошло]",
    "[не скажу]",
    "[слово удалено]",
    "[слово скрыто]",
    "[фраза скрыта]",
    "[цензурный соус]",
]


_CENSOR_DIR = Path(__file__).resolve().parents[2] / "assets" / "censor"


def _load_bank(filename: str, fallback: list[str]) -> list[str]:
    path = _CENSOR_DIR / filename
    if not path.is_file():
        return fallback
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        print(f"      Experimental censor bank fallback: {path} ({type(exc).__name__}: {exc})")
        return fallback
    values = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.append(stripped)
    return values or fallback


def _compile_terms(terms: list[str]) -> re.Pattern[str]:
    if not terms:
        return re.compile(r"(?!x)x")
    return re.compile(
        r"(?<![\w])(" + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)) + r")(?![\w])",
        re.IGNORECASE | re.UNICODE,
    )


def _compile_regexes(regexes: list[str]) -> re.Pattern[str]:
    if not regexes:
        return re.compile(r"(?!x)x")
    return re.compile("(" + "|".join(regexes) + ")", re.IGNORECASE | re.UNICODE)


_ACTIVE_BANNED_TERMS = _load_bank("banned_terms.txt", _BANNED_TERMS)
_ACTIVE_BANNED_PHRASES = _load_bank("banned_phrases.txt", _BANNED_PHRASES)
_ACTIVE_BANNED_REGEXES = _load_bank("banned_regexes.txt", _BANNED_REGEXES)
_ACTIVE_REPLACEMENTS = _load_bank("replacements.txt", _REPLACEMENTS)

_TERM_PATTERN = _compile_terms(_ACTIVE_BANNED_TERMS)
_PHRASE_PATTERN = _compile_terms(_ACTIVE_BANNED_PHRASES)
_REGEX_PATTERN = _compile_regexes(_ACTIVE_BANNED_REGEXES)


def apply_censor_to_segments(
    segments: list[Segment],
    *,
    percent: int,
    seed: str | None,
) -> CensorResult:
    percent = max(0, min(100, int(percent or 0)))
    if percent <= 0 or not segments:
        return CensorResult(segments=segments, changed_segments=0, replacements=0)

    changed_segments = 0
    replacements = 0
    seed_base = str(seed or "")

    for segment_index, segment in enumerate(segments):
        source = segment.translated_text if segment.translated_text is not None else segment.text
        censored, count = censor_text(source, percent=percent, seed=f"{seed_base}|{segment_index}")
        if count:
            replacements += count
            changed_segments += 1
            if segment.translated_text is not None:
                segment.translated_text = censored
            else:
                segment.text = censored

    if replacements:
        print(f"      Experimental censor: changed={changed_segments}/{len(segments)}, replacements={replacements}, percent={percent}")
    return CensorResult(segments=segments, changed_segments=changed_segments, replacements=replacements)


def censor_text(text: str, *, percent: int, seed: str | None = None) -> tuple[str, int]:
    percent = max(0, min(100, int(percent or 0)))
    if percent <= 0 or not text:
        return text, 0

    replacements = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replacements
        token = match.group(0)
        if _stable_int(f"{seed}|chance|{match.start()}|{token}") % 100 >= percent:
            return token
        replacement_key = f"{seed}|replacement|{match.start()}|{token}"
        replacement_pool = _replacement_pool(token, replacement_key)
        replacement_index = _stable_int(replacement_key) % len(replacement_pool)
        replacements += 1
        return replacement_pool[replacement_index]

    current = _REGEX_PATTERN.sub(repl, text)
    current = _PHRASE_PATTERN.sub(repl, current)
    current = _TERM_PATTERN.sub(repl, current)
    return current, replacements


def _replacement_pool(token: str, key: str) -> list[str]:
    """Usually keep a one/two-word match compact, while preserving occasional
    long comedy warnings from the original bank."""
    token_words = re.findall(r"[\wёЁ]+", token, flags=re.UNICODE)
    if len(token_words) > 2:
        return _ACTIVE_REPLACEMENTS
    short = [
        replacement
        for replacement in _ACTIVE_REPLACEMENTS
        if 1 <= len(re.findall(r"[\wёЁ]+", replacement, flags=re.UNICODE)) <= 2
    ]
    if short and _stable_int(f"{key}|prefer-short") % 100 < 85:
        return short
    return _ACTIVE_REPLACEMENTS


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
