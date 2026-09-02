"""The language menus, shared by every process that needs to name a language.

Kept out of bot.py so job_runner can use them too: that module deliberately
avoids importing bot.py, which pulls in the whole ML pipeline. Duplicating the
list instead is how settings drift apart - see the artifact ratio, which was
hardcoded in three places and silently ignored the setting.
"""

from __future__ import annotations

SOURCE_LANGS = [
    ("auto", "Авто"),
    ("vi", "Вьетнамский"),
    ("ru", "Русский"),
    ("en", "Английский"),
    ("he", "Иврит"),
    ("ko", "Корейский"),
    ("zh", "Китайский"),
    ("tr", "Турецкий"),
    ("ms", "Малайзийский"),
    ("hi", "Хинди"),
    ("ja", "Японский"),
    ("th", "Тайский"),
    ("de", "Немецкий"),
    ("ar", "Арабский"),
    ("uk", "Украинский"),
    ("id", "Индонезийский"),
    ("pl", "Польский"),
    ("fr", "Французский"),
    ("az", "Азербайджанский"),
    ("es", "Испанский"),
    ("it", "Итальянский"),
    ("pt", "Португальский"),
    ("fa", "Персидский"),
    ("nl", "Нидерландский"),
    ("sv", "Шведский"),
    ("cs", "Чешский"),
    ("el", "Греческий"),
    ("ro", "Румынский"),
    ("hu", "Венгерский"),
    ("fi", "Финский"),
    ("da", "Датский"),
    ("no", "Норвежский"),
    ("bg", "Болгарский"),
    ("sk", "Словацкий"),
    ("sl", "Словенский"),
    ("lt", "Литовский"),
    ("lv", "Латышский"),
    ("et", "Эстонский"),
    ("sq", "Албанский"),
    ("ca", "Каталанский"),
    ("gl", "Галисийский"),
    ("eu", "Баскский"),
    ("bn", "Бенгальский"),
    ("ur", "Урду"),
    ("sw", "Суахили"),
    ("tl", "Тагальский"),
]

TARGET_LANGS = [
    ("ru", "Русский"),
    ("uk", "Украинский"),
    ("en", "Английский"),
]


def source_lang_label(value: object) -> str:
    code = str(value or "auto").strip()
    if code == "auto":
        return "Любой язык"
    return next((label for item, label in SOURCE_LANGS if item == code), code)


def target_lang_label(value: object) -> str:
    code = str(value or "").strip()
    return next((label for item, label in TARGET_LANGS if item == code), code)


def transcript_header(job: object) -> str:
    """Which languages a job used, for the top of its transcript.

    The bot shows this while a job runs and then it is gone, so afterwards
    there was no way to tell what a given video had been dubbed from.
    """
    if not isinstance(job, dict) or not job:
        return ""
    source = source_lang_label(job.get("source_lang") or "auto")
    target = target_lang_label(job.get("target_lang"))
    if not target:
        return f"[{source}]"
    return f"[{source} → {target}]"
