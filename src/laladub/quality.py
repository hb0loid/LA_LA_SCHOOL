from __future__ import annotations

import re

from .models import Segment


_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?…。！？]+[.!?…。！？]*", re.UNICODE)


def collapse_repetitions_in_segments(
    segments: list[Segment],
    *,
    max_phrase_repeats: int = 2,
    max_word_repeats: int = 3,
    max_ngram_words: int = 10,
) -> list[Segment]:
    result: list[Segment] = []
    previous_key = ""
    previous_count = 0
    changed_text = 0
    dropped_segments = 0

    for segment in segments:
        if segment.translated_text is not None:
            collapsed = collapse_repetitions(
                segment.translated_text,
                max_phrase_repeats=max_phrase_repeats,
                max_word_repeats=max_word_repeats,
                max_ngram_words=max_ngram_words,
            )
            if collapsed != segment.translated_text:
                changed_text += 1
            segment.translated_text = collapsed
        else:
            collapsed = collapse_repetitions(
                segment.text,
                max_phrase_repeats=max_phrase_repeats,
                max_word_repeats=max_word_repeats,
                max_ngram_words=max_ngram_words,
            )
            if collapsed != segment.text:
                changed_text += 1
            segment.text = collapsed

        key = _normalize_for_repeat_key(segment.spoken_text)
        if key and key == previous_key:
            previous_count += 1
        else:
            previous_key = key
            previous_count = 1

        if key and previous_count > max_phrase_repeats:
            dropped_segments += 1
            continue
        result.append(segment)

    if changed_text or dropped_segments:
        print(f"      Repetition cleanup: changed={changed_text}, dropped={dropped_segments}")
    return result


def clamp_obvious_word_repeats_in_segments(
    segments: list[Segment],
    *,
    max_word_repeats: int = 3,
) -> tuple[list[Segment], int]:
    changed = 0
    for segment in segments:
        if segment.translated_text is not None:
            clamped = clamp_obvious_word_repeats(segment.translated_text, max_word_repeats=max_word_repeats)
            if clamped != segment.translated_text:
                segment.translated_text = clamped
                changed += 1
        else:
            clamped = clamp_obvious_word_repeats(segment.text, max_word_repeats=max_word_repeats)
            if clamped != segment.text:
                segment.text = clamped
                changed += 1
    if changed:
        print(f"      Obvious word repeat clamp: changed={changed}/{len(segments)}, max={max_word_repeats}")
    return segments, changed


def clamp_obvious_word_repeats(text: str, *, max_word_repeats: int = 3) -> str:
    max_word_repeats = max(1, int(max_word_repeats))
    if not text:
        return text

    text = _clamp_obvious_phrase_repeats(text, max_phrase_repeats=max_word_repeats)

    pattern = re.compile(
        r"\b(?P<word>[^\W_]+(?:['вЂ™-][^\W_]+)*)\b"
        r"(?P<tail>(?:[\s,.;:!?вЂ¦гЂ‚пјЃпјџ-]+(?P=word)\b){"
        + str(max_word_repeats)
        + r",})",
        re.IGNORECASE | re.UNICODE,
    )

    def replace(match: re.Match[str]) -> str:
        word = match.group("word")
        return " ".join([word] * max_word_repeats)

    previous = None
    current = text
    while previous != current:
        previous = current
        current = pattern.sub(replace, current)
    return current


def _clamp_obvious_phrase_repeats(text: str, *, max_phrase_repeats: int = 3) -> str:
    max_phrase_repeats = max(1, int(max_phrase_repeats))
    word = r"[^\W_]+(?:['вЂ™-][^\W_]+)*"
    separator = r"[\s,.;:!?вЂ¦гЂ‚пјЃпјџ-]+"
    current = text

    for phrase_words in range(4, 1, -1):
        phrase = rf"\b{word}\b" + (rf"(?:\s+\b{word}\b)" * (phrase_words - 1))
        pattern = re.compile(
            rf"(?P<phrase>{phrase})(?P<tail>(?:{separator}(?P=phrase)\b){{{max_phrase_repeats},}})",
            re.IGNORECASE | re.UNICODE,
        )

        def replace(match: re.Match[str]) -> str:
            return " ".join([match.group("phrase")] * max_phrase_repeats)

        previous = None
        while previous != current:
            previous = current
            current = pattern.sub(replace, current)

    return current


def is_repetitive_loop(
    segments: list[Segment],
    *,
    min_segments: int = 8,
    repeated_share: float = 0.55,
) -> bool:
    keys = []
    for segment in segments:
        key = _normalize_for_repeat_key(segment.spoken_text or segment.text)
        if key:
            keys.append(key)
    if len(keys) < min_segments:
        return False

    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    most_common = max(counts.values(), default=0)
    return most_common / len(keys) >= repeated_share


def collapse_repetitions(
    text: str,
    *,
    max_phrase_repeats: int = 2,
    max_word_repeats: int = 3,
    max_ngram_words: int = 10,
) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    sentence_chunks = _SENTENCE_RE.findall(text) or [text]
    collapsed_sentences: list[str] = []
    previous_key = ""
    previous_count = 0

    for chunk in sentence_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        key = _normalize_for_repeat_key(chunk)
        if key and key == previous_key:
            previous_count += 1
        else:
            previous_key = key
            previous_count = 1
        if key and previous_count > max_phrase_repeats:
            continue
        collapsed_sentences.append(
            _collapse_ngram_repeats(
                chunk,
                max_phrase_repeats=max_phrase_repeats,
                max_word_repeats=max_word_repeats,
                max_ngram_words=max_ngram_words,
            )
        )

    return " ".join(part for part in collapsed_sentences if part).strip()


def _collapse_ngram_repeats(
    text: str,
    *,
    max_phrase_repeats: int,
    max_word_repeats: int,
    max_ngram_words: int,
) -> str:
    words = _WORD_RE.findall(text)
    if len(words) < 4:
        return _collapse_single_word_repeats(text, max_word_repeats=max_word_repeats)

    normalized = [_normalize_word(word) for word in words]
    output: list[str] = []
    changed = False
    index = 0

    while index < len(words):
        matched = False
        max_size = min(max_ngram_words, (len(words) - index) // (max_phrase_repeats + 1))
        for size in range(max_size, 0, -1):
            phrase = normalized[index : index + size]
            if not any(phrase):
                continue
            if size > 1 and len({word for word in phrase if word}) == 1:
                continue
            count = 1
            while (
                index + (count + 1) * size <= len(words)
                and normalized[index + count * size : index + (count + 1) * size] == phrase
            ):
                count += 1

            allowed = max_word_repeats if size == 1 else max_phrase_repeats
            if count > allowed:
                for _ in range(allowed):
                    output.extend(words[index : index + size])
                index += count * size
                changed = True
                matched = True
                break

        if matched:
            continue

        output.append(words[index])
        index += 1

    if not changed:
        return _collapse_single_word_repeats(text, max_word_repeats=max_word_repeats)

    suffix = _terminal_punctuation(text)
    collapsed = " ".join(output).strip()
    if suffix and not collapsed.endswith(suffix):
        collapsed += suffix
    return collapsed


def _collapse_single_word_repeats(text: str, *, max_word_repeats: int) -> str:
    pattern = re.compile(r"(\b[^\W_]+(?:['’-][^\W_]+)*\b)(?:\s+\1\b)+", re.IGNORECASE | re.UNICODE)

    def replace(match: re.Match[str]) -> str:
        words = match.group(0).split()
        if len(words) <= max_word_repeats:
            return match.group(0)
        return " ".join(words[:max_word_repeats])

    return pattern.sub(replace, text)


def _normalize_for_repeat_key(text: str) -> str:
    words = [_normalize_word(word) for word in _WORD_RE.findall(text)]
    return " ".join(word for word in words if word)


def _normalize_word(word: str) -> str:
    return word.casefold().strip(".,!?;:()[]{}\"'«»“”„…。！？")


def _terminal_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?…。！？":
        return stripped[-1]
    return ""
