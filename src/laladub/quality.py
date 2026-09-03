from __future__ import annotations

import re
from collections import Counter

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


def suppress_pathological_segment_loops(
    segments: list[Segment],
    *,
    window_segments: int = 20,
    window_seconds: float = 30.0,
    min_window_segments: int = 8,
    repeated_share: float = 0.65,
    low_diversity_share: float = 0.85,
    max_unique_loop_phrases: int = 3,
    max_kept_per_cluster: int = 3,
    max_signature_words: int = 16,
) -> list[Segment]:
    """Remove only dense cross-segment ASR loops.

    Whisper failures often arrive as hundreds of separate one-line segments, so
    the normal per-line repetition clamp cannot see them.  This pass marks a
    phrase only when it dominates a dense local window, or when almost the
    entire window alternates between no more than a few short phrases.  Sparse
    callbacks and ordinary repeated jokes are deliberately left alone.
    """
    if len(segments) < min_window_segments:
        return segments

    keys: list[str] = []
    for segment in segments:
        key = _normalize_for_repeat_key(segment.spoken_text)
        if not key or len(key.split()) > max_signature_words:
            key = ""
        keys.append(key)

    flagged: set[int] = set()
    left = 0
    for right, segment in enumerate(segments):
        left = max(left, right - max(1, window_segments) + 1)
        while left < right and segment.end - segments[left].start > window_seconds:
            left += 1
        if right - left + 1 < min_window_segments:
            continue

        window_keys = keys[left : right + 1]
        counts = Counter(key for key in window_keys if key)
        if not counts:
            continue
        window_size = len(window_keys)
        short_count = sum(counts.values())
        low_diversity_loop = (
            len(counts) <= max_unique_loop_phrases
            and short_count / window_size >= low_diversity_share
        )
        bad_keys = {
            key
            for key, count in counts.items()
            if count > max_kept_per_cluster
            and (count / window_size >= repeated_share or low_diversity_loop)
        }
        if bad_keys:
            flagged.update(index for index in range(left, right + 1) if keys[index] in bad_keys)

    if not flagged:
        return segments

    kept: list[Segment] = []
    cluster_counts: dict[str, int] = {}
    cluster_start: dict[str, float] = {}
    last_seen: dict[str, tuple[int, float]] = {}
    dropped = 0
    for index, segment in enumerate(segments):
        key = keys[index]
        if index not in flagged or not key:
            kept.append(segment)
            continue

        previous = last_seen.get(key)
        if (
            previous is None
            or index - previous[0] > window_segments
            or segment.start - previous[1] > window_seconds
            or segment.start - cluster_start.get(key, segment.start) >= window_seconds
        ):
            cluster_counts[key] = 0
            cluster_start[key] = segment.start
        last_seen[key] = (index, segment.end)
        cluster_counts[key] = cluster_counts.get(key, 0) + 1
        if cluster_counts[key] > max_kept_per_cluster:
            dropped += 1
            continue
        kept.append(segment)

    if dropped:
        print(f"      Pathological segment loop cleanup: dropped={dropped}/{len(segments)}")
    return kept


def limit_repeated_segment_phrases(
    segments: list[Segment],
    *,
    trigger_occurrences: int = 8,
    max_occurrences: int = 3,
    max_signature_words: int = 16,
) -> list[Segment]:
    """Globally cap exact phrase floods in an artifact candidate stream."""
    keys = [_normalize_for_repeat_key(segment.spoken_text) for segment in segments]
    totals = Counter(
        key for key in keys if key and len(key.split()) <= max_signature_words
    )
    flooded = {key for key, count in totals.items() if count >= trigger_occurrences}
    if not flooded:
        return segments

    seen: Counter[str] = Counter()
    kept: list[Segment] = []
    dropped = 0
    for segment, key in zip(segments, keys):
        if key in flooded:
            seen[key] += 1
            if seen[key] > max_occurrences:
                dropped += 1
                continue
        kept.append(segment)
    if dropped:
        print(f"      Repeated artifact phrase cleanup: dropped={dropped}/{len(segments)}")
    return kept


def clamp_obvious_word_repeats(text: str, *, max_word_repeats: int = 3) -> str:
    max_word_repeats = max(1, int(max_word_repeats))
    if not text:
        return text

    # Translation corruption can produce one very long token such as
    # "вакавакавака..." without separators.  The back-reference regexes below
    # are intended for repeated words and phrases; applying them to a single
    # giant token can trigger catastrophic backtracking for minutes or hours.
    # There is nothing word-level to clamp in that case, so leave it for the
    # later duration/character budget step.
    raw_words = text.split()
    if len(raw_words) < 2 or any(len(word) > 64 for word in raw_words):
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


def limit_phrase_repeats_across_segments(
    segments: list[Segment],
    *,
    max_repeats: int = 5,
    max_phrase_words: int = 4,
    min_segments: int = 40,
) -> list[Segment]:
    """Drop a short phrase once it has already been said max_repeats times.

    suppress_pathological_segment_loops only sees a phrase that dominates a
    dense local window. A translation that has collapsed spreads the same few
    phrases evenly across the whole video instead: in one job "Интервью" landed
    65 times and "Свяжитесь с нами" 29 across 308 lines, none of them dense
    enough locally to trip that pass.

    Only short phrases are counted - a long line repeating verbatim is far more
    likely to be genuine - and the first max_repeats of each are always kept,
    so a real catchphrase survives.
    """
    if max_repeats <= 0 or len(segments) < min_segments:
        return segments

    seen: dict[str, int] = {}
    kept: list[Segment] = []
    dropped = 0
    for segment in segments:
        key = _normalize_for_repeat_key(segment.spoken_text)
        if not key or len(key.split()) > max_phrase_words:
            kept.append(segment)
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > max_repeats:
            dropped += 1
            continue
        kept.append(segment)

    if dropped:
        worst = sorted(seen.items(), key=lambda item: item[1], reverse=True)[:3]
        # Plain ASCII only: this log goes to a cp1251 console on Windows, and a
        # UnicodeEncodeError here would take the whole pipeline down.
        summary = ", ".join(f"{phrase!r} x{count}" for phrase, count in worst if count > max_repeats)
        print(f"      Dropped {dropped} over-repeated line(s): {summary}", flush=True)
    return kept
