from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

from laladub.models import DubConfig, Segment
from laladub.srt import read_srt, write_srt
from laladub.translation import translate_text_chain


CHAINS = [
    ["ru", "vi", "en", "ru"],
    ["ru", "ja", "ko", "en", "ru"],
    ["ru", "tr", "ar", "he", "en", "ru"],
    ["ru", "zh", "ja", "en", "ru"],
]

SECOND_CHAINS = [
    ["ru", "ms", "he", "en", "ru"],
    ["ru", "ja", "tr", "en", "ru"],
    ["ru", "vi", "ko", "en", "ru"],
]

PHONETIC_VARIANTS = {
    "гена": ["Джена", "Генай", "Генна"],
    "геннадий": ["Генадия", "Дженнадий", "Генандий"],
    "чебурашка": ["Чевлашка", "Чебураха", "Чеворшка"],
    "полотенце": ["полотенза", "потенциг", "полотенсия"],
    "полотенца": ["полотензы", "потенцига", "полотенсии"],
    "крокодил": ["кракодил", "крокодиль", "крокадила"],
    "ванную": ["вандай", "ванную комнату", "ваннуй"],
    "ванной": ["ванней", "вандальной", "ванной комнате"],
    "косяк": ["косак", "косячек", "космик"],
    "косячок": ["косяжок", "косачок", "косячер"],
    "телевизор": ["телевизион", "телевизер", "телевизорный"],
    "шкаф": ["шкаб", "шкафий", "шкав"],
}


def _rng(seed: str, index: int, label: str) -> random.Random:
    raw = f"{seed}|{index}|{label}".encode("utf-8", errors="ignore")
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big"))


def _chain_for(seed: str, index: int, second: bool = False) -> list[str]:
    choices = SECOND_CHAINS if second else CHAINS
    return _rng(seed, index, "second" if second else "first").choice(choices)


def _cyrillic_fragment(text: str) -> str:
    tokens = re.findall(r"[А-Яа-яЁё]+|\d+|[,.:;!?-]", text)
    words = [token for token in tokens if re.fullmatch(r"[А-Яа-яЁё]+", token)]
    if len(words) < 5:
        return ""
    result = " ".join(tokens)
    result = re.sub(r"\s+([,.:;!?])", r"\1", result)
    return re.sub(r"\s+", " ", result).strip(" ,.-")


def _select_forced_fragments(forced: list[Segment], limit: int) -> dict[int, str]:
    ranked: list[tuple[int, int, str]] = []
    for index, segment in enumerate(forced):
        fragment = _cyrillic_fragment(segment.text)
        if not fragment:
            continue
        unique = len(set(fragment.casefold().split()))
        ranked.append((unique, index, fragment))
    ranked.sort(reverse=True)
    return {index: fragment for _score, index, fragment in ranked[:limit]}


def _phonetic_damage(text: str, *, seed: str, index: int, strength: float) -> str:
    rng = _rng(seed, index, f"phonetic-{strength:.2f}")

    def named_replacement(match: re.Match[str]) -> str:
        word = match.group(0)
        variants = PHONETIC_VARIANTS.get(word.casefold())
        if not variants or rng.random() > strength:
            return word
        replacement = rng.choice(variants)
        return replacement if word[:1].isupper() else replacement.casefold()

    names = "|".join(sorted((re.escape(item) for item in PHONETIC_VARIANTS), key=len, reverse=True))
    damaged = re.sub(rf"\b(?:{names})\b", named_replacement, text, flags=re.IGNORECASE)
    words = damaged.split()
    for position, word in enumerate(words):
        bare = re.sub(r"[^А-Яа-яЁё]", "", word)
        if len(bare) < 6 or rng.random() > strength * 0.38:
            continue
        vowels = [offset for offset, char in enumerate(word) if char.casefold() in "аеёиоуыэюя"]
        if len(vowels) < 2:
            continue
        offset = rng.choice(vowels[1:])
        replacement = rng.choice("аеоиуы")
        words[position] = word[:offset] + replacement + word[offset + 1 :]
    damaged = " ".join(words)
    if strength >= 0.5:
        damaged = re.sub(r"\bчто\b", "што", damaged, flags=re.IGNORECASE)
        damaged = re.sub(r"\bсейчас\b", "щас", damaged, flags=re.IGNORECASE)
        damaged = re.sub(r"\bпотому что\b", "патамушта", damaged, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", damaged).strip()


def _translate(text: str, chain: list[str], config: DubConfig) -> str:
    try:
        return translate_text_chain(text, chain, config).strip() or text
    except Exception as exc:
        print(f"translation fallback {' -> '.join(chain)}: {type(exc).__name__}: {exc}")
        return text


def build_variants(
    clean: list[Segment],
    forced: list[Segment],
    config: DubConfig,
    seed: str,
) -> list[list[Segment]]:
    artifact_limit = max(1, round(len(clean) * 0.20))
    forced_fragments = _select_forced_fragments(forced, artifact_limit)
    variants = [[], [], []]
    for index, segment in enumerate(clean):
        semantic = _translate(segment.text, _chain_for(seed, index), config)
        mild = _phonetic_damage(semantic, seed=seed, index=index, strength=0.16)

        mixed = semantic
        forced_fragment = forced_fragments.get(index)
        if forced_fragment:
            if _rng(seed, index, "mix").random() < 0.5:
                mixed = f"{semantic}. {forced_fragment}"
            else:
                mixed = f"{forced_fragment}. {semantic}"
        mixed = _phonetic_damage(mixed, seed=seed, index=index, strength=0.38)

        destroyed = mixed
        if _rng(seed, index, "pass2-enabled").random() < 0.72:
            destroyed = _translate(destroyed, _chain_for(seed, index, second=True), config)
        destroyed = _phonetic_damage(destroyed, seed=seed, index=index, strength=0.68)

        for bucket, text in zip(variants, (mild, mixed, destroyed)):
            bucket.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    translated_text=text,
                )
            )
    return variants


def _write_plain(path: Path, segments: list[Segment]) -> None:
    path.write_text("\n".join(segment.spoken_text for segment in segments) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build text-only LaLaDub chaos variants from a saved job.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default="laladub-text-chaos")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = read_srt(workdir / "debug" / "timing_reference.srt", translated=False)
    forced = read_srt(workdir / "source.srt", translated=False)
    if not clean:
        raise RuntimeError("timing_reference.srt is empty")

    config = DubConfig(
        output=output_dir / "unused.mp4",
        workdir=output_dir,
        translator="hybrid",
        source_lang="ru",
        target_lang="ru",
    )
    variants = build_variants(clean, forced, config, args.seed)
    names = ["01_semantic", "02_hybrid", "03_destroyed"]
    for name, segments in zip(names, variants):
        write_srt(output_dir / f"{name}.srt", segments, translated=True)
        _write_plain(output_dir / f"{name}.txt", segments)
        print(f"wrote {name}: segments={len(segments)} chars={sum(len(item.spoken_text) for item in segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
