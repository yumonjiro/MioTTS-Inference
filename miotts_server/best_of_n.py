from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

import nltk
import pyopenjtalk
import torch
from g2p_en import G2p

from .asr import ASRService
from .audio import ensure_1d

logger = logging.getLogger(__name__)

_TOKEN_RATE_HZ = 25.0

# Seconds-per-phoneme defaults and clamps
_SPP_DEFAULT = {"en": 0.085, "ja": 0.10, "other": 0.11}
_SPP_MINMAX = {
    "en": (0.06, 0.12),
    "ja": (0.07, 0.15),
    "other": (0.07, 0.18),
}

_SILENCE_RATIO_THRESHOLD = 0.2
_SILENCE_LONG_THRESHOLD_SEC = 2.0

_WEIGHT_LEN = 0.4
_WEIGHT_SILENCE = 0.4
_WEIGHT_REPEAT = 0.2
_HYBRID_HEUR_WEIGHT = 0.3

_g2p_en: G2p | None = None
_g2p_en_lock = threading.Lock()


def _shorten(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class BestOfNCandidate:
    tokens: list[int]
    audio: torch.Tensor
    asr_text: str | None = None
    asr_error: float | None = None
    repeat_penalty: float = 0.0
    length_penalty: float = 0.0
    silence_penalty: float = 0.0
    score: float | None = None


def detect_language(text: str) -> str:
    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return "auto"
    ja_count = sum(1 for ch in text if _is_japanese_char(ch))
    en_count = sum(1 for ch in text if _is_ascii_alpha(ch))
    ja_ratio = ja_count / total
    en_ratio = en_count / total
    logger.debug(
        "Language detect: ja_ratio=%.3f en_ratio=%.3f total=%d text='%s'",
        ja_ratio,
        en_ratio,
        total,
        _shorten(text),
    )
    if ja_ratio >= 0.2:
        return "ja"
    if en_ratio >= 0.5:
        return "en"
    return "auto"


def resolve_language(text: str, preferred: str | None) -> str:
    if preferred in {"ja", "en"}:
        return preferred
    return detect_language(text)


async def score_candidates(
    text: str,
    candidates: list[BestOfNCandidate],
    sample_rate: int,
    language: str,
    asr_service: ASRService | None,
) -> tuple[int, float]:
    if not candidates:
        raise ValueError("No candidates to score.")

    resolved_lang = resolve_language(text, language)
    logger.debug(
        "Scoring %d candidates (lang=%s)",
        len(candidates),
        resolved_lang,
    )
    for candidate in candidates:
        candidate.repeat_penalty = _repeat_penalty(candidate.tokens)
        candidate.length_penalty = _length_penalty(text, candidate.tokens, resolved_lang)
        candidate.silence_penalty = _silence_penalty(candidate.audio, sample_rate)

    if asr_service is None:
        raise RuntimeError("ASR is required but not available.")
    asr_sec = 0.0
    asr_sec, asr_texts = await _run_asr(asr_service, candidates, sample_rate, resolved_lang)
    ref_norm = _normalize_reference(text, resolved_lang)
    for candidate, asr_text in zip(candidates, asr_texts, strict=False):
        candidate.asr_text = asr_text
        candidate.asr_error = _asr_error(ref_norm, asr_text, resolved_lang)
    logger.debug("ASR completed in %.3fs", asr_sec)

    for candidate in candidates:
        heuristic = (
            _WEIGHT_LEN * candidate.length_penalty
            + _WEIGHT_SILENCE * candidate.silence_penalty
            + _WEIGHT_REPEAT * candidate.repeat_penalty
        )
        asr_score = candidate.asr_error if candidate.asr_error is not None else 1.0
        candidate.score = asr_score + _HYBRID_HEUR_WEIGHT * heuristic
        logger.debug(
            "Candidate score: score=%.4f asr=%.4f len=%.4f silence=%.4f repeat=%.4f",
            candidate.score if candidate.score is not None else -1.0,
            candidate.asr_error if candidate.asr_error is not None else -1.0,
            candidate.length_penalty,
            candidate.silence_penalty,
            candidate.repeat_penalty,
        )

    best_idx = min(range(len(candidates)), key=lambda i: candidates[i].score or 0.0)
    logger.debug(
        "Selected candidate index=%d score=%.4f", best_idx, candidates[best_idx].score or 0.0
    )
    return best_idx, asr_sec


def _is_ascii_alpha(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_japanese_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def _repeat_penalty(tokens: list[int]) -> float:
    ratios = []
    for n in (2, 3, 4):
        ratios.append(_ngram_repeat_ratio(tokens, n))
    return max(ratios) if ratios else 0.0


def _ngram_repeat_ratio(tokens: list[int], n: int) -> float:
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    counts: dict[tuple[int, ...], int] = {}
    for i in range(total):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    unique = len(counts)
    return 1.0 - (unique / total)


def _length_penalty(text: str, tokens: list[int], language: str) -> float:
    duration_sec = len(tokens) / _TOKEN_RATE_HZ
    lang = language if language in {"ja", "en"} else detect_language(text)
    phonemes = max(_phoneme_count(text, lang), 1)
    min_spp, max_spp = _SPP_MINMAX.get(lang, _SPP_MINMAX["other"])
    bonus = _punctuation_bonus_sec(text)
    min_expected = phonemes * min_spp + bonus
    max_expected = phonemes * max_spp + bonus
    if duration_sec <= 0 or min_expected <= 0:
        return 0.0
    if duration_sec < min_expected:
        return (min_expected - duration_sec) / min_expected
    if duration_sec > max_expected:
        return (duration_sec - max_expected) / max_expected
    return 0.0


def _ensure_nltk_data() -> None:
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("corpora/cmudict")
    except LookupError:
        try:
            nltk.download("cmudict", quiet=True)
        except Exception:
            pass


def _phoneme_count(text: str, lang: str) -> int:
    if lang == "en":
        return _phoneme_count_en(text)
    if lang == "ja":
        return _phoneme_count_ja(text)
    return max(len(text), 1)


def _phoneme_count_en(text: str) -> int:
    global _g2p_en
    if _g2p_en is None:
        with _g2p_en_lock:
            if _g2p_en is None:
                _ensure_nltk_data()
                _g2p_en = G2p()
    try:
        phonemes = _g2p_en(text)
        return len([p for p in phonemes if p and p not in {" ", "<pad>", "<s>", "</s>", "<unk>"}])
    except Exception:
        return max(len(text), 1)


def _phoneme_count_ja(text: str) -> int:
    try:
        ph = pyopenjtalk.g2p(text)
        return len([p for p in ph.split(" ") if p and p not in {"pau", "sil"}])
    except Exception:
        return max(len(text), 1)


def _punctuation_bonus_sec(text: str) -> float:
    t = text.strip()
    if not t:
        return 0.0
    major_chars = ".!?。！？"
    major = len(re.findall(r"[.!?。！？]", t))
    minor = len(re.findall(r"[、，,;；:]", t))
    if t[-1] in major_chars:
        major = max(0, major - 1)
    ellipsis = len(re.findall(r"(…|\.\.\.)", t))
    dash_pause = len(re.findall(r"(—|--)", t))
    major_bonus = major * 0.40
    minor_bonus = minor * 0.20
    ellipsis_bonus = ellipsis * 1.0
    dash_bonus = dash_pause * 0.12
    return min(10.0, major_bonus + minor_bonus + ellipsis_bonus + dash_bonus)


def _silence_penalty(audio: torch.Tensor, sample_rate: int) -> float:
    ratio, longest = _silence_stats(audio, sample_rate)
    penalty = 0.0
    if ratio > _SILENCE_RATIO_THRESHOLD:
        penalty += (ratio - _SILENCE_RATIO_THRESHOLD) / max(1e-6, 1.0 - _SILENCE_RATIO_THRESHOLD)
    if longest > _SILENCE_LONG_THRESHOLD_SEC:
        penalty += (longest - _SILENCE_LONG_THRESHOLD_SEC) / max(1e-6, _SILENCE_LONG_THRESHOLD_SEC)
    return penalty


def _silence_stats(audio: torch.Tensor, sample_rate: int) -> tuple[float, float]:
    audio = ensure_1d(audio)
    if audio.is_cuda:
        audio = audio.cpu()
    audio = audio.detach().abs()
    if audio.numel() == 0:
        return 1.0, 0.0
    frame_size = max(1, int(sample_rate * 0.02))
    frames = audio.numel() // frame_size
    if frames == 0:
        energy = audio.mean().item()
        return (1.0 if energy < 1e-4 else 0.0), audio.numel() / sample_rate
    trimmed = audio[: frames * frame_size].view(frames, frame_size)
    energy = trimmed.mean(dim=1)
    silent = energy < 1e-4
    ratio = silent.float().mean().item()
    longest = _longest_run(silent.tolist()) * frame_size / sample_rate
    return ratio, longest


def _longest_run(flags: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _normalize_reference(text: str, language: str) -> str | list[str]:
    if language == "en":
        return _normalize_for_wer(text)
    return _normalize_for_cer(text)


def _normalize_for_cer(text: str) -> str:
    lowered = text.lower()
    cleaned = []
    for ch in lowered:
        if ch.isalnum() or _is_japanese_char(ch):
            cleaned.append(ch)
    return "".join(cleaned)


def _normalize_for_wer(text: str) -> list[str]:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9']+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return []
    return lowered.split()


def _asr_error(ref_norm: str | list[str], hyp_text: str, language: str) -> float:
    if language == "en":
        hyp_norm = _normalize_for_wer(hyp_text)
        return _wer(ref_norm if isinstance(ref_norm, list) else [], hyp_norm)
    hyp_norm = _normalize_for_cer(hyp_text)
    return _cer(ref_norm if isinstance(ref_norm, str) else "", hyp_norm)


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    dist = _edit_distance(list(ref), list(hyp))
    return dist / max(1, len(ref))


def _wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    dist = _edit_distance(ref, hyp)
    return dist / max(1, len(ref))


def _edit_distance(seq_a: list, seq_b: list) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    dp = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, b in enumerate(seq_b, start=1):
            temp = dp[j]
            cost = 0 if a == b else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[-1]


def max_allowed_tokens(text: str, language: str, threshold: float = 2.0) -> int:
    """入力テキストの音素数から、許容される最大スピーチトークン数を算出する。

    計算: phonemes × max_spp × threshold × TOKEN_RATE_HZ + bonus
    threshold は max_expected_sec に対する倍率。
    """
    lang = language if language in {"ja", "en"} else detect_language(text)
    phonemes = max(_phoneme_count(text, lang), 1)
    _, max_spp = _SPP_MINMAX.get(lang, _SPP_MINMAX["other"])
    bonus = _punctuation_bonus_sec(text)
    max_expected_sec = phonemes * max_spp + bonus
    return int(max_expected_sec * threshold * _TOKEN_RATE_HZ)


def max_expected_duration(text: str, language: str, threshold: float = 2.0) -> float:
    """入力テキストの音素数から、許容される最大音声秒数を算出する。"""
    lang = language if language in {"ja", "en"} else detect_language(text)
    phonemes = max(_phoneme_count(text, lang), 1)
    _, max_spp = _SPP_MINMAX.get(lang, _SPP_MINMAX["other"])
    bonus = _punctuation_bonus_sec(text)
    return (phonemes * max_spp + bonus) * threshold


def is_token_hallucination(
    tokens: list[int], text: str, language: str, threshold: float = 2.0,
) -> bool:
    """LLM生成のスピーチトークン数が音素数ベースの動的閾値を超えていれば True。"""
    return len(tokens) > max_allowed_tokens(text, language, threshold)


def is_audio_hallucination(
    audio_sec: float, text: str, language: str, threshold: float = 2.0,
) -> bool:
    """デコード後の音声秒数が音素数ベースの動的閾値を超えていれば True。"""
    return audio_sec > max_expected_duration(text, language, threshold)


async def _run_asr(
    asr_service: ASRService,
    candidates: list[BestOfNCandidate],
    sample_rate: int,
    language: str,
) -> tuple[float, list[str]]:
    start = time.perf_counter()
    lang = language if language in {"ja", "en"} else None
    audio_list = [candidate.audio for candidate in candidates]
    texts = await asyncio.to_thread(asr_service.transcribe_batch, audio_list, sample_rate, lang)
    end = time.perf_counter()
    return end - start, texts
