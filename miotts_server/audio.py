from __future__ import annotations

import io
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import soundfile as sf
import torch
from miocodec.util import load_audio


@contextmanager
def _temp_audio_file(data: bytes, suffix: str = ".wav") -> Generator[Path, None, None]:
    """Create a temporary file with audio data that is automatically cleaned up."""
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            path = Path(tmp.name)
        yield path
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def load_reference_audio_bytes(data: bytes, sample_rate: int) -> torch.Tensor:
    with _temp_audio_file(data, suffix=".wav") as path:
        return load_audio(str(path), sample_rate=sample_rate)


def load_reference_audio_path(path: str, sample_rate: int) -> torch.Tensor:
    return load_audio(path, sample_rate=sample_rate)


def write_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio = ensure_1d(audio)
    if audio.dtype not in (torch.float32, torch.float64):
        audio = audio.float()
    buffer = io.BytesIO()
    sf.write(buffer, audio.cpu().numpy(), sample_rate, format="WAV")
    return buffer.getvalue()


def write_pcm_f32_bytes(audio: torch.Tensor) -> bytes:
    """float32 PCMバイト列を返す（ヘッダなし、ストリーミング用）。"""
    audio = ensure_1d(audio)
    if audio.dtype != torch.float32:
        audio = audio.float()
    return audio.cpu().numpy().tobytes()


def apply_crossfade(prev_tail: torch.Tensor, next_head: torch.Tensor) -> torch.Tensor:
    """同じ長さの prev_tail と next_head をクロスフェードで合成する。

    equal-power crossfade を使い、音量の谷間を防ぐ。
    """
    n = min(len(prev_tail), len(next_head))
    if n == 0:
        return prev_tail if len(next_head) == 0 else next_head
    t = torch.linspace(0.0, 1.0, n, device=prev_tail.device)
    # equal-power: sin/cos カーブで音量を維持
    fade_out = torch.cos(t * (torch.pi / 2))
    fade_in = torch.sin(t * (torch.pi / 2))
    return prev_tail[:n] * fade_out + next_head[:n] * fade_in


def ensure_1d(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 2 and audio.shape[0] == 1:
        return audio.squeeze(0)
    if audio.dim() == 1:
        return audio
    return audio.flatten()


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for resampling.") from exc
    if audio.is_cuda:
        audio = audio.cpu()
    if audio.dtype not in (torch.float32, torch.float64):
        audio = audio.float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    resampled = resampler(audio)
    return resampled.squeeze(0)


def compress_silence(
    audio: torch.Tensor,
    sample_rate: int,
    max_silence_sec: float = 0.3,
    silence_threshold: float = 1e-4,
    frame_sec: float = 0.02,
) -> torch.Tensor:
    """無音区間が max_silence_sec を超えないように圧縮する。

    無音判定は frame_sec 単位のフレームエネルギーで行う。
    max_silence_sec を超える連続無音フレームを max_silence_sec 分に短縮する。
    有音部分はそのまま保持する。
    """
    audio = ensure_1d(audio)
    if audio.is_cuda:
        audio = audio.cpu()
    if audio.dtype not in (torch.float32, torch.float64):
        audio = audio.float()

    frame_size = max(1, int(sample_rate * frame_sec))
    max_silent_frames = max(1, int(max_silence_sec / frame_sec))
    n_frames = audio.numel() // frame_size
    remainder = audio.numel() % frame_size

    if n_frames == 0:
        return audio

    frames = audio[: n_frames * frame_size].view(n_frames, frame_size)
    energy = frames.abs().mean(dim=1)  # shape: (n_frames,)
    is_silent = energy < silence_threshold

    kept_chunks: list[torch.Tensor] = []
    silent_run = 0

    for i, silent in enumerate(is_silent.tolist()):
        frame = frames[i]
        if silent:
            silent_run += 1
            if silent_run <= max_silent_frames:
                kept_chunks.append(frame)
            # max_silent_frames を超えたフレームは捨てる
        else:
            silent_run = 0
            kept_chunks.append(frame)

    if remainder > 0:
        kept_chunks.append(audio[n_frames * frame_size :])

    if not kept_chunks:
        return audio

    return torch.cat(kept_chunks)
