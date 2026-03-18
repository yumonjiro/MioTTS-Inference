from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class ServerConfig:
    llm_base_url: str
    llm_api_key: str | None
    llm_model: str | None
    llm_timeout: float
    codec_model_id: str
    device: str
    max_text_length: int
    presets_dir: Path
    max_reference_mb: int
    max_reference_seconds: float
    best_of_n_enabled: bool
    best_of_n_default: int
    best_of_n_max: int
    best_of_n_language: str
    asr_model: str
    asr_device: str
    asr_compute_type: str
    asr_batch_size: int
    asr_language: str
    # ハルシネーション検出リトライ
    hallucination_max_retries: int
    hallucination_token_threshold: float   # LLMトークン数の閾値倍率
    hallucination_audio_threshold: float   # デコード後音声秒数の閾値倍率


@dataclass
class DefaultLLMParams:
    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float


@dataclass
class AudioConfig:
    allowed_extensions: tuple[str, ...]


DEFAULT_ALLOWED_EXTENSIONS = (".wav", ".flac", ".ogg")


_config: ServerConfig | None = None
_llm_defaults: DefaultLLMParams | None = None
_audio_config: AudioConfig | None = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        device = os.getenv("MIOTTS_DEVICE")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_device = os.getenv("MIOTTS_ASR_DEVICE")
        if not asr_device:
            asr_device = device
        default_asr_compute_type = "float16" if asr_device.startswith("cuda") else "int8"
        presets_dir = Path(os.getenv("MIOTTS_PRESETS_DIR", "presets"))
        presets_dir = presets_dir.expanduser().resolve()
        _config = ServerConfig(
            llm_base_url=os.getenv("MIOTTS_LLM_BASE_URL", "http://localhost:8000/v1"),
            llm_api_key=os.getenv("MIOTTS_LLM_API_KEY"),
            llm_model=os.getenv("MIOTTS_LLM_MODEL"),
            llm_timeout=_env_float("MIOTTS_LLM_TIMEOUT", 120.0),
            codec_model_id=os.getenv("MIOTTS_CODEC_MODEL", "Aratako/MioCodec-25Hz-44.1kHz-v2"),
            device=device,
            max_text_length=_env_int("MIOTTS_MAX_TEXT_LENGTH", 300),
            presets_dir=presets_dir,
            max_reference_mb=_env_int("MIOTTS_MAX_REFERENCE_MB", 20),
            max_reference_seconds=20.0,
            best_of_n_enabled=_env_bool("MIOTTS_BEST_OF_N_ENABLED", False),
            best_of_n_default=_env_int("MIOTTS_BEST_OF_N_DEFAULT", 1),
            best_of_n_max=_env_int("MIOTTS_BEST_OF_N_MAX", 8),
            best_of_n_language=os.getenv("MIOTTS_BEST_OF_N_LANGUAGE", "auto"),
            asr_model=os.getenv("MIOTTS_ASR_MODEL", "openai/whisper-large-v3-turbo"),
            asr_device=asr_device,
            asr_compute_type=os.getenv("MIOTTS_ASR_COMPUTE_TYPE", default_asr_compute_type),
            asr_batch_size=_env_int("MIOTTS_ASR_BATCH_SIZE", 0),
            asr_language=os.getenv("MIOTTS_ASR_LANGUAGE", "auto"),
            hallucination_max_retries=_env_int("MIOTTS_HALLUCINATION_MAX_RETRIES", 2),
            hallucination_token_threshold=_env_float("MIOTTS_HALLUCINATION_TOKEN_THRESHOLD", 1.1),
            hallucination_audio_threshold=_env_float("MIOTTS_HALLUCINATION_AUDIO_THRESHOLD", 1.1),
        )
    return _config


def get_llm_defaults() -> DefaultLLMParams:
    global _llm_defaults
    if _llm_defaults is None:
        _llm_defaults = DefaultLLMParams(
            temperature=_env_float("MIOTTS_LLM_TEMPERATURE", 0.8),
            top_p=_env_float("MIOTTS_LLM_TOP_P", 1.0),
            max_tokens=_env_int("MIOTTS_LLM_MAX_TOKENS", 700),
            repetition_penalty=_env_float("MIOTTS_LLM_REPETITION_PENALTY", 1.0),
            presence_penalty=_env_float("MIOTTS_LLM_PRESENCE_PENALTY", 0.0),
            frequency_penalty=_env_float("MIOTTS_LLM_FREQUENCY_PENALTY", 0.0),
        )
    return _llm_defaults


def get_audio_config() -> AudioConfig:
    global _audio_config
    if _audio_config is None:
        extensions = os.getenv("MIOTTS_ALLOWED_AUDIO_EXTS")
        if extensions:
            exts = tuple(ext.strip().lower() for ext in extensions.split(",") if ext.strip())
        else:
            exts = DEFAULT_ALLOWED_EXTENSIONS
        _audio_config = AudioConfig(allowed_extensions=exts)
    return _audio_config


def reset_config() -> None:
    global _config, _llm_defaults, _audio_config
    _config = None
    _llm_defaults = None
    _audio_config = None
