from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMParams(BaseModel):
    model: str | None = Field(default=None, description="LLM model id")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=1.5)
    presence_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    format: Literal["wav", "base64"] | None = None
    speed: float | None = Field(default=None, ge=0.5, le=2.0, description="Speech speed factor (1.0 = normal, >1.0 = faster, <1.0 = slower)")
    max_silence_sec: float | None = Field(default=None, ge=0.05, le=2.0, description="Compress silent regions longer than this value (seconds). None = no compression.")


class ReferenceConfig(BaseModel):
    type: Literal["base64", "preset"]
    data: str | None = None
    preset_id: str | None = None


class BestOfNConfig(BaseModel):
    enabled: bool | None = Field(
        default=None, description="Enable best-of-n selection for this request."
    )
    n: int | None = Field(default=None, ge=1, description="Number of candidates to generate.")
    language: Literal["ja", "en", "auto"] | None = None


class TTSRequest(BaseModel):
    text: str
    reference: ReferenceConfig | None = None
    llm: LLMParams | None = None
    output: OutputConfig | None = None
    best_of_n: BestOfNConfig | None = None


class TTSTimings(BaseModel):
    llm_sec: float
    parse_sec: float
    codec_sec: float
    total_sec: float
    best_of_n_sec: float | None = None
    asr_sec: float | None = None


class TTSResponse(BaseModel):
    audio: str
    format: Literal["base64"]
    sample_rate: int
    token_count: int
    timings: TTSTimings
    normalized_text: str


class TTSStreamChunk(BaseModel):
    """ストリーミングTTSの音声チャンクイベント。"""
    event: Literal["chunk"] = "chunk"
    audio: str  # base64エンコードされたraw PCM (float32, mono)
    chunk_index: int
    sample_rate: int
    token_count: int  # このチャンクのトークン数


class TTSStreamDone(BaseModel):
    """ストリーミングTTS完了イベント。"""
    event: Literal["done"] = "done"
    total_chunks: int
    total_token_count: int
    sample_rate: int
    timings: TTSTimings
    normalized_text: str


class TTSStreamError(BaseModel):
    """ストリーミングTTS中のエラーイベント。"""
    event: Literal["error"] = "error"
    detail: str
