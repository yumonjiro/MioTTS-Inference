from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .asr import ASRConfig, ASRService
from .audio import load_reference_audio_bytes, write_wav_bytes, compress_silence, trim_silence, time_stretch
from .best_of_n import (
    BestOfNCandidate,
    detect_language,
    is_audio_hallucination,
    is_token_hallucination,
    max_allowed_tokens,
    max_expected_duration,
    score_candidates,
)
from .codec import MioCodecService
from .config import get_audio_config, get_config, get_llm_defaults
from .llm_client import LLMClient
from .schemas import (
    BestOfNConfig,
    LLMParams,
    OutputConfig,
    ReferenceConfig,
    TTSRequest,
    TTSResponse,
    TTSTimings,
)
from .text import normalize_text, is_symbol_only
from .token_parser import parse_speech_tokens

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    _configure_torch(config)
    llm_client = LLMClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        timeout=config.llm_timeout,
    )
    codec_service = MioCodecService(
        model_id=config.codec_model_id,
        device=config.device,
        presets_dir=config.presets_dir,
    )
    codec_service.load()
    asr_service = None
    if config.best_of_n_enabled:
        try:
            asr_config = ASRConfig(
                model_id=config.asr_model,
                device=config.asr_device,
                compute_type=config.asr_compute_type,
                batch_size=config.asr_batch_size,
                language=config.asr_language,
            )
            asr_service = ASRService(asr_config)
            asr_service.load()
        except Exception as exc:
            logger.warning("ASR unavailable: %s", exc)
            asr_service = None
    app.state.llm_client = llm_client
    app.state.codec_service = codec_service
    app.state.asr_service = asr_service

    yield

    await llm_client.close()


app = FastAPI(title="MioTTS API Server", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/presets")
async def list_presets() -> dict[str, Any]:
    codec_service: MioCodecService = app.state.codec_service
    return {"presets": codec_service.list_presets()}


@app.post("/v1/tts")
async def tts_json(request: TTSRequest):
    output_format = _resolve_output_format(request.output, default_format="base64")
    result = await _run_tts(request, output_format)
    return result


@app.post("/v1/tts/file")
async def tts_file(
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    reference_preset_id: str | None = Form(None),
    model: str | None = Form(None),
    temperature: float | None = Form(None),
    top_p: float | None = Form(None),
    max_tokens: int | None = Form(None),
    repetition_penalty: float | None = Form(None),
    presence_penalty: float | None = Form(None),
    frequency_penalty: float | None = Form(None),
    output_format: str | None = Form(None),
    best_of_n_enabled: bool | None = Form(None),
    best_of_n_n: int | None = Form(None),
    best_of_n_language: str | None = Form(None),
):
    reference: ReferenceConfig | None = None
    reference_bytes: bytes | None = None
    if reference_audio is not None:
        reference_bytes = await _read_reference_file(reference_audio)
        reference = ReferenceConfig(type="base64", data="")
    elif reference_preset_id:
        reference = ReferenceConfig(type="preset", preset_id=reference_preset_id)

    try:
        best_of_n = None
        if any(
            value is not None
            for value in (
                best_of_n_enabled,
                best_of_n_n,
                best_of_n_language,
            )
        ):
            best_of_n = BestOfNConfig(
                enabled=best_of_n_enabled,
                n=best_of_n_n,
                language=best_of_n_language,
            )

        request = TTSRequest(
            text=text,
            reference=reference,
            llm=LLMParams(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            ),
            output=OutputConfig(format=output_format),
            best_of_n=best_of_n,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    output_format = _resolve_output_format(request.output, default_format="wav")
    return await _run_tts(request, output_format, reference_bytes=reference_bytes)


async def _run_tts(
    request: TTSRequest,
    output_format: str,
    reference_bytes: bytes | None = None,
):
    config = get_config()
    llm_defaults = get_llm_defaults()
    codec_service: MioCodecService = app.state.codec_service
    llm_client: LLMClient = app.state.llm_client

    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(request.text) > config.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"text is too long (max {config.max_text_length} characters)",
        )

    detected_language = detect_language(request.text)
    if detected_language == "ja":
        normalized = normalize_text(request.text)
    else:
        normalized = request.text.strip()

    # 記号のみのテキストは無音WAVを返す（LLM/codec をスキップ）
    if is_symbol_only(normalized):
        codec_service_: MioCodecService = app.state.codec_service
        sr = codec_service_.sample_rate
        silent_audio = torch.zeros(sr // 10, dtype=torch.float32)  # 0.1秒の無音
        wav_bytes = write_wav_bytes(silent_audio, sr)
        if output_format == "wav":
            return StreamingResponse(
                io.BytesIO(wav_bytes),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=tts.wav"},
            )
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        return JSONResponse(content=TTSResponse(
            audio=audio_b64,
            format="base64",
            sample_rate=sr,
            token_count=0,
            timings=TTSTimings(llm_sec=0, parse_sec=0, codec_sec=0, total_sec=0),
            normalized_text=normalized,
        ).model_dump())

    llm_params = request.llm or LLMParams()
    model = llm_params.model or config.llm_model
    temperature = (
        llm_params.temperature if llm_params.temperature is not None else llm_defaults.temperature
    )
    top_p = llm_params.top_p if llm_params.top_p is not None else llm_defaults.top_p
    max_tokens = (
        llm_params.max_tokens if llm_params.max_tokens is not None else llm_defaults.max_tokens
    )
    repetition_penalty = (
        llm_params.repetition_penalty
        if llm_params.repetition_penalty is not None
        else llm_defaults.repetition_penalty
    )
    presence_penalty = (
        llm_params.presence_penalty
        if llm_params.presence_penalty is not None
        else llm_defaults.presence_penalty
    )
    frequency_penalty = (
        llm_params.frequency_penalty
        if llm_params.frequency_penalty is not None
        else llm_defaults.frequency_penalty
    )
    messages: list[dict[str, Any]] = []
    messages.append({"role": "user", "content": normalized})

    best_of_n = _resolve_best_of_n(request, config)
    asr_service: ASRService | None = app.state.asr_service
    logger.debug(
        "Best-of-n resolved: enabled=%s n=%d lang=%s",
        best_of_n.enabled,
        best_of_n.n,
        best_of_n.language,
    )

    if not model:
        try:
            model = await llm_client.resolve_model(model)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to resolve LLM model: {exc}"
            ) from exc

    # ── max_tokens を音素数ベースで動的に制限 ─────────────────────────────
    # 入力テキストから許容トークン上限を算出し、ユーザ指定/デフォルトの
    # max_tokens と比較して小さい方を採用する。
    # これにより短い文で LLM が暴走的に長いトークン列を生成するのを防ぐ。
    hal_token_threshold = config.hallucination_token_threshold
    dynamic_limit = max_allowed_tokens(normalized, detected_language, hal_token_threshold)
    effective_max_tokens = min(max_tokens, dynamic_limit)
    if effective_max_tokens < max_tokens:
        logger.debug(
            "Dynamic max_tokens: %d → %d (phoneme-based limit for '%s')",
            max_tokens, effective_max_tokens, normalized[:40],
        )

    # ── ハルシネーション検出付き LLM リトライループ ──────────────────────
    # LLM出力のスピーチトークン数が、入力テキストの音素数から動的に算出した
    # 閾値を超えたら再生成する。best-of-n 有効時はスコアリングに任せる。
    hal_max_retries = config.hallucination_max_retries
    tokens_list: list[list[int]] = []
    hal_attempt = 0

    t0 = time.perf_counter()
    for hal_attempt in range(hal_max_retries + 1):
        try:
            llm_texts = await _fetch_llm_candidates(
                llm_client=llm_client,
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=effective_max_tokens,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                n=best_of_n.n if best_of_n.enabled else 1,
            )
        except Exception as exc:
            logger.exception("LLM request failed")
            raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

        tokens_list = []
        for llm_text in llm_texts:
            try:
                tokens_list.append(parse_speech_tokens(llm_text))
            except ValueError as exc:
                logger.warning("Skipping candidate with invalid tokens: %s", exc)
        if not tokens_list:
            raise HTTPException(status_code=422, detail="No speech tokens found in LLM output.")

        # best-of-n 有効時はスコアリングに任せてリトライしない
        if best_of_n.enabled and best_of_n.n > 1:
            break

        candidate_tokens = tokens_list[0]
        if not is_token_hallucination(
            candidate_tokens, normalized, detected_language, hal_token_threshold,
        ):
            break  # 正常範囲

        limit = max_allowed_tokens(normalized, detected_language, hal_token_threshold)
        logger.warning(
            "Token hallucination detected (attempt %d/%d): "
            "speech_tokens=%d limit=%d (%.1fs vs %.1fs max) text='%s'",
            hal_attempt + 1,
            hal_max_retries + 1,
            len(candidate_tokens),
            limit,
            len(candidate_tokens) / 25.0,
            limit / 25.0,
            normalized[:60],
        )
        if hal_attempt >= hal_max_retries:
            logger.warning("Max token-hallucination retries reached, proceeding")

    t1 = time.perf_counter()

    logger.debug(
        "LLM candidates: count=%d token_lengths=%s attempts=%d",
        len(tokens_list),
        [len(t) for t in tokens_list],
        hal_attempt + 1,
    )
    t2 = time.perf_counter()

    reference_waveform = None
    global_embedding = None
    if request.reference is None:
        raise HTTPException(status_code=400, detail="reference is required")

    if request.reference.type == "base64":
        if reference_bytes is None:
            if not request.reference.data:
                raise HTTPException(status_code=400, detail="reference.data is required")
            try:
                payload = request.reference.data
                if "base64," in payload:
                    payload = payload.split("base64,", 1)[1]
                max_bytes = config.max_reference_mb * 1024 * 1024
                payload = _strip_base64_whitespace(payload)
                estimated_size = _estimate_base64_decoded_size(payload)
                if estimated_size > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"reference audio too large (max {config.max_reference_mb} MB)",
                    )
                reference_bytes = base64.b64decode(payload, validate=True)
                if len(reference_bytes) > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"reference audio too large (max {config.max_reference_mb} MB)",
                    )
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail="invalid base64 reference") from exc
        try:
            reference_waveform = load_reference_audio_bytes(reference_bytes, codec_service.sample_rate)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid reference audio") from exc
        reference_waveform = _trim_reference(
            reference_waveform, codec_service.sample_rate, config.max_reference_seconds
        )
    elif request.reference.type == "preset":
        preset_id = request.reference.preset_id
        if not preset_id:
            raise HTTPException(status_code=400, detail="reference.preset_id is required")
        try:
            global_embedding = codec_service.load_preset_embedding(preset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=400, detail="unsupported reference.type")

    best_of_n_sec = None
    asr_sec = None

    if best_of_n.enabled and best_of_n.n > 1 and len(tokens_list) > 1:
        if asr_service is None:
            raise HTTPException(status_code=400, detail="ASR is not available on the server")
        try:
            audio_batch, audio_lengths = codec_service.synthesize_batch(
                tokens_list,
                reference_waveform,
                global_embedding,
            )
        except Exception as exc:
            logger.exception("Codec synthesis failed")
            raise HTTPException(status_code=500, detail=f"Codec synthesis failed: {exc}") from exc

        t3 = time.perf_counter()

        lengths = (
            audio_lengths.tolist() if hasattr(audio_lengths, "tolist") else list(audio_lengths)
        )
        candidates: list[BestOfNCandidate] = []
        for idx, tokens in enumerate(tokens_list):
            audio_len = int(lengths[idx]) if lengths else audio_batch.shape[1]
            audio = audio_batch[idx, :audio_len]
            candidates.append(BestOfNCandidate(tokens=tokens, audio=audio))
        logger.debug("Decoded batch: audio_lengths=%s", lengths)

        rank_start = time.perf_counter()
        try:
            best_idx, asr_sec = await score_candidates(
                text=normalized,
                candidates=candidates,
                sample_rate=codec_service.sample_rate,
                language=best_of_n.language,
                asr_service=asr_service,
            )
        except Exception as exc:
            logger.exception("Best-of-n scoring failed")
            raise HTTPException(status_code=500, detail=f"Best-of-n scoring failed: {exc}") from exc
        rank_end = time.perf_counter()
        total_rank_sec = rank_end - rank_start
        best_of_n_sec = total_rank_sec
        if asr_sec is not None:
            best_of_n_sec = max(0.0, total_rank_sec - asr_sec)

        selected = candidates[best_idx]
        tokens = selected.tokens
        audio = selected.audio
        logger.debug("Best-of-n selected index=%d tokens=%d", best_idx, len(tokens))
    else:
        # ── デコード後の音声秒数ハルシネーション検出付きリトライ ──
        # codec デコード後の実際の音声長が期待値を大幅に超えた場合、
        # LLM→codec を再試行する
        hal_audio_threshold = config.hallucination_audio_threshold

        for audio_attempt in range(hal_max_retries + 1):
            tokens = tokens_list[0]
            try:
                audio = codec_service.synthesize(tokens, reference_waveform, global_embedding)
            except Exception as exc:
                logger.exception("Codec synthesis failed")
                raise HTTPException(status_code=500, detail=f"Codec synthesis failed: {exc}") from exc

            # デコード後の音声秒数チェック
            decoded_sec = float(audio.numel()) / float(codec_service.sample_rate)
            if not is_audio_hallucination(
                decoded_sec, normalized, detected_language, hal_audio_threshold,
            ):
                break  # 正常

            max_dur = max_expected_duration(
                normalized, detected_language, hal_audio_threshold,
            )
            logger.warning(
                "Audio hallucination detected (attempt %d/%d): "
                "audio=%.1fs limit=%.1fs tokens=%d text='%s'",
                audio_attempt + 1,
                hal_max_retries + 1,
                decoded_sec,
                max_dur,
                len(tokens),
                normalized[:60],
            )
            if audio_attempt >= hal_max_retries:
                logger.warning(
                    "Max audio-hallucination retries reached, proceeding"
                )
                break

            # LLM から再生成してリトライ
            try:
                llm_texts = await _fetch_llm_candidates(
                    llm_client=llm_client,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=effective_max_tokens,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    n=1,
                )
            except Exception as exc:
                logger.exception("LLM retry request failed")
                break
            tokens_list = []
            for llm_text in llm_texts:
                try:
                    tokens_list.append(parse_speech_tokens(llm_text))
                except ValueError:
                    pass
            if not tokens_list:
                logger.warning("No valid tokens on audio-hallucination retry")
                break

        t3 = time.perf_counter()

    codec_sample_rate = codec_service.sample_rate

    t4 = time.perf_counter()

    audio_sec = 0.0
    if codec_sample_rate > 0:
        audio_sec = float(audio.numel()) / float(codec_sample_rate)
    rtf = (t4 - t0) / audio_sec if audio_sec > 0 else 0.0

    timings = TTSTimings(
        llm_sec=round(t1 - t0, 4),
        parse_sec=round(t2 - t1, 4),
        codec_sec=round(t3 - t2, 4),
        total_sec=round(t4 - t0, 4),
        best_of_n_sec=round(best_of_n_sec, 4) if best_of_n_sec is not None else None,
        asr_sec=round(asr_sec, 4) if asr_sec is not None else None,
    )
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.info(
        "TTS timings: total=%.3fs llm=%.3fs parse=%.3fs codec=%.3fs best_of_n=%.3fs asr=%.3fs rtf=%.3f tokens=%d",
        timings.total_sec,
        timings.llm_sec,
        timings.parse_sec,
        timings.codec_sec,
        timings.best_of_n_sec or 0.0,
        timings.asr_sec or 0.0,
        rtf,
        len(tokens),
    )

    max_silence_sec = request.output.max_silence_sec if request.output else None
    if max_silence_sec is not None:
        before_samples = audio.numel()
        audio = compress_silence(audio, codec_sample_rate, max_silence_sec=max_silence_sec)
        logger.debug(
            "Silence compression: max_silence_sec=%.2f %d→%d samples (%.1f%%)",
            max_silence_sec, before_samples, audio.numel(),
            100.0 * (before_samples - audio.numel()) / max(before_samples, 1),
        )

    # 前後の無音をトリム
    before_trim = audio.numel()
    audio = trim_silence(audio, codec_sample_rate)
    if before_trim != audio.numel():
        logger.debug(
            "Trim silence: %d→%d samples",
            before_trim, audio.numel(),
        )

    # ポストプロセスで話速制御 (WSOLA)
    speed = (request.output.speed if request.output and request.output.speed else None) or 1.0
    if abs(speed - 1.0) > 1e-6:
        before_stretch = audio.numel()
        audio = time_stretch(audio, speed, codec_sample_rate)
        logger.debug(
            "Time stretch: speed=%.2f %d→%d samples",
            speed, before_stretch, audio.numel(),
        )

    wav_bytes = write_wav_bytes(audio, codec_sample_rate)
    if output_format == "wav":
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts.wav"},
        )

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    response = TTSResponse(
        audio=audio_b64,
        format="base64",
        sample_rate=codec_sample_rate,
        token_count=len(tokens),
        timings=timings,
        normalized_text=normalized,
    )
    return JSONResponse(content=response.model_dump())


@dataclass
class _ResolvedBestOfN:
    enabled: bool
    n: int
    language: str


def _resolve_best_of_n(request: TTSRequest, config) -> _ResolvedBestOfN:
    if not config.best_of_n_enabled:
        if request.best_of_n and request.best_of_n.enabled:
            raise HTTPException(status_code=400, detail="best_of_n is disabled on the server")
        return _ResolvedBestOfN(
            enabled=False,
            n=1,
            language=config.best_of_n_language,
        )

    req = request.best_of_n or BestOfNConfig()
    n = req.n if req.n is not None else config.best_of_n_default
    n = max(1, min(n, config.best_of_n_max))
    enabled = req.enabled if req.enabled is not None else (n > 1)
    language = (req.language or config.best_of_n_language).lower()
    if language not in {"ja", "en", "auto"}:
        language = "auto"
    if not enabled:
        n = 1
    return _ResolvedBestOfN(
        enabled=enabled,
        n=n,
        language=language,
    )


async def _fetch_llm_candidates(
    llm_client: LLMClient,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    n: int,
) -> list[str]:
    if n <= 1:
        text = await llm_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        return [text]

    tasks = [
        llm_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        for _ in range(n)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    texts: list[str] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("LLM candidate failed: %s", result)
            continue
        texts.append(result)
    if not texts:
        raise RuntimeError("All LLM candidate requests failed.")
    return texts


async def _read_reference_file(file: UploadFile) -> bytes:
    config = get_config()
    audio_config = get_audio_config()
    max_bytes = config.max_reference_mb * 1024 * 1024
    if file.filename:
        ext = ("." + file.filename.rsplit(".", 1)[-1]).lower() if "." in file.filename else ""
        if ext not in audio_config.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported audio extension: {ext}",
            )
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"reference audio too large (max {config.max_reference_mb} MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _strip_base64_whitespace(data: str) -> str:
    return "".join(data.split())


def _estimate_base64_decoded_size(data: str) -> int:
    if not data:
        return 0
    padding_len = len(data) - len(data.rstrip("="))
    return max(0, (len(data) * 3) // 4 - padding_len)


def _resolve_output_format(output: OutputConfig | None, default_format: str) -> str:
    if output and output.format:
        return output.format
    return default_format


def _trim_reference(waveform: torch.Tensor, sample_rate: int, max_seconds: float) -> torch.Tensor:
    if max_seconds <= 0:
        return waveform
    max_samples = int(sample_rate * max_seconds)
    if waveform.numel() > max_samples:
        original_sec = waveform.numel() / sample_rate
        logger.info(
            "Reference audio trimmed: %.2fs -> %.2fs (max %.2fs)",
            original_sec,
            max_seconds,
            max_seconds,
        )
        return waveform[:max_samples]
    return waveform


def _configure_torch(config):
    try:
        import torch
    except Exception:
        return
    uvicorn_logger = logging.getLogger("uvicorn.error")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    uvicorn_logger.info("Enabled TF32 matmul/cudnn")
