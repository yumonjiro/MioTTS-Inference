from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

DEFAULT_API_BASE = os.getenv("MIOTTS_API_BASE", "http://localhost:8001")


def _fetch_presets(api_base: str) -> list[str]:
    try:
        res = httpx.get(f"{api_base}/v1/presets", timeout=5.0)
        res.raise_for_status()
        data = res.json()
        presets = data.get("presets", [])
        if isinstance(presets, list):
            return presets
    except Exception:
        pass
    return []


def _refresh_presets(api_base: str) -> gr.Dropdown:
    presets = _fetch_presets(api_base)
    value = presets[0] if presets else None
    return gr.update(choices=presets, value=value)


def _decode_wav_bytes(data: bytes) -> tuple[int, np.ndarray]:
    with io.BytesIO(data) as buff:
        audio, sr = sf.read(buff, dtype="float32")
    return sr, audio


def _call_tts(
    api_base: str,
    text: str,
    reference_mode: str,
    reference_audio: tuple[int, np.ndarray] | None,
    preset_id: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    speed: float,
    max_silence_sec: float,
    best_of_n_enabled: bool,
    best_of_n_n: int,
    best_of_n_language: str,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    if not text:
        return None, ""
    api_base = api_base.rstrip("/")

    payload: dict[str, Any] = {
        "text": text,
        "llm": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 700,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        },
        "output": {
            "speed": speed if speed != 1.0 else None,
            "max_silence_sec": max_silence_sec if max_silence_sec > 0 else None,
        },
    }
    if reference_mode == "upload" and reference_audio is not None:
        sr, audio = reference_audio
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload["reference"] = {"type": "base64", "data": audio_b64}
    elif preset_id:
        payload["reference"] = {"type": "preset", "preset_id": preset_id}
    if best_of_n_enabled:
        payload["best_of_n"] = {
            "enabled": True,
            "n": best_of_n_n,
            "language": best_of_n_language,
        }
    response = httpx.post(f"{api_base}/v1/tts", json=payload, timeout=120.0)

    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("audio/"):
        return _decode_wav_bytes(response.content), ""
    data = response.json()
    audio_b64 = data.get("audio")
    if not audio_b64:
        return None, "No audio in response."
    audio_bytes = base64.b64decode(audio_b64)
    sr, audio = _decode_wav_bytes(audio_bytes)
    audio_samples = audio.shape[0] if hasattr(audio, "shape") else len(audio)
    audio_sec = float(audio_samples) / float(sr) if sr else 0.0
    timings = data.get("timings") or {}
    total_sec = timings.get("total_sec") or 0.0
    rtf = (float(total_sec) / audio_sec) if audio_sec > 0 else 0.0

    def _fmt(label: str, value: Any) -> str:
        if value is None:
            return f"- {label}: n/a"
        try:
            return f"- {label}: {float(value):.3f}s"
        except Exception:
            return f"- {label}: {value}"

    total_sec = timings.get("total_sec")
    llm_sec = timings.get("llm_sec")
    parse_sec = timings.get("parse_sec")
    codec_sec = timings.get("codec_sec")
    best_of_n_sec = timings.get("best_of_n_sec")
    asr_sec = timings.get("asr_sec")
    rtf_line = f"- RTF: {rtf:.3f}" if rtf else "- RTF: n/a"

    info_text = "\n".join(
        [
            "Timings",
            _fmt("Total", total_sec),
            _fmt("LLM", llm_sec),
            _fmt("Parse", parse_sec),
            _fmt("Codec", codec_sec),
            _fmt("Best-of-N", best_of_n_sec),
            _fmt("ASR", asr_sec),
            rtf_line,
        ]
    )
    return (sr, audio), info_text


def _call_tts_stream(
    api_base: str,
    text: str,
    reference_mode: str,
    reference_audio: tuple[int, np.ndarray] | None,
    preset_id: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    speed: float,
    max_silence_sec: float,
):
    """ストリーミングTTSのジェネレータ。チャンク受信ごとに累積音声とログをyieldする。"""
    if not text:
        yield None, "No text provided."
        return
    api_base = api_base.rstrip("/")

    payload: dict[str, Any] = {
        "text": text,
        "llm": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 700,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        },
        "output": {
            "speed": speed if speed != 1.0 else None,
            "max_silence_sec": max_silence_sec if max_silence_sec > 0 else None,
        },
    }
    if reference_mode == "upload" and reference_audio is not None:
        sr, audio = reference_audio
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload["reference"] = {"type": "base64", "data": audio_b64}
    elif preset_id:
        payload["reference"] = {"type": "preset", "preset_id": preset_id}

    all_chunks: list[np.ndarray] = []
    sample_rate = 44100
    log_lines: list[str] = ["Streaming..."]
    import time
    t_start = time.perf_counter()

    try:
        with httpx.stream(
            "POST", f"{api_base}/v1/tts/stream", json=payload,
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) as response:
            if response.status_code != 200:
                response.read()
                yield None, f"Error: HTTP {response.status_code}\n{response.text}"
                return

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("event")

                if event_type == "chunk":
                    sample_rate = data["sample_rate"]
                    pcm_bytes = base64.b64decode(data["audio"])
                    audio_np = np.frombuffer(pcm_bytes, dtype=np.float32)
                    all_chunks.append(audio_np)
                    chunk_sec = len(audio_np) / sample_rate
                    elapsed = time.perf_counter() - t_start
                    log_lines.append(
                        f"chunk[{data['chunk_index']}]: "
                        f"{data['token_count']} tokens, {chunk_sec:.2f}s audio "
                        f"(elapsed: {elapsed:.2f}s)"
                    )
                    # チャンク受信ごとに累積音声をyield
                    combined = np.concatenate(all_chunks)
                    yield (sample_rate, combined), "\n".join(log_lines)

                elif event_type == "done":
                    t_end = time.perf_counter()
                    timings = data.get("timings", {})
                    total_tokens = data.get("total_token_count", 0)
                    total_chunks = data.get("total_chunks", 0)
                    sample_rate = data.get("sample_rate", sample_rate)
                    combined = np.concatenate(all_chunks) if all_chunks else np.array([], dtype=np.float32)
                    total_sec = len(combined) / sample_rate if sample_rate else 0
                    wall = t_end - t_start
                    rtf = wall / total_sec if total_sec > 0 else 0
                    log_lines.append("")
                    log_lines.append(
                        f"Done: {total_chunks} chunks, {total_tokens} tokens, "
                        f"{total_sec:.2f}s audio, {wall:.3f}s wall, RTF={rtf:.3f}"
                    )
                    for k, v in timings.items():
                        if v is not None:
                            log_lines.append(f"  {k}: {v}s")
                    yield (sample_rate, combined), "\n".join(log_lines)

                elif event_type == "error":
                    log_lines.append(f"ERROR: {data.get('detail', 'unknown')}")
                    combined = np.concatenate(all_chunks) if all_chunks else None
                    result = (sample_rate, combined) if combined is not None else None
                    yield result, "\n".join(log_lines)
                    return

    except httpx.ConnectError:
        yield None, f"Cannot connect to {api_base}. Is the server running?"
    except Exception as exc:
        yield None, f"Error: {exc}"


def build_app() -> gr.Blocks:
    presets = _fetch_presets(DEFAULT_API_BASE)

    with gr.Blocks(title="MioTTS Demo") as demo:
        gr.Markdown("# MioTTS Demo")

        with gr.Accordion("Advanced Settings", open=False):
            api_base = gr.Textbox(
                label="API Base URL",
                value=DEFAULT_API_BASE,
                placeholder="http://localhost:8001",
            )

        text = gr.Textbox(label="Text", lines=6, placeholder="Type text to synthesize...")

        with gr.Row():
            reference_mode = gr.Radio(
                choices=["preset", "upload"],
                value="preset",
                label="Reference Mode",
            )
            preset_id = gr.Dropdown(
                choices=presets,
                value=presets[0] if presets else None,
                label="Preset ID",
                allow_custom_value=True,
                visible=True,
            )
            with gr.Column(scale=0, min_width=72):
                refresh_presets = gr.Button("↻", size="md")

        reference_audio = gr.Audio(
            label="Reference Audio",
            sources=["upload"],
            type="numpy",
            visible=False,
        )

        def _update_reference_visibility(mode):
            if mode == "preset":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        reference_mode.change(
            fn=_update_reference_visibility,
            inputs=[reference_mode],
            outputs=[preset_id, reference_audio],
        )

        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p")
            repetition_penalty = gr.Slider(
                1.0, 1.5, value=1.0, step=0.05, label="Repetition Penalty"
            )
            presence_penalty = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Presence Penalty")
            frequency_penalty = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Frequency Penalty")

        with gr.Row():
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
            max_silence_sec = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Max Silence (sec, 0=off)")

        with gr.Row():
            best_of_n_enabled = gr.Checkbox(value=False, label="Best-of-N")
            best_of_n_n = gr.Slider(1, 8, value=2, step=1, label="N")
            best_of_n_language = gr.Dropdown(
                choices=["auto", "ja", "en"],
                value="auto",
                label="Language",
            )

        synth_btn = gr.Button("Synthesize")
        stream_btn = gr.Button("Synthesize (Stream)", variant="secondary")
        output_audio = gr.Audio(label="Output", type="numpy")
        output_info = gr.Markdown(label="Timings")

        refresh_presets.click(
            _refresh_presets,
            inputs=api_base,
            outputs=preset_id,
        )

        synth_btn.click(
            _call_tts,
            inputs=[
                api_base,
                text,
                reference_mode,
                reference_audio,
                preset_id,
                temperature,
                top_p,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                speed,
                max_silence_sec,
                best_of_n_enabled,
                best_of_n_n,
                best_of_n_language,
            ],
            outputs=[output_audio, output_info],
        )

        stream_btn.click(
            _call_tts_stream,
            inputs=[
                api_base,
                text,
                reference_mode,
                reference_audio,
                preset_id,
                temperature,
                top_p,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                speed,
                max_silence_sec,
            ],
            outputs=[output_audio, output_info],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
