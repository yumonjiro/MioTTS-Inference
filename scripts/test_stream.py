#!/usr/bin/env python3
"""MioTTS ストリーミングTTSのテスト用CLIスクリプト。

Usage:
    # リアルタイム再生 (sounddeviceが必要)
    python scripts/test_stream.py "こんにちは、今日はいい天気ですね。"

    # ファイルに保存
    python scripts/test_stream.py "Hello world." --output out.wav

    # プリセット指定
    python scripts/test_stream.py "テスト" --preset en_female

    # サーバURL指定
    python scripts/test_stream.py "テスト" --api-base http://localhost:8001
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import sys
import time

import httpx
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MioTTS streaming endpoint")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--api-base", default="http://localhost:8001", help="MioTTS API base URL")
    parser.add_argument("--preset", default="jp_female", help="Preset ID (default: jp_female)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0)")
    parser.add_argument("--max-silence", type=float, default=0.3, help="Max silence sec (0=off)")
    parser.add_argument("--output", "-o", help="Save to WAV file instead of playing")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio, just show events")
    parser.add_argument("--chunk-tokens", type=int, default=50, help="(info only) Expected chunk size")
    return parser.parse_args()


def build_request(args: argparse.Namespace) -> dict:
    payload: dict = {
        "text": args.text,
        "reference": {"type": "preset", "preset_id": args.preset},
        "output": {},
    }
    if args.speed != 1.0:
        payload["output"]["speed"] = args.speed
    if args.max_silence > 0:
        payload["output"]["max_silence_sec"] = args.max_silence
    return payload


def main() -> None:
    args = parse_args()
    api_base = args.api_base.rstrip("/")
    url = f"{api_base}/v1/tts/stream"
    payload = build_request(args)

    print(f"=== MioTTS Streaming Test ===")
    print(f"URL: {url}")
    print(f"Text: {args.text}")
    print(f"Preset: {args.preset}")
    print(f"Speed: {args.speed}")
    print()

    # sounddevice はオプション (--no-play / --output 時は不要)
    sd = None
    if not args.no_play and not args.output:
        try:
            import sounddevice as sd_module
            sd = sd_module
        except ImportError:
            print("Warning: sounddevice not installed. Install with: pip install sounddevice")
            print("         Falling back to --no-play mode.")
            args.no_play = True

    all_audio: list[np.ndarray] = []
    sample_rate = 44100  # doneイベントで更新される
    stream = None
    total_samples = 0
    t_start = time.perf_counter()
    first_chunk_time: float | None = None

    try:
        with httpx.stream(
            "POST", url, json=payload, timeout=httpx.Timeout(120.0, connect=10.0)
        ) as response:
            if response.status_code != 200:
                response.read()
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
                sys.exit(1)

            for line in response.iter_lines():
                if not line:
                    continue

                # SSEパース: "event: xxx" と "data: {...}" の組
                if line.startswith("event: "):
                    continue  # event行はスキップ、dataで処理
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    print(f"  [warn] Invalid JSON: {data_str[:80]}")
                    continue

                event_type = data.get("event")

                if event_type == "chunk":
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        latency = first_chunk_time - t_start
                        print(f"  First chunk latency: {latency:.3f}s")

                    chunk_index = data["chunk_index"]
                    token_count = data["token_count"]
                    sample_rate = data["sample_rate"]
                    audio_b64 = data["audio"]

                    # base64 → float32 PCM
                    pcm_bytes = base64.b64decode(audio_b64)
                    audio_np = np.frombuffer(pcm_bytes, dtype=np.float32)
                    chunk_sec = len(audio_np) / sample_rate
                    total_samples += len(audio_np)

                    print(
                        f"  chunk[{chunk_index}]: "
                        f"{token_count} tokens, "
                        f"{len(audio_np)} samples, "
                        f"{chunk_sec:.2f}s audio"
                    )

                    all_audio.append(audio_np)

                    # リアルタイム再生
                    if sd is not None and not args.no_play and not args.output:
                        if stream is None:
                            stream = sd.OutputStream(
                                samplerate=sample_rate,
                                channels=1,
                                dtype="float32",
                            )
                            stream.start()
                        stream.write(audio_np.reshape(-1, 1))

                elif event_type == "done":
                    t_end = time.perf_counter()
                    timings = data.get("timings", {})
                    total_tokens = data.get("total_token_count", 0)
                    total_chunks = data.get("total_chunks", 0)
                    sample_rate = data.get("sample_rate", sample_rate)
                    normalized = data.get("normalized_text", "")

                    total_audio_sec = total_samples / sample_rate if sample_rate else 0
                    wall_time = t_end - t_start
                    rtf = wall_time / total_audio_sec if total_audio_sec > 0 else 0

                    print()
                    print(f"=== Done ===")
                    print(f"  Normalized: {normalized}")
                    print(f"  Chunks: {total_chunks}")
                    print(f"  Tokens: {total_tokens}")
                    print(f"  Audio: {total_audio_sec:.2f}s @ {sample_rate}Hz")
                    print(f"  Wall time: {wall_time:.3f}s")
                    print(f"  RTF: {rtf:.3f}")
                    if first_chunk_time:
                        print(f"  First chunk latency: {first_chunk_time - t_start:.3f}s")
                    print(f"  Timings:")
                    for k, v in timings.items():
                        if v is not None:
                            print(f"    {k}: {v}s")

                elif event_type == "error":
                    detail = data.get("detail", "unknown error")
                    print(f"\n  [ERROR] {detail}")
                    sys.exit(1)

    except httpx.ConnectError:
        print(f"Error: Cannot connect to {api_base}")
        print("  Is the MioTTS server running?")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    # 保存
    if args.output and all_audio:
        import soundfile as sf

        combined = np.concatenate(all_audio)
        sf.write(args.output, combined, sample_rate)
        print(f"\n  Saved to: {args.output}")

    # --no-play でも --output なしの場合、サマリーだけ表示
    if not all_audio:
        print("\n  No audio chunks received.")


if __name__ == "__main__":
    main()
