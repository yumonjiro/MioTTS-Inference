from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 0.5
RETRYABLE_STATUS_CODES = {502, 503, 504, 429}


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self._chat_url = _resolve_chat_url(base_url)
        self._models_url = _resolve_models_url(base_url)
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=timeout)
        self._model_lock = asyncio.Lock()
        self._default_model: str | None = None
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repeat_penalty": repetition_penalty,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        logger.debug("Sending chat request to %s", self._chat_url)
        return await self._post_with_retry(self._chat_url, payload)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> AsyncIterator[str]:
        """LLMにストリーミングリクエストを送り、テキストデルタを逐次yieldする。"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repeat_penalty": repetition_penalty,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": True,
        }
        logger.debug("Sending streaming chat request to %s", self._chat_url)
        async for delta in self._stream_with_retry(self._chat_url, payload):
            yield delta

    async def _post_with_retry(self, url: str, payload: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.post(url, json=payload, headers=self._headers)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    logger.warning(
                        "Retryable status %d on attempt %d/%d",
                        response.status_code,
                        attempt + 1,
                        self._max_retries,
                    )
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(self._retry_delay * (2**attempt))
                        continue
                response.raise_for_status()
                data = response.json()
                return _extract_content(data)
            except httpx.TimeoutException as exc:
                logger.warning("Timeout on attempt %d/%d: %s", attempt + 1, self._max_retries, exc)
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (2**attempt))
            except httpx.ConnectError as exc:
                logger.warning(
                    "Connection error on attempt %d/%d: %s", attempt + 1, self._max_retries, exc
                )
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (2**attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All retry attempts failed.")

    async def _stream_with_retry(
        self, url: str, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        """SSEストリーミングレスポンスをパースし、テキストデルタをyieldする。

        リトライはストリーム開始前（接続エラー・リトライ可能ステータス）のみ行う。
        ストリーム開始後のエラーはそのまま伝播させる。
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with self._client.stream(
                    "POST", url, json=payload, headers=self._headers
                ) as response:
                    if response.status_code in RETRYABLE_STATUS_CODES:
                        logger.warning(
                            "Retryable status %d on stream attempt %d/%d",
                            response.status_code,
                            attempt + 1,
                            self._max_retries,
                        )
                        # レスポンスボディを消費してからリトライ
                        await response.aread()
                        if attempt < self._max_retries - 1:
                            await asyncio.sleep(self._retry_delay * (2**attempt))
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    # ストリーム開始成功 — SSEイベントをパース
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in SSE data: %s", data_str[:100])
                            continue
                        delta = _extract_stream_delta(data)
                        if delta:
                            yield delta
                    return
            except httpx.TimeoutException as exc:
                logger.warning(
                    "Timeout on stream attempt %d/%d: %s", attempt + 1, self._max_retries, exc
                )
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (2**attempt))
            except httpx.ConnectError as exc:
                logger.warning(
                    "Connection error on stream attempt %d/%d: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (2**attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All stream retry attempts failed.")

    async def resolve_model(self, model: str | None) -> str:
        if model:
            return model
        if self._default_model:
            return self._default_model
        async with self._model_lock:
            if self._default_model:
                return self._default_model
            models = await self.list_models()
            if not models:
                raise RuntimeError("No models available from /v1/models.")
            self._default_model = models[0]
            logger.info("Resolved LLM model from /v1/models: %s", self._default_model)
            return self._default_model

    async def list_models(self) -> list[str]:
        response = await self._client.get(self._models_url, headers=self._headers)
        response.raise_for_status()
        payload = response.json()
        return _extract_model_ids(payload)


def _resolve_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _resolve_models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/models"):
        return base
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _extract_model_ids(payload: Any) -> list[str]:
    models: list[str] = []
    if not isinstance(payload, dict):
        return models
    for key in ("data", "models"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for field in ("id", "model", "name"):
                value = item.get(field)
                if value:
                    models.append(str(value))
                    break
    return models


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not choices:
        raise ValueError("LLM response missing 'choices'.")
    choice = choices[0]
    if "message" in choice:
        content = choice["message"].get("content", "")
    else:
        content = choice.get("text", "")
    return _flatten_content(content)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                parts.append(str(text))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _extract_stream_delta(payload: dict[str, Any]) -> str:
    """OpenAI互換ストリーミングレスポンスからデルタテキストを抽出する。"""
    choices = payload.get("choices")
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta", {})
    content = delta.get("content", "")
    return _flatten_content(content) if content else ""
