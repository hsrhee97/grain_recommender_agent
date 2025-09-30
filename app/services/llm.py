"""Thin wrapper around the OpenAI Responses API used by the extractor."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency at runtime
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled lazily for environments without openai
    OpenAI = None  # type: ignore[assignment]


class LLMNotConfiguredError(RuntimeError):
    """Raised when the LLM client cannot be instantiated due to missing setup."""


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatLLMClient:
    """Simple OpenAI client wrapper for JSON-style responses."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover - executed only when dependency missing
            raise LLMNotConfiguredError(
                "openai 패키지를 찾을 수 없습니다. `pip install openai` 후 다시 시도하세요."
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMNotConfiguredError(
                "OPENAI_API_KEY 환경 변수를 설정해야 LLM 모드를 사용할 수 있습니다."
            )

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float = 0.2,
        response_format: Optional[dict] = None,
    ) -> str:
        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

        response = self._client.responses.create(
            model=self.model,
            input=payload,
            temperature=temperature,
            response_format=response_format or {"type": "text"},
        )

        text = getattr(response, "output_text", None)
        if text:
            return text

        for block in getattr(response, "output", []) or []:
            for content in getattr(block, "content", []) or []:
                if getattr(content, "type", None) == "text":
                    return getattr(content, "text", "")

        raise RuntimeError("LLM 응답에서 텍스트를 찾을 수 없습니다.")


__all__ = ["ChatLLMClient", "ChatMessage", "LLMNotConfiguredError"]
