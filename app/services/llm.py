"""LLM client abstraction with OpenAI and Gemini backends."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency at runtime
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled lazily for environments without openai
    OpenAI = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency at runtime
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled lazily for environments without gemini
    genai = None  # type: ignore[assignment]


class LLMNotConfiguredError(RuntimeError):
    """Raised when the LLM client cannot be instantiated due to missing setup."""


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatLLMClient:
    """Simple LLM client wrapper supporting multiple providers."""

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()

        if self.provider in {"openai", "openai-compatible"}:
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
            self._backend = "openai"

        elif self.provider in {"gemini", "google", "googleai"}:
            if genai is None:  # pragma: no cover - executed only when dependency missing
                raise LLMNotConfiguredError(
                    "google-generativeai 패키지를 찾을 수 없습니다. `pip install google-generativeai` 후 다시 시도하세요."
                )

            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise LLMNotConfiguredError(
                    "GEMINI_API_KEY 환경 변수를 설정해야 Gemini 기반 LLM 모드를 사용할 수 있습니다."
                )

            genai.configure(api_key=self.api_key)
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            self._client = genai.GenerativeModel(self.model)
            self._backend = "gemini"
        else:
            raise LLMNotConfiguredError(
                "지원하지 않는 LLM_PROVIDER 값입니다. openai 또는 gemini 중 하나를 사용하세요."
            )

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float = 0.2,
        response_format: Optional[dict] = None,
    ) -> str:
        if self._backend == "openai":
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

        # Gemini backend
        assert self._backend == "gemini"

        system_prompts = [msg.content for msg in messages if msg.role == "system"]
        other_messages = [msg for msg in messages if msg.role != "system"]

        if system_prompts:
            combined_system = "\n\n".join(part.strip() for part in system_prompts if part.strip())
            if other_messages and other_messages[0].role == "user":
                first = other_messages[0]
                other_messages[0] = ChatMessage(
                    role=first.role,
                    content=f"{combined_system}\n\n{first.content}",
                )
            else:
                other_messages.insert(0, ChatMessage(role="user", content=combined_system))

        contents = []
        for message in other_messages:
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [message.content]})

        generation_config = {
            "temperature": temperature,
        }

        if response_format and response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"

        response = self._client.generate_content(
            contents,
            generation_config=generation_config,
        )

        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            for content in getattr(candidate, "content", []) or []:
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    if isinstance(part, str) and part.strip():
                        return part
                    text_part = getattr(part, "text", None)
                    if text_part:
                        return text_part

        raise RuntimeError("LLM 응답에서 텍스트를 찾을 수 없습니다.")


__all__ = ["ChatLLMClient", "ChatMessage", "LLMNotConfiguredError"]
