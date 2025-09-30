"""Optional ASR wrapper for turning audio inputs into text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class ASRClientProtocol:
    """Minimal protocol implemented by ASR client wrappers."""

    def transcribe(
        self,
        *,
        audio_bytes: Optional[bytes] = None,
        audio_url: Optional[str] = None,
        lang: str = "ko-KR",
    ) -> Dict[str, Any]:  # pragma: no cover - interface definition only
        raise NotImplementedError


@dataclass
class EchoASRClient(ASRClientProtocol):
    """Fallback client that simply returns an empty transcription.

    The production system can inject a real client (e.g. Whisper, Google
    Speech-to-Text).  During tests or notebooks the echo client keeps behaviour
    deterministic by acknowledging the request without attempting to decode the
    audio payload.
    """

    default_text: str = ""

    def transcribe(
        self,
        *,
        audio_bytes: Optional[bytes] = None,
        audio_url: Optional[str] = None,
        lang: str = "ko-KR",
    ) -> Dict[str, Any]:
        if audio_bytes is None and audio_url is None:
            raise ValueError("audio_bytes 또는 audio_url 중 하나는 필요합니다.")

        return {
            "text": self.default_text,
            "confidence": 0.0,
            "lang": lang,
            "trace_id": "echo-asr",
        }


def transcribe_audio(
    *,
    audio_bytes: Optional[bytes] = None,
    audio_url: Optional[str] = None,
    lang: str = "ko-KR",
    client: Optional[ASRClientProtocol] = None,
) -> Dict[str, Any]:
    """Run the configured ASR client and normalise the response."""

    client = client or EchoASRClient()
    return client.transcribe(audio_bytes=audio_bytes, audio_url=audio_url, lang=lang)


__all__ = ["ASRClientProtocol", "EchoASRClient", "transcribe_audio"]