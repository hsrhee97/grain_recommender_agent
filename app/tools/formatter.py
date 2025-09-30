"""Final message formatter for recommendations."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _format_mix(mix: Dict[str, int]) -> str:
    ordered = sorted(mix.items(), key=lambda x: (-x[1], x[0]))
    lines = [f"• {grain} {percent}%" for grain, percent in ordered]
    return "\n".join(lines)


def format_recommendation(primary: Dict[str, Any],
                           candidates: Iterable[Dict[str, Any]] | None = None,
                           explanation: Optional[str] = None) -> Dict[str, Any]:
    """Create the user-facing message and payload."""
    candidates = list(candidates or [])
    message_lines: List[str] = ["✅ 추천 배합 (R0)"]
    message_lines.append(_format_mix(primary.get("mix", {})))

    if explanation:
        message_lines.append("")
        message_lines.append(f"📎 이유: {explanation}")

    if candidates:
        message_lines.append("")
        message_lines.append("대안 제안")
        for cand in candidates:
            message_lines.append(f"- {cand.get('id', 'R?')} : {', '.join(f'{k} {v}%' for k, v in cand.get('mix', {}).items())}")

    payload = {
        "primary": primary,
        "candidates": candidates,
    }

    return {
        "message": "\n".join(message_lines).strip(),
        "payload": payload,
    }


__all__ = ["format_recommendation"]
