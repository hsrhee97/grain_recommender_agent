"""Utilities for generating follow-up (reask) messages."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

_FIELD_LABELS = {
    "purpose": "취식 목적",
    "frequency": "섭취 빈도",
    "texture_pref": "선호 식감",
    "disliked_grains": "기피 곡물",
    "avoid_gluten": "글루텐 회피 여부",
    "health_issue": "건강 이슈",
}

_FIELD_HINTS = {
    "purpose": "혈당관리, 체중관리, 근력, 맛중심",
    "frequency": "주 1-2회 / 주 3-4회 / 주 5-7회",
    "texture_pref": "고슬밥 / 찰진밥 / 무관",
}


def build_reask_message(
    missing_fields: Sequence[str],
    conflicts: Sequence[str] | None = None,
    *,
    survey: Dict[str, Dict[str, Any]] | None = None,
    summary: str | None = None,
    locale: str = "ko-KR",
) -> str:
    """Create a conversational follow-up prompt for missing or conflicting answers."""

    conflicts = list(conflicts or [])
    parts: List[str] = []

    if summary:
        parts.append(summary)

    if conflicts:
        parts.append("입력에 모순이 있어 다시 확인하고 싶어요: " + ", ".join(conflicts))

    if missing_fields:
        primary = missing_fields[0]
        label = _FIELD_LABELS.get(primary, primary)
        hint = _FIELD_HINTS.get(primary)
        entry = (survey or {}).get(primary, {}) if isinstance(survey, dict) else {}
        alternatives = []
        if isinstance(entry, dict):
            for alt in entry.get("alternatives", []) or []:
                if not isinstance(alt, dict):
                    continue
                value = alt.get("value")
                conf = alt.get("confidence")
                if value is None:
                    continue
                if isinstance(conf, (int, float)):
                    pct = int(round(conf * 100))
                    alternatives.append(f"{value} ({pct}%)")
                else:
                    alternatives.append(str(value))
        other_text = entry.get("other_text") if isinstance(entry, dict) else None

        if alternatives:
            question = f"{label}는 {' / '.join(alternatives[:2])} 중 어느 쪽이 더 가까울까요?"
        elif other_text:
            question = f"{label}에 대해 '{other_text}'라고 이해했어요. 가까운 선택지를 알려주실 수 있을까요?"
        else:
            question = f"{label}를 알려주실 수 있을까요?"
            if hint:
                question += f" (예: {hint})"
        parts.append(question)

    if not parts:
        return "추가 확인을 위해 몇 가지 정보를 더 알려주세요."

    if locale.startswith("ko"):
        prefix = "확인을 위해"
    else:
        prefix = "To proceed"

    return prefix + " " + "\n".join(parts)


__all__ = ["build_reask_message"]
