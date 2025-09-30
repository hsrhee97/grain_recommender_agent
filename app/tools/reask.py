"""Utilities for generating follow-up (reask) messages."""
from __future__ import annotations

from typing import List, Sequence

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


def build_reask_message(missing_fields: Sequence[str], conflicts: Sequence[str] | None = None,
                         locale: str = "ko-KR") -> str:
    """Create a concise follow-up prompt for missing or conflicting answers."""
    conflicts = list(conflicts or [])
    parts: List[str] = []

    if conflicts:
        parts.append("입력에 모순이 있어 다시 확인하고 싶어요: " + ", ".join(conflicts))

    if missing_fields:
        primary = missing_fields[0]
        label = _FIELD_LABELS.get(primary, primary)
        hint = _FIELD_HINTS.get(primary)
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

    return prefix + " " + " ".join(parts)


__all__ = ["build_reask_message"]
