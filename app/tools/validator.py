"""Validator tool for checking survey completeness and conflicts."""
from __future__ import annotations

from typing import Any, Dict, List

from .reask import build_reask_message


def _flatten_survey(survey: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, entry in (survey or {}).items():
        if isinstance(entry, dict):
            flat[key] = entry.get("value")
        else:
            flat[key] = entry
    return flat


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return True


def _detect_conflicts(values: Dict[str, Any]) -> List[str]:
    conflicts: List[str] = []
    if values.get("health_issue") == "diabetes" and values.get("purpose") == "맛중심":
        conflicts.append("diabetes vs 맛중심")
    return conflicts


def validate_survey(survey: Dict[str, Any], schema: Dict[str, Any],
                    rules: Dict[str, Any] | None = None,
                    *, extraction: Dict[str, Any] | None = None,
                    locale: str = "ko-KR") -> Dict[str, Any]:
    """Validate survey answers against schema requirements."""
    flat = _flatten_survey(survey)
    required = schema.get("required", [])
    high_weight = schema.get("high_weight", [])

    missing_required = [field for field in required if not _has_value(flat.get(field))]
    high_weight_missing = [field for field in high_weight if not _has_value(flat.get(field))]

    conflicts = _detect_conflicts(flat)

    ok = not missing_required and not conflicts
    reask_message = None
    assistant_summary = None
    if extraction:
        assistant_summary = extraction.get("assistant_utterance")

    if not ok:
        reask_message = build_reask_message(
            missing_required,
            conflicts,
            survey=survey,
            summary=assistant_summary,
            locale=locale,
        )

    return {
        "ok": ok,
        "missing_required": missing_required,
        "conflicts": conflicts,
        "high_weight_missing": high_weight_missing,
        "reask_message": reask_message,
        "assistant_summary": assistant_summary,
    }


__all__ = ["validate_survey"]
