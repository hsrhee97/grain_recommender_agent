"""Feedback collection utilities."""

from __future__ import annotations

from typing import Any, Dict

from app.services.storage import MEMORY_STORAGE
from app.tools.memory import update_memory

VALID_SATISFACTION = {"like", "neutral", "dislike"}


class FeedbackValidationError(ValueError):
    """Raised when the feedback payload does not satisfy the schema."""


def validate_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the provided feedback dictionary matches expectations."""

    if not isinstance(feedback, dict):  # pragma: no cover - defensive
        raise FeedbackValidationError("feedback must be a dictionary")

    satisfaction = str(feedback.get("satisfaction", "")).strip().lower()
    if satisfaction not in VALID_SATISFACTION:
        raise FeedbackValidationError("satisfaction must be like, neutral, or dislike")

    reason = feedback.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise FeedbackValidationError("reason must be a string if provided")

    mix = feedback.get("mix")
    if mix is not None and not isinstance(mix, dict):
        raise FeedbackValidationError("mix must be a dict of grain ratios when provided")

    payload = {
        "satisfaction": satisfaction,
        "reason": reason.strip() if isinstance(reason, str) else None,
        "mix": mix,
    }
    return payload


def record_feedback(user_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Persist feedback information and update memory state."""

    payload = validate_feedback(feedback)

    entry = MEMORY_STORAGE.append_feedback(
        user_id,
        payload["satisfaction"],
        payload.get("reason"),
        payload.get("mix"),
    )

    baseline = update_memory(user_id, {"last_feedback": entry})

    return {
        "ok": True,
        "feedback": entry,
        "baseline": baseline,
    }


__all__ = ["record_feedback", "validate_feedback", "FeedbackValidationError", "VALID_SATISFACTION"]