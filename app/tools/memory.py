"""Helper functions for reading and updating user memory."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from app.services.storage import MEMORY_STORAGE


def get_user_memory(user_id: str) -> Dict[str, Any]:
    """Return the current memory snapshot for the given user."""

    return MEMORY_STORAGE.get_user_memory(user_id)


def _ensure_list(values: Optional[Iterable[str]]) -> list[str]:
    return sorted({str(item) for item in (values or []) if item is not None})


def derive_update_from_survey(survey: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a structured survey into memory updates.

    - disliked grains are persisted so future runs can bias against them
    - preferred texture/purpose are stored under the `preferences` section
    """

    update: Dict[str, Any] = {}
    disliked = None
    texture = None
    purpose = None

    for key, value in (survey or {}).items():
        normalized = value.get("value") if isinstance(value, dict) else value
        if key == "disliked_grains" and normalized:
            disliked = _ensure_list(normalized if isinstance(normalized, list) else [normalized])
        elif key == "texture_pref" and normalized:
            texture = str(normalized)
        elif key == "purpose" and normalized:
            purpose = str(normalized)

    if disliked:
        update["disliked"] = disliked

    preferences: Dict[str, Any] = {}
    if texture:
        preferences["texture_pref"] = texture
    if purpose:
        preferences["purpose"] = purpose

    if preferences:
        update["preferences"] = preferences

    return update


def update_memory_from_survey(user_id: str, survey: Dict[str, Any]) -> Dict[str, Any]:
    """Persist survey-derived preferences and return the updated snapshot."""

    update = derive_update_from_survey(survey)
    if not update:
        return get_user_memory(user_id)
    return MEMORY_STORAGE.update_user_memory(user_id, update)


def update_memory(user_id: str, update: Dict[str, Any]) -> Dict[str, Any]:
    """General-purpose updater used by other tools (e.g. feedback)."""

    return MEMORY_STORAGE.update_user_memory(user_id, update)


__all__ = [
    "derive_update_from_survey",
    "get_user_memory",
    "update_memory_from_survey",
    "update_memory",
]