"""Rule-based survey extractor for Korean free-form inputs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEMA_PATH = Path("app/schemas/survey_schema.json")

with SCHEMA_PATH.open("r", encoding="utf-8") as fp:
    SURVEY_SCHEMA: Dict[str, Any] = json.load(fp)

# 기본 곡물 키워드
GRAIN_KEYWORDS = ["현미", "백미", "귀리", "보리", "퀴노아", "수수", "조", "흑미", "찹쌀"]

_PURPOSE_MAP: List[Tuple[str, str]] = [
    (r"혈당|당뇨", "혈당관리"),
    (r"체중|다이어트", "체중관리"),
    (r"근육|근력|헬스", "근력"),
    (r"맛있|취향|맛", "맛중심"),
]

_FREQUENCY_PATTERNS: List[Tuple[str, str]] = [
    (r"주\s*[56-9]|매일|하루에", "주 5-7회"),
    (r"주\s*[34]|이틀", "주 3-4회"),
    (r"주\s*[12]|한?번", "주 1-2회"),
]

_TEXTURE_PATTERNS: List[Tuple[str, str]] = [
    (r"찰|찹쌀|쫀득", "찰진밥"),
    (r"고슬|퍽퍽", "고슬밥"),
]

_HEALTH_PATTERNS: List[Tuple[str, str]] = [
    (r"당뇨|혈당", "diabetes"),
]

_GLUTEN_PATTERN = re.compile(r"글루텐|celiac|씨리악", re.IGNORECASE)
_NEGATION_PATTERN = re.compile(r"싫|별로|피하|안 먹|못 먹|빼", re.IGNORECASE)


def _extract_with_patterns(text: str, patterns: List[Tuple[str, str]]) -> Tuple[str | None, float]:
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return label, 0.9
    return None, 0.0


def _detect_frequency(text: str) -> Tuple[str | None, float]:
    for pattern, label in _FREQUENCY_PATTERNS:
        if re.search(pattern, text):
            return label, 0.85
    return None, 0.0


def _detect_texture(text: str) -> Tuple[str | None, float]:
    for pattern, label in _TEXTURE_PATTERNS:
        if re.search(pattern, text):
            return label, 0.8
    return None, 0.0


def _detect_disliked(text: str) -> Tuple[List[str] | None, float]:
    hits: List[str] = []
    for grain in GRAIN_KEYWORDS:
        if grain in text and re.search(_NEGATION_PATTERN, text):
            hits.append(grain)
        else:
            # allow expressions like "보리는 빼고"
            if re.search(fr"{grain}[^가-힣]*(빼|제외)", text):
                hits.append(grain)
    if hits:
        return sorted(set(hits)), 0.95
    return None, 0.0


def _detect_gluten(text: str) -> Tuple[bool | None, float]:
    if re.search(_GLUTEN_PATTERN, text):
        if re.search(r"피하|없|알레르기|못 먹", text):
            return True, 0.85
        return True, 0.7
    return None, 0.0


def _detect_health_issue(text: str) -> Tuple[str | None, float]:
    value, confidence = _extract_with_patterns(text, _HEALTH_PATTERNS)
    if value:
        return value, max(confidence, 0.7)
    if "건강" in text:
        return "none", 0.4
    return None, 0.0


def _ensure_schema_value(field: str, value: Any) -> Any:
    options = (SURVEY_SCHEMA.get("fields", {}).get(field, {}) or {}).get("options")
    if options and value not in options:
        return None
    return value


def extract_survey(user_text: str, schema: Dict[str, Any] | None = None,
                   hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Convert user free-form text into survey-style JSON structure."""
    schema = schema or SURVEY_SCHEMA
    normalized = user_text.strip()

    purpose, purpose_conf = _extract_with_patterns(normalized, _PURPOSE_MAP)
    freq, freq_conf = _detect_frequency(normalized)
    texture, texture_conf = _detect_texture(normalized)
    disliked, disliked_conf = _detect_disliked(normalized)
    gluten, gluten_conf = _detect_gluten(normalized)
    health, health_conf = _detect_health_issue(normalized)

    survey: Dict[str, Dict[str, Any]] = {}

    def _pack(field: str, value: Any, confidence: float | None, source: str = "rule") -> None:
        survey[field] = {
            "value": _ensure_schema_value(field, value),
            "confidence": round(confidence or 0.0, 2),
            "source": source,
        }
        if survey[field]["value"] is None:
            survey[field]["reason"] = "not_confident"

    _pack("purpose", purpose, purpose_conf)
    _pack("frequency", freq, freq_conf)
    _pack("texture_pref", texture, texture_conf)
    if disliked is not None:
        _pack("disliked_grains", disliked, disliked_conf, source="user")
    else:
        _pack("disliked_grains", None, 0.0)
    if gluten is not None:
        _pack("avoid_gluten", gluten, gluten_conf, source="user")
    else:
        _pack("avoid_gluten", None, 0.0)
    _pack("health_issue", health, health_conf)

    raw_entities = {
        "purpose": purpose,
        "frequency": freq,
        "texture": texture,
        "disliked": disliked or [],
        "gluten_flag": gluten,
        "health": health,
    }

    return {"survey": survey, "raw_entities": raw_entities}


__all__ = ["extract_survey", "SURVEY_SCHEMA"]
