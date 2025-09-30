"""Survey extractor that can operate in rule-based or LLM-backed modes."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.services.llm import ChatLLMClient, ChatMessage, LLMNotConfiguredError

LOGGER = logging.getLogger(__name__)

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

DEFAULT_USE_LLM = os.getenv("GRAIN_AGENT_USE_LLM", "").lower() in {"1", "true", "yes"}
DEFAULT_LLM_FALLBACK = os.getenv("GRAIN_AGENT_LLM_FALLBACK", "true").lower() not in {"0", "false", "no"}


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
            # "보리는 빼고" 같은 표현 허용
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


def _ensure_schema_value(field: str, value: Any, schema: Dict[str, Any]) -> Any:
    if value is None:
        return None
    field_schema = (schema.get("fields", {}).get(field, {}) or {})
    field_type = field_schema.get("type")
    if field_type == "bool" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"yes", "y", "true", "1", "네", "응", "맞아", "그래"}:
            value = True
        elif lowered in {"no", "n", "false", "0", "아니오", "아니요", "아냐"}:
            value = False
    if field_type == "list" and isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,/\n]+", value) if part.strip()]
        value = parts or None
    options = field_schema.get("options")
    if options and value not in options:
        return None
    if field_type == "list" and isinstance(value, list):
        filtered = [item for item in value if not options or item in options]
        return filtered or None
    return value


def _pack_field(
    survey: Dict[str, Dict[str, Any]],
    field: str,
    value: Any,
    confidence: float | None,
    source: str,
    schema: Dict[str, Any],
) -> None:
    cleaned = _ensure_schema_value(field, value, schema)
    survey[field] = {
        "value": cleaned,
        "confidence": round(confidence or 0.0, 2),
        "source": source,
    }
    if cleaned is None:
        survey[field]["reason"] = "not_confident"


def _rule_based_extract(user_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    normalized = user_text.strip()

    purpose, purpose_conf = _extract_with_patterns(normalized, _PURPOSE_MAP)
    freq, freq_conf = _detect_frequency(normalized)
    texture, texture_conf = _detect_texture(normalized)
    disliked, disliked_conf = _detect_disliked(normalized)
    gluten, gluten_conf = _detect_gluten(normalized)
    health, health_conf = _detect_health_issue(normalized)

    survey: Dict[str, Dict[str, Any]] = {}

    _pack_field(survey, "purpose", purpose, purpose_conf, "rule", schema)
    _pack_field(survey, "frequency", freq, freq_conf, "rule", schema)
    _pack_field(survey, "texture_pref", texture, texture_conf, "rule", schema)
    if disliked is not None:
        _pack_field(survey, "disliked_grains", disliked, disliked_conf, "user", schema)
    else:
        _pack_field(survey, "disliked_grains", None, 0.0, "rule", schema)
    if gluten is not None:
        _pack_field(survey, "avoid_gluten", gluten, gluten_conf, "user", schema)
    else:
        _pack_field(survey, "avoid_gluten", None, 0.0, "rule", schema)
    _pack_field(survey, "health_issue", health, health_conf, "rule", schema)

    raw_entities = {
        "purpose": purpose,
        "frequency": freq,
        "texture": texture,
        "disliked": disliked or [],
        "gluten_flag": gluten,
        "health": health,
    }

    return {"survey": survey, "raw_entities": raw_entities, "meta": {"mode": "rule"}}


LLM_EXTRACTION_PROMPT = (
    "당신은 곡물 추천 설문을 구조화하는 어시스턴트입니다. 사용자의 한국어 입력을 읽고 "
    "주어진 스키마에 맞추어 JSON으로 응답하세요. 모든 필드는 다음 형식을 따라야 합니다:\n"
    "{\n"
    '  "survey": {\n'
    '    "field_name": {"value": <값 또는 null>, "confidence": <0~1 숫자>, "source": "llm"}\n'
    "  },\n"
    '  "raw_entities": {"field_name": <추출된 원본 값>, ...}\n'
    "}\n"
    "confidence는 0과 1 사이의 숫자로 작성하고, 응답은 JSON 객체 하나만 출력하세요."
)


def _normalize_llm_response(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    survey_section = payload.get("survey") or payload.get("fields") or {}
    survey: Dict[str, Dict[str, Any]] = {}
    for field in schema.get("fields", {}):
        entry = survey_section.get(field)
        value = None
        confidence: float | None = None
        source = "llm"
        reason = None
        if isinstance(entry, dict):
            value = entry.get("value", entry.get("answer"))
            confidence = entry.get("confidence") or entry.get("score")
            source = entry.get("source") or source
            reason = entry.get("reason")
        elif entry is not None:
            value = entry
        cleaned = _ensure_schema_value(field, value, schema)
        if cleaned is None and value not in {None, [], ""}:
            reason = (reason or "not_in_schema")
        _pack_field(
            survey,
            field,
            cleaned,
            confidence if confidence is not None else 0.75,
            source,
            schema,
        )
        if reason and "reason" not in survey[field]:
            survey[field]["reason"] = reason

    raw_entities = payload.get("raw_entities")
    if raw_entities is None:
        raw_entities = {
            field: (survey_section.get(field) if not isinstance(survey_section.get(field), dict)
                    else survey_section.get(field).get("value"))
            for field in schema.get("fields", {})
            if field in survey_section
        }

    return {"survey": survey, "raw_entities": raw_entities, "meta": {"mode": "llm"}}


def _llm_extract(user_text: str, schema: Dict[str, Any], hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    hints = hints or {}
    client = ChatLLMClient()
    messages: Iterable[ChatMessage] = [
        ChatMessage(role="system", content=LLM_EXTRACTION_PROMPT),
        ChatMessage(
            role="user",
            content=(
                "사용자 입력:\n"
                f"{user_text.strip()}\n\n"
                "스키마:\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"추가 힌트: {json.dumps(hints, ensure_ascii=False)}"
            ),
        ),
    ]

    response_text = client.complete(messages, temperature=0.0, response_format={"type": "json_object"})
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as err:
        raise ValueError("LLM 응답을 JSON으로 파싱할 수 없습니다.") from err

    return _normalize_llm_response(parsed, schema)


def extract_survey(
    user_text: str,
    schema: Dict[str, Any] | None = None,
    hints: Dict[str, Any] | None = None,
    *,
    use_llm: bool | None = None,
    fallback_to_rules: bool | None = None,
) -> Dict[str, Any]:
    """Convert user free-form text into survey-style JSON structure."""

    schema = schema or SURVEY_SCHEMA
    use_llm = DEFAULT_USE_LLM if use_llm is None else use_llm
    fallback = DEFAULT_LLM_FALLBACK if fallback_to_rules is None else fallback_to_rules

    if use_llm:
        try:
            result = _llm_extract(user_text, schema, hints=hints)
            result.setdefault("meta", {})["mode"] = "llm"
            return result
        except LLMNotConfiguredError:
            if not fallback:
                raise
            LOGGER.warning("LLM 설정이 없어 규칙 기반 추출로 대체합니다.")
        except Exception as err:  # pragma: no cover - best effort logging
            if not fallback:
                raise
            LOGGER.warning("LLM 추출 실패로 규칙 기반으로 전환합니다: %s", err)

    rule_result = _rule_based_extract(user_text, schema)
    if use_llm and fallback:
        rule_result.setdefault("meta", {})["mode"] = "llm_fallback_rule"
    return rule_result


__all__ = ["extract_survey", "SURVEY_SCHEMA"]
