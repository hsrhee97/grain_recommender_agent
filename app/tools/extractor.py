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
    extras: Dict[str, Any] | None = None,
) -> None:
    cleaned = _ensure_schema_value(field, value, schema)
    entry: Dict[str, Any] = {
        "value": cleaned,
        "confidence": round(confidence or 0.0, 2),
        "source": source,
    }
    if extras:
        entry.update({k: v for k, v in extras.items() if v is not None})
    if cleaned is None and "reason" not in entry:
        entry["reason"] = "not_confident"
    survey[field] = entry


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

LLM_SUMMARY_PROMPT = (
    "당신은 곡물 추천 챗봇의 보조 설명가입니다. 아래 설문 구조화 결과와 사용자 입력, "
    "추가 힌트를 참고해 2문장 이내의 한국어 요약을 작성하세요. 첫 문장에는 사용자가 "
    "무엇을 원하는지(목적·건강 등)를 부드럽게 인정하고, 두 번째 문장에서는 아직 확실하지 "
    "않은 선택지나 확인이 필요한 부분을 안내해 주세요. 버튼 텍스트 없이 자연어로만 답하세요."
)

SUMMARY_FIELD_LABELS = {
    "purpose": "목적",
    "frequency": "섭취 빈도",
    "texture_pref": "식감 선호",
    "disliked_grains": "기피 곡물",
    "avoid_gluten": "글루텐 회피",
    "health_issue": "건강 이슈",
}


def _normalize_llm_response(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    survey_section = payload.get("survey") or payload.get("fields") or {}
    survey: Dict[str, Dict[str, Any]] = {}
    raw_entities = payload.get("raw_entities", {}) or {}

    for field in schema.get("fields", {}):
        entry = survey_section.get(field)
        value = None
        confidence: float | None = None
        source = "llm"
        reason = None
        other_text = None
        alternatives: List[Dict[str, Any]] = []

        if isinstance(entry, dict):
            value = entry.get("value", entry.get("answer"))
            confidence = entry.get("confidence") or entry.get("score")
            source = entry.get("source") or source
            reason = entry.get("reason")
            other_text = entry.get("other_text")
            alt_entries = entry.get("alternatives") or entry.get("alt") or []
            if isinstance(alt_entries, dict):
                alt_entries = [alt_entries]
            for alt in alt_entries:
                alt_value = None
                alt_conf: float | None = None
                alt_reason = None
                alt_other = None
                if isinstance(alt, dict):
                    alt_value = alt.get("value", alt.get("answer"))
                    alt_conf = alt.get("confidence") or alt.get("score")
                    alt_reason = alt.get("reason")
                    alt_other = alt.get("other_text")
                else:
                    alt_value = alt
                cleaned_alt = _ensure_schema_value(field, alt_value, schema)
                alt_entry: Dict[str, Any] = {}
                if cleaned_alt is not None:
                    alt_entry["value"] = cleaned_alt
                if alt_conf is not None:
                    alt_entry["confidence"] = round(float(alt_conf), 2)
                if alt_reason:
                    alt_entry["reason"] = alt_reason
                if cleaned_alt is None and alt_value not in {None, "", []}:
                    alt_entry.setdefault("other_text", alt_other or alt_value)
                elif alt_other:
                    alt_entry["other_text"] = alt_other
                if alt_entry:
                    alternatives.append(alt_entry)
        elif entry is not None:
            value = entry

        cleaned = _ensure_schema_value(field, value, schema)
        extras: Dict[str, Any] = {}
        if cleaned is None and value not in {None, [], ""}:
            reason = reason or "not_in_schema"
            extras["other_text"] = value if other_text is None else other_text
        elif other_text is not None:
            extras["other_text"] = other_text
        if reason:
            extras["reason"] = reason
        if alternatives:
            extras["alternatives"] = alternatives

        _pack_field(
            survey,
            field,
            cleaned,
            confidence if confidence is not None else 0.75,
            source,
            schema,
            extras=extras,
        )

        if field not in raw_entities:
            raw_entities[field] = value

    return {"survey": survey, "raw_entities": raw_entities, "meta": {"mode": "llm"}}


def _serialize_for_summary(survey: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for field, entry in (survey or {}).items():
        if not isinstance(entry, dict):
            continue
        data: Dict[str, Any] = {
            "value": entry.get("value"),
            "confidence": entry.get("confidence"),
        }
        if entry.get("other_text"):
            data["other_text"] = entry.get("other_text")
        if entry.get("alternatives"):
            data["alternatives"] = entry.get("alternatives")
        if entry.get("reason"):
            data["reason"] = entry.get("reason")
        serialized[field] = data
    return serialized


def _fallback_summary(survey: Dict[str, Dict[str, Any]]) -> str:
    pieces: List[str] = []
    for field in ("purpose", "frequency", "texture_pref", "disliked_grains", "avoid_gluten", "health_issue"):
        entry = survey.get(field, {}) if isinstance(survey, dict) else {}
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        other = entry.get("other_text")
        label = SUMMARY_FIELD_LABELS.get(field, field)
        if value in (None, "", [], ()):  # type: ignore[arg-type]
            if other:
                pieces.append(f"{label}는 '{other}'라고 하셨어요")
            else:
                pieces.append(f"{label} 정보는 아직 확실하지 않아요")
        else:
            if isinstance(value, list):
                formatted = ", ".join(map(str, value))
            elif isinstance(value, bool):
                formatted = "예" if value else "아니오"
            else:
                formatted = str(value)
            pieces.append(f"{label}: {formatted}")
    if not pieces:
        return "사용자 의도를 파악 중이에요. 더 자세히 알려주시면 정확도가 높아져요."
    if len(pieces) == 1:
        return pieces[0]
    return " / ".join(pieces[:2])


def generate_survey_summary(
    user_text: str,
    survey: Dict[str, Dict[str, Any]],
    hints: Dict[str, Any] | None = None,
) -> str:
    """Produce a short, conversational summary of the extraction result."""

    summary_payload = {
        "user_text": user_text.strip(),
        "survey": _serialize_for_summary(survey),
        "hints": hints or {},
    }

    try:
        client = ChatLLMClient()
        messages: Iterable[ChatMessage] = [
            ChatMessage(role="system", content=LLM_SUMMARY_PROMPT),
            ChatMessage(
                role="user",
                content=json.dumps(summary_payload, ensure_ascii=False),
            ),
        ]
        response_text = client.complete(messages, temperature=0.7)
        text = response_text.strip()
        if text:
            return text
    except LLMNotConfiguredError:
        LOGGER.debug("LLM summary is not configured; falling back to template summary.")
    except Exception as exc:  # pragma: no cover - logging best effort
        LOGGER.warning("LLM summary generation failed: %s", exc)

    return _fallback_summary(survey)


def _llm_extract(user_text: str, schema: Dict[str, Any], hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    hints = hints or {}
    client = ChatLLMClient()
    messages: Iterable[ChatMessage] = [
        ChatMessage(
            role="system",
            content=LLM_EXTRACTION_PROMPT,
        ),

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

    extraction_result: Dict[str, Any] | None = None

    if use_llm:
        try:
            extraction_result = _llm_extract(user_text, schema, hints=hints)
            extraction_result.setdefault("meta", {})["mode"] = "llm"
        except LLMNotConfiguredError:
            if not fallback:
                raise
            LOGGER.warning("LLM 설정이 없어 규칙 기반 추출로 대체합니다.")
        except Exception as err:  # pragma: no cover - best effort logging
            if not fallback:
                raise
            LOGGER.warning("LLM 추출 실패로 규칙 기반으로 전환합니다: %s", err)

    if extraction_result is None:
        extraction_result = _rule_based_extract(user_text, schema)
        if use_llm and fallback:
            extraction_result.setdefault("meta", {})["mode"] = "llm_fallback_rule"

    summary_text = generate_survey_summary(user_text, extraction_result.get("survey", {}), hints=hints)
    extraction_result["assistant_utterance"] = summary_text
    extraction_result.setdefault("meta", {})["assistant_utterance"] = summary_text

    return extraction_result


__all__ = ["extract_survey", "SURVEY_SCHEMA"]
