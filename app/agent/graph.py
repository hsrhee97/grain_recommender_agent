"""Orchestration helpers for the grain recommender agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .state import AgentState
from app.tools.extractor import extract_survey
from app.tools.validator import validate_survey
from app.tools.rules_engine import rules_engine
from app.tools.explainer import generate_explanation
from app.tools.formatter import format_recommendation
from app.tools.memory import get_user_memory, update_memory_from_survey
from app.tools.feedback import record_feedback, FeedbackValidationError
from app.tools.candidates import generate_alternatives
from app.tools.asr import transcribe_audio

SCHEMA_PATH = Path("app/schemas/survey_schema.json")
RULES_PATH = Path("app/schemas/rule_weights.json")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


SURVEY_SCHEMA = _load_json(SCHEMA_PATH)
RULE_WEIGHTS = _load_json(RULES_PATH)


def run_agent_flow(
    user_text: Optional[str] = None,
    user_id: str = "demo",
    locale: str = "ko-KR",
    *,
    audio_bytes: bytes | None = None,
    audio_url: str | None = None,
    asr_lang: str | None = None,
    use_llm: bool | None = None,
    fallback_to_rules: bool | None = None,
    generate_alternative_candidates: bool = True,
    feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the single-turn recommendation flow."""

    if user_text is None and audio_bytes is None and audio_url is None:
        raise ValueError("user_text 또는 음성 입력 중 하나는 필요합니다.")

    transcription_meta = None
    if (audio_bytes is not None or audio_url is not None) and not user_text:
        transcription = transcribe_audio(
            audio_bytes=audio_bytes,
            audio_url=audio_url,
            lang=asr_lang or locale,
        )
        user_text = transcription.get("text", "")
        transcription_meta = transcription

    state = AgentState(user_id=user_id, input_text=user_text or "")

    memory_before = get_user_memory(user_id)

    extraction = extract_survey(
        user_text or "",
        schema=SURVEY_SCHEMA,
        use_llm=use_llm,
        fallback_to_rules=fallback_to_rules,
    )
    state.survey = extraction["survey"]

    report = validate_survey(
        state.survey,
        SURVEY_SCHEMA,
        RULE_WEIGHTS,
        extraction=extraction,
        locale=locale,
    )
    state.validator_report = report

    if not report["ok"]:
        payload = {
            "type": "reask",
            "missing_required": report.get("missing_required", []),
            "conflicts": report.get("conflicts", []),
        }
        if extraction.get("assistant_utterance"):
            payload["assistant_summary"] = extraction["assistant_utterance"]
        return {
            "message": report.get("reask_message", "추가 정보가 필요합니다."),
            "payload": payload,
        }

    recommendation = rules_engine(state)
    state.recommendation = recommendation

    if generate_alternative_candidates:
        extras = generate_alternatives(state.survey, recommendation["primary"])
        if extras:
            recommendation.setdefault("candidates", []).extend(extras)

    explanation = generate_explanation(recommendation["primary"], state.survey)
    state.explanation = explanation

    final = format_recommendation(
        recommendation["primary"],
        recommendation.get("candidates", []),
        explanation,
    )

    meta = final.setdefault("payload", {}).setdefault("meta", {})
    meta["raw_extraction"] = extraction
    if extraction.get("assistant_utterance"):
        meta["assistant_summary"] = extraction["assistant_utterance"]
    if transcription_meta:
        meta["transcription"] = transcription_meta

    meta["memory_before"] = memory_before
    memory_after = update_memory_from_survey(user_id, state.survey)
    meta["memory_after"] = memory_after

    if feedback:
        try:
            feedback_result = record_feedback(user_id, feedback)
            meta["feedback"] = feedback_result
            meta["memory_after"] = feedback_result.get("baseline", memory_after)
        except FeedbackValidationError as exc:
            meta["feedback_error"] = str(exc)

    return final


__all__ = ["run_agent_flow", "SURVEY_SCHEMA", "RULE_WEIGHTS"]
