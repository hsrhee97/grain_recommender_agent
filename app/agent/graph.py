"""Orchestration helpers for the grain recommender agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .state import AgentState
from app.tools.extractor import extract_survey
from app.tools.validator import validate_survey
from app.tools.rules_engine import rules_engine
from app.tools.explainer import generate_explanation
from app.tools.formatter import format_recommendation

SCHEMA_PATH = Path("app/schemas/survey_schema.json")
RULES_PATH = Path("app/schemas/rule_weights.json")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


SURVEY_SCHEMA = _load_json(SCHEMA_PATH)
RULE_WEIGHTS = _load_json(RULES_PATH)


def run_agent_flow(user_text: str, user_id: str = "demo", locale: str = "ko-KR") -> Dict[str, Any]:
    """Execute the single-turn recommendation flow."""
    state = AgentState(user_id=user_id, input_text=user_text)

    extraction = extract_survey(user_text, schema=SURVEY_SCHEMA)
    state.survey = extraction["survey"]

    report = validate_survey(state.survey, SURVEY_SCHEMA, RULE_WEIGHTS, locale=locale)
    state.validator_report = report

    if not report["ok"]:
        return {
            "message": report.get("reask_message", "추가 정보가 필요합니다."),
            "payload": {
                "type": "reask",
                "missing_required": report.get("missing_required", []),
                "conflicts": report.get("conflicts", []),
            },
        }

    recommendation = rules_engine(state)
    state.recommendation = recommendation

    explanation = generate_explanation(recommendation["primary"], state.survey)
    state.explanation = explanation

    final = format_recommendation(
        recommendation["primary"],
        recommendation.get("candidates", []),
        explanation,
    )

    final.setdefault("payload", {}).setdefault("meta", {})["raw_extraction"] = extraction
    return final


__all__ = ["run_agent_flow", "SURVEY_SCHEMA", "RULE_WEIGHTS"]
