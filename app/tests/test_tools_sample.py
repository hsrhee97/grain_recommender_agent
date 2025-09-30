import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.state import AgentState
from app.tools.validator import validate_survey
from app.tools.rules_engine import rules_engine
from app.tools.formatter import format_recommendation

SCHEMA_PATH = Path("app/schemas/survey_schema.json")
with SCHEMA_PATH.open("r", encoding="utf-8") as fp:
    SURVEY_SCHEMA = json.load(fp)


def test_validator_detects_missing_required():
    survey = {
        "purpose": {"value": "혈당관리"},
        "frequency": {"value": None},
        "texture_pref": {"value": None},
    }
    report = validate_survey(survey, SURVEY_SCHEMA, rules=None, extraction=None)
    assert not report["ok"]
    assert "frequency" in report["missing_required"]
    assert report["reask_message"] is not None


def test_rules_engine_sum_and_gluten():
    state = AgentState(user_id="u1", input_text="demo")
    state.survey = {
        "purpose": {"value": "혈당관리"},
        "frequency": {"value": "주 5-7회"},
        "texture_pref": {"value": "찰진밥"},
        "disliked_grains": {"value": ["보리"]},
        "avoid_gluten": {"value": True},
        "health_issue": {"value": "diabetes"},
    }
    rec = rules_engine(state)
    mix = rec["primary"]["mix"]
    assert sum(mix.values()) == 100
    assert "보리" not in mix


def test_formatter_adds_explanation():
    primary = {"id": "R0", "mix": {"현미": 40, "귀리": 30, "백미": 30}}
    result = format_recommendation(primary, explanation="테스트 설명")
    assert "✅" in result["message"]
    assert "📎 이유" in result["message"]
    assert result["payload"]["primary"]["mix"]["현미"] == 40
