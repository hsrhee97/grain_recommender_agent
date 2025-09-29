import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import run_agent_flow


def test_e2e_success_flow():
    text = "혈당 관리를 위해 주 6번 정도 밥을 먹어요. 찰진 식감을 원하고 보리는 빼주세요. 글루텐은 피하고 싶어요."
    result = run_agent_flow(text, user_id="test")
    assert "추천 배합" in result["message"]
    assert "보리" not in result["payload"]["primary"]["mix"]
    assert result["payload"]["primary"]["mix"]


def test_e2e_reask_when_missing():
    text = "다이어트 목적이에요."
    result = run_agent_flow(text, user_id="test")
    assert result["payload"]["type"] == "reask"
    assert "섭취" in result["message"]
