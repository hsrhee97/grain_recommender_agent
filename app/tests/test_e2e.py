import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import run_agent_flow


def test_e2e_success_flow():
    text = "혈당 관리를 위해 주 6번 정도 밥을 먹어요. 찰진 식감을 원하고 보리는 빼주세요. 글루텐은 피하고 싶어요."
    result = run_agent_flow(user_text=text, user_id="test")
    assert "추천 배합" in result["message"]
    assert "보리" not in result["payload"]["primary"]["mix"]
    assert result["payload"]["primary"]["mix"]

    meta = result["payload"].get("meta", {})
    assert meta.get("memory_after", {}).get("user_id") == "test"
    assert "feedback_error" not in meta


def test_e2e_reask_when_missing():
    text = "다이어트 목적이에요."
    result = run_agent_flow(user_text=text, user_id="test")
    assert result["payload"]["type"] == "reask"
    assert "섭취" in result["message"]  # 누락 필드 가이드


def test_feedback_collection_updates_memory():
    text = "혈당 관리를 위해 주 6번 정도 밥을 먹어요. 찰진 식감을 원하고 보리는 빼주세요. 글루텐은 피하고 싶어요."
    result = run_agent_flow(
        user_text=text,
        user_id="tester",
        feedback={"satisfaction": "like", "reason": "good"},
    )
    meta = result["payload"]["meta"]
    assert meta["feedback"]["ok"] is True
    assert meta["feedback"]["feedback"]["satisfaction"] == "like"
    assert meta["memory_after"].get("last_feedback")


def test_audio_input_requires_payload():
    # 음성만 주어질 때도 흐름이 동작하고(ASR 후) 유효성 검사 결과에 따라 reask가 나올 수 있음
    result = run_agent_flow(
        user_text=None,
        audio_bytes=b"dummy",
        user_id="audio",
    )
    assert result["payload"]["type"] in ("reask", "success")


def test_missing_inputs_raise_error():
    # 텍스트/오디오 모두 없으면 에러
    with pytest.raises(ValueError):
        run_agent_flow(user_text=None)
