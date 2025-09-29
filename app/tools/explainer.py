"""Simple explanation generator for recommendations."""
from __future__ import annotations

from typing import Any, Dict, List


def _flatten_survey(survey: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (survey or {}).items():
        if isinstance(value, dict):
            flat[key] = value.get("value")
        else:
            flat[key] = value
    return flat


def generate_explanation(primary: Dict[str, Any], survey: Dict[str, Any]) -> str:
    flat = _flatten_survey(survey)

    reasons: List[str] = []

    purpose = flat.get("purpose")
    if purpose == "혈당관리":
        reasons.append("혈당 관리를 돕는 낮은 GI 곡물을 중심으로 구성했어요")
    elif purpose == "체중관리":
        reasons.append("포만감을 주는 섬유질 곡물을 우선 배치했어요")
    elif purpose == "근력":
        reasons.append("단백질이 풍부한 곡물을 강화했어요")
    else:
        reasons.append("균형 잡힌 맛을 위한 조합이에요")

    if flat.get("avoid_gluten"):
        reasons.append("글루텐이 포함된 곡물은 제외했어요")

    disliked = flat.get("disliked_grains") or []
    if disliked:
        reasons.append(f"사용자가 기피한 {', '.join(disliked)} 는 넣지 않았어요")

    freq = flat.get("frequency")
    if freq == "주 5-7회":
        reasons.append("잦은 섭취에도 부담이 적도록 섬유질을 적절히 맞췄어요")

    return " ".join(reasons[:3])


__all__ = ["generate_explanation"]
