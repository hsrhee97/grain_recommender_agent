# app/tools/rules_engine.py
# ─────────────────────────────────────────────────────────────────
# 룰 코어(점수/배합) + 어댑터(AgentState↔설문키 매핑)를 한 파일에 통합
# ─────────────────────────────────────────────────────────────────
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from typing import Dict, Any

pd.set_option("display.precision", 2)

# ===================== 경로/로더 =====================
SCHEMAS_DIR = Path("app/schemas")
GRAINS_CSV              = SCHEMAS_DIR / "grains_catalog.csv"
SECTION_WEIGHTS_CSV     = SCHEMAS_DIR / "section_weights.csv"
SURVEY_TO_NUTRIENTS_CSV = SCHEMAS_DIR / "survey_to_nutrients.csv"
NUTRIENT_TO_GRAINS_CSV  = SCHEMAS_DIR / "nutrient_to_grains.csv"
SURVEY_SCHEMA_JSON      = SCHEMAS_DIR / "survey_schema.json"
RULE_WEIGHTS_JSON       = SCHEMAS_DIR / "rule_weights.json"

def _load_grains(path=GRAINS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "태그" in df.columns:
        df["태그"] = df["태그"].fillna("").map(lambda s: str(s).split(";") if isinstance(s, str) else [])
    return df

def _load_rule_weights(path=RULE_WEIGHTS_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_section_weights(path=SECTION_WEIGHTS_CSV, fallback_json=None) -> dict:
    if Path(path).exists():
        df = pd.read_csv(path)
        return {str(r["section"]): float(r["weight"]) for _, r in df.iterrows()}
    return dict((fallback_json or {}).get("section_weights", {}))

def _load_survey_to_nutrients(path=SURVEY_TO_NUTRIENTS_CSV, fallback_json=None) -> dict:
    if Path(path).exists():
        df = pd.read_csv(path).fillna("")
        out = {}
        for (sec, opt), g in df.groupby(["section","option"]):
            ranks = g.sort_values("rank")
            lst = [x for x in ranks["nutrient"].tolist() if str(x).strip() != ""]
            out.setdefault(sec, {})[opt] = lst
        return out
    return dict((fallback_json or {}).get("survey_to_nutrients", {}))

def _load_nutrient_to_grains(path=NUTRIENT_TO_GRAINS_CSV, fallback_json=None) -> dict:
    if Path(path).exists():
        df = pd.read_csv(path)
        
        out = {}
        for nut, g in df.groupby("nutrient"):
            ranks = g.sort_values("rank")
            out[nut] = ranks["grain"].tolist()
        return out
    return dict((fallback_json or {}).get("nutrient_to_grains", {}))

def _norm_section_weights(sw: dict) -> dict:
    if not sw: return {}
    s = sum(sw.values()) or 1.0
    return {k: float(v)/s for k,v in sw.items()}

# 전역 로드(임포트 1회)
GRAINS               = _load_grains()
WEIGHTS              = _load_rule_weights()
SECTION_WEIGHTS      = _norm_section_weights(_load_section_weights(fallback_json=WEIGHTS))
SURVEY_TO_NUTRIENTS  = _load_survey_to_nutrients(fallback_json=WEIGHTS)
NUTRIENT_TO_GRAINS   = _load_nutrient_to_grains(fallback_json=WEIGHTS)
NUTRIENT_RANK_WEIGHTS= WEIGHTS.get("nutrient_rank_weights", [1.0, 2/3, 1/3])
NUTRIENT_GAIN        = float(WEIGHTS.get("nutrient_gain", 0.08))

# ===================== 코어 유틸/스코어 =====================
def _feature_sum(row, feature_multipliers: dict, base_multiplier: float = 1.0) -> float:
    mult = base_multiplier
    for feat, coef in (feature_multipliers or {}).items():
        mult *= (1.0 + float(coef) * float(row.get(feat, 0)))
    return mult

def _apply_grain_factor(name: str, mapping: dict) -> float:
    if not mapping: return 1.0
    return float(mapping.get(name, 1.0))

def _penalize_if_threshold(value: float, threshold: float, multiplier: float) -> float:
    if threshold is None or multiplier is None: return 1.0
    return float(multiplier) if float(value) >= float(threshold) else 1.0

def _pref_to_texture_fields(survey: dict) -> dict:
    if "선호식감/맛" not in survey: 
        return survey
    prefer = survey.get("선호식감/맛")
    s = dict(survey)
    if prefer == "고슬밥":
        s.setdefault("식감", "부드러움")
        s.setdefault("맛", "담백/중성")
    elif prefer == "찰진밥":
        s.setdefault("식감", "쫀득/단단")
        s.setdefault("맛", "담백/중성")
    else:
        s.setdefault("식감", "중간")
        s.setdefault("맛", "담백/중성")
    return s

def _score_purpose(row, survey, W=WEIGHTS):
    key = survey.get("취식 목적") or survey.get("목적") or "맛중심"
    pconf = W["purpose"].get(key, W["purpose"]["맛중심"])
    mult = _feature_sum(row, pconf.get("feature_multipliers", {}), pconf.get("base_multiplier",1.0))
    mult *= _apply_grain_factor(row["곡물"], pconf.get("grain_multipliers", {}))
    mult *= _apply_grain_factor(row["곡물"], pconf.get("grain_bonuses", {}))
    return mult

def _score_texture_taste(row, survey, W=WEIGHTS):
    tconf = W["texture_taste"]
    mult = 1.0
    sopt = tconf["식감"].get(survey.get("식감","중간"), {})
    mult *= _feature_sum(row, sopt.get("feature_multipliers", {}))
    mult *= _apply_grain_factor(row["곡물"], sopt.get("grain_bonuses", {}))
    mult *= _penalize_if_threshold(row.get("쫀득",0), sopt.get("chewy_penalty_threshold"), sopt.get("chewy_penalty_multiplier"))
    mopt = tconf["맛"].get(survey.get("맛","담백/중성"), {})
    mult *= _feature_sum(row, mopt.get("feature_multipliers", {}))
    mult *= _apply_grain_factor(row["곡물"], mopt.get("grain_bonuses", {}))
    mult *= _apply_grain_factor(row["곡물"], mopt.get("grain_penalties", {}))
    return mult

def _score_constitution_gut(row, survey, W=WEIGHTS):
    cg = W["constitution_gut"]
    mult = 1.0
    copt = cg["체질"].get(survey.get("체질","보통"), {})
    mult *= _feature_sum(row, copt.get("feature_multipliers", {}))
    mult *= _penalize_if_threshold(row.get("섬유",0), copt.get("fiber_penalty_at_least"), copt.get("fiber_penalty_multiplier"))
    gopt = cg["장건강"].get(survey.get("장건강","보통"), {})
    soft_bonus = gopt.get("soft_bonus", {})
    if soft_bonus:
        mult *= float(soft_bonus.get("base", 1.0))
        if "부드러움" in soft_bonus:
            mult *= (1.0 + float(soft_bonus["부드러움"]) * float(row.get("부드러움",0)))
    mult *= _feature_sum(row, gopt.get("feature_multipliers", {}))
    mult *= _penalize_if_threshold(row.get("섬유",0), gopt.get("fiber_penalty_at_least"), gopt.get("fiber_penalty_multiplier"))
    return mult

def _score_frequency(row, survey, W=WEIGHTS):
    fopt = W["frequency"].get(survey.get("섭취 빈도","주 3-4회"), {})
    mult = 1.0
    mult *= _apply_grain_factor(row["곡물"], fopt.get("grain_bonuses", {}))
    mult *= _apply_grain_factor(row["곡물"], fopt.get("grain_penalties", {}))
    return mult

def _allergen_filter(row, survey, W=WEIGHTS):
    if not survey.get("알레르겐 회피(글루텐)", False):
        return 1.0
    if W["allergen"].get("exclude_gluten_grains_if_true", True):
        if row["곡물"] in W["allergen"]["gluten_grains"]:
            return 0.0
    return 1.0

def _nutrient_bonus_per_grain(survey, section_weights=SECTION_WEIGHTS,
                              s2n=SURVEY_TO_NUTRIENTS, n2g=NUTRIENT_TO_GRAINS,
                              rank_w=NUTRIENT_RANK_WEIGHTS, gain=NUTRIENT_GAIN):
    score = {g: 0.0 for g in GRAINS["곡물"].tolist()}
    for section, ans in survey.items():
        if section not in s2n or ans in (None, ""):
            continue
        targets = s2n.get(section, {}).get(ans, [])
        if not targets:
            continue
        sec_w = float(section_weights.get(section, 0.0))
        for idx, nutrient in enumerate(targets[:3]):
            rw = rank_w[idx] if idx < len(rank_w) else 0.0
            grains = n2g.get(nutrient, [])
            for g in grains[:3]:
                score[g] = score.get(g, 0.0) + sec_w * float(rw)
    return {g: (1.0 + float(gain) * v) for g, v in score.items()}

def _choose_base(survey: dict, W=WEIGHTS):
    br = W["base_rules"]
    triggers = br.get("soft_triggers", {})
    soft = any(survey.get(sec) in (vals or []) for sec, vals in triggers.items())
    avoid = set(survey.get("기피", []) or survey.get("기피곡물", []) or [])
    def _allowed(x):
        if "없음" in avoid: return True
        if x in avoid: return False
        if survey.get("알레르겐 회피(글루텐)", False) and x in W["allergen"]["gluten_grains"]:
            return False
        return True
    cand = br["soft_base"] if soft else br["default_base"]
    if _allowed(cand): return cand
    alt = "현미" if cand == "백미" else "백미"
    if _allowed(alt): return alt
    return br["fallback"]

def _base_min_percent(base: str, survey: dict, W=WEIGHTS) -> int:
    br = W["base_rules"]
    if base == "현미" and survey.get("식감") != "부드러움" and survey.get("장건강") != "민감":
        return br["base_min_percent"].get("현미_relaxed", 35)
    return br["base_min_percent"]["default"]

def _caps_and_mins(W=WEIGHTS):
    return W.get("caps", {}), W.get("mins", {})

def _compute_scores(survey, purpose_intensity=None, texture_intensity=None, W=WEIGHTS):
    survey = _pref_to_texture_fields(survey)
    sw = SECTION_WEIGHTS
    if purpose_intensity is None:
        purpose_intensity = 1.0 + float(sw.get("취식 목적", 0.0))
    if texture_intensity is None:
        texture_intensity = 1.0 + float(sw.get("선호식감/맛", 0.0))

    nut_bonus = _nutrient_bonus_per_grain(survey)
    scores = {}
    for _, row in GRAINS.iterrows():
        name = row["곡물"]
        avoid = set(survey.get("기피", []) or survey.get("기피곡물", []) or [])
        if "없음" not in avoid and name in avoid:
            continue
        base = _allergen_filter(row, survey, W)
        if base == 0.0:
            continue
        purpose_score = max(_score_purpose(row, survey, W), 1e-6)
        texture_score = max(_score_texture_taste(row, survey, W), 1e-6)
        constitution_score = max(_score_constitution_gut(row, survey, W), 1e-6)
        frequency_score = max(_score_frequency(row, survey, W), 1e-6)
        nutrient_score = max(nut_bonus.get(name, 1.0), 1e-6)

        base *= (purpose_score ** purpose_intensity)
        base *= (texture_score ** texture_intensity)
        base *= constitution_score
        base *= frequency_score
        base *= nutrient_score
        scores[name] = max(float(base), 1e-6)
    return scores

def _select_top_grains(scores: dict, base: str, n: int):
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    names = [k for k,_ in ordered]
    if base in names: names.remove(base)
    return [base] + names[:max(0, n-1)]

def _distribute_with_caps(chosen, scores, base, survey, W=WEIGHTS):
    caps, mins = _caps_and_mins(W)
    base_min = _base_min_percent(base, survey, W)
    base_share = base_min
    remaining = 100 - base_share

    others = [g for g in chosen if g != base]
    if len(others) == 0:
        return {base: 100}

    w = np.array([scores[g] for g in others], float)
    if w.sum() == 0:
        alloc = {g: remaining/len(others) for g in others}
    else:
        w = w / w.sum()
        alloc = {g: float(remaining * w[i]) for i,g in enumerate(others)}
    alloc[base] = float(base_share)

    # 캡 적용 → 초과분 재분배
    pool = 0.0
    for g in list(alloc.keys()):
        cap = caps.get(g, 100)
        if alloc[g] > cap:
            pool += alloc[g] - cap
            alloc[g] = float(cap)
    if pool > 1e-6:
        redis = [g for g in alloc if alloc[g] < caps.get(g, 100)]
        if redis:
            room = np.array([caps.get(g,100)-alloc[g] for g in redis], float)
            room_sum = room.sum()
            if room_sum > 0:
                room = room / room_sum
                for i,g in enumerate(redis):
                    alloc[g] += float(pool * room[i])

    # 최소치 강제
    for g, minv in (mins or {}).items():
        if g in alloc and alloc[g] < minv:
            need = minv - alloc[g]
            alloc[g] = float(minv)
            donors = [x for x in alloc if x != g and alloc[x] > 0]
            if donors:
                weights = np.array([alloc[x] for x in donors], float)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    for i,x in enumerate(donors):
                        alloc[x] = max(0.0, alloc[x] - need * weights[i])

    # 합 100 정규화 + 정수화
    total = sum(alloc.values())
    if abs(total-100) > 1e-6:
        scale = 100.0 / total
        for g in alloc:
            alloc[g] *= scale
    floats = alloc.copy()
    ints = {g:int(math.floor(v)) for g,v in floats.items()}
    rem = 100 - sum(ints.values())
    rema = sorted([(g, floats[g]-ints[g]) for g in floats], key=lambda x:x[1], reverse=True)
    for i in range(rem):
        ints[rema[i % len(rema)][0]] += 1
    return ints

def _choose_base_dynamic(survey: dict, scores: dict, W=WEIGHTS):
    avoid = set(survey.get("기피", []) or survey.get("기피곡물", []) or [])
    allowed = []
    for _, row in GRAINS.iterrows():
        name = row["곡물"]
        tags = row.get("태그", []) or []
        if "base" not in tags:
            continue
        if "없음" not in avoid and name in avoid:
            continue
        if survey.get("알레르겐 회피(글루텐)", False) and name in W["allergen"]["gluten_grains"]:
            continue
        allowed.append(name)
    if allowed:
        return max(allowed, key=lambda g: scores.get(g, 0.0))
    return _choose_base(survey, W)

def _generate_candidates_core(survey: dict):
    pi = float(WEIGHTS["candidate"].get("purpose_intensity", 1.0))
    ti = float(WEIGHTS["candidate"].get("texture_intensity", 1.0))
    scores = _compute_scores(survey, pi, ti, WEIGHTS)
    base = _choose_base_dynamic(survey, scores, WEIGHTS)
    n = max(WEIGHTS["candidate"]["n_min"],
            min(WEIGHTS["candidate"]["n_max"], int(survey.get("곡물 수", 5))))
    chosen = _select_top_grains(scores, base, n)
    mix = _distribute_with_caps(chosen, scores, base, survey, WEIGHTS)
    return mix, scores

# ===================== 어댑터(Agent ↔ 코어) =====================
_KEY_MAP = {
    # Agent측 키 → 룰 코어 측 한글 키
    "purpose": "취식 목적",
    "frequency": "섭취 빈도",
    "texture_pref": "선호식감/맛",
    "disliked_grains": "기피곡물",
    "avoid_gluten": "알레르겐 회피(글루텐)",
}

def _flatten_survey(agent_survey: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (agent_survey or {}).items():
        val = v.get("value") if isinstance(v, dict) else v
        out[_KEY_MAP.get(k, k)] = val
    out.setdefault("체질", "보통")
    out.setdefault("장건강", "보통")
    out.setdefault("맛", "담백/중성")
    return out

# ===================== Agent에서 호출하는 진입점 =====================
def rules_engine(state) -> Dict[str, Any]:
    """
    Agent(그래프)에서 호출되는 함수.
    - 입력: AgentState (state.survey 사용)
    - 출력: {"primary": {...}, "candidates": [...]}
    """
    survey_plain = _flatten_survey(state.survey)
    mix, _scores = _generate_candidates_core(survey_plain)
    primary = {
        "id": "R0",
        "mix": mix,
        "score": 0.0,
        "rationale": None
    }
    return {"primary": primary, "candidates": []}
