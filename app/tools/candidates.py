"""Candidate generation helpers for additional recommendation options."""

from __future__ import annotations

from typing import Dict, List

from .rules_engine import generate_candidate_variants


def generate_alternatives(
    survey: Dict[str, Any],
    primary: Dict[str, Any],
    limit: int = 2,
) -> List[Dict[str, Any]]:
    """Produce up to ``limit`` alternative mixes based on the survey context."""

    mix = primary.get("mix") if isinstance(primary, dict) else None
    if not mix:
        return []

    variants = generate_candidate_variants(survey, primary_mix=mix, limit=limit)
    numbered = []
    for idx, variant in enumerate(variants, start=1):
        numbered.append({"id": f"R{idx}", "mix": variant})
    return numbered


__all__ = ["generate_alternatives"]