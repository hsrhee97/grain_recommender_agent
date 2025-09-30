"""Data containers for the grain recommendation agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentState:
    user_id: str
    input_text: str
    survey: Dict[str, Any] = field(default_factory=dict)
    validator_report: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None

    def flatten_survey(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for key, value in (self.survey or {}).items():
            if isinstance(value, dict):
                flat[key] = value.get("value")
            else:
                flat[key] = value
        return flat


__all__ = ["AgentState"]
