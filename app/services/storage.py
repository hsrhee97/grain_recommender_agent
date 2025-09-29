"""Storage helpers for feedback and memory persistence.

The implementation intentionally keeps the abstraction thin so that the
in-memory agent code can rely on a single module regardless of whether the
backing store is SQLite or a future JSON/remote service.  For the current
project the requirements are modest, so a lightweight SQLite database is more
than sufficient and ships with Python.

The storage layer exposes a `MemoryStorage` class with three capabilities:

* fetch the current memory snapshot for a user
* upsert (merge) memory attributes such as liked/disliked grains or
  preferences
* append explicit feedback events while also keeping `last_feedback` in the
  user memory record

The merging rules follow the product specification: grain lists are treated as
sets, preference dictionaries are shallow-merged, and timestamps are stamped in
ISO8601 (UTC) format for traceability.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()


def _utcnow() -> str:
    """Return the current UTC timestamp in ISO8601 format."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryStorage:
    """SQLite-backed persistence for user memory and feedback."""

    db_path: Path = Path("data/memory.sqlite")

    def __post_init__(self) -> None:  # pragma: no cover - exercised indirectly
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    satisfaction TEXT NOT NULL,
                    reason TEXT,
                    mix TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_user_memory(self, user_id: str) -> Dict[str, Any]:
        """Fetch the persisted memory snapshot for a user.

        If the user has not been seen before an empty baseline matching the
        documented schema is returned.
        """

        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM user_memory WHERE user_id = ?", (user_id,)
            ).fetchone()

        if not row:
            return {
                "user_id": user_id,
                "liked": [],
                "disliked": [],
                "preferences": {},
                "last_feedback": None,
            }

        try:
            payload = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):  # pragma: no cover - defensive
            payload = {}

        payload.setdefault("user_id", user_id)
        payload.setdefault("liked", [])
        payload.setdefault("disliked", [])
        payload.setdefault("preferences", {})
        payload.setdefault("last_feedback", None)
        return payload

    def update_user_memory(self, user_id: str, update: Dict[str, Any]) -> Dict[str, Any]:
        """Merge the provided update into the stored memory record."""

        baseline = self.get_user_memory(user_id)
        if not update:
            return baseline

        merged = dict(baseline)

        if "liked" in update and update["liked"]:
            merged_liked = set(map(str, baseline.get("liked", [])))
            merged_liked.update(map(str, update.get("liked", []) or []))
            merged["liked"] = sorted(merged_liked)

        if "disliked" in update and update["disliked"]:
            merged_disliked = set(map(str, baseline.get("disliked", [])))
            merged_disliked.update(map(str, update.get("disliked", []) or []))
            merged["disliked"] = sorted(merged_disliked)

        if "preferences" in update and update["preferences"]:
            prefs = dict(baseline.get("preferences", {}))
            prefs.update({k: v for k, v in (update.get("preferences") or {}).items() if v is not None})
            merged["preferences"] = prefs

        if "last_feedback" in update and update["last_feedback"]:
            merged["last_feedback"] = update["last_feedback"]

        payload = json.dumps(merged, ensure_ascii=False)
        stamp = _utcnow()

        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memory (user_id, data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    data=excluded.data,
                    updated_at=excluded.updated_at
                """,
                (user_id, payload, stamp),
            )

        return merged

    def append_feedback(
        self,
        user_id: str,
        satisfaction: str,
        reason: Optional[str] = None,
        mix: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a feedback entry and update the memory snapshot."""

        created_at = _utcnow()
        mix_json = json.dumps(mix, ensure_ascii=False) if mix is not None else None

        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_log (user_id, satisfaction, reason, mix, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, satisfaction, reason, mix_json, created_at),
            )

        feedback_entry = {
            "user_id": user_id,
            "satisfaction": satisfaction,
            "reason": reason,
            "mix": mix,
            "created_at": created_at,
        }

        self.update_user_memory(user_id, {"last_feedback": feedback_entry})
        return feedback_entry


# Singleton instance used across the process to avoid repeatedly opening files.
MEMORY_STORAGE = MemoryStorage()


__all__ = ["MemoryStorage", "MEMORY_STORAGE"]