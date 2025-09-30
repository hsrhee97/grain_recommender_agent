import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.services.storage import MemoryStorage
import app.services.storage as storage_module
import app.tools.memory as memory_module
import app.tools.feedback as feedback_module


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Ensure each test interacts with an isolated memory database."""

    db_path = tmp_path / "memory.sqlite"
    storage = MemoryStorage(db_path=db_path)
    monkeypatch.setattr(storage_module, "MEMORY_STORAGE", storage, raising=False)
    monkeypatch.setattr(memory_module, "MEMORY_STORAGE", storage, raising=False)
    monkeypatch.setattr(feedback_module, "MEMORY_STORAGE", storage, raising=False)
    yield
