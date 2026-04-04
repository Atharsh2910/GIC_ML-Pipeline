"""
GigShield FastAPI entrypoint.

Run from repository root (recommended):

  uvicorn app:app --host 0.0.0.0 --port 8000

Reload during development:

  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Implementation lives in ``src.api.main`` (models, RAG, classic + LangGraph routes, optional /partner-mock).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.main import app  # noqa: E402

__all__ = ["app"]
