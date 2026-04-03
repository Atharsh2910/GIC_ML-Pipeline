"""SQLite persistence for agent decisions (LangChain SQLAgent target)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _db_path() -> Path:
    p = Path("data/gigshield_agents.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_sqlite() -> None:
    conn = sqlite3.connect(_db_path())
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_decisions (
                trace_id TEXT PRIMARY KEY,
                worker_id INTEGER,
                decision TEXT,
                confidence REAL,
                payout_amount REAL,
                payload_json TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT,
                query_type TEXT,
                snippet TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_decision(
    trace_id: str,
    worker_id: int,
    decision: str,
    confidence: float,
    payout_amount: float,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    init_sqlite()
    conn = sqlite3.connect(_db_path())
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO agent_decisions
            (trace_id, worker_id, decision, confidence, payout_amount, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                worker_id,
                decision,
                confidence,
                payout_amount,
                json.dumps(payload or {}, default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_rag_query(trace_id: str, query_type: str, snippet: str) -> None:
    init_sqlite()
    conn = sqlite3.connect(_db_path())
    try:
        conn.execute(
            "INSERT INTO rag_queries (trace_id, query_type, snippet, created_at) VALUES (?, ?, ?, ?)",
            (trace_id, query_type, snippet[:2000], datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
