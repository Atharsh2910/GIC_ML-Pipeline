"""
Supabase client initialization and typed writes for GigShield workflow.
Requires: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in environment.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from config.supabase_config import (
    ENV_SERVICE_KEY,
    ENV_URL,
    TABLE_AGENT_EVENTS,
    TABLE_DECISIONS,
    TABLE_RAG_QUERIES,
    TABLE_WORKERS,
)

_client = None


def get_supabase():
    """Lazy singleton Supabase client (PostgREST + service role)."""
    global _client
    if _client is not None:
        return _client
    url = (os.getenv(ENV_URL) or "").strip().rstrip("/")
    key = (os.getenv(ENV_SERVICE_KEY) or os.getenv("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError(
            f"Set {ENV_URL} and {ENV_SERVICE_KEY} in .env (service role key from Supabase → Settings → API)."
        )
    from supabase import create_client

    _client = create_client(url, key)
    return _client


def is_configured() -> bool:
    return bool((os.getenv(ENV_URL) or "").strip() and (os.getenv(ENV_SERVICE_KEY) or os.getenv("SUPABASE_KEY") or "").strip())


def insert_decision(
    trace_id: str,
    worker_id: int,
    decision: str,
    confidence: float,
    payout_amount: float,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    sb = get_supabase()
    sb.table(TABLE_DECISIONS).upsert(
        {
            "trace_id": trace_id,
            "worker_id": worker_id,
            "decision": decision,
            "confidence": confidence,
            "payout_amount": payout_amount,
            "payload_json": payload or {},
        },
        on_conflict="trace_id",
    ).execute()


def log_rag_query(trace_id: str, query_type: str, snippet: str) -> None:
    sb = get_supabase()
    sb.table(TABLE_RAG_QUERIES).insert(
        {
            "trace_id": trace_id,
            "query_type": query_type,
            "snippet": (snippet or "")[:8000],
        }
    ).execute()


def log_agent_event(trace_id: str, agent_name: str, event_type: str, payload: Dict[str, Any]) -> None:
    sb = get_supabase()
    sb.table(TABLE_AGENT_EVENTS).insert(
        {
            "trace_id": trace_id,
            "agent_name": agent_name,
            "event_type": event_type,
            "payload": payload,
        }
    ).execute()


def healthcheck() -> Dict[str, Any]:
    """Verify URL/key; optionally that decisions table exists."""
    if not is_configured():
        return {"ok": False, "error": "supabase env not configured"}
    try:
        sb = get_supabase()
        sb.table(TABLE_DECISIONS).select("trace_id").limit(1).execute()
        return {"ok": True, "table": TABLE_DECISIONS}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def insert_decision_any(
    trace_id: str,
    worker_id: int,
    decision: str,
    confidence: float,
    payout_amount: float,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Write to Supabase when configured; otherwise SQLite (`data/gigshield_agents.db`)."""
    if is_configured():
        insert_decision(trace_id, worker_id, decision, confidence, payout_amount, payload)
        return
    from src.agents.sql_store import insert_decision as sqlite_insert

    sqlite_insert(trace_id, worker_id, decision, confidence, payout_amount, payload)


def log_rag_query_any(trace_id: str, query_type: str, snippet: str) -> None:
    if is_configured():
        log_rag_query(trace_id, query_type, snippet)
        return
    from src.agents.sql_store import log_rag_query as sqlite_log

    sqlite_log(trace_id, query_type, snippet)


def bulk_upsert_workers(rows: List[Dict[str, Any]], batch_size: int = 250) -> int:
    """
    Upsert worker snapshots into gigshield_workers. Each row: {"worker_id": int, "record": dict}.
    Requires migration 002 and TABLE_WORKERS.
    """
    if not rows:
        return 0
    sb = get_supabase()
    total = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        sb.table(TABLE_WORKERS).upsert(chunk, on_conflict="worker_id").execute()
        total += len(chunk)
    return total
