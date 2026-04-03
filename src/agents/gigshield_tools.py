"""
LangChain @tool callables for GigShield agents (RAG, persistence, external signals).
Bindings are created via factory so each tool closes over RAGRetriever / optional Supabase / trace context.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from langchain_core.tools import StructuredTool, tool

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.rag_system import RAGRetriever


def _maybe_requests_weather(city: str) -> str:
    """Optional Open-Meteo (no API key). Falls back to stub."""
    try:
        import requests

        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=8,
        )
        r.raise_for_status()
        geo = r.json().get("results") or []
        if not geo:
            return json.dumps({"city": city, "note": "geocode miss", "rainfall_cm": 0, "temperature_c": None})
        lat, lon = geo[0]["latitude"], geo[0]["longitude"]
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m,precipitation"},
            timeout=8,
        )
        w.raise_for_status()
        cur = w.json().get("current", {})
        rain_mm = float(cur.get("precipitation") or 0)
        temp = cur.get("temperature_2m")
        return json.dumps(
            {
                "city": city,
                "rainfall_cm": round(rain_mm / 10.0, 3),
                "temperature_c": temp,
                "source": "open-meteo",
            }
        )
    except Exception as exc:
        return json.dumps({"city": city, "error": str(exc), "rainfall_cm": 0, "temperature_c": 22.0, "source": "stub"})


def build_gigshield_toolkit(
    rag: RAGRetriever,
    trace_id_holder: Dict[str, str],
) -> Dict[str, List[StructuredTool]]:
    """
    Returns tool lists keyed by agent role for create_tool_calling_agent.
    trace_id_holder: mutable dict e.g. {'trace_id': '...'} updated by graph before agent runs.
    """

    @tool
    def fetch_live_disruption_signals(city: str) -> str:
        """Fetch near-real-time weather-style signals for a city (rain proxy, temperature). Use before asserting triggers."""
        return _maybe_requests_weather(city)

    @tool
    def retrieve_disruption_knowledge(query: str) -> str:
        """Retrieve GigShield parametric disruption rules and historical disruption context from the vector store."""
        ctx = rag.retrieve_context(query, categories=["disruption_events", "regional_data"])
        return ctx.get("context_text") or ""

    @tool
    def retrieve_policy_knowledge(query: str) -> str:
        """Retrieve insurance policies, 75% threshold, cooling period, and slab rules from the vector store."""
        ctx = rag.retrieve_context(query, categories=["insurance_policies", "historical_claims"])
        return ctx.get("context_text") or ""

    @tool
    def retrieve_fraud_playbooks(query: str) -> str:
        """Retrieve fraud patterns, GPS spoofing, peer validation, and coordinated fraud guidance."""
        ctx = rag.retrieve_context(query, categories=["fraud_cases"])
        return ctx.get("context_text") or ""

    @tool
    def record_structured_observation(
        agent_name: str,
        event_type: str,
        observation_json: str,
    ) -> str:
        """Persist a structured agent observation to Supabase gigshield_agent_events (no secrets in payload)."""
        tid = trace_id_holder.get("trace_id", "")
        if not tid:
            return "skipped: no trace_id"
        try:
            from src.persistence import supabase_client

            if supabase_client.is_configured():
                payload = json.loads(observation_json) if observation_json else {}
                supabase_client.log_agent_event(tid, agent_name, event_type, payload)
                return "recorded"
        except Exception as exc:
            return f"record_failed:{exc}"
        return "supabase_not_configured"

    @tool
    def persist_underwriter_decision_stub(
        decision: str,
        confidence_0_to_1: float,
        rationale: str,
    ) -> str:
        """Reserved for human approval workflows; logs intent only when DB configured."""
        tid = trace_id_holder.get("trace_id", "")
        if not tid:
            return "skipped: no trace_id"
        try:
            from src.persistence import supabase_client

            if supabase_client.is_configured():
                supabase_client.log_agent_event(
                    tid,
                    "DecisionAgent",
                    "intent",
                    {"decision": decision, "confidence": confidence_0_to_1, "rationale": rationale},
                )
                return "recorded"
        except Exception as exc:
            return f"record_failed:{exc}"
        return "supabase_not_configured"

    monitor_tools = [fetch_live_disruption_signals, retrieve_disruption_knowledge]
    validation_tools = [retrieve_disruption_knowledge, retrieve_policy_knowledge, record_structured_observation]
    context_tools = [retrieve_policy_knowledge, retrieve_disruption_knowledge, retrieve_fraud_playbooks]
    fraud_tools = [retrieve_fraud_playbooks, retrieve_policy_knowledge, record_structured_observation]
    risk_tools = [retrieve_policy_knowledge, record_structured_observation]
    rules_tools = [retrieve_policy_knowledge, record_structured_observation]
    decision_tools = [retrieve_policy_knowledge, retrieve_fraud_playbooks, persist_underwriter_decision_stub]

    return {
        "monitor": monitor_tools,
        "validation": validation_tools,
        "context": context_tools,
        "fraud": fraud_tools,
        "risk": risk_tools,
        "rules": rules_tools,
        "decision": decision_tools,
    }
