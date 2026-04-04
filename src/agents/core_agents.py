"""
Multi-agent layer: monitoring, validation, RAG context, fraud/risk/rule checks, decision, SQL logging.
LLM calls are optional (env API keys); agents still return structured JSON without an LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import sys

sys.path.append("..")

from config.agent_config import (
    MONITOR_AGENT_CONFIG,
    VALIDATION_AGENT_CONFIG,
    CONTEXT_AGENT_CONFIG,
    FRAUD_DETECTION_AGENT_CONFIG,
    DECISION_AGENT_CONFIG,
    RULE_VALIDATION_AGENT_CONFIG,
    RISK_SCORING_AGENT_CONFIG,
    SQL_AGENT_CONFIG,
    LANGCHAIN_CONFIG,
)


@dataclass
class AgentMessage:
    agent_name: str
    timestamp: datetime
    message_type: str
    data: Dict[str, Any]
    trace_id: str
    priority: str = "medium"


def get_llm():
    """Return a LangChain chat model if API key available; else None."""
    provider = LANGCHAIN_CONFIG.get("llm_provider", "openai")
    if provider == "openai":
        key = os.getenv(LANGCHAIN_CONFIG.get("api_key_env", "OPENAI_API_KEY"))
        if not key:
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=LANGCHAIN_CONFIG.get("model_name", "gpt-4o-mini"),
                temperature=LANGCHAIN_CONFIG.get("temperature", 0.1),
                api_key=key,
            )
        except Exception:
            return None
    return None


_optional_llm = get_llm  # backward compat


class BaseAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config["name"]
        self.description = config.get("description", "")
        self.logger = logging.getLogger(self.name)

    def create_message(self, message_type: str, data: Dict[str, Any], trace_id: str, priority: str = "medium") -> AgentMessage:
        return AgentMessage(
            agent_name=self.name,
            timestamp=datetime.now(),
            message_type=message_type,
            data=data,
            trace_id=trace_id,
            priority=priority,
        )

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        raise NotImplementedError


class MonitorAgent(BaseAgent):
    """
    Layer-1 monitoring. When `mcp_client` is set (e.g. MockMCPClient), weather/regional signals
    simulate the MCP → external API path from the product architecture.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, mcp_client: Any = None):
        super().__init__(config or MONITOR_AGENT_CONFIG)
        self.trigger_thresholds = self.config["trigger_thresholds"]
        self.mcp_client = mcp_client

    async def fetch_weather_data(self, city: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.mcp_client is not None:
            bundle = await self.mcp_client.get_monitoring_signals(city, context=context or {})
            return dict(bundle.get("weather") or {})
        out = {"city": city, "rainfall_cm": 0.0, "temperature": 25.0, "alerts": []}
        wr = (context or {}).get("worker_row") or {}
        if wr.get("rainfall_cm") is not None and str(wr.get("rainfall_cm", "")) != "":
            try:
                out["rainfall_cm"] = float(wr["rainfall_cm"])
            except (TypeError, ValueError):
                pass
        if wr.get("temperature_extreme") is not None and str(wr.get("temperature_extreme", "")) != "":
            try:
                out["temperature"] = float(wr["temperature_extreme"])
            except (TypeError, ValueError):
                pass
        return out

    async def check_trigger_conditions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        triggers: List[Dict[str, Any]] = []
        if data.get("rainfall_cm", 0) >= self.trigger_thresholds["rainfall_cm"]:
            triggers.append({"type": "rainfall", "severity": data["rainfall_cm"], "threshold": self.trigger_thresholds["rainfall_cm"]})
        temp = data.get("temperature", 25.0)
        if temp >= self.trigger_thresholds["temperature_high"]:
            triggers.append({"type": "heat", "severity": temp})
        elif temp <= self.trigger_thresholds["temperature_low"]:
            triggers.append({"type": "cold", "severity": temp})
        has_rain = any(t.get("type") == "rainfall" for t in triggers)
        has_heat = any(t.get("type") == "heat" for t in triggers)
        for a in data.get("alerts") or []:
            if not isinstance(a, dict):
                continue
            at = str(a.get("type", ""))
            if at == "Heavy_Rain" and has_rain:
                continue
            if at == "Extreme_Heat" and has_heat:
                continue
            if at in ("Cyclone", "Labor_Action", "Extreme_Heat", "Heavy_Rain"):
                triggers.append(
                    {
                        "type": at.lower() if at != "Labor_Action" else "strike",
                        "severity": a.get("severity", "unknown"),
                        "source": a.get("source", "alert_feed"),
                        "detail": a,
                    }
                )
        return triggers

    def _enrich_triggers(self, triggers: List[Dict[str, Any]], regional: Dict[str, Any]) -> None:
        aw = int(regional.get("affected_workers_estimate", 50))
        dh = float(regional.get("disruption_duration_hours", 4.0))
        for t in triggers:
            t.setdefault("affected_workers", aw)
            t.setdefault("duration_hours", dh)

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        city = input_data.get("city", "unknown")
        ctx = {k: input_data[k] for k in ("worker_id", "outlet_id", "worker_row") if k in input_data}
        mcp_bundle: Dict[str, Any] = {}
        if self.mcp_client is not None:
            mcp_bundle = await self.mcp_client.get_monitoring_signals(str(city), context=ctx)
            weather = dict(mcp_bundle.get("weather") or {})
            regional = dict(mcp_bundle.get("regional") or {})
        else:
            weather = await self.fetch_weather_data(str(city), context=ctx)
            regional = {}
        triggers = await self.check_trigger_conditions(weather)
        self._enrich_triggers(triggers, regional)
        mtype = "trigger_detected" if triggers else "no_trigger"
        return self.create_message(
            mtype,
            {
                "triggers": triggers,
                "weather_data": weather,
                "mcp_bundle": mcp_bundle if mcp_bundle else None,
            },
            trace_id,
            priority="high" if triggers else "low",
        )


class ValidationAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or VALIDATION_AGENT_CONFIG)
        self.validation_rules = self.config["validation_rules"]

    async def validate_trigger(self, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        affected = trigger_data.get("affected_workers", 0)
        duration = trigger_data.get("duration_hours", 0)
        checks = {
            "min_workers": affected >= self.validation_rules["min_affected_workers"],
            "min_duration": duration >= self.validation_rules["min_disruption_duration"],
            "geo_consistent": True,
        }
        ok = all(checks.values())
        return {"is_valid": ok, "confidence": sum(checks.values()) / len(checks), "checks": checks}

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        triggers = input_data.get("triggers", [])
        validated = []
        for t in triggers:
            v = await self.validate_trigger(t)
            if v["is_valid"]:
                validated.append({**t, "validation": v})
        return self.create_message(
            "validation_complete",
            {"validated_triggers": validated, "validation_summary": {"total": len(triggers), "valid": len(validated)}},
            trace_id,
            priority="high" if validated else "medium",
        )


class ContextAgent(BaseAgent):
    """RAG-backed context enrichment; optional LLM summarization."""

    def __init__(self, rag_retriever, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or CONTEXT_AGENT_CONFIG)
        self.rag = rag_retriever
        self.llm = get_llm()

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        query = input_data.get("query") or input_data.get("context_query", "insurance eligibility disruption")
        cats = input_data.get("categories") or self.config.get("context_types", [])
        ctx = self.rag.retrieve_context(query=query, categories=cats if cats else None)
        summary = ctx["context_text"]
        if self.llm and self.config.get("enable_llm_summary", True):
            try:
                from langchain_core.messages import HumanMessage

                resp = await self.llm.ainvoke(
                    [HumanMessage(content=f"Summarize for underwriters in 3 bullets:\n{summary}")]
                )
                summary = getattr(resp, "content", str(resp))
            except Exception:
                pass
        return self.create_message("context_enriched", {"raw": ctx, "summary": summary}, trace_id, priority="medium")


class FraudDetectionAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None, fraud_model=None):
        super().__init__(config or FRAUD_DETECTION_AGENT_CONFIG)
        self.fraud_model = fraud_model
        self.weights = self.config["fraud_scoring"]["weights"]

    async def calculate_fraud_score(self, worker_data: Dict[str, Any]) -> Dict[str, Any]:
        scores = {
            "gps_spoofing": float(worker_data.get("gps_spoofing_score", 0.0)),
            "movement_anomaly": 1.0 - float(worker_data.get("movement_realism_score", 1.0)),
            "peer_deviation": abs(1.0 - float(worker_data.get("peer_group_activity_ratio", 1.0))),
            "behavior_inconsistency": 0.0,
            "device_sharing": float(worker_data.get("device_sharing_flag", 0)),
        }
        orders = float(worker_data.get("orders_completed_week", 0))
        hours = float(worker_data.get("active_hours_week", 1e-6))
        oph = orders / hours
        if oph < 0.2 or oph > 5.0:
            scores["behavior_inconsistency"] = 0.7
        rule_score = sum(scores[k] * self.weights[k] for k in self.weights)

        ml_prob = None
        if self.fraud_model is not None:
            import pandas as pd

            pred = self.fraud_model.predict(pd.DataFrame([worker_data]))
            ml_prob = float(pred["fraud_probability"].iloc[0])
            combined = 0.45 * rule_score + 0.55 * ml_prob
        else:
            combined = rule_score

        thr_r = self.config["fraud_scoring"]["threshold_reject"]
        thr_f = self.config["fraud_scoring"]["threshold_flag"]
        if combined >= thr_r:
            action = "reject"
        elif combined >= thr_f:
            action = "flag_for_review"
        else:
            action = "approve"

        return {
            "fraud_score": float(combined),
            "component_scores": scores,
            "ml_fraud_probability": ml_prob,
            "action": action,
            "trust_rating": float(1.0 - combined),
        }

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        worker = input_data.get("worker_data", {})
        out = await self.calculate_fraud_score(worker)
        return self.create_message(

            "fraud_analysis_complete",

            out,

            trace_id,

            priority="high" if out["action"] != "approve" else "medium",

        )


class RiskScoringAgent(BaseAgent):
    """Wraps trained risk model if provided; else heuristic."""

    def __init__(self, risk_model=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or RISK_SCORING_AGENT_CONFIG)
        self.risk_model = risk_model

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        import pandas as pd

        worker = input_data.get("worker_data", {})
        if self.risk_model is not None:
            pred = self.risk_model.predict(pd.DataFrame([worker]))
            score = float(pred["risk_score"].iloc[0])
        else:
            score = float(worker.get("overall_risk_score", 0.5))
        return self.create_message("risk_scoring_complete", {"risk_score": score}, trace_id, priority="medium")


class RuleValidationAgent(BaseAgent):
    """Business rules: cooling, slab, employment — complements deterministic ClaimEligibility."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or RULE_VALIDATION_AGENT_CONFIG)

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        w = input_data.get("worker_data", {})
        violations = []
        if w.get("cooling_period_completed", 0) != 1:
            violations.append("cooling_period")
        if w.get("premium_paid", 0) != 1:
            violations.append("premium_not_paid")
        et = str(w.get("employment_type", "")).lower().replace("_", "-")
        allowed_employment = ("full-time", "part-time", "occasional", "full_time", "part_time")
        if et and et not in allowed_employment:
            violations.append("employment_type")
        ok = len(violations) == 0
        return self.create_message(
            "rule_validation",
            {"ok": ok, "violations": violations},
            trace_id,
            priority="high" if not ok else "low",
        )


class DecisionAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or DECISION_AGENT_CONFIG)
        self.decision_matrix = self.config["decision_matrix"]

    def aggregate_signals(self, agent_outputs: List[AgentMessage]) -> Dict[str, Any]:
        agg: Dict[str, Any] = {"fraud_score": None, "risk_score": None, "rules_ok": None}
        for m in agent_outputs:
            if m.message_type == "fraud_analysis_complete":
                agg["fraud_score"] = m.data.get("fraud_score")
            if m.message_type == "risk_scoring_complete":
                agg["risk_score"] = m.data.get("risk_score")
            if m.message_type == "rule_validation":
                agg["rules_ok"] = m.data.get("ok")
        return agg

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        outputs: List[AgentMessage] = input_data.get("agent_outputs", [])
        agg = self.aggregate_signals(outputs)
        fraud = agg.get("fraud_score")
        risk = agg.get("risk_score")
        rules_ok = agg.get("rules_ok")

        if rules_ok is False:
            decision = "auto_reject"
            confidence = 0.95
        elif fraud is not None and fraud >= self.decision_matrix.get("auto_reject_threshold", 0.3) * 2.5:
            decision = "auto_reject"
            confidence = 0.9
        elif fraud is not None and fraud < 0.25 and (risk is None or risk < 0.6):
            decision = "auto_approve"
            confidence = 0.88
        else:
            decision = "manual_review"
            confidence = 0.55

        return self.create_message("decision_final", {"decision": decision, "confidence": confidence, "aggregated_signals": agg}, trace_id, priority="critical")


class SQLAgent(BaseAgent):
    """Structured primary persistence via Supabase when configured; SQLite + JSONL mirror for audit."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or SQL_AGENT_CONFIG)

    async def process(self, input_data: Dict[str, Any], trace_id: str) -> AgentMessage:
        payload = input_data.get("record", {})
        try:
            from src.persistence.supabase_client import insert_decision_any

            insert_decision_any(
                trace_id=trace_id,
                worker_id=int(payload.get("worker_id", 0)),
                decision=str(payload.get("decision", "")),
                confidence=float(payload.get("confidence", 0.0)),
                payout_amount=float(payload.get("payout_amount", 0.0)),
                payload=payload,
            )
        except Exception:
            pass
        os.makedirs("logs", exist_ok=True)
        path = os.path.join("logs", "agent_decisions.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"trace_id": trace_id, **payload}, default=str) + "\n")
        return self.create_message(
            "sql_recorded",
            {"path": path, "sqlite_fallback": "data/gigshield_agents.db", "supabase": "when SUPABASE_* set"},
            trace_id,
            priority="low",
        )


async def run_parallel_agents(
    worker_data: Dict[str, Any],
    trace_id: str,
    fraud_agent: FraudDetectionAgent,
    risk_agent: RiskScoringAgent,
    rule_agent: RuleValidationAgent,
) -> List[AgentMessage]:
    tasks = [
        fraud_agent.process({"worker_data": worker_data}, trace_id),
        risk_agent.process({"worker_data": worker_data}, trace_id),
        rule_agent.process({"worker_data": worker_data}, trace_id),
    ]
    return await asyncio.gather(*tasks)
