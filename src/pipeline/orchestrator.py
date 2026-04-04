"""
Orchestrator: monitoring → validation → RAG context → parallel ML-backed agents → decision → deterministic payout.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.core_agents import (
    AgentMessage,
    ContextAgent,
    DecisionAgent,
    FraudDetectionAgent,
    MonitorAgent,
    RiskScoringAgent,
    RuleValidationAgent,
    ValidationAgent,
    run_parallel_agents,
)
from src.rag.rag_system import RAGRetriever, VectorStore
from src.pipeline.training_pipeline import InferencePipeline
from src.models.deterministic_models import ClaimEligibilityModel, PayoutOptimizationModel
from src.integrations.mock_mcp_client import default_mcp_client


@dataclass
class WorkflowResult:
    trace_id: str
    worker_id: int
    decision: str
    confidence: float
    payout_amount: float
    processing_time_ms: float
    agent_outputs: List[AgentMessage]
    timestamp: datetime
    extras: Optional[Dict[str, Any]] = None


class GigShieldOrchestrator:
    def __init__(
        self,
        inference_pipeline: Optional[InferencePipeline] = None,
        vector_store: Optional[VectorStore] = None,
        mcp_client: Any = None,
    ):
        self.inference_pipeline = inference_pipeline
        self.vector_store = vector_store or VectorStore()
        self.rag_retriever = RAGRetriever(self.vector_store)

        self.mcp_client = mcp_client if mcp_client is not None else default_mcp_client()
        self.monitor_agent = MonitorAgent(mcp_client=self.mcp_client)
        self.validation_agent = ValidationAgent()
        self.context_agent = ContextAgent(self.rag_retriever)

        fraud_model = inference_pipeline.models.get("fraud_detection") if inference_pipeline else None
        risk_model = inference_pipeline.models.get("risk_scoring") if inference_pipeline else None

        self.fraud_detection_agent = FraudDetectionAgent(fraud_model=fraud_model)
        self.risk_scoring_agent = RiskScoringAgent(risk_model=risk_model)
        self.rule_validation_agent = RuleValidationAgent()
        self.decision_agent = DecisionAgent()

        self.claim_eligibility = ClaimEligibilityModel()
        self.payout_calculator = PayoutOptimizationModel()

    async def layer_1_monitoring(self, city: str, trace_id: str, worker_row: Optional[Dict[str, Any]] = None) -> AgentMessage:
        payload: Dict[str, Any] = {"city": city}
        if worker_row:
            payload["worker_id"] = worker_row.get("worker_id")
            payload["outlet_id"] = worker_row.get("outlet_id")
            payload["worker_row"] = worker_row
        return await self.monitor_agent.process(payload, trace_id)

    async def layer_2_validation(self, monitor_output: AgentMessage, trace_id: str) -> AgentMessage:
        rag_context = self.rag_retriever.retrieve_context(
            query=f"disruption validation {monitor_output.data.get('triggers', [])}",
            categories=["disruption_events", "insurance_policies"],
        )
        return await self.validation_agent.process({**monitor_output.data, "rag_context": rag_context}, trace_id)

    async def layer_3_context(self, worker_dict: Dict[str, Any], trace_id: str) -> AgentMessage:
        q = f"claim worker {worker_dict.get('worker_id')} city {worker_dict.get('city')} disruption {worker_dict.get('disruption_type')}"
        return await self.context_agent.process({"query": q, "categories": ["insurance_policies", "historical_claims"]}, trace_id)

    async def layer_4_parallel(self, worker_data: pd.DataFrame, trace_id: str) -> Dict[str, Any]:
        wdict = worker_data.iloc[0].to_dict()
        parallel_msgs = await run_parallel_agents(
            wdict,
            trace_id,
            self.fraud_detection_agent,
            self.risk_scoring_agent,
            self.rule_validation_agent,
        )
        ml_predictions: Dict[str, Any] = {}
        if self.inference_pipeline:
            ml_predictions = self.inference_pipeline.predict_for_worker(worker_data)
        return {"parallel_messages": parallel_msgs, "ml_predictions": ml_predictions}

    async def layer_5_decision(self, parallel_msgs: List[AgentMessage], trace_id: str) -> AgentMessage:
        return await self.decision_agent.process({"agent_outputs": parallel_msgs}, trace_id)

    async def process_claim(self, worker_data: pd.DataFrame, city: Optional[str] = None) -> WorkflowResult:
        start = datetime.now()
        trace_id = str(uuid.uuid4())
        if city is None:
            city = str(worker_data.iloc[0].get("city", "Mumbai"))

        wdict = worker_data.iloc[0].to_dict()
        mon = await self.layer_1_monitoring(city, trace_id, worker_row=wdict)
        val = await self.layer_2_validation(mon, trace_id)
        ctx = await self.layer_3_context(wdict, trace_id)
        layer4 = await self.layer_4_parallel(worker_data, trace_id)
        decision_msg = await self.layer_5_decision(layer4["parallel_messages"], trace_id)

        eligibility = self.claim_eligibility.evaluate_eligibility(wdict)

        payout_amount = 0.0
        payout_breakdown: Dict[str, Any] = {}
        if eligibility.is_eligible and decision_msg.data.get("decision") == "auto_approve":
            payout_breakdown = self.payout_calculator.calculate_payout(wdict, eligibility)
            payout_amount = float(payout_breakdown.get("final_payout", 0.0))

        end = datetime.now()
        ms = (end - start).total_seconds() * 1000.0

        agent_outputs = [mon, val, ctx, decision_msg] + layer4["parallel_messages"]

        if self.mcp_client is not None and hasattr(self.mcp_client, "submit_claim_rollout"):
            try:
                await self.mcp_client.submit_claim_rollout(
                    {
                        "trace_id": trace_id,
                        "worker_id": int(wdict.get("worker_id", 0)),
                        "decision": str(decision_msg.data.get("decision", "")),
                        "payout_amount": payout_amount,
                        "city": city,
                        "eligibility": eligibility.is_eligible,
                    }
                )
            except Exception:
                pass

        return WorkflowResult(
            trace_id=trace_id,
            worker_id=int(wdict.get("worker_id", 0)),
            decision=str(decision_msg.data.get("decision", "manual_review")),
            confidence=float(decision_msg.data.get("confidence", 0.0)),
            payout_amount=payout_amount,
            processing_time_ms=ms,
            agent_outputs=agent_outputs,
            timestamp=end,
            extras={
                "claim_eligibility": eligibility,
                "payout_breakdown": payout_breakdown,
                "ml_predictions": layer4.get("ml_predictions"),
            },
        )

    def generate_report(self, results: List[WorkflowResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append(
                {
                    "trace_id": r.trace_id,
                    "worker_id": r.worker_id,
                    "decision": r.decision,
                    "confidence": r.confidence,
                    "payout_amount": r.payout_amount,
                    "processing_time_ms": r.processing_time_ms,
                    "timestamp": r.timestamp,
                }
            )
        return pd.DataFrame(rows)
