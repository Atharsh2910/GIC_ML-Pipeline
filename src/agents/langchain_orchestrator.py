"""
GigShield LangChain + RAG orchestrator: all required agents wired per product architecture.

Flow: Monitor → Validation (+RAG) → Context (RAG+optional LLM) → parallel
(Fraud+RAG?, Risk, Rules) → Decision (optional LLM JSON) → Supabase (SQLite if env unset).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
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
    get_llm,
)
from src.persistence.supabase_client import insert_decision_any, log_rag_query_any
from src.rag.langchain_rag import (
    build_decision_chain,
    build_fraud_reasoning_chain,
    build_rag_qa_chain,
    ainvoke_chain,
)
from src.rag.rag_system import RAGRetriever, VectorStore, populate_knowledge_base
from src.pipeline.training_pipeline import InferencePipeline
from src.models.deterministic_models import ClaimEligibilityModel, PayoutOptimizationModel
from src.utils.schema import ensure_worker_columns


logger = logging.getLogger(__name__)


@dataclass
class LangChainWorkflowResult:
    trace_id: str
    worker_id: int
    decision: str
    confidence: float
    payout_amount: float
    processing_time_ms: float
    agent_messages: List[AgentMessage] = field(default_factory=list)
    rag_answers: Dict[str, str] = field(default_factory=dict)
    llm_decision_raw: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class GigShieldLangChainOrchestrator:
    """
    Full pipeline with LangChain RAG chains when OPENAI_API_KEY (or configured provider) is set.
    Without LLM: same agent flow, RAG retrieval still runs for validation/context (text only).
    """

    def __init__(
        self,
        inference_pipeline: Optional[InferencePipeline] = None,
        vector_store: Optional[VectorStore] = None,
        ensure_kb: bool = True,
    ):
        self.llm = get_llm()
        self.inference_pipeline = inference_pipeline

        self.vector_store = vector_store or VectorStore()
        if ensure_kb:
            try:
                populate_knowledge_base(self.vector_store)
            except Exception as e:
                logger.warning("Knowledge base populate skipped: %s", e)

        self.rag_retriever = RAGRetriever(self.vector_store)

        self.monitor_agent = MonitorAgent()
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

        # LangChain chains (only if LLM)
        self._rag_qa = None
        self._fraud_lc = None
        self._decision_lc = None
        if self.llm:
            self._rag_qa = build_rag_qa_chain(self.llm, self.rag_retriever, categories=["insurance_policies", "historical_claims"])
            self._fraud_lc = build_fraud_reasoning_chain(self.llm, self.rag_retriever)
            self._decision_lc = build_decision_chain(self.llm)

    async def run_full_workflow(
        self,
        worker_data: pd.DataFrame,
        city: Optional[str] = None,
        context_question: Optional[str] = None,
    ) -> LangChainWorkflowResult:
        start = datetime.now()
        trace_id = str(uuid.uuid4())
        worker_data = ensure_worker_columns(worker_data)
        row = worker_data.iloc[0].to_dict()
        wid = int(row.get("worker_id", 0))
        city = city or str(row.get("city", "Mumbai"))

        rag_answers: Dict[str, str] = {}
        messages: List[AgentMessage] = []

        # 1 Monitor
        mon = await self.monitor_agent.process({"city": city}, trace_id)
        messages.append(mon)

        # 2 Validation (+ RAG context in data)
        val_in = {**mon.data, "rag_context": self.rag_retriever.retrieve_context(
            query=f"validate disruption triggers {mon.data.get('triggers', [])}",
            categories=["disruption_events", "insurance_policies"],
        )}
        val = await self.validation_agent.process(val_in, trace_id)
        messages.append(val)

        # 3 Context agent (RAG + optional LLM summary)
        cq = context_question or (
            f"Summarize policy implications for worker {wid} city {city} disruption {row.get('disruption_type')} "
            f"income loss {row.get('income_loss_percentage')}"
        )
        ctx = await self.context_agent.process(
            {"query": cq, "categories": ["insurance_policies", "historical_claims", "regional_data"]},
            trace_id,
        )
        messages.append(ctx)
        log_rag_query(trace_id, "context_agent", cq)

        # Optional: LangChain RAG QA over same question
        if self._rag_qa:
            try:
                rag_answers["policy_qa"] = await ainvoke_chain(self._rag_qa, cq)
            except Exception as e:
                logger.exception("RAG QA chain failed: %s", e)
                rag_answers["policy_qa"] = f"(error: {e})"

        # 4 Parallel: fraud, risk, rules + ML pipeline
        parallel = await run_parallel_agents(
            row,
            trace_id,
            self.fraud_detection_agent,
            self.risk_scoring_agent,
            self.rule_validation_agent,
        )
        messages.extend(parallel)

        ml_predictions: Dict[str, Any] = {}
        if self.inference_pipeline:
            ml_predictions = self.inference_pipeline.predict_for_worker(worker_data)

        # Optional: LLM fraud reasoning on top of rule+ML scores
        if self._fraud_lc:
            try:
                fraud_msg = next(m for m in parallel if m.message_type == "fraud_analysis_complete")
                packet = {"worker": row, "fraud_agent": fraud_msg.data}
                rag_answers["fraud_llm"] = await ainvoke_chain(
                    self._fraud_lc, {"query": "fraud assessment", "worker_json": json.dumps(packet, default=str)}
                )
                log_rag_query_any(trace_id, "fraud_reasoning", rag_answers["fraud_llm"][:500])
            except Exception as e:
                logger.exception("Fraud LC chain failed: %s", e)
                rag_answers["fraud_llm"] = str(e)

        # 5 Decision (rule-based agent first)
        dec_msg = await self.decision_agent.process({"agent_outputs": parallel}, trace_id)
        decision = str(dec_msg.data.get("decision", "manual_review"))
        confidence = float(dec_msg.data.get("confidence", 0.5))

        llm_raw: Optional[str] = None
        if self._decision_lc:
            try:
                packet = {
                    "parallel_agents": [m.data for m in parallel],
                    "validation": val.data,
                    "context_summary": ctx.data.get("summary") or ctx.data.get("raw", {}).get("context_text", ""),
                    "fraud_rag": rag_answers.get("fraud_llm"),
                }
                llm_raw = await ainvoke_chain(self._decision_lc, packet)
                # Soft-merge: if LLM says manual_review keep conservative
                rag_answers["decision_llm"] = llm_raw
            except Exception as e:
                logger.exception("Decision LC failed: %s", e)

        # Eligibility + payout (deterministic)
        elig = self.claim_eligibility.evaluate_eligibility(row)
        payout_amount = 0.0
        payout_breakdown: Dict[str, Any] = {}
        if elig.is_eligible and decision == "auto_approve":
            payout_breakdown = self.payout_calculator.calculate_payout(row, elig)
            payout_amount = float(payout_breakdown.get("final_payout", 0.0))

        # 6 Supabase (or SQLite fallback when env not set)
        insert_decision_any(
            trace_id=trace_id,
            worker_id=wid,
            decision=decision,
            confidence=confidence,
            payout_amount=payout_amount,
            payload={
                "eligibility": elig.is_eligible,
                "rag_answers": rag_answers,
                "ml_keys": list(ml_predictions.keys()) if ml_predictions else [],
            },
        )

        end = datetime.now()
        ms = (end - start).total_seconds() * 1000.0

        return LangChainWorkflowResult(
            trace_id=trace_id,
            worker_id=wid,
            decision=decision,
            confidence=confidence,
            payout_amount=payout_amount,
            processing_time_ms=ms,
            agent_messages=[mon, val, ctx, *parallel, dec_msg],
            rag_answers=rag_answers,
            llm_decision_raw=llm_raw,
            extras={
                "claim_eligibility": elig,
                "payout_breakdown": payout_breakdown,
                "ml_predictions": ml_predictions,
                "validation": val.data,
                "monitor": mon.data,
            },
            timestamp=end,
        )
