"""
LangGraph multi-agent orchestration with LangChain tool-calling agents (create_tool_calling_agent + AgentExecutor).

Flow: Monitor → Validation → Context → ML/rule scoring (deterministic + trained models) →
parallel specialist agents (Fraud, Risk, Rules) → Decision agent → deterministic eligibility/payout → Supabase persist.

Requires a modern LangChain stack (see requirements.txt). GROQ_API_KEY must be set for LLM agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.graph import END, START, StateGraph

from src.agents.core_agents import (
    AgentMessage,
    DecisionAgent,
    FraudDetectionAgent,
    RiskScoringAgent,
    RuleValidationAgent,
    get_llm,
    run_parallel_agents,
)
from src.agents.gigshield_tools import build_gigshield_toolkit
from src.models.deterministic_models import ClaimEligibilityModel, PayoutOptimizationModel
from src.pipeline.training_pipeline import InferencePipeline
from src.rag.rag_system import RAGRetriever, VectorStore, populate_knowledge_base
from src.utils.schema import ensure_worker_columns

logger = logging.getLogger(__name__)


def _final_agent_text(output: Dict[str, Any]) -> str:
    msgs = output.get("messages") or []
    if not msgs:
        return str(output)
    last = msgs[-1]
    content = getattr(last, "content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content).strip()


class _ToolAgentGraph:
    """Wraps LangChain v1 `create_agent` graph with AgentExecutor-like ainvoke({\"input\": ...})."""

    def __init__(self, compiled_graph):
        self._g = compiled_graph

    def _messages(self, user_text: str):
        from langchain_core.messages import HumanMessage

        return [HumanMessage(content=user_text)]

    async def ainvoke(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        user_text = str(input_dict.get("input", ""))
        out = await self._g.ainvoke({"messages": self._messages(user_text)})
        return {"output": _final_agent_text(out)}

    def invoke(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        user_text = str(input_dict.get("input", ""))
        out = self._g.invoke({"messages": self._messages(user_text)})
        return {"output": _final_agent_text(out)}


def _agent_executor_factory(llm, system_prompt: str, tools: List[Any]):
    """Tool-calling loop: LangChain v1 `create_agent` (preferred) or legacy AgentExecutor."""
    try:
        from langchain.agents import create_agent

        graph = create_agent(llm, tools, system_prompt=system_prompt)
        return _ToolAgentGraph(graph)
    except Exception:
        pass

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=12,
        )
    except Exception as e:
        raise ImportError(
            "Need LangChain v1 (pip install 'langchain>=1') or legacy create_tool_calling_agent."
        ) from e


class GigShieldGraphState(TypedDict, total=False):
    trace_id: str
    worker_row: Dict[str, Any]
    city: str
    context_question: Optional[str]
    monitor_report: str
    validation_report: str
    context_report: str
    fraud_specialist_report: str
    risk_specialist_report: str
    rules_specialist_report: str
    decision_agent_report: str
    ml_bundle: Dict[str, Any]
    decision_code: str
    confidence: float
    payout_amount: float
    eligibility_snapshot: Dict[str, Any]
    payout_breakdown: Dict[str, Any]
    errors: List[str]


@dataclass
class GigShieldGraphResult:
    trace_id: str
    worker_id: int
    final_state: Dict[str, Any]
    processing_time_ms: float


class GigShieldLangGraphOrchestrator:
    """Compiles and runs the LangGraph; each node invokes a real tool-calling agent except ML/eligibility."""

    def __init__(
        self,
        inference_pipeline: Optional[InferencePipeline] = None,
        vector_store: Optional[VectorStore] = None,
        ensure_kb: bool = True,
    ):
        self.llm = get_llm()
        if self.llm is None:
            raise RuntimeError("GigShieldLangGraphOrchestrator requires GROQ_API_KEY (or configured Chat LLM).")

        self.inference_pipeline = inference_pipeline
        self.vector_store = vector_store or VectorStore()
        if ensure_kb:
            try:
                populate_knowledge_base(self.vector_store)
            except Exception as e:
                logger.warning("Knowledge base populate skipped: %s", e)

        self.rag = RAGRetriever(self.vector_store)
        self.trace_holder: Dict[str, str] = {}

        fraud_model = inference_pipeline.models.get("fraud_detection") if inference_pipeline else None
        risk_model = inference_pipeline.models.get("risk_scoring") if inference_pipeline else None
        self._fraud_agent = FraudDetectionAgent(fraud_model=fraud_model)
        self._risk_agent = RiskScoringAgent(risk_model=risk_model)
        self._rule_agent = RuleValidationAgent()
        self._decision_rule = DecisionAgent()
        self._elig = ClaimEligibilityModel()
        self._payout = PayoutOptimizationModel()

        tools_map = build_gigshield_toolkit(self.rag, self.trace_holder)

        self._ex_monitor = _agent_executor_factory(
            self.llm,
            "You are GigShield MonitorAgent. Use tools to pull live disruption-style signals and knowledge. "
            "Output a concise Markdown report: observed triggers, city, citing tool results.",
            tools_map["monitor"],
        )
        self._ex_validation = _agent_executor_factory(
            self.llm,
            "You are GigShield ValidationAgent. Cross-check monitor findings vs policy/disruption KB using tools. "
            "State whether triggers are plausible and what evidence you used.",
            tools_map["validation"],
        )
        self._ex_context = _agent_executor_factory(
            self.llm,
            "You are GigShield ContextAgent. Gather policy/regional/fraud context for underwriting using tools. "
            "Summarize implications in bullets.",
            tools_map["context"],
        )
        self._ex_fraud = _agent_executor_factory(
            self.llm,
            "You are GigShield Fraud specialist. Combine ML/rule fraud signals below with fraud playbooks via tools. "
            "Do not invent numeric scores; cite tools and given JSON.",
            tools_map["fraud"],
        )
        self._ex_risk = _agent_executor_factory(
            self.llm,
            "You are GigShield Risk specialist. Explain risk posture using policy KB and the supplied risk JSON.",
            tools_map["risk"],
        )
        self._ex_rules = _agent_executor_factory(
            self.llm,
            "You are GigShield Rules specialist. Map business rules (cooling, premium, employment) to policy text using tools.",
            tools_map["rules"],
        )
        self._ex_decision = _agent_executor_factory(
            self.llm,
            "You are GigShield DecisionAgent. Read all prior agent reports and tools. "
            "Finish with a single JSON object: {\"recommendation\":\"auto_approve|auto_reject|manual_review\", "
            "\"confidence\":0-1,\"rationale\":\"...\"} and no other trailing text.",
            tools_map["decision"],
        )

        self._graph = self._compile_graph()

    def _compile_graph(self):
        builder = StateGraph(GigShieldGraphState)

        builder.add_node("monitor", self._node_monitor)
        builder.add_node("validation", self._node_validation)
        builder.add_node("context", self._node_context)
        builder.add_node("ml_core", self._node_ml_core)
        builder.add_node("specialists", self._node_specialists)
        builder.add_node("decision", self._node_decision)
        builder.add_node("deterministic", self._node_deterministic)
        builder.add_node("persist", self._node_persist)

        builder.add_edge(START, "monitor")
        builder.add_edge("monitor", "validation")
        builder.add_edge("validation", "context")
        builder.add_edge("context", "ml_core")
        builder.add_edge("ml_core", "specialists")
        builder.add_edge("specialists", "decision")
        builder.add_edge("decision", "deterministic")
        builder.add_edge("deterministic", "persist")
        builder.add_edge("persist", END)

        return builder.compile()

    async def _node_monitor(self, state: GigShieldGraphState) -> Dict[str, Any]:
        self.trace_holder["trace_id"] = state["trace_id"]
        city = state.get("city") or "unknown"
        out = await self._ex_monitor.ainvoke({"input": f"Assess {city} for parametric triggers. worker_id={state['worker_row'].get('worker_id')}"})
        return {"monitor_report": str(out.get("output", ""))}

    async def _node_validation(self, state: GigShieldGraphState) -> Dict[str, Any]:
        blob = state.get("monitor_report", "")
        out = await self._ex_validation.ainvoke({"input": f"Monitor report:\n{blob[:12000]}\nValidate with tools."})
        return {"validation_report": str(out.get("output", ""))}

    async def _node_context(self, state: GigShieldGraphState) -> Dict[str, Any]:
        row = state["worker_row"]
        q = state.get("context_question") or (
            f"Worker {row.get('worker_id')} city {state.get('city')} disruption {row.get('disruption_type')} "
            f"income_loss_pct {row.get('income_loss_percentage')}"
        )
        out = await self._ex_context.ainvoke({"input": q})
        try:
            from src.persistence.supabase_client import log_rag_query_any

            log_rag_query_any(state["trace_id"], "graph_context", q[:2000])
        except Exception:
            pass
        return {"context_report": str(out.get("output", ""))}

    async def _node_ml_core(self, state: GigShieldGraphState) -> Dict[str, Any]:
        row = state["worker_row"]
        bundle: Dict[str, Any] = {"parallel_agents": [], "ml_predictions": {}}
        parallel = await run_parallel_agents(row, state["trace_id"], self._fraud_agent, self._risk_agent, self._rule_agent)
        bundle["parallel_agents"] = [
            {"agent": m.agent_name, "type": m.message_type, "data": m.data} for m in parallel
        ]
        if self.inference_pipeline is not None:
            df = pd.DataFrame([row])
            try:
                bundle["ml_predictions"] = self.inference_pipeline.predict_for_worker(df)
            except Exception as e:
                logger.exception("Inference failed: %s", e)
                bundle["ml_predictions"] = {"error": str(e)}
        return {"ml_bundle": bundle}

    async def _node_specialists(self, state: GigShieldGraphState) -> Dict[str, Any]:
        mb = state.get("ml_bundle") or {}
        fraud_blob = json.dumps(mb, default=str)[:12000]
        tasks = [
            self._ex_fraud.ainvoke({"input": f"Fraud assessment context JSON:\n{fraud_blob}"}),
            self._ex_risk.ainvoke({"input": f"Risk assessment context JSON:\n{fraud_blob}"}),
            self._ex_rules.ainvoke(
                {
                    "input": (
                        f"Rules assessment JSON:\n{fraud_blob}\n"
                        f"Validation narrative:\n{(state.get('validation_report') or '')[:4000]}"
                    )
                }
            ),
        ]
        fraud_o, risk_o, rules_o = await asyncio.gather(*tasks)
        return {
            "fraud_specialist_report": str(fraud_o.get("output", "")),
            "risk_specialist_report": str(risk_o.get("output", "")),
            "rules_specialist_report": str(rules_o.get("output", "")),
        }

    async def _node_decision(self, state: GigShieldGraphState) -> Dict[str, Any]:
        packet = {
            "monitor": state.get("monitor_report"),
            "validation": state.get("validation_report"),
            "context": state.get("context_report"),
            "fraud_specialist": state.get("fraud_specialist_report"),
            "risk_specialist": state.get("risk_specialist_report"),
            "rules_specialist": state.get("rules_specialist_report"),
            "ml_bundle": state.get("ml_bundle"),
        }
        out = await self._ex_decision.ainvoke(
            {"input": "Synthesize the following JSON into a final underwriting decision object:\n" + json.dumps(packet, default=str)[:14000]}
        )
        return {"decision_agent_report": str(out.get("output", ""))}

    async def _node_deterministic(self, state: GigShieldGraphState) -> Dict[str, Any]:
        row = ensure_worker_columns(pd.DataFrame([state["worker_row"]])).iloc[0].to_dict()
        mb = state.get("ml_bundle") or {}
        msgs: List[AgentMessage] = []
        for item in mb.get("parallel_agents") or []:
            msgs.append(
                AgentMessage(
                    agent_name=str(item.get("agent")),
                    timestamp=datetime.now(),
                    message_type=str(item.get("type")),
                    data=item.get("data") or {},
                    trace_id=state["trace_id"],
                )
            )
        dec_msg = await self._decision_rule.process({"agent_outputs": msgs}, state["trace_id"])
        decision_code = str(dec_msg.data.get("decision", "manual_review"))
        confidence = float(dec_msg.data.get("confidence", 0.5))

        llm_rec = _parse_decision_json(state.get("decision_agent_report", ""))
        if llm_rec and llm_rec.get("recommendation") == "manual_review":
            decision_code = "manual_review"
            confidence = min(confidence, float(llm_rec.get("confidence") or confidence))

        elig = self._elig.evaluate_eligibility(row)
        payout_amount = 0.0
        payout_breakdown: Dict[str, Any] = {}
        if elig.is_eligible and decision_code == "auto_approve":
            payout_breakdown = self._payout.calculate_payout(row, elig)
            payout_amount = float(payout_breakdown.get("final_payout", 0.0))

        return {
            "decision_code": decision_code,
            "confidence": confidence,
            "payout_amount": payout_amount,
            "eligibility_snapshot": {"eligible": elig.is_eligible, "reasons": list(elig.reasons)},
            "payout_breakdown": payout_breakdown,
        }

    async def _node_persist(self, state: GigShieldGraphState) -> Dict[str, Any]:
        row = state["worker_row"]
        wid = int(row.get("worker_id", 0))
        payload = {
            "graph": True,
            "monitor_report": (state.get("monitor_report") or "")[:4000],
            "validation_report": (state.get("validation_report") or "")[:4000],
            "context_report": (state.get("context_report") or "")[:4000],
            "fraud_specialist_report": (state.get("fraud_specialist_report") or "")[:4000],
            "risk_specialist_report": (state.get("risk_specialist_report") or "")[:4000],
            "rules_specialist_report": (state.get("rules_specialist_report") or "")[:4000],
            "decision_agent_report": (state.get("decision_agent_report") or "")[:4000],
            "ml_keys": list((state.get("ml_bundle") or {}).get("ml_predictions", {}).keys()),
        }
        try:
            from src.persistence.supabase_client import insert_decision_any

            insert_decision_any(
                trace_id=state["trace_id"],
                worker_id=wid,
                decision=state.get("decision_code", ""),
                confidence=float(state.get("confidence") or 0.0),
                payout_amount=float(state.get("payout_amount") or 0.0),
                payload=payload,
            )
        except Exception as e:
            err = state.get("errors") or []
            err.append(f"persist:{e}")
            return {"errors": err}
        return {}

    async def run(self, worker_data: pd.DataFrame, city: Optional[str] = None, context_question: Optional[str] = None) -> GigShieldGraphResult:
        start = datetime.now()
        worker_data = ensure_worker_columns(worker_data)
        row = worker_data.iloc[0].to_dict()
        trace_id = str(uuid.uuid4())
        self.trace_holder["trace_id"] = trace_id

        init: GigShieldGraphState = {
            "trace_id": trace_id,
            "worker_row": row,
            "city": city or str(row.get("city", "Mumbai")),
            "context_question": context_question,
            "errors": [],
        }
        final = await self._graph.ainvoke(init)
        ms = (datetime.now() - start).total_seconds() * 1000.0
        return GigShieldGraphResult(
            trace_id=trace_id,
            worker_id=int(row.get("worker_id", 0)),
            final_state=dict(final),
            processing_time_ms=ms,
        )


def _parse_decision_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        start = text.rfind("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start : end + 1])
    except Exception:
        return None
