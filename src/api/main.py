"""
GigShield FastAPI application.

Run from repository root:
  uvicorn app:app --host 0.0.0.0 --port 8000

Or:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Partner mock APIs (weather/news/telecom/...) are mounted at /partner-mock (e.g. /partner-mock/api/weather).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import (
    ClassicClaimRequest,
    EvaluateWorkerRequest,
    InferencePredictRequest,
    OrchestrateBatchRequest,
    PaymentVerifyRequest,
    RAGRetrieveRequest,
)
import razorpay
from supabase import create_client, Client

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)

# --- Lifespan globals ---
inference_pipeline: Any = None
vector_store: Any = None
classic_orchestrator: Any = None
langgraph_orchestrator: Any = None
supabase: Client = None


def _model_paths(model_dir: str) -> Dict[str, str]:
    base = (model_dir or "models").strip().rstrip("/\\")
    return {
        "income_forecasting": f"{base}/income_forecasting",
        "risk_scoring": f"{base}/risk_scoring",
        "fraud_detection": f"{base}/fraud_detection",
        "disruption_impact": f"{base}/disruption_impact",
        "behavior_analysis": f"{base}/behavior_analysis",
        "premium_prediction": f"{base}/premium_prediction",
    }


def json_safe(obj: Any) -> Any:
    """Make ML outputs JSON-serializable."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        if len(obj) == 1:
            return json_safe(obj.iloc[0].to_dict())
        return json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())
    if hasattr(obj, "tolist"):
        try:
            return json_safe(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    return str(obj)


def _workflow_to_dict(r: Any) -> Dict[str, Any]:
    from src.pipeline.orchestrator import WorkflowResult

    if not isinstance(r, WorkflowResult):
        return json_safe(r)
    extras = r.extras or {}
    elig = extras.get("claim_eligibility")
    elig_dict = None
    if elig is not None:
        try:
            elig_dict = {"is_eligible": bool(elig.is_eligible), "reasons": list(getattr(elig, "reasons", []) or [])}
        except Exception:
            elig_dict = str(elig)
    return {
        "trace_id": r.trace_id,
        "worker_id": r.worker_id,
        "decision": r.decision,
        "confidence": r.confidence,
        "payout_amount": r.payout_amount,
        "processing_time_ms": r.processing_time_ms,
        "timestamp": r.timestamp.isoformat() if hasattr(r.timestamp, "isoformat") else str(r.timestamp),
        "claim_eligibility": elig_dict,
        "payout_breakdown": json_safe(extras.get("payout_breakdown")),
        "ml_predictions": json_safe(extras.get("ml_predictions")),
    }


def load_models_sync():
    global inference_pipeline, vector_store, classic_orchestrator, langgraph_orchestrator

    model_dir = os.getenv("GIGSHIELD_MODEL_DIR", "models")
    paths = _model_paths(model_dir)

    print("\n[GigShield API] Background task starting — loading models and orchestrators...")

    try:
        from src.pipeline.training_pipeline import InferencePipeline

        inference_pipeline = InferencePipeline(paths)
        print("[GigShield API] InferencePipeline loaded.")
    except Exception as e:
        logger.warning("InferencePipeline not loaded: %s", e)
        print(f"[GigShield API] Warning: InferencePipeline unavailable: {e}")

    try:
        from src.rag.rag_system import VectorStore, populate_knowledge_base

        vector_store = VectorStore()
        populate_knowledge_base(vector_store)
        print("[GigShield API] Vector store + knowledge bundle ready.")
    except Exception as e:
        logger.warning("Vector store / KB: %s", e)
        print(f"[GigShield API] Warning: RAG vector store limited or offline: {e}")

    try:
        from src.pipeline.orchestrator import GigShieldOrchestrator

        classic_orchestrator = GigShieldOrchestrator(
            inference_pipeline=inference_pipeline,
            vector_store=vector_store,
        )
        print("[GigShield API] Classic GigShieldOrchestrator ready (no Groq/LLM required).")
    except Exception as e:
        logger.warning("Classic orchestrator: %s", e)
        print(f"[GigShield API] Warning: classic orchestrator failed: {e}")

    try:
        from src.agents.gigshield_langgraph import GigShieldLangGraphOrchestrator

        lang_ensure_kb = vector_store is None
        langgraph_orchestrator = GigShieldLangGraphOrchestrator(
            inference_pipeline=inference_pipeline,
            vector_store=vector_store,
            ensure_kb=lang_ensure_kb,
        )
        print("[GigShield API] LangGraph orchestrator ready.")
    except Exception as e:
        logger.warning("LangGraph orchestrator: %s", e)
        print(f"[GigShield API] Warning: LangGraph offline (check GROQ_API_KEY): {e}")

    try:
        supabase_url = os.getenv("VITE_SUPABASE_URL")
        supabase_key = os.getenv("VITE_SUPABASE_ANON_KEY")
        if supabase_url and supabase_key:
            supabase = create_client(supabase_url, supabase_key)
            print("[GigShield API] Supabase client initialized.")
        else:
            print("[GigShield API] Warning: Supabase credentials missing.")
    except Exception as e:
        logger.warning("Supabase init: %s", e)
        print(f"[GigShield API] Warning: Supabase initialization failed: {e}")

    print("[GigShield API] Background loading complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    
    print("\n[GigShield API] Starting FastAPI server. Models are loading in the background...")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_models_sync)
    yield

    print("[GigShield API] Shutdown.")


def create_app() -> FastAPI:
    application = FastAPI(
        title="GigShield API",
        description="ML inference, RAG, classic multi-agent pipeline, and LangGraph tool agents.",
        version="1.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("GIGSHIELD_CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional: mount local partner mock under /partner-mock
    if os.getenv("GIGSHIELD_MOUNT_PARTNER_MOCK", "true").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from mock_api.mock_api import app as partner_mock_app

            application.mount("/partner-mock", partner_mock_app)
            logger.info("Mounted partner mock at /partner-mock")
        except Exception as e:
            logger.warning("Partner mock not mounted: %s", e)

    @application.get("/", tags=["System"])
    def root():
        return {
            "service": "GigShield API",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "evaluate_worker": "POST /api/evaluate_worker",
                "orchestrate": "POST /api/orchestrate",
                "process_claim": "POST /api/claims/process-classic",
                "ml_predict": "POST /api/inference/predict",
                "rag": "POST /api/rag/retrieve",
                "payment_create": "POST /api/payment/create-order",
                "payment_verify": "POST /api/payment/verify",
                "payment_history": "GET /api/payment/history/{worker_id}"
            },
        }

    @application.get("/health", tags=["System"])
    def health_check():
        return {
            "status": "online",
            "models_loaded": inference_pipeline is not None,
            "classic_orchestrator": classic_orchestrator is not None,
            "langgraph_orchestrator": langgraph_orchestrator is not None,
            "vector_store": vector_store is not None,
        }

    @application.post("/api/evaluate_worker", tags=["LangGraph"])
    async def evaluate_worker(req: EvaluateWorkerRequest):
        if not langgraph_orchestrator:
            raise HTTPException(
                status_code=503,
                detail="LangGraph orchestrator offline. Set GROQ_API_KEY and restart.",
            )
        try:
            worker_dict = req.worker.model_dump()
            df_input = pd.DataFrame([worker_dict])
            target_city = req.city or req.worker.city
            result = await langgraph_orchestrator.run(
                worker_data=df_input,
                city=target_city,
                context_question=req.context_question,
            )
            out: Dict[str, Any] = {
                "trace_id": result.trace_id,
                "worker_id": result.worker_id,
                "processing_time_ms": result.processing_time_ms,
                "decision": result.final_state.get("decision_code"),
                "confidence": result.final_state.get("confidence"),
                "payout_amount": result.final_state.get("payout_amount"),
                "eligibility_snapshot": result.final_state.get("eligibility_snapshot"),
                "payout_breakdown": json_safe(result.final_state.get("payout_breakdown")),
            }
            if req.include_graph_state:
                out["final_state"] = json_safe(result.final_state)
            return out
        except Exception as e:
            logger.exception("evaluate_worker")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.post("/api/orchestrate", tags=["LangGraph"])
    async def orchestrate_batch(req: OrchestrateBatchRequest):
        if not langgraph_orchestrator:
            raise HTTPException(status_code=503, detail="LangGraph orchestrator offline.")
        if not req.workers:
            raise HTTPException(status_code=400, detail="workers list cannot be empty.")
        results: List[Dict[str, Any]] = []
        try:
            for worker in req.workers:
                worker_dict = worker.model_dump()
                df_input = pd.DataFrame([worker_dict])
                target_city = req.city or worker.city
                res = await langgraph_orchestrator.run(
                    worker_data=df_input,
                    city=target_city,
                    context_question=req.context_question,
                )
                item: Dict[str, Any] = {
                    "trace_id": res.trace_id,
                    "worker_id": res.worker_id,
                    "decision": res.final_state.get("decision_code"),
                    "confidence": res.final_state.get("confidence"),
                    "payout_amount": res.final_state.get("payout_amount"),
                    "processing_time_ms": res.processing_time_ms,
                }
                if req.include_graph_state:
                    item["final_state"] = json_safe(res.final_state)
                results.append(item)
            return {"batch_size": len(results), "results": results}
        except Exception as e:
            logger.exception("orchestrate_batch")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.post("/api/claims/process-classic", tags=["Classic pipeline"])
    async def process_claim_classic(req: ClassicClaimRequest):
        if not classic_orchestrator:
            raise HTTPException(
                status_code=503,
                detail="Classic orchestrator unavailable (check logs / RAG / models).",
            )
        try:
            worker_dict = req.worker.model_dump()
            df_input = pd.DataFrame([worker_dict])
            target_city = req.city or req.worker.city
            wf = await classic_orchestrator.process_claim(df_input, city=target_city)
            return _workflow_to_dict(wf)
        except Exception as e:
            logger.exception("process_claim_classic")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.post("/api/inference/predict", tags=["ML"])
    def inference_predict(req: InferencePredictRequest):
        if not inference_pipeline:
            raise HTTPException(status_code=503, detail="InferencePipeline not loaded (train models or fix GIGSHIELD_MODEL_DIR).")
        try:
            df_input = pd.DataFrame([req.worker.model_dump()])
            raw = inference_pipeline.predict_for_worker(df_input)
            return {"worker_id": req.worker.worker_id, "predictions": json_safe(raw)}
        except Exception as e:
            logger.exception("inference_predict")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.post("/api/rag/retrieve", tags=["RAG"])
    def rag_retrieve(req: RAGRetrieveRequest):
        try:
            from src.rag.rag_system import RAGRetriever, VectorStore

            vs = vector_store or VectorStore()
            rag = RAGRetriever(vs)
            ctx = rag.retrieve_context(query=req.query, categories=req.categories)
            results = ctx.get("results") or []
            if req.top_k is not None:
                k = max(1, int(req.top_k))
                results = results[:k]
                ctx["context_text"] = rag._format_context(results)
                ctx["num_results"] = len(results)
            return {
                "query": ctx.get("query"),
                "num_results": ctx.get("num_results"),
                "context_text": ctx.get("context_text"),
                "results": json_safe(results),
            }
        except Exception as e:
            logger.exception("rag_retrieve")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.post("/api/payment/create-order", tags=["Payment"])
    async def create_payment_order(amount: float, currency: str = "INR"):
        """
        Create a Razorpay order.
        """
        key_id = os.getenv("RAZORPAY_KEY_ID")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET")
        
        print(f"[Razorpay] Creating order for amount: {amount} {currency}")
        print(f"[Razorpay] Using Key ID: {key_id}")

        if not key_id or not key_secret:
            raise HTTPException(status_code=500, detail="Razorpay keys not configured on server. Check your .env file.")

        try:
            client = razorpay.Client(auth=(key_id, key_secret))
            data = {
                "amount": int(float(amount) * 100),  # amount in paise
                "currency": currency,
                "payment_capture": 1  # auto capture
            }
            order = client.order.create(data=data)
            print(f"[Razorpay] Order created: {order.get('id')}")
            return order
        except Exception as e:
            logger.error(f"Order creation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create Razorpay order: {str(e)}")

    @application.post("/api/payment/verify", tags=["Payment"])
    async def verify_payment(req: PaymentVerifyRequest):
        """
        Verify Razorpay payment signature.
        """
        key_id = os.getenv("RAZORPAY_KEY_ID")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET")
        
        if not key_id or not key_secret:
            raise HTTPException(status_code=500, detail="Razorpay keys not configured on server.")

        try:
            client = razorpay.Client(auth=(key_id, key_secret))
            
            # 1. Verify Signature
            params_dict = {
                'razorpay_order_id': req.razorpay_order_id,
                'razorpay_payment_id': req.razorpay_payment_id,
                'razorpay_signature': req.razorpay_signature
            }
            client.utility.verify_payment_signature(params_dict)
            
            logger.info(f"Payment verified successfully for worker {req.worker_id}")
            
            # 2. Update Database if Supabase is available
            if supabase:
                # Log transaction
                transaction_data = {
                    "worker_id": str(req.worker_id),
                    "razorpay_payment_id": req.razorpay_payment_id,
                    "razorpay_order_id": req.razorpay_order_id,
                    "amount": req.amount,
                    "status": "success"
                }
                supabase.table("payment_transactions").insert(transaction_data).execute()
                
                # Update worker coverage status
                # Assuming worker_id in DB is an integer or string matching req.worker_id
                # Removing any 'DB' prefix if present from frontend mapping (e.g. DB4 -> 4)
                numeric_worker_id = str(req.worker_id).replace("DB", "")
                
                supabase.table("gigshield_workers").update({"premium_paid": 1}).eq("worker_id", numeric_worker_id).execute()
                logger.info(f"Database updated for worker {numeric_worker_id}")

            return {
                "status": "success",
                "message": "Payment verified and record updated",
                "payment_id": req.razorpay_payment_id
            }
        except Exception as e:
            logger.error(f"Payment verification/update failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Payment verification failed: {str(e)}")

    @application.get("/api/payment/history/{worker_id}", tags=["Payment"])
    async def get_payment_history(worker_id: str):
        """
        Fetch successful payment transactions for a worker.
        """
        if not supabase:
            raise HTTPException(status_code=503, detail="Database connection offline.")
        
        try:
            # Handle both numeric and prefixed IDs
            search_id = str(worker_id)
            # We search for both '4' and 'DB4' to be safe
            base_id = search_id.replace("DB", "")
            prefixed_id = f"DB{base_id}"

            res = supabase.table("payment_transactions") \
                .select("*") \
                .or_(f"worker_id.eq.{base_id},worker_id.eq.{prefixed_id}") \
                .order("created_at", desc=True) \
                .execute()
            
            return res.data
        except Exception as e:
            logger.error(f"Failed to fetch payment history: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return application


app = create_app()
