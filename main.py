"""
GigShield ML Pipeline - Main Execution Script
Choose between training mode or inference mode
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load API keys from a `.env` file in the project root (create from `.env.example`)
load_dotenv(Path(__file__).resolve().parent / ".env")

sys.path.append('.')

from src.pipeline.training_pipeline import MLTrainingPipeline, InferencePipeline
from src.pipeline.orchestrator import GigShieldOrchestrator
from src.rag.rag_system import populate_knowledge_base, VectorStore
import pandas as pd


def train_models(data_path: str):
    """Train all ML models"""
    print("\n🚀 Starting GigShield Model Training...")
    
    pipeline = MLTrainingPipeline(data_path)
    models = pipeline.run_complete_training()
    
    print("\n✅ Training Complete!")
    print("Models saved to: models/")
    
    return models


def run_inference(worker_data_path: str, model_dir: str = 'models'):
    """Run inference on worker data"""
    print("\n🚀 Starting GigShield Inference Pipeline...")
    
    # Load worker data
    worker_df = pd.read_csv(worker_data_path)
    print(f"Loaded {len(worker_df)} worker records")
    
    # Load models
    model_paths = {
        'income_forecasting': f'{model_dir}/income_forecasting',
        'risk_scoring': f'{model_dir}/risk_scoring',
        'fraud_detection': f'{model_dir}/fraud_detection',
        'disruption_impact': f'{model_dir}/disruption_impact',
        'behavior_analysis': f'{model_dir}/behavior_analysis',
        'premium_prediction': f'{model_dir}/premium_prediction'
    }
    
    inference_pipeline = InferencePipeline(model_paths)
    
    # Run predictions
    for idx, row in worker_df.head(5).iterrows():
        worker_sample = pd.DataFrame([row])
        predictions = inference_pipeline.predict_for_worker(worker_sample)
        
        print(f"\nWorker {row['worker_id']} Predictions:")
        print(f"  Income Forecast: ₹{predictions['income_forecast']['ensemble']:.2f}")
        print(f"  Risk Score: {predictions['risk_score']['risk_score'].iloc[0]:.3f}")
        print(f"  Fraud Score: {predictions['fraud_analysis']['fraud_probability'].iloc[0]:.3f}")
        print(f"  Premium: ₹{predictions['premium']['final_premium'].iloc[0]:.2f}")
        dip = predictions.get("disruption_impact")
        if hasattr(dip, "iloc"):
            print(f"  Disruption loss % (pred): {float(dip.iloc[0]):.3f}")
        else:
            print(f"  Disruption loss % (pred): {np.asarray(dip).ravel()[0]:.3f}")
    
    print("\n✅ Inference Complete!")


async def run_autonomous_system(worker_data_path: str, model_dir: str = "models"):
    """Run complete autonomous insurance system"""
    print("\n🚀 Starting GigShield Autonomous System...")

    print("\nInitializing Knowledge Base...")
    vector_store = VectorStore()
    populate_knowledge_base(vector_store)

    worker_df = pd.read_csv(worker_data_path)

    model_paths = {
        "income_forecasting": f"{model_dir}/income_forecasting",
        "risk_scoring": f"{model_dir}/risk_scoring",
        "fraud_detection": f"{model_dir}/fraud_detection",
        "disruption_impact": f"{model_dir}/disruption_impact",
        "behavior_analysis": f"{model_dir}/behavior_analysis",
        "premium_prediction": f"{model_dir}/premium_prediction",
    }
    inference = InferencePipeline(model_paths)
    orchestrator = GigShieldOrchestrator(inference_pipeline=inference, vector_store=vector_store)
    
    # Process claims
    sample_workers = worker_df.head(3)
    results = []
    
    for idx, row in sample_workers.iterrows():
        worker_sample = pd.DataFrame([row])
        result = await orchestrator.process_claim(worker_sample)
        results.append(result)
    
    # Generate report
    report = orchestrator.generate_report(results)
    
    # Save report
    report.to_csv('data/outputs/claim_processing_report.csv', index=False)
    print("\n✅ Report saved to: data/outputs/claim_processing_report.csv")


async def run_autonomous_langchain_system(worker_data_path: str, model_dir: str = "models"):
    """Run RAG + LangChain orchestrator (requires GROQ_API_KEY for LLM chains; RAG still works)."""
    from src.agents.langchain_orchestrator import GigShieldLangChainOrchestrator

    print("\n🚀 GigShield LangChain + RAG pipeline...")
    from src.utils.schema import ensure_worker_columns

    worker_df = ensure_worker_columns(pd.read_csv(worker_data_path))
    model_paths = {
        "income_forecasting": f"{model_dir}/income_forecasting",
        "risk_scoring": f"{model_dir}/risk_scoring",
        "fraud_detection": f"{model_dir}/fraud_detection",
        "disruption_impact": f"{model_dir}/disruption_impact",
        "behavior_analysis": f"{model_dir}/behavior_analysis",
        "premium_prediction": f"{model_dir}/premium_prediction",
    }
    inference = InferencePipeline(model_paths)
    orch = GigShieldLangChainOrchestrator(inference_pipeline=inference, ensure_kb=True)

    sample = worker_df.head(2)
    results = []
    for _, row in sample.iterrows():
        worker_sample = pd.DataFrame([row])
        res = await orch.run_full_workflow(worker_sample)
        results.append(res)
        print(f"\n trace={res.trace_id} worker={res.worker_id} decision={res.decision} payout={res.payout_amount}")
        if res.rag_answers:
            print("  RAG/LLM keys:", list(res.rag_answers.keys()))

    out = Path("data/outputs/langchain_workflow_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "trace_id": r.trace_id,
                "worker_id": r.worker_id,
                "decision": r.decision,
                "confidence": r.confidence,
                "payout_amount": r.payout_amount,
                "ms": r.processing_time_ms,
            }
            for r in results
        ]
    ).to_csv(out, index=False)
    print(f"\n✅ LangChain workflow report: {out}")


async def run_autonomous_langgraph_system(worker_data_path: str, model_dir: str = "models"):
    """LangGraph + tool-calling agents + RAG tools + Supabase/SQLite persistence (requires GROQ_API_KEY)."""
    from src.agents.gigshield_langgraph import GigShieldLangGraphOrchestrator
    from src.utils.schema import ensure_worker_columns

    print("\n🚀 GigShield LangGraph multi-agent (tool-calling) pipeline...")
    worker_df = ensure_worker_columns(pd.read_csv(worker_data_path))
    model_paths = {
        "income_forecasting": f"{model_dir}/income_forecasting",
        "risk_scoring": f"{model_dir}/risk_scoring",
        "fraud_detection": f"{model_dir}/fraud_detection",
        "disruption_impact": f"{model_dir}/disruption_impact",
        "behavior_analysis": f"{model_dir}/behavior_analysis",
        "premium_prediction": f"{model_dir}/premium_prediction",
    }
    inference = InferencePipeline(model_paths)
    orch = GigShieldLangGraphOrchestrator(inference_pipeline=inference, ensure_kb=True)

    sample = worker_df.head(1)
    results = []
    for _, row in sample.iterrows():
        res = await orch.run(pd.DataFrame([row]))
        st = res.final_state
        results.append(res)
        print(
            f"\n trace={res.trace_id} worker={res.worker_id} "
            f"decision={st.get('decision_code')} payout={st.get('payout_amount')} ({res.processing_time_ms:.0f} ms)"
        )

    out = Path("data/outputs/langgraph_workflow_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "trace_id": r.trace_id,
                "worker_id": r.worker_id,
                "decision": r.final_state.get("decision_code"),
                "confidence": r.final_state.get("confidence"),
                "payout_amount": r.final_state.get("payout_amount"),
                "ms": r.processing_time_ms,
            }
            for r in results
        ]
    ).to_csv(out, index=False)
    print(f"\n✅ LangGraph workflow report: {out}")


def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'data/outputs',
        'models/income_forecasting',
        'models/risk_scoring',
        'models/fraud_detection',
        'models/disruption_impact',
        'models/behavior_analysis',
        'models/premium_prediction',
        'logs',
        'vector_store/chromadb'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")


def main():
    parser = argparse.ArgumentParser(description='GigShield ML Pipeline')
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'inference', 'autonomous', 'autonomous-lc', 'autonomous-lc-graph', 'populate-datastores', 'setup'],
        help='Execution mode: train, inference, autonomous, autonomous-lc, autonomous-lc-graph, populate-datastores, or setup'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/final_dataset.csv',
        help='Path to worker data CSV file'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models'
    )

    parser.add_argument(
        '--populate-limit',
        type=int,
        default=None,
        help='For populate-datastores: max CSV rows to load',
    )
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    if args.mode == 'setup':
        print("✅ Setup complete! Directory structure created.")
        print("\nNext steps:")
        print("1. Place your training data in: data/raw/final_dataset.csv (or pass --data)")
        print("2. Run training: python main.py --mode train")
        print("3. Run inference: python main.py --mode inference")
        print("4. Run autonomous system: python main.py --mode autonomous")
        print("5. RAG + LangChain: python main.py --mode autonomous-lc")
        print("6. LangGraph tool agents: python main.py --mode autonomous-lc-graph")
        print("7. Populate Supabase + Pinecone: python main.py --mode populate-datastores --data path/to.csv")
        
    elif args.mode == 'train':
        train_models(args.data)
        
    elif args.mode == 'inference':
        run_inference(args.data, args.model_dir)
        
    elif args.mode == 'autonomous':
        asyncio.run(run_autonomous_system(args.data, model_dir=args.model_dir))

    elif args.mode == 'autonomous-lc':
        asyncio.run(run_autonomous_langchain_system(args.data, model_dir=args.model_dir))

    elif args.mode == 'autonomous-lc-graph':
        asyncio.run(run_autonomous_langgraph_system(args.data, model_dir=args.model_dir))

    elif args.mode == 'populate-datastores':
        from src.pipeline.populate_data_stores import run_populate

        csv_path = Path(args.data)
        if not csv_path.is_file():
            alt = Path("data/raw/quick_commerce_synthetic_data52k.csv")
            if alt.is_file():
                print(f"Using {alt} (not found: {csv_path})")
                csv_path = alt
            else:
                raise SystemExit(f"CSV not found: {csv_path}")
        run_populate(
            csv_path=csv_path,
            limit=args.populate_limit,
            do_supabase=True,
            do_pinecone_kb=True,
            do_worker_vectors=False,
            worker_vector_limit=2000,
        )


if __name__ == "__main__":
    main()