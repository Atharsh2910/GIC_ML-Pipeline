# 🚀 Gig Insurance Company — Quick Start Guide

> **Gig Insurance Company** is an AI-powered parametric insurance platform for gig workers. It uses ML models, multi-agent orchestration (LangChain + LangGraph), RAG-augmented decision-making, and MCP-integrated external signals to automatically detect weather/disruption events and trigger payouts — no paperwork required.

---

## 📋 Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| **Python** | 3.10 – 3.13 | `python --version` |
| **Node.js** | 18+ | `node --version` |
| **npm** | 9+ | `npm --version` |
| **Git** | Any | `git --version` |

### Required API Keys

| Key | Required For | Get It From |
|-----|-------------|-------------|
| `GROQ_API_KEY` | LLM agents (LangGraph / LangChain orchestrator) | [console.groq.com](https://console.groq.com) |
| `PINECONE_API_KEY` | Vector store (production RAG) | [app.pinecone.io](https://app.pinecone.io) |
| `PINECONE_HOST` | Pinecone serverless index URL | Pinecone Console → Index → Host |
| `HF_TOKEN` | Embedding model download | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `VITE_SUPABASE_URL` | Frontend auth & persistence | [supabase.com](https://supabase.com) → Project → Settings → API |
| `VITE_SUPABASE_ANON_KEY` | Frontend auth & persistence | Supabase → Project → Settings → API |

> **Minimum viable demo**: The classic orchestrator works **without** `GROQ_API_KEY`. LangGraph/LangChain agents require it. ChromaDB is used as a local vector store fallback if Pinecone is not configured.

---

## ⚡ Setup (One-Time)

### 1. Clone the Repository

```bash
git clone https://github.com/Status-Code-401/GIG_INSURANCE_COMPANY.git
cd GIG_INSURANCE_COMPANY
```

### 2. Backend — Python Environment

```bash
# Create virtual environment
python -m venv src/venv

# Activate it
# Windows (PowerShell):
.\src\venv\Scripts\activate
# macOS/Linux:
source src/venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### 3. Backend — Environment Variables

Create a `.env` file in the **project root** (`GIC_ML-Pipeline/.env`):

```env
# ─── LLM (powers LangGraph + LangChain agent reasoning) ───
GROQ_API_KEY="your_groq_api_key_here"

# ─── Vector Store (RAG retrieval) ───
VECTOR_STORE_PROVIDER="pinecone"          # or "chromadb" for local-only
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_HOST="your_pinecone_host_url"
PINECONE_INDEX_NAME="your_index_name"

# ─── Embeddings ───
HF_TOKEN="your_huggingface_token_here"

# ─── Supabase Backend Persistence (optional) ───
# SUPABASE_URL="your_supabase_url"
# SUPABASE_SERVICE_ROLE_KEY="your_service_role_key"

# ─── MCP Mock Layer ───
# GIGSHIELD_USE_MOCK_MCP="true"           # default: true
# GIGSHIELD_MOCK_SCENARIO="heavy_rain"    # heavy_rain | cyclone | heat | strike | clear
# GIGSHIELD_MOCK_API_BASE=""              # set to http://127.0.0.1:8000 to use HTTP mock APIs
```

### 4. Frontend — Install Dependencies

```bash
cd frontend
npm install
```

### 5. Frontend — Environment Variables

Create a `.env` file inside the `frontend/` folder (`frontend/.env`):

```env
VITE_SUPABASE_URL="your_supabase_project_url"
VITE_SUPABASE_ANON_KEY="your_supabase_anon_key"
```

> **Get Supabase keys from:** https://supabase.com → Project → Settings → API

---

## 🏃 Running the App

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Start Backend (FastAPI)

```bash
# From the project root: GIC_ML-Pipeline/
.\src\venv\Scripts\activate          # Windows
# source src/venv/bin/activate       # macOS/Linux

uvicorn app:app --host 0.0.0.0 --port 8000
```

Wait for the models to load (you'll see "✓ All models loaded" and "Uvicorn running on http://0.0.0.0:8000").  
**First run takes 1-2 minutes** as it downloads embedding models.

On startup the API initializes in this order:
1. **InferencePipeline** — loads 6 trained ML model artifacts from `models/`
2. **VectorStore** — connects to Pinecone (or creates local ChromaDB) and populates the knowledge base
3. **Classic GigShieldOrchestrator** — ready immediately (no LLM needed)
4. **LangGraph Orchestrator** — requires `GROQ_API_KEY`; skipped if key is missing

### Terminal 2 — Start Frontend (Vite + React)

```bash
# From the frontend folder: GIC_ML-Pipeline/frontend/
npm run dev
```

The frontend runs at: **http://localhost:5173**

---

## 🔑 Login Credential

### Registered Accounts

You can also register a new worker via the **Register** page. These are stored in Supabase.

---

## 🌧️ Running the Disruption Simulation

This is the core demo — simulating a rainfall event and watching the ML pipeline process a claim.

### Steps:

1. **Login** with `Z001` / `demo`
2. **Scroll down** on the Dashboard and click **"Run Simulation"**
3. **Select scenario:**
   - 🌧️ **Heavy Rainfall / Flood** — simulates 150-250mm rain, income drops to 45%
   - 🌀 **Cyclone Alert** — simulates 200-300mm rain + high winds, income drops to 30%
   - ☀️ **Normal Conditions** — no disruption, claim won't trigger
4. **Click "Run Simulation"** and watch the pipeline:

```
MCP Layer → Weather API → ML Inference → Agent Analysis → Eligibility Check → Decision + Payout
```

### Expected Result (Flood scenario):

| Field | Value |
|-------|-------|
| Decision | ✅ APPROVE |
| Payout | ~₹1,818 |
| Confidence | ~88% |
| Coverable Gap | ₹2,250 (30% income loss) |

---

## 🏗️ System Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL DATA SOURCES                        │
│  Weather APIs  ·  News Feeds  ·  Telecom Status  ·  Platform Data   │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│   MCP (Model Context     │   MockMCPClient / HttpMockApiMCPClient
│   Protocol) Layer        │   Aggregates weather, news, telecom, platform
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT ORCHESTRATION                       │
│                                                                     │
│   ┌─────────────┐    ┌────────────────┐    ┌───────────────┐        │
│   │  Monitor    │───▶│  Validation    │───▶│   Context     │        │
│   │  Agent      │    │  Agent         │    │   Agent (RAG) │        │
│   └─────────────┘    └────────────────┘    └───────┬───────┘        │
│                                                    │                │
│                                         ┌──────────┼──────────┐     │
│                                         ▼          ▼          ▼     │
│                                   ┌──────────┐ ┌────────┐ ┌──────┐  │
│                                   │  Fraud   │ │  Risk  │ │Rules │  │
│                                   │  Agent   │ │ Agent  │ │Agent │  │
│                                   └────┬─────┘ └───┬────┘ └──┬───┘  │
│                                        └───────────┼─────────┘      │
│                                                    ▼                │
│                                           ┌────────────────┐        │
│                                           │ Decision Agent │        │
│                                           └───────┬────────┘        │
│                                                   │                 │
│                                                   ▼                 │
│                                        ┌─────────────────────┐      │
│                                        │  Deterministic      │      │
│                                        │  Eligibility +      │      │
│                                        │  Payout Calculator  │      │
│                                        └──────────┬──────────┘      │
│                                                   │                 │
│                                                   ▼                 │
│                                        ┌──────────────────┐         │
│                                        │  SQL / Supabase  │         │
│                                        │  Persistence     │         │
│                                        └──────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│    6 ML Models (Parallel)  ·  RAG Vector Store  ·  Knowledge Base   │
│    Income · Risk · Fraud · Disruption · Behavior · Premium          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Multi-Agent System

GigShield implements **8 specialized agents** that communicate via structured `AgentMessage` objects with trace IDs, priorities, and typed payloads.

### Agent Roster

| # | Agent | Role | Key Capabilities |
|---|-------|------|-------------------|
| 1 | **MonitorAgent** | Layer 1 — Trigger detection | Fetches weather/regional signals via MCP; checks rainfall, temperature, cyclone thresholds |
| 2 | **ValidationAgent** | Layer 2 — Cross-validation | Validates triggers meet minimum criteria (affected workers ≥ 5, duration ≥ 2 hrs, geo-consistency) |
| 3 | **ContextAgent** | Layer 3 — RAG enrichment | Retrieves policy, historical claims, and regional data from vector store; optional LLM summarization |
| 4 | **FraudDetectionAgent** | Layer 4 — Fraud scoring | Weighted rule engine (GPS spoofing, movement anomaly, peer deviation, device sharing) + ML model fusion |
| 5 | **RiskScoringAgent** | Layer 4 — Risk assessment | Wraps trained XGBoost/LightGBM risk model; heuristic fallback |
| 6 | **RuleValidationAgent** | Layer 4 — Business rules | Checks cooling period, premium payment status, employment type eligibility |
| 7 | **DecisionAgent** | Layer 5 — Final verdict | Aggregates fraud/risk/rules signals → `auto_approve`, `auto_reject`, or `manual_review` |
| 8 | **SQLAgent** | Layer 6 — Persistence | Writes decisions to Supabase (primary) + SQLite + JSONL audit log |

### Agent Communication Protocol

```python
@dataclass
class AgentMessage:
    agent_name: str          # e.g. "MonitorAgent"
    timestamp: datetime
    message_type: str        # e.g. "trigger_detected", "fraud_analysis_complete"
    data: Dict[str, Any]     # structured payload
    trace_id: str            # end-to-end correlation
    priority: str            # "critical" | "high" | "medium" | "low"
```

Agents in **Layer 4** (Fraud, Risk, Rules) run in **parallel** via `asyncio.gather()`, then their outputs are fanned-in to the Decision Agent.

---

## 🎭 Three Orchestration Modes

GigShield offers **three** progressively-richer orchestration strategies:

### 1. Classic Orchestrator (`GigShieldOrchestrator`)
**File:** `src/pipeline/orchestrator.py`  
**API:** `POST /api/claims/process-classic`  
**Requires:** No LLM — works out of the box

A 5-layer deterministic pipeline:
```
Layer 1: Monitor (MCP signals) → Layer 2: Validation (threshold checks)
→ Layer 3: Context (RAG retrieval) → Layer 4: Parallel agents (Fraud + Risk + Rules) + ML inference
→ Layer 5: Decision → Deterministic eligibility + payout
```

### 2. LangChain Orchestrator (`GigShieldLangChainOrchestrator`)
**File:** `src/agents/langchain_orchestrator.py`  
**Requires:** `GROQ_API_KEY` for LLM chains (RAG QA, fraud reasoning, decision)

Adds **LangChain LCEL chains** on top of the classic flow:
- **RAG QA Chain** — grounded policy question-answering over vector store
- **Fraud Reasoning Chain** — LLM interprets fraud signals with RAG context
- **Decision Chain** — LLM synthesizes agent outputs into structured JSON verdict
- All chains are optional; if LLM is unavailable, the classic agent logic still runs

### 3. LangGraph Orchestrator (`GigShieldLangGraphOrchestrator`)
**File:** `src/agents/gigshield_langgraph.py`  
**API:** `POST /api/evaluate_worker` · `POST /api/orchestrate`  
**Requires:** `GROQ_API_KEY`

The most advanced mode — a **LangGraph `StateGraph`** with 8 nodes:

```
START → monitor → validation → context → ml_core → specialists → decision → deterministic → persist → END
```

Each node is a **tool-calling agent** (via `create_tool_calling_agent` + `AgentExecutor`) with access to domain-specific LangChain tools. The LLM autonomously decides which tools to call at each step.

#### LangGraph Node Details

| Node | Agent Executor | Tools Available |
|------|---------------|-----------------|
| `monitor` | MonitorAgent executor | `fetch_live_disruption_signals`, `retrieve_disruption_knowledge` |
| `validation` | ValidationAgent executor | `retrieve_disruption_knowledge`, `retrieve_policy_knowledge`, `record_structured_observation` |
| `context` | ContextAgent executor | `retrieve_policy_knowledge`, `retrieve_disruption_knowledge`, `retrieve_fraud_playbooks` |
| `ml_core` | Deterministic | Parallel core agents (Fraud/Risk/Rules) + InferencePipeline |
| `specialists` | 3x parallel executors | Fraud: `retrieve_fraud_playbooks` · Risk: `retrieve_policy_knowledge` · Rules: `retrieve_policy_knowledge` |
| `decision` | DecisionAgent executor | `retrieve_policy_knowledge`, `retrieve_fraud_playbooks`, `persist_underwriter_decision_stub` |
| `deterministic` | Rule-based | `ClaimEligibilityModel`, `PayoutOptimizationModel` |
| `persist` | SQLAgent | Supabase / SQLite write |

#### State Schema

```python
class GigShieldGraphState(TypedDict):
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
    decision_code: str          # auto_approve | auto_reject | manual_review
    confidence: float
    payout_amount: float
    eligibility_snapshot: Dict[str, Any]
    payout_breakdown: Dict[str, Any]
    errors: List[str]
```

---

## 🔧 LangChain Tools

All tools are defined in `src/agents/gigshield_tools.py` and bound to agents via `build_gigshield_toolkit()`.

| Tool | Description |
|------|-------------|
| `fetch_live_disruption_signals(city)` | Fetches near-real-time weather via Open-Meteo API (no key needed) |
| `retrieve_disruption_knowledge(query)` | RAG retrieval over disruption events & regional data |
| `retrieve_policy_knowledge(query)` | RAG retrieval over insurance policies, thresholds, slab rules |
| `retrieve_fraud_playbooks(query)` | RAG retrieval over fraud patterns (GPS spoofing, coordinated rings) |
| `record_structured_observation(...)` | Persists agent observations to Supabase `gigshield_agent_events` |
| `persist_underwriter_decision_stub(...)` | Logs decision intent for human approval workflows |

---

## 📚 RAG System

**Files:** `src/rag/rag_system.py` · `src/rag/langchain_rag.py` · `src/rag/knowledge_bundle.py`

### Vector Store

- **Primary:** Pinecone (serverless, set via `PINECONE_API_KEY` + `PINECONE_HOST`)
- **Fallback:** ChromaDB (local persistent store at `chroma_data/`)
- **Embedding Models:** Auto-selected to match Pinecone index dimension:

| Index Dimension | Embedding Model |
|-----------------|-----------------|
| 1024 | `intfloat/e5-large-v2` |
| 768 | `sentence-transformers/all-mpnet-base-v2` |
| 384 (default) | `sentence-transformers/all-MiniLM-L6-v2` |

### Knowledge Base Categories

| Category | Contents |
|----------|----------|
| `insurance_policies` | 75% threshold rule, 8-week cooling period, slab coverage caps, proof requirements |
| `fraud_cases` | GPS spoofing patterns, coordinated fraud clusters, adverse selection flags |
| `disruption_events` | Parametric rain bands (5-20cm+), cyclone scaling, extreme heat triggers |
| `historical_claims` | Example payout calculations, penalty/defaulter rules |
| `regional_data` | City-specific patterns (Mumbai monsoon, Delhi heat waves) |

### LangChain LCEL Chains

| Chain | Input | Output | Used By |
|-------|-------|--------|---------|
| `build_rag_qa_chain` | User question (string) | Grounded answer from policy docs | LangChain Orchestrator |
| `build_fraud_reasoning_chain` | `{query, worker_json}` | `{rationale, escalate_review}` JSON | Fraud reasoning layer |
| `build_decision_chain` | Agent packet (dict/string) | `{decision, confidence, rationale}` JSON | Final LLM decision synthesis |

---

## 🔗 MCP (Model Context Protocol) Integration

**Files:** `src/integrations/mock_mcp_client.py` · `src/integrations/http_mock_api_client.py`

The MCP layer sits between external data sources and the MonitorAgent, aggregating signals into a unified bundle:

```python
{
    "weather":  {"city": "Chennai", "rainfall_cm": 16.0, "temperature": 28.0, "alerts": [...]},
    "regional": {"affected_workers_estimate": 180, "disruption_duration_hours": 9.0, "inventory_stress_index": 0.82},
    "platform_pulse": {"worker_id": 1, "orders_completed_proxy": 50, "source": "mock_platform_api"}
}
```

### Two MCP Client Implementations

| Client | When Used | Data Source |
|--------|-----------|-------------|
| `MockMCPClient` | Default (`GIGSHIELD_USE_MOCK_MCP=true`) | In-memory scenario simulation (heavy_rain, cyclone, heat, strike, clear) |
| `HttpMockApiMCPClient` | When `GIGSHIELD_MOCK_API_BASE` is set | Calls the FastAPI mock endpoints (`/api/weather`, `/api/news`, `/api/telecom`, `/api/platform`) via HTTP |

### Mock Scenarios

| Scenario | Rainfall | Temperature | Alerts | Affected Workers |
|----------|----------|-------------|--------|-------------------|
| `heavy_rain` | 16 cm | 28°C | Heavy_Rain (high) | ~180 |
| `cyclone` | 8 cm | 28°C | Cyclone (critical, level 4) | ~400 |
| `heat` | 0 cm | 46°C | Extreme_Heat (high) | ~90 |
| `strike` | 0 cm | 28°C | Labor_Action (medium) | ~220 |
| `clear` | 0 cm | 28°C | None | ~12 |

---

## 🧠 ML Models

The pipeline runs **6 ML models** in parallel:

| Model | Algorithm | Purpose |
|-------|-----------|---------| 
| **Income Forecasting** | Rolling Mean + LSTM + SARIMAX | Predict expected weekly income |
| **Risk Scoring** | XGBoost + LightGBM | Overall risk score for the worker |
| **Fraud Detection** | Isolation Forest + XGBoost | Detect fraudulent claims |
| **Disruption Impact** | XGBoost + LightGBM | Measure disruption severity |
| **Behavior Analysis** | XGBoost + LightGBM | Worker behavior score |
| **Premium Prediction** | Rule-based + ML adjustment | Calculate optimal premium |

### Deterministic Models

In addition to the trained ML models, two rule-based models handle the final eligibility and payout:

| Model | Purpose |
|-------|---------|
| **ClaimEligibilityModel** | Evaluates: income below 75% of 52-week average, cooling period, premium status, disruption proof |
| **PayoutOptimizationModel** | Calculates payout: coverable income gap × disruption factor × slab coverage × loyalty/penalty adjustments |

---

## 💾 Persistence Layer

**File:** `src/persistence/supabase_client.py` · `src/agents/sql_store.py`

### Dual-Write Strategy

| Target | When | Tables |
|--------|------|--------|
| **Supabase** (primary) | `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` set | `gigshield_decisions`, `gigshield_rag_queries`, `gigshield_agent_events`, `gigshield_workers` |
| **SQLite** (fallback) | Supabase not configured | `data/gigshield_agents.db` → `agent_decisions`, `rag_queries` |
| **JSONL** (audit) | Always | `logs/agent_decisions.jsonl` |

Every decision is written to **all available** stores for auditability.

---

## 🏗️ Project Structure

```
GIC_ML-Pipeline/
├── app.py                    # FastAPI entrypoint (run with uvicorn)
├── main.py                   # ML pipeline runner (training, inference)
├── requirements.txt          # Python dependencies
├── .env                      # Backend environment variables
│
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI routes, lifespan, CORS
│   │   └── schemas.py        # Pydantic request/response models
│   │
│   ├── agents/               # 🤖 Multi-Agent System
│   │   ├── core_agents.py    # 8 agent classes (Monitor→SQL) + AgentMessage protocol
│   │   ├── gigshield_langgraph.py  # LangGraph StateGraph orchestrator (8-node DAG)
│   │   ├── gigshield_tools.py      # LangChain @tool definitions (RAG, weather, persist)
│   │   ├── langchain_orchestrator.py  # LangChain LCEL orchestrator (RAG chains + agents)
│   │   └── sql_store.py      # SQLite persistence backend
│   │
│   ├── pipeline/             # 🔄 Orchestration
│   │   ├── orchestrator.py   # Classic 5-layer orchestrator (no LLM required)
│   │   ├── training_pipeline.py    # Model training & InferencePipeline
│   │   ├── populate_data_stores.py # Bulk data ingestion utilities
│   │   └── populate_pinecone_csv.py  # CSV → Pinecone vector uploader
│   │
│   ├── models/               # 🧠 ML Models
│   │   ├── deterministic_models.py  # ClaimEligibility + PayoutOptimization
│   │   ├── income_forecasting.py    # Income prediction (Rolling Mean/LSTM/SARIMAX)
│   │   ├── risk_scoring.py          # Risk scoring (XGBoost/LightGBM)
│   │   ├── fraud_detection.py       # Fraud detection (IsolationForest/XGBoost)
│   │   └── additional_models.py     # Disruption, behavior, premium models
│   │
│   ├── rag/                  # 📚 RAG System
│   │   ├── rag_system.py     # VectorStore (Chroma/Pinecone), EmbeddingGenerator, RAGRetriever
│   │   ├── langchain_rag.py  # LCEL chains (QA, Fraud Reasoning, Decision)
│   │   └── knowledge_bundle.py  # Curated knowledge chunks (5 categories)
│   │
│   ├── integrations/         # 🔗 MCP & External APIs
│   │   ├── mock_mcp_client.py       # MockMCPClient (in-memory scenario sim)
│   │   └── http_mock_api_client.py  # HttpMockApiMCPClient (calls mock_api over HTTP)
│   │
│   ├── persistence/          # 💾 Data Layer
│   │   ├── supabase_client.py  # Supabase + SQLite dual-write
│   │   └── __init__.py
│   │
│   └── utils/                # 🛠 Utilities
│       └── schema.py         # DataFrame column normalization
│
├── config/
│   ├── agent_config.py       # Agent system, LangChain, vector store, MCP configs
│   ├── model_config.py       # ML hyperparameters & thresholds
│   ├── data_config.py        # Data pipeline settings
│   ├── dataset_schema.py     # Data schema definitions
│   ├── supabase_config.py    # Supabase table names & env var keys
│   └── artifacts.py          # Model artifact paths
│
├── mock_api/
│   └── mock_api.py           # FastAPI app: weather, news, telecom, fuel, platform mock APIs
│
├── frontend/
│   ├── .env                  # Frontend environment variables
│   ├── package.json          # Node.js dependencies
│   └── src/
│       ├── App.jsx           # Main app (Dashboard, Login, Plans, Claims)
│       ├── SimulationPage.jsx # Disruption simulation engine
│       ├── OnboardingPage.jsx # Worker registration
│       ├── supabaseClient.js  # Supabase connection
│       └── index.css         # All styles
│
├── models/                   # Trained model artifacts (.pkl files)
├── data/                     # SQLite DB + generated data
├── supabase/                 # Migration SQL files
└── scripts/                  # Utility scripts
```

---

## 🔌 API Endpoints

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info and available endpoints |
| `/health` | GET | Health check (models, orchestrators, vector store status) |
| `/docs` | GET | Swagger UI (auto-generated API docs) |

### LangGraph (requires `GROQ_API_KEY`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evaluate_worker` | POST | Full LangGraph 8-node agent evaluation for a single worker |
| `/api/orchestrate` | POST | Batch LangGraph evaluation for multiple workers |

### Classic Pipeline

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/claims/process-classic` | POST | Classic 5-layer orchestrator: MCP → agents → eligibility + payout |

### ML & RAG

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inference/predict` | POST | Run all 6 ML models on a worker (no orchestration) |
| `/api/rag/retrieve` | POST | Direct RAG retrieval from vector store |

### Partner Mock APIs (mounted at `/partner-mock`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/partner-mock/api/weather` | GET | Mock weather data (`?city=Chennai&scenario=flood`) |
| `/partner-mock/api/news` | GET | Mock news articles (`?scenario=strike`) |
| `/partner-mock/api/telecom` | GET | Mock telecom outage data (`?scenario=cyclone`) |
| `/partner-mock/api/fuel` | GET | Mock fuel supply data |
| `/partner-mock/api/platform` | GET | Mock platform status data |
| `/partner-mock/api/claims/rollout` | POST | Mock claim rollout acknowledgement |

---

## 🧪 Testing the Agent Pipelines

### 1. Classic Pipeline (No LLM needed)

```bash
curl -X POST http://localhost:8000/api/claims/process-classic \
  -H "Content-Type: application/json" \
  -d '{
    "worker": {
      "worker_id": 1,
      "city": "Chennai",
      "avg_52week_income": 7500,
      "weekly_income": 3500,
      "income_loss_percentage": 35,
      "disruption_type": "Heavy_Rain",
      "rainfall_cm": 15,
      "selected_slab": "Slab_100",
      "premium_paid": 1,
      "cooling_period_completed": 1
    },
    "city": "Chennai"
  }'
```

### 2. LangGraph Pipeline (Requires `GROQ_API_KEY`)

```bash
curl -X POST http://localhost:8000/api/evaluate_worker \
  -H "Content-Type: application/json" \
  -d '{
    "worker": {
      "worker_id": 1,
      "city": "Chennai",
      "avg_52week_income": 7500,
      "weekly_income": 3500,
      "income_loss_percentage": 35,
      "disruption_type": "Heavy_Rain",
      "rainfall_cm": 15,
      "selected_slab": "Slab_100",
      "premium_paid": 1,
      "cooling_period_completed": 1
    },
    "city": "Chennai",
    "include_graph_state": true
  }'
```

### 3. Direct RAG Query

```bash
curl -X POST http://localhost:8000/api/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the 75 percent income threshold rule?", "categories": ["insurance_policies"]}'
```

### 4. Raw ML Inference

```bash
curl -X POST http://localhost:8000/api/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"worker": {"worker_id": 1, "city": "Mumbai", "avg_52week_income": 7500, "weekly_income": 3500}}'
```

---

## ❓ Troubleshooting

### Backend won't start
- Make sure you're in the **project root**, not inside `src/`
- Check that `.env` file has valid API keys
- Run `pip install -r requirements.txt` again

### LangGraph orchestrator offline
- Ensure `GROQ_API_KEY` is set in `.env` — the LangGraph mode **requires** an LLM
- The classic orchestrator (`/api/claims/process-classic`) still works without it
- Check console for: `[GigShield API] Warning: LangGraph offline (check GROQ_API_KEY)`

### Pinecone dimension mismatch
- Set `PINECONE_INDEX_NAME` to your **exact** index name from the Pinecone console
- Or explicitly set `PINECONE_INDEX_DIMENSION=1024` and `GIGSHIELD_EMBEDDING_MODEL=intfloat/e5-large-v2`
- Switch to ChromaDB by setting `VECTOR_STORE_PROVIDER=chromadb` (no Pinecone needed)

### Frontend shows "Simulation Error"
- Backend must be running on port **8000**
- Check the terminal for backend errors
- First startup takes 1-2 min to download models

### Claim always rejected
- Make sure you're using **Flood** or **Cyclone** scenario (Normal won't trigger claims)
- The simulation automatically drops weekly income below the 75% threshold

### Port already in use
```bash
# Windows — kill process on port 8000:
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Then restart:
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📝 Notes

- **First startup** downloads the `intfloat/e5-large-v2` embedding model (~1.3 GB). This is cached for future runs.
- The **LangGraph endpoint** (`/api/evaluate_worker`) requires a valid `GROQ_API_KEY`. The classic orchestrator works without it.
- **Supabase** is only needed for the Register/Onboarding flow. The 3 demo accounts (Z001-Z003) work without it.
- The **LLM** is `llama-3.3-70b-versatile` via Groq by default. Configurable in `config/agent_config.py` → `LANGCHAIN_CONFIG`.
- **Mock MCP** is enabled by default. Set `GIGSHIELD_USE_MOCK_MCP=false` to disable and use only worker-row signals.
- All agent decisions are audited in `logs/agent_decisions.jsonl` regardless of Supabase configuration.
