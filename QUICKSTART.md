# 🚀 GigShield — Quick Start Guide

> **GigShield** is an AI-powered parametric insurance platform for gig workers. It uses ML models to automatically detect weather disruptions (rainfall, cyclones) and trigger payouts — no paperwork required.

---

## 📋 Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| **Python** | 3.10 – 3.13 | `python --version` |
| **Node.js** | 18+ | `node --version` |
| **npm** | 9+ | `npm --version` |
| **Git** | Any | `git --version` |

---

## ⚡ Setup (One-Time)

### 1. Clone the Repository

```bash
git clone https://github.com/Atharsh2910/GIC_ML-Pipeline.git
cd GIC_ML-Pipeline
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
GROQ_API_KEY="your_groq_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_HOST="your_pinecone_host_url"
PINECONE_INDEX_NAME="your_index_name"
VECTOR_STORE_PROVIDER="pinecone"
HF_TOKEN="your_huggingface_token_here"
```

> **Get API keys from:**
> - Groq → https://console.groq.com
> - Pinecone → https://app.pinecone.io
> - HuggingFace → https://huggingface.co/settings/tokens

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

### Terminal 2 — Start Frontend (Vite + React)

```bash
# From the frontend folder: GIC_ML-Pipeline/frontend/
npm run dev
```

The frontend runs at: **http://localhost:5173**

---



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
Weather API → ML Inference → Eligibility Check → Decision + Payout
```

### Expected Result (Flood scenario):

| Field | Value |
|-------|-------|
| Decision | ✅ APPROVE |
| Payout | ~₹1,818 |
| Confidence | ~88% |
| Coverable Gap | ₹2,250 (30% income loss) |

---

## 🏗️ Project Structure

```
GIC_ML-Pipeline/
├── app.py                    # FastAPI entrypoint (run this with uvicorn)
├── main.py                   # ML pipeline runner (training, inference)
├── requirements.txt          # Python dependencies
├── .env                      # Backend environment variables
│
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI routes & lifespan
│   │   └── schemas.py        # Pydantic request/response models
│   ├── pipeline/
│   │   ├── orchestrator.py   # Classic orchestrator (5-layer pipeline)
│   │   └── training_pipeline.py  # Model training & inference
│   ├── models/
│   │   ├── deterministic_models.py  # Claim eligibility & payout calc
│   │   ├── income_forecasting.py    # Income prediction models
│   │   ├── risk_scoring.py          # Risk scoring models
│   │   └── fraud_detection.py       # Fraud detection models
│   ├── agents/               # LangGraph & LangChain agents
│   ├── rag/                  # RAG system (vector store + retriever)
│   └── integrations/         # MCP client & external integrations
│
├── config/
│   ├── model_config.py       # ML hyperparameters & thresholds
│   └── dataset_schema.py     # Data schema definitions
│
├── mock_api/
│   └── mock_api.py           # Weather, news, telecom mock APIs
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
└── models/                   # Trained model artifacts (.pkl files)
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inference/predict` | POST | Run all 6 ML models on a worker |
| `/api/claims/process-classic` | POST | Full orchestrator: eligibility + payout |
| `/api/evaluate_worker` | POST | LangGraph agent evaluation (needs GROQ_API_KEY) |
| `/partner-mock/api/weather` | GET | Mock weather data (`?city=Chennai&scenario=flood`) |
| `/partner-mock/api/news` | GET | Mock news articles |
| `/partner-mock/api/telecom` | GET | Mock telecom outage data |
| `/partner-mock/api/fuel` | GET | Mock fuel supply data |
| `/partner-mock/api/platform` | GET | Mock platform status data |
| `/docs` | GET | Swagger UI (auto-generated API docs) |

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

---

## ❓ Troubleshooting

### Backend won't start
- Make sure you're in the **project root**, not inside `src/`
- Check that `.env` file has valid API keys
- Run `pip install -r requirements.txt` again

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
