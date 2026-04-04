@echo off
echo ===========================================
echo Starting GigShield Central Backend...
echo ===========================================

if not exist ".venv" (
    echo [INFO] Creating Python Virtual Environment. This might take a minute...
    python -m venv .venv
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [INFO] Installing required dependencies...
pip install -r requirements.txt

echo [INFO] Starting FastAPI Uvicorn Server...
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
