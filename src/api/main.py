import sys
import os
from pathlib import Path

# Add project root to path for absolute imports
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from src.api.routers import claims, workers, predict
from mock_api.mock_api import router as mock_router

app = FastAPI(
    title="GigShield Central Backend API",
    description="Main Backend for GigShield ML Pipeline and Dashboard App",
    version="1.0.0"
)

# CORS Configuration for the Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modules
app.include_router(claims.router, prefix="/api/v1")
app.include_router(workers.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")

# Include the mock API router directly
app.include_router(mock_router)

# Note: HttpMockApiMCPClient is looking for /api/weather, so we can also 
# explicitly include mock_router if we refactor mock_api.py, but mounting is easiest.

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "GigShield Backend"}
