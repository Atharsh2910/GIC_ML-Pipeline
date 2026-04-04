"""
Mock MCP / partner API layer for GigShield workflow simulation.

Placement (per architecture doc):
  External APIs → MCP Layer → MonitorAgent → ValidationAgent → RAG → …

This module stands in for live Weather, News, Inventory, and Platform streams until
real B2B2C integrations exist. The orchestrators inject MockMCPClient into MonitorAgent.

Env:
  GIGSHIELD_MOCK_API_BASE  — if set (e.g. http://127.0.0.1:8000), use HttpMockApiMCPClient against mock_api
  GIGSHIELD_USE_MOCK_MCP   — default true; set false to use MonitorAgent stubs only
  GIGSHIELD_MOCK_SCENARIO  — heavy_rain | clear | heat | cyclone | strike (default heavy_rain)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from src.integrations.http_mock_api_client import HttpMockApiMCPClient

# In-memory "rollout log" for demos (not Supabase)
_MOCK_CLAIM_LOG: List[Dict[str, Any]] = []


def _scenario() -> str:
    return (os.getenv("GIGSHIELD_MOCK_SCENARIO") or "heavy_rain").strip().lower()


class MockMCPClient:
    """Async facade mimicking MCP-fed monitoring + claim acknowledgement APIs."""

    def __init__(self, scenario: Optional[str] = None):
        self.scenario = (scenario or _scenario()).strip().lower()

    async def get_monitoring_signals(
        self,
        city: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns weather + regional impact estimates (inventory stress, peer outage proxy)
        as if pulled from partner + meteorological feeds.
        """
        ctx = context or {}
        wid = ctx.get("worker_id")
        oid = ctx.get("outlet_id")

        weather: Dict[str, Any] = {"city": city, "rainfall_cm": 0.0, "temperature": 28.0, "alerts": []}
        regional = {
            "affected_workers_estimate": 12,
            "disruption_duration_hours": 3.0,
            "inventory_stress_index": 0.25,
            "news_flags": [],
        }

        if self.scenario == "clear":
            pass
        elif self.scenario == "heavy_rain":
            weather["rainfall_cm"] = 16.0
            weather["alerts"] = [{"type": "Heavy_Rain", "severity": "high", "source": "mock_meteo"}]
            regional["affected_workers_estimate"] = 180
            regional["disruption_duration_hours"] = 9.0
            regional["inventory_stress_index"] = 0.82
        elif self.scenario == "heat":
            weather["temperature"] = 46.0
            weather["alerts"] = [{"type": "Extreme_Heat", "severity": "high", "source": "mock_meteo"}]
            regional["affected_workers_estimate"] = 90
            regional["disruption_duration_hours"] = 6.0
        elif self.scenario == "cyclone":
            weather["rainfall_cm"] = 8.0
            weather["alerts"] = [{"type": "Cyclone", "severity": "critical", "level": 4, "source": "mock_imd"}]
            regional["affected_workers_estimate"] = 400
            regional["disruption_duration_hours"] = 24.0
            regional["inventory_stress_index"] = 0.95
        elif self.scenario == "strike":
            weather["alerts"] = [{"type": "Labor_Action", "severity": "medium", "source": "mock_news"}]
            regional["news_flags"] = ["transit_strike", "localized_routing_failure"]
            regional["affected_workers_estimate"] = 220
            regional["disruption_duration_hours"] = 12.0
            regional["inventory_stress_index"] = 0.55

        return {
            "weather": weather,
            "regional": regional,
            "platform_pulse": {
                "worker_id": wid,
                "outlet_id": oid,
                "orders_completed_proxy": ctx.get("worker_row", {}).get("orders_completed_week"),
                "source": "mock_platform_api",
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "scenario": self.scenario,
        }

    async def submit_claim_rollout(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Layer-4 execution acknowledgement (structured memory / partner webhook stand-in)."""
        rec = {
            "ack_id": str(uuid.uuid4()),
            "received_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        _MOCK_CLAIM_LOG.append(rec)
        return {"status": "accepted", "ack_id": rec["ack_id"], "queue": "mock_rollout"}


def mock_claim_log_snapshot() -> List[Dict[str, Any]]:
    return list(_MOCK_CLAIM_LOG)


def should_use_mock_mcp() -> bool:
    return os.getenv("GIGSHIELD_USE_MOCK_MCP", "true").strip().lower() in ("1", "true", "yes", "on")


def default_mcp_client() -> Optional[Union[MockMCPClient, "HttpMockApiMCPClient"]]:
    base = (os.getenv("GIGSHIELD_MOCK_API_BASE") or "").strip().rstrip("/")
    if base:
        from src.integrations.http_mock_api_client import HttpMockApiMCPClient

        return HttpMockApiMCPClient(base_url=base)
    return MockMCPClient() if should_use_mock_mcp() else None
