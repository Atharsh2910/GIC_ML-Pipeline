"""
HTTP client for the local FastAPI mock in `mock_api/mock_api.py`.

Set GIGSHIELD_MOCK_API_BASE=http://127.0.0.1:8000 (no trailing slash) and run:
  uvicorn mock_api.mock_api:app --host 127.0.0.1 --port 8000

Aggregates /api/weather, /api/news, /api/telecom, /api/platform into the same bundle
shape as MockMCPClient so MonitorAgent needs no changes.

Env:
  GIGSHIELD_MOCK_API_BASE   — if set, default_mcp_client() uses this client
  GIGSHIELD_MOCK_SCENARIO   — maps to mock_api query scenario (heavy_rain→flood, etc.)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

# Internal scenario name → FastAPI `scenario` query (weather/news/telecom)
_SCENARIO_TO_API: Dict[str, Optional[str]] = {
    "heavy_rain": "flood",
    "cyclone": "cyclone",
    "strike": "strike",
    "curfew": "curfew",
    "protest": "protest",
    "clear": None,
    "heat": None,
}


def _internal_scenario() -> str:
    return (os.getenv("GIGSHIELD_MOCK_SCENARIO") or "heavy_rain").strip().lower()


def _api_weather_scenario(internal: str) -> Optional[str]:
    return _SCENARIO_TO_API.get(internal, internal if internal in ("flood", "cyclone", "strike") else None)


def _telecom_scenario(internal: str) -> Optional[str]:
    if internal == "cyclone":
        return "cyclone"
    if internal in ("heavy_rain", "flood"):
        return "storm"
    if internal == "clear":
        return None
    return "outage"


class HttpMockApiMCPClient:
    """Calls mock_api FastAPI routes; same interface as MockMCPClient."""

    def __init__(self, base_url: str, scenario: Optional[str] = None):
        self.base = base_url.rstrip("/")
        self.scenario = (scenario or _internal_scenario()).strip().lower()

    def _weather_params(self, city: str) -> Dict[str, Any]:
        scen = _api_weather_scenario(self.scenario)
        p: Dict[str, Any] = {"city": city or "Mumbai"}
        if scen:
            p["scenario"] = scen
        return p

    async def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params or {}) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def get_monitoring_signals(
        self,
        city: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = context or {}
        wid = ctx.get("worker_id")
        oid = ctx.get("outlet_id")
        c = (city or "Mumbai").strip()
        api_scen = _api_weather_scenario(self.scenario)

        weather_task = asyncio.create_task(self._get_json("/api/weather", self._weather_params(c)))
        news_task = asyncio.create_task(
            self._get_json("/api/news", {"city": c, "scenario": api_scen, "limit": 5} if api_scen else {"city": c, "limit": 5})
        )
        telecom_task = asyncio.create_task(
            self._get_json(
                "/api/telecom",
                {"city": c, "scenario": _telecom_scenario(self.scenario)},
            )
        )
        platform_task = asyncio.create_task(self._get_json("/api/platform", {"city": c, "limit": 8}))

        weather_raw, news_raw, telecom_raw, platform_raw = await asyncio.gather(
            weather_task, news_task, telecom_task, platform_task, return_exceptions=True
        )

        weather = self._normalize_weather(c, weather_raw)
        regional = self._normalize_regional(news_raw, telecom_raw, platform_raw, c)

        if self.scenario == "heat":
            weather["temperature"] = max(float(weather.get("temperature", 28)), 44.0)
            weather.setdefault("alerts", []).append(
                {"type": "Extreme_Heat", "severity": "high", "source": "mock_api_derived"}
            )

        return {
            "weather": weather,
            "regional": regional,
            "platform_pulse": {
                "worker_id": wid,
                "outlet_id": oid,
                "orders_completed_proxy": (ctx.get("worker_row") or {}).get("orders_completed_week"),
                "source": "http_mock_api",
                "raw": {"weather": weather_raw if not isinstance(weather_raw, Exception) else None},
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "scenario": self.scenario,
        }

    def _normalize_weather(self, city: str, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, Exception) or not isinstance(raw, dict):
            return {"city": city, "rainfall_cm": 0.0, "temperature": 28.0, "alerts": []}
        reports = raw.get("reports") or []
        if not reports:
            return {"city": raw.get("city", city), "rainfall_cm": 0.0, "temperature": 28.0, "alerts": []}
        max_mm = 0.0
        max_temp = 0.0
        alerts: List[Dict[str, Any]] = []
        for r in reports:
            m = r.get("measurements") or {}
            max_mm = max(max_mm, float(m.get("rainfall_mm_24h") or 0))
            max_temp = max(max_temp, float(m.get("temperature_c") or 0))
            summ = r.get("condition_summary")
            if summ:
                alerts.append({"type": "weather_report", "summary": summ, "area": r.get("area")})
            al = r.get("alerts") or {}
            if isinstance(al, dict) and al.get("level"):
                alerts.append(
                    {
                        "type": "Heavy_Rain" if max_mm >= 100 else "weather_alert",
                        "severity": str(al.get("level", "")).lower(),
                        "source": al.get("issued_by", "IMD"),
                        "detail": al,
                    }
                )
        rainfall_cm = max_mm / 10.0
        return {
            "city": raw.get("city", city),
            "rainfall_cm": rainfall_cm,
            "temperature": max_temp or 28.0,
            "alerts": alerts[:12],
            "reports_count": len(reports),
        }

    def _normalize_regional(
        self,
        news_raw: Any,
        telecom_raw: Any,
        platform_raw: Any,
        city: str,
    ) -> Dict[str, Any]:
        news_flags: List[str] = []
        affected = 50
        duration_h = 4.0
        stress = 0.3

        if isinstance(news_raw, dict):
            for a in news_raw.get("articles") or []:
                h = (a.get("headline") or "").lower()
                if "strike" in h:
                    news_flags.append("strike_news")
                if "curfew" in h or "144" in h:
                    news_flags.append("curfew_news")
                if "protest" in h or "rally" in h:
                    news_flags.append("protest_news")
            affected += min(300, len(news_raw.get("articles") or []) * 40)

        if isinstance(telecom_raw, dict):
            down = sum(1 for o in telecom_raw.get("outages") or [] if o.get("status") == "down")
            deg = sum(1 for o in telecom_raw.get("outages") or [] if o.get("status") == "degraded")
            affected += down * 80 + deg * 35
            duration_h += down * 2.5 + deg * 0.8
            stress = min(0.98, stress + down * 0.12 + deg * 0.05)

        if isinstance(platform_raw, dict):
            inc = platform_raw.get("incidents") or []
            bad = [i for i in inc if str(i.get("city", "")).lower() == city.lower() and i.get("status") in ("down", "degraded")]
            affected += len(bad) * 120
            duration_h += len(bad) * 1.5
            stress = min(0.99, stress + len(bad) * 0.08)

        return {
            "affected_workers_estimate": int(min(5000, affected)),
            "disruption_duration_hours": float(min(48.0, duration_h)),
            "inventory_stress_index": float(stress),
            "news_flags": list(dict.fromkeys(news_flags)),
        }

    async def submit_claim_rollout(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/api/claims/rollout"
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        raise RuntimeError(f"{resp.status}: {text}")
                    return await resp.json()
        except Exception as exc:
            return {
                "status": "accepted",
                "ack_id": str(uuid.uuid4()),
                "queue": "local_fallback",
                "reason": "http_rollout_failed",
                "detail": str(exc)[:500],
            }
