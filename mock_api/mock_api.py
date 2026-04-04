from datetime import datetime, timedelta
import random
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(tags=["Mock Realistic Multi-City API"])

# -------------------------------
# CONFIG
# -------------------------------

CITIES = ["Chennai", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Delhi", "Pune", "Ahmedabad"]

AREAS_BY_CITY = {
    "Chennai": ["OMR", "Velachery", "Tambaram"],
    "Mumbai": ["Bandra", "Andheri", "Dadar"],
    "Bengaluru": ["Whitefield", "BTM", "Electronic City"],
    "Hyderabad": ["Gachibowli", "Hitech City", "Secunderabad"],
    "Delhi": ["Rohini", "Dwarka", "Connaught Place"],
    "Kolkata": ["Salt Lake", "Howrah", "Park Street"],
    "Pune": ["Hinjewadi", "Kothrud", "Viman Nagar"],
    "Ahmedabad": ["Navrangpura", "SG Highway", "Maninagar"]
}

# In-memory rollout log (mirrors in-process MockMCPClient for demos)
_CLAIM_ROLLOUT_LOG: List[Dict[str, Any]] = []


def _areas_for_city(city: str) -> List[str]:
    return AREAS_BY_CITY.get(city, ["Central"])


NEWS_SOURCES = ["The Hindu", "NDTV", "ANI", "Times of India", "PTI"]

def now():
    return datetime.utcnow().isoformat() + "Z"

# -------------------------------
#  REALISTIC NEWS
# -------------------------------

HARDCODED_NEWS = {
    "flood": [
        "IMD issues red alert as Chennai records 200mm rainfall in 24 hours",
        "Severe waterlogging reported across OMR and Velachery disrupting deliveries",
        "Flooded roads bring logistics operations to a halt in multiple zones"
    ],

    "strike": [
        "Transport strike disrupts supply chain across major city routes",
        "Thousands of drivers join union strike affecting gig workers",
        "Delivery operations impacted as driver unions halt services"
    ],

    "cyclone": [
        "Cyclone warning issued along coastal regions, operations suspended",
        "High wind speeds and heavy rain expected as storm approaches landfall"
    ],

    
    "curfew": [
        "Section 144 imposed in multiple areas restricting movement",
        "Night curfew announced by authorities amid rising tensions",
        "Authorities enforce movement restrictions across key zones"
    ],

    "protest": [
        "Mass protest blocks arterial roads causing major traffic disruptions",
        "Political rally leads to road closures across city center",
        "Thousands gather for protest, affecting logistics routes",
        "Public gathering causes severe congestion in multiple areas"
    ]
}
# -------------------------------
# WEATHER API
# -------------------------------

@router.get("/api/weather")
def weather(city: str = "Chennai", scenario: Optional[str] = None):
    areas = _areas_for_city(city)
    reports = []

    for area in areas:
        if scenario == "flood":
            rainfall = random.randint(150, 250)
        elif scenario == "cyclone":
            rainfall = random.randint(200, 300)
        else:
            rainfall = random.randint(0, 120)

        reports.append({
            "station_id": f"IMD-{random.randint(1000,9999)}",
            "area": area,
            "observation_time": now(),
            "coordinates": {
                "lat": round(10 + random.random()*10, 4),
                "lon": round(75 + random.random()*10, 4)
            },
            "measurements": {
                "rainfall_mm_24h": rainfall,
                "temperature_c": random.randint(24, 35),
                "humidity_percent": random.randint(60, 100),
                "wind_speed_kmph": random.randint(5, 80),
                "pressure_hpa": random.randint(990, 1015)
            },
            "condition_summary": random.choice([
                "Heavy to very heavy rainfall",
                "Thunderstorm with gusty winds",
                "Light to moderate rain",
                "Extremely heavy rainfall in isolated pockets"
            ]),
            "alerts": {
                "level": random.choice(["YELLOW", "ORANGE", "RED"]),
                "issued_by": "India Meteorological Department",
                "bulletin_no": f"IMD/BUL/{random.randint(100,999)}",
                "validity": (datetime.utcnow() + timedelta(hours=6)).isoformat() + "Z"
            }
        })

    return {"city": city, "reports": reports, "issued_at": now()}

# -------------------------------
# NEWS API (HYBRID)
# -------------------------------

@router.get("/api/news")
def news(city: Optional[str] = None, scenario: Optional[str] = None, limit: int = 10):
    articles = []

    if scenario in HARDCODED_NEWS:
        pool = HARDCODED_NEWS[scenario]
    else:
        pool = sum(HARDCODED_NEWS.values(), [])

    for _ in range(limit):
        selected_city = city if city else random.choice(CITIES)

        articles.append({
            "id": str(uuid.uuid4()),
            "headline": random.choice(pool),
            "source": random.choice(NEWS_SOURCES),
            "location": {
                "city": selected_city,
                "area": random.choice(_areas_for_city(selected_city)),
            },
            "published_at": (datetime.utcnow() - timedelta(hours=random.randint(1, 24))).isoformat() + "Z"
        })

    return {
        "total_articles": len(articles),
        "articles": articles
    }

# -------------------------------
# TELECOM API
# -------------------------------
@router.get("/api/telecom")
def telecom(city: str = "Chennai", scenario: Optional[str] = None):
    outages = []

    for _ in range(10):
        # base values
        latency = random.randint(20, 200)
        packet_loss = random.randint(0, 30)
        status = "operational"

        # scenario impact (light logic, not too obvious)
        if scenario == "storm" or scenario == "cyclone":
            latency = random.randint(150, 500)
            packet_loss = random.randint(20, 80)
        elif scenario == "outage":
            status = random.choice(["down", "degraded"])
            latency = random.randint(200, 500)
            packet_loss = random.randint(30, 80)

        # infer status (instead of pure random)
        if packet_loss > 60 or latency > 400:
            status = "down"
        elif packet_loss > 30 or latency > 200:
            status = "degraded"

        outages.append({
            "provider": random.choice(["Airtel", "Jio", "BSNL"]),
            "area": random.choice(_areas_for_city(city)),

            # core signals
            "status": status,
            "latency_ms": latency,
            "packet_loss_percent": packet_loss,

            # realistic but not too direct
            "jitter_ms": random.randint(5, 100),
            "outage_duration_min": random.randint(0, 120),
            "affected_users": random.randint(100, 5000),

            # mild hint (not too revealing)
            "issue_hint": random.choice([
                "network congestion",
                "possible fiber issue",
                "power instability",
                "unknown fluctuation"
            ])
        })

    return {
        "city": city,
        "scenario": scenario,
        "outages": outages,
        "generated_at": now()
    }

# -------------------------------
# FUEL API
# -------------------------------

@router.get("/api/fuel")
def fuel(city: str = "Chennai", limit: int = 10):
    headlines = [
        "Long queues observed at fuel stations across the city",
        "Several petrol pumps report shortage amid supply delays",
        "Drivers wait hours as fuel availability drops in key areas",
        "Fuel supply disruption reported due to logistics delays",
        "High demand leads to temporary stockouts in multiple stations",
        "Panic buying reported as fuel availability declines",
        "Tankers delayed due to heavy rains impacting supply chain"
    ]

    articles = []

    for _ in range(limit):
        articles.append({
            "id": str(uuid.uuid4()),
            "headline": random.choice(headlines),
            "source": random.choice(NEWS_SOURCES),
            "location": {
                "city": city,
                "area": random.choice(_areas_for_city(city)),
            },
            "published_at": now()
        })

    return {
        "city": city,
        "articles": articles
    }

# -------------------------------
# PLATFORM API
# -------------------------------

@router.get("/api/platform")
def platform(city: Optional[str] = None, limit: int = 10):
    incidents = []

    for _ in range(limit):
        error_rate = random.randint(0, 100)

        incidents.append({
            "platform": random.choice(["Swiggy", "Zomato", "Blinkit", "Zepto"]),
            "region": random.choice(["North", "Central", "South"]),
            "city": random.choice(CITIES),
            "status": (
                "down" if error_rate > 85 else
                "degraded" if error_rate > 50 else
                "operational"
            ),
            "metrics": {
                "error_rate_percent": error_rate,
                "api_latency_ms": random.randint(50, 2000),
                "success_rate_percent": 100 - error_rate,
                "active_orders": random.randint(0, 5000),
                "failed_orders": random.randint(0, 2000)
            },
            "incident_details": {
                "error_code": random.choice(["500", "502", "503"]),
                "root_cause": random.choice([
                    "server overload",
                    "database timeout",
                    "payment gateway failure",
                    "cloud outage"
                ]),
                "started_at": (datetime.utcnow() - timedelta(minutes=random.randint(5, 120))).isoformat() + "Z"
            }
        })

    return {
        "total_incidents": len(incidents),
        "incidents": incidents,
        "timestamp": now()
    }


# -------------------------------
# CLAIMS / ROLLOUT (MCP-style ack)
# -------------------------------


@router.post("/api/claims/rollout")
def claims_rollout(payload: Dict[str, Any]) -> Dict[str, Any]:
    ack_id = str(uuid.uuid4())
    _CLAIM_ROLLOUT_LOG.append(
        {
            "ack_id": ack_id,
            "received_at": now(),
            "payload": payload,
        }
    )
    return {"status": "accepted", "ack_id": ack_id, "queue": "mock_api_rollout"}