"""
Curated text chunks for RAG — maps to Chroma categories (insurance_policies, fraud_cases, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Each item: id, text, metadata (Chroma: values must be str/int/float)
KNOWLEDGE_BY_CATEGORY: Dict[str, List[Dict[str, Any]]] = {
    "insurance_policies": [
        {
            "id": "pol_75_threshold",
            "text": "Payout eligibility requires actual weekly income below 75 percent of the worker 52-week moving average. The coverable amount is the gap between actual income and the 75 percent threshold line.",
            "metadata": {"topic": "threshold", "version": "1"},
        },
        {
            "id": "pol_cooling",
            "text": "Eight-week cooling period: new enrollees must pay premiums for eight consecutive weeks before any claim can be paid. This mitigates adverse selection before forecast disruptions.",
            "metadata": {"topic": "cooling"},
        },
        {
            "id": "pol_slabs",
            "text": "Coverage slabs Basic Standard Premium Elite map to different premium multipliers and payout coverage caps. Base actuarial rate is 4 percent of average weekly income before slab adjustment.",
            "metadata": {"topic": "premium"},
        },
        {
            "id": "pol_proof",
            "text": "Claims require proof of presence in disruption zone, proof of work activity orders or hours, and proof of income impact. GPS alone is insufficient.",
            "metadata": {"topic": "eligibility"},
        },
    ],
    "fraud_cases": [
        {
            "id": "fr_gps",
            "text": "High gps_spoofing_score with low movement_realism_score and low presence_score indicates synthetic location; cross-check with peer_group_activity_ratio at outlet.",
            "metadata": {"severity": "high"},
        },
        {
            "id": "fr_cluster",
            "text": "coordinated_fraud_cluster_id greater than zero with multiple workers claiming same window may indicate rings; combine with device_sharing_flag and ip_gps_mismatch.",
            "metadata": {"severity": "critical"},
        },
        {
            "id": "fr_adverse",
            "text": "Workers enrolling only before major weather events without prior payment history are flagged for adverse selection review.",
            "metadata": {"severity": "medium"},
        },
    ],
    "disruption_events": [
        {
            "id": "dis_rain_bands",
            "text": "Heavy rain parametric schedule: under 5cm no trigger; 5-10cm 40 percent of eligible loss; 10-15cm 60 percent; 15-20cm 80 percent; 20cm plus 100 percent of eligible loss subject to slab.",
            "metadata": {"type": "Heavy_Rain"},
        },
        {
            "id": "dis_cyclone",
            "text": "Cyclone payouts scale with cyclone_alert_level 0-5 and worker proximity; combine with disruption_duration_hours for infrastructure-type events.",
            "metadata": {"type": "Cyclone"},
        },
        {
            "id": "dis_heat",
            "text": "Extreme heat uses temperature_extreme relative to regional norms; income loss must still breach the 75 percent floor versus 52-week average.",
            "metadata": {"type": "Extreme_Heat"},
        },
    ],
    "historical_claims": [
        {
            "id": "hc_001",
            "text": "Example: moving average 8000, actual 4500, rainfall 15cm, slab Elite, loyalty bonus applied; disruption factor 80 percent on coverable gap below threshold.",
            "metadata": {"city": "example"},
        },
        {
            "id": "hc_002",
            "text": "Defaulters: penalty reduces effective coverage on the eligible loss before disruption tier multiplication.",
            "metadata": {"topic": "penalty"},
        },
    ],
    "regional_data": [
        {
            "id": "reg_mumbai",
            "text": "Mumbai monsoon season elevated rainfall_cm variance; peer validation compares outlet cohort activity recovery after disruption windows.",
            "metadata": {"city": "Mumbai"},
        },
        {
            "id": "reg_delhi",
            "text": "Delhi summer heat waves correlate with Extreme_heat disruption_type and reduced active_hours_week across peer groups.",
            "metadata": {"city": "Delhi"},
        },
    ],
}


def all_documents_flat() -> List[Dict[str, Any]]:
    """Flatten for bulk ingest."""
    out: List[Dict[str, Any]] = []
    for category, docs in KNOWLEDGE_BY_CATEGORY.items():
        for d in docs:
            out.append({"category": category, **d})
    return out
