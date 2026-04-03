"""
Deterministic engines: Claim Eligibility & Payout Optimization.
Actuarial rules are expressed as closed-form formulas and vectorized tier tables (not ad-hoc if-chains).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.dataset_schema import DISRUPTION_NONE
from config.model_config import CLAIM_ELIGIBILITY_CONFIG, PAYOUT_OPTIMIZATION_CONFIG


@dataclass
class ClaimDecision:
    """Claim eligibility decision with reasoning."""

    is_eligible: bool
    reasons: List[str]
    threshold_check: Dict[str, bool]
    calculated_loss: float
    coverage_eligible: float


def _trust_on_0_5_scale(fraud_trust_rating: float) -> float:
    """Map dataset trust to 0–5 scale (doc); if values are in [0,1], scale up."""
    if fraud_trust_rating is None or (isinstance(fraud_trust_rating, float) and np.isnan(fraud_trust_rating)):
        return 2.5
    if fraud_trust_rating <= 1.0:
        return float(fraud_trust_rating) * 5.0
    return float(fraud_trust_rating)


def rainfall_coverage_vectorized(rainfall_cm: np.ndarray) -> np.ndarray:
    """Piecewise coverage for precipitation (doc §12), vectorized."""
    edges = np.array([0.0, 5.0, 10.0, 15.0, 20.0, np.inf])
    levels = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
    idx = np.searchsorted(edges, rainfall_cm, side="right") - 1
    idx = np.clip(idx, 0, len(levels) - 1)
    return levels[idx]


def cyclone_coverage_vectorized(level: np.ndarray) -> np.ndarray:
    """Alert level 0–5 → coverage 0–1."""
    level = np.clip(level, 0, 5)
    return np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])[level.astype(int)]


class ClaimEligibilityModel:
    """
    Deterministic eligibility: 75% income floor, cooling period, premium status.
    Coverable loss = max(0, 0.52w_avg - actual) when actual < 0.75 * avg.
    """

    def __init__(self, config: Dict | None = None):
        self.config = config or CLAIM_ELIGIBILITY_CONFIG

    def evaluate_eligibility(self, worker_data: Dict) -> ClaimDecision:
        reasons: List[str] = []
        checks: Dict[str, bool] = {}

        cooling_ok = worker_data.get("cooling_period_completed", 0) == 1
        checks["cooling_period"] = cooling_ok
        if not cooling_ok:
            reasons.append(
                f"Failed: Cooling period not completed (requires {self.config['cooling_period_weeks']} weeks)"
            )

        active_weeks_ok = worker_data.get("weeks_active", 0) >= self.config["minimum_active_weeks"]
        checks["minimum_active_weeks"] = active_weeks_ok
        if not active_weeks_ok:
            reasons.append(
                f"Failed: Insufficient active weeks (requires {self.config['minimum_active_weeks']} weeks)"
            )

        premium_ok = worker_data.get("premium_paid", 0) == 1
        checks["premium_paid"] = premium_ok
        if not premium_ok:
            reasons.append("Failed: Premium not paid for current week")

        actual = float(worker_data.get("weekly_income", 0) or 0)
        avg = float(worker_data.get("avg_52week_income", 0) or 0)
        threshold_income = avg * self.config["income_threshold"]

        threshold_ok = (avg > 0) and (actual < threshold_income)
        checks["income_threshold"] = threshold_ok
        if avg <= 0:
            reasons.append("Failed: Invalid average income")
        elif not threshold_ok:
            reasons.append(
                f"Failed: Income ({actual:.2f}) not below 75% threshold ({threshold_income:.2f})"
            )

        coverable = float(max(0.0, threshold_income - actual)) if avg > 0 else 0.0
        loss_pct = coverable / avg if avg > 0 else 0.0
        min_loss_ok = loss_pct >= self.config["minimum_loss_percentage"] - 1e-9
        checks["minimum_loss"] = min_loss_ok
        if avg > 0 and not min_loss_ok:
            reasons.append(
                f"Failed: Coverable loss fraction ({loss_pct:.1%}) below minimum ({self.config['minimum_loss_percentage']:.1%})"
            )

        is_eligible = bool(cooling_ok and active_weeks_ok and premium_ok and threshold_ok and min_loss_ok and avg > 0)
        if is_eligible:
            reasons.append(
                f"ELIGIBLE: Coverable gap below 75% floor: {coverable:.2f} ({loss_pct:.1%} of average income)"
            )

        return ClaimDecision(
            is_eligible=is_eligible,
            reasons=reasons,
            threshold_check=checks,
            calculated_loss=coverable,
            coverage_eligible=coverable,
        )

    def batch_evaluate(self, workers_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in workers_df.iterrows():
            d = self.evaluate_eligibility(row.to_dict())
            rows.append(
                {
                    "worker_id": row.get("worker_id"),
                    "is_eligible": d.is_eligible,
                    "calculated_loss": d.calculated_loss,
                    "coverage_eligible": d.coverage_eligible,
                    "reasons": "; ".join(d.reasons),
                }
            )
        return pd.DataFrame(rows)


class PayoutOptimizationModel:
    """
    Deterministic payout: disruption tier × slab × loyalty / penalties.
    Uses vectorized tier lookups for rainfall and cyclone; other types use duration-based scaling.
    """

    def __init__(self, config: Dict | None = None):
        self.config = config or PAYOUT_OPTIMIZATION_CONFIG

    def disruption_coverage_fraction(self, disruption_type: str, worker_row: Dict) -> float:
        """Map final_dataset disruption labels + legacy names to a 0–1 parametric factor."""
        raw = str(disruption_type or DISRUPTION_NONE).strip()
        if raw.lower() in ("", "nan", "none"):
            return 0.0
        key = raw.lower().replace(" ", "_")

        if key in ("heavy_rain", "flood", "rainfall"):
            return float(rainfall_coverage_vectorized(np.array([float(worker_row.get("rainfall_cm", 0) or 0)]))[0])
        if key in ("cyclone",):
            return float(cyclone_coverage_vectorized(np.array([float(worker_row.get("cyclone_alert_level", 0) or 0)]))[0])
        if key in ("extreme_heat", "heat"):
            te = abs(float(worker_row.get("temperature_extreme", 0) or 0))
            return float(np.clip(te / 45.0, 0.0, 1.0))
        if key in ("infrastructure", "strike", "curfew", "rally", "fuel_shortage", "network_outage", "platform_glitch", "cold"):
            hours = float(worker_row.get("disruption_duration_hours", 0) or 0)
            return float(np.clip(hours / 24.0, 0.0, 1.0))
        return 0.0

    def get_slab_coverage(self, slab: str) -> float:
        return float(self.config["slab_coverage"].get(slab, 0.75))

    def loyalty_bonus_percentage(self, consecutive_weeks: int) -> float:
        bonus_cycles = int(consecutive_weeks) // 4
        raw = bonus_cycles * self.config["loyalty_bonus_rate"]
        return float(min(raw, self.config["max_loyalty_bonus"]))

    def penalty_multiplier(self, default_weeks: int, fraud_trust_rating: float) -> float:
        """Reduces payable amount for defaulters; fraud uses 0–5 trust per product doc."""
        dw = int(default_weeks or 0)
        default_penalty = min(dw * 0.02, 0.30)
        tr5 = _trust_on_0_5_scale(fraud_trust_rating)
        fraud_component = max(0.0, 0.5 * (2.5 - min(tr5, 2.5)))
        total = float(np.clip(default_penalty + 0.2 * fraud_component, 0.0, 0.30))
        return 1.0 - total

    def calculate_payout(self, worker_data: Dict, eligibility_decision: ClaimDecision) -> Dict:
        if not eligibility_decision.is_eligible:
            return {"final_payout": 0.0, "eligible": False, "reason": "Not eligible for claim"}

        base_coverable = float(eligibility_decision.coverage_eligible)
        disruption_type = str(worker_data.get("disruption_type", DISRUPTION_NONE))
        d_cov = self.disruption_coverage_fraction(disruption_type, worker_data)

        slab = worker_data.get("selected_slab", "Standard")
        slab_cov = self.get_slab_coverage(slab)

        consec = int(worker_data.get("consecutive_payment_weeks", 0) or 0)
        loy_pct = self.loyalty_bonus_percentage(consec)
        coverable_adj = base_coverable * (1.0 + loy_pct)

        after_disruption = coverable_adj * d_cov
        after_slab = after_disruption * slab_cov

        default_weeks = int(worker_data.get("default_weeks", 0) or 0)
        fraud_trust = float(worker_data.get("fraud_trust_rating", 0.9) or 0.0)
        if self.config["penalty_impact_on_coverage"]:
            pen_mult = self.penalty_multiplier(default_weeks, fraud_trust)
            final_payout = after_slab * pen_mult
        else:
            pen_mult = 1.0
            final_payout = after_slab

        return {
            "final_payout": round(float(final_payout), 2),
            "eligible": True,
            "base_coverable_amount": round(base_coverable, 2),
            "disruption_coverage_pct": d_cov,
            "slab_coverage_pct": slab_cov,
            "loyalty_bonus_pct": loy_pct,
            "coverable_with_bonus": round(coverable_adj, 2),
            "after_disruption_tier": round(after_disruption, 2),
            "after_slab_coverage": round(after_slab, 2),
            "penalty_multiplier": pen_mult,
            "breakdown": (
                f"Base coverable {base_coverable:.2f} → +loyalty {loy_pct:.1%} → "
                f"×disruption {d_cov:.0%} → ×slab {slab_cov:.0%} → ×penalty {pen_mult:.2f} → {final_payout:.2f}"
            ),
        }

    def batch_calculate(self, workers_df: pd.DataFrame, eligibility_results: pd.DataFrame) -> pd.DataFrame:
        out_rows = []
        for _, row in workers_df.iterrows():
            wid = row.get("worker_id")
            elig = eligibility_results[eligibility_results["worker_id"] == wid]
            if elig.empty:
                continue
            er = elig.iloc[0]
            decision = ClaimDecision(
                is_eligible=bool(er["is_eligible"]),
                reasons=[],
                threshold_check={},
                calculated_loss=float(er.get("calculated_loss", 0)),
                coverage_eligible=float(er.get("coverage_eligible", 0)),
            )
            pr = self.calculate_payout(row.to_dict(), decision)
            pr["worker_id"] = wid
            out_rows.append(pr)
        return pd.DataFrame(out_rows)


if __name__ == "__main__":
    elig = ClaimEligibilityModel()
    pay = PayoutOptimizationModel()
    w = {
        "worker_id": 1,
        "weeks_active": 20,
        "cooling_period_completed": 1,
        "premium_paid": 1,
        "avg_52week_income": 8000,
        "weekly_income": 4500,
        "disruption_type": "rainfall",
        "rainfall_cm": 15,
        "selected_slab": "Slab 3 (100%)",
        "consecutive_payment_weeks": 20,
        "default_weeks": 0,
        "fraud_trust_rating": 0.9,
    }
    d = elig.evaluate_eligibility(w)
    print("eligible:", d.is_eligible, "coverable:", d.coverage_eligible)
    if d.is_eligible:
        print(pay.calculate_payout(w, d))
