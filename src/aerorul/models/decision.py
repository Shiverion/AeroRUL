"""Translate a raw RUL prediction into the business layer: failure risk tier and a
maintenance recommendation. This is what turns "24.3" into something a maintenance
planner can act on.
"""

from __future__ import annotations

from dataclasses import dataclass

# Thresholds are in engine cycles remaining. Tuned relative to the training RUL cap (125,
# see engineering.DEFAULT_RUL_CAP) so "low risk" means comfortably clear of the regime the
# model was trained to be precise in near end-of-life.
RISK_THRESHOLDS = {
    "critical": 15,
    "high": 30,
    "medium": 60,
    # anything above `medium` is "low"
}


@dataclass(frozen=True)
class MaintenanceAssessment:
    predicted_rul: float
    risk_tier: str
    recommendation: str


_RECOMMENDATIONS = {
    "critical": "Ground engine for inspection immediately — failure risk within the next few cycles.",
    "high": "Schedule maintenance within the next few operating cycles; increase sensor monitoring frequency.",
    "medium": "Plan maintenance during the next scheduled service window.",
    "low": "No action needed — continue routine monitoring.",
}


def assess(predicted_rul: float) -> MaintenanceAssessment:
    if predicted_rul <= RISK_THRESHOLDS["critical"]:
        tier = "critical"
    elif predicted_rul <= RISK_THRESHOLDS["high"]:
        tier = "high"
    elif predicted_rul <= RISK_THRESHOLDS["medium"]:
        tier = "medium"
    else:
        tier = "low"

    return MaintenanceAssessment(
        predicted_rul=predicted_rul,
        risk_tier=tier,
        recommendation=_RECOMMENDATIONS[tier],
    )
