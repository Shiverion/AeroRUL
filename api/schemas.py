"""Pydantic request/response models for the AeroRUL prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field

SUBSET_PATTERN = "^FD00[1-4]$"


class SensorReading(BaseModel):
    """One engine-cycle snapshot: 3 operational settings + 21 sensor channels."""

    unit_number: int
    time_cycles: int
    setting_1: float
    setting_2: float
    setting_3: float
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float


class PredictRequest(BaseModel):
    subset: str = Field(pattern=SUBSET_PATTERN, description="Which trained CMAPSS model to use")
    readings: list[SensorReading] = Field(
        min_length=1,
        description="Chronological cycle history for one engine (oldest first). More "
        "history gives more accurate rolling-window features; a single reading still works.",
    )


class MaintenancePrediction(BaseModel):
    unit_number: int
    latest_cycle: int
    predicted_rul: float
    risk_tier: str
    recommendation: str


class FleetEngineSummary(BaseModel):
    unit_number: int
    latest_cycle: int
    predicted_rul: float
    true_rul: float | None = None
    risk_tier: str
    recommendation: str


class EngineHistory(BaseModel):
    unit_number: int
    subset: str
    cycles: list[int]
    sensors: dict[str, list[float]]
    predicted_rul: float
    true_rul: float | None = None
    risk_tier: str
    recommendation: str
