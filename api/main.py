"""AeroRUL prediction API: sensor readings in, RUL + failure risk + maintenance
recommendation out. Also exposes precomputed fleet views for the dashboard.

Run: uv run uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.inference import (
    ModelRegistry,
    engine_history,
    fleet_summary,
    model_comparison,
    predict_from_readings,
)
from api.schemas import EngineHistory, FleetEngineSummary, MaintenancePrediction, PredictRequest

app = FastAPI(
    title="AeroRUL API",
    description="Turbofan engine Remaining Useful Life prediction and fleet monitoring.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "available_subsets": ModelRegistry.available_subsets()}


@app.post("/predict", response_model=MaintenancePrediction)
def predict_endpoint(request: PredictRequest) -> MaintenancePrediction:
    readings_df = pd.DataFrame([r.model_dump() for r in request.readings])
    try:
        result = predict_from_readings(request.subset, readings_df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MaintenancePrediction(**result)


@app.get("/fleet/{subset}", response_model=list[FleetEngineSummary])
def fleet_endpoint(subset: str) -> list[FleetEngineSummary]:
    try:
        return [FleetEngineSummary(**row) for row in fleet_summary(subset)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/engine/{subset}/{unit_number}", response_model=EngineHistory)
def engine_endpoint(subset: str, unit_number: int) -> EngineHistory:
    try:
        return EngineHistory(**engine_history(subset, unit_number))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models/{subset}")
def models_endpoint(subset: str) -> dict:
    """The model-comparison leaderboard for a subset: every architecture's test metrics and
    which one was picked as the deployed champion. None if scripts/compare_models.py hasn't
    been run for this subset yet.
    """
    comparison = model_comparison(subset)
    if comparison is None:
        raise HTTPException(
            status_code=404,
            detail=f"No model comparison found for {subset}. Run scripts/compare_models.py.",
        )
    return comparison
