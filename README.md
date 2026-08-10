# AeroRUL — Turbofan Engine Remaining Useful Life Prediction

Predictive maintenance system for jet engines: 21-sensor time series from the NASA CMAPSS
turbofan degradation dataset, run end-to-end from raw sensor readings to a maintenance
recommendation.

```
Raw Sensor Data -> Health Indicator -> RUL Prediction -> Failure Risk -> Maintenance Recommendation -> API / Dashboard
```

## Status

Working end-to-end skeleton: raw sensor data -> trained model -> API -> dashboard, for all
4 CMAPSS subsets. Deeper modeling (sequence models, uncertainty, survival analysis) is next.

- [x] Data ingestion (NASA CMAPSS, FD001-FD004)
- [x] Feature engineering (RUL labeling, per-condition normalization, rolling stats, health indicator)
- [x] Baseline model (XGBoost) + NASA PHM08 scoring evaluation, tracked in MLflow
- [x] FastAPI prediction + fleet-monitoring service
- [x] React fleet dashboard (risk-tiered engine list + per-engine sensor trends)
- [x] Dockerized deployment (API + dashboard via docker-compose)
- [ ] Sequence models (LSTM, Temporal CNN, Transformer)
- [ ] Uncertainty quantification (conformal prediction / quantile regression)
- [ ] Survival analysis comparison

### Current baseline results (XGBoost, test set)

| Subset | RMSE | MAE | NASA score | Test engines |
|--------|-----:|----:|-----------:|-------------:|
| FD001  | 18.1 | 12.8 | 1108 | 100 |
| FD002  | 15.2 | 11.2 | 1174 | 259 |
| FD003  | 17.8 | 12.1 | 1191 | 100 |
| FD004  | 17.8 | 13.2 | 2068 | 248 |

## Project layout

```
src/aerorul/
  data/          raw-file loading, schema/constants
  features/      RUL labeling, rolling-window feature engineering
  models/        XGBoost baseline, sequence models, survival model
  evaluation/    RMSE + NASA asymmetric scoring function
  uncertainty/   conformal prediction / quantile intervals
api/             FastAPI service (predict / fleet / health)
dashboard/       React fleet-health dashboard
scripts/         data download, training, evaluation entry points
data/            raw + processed CMAPSS data (gitignored, see data/README.md)
models_store/    trained model artifacts (gitignored)
```

## Getting started

### Backend (data, training, API)

```bash
scripts/download_data.sh              # fetch raw CMAPSS data into data/raw/
uv sync --extra api --extra dev
uv run python scripts/train.py --subset all    # trains + evaluates + saves all 4 subsets
uv run uvicorn api.main:app --reload --port 8000
```

Try it: `curl http://localhost:8000/fleet/FD001` or open http://localhost:8000/docs for the
interactive OpenAPI UI.

### Dashboard

```bash
cd dashboard
npm install
npm run dev          # http://localhost:5173, expects the API at http://localhost:8000
```

### Everything via Docker

```bash
uv run python scripts/train.py --subset all   # models_store/ must be populated before building
docker compose up --build
```
API on http://localhost:8000, dashboard on http://localhost:5173.

## Dataset

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) — simulated turbofan
degradation runs across 4 subsets varying in operating conditions and fault modes. Each
engine is described by 3 operational settings and 21 sensor channels, run from healthy to
failure (train) or truncated before failure with a true RUL label (test). See
[data/README.md](data/README.md) for details.

Citation: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation
modeling for aircraft engine run-to-failure simulation.* International Conference on
Prognostics and Health Management.
