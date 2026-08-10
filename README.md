# AeroRUL — Turbofan Engine Remaining Useful Life Prediction

Predictive maintenance system for jet engines: 21-sensor time series from the NASA CMAPSS
turbofan degradation dataset, run end-to-end from raw sensor readings to a maintenance
recommendation.

```
Raw Sensor Data -> Health Indicator -> RUL Prediction -> Failure Risk -> Maintenance Recommendation -> API / Dashboard
```

## Status

Raw sensor data -> multiple competing models -> a per-subset champion -> API -> dashboard,
for all 4 CMAPSS subsets.

- [x] Data ingestion (NASA CMAPSS, FD001-FD004)
- [x] Feature engineering (RUL labeling, per-condition normalization, rolling stats, health indicator)
- [x] Baseline model (XGBoost) + NASA PHM08 scoring evaluation, tracked in MLflow
- [x] Sequence models (LSTM, Temporal CNN, Transformer)
- [x] Uncertainty quantification (split conformal prediction intervals)
- [x] Survival analysis (Weibull AFT, landmark time-to-failure formulation)
- [x] Model comparison + automatic per-subset champion selection
- [x] FastAPI prediction + fleet-monitoring service (serves each subset's champion model)
- [x] React fleet dashboard (risk-tiered engine list, uncertainty bands, per-engine sensor trends)
- [x] Dockerized deployment (API + dashboard via docker-compose)

Run `uv run python scripts/compare_models.py --subset all` after training to see the
current per-subset leaderboard and regenerate `models_store/champion.json`.

### Model comparison (test set, evaluated against true RUL from `RUL_*.txt`)

| Subset | Champion | RMSE | MAE | NASA score | Runner-up |
|--------|----------|-----:|----:|-----------:|-----------|
| FD001  | transformer | 16.0 | 12.2 | 439 | tcn (558) |
| FD002  | tcn         | 28.3 | 19.4 | 10486 | xgboost (11118) |
| FD003  | transformer | 16.0 | 11.7 | 549 | xgboost (1225) |
| FD004  | transformer | 26.6 | 19.5 | 5647 | lstm (5869) |

Two things worth calling out:
- **FD002/FD004 are genuinely harder** (6 operating conditions, 2 fault modes) — every
  model's error roughly doubles versus FD001/FD003. On FD002, XGBoost is essentially tied
  with the deep models; the sequence models' edge shows up mainly on the single-condition
  subsets.
- **Survival analysis (Weibull AFT) loses on NASA score everywhere** despite a solid
  ~0.80-0.82 concordance index — a handful of engines get a badly-overestimated median
  remaining life, and the NASA score's asymmetric penalty (late predictions cost far more
  than early ones) punishes those outliers heavily. It's kept as a deliberate comparison
  point, not a strawman: it's the only model here that natively produces a full survival
  distribution rather than a point estimate, which matters for some questions even when its
  point-estimate accuracy trails.
- **Uncertainty intervals under-cover on FD002/FD004** (~76-80% empirical vs. a 90% target;
  FD001/FD003 hit ~89-96%). The conformal calibration is filtered to avoid the RUL-cap
  artifact (see `scripts/calibrate_uncertainty.py`), but residual distribution shift between
  the calibration split and the true test set is larger on the multi-condition subsets.
  Treat interval widths on those two subsets as optimistic.

All figures come from `scripts/compare_models.py`, which evaluates every model consistently
against the true, uncapped RUL truth — training itself still targets the RUL-capped label
(a standard loss-shaping trick, see `DEFAULT_RUL_CAP` in `engineering.py`), but scoring
against anything other than the real ground truth would understate error for engines with
a lot of life left, which is exactly the kind of bug worth naming rather than hiding.

## Project layout

```
src/aerorul/
  data/          raw-file loading, schema/constants
  features/      RUL labeling, per-condition normalization, rolling-window features
  models/        XGBoost, LSTM, TCN, Transformer, Weibull AFT survival, decision layer
  evaluation/    RMSE + NASA asymmetric scoring function
  uncertainty/   split conformal prediction intervals
api/             FastAPI service (predict / fleet / engine / health)
dashboard/       React fleet-health dashboard
scripts/         data download, training (per model type), comparison, evaluation
data/            raw + processed CMAPSS data (gitignored, see data/README.md)
models_store/    trained model artifacts + champion.json (gitignored)
```

## Getting started

### Backend (data, training, API)

```bash
scripts/download_data.sh              # fetch raw CMAPSS data into data/raw/
uv sync --extra api --extra dev --extra dl --extra survival

uv run python scripts/train.py --subset all              # XGBoost baseline
uv run python scripts/train_sequence.py --subset all --model all   # LSTM, TCN, Transformer
uv run python scripts/train_survival.py --subset all     # Weibull AFT survival model
uv run python scripts/calibrate_uncertainty.py --subset all        # conformal intervals
uv run python scripts/compare_models.py --subset all     # picks each subset's champion

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
