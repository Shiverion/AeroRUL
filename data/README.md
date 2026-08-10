# Data

## Raw

The NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan degradation
dataset — 4 subsets (FD001-FD004), each with `train_*.txt`, `test_*.txt`, `RUL_*.txt`.

Source: NASA Prognostics Center of Excellence data repository
(https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data).

Not checked into git (raw text files, ~40MB total, easy to regenerate). Fetch with:

```bash
scripts/download_data.sh
```

Each row is one engine-cycle with: unit number, time cycle, 3 operational settings,
21 sensor readings. `train_*` runs each engine to failure; `test_*` is truncated before
failure and `RUL_*` gives the true remaining cycles at truncation (test-set labels).

| Subset | Train units | Test units | Conditions | Fault modes |
|--------|------------:|-----------:|------------|-------------|
| FD001  | 100 | 100 | 1 | 1 (HPC degradation) |
| FD002  | 260 | 259 | 6 | 1 (HPC degradation) |
| FD003  | 100 | 100 | 1 | 2 (HPC + Fan degradation) |
| FD004  | 248 | 249 | 6 | 2 (HPC + Fan degradation) |

## Processed

`data/processed/` holds feature-engineered parquet files produced by
`src/aerorul/features/engineering.py` — also gitignored, regenerate via
`scripts/build_features.py`.
