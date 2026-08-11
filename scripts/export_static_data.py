"""Export static JSON snapshots of /fleet, /engine, and /models for a backend-free dashboard
deployment (e.g. Vercel) -- dashboard/src/api.ts reads these directly when no live API base
is configured at build time, instead of hitting a running FastAPI service.

Fleet/engine data is inherently static anyway: it's predictions over the fixed CMAPSS test
set, computed once, not live inference on a moving feed. The one thing that genuinely needs
a running server is POST /predict (arbitrary raw sensor readings) -- the dashboard doesn't
call that endpoint, so nothing is lost by freezing everything else.

Usage:
    uv run python scripts/export_static_data.py --subset all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aerorul.data.schema import SUBSETS
from api.inference import engine_history, fleet_summary, model_comparison

DATA_DIR = PROJECT_ROOT / "dashboard" / "public" / "data"


def run(subset: str, data_dir: Path = DATA_DIR) -> int:
    subset_dir = data_dir / subset
    engines_dir = subset_dir / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)

    fleet = fleet_summary(subset)
    (subset_dir / "fleet.json").write_text(json.dumps(fleet, indent=2))

    comparison = model_comparison(subset)
    if comparison is not None:
        (subset_dir / "models.json").write_text(json.dumps(comparison, indent=2))

    for row in fleet:
        history = engine_history(subset, row["unit_number"])
        (engines_dir / f"{row['unit_number']}.json").write_text(json.dumps(history))

    print(
        f"[{subset}] wrote fleet.json ({len(fleet)} engines), "
        f"{'models.json, ' if comparison else '(no models.json -- run compare_models.py) '}"
        f"{len(fleet)} engine detail files"
    )
    return len(fleet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=[*SUBSETS, "all"])
    args = parser.parse_args()

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    available = []
    for subset in subsets:
        try:
            run(subset)
            available.append(subset)
        except FileNotFoundError as exc:
            print(f"[{subset}] skipped: {exc}")

    manifest_path = DATA_DIR / "manifest.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"available_subsets": available}, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
