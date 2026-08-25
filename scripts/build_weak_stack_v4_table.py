"""Build ``data/processed/game_features_weak_stack_v4.parquet``.

Pure additive merge-by-``game_id`` enrichment of the PRODUCTION
``game_features_weak_stack.parquet`` with the six forecast-weather columns
from ``nfl_ats.forecast_weather_features.attach_forecast_weather_features``.
See ``docs/weak_stack_v4.md``.

Built on the production table on purpose, NOT on
``game_features_weak_stack_surface.parquet`` or the v3 table: the question is
whether forecast weather adds to PRODUCTION, and stacking it onto a profile
already refused at the opener would confound the answer.

Never touches ``game_features_weak_stack.parquet`` or any other existing file.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_v4_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SOURCE = REPO / "data/processed/game_features_weak_stack.parquet"
DEST = REPO / "data/processed/game_features_weak_stack_v4.parquet"


def main() -> None:
    from nfl_ats.forecast_weather_features import (
        FORECAST_WEATHER_COLUMNS,
        attach_forecast_weather_features,
        coverage_summary,
    )

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")

    v4 = attach_forecast_weather_features(base, repo_root=REPO)
    new_cols = sorted(set(v4.columns) - set(base.columns))
    print(f"new columns ({len(new_cols)}): {new_cols}")
    assert set(new_cols) == set(FORECAST_WEATHER_COLUMNS), (
        f"expected exactly the declared family, got {new_cols}"
    )

    pre_existing = [c for c in base.columns if c in v4.columns]
    pd.testing.assert_frame_equal(base[pre_existing], v4[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    print("coverage:", coverage_summary(v4))

    v4.to_parquet(DEST)
    print(f"wrote {DEST} rows={len(v4)} cols={len(v4.columns)}")


if __name__ == "__main__":
    main()
