"""Build ``data/processed/game_features_weak_stack_fluview.parquet``.

Pure additive merge-by-``game_id`` enrichment of the PRODUCTION
``game_features_weak_stack.parquet`` with the two FluView elevated-illness
indicator columns from
``nfl_ats.fluview_production_feature.attach_fluview_elevated_features``. See
``docs/fluview_on_production.md``.

Built on the production table on purpose, NOT on
``game_features_weak_stack_surface/_v3/_v4/_graph_sack.parquet``: the
question is whether the FluView feature adds to PRODUCTION, and stacking it
onto a profile already refused or still undecided at the opener would
confound the answer -- the same reason ``scripts/build_weak_stack_v4_table.py``
and ``scripts/build_weak_stack_graph_sack_table.py`` give for their own
choice.

Never touches ``game_features_weak_stack.parquet`` or any other existing file.

Reuses the already-ingested FluView snapshot
(``data/raw/fluview/*/fluview_raw.parquet``) and the frozen battery's own
recorded per-state thresholds (``artifacts/fluview_battery/*/results.json``)
-- no new API calls.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_fluview_table.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SOURCE = REPO / "data/processed/game_features_weak_stack.parquet"
DEST = REPO / "data/processed/game_features_weak_stack_fluview.parquet"


def main() -> None:
    from nfl_ats.fluview_production_feature import (
        FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS,
        attach_fluview_elevated_features,
        default_fluview_raw_path,
        default_fluview_results_path,
    )

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")
    print(f"fluview raw snapshot: {default_fluview_raw_path()}")
    print(f"frozen thresholds from: {default_fluview_results_path()}")

    started = time.time()
    widened = attach_fluview_elevated_features(base)
    print(f"fluview features computed in {time.time() - started:.1f}s")

    new_cols = sorted(set(widened.columns) - set(base.columns))
    print(f"new columns ({len(new_cols)}): {new_cols}")
    assert new_cols == sorted(FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS), (
        f"expected exactly the two frozen columns, got {new_cols}"
    )

    pre_existing = [c for c in base.columns if c in widened.columns]
    pd.testing.assert_frame_equal(base[pre_existing], widened[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    for column in FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS:
        coverage = widened[column].notna().mean()
        print(f"coverage: {coverage:.3%} of {len(widened)} rows have a non-missing {column}")

    coverage_by_season = (
        widened.assign(_notna=widened[FLUVIEW_ELEVATED_ON_PRODUCTION_FEATURE_COLUMNS[0]].notna())
        .groupby("season")["_notna"]
        .mean()
    )
    print("coverage by season (home column):")
    for season, cov in coverage_by_season.items():
        print(f"  {int(season)}: {cov:.1%}")

    widened.to_parquet(DEST)
    print(f"wrote {DEST} rows={len(widened)} cols={len(widened.columns)}")


if __name__ == "__main__":
    main()
