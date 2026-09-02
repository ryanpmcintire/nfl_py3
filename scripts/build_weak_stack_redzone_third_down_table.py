"""Build ``data/processed/game_features_weak_stack_redzone_third_down.parquet``.

Pure additive merge-by-``game_id`` enrichment of the PRODUCTION
``game_features_weak_stack.parquet`` with the one third-down mean-reversion
fade column from
``nfl_ats.redzone_reversion_production_feature.attach_redzone_third_down_features``.
See ``docs/redzone_reversion_on_production.md``. Mirrors
``scripts/build_weak_stack_illness_table.py``.

Built on the production table on purpose, NOT on
``game_features_weak_stack_surface/_v3/_v4/_graph_*/_fluview/_illness.parquet``:
the question is whether the fade column adds to PRODUCTION, and stacking it
onto a profile already refused or still undecided at the opener would confound
the answer -- the same reason ``scripts/build_weak_stack_v4_table.py`` gives
for its own choice.

Never touches ``game_features_weak_stack.parquet`` or any other existing file.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_redzone_third_down_table.py
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
DEST = REPO / "data/processed/game_features_weak_stack_redzone_third_down.parquet"


def main() -> None:
    from nfl_ats.redzone_reversion_production_feature import (
        REDZONE_REVERSION_ON_PRODUCTION_FEATURE_COLUMNS,
        attach_redzone_third_down_features,
    )

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")

    started = time.time()
    widened = attach_redzone_third_down_features(base)
    print(f"red-zone reversion feature computed in {time.time() - started:.1f}s")

    new_cols = sorted(set(widened.columns) - set(base.columns))
    expected = sorted(REDZONE_REVERSION_ON_PRODUCTION_FEATURE_COLUMNS)
    assert new_cols == expected, f"expected exactly {expected}, got {new_cols}"

    pre_existing = [c for c in base.columns if c in widened.columns]
    pd.testing.assert_frame_equal(base[pre_existing], widened[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    for column in expected:
        values = widened[column]
        covered = values.dropna()
        coverage = values.notna().mean()
        print(f"{column}: coverage {coverage:.3%} of {len(widened)} rows ({len(covered)} covered)")
        for state in (-1.0, 0.0, 1.0):
            hits = int((covered == state).sum())
            share = float((covered == state).mean()) if len(covered) else float("nan")
            print(f"  value {state:+.0f}: {share:.3%} of covered rows ({hits})")
        nonzero = float((covered != 0.0).mean()) if len(covered) else float("nan")
        print(f"  non-zero on {nonzero:.3%} of covered rows (construction predicts ~37.5%)")
        seasons_missing = sorted(widened.loc[values.isna(), "season"].astype(int).unique().tolist())
        print(f"  seasons carrying any missing value: {seasons_missing}")

    widened.to_parquet(DEST)
    print(f"wrote {DEST} rows={len(widened)} cols={len(widened.columns)}")


if __name__ == "__main__":
    main()
