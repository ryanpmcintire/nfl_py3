"""Build the two Phase 12 market-lead ``weak_stack`` tables.

Pure additive merge-by-``game_id`` enrichment of the PRODUCTION
``game_features_weak_stack.parquet`` with the two market microstructure
columns from ``nfl_ats.market_lead_features`` (LEAD-05 opener-softness fade,
LEAD-03 moneyline-spread divergence). See ``docs/market_lead_battery.md``.
Mirrors ``scripts/build_weak_stack_illness_table.py``.

Built on the production table directly, NOT on any other candidate profile:
the question is whether each market column adds to what is actually PLAYED,
and stacking it onto a profile already refused or still undecided would
confound the answer.

Never touches ``game_features_weak_stack.parquet`` or any other existing
file. Reads only the committed local snapshots under ``data/market/raw/`` --
the paid Odds API is never called.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_market_lead_tables.py
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
DEST_OPENER_SOFTNESS = REPO / "data/processed/game_features_weak_stack_opener_softness.parquet"
DEST_ML_DIVERGENCE = REPO / "data/processed/game_features_weak_stack_ml_divergence.parquet"


def _build_one(
    base: pd.DataFrame, *, attach, expected_columns: tuple[str, ...], dest: Path
) -> None:
    started = time.time()
    widened = attach(base)
    print(f"  computed in {time.time() - started:.1f}s")

    new_cols = sorted(set(widened.columns) - set(base.columns))
    expected = sorted(expected_columns)
    assert new_cols == expected, f"expected exactly {expected}, got {new_cols}"

    pre_existing = [c for c in base.columns if c in widened.columns]
    pd.testing.assert_frame_equal(base[pre_existing], widened[pre_existing], check_exact=True)
    print("  additivity check passed: every pre-existing column is bit-identical")

    for column in expected:
        values = widened[column]
        coverage = values.notna().mean()
        print(f"  {column}: coverage {coverage:.3%} of {len(widened)} rows")
        if values.notna().any():
            counts = values.dropna().value_counts().sort_index()
            print(f"    value counts: {counts.to_dict()}")

    widened.to_parquet(dest)
    print(f"  wrote {dest} rows={len(widened)} cols={len(widened.columns)}")


def main() -> None:
    from nfl_ats.market_lead_features import (
        ML_SPREAD_DIVERGENCE_COLUMN,
        OPENER_SOFTNESS_FADE_COLUMN,
        attach_ml_spread_divergence_features,
        attach_opener_softness_fade_features,
    )

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")

    print("\nopener_softness_fade_signal (LEAD-05):")
    _build_one(
        base,
        attach=attach_opener_softness_fade_features,
        expected_columns=(OPENER_SOFTNESS_FADE_COLUMN,),
        dest=DEST_OPENER_SOFTNESS,
    )

    print("\nml_spread_divergence_signal (LEAD-03):")
    _build_one(
        base,
        attach=attach_ml_spread_divergence_features,
        expected_columns=(ML_SPREAD_DIVERGENCE_COLUMN,),
        dest=DEST_ML_DIVERGENCE,
    )


if __name__ == "__main__":
    main()
