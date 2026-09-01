"""Build ``data/processed/game_features_weak_stack_graph_sack.parquet``.

Pure additive merge-by-``game_id`` enrichment of the PRODUCTION
``game_features_weak_stack.parquet`` with the one graph-propagated
``off_sack_rate`` column from
``nfl_ats.graph_team_stat_production_feature.attach_graph_off_sack_rate_feature``.
See ``docs/graph_team_stat_on_production.md``.

Built on the production table on purpose, NOT on
``game_features_weak_stack_surface/_v3/_v4.parquet``: the question is whether
the graph feature adds to PRODUCTION, and stacking it onto a profile already
refused or still undecided at the opener would confound the answer -- the
same reason ``scripts/build_weak_stack_v4_table.py`` gives for its own choice.

Never touches ``game_features_weak_stack.parquet`` or any other existing file.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_graph_sack_table.py
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
DEST = REPO / "data/processed/game_features_weak_stack_graph_sack.parquet"


def main() -> None:
    from nfl_ats.graph_team_stat_production_feature import (
        GRAPH_OFF_SACK_RATE_COLUMN,
        attach_graph_off_sack_rate_feature,
    )

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")

    started = time.time()
    widened = attach_graph_off_sack_rate_feature(base)
    print(f"graph feature computed in {time.time() - started:.1f}s")

    new_cols = sorted(set(widened.columns) - set(base.columns))
    print(f"new columns ({len(new_cols)}): {new_cols}")
    assert new_cols == [GRAPH_OFF_SACK_RATE_COLUMN], (
        f"expected exactly one new column, got {new_cols}"
    )

    pre_existing = [c for c in base.columns if c in widened.columns]
    pd.testing.assert_frame_equal(base[pre_existing], widened[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    coverage = widened[GRAPH_OFF_SACK_RATE_COLUMN].notna().mean()
    print(f"coverage: {coverage:.3%} of {len(widened)} rows have a rated graph value")

    widened.to_parquet(DEST)
    print(f"wrote {DEST} rows={len(widened)} cols={len(widened.columns)}")


if __name__ == "__main__":
    main()
