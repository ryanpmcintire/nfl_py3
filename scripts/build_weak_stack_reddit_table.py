"""Build ``data/processed/game_features_weak_stack_reddit.parquet``.

Pure additive merge-by-``game_id`` enrichment of the PRODUCTION
``game_features_weak_stack.parquet`` with the two Arctic Shift subreddit
fan-attention indicators from
``nfl_ats.reddit_attention_production_feature.attach_reddit_attention_features``.
See ``docs/reddit_attention_on_production.md``. Mirrors
``scripts/build_weak_stack_illness_table.py``.

Built on the production table on purpose, NOT on
``game_features_weak_stack_surface/_v3/_v4/_graph_*/_fluview/_illness.parquet``:
the question is whether the attention feature adds to PRODUCTION, and stacking
it onto a profile already refused or still undecided at the opener would
confound the answer -- the same reason ``scripts/build_weak_stack_v4_table.py``
gives for its own choice.

Never touches ``game_features_weak_stack.parquet`` or any other existing file.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_reddit_table.py
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
DEST = REPO / "data/processed/game_features_weak_stack_reddit.parquet"


def main() -> None:
    from nfl_ats.reddit_attention_production_feature import (
        REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS,
        attach_reddit_attention_features,
    )

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")

    started = time.time()
    widened = attach_reddit_attention_features(base)
    print(f"reddit attention features computed in {time.time() - started:.1f}s")

    new_cols = sorted(set(widened.columns) - set(base.columns))
    expected = sorted(REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS)
    assert new_cols == expected, f"expected exactly {expected}, got {new_cols}"

    pre_existing = [c for c in base.columns if c in widened.columns]
    pd.testing.assert_frame_equal(base[pre_existing], widened[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    for column in expected:
        values = widened[column]
        coverage = values.notna().mean()
        firing = values.mean() if values.notna().any() else float("nan")
        print(
            f"{column}: coverage {coverage:.3%} of {len(widened)} rows, "
            f"fires on {firing:.3%} of covered rows"
        )

    # Per-season coverage / firing inside the seasons a close-graded family can
    # draw, printed BEFORE any model is fit -- the disclosure
    # docs/reddit_attention_on_production.md section 6.1(c) commits to.
    reg = widened.loc[widened["game_type"].astype(str).eq("REG")]
    for column in expected:
        by_season = reg.groupby("season")[column].agg(["count", "mean", "size"])
        head = by_season.loc[by_season.index <= 2015]
        print(f"\n{column} by season (REG only, through 2015):")
        for season, row in head.iterrows():
            covered = int(row["count"])
            total = int(row["size"])
            rate = float(row["mean"]) if covered else float("nan")
            print(f"  {season}: covered {covered}/{total}, fires {rate:.3%}")

    widened.to_parquet(DEST)
    print(f"\nwrote {DEST} rows={len(widened)} cols={len(widened.columns)}")


if __name__ == "__main__":
    main()
