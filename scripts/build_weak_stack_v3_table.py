"""Build ``data/processed/game_features_weak_stack_v3.parquet``.

Pure additive merge-by-``game_id`` enrichment of the already-built
``game_features_weak_stack_surface.parquet`` (weak_stack + surface_switch_flag)
with the three new gap_v3 sub-families from
``nfl_ats.weak_stack_v3_features.attach_weak_stack_v3_gap_features`` --
division revenge, sandwich spot, post-blowout letdown/bounce, penalty rate,
and the two travel/rest flags. See ``docs/weak_stack_v3.md``.

Never touches ``game_features_weak_stack.parquet`` (the production table) or
any other existing file. Measure-only precedent, matching
``scripts/surface_profile_opener_eval.py``'s own note that the surface table
was built as "a pure, additive merge-by-game_id enrichment of the existing
weak_stack table".

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_v3_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

SOURCE = REPO / "data/processed/game_features_weak_stack_surface.parquet"
DEST = REPO / "data/processed/game_features_weak_stack_v3.parquet"


def main() -> None:
    from nfl_ats.weak_stack_v3_features import attach_weak_stack_v3_gap_features

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")

    v3 = attach_weak_stack_v3_gap_features(base, repo_root=REPO)
    new_cols = sorted(set(v3.columns) - set(base.columns))
    print(f"new columns ({len(new_cols)}): {new_cols}")

    pre_existing = [c for c in base.columns if c in v3.columns]
    pd.testing.assert_frame_equal(base[pre_existing], v3[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    v3.to_parquet(DEST)
    print(f"wrote {DEST} rows={len(v3)} cols={len(v3.columns)}")


if __name__ == "__main__":
    main()
