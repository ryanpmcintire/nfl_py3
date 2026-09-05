r"""Build ``data/processed/game_features_weak_stack_oracle_weather.parquet``.

POSITIVE CONTROL ONLY. Additive merge-by-``game_id`` enrichment of the
PRODUCTION ``game_features_weak_stack.parquet`` with the weather that ACTUALLY
happened (``nfl_ats.forecast_weather_features.attach_observed_weather_features``).

The resulting table is deliberately LEAKY and must never be promoted or
published from. It exists to answer the one question a better forecast cannot:
if the model is handed weather of infinite forecast skill, does forced-pick
accuracy move at all? If not, the whole weather channel is bounded by a
positive control -- one of only two admissible closing grounds in AGENTS.md.
If it does, the oracle-minus-forecast gap is exactly the headroom a better
wind source could buy. See ``docs/weak_stack_v4.md``.

Run:  .\.tools\uv.exe run --no-sync python scripts/build_weak_stack_oracle_weather_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import stamp_sidecar  # noqa: E402

SOURCE = REPO / "data/processed/game_features_weak_stack.parquet"
DEST = REPO / "data/processed/game_features_weak_stack_oracle_weather.parquet"


def main() -> None:
    from nfl_ats.forecast_weather_features import (
        OBSERVED_WEATHER_COLUMNS,
        attach_observed_weather_features,
    )

    base = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(base)} cols={len(base.columns)}")

    oracle = attach_observed_weather_features(base, repo_root=REPO)
    new_cols = sorted(set(oracle.columns) - set(base.columns))
    print(f"new columns ({len(new_cols)}): {new_cols}")
    assert set(new_cols) == set(OBSERVED_WEATHER_COLUMNS)

    pre_existing = [c for c in base.columns if c in oracle.columns]
    pd.testing.assert_frame_equal(base[pre_existing], oracle[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    present = oracle["observed_temp_f"].notna()
    print(f"observed-weather coverage: {int(present.sum())}/{len(oracle)} ({present.mean():.1%})")

    oracle.to_parquet(DEST)
    stamp_sidecar(DEST)  # ENG-38
    print(f"wrote {DEST} rows={len(oracle)} cols={len(oracle.columns)}")


if __name__ == "__main__":
    main()
