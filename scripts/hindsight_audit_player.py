from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod22_qb_input_repair as m22  # noqa: E402

import nfl_ats.margin as margin  # noqa: E402
from nfl_ats.cli_commands import features as feature_cli  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/hindsight_audit_player"
ACTIVE = json.loads((ROOT / "artifacts/active_ats_model.json").read_text(encoding="utf-8"))
SET_KEY = margin.margin_feature_set("market_residual", ACTIVE["feature_profile"])
MARKERS = ("_qb_", "injury", "continuity", "active_roster")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    columns = [c for c in margin.FEATURE_SETS[SET_KEY] if any(m in c for m in MARKERS)]
    inputs = m22.load_inputs()
    canon = feature_cli.canonicalize_injuries(
        inputs["injuries"], timestamp_fallback=inputs["fallback"], first_seen=inputs["first_seen"]
    )
    canon = canon.loc[canon["season"].between(2019, 2026)]
    basis = pd.crosstab(canon["season"], canon["observed_at_basis"])
    modified = canon.loc[canon["observed_at_basis"].eq("date_modified"), "date_modified"]
    rosters = inputs["rosters"]
    scoped = rosters.loc[rosters["season"].between(2019, 2025)]
    within = scoped.groupby(["season", "gsis_id"])["years_exp"].nunique()
    per_season = (
        scoped.groupby(["gsis_id", "season"])["years_exp"].first().reset_index()
    ).sort_values(["gsis_id", "season"])
    per_season["step"] = per_season.groupby("gsis_id")["years_exp"].diff()
    per_season["gap"] = per_season.groupby("gsis_id")["season"].diff()
    depth = inputs["depth"]
    depth_seen = pd.to_datetime(depth["observed_at_utc"], utc=True, errors="coerce")
    rates = inputs["rates"]
    report = {
        "columns": columns,
        "injury_basis_by_season": {
            str(k): {c: int(v) for c, v in row.items()} for k, row in basis.iterrows()
        },
        "injury_rows_per_player_week_max": int(
            canon.groupby(["season", "week", "team", "gsis_id"]).size().max()
        ),
        "date_modified_weekday_counts": {
            str(k): int(v) for k, v in modified.dt.dayofweek.value_counts().sort_index().items()
        },
        "roster_years_exp_varying_within_season": int((within > 1).sum()),
        "roster_player_seasons": len(within),
        "roster_years_exp_step_consecutive": {
            str(k): int(v)
            for k, v in per_season.loc[per_season["gap"].eq(1), "step"].value_counts().items()
        },
        "depth_rows": len(depth),
        "depth_first_observed": str(depth_seen.min()),
        "depth_last_observed": str(depth_seen.max()),
        "availability_rates_source_before_target": bool(
            (rates["source_end_season"] < rates["target_season"]).all()
        ),
    }
    (OUT / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
