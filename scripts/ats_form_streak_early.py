from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ats_form_and_team_record as base

OUT = base.OUT / "streak_early.json"
REGRESS_STREAK = -3


def main():
    rng = np.random.default_rng(base.SEED + 2)
    wide, _, _ = base.build_games()
    home_cold = wide["home_ats_streak"].le(REGRESS_STREAK)
    away_cold = wide["away_ats_streak"].le(REGRESS_STREAK)
    wide["regress_flag"] = home_cold.astype(float) * (~away_cold) - away_cold.astype(float) * (
        ~home_cold
    )
    late = (base.EARLY_WEEKS[1] + 1, 22)
    screen = {
        "streak_diff_weeks_2_8": base.screen_term(wide, "ats_streak_diff", rng, base.EARLY_WEEKS),
        "streak_diff_weeks_9_plus": base.screen_term(wide, "ats_streak_diff", rng, late),
        "regress_flag_weeks_2_8": base.screen_term(wide, "regress_flag", rng, base.EARLY_WEEKS),
        "regress_flag_weeks_9_plus_descriptive": base.screen_term(wide, "regress_flag", rng, late),
    }
    early_flag = wide["week"].between(*base.EARLY_WEEKS)
    screen["regress_flag_fires_weeks_2_8"] = int(wide.loc[early_flag, "regress_flag"].ne(0).sum())
    pg = base.model_frame()
    pg = pg.merge(wide[["game_id", "ats_streak_diff", "regress_flag"]], on="game_id", how="left")
    early = pg["week"].between(*base.EARLY_WEEKS).astype(float)
    pg["early_streak_diff"] = pg["ats_streak_diff"].fillna(0.0) * early
    pg["early_regress_flag"] = pg["regress_flag"].fillna(0.0) * early
    model = base.on_model(pg, ("early_streak_diff", "early_regress_flag"), rng)
    model["early_flag_fires"] = int(pg["early_regress_flag"].ne(0).sum())
    res = {"screen_vs_opener_2011_2025": screen, "on_model_2020_2025": model}
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
