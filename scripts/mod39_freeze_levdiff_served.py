import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts", "src"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import sim09_stakes as stakes  # noqa: E402

TERM = "mod39_levdiff_served"
FIT_FIRST = 2020
FIT_LAST = 2025
TARGET_SEASON = 2026
DOMAIN_WEEKS = (10, 16)
OFFSET_COEFFICIENT = 1.0
SERVED_COLUMN = "out_of_season_home_probability"
HISTORICAL_OFFSET_COLUMN = "home_cover_probability_excluding_push_at_open"
PICK = ROOT / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
RATING = ROOT / "artifacts" / "sim09" / "stakes_summary.json"
OUTPATH = ROOT / "artifacts" / "forward_terms" / TERM / "frozen_coefficients.json"
HASH_KEYS = ("b_logit_per_unit", "intercept", "served_logit_coefficient", "rating_params")
SCORE_KEYS = (
    "logloss_gain",
    "brier_gain",
    "acc_gain",
    "acc_full",
    "acc_ctl",
    "logloss_full",
    "logloss_ctl",
    "decisive_games",
    "decisive_full_wins",
    "decisive_ctl_wins",
    "insample_logloss_gain",
    "gap_logloss",
)
FRAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "hi",
    "ai",
    "margin",
    "neutral",
]


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def season_frame(sc, season, week):
    s = sc[sc.season.isin([season - 1, season]) & (sc.game_type == "REG")].copy()
    s["home_team"] = stakes.norm_team(s.home_team)
    s["away_team"] = stakes.norm_team(s.away_team)
    s = s.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    s["hi"] = s.home_team.map(stakes.TI)
    s["ai"] = s.away_team.map(stakes.TI)
    assert bool(s.hi.notna().all()) and bool(s.ai.notna().all())
    s["hi"] = s.hi.astype(int)
    s["ai"] = s.ai.astype(int)
    s["margin"] = s.home_score - s.away_score
    s["neutral"] = (s.location == "Neutral").astype(float)
    s.loc[(s.season == season) & (s.week >= week), "margin"] = np.nan
    return s[FRAME_COLUMNS]


def week_levdiff(sc, season, week, params, sims):
    reg = season_frame(sc, season, week)
    prior = reg[reg.season == season - 1]
    cur = reg[reg.season == season].reset_index(drop=True)
    played = cur[cur.week < week]
    if prior.empty or bool(prior.margin.isna().any()):
        raise RuntimeError(f"MISSING INPUT: complete {season - 1} regular-season results")
    if played.empty or bool(played.margin.isna().any()):
        raise RuntimeError(f"MISSING INPUT: {season} results through week {week - 1}")
    assert bool(cur[cur.week >= week].margin.isna().all())
    tabs, last = stakes.strength_tables(reg)
    seed_key = [int(season), int(week)]
    stakes.RNG = np.random.default_rng(seed_key)
    seed, x, rem, _ = stakes.simulate(cur, season, week, tabs, last, params, sims)
    made = seed > 0
    column = {idx: i for i, idx in enumerate(rem.index)}
    rows = {}
    for idx, gm in rem[rem.week == week].iterrows():
        win = x[:, column[idx]]
        lev = {}
        for side, team, won in (("home", gm.hi, win), ("away", gm.ai, ~win)):
            a = made[won, team].mean() if won.any() else np.nan
            b = made[~won, team].mean() if (~won).any() else np.nan
            lev[side] = 0.0 if np.isnan(a) or np.isnan(b) else float(a - b)
        rows[gm.game_id] = {
            "lev_home": lev["home"],
            "lev_away": lev["away"],
            "levdiff": lev["home"] - lev["away"],
            "p_make_home": float(made[:, gm.hi].mean()),
            "p_make_away": float(made[:, gm.ai].mean()),
        }
    meta = {
        "rng_seed": seed_key,
        "simulations": int(sims),
        "standings_games": len(played),
        "results_through_week": int(played.week.max()),
    }
    return rows, meta


def main():
    import mod39_leverage as m39

    g = m39.games()
    op = g[g.season.between(FIT_FIRST, FIT_LAST)].dropna(
        subset=["margin_vs_open", HISTORICAL_OFFSET_COLUMN]
    )
    pk = pd.read_parquet(PICK, columns=["game_id", SERVED_COLUMN])
    d = op.merge(pk, on="game_id", how="inner").reset_index(drop=True)
    assert int(d[SERVED_COLUMN].isna().sum()) == 0
    assert int(d.season.min()) == FIT_FIRST and int(d.season.max()) == FIT_LAST
    assert int((d.margin_vs_open == 0).sum()) == 0
    served = d.assign(**{HISTORICAL_OFFSET_COLUMN: d[SERVED_COLUMN]})
    rep = m39.loso_report(served, ["levdiff"])
    control = m39.loso_report(served, [])
    folds = rep["coefs_by_heldout_season"]
    slope_folds = {str(s): float(v[1]) for s, v in folds.items()}
    intercept_folds = {str(s): float(v[0]) for s, v in folds.items()}
    body = {
        "term": TERM,
        "term_definition": "home minus away of the playoff-chance swing on this game: each team's simulated chance of making the playoffs if it wins this game minus its chance if it loses, from a standings simulation over the remaining schedule that uses results through the previous week and margin-based team ratings; Weeks 10 to 16 only; added with an intercept to the logit of the served home probability, which enters as an offset",  # noqa: E501
        "b_logit_per_unit": float(rep["insample_coefs"][1]),
        "intercept": float(rep["insample_coefs"][0]),
        "served_logit_coefficient": OFFSET_COEFFICIENT,
        "control_intercept": float(control["insample_coefs"][0]),
        "loso_coefficient_by_heldout_season": slope_folds,
        "loso_coefficient_min": float(min(slope_folds.values())),
        "loso_coefficient_max": float(max(slope_folds.values())),
        "loso_intercept_by_heldout_season": intercept_folds,
        "loso_score_vs_intercept_only": {k: rep[k] for k in SCORE_KEYS},
        "rating_params": json.loads(RATING.read_text())["rating_params"],
        "rating_params_source": "artifacts/sim09/stakes_summary.json, fit on 2009-2017 margins by scripts/sim09_stakes.py",  # noqa: E501
        "rating_params_sha256": sha_file(RATING),
        "simulations": int(stakes.NSIM),
        "domain_weeks": list(DOMAIN_WEEKS),
        "levdiff_sd_fit_sample": float(d.levdiff.std()),
        "fit_seasons": [int(d.season.min()), int(d.season.max())],
        "fit_games": len(d),
        "opener_pushes_excluded": len(op) - len(d),
        "fit_base": f"logit of {SERVED_COLUMN} from the pick_probability artifact as an offset; opener grade; opener pushes excluded",  # noqa: E501
        "base_artifact": str(PICK.relative_to(ROOT)),
        "base_artifact_sha256": sha_file(PICK),
        "swing_artifact": str(m39.TW.relative_to(ROOT)),
        "swing_artifact_sha256": sha_file(m39.TW),
        "opener_artifact": str(m39.OPEN.relative_to(ROOT)),
        "fit_script": "scripts/mod39_freeze_levdiff_served.py",
        "historical_unit": "scripts/mod39_leverage.py, artifacts/mod39/mod39_results.json",
        "target_season": TARGET_SEASON,
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj({k: body[k] for k in HASH_KEYS})
    body["created_at_utc"] = datetime.now(UTC).isoformat()
    if OUTPATH.exists():
        raise SystemExit(f"already frozen: {OUTPATH}")
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps(body, indent=1))
    print(json.dumps({k: v for k, v in body.items() if k != "term_definition"}, indent=1))
    early = g[g.season < FIT_FIRST]
    context = {
        "historical_games": len(op),
        "historical_opener_pushes": int((op.margin_vs_open == 0).sum()),
        "levdiff_sd_historical_opener_sample": float(op.levdiff.std()),
        "levdiff_sd_before_first_fit_season": float(early.levdiff.std()),
        "games_before_first_fit_season": len(early),
    }
    print(json.dumps(context, indent=1))


if __name__ == "__main__":
    main()
