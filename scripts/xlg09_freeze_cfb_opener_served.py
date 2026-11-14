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

import opener_error_transfer_unit3 as u3  # noqa: E402

TERM = "xlg09_cfb_opener_served"
FIT_FIRST = 2020
FIT_LAST = 2025
TARGET_SEASON = 2026
DOMAIN_WEEKS = (11, 18)
OFFSET_COEFFICIENT = 1.0
SERVED_COLUMN = "out_of_season_home_probability"
HISTORICAL_OFFSET_COLUMN = "home_cover_probability_excluding_push_at_open"
TERM_COLUMN = "cfb_transfer_logit"
PICK = ROOT / "artifacts" / "pick_probability" / "20261114T184614Z" / "per_game.parquet"
OUTPATH = ROOT / "artifacts" / "forward_terms" / TERM / "frozen_coefficients.json"
HASH_KEYS = ("b_logit_per_unit", "intercept", "served_logit_coefficient", "college_rule")
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
TERM_DEFINITION = (
    "logit of a home-cover rule learned on college football openers only (ridge logistic on "
    "home favourite flag, spread size, distance to the nearest key number, home minus away "
    "prior-game line move, home minus away rest days; college seasons before the target NFL "
    "season), evaluated on the NFL game's Tuesday pool line with college means and standard "
    "deviations; added with an intercept to the logit of the served home probability, which "
    "enters as an offset"
)
FORWARD_INPUTS = {
    "opener": (
        "market_spread of the last card record published at or before the pick deadline and "
        "the run time, artifacts/clv_ledger/published_picks.parquet, home spread positive "
        "when the home team is favoured"
    ),
    "prior_game": (
        "each team's most recent earlier regular-season game of the same season in the newest "
        "schedule snapshot; it must carry a result there, else the term fails closed"
    ),
    "prior_move": (
        "prior game's schedule closing spread minus its published card line, signed for the "
        "team (negated when the team was away); zero when the team has no earlier game"
    ),
    "rest_days": "calendar days from the prior game's date to this game's date",
    "difference_from_history": (
        "the 2020-25 fit took the prior game from the fitted population, which leaves out "
        "opener pushes and games without an archived opener, and took the close from the "
        "captured consensus where one existed; forward uses the schedule close only"
    ),
}


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def college_inputs():
    paths = []
    for season in u3.CFB_TRAIN_SEASONS_ALL:
        paths.append(u3.CFB_LINES_ROOT / f"season={season}" / "lines.parquet")
        paths.append(u3.CFB_SCHEDULES_ROOT / f"season={season}" / "schedules.parquet")
    return paths


def college_rule(cfb, before_season):
    train = cfb.loc[cfb["season"].lt(before_season)]
    means, stds = u3.standardize(train, u3.FEATURES)
    beta = u3.fit_logit_beta(
        u3.design(train, u3.FEATURES, means, stds), train["home_covered"].to_numpy()
    )
    return {
        "features": list(u3.FEATURES),
        "means": means,
        "stds": stds,
        "beta": [float(v) for v in beta],
        "ridge": float(u3.RIDGE),
        "key_numbers": [float(k) for k in u3.KEY_NUMBERS],
        "train_seasons": sorted(int(s) for s in train["season"].unique()),
        "train_games": len(train),
    }


def rule_values(frame, rule):
    beta = np.asarray(rule["beta"], dtype=float)
    return u3.predict_linear(frame, rule["features"], beta, rule["means"], rule["stds"])


def card_lines(published, season, now):
    pub = published.loc[published["season"].eq(season)].copy()
    pub["published_at_utc"] = pd.to_datetime(pub["published_at_utc"], utc=True)
    pub["pick_deadline_utc"] = pd.to_datetime(pub["pick_deadline_utc"], utc=True)
    pub = pub.loc[
        pub["published_at_utc"].le(pub["pick_deadline_utc"]) & pub["published_at_utc"].le(now)
    ].sort_values("published_at_utc")
    last = pub.groupby("game_id").tail(1).set_index("game_id")
    first = pub.groupby("game_id").head(1).set_index("game_id")
    return pd.DataFrame(
        {
            "line": last["market_spread"].astype(float),
            "published_at_utc": last["published_at_utc"],
            "first_line": first["market_spread"].astype(float),
        }
    )


def team_prior(reg, lines, team, gameday):
    earlier = reg.loc[
        (reg["home_team"].eq(team) | reg["away_team"].eq(team)) & reg["gameday"].lt(gameday)
    ]
    if earlier.empty:
        return None
    prev = earlier.sort_values("gameday").iloc[-1]
    if pd.isna(prev["result"]) or pd.isna(prev["spread_line"]):
        raise RuntimeError(f"MISSING INPUT: result and closing line for {prev['game_id']}")
    if prev["game_id"] not in lines.index:
        raise RuntimeError(f"MISSING INPUT: published card line for {prev['game_id']}")
    move = float(prev["spread_line"]) - float(lines.loc[prev["game_id"], "line"])
    return {
        "game_id": str(prev["game_id"]),
        "week": int(prev["week"]),
        "signed_move": move if prev["home_team"] == team else -move,
        "rest_days": float((gameday - prev["gameday"]).days),
    }


def week_terms(schedule, published, season, week, rule, now):
    assert [float(k) for k in u3.KEY_NUMBERS] == rule["key_numbers"]
    reg = schedule.loc[schedule["season"].eq(season) & schedule["game_type"].eq("REG")].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    lines = card_lines(published, season, now)
    rows = {}
    missing = []
    prior_weeks = []
    for game in reg.loc[reg["week"].eq(week)].itertuples():
        if game.game_id not in lines.index:
            missing.append({"game_id": game.game_id, "reason": "no published card line"})
            continue
        home = team_prior(reg, lines, game.home_team, game.gameday)
        away = team_prior(reg, lines, game.away_team, game.gameday)
        if (home is None) != (away is None):
            missing.append({"game_id": game.game_id, "reason": "one team has no earlier game"})
            continue
        opener = float(lines.loc[game.game_id, "line"])
        prior_weeks.extend(p["week"] for p in (home, away) if p is not None)
        rows[game.game_id] = {
            "opener_home_spread": opener,
            "opener_published_at_utc": lines.loc[game.game_id, "published_at_utc"].isoformat(),
            "opener_first_published_line": float(lines.loc[game.game_id, "first_line"]),
            "home_favorite": float(opener > 0.0),
            "spread_size": abs(opener),
            "prior_move_diff": (home["signed_move"] - away["signed_move"]) if home else 0.0,
            "rest_diff": (home["rest_days"] - away["rest_days"]) if home else 0.0,
            "home_prior_game": home["game_id"] if home else None,
            "away_prior_game": away["game_id"] if away else None,
            "home_prior_signed_move": home["signed_move"] if home else None,
            "away_prior_signed_move": away["signed_move"] if away else None,
        }
    if rows:
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame["key_number_distance"] = u3.key_number_distance(frame["spread_size"])
        values = rule_values(frame, rule)
        for game_id, distance, value in zip(
            frame.index, frame["key_number_distance"], values, strict=True
        ):
            rows[game_id]["key_number_distance"] = float(distance)
            rows[game_id][TERM_COLUMN] = float(value)
    meta = {
        "trait_data_through_week": max(prior_weeks) if prior_weeks else None,
        "card_lines_changed_after_first_publication": sorted(
            g
            for g, r in rows.items()
            if r["opener_first_published_line"] != r["opener_home_spread"]
        ),
        "close_source": "spread_line of the completed prior game in the schedule snapshot",
        "opener_source": "artifacts/clv_ledger/published_picks.parquet market_spread",
    }
    return rows, missing, meta


def schedule_rest_mismatch(frame):
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    schedule, _ = load_snapshot(latest_snapshot(ROOT / "data" / "raw"))
    sched = schedule.loc[schedule["season"].isin(frame["season"].unique())].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    parts = [
        sched[["season", "gameday", side]].rename(columns={side: "team"})
        for side in ("home_team", "away_team")
    ]
    long = pd.concat(parts, ignore_index=True).sort_values(["team", "season", "gameday"])
    long["rest"] = long.groupby(["team", "season"])["gameday"].diff().dt.days
    true_rest = long.set_index(["team", "season", "gameday"])["rest"]
    out = {"rows": len(frame)}
    differs = np.zeros(len(frame), dtype=bool)
    unmatched = 0
    for side in ("home", "away"):
        keys = list(zip(frame[f"{side}_team"], frame["season"], frame["gameday"], strict=True))
        unmatched += sum(1 for k in keys if k not in true_rest.index)
        actual = true_rest.reindex(keys).to_numpy(dtype=float)
        fitted = frame[f"{side}_rest"].to_numpy(dtype=float)
        differs |= ~((actual == fitted) | (np.isnan(actual) & np.isnan(fitted)))
    out["rows_whose_fitted_prior_game_is_not_the_schedule_prior_game"] = int(differs.sum())
    out["team_rows_not_found_in_schedule"] = int(unmatched)
    return out


def main():
    import mod39_leverage as m39

    nfl, _, _ = u3.load_extended_nfl_population()
    cfb = u3.load_cfb_population()
    nfl = nfl.loc[nfl["season"].between(FIT_FIRST, FIT_LAST)].reset_index(drop=True)
    nfl[TERM_COLUMN] = np.nan
    fold_rules = {}
    for held in sorted(int(s) for s in nfl["season"].unique()):
        rule = college_rule(cfb, held)
        rows = nfl["season"].eq(held)
        nfl.loc[rows, TERM_COLUMN] = rule_values(nfl.loc[rows], rule)
        fold_rules[str(held)] = {k: rule[k] for k in ("beta", "train_seasons", "train_games")}
    pk = pd.read_parquet(
        PICK, columns=["game_id", SERVED_COLUMN, "margin_vs_open", "tue_open_home_spread"]
    )
    pk["game_id"] = pk["game_id"].astype(str)
    d = nfl.merge(pk, on="game_id", how="inner").reset_index(drop=True)
    assert len(d) == len(nfl)
    assert int(d[SERVED_COLUMN].isna().sum()) == 0 and int(d[TERM_COLUMN].isna().sum()) == 0
    assert int(d.season.min()) == FIT_FIRST and int(d.season.max()) == FIT_LAST
    assert int((d.margin_vs_open == 0).sum()) == 0
    assert bool(np.isclose(d.home_open, d.tue_open_home_spread).all())
    assert bool(((d.margin_vs_open > 0).astype(float) == d.home_covered).all())
    served = d.assign(**{HISTORICAL_OFFSET_COLUMN: d[SERVED_COLUMN]})
    rep = m39.loso_report(served, [TERM_COLUMN])
    control = m39.loso_report(served, [])
    folds = rep["coefs_by_heldout_season"]
    slope_folds = {str(s): float(v[1]) for s, v in folds.items()}
    intercept_folds = {str(s): float(v[0]) for s, v in folds.items()}
    forward_rule = college_rule(cfb, TARGET_SEASON)
    assert max(forward_rule["train_seasons"]) < TARGET_SEASON
    body = {
        "term": TERM,
        "term_definition": TERM_DEFINITION,
        "b_logit_per_unit": float(rep["insample_coefs"][1]),
        "intercept": float(rep["insample_coefs"][0]),
        "served_logit_coefficient": OFFSET_COEFFICIENT,
        "control_intercept": float(control["insample_coefs"][0]),
        "loso_coefficient_by_heldout_season": slope_folds,
        "loso_coefficient_min": float(min(slope_folds.values())),
        "loso_coefficient_max": float(max(slope_folds.values())),
        "loso_intercept_by_heldout_season": intercept_folds,
        "loso_score_vs_intercept_only": {k: rep[k] for k in SCORE_KEYS},
        "college_rule": forward_rule,
        "college_rule_by_fit_season": fold_rules,
        "college_inputs_sha256": sha_obj([sha_file(p) for p in college_inputs()]),
        "college_lines_root": str(u3.CFB_LINES_ROOT.relative_to(ROOT)),
        "college_schedules_root": str(u3.CFB_SCHEDULES_ROOT.relative_to(ROOT)),
        "college_book_by_season": {str(k): v for k, v in u3.CFB_BOOK_BY_SEASON.items()},
        "forward_inputs": FORWARD_INPUTS,
        "domain_weeks": list(DOMAIN_WEEKS),
        "term_sd_fit_sample": float(d[TERM_COLUMN].std()),
        "term_abs_max_fit_sample": float(d[TERM_COLUMN].abs().max()),
        "prior_move_diff_sd_fit_sample": float(d["prior_move_diff"].std()),
        "fit_prior_game_check": schedule_rest_mismatch(d),
        "fit_seasons": [int(d.season.min()), int(d.season.max())],
        "fit_games": len(d),
        "fit_base": (
            f"logit of {SERVED_COLUMN} from the pick_probability artifact as an offset; "
            "true Tuesday opener grade; opener pushes are not in the population"
        ),
        "base_artifact": str(PICK.relative_to(ROOT)),
        "base_artifact_sha256": sha_file(PICK),
        "nfl_population": str(u3.EXTENDED_POPULATION_PATH.relative_to(ROOT)),
        "nfl_population_sha256": sha_file(u3.EXTENDED_POPULATION_PATH),
        "fit_script": "scripts/xlg09_freeze_cfb_opener_served.py",
        "fit_script_sha256": sha_file(__file__),
        "rule_script": "scripts/opener_error_transfer_unit3.py",
        "rule_script_sha256": sha_file(u3.__file__),
        "historical_unit": "scripts/opener_error_transfer_unit3.py (opener_error_transfer_v3)",
        "target_season": TARGET_SEASON,
        "no_2026_outcomes": True,
    }
    body = json.loads(json.dumps(body, default=str))
    body["coefficient_hash"] = sha_obj({k: body[k] for k in HASH_KEYS})
    body["created_at_utc"] = datetime.now(UTC).isoformat()
    if OUTPATH.exists():
        raise SystemExit(f"already frozen: {OUTPATH}")
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps(body, indent=1))
    skip = ("term_definition", "forward_inputs", "college_rule_by_fit_season")
    print(json.dumps({k: v for k, v in body.items() if k not in skip}, indent=1))
    print("frozen_file_sha256", sha_file(OUTPATH))


if __name__ == "__main__":
    main()
