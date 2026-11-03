import json
import sys

import mod24_ftn_turnover_luck as m
import numpy as np
import pandas as pd

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS  # noqa: F401
from nfl_ats.margin import (
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    _target_values,
    make_margin_estimator,
)
from nfl_ats.purged_cv import TEAM_STATE_SPAN

OUT_DIR = m.ROOT / "artifacts" / "mod23_unit12_ftn_scheme_v2"
SCORE_SEASONS = (2023, 2024, 2025)
CUTOFF = pd.Timestamp("2024-11-10")
PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "posteam",
    "defteam",
    "qb_dropback",
    "pass_attempt",
    "rush_attempt",
]
OFF_FEATURES = (
    "off_motion",
    "off_no_huddle",
    "off_play_action",
    "off_rpo",
    "off_screen",
    "off_out_of_pocket",
    "off_catchable",
    "off_created",
    "off_contested",
)
DEF_FEATURES = (
    "def_blitz",
    "def_rushers",
    "def_play_action_faced",
    "def_motion_faced",
    "def_run_box",
    "def_catchable_allowed",
)
TEAM_FEATURES = OFF_FEATURES + DEF_FEATURES
DIFF_COLUMNS = tuple(f"ftn_scheme_{n}_diff" for n in TEAM_FEATURES)


def load_plays():
    ftn = pd.concat(
        [
            pd.read_parquet(m.FTN_HIST / f"ftn_charting_{s}.parquet")
            for s in (2022, 2023, 2024, 2025)
        ]
        + [pd.read_parquet(m.FTN_2026)],
        ignore_index=True,
    ).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    pbp = pd.concat(
        [
            pd.read_parquet(m.PBP_ROOT / f"season={s}", columns=PBP_COLUMNS)
            for s in range(2022, 2027)
        ],
        ignore_index=True,
    )
    pbp = pbp.dropna(subset=["posteam", "defteam"])
    pbp["play_id"] = pbp["play_id"].astype(float)
    ftn["play_id"] = ftn["play_id"].astype(float)
    keep = [
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "qb_dropback",
        "pass_attempt",
        "rush_attempt",
    ]
    plays = ftn.merge(pbp[keep], on=["game_id", "play_id"], how="inner")
    for c in ("qb_dropback", "pass_attempt", "rush_attempt"):
        plays[c] = plays[c].fillna(0).astype(float)
    for c in (
        "is_motion",
        "is_no_huddle",
        "is_play_action",
        "is_rpo",
        "is_screen_pass",
        "is_qb_out_of_pocket",
        "is_catchable_ball",
        "is_created_reception",
        "is_contested_ball",
    ):
        plays[c] = plays[c].fillna(False).astype(float)
    plays["blitz"] = (plays["n_blitzers"].fillna(0) >= 1).astype(float)
    return plays


def side_table(plays):
    p = plays.copy()
    p["scrim"] = ((p["qb_dropback"] > 0) | (p["rush_attempt"] > 0)).astype(float)
    p = p.loc[p["scrim"] > 0]
    p["db_pa"] = p["is_play_action"] * p["qb_dropback"]
    p["db_scr"] = p["is_screen_pass"] * p["qb_dropback"]
    p["db_oop"] = p["is_qb_out_of_pocket"] * p["qb_dropback"]
    p["db_blitz"] = p["blitz"] * p["qb_dropback"]
    p["db_rush"] = p["n_pass_rushers"].where(p["qb_dropback"] > 0)
    p["run_box"] = p["n_defense_box"].where(p["rush_attempt"] > 0)
    p["cat"] = p["is_catchable_ball"] * p["pass_attempt"]
    p["crt"] = p["is_created_reception"] * p["pass_attempt"]
    p["cnt"] = p["is_contested_ball"] * p["pass_attempt"]
    g = p.groupby(["game_id", "posteam", "defteam"], as_index=False).agg(
        scrim=("scrim", "sum"),
        db=("qb_dropback", "sum"),
        att=("pass_attempt", "sum"),
        runs=("rush_attempt", "sum"),
        motion=("is_motion", "sum"),
        nohuddle=("is_no_huddle", "sum"),
        rpo=("is_rpo", "sum"),
        pa=("db_pa", "sum"),
        scr=("db_scr", "sum"),
        oop=("db_oop", "sum"),
        blitz=("db_blitz", "sum"),
        rush_sum=("db_rush", "sum"),
        rush_n=("db_rush", "count"),
        box_sum=("run_box", "sum"),
        box_n=("run_box", "count"),
        cat=("cat", "sum"),
        crt=("crt", "sum"),
        cnt=("cnt", "sum"),
    )
    off = pd.DataFrame(
        {
            "game_id": g["game_id"],
            "team": g["posteam"],
            "off_motion": g["motion"] / g["scrim"],
            "off_no_huddle": g["nohuddle"] / g["scrim"],
            "off_play_action": g["pa"] / g["db"].where(g["db"] > 0),
            "off_rpo": g["rpo"] / g["scrim"],
            "off_screen": g["scr"] / g["db"].where(g["db"] > 0),
            "off_out_of_pocket": g["oop"] / g["db"].where(g["db"] > 0),
            "off_catchable": g["cat"] / g["att"].where(g["att"] > 0),
            "off_created": g["crt"] / g["att"].where(g["att"] > 0),
            "off_contested": g["cnt"] / g["att"].where(g["att"] > 0),
        }
    )
    de = pd.DataFrame(
        {
            "game_id": g["game_id"],
            "team": g["defteam"],
            "def_blitz": g["blitz"] / g["db"].where(g["db"] > 0),
            "def_rushers": g["rush_sum"] / g["rush_n"].where(g["rush_n"] > 0),
            "def_play_action_faced": g["pa"] / g["db"].where(g["db"] > 0),
            "def_motion_faced": g["motion"] / g["scrim"],
            "def_run_box": g["box_sum"] / g["box_n"].where(g["box_n"] > 0),
            "def_catchable_allowed": g["cat"] / g["att"].where(g["att"] > 0),
        }
    )
    return off.merge(de, on=["game_id", "team"], how="outer")


WARMUP_SEASON = 2022


def prior_game_weight(raw, name):
    frame = raw.loc[raw["gameday"].dt.year.eq(WARMUP_SEASON), ["team", name]].dropna()
    groups = frame.groupby("team")[name]
    counts = groups.count()
    within = float((groups.var(ddof=1) * (counts - 1)).sum() / (counts - 1).sum())
    between = float(groups.mean().var(ddof=1)) - within / float(counts.mean())
    return within / between if between > 0 else float("inf")


def league_prior_mean(frame, name):
    daily = frame.groupby("gameday")[name].agg(["sum", "count"]).sort_index().cumsum().shift(1)
    return frame["gameday"].map(daily["sum"] / daily["count"])


def shrunk_prior(frame, name, n0):
    frame = frame.sort_values(["team", "gameday"])
    shifted = frame.groupby("team")[name].shift(1)
    smooth = shifted.groupby(frame["team"]).transform(
        lambda s: s.ewm(span=TEAM_STATE_SPAN, min_periods=1, adjust=False).mean()
    )
    seen = shifted.notna().groupby(frame["team"]).cumsum()
    weight = seen / (seen + n0)
    mean = league_prior_mean(frame, name)
    return (mean + weight * (smooth - mean)).where(seen > 0, mean)


def build_scheme(plays, games):
    dates = dict(zip(games["game_id"], pd.to_datetime(games["gameday"]), strict=True))
    per_team = side_table(plays)
    per_team["gameday"] = per_team["game_id"].map(dates)
    per_team = per_team.dropna(subset=["gameday"]).sort_values(["team", "gameday"])
    raw = per_team.copy()
    n0 = {name: prior_game_weight(raw, name) for name in TEAM_FEATURES}
    for name in TEAM_FEATURES:
        per_team[name] = shrunk_prior(per_team, name, n0[name])
    per_team.attrs["n0"] = n0
    return raw, per_team[["game_id", "team", "gameday", *TEAM_FEATURES]], n0


def attach(features, per_team):
    out = features.copy()
    home = per_team.rename(columns={"team": "home_team"}).drop(columns="gameday")
    away = per_team.rename(columns={"team": "away_team"}).drop(columns="gameday")
    out = out.merge(home, on=["game_id", "home_team"], how="left")
    out = out.merge(away, on=["game_id", "away_team"], how="left", suffixes=("", "_away"))
    for name in TEAM_FEATURES:
        out[f"ftn_scheme_{name}_diff"] = (
            out[name].fillna(0.0) - out[f"{name}_away"].fillna(0.0)
        ).where(out["season"].ge(2022), 0.0)
    return out.drop(columns=[*TEAM_FEATURES, *(f"{n}_away" for n in TEAM_FEATURES)])


def assert_pregame(plays, features, per_team):
    dates = dict(zip(features["game_id"], pd.to_datetime(features["gameday"]), strict=True))
    day = plays["game_id"].map(dates)
    truncated = plays.loc[day.le(CUTOFF)]
    _, trunc, _ = build_scheme(truncated, features)
    ids = set(features.loc[pd.to_datetime(features["gameday"]).eq(CUTOFF), "game_id"])
    a = per_team.loc[per_team["game_id"].isin(ids)].set_index(["game_id", "team"])
    b = trunc.loc[trunc["game_id"].isin(ids)].set_index(["game_id", "team"])
    both = a.index.intersection(b.index)
    same = np.isclose(
        a.loc[both, list(TEAM_FEATURES)].to_numpy(dtype=float),
        b.loc[both, list(TEAM_FEATURES)].to_numpy(dtype=float),
        equal_nan=True,
        rtol=0,
        atol=1e-9,
    )
    if len(both) == 0 or not same.all():
        raise AssertionError("future information reaches the scheme features")
    print("pregame truncation check passed", len(both), "team-games", flush=True)


def split_half(raw, games):
    season = dict(zip(games["game_id"], games["season"], strict=True))
    raw = raw.copy()
    raw["season"] = raw["game_id"].map(season)
    raw = raw.loc[raw["season"].between(2022, 2025)].sort_values(["team", "gameday"])
    raw["k"] = raw.groupby(["team", "season"]).cumcount()
    rng = np.random.default_rng(m.BOOTSTRAP_SEED)
    rows = {}
    for name in TEAM_FEATURES:
        odd = raw.loc[raw["k"] % 2 == 1].groupby(["team", "season"])[name].mean()
        even = raw.loc[raw["k"] % 2 == 0].groupby(["team", "season"])[name].mean()
        both = pd.concat([odd, even], axis=1, keys=["o", "e"]).dropna()
        for c in ("o", "e"):
            both[c] = both[c] - both.groupby(level="season")[c].transform("mean")
        x, y = both["o"].to_numpy(), both["e"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        draws = []
        for _ in range(m.BOOTSTRAP_SAMPLES):
            i = rng.integers(0, len(x), len(x))
            draws.append(np.corrcoef(x[i], y[i])[0, 1])

        def sb(v):
            return 2 * v / (1 + v)

        rows[name] = {
            "n_team_seasons": len(both),
            "split_half_r": r,
            "spearman_brown": sb(r),
            "sb_low": float(sb(np.percentile(draws, 2.5))),
            "sb_high": float(sb(np.percentile(draws, 97.5))),
        }
    return rows


def fold_coefficients(table):
    columns = (*m.BASE_COLUMNS, *DIFF_COLUMNS)
    suppressed = tuple(
        c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(m.PROFILE, ()) if c in columns
    )
    frame = table.loc[table["game_type"].eq("REG") & table["ats_margin"].notna()]
    rows = {}
    for season in SCORE_SEASONS:
        train = frame.loc[frame["season"].lt(season)]
        model = make_margin_estimator(
            "ridge", ridge_alpha=m.BASE_ALPHA, suppressed_indicator_columns=suppressed
        )
        model.fit(train.loc[:, list(columns)], _target_values(train, "market_residual"))
        names = list(model[:-1].get_feature_names_out())
        coef = np.ravel(model[-1].coef_)
        rows[season] = {n: float(c) for n, c in zip(names, coef, strict=False) if "ftn_scheme" in n}
    return rows


def main():
    if sys.argv[1:] != ["run"]:
        raise SystemExit("usage: mod23_unit12_ftn_scheme.py run")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(m.SERVED_FEATURES)
    plays = load_plays()
    print("matched plays", len(plays), flush=True)
    raw, per_team, n0 = build_scheme(plays, served)
    print("prior games n0", n0, flush=True)
    assert_pregame(plays, served, per_team)
    reliability = split_half(raw, served)
    for k, v in reliability.items():
        print(
            k,
            round(v["spearman_brown"], 3),
            round(v["sb_low"], 3),
            round(v["sb_high"], 3),
            flush=True,
        )
    table = attach(served, per_team)
    base = m.decisive(m.run_eval(table, m.BASE_COLUMNS, "base12"))
    arm = m.decisive(m.run_eval(table, (*m.BASE_COLUMNS, *DIFF_COLUMNS), "scheme12"))
    a = arm.loc[arm["season"].isin(SCORE_SEASONS)]
    b = base.loc[base["season"].isin(SCORE_SEASONS)]
    result = m.compare("scheme v base 2023_2025", a, b)
    common = a.index.intersection(b.index)
    ca = a.loc[common, "correct_at_open_probability_rule"]
    cb = b.loc[common, "correct_at_open_probability_rule"]
    diff = ca != cb
    result["flip_arm_wins"] = int(ca[diff].sum())
    result["flip_arm_losses"] = int((1 - ca[diff]).sum())
    result["ref_ll"] = float(b.loc[common, "ll"].mean())
    result["arm_ll"] = float(a.loc[common, "ll"].mean())
    result["ref_brier"] = float(b.loc[common, "brier"].mean())
    result["arm_brier"] = float(a.loc[common, "brier"].mean())
    result["reliability"] = reliability
    result["prior_games_n0"] = n0
    result["ewma_span"] = TEAM_STATE_SPAN
    result["fold_coefficients"] = fold_coefficients(table)
    result["looks"] = {"fits": 1, "features": len(TEAM_FEATURES), "scored_seasons": 3}
    (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("reliability",)}, indent=1, default=str
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
