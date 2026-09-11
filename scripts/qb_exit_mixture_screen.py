from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES as TEAM_ALIASES  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402

PBP_ROOT = REPO / "data" / "pbp" / "raw"
INJ_ROOT = REPO / "data" / "raw" / "nflverse_injuries"
OUT_LABELS = REPO / "data" / "processed" / "qb_starter_exits.parquet"
OUT_FEATURES = REPO / "data" / "processed" / "qb_exit_pregame_features.parquet"

PBP_COLUMNS = [
    "play_id",
    "game_id",
    "season",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "qtr",
    "game_seconds_remaining",
    "qb_dropback",
    "passer_player_id",
    "passer_player_name",
    "score_differential",
    "sack",
    "pass_attempt",
]

BLOWOUT_MARGIN = 14
MIN_TEAM_DROPBACKS = 10
RELIEF_DROPBACKS_STRICT = 5


def newest_stamp(root: Path) -> str:
    stamps = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not stamps:
        raise SystemExit(f"no snapshots under {root}")
    return stamps[-1]


def load_pbp(stamp: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(PBP_ROOT / stamp / "season=*" / "*.parquet"))):
        season = int(os.path.basename(os.path.dirname(path)).split("=")[1])
        frame = pd.read_parquet(path, columns=PBP_COLUMNS)
        if "season" not in frame or frame["season"].isna().all():
            frame["season"] = season
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def label_exits(pbp: pd.DataFrame) -> pd.DataFrame:
    drops = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()].copy()
    drops = drops.sort_values(["game_id", "play_id"], kind="mergesort")
    drops["team_order"] = drops.groupby(["game_id", "posteam"]).cumcount()

    grouped = drops.groupby(["game_id", "posteam"], sort=False)
    rows = []
    for (game_id, posteam), frame in grouped:
        n_team = len(frame)
        first = frame.iloc[0]
        starter_id = first["passer_player_id"]
        starter_mask = frame["passer_player_id"] == starter_id
        n_starter = int(starter_mask.sum())
        last_starter = frame[starter_mask].iloc[-1]
        after = frame[frame["team_order"] > last_starter["team_order"]]
        relief = after[after["passer_player_id"] != starter_id]
        n_relief = len(relief)
        relief_id = relief.iloc[0]["passer_player_id"] if n_relief else None
        relief_name = relief.iloc[0]["passer_player_name"] if n_relief else None
        exit_qtr = float(last_starter["qtr"])
        exit_diff = float(last_starter["score_differential"])
        pre_q4 = exit_qtr <= 3
        competitive = abs(exit_diff) <= BLOWOUT_MARGIN
        eligible = n_team >= MIN_TEAM_DROPBACKS
        rows.append(
            {
                "game_id": game_id,
                "season": int(first["season"]),
                "season_type": first["season_type"],
                "week": int(first["week"]),
                "posteam": posteam,
                "defteam": first["defteam"],
                "home_team": first["home_team"],
                "away_team": first["away_team"],
                "is_home": bool(posteam == first["home_team"]),
                "starter_id": starter_id,
                "starter_name": first["passer_player_name"],
                "team_dropbacks": n_team,
                "starter_dropbacks": n_starter,
                "starter_share": n_starter / n_team if n_team else np.nan,
                "relief_dropbacks": n_relief,
                "relief_id": relief_id,
                "relief_name": relief_name,
                "exit_qtr": exit_qtr,
                "exit_seconds_remaining": float(last_starter["game_seconds_remaining"]),
                "exit_score_diff": exit_diff,
                "eligible": eligible,
                "early_exit": bool(eligible and pre_q4 and n_relief >= 1 and competitive),
                "early_exit_strict": bool(
                    eligible and pre_q4 and n_relief >= RELIEF_DROPBACKS_STRICT and competitive
                ),
                "early_exit_any_margin": bool(eligible and pre_q4 and n_relief >= 1),
            }
        )
    return pd.DataFrame(rows)


def cmd_label(args: argparse.Namespace) -> None:
    stamp = args.pbp_stamp or newest_stamp(PBP_ROOT)
    pbp = load_pbp(stamp)
    labels = label_exits(pbp)
    labels["pbp_stamp"] = stamp
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(OUT_LABELS, index=False)

    reg = labels[(labels["season_type"] == "REG") & labels["eligible"]]
    overall = {
        "pbp_stamp": stamp,
        "team_games_all": len(labels),
        "team_games_reg_eligible": len(reg),
        "ineligible_team_games": int((~labels["eligible"]).sum()),
        "base_rate_early_exit": float(reg["early_exit"].mean()),
        "base_rate_early_exit_strict": float(reg["early_exit_strict"].mean()),
        "base_rate_early_exit_any_margin": float(reg["early_exit_any_margin"].mean()),
        "mean_starter_share": float(reg["starter_share"].mean()),
    }
    by_season = (
        reg.groupby("season")
        .agg(
            team_games=("early_exit", "size"),
            exits=("early_exit", "sum"),
            rate=("early_exit", "mean"),
            rate_strict=("early_exit_strict", "mean"),
            rate_any_margin=("early_exit_any_margin", "mean"),
            mean_starter_share=("starter_share", "mean"),
        )
        .reset_index()
    )
    odd = reg[reg["season"] % 2 == 1]
    even = reg[reg["season"] % 2 == 0]
    split_half = {
        "odd_seasons_rate": float(odd["early_exit"].mean()),
        "even_seasons_rate": float(even["early_exit"].mean()),
        "odd_n": len(odd),
        "even_n": len(even),
    }
    print(json.dumps(overall, indent=2))
    print(by_season.to_string(index=False))
    print(json.dumps(split_half, indent=2))
    print(f"wrote {OUT_LABELS}")


REPORT_RANK = {
    "Out": 4,
    "Doubtful": 3,
    "Questionable": 2,
    "Probable": 1,
}
PRACTICE_RANK = {
    "Out (Definitely Will Not Play)": 3,
    "Did Not Participate In Practice": 2,
    "Limited Participation in Practice": 1,
    "Full Participation in Practice": 0,
}
EXIT_SHRINKAGE_STARTS = 20.0
SACK_SHRINKAGE_DROPBACKS = 200.0

FEATURE_COLUMNS = [
    "on_injury_report",
    "inj_report_rank",
    "inj_practice_rank",
    "experience_years",
    "log_career_starts_prior",
    "prior_missed_starts",
    "has_prior_starts",
    "qb_prior_exit_rate",
    "opp_sack_rate",
]


def _kickoff_utc(schedules: pd.DataFrame) -> pd.Series:
    gameday = schedules["gameday"].astype(str)
    gametime = (
        schedules["gametime"].astype(str).replace({"nan": "13:00", "None": "13:00", "": "13:00"})
    )
    stamps = pd.to_datetime(gameday + " " + gametime, errors="coerce")
    stamps = stamps.fillna(pd.to_datetime(gameday, errors="coerce") + pd.Timedelta(hours=13))
    return stamps.dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def _opponent_sack_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    drops = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    faced = (
        drops.groupby(["season", "week", "game_id", "defteam"], as_index=False)
        .agg(dropbacks_faced=("sack", "size"), sacks=("sack", "sum"))
        .rename(columns={"defteam": "team"})
    )
    faced = faced.sort_values(["team", "season", "week", "game_id"], kind="mergesort")
    grouped = faced.groupby(["team", "season"], sort=False)
    faced["prior_dropbacks"] = grouped["dropbacks_faced"].cumsum() - faced["dropbacks_faced"]
    faced["prior_sacks"] = grouped["sacks"].cumsum() - faced["sacks"]

    season_total = faced.groupby(["team", "season"], as_index=False).agg(
        season_dropbacks=("dropbacks_faced", "sum"), season_sacks=("sacks", "sum")
    )
    season_total["season"] = season_total["season"] + 1
    season_total = season_total.rename(
        columns={"season_dropbacks": "prev_dropbacks", "season_sacks": "prev_sacks"}
    )
    faced = faced.merge(season_total, on=["team", "season"], how="left")
    faced[["prev_dropbacks", "prev_sacks"]] = faced[["prev_dropbacks", "prev_sacks"]].fillna(0.0)

    league = faced.groupby("season", as_index=False).agg(
        league_dropbacks=("dropbacks_faced", "sum"), league_sacks=("sacks", "sum")
    )
    league["league_rate"] = league["league_sacks"] / league["league_dropbacks"]
    league["season"] = league["season"] + 1
    faced = faced.merge(league[["season", "league_rate"]], on="season", how="left")
    faced["league_rate"] = faced["league_rate"].fillna(
        faced["sacks"].sum() / faced["dropbacks_faced"].sum()
    )

    numerator = (
        faced["prior_sacks"] + faced["prev_sacks"] + SACK_SHRINKAGE_DROPBACKS * faced["league_rate"]
    )
    denominator = faced["prior_dropbacks"] + faced["prev_dropbacks"] + SACK_SHRINKAGE_DROPBACKS
    faced["opp_sack_rate"] = numerator / denominator
    return faced[["game_id", "team", "opp_sack_rate"]].rename(columns={"team": "defteam"})


def _career_history(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels.sort_values(["season", "week", "game_id"], kind="mergesort").copy()
    frame["one"] = 1.0
    grouped = frame.groupby("starter_id", sort=False)
    frame["career_starts_prior"] = grouped["one"].cumsum() - 1.0
    frame["first_season"] = grouped["season"].transform("min")
    frame["experience_years"] = frame["season"] - frame["first_season"]

    season_starts = frame.groupby(["starter_id", "season", "posteam"], as_index=False).agg(
        starts=("one", "sum"), exits=("early_exit", "sum")
    )
    season_starts = season_starts.sort_values(
        ["starter_id", "season", "starts"], ascending=[True, True, False]
    )
    primary = season_starts.drop_duplicates(["starter_id", "season"]).rename(
        columns={"posteam": "primary_team", "starts": "primary_starts"}
    )[["starter_id", "season", "primary_team", "primary_starts"]]

    team_games = frame.groupby(["posteam", "season"], as_index=False).agg(team_games=("one", "sum"))
    primary = primary.merge(
        team_games.rename(columns={"posteam": "primary_team"}),
        on=["primary_team", "season"],
        how="left",
    )
    primary["missed"] = primary["team_games"] - primary["primary_starts"]
    primary["season"] = primary["season"] + 1
    prior = primary[["starter_id", "season", "missed"]].rename(
        columns={"missed": "prior_missed_starts"}
    )
    frame = frame.merge(prior, on=["starter_id", "season"], how="left")
    frame["has_prior_starts"] = frame["prior_missed_starts"].notna().astype(float)
    frame["prior_missed_starts"] = frame["prior_missed_starts"].fillna(0.0)

    career = (
        season_starts.groupby(["starter_id", "season"], as_index=False)
        .agg(starts=("starts", "sum"), exits=("exits", "sum"))
        .sort_values(["starter_id", "season"])
    )
    grouped_career = career.groupby("starter_id", sort=False)
    career["starts_prior"] = grouped_career["starts"].cumsum() - career["starts"]
    career["exits_prior"] = grouped_career["exits"].cumsum() - career["exits"]
    base = float(labels["early_exit"].mean())
    career["qb_prior_exit_rate"] = (career["exits_prior"] + EXIT_SHRINKAGE_STARTS * base) / (
        career["starts_prior"] + EXIT_SHRINKAGE_STARTS
    )
    frame = frame.merge(
        career[["starter_id", "season", "qb_prior_exit_rate"]],
        on=["starter_id", "season"],
        how="left",
    )
    frame["qb_prior_exit_rate"] = frame["qb_prior_exit_rate"].fillna(base)
    frame["log_career_starts_prior"] = np.log1p(frame["career_starts_prior"])
    return frame.drop(columns=["one", "first_season"])


def _injury_join(
    frame: pd.DataFrame, injuries: pd.DataFrame, kickoff: pd.DataFrame
) -> pd.DataFrame:
    inj = injuries.copy()
    inj = inj[inj["season_type"].isna() | inj["season_type"].eq("REG") | inj["game_type"].eq("REG")]
    inj["team"] = inj["team"].map(lambda code: TEAM_ALIASES.get(code, code))
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj = inj.dropna(subset=["week"])
    inj["week"] = inj["week"].astype(int)
    inj["report_rank"] = inj["report_status"].map(REPORT_RANK).fillna(0.0)
    inj["practice_rank"] = inj["practice_status"].map(PRACTICE_RANK).fillna(0.0)
    inj = inj.rename(columns={"gsis_id": "starter_id", "team": "posteam"})

    merged = frame.merge(
        inj[
            [
                "season",
                "week",
                "posteam",
                "starter_id",
                "report_rank",
                "practice_rank",
                "date_modified",
            ]
        ],
        on=["season", "week", "posteam", "starter_id"],
        how="left",
    )
    merged = merged.merge(kickoff, on="game_id", how="left")
    late = merged["date_modified"].notna() & merged["kickoff_utc"].notna()
    after = late & (merged["date_modified"] >= merged["kickoff_utc"])
    merged.loc[after, ["report_rank", "practice_rank"]] = np.nan
    merged["dropped_late_injury_rows"] = after.astype(int)
    merged["on_injury_report"] = merged["report_rank"].notna().astype(float)
    merged["inj_report_rank"] = merged["report_rank"].fillna(0.0)
    merged["inj_practice_rank"] = merged["practice_rank"].fillna(0.0)
    return merged.drop(columns=["report_rank", "practice_rank", "date_modified"])


def cmd_features(args: argparse.Namespace) -> None:
    labels = pd.read_parquet(OUT_LABELS)
    reg = labels[(labels["season_type"] == "REG") & labels["eligible"]].copy()

    stamp = args.pbp_stamp or newest_stamp(PBP_ROOT)
    pbp = load_pbp(stamp)
    pbp = pbp[pbp["season_type"] == "REG"]
    sack_rate = _opponent_sack_rate(pbp)

    schedules_path = sorted((REPO / "data" / "raw").glob("*/schedules.parquet"))[-1]
    schedules = pd.read_parquet(schedules_path)
    kickoff = pd.DataFrame(
        {"game_id": schedules["game_id"].astype(str), "kickoff_utc": _kickoff_utc(schedules)}
    )

    inj_stamp = args.injury_stamp or newest_stamp(INJ_ROOT)
    injuries = pd.read_parquet(INJ_ROOT / inj_stamp / "injuries.parquet")

    frame = _career_history(reg)
    frame = frame.merge(sack_rate, on=["game_id", "defteam"], how="left")
    frame["opp_sack_rate"] = frame["opp_sack_rate"].fillna(frame["opp_sack_rate"].median())
    frame = _injury_join(frame, injuries, kickoff)
    frame["pbp_stamp"] = stamp
    frame["injury_stamp"] = inj_stamp
    frame["schedules_path"] = str(schedules_path.relative_to(REPO))
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_FEATURES, index=False)

    summary = {
        "rows": len(frame),
        "pbp_stamp": stamp,
        "injury_stamp": inj_stamp,
        "schedules": str(schedules_path.relative_to(REPO)),
        "starters_on_injury_report": float(frame["on_injury_report"].mean()),
        "injury_rows_dropped_after_kickoff": int(frame["dropped_late_injury_rows"].sum()),
        "exit_rate_on_report": float(
            frame.loc[frame["on_injury_report"] == 1.0, "early_exit"].mean()
        ),
        "exit_rate_off_report": float(
            frame.loc[frame["on_injury_report"] == 0.0, "early_exit"].mean()
        ),
        "n_on_report": int((frame["on_injury_report"] == 1.0).sum()),
        "n_off_report": int((frame["on_injury_report"] == 0.0).sum()),
        "mean_relief_dropback_share_given_exit": float(
            (1.0 - frame.loc[frame["early_exit"], "starter_share"]).mean()
        ),
    }
    print(json.dumps(summary, indent=2))
    print(json.dumps(reliability_report(frame), indent=2))
    print(f"wrote {OUT_FEATURES}")


def _split_half(frame: pd.DataFrame, key: str, min_games: int, seed: int, draws: int) -> dict:
    frame = frame.copy()
    frame["half"] = np.where(frame["season"] % 2 == 1, "odd", "even")
    grouped = frame.groupby([key, "half"])["early_exit"].agg(["mean", "size"]).unstack("half")
    if ("mean", "odd") not in grouped.columns or ("mean", "even") not in grouped.columns:
        return {"key": key, "n_units": 0}
    keep = (grouped[("size", "odd")].fillna(0) >= min_games) & (
        grouped[("size", "even")].fillna(0) >= min_games
    )
    means = grouped.loc[keep, [("mean", "odd"), ("mean", "even")]].dropna()
    n = len(means)
    if n < 3:
        return {"key": key, "n_units": n, "pearson_r": float("nan")}
    odd = means[("mean", "odd")].to_numpy(dtype=float)
    even = means[("mean", "even")].to_numpy(dtype=float)
    r = float(np.corrcoef(odd, even)[0, 1])
    rng = np.random.default_rng(seed)
    boots = np.empty(draws, dtype=float)
    for i in range(draws):
        pick = rng.integers(0, n, size=n)
        with np.errstate(invalid="ignore"):
            boots[i] = np.corrcoef(odd[pick], even[pick])[0, 1]
    spearman_brown = (2.0 * r) / (1.0 + r) if r > -1.0 else float("nan")
    return {
        "key": key,
        "n_units": n,
        "min_games_per_half": min_games,
        "pearson_r": r,
        "pearson_r_ci95": [
            float(np.nanquantile(boots, 0.025)),
            float(np.nanquantile(boots, 0.975)),
        ],
        "spearman_brown_full_length_reliability": spearman_brown,
        "probability_positive": float(probability_positive_from_draws(boots, ignore_nan=True)),
    }


def reliability_report(frame: pd.DataFrame) -> dict:
    return {
        "starter_qb_odd_even_seasons": _split_half(frame, "starter_id", 8, 20260910, 4000),
        "starter_qb_odd_even_seasons_min16": _split_half(frame, "starter_id", 16, 20260910, 4000),
        "team_odd_even_seasons": _split_half(frame, "posteam", 40, 20260910, 4000),
    }


def _fit_chronological(frame: pd.DataFrame, min_train_seasons: int) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frame = frame.sort_values(["season", "week", "game_id", "posteam"]).reset_index(drop=True)
    frame["p_exit"] = np.nan
    seasons = sorted(frame["season"].unique())
    for season in seasons:
        train = frame[frame["season"] < season]
        if train["season"].nunique() < min_train_seasons or train["early_exit"].sum() < 20:
            continue
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        )
        model.fit(
            train[FEATURE_COLUMNS].to_numpy(dtype=float), train["early_exit"].to_numpy(dtype=int)
        )
        mask = frame["season"] == season
        frame.loc[mask, "p_exit"] = model.predict_proba(
            frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=float)
        )[:, 1]
    return frame


def _calibration(frame: pd.DataFrame, bins: int) -> dict:
    scored = frame.dropna(subset=["p_exit"])
    if scored.empty:
        return {}
    edges = np.quantile(scored["p_exit"], np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    cut = pd.cut(scored["p_exit"], edges, include_lowest=True, duplicates="drop")
    table = (
        scored.groupby(cut, observed=True)
        .agg(
            n=("early_exit", "size"), predicted=("p_exit", "mean"), observed=("early_exit", "mean")
        )
        .reset_index(drop=True)
    )
    base = float(scored["early_exit"].mean())
    brier = float(((scored["p_exit"] - scored["early_exit"].astype(float)) ** 2).mean())
    brier_base = float(((base - scored["early_exit"].astype(float)) ** 2).mean())
    logit = np.log(
        np.clip(scored["p_exit"], 1e-6, 1 - 1e-6) / (1 - np.clip(scored["p_exit"], 1e-6, 1 - 1e-6))
    )
    slope = (
        float(np.polyfit(logit, scored["early_exit"].astype(float), 1)[0])
        if logit.std() > 0
        else float("nan")
    )
    from sklearn.metrics import roc_auc_score

    outcomes = scored["early_exit"].astype(int)
    auc = (
        float(roc_auc_score(outcomes, scored["p_exit"])) if outcomes.nunique() > 1 else float("nan")
    )
    return {
        "scored_rows": len(scored),
        "seasons": [int(scored["season"].min()), int(scored["season"].max())],
        "base_rate": base,
        "mean_predicted": float(scored["p_exit"].mean()),
        "auc": auc,
        "brier": brier,
        "brier_base_rate": brier_base,
        "brier_skill_score": float(1.0 - brier / brier_base) if brier_base else float("nan"),
        "linear_probability_slope_on_logit": slope,
        "p_exit_quantiles": {
            q: float(scored["p_exit"].quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "decile_table": table.to_dict(orient="records"),
    }


def cmd_fit(args: argparse.Namespace) -> None:
    frame = pd.read_parquet(OUT_FEATURES)
    fitted = _fit_chronological(frame, args.min_train_seasons)
    fitted.to_parquet(OUT_FEATURES, index=False)
    report = {
        "calibration_all_scored_seasons": _calibration(fitted, args.bins),
        "calibration_window_2020_2021": _calibration(
            fitted[fitted["season"].isin([2020, 2021])], args.bins
        ),
    }
    print(json.dumps(report, indent=2, default=float))
    print(f"wrote {OUT_FEATURES}")


FAMILY = "qb_exit_mixture_on_production"
BACKUP_QB_RAW_GAP = -0.029095
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260910
NULL_PERMUTATIONS = 200
LARGE_CONTROL_DELTA = -0.50
PRODUCTION_FEATURES = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
MARKET_ROOT = REPO / "data" / "market" / "raw"
SCORE_OUTPUT = REPO / "artifacts" / "experiments" / "qb_exit_mixture"


def _accuracy_metric(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["baseline_correct", "candidate_correct"])
    return {
        "accuracy_points": 100.0
        * float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": 100.0 * float(valid["candidate_correct"].mean()),
        "baseline_accuracy": 100.0 * float(valid["baseline_correct"].mean()),
    }


def _brier_metric(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["home_cover_open"])
    baseline = float(((valid["p_production"] - valid["home_cover_open"]) ** 2).mean())
    candidate = float(((valid["p_mixture"] - valid["home_cover_open"]) ** 2).mean())
    return {"brier_improvement": baseline - candidate}


def _summarize(frame: pd.DataFrame, metric_fn, samples: int, seed: int) -> dict:
    from nfl_ats.clv import week_blocked_bootstrap

    point = metric_fn(frame)
    week = week_blocked_bootstrap(frame, metric_fn, block="week", samples=samples, seed=seed)
    season = week_blocked_bootstrap(frame, metric_fn, block="season", samples=samples, seed=seed)
    primary = next(iter(point))
    w = week.loc[week["metric"].eq(primary)].iloc[0]
    s = season.loc[season["metric"].eq(primary)].iloc[0]
    return {
        **point,
        "week_blocked_ci95": [float(w["lower"]), float(w["upper"])],
        "week_blocked_probability_positive": float(w["probability_positive"]),
        "season_blocked_ci95": [float(s["lower"]), float(s["upper"])],
        "season_blocked_probability_positive": float(s["probability_positive"]),
        "n_games": len(frame),
        "n_weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(frame["season"].nunique()),
    }


def _mixture_frame(
    scored: pd.DataFrame, exits: pd.DataFrame, teams: pd.DataFrame, delta: float, oracle: bool
) -> pd.DataFrame:
    from nfl_ats.clv import pick_correct

    column = "early_exit" if oracle else "p_exit"
    side = exits[["game_id", "posteam", column, "on_injury_report"]].copy()
    side[column] = side[column].astype(float)
    frame = scored.merge(teams, on="game_id", how="inner")
    home = side.rename(
        columns={"posteam": "home_team", column: "p_exit_home", "on_injury_report": "report_home"}
    )
    away = side.rename(
        columns={"posteam": "away_team", column: "p_exit_away", "on_injury_report": "report_away"}
    )
    frame = frame.merge(home, on=["game_id", "home_team"], how="inner")
    frame = frame.merge(away, on=["game_id", "away_team"], how="inner")
    frame = frame.dropna(subset=["p_exit_home", "p_exit_away"]).copy()
    frame["p_production"] = frame["home_cover_probability_at_open"]
    frame["p_mixture"] = frame["p_production"] + delta * (
        frame["p_exit_home"] - frame["p_exit_away"]
    )
    frame["baseline_pick_home"] = frame["p_production"].ge(0.5)
    frame["candidate_pick_home"] = frame["p_mixture"].ge(0.5)
    frame["baseline_correct"] = pick_correct(frame["baseline_pick_home"], frame["margin_vs_open"])
    frame["candidate_correct"] = pick_correct(frame["candidate_pick_home"], frame["margin_vs_open"])
    frame["home_cover_open"] = np.where(
        frame["margin_vs_open"].eq(0.0), np.nan, frame["margin_vs_open"].gt(0.0).astype(float)
    )
    frame["asymmetric_cell"] = frame["report_home"].ne(frame["report_away"])
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _null_distribution(frame: pd.DataFrame, permutations: int, seed: int) -> dict:
    from nfl_ats.clv import pick_correct

    positions = [
        np.asarray(v, dtype=np.intp)
        for v in frame.groupby(["season", "week"], sort=False).indices.values()
    ]
    rng = np.random.default_rng(seed)
    original = frame["margin_vs_open"].to_numpy(dtype=float)
    deltas = []
    for _ in range(permutations):
        shuffled = original.copy()
        for group in positions:
            shuffled[group] = rng.permutation(shuffled[group])
        margin = pd.Series(shuffled, index=frame.index)
        table = pd.DataFrame(
            {
                "b": pick_correct(frame["baseline_pick_home"], margin),
                "c": pick_correct(frame["candidate_pick_home"], margin),
            }
        ).dropna()
        deltas.append(100.0 * float((table["c"] - table["b"]).mean()))
    values = np.asarray(deltas, dtype=float)
    observed = _accuracy_metric(frame)["accuracy_points"]
    return {
        "permutations": permutations,
        "null_mean_delta": float(values.mean()),
        "null_sd_delta": float(values.std(ddof=1)),
        "null_q025": float(np.quantile(values, 0.025)),
        "null_q975": float(np.quantile(values, 0.975)),
        "observed_delta": observed,
        "fraction_of_null_below_observed": float((values < observed).mean()),
    }


def cmd_score(args: argparse.Namespace) -> None:
    from nfl_ats.clv import opener_pick_evaluation, resolve_active_probability_method
    from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
    from nfl_ats.rotation import confirmation_split, load_registry

    features = pd.read_parquet(PRODUCTION_FEATURES)
    training, window = confirmation_split(features, load_registry(), FAMILY)
    if pd.to_datetime(training["gameday"]).max() >= pd.to_datetime(window["gameday"]).min():
        raise SystemExit("confirmation split leaked a training row into the assigned window")
    seasons = tuple(sorted(int(x) for x in window["season"].unique()))
    scoped = pd.concat([training, window], ignore_index=True)

    config = {
        "probability_method": resolve_active_probability_method(),
        "feature_profile": "weak_stack",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "target": "market_residual",
    }
    scored = opener_pick_evaluation(
        MARKET_ROOT, scoped, active_model_config=config, min_train_games=DEFAULT_MIN_TRAIN_GAMES
    )
    scored = scored.loc[scored["season"].astype(int).isin(seasons)].reset_index(drop=True)

    teams = features[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    exits = pd.read_parquet(OUT_FEATURES)

    arms = {
        "screen": (BACKUP_QB_RAW_GAP, False),
        "positive_control_oracle_exit": (BACKUP_QB_RAW_GAP, True),
        "positive_control_oracle_exit_large_delta": (LARGE_CONTROL_DELTA, True),
        "snap_share_scaled": (BACKUP_QB_RAW_GAP * args.relief_share, False),
    }
    result: dict = {
        "window_seasons": list(seasons),
        "grade": "opener",
        "delta_backup_qb_cover_gap": BACKUP_QB_RAW_GAP,
        "relief_dropback_share_given_exit": args.relief_share,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "active_model": json.loads((REPO / "artifacts" / "active_ats_model.json").read_text())[
            "model_id"
        ],
        "probability_method": config["probability_method"],
        "arms": {},
    }
    for name, (delta, oracle) in arms.items():
        frame = _mixture_frame(scored, exits, teams, delta, oracle)
        graded = frame.dropna(subset=["baseline_correct", "candidate_correct"])
        arm = {
            "delta": delta,
            "oracle_exit": oracle,
            "paired_games": len(frame),
            "non_push_games": len(graded),
            "picks_changed": int(
                frame["baseline_pick_home"].ne(frame["candidate_pick_home"]).sum()
            ),
            "picks_changed_non_push": int(
                graded["baseline_pick_home"].ne(graded["candidate_pick_home"]).sum()
            ),
            "mean_abs_probability_shift": float(
                (frame["p_mixture"] - frame["p_production"]).abs().mean()
            ),
            "max_abs_probability_shift": float(
                (frame["p_mixture"] - frame["p_production"]).abs().max()
            ),
            "accuracy": _summarize(graded, _accuracy_metric, args.bootstrap_samples, args.seed),
            "brier": _summarize(
                frame.dropna(subset=["home_cover_open"]),
                _brier_metric,
                args.bootstrap_samples,
                args.seed,
            ),
            "per_season": {
                str(season): _summarize(rows, _accuracy_metric, args.bootstrap_samples, args.seed)
                for season, rows in graded.groupby("season")
            },
        }
        asymmetric = graded[graded["asymmetric_cell"]]
        arm["asymmetric_cell"] = {
            "n_games": len(asymmetric),
            "picks_changed": int(
                asymmetric["baseline_pick_home"].ne(asymmetric["candidate_pick_home"]).sum()
            ),
            **(
                _summarize(asymmetric, _accuracy_metric, args.bootstrap_samples, args.seed)
                if len(asymmetric) >= 20
                else {}
            ),
        }
        if name == "screen":
            arm["permutation_null"] = _null_distribution(graded, args.permutations, args.seed)
            SCORE_OUTPUT.mkdir(parents=True, exist_ok=True)
            frame.to_csv(SCORE_OUTPUT / "paired_predictions.csv", index=False)
        result["arms"][name] = arm

    SCORE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (SCORE_OUTPUT / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))
    print(f"wrote {SCORE_OUTPUT / 'results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QB starter early-exit mixture screen (LEAD-63)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_label = sub.add_parser("label", help="label starter early exits from play-by-play")
    p_label.add_argument("--pbp-stamp", default=None)
    p_label.set_defaults(func=cmd_label)
    p_feat = sub.add_parser("features", help="build deadline-visible pregame exit features")
    p_feat.add_argument("--pbp-stamp", default=None)
    p_feat.add_argument("--injury-stamp", default=None)
    p_feat.set_defaults(func=cmd_features)
    p_fit = sub.add_parser("fit", help="chronological logistic exit probability plus calibration")
    p_fit.add_argument("--min-train-seasons", type=int, default=4)
    p_fit.add_argument("--bins", type=int, default=10)
    p_fit.set_defaults(func=cmd_fit)
    p_score = sub.add_parser(
        "score", help="grade the mixture pick against production at the opener"
    )
    p_score.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    p_score.add_argument("--permutations", type=int, default=NULL_PERMUTATIONS)
    p_score.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p_score.add_argument("--relief-share", type=float, default=0.5969)
    p_score.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
