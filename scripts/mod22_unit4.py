from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from players_on_field_rating_eval import (  # noqa: E402
    ARTIFACTS_ROOT,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DATA_ROOT,
    INTERVAL_LEVEL,
    RELIABILITY_EDGES,
    REPO_ROOT,
    in_sample_fit,
    loso,
)
from players_on_field_rating_unit2 import evaluate_candidate  # noqa: E402

from nfl_ats.pick_probability_fit import FIT_FEATURES, build_fit_population  # noqa: E402

SNAPSHOT = DATA_ROOT / "players" / "raw" / "20260910T205112Z"
FEATURE_TABLE = DATA_ROOT / "processed" / "game_features_weak_stack.parquet"
OUTPUT_ROOT = ARTIFACTS_ROOT / "mod22_unit4"
OUT_STATUSES = {"Out", "Doubtful"}
DECISION_HOURS = 24
SEASON_START, SEASON_END = 2020, 2025


def norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower().replace(" jr.", "").replace(" iii", ""))


def load_games() -> pd.DataFrame:
    games = pd.read_parquet(
        FEATURE_TABLE,
        columns=["game_id", "season", "week", "game_type", "kickoff", "home_team", "away_team"],
    )
    games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True, errors="coerce")
    games = games.loc[(games["game_type"] == "REG") & games["kickoff"].notna()].copy()
    games["decision_at"] = games["kickoff"] - pd.Timedelta(hours=DECISION_HOURS)
    return games


def starters(games: pd.DataFrame) -> pd.DataFrame:
    snaps = pd.read_parquet(
        SNAPSHOT / "snap_counts.parquet",
        columns=["game_id", "season", "team", "position", "player", "offense_snaps"],
    )
    qb = snaps.loc[snaps["position"] == "QB"]
    qb = qb.sort_values("offense_snaps", ascending=False).drop_duplicates(["game_id", "team"])
    qb = qb.merge(games[["game_id", "kickoff"]], on="game_id", how="inner")
    return qb.sort_values(["team", "kickoff"]).reset_index(drop=True)


def main() -> None:
    now = datetime.now(UTC)
    games = load_games()
    starts = starters(games)
    starts_by_team = {team: frame.reset_index(drop=True) for team, frame in starts.groupby("team")}
    rosters = pd.read_parquet(
        SNAPSHOT / "weekly_rosters.parquet",
        columns=["season", "team", "position", "full_name", "gsis_id"],
    )
    rosters = rosters.loc[rosters["position"] == "QB"].copy()
    rosters["key"] = rosters["full_name"].map(norm)
    gsis_by_key = (
        rosters.dropna(subset=["gsis_id"]).groupby(["team", "key"])["gsis_id"].agg(lambda s: set(s))
    )
    injuries = pd.read_parquet(
        SNAPSHOT / "injuries.parquet",
        columns=["season", "week", "team", "gsis_id", "report_status", "effective_observed_at"],
    )
    injuries["effective_observed_at"] = pd.to_datetime(
        injuries["effective_observed_at"], utc=True, errors="coerce"
    )
    injuries = injuries.loc[injuries["report_status"].isin(OUT_STATUSES)]
    inj_index = {key: frame for key, frame in injuries.groupby(["season", "week", "team"])}

    match_total = match_hit = 0
    rows = []
    window = games.loc[(games["season"] >= SEASON_START) & (games["season"] <= SEASON_END)]
    for game in window.itertuples():
        flags = {}
        for side, team in (("home", game.home_team), ("away", game.away_team)):
            flag = 0.0
            frame_team = starts_by_team.get(team)
            prior = None
            if frame_team is not None:
                earlier = frame_team.loc[frame_team["kickoff"] < game.kickoff]
                if not earlier.empty:
                    prior = earlier.iloc[-1]
            match_total += 1
            if prior is not None:
                assert prior["kickoff"] < game.decision_at
                ids = gsis_by_key.get((team, norm(prior["player"])), set())
                if ids:
                    match_hit += 1
                    frame = inj_index.get((game.season, game.week, team))
                    if frame is not None:
                        hit = frame.loc[
                            frame["gsis_id"].isin(ids)
                            & (frame["effective_observed_at"] <= game.decision_at)
                        ]
                        assert (hit["effective_observed_at"] <= game.decision_at).all()
                        flag = float(len(hit) > 0)
            flags[side] = flag
        rows.append((game.game_id, flags["home"], flags["away"]))
    qb_flags = pd.DataFrame(rows, columns=["game_id", "home_qb_out", "away_qb_out"])
    qb_flags["qb_out_diff"] = qb_flags["away_qb_out"] - qb_flags["home_qb_out"]

    graded, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    population = graded.merge(qb_flags, on="game_id", how="inner")
    population = population.loc[population["tue_open_home_spread"].notna()].reset_index(drop=True)

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": RELIABILITY_EDGES,
    }
    base_oos, base_folds = loso(population, FIT_FEATURES)
    base_is, _ = in_sample_fit(population, FIT_FEATURES)
    population = population.assign(base_oos_probability=base_oos, base_is_probability=base_is)
    features = (*FIT_FEATURES, "qb_out_diff")
    cand_oos, cand_folds = loso(population, features)
    cand_is, cand_is_coef = in_sample_fit(population, features)
    result = evaluate_candidate(
        population, cand_oos, cand_is, cand_folds, base_folds, base_is, declaration
    )
    result["in_sample_coefficients"] = cand_is_coef

    summary = {
        "created_at_utc": now.isoformat(),
        "command": "python scripts/mod22_unit4.py",
        "provenance": provenance,
        "games": len(population),
        "games_home_qb_out": int(population["home_qb_out"].sum()),
        "games_away_qb_out": int(population["away_qb_out"].sum()),
        "games_nonzero_term": int((population["qb_out_diff"] != 0).sum()),
        "name_to_gsis_match_rate": match_hit / match_total,
        "look_qb_out_diff": result,
    }
    out = OUTPUT_ROOT / now.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    population.to_parquet(out / "per_game.parquet", index=False)
    print(str(out.relative_to(REPO_ROOT)))
    trimmed = {key: value for key, value in summary.items() if key != "provenance"}
    print(json.dumps(trimmed, indent=1, sort_keys=True, default=str)[:7000])


if __name__ == "__main__":
    main()
