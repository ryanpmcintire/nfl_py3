from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)
from nfl_ats.public_board import find_matching_opener_evaluation  # noqa: E402
from nfl_ats.snapshots import latest_snapshot, load_snapshot  # noqa: E402

OUTPUT = Path("artifacts/experiments/veteran_rest_day_tell")
SEED = 20260910
SAMPLES = 20000
AGE_THRESHOLD = 30.0
STARTER_THRESHOLD = 0.5
TRAILING_GAMES = 4
FOLLOWUP_GAMES = 4
RELIABLE_SEASONS = (2021, 2022, 2023, 2024, 2025)
ODD_SEASONS = (2021, 2023, 2025)
EVEN_SEASONS = (2022, 2024)

REST_PATTERN = re.compile(r"\brest(?:ed|ing)?\b|load management", re.I)
REASON_COLUMNS = (
    "report_primary_injury",
    "report_secondary_injury",
    "practice_primary_injury",
    "practice_secondary_injury",
)
DNP_STATUS = "Did Not Participate In Practice"


def canonical_team(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def latest_under(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return candidates[-1]


def load_players_master() -> tuple[pd.DataFrame, Path]:
    existing = sorted((REPO / "data/players/raw").glob("*/players.parquet"))
    if existing:
        path = existing[-1]
        return pd.read_parquet(path), path
    import nflreadpy as nfl

    players = nfl.load_players().to_pandas()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = REPO / "data/players/raw" / stamp
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / "players.parquet"
    atomic_parquet(players, out_path)
    atomic_json(
        {
            "schema": "nflreadpy_players_snapshot/1",
            "source": "nflreadpy.load_players()",
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows": len(players),
            "columns": players.columns.tolist(),
            "output_parquet_sha256": sha256_file(out_path),
        },
        destination / "manifest.json",
    )
    return players, out_path


def build_team_game_index(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules.loc[schedules.game_type == "REG"].copy()
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    local = pd.to_datetime(reg.gameday.astype(str).str[:10] + " " + reg.gametime.astype(str))
    reg["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    home = reg[["game_id", "season", "week", "home_team", "away_team", "kickoff"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    home["is_home"] = True
    away = reg[["game_id", "season", "week", "home_team", "away_team", "kickoff"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    away["is_home"] = False
    long = pd.concat([home, away], ignore_index=True)
    long["team"] = canonical_team(long["team"])
    long["opponent"] = canonical_team(long["opponent"])
    return long.sort_values(["season", "week", "team"]).reset_index(drop=True)


def load_injury_reports() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path = latest_under(REPO / "data/raw/nflverse_injuries", "*/injuries.parquet")
    canonical_path = latest_under(REPO / "data/players/raw", "*/injuries.parquet")
    raw = pd.read_parquet(raw_path)
    canonical = pd.read_parquet(canonical_path)
    keys = ["season", "game_type", "week", "team", "gsis_id", "date_modified"]
    basis = canonical[[*keys, "observed_at_basis", "observed_at_is_proxy"]].drop_duplicates()
    merged = raw.merge(basis, on=keys, how="left", validate="many_to_one")
    merged["team"] = canonical_team(merged["team"])
    merged = merged.loc[merged.game_type == "REG"].copy()
    merged["season"] = merged["season"].astype(int)
    merged["week"] = merged["week"].astype(int)
    rest_mask = pd.Series(False, index=merged.index)
    for column in REASON_COLUMNS:
        rest_mask |= merged[column].fillna("").str.contains(REST_PATTERN)
    merged["rest_reason"] = rest_mask
    merged["weekday_modified"] = merged["date_modified"].dt.day_name()
    provenance = {
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "canonical_path": str(canonical_path),
        "canonical_sha256": sha256_file(canonical_path),
    }
    return merged, provenance


def attach_kickoff_and_age(
    frame: pd.DataFrame, team_game_index: pd.DataFrame, birth: pd.DataFrame
) -> pd.DataFrame:
    out = frame.merge(
        team_game_index[["season", "week", "team", "game_id", "kickoff", "opponent", "is_home"]],
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(birth, on="gsis_id", how="left", validate="many_to_one")
    birth_date = pd.to_datetime(out["birth_date"], errors="coerce", utc=True)
    out["age_years"] = (out["kickoff"] - birth_date).dt.days / 365.25
    out["age_known"] = out["birth_date"].notna()
    out["has_game"] = out["game_id"].notna()
    out["age_eligible"] = out["age_known"] & out["age_years"].ge(AGE_THRESHOLD)
    return out


def build_flags(reports: pd.DataFrame) -> pd.DataFrame:
    frame = reports.copy()
    frame["not_dnp"] = frame["practice_status"].notna() & frame["practice_status"].ne(DNP_STATUS)
    frame["flagged"] = (
        frame["rest_reason"] & frame["age_eligible"] & frame["not_dnp"] & frame["has_game"]
    )
    return frame


def season_flag_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in frame.groupby("season"):
        rest_rows = group.loc[group.rest_reason]
        rows.append(
            {
                "season": int(season),
                "reg_injury_rows": len(group),
                "rest_tagged_rows": len(rest_rows),
                "rest_tagged_age30plus": int(
                    (rest_rows.age_known & rest_rows.age_years.ge(AGE_THRESHOLD)).sum()
                ),
                "rest_tagged_missing_birthdate": int((~rest_rows.age_known).sum()),
                "flagged_player_weeks": int(group.flagged.sum()),
                "real_timestamp_share": (
                    float((~group.observed_at_is_proxy.fillna(True).astype(bool)).mean())
                    if len(group)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def load_snap_counts() -> pd.DataFrame:
    path = latest_under(REPO / "data/players/raw", "*/snap_counts.parquet")
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.game_type == "REG"].copy()
    frame["team"] = canonical_team(frame["team"])
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["role_share"] = frame[["offense_pct", "defense_pct"]].max(axis=1).clip(0.0, 1.0)
    frame["played"] = (
        frame[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0.0).sum(axis=1).gt(0.0)
    )
    return frame[
        ["season", "week", "team", "game_id", "pfr_player_id", "position", "role_share", "played"]
    ].drop_duplicates(["season", "week", "pfr_player_id"])


def load_weekly_rosters() -> pd.DataFrame:
    path = latest_under(REPO / "data/players/raw", "*/weekly_rosters.parquet")
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.game_type == "REG"].copy()
    frame["team"] = canonical_team(frame["team"])
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    return frame.drop_duplicates(["season", "week", "team", "gsis_id"])


def crosswalk_gsis_to_pfr(players: pd.DataFrame, weekly_rosters: pd.DataFrame) -> pd.Series:
    from_rosters = weekly_rosters.loc[
        weekly_rosters.gsis_id.notna() & weekly_rosters.pfr_id.notna(), ["gsis_id", "pfr_id"]
    ].drop_duplicates("gsis_id")
    from_players = players.loc[
        players.gsis_id.notna() & players.pfr_id.notna(), ["gsis_id", "pfr_id"]
    ].drop_duplicates("gsis_id")
    combined = pd.concat([from_rosters, from_players], ignore_index=True).drop_duplicates("gsis_id")
    return combined.set_index("gsis_id")["pfr_id"].rename("pfr_player_id")


def trailing_share_table(snap_counts: pd.DataFrame) -> pd.DataFrame:
    ordered = snap_counts.sort_values(["pfr_player_id", "season", "week"]).reset_index(drop=True)
    grouped = ordered.groupby(["pfr_player_id", "season"], sort=False)["role_share"]
    ordered["trailing_share"] = grouped.transform(
        lambda values: values.shift(1).rolling(TRAILING_GAMES, min_periods=1).mean()
    )
    ordered["trailing_games_seen"] = ordered.groupby(["pfr_player_id", "season"]).cumcount()
    return ordered


def team_played_index(team_game_index: pd.DataFrame) -> set[tuple[int, int, str]]:
    return set(
        zip(
            team_game_index["season"].tolist(),
            team_game_index["week"].tolist(),
            team_game_index["team"].tolist(),
            strict=True,
        )
    )


def player_active_index(snap_history: pd.DataFrame) -> set[tuple[str, int, int]]:
    active = snap_history.loc[snap_history.role_share.gt(0.0)]
    return set(
        zip(
            active["pfr_player_id"].tolist(),
            active["season"].tolist(),
            active["week"].tolist(),
            strict=True,
        )
    )


def missed_within_followup(
    rows: pd.DataFrame,
    team_played: set[tuple[int, int, str]],
    player_active: set[tuple[str, int, int]],
) -> pd.Series:
    missed = []
    valid = []
    for row in rows.itertuples():
        any_missed = False
        n_valid = 0
        for offset in range(1, FOLLOWUP_GAMES + 1):
            candidate_week = row.week + offset
            if (row.season, candidate_week, row.team) not in team_played:
                continue
            n_valid += 1
            if (row.pfr_player_id, row.season, candidate_week) not in player_active:
                any_missed = True
        missed.append(any_missed if n_valid else None)
        valid.append(n_valid)
    return pd.DataFrame({"missed_within_4": missed, "n_valid_followups": valid}, index=rows.index)


def summarize_group(rows: pd.DataFrame) -> dict[str, Any]:
    covered = rows.loc[rows.role_share.notna()]
    with_baseline = rows.loc[rows.trailing_games_seen.notna() & rows.trailing_games_seen.ge(1)]
    with_followup = rows.loc[rows.n_valid_followups.fillna(0).gt(0)]
    return {
        "n_player_weeks": len(rows),
        "n_same_week_observed": len(covered),
        "mean_same_week_role_share": (float(covered.role_share.mean()) if len(covered) else None),
        "n_with_trailing_baseline": len(with_baseline),
        "mean_production_drop": (
            float((with_baseline.role_share - with_baseline.trailing_share).mean())
            if len(with_baseline)
            else None
        ),
        "n_with_followup_data": len(with_followup),
        "missed_within_4_rate": (
            float(with_followup.missed_within_4.mean()) if len(with_followup) else None
        ),
    }


def outcome_comparison(
    flags: pd.DataFrame,
    weekly_rosters: pd.DataFrame,
    snap_history: pd.DataFrame,
    gsis_to_pfr: pd.Series,
    team_played: set[tuple[int, int, str]],
    player_active: set[tuple[str, int, int]],
    team_game_index: pd.DataFrame,
    birth: pd.DataFrame,
) -> dict[str, Any]:
    flagged_keys = flags.loc[
        flags.flagged & flags.season.isin(RELIABLE_SEASONS),
        ["season", "week", "team", "gsis_id"],
    ].drop_duplicates()
    flagged_keys["is_flagged"] = True

    universe = weekly_rosters.loc[
        weekly_rosters.status.eq("ACT") & weekly_rosters.season.isin(RELIABLE_SEASONS)
    ].copy()
    universe = attach_kickoff_and_age(universe, team_game_index, birth)
    universe = universe.loc[universe.age_eligible].copy()
    universe = universe.merge(flagged_keys, on=["season", "week", "team", "gsis_id"], how="left")
    universe["is_flagged"] = universe["is_flagged"].fillna(False).astype(bool)
    universe["pfr_player_id"] = universe["gsis_id"].map(gsis_to_pfr)
    universe = universe.loc[universe.pfr_player_id.notna()].copy()

    snap_lookup = snap_history.set_index(["pfr_player_id", "season", "week"])[
        ["role_share", "trailing_share", "trailing_games_seen"]
    ]
    universe = universe.merge(snap_lookup, on=["pfr_player_id", "season", "week"], how="left")

    universe = universe.reset_index(drop=True)
    followup = missed_within_followup(
        universe[["pfr_player_id", "season", "week", "team"]], team_played, player_active
    )
    universe = pd.concat([universe, followup.reset_index(drop=True)], axis=1)

    flagged_rows = universe.loc[universe.is_flagged]
    control_rows = universe.loc[~universe.is_flagged]
    starter_mask = universe.trailing_share.fillna(0.0).ge(STARTER_THRESHOLD)
    flagged_starters = universe.loc[universe.is_flagged & starter_mask]
    control_starters = universe.loc[(~universe.is_flagged) & starter_mask]

    result: dict[str, Any] = {
        "population_seasons": list(RELIABLE_SEASONS),
        "flagged": summarize_group(flagged_rows),
        "control_all_others": summarize_group(control_rows),
        "flagged_starters_only": summarize_group(flagged_starters),
        "control_starters_only": summarize_group(control_starters),
    }
    for half_name, seasons in (("odd", ODD_SEASONS), ("even", EVEN_SEASONS)):
        result[f"flagged_{half_name}"] = summarize_group(
            flagged_rows.loc[flagged_rows.season.isin(seasons)]
        )
        result[f"control_{half_name}"] = summarize_group(
            control_rows.loc[control_rows.season.isin(seasons)]
        )
        result[f"flagged_starters_only_{half_name}"] = summarize_group(
            flagged_starters.loc[flagged_starters.season.isin(seasons)]
        )
        result[f"control_starters_only_{half_name}"] = summarize_group(
            control_starters.loc[control_starters.season.isin(seasons)]
        )
    return result


def reliability_flag_rate(flags: pd.DataFrame, team_game_index: pd.DataFrame) -> dict[str, Any]:
    flagged_team_games = (
        flags.loc[flags.flagged, ["season", "week", "team"]]
        .drop_duplicates()
        .assign(is_flagged_game=True)
    )
    universe = team_game_index.loc[team_game_index.season.isin(RELIABLE_SEASONS)].merge(
        flagged_team_games, on=["season", "week", "team"], how="left"
    )
    universe["is_flagged_game"] = universe["is_flagged_game"].fillna(False)

    def team_rates(seasons: tuple[int, ...]) -> pd.Series:
        subset = universe.loc[universe.season.isin(seasons)]
        return subset.groupby("team")["is_flagged_game"].mean()

    odd_rates = team_rates(ODD_SEASONS)
    even_rates = team_rates(EVEN_SEASONS)
    joined = pd.concat([odd_rates.rename("odd"), even_rates.rename("even")], axis=1).dropna()
    if len(joined) < 4:
        return {"n_teams": len(joined), "pearson_r": None, "interval": None}
    pearson_r = float(joined["odd"].corr(joined["even"]))
    rng = np.random.default_rng(SEED)
    draws = np.empty(SAMPLES)
    values = joined.to_numpy(dtype=float)
    n_units = len(values)
    for i in range(SAMPLES):
        idx = rng.integers(0, n_units, size=n_units)
        odd_s = values[idx, 0]
        even_s = values[idx, 1]
        odd_std = odd_s.std()
        even_std = even_s.std()
        if odd_std == 0.0 or even_std == 0.0:
            draws[i] = 0.0
            continue
        draws[i] = float(
            np.mean((odd_s - odd_s.mean()) * (even_s - even_s.mean())) / (odd_std * even_std)
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "n_teams": len(joined),
        "pearson_r": pearson_r,
        "interval": [float(low), float(high)],
        "probability_positive": float(probability_positive_from_draws(draws)),
    }


def build_team_value(
    flags: pd.DataFrame,
    snap_history: pd.DataFrame,
    gsis_to_pfr: pd.Series,
    team_game_index: pd.DataFrame,
) -> pd.DataFrame:
    flagged = flags.loc[flags.flagged].copy()
    flagged["pfr_player_id"] = flagged["gsis_id"].map(gsis_to_pfr)
    flagged = flagged.loc[flagged.pfr_player_id.notna()]
    lookup = snap_history.set_index(["pfr_player_id", "season", "week"])[
        ["trailing_share", "trailing_games_seen"]
    ]
    flagged = flagged.merge(lookup, on=["pfr_player_id", "season", "week"], how="left")
    starters = flagged.loc[
        flagged.trailing_games_seen.fillna(0).ge(1)
        & flagged.trailing_share.fillna(0).ge(STARTER_THRESHOLD)
    ]
    team_value = (
        starters.groupby(["season", "week", "team"])["trailing_share"]
        .sum()
        .rename("team_value")
        .reset_index()
    )
    indexed = team_game_index.merge(team_value, on=["season", "week", "team"], how="left")
    indexed["team_value"] = indexed["team_value"].fillna(0.0)
    return indexed[["game_id", "season", "week", "team", "is_home", "team_value"]]


def production_screen(
    per_game: pd.DataFrame, team_value: pd.DataFrame, season_start: int, season_end: int
) -> dict[str, Any]:
    home_value = team_value.loc[team_value.is_home, ["game_id", "team_value"]].rename(
        columns={"team_value": "home_value"}
    )
    away_value = team_value.loc[~team_value.is_home, ["game_id", "team_value"]].rename(
        columns={"team_value": "away_value"}
    )
    frame = per_game.merge(home_value, on="game_id", how="left").merge(
        away_value, on="game_id", how="left"
    )
    frame["home_value"] = frame["home_value"].fillna(0.0)
    frame["away_value"] = frame["away_value"].fillna(0.0)
    frame["home_flagged"] = frame["home_value"].gt(0.0)
    frame["away_flagged"] = frame["away_value"].gt(0.0)
    frame["production_pick_home"] = frame["pick_home_at_open_probability_rule"].astype(bool)
    frame["flip"] = frame["home_flagged"].ne(frame["away_flagged"]) & frame[
        "production_pick_home"
    ].eq(frame["home_flagged"])
    frame["candidate_pick_home"] = frame["production_pick_home"] ^ frame["flip"]
    frame["production_correct"] = pick_correct(
        frame["production_pick_home"], frame["margin_vs_open"]
    )
    frame["candidate_correct"] = pick_correct(frame["candidate_pick_home"], frame["margin_vs_open"])
    frame["oracle_pick_home"] = frame["margin_vs_open"].gt(0.0)
    frame["oracle_correct"] = pick_correct(frame["oracle_pick_home"], frame["margin_vs_open"])

    def score(subset: pd.DataFrame, correct_col: str, other_col: str) -> pd.DataFrame:
        valid = subset.dropna(subset=[correct_col, other_col]).copy()
        valid["delta"] = (valid[correct_col] - valid[other_col]) * 100.0

        def metric_fn(block: pd.DataFrame) -> dict[str, float]:
            return {"delta": float(block["delta"].mean())}

        return week_blocked_bootstrap(valid, metric_fn, block="week", samples=SAMPLES, seed=SEED)

    full = frame
    window = frame.loc[frame.season.between(season_start, season_end)]

    results: dict[str, Any] = {}
    for label, subset in (("full_archive", full), ("assigned_window", window)):
        n_games = len(subset)
        n_flips = int(subset["flip"].sum())
        candidate_bootstrap = score(subset, "candidate_correct", "production_correct")
        control_bootstrap = score(subset, "oracle_correct", "production_correct")
        results[label] = {
            "n_games": n_games,
            "n_flips": n_flips,
            "production_accuracy": float(subset["production_correct"].mean() * 100),
            "candidate_accuracy": float(subset["candidate_correct"].mean() * 100),
            "candidate_effect": float(candidate_bootstrap.loc[0, "estimate"]),
            "candidate_interval": [
                float(candidate_bootstrap.loc[0, "lower"]),
                float(candidate_bootstrap.loc[0, "upper"]),
            ],
            "candidate_probability_positive": float(
                candidate_bootstrap.loc[0, "probability_positive"]
            ),
            "positive_control_effect": float(control_bootstrap.loc[0, "estimate"]),
            "positive_control_probability_positive": float(
                control_bootstrap.loc[0, "probability_positive"]
            ),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="LEAD-11 veteran rest-day tell")
    parser.add_argument("--season-start", type=int, default=2020)
    parser.add_argument("--season-end", type=int, default=2021)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()

    players, players_path = load_players_master()
    weekly_rosters = load_weekly_rosters()
    gsis_to_pfr = crosswalk_gsis_to_pfr(players, weekly_rosters)

    snapshot = latest_snapshot(REPO / "data" / "raw")
    schedules, _ = load_snapshot(snapshot)
    team_game_index = build_team_game_index(schedules)

    reports, injury_provenance = load_injury_reports()
    birth = players.loc[players.gsis_id.notna(), ["gsis_id", "birth_date"]].drop_duplicates(
        "gsis_id"
    )
    reports = attach_kickoff_and_age(reports, team_game_index, birth)
    flags = build_flags(reports)
    season_table = season_flag_table(flags)
    print(season_table.to_string(index=False), flush=True)

    if args.coverage_only:
        return 0

    snap_counts = load_snap_counts()
    snap_history = trailing_share_table(snap_counts)
    team_played = team_played_index(team_game_index)
    player_active = player_active_index(snap_history)

    outcome = outcome_comparison(
        flags,
        weekly_rosters,
        snap_history,
        gsis_to_pfr,
        team_played,
        player_active,
        team_game_index,
        birth,
    )
    reliability = reliability_flag_rate(flags, team_game_index)

    team_value = build_team_value(flags, snap_history, gsis_to_pfr, team_game_index)
    match = find_matching_opener_evaluation(REPO / "artifacts")
    if match is None:
        raise ValueError("No opener evaluation matches the active model")
    metadata, per_game_path_dir = match
    per_game_path = per_game_path_dir / "per_game.parquet"
    per_game = pd.read_parquet(per_game_path)
    per_game["season"] = per_game["season"].astype(int)
    per_game["week"] = per_game["week"].astype(int)

    screen = production_screen(per_game, team_value, args.season_start, args.season_end)

    destination = OUTPUT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    metadata_out = {
        "active_model_id": metadata.get("active_model_id"),
        "source_per_game": str(per_game_path),
        "source_per_game_sha256": sha256_file(per_game_path),
        "players_master": str(players_path),
        "players_master_sha256": sha256_file(players_path),
        "injury_provenance": injury_provenance,
        "season_flag_table": season_table.to_dict(orient="records"),
        "outcome_comparison": outcome,
        "reliability_flag_rate_odd_even_season": reliability,
        "production_screen": screen,
        "assigned_window": [args.season_start, args.season_end],
        "seed": SEED,
        "bootstrap_samples": SAMPLES,
        "age_threshold": AGE_THRESHOLD,
        "starter_threshold": STARTER_THRESHOLD,
        "trailing_games": TRAILING_GAMES,
        "followup_games": FOLLOWUP_GAMES,
        "reliable_seasons": list(RELIABLE_SEASONS),
        "odd_seasons": list(ODD_SEASONS),
        "even_seasons": list(EVEN_SEASONS),
        "provenance": artifact_provenance(
            {
                "command": "veteran-rest-day-tell",
                "seed": SEED,
                "samples": SAMPLES,
                "season_start": args.season_start,
                "season_end": args.season_end,
            },
            per_game_path,
        ),
    }
    write_experiment_artifact(
        destination,
        "results.json",
        metadata_out,
        command="veteran-rest-day-tell",
        metrics={
            "assigned_window_effect": screen["assigned_window"]["candidate_effect"],
            "full_archive_effect": screen["full_archive"]["candidate_effect"],
        },
        notes="LEAD-11 veteran rest-day tell: flag build, outcome comparison, production screen",
        registry_root=OUTPUT / "registry",
    )
    atomic_parquet(flags, destination / "flags.parquet")
    atomic_parquet(team_value, destination / "team_value.parquet")
    print(f"wrote {destination}")
    print(
        {
            "outcome": outcome,
            "reliability": reliability,
            "screen": screen,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
