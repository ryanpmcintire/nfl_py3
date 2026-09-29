from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from players_on_field_rating_eval import (
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
from players_on_field_rating_unit2 import evaluate_candidate

from nfl_ats.pick_probability_fit import FIT_FEATURES, build_fit_population
from nfl_ats.signal_atlas import _metrics as signal_metrics

SNAPSHOT = DATA_ROOT / "players" / "raw" / "20260910T205112Z"
FEATURE_TABLE = DATA_ROOT / "processed" / "game_features_weak_stack.parquet"
OUTPUT_ROOT = REPO_ROOT / "tests" / "scratch" / "codex" / "mod22_unit4"
OUT_STATUSES = {"Out", "Doubtful"}
DECISION_HOURS = 24
SEASON_START, SEASON_END = 2020, 2025


def norm(name: str) -> str:
    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    parts = re.sub(r"[^a-z\s]", "", value).split()
    while parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        parts.pop()
    return "".join(parts)


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
        columns=[
            "season",
            "week",
            "team",
            "gsis_id",
            "report_status",
            "date_modified",
            "effective_observed_at",
            "observed_at_basis",
            "observed_at_is_proxy",
        ],
    )
    injuries["effective_observed_at"] = pd.to_datetime(
        injuries["effective_observed_at"], utc=True, errors="coerce"
    )
    injuries["date_modified"] = pd.to_datetime(injuries["date_modified"], utc=True, errors="coerce")
    injuries = injuries.loc[injuries["report_status"].isin(OUT_STATUSES)].copy()
    evidenced = (
        injuries["observed_at_basis"].eq("date_modified")
        & injuries["observed_at_is_proxy"].eq(False)
        & injuries["date_modified"].notna()
        & injuries["effective_observed_at"].eq(injuries["date_modified"])
    )
    injuries["evidenced_report_at"] = injuries["date_modified"].where(evidenced)
    inj_index = {key: frame for key, frame in injuries.groupby(["season", "week", "team"])}  # noqa: C416

    match_total = match_eligible = match_hit = 0
    matched_names = Counter()
    unmatched_names = Counter()
    audits = []
    rows = []
    window = games.loc[(games["season"] >= SEASON_START) & (games["season"] <= SEASON_END)]
    for game in window.itertuples():
        flags = {}
        for side, team in (("home", game.home_team), ("away", game.away_team)):
            flag = 0.0
            audit = {
                "game_id": game.game_id,
                "side": side,
                "team": team,
                "decision_at": game.decision_at,
                "expected_starter": None,
                "matched_ids": 0,
                "cutoff_only_flag": False,
                "excluded_proxy_reports": 0,
                "excluded_unevidenced_reports": 0,
                "excluded_late_reports": 0,
                "earliest_evidenced_report_at": pd.NaT,
            }
            frame_team = starts_by_team.get(team)
            prior = None
            if frame_team is not None:
                earlier = frame_team.loc[frame_team["kickoff"] < game.kickoff]
                if not earlier.empty:
                    prior = earlier.iloc[-1]
            match_total += 1
            if prior is not None:
                match_eligible += 1
                audit["expected_starter"] = prior["player"]
                assert prior["kickoff"] < game.decision_at
                ids = gsis_by_key.get((team, norm(prior["player"])), set())
                audit["matched_ids"] = len(ids)
                if ids:
                    match_hit += 1
                    matched_names[str(prior["player"])] += 1
                    frame = inj_index.get((game.season, game.week, team))
                    if frame is not None:
                        reports = frame.loc[frame["gsis_id"].isin(ids)]
                        audit["cutoff_only_flag"] = bool(
                            reports["effective_observed_at"].le(game.decision_at).any()
                        )
                        audit["excluded_proxy_reports"] = int(
                            reports["observed_at_is_proxy"].eq(True).sum()
                        )
                        audit["excluded_unevidenced_reports"] = int(
                            reports["evidenced_report_at"].isna().sum()
                        )
                        audit["excluded_late_reports"] = int(
                            reports["evidenced_report_at"].gt(game.decision_at).sum()
                        )
                        hit = reports.loc[reports["evidenced_report_at"].le(game.decision_at)]
                        assert hit["observed_at_basis"].eq("date_modified").all()
                        assert hit["observed_at_is_proxy"].eq(False).all()
                        assert hit["date_modified"].le(game.decision_at).all()
                        audit["earliest_evidenced_report_at"] = hit["evidenced_report_at"].min()
                        flag = float(len(hit) > 0)
                else:
                    unmatched_names[str(prior["player"])] += 1
            audit["qb_out"] = bool(flag)
            audits.append(audit)
            flags[side] = flag
        rows.append((game.game_id, flags["home"], flags["away"]))
    qb_flags = pd.DataFrame(rows, columns=["game_id", "home_qb_out", "away_qb_out"])
    qb_flags["qb_out_diff"] = qb_flags["away_qb_out"] - qb_flags["home_qb_out"]
    audit_frame = pd.DataFrame(audits)

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
    outcomes = population["home_covered"].to_numpy(dtype=float)
    candidate_is_metrics = signal_metrics(cand_is, outcomes, RELIABILITY_EDGES)
    base_is_metrics = signal_metrics(base_is, outcomes, RELIABILITY_EDGES)
    for metric in ("brier", "log_loss"):
        result["is_vs_oos_gap"][f"candidate_in_sample_{metric}"] = candidate_is_metrics[metric]
        result["is_vs_oos_gap"][f"base_in_sample_{metric}"] = base_is_metrics[metric]
        improvement = base_is_metrics[metric] - candidate_is_metrics[metric]
        result["is_vs_oos_gap"][f"in_sample_{metric}_improvement"] = improvement
        result["is_vs_oos_gap"][f"{metric}_improvement_gap"] = (
            improvement - result["paired_cell"][f"{metric}_improvement"]
        )
    population["candidate_oos_probability"] = cand_oos
    population["candidate_is_probability"] = cand_is
    scored_audit = audit_frame.loc[audit_frame["game_id"].isin(population["game_id"])]

    summary = {
        "created_at_utc": now.isoformat(),
        "command": ".tools/uv.exe run --no-sync python scripts/mod22_unit4.py",
        "measurement": "evidenced_timestamp_and_suffix_remeasurement",
        "family": "qb_expected_starter_v1",
        "look_count": 1,
        "declaration": declaration,
        "timestamp_evidence": "non-proxy date_modified matching effective_observed_at",
        "provenance": provenance,
        "games": len(population),
        "games_home_qb_out": int(population["home_qb_out"].sum()),
        "games_away_qb_out": int(population["away_qb_out"].sum()),
        "games_nonzero_term": int((population["qb_out_diff"] != 0).sum()),
        "name_to_gsis_match_rate": match_hit / match_total,
        "name_matching": {
            "team_games": match_total,
            "eligible_previous_starters": match_eligible,
            "matched": match_hit,
            "eligible_match_rate": match_hit / match_eligible if match_eligible else None,
            "without_previous_starter": match_total - match_eligible,
            "ambiguous_team_games": int(audit_frame["matched_ids"].gt(1).sum()),
            "matched_names": dict(sorted(matched_names.items())),
            "unmatched_names": dict(sorted(unmatched_names.items())),
        },
        "timestamp_audit_scored_team_games": {
            "cutoff_only_flags": int(scored_audit["cutoff_only_flag"].sum()),
            "evidenced_flags": int(scored_audit["qb_out"].sum()),
            "flags_removed_without_evidence": int(
                (scored_audit["cutoff_only_flag"] & ~scored_audit["qb_out"]).sum()
            ),
            "excluded_proxy_reports": int(scored_audit["excluded_proxy_reports"].sum()),
            "excluded_unevidenced_reports": int(scored_audit["excluded_unevidenced_reports"].sum()),
            "excluded_late_reports": int(scored_audit["excluded_late_reports"].sum()),
        },
        "look_qb_out_diff": result,
    }
    out = OUTPUT_ROOT / now.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    population.to_parquet(out / "per_game.parquet", index=False)
    audit_frame.to_parquet(out / "team_game_audit.parquet", index=False)
    print(str(out.relative_to(REPO_ROOT)))
    cell = result["paired_cell"]
    print(
        json.dumps(
            {
                "games": len(population),
                "name_to_gsis_match_rate": summary["name_to_gsis_match_rate"],
                "timestamp_audit": summary["timestamp_audit_scored_team_games"],
                "brier_improvement": cell["brier_improvement"],
                "brier_interval": cell["brier_interval"],
                "probability_positive": cell["brier_probability_positive"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
