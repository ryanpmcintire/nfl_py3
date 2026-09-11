from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from nfl_ats.availability import availability_rate_lookup, resolve_unavailability
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.io import atomic_json, run_id
from nfl_ats.participation import (
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    build_participation_play_table,
    build_season_lagged_player_ratings,
    load_participation_snapshot,
    participation_snapshot_from_root,
)
from nfl_ats.pbp import PbpSnapshot, load_pbp_snapshot, snapshot_from_root
from nfl_ats.players import (
    _prior_rosters,
    _roster_history,
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "latent_ratings_on_production"

PARTICIPATION_SNAPSHOT = (
    REPO_ROOT / "data" / "players" / "participation" / "raw" / "20260813T131635Z"
)
PBP_SNAPSHOT = REPO_ROOT / "data" / "pbp" / "raw" / "20260817T184927Z"
PLAYER_SNAPSHOT = REPO_ROOT / "data" / "players" / "raw" / "20260910T205112Z"
RATINGS_PATH = REPO_ROOT / "data" / "processed" / "player_participation_ratings.parquet"
AVAILABILITY_RATES_PATH = REPO_ROOT / "data" / "processed" / "weak_stack_availability_rates.parquet"
PRODUCTION_FEATURES = REPO_ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
OPENER_PER_GAME = (
    REPO_ROOT / "artifacts" / "opener_evaluation" / "20260910T211255Z" / "per_game.parquet"
)
FEATURE_TABLE = OUTPUT_ROOT / "expected_lineup_ratings.parquet"

ACTIVE_MODEL_ID = "d49194e04945a5e5"
FEATURE_SEASONS = tuple(range(2017, 2026))
THRESHOLD_SEASONS = (2017, 2018, 2019)
WINDOW_SEASONS = (2020, 2021)
THRESHOLD_QUANTILE = 0.75
ROLE_SPAN = 8
DECISION_HOURS_BEFORE_KICKOFF = 24
RELIABILITY_MIN_PLAYS = 500
ODD_SEASONS = (2017, 2019, 2021, 2023, 2025)
EVEN_SEASONS = (2016, 2018, 2020, 2022, 2024)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260911
NULL_DRAWS = 200
NULL_SEED = 20260911
ACTIVE_ROSTER_STATUSES = frozenset(("ACT", "INA"))


def _load_pbp(seasons: Sequence[int]) -> pd.DataFrame:
    snapshot = snapshot_from_root(PBP_SNAPSHOT)
    wanted = tuple(season for season in snapshot.seasons if season in set(seasons))
    return load_pbp_snapshot(PbpSnapshot(snapshot.snapshot_id, snapshot.root, wanted))


def _play_table() -> pd.DataFrame:
    participation = load_participation_snapshot(
        participation_snapshot_from_root(PARTICIPATION_SNAPSHOT)
    )
    seasons = sorted(participation["season"].astype(int).unique())
    return build_participation_play_table(participation, _load_pbp(seasons))


def _player_ids(raw: Any) -> tuple[str, ...]:
    if pd.isna(raw):
        return ()
    return tuple(value.strip() for value in str(raw).split(";") if value.strip())


def _half_rows(
    rows: Iterable[Any],
    *,
    offense_counts: Counter[str],
    defense_counts: Counter[str],
) -> Iterator[dict[str, float]]:
    for row in rows:
        offense_players = _player_ids(row.offense_players)
        defense_players = _player_ids(row.defense_players)
        offense_counts.update(offense_players)
        defense_counts.update(defense_players)
        values = {f"offense_player::{player_id}": 1.0 for player_id in offense_players}
        values.update({f"defense_player::{player_id}": -1.0 for player_id in defense_players})
        values[f"offense_team::{row.posteam}"] = PARTICIPATION_RATING_TEAM_FEATURE_SCALE
        values[f"defense_team::{row.defteam}"] = -PARTICIPATION_RATING_TEAM_FEATURE_SCALE
        yield values


def _fit_half(
    play_table: pd.DataFrame, seasons: Sequence[int]
) -> tuple[dict[str, float], Counter[str], Counter[str], int]:
    subset = play_table.loc[play_table["season"].astype(int).isin(set(seasons))]
    offense_counts: Counter[str] = Counter()
    defense_counts: Counter[str] = Counter()
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(
        _half_rows(
            subset.itertuples(index=False),
            offense_counts=offense_counts,
            defense_counts=defense_counts,
        )
    )
    target = (
        pd.to_numeric(subset["epa"], errors="raise")
        .clip(-PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_EPA_CLIP)
        .to_numpy(dtype=float)
    )
    estimator = Ridge(
        alpha=PARTICIPATION_RATING_RIDGE_ALPHA, solver="lsqr", fit_intercept=True, tol=1e-6
    )
    estimator.fit(matrix, target)
    coefficients = dict(
        zip(vectorizer.get_feature_names_out(), np.asarray(estimator.coef_), strict=True)
    )
    return coefficients, offense_counts, defense_counts, len(subset)


def _spearman_brown(value: float) -> float:
    if not np.isfinite(value) or value <= -1.0:
        return float("nan")
    return float(2.0 * value / (1.0 + value))


def run_reliability(output_dir: Path) -> dict[str, Any]:
    play_table = _play_table()
    odd = _fit_half(play_table, ODD_SEASONS)
    even = _fit_half(play_table, EVEN_SEASONS)
    result: dict[str, Any] = {
        "mode": "reliability",
        "population": "competitive 11-on-11 participation plays, 2016-2025",
        "half_a_seasons": list(ODD_SEASONS),
        "half_b_seasons": list(EVEN_SEASONS),
        "half_a_plays": odd[3],
        "half_b_plays": even[3],
        "minimum_plays_per_half": RELIABILITY_MIN_PLAYS,
        "configuration": {
            "ridge_alpha": PARTICIPATION_RATING_RIDGE_ALPHA,
            "team_feature_scale": PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
            "epa_clip": PARTICIPATION_RATING_EPA_CLIP,
            "reliability_prior_plays": PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
        },
        "channels": {},
    }
    for channel, prefix, counts_index in (
        ("offense", "offense_player::", 1),
        ("defense", "defense_player::", 2),
    ):
        odd_counts = odd[counts_index]
        even_counts = even[counts_index]
        players = sorted(
            player_id
            for player_id in set(odd_counts) & set(even_counts)
            if odd_counts[player_id] >= RELIABILITY_MIN_PLAYS
            and even_counts[player_id] >= RELIABILITY_MIN_PLAYS
        )
        left = np.asarray([odd[0].get(f"{prefix}{player}", 0.0) for player in players], dtype=float)
        right = np.asarray(
            [even[0].get(f"{prefix}{player}", 0.0) for player in players], dtype=float
        )
        pearson = float(np.corrcoef(left, right)[0, 1]) if len(players) > 2 else float("nan")
        ranks_left = pd.Series(left).rank().to_numpy(dtype=float)
        ranks_right = pd.Series(right).rank().to_numpy(dtype=float)
        spearman = (
            float(np.corrcoef(ranks_left, ranks_right)[0, 1]) if len(players) > 2 else float("nan")
        )
        result["channels"][channel] = {
            "players": len(players),
            "pearson": pearson,
            "spearman": spearman,
            "spearman_brown": _spearman_brown(pearson),
        }
    atomic_json(result, output_dir / "reliability.json")
    return result


def run_reproducibility(output_dir: Path) -> dict[str, Any]:
    participation = load_participation_snapshot(
        participation_snapshot_from_root(PARTICIPATION_SNAPSHOT)
    )
    seasons = sorted(participation["season"].astype(int).unique())
    refit = build_season_lagged_player_ratings(
        participation, _load_pbp(seasons), target_seasons=list(WINDOW_SEASONS)
    )
    stored = pd.read_parquet(RATINGS_PATH)
    stored = stored.loc[stored["target_season"].isin(WINDOW_SEASONS)]
    merged = refit.merge(
        stored,
        on=["target_season", "player_id"],
        how="outer",
        suffixes=("_refit", "_stored"),
        indicator=True,
    )
    both = merged.loc[merged["_merge"].eq("both")]
    result = {
        "mode": "reproducibility",
        "target_seasons": list(WINDOW_SEASONS),
        "refit_rows": len(refit),
        "stored_rows": len(stored),
        "matched_rows": len(both),
        "unmatched_rows": int(len(merged) - len(both)),
        "max_abs_offense_difference": float(
            (both["offense_rating_refit"] - both["offense_rating_stored"]).abs().max()
        ),
        "max_abs_defense_difference": float(
            (both["defense_rating_refit"] - both["defense_rating_stored"]).abs().max()
        ),
    }
    atomic_json(result, output_dir / "reproducibility.json")
    return result


def _visible_injury_index() -> tuple[dict[tuple[int, int, str], list[tuple[str, Any, float]]], int]:
    injuries = pd.read_parquet(PLAYER_SNAPSHOT / "injuries.parquet")
    injuries = injuries.loc[
        injuries["game_type"].eq("REG") & injuries["season"].astype(int).ge(FEATURE_SEASONS[0])
    ].copy()
    lookup = availability_rate_lookup(pd.read_parquet(AVAILABILITY_RATES_PATH))
    severities: list[float] = []
    for row in injuries.itertuples(index=False):
        unavailable, _basis = resolve_unavailability(
            lookup,
            target_season=int(row.season),
            report_status=row.report_status,
            practice_status=row.practice_status,
            position=row.position,
        )
        severities.append(float(unavailable))
    injuries["unavailability"] = severities
    injuries = injuries.sort_values("effective_observed_at")
    index: dict[tuple[int, int, str], list[tuple[str, Any, float]]] = defaultdict(list)
    for row in injuries.itertuples(index=False):
        index[(int(row.season), int(row.week), str(row.team))].append(
            (str(row.gsis_id), row.effective_observed_at, float(row.unavailability))
        )
    return dict(index), len(injuries)


def _snap_index(snaps: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    linked = snaps.loc[snaps["gsis_id"].notna()]
    index: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for row in linked.itertuples(index=False):
        index[(str(row.game_id), str(row.team))].append(
            (str(row.gsis_id), float(row.offense_pct), float(row.defense_pct))
        )
    return dict(index)


def build_feature_table(output_dir: Path) -> pd.DataFrame:
    games = pd.read_parquet(
        PRODUCTION_FEATURES,
        columns=["game_id", "season", "week", "game_type", "kickoff", "home_team", "away_team"],
    )
    games = games.loc[
        games["game_type"].eq("REG") & games["season"].astype(int).isin(set(FEATURE_SEASONS))
    ].copy()
    games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True, errors="coerce")
    games = games.sort_values(["kickoff", "game_id"]).reset_index(drop=True)

    rosters = canonicalize_rosters(pd.read_parquet(PLAYER_SNAPSHOT / "weekly_rosters.parquet"))
    snaps = canonicalize_snaps(pd.read_parquet(PLAYER_SNAPSHOT / "snap_counts.parquet"))
    snaps = attach_snap_player_ids(snaps, rosters)
    snap_index = _snap_index(snaps)
    roster_history = _roster_history(rosters)
    injury_index, injury_rows = _visible_injury_index()

    ratings_frame = pd.read_parquet(RATINGS_PATH)
    ratings: dict[tuple[int, str], tuple[float, float]] = {
        (int(row.target_season), str(row.player_id)): (
            float(row.offense_rating),
            float(row.defense_rating),
        )
        for row in ratings_frame.itertuples(index=False)
    }

    alpha = 2.0 / (ROLE_SPAN + 1.0)
    role_states: dict[str, dict[str, list[float]]] = defaultdict(dict)
    records: list[dict[str, Any]] = []

    for game in games.itertuples(index=False):
        season = int(game.season)
        week = int(game.week)
        kickoff = game.kickoff
        decision_at = (
            pd.NaT
            if pd.isna(kickoff)
            else kickoff - pd.Timedelta(hours=DECISION_HOURS_BEFORE_KICKOFF)
        )
        record: dict[str, Any] = {
            "game_id": str(game.game_id),
            "season": season,
            "week": week,
            "home_team": str(game.home_team),
            "away_team": str(game.away_team),
        }
        for side in ("home", "away"):
            team = str(getattr(game, f"{side}_team"))
            latest_roster, _previous = _prior_rosters(roster_history.get(team, []), (season, week))
            eligible: set[str] | None = None
            if latest_roster is not None:
                eligible = set(
                    latest_roster.loc[
                        latest_roster["status"].isin(ACTIVE_ROSTER_STATUSES), "gsis_id"
                    ]
                    .dropna()
                    .astype(str)
                )
            unavailability: dict[str, float] = {}
            visible_rows = 0
            for player_id, observed_at, severity in injury_index.get((season, week, team), []):
                if pd.isna(decision_at) or pd.isna(observed_at) or observed_at > decision_at:
                    continue
                unavailability[player_id] = severity
                visible_rows += 1
            lineup_offense = 0.0
            lineup_defense = 0.0
            full_offense = 0.0
            full_defense = 0.0
            offense_share = 0.0
            defense_share = 0.0
            rated_players = 0
            for player_id, shares in role_states[team].items():
                if eligible is not None and player_id not in eligible:
                    continue
                rating = ratings.get((season, player_id))
                if rating is None:
                    continue
                play_probability = 1.0 - float(unavailability.get(player_id, 0.0))
                rated_players += 1
                offense_share += shares[0]
                defense_share += shares[1]
                full_offense += shares[0] * rating[0]
                full_defense += shares[1] * rating[1]
                lineup_offense += shares[0] * play_probability * rating[0]
                lineup_defense += shares[1] * play_probability * rating[1]
            record[f"{side}_lineup_offense"] = lineup_offense
            record[f"{side}_lineup_defense"] = lineup_defense
            record[f"{side}_lineup_total"] = lineup_offense + lineup_defense
            record[f"{side}_full_strength_total"] = full_offense + full_defense
            record[f"{side}_divergence"] = (lineup_offense + lineup_defense) - (
                full_offense + full_defense
            )
            record[f"{side}_rated_players"] = rated_players
            record[f"{side}_offense_share_sum"] = offense_share
            record[f"{side}_defense_share_sum"] = defense_share
            record[f"{side}_visible_injury_rows"] = visible_rows
        records.append(record)

        for side in ("home", "away"):
            team = str(getattr(game, f"{side}_team"))
            for player_id, offense_pct, defense_pct in snap_index.get(
                (str(game.game_id), team), []
            ):
                existing = role_states[team].get(player_id)
                if existing is None:
                    role_states[team][player_id] = [offense_pct, defense_pct]
                else:
                    existing[0] = alpha * offense_pct + (1.0 - alpha) * existing[0]
                    existing[1] = alpha * defense_pct + (1.0 - alpha) * existing[1]

    frame = pd.DataFrame.from_records(records)
    frame["diff_lineup_total"] = frame["home_lineup_total"] - frame["away_lineup_total"]
    frame["diff_divergence"] = frame["home_divergence"] - frame["away_divergence"]
    frame["max_abs_divergence"] = np.maximum(
        frame["home_divergence"].abs(), frame["away_divergence"].abs()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURE_TABLE, index=False)
    summary = {
        "mode": "feature",
        "games": len(frame),
        "seasons": [int(frame["season"].min()), int(frame["season"].max())],
        "injury_rows_indexed": injury_rows,
        "mean_rated_players_home": float(frame["home_rated_players"].mean()),
        "mean_offense_share_sum_home": float(frame["home_offense_share_sum"].mean()),
        "mean_defense_share_sum_home": float(frame["home_defense_share_sum"].mean()),
        "destination": str(FEATURE_TABLE),
    }
    atomic_json(summary, output_dir / "feature_summary.json")
    return frame


def _thresholds(frame: pd.DataFrame) -> dict[str, float]:
    pre = frame.loc[frame["season"].isin(THRESHOLD_SEASONS)]
    return {
        "M1": float(pre["diff_lineup_total"].abs().quantile(THRESHOLD_QUANTILE)),
        "M2": float(pre["diff_divergence"].abs().quantile(THRESHOLD_QUANTILE)),
        "C": float(pre["max_abs_divergence"].quantile(THRESHOLD_QUANTILE)),
        "threshold_games": len(pre),
    }


def _paired_frame(feature: pd.DataFrame, seasons: Sequence[int]) -> pd.DataFrame:
    per_game = pd.read_parquet(OPENER_PER_GAME)
    per_game = per_game.loc[per_game["season"].astype(int).isin(set(seasons))]
    merged = per_game.merge(
        feature.drop(columns=["season", "week"]), on="game_id", how="left", validate="one_to_one"
    )
    if merged["diff_lineup_total"].isna().any():
        missing = merged.loc[merged["diff_lineup_total"].isna(), "game_id"].tolist()
        raise ValueError(f"feature coverage gap on {len(missing)} games: {missing[:5]}")
    merged = merged.loc[merged["correct_at_open_probability_rule"].notna()].copy()
    merged["home_cover"] = (merged["margin_vs_open"] > 0).astype(float)
    merged["baseline_pick_home"] = merged["pick_home_at_open_probability_rule"].astype(bool)
    merged["baseline_correct"] = merged["correct_at_open_probability_rule"].astype(float)
    merged["baseline_probability"] = merged["home_cover_probability_at_open"].astype(float)
    return merged.reset_index(drop=True)


def _apply_tilt(
    frame: pd.DataFrame, signal: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    baseline_pick = frame["baseline_pick_home"].to_numpy(dtype=bool)
    tilted = baseline_pick.copy()
    active = np.abs(signal) >= threshold
    tilted[active] = signal[active] > 0
    flipped = tilted != baseline_pick
    return tilted, flipped


def _scored(frame: pd.DataFrame, tilted: np.ndarray, flipped: np.ndarray) -> pd.DataFrame:
    home_cover = frame["home_cover"].to_numpy(dtype=float)
    candidate_correct = np.where(tilted, home_cover, 1.0 - home_cover)
    probability = frame["baseline_probability"].to_numpy(dtype=float)
    candidate_probability = np.where(flipped, 1.0 - probability, probability)
    return pd.DataFrame(
        {
            "season": frame["season"].to_numpy(),
            "week": frame["week"].to_numpy(),
            "baseline_correct": frame["baseline_correct"].to_numpy(dtype=float),
            "candidate_correct": candidate_correct,
            "baseline_brier": (probability - home_cover) ** 2,
            "candidate_brier": (candidate_probability - home_cover) ** 2,
            "flipped": flipped.astype(float),
        }
    )


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy_delta_points": 100.0
        * float(frame["candidate_correct"].mean() - frame["baseline_correct"].mean()),
        "baseline_accuracy": float(frame["baseline_correct"].mean()),
        "candidate_accuracy": float(frame["candidate_correct"].mean()),
        "brier_improvement": float(
            frame["baseline_brier"].mean() - frame["candidate_brier"].mean()
        ),
    }


def _uncertainty(scored: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for block in ("week", "season"):
        table = week_blocked_bootstrap(
            scored,
            _metrics,
            block=block,
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        output[block] = {
            str(row.metric): {
                "estimate": float(row.estimate),
                "lower": float(row.lower),
                "upper": float(row.upper),
                "probability_positive": float(row.probability_positive),
            }
            for row in table.itertuples(index=False)
        }
    return output


def _screen_arm(
    paired: pd.DataFrame, signal_column: str, threshold: float, label: str
) -> dict[str, Any]:
    signal = paired[signal_column].to_numpy(dtype=float)
    tilted, flipped = _apply_tilt(paired, signal, threshold)
    scored = _scored(paired, tilted, flipped)
    metrics = _metrics(scored)
    per_season = {
        str(season): {
            "games": len(group),
            "baseline_accuracy": float(group["baseline_correct"].mean()),
            "candidate_accuracy": float(group["candidate_correct"].mean()),
            "accuracy_delta_points": 100.0
            * float(group["candidate_correct"].mean() - group["baseline_correct"].mean()),
            "picks_changed": int(group["flipped"].sum()),
        }
        for season, group in scored.groupby("season", sort=True)
    }
    flipped_frame = scored.loc[scored["flipped"].eq(1.0)]
    return {
        "label": label,
        "signal_column": signal_column,
        "threshold": threshold,
        "games": len(scored),
        "weeks": int(scored.groupby(["season", "week"]).ngroups),
        "picks_changed": int(scored["flipped"].sum()),
        "tilt_eligible_games": int((np.abs(signal) >= threshold).sum()),
        "flipped_baseline_accuracy": (
            float(flipped_frame["baseline_correct"].mean()) if len(flipped_frame) else float("nan")
        ),
        "flipped_candidate_accuracy": (
            float(flipped_frame["candidate_correct"].mean()) if len(flipped_frame) else float("nan")
        ),
        "metrics": metrics,
        "uncertainty": _uncertainty(scored),
        "per_season": per_season,
    }


def _cell_arm(
    paired: pd.DataFrame, signal_column: str, threshold: float, cell_mask: np.ndarray, label: str
) -> dict[str, Any]:
    signal = paired[signal_column].to_numpy(dtype=float)
    tilted, flipped = _apply_tilt(paired, signal, threshold)
    scored = _scored(paired, tilted, flipped)
    inside = scored.loc[cell_mask].reset_index(drop=True)
    outside = scored.loc[~cell_mask].reset_index(drop=True)
    payload: dict[str, Any] = {"label": label, "signal_column": signal_column}
    for name, subset in (("inside_cell", inside), ("outside_cell", outside)):
        if subset.empty:
            payload[name] = {"games": 0}
            continue
        payload[name] = {
            "games": len(subset),
            "weeks": int(subset.groupby(["season", "week"]).ngroups),
            "picks_changed": int(subset["flipped"].sum()),
            "metrics": _metrics(subset),
            "uncertainty": _uncertainty(subset),
        }
    return payload


def _null_distribution(
    paired: pd.DataFrame, signal_column: str, threshold: float, seed: int
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    week_positions = list(paired.groupby(["season", "week"], sort=False).indices.values())
    base = paired[signal_column].to_numpy(dtype=float)
    deltas: list[float] = []
    flips: list[int] = []
    for _ in range(NULL_DRAWS):
        permuted = base.copy()
        for positions in week_positions:
            permuted[positions] = generator.permutation(base[positions])
        tilted, flipped = _apply_tilt(paired, permuted, threshold)
        scored = _scored(paired, tilted, flipped)
        deltas.append(_metrics(scored)["accuracy_delta_points"])
        flips.append(int(flipped.sum()))
    values = np.asarray(deltas, dtype=float)
    observed_tilted, observed_flipped = _apply_tilt(paired, base, threshold)
    observed = _metrics(_scored(paired, observed_tilted, observed_flipped))["accuracy_delta_points"]
    return {
        "draws": NULL_DRAWS,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "mean_picks_changed": float(np.mean(flips)),
        "observed": float(observed),
        "observed_picks_changed": int(observed_flipped.sum()),
        "observed_percentile": float(100.0 * np.mean(values <= observed)),
    }


def run_screen(output_dir: Path, *, mode: str) -> dict[str, Any]:
    feature = pd.read_parquet(FEATURE_TABLE)
    thresholds = _thresholds(feature)
    paired = _paired_frame(feature, WINDOW_SEASONS)
    payload: dict[str, Any] = {
        "mode": mode,
        "active_model_id": ACTIVE_MODEL_ID,
        "opener_archive": str(OPENER_PER_GAME.relative_to(REPO_ROOT)),
        "window_seasons": list(WINDOW_SEASONS),
        "thresholds": thresholds,
        "paired_games": len(paired),
        "predeclaration": "docs/latent_ratings_on_production.md",
    }

    if mode == "positive-control":
        paired = paired.copy()
        paired["oracle_signal"] = paired["margin_vs_open"].astype(float)
        oracle_threshold = float(paired["oracle_signal"].abs().quantile(THRESHOLD_QUANTILE))
        payload["oracle_threshold"] = oracle_threshold
        payload["arms"] = {
            "oracle": _screen_arm(paired, "oracle_signal", oracle_threshold, "positive control")
        }
        atomic_json(payload, output_dir / "positive_control.json")
        return payload

    if mode == "null":
        payload["arms"] = {
            "screen_a": _null_distribution(
                paired, "diff_lineup_total", thresholds["M1"], NULL_SEED
            ),
            "screen_b": _null_distribution(
                paired, "diff_divergence", thresholds["M2"], NULL_SEED + 1
            ),
        }
        atomic_json(payload, output_dir / "null.json")
        return payload

    cell_mask = paired["max_abs_divergence"].to_numpy(dtype=float) >= thresholds["C"]
    payload["cell_games"] = int(cell_mask.sum())
    payload["arms"] = {
        "screen_a": _screen_arm(
            paired, "diff_lineup_total", thresholds["M1"], "expected-lineup rating tilt"
        ),
        "screen_b": _screen_arm(
            paired, "diff_divergence", thresholds["M2"], "lineup-divergence tilt"
        ),
    }
    payload["cells"] = {
        "screen_a": _cell_arm(
            paired, "diff_lineup_total", thresholds["M1"], cell_mask, "expected-lineup rating tilt"
        ),
        "screen_b": _cell_arm(
            paired, "diff_divergence", thresholds["M2"], cell_mask, "lineup-divergence tilt"
        ),
    }
    payload["baseline"] = {
        "accuracy": float(paired["baseline_correct"].mean()),
        "games": len(paired),
        "home_pick_rate": float(paired["baseline_pick_home"].mean()),
    }
    payload["diagnostics"] = {
        column: {
            "correlation_with_opener_home_spread": float(
                np.corrcoef(paired[column], paired["tue_open_home_spread"])[0, 1]
            ),
            "correlation_with_opener_ats_margin": float(
                np.corrcoef(paired[column], paired["margin_vs_open"])[0, 1]
            ),
        }
        for column in ("diff_lineup_total", "diff_divergence")
    }
    atomic_json(payload, output_dir / "results.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PER-09 latent player ratings, aggregated to an expected-lineup rating sum and "
            "screened on top of the served production opener pick "
            "(docs/latent_ratings_on_production.md)"
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("reliability", "reproducibility", "feature", "positive-control", "null", "screen"),
        help="which stage to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory; defaults to a new timestamped run directory",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (OUTPUT_ROOT / run_id())
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "reliability":
        payload: dict[str, Any] = run_reliability(output_dir)
    elif args.mode == "reproducibility":
        payload = run_reproducibility(output_dir)
    elif args.mode == "feature":
        build_feature_table(output_dir)
        payload = json.loads((output_dir / "feature_summary.json").read_text(encoding="utf-8"))
    else:
        payload = run_screen(output_dir, mode=args.mode)

    payload["output_dir"] = str(output_dir)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
