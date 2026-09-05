"""Movement leads battery: predeclared measurement (measure-only).

Predeclaration: ``docs/movement_leads_battery.md`` (frozen before any
accuracy sign below was computed). Answers three Phase 12 ROADMAP leads --
LEAD-07 (movement-timing day-part decomposition), LEAD-01 (Wednesday-
revision follow, which turns out to be LEAD-07 cell (a) in its playable
form), and LEAD-06 (rising-total, stable-spread underdog) -- on the local
point-in-time odds archive. Reuses the observed-movement threshold-overlay
construction from ``scripts/observed_movement_channel.py`` and the
rotation-window-governed scoring shape from
``scripts/movement_expansion_battery.py`` -- ``oracle``/``threshold``-style
picks, ``score_cell``, and the 200-draw within-week permutation null are
imported UNMODIFIED from that sibling script rather than re-implemented,
since neither file is edited by this change.

New rotation-registry family (``movement_leads_v1``, grade ``opener``,
``--acknowledge-mined``), declared and assigned ``[2020, 2021]`` before this
script ran (see the predeclaration doc's "Rotation-registry window"
section). That window covers only the ONE cell that does not need a
Wednesday reading (``movement_leads_rising_total_dog``); the other four
cells use the full 2023-2025 ``intraday_hourly``-covered population instead,
since no Wednesday checkpoint exists anywhere before 2023 -- a disclosed,
predeclared scope decision (see the doc's "Disclosed tension" section), not
a silent widening.

Five predeclared cells:
  1. movement_leads_wed_follow_1_0       -- day-part (a) / LEAD-01, playable
  2. movement_leads_sat_follow_1_0       -- day-part (b), playable
  3. movement_leads_sun_am_follow_1_0    -- day-part (c), playable
  4. movement_leads_sun_vs_wed_per_point -- PRIMARY: (c) minus (a) per-point value
  5. movement_leads_rising_total_dog     -- LEAD-06

Plus perfect-foresight positive controls (instrument diagnostics, NOT
recorded to the weak-signal registry) on both population sizes this family
uses.

Every threshold cell's paired flip-value = candidate pick minus production
pick (``pick_home_at_open_probability_rule``), both graded at
``margin_vs_open`` (the frozen Tuesday line), reported regardless of sign
(AGENTS.md). Week-blocked bootstrap primary, season-blocked secondary
(``nfl_ats.clv.week_blocked_bootstrap``, seed 20260905, 20,000 samples),
plus a 200-draw within-week permutation null.

Writes ``artifacts/movement_leads_battery/<run_id>/`` (``per_game_dayparts.parquet``,
``per_game_rising_total.parquet``, ``cells_summary.csv``, ``metadata.json``
via ``write_experiment_artifact``, which also stamps
``registry/experiments/``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.clv import (  # noqa: E402
    DECISION_LABEL_ORDER,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    decision_market_consensus,
    load_decision_quotes,
    pick_correct,
    week_blocked_bootstrap,
)
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import load_registry  # noqa: E402

# Reused unmodified from the sibling template (see the module docstring and
# docs/movement_leads_battery.md's "Construction" section for why these are
# imported rather than re-implemented).
from scripts.movement_expansion_battery import (  # noqa: E402
    NULL_PERMUTATIONS,
    PRODUCTION_PICK_COL,
    permuted_margins,
    row_or_nan,
    score_cell,
    threshold_pick,
    week_positions,
)

DEFAULT_ARCHIVE = REPO_ROOT / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"
DEFAULT_GAME_FEATURES = REPO_ROOT / "data/processed/game_features.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/movement_leads_battery"
DEFAULT_REGISTRY_ROOT = REPO_ROOT / "registry"
ROTATION_FAMILY = "movement_leads_v1"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260905  # this document's own seed (docs/movement_leads_battery.md)
THRESHOLD = 1.0
RISING_TOTAL_THRESHOLD = 2.0
STABLE_SPREAD_THRESHOLD = 0.5

INTRADAY_LABEL = "intraday_hourly"
INTRADAY_SEASONS: tuple[int, ...] = (2023, 2024, 2025)
EASTERN = ZoneInfo("America/New_York")
# Day offsets match odds_backfill.DECISION_TIMES' own convention
# (tue_open=-5, thu_pre_tnf=-3, sat_midday=-1): negative = days before that
# week's anchor Sunday. Wednesday sits between Tuesday and Thursday, at -4.
WEDNESDAY_DAY_OFFSET = -4
WEDNESDAY_HOUR_ET = 12  # "Wed-noon", LEAD-01's own ROADMAP-declared instrument
SUNDAY_DAY_OFFSET = 0
SUNDAY_HOUR_ET = (
    16  # true deadline; real archive ceiling ~10:55 ET (docs/observed_movement_channel.md)
)
# tue_open < thu_pre_tnf < sat_midday < sun_early_close: sun_late_close and
# mon_pre_mnf are excluded even when they precede a game's own kickoff --
# both sit structurally after the owner's Sunday 16:00 ET pick deadline.
PRE_DEADLINE_LABELS: tuple[str, ...] = ("tue_open", "thu_pre_tnf", "sat_midday", "sun_early_close")


# ---------------------------------------------------------------------------
# Intraday-hourly loading: duplicated (not imported) from
# observed_movement_channel.py, whose own docstring explains the identical
# choice for its ``_true_week_correct`` helper -- module-private, not meant
# for cross-script reuse.
# ---------------------------------------------------------------------------


def _true_week_correct(quotes: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Restrict quotes to each game's own scheduled (season, week)."""

    if quotes.empty:
        return quotes
    required = {"game_id", "season", "week"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Correction schedule is missing columns: {', '.join(missing)}")
    true_week = (
        schedule[["game_id", "season", "week"]]
        .drop_duplicates("game_id")
        .rename(
            columns={"game_id": "nflverse_game_id", "season": "_true_season", "week": "_true_week"}
        )
    )
    merged = quotes.merge(true_week, on="nflverse_game_id", how="left")
    return merged.loc[
        merged["season"].eq(merged["_true_season"]) & merged["week"].eq(merged["_true_week"])
    ].drop(columns=["_true_season", "_true_week"])


def _weekday_cutoff_et_utc(
    week_first_commence_utc: pd.Series, day_offset_from_sunday: int, hour: int
) -> pd.Series:
    """Per-row: the America/New_York day (offset from that week's anchor
    Sunday) at ``hour``:00 -- generalizes
    ``observed_movement_channel.py``'s ``_sunday_cutoff_et_utc``
    (``day_offset_from_sunday=0`` there, always).
    """

    local = week_first_commence_utc.dt.tz_convert(EASTERN)
    naive_date = local.dt.tz_localize(None).dt.normalize()
    days_ahead = (6 - local.dt.weekday) % 7
    sunday_naive_date = naive_date + pd.to_timedelta(days_ahead, unit="D")
    target_naive_date = sunday_naive_date + pd.Timedelta(days=day_offset_from_sunday)
    target_naive = target_naive_date + pd.Timedelta(hours=hour)
    return target_naive.dt.tz_localize(EASTERN).dt.tz_convert("UTC")


def _load_intraday_with_kickoff(
    root: Path, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 2023-2025 ``intraday_hourly`` archive once, plus a per-game kickoff table."""

    raw = load_decision_quotes(
        root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=(INTRADAY_LABEL,),
        seasons=INTRADAY_SEASONS,
    )
    if raw.empty:
        return raw, pd.DataFrame(columns=["nflverse_game_id", "commence_time_utc"])
    corrected = _true_week_correct(raw, schedule)
    kickoff = (
        corrected[["nflverse_game_id", "season", "week", "commence_time_utc"]]
        .dropna(subset=["nflverse_game_id"])
        .groupby("nflverse_game_id", as_index=False)
        .agg(
            commence_time_utc=("commence_time_utc", "min"),
            season=("season", "first"),
            week=("week", "first"),
        )
    )
    kickoff["week_first_commence_utc"] = kickoff.groupby(["season", "week"])[
        "commence_time_utc"
    ].transform("min")
    return corrected, kickoff


def _home_spread_at_weekday_cutoff(
    corrected: pd.DataFrame,
    kickoff: pd.DataFrame,
    *,
    day_offset_from_sunday: int,
    hour: int,
    column_name: str,
) -> tuple[pd.DataFrame, int]:
    """Per-game HOME spread from the last intraday capture at or before
    ``min(kickoff, that week's target weekday/hour ET)`` -- the deadline
    guard (AGENTS.md: every "follow the move" cell uses only snapshots at or
    before ``min(kickoff, Sunday 16:00 ET)``). Generalizes
    ``observed_movement_channel.py``'s ``_spread_at_cutoff`` (Sunday-only
    there) to an arbitrary day offset from the week's anchor Sunday.

    Returns ``(frame[game_id, column_name], n_games_missing_a_qualifying_capture)``.
    """

    if corrected.empty or kickoff.empty:
        return pd.DataFrame(columns=["game_id", column_name]), 0

    working_kickoff = kickoff.copy()
    working_kickoff["_target_et_utc"] = _weekday_cutoff_et_utc(
        working_kickoff["week_first_commence_utc"], day_offset_from_sunday, hour
    )
    working_kickoff["cutoff_utc"] = working_kickoff[["_target_et_utc", "commence_time_utc"]].min(
        axis=1
    )

    cutoff_lookup = working_kickoff.set_index("nflverse_game_id")["cutoff_utc"]
    quotes = corrected.merge(
        cutoff_lookup.rename("cutoff_utc"),
        left_on="nflverse_game_id",
        right_index=True,
        how="inner",
    )
    eligible_quotes = quotes.loc[quotes["observed_at_utc"].le(quotes["cutoff_utc"])].copy()
    n_games_total = quotes["nflverse_game_id"].nunique()

    if eligible_quotes.empty:
        return pd.DataFrame(columns=["game_id", column_name]), n_games_total

    consensus = decision_market_consensus(eligible_quotes.drop(columns=["cutoff_utc"]))
    spread = consensus.loc[
        consensus["market"].eq("spreads") & consensus["outcome_side"].eq("HOME")
    ][["nflverse_game_id", "consensus_line"]].rename(
        columns={"nflverse_game_id": "game_id", "consensus_line": column_name}
    )
    n_missing = n_games_total - spread["game_id"].nunique()
    return spread, n_missing


# ---------------------------------------------------------------------------
# LEAD-06: latest pre-deadline historical_backfill checkpoint
# ---------------------------------------------------------------------------


def load_latest_pre_deadline(
    market_root: Path, game_features_path: Path, seasons: tuple[int, int]
) -> pd.DataFrame:
    """Per-game latest pre-Sunday-16:00-ET-deadline home spread + total line.

    Priority order ``tue_open`` < ``thu_pre_tnf`` < ``sat_midday`` <
    ``sun_early_close`` (``DECISION_LABEL_ORDER``); ``sun_late_close`` and
    ``mon_pre_mnf`` are excluded even for games whose own kickoff is later,
    because both checkpoints sit structurally after the owner's Sunday
    16:00 ET pick deadline (``min(kickoff, Sunday 16:00 ET)``) -- the
    deadline guard.
    """

    features = pd.read_parquet(game_features_path)
    schedule = features.loc[features["game_type"].eq("REG"), ["game_id", "season", "week"]].dropna()
    schedule = schedule.assign(
        season=schedule["season"].astype(int), week=schedule["week"].astype(int)
    )
    season_list = list(range(seasons[0], seasons[1] + 1))
    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=PRE_DEADLINE_LABELS,
        seasons=season_list,
        schedule=schedule,
    )
    ordered = pairing.assign(_order=pairing["decision_label"].map(DECISION_LABEL_ORDER))
    latest = ordered.sort_values(["game_id", "_order"]).groupby("game_id", as_index=False).tail(1)
    latest = latest[["game_id", "decision_label", "home_spread", "total_line"]].rename(
        columns={
            "decision_label": "latest_pre_deadline_label",
            "home_spread": "latest_pre_deadline_home_spread",
            "total_line": "latest_pre_deadline_total_line",
        }
    )
    tue_open_total = ordered.loc[ordered["decision_label"].eq("tue_open")][
        ["game_id", "total_line"]
    ].rename(columns={"total_line": "tue_open_total_line"})
    return latest.merge(tue_open_total, on="game_id", how="left")


def rising_total_dog_pick(
    tue_open_total: pd.Series,
    latest_total: pd.Series,
    tue_open_spread: pd.Series,
    latest_spread: pd.Series,
    production_home: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """LEAD-06: total rose >= 2.0 with the spread stable (< 0.5 move) -> take the dog."""

    total_move = latest_total - tue_open_total
    spread_move = (latest_spread - tue_open_spread).abs()
    flagged = total_move.ge(RISING_TOTAL_THRESHOLD) & spread_move.lt(STABLE_SPREAD_THRESHOLD)
    dog_home = np.where(
        tue_open_spread.gt(0.0),
        True,
        np.where(tue_open_spread.lt(0.0), False, production_home.astype(bool)),
    )
    pick = np.where(flagged.fillna(False), dog_home, production_home.astype(bool))
    return (
        pd.Series(pick, index=tue_open_total.index).astype(bool),
        pd.Series(flagged.fillna(False), index=tue_open_total.index),
    )


# ---------------------------------------------------------------------------
# Cell 4: per-point value, paired difference between day-parts (c) and (a)
# ---------------------------------------------------------------------------


def per_point_value_diff_metric(
    *,
    pick_a_col: str,
    move_a_col: str,
    pick_c_col: str,
    move_c_col: str,
    production_col: str,
    margin_col: str,
    threshold: float,
):
    """Build a metric_fn recomputing correctness fresh from FIXED picks each
    call, so the same function drives both the real bootstrap (on the real
    ``margin_col``) and the permutation null (on a shuffled copy of it).
    """

    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        margin = rows[margin_col]
        prod_correct = pick_correct(rows[production_col].astype(bool), margin)
        a_correct = pick_correct(rows[pick_a_col].astype(bool), margin)
        c_correct = pick_correct(rows[pick_c_col].astype(bool), margin)
        delta_a = a_correct - prod_correct
        delta_c = c_correct - prod_correct

        a_move_abs = rows[move_a_col].abs()
        c_move_abs = rows[move_c_col].abs()
        a_elig = a_move_abs.ge(threshold) & delta_a.notna()
        c_elig = c_move_abs.ge(threshold) & delta_c.notna()

        ppv_a = (
            float(delta_a.loc[a_elig].mean() / a_move_abs.loc[a_elig].mean())
            if int(a_elig.sum())
            else 0.0
        )
        ppv_c = (
            float(delta_c.loc[c_elig].mean() / c_move_abs.loc[c_elig].mean())
            if int(c_elig.sum())
            else 0.0
        )
        return {
            "per_point_value_wed": ppv_a,
            "per_point_value_sun_am": ppv_c,
            "per_point_value_diff": ppv_c - ppv_a,
        }

    return _metric


def score_per_point_diff(
    frame: pd.DataFrame,
    *,
    pick_a_col: str,
    move_a_col: str,
    pick_c_col: str,
    move_c_col: str,
    production_col: str,
    margin_col: str,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metric = per_point_value_diff_metric(
        pick_a_col=pick_a_col,
        move_a_col=move_a_col,
        pick_c_col=pick_c_col,
        move_c_col=move_c_col,
        production_col=production_col,
        margin_col=margin_col,
        threshold=threshold,
    )
    week_bs = week_blocked_bootstrap(frame, metric, block="week", samples=samples, seed=seed)
    season_bs = week_blocked_bootstrap(frame, metric, block="season", samples=samples, seed=seed)
    week_row = week_bs.loc[week_bs["metric"].eq("per_point_value_diff")].iloc[0]
    season_row = season_bs.loc[season_bs["metric"].eq("per_point_value_diff")].iloc[0]
    point = metric(frame)
    n_a_eligible = int(frame[move_a_col].abs().ge(threshold).sum())
    n_c_eligible = int(frame[move_c_col].abs().ge(threshold).sum())
    return {
        "cell": "movement_leads_sun_vs_wed_per_point",
        "n_population": len(frame),
        "n_wed_eligible": n_a_eligible,
        "n_sun_am_eligible": n_c_eligible,
        "per_point_value_wed": point["per_point_value_wed"],
        "per_point_value_sun_am": point["per_point_value_sun_am"],
        "per_point_value_diff_points": row_or_nan(week_row, "estimate"),
        "week_blocked_ci95_points": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "season_blocked_ci95_points": [float(season_row["lower"]), float(season_row["upper"])],
        "season_blocked_probability_positive": float(season_row["probability_positive"]),
    }


def generic_null_distribution(
    frame: pd.DataFrame,
    *,
    metric_fn: Any,
    metric_name: str,
    margin_col: str,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """200-draw within-week permutation null for an arbitrary metric_fn.

    Generalizes ``movement_expansion_battery.null_distribution`` (which is
    hard-wired to a single candidate-vs-production accuracy delta) to any
    metric that recomputes correctness fresh from FIXED picks and a margin
    column -- exactly ``per_point_value_diff_metric`` above.
    """

    working = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = week_positions(working)
    values: list[float] = []
    for _ in range(permutations):
        trial = working.copy()
        trial[margin_col] = permuted_margins(working, margin_col, rng, groups)
        values.append(metric_fn(trial)[metric_name])
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    observed = float(metric_fn(working)[metric_name])
    return {
        "permutations": len(finite),
        "null_mean": float(finite.mean()) if len(finite) else float("nan"),
        "null_q025": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
        "null_q975": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        "observed": observed,
        "fraction_of_null_below_observed": float((finite < observed).mean())
        if len(finite)
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--game-features", type=Path, default=DEFAULT_GAME_FEATURES)
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_ROOT / "rotation_registry.json"
    )
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    family = registry.families[ROTATION_FAMILY]
    window = family.assigned_window
    if window is None:
        raise SystemExit(f"{ROTATION_FAMILY} has no assigned window; run `rotation assign` first")
    window_seasons = (window.seasons[0], window.seasons[1])
    print(f"family={ROTATION_FAMILY} assigned window seasons={window_seasons}")

    started = time.time()
    archive = pd.read_parquet(args.archive)

    features = pd.read_parquet(args.game_features)
    schedule = features.loc[features["game_type"].eq("REG"), ["game_id", "season", "week"]].dropna()
    schedule = schedule.assign(
        season=schedule["season"].astype(int), week=schedule["week"].astype(int)
    )

    # ================================================================
    # Day-part population (2023-2025) -- cells 1, 2, 3, 4
    # ================================================================
    day_part_base = archive.loc[archive["season"].isin(INTRADAY_SEASONS)].reset_index(drop=True)
    print(f"\nday-part base population (2023-2025): {len(day_part_base)} games")

    corrected, kickoff = _load_intraday_with_kickoff(args.market_root, schedule)
    wed_spread, n_missing_wed = _home_spread_at_weekday_cutoff(
        corrected,
        kickoff,
        day_offset_from_sunday=WEDNESDAY_DAY_OFFSET,
        hour=WEDNESDAY_HOUR_ET,
        column_name="wed_noon_home_spread",
    )
    sun_spread, n_missing_sun = _home_spread_at_weekday_cutoff(
        corrected,
        kickoff,
        day_offset_from_sunday=SUNDAY_DAY_OFFSET,
        hour=SUNDAY_HOUR_ET,
        column_name="sun_am_home_spread",
    )
    print(f"wed_noon missing a qualifying capture: {n_missing_wed}")
    print(f"sun_am (true deadline) missing a qualifying capture: {n_missing_sun}")

    sat_pairing = build_pairing_table(
        args.market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("sat_midday",),
        seasons=list(INTRADAY_SEASONS),
        schedule=schedule,
    )
    sat_spread = sat_pairing[["game_id", "home_spread"]].rename(
        columns={"home_spread": "sat_midday_home_spread"}
    )

    merged = day_part_base.merge(wed_spread, on="game_id", how="left")
    merged = merged.merge(sat_spread, on="game_id", how="left")
    merged = merged.merge(sun_spread, on="game_id", how="left")

    production_home = merged[PRODUCTION_PICK_COL].astype(bool)
    merged["_move_a"] = merged["wed_noon_home_spread"] - merged["tue_open_home_spread"]
    merged["_pick_a"] = threshold_pick(
        merged["wed_noon_home_spread"], merged["tue_open_home_spread"], production_home, THRESHOLD
    )

    pop_a = merged.loc[merged["wed_noon_home_spread"].notna()].reset_index(drop=True)

    pop_b = merged.loc[
        merged[["wed_noon_home_spread", "sat_midday_home_spread"]].notna().all(axis=1)
    ].reset_index(drop=True)
    pop_b["_move_b"] = pop_b["sat_midday_home_spread"] - pop_b["wed_noon_home_spread"]
    pop_b["_pick_b"] = threshold_pick(
        pop_b["sat_midday_home_spread"],
        pop_b["wed_noon_home_spread"],
        pop_b[PRODUCTION_PICK_COL].astype(bool),
        THRESHOLD,
    )

    pop_c = merged.loc[
        merged[["sat_midday_home_spread", "sun_am_home_spread"]].notna().all(axis=1)
    ].reset_index(drop=True)
    pop_c["_move_c"] = pop_c["sun_am_home_spread"] - pop_c["sat_midday_home_spread"]
    pop_c["_pick_c"] = threshold_pick(
        pop_c["sun_am_home_spread"],
        pop_c["sat_midday_home_spread"],
        pop_c[PRODUCTION_PICK_COL].astype(bool),
        THRESHOLD,
    )

    joint = merged.loc[
        merged[["wed_noon_home_spread", "sat_midday_home_spread", "sun_am_home_spread"]]
        .notna()
        .all(axis=1)
    ].reset_index(drop=True)
    joint["_move_a"] = joint["wed_noon_home_spread"] - joint["tue_open_home_spread"]
    joint["_move_c"] = joint["sun_am_home_spread"] - joint["sat_midday_home_spread"]
    joint["_pick_a"] = threshold_pick(
        joint["wed_noon_home_spread"],
        joint["tue_open_home_spread"],
        joint[PRODUCTION_PICK_COL].astype(bool),
        THRESHOLD,
    )
    joint["_pick_c"] = threshold_pick(
        joint["sun_am_home_spread"],
        joint["sat_midday_home_spread"],
        joint[PRODUCTION_PICK_COL].astype(bool),
        THRESHOLD,
    )

    print(f"\npopulation (a) Tue->Wed: {len(pop_a)}")
    print(f"population (b) Wed->Sat: {len(pop_b)}")
    print(f"population (c) Sat->Sun-AM: {len(pop_c)}")
    print(f"joint (a)+(c) population: {len(joint)}")

    cell_a = score_cell(
        pop_a,
        name="movement_leads_wed_follow_1_0",
        candidate_col="_pick_a",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    cell_b = score_cell(
        pop_b,
        name="movement_leads_sat_follow_1_0",
        candidate_col="_pick_b",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    cell_c = score_cell(
        pop_c,
        name="movement_leads_sun_am_follow_1_0",
        candidate_col="_pick_c",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    cell_4 = score_per_point_diff(
        joint,
        pick_a_col="_pick_a",
        move_a_col="_move_a",
        pick_c_col="_pick_c",
        move_c_col="_move_c",
        production_col=PRODUCTION_PICK_COL,
        margin_col="margin_vs_open",
        threshold=THRESHOLD,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    cell_4_metric = per_point_value_diff_metric(
        pick_a_col="_pick_a",
        move_a_col="_move_a",
        pick_c_col="_pick_c",
        move_c_col="_move_c",
        production_col=PRODUCTION_PICK_COL,
        margin_col="margin_vs_open",
        threshold=THRESHOLD,
    )
    cell_4_null = generic_null_distribution(
        joint,
        metric_fn=cell_4_metric,
        metric_name="per_point_value_diff",
        margin_col="margin_vs_open",
        permutations=NULL_PERMUTATIONS,
        seed=BOOTSTRAP_SEED,
    )
    cell_4["permutation_null"] = cell_4_null

    for cell in (cell_a, cell_b, cell_c):
        print(
            f"{cell['cell']}: n={cell['n_graded']} delta={cell['paired_delta_points']:+.4f}pts "
            f"weekCI={cell['week_blocked_ci95_points']} "
            f"P+={cell['week_blocked_probability_positive']:.4f} "
            f"seasonP+={cell['season_blocked_probability_positive']:.4f}"
        )
    print(
        f"{cell_4['cell']}: n={cell_4['n_population']} "
        f"ppv_wed={cell_4['per_point_value_wed']:+.4f} "
        f"ppv_sun_am={cell_4['per_point_value_sun_am']:+.4f} "
        f"diff={cell_4['per_point_value_diff_points']:+.4f} "
        f"weekCI={cell_4['week_blocked_ci95_points']} "
        f"P+={cell_4['week_blocked_probability_positive']:.4f}"
    )

    # ================================================================
    # LEAD-06: rising-total, stable-spread dog -- assigned window population
    # ================================================================
    window_population = archive.loc[archive["season"].between(*window_seasons)].reset_index(
        drop=True
    )
    print(f"\nLEAD-06 window population {window_seasons}: {len(window_population)} games")
    latest = load_latest_pre_deadline(args.market_root, args.game_features, window_seasons)
    wmerged = window_population.merge(latest, on="game_id", how="left")
    wproduction_home = wmerged[PRODUCTION_PICK_COL].astype(bool)
    pick5, flagged5 = rising_total_dog_pick(
        wmerged["tue_open_total_line"],
        wmerged["latest_pre_deadline_total_line"],
        wmerged["tue_open_home_spread"],
        wmerged["latest_pre_deadline_home_spread"],
        wproduction_home,
    )
    wmerged["_pick_rising_dog"] = pick5
    wmerged["_flagged_rising_dog"] = flagged5
    cell_5 = score_cell(
        wmerged,
        name="movement_leads_rising_total_dog",
        candidate_col="_pick_rising_dog",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    flagged_by_season = (
        wmerged.loc[wmerged["_flagged_rising_dog"], "season"].value_counts().sort_index()
    )
    cell_5["n_flagged"] = int(wmerged["_flagged_rising_dog"].sum())
    cell_5["flagged_by_season"] = {str(k): int(v) for k, v in flagged_by_season.items()}
    print(
        f"{cell_5['cell']}: n={cell_5['n_graded']} flagged={cell_5['n_flagged']} "
        f"by_season={cell_5['flagged_by_season']} "
        f"delta={cell_5['paired_delta_points']:+.4f}pts "
        f"weekCI={cell_5['week_blocked_ci95_points']} "
        f"P+={cell_5['week_blocked_probability_positive']:.4f}"
    )

    # ================================================================
    # Positive controls (perfect-foresight, NOT recorded to the signal
    # registry -- instrument sensitivity diagnostics only)
    # ================================================================
    joint_control = joint.copy()
    joint_control["_pick_perfect_foresight"] = joint_control["margin_vs_open"].gt(0.0)
    control_dayparts = score_cell(
        joint_control,
        name="movement_leads_positive_control_perfect_foresight_dayparts",
        candidate_col="_pick_perfect_foresight",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    window_control = wmerged.copy()
    window_control["_pick_perfect_foresight"] = window_control["margin_vs_open"].gt(0.0)
    control_window = score_cell(
        window_control,
        name="movement_leads_positive_control_perfect_foresight_window",
        candidate_col="_pick_perfect_foresight",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    print(
        f"\npositive control (day-parts population, n={control_dayparts['n_graded']}): "
        f"delta={control_dayparts['paired_delta_points']:+.4f}pts "
        f"P+={control_dayparts['week_blocked_probability_positive']:.4f}"
    )
    print(
        f"positive control (window population, n={control_window['n_graded']}): "
        f"delta={control_window['paired_delta_points']:+.4f}pts "
        f"P+={control_window['week_blocked_probability_positive']:.4f}"
    )

    # ================================================================
    # Write artifacts
    # ================================================================
    run_dir_name = args.out or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = DEFAULT_OUTPUT_ROOT / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(output_dir / "per_game_dayparts.parquet", index=False)
    joint.to_parquet(output_dir / "per_game_dayparts_joint.parquet", index=False)
    wmerged.to_parquet(output_dir / "per_game_rising_total.parquet", index=False)

    cells_frame = pd.json_normalize([cell_a, cell_b, cell_c, cell_4, cell_5])
    cells_frame.to_csv(output_dir / "cells_summary.csv", index=False)

    configuration = {
        "rotation_family": ROTATION_FAMILY,
        "window_seasons": list(window_seasons),
        "day_part_seasons": list(INTRADAY_SEASONS),
        "archive": str(args.archive),
        "market_root": str(args.market_root),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "null_permutations": NULL_PERMUTATIONS,
        "threshold": THRESHOLD,
        "rising_total_threshold": RISING_TOTAL_THRESHOLD,
        "stable_spread_threshold": STABLE_SPREAD_THRESHOLD,
        "cells": [
            cell_a["cell"],
            cell_b["cell"],
            cell_c["cell"],
            cell_4["cell"],
            cell_5["cell"],
        ],
        "primary_cell": "movement_leads_sun_vs_wed_per_point",
    }
    metadata: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **configuration,
        "n_missing_wed_noon": int(n_missing_wed),
        "n_missing_sun_am": int(n_missing_sun),
        "cell_movement_leads_wed_follow_1_0": cell_a,
        "cell_movement_leads_sat_follow_1_0": cell_b,
        "cell_movement_leads_sun_am_follow_1_0": cell_c,
        "cell_movement_leads_sun_vs_wed_per_point": cell_4,
        "cell_movement_leads_rising_total_dog": cell_5,
        "positive_control_dayparts": control_dayparts,
        "positive_control_window": control_window,
        "elapsed_seconds": time.time() - started,
        "provenance": artifact_provenance(configuration, args.archive, project_root=REPO_ROOT),
    }
    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="movement-leads-battery",
        metrics={
            "cell_movement_leads_wed_follow_1_0": cell_a,
            "cell_movement_leads_sat_follow_1_0": cell_b,
            "cell_movement_leads_sun_am_follow_1_0": cell_c,
            "cell_movement_leads_sun_vs_wed_per_point": cell_4,
            "cell_movement_leads_rising_total_dog": cell_5,
            "positive_control_dayparts": control_dayparts,
            "positive_control_window": control_window,
        },
        notes="Predeclared movement leads battery (LEAD-01/06/07); docs/movement_leads_battery.md",
        source="docs/movement_leads_battery.md",
        rotation_family=ROTATION_FAMILY,
        project_root=REPO_ROOT,
        registry_root=DEFAULT_REGISTRY_ROOT,
    )
    print(f"\nwrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
