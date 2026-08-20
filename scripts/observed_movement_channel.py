"""Observed line-movement channel: predeclared measurement (measure-only).

Predeclaration: `docs/observed_movement_channel.md` (frozen before any
accuracy sign below was computed). That document also reconciles this
experiment against two already-recorded families
(`movement_direction_tilt_*`, which uses MKT-06's PREDICTED movement, and
`odds_microstructure_H3_*`, which already measures OBSERVED movement in a
different, always-flip / 50%-baseline design) -- read it first.

## What this measures

The owner corrected (2026-08-20) a standing error: the pool's PICKS are
editable until each game's own kickoff; only the GRADING LINE freezes
Tuesday noon. That means a pick can legitimately be refreshed later in the
week using market information that has, by then, already been observed
(no forecasting, no leakage -- every input here is a realized quote), and
still graded against the frozen Tuesday line. This script measures what
that refresh channel is worth, reusing the exact frozen production recipe
`docs/opener_evaluation.md` runs (`nfl_ats.clv.opener_pick_evaluation`,
`weak_stack`/ridge/alpha-10, weekly refits on strictly-earlier games) so
"production pick" here means the identical pick production actually plays
(`pick_home_at_open_probability_rule`), not a re-derived stand-in.

**Amendment 2026-08-20 (owner refinement, applied before any run's numbers
were used):** picks cannot change after Sunday 16:00 ET -- the true
per-game decision deadline is `min(kickoff, Sunday 16:00 ET)`, not a
13:00 ET guess. Arm 3 below now reports BOTH: `sunday_1600` (the true
deadline -- the realizable, primary read) and `sunday_1300` (a
deliberately conservative comparison point, kept because it was already
built and is strictly inside the true deadline for every game). For
SNF/MNF (and any late-Sunday-afternoon kickoff), Arm 1's close-based
oracle prints AFTER the true pick deadline and so overstates what is
reachable on those games specifically; Arm 1's cell discloses the
deadline-bound game count measured on the 2023-2025 subsample, and the
`sunday_1600` oracle cell in Arm 3 is the deadline-respecting variant for
those same games. See `docs/observed_movement_channel.md`'s dated
amendment for the full text.

Three arms, each reported regardless of sign (AGENTS.md):

1. **Oracle (full-slate)**: pick the side the CLOSE favors relative to the
   Tuesday opener; when the line did not move, keep the production pick
   (no games dropped, unlike the H3 family). Graded at the Tuesday line.
   Full-slate accuracy plus the PAIRED delta vs production on the same
   games. Caveat (see amendment above): for SNF/MNF/late-Sunday games the
   close is observed after the true Sunday 16:00 ET pick deadline.
2. **Threshold overlays**: flip the production pick to the movement side
   only when `|close - tue_open| >= 0.5` and `>= 1.0` (two cells). Reports
   flip-eligible count, flip-changed count, and the model-agrees-already
   fraction (of flip-eligible games, how many the production pick already
   had right without any overlay).
3. **Sunday realism** (2023-2025 `intraday_hourly` archive only): arms
   1+2 repeated with the close replaced by the last intraday capture at or
   before a per-game cutoff, for two cutoff variants -- `min(kickoff,
   Sunday 16:00 ET)` (the TRUE decision deadline, primary/realizable) and
   `min(kickoff, Sunday 13:00 ET)` (conservative comparison). For
   SNF/MNF, the 16:00 cutoff correctly uses Sunday-afternoon lines, NOT
   their own close. Games with no qualifying capture before a cutoff are
   dropped from that cutoff's cells only (count reported per cutoff).

Bootstrap: `nfl_ats.clv.week_blocked_bootstrap`, `samples=20_000`,
`seed=20260819` (this script's own seed), `block="week"` primary and
`block="season"` secondary. Per AGENTS.md, every cell is reported
regardless of whether its interval contains zero; `probability_positive`
is the reported continuous read, never a binary "contains zero" verdict.

Writes `artifacts/observed_movement_channel/<run_id>/` (per-game parquet
+ `cells_summary.csv` + `metadata.json` via `write_experiment_artifact`,
which also stamps `registry/experiments/`). Proposes (never executes)
`nfl-ats weak-signals record` commands, printed and written to
`metadata.json` -- effect/interval values are pre-scaled to
`accuracy_points` (fraction * 100) to avoid the x100 unit bug three prior
scripts in this repo were caught making (see
`odds_microstructure_H3_*` entries' notes in `registry/weak_signals.json`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.clv import (
    HISTORICAL_CAPTURE_KIND,
    decision_market_consensus,
    load_decision_quotes,
    opener_pick_evaluation,
    pick_correct,
    resolve_active_model_config,
    week_blocked_bootstrap,
)
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data/market/raw"
DEFAULT_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
DEFAULT_ARTIFACTS_ROOT = REPO / "artifacts"
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/observed_movement_channel"
DEFAULT_REGISTRY_ROOT = REPO / "registry"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819

INTRADAY_LABEL = "intraday_hourly"
INTRADAY_SEASONS: tuple[int, ...] = (2023, 2024, 2025)
EASTERN = ZoneInfo("America/New_York")
THRESHOLDS: tuple[float, ...] = (0.5, 1.0)

# Amendment 2026-08-20 (owner refinement): the true per-game pick deadline
# is min(kickoff, Sunday 16:00 ET) -- picks cannot change after that. 16 is
# the realizable/primary cutoff; 13 is kept as a deliberately conservative
# comparison point (strictly inside the true deadline for every game).
SUNDAY_CUTOFFS: tuple[tuple[int, str, bool], ...] = (
    (16, "sunday_1600_realism", True),
    (13, "sunday_1300_conservative", False),
)


# ---------------------------------------------------------------------------
# Pick-rule helpers
# ---------------------------------------------------------------------------


def _oracle_pick(open_move: pd.Series, production_pick: pd.Series) -> pd.Series:
    """Movement side when the line moved; production pick on an exact tie."""

    home = np.where(
        open_move.gt(0.0), True, np.where(open_move.lt(0.0), False, production_pick.astype(bool))
    )
    return pd.Series(home, index=open_move.index)


def _threshold_pick(
    open_move: pd.Series, production_pick: pd.Series, threshold: float
) -> tuple[pd.Series, pd.Series]:
    """Flip to the movement side only when |move| clears ``threshold``."""

    eligible = open_move.abs().ge(threshold)
    movement_home = open_move.gt(0.0)
    pick = np.where(eligible, movement_home, production_pick.astype(bool))
    return pd.Series(pick, index=open_move.index), eligible


def _paired_metric_fn(candidate_col: str, production_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        cand = pd.to_numeric(rows[candidate_col], errors="coerce")
        prod = pd.to_numeric(rows[production_col], errors="coerce")
        both = rows.loc[cand.notna() & prod.notna()]
        cand_v = pd.to_numeric(both[candidate_col], errors="coerce")
        prod_v = pd.to_numeric(both[production_col], errors="coerce")
        if len(cand_v) == 0:
            return {
                "candidate_accuracy": float("nan"),
                "production_accuracy": float("nan"),
                "paired_delta": float("nan"),
            }
        return {
            "candidate_accuracy": float(cand_v.mean()),
            "production_accuracy": float(prod_v.mean()),
            "paired_delta": float((cand_v - prod_v).mean()),
        }

    return _metric


def _both_blocks(frame: pd.DataFrame, metric_fn) -> dict[str, pd.DataFrame]:
    return {
        "week": week_blocked_bootstrap(
            frame, metric_fn, block="week", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        ),
        "season": week_blocked_bootstrap(
            frame, metric_fn, block="season", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        ),
    }


def _extract(ci: dict[str, pd.DataFrame], metric: str) -> dict[str, float]:
    week_row = ci["week"].loc[ci["week"]["metric"].eq(metric)].iloc[0]
    season_row = ci["season"].loc[ci["season"]["metric"].eq(metric)].iloc[0]
    return {
        "estimate": float(week_row["estimate"]),
        "week_lower": float(week_row["lower"]),
        "week_upper": float(week_row["upper"]),
        "week_probability_positive": float(week_row["probability_positive"]),
        "season_lower": float(season_row["lower"]),
        "season_upper": float(season_row["upper"]),
        "season_probability_positive": float(season_row["probability_positive"]),
    }


def _score_cell(
    frame: pd.DataFrame,
    *,
    name: str,
    candidate_pick_col: str,
    margin_col: str,
    production_correct_col: str,
    eligible: pd.Series | None,
    production_pick_col: str,
) -> dict[str, Any]:
    working = frame.copy()
    candidate_correct_col = f"_candidate_correct_{name}"
    working[candidate_correct_col] = pick_correct(working[candidate_pick_col], working[margin_col])
    ci = _both_blocks(working, _paired_metric_fn(candidate_correct_col, production_correct_col))
    candidate_row = _extract(ci, "candidate_accuracy")
    production_row = _extract(ci, "production_accuracy")
    paired_row = _extract(ci, "paired_delta")

    both = working.dropna(subset=[candidate_correct_col, production_correct_col])
    cell: dict[str, Any] = {
        "cell": name,
        "n_graded": len(both),
        "candidate_point_accuracy": float(both[candidate_correct_col].mean()),
        "production_point_accuracy": float(both[production_correct_col].mean()),
        "candidate_week_ci": [candidate_row["week_lower"], candidate_row["week_upper"]],
        "candidate_season_ci": [candidate_row["season_lower"], candidate_row["season_upper"]],
        "production_week_ci": [production_row["week_lower"], production_row["week_upper"]],
        "paired_delta_point": float(paired_row["estimate"]),
        "paired_delta_week_ci": [paired_row["week_lower"], paired_row["week_upper"]],
        "paired_delta_week_probability_positive": paired_row["week_probability_positive"],
        "paired_delta_season_ci": [paired_row["season_lower"], paired_row["season_upper"]],
        "paired_delta_season_probability_positive": paired_row["season_probability_positive"],
    }
    if eligible is not None:
        elig = working.loc[eligible.reindex(working.index).fillna(False)]
        n_eligible = len(elig)
        if n_eligible:
            agrees = elig[candidate_pick_col].astype(bool) == elig[production_pick_col].astype(bool)
            n_changed = int((~agrees).sum())
            cell["n_flip_eligible"] = n_eligible
            cell["n_flip_changed"] = n_changed
            cell["model_agrees_already_fraction"] = float(agrees.mean())
        else:
            cell["n_flip_eligible"] = 0
            cell["n_flip_changed"] = 0
            cell["model_agrees_already_fraction"] = float("nan")
    return cell


# ---------------------------------------------------------------------------
# Sunday-morning realism: per-game cutoff + intraday consensus
# ---------------------------------------------------------------------------


def _true_week_correct(quotes: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Restrict quotes to each game's own scheduled (season, week).

    Duplicated from `scripts/odds_microstructure_battery.py`'s private
    helper of the same name/logic rather than imported (that script's
    version is module-private) -- corrects for a book posting a game's
    board a week or more ahead of its own scheduled week ("early
    sighting").
    """

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


def _sunday_cutoff_et_utc(commence_utc: pd.Series, hour: int) -> pd.Series:
    """Per-row: the America/New_York Sunday `hour`:00 on/after that row's own week.

    Derived from each row's OWN earliest-in-week kickoff via the caller's
    groupby, not from the `gameday`/`gametime` feature columns, to keep a
    single timestamp convention (`commence_time_utc`, already used
    throughout `nfl_ats.clv` for pregame filtering).
    """

    local = commence_utc.dt.tz_convert(EASTERN)
    # All arithmetic done in naive wall-clock time, then localized exactly
    # once -- adding a fixed Timedelta to an already tz-aware timestamp
    # would add absolute duration, not wall-clock hours, and silently drift
    # by an hour across a DST boundary.
    naive_date = local.dt.tz_localize(None).dt.normalize()
    days_ahead = (6 - local.dt.weekday) % 7
    sunday_naive_date = naive_date + pd.to_timedelta(days_ahead, unit="D")
    sunday_hour_naive = sunday_naive_date + pd.Timedelta(hours=hour)
    return sunday_hour_naive.dt.tz_localize(EASTERN).dt.tz_convert("UTC")


def _load_intraday_with_kickoff(
    root: Path, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 2023-2025 `intraday_hourly` archive once, plus a per-game kickoff table.

    Loading and true-week-correcting the archive is the expensive step;
    every cutoff-hour variant (16:00 realizable, 13:00 conservative) reuses
    this single load rather than re-reading ~7,000 manifests per variant.
    """

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


def _spread_at_cutoff(
    corrected: pd.DataFrame, kickoff: pd.DataFrame, *, hour: int
) -> tuple[pd.DataFrame, int, int]:
    """Per-game HOME spread from the last intraday capture before `min(kickoff, Sunday hour:00 ET)`.

    Returns (frame[game_id, sunday_home_spread, ...],
    n_games_with_zero_qualifying_capture, n_games_deadline_bound) where
    "deadline bound" means the cutoff was set by the Sunday clock, not by
    the game's own (earlier) kickoff -- i.e. SNF/MNF/late-Sunday-afternoon
    games whose own close/kickoff falls after this cutoff.
    """

    if corrected.empty or kickoff.empty:
        return pd.DataFrame(columns=["game_id", "sunday_home_spread"]), 0, 0

    working_kickoff = kickoff.copy()
    working_kickoff["sunday_cutoff_et_utc"] = _sunday_cutoff_et_utc(
        working_kickoff["week_first_commence_utc"], hour
    )
    working_kickoff["cutoff_utc"] = working_kickoff[
        ["sunday_cutoff_et_utc", "commence_time_utc"]
    ].min(axis=1)
    working_kickoff["deadline_bound"] = working_kickoff["sunday_cutoff_et_utc"].lt(
        working_kickoff["commence_time_utc"]
    )
    n_deadline_bound = int(working_kickoff["deadline_bound"].sum())

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
        return (
            pd.DataFrame(columns=["game_id", "sunday_home_spread"]),
            n_games_total,
            n_deadline_bound,
        )

    consensus = decision_market_consensus(eligible_quotes.drop(columns=["cutoff_utc"]))
    spread = consensus.loc[
        consensus["market"].eq("spreads") & consensus["outcome_side"].eq("HOME")
    ][["nflverse_game_id", "consensus_line", "books", "snapshot_timestamp_utc"]].rename(
        columns={
            "nflverse_game_id": "game_id",
            "consensus_line": "sunday_home_spread",
            "books": "sunday_books",
            "snapshot_timestamp_utc": "sunday_snapshot_utc",
        }
    )
    n_missing = n_games_total - spread["game_id"].nunique()
    return spread, int(n_missing), n_deadline_bound


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    args = parser.parse_args()
    started = time.time()

    print(f"Loading features: {args.features}")
    features_raw = pd.read_parquet(args.features)
    features = regular_season_rows(features_raw)

    active_model_config = resolve_active_model_config(args.artifacts_root)
    print(f"Active model config (production pick source): {active_model_config}")

    print("Running opener_pick_evaluation (frozen production recipe, docs/opener_evaluation.md)")
    scored = opener_pick_evaluation(args.root, features, active_model_config=active_model_config)
    season_min, season_max = scored["season"].min(), scored["season"].max()
    print(f"Base paired population: {len(scored)} games, seasons {season_min}-{season_max}")

    production_pick_col = "pick_home_at_open_probability_rule"
    production_correct_col = "correct_at_open_probability_rule"
    margin_col = "margin_vs_open"

    n_pushes = int(scored[production_correct_col].isna().sum())
    print(f"Pushes at the Tuesday grading line (excluded from accuracy): {n_pushes}")

    cells: list[dict[str, Any]] = []

    # Load the intraday archive + per-game kickoff table once, shared by
    # Arm 1's deadline-bound disclosure and every Arm 3 cutoff variant.
    print("\nLoading intraday_hourly archive (2023-2025) + per-game kickoff table")
    intraday_quotes, kickoff_table = _load_intraday_with_kickoff(args.root, features)
    n_deadline_bound_1600 = 0
    if not kickoff_table.empty:
        _diag_kickoff = kickoff_table.copy()
        _diag_kickoff["sunday_1600_et_utc"] = _sunday_cutoff_et_utc(
            _diag_kickoff["week_first_commence_utc"], 16
        )
        n_deadline_bound_1600 = int(
            _diag_kickoff["sunday_1600_et_utc"].lt(_diag_kickoff["commence_time_utc"]).sum()
        )
        n_games_2023_2025 = int(_diag_kickoff["nflverse_game_id"].nunique())
    else:
        n_games_2023_2025 = 0
    print(
        f"2023-2025 games whose own kickoff is AFTER Sunday 16:00 ET (deadline-bound; "
        f"SNF/MNF/late-Sunday-afternoon): {n_deadline_bound_1600} of {n_games_2023_2025}"
    )

    # ===================================================================
    # Arm 1 -- Oracle, full-slate (Tuesday -> Close)
    # ===================================================================
    print("\n=== Arm 1: Oracle (Tuesday -> Close), full-slate ===")
    scored["oracle_pick_home"] = _oracle_pick(scored["open_move"], scored[production_pick_col])
    n_zero_move = int(scored["open_move"].eq(0.0).sum())
    print(f"Zero-movement games (production pick kept): {n_zero_move} of {len(scored)}")
    agrees_nonzero = scored.loc[scored["open_move"].ne(0.0)]
    if len(agrees_nonzero):
        movement_home = agrees_nonzero["open_move"].gt(0.0)
        agreement_all = float(
            (movement_home == agrees_nonzero[production_pick_col].astype(bool)).mean()
        )
    else:
        agreement_all = float("nan")
    cell1 = _score_cell(
        scored,
        name="oracle_full_slate",
        candidate_pick_col="oracle_pick_home",
        margin_col=margin_col,
        production_correct_col=production_correct_col,
        eligible=scored["open_move"].ne(0.0),
        production_pick_col=production_pick_col,
    )
    cell1["n_zero_movement_games"] = n_zero_move
    cell1["model_agrees_already_fraction_all_nonzero_movement"] = agreement_all
    deadline_bound_pct = (
        (n_deadline_bound_1600 / n_games_2023_2025 * 100.0) if n_games_2023_2025 else float("nan")
    )
    cell1["deadline_caveat"] = (
        "This arm grades the close, which for SNF/MNF/late-Sunday-afternoon games prints AFTER "
        "the true Sunday 16:00 ET pick deadline (owner refinement, 2026-08-20); those games' "
        "oracle picks are not actually reachable. Measured on the 2023-2025 intraday_hourly "
        f"subsample: {n_deadline_bound_1600} of {n_games_2023_2025} games "
        f"({deadline_bound_pct:.1f}%) "
        "are deadline-bound at the 16:00 ET cutoff. The deadline-respecting variant for those "
        "same games is Arm 3's oracle_sunday_1600_realism cell."
    )
    cells.append({"arm": "1_oracle", **cell1})
    print(
        f"  candidate={cell1['candidate_point_accuracy']:.4f} "
        f"production={cell1['production_point_accuracy']:.4f} "
        f"paired_delta={cell1['paired_delta_point']:+.4f} "
        f"week_P+={cell1['paired_delta_week_probability_positive']:.3f}"
    )

    # ===================================================================
    # Arm 2 -- Threshold overlays (Tuesday -> Close)
    # ===================================================================
    for threshold in THRESHOLDS:
        label = f"threshold_{str(threshold).replace('.', '_')}"
        print(f"\n=== Arm 2: threshold overlay >= {threshold} (Tuesday -> Close) ===")
        pick_col = f"_pick_{label}"
        pick, eligible = _threshold_pick(
            scored["open_move"], scored[production_pick_col], threshold
        )
        scored[pick_col] = pick
        cell = _score_cell(
            scored,
            name=label,
            candidate_pick_col=pick_col,
            margin_col=margin_col,
            production_correct_col=production_correct_col,
            eligible=eligible,
            production_pick_col=production_pick_col,
        )
        cells.append({"arm": f"2_{label}", "threshold": threshold, **cell})
        print(
            f"  candidate={cell['candidate_point_accuracy']:.4f} "
            f"paired_delta={cell['paired_delta_point']:+.4f} "
            f"week_P+={cell['paired_delta_week_probability_positive']:.3f} "
            f"flip_eligible={cell['n_flip_eligible']} flip_changed={cell['n_flip_changed']} "
            f"agrees_already={cell['model_agrees_already_fraction']:.3f}"
        )

    # ===================================================================
    # Arm 3 -- Sunday realism (2023-2025 intraday_hourly only), two cutoffs:
    # 16:00 ET (true deadline, realizable/primary) and 13:00 ET
    # (conservative comparison, strictly inside the true deadline).
    # ===================================================================
    realism_frames: dict[str, pd.DataFrame] = {}
    realism_coverage: dict[str, dict[str, int]] = {}
    primary_cell_r1: dict[str, Any] | None = None
    for hour, cutoff_label, is_primary in SUNDAY_CUTOFFS:
        tag = "REALIZABLE, true deadline" if is_primary else "conservative comparison"
        print(f"\n=== Arm 3: Sunday cutoff {hour}:00 ET [{tag}] (2023-2025 intraday_hourly) ===")
        sunday_spread, n_missing, n_deadline_bound = _spread_at_cutoff(
            intraday_quotes, kickoff_table, hour=hour
        )
        print(
            f"Games missing any pre-cutoff intraday capture: {n_missing}; "
            f"deadline-bound (cutoff set by the {hour}:00 clock, not the game's own kickoff): "
            f"{n_deadline_bound}"
        )
        realism = (
            scored.loc[scored["season"].isin(INTRADAY_SEASONS)]
            .merge(sunday_spread, on="game_id", how="inner")
            .copy()
        )
        print(f"Population: {len(realism)} games (2023-2025 with a qualifying capture)")
        realism["sunday_open_move"] = (
            realism["sunday_home_spread"] - realism["tue_open_home_spread"]
        )
        realism_frames[cutoff_label] = realism
        realism_coverage[cutoff_label] = {
            "hour_et": hour,
            "n_missing_capture": n_missing,
            "n_deadline_bound": n_deadline_bound,
        }

        realism["oracle_pick_home_sunday"] = _oracle_pick(
            realism["sunday_open_move"], realism[production_pick_col]
        )
        n_zero_move_sunday = int(realism["sunday_open_move"].eq(0.0).sum())
        cell_r1 = _score_cell(
            realism,
            name=f"oracle_{cutoff_label}",
            candidate_pick_col="oracle_pick_home_sunday",
            margin_col=margin_col,
            production_correct_col=production_correct_col,
            eligible=realism["sunday_open_move"].ne(0.0),
            production_pick_col=production_pick_col,
        )
        cell_r1["n_zero_movement_games"] = n_zero_move_sunday
        cell_r1["n_games_missing_intraday_capture"] = n_missing
        cell_r1["n_games_deadline_bound"] = n_deadline_bound
        cell_r1["cutoff_hour_et"] = hour
        cell_r1["is_primary_realizable_cutoff"] = is_primary
        cells.append({"arm": f"3_oracle_{cutoff_label}", **cell_r1})
        if is_primary:
            primary_cell_r1 = cell_r1
        print(
            f"  candidate={cell_r1['candidate_point_accuracy']:.4f} "
            f"production={cell_r1['production_point_accuracy']:.4f} "
            f"paired_delta={cell_r1['paired_delta_point']:+.4f} "
            f"week_P+={cell_r1['paired_delta_week_probability_positive']:.3f}"
        )

        for threshold in THRESHOLDS:
            label = f"threshold_{str(threshold).replace('.', '_')}_{cutoff_label}"
            print(f"=== Arm 3: threshold overlay >= {threshold} ({cutoff_label}) ===")
            pick_col = f"_pick_{label}"
            pick, eligible = _threshold_pick(
                realism["sunday_open_move"], realism[production_pick_col], threshold
            )
            realism[pick_col] = pick
            cell = _score_cell(
                realism,
                name=label,
                candidate_pick_col=pick_col,
                margin_col=margin_col,
                production_correct_col=production_correct_col,
                eligible=eligible,
                production_pick_col=production_pick_col,
            )
            cell["n_games_missing_intraday_capture"] = n_missing
            cell["n_games_deadline_bound"] = n_deadline_bound
            cell["cutoff_hour_et"] = hour
            cell["is_primary_realizable_cutoff"] = is_primary
            cells.append({"arm": f"3_{label}", "threshold": threshold, **cell})
            print(
                f"  candidate={cell['candidate_point_accuracy']:.4f} "
                f"paired_delta={cell['paired_delta_point']:+.4f} "
                f"week_P+={cell['paired_delta_week_probability_positive']:.3f} "
                f"flip_eligible={cell['n_flip_eligible']} flip_changed={cell['n_flip_changed']} "
                f"agrees_already={cell['model_agrees_already_fraction']:.3f}"
            )

    assert primary_cell_r1 is not None, "16:00 ET cutoff must be present in SUNDAY_CUTOFFS"
    realism = realism_frames[
        "sunday_1600_realism"
    ]  # the primary/realizable population, for artifacts

    # ------------------------------------------------------------------
    # EV statement: what a Sunday refresh would have been worth, plainly.
    # Primary read is the TRUE deadline (16:00 ET); the 13:00 conservative
    # cutoff is reported alongside for comparison, not as the headline.
    # ------------------------------------------------------------------
    mean_abs_move_primary = (
        float(realism["sunday_open_move"].abs().mean()) if len(realism) else float("nan")
    )
    conservative_cell = next(c for c in cells if c["arm"] == "3_oracle_sunday_1300_conservative")
    ev_statement = {
        "arm": "3_oracle_sunday_1600_realism",
        "games": primary_cell_r1["n_graded"],
        "paired_delta_accuracy_points": primary_cell_r1["paired_delta_point"] * 100.0,
        "paired_delta_week_probability_positive": primary_cell_r1[
            "paired_delta_week_probability_positive"
        ],
        "mean_absolute_move_points": mean_abs_move_primary,
        "conservative_1300_comparison_accuracy_points": conservative_cell["paired_delta_point"]
        * 100.0,
        "plain_statement": (
            f"Over {primary_cell_r1['n_graded']} paired 2023-2025 games with a qualifying capture "
            f"before the TRUE per-game pick deadline (min(kickoff, Sunday 16:00 ET)), refreshing "
            f"the pick to the observed movement side at that deadline (falling back to the "
            f"production pick on a tie) would have changed full-slate accuracy by "
            f"{primary_cell_r1['paired_delta_point'] * 100.0:+.2f} accuracy points vs never "
            f"refreshing, week-blocked probability_positive="
            f"{primary_cell_r1['paired_delta_week_probability_positive']:.3f}. The more "
            f"conservative 13:00 ET cutoff reads "
            f"{conservative_cell['paired_delta_point'] * 100.0:+.2f} accuracy points on the same "
            "design."
        ),
    }
    print(f"\nEV statement: {ev_statement['plain_statement']}")

    # ------------------------------------------------------------------
    # Propose (never execute) weak-signals record commands, pre-scaled to
    # accuracy_points (fraction * 100) -- see module docstring on the x100
    # unit bug this deliberately avoids.
    # ------------------------------------------------------------------
    proposed_records: list[dict[str, Any]] = []
    for c in cells:
        name = f"observed_movement_{c['cell']}"
        p_plus = c["paired_delta_week_probability_positive"]
        season_start, season_end = (2023, 2025) if str(c["arm"]).startswith("3_") else (2020, 2025)
        effect_pts = c["paired_delta_point"] * 100.0
        low_pts = c["paired_delta_week_ci"][0] * 100.0
        high_pts = c["paired_delta_week_ci"][1] * 100.0
        season_pp = c["paired_delta_season_probability_positive"]
        notes = (
            f"observed line-movement channel (docs/observed_movement_channel.md), arm={c['arm']}; "
            f"paired delta vs the production pick (pick_home_at_open_probability_rule) on the same "
            f"games, graded at the frozen Tuesday line; season-blocked P+={season_pp:.4f}; "
            "not independently confirmed on a rotated window."
        )
        proposed_records.append(
            {
                "cell": c["cell"],
                "week_probability_positive": p_plus,
                "proposed_command": (
                    "nfl-ats weak-signals record "
                    f"--name {name} "
                    f'--description "observed movement channel, arm {c["arm"]}, cell {c["cell"]}" '
                    "--source artifacts/observed_movement_channel/<run_id>/cells_summary.csv "
                    f"--effect {effect_pts:.6f} --effect-units accuracy_points "
                    "--classification unresolved_below_power --league nfl "
                    f"--season-start {season_start} --season-end {season_end} "
                    f"--interval-low {low_pts:.6f} --interval-high {high_pts:.6f} "
                    f"--probability-positive {p_plus:.6f} --sample-games {c['n_graded']} "
                    f'--notes "{notes}"'
                ),
            }
        )

    print(f"\n{len(proposed_records)} cells; proposed weak-signals record commands:")
    for record in proposed_records:
        print(f"  [P+={record['week_probability_positive']:.3f}] {record['cell']}")
        print(f"    {record['proposed_command']}")

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------
    output_dir = args.output_root / run_id()
    atomic_parquet(scored, output_dir / "per_game_tue_close.parquet")
    for cutoff_label, frame in realism_frames.items():
        atomic_parquet(frame, output_dir / f"per_game_{cutoff_label}.parquet")
    cells_frame = pd.json_normalize(cells)
    atomic_csv(cells_frame, output_dir / "cells_summary.csv")

    configuration = {
        "command": "observed-movement-channel",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "thresholds": list(THRESHOLDS),
        "sunday_cutoffs_hour_et": {label: hour for hour, label, _ in SUNDAY_CUTOFFS},
        "predeclaration": "docs/observed_movement_channel.md",
    }
    metadata: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **configuration,
        "active_model_config": active_model_config,
        "base_population_games": len(scored),
        "base_population_pushes": n_pushes,
        "deadline_bound_games_2023_2025_at_1600_et": n_deadline_bound_1600,
        "n_games_2023_2025_intraday_archive": n_games_2023_2025,
        "sunday_realism_coverage": realism_coverage,
        "cells": cells,
        "ev_statement": ev_statement,
        "proposed_weak_signal_records": proposed_records,
        "reconciliation": (
            "Tue-to-Wed oracle previously measured at +4.47 accuracy points "
            "(odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025, "
            "artifacts/odds_microstructure/20260818T225430Z/cells_summary.csv); "
            "movement_direction_tilt_* (2026-08-19) uses PREDICTED, not observed, "
            "movement -- no overlap. Full reconciliation in "
            "docs/observed_movement_channel.md."
        ),
        "deadline_amendment": (
            "Owner refinement 2026-08-20: the true per-game pick deadline is "
            "min(kickoff, Sunday 16:00 ET), not a 13:00 ET guess. sunday_1600_realism is the "
            "realizable/primary Arm 3 cutoff; sunday_1300_conservative is kept as a strictly-"
            "inside-the-deadline comparison point. Arm 1's close-based oracle is disclosed as "
            "overstating reachability for deadline-bound (SNF/MNF/late-Sunday) games."
        ),
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="observed-movement-channel",
        metrics=metadata,
        registry_root=args.registry_root,
        notes=(
            "Observed line-movement channel (owner correction 2026-08-20: picks editable "
            "till kickoff, only the grading line freezes Tuesday). 3 arms: oracle full-slate, "
            "threshold overlays (>=0.5, >=1.0), Sunday-morning realism (2023-2025 hourly "
            "archive). Every cell predeclared to record unresolved_below_power via a separate "
            "nfl-ats weak-signals record call regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nelapsed: {time.time() - started:.1f}s")
    print(f"artifacts: {output_dir}")


if __name__ == "__main__":
    main()
