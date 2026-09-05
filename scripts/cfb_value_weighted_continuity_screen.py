"""Value-weighted CFB role-continuity screen (standalone, src/ untouched).

Executes the predeclaration drafted in
``C:\\Users\\Ryan\\AppData\\Local\\Temp\\claude\\F--Repos-nfl-py3\\c8c5fbdd-027f-438d-b992-979e83a91c2e``
``\\scratchpad\\rapm_scope\\predeclaration_draft.md`` and its companion
``implementation_plan.md``, as adjudicated by the reviewing orchestrator:
reframed from "RAPM" to "value-weighted role continuity" (RAPM's full-lineup
design matrix has no CFB data source -- ``src/nfl_ats/cfb.py``'s
credited-actor-only contract), value weight = credited-play EPA (not CFBD
usage), a standalone candidate arm (not stacked on the existing plain
continuity family), NFL's participation-rating shrinkage constants mirrored
as an explicitly-underived first pass, season handling unchanged, and a
20,000-sample bootstrap.

This script is self-contained: it imports existing ``nfl_ats`` helpers
(including a few leading-underscore module-private ones -- reuse, not
modification) but adds no code to ``src/``, writes no registry file, and adds
no test. CFB-only; rotation registry rule 8 makes CFB screens free (no NFL
rotation window touched). Per AGENTS.md, an interval containing zero is never
grounds to close this line -- this script only measures and reports
``probability_positive``; classification is a human/orchestrator decision
made from the numbers below, not something this script asserts.

Design decisions made while executing (flagged here, expanded in the results
write-up):

1. **Column name correction.** The predeclaration draft cites the PBP value
   column as ``epa`` (lowercase); the real canonical column, confirmed on
   disk, is ``EPA`` (uppercase) -- same column, same semantics, just a
   spelling slip in the draft. This script reads ``EPA``.
2. **Weight formula.** The draft leaves open "state x value_rate" vs.
   "value_rate alone" (its DECISIONS NEEDING REVIEW #3, not resolved in the
   orchestrator's 7-item adjudication). This script implements **state x
   value_rate**: the hypothesis text says weighting continuity "by value...
   instead of by usage share ALONE" -- i.e. value augments the existing
   usage-share weight, it does not replace it. Reported as an inferred
   design choice, not a re-litigated adjudicated decision.
3. **EWM input granularity.** "trailing_span8_EWM(credited_play_epa)" is
   read as a PER-PLAY rate (the family is literally named "value RATE", and
   mirrors ``players.py::_player_value_rate``'s numerator/denominator
   structure), updated once per team-game a player has credited plays --
   same cadence as the existing usage-share state EWM. The per-game input is
   ``epa_sum / epa_valid_plays`` for that player's credited dropbacks/carries
   that game, clipped to +/-``VALUE_EPA_CLIP`` before the EWM update
   (mirrors ``participation.py``'s pre-ridge EPA clip).
4. **Reliability / career plays.** ``career_credited_plays`` is a running
   total of credited PLAYS (not games/appearances), reset when a player
   changes teams -- mirrors the existing state/appearances reset convention
   in ``cfb_role_features.py`` exactly ("player changing teams starts a
   fresh state").
5. **Sign safety.** Unlike the plain family's usage-share weight (always in
   [0, 1], so ``active_mass`` is always >= 0), ``value_rate`` can be
   negative (a below-average trailing EPA rate), so ``state * value_rate``
   can be negative. This script does NOT clip or rectify that -- it is
   read literally as specified -- but (a) guards the one real numerical
   trap (``0.0 * NaN`` when reliability is 0 but the value trail is still
   NaN) by defining ``value_rate = 0.0`` whenever reliability <= 0 or the
   value trail is not yet finite, and (b) keeps the existing family's
   ``active_mass > 0.0 -> neutral`` fallback unchanged, which now also
   catches sign-cancellation degeneracy, not just "zero qualified players."
   The frequency of that fallback and the resulting continuity column's
   descriptive stats are reported so any blow-up is visible, not hidden.

CFB seasons 2013-2025 (``FROZEN_ROLE_SEASONS``, unchanged). Season handling,
action types (dropback/carry only), thresholds, streak cap, EWM span, and
neutral imputation are all inherited unchanged from the frozen family per
predeclaration section 2 -- nothing here retunes them.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import nfl_ats.cli as cli  # noqa: E402
from nfl_ats.cfb_benchmark import (  # noqa: E402
    _PREDICTION_PASSTHROUGH,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    _score_week,
    cfb_evaluation_window,
    fit_cfb_residual_model,
    summarize_cfb_benchmark,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS, load_cfb_seasons  # noqa: E402
from nfl_ats.cfb_role_features import (  # noqa: E402
    _CONTINUITY_COLUMNS,
    CFB_ROLE_FEATURE_COLUMNS,
    CONTINUITY_NEUTRAL,
    FROZEN_STREAK_CAP,
    ROLE_BENCHMARK_BASELINE_ARM,
    ROLE_FEATURE_ACTION_TYPES,
    _iter_team_action_games,
    attach_role_continuity,
)
from nfl_ats.cfb_roles import (  # noqa: E402
    CFB_ROLE_PBP_LOAD_COLUMNS,
    FROZEN_MIN_PRIOR_APPEARANCES,
    FROZEN_ROLE_SEASONS,
    FROZEN_ROLE_SPAN,
    FROZEN_ROLE_THRESHOLDS,
    _regular_season_mask,
    cfb_role_actions,
)
from nfl_ats.data import DataContractError, require_columns  # noqa: E402
from nfl_ats.experiments import paired_feature_comparisons  # noqa: E402
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id  # noqa: E402
from nfl_ats.margin import fit_market_baseline  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "cfb_value_weighted_continuity"
CFB_FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"

# New, CFB-specific constants: mirrored from NFL's participation-rating
# shrinkage (participation.py:29-30) as a first pass per predeclaration
# section 2 / adjudicated decision 5. EXPLICITLY UNDERIVED for CFB's
# different play volume and EPA distribution -- flagged in every output.
VALUE_RELIABILITY_PRIOR_PLAYS = 500.0
VALUE_EPA_CLIP = 5.0

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819  # frozen for this run before any accuracy number was seen

CANDIDATE_ARM = "cfb_benchmark_v1_value_roles"
VALUE_ROLE_COLUMNS = (*CFB_MODEL_FEATURE_COLUMNS, *CFB_ROLE_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Step 1: load pbp (+EPA) and build the credited-action tables
# ---------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pbp with EPA, canonical CFB games) for the frozen role seasons."""

    seasons = list(range(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1] + 1))
    pbp = load_cfb_seasons(
        cli._data_root() / "cfb",
        "pbp",
        seasons,
        columns=[*CFB_ROLE_PBP_LOAD_COLUMNS, "pos_team_id", "EPA"],
    )
    canonical_games = pd.read_parquet(CFB_FEATURES_PATH)
    return pbp, canonical_games


def cfb_epa_actions(pbp: pd.DataFrame, canonical_games: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team, player, action_type) credited-play EPA, dropback/carry only.

    A read-only companion to ``nfl_ats.cfb_roles.cfb_role_actions`` -- not a
    modification of it. Replicates that function's own credited-row
    selection (same games join, season/week filter, and
    ``_regular_season_mask``, imported unchanged from ``cfb_roles.py`` so the
    exact same rows are eligible) but keeps the play-level ``EPA`` column
    that ``cfb_role_actions`` drops (``cfb_roles.py:606-611``, "no epa
    column is read"). Restricted to ``ROLE_FEATURE_ACTION_TYPES``
    (dropback, carry) -- this screen has no use for reception-level EPA.

    The coverage gate and ``FROZEN_MIN_TEAM_ACTIONS`` floor are deliberately
    NOT reapplied here; the caller inner/left-joins this frame onto the
    already-gated ``cfb_role_actions`` output, which restricts to exactly
    the keys that survive those gates.
    """

    require_columns(pbp, (*CFB_ROLE_PBP_LOAD_COLUMNS, "EPA"), "cfb play_by_play (+EPA)")
    require_columns(canonical_games, ("game_id", "gameday"), "cfb canonical games")

    games = canonical_games.loc[:, ["game_id", "gameday"]].copy()
    games["game_id"] = games["game_id"].astype(str)
    games = games.drop_duplicates("game_id")

    working = pbp.copy()
    working["game_id"] = working["game_id"].astype(str)
    working["season"] = pd.to_numeric(working["season"], errors="coerce")
    working["week"] = pd.to_numeric(working["week"], errors="coerce")
    working = working.merge(games, on="game_id", how="inner", validate="many_to_one")
    working = working.loc[
        working["season"].notna()
        & working["week"].notna()
        & working["season"].between(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1])
    ].copy()
    working = working.loc[_regular_season_mask(working["seasonType"])].copy()
    working["season"] = working["season"].astype("int64")
    working["week"] = working["week"].astype("int64")

    pass_flag = pd.to_numeric(working["pass"], errors="coerce").fillna(0).astype(bool)
    rush_flag = pd.to_numeric(working["rush"], errors="coerce").fillna(0).astype(bool)
    definitions = {
        "dropback": (pass_flag, working["passer_player_id"]),
        "carry": (rush_flag, working["rusher_player_id"]),
    }
    frames: list[pd.DataFrame] = []
    for action_type, (eligible, credit_column) in definitions.items():
        credited = eligible & credit_column.notna()
        rows = working.loc[credited, ["game_id", "season", "week", "pos_team", "EPA"]].copy()
        rows["action_type"] = action_type
        rows["player_id"] = credit_column.loc[credited].astype(str)
        rows["EPA"] = pd.to_numeric(rows["EPA"], errors="coerce")
        frames.append(rows)
    combined = pd.concat(frames, ignore_index=True)

    epa_actions = (
        (
            combined.groupby(
                ["game_id", "season", "week", "pos_team", "action_type", "player_id"], sort=False
            ).agg(
                epa_sum=("EPA", "sum"),
                epa_valid_plays=("EPA", "count"),
                credited_plays=("EPA", "size"),
            )
        )
        .reset_index()
        .rename(columns={"pos_team": "team"})
    )
    return epa_actions


def build_epa_lookup(
    actions: pd.DataFrame, epa_actions: pd.DataFrame
) -> tuple[dict[tuple[str, str], dict[str, dict[str, tuple[float, int, int]]]], dict[str, Any]]:
    """Nested (team, action_type) -> game_id -> player_id -> (epa_sum, epa_valid_plays, plays).

    Joins the script-local ``epa_actions`` onto the frozen, already-gated
    ``actions`` table (dropback/carry rows only) so every value this screen
    uses inherits the exact same eligibility gates as the plain continuity
    family (coverage >= 0.95, >= 10 team actions/game).
    """

    dc_actions = actions.loc[actions["action_type"].isin(ROLE_FEATURE_ACTION_TYPES)].copy()
    epa_actions = epa_actions.copy()
    for frame in (dc_actions, epa_actions):
        frame["game_id"] = frame["game_id"].astype(str)
        frame["team"] = frame["team"].astype(str)
        frame["player_id"] = frame["player_id"].astype(str)
        frame["action_type"] = frame["action_type"].astype(str)

    merged = dc_actions.merge(
        epa_actions[
            [
                "game_id",
                "team",
                "player_id",
                "action_type",
                "epa_sum",
                "epa_valid_plays",
                "credited_plays",
            ]
        ],
        on=["game_id", "team", "player_id", "action_type"],
        how="left",
        validate="one_to_one",
    )
    unmatched = merged["epa_sum"].isna()
    matched = merged.loc[~unmatched]
    mismatched_counts = int((matched["credited_plays"] != matched["count"]).sum())

    lookup: dict[tuple[str, str], dict[str, dict[str, tuple[float, int, int]]]] = {}
    for row in merged.itertuples(index=False):
        key = (str(row.team), str(row.action_type))
        epa_sum = float(row.epa_sum) if pd.notna(row.epa_sum) else 0.0
        epa_valid_plays = int(row.epa_valid_plays) if pd.notna(row.epa_valid_plays) else 0
        credited_plays = int(row.count)
        lookup.setdefault(key, {}).setdefault(str(row.game_id), {})[str(row.player_id)] = (
            epa_sum,
            epa_valid_plays,
            credited_plays,
        )

    diagnostics = {
        "dc_action_rows": len(dc_actions),
        "unmatched_epa_rows": int(unmatched.sum()),
        "unmatched_epa_fraction": float(unmatched.mean()) if len(merged) else math.nan,
        "count_mismatch_rows": mismatched_counts,
    }
    return lookup, diagnostics


# ---------------------------------------------------------------------------
# Step 2: the new trait -- value-weighted role continuity
# ---------------------------------------------------------------------------


@dataclass
class _ValuePlayerTrail:
    state: float = math.nan  # usage-share EWM state, identical to the plain family
    appearances: int = 0
    missed_streak: int = 0
    last_season: int = -1
    value_state: float = math.nan  # EWM of clipped per-game mean credited-play EPA
    career_plays: int = 0  # cumulative credited plays for this (team, player, action_type)


@dataclass
class _ValueDiagnostics:
    total_rows: int = 0
    neutral_fallback_rows: int = 0
    qualified_player_games: int = 0
    zero_value_rate_player_games: int = 0
    reliabilities: list[float] = field(default_factory=list)
    value_rates: list[float] = field(default_factory=list)


def build_value_weighted_continuity(
    actions: pd.DataFrame,
    team_games: pd.DataFrame,
    epa_lookup: dict[tuple[str, str], dict[str, dict[str, tuple[float, int, int]]]],
    *,
    thresholds: dict[str, float] = FROZEN_ROLE_THRESHOLDS,
    min_prior: int = FROZEN_MIN_PRIOR_APPEARANCES,
    span: int = FROZEN_ROLE_SPAN,
    streak_cap: int = FROZEN_STREAK_CAP,
    reliability_prior_plays: float = VALUE_RELIABILITY_PRIOR_PLAYS,
    epa_clip: float = VALUE_EPA_CLIP,
) -> tuple[pd.DataFrame, _ValueDiagnostics]:
    """Value-weighted analogue of ``cfb_role_features.build_role_continuity``.

    Qualification (season scoping, streak cap, threshold, min prior
    appearances) is IDENTICAL to the frozen plain-continuity family -- the
    only change is the per-player weight contributed to
    ``active_mass``/``appeared_mass``: ``state * value_rate`` instead of
    ``state`` alone, where ``value_rate = value_state * reliability`` (see
    the module docstring, design decisions 2-5).
    """

    alpha = 2.0 / (span + 1.0)
    rows: list[dict[str, Any]] = []
    diag = _ValueDiagnostics()

    for team, action_type, games, per_game in _iter_team_action_games(actions, team_games):
        threshold = thresholds[action_type]
        game_epa = epa_lookup.get((team, action_type), {})
        trails: dict[str, _ValuePlayerTrail] = {}
        for game in games:
            game_season = int(game["season"])
            active_mass = 0.0
            appeared_mass = 0.0
            qualified_players = 0
            for trail in trails.values():
                if not (
                    math.isfinite(trail.state)
                    and trail.state >= threshold
                    and trail.appearances >= min_prior
                    and trail.last_season == game_season
                    and trail.missed_streak < streak_cap
                ):
                    continue
                qualified_players += 1
                diag.qualified_player_games += 1
                reliability = (
                    trail.career_plays / (trail.career_plays + reliability_prior_plays)
                    if trail.career_plays > 0
                    else 0.0
                )
                if reliability <= 0.0 or not math.isfinite(trail.value_state):
                    value_rate = 0.0
                    diag.zero_value_rate_player_games += 1
                else:
                    value_rate = trail.value_state * reliability
                diag.reliabilities.append(reliability)
                diag.value_rates.append(value_rate)
                weight = trail.state * value_rate
                active_mass += weight
                if trail.missed_streak == 0:
                    appeared_mass += weight

            diag.total_rows += 1
            if active_mass > 0.0:
                continuity = appeared_mass / active_mass
            else:
                continuity = CONTINUITY_NEUTRAL
                diag.neutral_fallback_rows += 1

            rows.append(
                {
                    "game_id": str(game["game_id"]),
                    "season": game_season,
                    "week": int(game["week"]),
                    "team": team,
                    "action_type": action_type,
                    "continuity": continuity,
                    "active_mass": active_mass,
                    "absent_mass": active_mass - appeared_mass,
                    "qualified_players": qualified_players,
                }
            )

            appeared = per_game.get(str(game["game_id"]), {})
            epa_this_game = game_epa.get(str(game["game_id"]), {})
            for player, trail in trails.items():
                if player not in appeared:
                    trail.missed_streak += 1
            for player, share in appeared.items():
                trail = trails.setdefault(player, _ValuePlayerTrail())
                trail.state = (
                    share
                    if not math.isfinite(trail.state)
                    else alpha * share + (1.0 - alpha) * trail.state
                )
                trail.appearances += 1
                trail.missed_streak = 0
                trail.last_season = game_season

                epa_sum, epa_valid_plays, credited_plays = epa_this_game.get(player, (0.0, 0, 0))
                if epa_valid_plays > 0:
                    per_play = epa_sum / epa_valid_plays
                    per_play = max(-epa_clip, min(epa_clip, per_play))
                    trail.value_state = (
                        per_play
                        if not math.isfinite(trail.value_state)
                        else alpha * per_play + (1.0 - alpha) * trail.value_state
                    )
                trail.career_plays += credited_plays

    if not rows:
        return pd.DataFrame(columns=_CONTINUITY_COLUMNS), diag
    result = pd.DataFrame(rows, columns=_CONTINUITY_COLUMNS)
    if result.duplicated(["game_id", "team", "action_type"]).any():
        raise DataContractError("Value-weighted continuity produced duplicate team-game rows")
    result = result.sort_values(["season", "week", "game_id", "team", "action_type"]).reset_index(
        drop=True
    )
    return result, diag


def split_half_reliability(
    continuity: pd.DataFrame, action_type: str, *, seed: int, n_boot: int = 4000
) -> dict[str, Any]:
    """Odd/even-week split-half reliability of the new trait (predeclaration section 7).

    Same recipe as ``scripts/cfb_role_continuity_remeasurement.py``'s
    function of the same name: split each team-season's values by odd/even
    week, correlate the two halves' team-season means, Spearman-Brown
    correct to full-length reliability, bootstrap a CI and
    ``probability_positive``.
    """

    from scipy.stats import spearmanr

    subset = continuity.loc[continuity["action_type"].eq(action_type)].copy()
    subset["half"] = np.where(subset["week"] % 2 == 0, "even", "odd")
    means = subset.groupby(["team", "season", "half"])["continuity"].mean().unstack("half")
    counts = subset.groupby(["team", "season", "half"]).size().unstack("half")
    valid = (counts.get("odd", 0) >= 2) & (counts.get("even", 0) >= 2)
    means = means.dropna()
    valid = valid.reindex(means.index).fillna(False)
    means = means.loc[valid]

    odd_vals = means["odd"].to_numpy(dtype=float)
    even_vals = means["even"].to_numpy(dtype=float)
    n = len(means)
    r = float(np.corrcoef(odd_vals, even_vals)[0, 1])
    rho = float(spearmanr(odd_vals, even_vals).correlation)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sel = rng.integers(0, n, size=n)
        boots[i] = np.corrcoef(odd_vals[sel], even_vals[sel])[0, 1]
    sb = (2.0 * r) / (1.0 + r) if r > -1.0 else float("nan")
    return {
        "action_type": action_type,
        "n_team_seasons": n,
        "pearson_r": r,
        "pearson_r_ci95": [
            float(np.nanquantile(boots, 0.025)),
            float(np.nanquantile(boots, 0.975)),
        ],
        "spearman_rho": rho,
        "spearman_brown_full_length_reliability": sb,
        "probability_positive": float(np.mean(boots > 0.0)),
    }


# ---------------------------------------------------------------------------
# Step 3: accuracy screen -- baseline vs baseline + value-weighted continuity
# ---------------------------------------------------------------------------


def value_role_benchmark(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    """Walk-forward: market, frozen market_residual, and the value-continuity candidate.

    Structurally identical to ``cfb_role_features.cfb_role_benchmark`` (same
    weeks, same training-window/Ridge recipe, same paired-comparison
    machinery) with the plain-continuity candidate swapped for the
    value-weighted one -- a standalone arm, per adjudicated decision 3, not
    stacked with the plain continuity columns.
    """

    required = {*_PREDICTION_PASSTHROUGH, *CFB_MODEL_FEATURE_COLUMNS, *CFB_ROLE_FEATURE_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(
            f"CFB value-role features are missing columns: {', '.join(missing)}"
        )
    role_values = features.loc[:, list(CFB_ROLE_FEATURE_COLUMNS)]
    if bool(role_values.nunique(dropna=False).le(1).all()):
        raise DataContractError(
            "Every value-continuity column is constant; the feature join produced no "
            "information, so a benchmark comparison would be vacuous"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        models = {
            "market": fit_market_baseline(training),
            "market_residual": fit_cfb_residual_model(training, ridge_alpha=ridge_alpha),
            "market_residual_value_roles": fit_cfb_residual_model(
                training, ridge_alpha=ridge_alpha, feature_columns=VALUE_ROLE_COLUMNS
            ),
        }
        batches.extend(_score_week(weekly_games, models))
    if not batches:
        raise ValueError("No CFB week had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    summary, season_summary = summarize_cfb_benchmark(predictions)

    clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core")
        & predictions["method"].isin(("market_residual", "market_residual_value_roles"))
    ].copy()
    if clean.empty:
        raise ValueError("No clean-core predictions available for the paired comparison")
    clean["feature_set"] = clean["method"].map(
        {
            "market_residual": ROLE_BENCHMARK_BASELINE_ARM,
            "market_residual_value_roles": CANDIDATE_ARM,
        }
    )
    paired = pd.concat(
        [
            paired_feature_comparisons(
                clean,
                baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
                samples=bootstrap_samples,
                block="week",
                seed=bootstrap_seed,
            ),
            paired_feature_comparisons(
                clean,
                baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
                samples=bootstrap_samples,
                block="season",
                seed=bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    return {
        "predictions": predictions,
        "summary": summary,
        "season_summary": season_summary,
        "paired": paired,
        "clean_paired_games": clean["game_id"].nunique(),
    }


def _reliability_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def _value_rate_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
        "fraction_negative": float((arr < 0.0).mean()),
        "fraction_zero": float((arr == 0.0).mean()),
    }


def main() -> None:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    output = OUT_ROOT / run_id()
    print(f"Artifact directory: {output}", flush=True)

    print("=== Step 1: load pbp (+EPA) and build credited-action tables ===", flush=True)
    t0 = time.perf_counter()
    pbp, canonical_games = load_inputs()
    actions, team_games, coverage = cfb_role_actions(pbp, canonical_games)
    team_ids = (
        pbp.loc[
            pbp["pos_team"].notna() & pbp["pos_team_id"].notna(),
            ["game_id", "pos_team", "pos_team_id"],
        ]
        .rename(columns={"pos_team": "team", "pos_team_id": "team_id"})
        .drop_duplicates(["game_id", "team"])
    )
    epa_actions = cfb_epa_actions(pbp, canonical_games)
    epa_lookup, epa_join_diag = build_epa_lookup(actions, epa_actions)
    timings["step1_load_and_actions_seconds"] = time.perf_counter() - t0
    print(
        f"pbp rows={len(pbp)} actions rows={len(actions)} epa_actions rows={len(epa_actions)} "
        f"epa_join={json.dumps(epa_join_diag)}",
        flush=True,
    )

    print("\n=== Step 2: build value-weighted continuity (new trait) ===", flush=True)
    t0 = time.perf_counter()
    value_continuity, value_diag = build_value_weighted_continuity(actions, team_games, epa_lookup)
    timings["step2_build_continuity_seconds"] = time.perf_counter() - t0
    neutral_fallback_rate = (
        value_diag.neutral_fallback_rows / value_diag.total_rows
        if value_diag.total_rows
        else math.nan
    )
    continuity_desc = (
        value_continuity.groupby("action_type")["continuity"].describe().to_dict(orient="index")
    )
    diagnostics_report = {
        "total_team_game_action_rows": value_diag.total_rows,
        "neutral_fallback_rows": value_diag.neutral_fallback_rows,
        "neutral_fallback_rate": neutral_fallback_rate,
        "qualified_player_games": value_diag.qualified_player_games,
        "zero_value_rate_player_games": value_diag.zero_value_rate_player_games,
        "zero_value_rate_fraction": (
            value_diag.zero_value_rate_player_games / value_diag.qualified_player_games
            if value_diag.qualified_player_games
            else math.nan
        ),
        "reliability_summary": _reliability_summary(value_diag.reliabilities),
        "value_rate_summary": _value_rate_summary(value_diag.value_rates),
        "continuity_describe_by_action_type": continuity_desc,
    }
    print(json.dumps(diagnostics_report, indent=2, default=str), flush=True)

    print(
        "\n=== Step 3: split-half reliability of the value-weighted trait "
        "(BEFORE the accuracy screen) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    reliability_results = {
        "dropback": split_half_reliability(value_continuity, "dropback", seed=1),
        "carry": split_half_reliability(value_continuity, "carry", seed=2),
    }
    timings["step3_split_half_reliability_seconds"] = time.perf_counter() - t0
    print(json.dumps(reliability_results, indent=2, default=str), flush=True)

    print("\n=== Step 4: attach features and run the 3-arm accuracy screen ===", flush=True)
    t0 = time.perf_counter()
    value_features = attach_role_continuity(canonical_games, value_continuity, team_ids)
    benchmark = value_role_benchmark(value_features)
    timings["step4_accuracy_screen_seconds"] = time.perf_counter() - t0
    paired = benchmark["paired"]
    accuracy_week = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")
    ].iloc[0]
    accuracy_season = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("season")
    ].iloc[0]
    brier_week = paired.loc[
        paired["metric"].eq("brier_improvement") & paired["block"].eq("week")
    ].iloc[0]
    logloss_week = paired.loc[
        paired["metric"].eq("log_loss_improvement") & paired["block"].eq("week")
    ].iloc[0]
    screen_summary = {
        "clean_paired_games": int(benchmark["clean_paired_games"]),
        "accuracy_improvement_week": accuracy_week.to_dict(),
        "accuracy_improvement_season": accuracy_season.to_dict(),
        "brier_improvement_week": brier_week.to_dict(),
        "log_loss_improvement_week": logloss_week.to_dict(),
    }
    print(json.dumps(screen_summary, indent=2, default=str), flush=True)

    print("\n=== Step 5: write artifacts ===", flush=True)
    atomic_parquet(value_continuity, output / "value_continuity.parquet")
    stamp_sidecar(output / "value_continuity.parquet")  # ENG-38
    atomic_parquet(benchmark["predictions"], output / "predictions.parquet")
    stamp_sidecar(output / "predictions.parquet")  # ENG-38
    atomic_csv(benchmark["summary"], output / "summary.csv")
    stamp_sidecar(output / "summary.csv")  # ENG-38
    atomic_csv(benchmark["season_summary"], output / "season_summary.csv")
    stamp_sidecar(output / "season_summary.csv")  # ENG-38
    atomic_csv(paired, output / "paired_comparisons.csv")
    stamp_sidecar(output / "paired_comparisons.csv")  # ENG-38
    atomic_csv(coverage, output / "coverage.csv")
    stamp_sidecar(output / "coverage.csv")  # ENG-38
    atomic_json(reliability_results, output / "split_half_reliability.json")
    atomic_json(diagnostics_report, output / "value_diagnostics.json")

    timings["total_seconds"] = time.perf_counter() - started
    metadata = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/cfb_value_weighted_continuity_screen.py",
        "predeclaration_source": (
            "scratchpad rapm_scope/predeclaration_draft.md + implementation_plan.md "
            "(2026-08-18 scoping agent; DRAFT, not a committed docs/ file)"
        ),
        "candidate_arm": CANDIDATE_ARM,
        "baseline_arm": ROLE_BENCHMARK_BASELINE_ARM,
        "value_reliability_prior_plays": VALUE_RELIABILITY_PRIOR_PLAYS,
        "value_epa_clip": VALUE_EPA_CLIP,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "role_seasons": list(FROZEN_ROLE_SEASONS),
        "epa_join_diagnostics": epa_join_diag,
        "value_diagnostics": diagnostics_report,
        "split_half_reliability": reliability_results,
        "screen_summary": screen_summary,
        "timing": timings,
    }
    write_stamped_artifact(metadata, output / "metadata.json")  # ENG-38
    print(f"\nWrote artifacts to {output}", flush=True)
    print(f"Total runtime: {timings['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
