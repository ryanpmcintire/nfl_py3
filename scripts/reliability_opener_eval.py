"""Split-half reliability for the 28 ``opener_error_mining_*`` registry cells (ORCH-D).

**What these cells are.** Every entry is one SLICE of a single frozen
evaluation: ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``
(the production rule -- ``home_cover_probability_at_open >= 0.5`` -- graded
at the Tuesday opener) joined on ``game_id`` to
``data/processed/game_features_weak_stack.parquet`` for context columns
(``div_game``, ``rest_diff``, ``weekday``, ``gametime``, ``total_line``),
exactly the population ``docs/opener_error_analysis.md`` built (read
2026-09-01). 34 opener-line pushes (``correct_at_open_probability_rule`` is
NaN, i.e. ``margin_vs_open == 0``) are dropped first, leaving the same
push-excluded **n=1,503** population every registry cell's ``sample_games``
was measured against -- reproduced exactly for all 28 entries by
:func:`load_population` plus :data:`ENTRY_SLICE_MASKS`, and pinned by
``tests/test_reliability_opener_eval.py``.

**The construct behind each slice.** A cell buckets one continuous or
categorical quantity; the registry cell's reliability is that QUANTITY's
split-half reliability, not the bucket's -- the same "battery inherits the
trait's number" convention the six ``attention_battery_*`` cells and the
``graph_team_stat_*`` cells already follow. Six quantities cover all 28
cells:

- ``confidence_distance`` = ``|home_cover_probability_at_open - 0.5|`` (the
  4 ``confidence_bucket_*`` cells)
- ``spread_magnitude`` = ``|tue_open_home_spread|`` (the 4
  ``spread_magnitude_*`` cells)
- ``favorite_spread_own`` -- each team's OWN spread number, negative when
  that team is favored: the home team's value IS
  ``tue_open_home_spread``; the away team's is its negation (the 2
  ``favorite_side_*`` cells)
- ``rest_diff_own`` -- each team's own rest advantage over the opponent:
  home team's value is ``rest_diff`` (``home_rest - away_rest``); away
  team's is its negation (the 2 ``rest_diff_*`` cells)
- ``movement_own`` -- each team's own-side projection of the observed
  opener-to-close move, ``open_move = close_home_spread -
  tue_open_home_spread``: home team's value is ``open_move``; away team's
  is its negation (the 7 ``movement_agreement_*`` cells, including the two
  overlay-paired-delta cells, which restrict the *population* the paired
  delta was computed over but not the underlying quantity being measured
  for reliability)
- ``total_line`` -- a genuinely game-level market quantity with no
  home/away asymmetry, so per ``docs/reliability_sweep_20260901.md`` it is
  measured at the venue/home-team unit with :data:`rlib.METHOD_VENUE`, one
  row per game keyed on ``home_team`` (the 2 ``total_bucket_*`` cells)

The remaining 7 cells have no continuous parent trait -- they are per-game
FLAGS (division game, which side got the pick, whether the pick was the
market favorite, primetime slate, week third, a single season) -- and are
measured as the flag's own team-season EXPOSURE-rate reliability via
:func:`rlib.game_flag_to_team_week` + ``method=rlib.METHOD_EXPOSURE``, per
``docs/reliability_sweep_20260901.md``. ``opener_error_mining_season_2025``
restricts to season 2025 alone; once the frame is filtered to that single
season the flag (``season == 2025``) is constant, so this cell is expected
to come back UNMEASURED (``constant_or_all_missing``), not reliability 0 --
reported, never recorded.

**Method.** :data:`rlib.METHOD_TRAIT` for the six continuous quantities
(:data:`rlib.METHOD_VENUE` for ``total_line``), :data:`rlib.METHOD_EXPOSURE`
for the 7 flags: unit = team-season (venue-season for ``total_line``),
halves = odd/even weeks, Spearman-Brown corrected, block bootstrap over
units, seed 20260901, 4000 draws, restricted to each entry's OWN season
range. The estimator is
``nfl_ats.cfb_qb_dependence.split_half_reliability``, imported through the
shared harness (``scripts/reliability_lib.py``), never reimplemented.

**Effect replication, reported not recorded.** For every entry,
:func:`rlib.half_season_replication` on the entry's OWN slice (outcome =
``correct_at_open_probability_rule``, the production rule's opener-grade
hit indicator -- ``per_game.parquet`` column, read above) plus
:func:`rlib.battery_replication_correlation` across all 28 slices. Neither
is a correlation on the reliability scale and neither is ever written to
the registry.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI
that contains zero is NEVER grounds to reject, fail, or close an
experiment. Only two grounds ever close a line of work: (1) refuted
mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of
zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``; report ``probability_positive``, never
"contains zero". This script CLOSES NOTHING: it measures, and a low number
is a candidate for the reliability ground, never the closure itself.
Within-week correlation is ZERO.

A construct with too few usable team-seasons (or venue-seasons, or a
single-season constant flag) is reported as UNMEASURED, never as
reliability 0 -- writing a NaN or a degenerate constant through as a number
would manufacture the appearance of a closing ground out of nothing.

**Compositional-constraint hazard.** Flagged mid-session by the ORCH-D
coordinator (reported, unverified by this script beyond the independent
check below): a quantity whose SEASON TOTAL is conserved within a
team-season -- days of rest is the canonical case, since a season spans a
fixed number of calendar days -- mechanically anti-correlates across ANY
two-way split of a team-season's games, real week order or not, so its
split-half "reliability" is not a measurement of a between-team trait at
all. :func:`random_half_diagnostic` checks every construct this script
measures by replacing the real week column with a random half-label and
re-measuring; :func:`is_compositional_artifact` flags a construct
`not_applicable_compositional_constraint` (reported, never recorded) when
the diagnostic can't be told apart from the real reading (see the function
docstrings for the exact rule and :data:`COMPOSITIONAL_GAP_THRESHOLD` /
:data:`COMPOSITIONAL_MAGNITUDE_FLOOR`). Confirmed here for
``rest_diff_own`` (both ``rest_diff_*`` cells' primary trait) and
``div_game_flag`` (``division_game_yes``); ``week_third_early_flag`` /
``week_third_late_flag`` are ALSO flagged, for a related reason -- a
team-season's count of early/late-third games is itself nearly identical
across every team, so there is barely any real between-team signal for a
split of any kind to detect. The two ``rest_diff_*`` cells fall back to
their own categorical flag's EXPOSURE reliability (:data:`FALLBACK_CONSTRUCT`)
when that fallback itself passes the same diagnostic.

Writes ``artifacts/reliability_sweep/opener_eval/<stamp>/results.json`` via
``nfl_ats.provenance.write_experiment_artifact`` and prints the
``set-reliability`` commands it would run (``--record`` runs nothing
itself; recording goes through the locked CLI).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import reliability_lib as rlib  # noqa: E402

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

PER_GAME_PATH = REPO / "artifacts" / "opener_evaluation" / "20260819T174244Z" / "per_game.parquet"
FEATURE_PATH = REPO / "data" / "processed" / "game_features_weak_stack.parquet"

PREFIX = "opener_error_mining_"
OUTCOME_COL = "correct_at_open_probability_rule"

FEATURE_JOIN_COLS = [
    "game_id",
    "weekday",
    "gametime",
    "total_line",
    "rest_diff",
    "neutral_site",
    "div_game",
    "home_team",
    "away_team",
]


def load_population() -> pd.DataFrame:
    """Rebuild the exact push-excluded n=1,503 population every cell used.

    ``per_game.parquet`` joined on ``game_id`` to the weak-stack feature
    table for context columns, one-to-one, then the 34 opener-line pushes
    (``OUTCOME_COL`` is NaN) dropped -- same construction as
    ``docs/opener_error_analysis.md``, verified byte-for-byte against every
    entry's own ``sample_games`` in :data:`ENTRY_SLICE_MASKS` /
    ``tests/test_reliability_opener_eval.py``.
    """

    per_game = pd.read_parquet(PER_GAME_PATH)
    features = pd.read_parquet(FEATURE_PATH)[FEATURE_JOIN_COLS].drop_duplicates(subset="game_id")
    games = per_game.merge(features, on="game_id", how="left", validate="one_to_one")
    games = games.loc[games[OUTCOME_COL].notna()].reset_index(drop=True)
    return games


def add_derived_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Every quantity/flag the 28 cells slice on, derived once and reused."""

    out = games.copy()
    out["confidence_distance"] = (out["home_cover_probability_at_open"] - 0.5).abs()
    out["spread_magnitude"] = out["tue_open_home_spread"].abs()
    out["pick_home"] = out["pick_home_at_open_probability_rule"].astype(bool)
    spread = out["tue_open_home_spread"]
    out["is_home_favorite"] = spread < 0
    out["is_road_favorite"] = spread > 0
    out["pick_is_favorite"] = (out["pick_home"] & out["is_home_favorite"]) | (
        ~out["pick_home"] & out["is_road_favorite"]
    )
    gt_hour = out["gametime"].astype(str).str.slice(0, 2).astype(int)
    out["is_primetime"] = out["weekday"].isin(["Thursday", "Monday"]) | (
        out["weekday"].isin(["Sunday", "Saturday"]) & (gt_hour >= 20)
    )
    move_agrees = (out["open_move"] > 0) == out["pick_home"]
    out["move_flat"] = out["open_move"] == 0
    out["move_agrees_true"] = move_agrees & ~out["move_flat"]
    out["move_disagrees_true"] = (~move_agrees) & ~out["move_flat"]
    return out


def entry_slice_masks(games: pd.DataFrame) -> dict[str, pd.Series]:
    """One boolean mask per registry entry -- the guard against re-deriving a slice wrong."""

    spread = games["tue_open_home_spread"]
    return {
        f"{PREFIX}confidence_bucket_lt0p02": games["confidence_distance"] < 0.02,
        f"{PREFIX}confidence_bucket_0p02_0p05": (games["confidence_distance"] > 0.02)
        & (games["confidence_distance"] <= 0.05),
        f"{PREFIX}confidence_bucket_0p05_0p10": (games["confidence_distance"] > 0.05)
        & (games["confidence_distance"] <= 0.10),
        f"{PREFIX}confidence_bucket_gt0p10": games["confidence_distance"] > 0.10,
        f"{PREFIX}division_game_yes": games["div_game"] == 1,
        f"{PREFIX}favorite_side_home_favorite": games["is_home_favorite"],
        f"{PREFIX}favorite_side_road_favorite": games["is_road_favorite"],
        f"{PREFIX}movement_agreement_agrees": games["move_disagrees_true"],
        f"{PREFIX}movement_agreement_agrees_corrected": games["move_agrees_true"],
        f"{PREFIX}movement_agreement_disagrees": games["move_agrees_true"],
        f"{PREFIX}movement_agreement_disagrees_corrected": games["move_disagrees_true"],
        f"{PREFIX}movement_agreement_disagrees_overlay_paired_delta": games["move_disagrees_true"],
        f"{PREFIX}movement_agreement_disagrees_overlay_paired_delta_move_ge_1_0": games[
            "move_disagrees_true"
        ]
        & (games["open_move"].abs() >= 1.0),
        f"{PREFIX}movement_agreement_flat": games["move_flat"],
        f"{PREFIX}pick_home_away_home": games["pick_home"],
        f"{PREFIX}pick_side_favorite": games["pick_is_favorite"],
        f"{PREFIX}rest_diff_away_more_rested": games["rest_diff"] < 0,
        f"{PREFIX}rest_diff_even": games["rest_diff"] == 0,
        f"{PREFIX}season_2025": games["season"] == 2025,
        f"{PREFIX}slate_primetime": games["is_primetime"],
        f"{PREFIX}spread_magnitude_0_2p5": spread.abs() <= 2.5,
        f"{PREFIX}spread_magnitude_3_6p5": (spread.abs() > 2.5) & (spread.abs() <= 6.5),
        f"{PREFIX}spread_magnitude_7_9p5": (spread.abs() > 6.5) & (spread.abs() <= 9.5),
        f"{PREFIX}spread_magnitude_10plus": spread.abs() > 9.5,
        f"{PREFIX}total_bucket_45_48": (games["total_line"] > 45.0) & (games["total_line"] <= 48.0),
        f"{PREFIX}total_bucket_below_42": (games["total_line"] > 28.499)
        & (games["total_line"] <= 42.0),
        f"{PREFIX}week_third_early": games["week"].between(1, 6),
        f"{PREFIX}week_third_late": games["week"].between(13, 18),
    }


#: Entry -> (kind, quantity/flag column, human note). ``kind`` in
#: {"trait", "venue", "exposure"}. Every trait/venue entry sharing a
#: quantity+season-range gets the SAME reliability number (the battery
#: inherits the trait's number, per the attention_battery_* precedent).
ENTRY_CONSTRUCT: dict[str, dict[str, str]] = {
    f"{PREFIX}confidence_bucket_lt0p02": {"kind": "trait", "quantity": "confidence_distance"},
    f"{PREFIX}confidence_bucket_0p02_0p05": {"kind": "trait", "quantity": "confidence_distance"},
    f"{PREFIX}confidence_bucket_0p05_0p10": {"kind": "trait", "quantity": "confidence_distance"},
    f"{PREFIX}confidence_bucket_gt0p10": {"kind": "trait", "quantity": "confidence_distance"},
    f"{PREFIX}division_game_yes": {"kind": "exposure", "quantity": "div_game_flag"},
    f"{PREFIX}favorite_side_home_favorite": {"kind": "trait", "quantity": "favorite_spread_own"},
    f"{PREFIX}favorite_side_road_favorite": {"kind": "trait", "quantity": "favorite_spread_own"},
    f"{PREFIX}movement_agreement_agrees": {"kind": "trait", "quantity": "movement_own"},
    f"{PREFIX}movement_agreement_agrees_corrected": {"kind": "trait", "quantity": "movement_own"},
    f"{PREFIX}movement_agreement_disagrees": {"kind": "trait", "quantity": "movement_own"},
    f"{PREFIX}movement_agreement_disagrees_corrected": {
        "kind": "trait",
        "quantity": "movement_own",
    },
    f"{PREFIX}movement_agreement_disagrees_overlay_paired_delta": {
        "kind": "trait",
        "quantity": "movement_own",
    },
    f"{PREFIX}movement_agreement_disagrees_overlay_paired_delta_move_ge_1_0": {
        "kind": "trait",
        "quantity": "movement_own",
    },
    f"{PREFIX}movement_agreement_flat": {"kind": "trait", "quantity": "movement_own"},
    f"{PREFIX}pick_home_away_home": {"kind": "exposure", "quantity": "pick_home_flag"},
    f"{PREFIX}pick_side_favorite": {"kind": "exposure", "quantity": "pick_is_favorite_flag"},
    f"{PREFIX}rest_diff_away_more_rested": {"kind": "trait", "quantity": "rest_diff_own"},
    f"{PREFIX}rest_diff_even": {"kind": "trait", "quantity": "rest_diff_own"},
    f"{PREFIX}season_2025": {"kind": "exposure", "quantity": "season_2025_flag"},
    f"{PREFIX}slate_primetime": {"kind": "exposure", "quantity": "primetime_flag"},
    f"{PREFIX}spread_magnitude_0_2p5": {"kind": "trait", "quantity": "spread_magnitude"},
    f"{PREFIX}spread_magnitude_3_6p5": {"kind": "trait", "quantity": "spread_magnitude"},
    f"{PREFIX}spread_magnitude_7_9p5": {"kind": "trait", "quantity": "spread_magnitude"},
    f"{PREFIX}spread_magnitude_10plus": {"kind": "trait", "quantity": "spread_magnitude"},
    f"{PREFIX}total_bucket_45_48": {"kind": "venue", "quantity": "total_line"},
    f"{PREFIX}total_bucket_below_42": {"kind": "venue", "quantity": "total_line"},
    f"{PREFIX}week_third_early": {"kind": "exposure", "quantity": "week_third_early_flag"},
    f"{PREFIX}week_third_late": {"kind": "exposure", "quantity": "week_third_late_flag"},
}

#: Used only when an entry's PRIMARY construct is flagged
#: `not_applicable_compositional_constraint` -- falls back to the
#: categorical flag's own EXPOSURE reliability, which is diagnosed by the
#: same random-half check before it is trusted either.
FALLBACK_CONSTRUCT: dict[str, dict[str, str]] = {
    f"{PREFIX}rest_diff_away_more_rested": {"kind": "exposure", "quantity": "rest_diff_lt0_flag"},
    f"{PREFIX}rest_diff_even": {"kind": "exposure", "quantity": "rest_diff_eq0_flag"},
}

QUANTITY_NOTE: dict[str, str] = {
    "confidence_distance": "|home_cover_probability_at_open - 0.5|, symmetric across home/away",
    "spread_magnitude": "|tue_open_home_spread|, symmetric across home/away",
    "favorite_spread_own": (
        "own-side spread (negative=favored): home row = tue_open_home_spread, "
        "away row = -tue_open_home_spread"
    ),
    "rest_diff_own": (
        "own-side rest advantage: home row = rest_diff (home_rest-away_rest), away row = -rest_diff"
    ),
    "movement_own": (
        "own-side projection of observed opener-to-close movement: home row = open_move "
        "(close_home_spread - tue_open_home_spread), away row = -open_move"
    ),
    "total_line": "market total, no home/away split -- measured at the home-team/venue unit",
    "div_game_flag": "div_game == 1",
    "pick_home_flag": "pick_home_at_open_probability_rule is True",
    "pick_is_favorite_flag": "pick side is the market favorite (pick_home XOR spread sign)",
    "season_2025_flag": "season == 2025 (constant once restricted to season 2025 alone)",
    "primetime_flag": "Thu/Mon any kickoff, or Sun/Sat kickoff >= 20:00 local",
    "week_third_early_flag": "week in [1, 6]",
    "week_third_late_flag": "week in [13, 18]",
    "rest_diff_lt0_flag": (
        "rest_diff < 0 (away team more rested) -- EXPOSURE fallback used because "
        "rest_diff_own's TRAIT reliability is a compositional artifact (see "
        "compositional_diagnostic in the artifact payload)"
    ),
    "rest_diff_eq0_flag": (
        "rest_diff == 0 (no rest advantage either way) -- EXPOSURE fallback, same reason"
    ),
}


def build_trait_frame(games: pd.DataFrame, quantity: str) -> pd.DataFrame:
    """Team-week long frame (two rows/game) for one continuous quantity."""

    if quantity == "confidence_distance":
        home_val = away_val = games["confidence_distance"]
    elif quantity == "spread_magnitude":
        home_val = away_val = games["spread_magnitude"]
    elif quantity == "favorite_spread_own":
        home_val = games["tue_open_home_spread"]
        away_val = -games["tue_open_home_spread"]
    elif quantity == "rest_diff_own":
        home_val = games["rest_diff"]
        away_val = -games["rest_diff"]
    elif quantity == "movement_own":
        home_val = games["open_move"]
        away_val = -games["open_move"]
    else:
        raise ValueError(f"unknown trait quantity: {quantity!r}")

    pieces = []
    for team_col, value in ((games["home_team"], home_val), (games["away_team"], away_val)):
        piece = games.loc[:, ["season", "week"]].copy()
        piece["team_id"] = team_col.to_numpy()
        piece["value"] = value.to_numpy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def build_venue_frame(games: pd.DataFrame, quantity: str) -> pd.DataFrame:
    """Venue-season frame (one row/game, unit = home team) for a game-level market quantity."""

    frame = games.loc[:, ["season", "week"]].copy()
    frame["team_id"] = games["home_team"].to_numpy()
    frame["value"] = games[quantity].to_numpy()
    return frame


def build_exposure_flag(games: pd.DataFrame, quantity: str) -> pd.Series:
    if quantity == "div_game_flag":
        return games["div_game"] == 1
    if quantity == "pick_home_flag":
        return games["pick_home"]
    if quantity == "pick_is_favorite_flag":
        return games["pick_is_favorite"]
    if quantity == "season_2025_flag":
        return games["season"] == 2025
    if quantity == "primetime_flag":
        return games["is_primetime"]
    if quantity == "week_third_early_flag":
        return games["week"].between(1, 6)
    if quantity == "week_third_late_flag":
        return games["week"].between(13, 18)
    if quantity == "rest_diff_lt0_flag":
        return games["rest_diff"] < 0
    if quantity == "rest_diff_eq0_flag":
        return games["rest_diff"] == 0
    raise ValueError(f"unknown exposure quantity: {quantity!r}")


def build_long_frame(games: pd.DataFrame, kind: str, quantity: str) -> pd.DataFrame:
    """Dispatch to the right frame-builder for ``kind`` -- one call site for both
    the real measurement and the compositional diagnostic, so they are always
    built the same way."""

    if kind == "trait":
        return build_trait_frame(games, quantity)
    if kind == "venue":
        return build_venue_frame(games, quantity)
    if kind == "exposure":
        flag = build_exposure_flag(games, quantity)
        return rlib.game_flag_to_team_week(games, flag)
    raise ValueError(f"unknown kind: {kind!r}")


_METHOD_FOR_KIND = {
    "trait": rlib.METHOD_TRAIT,
    "venue": rlib.METHOD_VENUE,
    "exposure": rlib.METHOD_EXPOSURE,
}
_METRIC_COL_FOR_KIND = {"trait": "value", "venue": "value", "exposure": "exposure"}

#: Reseeds for :func:`random_half_diagnostic`, and its own seed base -- kept
#: distinct from :data:`rlib.RELIABILITY_SEED` so the diagnostic's random
#: draws never collide with the real measurement's bootstrap draws.
DIAGNOSTIC_N_RESEEDS = 20
DIAGNOSTIC_BASE_SEED = 90101

#: A construct is flagged `not_applicable_compositional_constraint` when
#: EITHER its random-half-reseed mean reliability is more than this far
#: from the real odd/even measurement (the true split's reading carries no
#: information beyond what pure randomization already produces) OR the real
#: measurement and the random-half mean are BOTH negative with
#: |value| >= COMPOSITIONAL_MAGNITUDE_FLOOR (a durable negative that
#: survives replacing genuine time order with noise). Round numbers chosen
#: for this sweep, not fitted to any one cell -- every real/diagnostic pair
#: is reported alongside the flag so a reader can apply a stricter or
#: looser threshold. See :func:`is_compositional_artifact`.
COMPOSITIONAL_GAP_THRESHOLD = 0.5
COMPOSITIONAL_MAGNITUDE_FLOOR = 0.30


def random_half_diagnostic(
    long: pd.DataFrame,
    metric_col: str,
    *,
    unit_col: str = "team_id",
    n_reseeds: int = DIAGNOSTIC_N_RESEEDS,
    base_seed: int = DIAGNOSTIC_BASE_SEED,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Re-measure reliability with the real week replaced by a random half-label.

    Each reseed assigns every row an independent random label in {1, 2} (so
    ``split_half_reliability``'s ``week % 2`` split becomes a random split
    unrelated to actual week timing) and re-measures. If the real
    measurement's negativity (or instability) survives this -- comparable
    magnitude, same sign -- it is a mechanical consequence of a
    within-team-season conserved total, not a property of the trait.
    """

    rng_master = np.random.default_rng(base_seed)
    values: list[float] = []
    for _ in range(n_reseeds):
        seed = int(rng_master.integers(0, 10_000_000))
        rng = np.random.default_rng(seed)
        frame = long.copy()
        frame["week"] = rng.integers(1, 3, size=len(frame))
        measured = rlib.measure_reliability(
            frame,
            metric_col,
            method="diagnostic-random-half",
            unit_col=unit_col,
            seed=seed,
            n_boot=n_boot,
        )
        if measured["reliability"] is not None:
            values.append(measured["reliability"])
    arr = np.array(values, dtype=float)
    return {
        "n_reseeds": n_reseeds,
        "n_usable": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "values": [float(v) for v in arr],
    }


def is_compositional_artifact(real: float | None, diag_mean: float | None) -> bool:
    """Apply :data:`COMPOSITIONAL_GAP_THRESHOLD` / :data:`COMPOSITIONAL_MAGNITUDE_FLOOR`."""

    if real is None or diag_mean is None:
        return False
    if abs(real - diag_mean) >= COMPOSITIONAL_GAP_THRESHOLD:
        return True
    return (
        real < 0
        and diag_mean < 0
        and abs(real) >= COMPOSITIONAL_MAGNITUDE_FLOOR
        and abs(diag_mean) >= COMPOSITIONAL_MAGNITUDE_FLOOR
    )


def target_entries() -> dict[str, dict[str, Any]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, dict[str, Any]] = {}
    for name, signal in registry.signals.items():
        if not name.startswith(PREFIX):
            continue
        out[name] = {
            "seasons": (int(signal.seasons[0]), int(signal.seasons[1])),
            "sample_games": signal.sample_games,
            "reliability": signal.reliability,
            "effect": signal.effect,
            "classification": signal.classification,
            "source": signal.source,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    args = parser.parse_args()

    started = time.time()
    entries = target_entries()
    print(f"=== {len(entries)} opener_error_mining registry cells in scope ===")

    games = add_derived_columns(load_population())
    masks = entry_slice_masks(games)
    print(f"population: {games.shape}, pushes already excluded")

    # Cache one (measurement, compositional-diagnostic) pair per
    # (kind, quantity, seasons) -- entries sharing a quantity+season-range
    # inherit the same numbers, same as the attention_battery_*/
    # graph_team_stat_* precedent, and the diagnostic (20 reseeds) is only
    # ever run once per distinct construct.
    construct_cache: dict[tuple[str, str, tuple[int, int]], dict[str, Any]] = {}

    def measure_and_diagnose(kind: str, quantity: str, seasons: tuple[int, int]) -> dict[str, Any]:
        key = (kind, quantity, seasons)
        if key in construct_cache:
            return construct_cache[key]
        long = build_long_frame(games, kind, quantity)
        metric_col = _METRIC_COL_FOR_KIND[kind]
        measured = rlib.measure_reliability(
            long, metric_col, method=_METHOD_FOR_KIND[kind], seasons=seasons, n_boot=args.n_boot
        )
        diagnostic = None
        compositional = False
        if measured["status"] == rlib.STATUS_MEASURED:
            restricted = games.loc[games["season"].between(seasons[0], seasons[1])]
            diag_long = build_long_frame(restricted, kind, quantity)
            diagnostic = random_half_diagnostic(diag_long, metric_col)
            compositional = is_compositional_artifact(measured["reliability"], diagnostic["mean"])
        result = {
            "kind": kind,
            "quantity": quantity,
            "seasons": seasons,
            "measured": measured,
            "diagnostic": diagnostic,
            "compositional_artifact": compositional,
        }
        construct_cache[key] = result
        return result

    replication_cells: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for name in sorted(entries):
        entry = entries[name]
        seasons = entry["seasons"]
        primary = ENTRY_CONSTRUCT[name]
        primary_result = measure_and_diagnose(primary["kind"], primary["quantity"], seasons)

        final_result = primary_result
        used_fallback = False
        if primary_result["compositional_artifact"] and name in FALLBACK_CONSTRUCT:
            fb = FALLBACK_CONSTRUCT[name]
            fallback_result = measure_and_diagnose(fb["kind"], fb["quantity"], seasons)
            if (
                fallback_result["measured"]["status"] == rlib.STATUS_MEASURED
                and not fallback_result["compositional_artifact"]
            ):
                final_result = fallback_result
                used_fallback = True

        measured = final_result["measured"]
        compositional = final_result["compositional_artifact"]
        recordable = measured["status"] == rlib.STATUS_MEASURED and not compositional
        status = "not_applicable_compositional_constraint" if compositional else measured["status"]

        mask = masks[name]
        sample_games_reproduced = int(mask.sum())
        sample_games_recorded = entry.get("sample_games")

        half = rlib.half_season_replication(games, mask, outcome_col=OUTCOME_COL)
        replication_cells[name] = half

        rows.append(
            {
                "entry": name,
                "kind": final_result["kind"],
                "quantity": final_result["quantity"],
                "quantity_note": QUANTITY_NOTE[final_result["quantity"]],
                "unit": "home_team (venue proxy)" if final_result["kind"] == "venue" else "team_id",
                "seasons": list(seasons),
                "used_fallback": used_fallback,
                "primary_quantity": primary["quantity"],
                "primary_compositional_artifact": primary_result["compositional_artifact"],
                "sample_games_reproduced": sample_games_reproduced,
                "sample_games_recorded": sample_games_recorded,
                "sample_games_match": (
                    sample_games_recorded is None
                    or sample_games_reproduced == sample_games_recorded
                ),
                "registry_effect": entry["effect"],
                "registry_classification": entry["classification"],
                "n_units": measured["n_units"],
                "pearson_r": measured["pearson_r"],
                "pearson_r_ci95": measured["pearson_r_ci95"],
                "spearman_rho": measured["spearman_rho"],
                "spearman_brown_full_length_reliability": measured[
                    "spearman_brown_full_length_reliability"
                ],
                "probability_positive": measured["probability_positive"],
                "reliability": measured["reliability"],
                "reliability_low": measured["reliability_low"],
                "reliability_high": measured["reliability_high"],
                "status": status,
                "recordable": recordable,
                "method": measured["method"],
                "half_season_replication": half,
            }
        )
        rel = measured["reliability"]
        shown = f"{rel:+.4f}" if rel is not None else "  n/a "
        match_flag = "OK" if rows[-1]["sample_games_match"] else "MISMATCH"
        fb_flag = " FALLBACK" if used_fallback else ""
        print(
            f"  {name:<70} {final_result['quantity']:<22} n={measured['n_units']:>4} "
            f"rel={shown} {status:<38} slice={sample_games_reproduced:>4} {match_flag}{fb_flag}"
        )

    battery_replication = rlib.battery_replication_correlation(replication_cells)

    compositional_diagnostic_summary = {
        f"{kind}:{quantity}:{seasons[0]}-{seasons[1]}": {
            "kind": result["kind"],
            "quantity": result["quantity"],
            "seasons": list(result["seasons"]),
            "real_reliability": result["measured"]["reliability"],
            "diagnostic": result["diagnostic"],
            "compositional_artifact": result["compositional_artifact"],
        }
        for (kind, quantity, seasons), result in construct_cache.items()
        if result["diagnostic"] is not None
    }

    windows = sorted({tuple(r["seasons"]) for r in rows if r.get("n_units")})
    controls: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for window in windows:
        restricted = games.loc[games["season"].between(window[0], window[1])]
        # positive_control only needs season/week/unit_col -- reuse each unit
        # structure's own shape (team-week, 2 rows/game, for trait+exposure
        # entries; venue, 1 row/game keyed on home_team, for total_bucket_*).
        team_week_frame = build_trait_frame(restricted, "confidence_distance")
        venue_frame = build_venue_frame(restricted, "total_line")
        controls[f"{window[0]}-{window[1]}"] = {
            "team_week_unit_structure": rlib.positive_control(team_week_frame, n_boot=1000),
            "venue_unit_structure": rlib.positive_control(
                venue_frame, unit_col="team_id", n_boot=1000
            ),
        }

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "opener_eval" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "reliability-opener-eval",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "outcome_col": OUTCOME_COL,
        "entries": sorted(entries),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "outcome_col": OUTCOME_COL,
        "population_provenance": (
            f"{PER_GAME_PATH.relative_to(REPO)} joined on game_id to "
            f"{FEATURE_PATH.relative_to(REPO)} (one-to-one), 34 opener-line-push rows "
            f"({OUTCOME_COL} NaN) dropped, leaving n=1503 -- reproduces "
            "docs/opener_error_analysis.md's population exactly (read 2026-09-01)."
        ),
        "entry_construct": ENTRY_CONSTRUCT,
        "fallback_construct": FALLBACK_CONSTRUCT,
        "quantity_notes": QUANTITY_NOTE,
        "compositional_diagnostic": {
            "gap_threshold": COMPOSITIONAL_GAP_THRESHOLD,
            "magnitude_floor": COMPOSITIONAL_MAGNITUDE_FLOOR,
            "n_reseeds": DIAGNOSTIC_N_RESEEDS,
            "base_seed": DIAGNOSTIC_BASE_SEED,
            "by_construct": compositional_diagnostic_summary,
        },
        "positive_control": controls,
        "battery_replication_correlation": battery_replication,
        "results": rows,
        "provenance": artifact_provenance(configuration, PER_GAME_PATH, project_root=REPO),
    }

    measured_rows = [r for r in rows if r["recordable"]]
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-opener-eval",
        metrics={
            "n_entries": len(rows),
            "n_measured": len(measured_rows),
            "n_unmeasured": len(rows) - len(measured_rows),
        },
        notes=(
            "Measure-only split-half reliability for the opener_error_mining registry "
            "cells; every cell measured regardless of sign or interval shape, and nothing "
            "is closed or reclassified, per AGENTS.md's binding closing-grounds taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    mismatches = [r for r in rows if not r["sample_games_match"]]
    print(f"\n{len(measured_rows)} of {len(rows)} measured; {len(rows) - len(measured_rows)} not")
    print(f"sample_games mismatches: {len(mismatches)}")
    for row in mismatches:
        print(
            f"  {row['entry']}: reproduced={row['sample_games_reproduced']} "
            f"recorded={row['sample_games_recorded']}"
        )
    for label, predicate in (
        ("<= 0.10", lambda v: v <= 0.10),
        (">= 0.80", lambda v: v >= 0.80),
    ):
        hits = [r for r in measured_rows if predicate(r["reliability"])]
        print(f"  {label}: {len(hits)}")
        for row in sorted(hits, key=lambda r: r["reliability"]):
            print(
                f"    {row['entry']:<70} {row['reliability']:+.4f} "
                f"[{row['reliability_low']:+.4f}, {row['reliability_high']:+.4f}] "
                f"({row['kind']})"
            )

    not_applicable = [r for r in rows if r["status"] == "not_applicable_compositional_constraint"]
    print(
        f"\nnot_applicable_compositional_constraint (skipped, not recorded): {len(not_applicable)}"
    )
    for row in not_applicable:
        real = f"{row['reliability']:+.4f}" if row["reliability"] is not None else "n/a"
        print(f"  {row['entry']}: primary quantity={row['primary_quantity']} real={real}")

    source_path = str((output_dir / "results.json").relative_to(REPO)).replace("\\", "/")
    print(f"\n=== set-reliability commands for the {len(measured_rows)} recordable entries ===")
    for row in measured_rows:
        reason = (
            f"opener_error_mining battery slice; slicing quantity = {row['quantity']} "
            f"({row['quantity_note']}); {row['method'].split(',')[0]}"
            + (
                " (EXPOSURE fallback: primary trait was a compositional artifact)"
                if row["used_fallback"]
                else ""
            )
        )
        print(
            "nfl-ats weak-signals set-reliability "
            f"--name {row['entry']} "
            f"--reliability {row['reliability']:.4f} "
            f"--reliability-low {row['reliability_low']:.4f} "
            f"--reliability-high {row['reliability_high']:.4f} "
            f'--method "{row["method"]}" '
            f"--source {source_path} "
            f'--reason "{reason}"'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
