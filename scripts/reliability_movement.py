"""Split-half reliability for the 26 line-movement registry cells (ORCH-D).

**What these cells are, and which builder made each one.** Four screens
produced them, and every construct below is imported from the screen that
built it rather than re-derived here:

``observed_movement_*`` (6 cells)
    ``scripts/observed_movement_channel.py``. Arms 1-3 flip the production
    pick (``pick_home_at_open_probability_rule``) toward the side the market
    moved to, either unconditionally (oracle) or only when the move clears
    0.5 / 1.0 points. The thresholded quantity is the signed HOME-side spread
    move: ``open_move = close_home_spread - tue_open_home_spread`` (read:
    ``src/nfl_ats/clv.py:2151``) for the 2020-2025 arms, and
    ``sunday_open_move = sunday_home_spread - tue_open_home_spread`` (read:
    ``scripts/observed_movement_channel.py:554-556``) for the 2023-2025
    ``*_sunday_am_realism`` arms, where ``sunday_home_spread`` is the last
    ``intraday_hourly`` capture at or before ``min(kickoff, Sunday 16:00
    ET)`` (read: ``scripts/observed_movement_channel.py:338-394``). A
    different timestamp pair is a DIFFERENT construct and is measured
    separately.

``movement_expansion_*`` (5 cells)
    ``scripts/movement_expansion_battery.py``, on the ``movement_expansion_v1``
    rotation window (2020-2021). Same overlay construction, three checkpoint
    windows: Tuesday open -> close, -> ``thu_pre_tnf``, -> ``sat_midday``
    (read: ``scripts/movement_expansion_battery.py:120-137`` for
    ``oracle_pick``/``threshold_pick``, which form ``move = cur - tue_open``
    inline, and ``:299-343`` for the checkpoint columns).

``movement_rule_composed_chain`` (1 cell)
    ``scripts/movement_composition_eval.py``: the same >= 1.0-point
    close-vs-Tuesday rule applied on top of the composed production chain
    (read: ``:296`` for ``open_move_reloaded``, verified equal to the
    archive's ``open_move`` inside that script at ``:297``; ``:315`` for the
    overlay; ``:323`` for ``movement_eligible``). Its construct is therefore
    the SAME quantity and window as ``observed_movement_threshold_1_0`` on
    the same 1,503 games, and its measured reliability is identical by
    construction -- stated, not hidden.

``movement_attribution_*`` (14 cells)
    ``scripts/movement_attribution.py``. These do not threshold a new
    quantity: they take the games where the market's side disagrees with the
    production pick (``load_population``, read: ``:187-211``), optionally
    restrict to ``|open_move| >= 1.0`` (``:487``), and cut that population by
    an attribution CLASS -- INJURY (``:339``), WEATHER (``:362``), PUBLIC and
    its ``book_shading_public`` / ``reverse_line_movement`` subtypes
    (``:396``), ATTRIBUTED_ANY / UNATTRIBUTED (``:484-485``). A class is a
    per-game flag with no continuous parent trait, so those cells get
    ``METHOD_EXPOSURE`` on the flag's per-team-season exposure rate, measured
    over the full 1,503-game non-push slate the parent family was scored on.

**Method.** ``scripts/reliability_lib.py``, imported, never reimplemented:
unit = team-season, halves = odd/even weeks, Spearman-Brown corrected, block
bootstrap over team-seasons, seed 20260901, 4,000 draws, restricted to each
registry cell's OWN seasons. For the movement quantities the long frame puts
each game in twice, carrying the move SIGNED TOWARD that side (+move for the
home row, -move for the away row -- the sign convention
``observed_movement_channel._oracle_pick`` reads at ``:132-138``, where a
positive move means the market moved onto the home side). The absolute
magnitude, which is what the thresholds actually compare against, is measured
alongside and reported as a secondary read.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. Only
two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED
wrong sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an effect
that size. Everything else is ``unresolved_below_power``; report
``probability_positive``, never "contains zero". This script CLOSES NOTHING
and RECLASSIFIES NOTHING: it measures, and a low number is a candidate for the
reliability ground, never the closure itself. A low ``METHOD_EXPOSURE`` value
is not even a candidate -- a schedule quirk with no stable team structure can
still move covers. Within-week correlation is ZERO.

**Oracle cells are ceilings by construction.** ``observed_movement_oracle_*``
and ``movement_expansion_thu_oracle_full_slate`` follow the market's realized
side on every game, and the close prints after the true per-game pick deadline
``min(kickoff, Sunday 16:00 ET)`` for SNF/MNF/late-Sunday games. Their EFFECT
numbers are upper bounds on what any playable rule could reach, so no
reliability read from them is evidence about a playable rule. Their INPUT --
the movement quantity itself -- is still a real trait and is measured here.

**Never write an unmeasurable reliability as a number.** Too few usable units
is reported as UNMEASURED, never as reliability 0.

**One further hazard, pre-stated before any number was read.** A near-constant
column (a flag almost no team-season ever carries) can return a large
|correlation| of either sign that flips with the season window; that is an
artifact of a handful of non-constant units, not a trait. Any measurement whose
usable units include fewer than ``reliability_lib.MIN_UNITS`` units with a
non-constant value, or whose cross-unit half-mean spread is numerically zero,
is reported as ``not_informative_near_constant`` and is NOT recorded.

Writes ``artifacts/reliability_sweep/movement/<stamp>/results.json`` and prints
the ``set-reliability`` commands it would run (recording itself goes through
the locked CLI, never from inside this script).
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

import movement_attribution as attribution  # noqa: E402
import movement_expansion_battery as expansion  # noqa: E402
import observed_movement_channel as channel  # noqa: E402
import reliability_lib as rlib  # noqa: E402

from nfl_ats.clv import pick_correct  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

# --- Persisted per-game outputs of the four builders (their own artifacts) ---
ANCHOR_TUE_CLOSE = (
    REPO / "artifacts/observed_movement_channel/20260820T093426Z/per_game_tue_close.parquet"
)
ANCHOR_SUNDAY_1600 = (
    REPO
    / "artifacts/observed_movement_channel/20260820T093426Z/per_game_sunday_1600_realism.parquet"
)
EXPANSION_PER_GAME = REPO / "artifacts/movement_expansion_battery/20260831T173822Z/per_game.parquet"
COMPOSITION_PER_GAME = (
    REPO / "artifacts/movement_composition_eval/20260822T144746Z/per_game.parquet"
)
GAME_FEATURES = REPO / "data/processed/game_features.parquet"

PRODUCTION_PICK = "pick_home_at_open_probability_rule"
PRODUCTION_CORRECT = "correct_at_open_probability_rule"

SIGNED_METRIC = "move_toward_team"
ABS_METRIC = "abs_move"


# ---------------------------------------------------------------------------
# Movement quantity + team-week frames
# ---------------------------------------------------------------------------


def checkpoint_move(current_home_spread: pd.Series, tue_open_home_spread: pd.Series) -> pd.Series:
    """Signed HOME-side spread move between the Tuesday opener and a checkpoint.

    This is the one place the difference is formed, and
    ``tests/test_reliability_movement.py`` proves that feeding it to
    ``observed_movement_channel._threshold_pick`` /``_oracle_pick`` reproduces
    ``movement_expansion_battery.threshold_pick`` / ``oracle_pick`` exactly,
    so it is the screens' own quantity rather than a look-alike.
    """

    current = pd.to_numeric(current_home_spread, errors="coerce")
    opener = pd.to_numeric(tue_open_home_spread, errors="coerce")
    return current - opener


def verify_move_reproduces_screen_picks(
    frame: pd.DataFrame, *, current_column: str, threshold: float, persisted_pick_column: str
) -> dict[str, Any]:
    """Runtime guard: the move formed here IS the one the screens threshold.

    Two independent checks on the real population, not a fixture: feeding
    :func:`checkpoint_move`'s output to
    ``observed_movement_channel._threshold_pick`` must reproduce (a)
    ``movement_expansion_battery.threshold_pick``, which forms the difference
    inline from the same two spread columns, and (b) the pick column the
    battery already persisted into its own artifact. Either mismatch would mean
    this sweep measured a look-alike quantity rather than the cell's own.
    """

    production = frame[PRODUCTION_PICK].astype(bool)
    move = checkpoint_move(frame[current_column], frame["tue_open_home_spread"])
    mine, _eligible = channel._threshold_pick(move, production, threshold)
    theirs = expansion.threshold_pick(
        frame[current_column], frame["tue_open_home_spread"], production, threshold
    )
    persisted = frame[persisted_pick_column].astype(bool)
    return {
        "current_column": current_column,
        "threshold": threshold,
        "matches_expansion_builder": bool(mine.astype(bool).equals(theirs.astype(bool))),
        "matches_persisted_artifact_pick": bool(mine.astype(bool).equals(persisted)),
        "n_games": len(frame),
    }


def team_week_movement(games: pd.DataFrame, move: pd.Series) -> pd.DataFrame:
    """Two rows per game carrying the move SIGNED TOWARD that side.

    ``observed_movement_channel._oracle_pick`` (read: ``:132-138``) reads a
    POSITIVE move as "the market moved onto the home side", so the home row
    carries ``+move`` and the away row ``-move``. ``abs_move`` -- the quantity
    the thresholds actually compare against -- is identical on both rows and
    is carried alongside as the secondary read.
    """

    values = pd.to_numeric(move, errors="coerce").to_numpy(dtype=float)
    pieces = []
    for column, sign in (("home_team", 1.0), ("away_team", -1.0)):
        piece = games.loc[:, ["season", "week", column]].rename(columns={column: "team_id"}).copy()
        piece[SIGNED_METRIC] = sign * values
        piece[ABS_METRIC] = np.abs(values)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def attach_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Join home/away team on ``game_id``; the per-game artifacts carry neither."""

    identity = pd.read_parquet(GAME_FEATURES)[["game_id", "home_team", "away_team"]]
    merged = frame.merge(identity, on="game_id", how="left", validate="one_to_one")
    missing = int(merged["home_team"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} games failed to join game_features on game_id")
    return merged


def non_push(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop pushes at the frozen Tuesday grading line, as every cell's scorer does."""

    return frame.loc[frame[PRODUCTION_CORRECT].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Near-constant diagnostics (pre-stated; see the module docstring)
# ---------------------------------------------------------------------------


def constancy_diagnostics(
    long: pd.DataFrame, metric: str, seasons: tuple[int, int]
) -> dict[str, Any]:
    """Cross-unit spread of the two half-means, and how many units vary at all."""

    frame = long.loc[:, ["team_id", "season", "week", metric]].dropna(subset=[metric]).copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["season"])
    frame["season"] = frame["season"].astype(int)
    frame = frame.loc[frame["season"].between(seasons[0], seasons[1])]
    if frame.empty:
        return {"n_usable_units": 0, "n_units_non_constant": 0}
    frame["half"] = np.where(frame["week"] % 2 == 0, "even", "odd")
    means = frame.groupby(["team_id", "season", "half"])[metric].mean().unstack("half")
    counts = frame.groupby(["team_id", "season", "half"]).size().unstack("half")
    usable = (counts.get("odd", pd.Series(dtype=float)).fillna(0) >= 2) & (
        counts.get("even", pd.Series(dtype=float)).fillna(0) >= 2
    )
    means = means.dropna()
    usable = usable.reindex(means.index).fillna(False)
    means = means.loc[usable]
    if means.empty:
        return {"n_usable_units": 0, "n_units_non_constant": 0}
    baseline = float(np.nanmin(means.to_numpy(dtype=float)))
    non_constant = int(((means["odd"] != baseline) | (means["even"] != baseline)).sum())
    return {
        "n_usable_units": len(means),
        "n_units_non_constant": non_constant,
        "odd_half_mean_std": float(means["odd"].std(ddof=1)),
        "even_half_mean_std": float(means["even"].std(ddof=1)),
        "odd_half_mean_distinct_values": int(means["odd"].nunique()),
        "even_half_mean_distinct_values": int(means["even"].nunique()),
    }


def near_constant(diagnostics: dict[str, Any]) -> bool:
    """The pre-stated artifact test: too few varying units, or no spread at all."""

    if diagnostics.get("n_units_non_constant", 0) < rlib.MIN_UNITS:
        return True
    spreads = [
        diagnostics.get("odd_half_mean_std", 0.0),
        diagnostics.get("even_half_mean_std", 0.0),
    ]
    return any((not np.isfinite(s)) or s <= 0.0 for s in spreads)


# ---------------------------------------------------------------------------
# Populations, each built from its own screen's persisted per-game artifact
# ---------------------------------------------------------------------------


def build_populations() -> dict[str, dict[str, Any]]:
    """Every population a cell in this group was scored on, plus its paired delta.

    ``paired_delta`` is per game: the candidate pick's correctness minus the
    incumbent's on the same game, ``NaN`` on a push. It is exactly 0 on any
    game the candidate does not change, so ``half_season_replication``'s
    flagged-minus-complement gap is the flip's own value inside the flagged
    subset. Every candidate pick comes from the screen's own builder.
    """

    populations: dict[str, dict[str, Any]] = {}

    # --- Tuesday open -> close, full 2020-2025 slate -----------------------
    close = non_push(attach_identity(pd.read_parquet(ANCHOR_TUE_CLOSE)))
    production_correct = close[PRODUCTION_CORRECT].astype(float)
    close["delta_oracle"] = (
        pick_correct(close["oracle_pick_home"].astype(bool), close["margin_vs_open"]).astype(float)
        - production_correct
    )
    for threshold, column in ((0.5, "_pick_threshold_0_5"), (1.0, "_pick_threshold_1_0")):
        label = f"delta_threshold_{str(threshold).replace('.', '_')}"
        close[label] = (
            pick_correct(close[column].astype(bool), close["margin_vs_open"]).astype(float)
            - production_correct
        )
        close[f"eligible_{str(threshold).replace('.', '_')}"] = (
            close["open_move"].abs().ge(threshold)
        )
    close["eligible_nonzero"] = close["open_move"].ne(0.0)
    populations["tue_close"] = {
        "games": close,
        "move": close["open_move"],
        "window": "Tuesday opener -> close (open_move = close_home_spread - tue_open_home_spread)",
        "long": team_week_movement(close, close["open_move"]),
        "builder_agreement": [
            verify_move_reproduces_screen_picks(
                close,
                current_column="close_home_spread",
                threshold=threshold,
                persisted_pick_column=column,
            )
            for threshold, column in (
                (0.5, "_pick_threshold_0_5"),
                (1.0, "_pick_threshold_1_0"),
            )
        ],
    }

    # --- Tuesday open -> last capture before min(kickoff, Sunday 16:00 ET) --
    sunday = non_push(attach_identity(pd.read_parquet(ANCHOR_SUNDAY_1600)))
    sunday_production = sunday[PRODUCTION_CORRECT].astype(float)
    sunday["delta_oracle"] = (
        pick_correct(
            sunday["oracle_pick_home_sunday"].astype(bool), sunday["margin_vs_open"]
        ).astype(float)
        - sunday_production
    )
    for threshold in (0.5, 1.0):
        tag = str(threshold).replace(".", "_")
        column = f"_pick_threshold_{tag}_sunday_1600_realism"
        sunday[f"delta_threshold_{tag}"] = (
            pick_correct(sunday[column].astype(bool), sunday["margin_vs_open"]).astype(float)
            - sunday_production
        )
        sunday[f"eligible_{tag}"] = sunday["sunday_open_move"].abs().ge(threshold)
    sunday["eligible_nonzero"] = sunday["sunday_open_move"].ne(0.0)
    populations["sunday_1600"] = {
        "games": sunday,
        "move": sunday["sunday_open_move"],
        "window": (
            "Tuesday opener -> last intraday_hourly capture at or before "
            "min(kickoff, Sunday 16:00 ET) (sunday_open_move)"
        ),
        "long": team_week_movement(sunday, sunday["sunday_open_move"]),
    }

    # --- movement_expansion_v1 window (2020-2021), three checkpoints -------
    battery = non_push(attach_identity(pd.read_parquet(EXPANSION_PER_GAME)))
    battery_production = battery[PRODUCTION_CORRECT].astype(float)
    for tag, column in (
        ("close_1_0", "_pick_close_thr_1_0"),
        ("close_2_0", "_pick_close_thr_2_0"),
    ):
        battery[f"delta_{tag}"] = (
            pick_correct(battery[column].astype(bool), battery["margin_vs_open"]).astype(float)
            - battery_production
        )
    close_move = checkpoint_move(battery["close_home_spread"], battery["tue_open_home_spread"])
    battery["eligible_1_0"] = close_move.abs().ge(1.0)
    battery["eligible_2_0"] = close_move.abs().ge(2.0)
    populations["expansion_close"] = {
        "games": battery,
        "move": close_move,
        "window": "Tuesday opener -> close, movement_expansion_v1 window",
        "long": team_week_movement(battery, close_move),
        "builder_agreement": [
            verify_move_reproduces_screen_picks(
                battery,
                current_column="close_home_spread",
                threshold=threshold,
                persisted_pick_column=column,
            )
            for threshold, column in ((1.0, "_pick_close_thr_1_0"), (2.0, "_pick_close_thr_2_0"))
        ],
    }

    for tag, spread_column, checkpoint in (
        ("thu", "thu_pre_tnf_home_spread", "Thursday pre-TNF"),
        ("sat", "sat_midday_home_spread", "Saturday midday"),
    ):
        frame = battery.loc[battery[spread_column].notna()].reset_index(drop=True)
        move = checkpoint_move(frame[spread_column], frame["tue_open_home_spread"])
        production_home = frame[PRODUCTION_PICK]
        incumbent = frame[PRODUCTION_CORRECT].astype(float)
        oracle = expansion.oracle_pick(
            frame[spread_column], frame["tue_open_home_spread"], production_home
        )
        thresholded = expansion.threshold_pick(
            frame[spread_column], frame["tue_open_home_spread"], production_home, 1.0
        )
        frame["delta_oracle"] = (
            pick_correct(oracle.astype(bool), frame["margin_vs_open"]).astype(float) - incumbent
        )
        frame["delta_threshold_1_0"] = (
            pick_correct(thresholded.astype(bool), frame["margin_vs_open"]).astype(float)
            - incumbent
        )
        frame["eligible_1_0"] = move.abs().ge(1.0)
        frame["eligible_nonzero"] = move.ne(0.0)
        populations[f"expansion_{tag}"] = {
            "games": frame,
            "move": move,
            "window": f"Tuesday opener -> {checkpoint} checkpoint, movement_expansion_v1 window",
            "long": team_week_movement(frame, move),
        }

    # --- composed chain (2020-2025), same window as threshold_1_0 ----------
    composed = non_push(attach_identity(pd.read_parquet(COMPOSITION_PER_GAME)))
    composed["delta_composed"] = composed["correct_b"].astype(float) - composed["correct_a"].astype(
        float
    )
    populations["composed_chain"] = {
        "games": composed,
        "move": composed["open_move_reloaded"],
        "window": (
            "Tuesday opener -> close (open_move_reloaded, verified equal to the archive's "
            "open_move inside movement_composition_eval.py:297)"
        ),
        "long": team_week_movement(composed, composed["open_move_reloaded"]),
    }

    return populations


def build_attribution_flags(slate: pd.DataFrame) -> dict[str, pd.Series]:
    """Every attribution class as a boolean over the full non-push slate.

    Runs ``movement_attribution``'s own flag builders (``load_population`` ->
    ``attach_injury_flag`` -> ``attach_weather_flag`` -> ``attach_public_flag``)
    and then expresses each cell as a per-game flag on the 1,503-game slate the
    parent ``observed_movement_*`` family was scored on: True exactly on the
    games that cell scored, False elsewhere. That denominator is what makes the
    exposure rate a per-team-season quantity at all -- inside the ~494-game
    attribution population a team-season carries only a handful of games.
    """

    population = attribution.load_population()
    population = attribution.attach_injury_flag(population)
    population = attribution.attach_weather_flag(population)
    population = attribution.attach_public_flag(population)
    population["ATTRIBUTED"] = population["INJURY"] | population["WEATHER"] | population["PUBLIC"]
    population["UNATTRIBUTED"] = ~population["ATTRIBUTED"]
    threshold = population["open_move"].abs().ge(1.0)

    classes: dict[str, pd.Series] = {}
    for population_name, subset in (
        ("pop_unfiltered", population),
        ("pop_threshold", population.loc[threshold]),
    ):
        named = {
            "injury": subset["INJURY"].astype(bool),
            "weather": subset["WEATHER"].astype(bool),
            "public": subset["PUBLIC"].astype(bool),
            "attributed_any": subset["ATTRIBUTED"].astype(bool),
            "unattributed": subset["UNATTRIBUTED"].astype(bool),
            "public_book_shading_public": subset["public_subtype"].eq("book_shading_public"),
            "public_reverse_line_movement": subset["public_subtype"].eq("reverse_line_movement"),
        }
        for class_name, mask in named.items():
            identifiers = set(subset.loc[mask, "game_id"])
            classes[f"{population_name}_{class_name}"] = slate["game_id"].isin(identifiers)
    return classes


# ---------------------------------------------------------------------------
# Entry -> construct specification
# ---------------------------------------------------------------------------

TRAIT_ENTRIES: tuple[dict[str, str], ...] = (
    {
        "entry": "observed_movement_oracle_full_slate",
        "population": "tue_close",
        "delta": "delta_oracle",
        "flag": "eligible_nonzero",
        "battery": "observed_movement",
        "note": "oracle ceiling",
    },
    {
        "entry": "observed_movement_threshold_0_5",
        "population": "tue_close",
        "delta": "delta_threshold_0_5",
        "flag": "eligible_0_5",
        "battery": "observed_movement",
        "note": "",
    },
    {
        "entry": "observed_movement_threshold_1_0",
        "population": "tue_close",
        "delta": "delta_threshold_1_0",
        "flag": "eligible_1_0",
        "battery": "observed_movement",
        "note": "",
    },
    {
        "entry": "observed_movement_oracle_sunday_am_realism",
        "population": "sunday_1600",
        "delta": "delta_oracle",
        "flag": "eligible_nonzero",
        "battery": "observed_movement",
        "note": "oracle ceiling",
    },
    {
        "entry": "observed_movement_threshold_0_5_sunday_am_realism",
        "population": "sunday_1600",
        "delta": "delta_threshold_0_5",
        "flag": "eligible_0_5",
        "battery": "observed_movement",
        "note": "",
    },
    {
        "entry": "observed_movement_threshold_1_0_sunday_am_realism",
        "population": "sunday_1600",
        "delta": "delta_threshold_1_0",
        "flag": "eligible_1_0",
        "battery": "observed_movement",
        "note": "",
    },
    {
        "entry": "movement_expansion_window_close_threshold_1_0",
        "population": "expansion_close",
        "delta": "delta_close_1_0",
        "flag": "eligible_1_0",
        "battery": "movement_expansion",
        "note": "",
    },
    {
        "entry": "movement_expansion_close_threshold_2_0",
        "population": "expansion_close",
        "delta": "delta_close_2_0",
        "flag": "eligible_2_0",
        "battery": "movement_expansion",
        "note": "",
    },
    {
        "entry": "movement_expansion_thu_oracle_full_slate",
        "population": "expansion_thu",
        "delta": "delta_oracle",
        "flag": "eligible_nonzero",
        "battery": "movement_expansion",
        "note": "oracle ceiling",
    },
    {
        "entry": "movement_expansion_thu_threshold_1_0",
        "population": "expansion_thu",
        "delta": "delta_threshold_1_0",
        "flag": "eligible_1_0",
        "battery": "movement_expansion",
        "note": "",
    },
    {
        "entry": "movement_expansion_sat_threshold_1_0",
        "population": "expansion_sat",
        "delta": "delta_threshold_1_0",
        "flag": "eligible_1_0",
        "battery": "movement_expansion",
        "note": "",
    },
    {
        "entry": "movement_rule_composed_chain",
        "population": "composed_chain",
        "delta": "delta_composed",
        "flag": "movement_eligible",
        "battery": "movement_rule",
        "note": "",
    },
)

ATTRIBUTION_CLASSES: tuple[str, ...] = (
    "injury",
    "weather",
    "public",
    "attributed_any",
    "unattributed",
    "public_book_shading_public",
    "public_reverse_line_movement",
)

BUILDER_PROVENANCE: dict[str, str] = {
    "observed_movement": (
        "scripts/observed_movement_channel.py:132-149 (_oracle_pick/_threshold_pick) thresholds "
        "open_move; src/nfl_ats/clv.py:2151 defines open_move = close_home_spread - "
        "tue_open_home_spread; scripts/observed_movement_channel.py:554-556 defines "
        "sunday_open_move and :338-394 the min(kickoff, Sunday 16:00 ET) capture rule. "
        "Read 2026-09-01."
    ),
    "movement_expansion": (
        "scripts/movement_expansion_battery.py:120-137 (oracle_pick/threshold_pick) forms "
        "move = cur - tue_open; :299-343 attaches the thu_pre_tnf / sat_midday checkpoint "
        "spreads and builds each cell's pick. Read 2026-09-01."
    ),
    "movement_rule": (
        "scripts/movement_composition_eval.py:296 (open_move_reloaded), :192-198 "
        "(movement_overlay), :315 (arm b = chain + movement rule), :323 (movement_eligible). "
        "Read 2026-09-01."
    ),
    "movement_attribution": (
        "scripts/movement_attribution.py:187-211 (load_population: the disagreement games), "
        ":339 attach_injury_flag, :362 attach_weather_flag, :396 attach_public_flag, "
        ":484-487 ATTRIBUTED/UNATTRIBUTED and the |open_move|>=1.0 POP_THRESHOLD cut, "
        ":544-556 the per-class cell loop. Read 2026-09-01."
    ),
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def registry_seasons() -> dict[str, tuple[int, int]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, tuple[int, int]] = {}
    for name, signal in registry.signals.items():
        out[name] = (int(signal.seasons[0]), int(signal.seasons[1]))
    return out


def measurement_row(
    *,
    entry: str,
    long: pd.DataFrame,
    metric: str,
    method: str,
    quantity: str,
    window: str,
    seasons: tuple[int, int],
    n_boot: int,
) -> dict[str, Any]:
    measured = rlib.measure_reliability(long, metric, method=method, seasons=seasons, n_boot=n_boot)
    diagnostics = constancy_diagnostics(long, metric, seasons)
    status = measured["status"]
    if status == rlib.STATUS_MEASURED and near_constant(diagnostics):
        status = "not_informative_near_constant"
    return {
        "entry": entry,
        "movement_quantity": quantity,
        "timestamp_window": window,
        "unit": "team-season",
        "seasons": [seasons[0], seasons[1]],
        "n_units": measured["n_units"],
        "pearson_r": measured["pearson_r"],
        "pearson_r_ci95": measured["pearson_r_ci95"],
        "spearman_rho": measured["spearman_rho"],
        "spearman_brown_full_length_reliability": measured[
            "spearman_brown_full_length_reliability"
        ],
        "probability_positive": measured["probability_positive"],
        "reliability": measured["reliability"] if status == rlib.STATUS_MEASURED else None,
        "reliability_low": measured["reliability_low"] if status == rlib.STATUS_MEASURED else None,
        "reliability_high": (
            measured["reliability_high"] if status == rlib.STATUS_MEASURED else None
        ),
        "status": status,
        "method": measured["method"],
        "constancy_diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    parser.add_argument("--control-n-boot", type=int, default=1000)
    args = parser.parse_args()

    started = time.time()
    seasons_by_entry = registry_seasons()
    populations = build_populations()
    for key, value in populations.items():
        print(f"population {key:<18} games={len(value['games']):>5}  {value['window']}")

    rows: list[dict[str, Any]] = []
    replications: dict[str, dict[str, Any]] = {}
    secondary: list[dict[str, Any]] = []

    # --- the twelve trait cells -------------------------------------------
    for spec in TRAIT_ENTRIES:
        entry = spec["entry"]
        seasons = seasons_by_entry[entry]
        population = populations[spec["population"]]
        long = population["long"]
        row = measurement_row(
            entry=entry,
            long=long,
            metric=SIGNED_METRIC,
            method=rlib.METHOD_TRAIT,
            quantity="signed spread move toward the team's own side",
            window=population["window"],
            seasons=seasons,
            n_boot=args.n_boot,
        )
        row["battery"] = spec["battery"]
        row["method_tag"] = "METHOD_TRAIT"
        row["builder_provenance"] = BUILDER_PROVENANCE[spec["battery"]]
        row["oracle_ceiling"] = spec["note"] == "oracle ceiling"
        rows.append(row)
        secondary.append(
            measurement_row(
                entry=entry,
                long=long,
                metric=ABS_METRIC,
                method=rlib.METHOD_TRAIT,
                quantity="absolute spread move magnitude (the thresholded quantity)",
                window=population["window"],
                seasons=seasons,
                n_boot=args.n_boot,
            )
        )
        games = population["games"]
        replication = rlib.half_season_replication(
            games, games[spec["flag"]].astype(bool), outcome_col=spec["delta"]
        )
        replication["outcome"] = (
            f"per-game paired delta ({spec['delta']}): candidate correctness minus the "
            "incumbent's on the same game, exactly 0 where the candidate changes nothing"
        )
        replication["flag"] = spec["flag"]
        replications[entry] = replication
        row["half_season_replication"] = replication
        shown = "  n/a  " if row["reliability"] is None else f"{row['reliability']:+.4f}"
        print(f"  {entry:<52} n={row['n_units']:>4} rel={shown} {row['status']}")

    # --- the fourteen attribution cells ------------------------------------
    slate = populations["tue_close"]["games"]
    flags = build_attribution_flags(slate)
    for population_name in ("pop_unfiltered", "pop_threshold"):
        for class_name in ATTRIBUTION_CLASSES:
            entry = f"movement_attribution_{population_name}_{class_name}"
            seasons = seasons_by_entry[entry]
            flag = flags[f"{population_name}_{class_name}"]
            long = rlib.game_flag_to_team_week(slate, flag)
            row = measurement_row(
                entry=entry,
                long=long,
                metric="exposure",
                method=rlib.METHOD_EXPOSURE,
                quantity=(
                    f"per-team-season EXPOSURE RATE of the {population_name.upper()} / "
                    f"{class_name.upper()} attribution class over the 2020-2025 non-push slate"
                ),
                window=(
                    "Tuesday opener -> close (the parent open_move the population is cut from); "
                    "the class itself is a per-game flag with no continuous parent trait"
                ),
                seasons=seasons,
                n_boot=args.n_boot,
            )
            row["battery"] = "movement_attribution"
            row["method_tag"] = "METHOD_EXPOSURE"
            row["builder_provenance"] = BUILDER_PROVENANCE["movement_attribution"]
            row["oracle_ceiling"] = False
            row["n_flagged_games"] = int(flag.sum())
            replication = rlib.half_season_replication(slate, flag, outcome_col="delta_oracle")
            replication["outcome"] = (
                "per-game paired delta (delta_oracle): the market-side flip's correctness minus "
                "the production pick's, exactly 0 outside the attribution population"
            )
            replication["flag"] = f"{population_name}_{class_name}"
            replications[entry] = replication
            row["half_season_replication"] = replication
            rows.append(row)
            shown = "  n/a  " if row["reliability"] is None else f"{row['reliability']:+.4f}"
            print(
                f"  {entry:<52} n={row['n_units']:>4} rel={shown} {row['status']} "
                f"(flagged games {row['n_flagged_games']})"
            )

    # --- positive controls, one per distinct unit structure ----------------
    controls: dict[str, Any] = {}
    control_frames = {
        "tue_close_2020_2025": (populations["tue_close"]["long"], (2020, 2025)),
        "sunday_1600_2023_2025": (populations["sunday_1600"]["long"], (2023, 2025)),
        "expansion_close_2020_2021": (populations["expansion_close"]["long"], (2020, 2021)),
        "expansion_thu_2020_2021": (populations["expansion_thu"]["long"], (2020, 2021)),
        "expansion_sat_2020_2021": (populations["expansion_sat"]["long"], (2020, 2021)),
    }
    for label, (long, window) in control_frames.items():
        restricted = long.loc[long["season"].between(window[0], window[1])]
        controls[label] = rlib.positive_control(restricted, n_boot=args.control_n_boot)
        print(f"positive control {label}: {controls[label]}")

    # --- battery-level replication correlations ----------------------------
    battery_correlations: dict[str, Any] = {}
    for battery in (
        "observed_movement",
        "movement_expansion",
        "movement_attribution",
        "movement_rule",
    ):
        members = {
            row["entry"]: replications[row["entry"]] for row in rows if row["battery"] == battery
        }
        battery_correlations[battery] = rlib.battery_replication_correlation(members)

    # --- artifact ----------------------------------------------------------
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "movement" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "reliability-movement",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "entries": sorted(row["entry"] for row in rows),
        "sources": {
            "tue_close": str(ANCHOR_TUE_CLOSE.relative_to(REPO)),
            "sunday_1600": str(ANCHOR_SUNDAY_1600.relative_to(REPO)),
            "expansion": str(EXPANSION_PER_GAME.relative_to(REPO)),
            "composition": str(COMPOSITION_PER_GAME.relative_to(REPO)),
        },
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "builder_provenance": BUILDER_PROVENANCE,
        "builder_agreement_checks": {
            key: value["builder_agreement"]
            for key, value in populations.items()
            if "builder_agreement" in value
        },
        "near_constant_rule": (
            "Pre-stated before any number was read: a measurement whose usable units include "
            f"fewer than {rlib.MIN_UNITS} units with a non-constant half-mean, or whose "
            "cross-unit half-mean spread is zero, is reported as not_informative_near_constant "
            "and NOT recorded -- a near-constant column can return a large |correlation| of "
            "either sign that flips with the season window."
        ),
        "oracle_ceiling_note": (
            "observed_movement_oracle_* and movement_expansion_thu_oracle_full_slate follow the "
            "market's realized side on every game and the close prints after the true per-game "
            "pick deadline min(kickoff, Sunday 16:00 ET) for SNF/MNF/late-Sunday games. Their "
            "EFFECT numbers are ceilings by construction, so no reliability read from them is "
            "evidence about a playable rule; their movement INPUT is still a real trait."
        ),
        "exposure_note": (
            "METHOD_EXPOSURE measures a flag's per-team-season exposure rate, not the trait "
            "reliability NO_SPLIT_HALF_RELIABILITY_MAX was calibrated against. A low value there "
            "is NOT an admissible no_split_half_reliability ground."
        ),
        "positive_control": controls,
        "battery_replication_correlation": battery_correlations,
        "results": rows,
        "secondary_absolute_magnitude_results": secondary,
        "provenance": artifact_provenance(configuration, GAME_FEATURES, project_root=REPO),
    }
    measured_rows = [row for row in rows if row["status"] == rlib.STATUS_MEASURED]
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-movement",
        metrics={
            "n_entries": len(rows),
            "n_measured": len(measured_rows),
            "n_unmeasured": len(rows) - len(measured_rows),
        },
        notes=(
            "Measure-only split-half reliability for the 26 line-movement registry cells; every "
            "cell measured regardless of sign or interval shape, and nothing is closed or "
            "reclassified, per AGENTS.md's binding closing-grounds taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    print(f"\n{len(measured_rows)} of {len(rows)} measured; {len(rows) - len(measured_rows)} not")
    for label, predicate in (
        ("<= 0.10", lambda value: value <= 0.10),
        (">= 0.80", lambda value: value >= 0.80),
    ):
        hits = [row for row in measured_rows if predicate(row["reliability"])]
        print(f"  {label}: {len(hits)}")
        for row in sorted(hits, key=lambda r: r["reliability"]):
            print(
                f"    {row['entry']:<52} {row['reliability']:+.4f} "
                f"[{row['reliability_low']:+.4f}, {row['reliability_high']:+.4f}] "
                f"{row['method_tag']}"
            )

    print("\nset-reliability commands (run through the locked CLI, not from here):")
    for row in measured_rows:
        reason = (
            f"movement quantity: {row['movement_quantity']}; timestamp window: "
            f"{row['timestamp_window']}"
        )
        print(
            "nfl-ats weak-signals set-reliability "
            f"--name {row['entry']} "
            f"--reliability {row['reliability']:.6f} "
            f"--reliability-low {row['reliability_low']:.6f} "
            f"--reliability-high {row['reliability_high']:.6f} "
            f'--method "{row["method"]}" '
            f"--source {output_dir.relative_to(REPO).as_posix()}/results.json "
            f'--reason "{reason}"'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
