"""Pace-mismatch dog tilt overlay: a parameter-free pick-level nudge.

Research chain (measured 2026-08-19, read from ``registry/weak_signals.json``
before this module was built):

**Registry cell** ``team_style_pace_mismatch_dog_cover`` (PBP-08 team
"personality" battery, one of 5 predeclared cells --
``scripts/team_style_screen.py``, predeclaration ``docs/team_style.md``, mined
lineage, uncorrected multiplicity across cells). Top-quartile absolute
home-minus-away PRIOR-SEASON, league-centered ``seconds_per_play_pace`` vs the
field, scored on ``dog_cover`` (the underdog's cover indicator; pick'ems
excluded, not folded into the complement). REG 2009-2025, n=4,313 games
(n_flag=1,018, n_missing_required_data=248): full-slate effect **+0.2292
accuracy points**, week-blocked 95% **[-0.5587, +1.0401]**,
``probability_positive`` **0.71125**, n_blocks=294. Season-blocked secondary
95% **[-0.1830, +0.6153]**, ``probability_positive`` **0.8711**. Reliability
(the highest in this PBP-08 battery): ``seconds_per_play_pace``'s YoY Pearson
r **+0.489**, 95% CI [+0.405, +0.567], n=512 team-season pairs
(``scripts/team_style_screen.py:374-376``).

**Direction check, done and PASSED, not skipped.** Measured directly from
``artifacts/team_style_screen/20260819T210011Z/results.json``: in flagged
(top-quartile pace-mismatch) games, the underdog covers ``subset_mean``
0.518664 (51.8664%) against a ``complement_mean`` 0.508953 (50.8953%) field --
the underdog covers MORE in the flagged population than outside it, exactly
the predeclared POSITIVE-on-``dog_cover`` direction
(``scripts/team_style_screen.py:479-483``: "Predicted POSITIVE on dog_cover
(variance mechanism: fewer possessions favour the dog; docs/team_style.md)").
A sibling cell in this same batch failed exactly this check (measured
opposite the predeclared direction) and was NOT built into an overlay; this
one passed and is.

**The interval crosses zero on the primary (week-blocked) read.** Per
AGENTS.md/CLAUDE.md, at this evaluator's ~2-point resolution that is the
EXPECTED shape for a real small signal, never grounds to decline building a
no-window-cost prospective challenger. Neither admissible closing ground
applies (no resolved wrong sign -- the point estimate and the season-blocked
secondary both sit on the predicted side; no positive-control bound), so this
remains ``unresolved_below_power`` in the registry.

**Battery-multiplicity caveat, stated up front, not buried.** This is one of
5 predeclared cells in the PBP-08 team-style battery (2 identity cells, 3
matchup cells: this one, a short-game-vs-pressure-defense cell, and a
deep-ball-outdoor-wind cell) -- see ``scripts/team_style_screen.py``'s module
docstring and ``docs/team_style.md``. No multiple-comparison correction is
applied across the battery; the ``probability_positive`` figures above are
this ONE cell's own bootstrap read, not adjusted for having looked at 5.

**Trait and quartile cut, transcribed verbatim from the screen, not
re-derived:**

* ``seconds_per_play_pace`` -- drive time-of-possession divided by drive play
  count, pooled directly from plays/drives per (season, team) (NOT an average
  of per-game rates, to avoid Simpson's-paradox bias from uneven per-game play
  counts) -- ``scripts/team_style_features.py:358-366``
  (``build_team_season_style``'s pace block) via
  ``scripts/team_style_features.py:162-184`` (``_drive_pace_table``, which
  reuses ``nfl_ats.pbp.build_drive_table`` verbatim, the same drive
  aggregation the production PBP-05 pipeline uses).
* "Centered" -- each dimension minus ITS OWN SEASON's unweighted across-team
  mean, so leaguewide pace drift over 2009-2025 does not read as a team
  identity -- ``scripts/team_style_features.py:413-422``
  (``add_league_centered``).
* Prior-season join -- ``scripts/team_style_screen.py:151-161`` (``_prior``):
  shift the (season, team) table forward one season, so joining on
  ``season`` pulls the PRIOR season's centered value onto this season's game.
  Ported here as :func:`pace_mismatch_flag_by_game`'s own prior-season merge,
  same shift-by-one-season construction.
* ``pace_diff_abs`` -- ``abs(home_prior_pace_centered - away_prior_pace_centered)``
  -- ``scripts/team_style_screen.py:203-216`` (``build_game_table``'s pace
  block).
* Quartile cut -- top quartile (0.75) of ``pace_diff_abs`` over the REG
  population, ``QUARTILE = 0.75`` (``scripts/team_style_screen.py:76``),
  ``pace_threshold = float(game["pace_diff_abs"].quantile(QUARTILE))``
  (``scripts/team_style_screen.py:366``, computed over ALL REG games
  including pick'ems, before the pick'em population restriction for the
  ``dog_cover`` value column). The MEASURED numeric threshold, frozen here
  exactly as measured and never recomputed (same discipline as
  ``spread_gap_zone_fade_overlay.SPREAD_GAP_LOWER_BOUND``/``UPPER_BOUND``):
  **2.1685022294778378**, read from
  ``artifacts/team_style_screen/20260819T210011Z/results.json:pace_diff_abs_threshold``.
* Flag comparator -- ``>=`` the threshold (``scripts/team_style_screen.py:468``:
  ``flag_b2 = game_b2["pace_diff_abs"] >= pace_threshold``).

**Spread convention, verified independently, not just trusted from a
comment.** ``scripts/team_style_screen.py:121-123``: "spread_line > 0 -> HOME
favored; spread_line < 0 -> AWAY favored", cross-checked there against
``nfl_ats.features.add_ats_outcomes``'s ``ats_margin = result - spread_line``
convention. Independently re-verified this session against real schedule data
(``data/raw/20260824T115346Z/schedules.parquet``): the 2013 week-6 DEN
(home)-vs-JAX (away) game carries ``spread_line=+27.0`` with Denver a known
lopsided home favorite; the 2019 week-2 MIA (home)-vs-NE (away) game carries
``spread_line=-18.0`` with New England a known lopsided ROAD favorite. Both
confirm ``spread_line > 0`` means HOME favored and ``spread_line < 0`` means
AWAY favored.

**The rule is parameter-free and frozen (assigned before any code was
written).** REG season only. Compute the absolute difference between the two
teams' PRIOR-SEASON centered ``seconds_per_play_pace``. If that difference is
in the top quartile (the screen's own frozen cut, above) AND the active
model's own forced pick is the FAVOURITE (by ``spread_line``), flip the pick
to the UNDERDOG. If the model already has the underdog, leave it. Pick'em
games (``spread_line == 0``, no defined underdog -- exactly how
``scripts/team_style_screen.py:124-128`` defines ``dog_cover`` as NaN for
``spread_line == 0``) are never touched. Missing prior-season pace data (a
new franchise's first tracked season, or an incomplete cache) means
``pace_mismatch_flag = False``, never an error.

This module is the no-window-cost path, built on the exact pattern of
``surface_switch_tilt_overlay.py`` (schedule/trait-derived flag, ported
verbatim from its source screen), ``spread_gap_zone_fade_overlay.py`` (flips
keyed off the card's own ``spread_line`` rather than team identity), and
``pbp08_protection_mismatch_tilt_overlay.py``/``pbp08_matchup_flags.py`` (the
PBP-derived team-trait flag pattern, including its FAIL-OPEN posture for a
data-dependent, gitignored, bespoke cache): a **pick-level, post-prediction
transform** of the active model's own forced pick, dual-tracked against that
same active model in the prospective challenger ledger
(``nfl_ats.prospective_scoring``), at no rotation-registry window cost and
with zero training-time feature changes. **Nothing in this module is wired
into ``publishing.py`` or the production pick path** -- no owner decision to
play this on the real card has been made; it is dual-tracked only.

**Fail-open, like the PBP-derived sibling.** The team-season pace cache
(``data/pbp/team_style/team_season_style.parquet``) is a bespoke,
gitignored, network-fetched research artifact -- NOT part of the standard
captured raw-schedule snapshot pipeline -- so a missing cache, a missing
schedule snapshot, or a broken build folds into ZERO flags and a documented
no-op, never an exception that could break a weekly record call.

Four things live here, mirroring the sibling overlays' structure:

1. :func:`pace_mismatch_flag_by_game` -- the pregame-safe, DATA-DERIVED
   signal, ported from the screen's own construction (see citations above).
2. :func:`pace_mismatch_flags_fail_open` -- loads the schedule snapshot and
   the team-season pace cache and calls (1), never raising.
3. :func:`apply_pace_mismatch_dog_tilt_overlay` -- the pick-level transform,
   plus :func:`overlay_disclosure_note` for the plain-English provenance
   sentence.
4. :func:`record_pace_mismatch_dog_tilt_challenger_decisions` -- writes the
   overlay's own arm to the prospective challenger ledger so 2026 scores it
   cleanly, independent of whether it is ever played on the real card.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    challenger_ledger_path,
    config_fingerprint,
    find_challenger,
    load_challenger_decisions,
)
from nfl_ats.provenance import sha256_file
from nfl_ats.snapshots import latest_snapshot, load_snapshot

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "pace_mismatch_dog_tilt_overlay"

#: Frozen, MEASURED cut -- scripts/team_style_screen.py's own
#: ``pace_diff_abs_threshold`` (QUARTILE=0.75 over the REG population),
#: read from artifacts/team_style_screen/20260819T210011Z/results.json.
#: Not a free parameter of this overlay; never recomputed here.
PACE_DIFF_ABS_THRESHOLD = 2.1685022294778378

#: The merged-in flag travels under this module-private name, matching
#: surface_switch_tilt_overlay's OVERLAY_FLAG_COLUMN convention -- a
#: predictions frame that already carries a same-named column collides
#: silently instead of crashing, and a missing flag column folds into the
#: documented no-op.
OVERLAY_FLAG_COLUMN = "_pace_mismatch_dog_tilt_flag"

#: Required columns of the team-season pace cache this module reads
#: (data/pbp/team_style/team_season_style.parquet, built by
#: scripts/team_style_features.py -- see module docstring for the exact
#: derivation of seconds_per_play_pace_centered).
TEAM_SEASON_STYLE_REQUIRED_COLUMNS = frozenset({"season", "team", "seconds_per_play_pace_centered"})


def team_season_style_path(data_root: Path) -> Path:
    """Matches ``scripts/team_style_features.py:71,75``'s
    ``CACHE_DIR`` / ``TEAM_SEASON_PATH`` construction verbatim (relative to
    ``data_root`` rather than the repo root, so tests can point this at a
    ``tmp_path`` fixture)."""

    return data_root / "pbp" / "team_style" / "team_season_style.parquet"


def _canonical_team(team: pd.Series) -> pd.Series:
    """Ported verbatim from ``surface_switch_tilt_overlay._canonical_team``."""

    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def pace_mismatch_flag_by_game(
    schedules: pd.DataFrame, team_season_style: pd.DataFrame
) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``pace_diff_abs`` and
    ``pace_mismatch_flag``.

    ``pace_mismatch_flag`` fires when the absolute difference between the
    home and away teams' PRIOR-SEASON, league-centered
    ``seconds_per_play_pace`` is at or above :data:`PACE_DIFF_ABS_THRESHOLD`
    -- the exact construction and frozen numeric cut transcribed in this
    module's docstring from ``scripts/team_style_screen.py``.

    **Why the prior-season shift is pregame-safe**: ``seconds_per_play_pace``
    for season ``S`` is joined onto games in season ``S + 1`` only (the same
    ``_prior``-style shift ``scripts/team_style_screen.py:151-161`` uses for
    every trait in this battery) -- a team's PRIOR season's pace is public,
    fully-realized information before that team's next Week 1 kicks off, and
    this function never reads ``result``, ``spread_line``, or any outcome
    column at all. Two leakage regression tests
    (``tests/test_pace_mismatch_dog_tilt_overlay.py``) prove this empirically:
    mutating a game's own outcome columns has no bearing, and mutating the
    CURRENT season's (not prior season's) pace row for a team never changes
    that game's already-computed flag.

    Team codes are canonicalized (``TEAM_ABBREVIATION_ALIASES``) on both the
    schedule and the style table before joining -- a merge-safety measure
    (franchise continuity: OAK->LV, SD->LAC, STL->LA) that does not change
    which prior-season pace value a team actually carries.
    """

    required_schedule = {"game_id", "season", "game_type", "home_team", "away_team"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            f"schedules is missing columns for pace-mismatch tracking: "
            f"{', '.join(missing_schedule)}"
        )
    missing_style = sorted(TEAM_SEASON_STYLE_REQUIRED_COLUMNS.difference(team_season_style.columns))
    if missing_style:
        raise DataContractError(
            f"team_season_style is missing columns for pace-mismatch tracking: "
            f"{', '.join(missing_style)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)

    style = team_season_style[["team", "season", "seconds_per_play_pace_centered"]].copy()
    style["team"] = _canonical_team(style["team"])
    style["season"] = style["season"].astype(int)
    # Shift one season forward so joining on `season` pulls the PRIOR
    # season's centered pace onto this season's game -- scripts/
    # team_style_screen.py:151-161 (`_prior`), applied once per side.
    style["season"] = style["season"] + 1

    prior_home = style.rename(
        columns={"team": "home_team", "seconds_per_play_pace_centered": "home_prior_pace_centered"}
    )
    reg = reg.merge(prior_home, on=["home_team", "season"], how="left")

    prior_away = style.rename(
        columns={"team": "away_team", "seconds_per_play_pace_centered": "away_prior_pace_centered"}
    )
    reg = reg.merge(prior_away, on=["away_team", "season"], how="left")

    reg["pace_diff_abs"] = (reg["home_prior_pace_centered"] - reg["away_prior_pace_centered"]).abs()
    # NaN >= threshold evaluates False in pandas/numpy, so missing prior-
    # season data for either side already yields flag=False; .fillna(False)
    # is kept for explicitness and mypy's benefit, not because it is load-
    # bearing.
    reg["pace_mismatch_flag"] = reg["pace_diff_abs"].ge(PACE_DIFF_ABS_THRESHOLD).fillna(False)

    return reg[["game_id", "season", "pace_diff_abs", "pace_mismatch_flag"]].reset_index(drop=True)


def pace_mismatch_flags_fail_open(data_root: Path) -> pd.DataFrame:
    """The flag table for the full local history, or an EMPTY frame on any
    missing input.

    Never raises. Same posture as
    ``pbp08_protection_mismatch_tilt_overlay.flags_for_week_fail_open``: an
    absent team-season pace cache or schedule snapshot is a no-op week, not a
    broken publish. The reason is surfaced as a warning so a silent zero is
    at least noisy.
    """

    try:
        style_path = team_season_style_path(data_root)
        if not style_path.is_file():
            raise FileNotFoundError(f"missing team-season pace cache: {style_path}")
        snapshot = latest_snapshot(data_root / "raw")
        schedules, _team_stats = load_snapshot(snapshot)
        team_season_style = pd.read_parquet(style_path)
        return pace_mismatch_flag_by_game(schedules, team_season_style)
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        DataContractError,
    ) as error:  # pragma: no cover - exercised via the warning path
        warnings.warn(
            f"{CHALLENGER_ID}: flag build failed, proceeding with zero flags ({error})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["game_id", "season", "pace_diff_abs", "pace_mismatch_flag"])


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    original_pick_team: str
    flipped_to_team: str
    spread_line: float
    pace_diff_abs: float


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring the sibling overlays' ``TiltResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_pace_mismatch_dog_tilt_overlay(
    predictions: pd.DataFrame,
    flags: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick from the FAVOURITE to the UNDERDOG wherever the
    flag fires.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the registry cell
      was scored on regular-season games only);
    * ``spread_line`` is present, numeric, and non-zero (pick'em games have
      no defined underdog and are never touched -- matching
      ``scripts/team_style_screen.py:124-128``'s ``dog_cover`` construction,
      which is NaN for ``spread_line == 0``);
    * :func:`pace_mismatch_flag_by_game` fires for the game (top-quartile
      absolute prior-season centered pace difference); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) is
      currently on the FAVOURITE side (``spread_line > 0`` and home picked,
      or ``spread_line < 0`` and away picked).

    A pick already on the underdog is left untouched -- there is nothing to
    flip TO. Flipping sets ``home_cover_probability`` to its complement,
    exactly as the sibling overlays do, so every existing reader of the
    column needs no overlay-aware branch.

    The flag is ALWAYS this call's own ``flags`` argument, merged under the
    private :data:`OVERLAY_FLAG_COLUMN` name: a predictions frame that
    already carries a same-named column collides silently instead of
    crashing, and if no flag column survives the merge at all the result is
    the documented no-op -- zero flips, never a KeyError.
    """

    required = {
        "game_id",
        "season",
        "home_team",
        "away_team",
        "home_cover_probability",
        "spread_line",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    flag_columns = ["game_id", "season", "pace_diff_abs", "pace_mismatch_flag"]
    if flags.empty or "pace_mismatch_flag" not in flags.columns:
        flags_to_merge = pd.DataFrame(columns=flag_columns)
    else:
        flags_to_merge = flags[flag_columns]

    merged = base.merge(
        flags_to_merge.rename(
            columns={"pace_mismatch_flag": OVERLAY_FLAG_COLUMN, "pace_diff_abs": "_pace_diff_abs"}
        ),
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    if OVERLAY_FLAG_COLUMN not in merged.columns:
        merged[OVERLAY_FLAG_COLUMN] = False
    merged[OVERLAY_FLAG_COLUMN] = merged[OVERLAY_FLAG_COLUMN].fillna(False).astype(bool)
    if "_pace_diff_abs" not in merged.columns:
        merged["_pace_diff_abs"] = np.nan

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    spread_line = pd.to_numeric(merged["spread_line"], errors="coerce")
    home_favored = spread_line.gt(0.0)
    away_favored = spread_line.lt(0.0)
    eligible &= home_favored | away_favored  # excludes NaN and exact-zero pick'ems

    pick_home = merged["home_cover_probability"].ge(0.5)
    model_has_favorite = (home_favored & pick_home) | (away_favored & ~pick_home)

    flip_mask = eligible & merged[OVERLAY_FLAG_COLUMN] & model_has_favorite

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for idx in merged.loc[flip_mask].index:
        row = merged.loc[idx]
        row_home_pick = bool(pick_home.loc[idx])
        original_team = str(row["home_team"] if row_home_pick else row["away_team"])
        flipped_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                original_pick_team=original_team,
                flipped_to_team=flipped_team,
                spread_line=float(spread_line.loc[idx]),
                pace_diff_abs=float(row["_pace_diff_abs"]),
            )
        )

    return TiltResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: TiltResult) -> str:
    """Plain-language provenance sentence, mirroring the sibling overlays'.

    Empty when the overlay is off or changed nothing this week. Not
    currently surfaced on the published card -- this overlay is dual-tracked
    only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup} ({flip.spread_line:+.1f}, pace gap {flip.pace_diff_abs:.2f}): "
        f"{flip.original_pick_team} -> {flip.flipped_to_team}"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** off the market favourite "
        "onto the underdog in a top-quartile prior-season pace mismatch (fewer, longer "
        f"possessions favour the dog). {detail}. See docs/pace_mismatch_dog_tilt_overlay.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_pace_mismatch_dog_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors ``surface_switch_tilt_overlay.record_surface_switch_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this
    challenger tracks the tilt's forced-pick (``decision_line``) accuracy
    only, never a fabricated paper-bet edge for the post-tilt side.
    """

    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError(
            "No synchronized active ATS model is available to record tilt decisions from"
        )
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this tilt -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    flags = pace_mismatch_flags_fail_open(data_root)
    tilt = apply_pace_mismatch_dog_tilt_overlay(card, flags)
    tilted_card = tilt.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = tilted_card.loc[keep]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": CHALLENGER_ID,
            "config_fingerprint": observed_fingerprint,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)

    return {
        "challenger_id": CHALLENGER_ID,
        "season": int(card["season"].iloc[0]),
        "week": int(card["week"].iloc[0]),
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
    }


__all__ = [
    "CHALLENGER_ID",
    "OVERLAY_FLAG_COLUMN",
    "PACE_DIFF_ABS_THRESHOLD",
    "TEAM_SEASON_STYLE_REQUIRED_COLUMNS",
    "TiltFlip",
    "TiltResult",
    "apply_pace_mismatch_dog_tilt_overlay",
    "overlay_disclosure_note",
    "pace_mismatch_flag_by_game",
    "pace_mismatch_flags_fail_open",
    "record_pace_mismatch_dog_tilt_challenger_decisions",
    "team_season_style_path",
]
