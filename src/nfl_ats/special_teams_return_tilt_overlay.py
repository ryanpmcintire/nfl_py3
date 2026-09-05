"""Special-teams return top-quartile tilt overlay: a parameter-free pick-level nudge.

Research chain (mined lineage, PBP-06 special-teams battery, all measured
2026-08-19, read from ``registry/weak_signals.json`` and
``artifacts/special_teams_battery/20260819T232856Z/results.json`` before this
module was built): ``special_teams_return_top_quartile`` -- one of 8
predeclared cells in the special-teams battery (``scripts/special_teams_screen.py``,
predeclaration ``docs/special_teams_battery.md``, mined, uncorrected
multiplicity across the 8 cells) -- flags teams whose PRIOR-season
``return_composite`` (mean z of punt-return and kickoff-return yards) ranks
in the top quartile league-wide. Week-blocked, REG 2009-2025, n=8,634
team-games (2,016 flagged, 496 with no trailing prior-season data): full-slate
effect **+0.4986 accuracy points**, 95% **[-0.0742, +1.0797]**,
``probability_positive`` **0.9547** (season-blocked secondary 95%
[-0.0229, +1.0225], ``probability_positive`` 0.9690).

**The interval crosses zero. Per AGENTS.md, at this evaluator's ~2-point
resolution that is the EXPECTED shape for a real small signal, never grounds
to decline building a no-window-cost prospective challenger.** Neither
admissible closing ground applies (no resolved wrong sign -- the interval is
not entirely below zero; no positive-control bound), so this stays
``unresolved_below_power`` in the registry. Wiring it here is an EV-positive
dual-tracked play (P+ 0.9547 far above the 0.5 that makes playing it the
favoured side of the bet), not a claim of a proven edge (AGENTS.md "a
promotion bar is not a decision bar").

**The reliability is LOW, and that is stated plainly, not hidden.**
Componentwise year-over-year Pearson reliability: punt_return_yards +0.109
[+0.019, +0.196] n=512 team-season pairs; kickoff_return_yards +0.158
[+0.073, +0.243] n=508 (``docs/special_teams_battery.md``). Both are
positive and both intervals exclude zero -- the trait persists across
seasons, weakly. A low-but-positive, interval-excluding-zero reliability
attenuates a real effect toward zero; it does not refute the mechanism (that
would require a reliability whose interval includes or sits at zero, which is
not what was measured here). No promotion is implied by either number; both
travel with every use of this overlay.

**Battery-multiplicity caveat.** ``special_teams_return_top_quartile`` is one
of 8 cells (4 raw dimensions x top/bottom quartile) predeclared together in
the same battery. The bottom-quartile mirror
(``special_teams_return_bottom_quartile``) and the other three raw-dimension
cells (``fg_oe``, ``punt_net_yards``, and the four-dimension
``special_teams_composite_edge``) are correlated siblings sharing overlapping
windows and legs, not independent votes -- multiplicity across the battery is
uncorrected, exactly as the PBP-08 protection-mismatch battery documents for
its own four cells (``src/nfl_ats/pbp08_protection_mismatch_tilt_overlay.py``).

This module is the no-window-cost path, built on the exact pattern of
``interim_hc_first_game_tilt_overlay.py`` (a "flip TOWARD the flagged team"
rule -- this overlay's own direction) and
``pbp08_protection_mismatch_tilt_overlay.py`` (a PBP-battery-derived team
trait read from a stored snapshot, fail-open on a missing source): a
**pick-level, post-prediction transform** of the active model's own forced
pick, dual-tracked against that same active model in the prospective
challenger ledger (``nfl_ats.prospective_scoring``), at no rotation-registry
window cost and with zero training-time feature changes. **Nothing in this
module is wired into ``publishing.py`` or the production pick path** -- like
the tilt siblings, no owner decision to play this on the real card has been
made; it is dual-tracked only.

**The rule is parameter-free and frozen, REG season only**: build the
prior-season top-quartile ``return_composite`` flag for both teams in a game.
If EXACTLY ONE of the two teams is flagged AND the active model's own forced
pick is NOT that team, flip the pick ONTO that team. Both-flagged games are
NEVER touched -- mirroring ``interim_hc_first_game_tilt_overlay``'s
``both_first_game_games`` handling and ``coach_fade_overlay``'s
``both_year_one_games`` handling: a mutual case has no measured direction to
pick between and is reported separately, never flipped. Direction is
PREDECLARED positive on ``team_covered`` -- back the elite return unit -- so
this overlay only ever flips ONTO the flagged side, never off it (unlike the
asymmetric-the-other-way PBP-08 protection overlay, which flips OFF a
flagged offense).

**Pregame-safe by construction.** The trait is a PRIOR-SEASON team-season
aggregate (``scripts/special_teams_features.py`` -> ``team_season.parquet``),
looked up for a game by shifting each team-season row forward exactly one
season (ported from ``scripts/special_teams_screen.py::_prior``, line 148) --
a season's own row is NEVER used as that season's own prior, only as the
PRIOR for the season immediately after it. This function never reads
``result``/``spread_line``/any outcome column at all. Two leakage regression
tests (``tests/test_special_teams_return_tilt_overlay.py``) prove this
empirically: mutating a team's CURRENT-season row never changes that same
season's already-computed flag, and a future season's team-season row never
changes an earlier season's already-computed flag.

**One accepted, stated dilution, mirroring ``fg_oe``'s own documented
convention** (``scripts/special_teams_features.py`` module docstring): the
top-quartile threshold is the ``return_composite_z`` quantile over the WHOLE
available team-season panel (544 rows in the 2009-2025 snapshot this module
reads by default), so a team's own most-recent season contributes a small
amount to the very threshold its PRIOR season is compared against. With
17+ seasons and 32 teams pooled into one global quantile, one row's
contribution is a small fraction of the panel -- the same "minor dilution"
argument ``special_teams_features.py`` already makes for ``fg_oe``'s
season-local baseline, and a materially SMALLER dilution ratio here because
the pool is the whole panel, not one season.

Two things live here, mirroring the sibling overlays exactly:

1. :func:`special_teams_return_flag_by_game` -- the pregame-safe,
   DATA-DERIVED signal, built from an already-loaded ``team_season`` frame
   (see :func:`return_composite_z_with_threshold` for the ported composite
   and quartile-cut construction) and the REG schedule, never hand-typed.
   :func:`special_teams_return_flag_by_game_fail_open` wraps it with the
   latest-snapshot loader and a FAIL-OPEN contract (a missing
   ``data/raw/special_teams/*/team_season.parquet`` snapshot yields zero
   flags and a ``RuntimeWarning``, never an exception -- mirrors
   ``pbp08_protection_mismatch_tilt_overlay.flags_for_week_fail_open`` and
   ``interim_hc_first_game_tilt_overlay.interim_first_game_flag_by_game_fail_open``).
2. :func:`apply_special_teams_return_tilt_overlay` -- the pick-level
   transform, plus :func:`overlay_disclosure_note` for the plain-English
   provenance sentence.

:func:`record_special_teams_return_tilt_challenger_decisions` writes the
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
from nfl_ats.provenance import sha256_file, stamp_sidecar
from nfl_ats.snapshots import latest_snapshot, load_snapshot

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "special_teams_return_tilt_overlay"

#: Ported verbatim from scripts/special_teams_screen.py:66. Applied to the
#: WHOLE team-season panel (global cut, not within-season), exactly as
#: scripts/special_teams_screen.py's own ``main()`` computes it (see that
#: script's printed "544-row 2009-2025 team-season panel" threshold line).
QUARTILE_TOP = 0.75

#: The two raw dimensions scripts/special_teams_screen.py::add_composites
#: (lines 126-145) averages into ``return_composite_z``. Ported verbatim,
#: including the leg order (punt leg first, matching the source).
RETURN_COMPOSITE_LEGS: tuple[str, ...] = ("punt_return_yards", "kickoff_return_yards")

REQUIRED_TEAM_SEASON_COLUMNS = {
    "season",
    "team",
    "punt_return_yards_centered",
    "kickoff_return_yards_centered",
}


def return_composite_z_with_threshold(team_season: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ported verbatim from ``scripts/special_teams_screen.py``.

    Z-scores each return leg's league-centered dimension against the WHOLE
    panel's own pooled standard deviation (``add_composites``, lines
    126-145: ``sd = float(result[centered].std(ddof=1))``,
    ``result[f"{dim}_z"] = result[centered] / sd if sd > 0 else np.nan``),
    then averages the two legs (line 139:
    ``result[["punt_return_yards_z", "kickoff_return_yards_z"]].mean(axis=1)``
    -- ``pandas.DataFrame.mean`` skips NaN by default, so a team missing one
    leg's centered value still gets a composite from the other leg alone,
    exactly as the source does). The top-quartile threshold is the
    ``QUARTILE_TOP`` quantile of ``return_composite_z`` over this WHOLE
    frame (``main()``, lines 322-333), never within-season.

    ``team_season`` must carry ``punt_return_yards_centered`` and
    ``kickoff_return_yards_centered`` -- the league-centered dimensions
    ``scripts/special_teams_features.py::add_league_centered`` (lines
    351-358) produces on its ``team_season.parquet`` output.
    """

    missing = REQUIRED_TEAM_SEASON_COLUMNS.difference(team_season.columns)
    if missing:
        raise DataContractError(
            f"team_season is missing columns for the return composite: {', '.join(sorted(missing))}"
        )

    result = team_season.copy()
    for leg in RETURN_COMPOSITE_LEGS:
        centered = f"{leg}_centered"
        sd = float(result[centered].std(ddof=1))
        result[f"{leg}_z"] = result[centered] / sd if sd > 0 else np.nan

    result["return_composite_z"] = result[[f"{leg}_z" for leg in RETURN_COMPOSITE_LEGS]].mean(
        axis=1
    )
    threshold = float(result["return_composite_z"].quantile(QUARTILE_TOP))
    return result, threshold


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def special_teams_return_flag_by_game(
    schedules: pd.DataFrame, team_season: pd.DataFrame
) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``home_return_top_quartile`` /
    ``away_return_top_quartile``, pregame-safe.

    A side is flagged when its PRIOR season's ``return_composite_z`` (see
    :func:`return_composite_z_with_threshold`) is at or above the top-quartile
    threshold computed over the whole ``team_season`` panel handed in --
    ported from ``scripts/special_teams_screen.py``'s own join
    (``_prior``, line 148: shift each team-season row's ``season`` forward by
    exactly one before joining on ``(team, season)``, so a row is only ever
    consulted as the PRIOR value for the season immediately after the one it
    describes) and flag rule (``flag = long_df[col] >= cutoff`` for the top
    quartile, line ~381).

    Team codes are canonicalized (``TEAM_ABBREVIATION_ALIASES``) before the
    join, mirroring every sibling overlay's merge-safety convention.

    A side with no prior-season row (the team's first tracked season, an
    expansion team, or any gap year) gets ``NaN`` for its prior composite,
    which compares False against the threshold and is folded into "not
    flagged" -- never an error, matching the screen's own
    ``n_missing_required_data`` handling (``score_cell``, ``flag.fillna(False)``).
    """

    required = {"game_id", "season", "game_type", "home_team", "away_team"}
    missing = required.difference(schedules.columns)
    if missing:
        raise DataContractError(
            "schedules is missing columns for special-teams return tracking: "
            f"{', '.join(sorted(missing))}"
        )

    composite, threshold = return_composite_z_with_threshold(team_season)
    composite = composite[["season", "team", "return_composite_z"]].copy()
    composite["team"] = _canonical_team(composite["team"])
    composite["season"] = composite["season"].astype(int)
    # Ported verbatim from scripts/special_teams_screen.py::_prior (line 148):
    # shift the team-season row's OWN season forward by one, so it is only
    # ever joined onto a game whose season is one greater than the one this
    # row describes -- i.e. it is consulted as the PRIOR value only.
    composite["season"] = composite["season"] + 1

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)

    home_prior = composite.rename(
        columns={"team": "home_team", "return_composite_z": "home_prior_return_composite_z"}
    )
    frame = reg[["game_id", "season", "home_team", "away_team"]].merge(
        home_prior, on=["home_team", "season"], how="left"
    )
    away_prior = composite.rename(
        columns={"team": "away_team", "return_composite_z": "away_prior_return_composite_z"}
    )
    frame = frame.merge(away_prior, on=["away_team", "season"], how="left")

    frame["home_return_top_quartile"] = (
        frame["home_prior_return_composite_z"].ge(threshold).fillna(False).astype(bool)
    )
    frame["away_return_top_quartile"] = (
        frame["away_prior_return_composite_z"].ge(threshold).fillna(False).astype(bool)
    )
    frame["return_composite_top_quartile_threshold"] = threshold
    return frame[
        [
            "game_id",
            "season",
            "home_return_top_quartile",
            "away_return_top_quartile",
            "home_prior_return_composite_z",
            "away_prior_return_composite_z",
            "return_composite_top_quartile_threshold",
        ]
    ]


_EMPTY_FLAGS_COLUMNS = ("game_id", "season", "home_return_top_quartile", "away_return_top_quartile")


def latest_special_teams_team_season(data_root: Path) -> Path | None:
    """Newest ``data_root/raw/special_teams/*/team_season.parquet`` snapshot,
    or ``None`` if the directory or every snapshot is absent."""

    root = data_root / "raw" / "special_teams"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/team_season.parquet"))
    return candidates[-1] if candidates else None


def special_teams_return_flag_by_game_fail_open(
    data_root: Path, schedules: pd.DataFrame
) -> pd.DataFrame:
    """The flag table for every REG game in ``schedules``, or an EMPTY frame
    on any missing/malformed input.

    **FAIL-OPEN**: any exception from locating or reading the team-season
    snapshot, or from :func:`special_teams_return_flag_by_game` itself, is
    caught, surfaced as a ``RuntimeWarning``, and folded into "zero games
    flagged" -- mirrors
    ``pbp08_protection_mismatch_tilt_overlay.flags_for_week_fail_open`` and
    ``interim_hc_first_game_tilt_overlay.interim_first_game_flag_by_game_fail_open``
    exactly: this overlay must never be able to block a publish.
    """

    try:
        team_season_path = latest_special_teams_team_season(data_root)
        if team_season_path is None:
            raise FileNotFoundError(
                f"no {data_root / 'raw' / 'special_teams'}/*/team_season.parquet snapshot found"
            )
        team_season = pd.read_parquet(team_season_path)
        return special_teams_return_flag_by_game(schedules, team_season)
    except (FileNotFoundError, KeyError, ValueError, DataContractError) as error:
        warnings.warn(
            f"{CHALLENGER_ID}: flag build failed, proceeding with zero flags "
            f"({type(error).__name__}: {error})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {column: pd.Series([], dtype=object) for column in _EMPTY_FLAGS_COLUMNS}
        )


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    flagged_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring every sibling tilt overlay.
    ``both_flagged_games`` lists games where BOTH teams are top-quartile by
    the prior-season return composite simultaneously -- no measured
    direction for that case (mirrors ``coach_fade_overlay``'s
    ``both_year_one_games`` and ``interim_hc_first_game_tilt_overlay``'s
    ``both_first_game_games``), so those games are flagged, never flipped.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_flagged_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_special_teams_return_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    data_root: Path,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick ONTO the flagged team's side.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the registered
      measurement -- 8,634 REG team-games -- is a regular-season read);
    * exactly ONE side is top-quartile by prior-season ``return_composite``
      (a simultaneous both-flagged case is left untouched -- see
      ``both_flagged_games``); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home)
      is NOT already on the flagged side.

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    the sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise DataContractError(
            f"predictions is missing overlay columns: {', '.join(sorted(missing))}"
        )

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not enabled:
        return TiltResult(base, (), (), enabled)

    flags = special_teams_return_flag_by_game_fail_open(data_root, schedules)
    merged = base.merge(
        flags[["game_id", "home_return_top_quartile", "away_return_top_quartile"]],
        on="game_id",
        how="left",
    )
    for column in ("home_return_top_quartile", "away_return_top_quartile"):
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    both_flagged = merged["home_return_top_quartile"] & merged["away_return_top_quartile"]

    flip_to_home = eligible & merged["home_return_top_quartile"] & ~both_flagged & ~home_pick
    flip_to_away = eligible & merged["away_return_top_quartile"] & ~both_flagged & home_pick
    flip_mask = flip_to_home | flip_to_away

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        flagged_is_home = bool(row["home_return_top_quartile"])
        flagged_team = str(row["home_team"] if flagged_is_home else row["away_team"])
        opponent_team = str(row["away_team"] if flagged_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                flagged_team=flagged_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_flagged, "game_id"].astype(str))
    return TiltResult(overlaid, tuple(flips), both_ids, enabled)


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
        f"{flip.matchup}: {flip.opponent_team} -> {flip.flagged_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** onto a team whose "
        "prior-season special-teams return composite (mean z of punt-return and "
        "kickoff-return yards) ranks in the top quartile league-wide, where the model's own "
        f"pick was not already on that side. {detail}. See "
        "docs/special_teams_return_tilt_overlay.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_special_teams_return_tilt_challenger_decisions(
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

    The special-teams team-season lookup is FAIL-OPEN (see
    :func:`special_teams_return_flag_by_game_fail_open`): a missing source
    snapshot never raises out of this function, it simply yields zero flags
    for the week.

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
    missing = required.difference(card.columns)
    if missing:
        raise DataContractError(
            f"Active forecast card is missing columns: {', '.join(sorted(missing))}"
        )
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    tilt = apply_special_teams_return_tilt_overlay(card, schedules, data_root)
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
        ledger_path = challenger_ledger_path(artifacts_root)
        atomic_parquet(combined[list(CHALLENGER_DECISION_COLUMNS)], ledger_path)
        # ENG-38: stamp which commit appended these rows -- a JSON sidecar,
        # not a rewrite of the parquet ledger itself.
        stamp_sidecar(
            ledger_path, extra={"challenger_id": CHALLENGER_ID, "rows_appended": len(decisions)}
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
        "both_flagged_game_ids": list(tilt.both_flagged_games),
    }


__all__ = [
    "CHALLENGER_ID",
    "QUARTILE_TOP",
    "RETURN_COMPOSITE_LEGS",
    "TiltFlip",
    "TiltResult",
    "apply_special_teams_return_tilt_overlay",
    "latest_special_teams_team_season",
    "overlay_disclosure_note",
    "record_special_teams_return_tilt_challenger_decisions",
    "return_composite_z_with_threshold",
    "special_teams_return_flag_by_game",
    "special_teams_return_flag_by_game_fail_open",
]
