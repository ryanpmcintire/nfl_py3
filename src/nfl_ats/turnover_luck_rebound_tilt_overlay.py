"""Turnover-luck rebound tilt overlay: a parameter-free pick-level nudge.

Research chain (all measured 2026-08-21, read from ``registry/weak_signals.json``
before this module was built): the close-game/turnover LUCK regression
battery (``scripts/close_game_luck_screen.py``, predeclared in
``docs/close_game_luck_screen.md``) cell ``turnover_under_rebound`` --
registered as ``close_game_luck_turnover_under_rebound`` -- flags a team
whose PRIOR-season centered turnover differential per game sits in the
bottom quartile of the pooled panel and predicts a POSITIVE cover rate on
``team_covered`` (a rebound: a team that turned the ball over unluckily last
season tends not to repeat it). Week-blocked, REG 2009-2025, n=8,634
team-games (n_flag 2,036, 496 missing required data): full-slate effect
**+0.4092 accuracy points**, 95% ``[-0.1526, +0.9692]``, ``probability_positive``
**0.92**; season-blocked secondary 95% ``[+0.0297, +0.7949]``,
``probability_positive`` 0.981.

**Direction check passed, stated up front.** The flagged group covers
51.33% against a 49.59% field (``subset_mean``/``complement_mean`` in
``artifacts/close_game_luck_screen/20260821T182234Z/results.json``, cell
``turnover_under_rebound``) -- exactly the REBOUND direction predeclared
before this cell was scored, not a post-hoc sign flip.

**The week-blocked interval crosses zero. Per AGENTS.md, at this evaluator's
~2-point resolution that is the EXPECTED shape for a real-but-small signal,
never grounds to decline building a no-window-cost prospective challenger.**
Neither admissible closing ground applies (no resolved wrong sign -- the
whole interval is not on the wrong side of zero -- and no positive-control
bound was run), so this stays ``unresolved_below_power`` in the registry;
wiring it here is an EV-positive dual-tracked play (``probability_positive``
0.92 > 0.5), not a claim of a proven edge (AGENTS.md "a promotion bar is not
a decision bar").

**Reliability is low but positive -- 0.1322, not a closing ground.** The
underlying trait's year-over-year Pearson correlation (season-centered
``turnover_diff_per_game``, ``scripts/close_game_luck_screen.py::reliability_table``)
is +0.1322, 95% CI ``[+0.0490, +0.2134]``, n=512 team-season pairs -- entirely
positive, so the trait persists weakly across seasons rather than being pure
noise. A reliability this low ATTENUATES the measurable effect (regression to
the mean erases most, but not all, of a team's turnover-luck signature
year over year); it does not refute the mechanism. No admissible closing
ground turns on reliability alone unless it is exactly zero, and it is not.

**Deliberately ASYMMETRIC -- the dead mirror is stated plainly, not buried.**
The sibling cell ``turnover_over_fade`` (top-quartile centered turnover
differential, predicted NEGATIVE/fade) reads full-slate **+0.0076 accuracy
points**, ``probability_positive`` **0.5008** -- a dead coin flip on the SAME
underlying trait. So this is ONE asymmetric signal (the bottom tail of the
turnover-luck distribution moves; the top tail does not), never two votes for
"turnover luck matters symmetrically." The registry's own note on
``close_game_luck_turnover_under_rebound`` already says this: "Correlated
decomposition of turnover trait, not independent confirmation." This overlay
therefore only ever flips a pick ONTO the bottom-quartile team, never away
from a top-quartile one -- there is no measured direction for that case.

This module is the no-window-cost path, built on the exact pattern of
``surface_switch_tilt_overlay.py`` (the canonical simplest tilt overlay),
``interim_hc_first_game_tilt_overlay.py`` (flip TOWARD the flagged side,
this overlay's own direction), and ``coach_fade_overlay.py`` /
``backup_qb_fade_overlay.py`` (both-flagged clean-case handling): a
**pick-level, post-prediction transform** of the active model's own forced
pick, dual-tracked against that same active model in the prospective
challenger ledger (``nfl_ats.prospective_scoring``), at no rotation-registry
window cost and with zero training-time feature changes. **Nothing in this
module is wired into ``publishing.py`` or the production pick path** -- like
the tilt siblings, no owner decision to play this on the real card has been
made; it is dual-tracked only.

**The rule is parameter-free and frozen given the screen's own numbers**: no
threshold is tuned here -- the bottom-quartile cutoff below is the screen's
own pooled-panel 25th percentile of ``turnover_diff_per_game_centered``
(``scripts/close_game_luck_screen.py:369``,
``thresholds["turnover_q25"] = -0.4026832217261905``, read from
``artifacts/close_game_luck_screen/20260821T182234Z/results.json``), carried
here as a frozen constant rather than a knob (AGENTS.md: "every overlay
parameter must be the registry cell's own measured value, cited"). REG season
only (every measured read above was scored on regular-season games). When
**exactly one** team in a game is flagged AND the active model's own forced
pick is NOT that team, flip the pick ONTO that team. Both-flagged games are
never touched -- no measured direction for a mutual case, mirroring
``coach_fade_overlay`` / ``backup_qb_fade_overlay``'s clean-case handling --
and a team already picked needs no flip.

**Pregame-safe by construction.** :func:`turnover_under_flag_by_game` looks
up each team's centered turnover differential from the season STRICTLY
BEFORE the game being flagged (the same ``season + 1`` shift
``scripts/close_game_luck_screen.py::_prior`` (line 193) uses), and a
missing prior season (an expansion team, or any gap year) yields ``False``,
never an error. The trait itself is a full PRIOR-season aggregate -- fully
known before that season's Week 1 -- so it can and should fire in Week 1 of
2026. Two leakage regression tests in
``tests/test_turnover_luck_rebound_tilt_overlay.py`` prove this empirically:
the flag is unchanged when the CURRENT season's or the target game's own
turnover events are mutated (the function never even loads current-season
play-by-play into the panel it looks the flag up from).

Trait and quartile construction, transcribed VERBATIM (not re-derived) from
``scripts/close_game_luck_screen.py``:

* **Giveaways** (``build_giveaways_table``, lines 91-106): for REG-season
  plays only, ``giveaways = interception + fumble_lost`` per play, summed by
  ``(game_id, posteam)``.
* **Takeaways** (``build_team_games``, lines 108-136): for each team-game,
  ``takeaways`` is the OPPONENT's ``giveaways`` in that same game (a merge on
  ``(game_id, opponent)``), so takeaways and giveaways are two views of the
  same turnover events, never independently measured.
* **Season aggregate** (``build_panel``, lines 139-163):
  ``turnover_diff_per_game = (takeaways - giveaways) / games`` per
  ``(season, team)``, summed/averaged across the WHOLE season.
* **"Centered"** (``build_panel``, lines 160-162): ``league_mean`` is that
  SAME season's mean ``turnover_diff_per_game`` across all teams
  (``panel.groupby("season")[trait].transform("mean")``), and
  ``turnover_diff_per_game_centered = turnover_diff_per_game - league_mean``
  -- centering is PER-SEASON (removes that season's overall turnover
  environment), never pooled across seasons.
* **Bottom-quartile cut** (``main``, line 369):
  ``thresholds["turnover_q25"] = panel["turnover_diff_per_game_centered"].quantile(0.25)``
  -- a single POOLED quantile taken across the ENTIRE 2009-2025 panel (every
  team-season at once, not a per-season or expanding cut), frozen here at
  its measured value ``-0.4026832217261905``.
* **Prior-season lookup** (``_prior``, lines 193-198): the CENTERED value is
  shifted ``season + 1`` and joined back onto the following season's games by
  ``team`` -- so a game in season *S* only ever sees the centered value
  computed from season *S-1*.

This module ports that construction on the ``turnover_diff_per_game`` leg
only (the ``one_score_luck`` and ``takeaway_share`` legs the screen also
carries belong to different, separately-registered cells and are out of
scope here).

Two things live here, mirroring the sibling overlays exactly:

1. :func:`turnover_under_flag_by_game` -- the pregame-safe, DATA-DERIVED
   signal, ported (trimmed to the turnover leg only) from
   ``scripts/close_game_luck_screen.py``, read from the newest local schedule
   and play-by-play snapshots, never hand-typed.
2. :func:`apply_turnover_luck_rebound_tilt_overlay` -- the pick-level
   transform, plus :func:`overlay_disclosure_note` for the plain-English
   provenance sentence.

:func:`record_turnover_luck_rebound_tilt_challenger_decisions` writes the
overlay's own arm to the prospective challenger ledger so 2026 scores it
cleanly, independent of whether it is ever played on the real card.

**Composition correlation warning.** This overlay's flag and the dead-mirror
``turnover_over_fade`` cell derive from the SAME underlying
``turnover_diff_per_game_centered`` trait (opposite tails of the same
distribution). If either cell is ever pooled with another weak-signal input,
it must never be treated as independent confirmation of a second, different
turnover-related finding -- see the registry's own note on
``close_game_luck_turnover_under_rebound``.
"""

from __future__ import annotations

import json
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
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot
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
CHALLENGER_ID = "turnover_luck_rebound_tilt_overlay"

#: Frozen, not tuned here: the screen's own pooled-panel 25th percentile of
#: ``turnover_diff_per_game_centered`` across the full 2009-2025 team-season
#: panel (scripts/close_game_luck_screen.py:369), read from
#: artifacts/close_game_luck_screen/20260821T182234Z/results.json ->
#: thresholds.turnover_q25. AGENTS.md: "every overlay parameter must be the
#: registry cell's own measured value, cited" -- this IS that value.
TURNOVER_UNDER_Q25_THRESHOLD = -0.4026832217261905

#: Required play-by-play columns for the giveaways leg (mirrors
#: scripts/close_game_luck_screen.py::build_giveaways_table).
_REQUIRED_PBP_COLUMNS = {"game_id", "season_type", "posteam", "interception", "fumble_lost"}

#: Required schedule columns for the team-game / panel construction (mirrors
#: scripts/close_game_luck_screen.py::build_team_games / build_panel).
_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "season", "game_type", "home_team", "away_team"}


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _giveaways_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """Ported verbatim from ``scripts/close_game_luck_screen.py::build_giveaways_table``
    (lines 91-106): REG-season plays only, ``giveaways = interception +
    fumble_lost`` per play, summed by ``(game_id, posteam)``.

    Unlike the screen, this never caps the season range -- the screen's
    ``SEASON_START``/``SEASON_END`` (2009-2025) bound its own HISTORICAL
    measurement population only; this function must also score the CURRENT
    (2026+) season, which the screen never saw.
    """

    missing = sorted(_REQUIRED_PBP_COLUMNS.difference(pbp.columns))
    if missing:
        raise DataContractError(
            f"play-by-play is missing columns for turnover tracking: {', '.join(missing)}"
        )

    plays = pbp.loc[pbp["season_type"].astype(str).eq("REG")].copy()
    plays["posteam"] = _canonical_team(plays["posteam"])
    plays["_turnover"] = pd.to_numeric(plays["interception"], errors="coerce").fillna(
        0.0
    ) + pd.to_numeric(plays["fumble_lost"], errors="coerce").fillna(0.0)
    giveaways = (
        plays.dropna(subset=["posteam"])
        .groupby(["game_id", "posteam"], sort=False)["_turnover"]
        .sum()
        .reset_index(name="giveaways")
    )
    return giveaways


def _team_game_turnovers(schedules: pd.DataFrame, giveaways: pd.DataFrame) -> pd.DataFrame:
    """Ported (trimmed to the turnover leg) from
    ``scripts/close_game_luck_screen.py::build_team_games`` (lines 108-136):
    one row per team-game with ``giveaways`` and ``takeaways`` (the
    OPPONENT's ``giveaways`` in that same game).
    """

    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for turnover tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)

    own = giveaways.rename(columns={"posteam": "team"})
    against = giveaways.rename(columns={"posteam": "opponent"})
    sides = []
    for is_home in (True, False):
        team_col = "home_team" if is_home else "away_team"
        opp_col = "away_team" if is_home else "home_team"
        side = pd.DataFrame(
            {
                "game_id": reg["game_id"],
                "season": reg["season"],
                "team": reg[team_col],
                "opponent": reg[opp_col],
            }
        )
        side = side.merge(own, on=["game_id", "team"], how="left")
        side = side.merge(against, on=["game_id", "opponent"], how="left", suffixes=("", "_opp"))
        side = side.rename(columns={"giveaways_opp": "takeaways"})
        sides.append(side)
    team_games = pd.concat(sides, ignore_index=True).reset_index(drop=True)
    team_games["giveaways"] = team_games["giveaways"].fillna(0.0)
    team_games["takeaways"] = team_games["takeaways"].fillna(0.0)
    return team_games


def _season_centered_turnover_panel(team_games: pd.DataFrame) -> pd.DataFrame:
    """Ported verbatim from ``scripts/close_game_luck_screen.py::build_panel``
    (lines 139-163), turnover leg only: per-``(season, team)``
    ``turnover_diff_per_game = (takeaways - giveaways) / games``, then
    centered by subtracting that SAME season's league mean.
    """

    panel = (
        team_games.groupby(["season", "team"], sort=False)
        .agg(
            games=("game_id", "size"),
            giveaways=("giveaways", "sum"),
            takeaways=("takeaways", "sum"),
        )
        .reset_index()
    )
    panel["turnover_diff_per_game"] = (panel["takeaways"] - panel["giveaways"]) / panel[
        "games"
    ].replace(0, np.nan)
    league_mean = panel.groupby("season")["turnover_diff_per_game"].transform("mean")
    panel["turnover_diff_per_game_centered"] = panel["turnover_diff_per_game"] - league_mean
    return panel


def turnover_under_flag_by_game(schedules: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``home_turnover_under_flag`` /
    ``away_turnover_under_flag``.

    A side is flagged when its PRIOR-season (``season - 1``) centered
    ``turnover_diff_per_game`` is at or below the frozen bottom-quartile
    threshold :data:`TURNOVER_UNDER_Q25_THRESHOLD` -- ported verbatim from
    ``scripts/close_game_luck_screen.py``'s ``_prior``
    (``season + 1`` shift, looked up by team) and ``turnover_under_rebound``
    cell (``flag = prior_turnover_diff_per_game_centered <= thresholds["turnover_q25"]``,
    line 449).

    **Pregame-safe by construction.** The panel this function builds is
    computed entirely from PRIOR-season play-by-play and schedule rows (the
    ``season + 1`` shift means a game in season *S* only ever reads season
    *S-1*'s already-completed, already-centered trait); it never reads a
    game's own outcome, ``result``, or ``spread_line`` at all. A team with no
    observed prior season in the data (an expansion team, or any data gap)
    gets ``NaN`` from the merge, which compares ``False`` against the
    threshold -- "missing prior-season data" always means "not flagged",
    never an error.
    """

    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for turnover tracking: {', '.join(missing)}"
        )

    giveaways = _giveaways_table(pbp)
    team_games = _team_game_turnovers(schedules, giveaways)
    panel = _season_centered_turnover_panel(team_games)

    prior = panel[["team", "season", "turnover_diff_per_game_centered"]].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(
        columns={"turnover_diff_per_game_centered": "prior_turnover_diff_per_game_centered"}
    )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    frame = reg[["game_id", "season", "home_team", "away_team"]].copy()

    home_prior = prior.rename(
        columns={
            "team": "home_team",
            "prior_turnover_diff_per_game_centered": "home_prior_centered",
        }
    )
    frame = frame.merge(home_prior, on=["home_team", "season"], how="left")
    away_prior = prior.rename(
        columns={
            "team": "away_team",
            "prior_turnover_diff_per_game_centered": "away_prior_centered",
        }
    )
    frame = frame.merge(away_prior, on=["away_team", "season"], how="left")

    frame["home_turnover_under_flag"] = (
        frame["home_prior_centered"].le(TURNOVER_UNDER_Q25_THRESHOLD).fillna(False).astype(bool)
    )
    frame["away_turnover_under_flag"] = (
        frame["away_prior_centered"].le(TURNOVER_UNDER_Q25_THRESHOLD).fillna(False).astype(bool)
    )
    return frame[["game_id", "season", "home_turnover_under_flag", "away_turnover_under_flag"]]


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    flagged_team: str
    original_pick_team: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``surface_switch_tilt_overlay.TiltResult``.
    ``both_flagged_games`` lists games where BOTH teams are bottom-quartile
    coming in -- no measured direction for that case (mirrors
    ``coach_fade_overlay``'s ``both_year_one_games``), so those games are
    flagged, never flipped.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_flagged_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_turnover_luck_rebound_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick ONTO the bottom-quartile-turnover-luck team.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (every measured
      read -- the battery cell and its reliability -- was scored on
      regular-season games only);
    * :func:`turnover_under_flag_by_game` fires for EXACTLY ONE side of the
      game; and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) is
      NOT already on the flagged side.

    Both-flagged games are left untouched (see ``both_flagged_games``) --
    no measured direction for a mutual case, mirroring
    ``coach_fade_overlay`` / ``backup_qb_fade_overlay``'s clean-case
    handling. **Deliberately ASYMMETRIC**, like the sibling tilts: this only
    ever flips a pick ONTO a flagged team, never away from one -- the dead
    mirror cell ``turnover_over_fade`` (module docstring) gives no measured
    direction for fading a top-quartile team.

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    the sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not enabled:
        return TiltResult(base, (), (), enabled)

    flags = turnover_under_flag_by_game(schedules, pbp)
    merged = base.merge(flags, on=["game_id", "season"], how="left", validate="one_to_one")
    merged["home_turnover_under_flag"] = (
        merged["home_turnover_under_flag"].fillna(False).astype(bool)
    )
    merged["away_turnover_under_flag"] = (
        merged["away_turnover_under_flag"].fillna(False).astype(bool)
    )

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    both_flagged = merged["home_turnover_under_flag"] & merged["away_turnover_under_flag"]
    home_pick = merged["home_cover_probability"].ge(0.5)

    flip_to_home = eligible & merged["home_turnover_under_flag"] & ~both_flagged & ~home_pick
    flip_to_away = eligible & merged["away_turnover_under_flag"] & ~both_flagged & home_pick
    flip_mask = flip_to_home | flip_to_away

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        flagged_is_home = bool(row["home_turnover_under_flag"])
        flagged_team = str(row["home_team"] if flagged_is_home else row["away_team"])
        original_pick_team = str(row["away_team"] if flagged_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                flagged_team=flagged_team,
                original_pick_team=original_pick_team,
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
        f"{flip.matchup}: {flip.original_pick_team} -> {flip.flagged_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** onto a team coming off a "
        "bottom-quartile prior-season turnover differential (a turnover-luck rebound; the "
        f"model's pick was not already on that side). {detail}. See "
        "docs/turnover_luck_rebound_tilt_overlay.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_turnover_luck_rebound_tilt_challenger_decisions(
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

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    pbp_snapshot = latest_pbp_snapshot(data_root / "pbp" / "raw")
    pbp = load_pbp_snapshot(pbp_snapshot)
    tilt = apply_turnover_luck_rebound_tilt_overlay(card, schedules, pbp)
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
        "both_flagged_game_ids": list(tilt.both_flagged_games),
    }


__all__ = [
    "CHALLENGER_ID",
    "TURNOVER_UNDER_Q25_THRESHOLD",
    "TiltFlip",
    "TiltResult",
    "apply_turnover_luck_rebound_tilt_overlay",
    "overlay_disclosure_note",
    "record_turnover_luck_rebound_tilt_challenger_decisions",
    "turnover_under_flag_by_game",
]
