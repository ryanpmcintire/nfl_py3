"""Third-down mean-reversion fade overlay: a parameter-free pick-level flip.

Research chain (mined lineage; measured 2026-08-21 by
``scripts/redzone_reversion_screen.py``, predeclared in
``docs/redzone_reversion_screen.md`` cell C3, and read out of
``artifacts/redzone_reversion_screen/20260821T181025Z/results.json`` and
``registry/weak_signals.json`` before this module was built):

``redzone_reversion_c3_third_down_over_fade`` flags a team whose PRIOR-season
centered 3rd-down conversion rate sits in the GLOBAL top quartile (pooled
across every 2009-2025 team-season, not recomputed per season -- see "The
frozen threshold" below), and predicts that team FADES (under-covers) the
following season. Population: NFL REG close-graded slate 2009-2025,
team-perspective long table, n=8,634 team-games (read,
``artifacts/redzone_reversion_screen/20260821T181025Z/results.json:196``).
Week-blocked primary: full-slate effect **+0.36652412950519364 accuracy
points**, 95% **[-0.2586740547946662, +0.9990199709809606]**,
``probability_positive`` **0.87185**, 294 week blocks (read, results.json
:191-197). Season-blocked secondary: **+0.36652412950519364**, 95%
**[-0.25761982268609834, +0.9651280242694]**, ``probability_positive``
**0.87135**, 17 season blocks (read, results.json:171-177). Registered in
``registry/weak_signals.json`` under
``redzone_reversion_c3_third_down_over_fade``: effect +0.36652412950519364,
week-blocked 95% [-0.2587, +0.999], ``probability_positive`` 0.87185,
**reliability 0.407** (trait year-over-year Pearson +0.407, 95%
[+0.337, +0.473], read results.json:52-61, n=512 team-season pairs),
n=8634 team-games, sample_blocks 294, seasons 2009-2025, category
``onfield``, classification ``unresolved_below_power``.

**The interval crosses zero.** Per AGENTS.md, at this evaluator's ~2-point
resolution that is the EXPECTED shape for a real-but-small signal and is
NEVER grounds to decline building a no-window-cost prospective challenger.
Neither admissible closing ground applies (no resolved wrong sign, no
positive-control bound), so the cell stays ``unresolved_below_power`` in the
registry; wiring it here is an EV-positive dual-tracked play (P+ 0.87185 >
0.5), not a claim of a proven edge.

**Direction check, verified from the artifact, not assumed.** Measured,
``artifacts/redzone_reversion_screen/20260821T181025Z/results.json`` cell
``third_down_over_fade``: ``sign_dir`` is ``-1`` (FADE -- predicted NEGATIVE
on ``team_covered``; see results.json:181), ``subset_mean``
**0.4884947267497603** (read, results.json:199) and ``complement_mean``
**0.503665241295052** (read, results.json:188). Flagged (prior-season elite
third-down) teams covered **48.85%** against a **50.37%** field -- the
predeclared FADE is exactly what the data shows. A sibling cell in this same
mined battery failed this exact check and was dropped from the batch; this
one passes.

**Shared-trait mirror, stated up front, not buried.** The registered cell's
own note flags this: ``third_down_under_rebound`` (results.json cell name
``third_down_under_rebound``, ``sign_dir`` 1, predicting POSITIVE on
``team_covered``) reads ``full_slate_effect_pts`` **-0.3564386470499031**,
``probability_positive`` **0.09885** (week-blocked; read, results.json:237,
243) -- the mirror's OWN prediction is CONTRADICTED (bottom-quartile teams
did not rebound; if anything they under-covered too). Both cells key off the
SAME underlying trait (prior-season centered 3rd-down conversion rate) split
at the SAME two tails of the SAME panel, so they are ONE signal read from two
ends, not two independent votes -- exactly the "mirror c4 shares trait, not
independent" caveat already in the registry entry. This module builds ONLY
the C3 (top-quartile fade) side; nothing here ever reads the bottom quartile,
and this signal must never be pooled with a hypothetical bottom-quartile
challenger as if the two were independent.

## The trait, transcribed VERBATIM (not re-derived)

Cited from ``scripts/redzone_reversion_screen.py``, the module this cell was
measured by (not importable as a library -- it is a standalone CLI with its
own ``sys.path`` hacks -- so the construction below is PORTED, exactly as
``pbp08_matchup_flags.py`` ports ``pbp08_matchup_screen.py``'s construction
rather than importing the screen):

1. **Third-down conversion rate**, per (season, team): every play with
   ``down == 3.0`` (``build_efficiency_panels``, line 126) is grouped by
   ``(season, posteam)``; ``n_third_downs`` is the play count and
   ``third_conversions`` is the sum of ``first_down`` (lines 127-132);
   ``third_down_conv_rate = third_conversions / n_third_downs`` (lines
   133-135). Plays are ``nfl_ats.pbp.analysis_plays``' documented v1
   efficiency filter (real scrimmage plays with an offense, EPA and win
   probability; no kneels/spikes/aborted/no-plays), REG season only (line
   94), team codes canonicalized via ``TEAM_ABBREVIATION_ALIASES`` (line 97).
2. **"Centered"** means: subtract that SEASON's own cross-team mean rate --
   ``league_mean = offense.groupby("season")[trait].transform("mean")`` then
   ``offense[f"{trait}_centered"] = offense[trait] - league_mean`` (lines
   151-153). A team's centered value is its OWN rate minus the average of
   every team THAT SAME SEASON -- never a rate compared across seasons or
   to a fixed league-wide historical average.
3. **The top-quartile cut is GLOBAL, not within-season.** ``thresholds =
   {..., "third_down_q75": float(offense["third_down_conv_rate_centered"]
   .quantile(0.75)), ...}`` (line 383) takes the 0.75 quantile of the ENTIRE
   pooled ``offense`` panel -- every team-season 2009-2025 at once -- not a
   quantile recomputed separately inside each season. One number, drawn from
   the whole panel, applied to every season alike.
4. **Prior-season lookup.** ``_prior`` (lines 191-196) shifts a team-season's
   row forward one season (``season = season + 1``) before joining onto the
   schedule by ``(team, season)`` -- so a game in season *S* reads the
   centered rate the team posted in season *S-1*, never season *S* itself.

## The frozen threshold (an underived constant would be a defect)

The measured GLOBAL top-quartile threshold is **0.03392624406886406** (read,
``artifacts/redzone_reversion_screen/20260821T181025Z/results.json:349``,
``thresholds.third_down_q75``; the same value
``scripts/redzone_reversion_screen.py:383`` computed). This module FREEZES
that exact value as :data:`THIRD_DOWN_TOP_QUARTILE_CENTERED` rather than
recomputing a quantile live, for two reasons, mirroring
``spread_gap_zone_fade_overlay.SPREAD_GAP_LOWER_BOUND`` /
``SPREAD_GAP_UPPER_BOUND``'s identical choice:

* **Pregame safety by construction.** The screen's own quantile is GLOBAL
  across the whole 2009-2025 panel (point 3 above) -- recomputing it live
  from an expanding, season-by-season pool would silently change the cutoff
  every season and would no longer be the measured cell AGENTS.md requires
  ("every overlay parameter must be the registry cell's own measured value,
  cited"). A frozen constant carries zero risk of ever reading a future
  season's data, by construction.
* **Parameter-free, not re-derived.** Nothing here is fit, tuned, or
  selected on 2026 outcomes -- the threshold is the registry cell's own
  number, transcribed with its citation, exactly as
  ``spread_gap_zone_fade_overlay`` transcribes its 7.5/10.0 bucket bounds
  "verbatim and adds no threshold of its own."

**The rule is parameter-free and frozen** -- no threshold tuning, nothing
fitted to outcomes. REG season only (every read above is a regular-season
measurement). Build the top-quartile prior-season-centered-3rd-down flag for
BOTH teams in a game. If EXACTLY ONE of the two teams is flagged AND the
active model's own forced pick IS that team, flip the pick to the other
side. Both-flagged games are never touched -- the same clean-case handling
``coach_fade_overlay``/``tank_zone_fade_tilt_overlay`` use: no measured
direction when both sides carry the flag. Never flip in any other
situation.

**Pregame-safe by construction, structurally, not just by convention.** The
trait is a PRIOR-SEASON aggregate -- fully known before Week 1 of the season
being flagged, since it depends only on plays from a season that has already
finished. Two leakage regression tests in
``tests/test_third_down_reversion_fade_overlay.py`` prove this empirically:
mutating a game's own current-season PBP/outcome data never changes its
flag, and a later season's PBP data never changes an earlier season's
already-computed flags.

This module is the no-window-cost path, built on the exact pattern of
``coach_fade_overlay.py`` (clean-case both-flagged handling) and
``tank_zone_fade_tilt_overlay.py`` / ``pbp08_protection_mismatch_tilt_overlay.py``
(a PBP-derived team-trait flag): a **pick-level, post-prediction transform**
of the active model's own forced pick, dual-tracked against that same active
model in the prospective challenger ledger (``nfl_ats.prospective_scoring``),
at no rotation-registry window cost and with zero training-time feature
changes. **Nothing in this module is wired into ``publishing.py`` or the
production pick path** -- no owner decision to play this on the real card
has been made; it is dual-tracked only.

Two things live here, mirroring the sibling overlays exactly:

1. :func:`third_down_over_flag_by_game` -- the pregame-safe, DATA-DERIVED
   signal, porting VERBATIM the trait/centering construction above.
2. :func:`apply_third_down_reversion_fade_overlay` -- the pick-level
   transform, plus :func:`overlay_disclosure_note` for the plain-English
   provenance sentence.

:func:`record_third_down_reversion_fade_challenger_decisions` writes the
overlay's own arm to the prospective challenger ledger so 2026 scores it
cleanly, independent of whether it is ever played on the real card.
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
from nfl_ats.pbp import analysis_plays, latest_pbp_snapshot, load_pbp_snapshot
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
CHALLENGER_ID = "third_down_reversion_fade_overlay"

#: The registered cell's own GLOBAL pooled top-quartile cutoff for the
#: prior-season centered 3rd-down conversion rate -- measured, read
#: artifacts/redzone_reversion_screen/20260821T181025Z/results.json:349
#: (thresholds.third_down_q75), the same value
#: scripts/redzone_reversion_screen.py:383 computed. Frozen, not
#: recomputed live -- see the module docstring's "The frozen threshold".
THIRD_DOWN_TOP_QUARTILE_CENTERED = 0.03392624406886406

#: The merged-in flags travel under module-private names so a predictions
#: frame carrying same-named columns collides with neither -- the same
#: defensive naming ``surface_switch_tilt_overlay.OVERLAY_FLAG_COLUMN``
#: adopted after a 2026-08-24 KeyError rehearsal.
HOME_FLAG_COLUMN = "_third_down_reversion_fade_home"
AWAY_FLAG_COLUMN = "_third_down_reversion_fade_away"

_REQUIRED_SCHEDULE_COLUMNS = frozenset({"game_id", "season", "game_type", "home_team", "away_team"})


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _third_down_conv_rate_centered_by_team_season(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team): the centered 3rd-down conversion rate.

    Ported VERBATIM from ``scripts/redzone_reversion_screen.py``'s
    ``build_efficiency_panels`` -- the 3rd-down leg only (that function's
    red-zone and defense legs are irrelevant to this cell and are not
    reproduced here):

    * third-down plays: ``down == 3.0`` (line 126), REG season only (line
      94), ``nfl_ats.pbp.analysis_plays``' documented v1 filter, team codes
      canonicalized (line 97);
    * ``n_third_downs`` / ``third_conversions`` / ``third_down_conv_rate``
      (lines 127-135);
    * "centered": each team-season's rate minus THAT SEASON's own cross-team
      mean (lines 151-153) -- never a cross-season comparison.
    """

    plays = analysis_plays(pbp)
    plays = plays.loc[plays["season_type"].astype(str).eq("REG")].copy()
    plays["season"] = pd.to_numeric(plays["season"], errors="raise").astype(int)
    plays["posteam"] = _canonical_team(plays["posteam"])
    plays["down"] = pd.to_numeric(plays["down"], errors="coerce")
    plays["first_down"] = pd.to_numeric(plays["first_down"], errors="coerce")

    third = plays.loc[plays["down"].eq(3.0)]
    off_third = (
        third.groupby(["season", "posteam"], sort=False)
        .agg(n_third_downs=("play_id", "size"), third_conversions=("first_down", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off_third["third_down_conv_rate"] = off_third["third_conversions"] / off_third[
        "n_third_downs"
    ].replace(0, np.nan)

    league_mean = off_third.groupby("season")["third_down_conv_rate"].transform("mean")
    off_third["third_down_conv_rate_centered"] = off_third["third_down_conv_rate"] - league_mean
    return off_third[["season", "team", "third_down_conv_rate_centered"]]


def third_down_over_flag_by_game(schedules: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per REG ``game_id``: ``third_down_over_home`` / ``third_down_over_away``.

    A side is flagged when its PRIOR season's centered 3rd-down conversion
    rate is at or above :data:`THIRD_DOWN_TOP_QUARTILE_CENTERED` -- the
    registry cell's own GLOBAL pooled cutoff (see the module docstring's
    "The frozen threshold"). "Prior season" is looked up by shifting the
    team-season panel forward one season before joining (ported from
    ``_prior``, ``scripts/redzone_reversion_screen.py:191-196``), so a game
    in season *S* only ever reads season *S-1*'s data.

    **Pregame-safe by construction.** This never reads the flagged game's
    own outcome, spread, or any CURRENT-season play -- only the PRIOR
    season's league-wide third-down plays, which are complete and public
    well before the current season's Week 1. A team with no observed PRIOR
    season in ``pbp`` (first year in the data, an expansion team, or any gap
    year) or with a missing/NaN centered rate is left UNFLAGGED, never
    raising. Two leakage regression tests in
    ``tests/test_third_down_reversion_fade_overlay.py`` prove both
    properties empirically (a game's own data never moves its flag; a later
    season's data never moves an earlier season's already-computed flag).
    """

    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for third-down reversion tracking: {', '.join(missing)}"
        )

    panel = _third_down_conv_rate_centered_by_team_season(pbp)
    prior = panel.copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"third_down_conv_rate_centered": "prior_centered_rate"})

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = pd.to_numeric(reg["season"], errors="coerce").astype(int)

    home_prior = prior.rename(
        columns={"team": "home_team", "prior_centered_rate": "home_prior_rate"}
    )
    away_prior = prior.rename(
        columns={"team": "away_team", "prior_centered_rate": "away_prior_rate"}
    )

    frame = reg[["game_id", "season", "home_team", "away_team"]].merge(
        home_prior, on=["home_team", "season"], how="left"
    )
    frame = frame.merge(away_prior, on=["away_team", "season"], how="left")

    frame["third_down_over_home"] = (
        frame["home_prior_rate"].ge(THIRD_DOWN_TOP_QUARTILE_CENTERED).fillna(False).astype(bool)
    )
    frame["third_down_over_away"] = (
        frame["away_prior_rate"].ge(THIRD_DOWN_TOP_QUARTILE_CENTERED).fillna(False).astype(bool)
    )
    return frame[["game_id", "season", "third_down_over_home", "third_down_over_away"]]


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
    byte-identical, mirroring ``surface_switch_tilt_overlay.TiltResult``.
    ``both_flagged_games`` lists eligible games where BOTH sides carry the
    flag; there is no measured direction for that case (mirroring
    ``coach_fade_overlay``'s ``both_year_one_games``), so those games are
    reported, never flipped.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_flagged_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_third_down_reversion_fade_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Fade the top-quartile-third-down team, and only in the clean case.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (every read behind
      this cell is a regular-season measurement);
    * EXACTLY ONE side of the game carries the flag (a both-flagged game has
      no measured direction and is reported in ``both_flagged_games``
      instead of flipped, mirroring ``coach_fade_overlay``'s clean-case
      handling); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) IS
      that flagged side.

    Deliberately one-directional: the overlay never flips a pick TOWARD a
    flagged team, because the measured evidence is a fade -- flagged teams
    covered 48.85% against a 50.37% complement (see the module docstring's
    direction check).

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    the sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch. A game with no schedule row, or with no flag after
    the merge, is the documented no-op -- zero flips, never a ``KeyError``.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not enabled:
        return TiltResult(base, (), (), enabled)

    flags = third_down_over_flag_by_game(schedules, pbp).rename(
        columns={
            "third_down_over_home": HOME_FLAG_COLUMN,
            "third_down_over_away": AWAY_FLAG_COLUMN,
        }
    )
    merged = base.merge(flags, on=["game_id", "season"], how="left", validate="one_to_one")
    for column in (HOME_FLAG_COLUMN, AWAY_FLAG_COLUMN):
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = pd.to_numeric(merged["home_cover_probability"], errors="coerce").ge(0.5)
    both_flagged = merged[HOME_FLAG_COLUMN] & merged[AWAY_FLAG_COLUMN]
    picked_is_flagged = merged[HOME_FLAG_COLUMN].where(home_pick, merged[AWAY_FLAG_COLUMN])

    flip_mask = eligible & picked_is_flagged & ~both_flagged

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(float(row["home_cover_probability"]) >= 0.5)
        flagged_team = str(row["home_team"] if row_home_pick else row["away_team"])
        opponent = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                flagged_team=flagged_team,
                opponent_team=opponent,
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
        f"{flip.matchup}: {flip.flagged_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the third-down "
        "mean-reversion fade (the model sided with a team whose PRIOR season's centered "
        "3rd-down conversion rate was in the league's top quartile, against an opponent "
        f"that was not). {detail}. See docs/third_down_reversion_fade_overlay.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_third_down_reversion_fade_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the fade overlay's picks to the prospective challenger ledger.

    Mirrors ``tank_zone_fade_tilt_overlay.record_tank_zone_fade_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast
    rather than searching ``artifacts/margin_predictions/`` by fingerprint,
    and it refuses to record if the active model's live fingerprint no
    longer matches the snapshot this challenger was registered against.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this
    challenger tracks the fade's forced-pick (``decision_line``) accuracy
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
            "No synchronized active ATS model is available to record fade decisions from"
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
            "changed underneath this fade -- re-register before recording"
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
    tilt = apply_third_down_reversion_fade_overlay(card, schedules, pbp)
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
    "AWAY_FLAG_COLUMN",
    "CHALLENGER_ID",
    "HOME_FLAG_COLUMN",
    "THIRD_DOWN_TOP_QUARTILE_CENTERED",
    "TiltFlip",
    "TiltResult",
    "apply_third_down_reversion_fade_overlay",
    "overlay_disclosure_note",
    "record_third_down_reversion_fade_challenger_decisions",
    "third_down_over_flag_by_game",
]
