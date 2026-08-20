"""Interim head-coach first-game tilt overlay: a parameter-free pick-level nudge.

Research chain (all measured 2026-08-20, read from ``registry/weak_signals.json``
and ``docs/interim_coach_screen.md`` before this module was built): a team's
FIRST REG-season game under a newly appointed interim head coach covers at
58.97% (n=39, 2009-2025), vs. 49.96% for the rest of the league --
``interim_hc_first_game`` in the registry, effect +0.0407 full-slate accuracy
points, week-blocked 95% ``[-0.0338, +0.1105]``, ``probability_positive``
0.8452 (season-blocked 0.8344). The interval crosses zero -- per AGENTS.md,
at this evaluator's ~2-point resolution that is the EXPECTED shape for a real
small signal, never grounds to decline building a no-window-cost prospective
challenger. Neither admissible closing ground applies (no resolved wrong
sign, no positive-control bound), so this stays ``unresolved_below_power`` in
the registry; wiring it here is an EV-positive dual-tracked play (P+ 0.845 >
0.5), not a claim of a proven edge (AGENTS.md "a promotion bar is not a
decision bar").

**The decomposition this cell rests on** (``docs/interim_coach_screen.md``
section 6): the whole-stint cell ``interim_hc_active`` (any game under an
interim, any point in the stint) is flat-to-slightly-negative (49.20% vs.
50.02%, P+ 0.386) -- the folklore does NOT hold broadly. Splitting by game
number within the stint tells a sharper story: game 1 only (n=39) covers
58.97%; games 2+ (n=211) cover only 47.39% -- BELOW the league baseline. So
this overlay fires ONLY on the first game of a stint, never on later games of
the same interim tenure, which is exactly where the (below-power) evidence
points and exactly where it stops pointing.

This module is the no-window-cost path, built on the exact pattern of
``surface_switch_tilt_overlay.py``, ``division_revenge_tilt_overlay.py``, and
``coach_fade_overlay.py`` (the original precedent): a **pick-level,
post-prediction transform** of the active model's own forced pick,
dual-tracked against that same active model in the prospective challenger
ledger (``nfl_ats.prospective_scoring``), at no rotation-registry window cost
and with zero training-time feature changes. **Nothing in this module is
wired into ``publishing.py`` or the production pick path** -- like the tilt
siblings, and unlike ``hc_year_one_fade_overlay``, no owner decision to play
this on the real card has been made; it is dual-tracked only.

**The rule is parameter-free**: whenever a team is in the FIRST REG-season
game of a newly appointed interim head-coach stint AND the model's own pick
does NOT already side with that team, flip the pick toward it (direction
implied by the measured 58.97% cover rate). No week restriction (unlike
``hc_year_one_fade_overlay``'s weeks 1-8 window) -- a mid-season firing can
land any week 1-18, and the registered cell carries no week-dependent claim.

**FAIL-OPEN, per this module's build task**: interim-coach status is derived
from :func:`nfl_ats.experiment_runner._build_interim_coach_trait_data`, a
local snapshot join (Pro Football Rumors' interim-coach list joined onto
``schedules.parquet``'s own per-game coach field -- see
``docs/interim_coach_screen.md`` sections 1-2 for the fetch/join provenance
and 6-of-6 spot-check cross-checks). If that join cannot run for any reason
(no interim-coaches snapshot fetched yet, a malformed source file, a missing
schedules snapshot), :func:`interim_first_game_flag_by_game_fail_open` catches
the exception, emits a ``RuntimeWarning``, and returns zero flags -- the
challenger simply falls back to the model's own pick for every game that
week, exactly mirroring ``forecast_cold_visitor_tilt_overlay``'s fail-open
live-fetch wrapper. This overlay must never be able to block a publish.

**Overlap with ``hc_year_one_fade_overlay`` -- explicitly checked, per
``docs/interim_coach_screen.md`` section 3.** ``hc_year_one_fade_overlay``
flags a team whose CURRENT-season coach differs from LAST season's (a
whole-season condition, active only weeks 1-8); THIS overlay flags a team
whose coach changed WITHIN the current season (an in-season firing/interim
appointment), any week. A genuinely early first-interim-game (weeks 1-8) will
almost always ALSO satisfy the year-one construct (the interim coach, by
definition, was not that team's coach last season either), so the two
overlays' eligible games can and do overlap -- measured: of the 39
``first_game`` team-games, 11 fall in weeks 1-8, of which 8 would also be
eligible for the other overlay's clean case. **Both overlays are dual-tracked
challengers only; neither is composed with the other.** Each transforms the
SAME un-overlaid base card independently and records its own arm to the
prospective challenger ledger -- this overlay never sees, and is never seen
by, ``hc_year_one_fade_overlay``'s flips, exactly the same independence
``spread_gap_zone_fade_overlay`` already documents for its own relationship
to that overlay. No precedence rule is needed for two dual-tracked arms that
never combine; a precedence rule would only become necessary if BOTH were
ever played on the real card simultaneously, which is not the case today (only
``hc_year_one_fade_overlay`` is).

Two things live here, mirroring the sibling overlays exactly:

1. :func:`interim_first_game_flag_by_game_fail_open` -- the pregame-safe,
   DATA-DERIVED, FAIL-OPEN signal, reusing
   ``nfl_ats.experiment_runner._build_interim_coach_trait_data`` verbatim
   (the exact join ``docs/interim_coach_screen.md`` validated), never
   hand-typed.
2. :func:`apply_interim_hc_first_game_tilt_overlay` -- the pick-level
   transform, plus :func:`overlay_disclosure_note` for the plain-English
   provenance sentence.

:func:`record_interim_hc_first_game_tilt_challenger_decisions` writes the
overlay's own arm to the prospective challenger ledger so 2026 scores it
cleanly, independent of whether it is ever played on the real card.

**Live availability**: interim-coach appointments are public news, announced
well before the team's next kickoff -- exactly as pregame-safe as the
already-wired ``hc_year_one_fade_overlay``. Week 1 of the 2026 season
trivially has zero interim coaches (every team starts the season with its
already-known Week 1 coach), so the first live evaluation opportunity for
this challenger arrives mid-season, whenever the first 2026 in-season firing
happens -- not a defect, just the honest shape of a rare event.
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

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "interim_hc_first_game_tilt_overlay"

#: One row per (game_id, team) with no interim-coach data available at all --
#: the empty, well-typed shape :func:`interim_first_game_flag_by_game_fail_open`
#: returns on a join failure, so every downstream merge sees a frame with the
#: right columns/dtypes rather than a bare empty DataFrame.
_EMPTY_FLAGS_COLUMNS = ("game_id", "team", "entry_id")


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def interim_first_game_flag_by_game_fail_open(repo_root: Path) -> pd.DataFrame:
    """One row per (``game_id``, ``team``): that team's FIRST REG-season game
    of a specific interim-head-coach stint, per
    ``nfl_ats.experiment_runner._build_interim_coach_trait_data``
    (``first_game_under_interim`` restricted to ``True``).

    **FAIL-OPEN**: any exception from the underlying join -- no
    ``data/raw/interim_coaches/*/parsed_table.csv`` snapshot fetched yet, a
    malformed source file, a missing schedules snapshot, an unresolved
    join-count mismatch -- is caught, logged as a ``RuntimeWarning``, and
    folded into "zero games flagged" (an empty-but-well-formed frame) so
    downstream flag computation naturally applies no tilt for the week rather
    than raising. Mirrors
    ``forecast_cold_visitor_tilt_overlay.fetch_tuesday_noon_forecast_temps_fail_open``'s
    fail-open contract exactly: this overlay must never be able to block a
    publish.
    """

    # Local import, mirroring _build_interim_coach_trait_data's own local
    # import of nfl_ats.coach_fade_overlay: keeps this module's import-time
    # footprint independent of experiment_runner's (a much heavier module),
    # and avoids any risk of a load-order cycle between the two.
    from nfl_ats.experiment_runner import _build_interim_coach_trait_data

    try:
        trait_data = _build_interim_coach_trait_data(repo_root)
    except Exception as exc:  # deliberate fail-open, see docstring
        warnings.warn(
            "interim_hc_first_game_tilt: interim-coach join failed, proceeding with zero "
            f"flags ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {column: pd.Series([], dtype=object) for column in _EMPTY_FLAGS_COLUMNS}
        )

    game_trait = trait_data.game_trait
    first_only = game_trait.loc[game_trait["first_game_under_interim"]].copy()
    first_only["game_id"] = first_only["game_id"].astype(str)
    first_only["team"] = _canonical_team(first_only["team"])
    return first_only[list(_EMPTY_FLAGS_COLUMNS)].reset_index(drop=True)


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    interim_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``surface_switch_tilt_overlay.TiltResult``.
    ``both_first_game_games`` lists games where BOTH teams happen to be in
    their own interim stint's first game simultaneously -- no measured
    direction for that case (mirrors ``coach_fade_overlay``'s
    ``both_year_one_games``), so those games are flagged, never flipped.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_first_game_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_interim_hc_first_game_tilt_overlay(
    predictions: pd.DataFrame,
    repo_root: Path,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick toward the interim-coached team's first game.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the registered
      measurement -- 250 REG-season, non-push team-games -- is a
      regular-season read);
    * exactly ONE side of the game is in the first game of an interim
      stint (a rare simultaneous case where BOTH sides qualify is left
      untouched -- see ``both_first_game_games``); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home)
      is NOT already on the interim-coached side.

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

    flags = interim_first_game_flag_by_game_fail_open(repo_root)

    merged = base.copy()
    merged["home_team_canon"] = _canonical_team(merged["home_team"])
    merged["away_team_canon"] = _canonical_team(merged["away_team"])

    if flags.empty:
        merged["home_first_game"] = False
        merged["away_first_game"] = False
    else:
        home_flags = flags.rename(columns={"team": "home_team_canon"})[
            ["game_id", "home_team_canon"]
        ].copy()
        home_flags["home_first_game"] = True
        merged = merged.merge(
            home_flags.drop_duplicates(subset=["game_id", "home_team_canon"]),
            on=["game_id", "home_team_canon"],
            how="left",
        )

        away_flags = flags.rename(columns={"team": "away_team_canon"})[
            ["game_id", "away_team_canon"]
        ].copy()
        away_flags["away_first_game"] = True
        merged = merged.merge(
            away_flags.drop_duplicates(subset=["game_id", "away_team_canon"]),
            on=["game_id", "away_team_canon"],
            how="left",
        )

        merged["home_first_game"] = merged["home_first_game"].fillna(False).astype(bool)
        merged["away_first_game"] = merged["away_first_game"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    both_first_game = merged["home_first_game"] & merged["away_first_game"]

    flip_to_home = eligible & merged["home_first_game"] & ~both_first_game & ~home_pick
    flip_to_away = eligible & merged["away_first_game"] & ~both_first_game & home_pick
    flip_mask = flip_to_home | flip_to_away

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        interim_is_home = bool(row["home_first_game"])
        interim_team = str(row["home_team"] if interim_is_home else row["away_team"])
        opponent_team = str(row["away_team"] if interim_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                interim_team=interim_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_first_game, "game_id"].astype(str))
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
        f"{flip.matchup}: {flip.opponent_team} -> {flip.interim_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the interim head-coach "
        "first-game tilt (a team playing its first game under a newly appointed interim head "
        "coach, and the model's own pick was not already on that side). "
        f"{detail}. See docs/interim_coach_screen.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_interim_hc_first_game_tilt_challenger_decisions(
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

    ``repo_root`` for the interim-coach join is derived as ``data_root.parent``
    -- every caller in this codebase already passes ``data_root`` as
    ``<repo_root>/data`` (see ``_data_root()`` in ``cli.py`` and every sibling
    ``record_*_challenger_decisions`` function's identical ``data_root``
    convention), so this holds without introducing a new CLI parameter.

    The interim-coach join is FAIL-OPEN (see
    :func:`interim_first_game_flag_by_game_fail_open`): a missing source
    snapshot or join failure never raises out of this function, it simply
    yields zero flags for the week.

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

    tilt = apply_interim_hc_first_game_tilt_overlay(card, data_root.parent)
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
        "both_first_game_games": list(tilt.both_first_game_games),
    }
