"""Grass-to-turf surface switch tilt overlay: a parameter-free pick-level nudge.

Research chain (mined lineage, all measured 2026-08-19, read from
``registry/weak_signals.json`` before this module was built):

1. **Weather battery cell.** ``weather_battery_surface_switch_grass_to_turf``
   -- one of 8 predeclared cells in the NFL weather/environment bias battery
   (``scripts/nfl_weather_battery_screen.py``, mined, uncorrected multiplicity)
   -- flags games where the AWAY team's modal home surface this season
   normalizes to grass AND this game's own surface normalizes to turf (a
   footing/speed mismatch). Week-blocked, REG 2009-2025, n=4,317 games:
   +1.1618 accuracy points, 95% [+0.2896, +2.038], ``probability_positive``
   0.995.
2. **Venue-controlled follow-up.** ``surface_familiarity_r1_turf_venue_visitor_split``
   isolates the same mechanism WITHIN turf-venue games only (holding venue
   fixed, so it cannot be confounded with "turf venues are just different"):
   grass-modal-home visitors vs. turf-modal-home visitors, both playing at a
   turf venue. Full-slate effect +1.4579 accuracy points, 95%
   [-0.4774, +3.3655], ``probability_positive`` 0.9332, n_pair=1,857
   (2009-2025). The gap SURVIVES holding venue fixed and is even larger than
   the parent cell's full-slate effect.
3. **CFB cross-league replication.** ``cfb_surface_familiarity_turf_venue_visitor_split``
   replicates the SAME venue-controlled construct on FBS games (2012-2025,
   n_pair=6,133): +1.5579 accuracy points, 95% [-0.6665, +3.7421],
   ``probability_positive`` 0.9156 -- same sign, a comparable (if anything
   larger) point estimate, cross-league, venue-controlled.

**All three intervals cross zero.** Per AGENTS.md, at this evaluator's
~2-point resolution that is the EXPECTED shape for a real small signal, never
grounds to decline building a no-window-cost prospective challenger. Neither
admissible closing ground applies to any of the three reads (no resolved
wrong sign, no positive-control bound), so all three remain
``unresolved_below_power`` in the registry.

**The symmetric mirror is null in BOTH leagues -- stated up front, not
buried.** If this were a clean bilateral "surface-switch cost" mechanism, the
GRASS-venue mirror (turf-modal visitors on grass, vs. grass-modal visitors)
should show the SAME-SIZED positive gap. It does not: NFL
``surface_familiarity_r2_grass_venue_mirror`` reads -0.4995 points,
``probability_positive`` 0.3205 (near a coin flip, leaning the wrong way);
CFB ``cfb_surface_familiarity_grass_venue_mirror`` reads -0.1218 points,
``probability_positive`` 0.4291 (also near a coin flip, also leaning the
wrong way). This is why the rule below is deliberately ASYMMETRIC -- it only
ever fires in the one direction (grass-modal visitor onto turf) that both
leagues' primary reads actually support, never the mirror direction that
neither league corroborates.

**Era caveat -- the NFL effect concentrates in 2018-2025.** The venue-
controlled follow-up's own era split
(``surface_familiarity_r3_era_2009_2017`` / ``surface_familiarity_r3_era_2018_2025``)
shows the SAME sign in both eras (sign-stable) but a roughly 4.5x larger
magnitude in the later half: 2009-2017 full-slate +0.5277 points,
``probability_positive`` 0.6482 (n=973); 2018-2025 full-slate +2.3869 points,
``probability_positive`` 0.9578 (n=884). Both eras stay ``unresolved_below_power``
-- this is reported as a caveat on the effect's stability, not as grounds to
restrict the overlay's eligible weeks (the overlay applies to all of 2026
regardless of era, since 2026 postdates both halves).

This module is the no-window-cost path, built on the exact pattern of
``backup_qb_fade_overlay.py``, ``division_revenge_tilt_overlay.py``,
``injury_value_tilt_overlay.py``, and ``coach_fade_overlay.py`` (the original
precedent): a **pick-level, post-prediction transform** of the active model's
own forced pick, dual-tracked against that same active model in the
prospective challenger ledger (``nfl_ats.prospective_scoring``), at no
rotation-registry window cost and with zero training-time feature changes.
**Nothing in this module is wired into ``publishing.py`` or the production
pick path** -- like the three tilt/fade siblings, and unlike the coach-fade
overlay, no owner decision to play this on the real card has been made; it is
dual-tracked only.

**The rule is parameter-free and frozen** -- no threshold, no tuning, nothing
derived from 2018-2025 outcomes: when the flag is set AND the active model's
own forced pick is the AWAY team, flip to the home team. REG season only
(every read above was scored on regular-season games); missing surface data
means no flip.

Two things live here, mirroring the sibling overlays exactly:

1. :func:`surface_switch_flag_by_game` -- the pregame-safe, DATA-DERIVED
   signal, ported VERBATIM from ``scripts/nfl_weather_battery_screen.py``'s
   ``_normalize_surface`` function and ``load_population``'s modal-home-
   surface derivation (same ``GRASS_SURFACES``/``TURF_SURFACES`` sets, same
   per-``(home_team, season)`` modal-surface aggregation, same
   ``away_modal_surface == "grass" and surface_norm == "turf"`` flag
   definition), read straight from the newest local schedule snapshot
   (``data/raw/<snapshot>/schedules.parquet``), never hand-typed.
2. :func:`apply_surface_switch_tilt_overlay` -- the pick-level transform, plus
   :func:`overlay_disclosure_note` for the plain-English provenance sentence.

:func:`record_surface_switch_tilt_challenger_decisions` writes the overlay's
own arm to the prospective challenger ledger so 2026 scores it cleanly,
independent of whether it is ever played on the real card.
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
CHALLENGER_ID = "surface_switch_tilt_overlay"

#: Ported verbatim from scripts/nfl_weather_battery_screen.py's own module
#: constants -- the same surface-name normalization the registry measurement
#: used. Kept exactly, not re-derived.
GRASS_SURFACES = frozenset({"grass", "dessograss"})
TURF_SURFACES = frozenset(
    {"fieldturf", "sportturf", "matrixturf", "astroturf", "a_turf", "astroplay"}
)

#: The merged-in flag travels under this module-private name. The model's own
#: feature table now carries ``surface_switch_flag`` as an input (features.py
#: ports this exact derivation), so the predictions frame can arrive with a
#: same-named column -- a bare left-merge would suffix both copies to
#: ``_x``/``_y`` and the old ``merged["surface_switch_flag"]`` read raised
#: KeyError on both production call sites (2026-08-24 rehearsal). This module
#: always derives its own flag from the schedules snapshot; a foreign column
#: of the same name is neither consumed nor overwritten.
OVERLAY_FLAG_COLUMN = "_surface_switch_tilt_flag"


def _normalize_surface(raw: object) -> str | None:
    """Ported verbatim from ``scripts/nfl_weather_battery_screen.py::_normalize_surface``.

    Lower-cases and strips whitespace (schedules carries at least one raw
    ``"grass "`` value with a trailing space) before matching against the
    frozen surface sets; anything unmatched (including the empty string and
    non-string/NaN values) normalizes to ``None`` -- "missing surface data",
    which :func:`surface_switch_flag_by_game` folds into "not flagged".
    """

    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in GRASS_SURFACES:
        return "grass"
    if value in TURF_SURFACES:
        return "turf"
    return None


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def surface_switch_flag_by_game(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``surface_switch_flag``.

    ``surface_switch_flag`` fires when the AWAY team's modal home surface
    THIS SEASON normalizes to grass AND this game's OWN surface normalizes
    to turf -- ported verbatim from
    ``scripts/nfl_weather_battery_screen.py``'s ``load_population`` (the
    ``away_modal_surface`` merge and the cell-6 flag definition in
    ``build_cells``): for every ``(home_team, season)``, the MODE of that
    team's normalized home-game surface across the FULL regular season
    (``s.mode().iat[0] if not s.mode(dropna=True).empty else None``, the
    exact aggregation the source script uses), then looked up for each
    game by its AWAY team and season.

    **Why a full-season aggregate is pregame-safe here, unlike the coach and
    QB overlays' strictly-prior-only aggregates**: a team's home-stadium
    surface is a STRUCTURAL, stadium-level fact fixed for essentially the
    entire season and public knowledge before Week 1 -- it is not an
    outcome, and this function never reads ``result`` or ``spread_line`` at
    all. The source script's own comment makes the same point: "roof/surface
    is a stadium fact, not a cover outcome". Two leakage regression tests
    (``tests/test_surface_switch_tilt_overlay.py``) prove this empirically:
    mutating a game's own ``result``/outcome columns has no bearing (the
    function does not even read them), and a future season's surface data
    never changes an earlier season's already-computed flags.

    Team codes are canonicalized (``TEAM_ABBREVIATION_ALIASES``) before the
    ``(home_team, season)`` grouping and the away-team lookup -- a merge-
    safety measure so this function's output joins cleanly against the
    predictions frame's own team codes; it does not change which surface a
    team's home games in a given season actually used, so it does not alter
    the measured construct itself.
    """

    required = {"game_id", "season", "game_type", "home_team", "away_team", "surface"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for surface-switch tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    reg["surface_norm"] = reg["surface"].map(_normalize_surface)

    modal_surface = (
        reg.groupby(["home_team", "season"])["surface_norm"]
        .agg(lambda s: s.mode().iat[0] if not s.mode(dropna=True).empty else None)  # type: ignore[type-var]
        .rename("away_modal_surface")
    )
    frame = reg[["game_id", "season", "away_team", "surface_norm"]].merge(
        modal_surface, left_on=["away_team", "season"], right_index=True, how="left"
    )
    frame["surface_switch_flag"] = frame["away_modal_surface"].eq("grass") & frame[
        "surface_norm"
    ].eq("turf")
    return frame[["game_id", "season", "surface_switch_flag"]]


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    grass_modal_visitor: str
    turf_venue_home: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``coach_fade_overlay.OverlayResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_surface_switch_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick from AWAY to HOME wherever the flag fires.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (every measured
      read -- the weather battery cell, the venue-controlled follow-up, the
      CFB replication -- was scored on regular-season games only);
    * :func:`surface_switch_flag_by_game` fires for the game (the AWAY
      team's modal home surface this season is grass AND this game's own
      surface is turf); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home)
      is currently on the AWAY side.

    **Deliberately ASYMMETRIC, unlike the sibling tilt overlays**: this
    never flips a HOME pick to AWAY when the flag fires, because the
    measured evidence never supports that direction -- the grass-venue
    mirror (the symmetric case) is near-null in both leagues (NFL
    ``probability_positive`` 0.3205, CFB 0.4291), so there is no measured
    direction to fade INTO the flagged side, only away from picking the
    grass-modal visitor on turf.

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    the sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch.

    The flag is ALWAYS this module's own schedules-derived
    :func:`surface_switch_flag_by_game` output, merged under the private
    :data:`OVERLAY_FLAG_COLUMN` name: a predictions frame that already
    carries a same-named ``surface_switch_flag`` column (the feature table
    ports this exact derivation as a model input) collides silently instead
    of crashing, and if no flag column survives the merge at all the result
    is the documented no-op -- zero flips, never a KeyError.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    flags = surface_switch_flag_by_game(schedules)
    merged = base.merge(
        flags.rename(columns={"surface_switch_flag": OVERLAY_FLAG_COLUMN}),
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    if OVERLAY_FLAG_COLUMN not in merged.columns:
        # Absence path: nothing to read means nothing flips -- the documented
        # no-op (missing surface data never flips), never a KeyError.
        merged[OVERLAY_FLAG_COLUMN] = False
    merged[OVERLAY_FLAG_COLUMN] = merged[OVERLAY_FLAG_COLUMN].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    away_pick = merged["home_cover_probability"].lt(0.5)
    flip_mask = eligible & merged[OVERLAY_FLAG_COLUMN] & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                grass_modal_visitor=str(row["away_team"]),
                turf_venue_home=str(row["home_team"]),
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
        f"{flip.matchup}: {flip.grass_modal_visitor} -> {flip.turf_venue_home}"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the grass-to-turf "
        "surface switch (the model's pick was on the away team, whose modal home surface "
        "this season is grass, playing on turf this week). "
        f"{detail}. See docs/surface_switch_tilt_overlay.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_surface_switch_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors ``division_revenge_tilt_overlay.record_division_revenge_tilt_challenger_decisions``
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
    tilt = apply_surface_switch_tilt_overlay(card, schedules)
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
