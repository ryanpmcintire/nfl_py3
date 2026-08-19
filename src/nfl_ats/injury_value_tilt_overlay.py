"""Injury value-lost tilt overlay: a parameter-free pick-level nudge.

Research chain: ``docs/injury_value_lost.md`` isolates
``injury_value_lost_narrowed`` -- the ``player_value`` profile's two
value-lost diff columns, fixed-prior severity, zero semantics-shift confound
-- at +1.316 accuracy points, ``probability_positive`` 0.8875 on the
already-spent 456-game ``[2020, 2021]`` opener window (``unresolved_below_power``:
the interval crosses zero, which AGENTS.md says is the EXPECTED shape for a
real small signal, not grounds to close the line). That document's own
predeclaration (section 7) explicitly defers spending the next NFL opener
window (``[2022, 2023]``, one of only two left in the project) until the free,
zero-cost 2026 prospective evidence for ``mod07_weak_signal_stack`` has
accrued enough weeks to be informative -- so this module does NOT spend a
window and does NOT touch the production pick path.

This is the no-window-cost path instead, built on the same pattern as
``coach_fade_overlay.py`` (year-1 head-coach fade) and
``best_pick_nomination.py`` (nomination v2): a **pick-level, post-prediction
transform** of the active model's own forced pick, dual-tracked against that
same active model in the prospective challenger ledger
(``nfl_ats.prospective_scoring``), at no rotation-registry window cost and
with zero training-time feature changes.

**The rule is parameter-free** -- no threshold, no tuning, nothing derived
from 2018-2025 outcomes. It reads the SAME two pregame-available columns
``docs/injury_value_lost.md`` section 4 isolated
(``diff_injury_skill_epa_value_lost`` + ``diff_injury_defense_disruption_value_lost``,
also named in ``nfl_ats.surgical_gating.VALUE_LOST_DIFF_COLUMNS`` -- imported
from there rather than re-declared, so the two modules can never drift on
which columns define the construct) from ``data/processed/game_features_player.parquet``
(the canonical, weekly-rebuilt, FIXED-prior-severity table -- manifest
``player_feature_version: "v2"`` means no learned-availability table was
passed at build time, matching the exact isolation the registry entry
measured). ``diff_X = home_X - away_X`` (``features.py``'s convention), so a
positive total means the HOME team lost strictly more value than the away
team.

The tilt: when the active model's own forced pick sits on the side that lost
STRICTLY MORE injury value than its opponent -- i.e. the model's pick
disagrees with "tilt toward the side with less injury value lost" -- the
overlay flips the pick to the healthier side. Ties (the differential is
exactly zero, the common case: injuries are sparse) are left untouched, since
a zero differential carries no directional information at all.

Two things live here, mirroring ``coach_fade_overlay.py`` exactly:

1. :func:`raw_value_lost_diff` -- the pregame-safe, DATA-DERIVED signal, read
   straight from the already-built feature table, never hand-typed.
2. :func:`apply_injury_value_tilt_overlay` -- the pick-level transform, plus
   :func:`overlay_disclosure_note` for the plain-English provenance sentence.

:func:`record_injury_value_tilt_challenger_decisions` writes the overlay's
own arm to the prospective challenger ledger so 2026 scores it cleanly,
independent of whether it is ever played on the real card. **Nothing in this
module is wired into ``publishing.py`` or the real card path** -- unlike
``coach_fade_overlay`` (an owner decision to play weeks 1-8 for real), no
such decision has been made for this candidate, and the task that built this
module was explicit: dual-track it, do not touch the production pick.
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
from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "injury_value_lost_tilt_overlay"

#: The canonical, fixed-prior-severity player feature table, rebuilt every
#: Tuesday by weekly-run's step 3 (`build-player-features`) regardless of
#: which profile the card path is currently using -- so this table is always
#: fresh for the current week even while `weak_stack` is the active profile.
#: Matches `nfl_ats.weekly.PLAYER_FEATURE_TABLE`; not imported from there to
#: avoid a dependency on the weekly-ops module from a research overlay.
PLAYER_FEATURE_TABLE_NAME = "game_features_player.parquet"


def raw_value_lost_diff(features: pd.DataFrame) -> pd.DataFrame:
    """One row per ``game_id``: the signed, pregame-available value-lost total.

    ``diff_total = diff_injury_skill_epa_value_lost + diff_injury_defense_disruption_value_lost``,
    each already ``home - away`` (``features.py``'s ``diff_`` convention).
    Positive means the HOME team lost strictly more value; negative means the
    AWAY team did; zero (the common case -- injuries are sparse) carries no
    signal. Reads the same ``VALUE_LOST_DIFF_COLUMNS`` the surgical gate uses,
    so the two modules can never silently disagree on the construct.
    """

    required = {"game_id", *VALUE_LOST_DIFF_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing value-lost columns: {', '.join(missing)}")

    total = pd.Series(0.0, index=features.index, dtype=float)
    for column in VALUE_LOST_DIFF_COLUMNS:
        total = total + pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return pd.DataFrame({"game_id": features["game_id"].astype(str), "value_lost_diff": total})


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    hurt_team: str
    healthier_team: str
    value_lost_diff: float


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


def apply_injury_value_tilt_overlay(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick wherever it sits on the strictly-more-hurt side.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the construct's
      split-half reliability and mechanism screen were both measured on
      regular-season games only -- ``docs/injury_value_lost.md`` section 3.1);
    * the value-lost differential for the game is strictly nonzero (a tie
      carries no directional information, and injuries are sparse -- 29.86%
      of games have an exactly-zero differential per
      ``docs/surgical_injury.md`` section 1.2); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home)
      lands on the side that lost STRICTLY MORE value than its opponent --
      i.e. the model's pick disagrees with "tilt toward the side with less
      injury value lost".

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    ``coach_fade_overlay.apply_coach_fade_overlay`` does, so every existing
    reader of the column needs no overlay-aware branch.
    """

    required = {"game_id", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    diff = raw_value_lost_diff(features)
    merged = base.merge(diff, on="game_id", how="left", validate="one_to_one")
    merged["value_lost_diff"] = merged["value_lost_diff"].fillna(0.0)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    home_lost_more = merged["value_lost_diff"].gt(0.0)
    away_lost_more = merged["value_lost_diff"].lt(0.0)

    flip_mask = eligible & ((home_pick & home_lost_more) | (~home_pick & away_lost_more))

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(row["home_cover_probability"] >= 0.5)
        hurt_team = str(row["home_team"] if row_home_pick else row["away_team"])
        healthier_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                hurt_team=hurt_team,
                healthier_team=healthier_team,
                value_lost_diff=float(row["value_lost_diff"]),
            )
        )

    return TiltResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: TiltResult) -> str:
    """Plain-language provenance sentence, mirroring
    ``coach_fade_overlay.overlay_disclosure_note``.

    Empty when the overlay is off or changed nothing this week. Not currently
    surfaced on the published card -- this overlay is dual-tracked only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.hurt_team} -> {flip.healthier_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the injury "
        "value-lost tilt (the model's pick sat on the side that lost strictly more "
        f"injury value than its opponent). {detail}. See "
        "docs/injury_value_lost_tilt_overlay.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_injury_value_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors ``coach_fade_overlay.record_overlay_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against (a
    promotion under the challenger's feet must not silently convert into
    "prospective evidence" for the tilt).

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

    feature_path = data_root / "processed" / PLAYER_FEATURE_TABLE_NAME
    if not feature_path.is_file():
        raise ValueError(f"Player feature table is not built yet: {feature_path}")
    features = pd.read_parquet(feature_path, columns=["game_id", *VALUE_LOST_DIFF_COLUMNS])

    tilt = apply_injury_value_tilt_overlay(card, features)
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
