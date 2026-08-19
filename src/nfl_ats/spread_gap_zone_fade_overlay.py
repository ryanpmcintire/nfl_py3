"""Spread-gap-zone fade overlay: a parameter-free pick-level nudge.

Research chain (read from ``registry/weak_signals.json:pick_conditioned_spread_gap_zone_pre2018``
before this module was built, 2026-08-19): the active model's own forced
picks (``home_cover_probability >= 0.5``), restricted to games where
``7.5 < abs(spread_line) <= 10.0``, were mined at 45.96% accuracy on the
opener line (2020-2025, n=198) and 47.97% on the close line (2018-2025,
n=271) -- BOTH below the model's own overall accuracy, in the SAME
direction. A predeclared, never-mined replication on 2011-2017 walk-forward
picks (before either mined window; the 7.5/10.0 bucket bounds were FROZEN
before this replication ran, exactly as stated in the registry entry's
description) reads 46.22% accuracy (n_bucket=238 of 1,743 scored games),
week-blocked 95% [-2.156, +11.698] accuracy points (hypothesis-signed:
positive means the replication agrees with the mined "lower" direction),
``probability_positive`` 0.9135 for the mined direction. **Three windows
(2011-2017 replication, 2018-2025 close, 2020-2025 opener), all the SAME
direction** -- the model underperforms its own baseline specifically inside
this spread-gap zone. The interval crosses zero; per AGENTS.md that is the
EXPECTED shape for a real small signal at this evaluator's resolution, never
grounds to decline building a no-window-cost prospective challenger.

**Frozen thresholds, not tuned on the replication.** The registry entry's
own description states the 7.5/10.0 bucket bounds were fixed BEFORE the
2011-2017 pre-2018 replication ran; this module reuses those same two
numbers verbatim (:data:`SPREAD_GAP_LOWER_BOUND`, :data:`SPREAD_GAP_UPPER_BOUND`)
and adds no threshold of its own. The rule itself is otherwise
parameter-free: inside the zone, EVERY forced pick flips, unconditionally on
which side the model favored.

**Critical caveat, stated here because it must not be buried: this is a
PICK-CONDITIONED construct, not a market-conditioned one.** The measured
46% figures are the accuracy of OUR OWN model's forced picks when restricted
to this spread-gap zone -- not the accuracy of any fixed market side (e.g.
"the underdog covers 54% of the time in this zone"). That means the flip's
expected in-zone accuracy is only the complement of the measured number
(roughly 54%) IF the lean is real and IF the active model's own pick-
generation process inside this zone stays stable going forward -- a change
to the active model's configuration could change which games and which
sides land in the zone, and would not automatically inherit this
measurement. The 2026 prospective ledger settles this empirically instead of
assuming it.

**Interaction with other overlays on the production card, stated explicitly
per the established pattern**: this challenger is tracked INDEPENDENTLY
against the active model's own UN-flipped card, exactly like every other
overlay challenger in this repository (``backup_qb_fade_overlay``,
``division_revenge_tilt_overlay``, ``injury_value_tilt_overlay``,
``surface_switch_tilt_overlay``). ``_cmd_publish_predictions`` calls every
overlay recorder in SEPARATE try/except blocks, each reading the SAME
un-flipped active-model card and applying its own transform independently --
this overlay never sees ``coach_fade_overlay``'s flips (or any other
overlay's), and no other overlay ever sees this one's. If ``coach_fade_overlay``
is ever played for real on the PUBLISHED card (it currently is, weeks 1-8,
per ``docs/coach_fade_overlay.md``) and a game happens to sit in both that
overlay's clean-case set AND this overlay's spread-gap zone, the two rules
could disagree about the published pick -- but that interaction is a
property of the PUBLISHED card, which this module never touches; this
challenger's own prospective evidence is always scored against the
un-flipped active model, never against whatever the published card actually
shows.

This module is the no-window-cost path, built on the exact pattern of
``surface_switch_tilt_overlay.py``, ``backup_qb_fade_overlay.py``,
``division_revenge_tilt_overlay.py``, ``injury_value_tilt_overlay.py``, and
``coach_fade_overlay.py`` (the original precedent): a **pick-level,
post-prediction transform** of the active model's own forced pick,
dual-tracked against that same active model in the prospective challenger
ledger (``nfl_ats.prospective_scoring``), at no rotation-registry window cost
and with zero training-time feature changes. **Nothing in this module is
wired into ``publishing.py`` or the production pick path** -- like the four
tilt/fade siblings, and unlike the coach-fade overlay, no owner decision to
play this on the real card has been made; it is dual-tracked only.

Unlike the other overlays, this one reads no schedule snapshot and no
separate feature table: the spread-gap zone is entirely a function of the
CARD's own ``spread_line`` column -- exactly the same decision-line field
the sibling overlays' recorders already read for ``decision_home_spread``
(``injury_value_tilt_overlay.record_injury_value_tilt_challenger_decisions``
and its siblings), so this module's flip logic uses the identical data
plumbing, just at the point of computing the flip rather than only at the
point of recording it.

Two things live here, mirroring the sibling overlays' structure:

1. :func:`apply_spread_gap_zone_fade_overlay` -- the pick-level transform,
   reading ``spread_line`` directly off the predictions/card frame, plus
   :func:`overlay_disclosure_note` for the plain-English provenance sentence.
2. :func:`record_spread_gap_zone_fade_challenger_decisions` -- writes the
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
CHALLENGER_ID = "spread_gap_zone_fade_overlay"

#: Frozen BEFORE the 2011-2017 pre-2018 replication ran
#: (registry/weak_signals.json:pick_conditioned_spread_gap_zone_pre2018's own
#: description states the bucket bounds were predeclared). Not free
#: parameters of this overlay -- kept exactly as measured.
SPREAD_GAP_LOWER_BOUND = 7.5
SPREAD_GAP_UPPER_BOUND = 10.0


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    original_pick_team: str
    flipped_to_team: str
    spread_line: float


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


def apply_spread_gap_zone_fade_overlay(
    predictions: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip EVERY forced pick whose market line sits in the spread-gap zone.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the pre-2018
      replication and both mined reads were scored on regular-season games
      only);
    * ``spread_line`` is present and numeric; and
    * ``SPREAD_GAP_LOWER_BOUND <= abs(spread_line) <= SPREAD_GAP_UPPER_BOUND``
      (the frozen zone).

    Unlike the sibling overlays, this rule does NOT condition on which side
    the model already picked -- inside the zone, EVERY forced pick flips,
    unconditionally, because the measured construct is a property of the
    ZONE itself (the model's own picks underperform inside it), not of a
    particular side.

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    the sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch.
    """

    required = {"game_id", "home_team", "away_team", "home_cover_probability", "spread_line"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    spread_line = pd.to_numeric(base["spread_line"], errors="coerce")
    abs_spread = spread_line.abs()

    eligible = pd.Series(True, index=base.index)
    if "game_type" in base.columns:
        eligible &= base["game_type"].astype(str).eq("REG")
    eligible &= abs_spread.notna()

    in_zone = (
        eligible & abs_spread.ge(SPREAD_GAP_LOWER_BOUND) & abs_spread.le(SPREAD_GAP_UPPER_BOUND)
    )

    original_home_pick = base["home_cover_probability"].ge(0.5)

    overlaid = base.copy()
    overlaid.loc[in_zone, "home_cover_probability"] = (
        1.0 - overlaid.loc[in_zone, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for idx in base.loc[in_zone].index:
        row = base.loc[idx]
        row_home_pick = bool(original_home_pick.loc[idx])
        original_team = str(row["home_team"] if row_home_pick else row["away_team"])
        flipped_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                original_pick_team=original_team,
                flipped_to_team=flipped_team,
                spread_line=float(spread_line.loc[idx]),
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
        f"{flip.matchup} ({flip.spread_line:+.1f}): "
        f"{flip.original_pick_team} -> {flip.flipped_to_team}"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the spread-gap-zone "
        f"fade (market line {SPREAD_GAP_LOWER_BOUND}-{SPREAD_GAP_UPPER_BOUND} points, a zone "
        "where the model's own forced picks have underperformed across three separate reads). "
        f"{detail}. See docs/spread_gap_zone_fade_overlay.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_spread_gap_zone_fade_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the fade overlay's picks to the prospective challenger ledger.

    Mirrors ``surface_switch_tilt_overlay.record_surface_switch_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    ``data_root`` is accepted for call-signature parity with every other
    overlay recorder (``_cmd_publish_predictions`` calls all of them
    uniformly as ``recorder(_artifacts_root(), _data_root())``) but is not
    read: unlike the schedule- or feature-table-dependent siblings, the
    spread-gap zone is entirely a function of the card's own ``spread_line``
    column, matching ``best_pick_nomination.record_nomination_challenger_decisions``'s
    identical, established precedent for an unused ``data_root`` parameter.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this
    challenger tracks the fade's forced-pick (``decision_line``) accuracy
    only, never a fabricated paper-bet edge for the post-fade side.
    """

    del data_root  # unused -- see docstring; kept for call-signature parity

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

    tilt = apply_spread_gap_zone_fade_overlay(card)
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
