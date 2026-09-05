"""Low-total divisional home-dog challenger: a parameter-free pick-level
nudge (LEAD-42).

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim per this
project's rule for any module that scores or adjudicates an experiment:** an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: record it with ``nfl-ats weak-signals
record``, report ``probability_positive``, never the binary "contains zero."

Research chain (read from ``docs/schedule_flag_battery.md`` "Wave 2" before
this module was built): ``low_total_div_home_dog_on_production``
(``src/nfl_ats/schedule_flag_features.py``, ``derive_low_total_div_home_dog_features``)
stacked ONE new column onto the exact production ``weak_stack`` ridge chain
and screened it against the Tuesday-opener consensus over the rotation-
assigned [2020, 2021] window: **+0.4386 accuracy points, week-blocked 95%
[-0.6608, +1.7544], probability_positive 0.68955**, 456 paired games / 35
weeks (8/456 forced picks flip). The interval crosses zero; per AGENTS.md
that is the EXPECTED shape for a real small signal at this evaluator's
resolution, never grounds to decline building a no-window-cost prospective
challenger. Neither admissible closing ground applies (no resolved wrong
sign -- the interval is not entirely below zero; no positive-control bound
was run for this pick-level challenger specifically), so this stays
``unresolved_below_power`` in the registry. Wiring it here is an EV-positive
dual-tracked play (``probability_positive`` 0.68955 above the 0.5 that makes
playing it the favoured side of the bet), not a claim of a proven edge
(AGENTS.md "a promotion bar is not a decision bar").

**The rule is parameter-free and frozen, REG divisional games only**: a game
qualifies when it is divisional (``div_game == 1``), the DECISION total is
``<= 42``, and the home team is the underdog at the decision spread
(``spread_line < 0``, this repository's uniform sign convention: positive =
home favored). Predeclared direction: BACK the home dog. Unlike the
on-production screen (which reads the Tuesday-opener consensus from a
separate historical market-archive store, per
``nfl_ats.schedule_flag_features.default_opener_lines``), this LIVE
challenger reads ``total_line``/``spread_line``/``div_game`` DIRECTLY off the
active weekly forecast card -- the SAME decision-line fields the sibling
tilt/fade overlays already read for ``decision_home_spread``
(``spread_gap_zone_fade_overlay.record_spread_gap_zone_fade_challenger_decisions``
is this module's direct precedent). This is the card's own DECISION line
(the "TUESDAY-lock" input the production model is fit against for the week),
not a separately re-derived opener quote -- disclosed here as the one
deliberate difference from the on-production screen's data source, not a
silent substitution.

**Asymmetric, matching the on-production construct's own shape**: this
overlay only ever flips the pick ONTO the home dog when a qualifying game's
model pick is currently on the away side -- never the reverse (an eligible
game with the model already picking the home dog is left untouched, and a
non-qualifying game is never touched at all). This mirrors
``forecast_weather_kn_precip_high_total_tilt_overlay``'s identical
"never flips a already-correct pick" convention.

This module is the no-window-cost path, built on the exact pattern of
``spread_gap_zone_fade_overlay.py`` (unlike the other overlays, it reads no
schedule snapshot and no separate feature table -- everything it needs is
already a column on the card): a **pick-level, post-prediction transform** of
the active model's own forced pick, dual-tracked against that same active
model in the prospective challenger ledger (``nfl_ats.prospective_scoring``),
at no rotation-registry window cost and with zero training-time feature
changes. **Nothing in this module is wired into ``publishing.py``'s
prediction path or the production pick path** -- it is dual-tracked only; no
owner decision to play this on the real card has been made.

Two things live here, mirroring the sibling overlays' structure:

1. :func:`apply_low_total_div_home_dog_overlay` -- the pick-level transform,
   reading ``div_game``/``total_line``/``spread_line`` directly off the
   predictions/card frame, plus :func:`overlay_disclosure_note` for the
   plain-English provenance sentence.
2. :func:`record_low_total_div_home_dog_challenger_decisions` -- writes the
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
CHALLENGER_ID = "low_total_div_home_dog_challenger"

#: Frozen, ported verbatim from nfl_ats.schedule_flag_features.LOW_TOTAL_DIV_DOG_TOTAL_MAX.
LOW_TOTAL_MAX = 42.0


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    total_line: float
    spread_line: float


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring the sibling overlays' result classes.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_low_total_div_home_dog_overlay(
    predictions: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick onto the home dog in a qualifying game.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the on-production
      screen's own construct is REG-scoped, via ``div_game``);
    * ``div_game == 1``;
    * ``total_line`` is present, numeric, and ``<= LOW_TOTAL_MAX``;
    * ``spread_line`` is present, numeric, and ``< 0`` (home is the
      underdog); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) is
      currently on the AWAY side.

    Deliberately ASYMMETRIC: never flips a HOME pick to AWAY, and a missing
    ``total_line``/``spread_line`` folds into "not eligible," never a
    fabricated qualifying value.
    """

    required = {
        "game_id",
        "home_team",
        "away_team",
        "home_cover_probability",
        "div_game",
        "total_line",
        "spread_line",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    div_game = pd.to_numeric(base["div_game"], errors="coerce").eq(1.0)
    total_line = pd.to_numeric(base["total_line"], errors="coerce")
    spread_line = pd.to_numeric(base["spread_line"], errors="coerce")
    low_total = total_line.notna() & total_line.le(LOW_TOTAL_MAX)
    home_dog = spread_line.notna() & spread_line.lt(0.0)

    eligible = div_game & low_total & home_dog
    if "game_type" in base.columns:
        eligible &= base["game_type"].astype(str).eq("REG")

    away_pick = base["home_cover_probability"].lt(0.5)
    flip_mask = eligible & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for idx in base.loc[flip_mask].index:
        row = base.loc[idx]
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                total_line=float(total_line.loc[idx]),
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
        f"{flip.matchup} (total {flip.total_line:.1f}, spread {flip.spread_line:+.1f}): "
        "AWAY -> HOME"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** onto the home underdog "
        f"in a divisional game with a decision total at or below {LOW_TOTAL_MAX:.0f}, where the "
        f"model's own pick was on the away side. {detail}. See docs/schedule_flag_battery.md "
        "(Wave 2, LEAD-42). Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_paired_overlay_arms(
    artifacts_root: Path, challenger_id: str, decisions: pd.DataFrame, card: pd.DataFrame
) -> None:
    """Freeze both arms together, without changing the shared challenger schema.

    The candidate also enters the standard scoring ledger; this companion
    preserves its exact contemporaneous baseline, even when called directly.
    First-write-wins makes a retry safe after either ledger write fails.
    """
    if decisions.empty:
        return
    paired = decisions.copy()
    baseline = card.set_index("game_id")["home_cover_probability"]
    paired["baseline_pick_side"] = np.where(paired["game_id"].map(baseline).ge(0.5), "HOME", "AWAY")
    path = artifacts_root / "prospective" / f"{challenger_id}_paired_decisions.parquet"
    if path.is_file():
        existing = pd.read_parquet(path)
        paired = pd.concat([existing, paired], ignore_index=True).drop_duplicates(
            subset=["challenger_id", "game_id"], keep="first"
        )
    atomic_parquet(paired, path)


def record_low_total_div_home_dog_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the overlay's picks to the prospective challenger ledger.

    Mirrors
    ``nfl_ats.spread_gap_zone_fade_overlay.record_spread_gap_zone_fade_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    ``data_root`` is accepted for call-signature parity with every other
    overlay recorder (``orchestrate_publish_predictions`` calls all of them
    uniformly as ``recorder(_artifacts_root(), _data_root())``) but is not
    read: the low-total-div-dog construct is entirely a function of the
    card's own ``div_game``/``total_line``/``spread_line`` columns, matching
    ``spread_gap_zone_fade_overlay``'s identical, established precedent for
    an unused ``data_root`` parameter.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this
    challenger tracks the overlay's forced-pick (``decision_line``) accuracy
    only, never a fabricated paper-bet edge for the post-flip side.
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
            "No synchronized active ATS model is available to record overlay decisions from"
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
            "changed underneath this overlay -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    source_columns = {"div_game", "total_line"}
    if card.empty or not source_columns.issubset(card.columns):
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": "decision total/divisional source is absent",
        }
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "total_line",
        "div_game",
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

    tilt = apply_low_total_div_home_dog_overlay(card)
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
        record_paired_overlay_arms(artifacts_root, CHALLENGER_ID, decisions, card)
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
    "LOW_TOTAL_MAX",
    "TiltFlip",
    "TiltResult",
    "apply_low_total_div_home_dog_overlay",
    "overlay_disclosure_note",
    "record_low_total_div_home_dog_challenger_decisions",
]
