"""PBP-08 protection-mismatch tilt overlay -- dual-tracked challenger only.

What it plays
-------------
When one team's offense has allowed pressure at a top-quartile rate over its
last four games AND the defense it is about to face generates pressure at a
top-quartile rate, back the DEFENSE. See
:mod:`nfl_ats.pbp08_matchup_flags` for the frozen flag construction and
``docs/pbp08_matchup_screen.md`` for the predeclaration and results.

Why it is a challenger and not a promotion
------------------------------------------
The screen measured +0.336 accuracy points full-slate, week-blocked 95%
[+0.014, +0.658], ``probability_positive`` 0.9785 (season-blocked 0.9797),
direction as predeclared, era-consistent (+0.445 / +0.225), with both
bottom-vs-bottom mirror controls landing near null (+0.019 / +0.033, P+
~0.56). That is the strongest mined mean-edge cell the project has, and P+
0.979 is far above the 0.5 that makes playing it the favoured side of the
bet -- but it is a MINED family with uncorrected multiplicity across four
cells plus two era splits, so it earns prospective evidence, not a claim on
the published card.

It is deliberately NOT run as a rotation-registry confirmation look. Measured
2026-08-25 (``nfl-ats rotation status``): the opener-graded pool has exactly
one unspent window left, ``[2024, 2025]``. Sizing that window against the
screen's own numbers -- the 2018-2025 era arm scored n_total 4,174 team-games
with a week-blocked half-width of ~0.48 points, and two opener seasons are
about 1,024 team-games -- puts the confirmation half-width near +/-0.97
points around a +0.23 effect, an interval roughly four times the effect it
would be testing. Spending the last virgin opener window on a test that
cannot resolve is the worse trade; the challenger route costs no window and
starts accruing 2026 evidence at the Week 1 lock.

Fail-open, like the sibling data-dependent overlays: a missing PBP snapshot,
a schedule that cannot reach back far enough for a four-game window, or a
week whose quartile pool never filled all fold into ZERO flags and a
documented no-op -- never an exception that could un-publish the card.
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
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pbp08_matchup_flags import build_flag_table, flag_summary
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
CHALLENGER_ID = "pbp08_protection_mismatch_tilt_overlay"

#: The screen's own first season. The expanding quartile thresholds are built
#: from ALL strictly-earlier week-blocks, so truncating the history changes the
#: thresholds and therefore the flags: measured 2026-08-25, feeding only three
#: seasons produced 3 leans in Week 1 2026 where the full 2009-onward pool
#: produces 4. Cost is not a reason to truncate -- the full build measured
#: 1.1 seconds -- so the flag build is always handed the whole history, exactly
#: as ``scripts/pbp08_matchup_screen.py`` does.
SCREEN_SEASON_START = 2009


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    original_pick_team: str
    flipped_to_team: str
    backed_side: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows, mirroring every sibling tilt.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool
    flag_summary: dict[str, Any]

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def latest_pbp_snapshot(data_root: Path) -> Path | None:
    root = data_root / "pbp" / "raw"
    if not root.is_dir():
        return None
    snapshots = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    return snapshots[0] if snapshots else None


def latest_schedules(data_root: Path) -> Path | None:
    candidates = sorted(data_root.glob("raw/*/schedules.parquet"), reverse=True)
    return candidates[0] if candidates else None


def flags_for_week_fail_open(data_root: Path, *, season: int, week: int) -> pd.DataFrame:
    """The flag table for one week, or an EMPTY frame on any missing input.

    Never raises. Same posture as the forecast and interim-coach overlays:
    an absent snapshot is a no-op week, not a broken publish. The reason is
    surfaced as a warning so a silent zero is at least noisy.
    """

    try:
        snapshot = latest_pbp_snapshot(data_root)
        schedules_path = latest_schedules(data_root)
        if snapshot is None or schedules_path is None:
            raise FileNotFoundError(
                f"missing pbp snapshot ({snapshot}) or schedules ({schedules_path})"
            )
        schedule = pd.read_parquet(schedules_path)
        schedule = schedule.loc[schedule["game_type"].astype(str).eq("REG")]
        schedule = schedule.loc[schedule["season"].between(SCREEN_SEASON_START, season)].copy()
        table = build_flag_table(schedule, snapshot)
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
        return pd.DataFrame(columns=["game_id", "back_side"])

    return table.loc[(table["season"] == season) & (table["week"] == week)].reset_index(drop=True)


def apply_pbp08_protection_mismatch_tilt(
    predictions: pd.DataFrame,
    flags: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip a pick that sits on the badly-protected offense, and only that.

    **Deliberately ASYMMETRIC**, like the sibling tilts: the measured
    construct is that the FLAGGED offense under-covers (45.19% against a
    50.45% complement in 2009-2017), so the overlay moves picks OFF that side.
    It never moves a pick ONTO a flagged offense, and it does nothing at all
    when the model already has the defense.

    A game with no lean -- neither side flagged, both flagged (a mutual
    mismatch, which is not the measured construct), or an incomplete window --
    is untouched.
    """

    required = {"game_id", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    summary = flag_summary(flags) if not flags.empty and "back_side" in flags.columns else {}
    if not enabled or flags.empty or "back_side" not in flags.columns:
        return TiltResult(base, (), enabled, summary)

    lean = (
        flags.loc[flags["back_side"].isin(("HOME", "AWAY")), ["game_id", "back_side"]]
        .drop_duplicates(subset="game_id")
        .set_index("game_id")["back_side"]
    )

    probabilities = pd.to_numeric(base["home_cover_probability"], errors="coerce")
    model_side = pd.Series(np.where(probabilities.ge(0.5), "HOME", "AWAY"), index=base.index)
    backed = base["game_id"].astype(str).map(lean)

    # Flip exactly the rows where a lean exists and the model is on the other
    # side of it -- the asymmetric case. A row with no lean, or one the model
    # already agrees with, is left byte-identical.
    should_flip = backed.notna() & backed.ne(model_side)

    overlaid = base.copy()
    overlaid.loc[should_flip, "home_cover_probability"] = 1.0 - probabilities.loc[should_flip]

    flips = tuple(
        TiltFlip(
            game_id=str(row.game_id),
            matchup=f"{row.away_team} at {row.home_team}",
            original_pick_team=str(
                row.home_team if model_side.loc[index] == "HOME" else row.away_team
            ),
            flipped_to_team=str(row.home_team if backed.loc[index] == "HOME" else row.away_team),
            backed_side=str(backed.loc[index]),
        )
        for index, row in zip(
            base.loc[should_flip].index, base.loc[should_flip].itertuples(), strict=True
        )
    )

    return TiltResult(overlaid, flips, enabled, summary)


def overlay_disclosure_note(result: TiltResult) -> str:
    """Plain-language provenance sentence. Empty when nothing moved."""

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.original_pick_team} -> {flip.flipped_to_team}"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** off an offense whose "
        "four-game pressure-allowed rate is top-quartile against a defense generating "
        f"pressure at a top-quartile rate. {detail}. See docs/pbp08_matchup_screen.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_pbp08_protection_mismatch_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the tilt's picks to the prospective challenger ledger.

    Mirrors ``spread_gap_zone_fade_overlay``'s recorder exactly: this
    challenger's "model" IS the active model transformed post-prediction, so
    it reads the active model's own synchronized weekly forecast rather than
    searching ``artifacts/margin_predictions/`` by fingerprint, and refuses to
    record if the active model's live fingerprint no longer matches the
    snapshot this challenger was registered against.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this tracks
    forced-pick accuracy at the decision line, never a fabricated paper-bet
    edge for the post-tilt side.
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
        raise ValueError("No synchronized active ATS model is available to record tilt decisions")
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
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration fingerprint "
            f"{declared_fingerprint}, but the current active forecast {forecast} was produced "
            f"with {observed_fingerprint}; the active model changed underneath this tilt -- "
            "re-register before recording"
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
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    flags = flags_for_week_fail_open(data_root, season=season, week=week)
    tilt = apply_pbp08_protection_mismatch_tilt(card, flags)
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
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
        "flag_summary": tilt.flag_summary,
    }


__all__ = [
    "CHALLENGER_ID",
    "TiltFlip",
    "TiltResult",
    "apply_pbp08_protection_mismatch_tilt",
    "flags_for_week_fail_open",
    "overlay_disclosure_note",
    "record_pbp08_protection_mismatch_tilt_challenger_decisions",
]
