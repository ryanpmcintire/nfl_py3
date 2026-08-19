"""Backup-QB fade overlay: a parameter-free pick-level nudge.

Research chain: ``scripts/nfl_bias_battery_screen.py``'s ``backup_qb_start``
cell -- "Starting QB differs from the team's modal QB this season (>=3 prior
starts)" -- is one of 17 predeclared situational cells in the NFL bias
battery, close-graded (2009-2025, 7,002 QB-baseline-eligible team-game rows)
at -0.2731 accuracy points, ``probability_positive`` 0.1684
(``registry/weak_signals.json:bias_battery_backup_qb_start``,
``unresolved_below_power``). The same construct, ported into
``nfl_ats.experiment_runner`` and re-screened at the opener grade (2020-2025,
2,436 team-games), reads -2.3578 accuracy points, ``probability_positive``
0.0982 (``registry/weak_signals.json:bias_battery_backup_qb_start_opener``,
also ``unresolved_below_power``). **Both grades lean the same way**: a
negative effect at BOTH grades means the flagged (backup-starter) side covers
LESS than its complement -- the backup-QB side under-covers, both close- and
opener-graded (``probability_positive`` well under 0.5 at both: 0.1684 and
0.0982). Per AGENTS.md an interval crossing zero at this evaluator's
~2-point resolution is the EXPECTED shape for a real small signal, never
grounds to close the line, and this task's own evidence-check gate ("if both
grades lean the same way ... build") is satisfied by this same-direction
agreement.

This module is the no-window-cost path, built on the exact pattern of
``division_revenge_tilt_overlay.py``, ``injury_value_tilt_overlay.py``, and
``coach_fade_overlay.py`` (the original precedent): a **pick-level,
post-prediction transform** of the active model's own forced pick,
dual-tracked against that same active model in the prospective challenger
ledger (``nfl_ats.prospective_scoring``), at no rotation-registry window cost
and with zero training-time feature changes. **Nothing in this module is
wired into ``publishing.py`` or the production pick path** -- like the
division-revenge and injury tilts, and unlike the coach-fade overlay, no
owner decision to play this on the real card has been made; it is dual-
tracked only.

**Important caveat, stated here because it must not be buried:** the active
``weak_stack`` model already carries QB-continuity and injury/availability
features (``docs/injury_value_lost.md``, the QB-continuity family in
``nfl_ats.experiment_runner``). This overlay may therefore be double-counting
information the model already prices into ``home_cover_probability`` --
fading a backup-QB team the model has ALREADY discounted for that same reason
would double-discount it. That is exactly the open question prospective
dual-tracking is built to measure: if the model already fully prices the
backup-QB effect, this overlay's flips should show no edge over the model's
own raw picks; if the model under-prices it (plausible, since the model's
features describe QB *continuity*/*value lost*, not the specific "modal
starter this season" construct the bias battery measured), the overlay's
flips should show a positive edge. The 2026 prospective ledger settles this
empirically instead of by assumption.

**The rule is parameter-free** -- no threshold, no tuning, nothing derived
from 2018-2025 outcomes, aside from the **battery's own frozen eligibility
rule**: a team's starting QB is only ever compared to a modal starter once at
least 3 prior starts this season have been observed (kept exactly as
measured -- see :func:`backup_qb_flag_by_game`).

Two things live here, mirroring ``division_revenge_tilt_overlay.py`` exactly:

1. :func:`backup_qb_flag_by_game` -- the pregame-safe, DATA-DERIVED signal,
   ported verbatim from ``nfl_ats.experiment_runner._bias_battery_qb_backup_flag``
   / ``scripts/nfl_bias_battery_screen.py``'s modal-QB tracker (same running
   counts, same ``>= 3 prior starts`` eligibility floor), read straight from
   the schedule snapshot's ``home_qb_name``/``away_qb_name`` columns, never
   hand-typed.
2. :func:`apply_backup_qb_fade_overlay` -- the pick-level transform, plus
   :func:`overlay_disclosure_note` for the plain-English provenance sentence.

:func:`record_backup_qb_fade_challenger_decisions` writes the overlay's own
arm to the prospective challenger ledger so 2026 scores it cleanly,
independent of whether it is ever played on the real card.
"""

from __future__ import annotations

import json
from collections import Counter
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
CHALLENGER_ID = "backup_qb_fade_overlay"

#: The battery's own frozen eligibility floor (scripts/nfl_bias_battery_screen.py
#: / nfl_ats.experiment_runner._bias_battery_qb_backup_flag): a team's
#: starting QB is only ever compared against a modal-starter baseline once at
#: least this many prior starts this season have been observed. Kept exactly
#: as measured -- not a free parameter of this overlay.
MIN_PRIOR_STARTS = 3


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _modal_backup_flag_for_group(qb_names: pd.Series) -> pd.Series:
    """Per (team, season) group, sorted by gameday.

    Ported verbatim from ``nfl_ats.experiment_runner._bias_battery_qb_backup_flag``
    / ``nfl_bias_battery_screen._qb_backup_flag``: a running counter of every
    QB who has STRICTLY PRIOR starts this team-season; once at least
    :data:`MIN_PRIOR_STARTS` prior starts have accumulated, the row is
    flagged when its OWN starter differs from the modal (most frequent)
    starter among those prior starts. Before the eligibility floor is
    reached, the row is simply not flagged (folded into ``False``, mirroring
    how ``coach_fade_overlay.year_one_by_game`` folds "no observed prior
    season" into ``False`` rather than a separate missing-data sentinel) --
    "insufficient data" and "confidently not a backup start" are
    indistinguishable to a fade rule that only ever asks "is this side
    flagged", so collapsing them is a design choice, not data loss: the
    eligibility floor is still the one the battery measured.
    """

    counts: Counter[str] = Counter()
    flags: list[bool] = []
    for qb_name in qb_names:
        total_prior = sum(counts.values())
        if total_prior >= MIN_PRIOR_STARTS:
            modal_qb = counts.most_common(1)[0][0]
            flags.append(bool(qb_name != modal_qb))
        else:
            flags.append(False)
        if isinstance(qb_name, str) and qb_name:
            counts[qb_name] += 1
    return pd.Series(flags, index=qb_names.index, dtype=bool)


def backup_qb_flag_by_game(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``backup_home``/``backup_away``.

    Pregame-safe by construction: at the time of any given game, the modal
    starter used to evaluate it is computed only from that same team's
    STRICTLY EARLIER starts this season (sorted by ``gameday``, matching the
    battery's own running-counter design). A later game in the season can
    never change an earlier game's flag -- the two leakage regression tests
    in ``tests/test_backup_qb_fade_overlay.py`` prove this empirically,
    mirroring ``coach_fade_overlay``'s pair.
    """

    required = {
        "game_id",
        "season",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "home_qb_name",
        "away_qb_name",
    }
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for backup-QB tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")
    reg["season"] = reg["season"].astype(int)

    home_side = pd.DataFrame(
        {
            "game_id": reg["game_id"].astype(str),
            "season": reg["season"],
            "team": reg["home_team"],
            "gameday": reg["gameday"],
            "own_qb_name": reg["home_qb_name"],
            "is_home": True,
        }
    )
    away_side = pd.DataFrame(
        {
            "game_id": reg["game_id"].astype(str),
            "season": reg["season"],
            "team": reg["away_team"],
            "gameday": reg["gameday"],
            "own_qb_name": reg["away_qb_name"],
            "is_home": False,
        }
    )
    long_df = pd.concat([home_side, away_side], ignore_index=True)
    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)

    long_df["backup_flag"] = False
    for _, group in long_df.groupby(["team", "season"], sort=False):
        long_df.loc[group.index, "backup_flag"] = _modal_backup_flag_for_group(
            group["own_qb_name"]
        ).to_numpy()

    home_rows = long_df.loc[long_df["is_home"], ["game_id", "backup_flag"]].rename(
        columns={"backup_flag": "backup_home"}
    )
    away_rows = long_df.loc[~long_df["is_home"], ["game_id", "backup_flag"]].rename(
        columns={"backup_flag": "backup_away"}
    )
    flags = home_rows.merge(away_rows, on="game_id", how="outer")
    flags["backup_home"] = flags["backup_home"].fillna(False).astype(bool)
    flags["backup_away"] = flags["backup_away"].fillna(False).astype(bool)

    seasons = reg[["game_id", "season"]].drop_duplicates(subset="game_id")
    return seasons.merge(flags, on="game_id", how="left")


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    backup_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``coach_fade_overlay.OverlayResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_backup_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_backup_qb_fade_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick away from a clean-case backup-QB start.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the construct's
      close- and opener-graded measurements were both scored on regular-
      season games only);
    * the model's own pick (``home_cover_probability >= 0.5`` picks home)
      lands on a side flagged as a backup start (:func:`backup_qb_flag_by_game`);
      and
    * the OPPONENT is not ALSO flagged as a backup start (the "clean case",
      mirroring ``coach_fade_overlay.apply_coach_fade_overlay``'s year-1
      clean-case gate exactly). A both-backup game has no measured direction
      to fade and is reported in ``both_backup_games`` instead of flipped.

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    the sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), (), enabled)

    flags = backup_qb_flag_by_game(schedules)
    merged = base.merge(flags, on=["game_id", "season"], how="left", validate="one_to_one")
    merged["backup_home"] = merged["backup_home"].fillna(False).astype(bool)
    merged["backup_away"] = merged["backup_away"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    picked_is_backup = merged["backup_home"].where(home_pick, merged["backup_away"])
    opponent_is_backup = merged["backup_away"].where(home_pick, merged["backup_home"])
    both_backup = merged["backup_home"] & merged["backup_away"]

    flip_mask = eligible & picked_is_backup & ~opponent_is_backup

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(row["home_cover_probability"] >= 0.5)
        backup_team = str(row["home_team"] if row_home_pick else row["away_team"])
        opponent_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                backup_team=backup_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_backup, "game_id"].astype(str))
    return TiltResult(overlaid, tuple(flips), both_ids, enabled)


def overlay_disclosure_note(result: TiltResult) -> str:
    """Plain-language provenance sentence, mirroring
    ``division_revenge_tilt_overlay.overlay_disclosure_note``.

    Empty when the overlay is off or changed nothing this week. Not currently
    surfaced on the published card -- this overlay is dual-tracked only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.backup_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the backup-QB "
        "fade (the model's pick sat on a side starting a QB other than its own modal "
        f"starter this season, against a non-backup opponent). {detail}. See "
        "docs/backup_qb_fade_overlay.md. Prospective evidence only -- not applied to "
        "the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_backup_qb_fade_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the fade overlay's picks to the prospective challenger ledger.

    Mirrors ``division_revenge_tilt_overlay.record_division_revenge_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this
    challenger tracks the fade's forced-pick (``decision_line``) accuracy
    only, never a fabricated paper-bet edge for the post-fade side.
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
    tilt = apply_backup_qb_fade_overlay(card, schedules)
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
