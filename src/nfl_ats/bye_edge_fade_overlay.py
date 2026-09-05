"""Bye-edge fade overlay: a parameter-free, post-prediction pick-level nudge.

Research chain (all measured this session, read from ``registry/weak_signals.json``
before this module was built): ``bye_overval_fade_full_slate_post2011`` --
seasons 2012-2025 restricted to week-blocks containing at least one strictly
off-bye team anywhere in the league; flag = exactly one of the two teams off a
STRICT bye (>=12-day gap to its own immediately preceding game this season);
value column is the FADE-side cover indicator. Full-slate effect
**+0.5508134037685816** accuracy points, week-blocked 95%
``[-0.5667823486802533, +1.6434387325997184]``, ``probability_positive``
**0.8375**, n_flag=498 of n_total=2171 (n_complement=1673), fraction of slate
0.22938737908797788 (season-blocked secondary: +0.5508134037685816 pts, 95%
``[-0.4285341983532842, +1.5664703739568875]``, ``probability_positive``
0.8603, 14 season blocks). The interval crosses zero -- per AGENTS.md, at
this evaluator's ~2-point resolution that is the EXPECTED shape for a real
small signal, never grounds to decline building a no-window-cost prospective
challenger. Neither admissible closing ground applies (no resolved wrong
sign, no positive-control bound), so this stays ``unresolved_below_power`` in
the registry; wiring it here is an EV-positive dual-tracked play (P+ 0.8375 >
0.5), not a claim of a proven edge (AGENTS.md "a promotion bar is not a
decision bar").

**This entry REPLACED a prior buggy-map entry** (``docs/bye_overvaluation_screen.md``,
"Correction 2026-08-22"): the original ``build_bye_maps`` sorted each team's
games ACROSS seasons, so every season opener inherited a >=12-day gap from
the PRIOR season's finale and was misflagged "off bye". Fixed by computing
gaps within ``(team, season)`` groups. The corrected instrument is
``artifacts/bye_overvaluation_screen/post_fix_seed20260822/results.json``
(seed 20260822); the numbers quoted above are read from that corrected
artifact, and this module ports the CORRECTED (within-season) gap logic,
never the buggy cross-season one.

**Direction check -- verified, not assumed.** The registered cell's
``value_column`` is ``"fade_side_cover"`` and its description ends
"Predicted direction POSITIVE". Read directly from the corrected artifact:
``subset_cover`` (the fade side's own cover rate) is **0.5261044176706827**,
strictly above ``complement_cover`` **0.502092050209205** -- fading the
bye-holding side is BOTH the predeclared direction AND the measured
direction. This check is stated explicitly because a sibling cell in this
same build batch failed exactly this check and was dropped for it.

**Instrument sanity control.** ``bye_overval_both_bye_sanity`` (BOTH teams
off strict bye, full window 2009-2025, two-sided predeclared null): effect
+0.01237162203848194 accuracy points, ``probability_positive`` 0.55985 -- a
clean null, exactly as a working instrument should read when rest cancels on
both sides. This overlay never touches a both-off-bye game (see
:func:`apply_bye_edge_fade_overlay`), consistent with that null control.

**Era caveat -- disclosed, not a week restriction.** The 2009-2011 control
cell ``bye_overval_home_edge_pre2011`` (identical HOME-off-strict-bye flag,
pre-CBA seasons) reads +0.2707765 accuracy points, ``probability_positive``
0.70695 -- the OPPOSITE lean from the post-2011 overvaluation cell this
module implements (which reads negative for the raw home-edge construct
post-2011: ``bye_overval_home_edge_post2011`` is -0.3304447 pts, P+ 0.06365).
This is exactly why the registered fade cell is restricted to 2012+. 2026
postdates that boundary, so the overlay applies to ALL of 2026 regardless of
week -- a caveat on the effect's era-stability, reported here, not grounds to
restrict which weeks the overlay is eligible to fire on.

This module is the no-window-cost path, built on the exact pattern of
``surface_switch_tilt_overlay.py`` and ``interim_hc_first_game_tilt_overlay.py``:
a **pick-level, post-prediction transform** of the active model's own forced
pick, dual-tracked against that same active model in the prospective
challenger ledger (``nfl_ats.prospective_scoring``), at no rotation-registry
window cost and with zero training-time feature changes. **Nothing in this
module is wired into ``publishing.py`` or the production pick path** -- like
the tilt siblings, no owner decision to play this on the real card has been
made; it is dual-tracked only.

**The rule is parameter-free and frozen** -- no threshold, no tuning, nothing
derived from outcomes: when exactly one of the two teams is off a strict bye
(>=12-day gap to its own immediately preceding game this season) AND the
active model's own forced pick IS that bye-holding team, flip the pick to the
other side (fade the bye-holding side). REG season only (every measured read
above was scored on regular-season games); both-off-bye and neither-off-bye
games are never touched -- ``bye_overval_both_bye_sanity`` is the null
control for the both-bye case, not a mechanism this overlay claims to trade.

Two things live here, mirroring the sibling overlays exactly:

1. :func:`bye_edge_flag_by_game` -- the pregame-safe, DATA-DERIVED signal,
   ported verbatim from ``scripts/bye_overvaluation_screen.py``'s
   ``build_bye_maps`` (the corrected, within-``(team, season)`` gap
   computation and the ``POST_BYE_GAP_DAYS = 12`` threshold,
   ``scripts/bye_overvaluation_screen.py:58``), read straight from the
   newest local schedule snapshot (``data/raw/<snapshot>/schedules.parquet``),
   never hand-typed. **Unlike the screen script**, this function is built
   directly off the raw schedule snapshot rather than the screen's own
   outcome-filtered measurement population (which drops push/no-line games
   via ``add_ats_outcomes`` before computing gaps): a live overlay must never
   let a game's own outcome or push status change an earlier game's already
   -computed bye-gap sequence, so this function never reads ``result`` or
   ``spread_line`` at all -- proven by the leakage regression tests in
   ``tests/test_bye_edge_fade_overlay.py``.
2. :func:`apply_bye_edge_fade_overlay` -- the pick-level transform, plus
   :func:`overlay_disclosure_note` for the plain-English provenance sentence.

:func:`record_bye_edge_fade_challenger_decisions` writes the overlay's own
arm to the prospective challenger ledger so 2026 scores it cleanly,
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
from nfl_ats.provenance import sha256_file, stamp_sidecar
from nfl_ats.snapshots import latest_snapshot, load_snapshot

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "bye_edge_fade_overlay"

#: Ported verbatim from scripts/bye_overvaluation_screen.py:58 -- the STRICT
#: bye definition: a >=12-calendar-day gap to a team's own immediately
#: preceding REG-season game. Kept exactly, not re-derived.
POST_BYE_GAP_DAYS = 12

#: The merged-in flags travel under these module-private names -- a
#: collision-safety measure mirroring surface_switch_tilt_overlay's
#: OVERLAY_FLAG_COLUMN: a predictions frame that happens to already carry a
#: same-named ``home_off_bye``/``away_off_bye`` column collides silently
#: instead of crashing, and this module's own schedules-derived flag always
#: wins.
HOME_OFF_BYE_COLUMN = "_bye_edge_fade_home_off_bye"
AWAY_OFF_BYE_COLUMN = "_bye_edge_fade_away_off_bye"


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def bye_edge_flag_by_game(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``home_off_bye``, ``away_off_bye``.

    Ported verbatim from ``scripts/bye_overvaluation_screen.py``'s
    ``build_bye_maps`` (the CORRECTED, post-2026-08-22 version --
    ``docs/bye_overvaluation_screen.md``, "Correction 2026-08-22"): melt each
    REG-season game into two team-rows, sort each team's rows within its own
    season by ``gameday``, take the day-gap to the immediately PRECEDING
    game in that ``(team, season)`` group
    (``scripts/bye_overvaluation_screen.py:92-94``:
    ``long_df.groupby(["team", "season"])["gameday_dt"].diff().dt.days``),
    and flag ``gap_days >= POST_BYE_GAP_DAYS`` (12). A team's first REG-season
    game has no preceding game in the SAME season, so its gap is undefined
    (``NaN``) and is folded to ``False`` (``.fillna(False)``) -- never a bye,
    exactly as the source script does. Grouping by ``(team, season)`` rather
    than ``team`` alone is what the 2026-08-22 fix corrected: the prior,
    buggy version sorted across season boundaries, so every season opener
    inherited the prior season's off-season gap and was misflagged off-bye.

    **Read from the raw schedule snapshot, never the screen's own
    outcome-filtered measurement population.** ``scripts/bye_overvaluation_screen.py``
    computes this same gap sequence on a population that has already dropped
    push/no-line games (``add_ats_outcomes``, then ``home_cover.notna()``) --
    acceptable for a measure-only screen, but wrong for a live, pregame-safe
    overlay: excluding a game from the sequence because it happened to push
    requires knowing its RESULT, which is not available before kickoff for
    any game still to be played, and must never be allowed to shift an
    earlier team's already-computed gap. This function instead runs the
    identical gap/threshold logic against every REG-season row in the raw
    schedule snapshot, structural-fact only. Two leakage regression tests
    (``tests/test_bye_edge_fade_overlay.py``) prove this empirically: this
    function does not even require or read ``result``/``spread_line``, and a
    future season's schedule data never changes an earlier season's
    already-computed flags.

    Team codes are canonicalized (``TEAM_ABBREVIATION_ALIASES``) before the
    ``(team, season)`` grouping -- a merge-safety measure mirroring
    ``surface_switch_flag_by_game``, so the output joins cleanly against the
    predictions frame's own team codes; it does not change which games count
    as a team's own REG-season sequence, so it does not alter the measured
    construct itself.
    """

    required = {"game_id", "season", "game_type", "gameday", "home_team", "away_team"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for bye-edge tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="coerce")

    long_rows = []
    for _, game in reg.iterrows():
        for side, team in (("home", game["home_team"]), ("away", game["away_team"])):
            long_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "team": team,
                    "side": side,
                    "gameday_dt": game["gameday_dt"],
                }
            )
    long_df = pd.DataFrame(
        long_rows, columns=["game_id", "season", "team", "side", "gameday_dt"]
    ).sort_values(["team", "season", "gameday_dt"])
    long_df["gap_days"] = long_df.groupby(["team", "season"])["gameday_dt"].diff().dt.days
    long_df["post_bye"] = (long_df["gap_days"] >= POST_BYE_GAP_DAYS).fillna(False).astype(bool)

    def side_map(side: str) -> pd.Series:
        joined = reg[["game_id"]].merge(
            long_df.loc[long_df["side"] == side, ["game_id", "post_bye"]],
            on="game_id",
            how="left",
        )
        return joined["post_bye"].fillna(False).astype(bool)

    frame = reg[["game_id", "season"]].reset_index(drop=True).copy()
    frame["home_off_bye"] = side_map("home").to_numpy()
    frame["away_off_bye"] = side_map("away").to_numpy()
    return frame


@dataclass(frozen=True)
class TiltFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    bye_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``surface_switch_tilt_overlay.TiltResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_bye_edge_fade_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    """Flip the forced pick away from the strict-bye-holding side.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present (the registered
      measurement -- the full-slate fade arm -- was scored on regular-season
      games only);
    * :func:`bye_edge_flag_by_game` finds EXACTLY ONE of the two teams off a
      strict bye (both-off-bye and neither-off-bye games are never touched --
      ``bye_overval_both_bye_sanity`` is the null control for the both-bye
      case, not a mechanism this overlay claims); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) is
      currently ON the bye-holding side.

    Flipping sets ``home_cover_probability`` to its complement, exactly as
    the sibling overlays do, so every existing reader of the column needs no
    overlay-aware branch.

    The flags are ALWAYS this module's own schedules-derived
    :func:`bye_edge_flag_by_game` output, merged under the private
    :data:`HOME_OFF_BYE_COLUMN`/:data:`AWAY_OFF_BYE_COLUMN` names: a
    predictions frame that already carries same-named columns collides
    silently instead of crashing, and if no flag survives the merge at all
    the result is the documented no-op -- zero flips, never a KeyError.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    flags = bye_edge_flag_by_game(schedules)
    merged = base.merge(
        flags.rename(
            columns={
                "home_off_bye": HOME_OFF_BYE_COLUMN,
                "away_off_bye": AWAY_OFF_BYE_COLUMN,
            }
        ),
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    for column in (HOME_OFF_BYE_COLUMN, AWAY_OFF_BYE_COLUMN):
        if column not in merged.columns:
            # Absence path: nothing to read means nothing flips -- the
            # documented no-op, never a KeyError.
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_is_bye_team = merged[HOME_OFF_BYE_COLUMN] & ~merged[AWAY_OFF_BYE_COLUMN]
    away_is_bye_team = merged[AWAY_OFF_BYE_COLUMN] & ~merged[HOME_OFF_BYE_COLUMN]
    home_pick = merged["home_cover_probability"].ge(0.5)

    flip_mask = eligible & ((home_is_bye_team & home_pick) | (away_is_bye_team & ~home_pick))

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        bye_is_home = bool(row[HOME_OFF_BYE_COLUMN])
        bye_team = str(row["home_team"] if bye_is_home else row["away_team"])
        opponent_team = str(row["away_team"] if bye_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                bye_team=bye_team,
                opponent_team=opponent_team,
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
        f"{flip.matchup}: {flip.bye_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the bye-edge fade "
        "(the model's pick was on the team coming off a strict, >=12-day-gap bye, and its "
        "opponent was not also off a bye this week). "
        f"{detail}. See docs/bye_edge_fade_overlay.md. Prospective evidence only -- not "
        "applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_bye_edge_fade_challenger_decisions(
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
    fade = apply_bye_edge_fade_overlay(card, schedules)
    faded_card = fade.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = faded_card.loc[keep]

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
        "flip_count": fade.flip_count,
        "flipped_game_ids": [flip.game_id for flip in fade.flips],
    }
