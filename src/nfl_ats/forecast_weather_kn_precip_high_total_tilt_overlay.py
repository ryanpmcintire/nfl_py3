"""Forecast (kickoff-nearest) precip-high-total tilt overlay: a
parameter-free pick-level nudge sharing the SAME live kickoff-nearest
GFS-MOS fetch ``forecast_weather_kn_warm_team_cold_late_tilt_overlay``
already makes.

Research chain: ``docs/forecast_weather_screen.md``'s 2026-08-20 extension
("2009-2019 archive backward + a 6-cell family"), cell 4,
``forecast_weather_kn_precip_high_total``. Measured this session (read
directly from ``registry/weak_signals.json``, current after the archive's
full 2009-2025 fetch completed and all 12 kickoff_nearest specs were
re-run with ``--replace``): full window (REG 2009-2025) +0.0832 accuracy
points, week-blocked 95% [-0.0898, +0.2462], ``probability_positive``
0.8324, n_flag=50 of n_total=4,317; pre2020 window (REG 2009-2019) +0.1086
accuracy points, 95% [-0.1107, +0.3065], ``probability_positive`` 0.8406,
n_flag=29 of n_total=2,735. Both intervals cross zero. Per AGENTS.md that
is the EXPECTED shape for a real small signal at this evaluator's ~2-point
resolution, never grounds to decline a no-window-cost prospective
challenger; neither admissible closing ground applies (no resolved wrong
sign, no positive-control bound was run), so this stays
``unresolved_below_power`` in the registry. Wiring it here is an
EV-positive dual-tracked play (``probability_positive`` 0.83 > 0.5), not a
claim of a proven edge (AGENTS.md "a promotion bar is not a decision bar").
Per ``docs/forecast_weather_screen.md``'s revised "Wiring recommendations"
section this is recommendation #3 -- "moderate EV, genuinely new
mechanism" -- wired here alongside the higher-EV
``forecast_weather_kn_warm_team_cold_late_tilt`` (recommendation #1)
specifically because it shares that challenger's live fetch at no extra
network cost (see below), not because it independently cleared a higher
bar; the doc's own text frames wiring it as conditional on exactly that
cheapness ("if cheap, wire it").

**No tuesday_noon sibling exists for this cell** -- the tuesday_noon
forecast archive never captured a precipitation field
(``docs/forecast_weather_screen.md``'s 2026-08-20 extension section, "Field
extraction extended to capture precipitation probability"), so this is a
genuinely new live signal, not a rerun of an already-wired mechanism.

**Flag, ported from ``src/nfl_ats/experiment_runner.py``'s
``_flag_forecast_weather_kn_precip_high_total`` (the runner builder that
produced the registry numbers above):** outdoor AND this game's
kickoff-nearest forecast precipitation probability (``p06``, falling back
to ``p12``) ``>= 60`` percent AND this game's own ``total_line`` ``>= 47``.
``total_line`` is read directly from the active weekly card
(``recommendations.csv`` already carries it, like ``spread_line``) rather
than the schedules snapshot -- the same "use the card's own live market
column" convention every other overlay in this package uses for
``spread_line``.

**Predicted direction is an UNVERIFIED folk mechanism, disclosed as such by
the predeclaration doc itself, not a validated one**: "a high total suggests
the market has not fully priced in precip-driven scoring suppression; home
teams are conventionally assumed better adapted to their own site's
weather" -- the SAME unverified assumption the sibling cold-weather cells
already carry, reported here rather than re-litigated.

**Live data path: SHARED with
``forecast_weather_kn_warm_team_cold_late_tilt_overlay``, not duplicated.**
Both cells consume the identical kickoff-nearest, model=GFS GFS-MOS
bulletin for the identical set of games -- that sibling module's fetch
already captures ``forecast_precip_prob_pct`` alongside
``forecast_temp_f`` from the SAME HTTP response (GFS MOS bulletins carry
both fields per row; no second request), specifically so this module could
reuse it. This module imports that fetch machinery directly
(:func:`nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.fetch_kickoff_nearest_forecasts_fail_open`,
:func:`nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.games_for_forecast_fetch`,
:func:`nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.fetch_shared_kickoff_nearest_forecasts_fail_open`)
rather than reimplementing it, and
:func:`record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions`
accepts an already-fetched ``forecasts`` frame so a caller wiring both
challengers in the same publish call hits the Iowa Environmental Mesonet's
public MOS JSON API once, not twice, per
``docs/forecast_weather_screen.md``'s "one fetch, several consumers" wiring
note. Passing no ``forecasts`` (the default) still works standalone -- this
module fetches for itself in that case, using the exact same fail-open
contract.

**FAIL-OPEN, unconditionally**, inherited from the shared fetch layer: any
failure fetching or parsing live forecast data is caught inside
:func:`nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.fetch_kickoff_nearest_forecasts_fail_open`,
logged as a ``RuntimeWarning``, and folded into "every game gets no
forecast, the flag is False everywhere" rather than raised. This overlay
must never be able to block a publish.

**Tilt direction: HOME**, matching the cell's own predicted direction and
mirroring every sibling overlay's deliberately ASYMMETRIC pattern: flip
AWAY -> HOME only when the flag fires AND the model's own pick is currently
on the away side; never the reverse.

This module is the no-window-cost path, built on the exact pattern of
``forecast_weather_kn_warm_team_cold_late_tilt_overlay.py``,
``forecast_cold_visitor_tilt_overlay.py``, and
``interim_hc_first_game_tilt_overlay.py``: a **pick-level, post-prediction
transform** of the active model's own forced pick, dual-tracked against
that same active model in the prospective challenger ledger
(``nfl_ats.prospective_scoring``), at no rotation-registry window cost and
with zero training-time feature changes. **Nothing in this module is wired
into ``publishing.py`` or the production pick path** -- no owner decision to
play this on the real card has been made; it is dual-tracked only.

:func:`record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions`
writes the overlay's own arm to the prospective challenger ledger so 2026
scores it cleanly, independent of whether it is ever played on the real
card.

**Operational cutoff correction, 2026-09-02:** historical evidence and
public function names retain their ``kickoff_nearest`` identity for
provenance, but the shared live fetch now uses ``pool_decision``:
``min(kickoff, Sunday 16:00 America/New_York)``. It does not depend on the
historical replacement archive; supplied frames must prove the new cutoff.
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
from nfl_ats.forecast_cold_visitor_tilt_overlay import OUTDOOR_ROOFS, STATION_MAP_RELATIVE_PATH
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    LIVE_FORECAST_CUTOFF_MODE,
    FetchBulletin,
    fetch_kickoff_nearest_forecasts_fail_open,
    fetch_mos_bulletin,
    games_for_forecast_fetch,
    validate_live_forecast_provenance,
)
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
CHALLENGER_ID = "forecast_weather_kn_precip_high_total_tilt"

#: Reused verbatim from src/nfl_ats/experiment_runner.py's
#: _FORECAST_PRECIP_PROB_THRESHOLD_PCT / _FORECAST_HIGH_TOTAL_THRESHOLD.
PRECIP_PROB_THRESHOLD_PCT = 60.0
HIGH_TOTAL_THRESHOLD = 47.0


def precip_high_total_flag_by_game(
    schedules: pd.DataFrame, forecasts: pd.DataFrame, total_lines: pd.DataFrame
) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``precip_high_total_flag``,
    ``forecast_precip_prob_pct``, ``total_line``.

    ``forecasts`` needs ``game_id``, ``forecast_precip_prob_pct`` (one row
    per game; a missing/NaN value folds into "not flagged"). ``total_lines``
    needs ``game_id``, ``total_line`` -- kept as a SEPARATE small input
    (rather than merged into ``forecasts``) because it comes from a
    different source (the active weekly card's own live market column, not
    the forecast fetch), mirroring how ``forecast_cold_visitor_tilt_overlay``
    keeps ``climate_temp`` and ``forecast_temp_f`` as logically distinct
    inputs even though both end up in the same flag test. Mirrors the 6-cell
    family's cell 4 definition (``docs/forecast_weather_screen.md``):
    outdoor AND kickoff-nearest forecast precip prob>=60% AND
    total_line>=47.
    """

    required_forecast = {"game_id", "forecast_precip_prob_pct"}
    missing = sorted(required_forecast.difference(forecasts.columns))
    if missing:
        raise DataContractError(f"forecasts is missing columns: {', '.join(missing)}")
    required_schedule = {"game_id", "game_type", "roof"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            "schedules is missing columns for precip high-total tracking: "
            f"{', '.join(missing_schedule)}"
        )
    required_total_line = {"game_id", "total_line"}
    missing_total_line = sorted(required_total_line.difference(total_lines.columns))
    if missing_total_line:
        raise DataContractError(f"total_lines is missing columns: {', '.join(missing_total_line)}")

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)

    forecasts = forecasts.copy()
    forecasts["game_id"] = forecasts["game_id"].astype(str)
    forecasts["forecast_precip_prob_pct"] = pd.to_numeric(
        forecasts["forecast_precip_prob_pct"], errors="coerce"
    )

    total_lines = total_lines.copy()
    total_lines["game_id"] = total_lines["game_id"].astype(str)
    total_lines["total_line"] = pd.to_numeric(total_lines["total_line"], errors="coerce")

    frame = (
        reg[["game_id", "outdoor"]]
        .merge(
            forecasts[["game_id", "forecast_precip_prob_pct"]].drop_duplicates(subset="game_id"),
            on="game_id",
            how="left",
        )
        .merge(
            total_lines[["game_id", "total_line"]].drop_duplicates(subset="game_id"),
            on="game_id",
            how="left",
        )
    )

    flag = (
        frame["outdoor"]
        & frame["forecast_precip_prob_pct"].ge(PRECIP_PROB_THRESHOLD_PCT)
        & frame["total_line"].ge(HIGH_TOTAL_THRESHOLD)
    )
    frame["precip_high_total_flag"] = flag.fillna(False).astype(bool)
    return frame[["game_id", "precip_high_total_flag", "forecast_precip_prob_pct", "total_line"]]


@dataclass(frozen=True)
class PrecipHighTotalFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    away_team: str
    home_team: str
    forecast_precip_prob_pct: float
    total_line: float


@dataclass(frozen=True)
class PrecipHighTotalResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring the sibling overlays' result classes.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[PrecipHighTotalFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_precip_high_total_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    enabled: bool = True,
) -> PrecipHighTotalResult:
    """Flip the forced pick from AWAY to HOME wherever the flag fires.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present;
    * :func:`precip_high_total_flag_by_game` fires for the game (outdoor,
      kickoff-nearest forecast precip prob>=60%, this game's own
      ``total_line``>=47, read directly from ``predictions``); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) is
      currently on the AWAY side.

    **Deliberately ASYMMETRIC**, mirroring every sibling overlay: never
    flips a HOME pick to AWAY. Missing forecast/total_line data (including a
    total fetch failure upstream) folds into "not flagged", never an error.
    """

    required = {
        "game_id",
        "season",
        "home_team",
        "away_team",
        "home_cover_probability",
        "total_line",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return PrecipHighTotalResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    total_lines = base[["game_id", "total_line"]]
    flags = precip_high_total_flag_by_game(schedules, forecasts, total_lines)
    merged = base.merge(
        flags[["game_id", "precip_high_total_flag", "forecast_precip_prob_pct"]],
        on="game_id",
        how="left",
    )
    merged["precip_high_total_flag"] = merged["precip_high_total_flag"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    away_pick = merged["home_cover_probability"].lt(0.5)
    flip_mask = eligible & merged["precip_high_total_flag"] & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[PrecipHighTotalFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        precip = row["forecast_precip_prob_pct"]
        total_line = row["total_line"]
        flips.append(
            PrecipHighTotalFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                away_team=str(row["away_team"]),
                home_team=str(row["home_team"]),
                forecast_precip_prob_pct=float(precip) if pd.notna(precip) else float("nan"),
                total_line=float(total_line) if pd.notna(total_line) else float("nan"),
            )
        )

    return PrecipHighTotalResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: PrecipHighTotalResult) -> str:
    """Plain-language provenance sentence, mirroring the sibling overlays'.

    Empty when the overlay is off or changed nothing this week. Not
    currently surfaced on the published card -- this overlay is dual-tracked
    only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: AWAY -> HOME (precip {flip.forecast_precip_prob_pct:.0f}%, "
        f"total {flip.total_line:.1f})"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the forecast "
        "(pool-decision) precip-high-total tilt (the model's pick was on an away team "
        "in an outdoor game with a high forecast precipitation probability and a high "
        "total line). "
        f"{detail}. See docs/forecast_weather_screen.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    registry_root: Path,
    *,
    now: datetime | None = None,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
    forecasts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors
    ``forecast_weather_kn_warm_team_cold_late_tilt_overlay.record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    ``forecasts``, when supplied, is used AS-IS instead of fetching again --
    see
    :func:`nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.fetch_shared_kickoff_nearest_forecasts_fail_open`
    for the "one fetch, several consumers" path this parameter exists for.
    ``None`` (the default) fetches for itself, using the exact same
    fail-open live path
    (:func:`nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.fetch_kickoff_nearest_forecasts_fail_open`):
    a network or station-mapping failure never raises out of this function,
    it simply yields zero flags for the week.

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
        "total_line",
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

    if forecasts is None:
        games_for_fetch = games_for_forecast_fetch(card, schedules)
        station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
        forecasts = fetch_kickoff_nearest_forecasts_fail_open(
            games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
        )

    validate_live_forecast_provenance(forecasts, card)

    tilt = apply_precip_high_total_tilt_overlay(card, schedules, forecasts)
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
        "forecast_fetch_status_counts": (
            forecasts["fetch_status"].value_counts().to_dict() if not forecasts.empty else {}
        ),
        "forecast_cutoff_mode": LIVE_FORECAST_CUTOFF_MODE,
    }
