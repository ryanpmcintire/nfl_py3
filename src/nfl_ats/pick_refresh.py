from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats import mass_preserving_lattice
from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.calibration import ResidualSmoothingMethod
from nfl_ats.clv import (
    LIVE_CAPTURE_KIND,
    load_decision_quotes,
    load_paper_decisions,
    refuse_if_outside_recording_lock_window,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.home_side_location import load_forecast_home_side_offsets
from nfl_ats.io import atomic_parquet, atomic_text, run_id
from nfl_ats.key_line_pick_read import (
    KEY_LINE_ATOMS,
    KeyLinePickRead,
    apply_key_line_pick_read,
    apply_pick_overrides,
    load_forecast_key_line_pick_read,
    served_pick_overrides,
)
from nfl_ats.lines import apply_external_lines
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, MarginFeatureProfile
from nfl_ats.market_data import load_quote_history, spread_consensus
from nfl_ats.mass_preserving_lattice import (
    THREE_WAY_COLUMNS,
    load_forecast_discrete_push_read,
    serve_discrete_three_way,
)
from nfl_ats.nfl_week import week_cycle_sunday
from nfl_ats.outcomes import MARGIN_DISTRIBUTION_METHODS, fit_margin_models_for_week
from nfl_ats.prediction_safety import validate_three_way_split
from nfl_ats.provenance import sha256_file
from nfl_ats.public_betting_live import HandleReading, load_latest_public_handle
from nfl_ats.sharp_book_movement_features import (
    LEADER_FOLLOW_BIG_SPREAD_LINE,
    LEADER_FOLLOW_BIG_SPREAD_THRESHOLD,
    late_week_follow_frame,
    leader_follow_threshold,
)
from nfl_ats.sharp_book_movement_features import (
    LEADER_FOLLOW_THRESHOLD as LATE_WEEK_FOLLOW_THRESHOLD,
)
from nfl_ats.sharp_book_movement_features import (
    THRESHOLD as LATE_WEEK_FOLLOW_OFF_THRESHOLD,
)
from nfl_ats.weekly import CARD_PATH_TABLES

if TYPE_CHECKING:
    from nfl_ats.best_pick_renomination import SundayRenomination

PICK_LOCK_TIMEZONE = ZoneInfo("America/New_York")
SUNDAY_PICK_LOCK_LOCAL_TIME = time(16, 0)


def sunday_pick_lock(kickoffs: pd.Series) -> pd.Timestamp:

    valid = pd.to_datetime(kickoffs, utc=True, errors="coerce").dropna()
    if valid.empty:
        raise ValueError("Cannot anchor a Sunday pick lock with no kickoffs")
    local_dates = valid.dt.tz_convert(PICK_LOCK_TIMEZONE).dt.date
    cycle_sundays: Counter[date] = Counter(week_cycle_sunday(day) for day in local_dates)
    anchor = min(cycle_sundays.items(), key=lambda item: (-item[1], item[0]))[0]
    local = datetime.combine(anchor, SUNDAY_PICK_LOCK_LOCAL_TIME, tzinfo=PICK_LOCK_TIMEZONE)
    return pd.Timestamp(local).tz_convert("UTC")


PRODUCTION_COMPOSITION_POLICY_IDS: tuple[str, ...] = (
    "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1",
    "overlay_union_coach_division_revenge_player_arrests_v2",
    "overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3",
)


def pick_deadline(kickoff: pd.Timestamp, sunday_lock: pd.Timestamp) -> pd.Timestamp:

    return min(kickoff, sunday_lock)


MOVEMENT_POLICY_THRESHOLD = 1.0
MOVEMENT_POLICY_MOVEMENT = "movement_ge_1.0"
MOVEMENT_POLICY_MODEL_ONLY = "model_only"
CONSENSUS_MOVEMENT_OFF_CHALLENGER_ID = "consensus_movement_1_0_off_incumbent"

LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY = "late_week_leader_median_follow_1_0_big_spread_0_5"
LATE_WEEK_FOLLOW_NEWS_VETO_POLICY = "late_week_leader_median_follow_1_0_big_spread_0_5_news_veto"
FOLLOW_NEWS_VETO_REASON = "The line moved, but the injury report points the other way."
FOLLOW_BIG_SPREAD_REASON = (
    "On a spread this big, even half a point from the books that move first is a signal."
)
LATE_WEEK_FOLLOW_THRESHOLD_DESCRIPTION = (
    "a full point, or half a point once the frozen line reaches 10.5"
)

MOVEMENT_GOVERNED_POLICIES = (
    MOVEMENT_POLICY_MOVEMENT,
    LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
    LATE_WEEK_FOLLOW_NEWS_VETO_POLICY,
)

HANDLE_FOLLOW_POLICY = "handle_follow_0_70"
HANDLE_FOLLOW_MONEY_THRESHOLD = 70.0
HANDLE_FOLLOW_REASON = "Followed the heavy-money side"
HANDLE_READING_LOCAL_TIME = time(12, 0)


def handle_reading_opens(sunday_lock: pd.Timestamp) -> pd.Timestamp:

    saturday = sunday_lock.tz_convert(PICK_LOCK_TIMEZONE).date() - timedelta(days=1)
    local = datetime.combine(saturday, HANDLE_READING_LOCAL_TIME, tzinfo=PICK_LOCK_TIMEZONE)
    return pd.Timestamp(local).tz_convert("UTC")


BEST_PICK_RENOMINATION_REASON = "The Best Pick moved to this game on the Sunday refresh"

ROOKIE_CREW_POLICY = "rookie_crew_underdog_v1"
ROOKIE_CREW_SEASON_FLOOR = 2010
ROOKIE_CREW_PROFILE = "weak_stack_rookie_crew_underdog"
ROOKIE_CREW_REASON = "The officiating crew is new this season."


def _movement_side(delta: float) -> str:

    return "HOME" if delta > 0.0 else "AWAY"


def current_captured_home_spread(
    data_root: Path, *, now: pd.Timestamp
) -> tuple[dict[str, float], dict[str, Any]]:

    market_root = data_root / "market" / "raw"
    now_floor = pd.Timestamp(now)
    now_floor = (
        now_floor.tz_localize("UTC") if now_floor.tzinfo is None else now_floor.tz_convert("UTC")
    )
    quotes = load_quote_history(market_root, since=now_floor - pd.Timedelta(days=21))
    if quotes.empty:
        return {}, {
            "fresh": False,
            "reason": "no_market_snapshots",
            "latest_observed_at_utc": None,
            "games_with_current_line": 0,
        }
    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    latest_observed = observed.max()
    now_ts = pd.Timestamp(now)
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    if (
        latest_observed.tz_convert(PICK_LOCK_TIMEZONE).date()
        != now_ts.tz_convert(PICK_LOCK_TIMEZONE).date()
    ):
        return {}, {
            "fresh": False,
            "reason": "latest_capture_not_from_today",
            "latest_observed_at_utc": latest_observed.isoformat(),
            "games_with_current_line": 0,
        }
    consensus = spread_consensus(quotes)
    lines = {
        str(game_id): float(spread)
        for game_id, spread in zip(
            consensus["nflverse_game_id"], consensus["consensus_home_spread"], strict=True
        )
        if pd.notna(spread)
    }
    return lines, {
        "fresh": True,
        "reason": "",
        "latest_observed_at_utc": latest_observed.isoformat(),
        "games_with_current_line": len(lines),
    }


PICK_REVISION_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "decision_home_spread",
    "original_recorded_at_utc",
    "previous_pick_side",
    "previous_home_cover_probability",
    "new_pick_side",
    "new_home_cover_probability",
    "decision_policy_id",
    "decision_policy_fingerprint",
    "coach_fade_flip",
    "division_revenge_flip",
    "player_arrests_flip",
    "spread_gap_zone_flip",
    "composed_overlay_flip",
    "player_arrests_snapshot_id",
    "player_arrests_safe_index_sha256",
    "movement_policy",
    "movement_delta",
    "movement_pick_side",
    "model_only_pick_side",
    "late_week_net_move",
    "late_week_pick_side",
    "late_week_eligible_books",
    "late_week_threshold_applied",
    "consensus_delta",
    "consensus_pick_side",
    "handle_pick_side",
    "handle_money_pct",
    "handle_ticket_pct",
    "handle_pre_rule_pick_side",
    "rookie_crew_flag",
    "rookie_crew_referee",
    "rookie_crew_pick_side",
    "follow_news_veto",
    "follow_news_source",
    "follow_news_team",
    "best_pick_before",
    "best_pick_after",
    "model_id",
    "feature_table_sha256",
    "reason",
    "trigger_type",
    "trigger_source",
    "trigger_observed_at_utc",
)


TRIGGER_CLOCK_DISPATCH = "clock_dispatch"
TRIGGER_NEWS_EVENT = "news_event"
TRIGGER_UNKNOWN = "unknown"


def pick_revision_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "pick_revisions.parquet"


def load_pick_revisions(artifacts_root: Path) -> pd.DataFrame:

    path = pick_revision_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(PICK_REVISION_COLUMNS))
    ledger = pd.read_parquet(path)
    legacy_defaults: dict[str, Any] = {
        "player_arrests_flip": False,
        "division_revenge_flip": False,
        "spread_gap_zone_flip": False,
        "composed_overlay_flip": False,
        "decision_policy_id": "legacy_model_only",
        "decision_policy_fingerprint": "",
        "player_arrests_snapshot_id": "",
        "player_arrests_safe_index_sha256": "",
        "trigger_type": TRIGGER_UNKNOWN,
        "trigger_source": "",
        "trigger_observed_at_utc": pd.NaT,
        "late_week_net_move": None,
        "late_week_pick_side": "",
        "late_week_eligible_books": 0,
        "late_week_threshold_applied": LATE_WEEK_FOLLOW_THRESHOLD,
        "consensus_delta": None,
        "consensus_pick_side": "",
        "handle_pick_side": "",
        "handle_money_pct": None,
        "handle_ticket_pct": None,
        "handle_pre_rule_pick_side": "",
        "rookie_crew_flag": 0.0,
        "rookie_crew_referee": "",
        "rookie_crew_pick_side": "",
        "follow_news_veto": False,
        "follow_news_source": "",
        "follow_news_team": "",
        "best_pick_before": "",
        "best_pick_after": "",
    }
    for column, default in legacy_defaults.items():
        if column not in ledger.columns:
            ledger[column] = default
    missing = sorted(set(PICK_REVISION_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(f"Pick-revision ledger is missing columns: {', '.join(missing)}")
    return ledger[list(PICK_REVISION_COLUMNS)]


def served_best_pick(artifacts_root: Path, *, season: int, week: int) -> str | None:

    try:
        revisions = load_pick_revisions(artifacts_root)
    except (OSError, ValueError, DataContractError):
        return None
    if revisions.empty:
        return None
    after = revisions["best_pick_after"].fillna("").astype(str)
    rows = revisions.loc[
        revisions["season"].astype(int).eq(season)
        & revisions["week"].astype(int).eq(week)
        & after.ne("")
    ]
    if rows.empty:
        return None
    latest = rows.sort_values("revision_recorded_at_utc").iloc[-1]
    return str(latest["best_pick_after"])


def describe_week_revisions(
    revisions: pd.DataFrame,
    games: tuple[tuple[str, str, str], ...],
    *,
    season: int | None,
    week: int | None,
) -> tuple[str, ...]:

    if revisions.empty or season is None or week is None:
        return ()
    scoped = revisions.loc[
        revisions["season"].astype(int).eq(int(season))
        & revisions["week"].astype(int).eq(int(week))
    ]
    if scoped.empty:
        return ()
    matchup = {game_id: (away, home) for game_id, away, home in games}
    ordered = scoped.sort_values("revision_recorded_at_utc").groupby("game_id").tail(1)
    lines = []
    for _, revision in ordered.iterrows():
        game_id = str(revision["game_id"])
        teams = matchup.get(game_id)
        if teams is None:
            continue
        away, home = teams
        new_side = str(revision["new_pick_side"])
        previous_side = str(revision["previous_pick_side"])
        spread = revision["decision_home_spread"]
        spread_text = (
            f"frozen Tuesday line (home {float(spread):+g})"
            if pd.notna(spread)
            else "frozen Tuesday line"
        )
        run_id = str(revision.get("refresh_run_id", "") or "refresh pass")
        trigger = str(revision.get("trigger_type", "") or "")
        trigger_text = " (news-triggered)" if trigger == "news_event" else ""
        policy_name = str(revision.get("movement_policy", "") or "")
        crew_text = f" {ROOKIE_CREW_REASON}" if policy_name == ROOKIE_CREW_POLICY else ""
        if policy_name == LATE_WEEK_FOLLOW_NEWS_VETO_POLICY:
            crew_text = f" {FOLLOW_NEWS_VETO_REASON}"
        if policy_name == LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY and (
            pd.notna(spread) and abs(float(spread)) >= LEADER_FOLLOW_BIG_SPREAD_LINE
        ):
            crew_text = f" {FOLLOW_BIG_SPREAD_REASON}"
        if new_side == previous_side:
            lines.append(
                f"{away} at {home} refresh ({run_id}){trigger_text}: "
                f"refresh confirmed {new_side}, no change from Tuesday; {spread_text}.{crew_text}"
            )
        else:
            delta = revision.get("movement_delta")
            movement_text = f"; line moved {float(delta):+g} points" if pd.notna(delta) else ""
            lines.append(
                f"{away} at {home} refresh ({run_id}){trigger_text}: "
                f"pick now {new_side} (Tuesday card: {previous_side}); "
                f"{spread_text}{movement_text}.{crew_text}"
            )
    return tuple(lines)


def original_card(artifacts_root: Path, *, season: int, week: int) -> pd.DataFrame:

    ledger = load_paper_decisions(artifacts_root)
    if ledger.empty:
        return ledger
    rows = ledger.loc[ledger["season"].astype(int).eq(season) & ledger["week"].astype(int).eq(week)]
    return rows.reset_index(drop=True)


MODEL_CONFIGURATION_KEYS = (
    "method",
    "feature_profile",
    "regressor",
    "ridge_alpha",
    "calibration_method",
    "probability_method",
)


def model_configuration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    method = manifest.get("method", manifest.get("ats_method"))
    ridge_alpha = manifest.get("ridge_alpha")
    return {
        "method": None if method is None else str(method),
        "feature_profile": (
            None if manifest.get("feature_profile") is None else str(manifest["feature_profile"])
        ),
        "regressor": None if manifest.get("regressor") is None else str(manifest["regressor"]),
        "ridge_alpha": None if ridge_alpha is None else float(ridge_alpha),
        "calibration_method": str(manifest.get("calibration_method") or "none"),
        "probability_method": (
            None
            if manifest.get("probability_method") is None
            else str(manifest["probability_method"])
        ),
    }


def recorded_card_configuration(
    artifacts_root: Path, original: pd.DataFrame
) -> dict[str, Any] | None:
    if "forecast_artifact" not in original.columns:
        return None
    configurations: list[dict[str, Any]] = []
    for artifact in sorted(set(original["forecast_artifact"].astype(str))):
        metadata_path = artifacts_root / artifact / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        configuration = model_configuration(metadata)
        if configuration not in configurations:
            configurations.append(configuration)
    if len(configurations) != 1:
        return None
    return configurations[0]


def _utc(instant: datetime | None) -> pd.Timestamp:
    value = pd.Timestamp(instant if instant is not None else datetime.now(UTC))
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _published_pick_side(original: pd.DataFrame) -> pd.Series:

    return original.set_index("game_id")["pick_side"].astype(str)


@dataclass(frozen=True)
class RefreshedGame:
    game_id: str
    home_team: str
    away_team: str
    kickoff: pd.Timestamp
    deadline: pd.Timestamp
    decision_home_spread: float
    original_recorded_at_utc: pd.Timestamp
    previous_pick_side: str
    previous_home_cover_probability: float | None
    new_pick_side: str
    new_home_cover_probability: float
    decision_policy_id: str
    decision_policy_fingerprint: str
    coach_fade_flip: bool
    division_revenge_flip: bool
    player_arrests_flip: bool
    spread_gap_zone_flip: bool
    composed_overlay_flip: bool
    player_arrests_snapshot_id: str
    player_arrests_safe_index_sha256: str
    movement_policy: str
    movement_delta: float | None
    movement_pick_side: str
    model_only_pick_side: str
    eligible: bool
    ineligible_reason: str
    changed: bool
    late_week_net_move: float | None = None
    late_week_pick_side: str = ""
    late_week_eligible_books: int = 0
    late_week_threshold_applied: float = LATE_WEEK_FOLLOW_THRESHOLD
    consensus_delta: float | None = None
    consensus_pick_side: str = ""
    consensus_arm_pick_side: str = ""
    handle_pick_side: str = ""
    handle_money_pct: float | None = None
    handle_ticket_pct: float | None = None
    handle_pre_rule_pick_side: str = ""
    rookie_crew_flag: float = 0.0
    rookie_crew_referee: str = ""
    rookie_crew_pick_side: str = ""
    follow_news_veto: bool = False
    follow_news_source: str = ""
    follow_news_team: str = ""
    published_pick_side: str = ""


@dataclass(frozen=True)
class RefreshResult:
    season: int
    week: int
    refresh_run_id: str
    computed_at_utc: pd.Timestamp
    model_id: str
    feature_table_path: str
    feature_table_sha256: str
    games: tuple[RefreshedGame, ...]
    unrefreshable_game_ids: tuple[str, ...]
    missing_from_features_game_ids: tuple[str, ...]
    current_line_metadata: dict[str, Any] = field(default_factory=dict)
    late_week_metadata: dict[str, Any] = field(default_factory=dict)
    follow_news_metadata: dict[str, Any] = field(default_factory=dict)
    handle_metadata: dict[str, Any] = field(default_factory=dict)
    rookie_crew_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_games(self) -> tuple[RefreshedGame, ...]:
        return tuple(game for game in self.games if game.changed)

    @property
    def card_changed_games(self) -> tuple[RefreshedGame, ...]:
        return tuple(
            game
            for game in self.games
            if game.eligible
            and game.published_pick_side in ("HOME", "AWAY")
            and game.new_pick_side != game.published_pick_side
        )

    @property
    def ineligible_games(self) -> tuple[RefreshedGame, ...]:
        return tuple(game for game in self.games if not game.eligible)


def _active_model_config(
    artifacts_root: Path,
) -> tuple[dict[str, Any], str, MarginFeatureProfile, str, float, ResidualSmoothingMethod]:

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to refresh picks from")
    method = str(active.get("method"))
    if method not in MARGIN_DISTRIBUTION_METHODS:
        raise ValueError(
            "refresh-picks only supports margin-distribution methods "
            f"{MARGIN_DISTRIBUTION_METHODS}; the active model uses {method!r}"
        )
    feature_profile_raw = str(active.get("feature_profile") or "")
    if feature_profile_raw not in MARGIN_FEATURE_PROFILES:
        raise ValueError(
            f"Active model feature_profile {feature_profile_raw!r} is not a known margin "
            f"feature profile ({', '.join(MARGIN_FEATURE_PROFILES)})"
        )
    regressor = str(active.get("regressor") or "ridge")
    ridge_alpha = float(active.get("ridge_alpha") or 10.0)
    probability_method_raw = str(active.get("probability_method", "ecdf"))
    return (
        active,
        method,
        feature_profile_raw,
        regressor,
        ridge_alpha,
        cast(ResidualSmoothingMethod, probability_method_raw),
    )


def _late_week_follow_lookup(
    original_indexed: pd.DataFrame,
    overlaid: pd.DataFrame,
    data_root: Path,
    *,
    sunday_lock: pd.Timestamp,
    now: pd.Timestamp,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:

    def _unavailable(reason: str, refused: int = 0) -> tuple[dict, dict[str, Any]]:
        return {}, {
            "available": False,
            "reason": reason,
            "games_with_exposure": 0,
            "games_followed": 0,
            "refused_quote_rows": refused,
        }

    try:
        quotes = load_decision_quotes(data_root / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    except (ValueError, FileNotFoundError, DataContractError) as error:
        return _unavailable(f"live intraday odds archive is unreadable: {error}")
    if quotes.empty:
        return _unavailable("live intraday odds archive is absent")
    games: list[dict[str, Any]] = []
    for game_id in overlaid.index.astype(str):
        game_id = str(game_id)
        if game_id not in original_indexed.index:
            continue
        kickoff = pd.Timestamp(cast(Any, overlaid.loc[game_id, "kickoff"]))
        if now >= pick_deadline(kickoff, sunday_lock):
            continue
        games.append(
            {
                "game_id": game_id,
                "commence_time_utc": kickoff,
                "week_first_commence_utc": sunday_lock,
                "cutoff_utc": now,
                "decision_home_spread": original_indexed.loc[game_id, "decision_home_spread"],
            }
        )
    if not games:
        return _unavailable("No games remain before their pick deadline.")
    try:
        exposure, refused = late_week_follow_frame(
            quotes,
            pd.DataFrame(games),
            now=now,
            tuesday_pick_side=original_indexed["pick_side"].astype(str),
        )
    except (ValueError, FileNotFoundError, DataContractError, KeyError) as error:
        return _unavailable(f"late-week exposure is unusable: {error}")
    if not exposure.eligible_books.gt(0).any():
        return _unavailable("No pre-deadline late-week book changes are available.", refused)
    lookup: dict[str, dict[str, Any]] = {}
    for row in exposure.itertuples():
        lookup[str(row.game_id)] = {
            "net_move": float(cast(Any, row.leader_median_net_move)),
            "pick_side": str(row.movement_would_be_pick_side),
            "tuesday_pick_side": str(row.tuesday_pick_side),
            "eligible_books": int(cast(Any, row.leader_books)),
            "threshold": float(cast(Any, row.late_week_threshold_applied)),
            "off_threshold_pick_side": str(row.leader_median_half_would_be_pick_side),
            "flat_threshold_pick_side": str(row.leader_median_flat_would_be_pick_side),
            "equal_net_move": float(cast(Any, row.equal_net_move)),
            "equal_pick_side": str(row.equal_would_be_pick_side),
            "equal_eligible_books": int(cast(Any, row.eligible_books)),
        }

    def _fires(value: dict[str, Any], threshold: float) -> bool:
        return (
            cast(int, value["eligible_books"]) > 0
            and abs(cast(float, value["net_move"])) >= threshold
        )

    followed = sum(1 for value in lookup.values() if _fires(value, cast(float, value["threshold"])))
    flat_followed = sum(1 for value in lookup.values() if _fires(value, LATE_WEEK_FOLLOW_THRESHOLD))
    off_followed = sum(
        1 for value in lookup.values() if _fires(value, LATE_WEEK_FOLLOW_OFF_THRESHOLD)
    )
    return lookup, {
        "available": True,
        "reason": "",
        "games_with_exposure": int(exposure.eligible_books.gt(0).sum()),
        "games_followed": followed,
        "games_followed_at_flat_threshold": flat_followed,
        "flat_threshold": LATE_WEEK_FOLLOW_THRESHOLD,
        "games_at_big_spread_gate": sum(
            1 for value in lookup.values() if value["threshold"] < LATE_WEEK_FOLLOW_THRESHOLD
        ),
        "games_followed_at_off_threshold": off_followed,
        "off_threshold": LATE_WEEK_FOLLOW_OFF_THRESHOLD,
        "refused_quote_rows": refused,
        "games": [
            {
                "game_id": game_id,
                "leader_median_net_move": value["net_move"],
                "leader_books": value["eligible_books"],
                "leader_pick_side": value["pick_side"],
                "late_week_threshold_applied": value["threshold"],
                "off_threshold_pick_side": value["off_threshold_pick_side"],
                "flat_threshold_pick_side": value["flat_threshold_pick_side"],
                "equal_net_move": value["equal_net_move"],
                "eligible_books": value["equal_eligible_books"],
                "equal_pick_side": value["equal_pick_side"],
            }
            for game_id, value in lookup.items()
            if cast(int, value["equal_eligible_books"]) > 0
        ],
    }


def _follow_news_veto_lookup(
    data_root: Path,
    late_week_lookup: Mapping[str, dict[str, Any]],
    overlaid: pd.DataFrame,
    *,
    season: int,
    week: int,
    now: pd.Timestamp,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:

    from nfl_ats.injury_signal_refresh_tilt import follow_news_for_game, load_news_sources

    firing = {
        game_id: value
        for game_id, value in late_week_lookup.items()
        if cast(int, value["eligible_books"]) > 0
        and abs(cast(float, value["net_move"])) >= cast(float, value["threshold"])
    }
    if not firing:
        return {}, {
            "available": False,
            "reason": "the late-week follow did not fire on any game this pass",
            "games_evaluated": 0,
            "games_vetoed": 0,
            "games": [],
        }
    try:
        injuries, pft = load_news_sources(data_root)
        lookup: dict[str, dict[str, Any]] = {}
        for game_id, value in firing.items():
            if game_id not in overlaid.index:
                continue
            row = overlaid.loc[game_id]
            reading = follow_news_for_game(
                game_id=game_id,
                season=season,
                week=week,
                kickoff=pd.Timestamp(cast(Any, row["kickoff"])),
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                leader_median_net_move=cast(float, value["net_move"]),
                now=now,
                injuries=injuries,
                pft=pft,
            )
            lookup[game_id] = {
                "source": reading.source,
                "net_toward_market": reading.net_toward_market,
                "moved_toward_team": reading.moved_toward_team,
                "confirms": reading.confirms,
                "contradicts": reading.contradicts,
            }
    except (OSError, ValueError, KeyError, TypeError, DataContractError) as error:
        return {}, {
            "available": False,
            "reason": f"the injury-news reader is unusable: {error}",
            "games_evaluated": 0,
            "games_vetoed": 0,
            "games": [],
        }
    return lookup, {
        "available": True,
        "reason": "",
        "games_evaluated": len(lookup),
        "games_vetoed": sum(1 for value in lookup.values() if value["contradicts"]),
        "games": [
            {
                "game_id": game_id,
                "follow_news_source": value["source"],
                "news_toward_market": value["net_toward_market"],
                "moved_toward_team": value["moved_toward_team"],
                "confirms_the_move": value["confirms"],
                "contradicts_the_move": value["contradicts"],
            }
            for game_id, value in lookup.items()
        ],
    }


def _handle_follow_lookup(
    data_root: Path,
    *,
    season: int,
    week: int,
    sunday_lock: pd.Timestamp,
    now: pd.Timestamp,
) -> tuple[dict[str, HandleReading], dict[str, Any]]:

    opens = handle_reading_opens(sunday_lock)
    if now < opens:
        return {}, {
            "available": False,
            "reason": "before_saturday_noon_et",
            "snapshot": "",
            "captured_at_utc": None,
            "reading_opens_utc": opens.isoformat(),
            "games_with_reading": 0,
        }
    try:
        readings, metadata = load_latest_public_handle(
            data_root, season=season, week=week, before=now
        )
    except (OSError, ValueError, KeyError, DataContractError) as error:
        return {}, {
            "available": False,
            "reason": f"public betting store is unusable: {error}",
            "snapshot": "",
            "captured_at_utc": None,
            "reading_opens_utc": opens.isoformat(),
            "games_with_reading": 0,
        }
    return readings, {**metadata, "reading_opens_utc": opens.isoformat()}


def _rookie_crew_lookup(
    features: pd.DataFrame,
    overridden: pd.DataFrame,
    data_root: Path,
    *,
    season: int,
    week: int,
    decision_spreads: Mapping[str, float],
    deadlines: Mapping[str, pd.Timestamp],
    overlay_flip_game_ids: frozenset[str],
    regressor: str,
    ridge_alpha: float,
    min_train_games: int,
    method: str,
    probability_method: ResidualSmoothingMethod,
    center_offset: np.ndarray | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:

    def _unavailable(reason: str, **extra: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        return {}, {
            "available": False,
            "reason": reason,
            "games_with_rookie_crew": 0,
            "rookie_crew_game_ids": [],
            **extra,
        }

    from nfl_ats.crew_tilt_refresh_overlay import latest_crew_snapshot

    try:
        snapshot = latest_crew_snapshot(data_root, season=season, week=week)
    except (OSError, ValueError, DataContractError) as error:
        return _unavailable(f"the crew-assignment snapshot is unreadable: {error}")
    if snapshot is None or not snapshot.referee_by_game_id:
        return _unavailable("no officiating-crew assignment is published for this week")

    in_window = {
        game_id: referee
        for game_id, referee in snapshot.referee_by_game_id.items()
        if game_id in deadlines and snapshot.captured_at_utc < deadlines[game_id]
    }
    if not in_window:
        return _unavailable(
            "the published crew assignment arrived at or after every game's pick deadline",
            crew_snapshot_id=snapshot.snapshot_id,
        )

    repo_root = data_root.parent
    try:
        from nfl_ats.experiment_runner import _REFEREE_POSITION, _REFEREE_SEASON_TYPE
        from nfl_ats.officials_archive import load_officials
        from nfl_ats.officials_flag_features import (
            ROOKIE_CREW_UNDERDOG_COLUMN,
            ROOKIE_PRIOR_EXPERIENCE_MAX,
        )
        from nfl_ats.schedule_flag_features import default_opener_lines
        from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

        officials = load_officials(repo_root, include_archive=True)
        crews = officials.loc[
            officials["position"].eq(_REFEREE_POSITION)
            & officials["season_type"].eq(_REFEREE_SEASON_TYPE),
            ["game_id", "official_name"],
        ]
        schedules = pd.read_parquet(latest_schedules_snapshot(repo_root)).loc[
            :, ["game_id", "old_game_id", "season", "week"]
        ]
        history = (
            crews.merge(
                schedules.rename(columns={"season": "crew_season"}),
                left_on="game_id",
                right_on="old_game_id",
                how="inner",
                suffixes=("_legacy", ""),
            )
            .loc[:, ["game_id", "official_name", "crew_season"]]
            .rename(columns={"crew_season": "season"})
            .astype({"season": int})
        )
        tenure = (
            history.loc[:, ["official_name", "season"]]
            .drop_duplicates()
            .sort_values(["official_name", "season"])
            .reset_index(drop=True)
        )
        tenure["prior_seasons_experience"] = tenure.groupby("official_name").cumcount()
        history = history.merge(tenure, on=["official_name", "season"], how="left")
    except (OSError, ValueError, KeyError, DataContractError) as error:
        return _unavailable(f"the officiating-crew history is unreadable: {error}")

    prior_seasons = tenure.loc[tenure["season"].lt(int(season))].groupby("official_name").size()
    rookie_sides: dict[str, dict[str, Any]] = {}
    for game_id, referee in in_window.items():
        spread = decision_spreads.get(game_id)
        if spread is None or not np.isfinite(spread) or float(spread) == 0.0:
            continue
        if int(prior_seasons.get(str(referee), 0)) > ROOKIE_PRIOR_EXPERIENCE_MAX:
            continue
        rookie_sides[game_id] = {
            "referee": str(referee),
            "flag": 1.0 if float(spread) < 0.0 else -1.0,
        }
    if not rookie_sides:
        return _unavailable(
            "no game this week is worked by a first- or second-season officiating crew",
            crew_snapshot_id=snapshot.snapshot_id,
        )

    try:
        openers = default_opener_lines(
            schedules.loc[:, ["game_id", "season", "week"]],
            market_root=data_root / "market" / "raw",
        )
        proxy = features.loc[:, ["game_id", "spread_line"]].copy()
        proxy["game_id"] = proxy["game_id"].astype(str)
        proxy = proxy.merge(
            openers.loc[:, ["game_id", "tue_open_home_spread"]], on="game_id", how="left"
        )
        graded_line = proxy["tue_open_home_spread"].fillna(proxy["spread_line"])
        line_by_game = dict(zip(proxy["game_id"], graded_line, strict=True))

        history["game_id"] = history["game_id"].astype(str)
        line = pd.to_numeric(history["game_id"].map(line_by_game), errors="coerce")
        rookie = history["prior_seasons_experience"].le(ROOKIE_PRIOR_EXPERIENCE_MAX) & history[
            "season"
        ].ge(ROOKIE_CREW_SEASON_FLOOR)
        history_flag = pd.Series(
            np.where(rookie & line.lt(0.0), 1.0, np.where(rookie & line.gt(0.0), -1.0, 0.0)),
            index=history.index,
        )
        archive = (
            pd.DataFrame({"game_id": history["game_id"], "flag": history_flag})
            .drop_duplicates("game_id")
            .set_index("game_id")["flag"]
        )
        forward = pd.Series({key: float(read["flag"]) for key, read in rookie_sides.items()})

        def _flag_for(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
            ids = frame["game_id"].astype(str)
            signed = ids.map(forward).fillna(ids.map(archive)).fillna(0.0)
            return np.asarray(signed.to_numpy(), dtype=np.float64)

        trained = features.assign(**{ROOKIE_CREW_UNDERDOG_COLUMN: _flag_for(features)})
        scoring = overridden.assign(**{ROOKIE_CREW_UNDERDOG_COLUMN: _flag_for(overridden)})
        _target, refit = fit_margin_models_for_week(
            trained,
            season=season,
            week=week,
            regressor=regressor,
            min_train_games=min_train_games,
            feature_profile=cast(MarginFeatureProfile, ROOKIE_CREW_PROFILE),
            ridge_alpha=ridge_alpha,
            methods=(method,),
        )
        forecasts = refit[method].predict(
            scoring, probability_method=probability_method, center_offset=center_offset
        )
    except (OSError, ValueError, KeyError, DataContractError) as error:
        return _unavailable(
            f"the rookie-crew refit is unavailable: {error}",
            crew_snapshot_id=snapshot.snapshot_id,
        )

    probabilities = pd.to_numeric(forecasts["home_cover_probability"], errors="coerce").to_numpy(
        dtype=float
    )
    lookup: dict[str, dict[str, Any]] = {}
    for game_id, probability in zip(
        scoring["game_id"].astype(str).to_numpy(), probabilities, strict=True
    ):
        read = rookie_sides.get(str(game_id))
        if read is None or not np.isfinite(probability):
            continue
        composed = 1.0 - probability if str(game_id) in overlay_flip_game_ids else probability
        lookup[str(game_id)] = {
            "referee": read["referee"],
            "flag": float(read["flag"]),
            "side": "HOME" if composed >= 0.5 else "AWAY",
        }
    return lookup, {
        "available": True,
        "reason": "",
        "crew_snapshot_id": snapshot.snapshot_id,
        "crew_captured_at_utc": snapshot.captured_at_utc.isoformat(),
        "games_with_rookie_crew": len(lookup),
        "rookie_crew_game_ids": sorted(lookup),
    }


def plan_refresh(
    artifacts_root: Path,
    data_root: Path,
    *,
    season: int,
    week: int,
    features_path: Path | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    now: datetime | None = None,
) -> RefreshResult:

    active, method, feature_profile, regressor, ridge_alpha, probability_method = (
        _active_model_config(artifacts_root)
    )
    model_id = str(active.get("model_id"))

    original = original_card(artifacts_root, season=season, week=week)
    if original.empty:
        raise ValueError(
            f"No recorded original card for {season} week {week}: refresh-picks grades "
            "against the Tuesday paper-decision ledger, which is only written by "
            "`publish-predictions --record-decisions`. Run that first."
        )
    mismatched_model = sorted(set(original["model_id"].astype(str)) - {model_id})
    if mismatched_model:
        recorded_configuration = recorded_card_configuration(artifacts_root, original)
        active_configuration = model_configuration(active)
        if recorded_configuration is None or recorded_configuration != active_configuration:
            raise ValueError(
                "The active model has changed since this week's original card was recorded "
                f"(recorded model_id(s) {mismatched_model}, active model_id {model_id!r}; "
                f"recorded configuration {recorded_configuration}, active configuration "
                f"{active_configuration}); refresh-picks refuses to recompute picks under a "
                "different model identity than the one the pool's grading lines were "
                "locked against."
            )

    if features_path is not None:
        resolved_features_path = features_path
    elif feature_profile in CARD_PATH_TABLES:
        resolved_features_path = data_root / "processed" / CARD_PATH_TABLES[feature_profile]
    else:
        raise ValueError(
            f"No --features path was given and {feature_profile!r} has no known default "
            f"card-path table (nfl_ats.weekly.CARD_PATH_TABLES only knows "
            f"{sorted(CARD_PATH_TABLES)}); pass --features explicitly."
        )
    if not resolved_features_path.is_file():
        raise FileNotFoundError(f"Feature table not found: {resolved_features_path}")
    features = pd.read_parquet(resolved_features_path)

    target, margin_models = fit_margin_models_for_week(
        features,
        season=season,
        week=week,
        regressor=regressor,
        min_train_games=min_train_games,
        feature_profile=feature_profile,
        ridge_alpha=ridge_alpha,
        methods=(method,),
    )
    model = margin_models[method]
    target = target.copy()
    target["game_id"] = target["game_id"].astype(str)
    target["kickoff"] = pd.to_datetime(target["kickoff"], utc=True, errors="coerce")
    if target["kickoff"].isna().any():
        raise DataContractError("Current feature table has games without a kickoff timestamp")

    original_game_ids = set(original["game_id"].astype(str))
    target_game_ids = set(target["game_id"])
    unrefreshable = tuple(sorted(target_game_ids - original_game_ids))
    missing_from_features = tuple(sorted(original_game_ids - target_game_ids))
    refreshable = target.loc[target["game_id"].isin(original_game_ids)].copy()

    computed_at = _utc(now)
    refresh_id = run_id(computed_at.to_pydatetime())
    feature_sha = sha256_file(resolved_features_path)

    games: tuple[RefreshedGame, ...] = ()
    line_metadata: dict[str, Any] = {}
    late_week_metadata: dict[str, Any] = {}
    follow_news_metadata: dict[str, Any] = {}
    handle_metadata: dict[str, Any] = {}
    rookie_crew_metadata: dict[str, Any] = {}
    if not refreshable.empty:
        lines = original[["game_id", "decision_home_spread"]].rename(
            columns={"decision_home_spread": "home_spread"}
        )
        overridden = apply_external_lines(refreshable, lines)
        center_offset = _served_home_side_center_offset(artifacts_root, active, overridden)
        forecasts = model.predict(
            overridden, probability_method=probability_method, center_offset=center_offset
        )
        forecasts = _served_lattice_reads(
            artifacts_root,
            active,
            features,
            overridden,
            forecasts,
            residuals=model.residuals,
            probability_method=probability_method,
            season=season,
            week=week,
        )
        identity_columns = [
            column
            for column in (
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "kickoff",
                "spread_line",
                "game_type",
            )
            if column in overridden.columns
        ]
        scored = pd.concat(
            [
                overridden.loc[:, identity_columns].reset_index(drop=True),
                forecasts.reset_index(drop=True),
            ],
            axis=1,
        )
        validate_three_way_split(scored, line_column="spread_line")

        original_indexed = original.set_index("game_id")
        sunday_lock = sunday_pick_lock(original["kickoff"])
        original_deadlines = pd.to_datetime(original["kickoff"], utc=True, errors="coerce").map(
            lambda kickoff: pick_deadline(kickoff, sunday_lock)
        )
        revisable = original.loc[original_deadlines.gt(computed_at)]
        policy_ids = set(
            (revisable if not revisable.empty else original)["decision_policy_id"].astype(str)
        )
        if len(policy_ids) != 1 or not policy_ids <= set(PRODUCTION_COMPOSITION_POLICY_IDS):
            raise DataContractError(
                "Refresh requires one frozen production composition policy across the games it "
                f"can still revise; found {sorted(policy_ids)}"
            )
        overlaid_frame = scored.reset_index(drop=True).copy()
        frozen_union = (
            overlaid_frame["game_id"]
            .astype(str)
            .map(original_indexed["composed_overlay_flip"].astype(bool))
        )
        if frozen_union.isna().any():
            raise DataContractError("Tuesday paper ledger is missing frozen composition flags")
        overlaid_frame.loc[frozen_union.astype(bool), "home_cover_probability"] = (
            1.0 - overlaid_frame.loc[frozen_union.astype(bool), "home_cover_probability"]
        )
        overlaid = overlaid_frame.set_index("game_id")

        published_side = _published_pick_side(original)
        current_lines, line_metadata = current_captured_home_spread(data_root, now=computed_at)
        late_week_lookup, late_week_metadata = _late_week_follow_lookup(
            original_indexed,
            overlaid,
            data_root,
            sunday_lock=sunday_lock,
            now=computed_at,
        )
        follow_news_lookup, follow_news_metadata = _follow_news_veto_lookup(
            data_root,
            late_week_lookup,
            overlaid,
            season=season,
            week=week,
            now=computed_at,
        )
        handle_lookup, handle_metadata = _handle_follow_lookup(
            data_root,
            season=season,
            week=week,
            sunday_lock=sunday_lock,
            now=computed_at,
        )
        rookie_crew_lookup, rookie_crew_metadata = _rookie_crew_lookup(
            features,
            overridden,
            data_root,
            season=season,
            week=week,
            decision_spreads={
                str(game_id): float(cast(Any, value))
                for game_id, value in original_indexed["decision_home_spread"].items()
                if pd.notna(value)
            },
            deadlines={
                str(game_id): pick_deadline(pd.Timestamp(row["kickoff"]), sunday_lock)
                for game_id, row in overlaid.iterrows()
            },
            overlay_flip_game_ids=frozenset(
                overlaid_frame.loc[frozen_union.astype(bool), "game_id"].astype(str)
            ),
            regressor=regressor,
            ridge_alpha=ridge_alpha,
            min_train_games=min_train_games,
            method=method,
            probability_method=probability_method,
            center_offset=center_offset,
        )

        existing_revisions = load_pick_revisions(artifacts_root)
        week_revisions = existing_revisions.loc[
            existing_revisions["season"].astype(int).eq(season)
            & existing_revisions["week"].astype(int).eq(week)
        ]
        latest_revision = (
            week_revisions.sort_values("revision_recorded_at_utc")
            .groupby("game_id", as_index=False)
            .tail(1)
            .set_index("game_id")
            if not week_revisions.empty
            else week_revisions.set_index("game_id")
        )

        rows: list[RefreshedGame] = []
        for game_id, row in overlaid.iterrows():
            game_id = str(game_id)
            orig_row = original_indexed.loc[game_id]
            kickoff = pd.Timestamp(row["kickoff"])
            deadline = pick_deadline(kickoff, sunday_lock)
            eligible = computed_at < deadline
            if computed_at >= kickoff:
                reason = "kickoff_passed"
            elif computed_at >= sunday_lock:
                reason = "sunday_pick_lock_passed"
            else:
                reason = ""

            if game_id in latest_revision.index:
                prev_side = str(latest_revision.loc[game_id, "new_pick_side"])
                prev_prob_raw = cast(
                    Any, latest_revision.loc[game_id, "new_home_cover_probability"]
                )
                prev_prob = float(prev_prob_raw) if pd.notna(prev_prob_raw) else None
            else:
                prev_side = str(published_side[game_id])
                prev_prob = None

            new_prob = float(row["home_cover_probability"])
            model_only_side = "HOME" if new_prob >= 0.5 else "AWAY"

            decision_home_spread = float(cast(Any, orig_row["decision_home_spread"]))
            current_line = current_lines.get(game_id)
            if current_line is None:
                consensus_delta: float | None = None
                consensus_side = ""
            else:
                consensus_delta = float(current_line) - decision_home_spread
                consensus_side = _movement_side(consensus_delta)
            consensus_fires = (
                consensus_delta is not None and abs(consensus_delta) >= MOVEMENT_POLICY_THRESHOLD
            )
            late_week_threshold = leader_follow_threshold(decision_home_spread)
            late_week = late_week_lookup.get(game_id)
            if late_week is None:
                late_week_net: float | None = None
                late_week_side = ""
                late_week_tuesday_side = ""
                late_week_books = 0
                late_week_fires = False
            else:
                late_week_net = late_week["net_move"]
                late_week_side = late_week["pick_side"]
                late_week_tuesday_side = late_week["tuesday_pick_side"]
                late_week_books = late_week["eligible_books"]
                late_week_threshold = float(late_week["threshold"])
                late_week_fires = (
                    late_week_books > 0
                    and late_week_net is not None
                    and abs(late_week_net) >= late_week_threshold
                )
            follow_news = follow_news_lookup.get(game_id)
            follow_news_source = "" if follow_news is None else str(follow_news["source"])
            follow_news_team = "" if follow_news is None else str(follow_news["moved_toward_team"])
            follow_news_veto = bool(
                late_week_fires and follow_news is not None and follow_news["contradicts"]
            )
            rookie_crew = rookie_crew_lookup.get(game_id)
            rookie_crew_flag = 0.0 if rookie_crew is None else float(rookie_crew["flag"])
            rookie_crew_referee = "" if rookie_crew is None else str(rookie_crew["referee"])
            rookie_crew_side = "" if rookie_crew is None else str(rookie_crew["side"])
            rookie_crew_fires = rookie_crew_side not in ("", model_only_side)

            if late_week_fires:
                policy = (
                    LATE_WEEK_FOLLOW_NEWS_VETO_POLICY
                    if follow_news_veto
                    else LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY
                )
                new_side = (
                    (late_week_tuesday_side or model_only_side)
                    if follow_news_veto
                    else late_week_side
                )
                movement_delta = late_week_net
                movement_pick_side = late_week_side
            else:
                policy = ROOKIE_CREW_POLICY if rookie_crew_fires else MOVEMENT_POLICY_MODEL_ONLY
                new_side = rookie_crew_side if rookie_crew_fires else model_only_side
                if consensus_delta is not None:
                    movement_delta = consensus_delta
                    movement_pick_side = consensus_side
                elif late_week_net is not None:
                    movement_delta = late_week_net
                    movement_pick_side = late_week_side
                else:
                    movement_delta = None
                    movement_pick_side = ""
            handle = handle_lookup.get(game_id)
            handle_side = "" if handle is None else handle.heavy_side
            handle_money = None if handle is None else handle.heavy_money_pct
            handle_ticket = None if handle is None else handle.heavy_ticket_pct
            handle_pre_rule_side = new_side
            if (
                policy == MOVEMENT_POLICY_MODEL_ONLY
                and handle_money is not None
                and handle_money >= HANDLE_FOLLOW_MONEY_THRESHOLD
                and handle_side != new_side
            ):
                policy = HANDLE_FOLLOW_POLICY
                new_side = handle_side
            consensus_arm_side = (
                consensus_side if consensus_fires and not late_week_fires else new_side
            )
            changed = eligible and new_side != prev_side

            rows.append(
                RefreshedGame(
                    game_id=game_id,
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    kickoff=kickoff,
                    deadline=deadline,
                    decision_home_spread=decision_home_spread,
                    original_recorded_at_utc=pd.Timestamp(cast(Any, orig_row["recorded_at_utc"])),
                    previous_pick_side=prev_side,
                    previous_home_cover_probability=prev_prob,
                    new_pick_side=new_side,
                    new_home_cover_probability=new_prob,
                    decision_policy_id=str(orig_row["decision_policy_id"]),
                    decision_policy_fingerprint=str(orig_row["decision_policy_fingerprint"]),
                    coach_fade_flip=bool(orig_row["coach_fade_flip"]),
                    division_revenge_flip=bool(orig_row["division_revenge_flip"]),
                    player_arrests_flip=bool(orig_row["player_arrests_flip"]),
                    spread_gap_zone_flip=bool(orig_row["spread_gap_zone_flip"]),
                    composed_overlay_flip=bool(orig_row["composed_overlay_flip"]),
                    player_arrests_snapshot_id=str(orig_row["player_arrests_snapshot_id"]),
                    player_arrests_safe_index_sha256=str(
                        orig_row["player_arrests_safe_index_sha256"]
                    ),
                    movement_policy=policy,
                    movement_delta=movement_delta,
                    movement_pick_side=movement_pick_side,
                    model_only_pick_side=model_only_side,
                    late_week_net_move=late_week_net,
                    late_week_pick_side=late_week_side,
                    late_week_eligible_books=late_week_books,
                    late_week_threshold_applied=late_week_threshold,
                    consensus_delta=consensus_delta,
                    consensus_pick_side=consensus_side,
                    consensus_arm_pick_side=consensus_arm_side,
                    handle_pick_side=handle_side,
                    handle_money_pct=handle_money,
                    handle_ticket_pct=handle_ticket,
                    handle_pre_rule_pick_side=handle_pre_rule_side,
                    rookie_crew_flag=rookie_crew_flag,
                    rookie_crew_referee=rookie_crew_referee,
                    rookie_crew_pick_side=rookie_crew_side,
                    follow_news_veto=follow_news_veto,
                    follow_news_source=follow_news_source,
                    follow_news_team=follow_news_team,
                    eligible=eligible,
                    ineligible_reason=reason,
                    changed=changed,
                    published_pick_side=str(published_side[game_id]),
                )
            )
        games = tuple(sorted(rows, key=lambda game: game.game_id))

    return RefreshResult(
        season=season,
        week=week,
        refresh_run_id=refresh_id,
        computed_at_utc=computed_at,
        model_id=model_id,
        feature_table_path=str(resolved_features_path),
        feature_table_sha256=feature_sha,
        games=games,
        unrefreshable_game_ids=unrefreshable,
        missing_from_features_game_ids=missing_from_features,
        current_line_metadata=line_metadata,
        late_week_metadata=late_week_metadata,
        follow_news_metadata=follow_news_metadata,
        handle_metadata=handle_metadata,
        rookie_crew_metadata=rookie_crew_metadata,
    )


def refresh_summary(plan: RefreshResult, *, record_decisions: bool) -> dict[str, Any]:

    return {
        "season": plan.season,
        "week": plan.week,
        "refresh_run_id": plan.refresh_run_id,
        "computed_at_utc": plan.computed_at_utc.isoformat(),
        "model_id": plan.model_id,
        "games_considered": len(plan.games),
        "changed_game_ids": [game.game_id for game in plan.changed_games],
        "changed_picks": [
            {
                "game_id": game.game_id,
                "away_team": game.away_team,
                "home_team": game.home_team,
                "kickoff": game.kickoff.isoformat(),
                "previous_pick_side": game.previous_pick_side,
                "new_pick_side": game.new_pick_side,
                "new_home_cover_probability": game.new_home_cover_probability,
                "decision_home_spread": game.decision_home_spread,
                "movement_policy": game.movement_policy,
                "movement_delta": game.movement_delta,
                "model_only_pick_side": game.model_only_pick_side,
                "eligible": game.eligible,
            }
            for game in plan.changed_games
        ],
        "post_kickoff_skipped": [
            game.game_id
            for game in plan.ineligible_games
            if game.ineligible_reason == "kickoff_passed"
        ],
        "sunday_lock_skipped": [
            game.game_id
            for game in plan.ineligible_games
            if game.ineligible_reason == "sunday_pick_lock_passed"
        ],
        "unrefreshable_game_ids": list(plan.unrefreshable_game_ids),
        "missing_from_features_game_ids": list(plan.missing_from_features_game_ids),
        "record_decisions": record_decisions,
        "movement_policy": {
            "threshold": MOVEMENT_POLICY_THRESHOLD,
            "current_line_fresh": bool(plan.current_line_metadata.get("fresh", False)),
            "current_line_reason": plan.current_line_metadata.get("reason", ""),
            "latest_observed_at_utc": plan.current_line_metadata.get("latest_observed_at_utc"),
            "games_with_current_line": plan.current_line_metadata.get("games_with_current_line", 0),
            "games_movement_applied": [
                game.game_id
                for game in plan.games
                if game.movement_policy in MOVEMENT_GOVERNED_POLICIES
            ],
            "games_model_only": [
                game.game_id
                for game in plan.games
                if game.movement_policy == MOVEMENT_POLICY_MODEL_ONLY
            ],
            "consensus_movement": {
                "served": False,
                "retired_policy_id": MOVEMENT_POLICY_MOVEMENT,
                "challenger_id": CONSENSUS_MOVEMENT_OFF_CHALLENGER_ID,
                "games_consensus_applied": [],
                "games_consensus_would_fire": [
                    game.game_id
                    for game in plan.games
                    if game.consensus_delta is not None
                    and abs(game.consensus_delta) >= MOVEMENT_POLICY_THRESHOLD
                ],
                "games_consensus_would_govern": [
                    game.game_id
                    for game in plan.games
                    if game.consensus_delta is not None
                    and abs(game.consensus_delta) >= MOVEMENT_POLICY_THRESHOLD
                    and game.movement_policy
                    not in (
                        LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
                        LATE_WEEK_FOLLOW_NEWS_VETO_POLICY,
                    )
                ],
                "games_consensus_would_change_pick": [
                    game.game_id
                    for game in plan.games
                    if game.consensus_arm_pick_side
                    and game.consensus_arm_pick_side != game.new_pick_side
                ],
            },
            "late_week_follow": {
                "threshold": LATE_WEEK_FOLLOW_THRESHOLD_DESCRIPTION,
                "big_spread_line": LEADER_FOLLOW_BIG_SPREAD_LINE,
                "big_spread_threshold": LEADER_FOLLOW_BIG_SPREAD_THRESHOLD,
                "available": bool(plan.late_week_metadata.get("available", False)),
                "reason": plan.late_week_metadata.get("reason", ""),
                "games_with_exposure": plan.late_week_metadata.get("games_with_exposure", 0),
                "games_followed": plan.late_week_metadata.get("games_followed", 0),
                "games_at_big_spread_gate": plan.late_week_metadata.get(
                    "games_at_big_spread_gate", 0
                ),
                "flat_threshold": plan.late_week_metadata.get(
                    "flat_threshold", LATE_WEEK_FOLLOW_THRESHOLD
                ),
                "games_followed_at_flat_threshold": plan.late_week_metadata.get(
                    "games_followed_at_flat_threshold", 0
                ),
                "off_threshold": plan.late_week_metadata.get(
                    "off_threshold", LATE_WEEK_FOLLOW_OFF_THRESHOLD
                ),
                "games_followed_at_off_threshold": plan.late_week_metadata.get(
                    "games_followed_at_off_threshold", 0
                ),
                "refused_quote_rows": plan.late_week_metadata.get("refused_quote_rows", 0),
                "games": plan.late_week_metadata.get("games", []),
                "games_late_week_follow_applied": [
                    game.game_id
                    for game in plan.games
                    if game.movement_policy == LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY
                ],
                "news_veto": {
                    "available": bool(plan.follow_news_metadata.get("available", False)),
                    "reason": plan.follow_news_metadata.get("reason", ""),
                    "games_evaluated": plan.follow_news_metadata.get("games_evaluated", 0),
                    "games_vetoed": plan.follow_news_metadata.get("games_vetoed", 0),
                    "games": plan.follow_news_metadata.get("games", []),
                    "games_news_veto_applied": [
                        game.game_id
                        for game in plan.games
                        if game.movement_policy == LATE_WEEK_FOLLOW_NEWS_VETO_POLICY
                    ],
                },
            },
            "handle_follow": {
                "threshold_money_pct": HANDLE_FOLLOW_MONEY_THRESHOLD,
                "available": bool(plan.handle_metadata.get("available", False)),
                "reason": plan.handle_metadata.get("reason", ""),
                "snapshot": plan.handle_metadata.get("snapshot", ""),
                "captured_at_utc": plan.handle_metadata.get("captured_at_utc"),
                "reading_opens_utc": plan.handle_metadata.get("reading_opens_utc"),
                "games_with_reading": plan.handle_metadata.get("games_with_reading", 0),
                "games_handle_follow_applied": [
                    game.game_id
                    for game in plan.games
                    if game.movement_policy == HANDLE_FOLLOW_POLICY
                ],
            },
        },
        "rookie_crew": {
            "available": bool(plan.rookie_crew_metadata.get("available", False)),
            "reason": plan.rookie_crew_metadata.get("reason", ""),
            "crew_snapshot_id": plan.rookie_crew_metadata.get("crew_snapshot_id"),
            "crew_captured_at_utc": plan.rookie_crew_metadata.get("crew_captured_at_utc"),
            "games_with_rookie_crew": plan.rookie_crew_metadata.get("games_with_rookie_crew", 0),
            "rookie_crew_game_ids": plan.rookie_crew_metadata.get("rookie_crew_game_ids", []),
            "games_rookie_crew_applied": [
                game.game_id for game in plan.games if game.movement_policy == ROOKIE_CREW_POLICY
            ],
            "rookie_crew_reads": [
                {
                    "game_id": game.game_id,
                    "matchup": f"{game.away_team} at {game.home_team}",
                    "referee": game.rookie_crew_referee,
                    "underdog_is_home": game.rookie_crew_flag > 0.0,
                    "decision_home_spread": game.decision_home_spread,
                    "rookie_crew_pick_side": game.rookie_crew_pick_side,
                    "model_only_pick_side": game.model_only_pick_side,
                    "served_pick_side": game.new_pick_side,
                    "changes_the_served_pick": game.movement_policy == ROOKIE_CREW_POLICY,
                }
                for game in plan.games
                if game.rookie_crew_flag != 0.0
            ],
        },
    }


def record_plan(
    artifacts_root: Path,
    plan: RefreshResult,
    *,
    note: str = "",
    record_decisions: bool = False,
    trigger_type: str = TRIGGER_CLOCK_DISPATCH,
    trigger_source: str = "",
    trigger_observed_at_utc: datetime | None = None,
    renomination: SundayRenomination | None = None,
) -> dict[str, Any]:

    if not record_decisions:
        return {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append changed picks to the pick-revision ledger",
        }

    original = original_card(artifacts_root, season=plan.season, week=plan.week)
    refuse_if_outside_recording_lock_window(
        original["kickoff"], plan.computed_at_utc, ledger="pick-revision"
    )
    existing = load_pick_revisions(artifacts_root)
    changed = [game for game in plan.changed_games if game.eligible]
    star_moved = renomination is not None and renomination.moved
    star_game_id = renomination.game_id if star_moved and renomination is not None else ""
    if star_moved and star_game_id not in {game.game_id for game in changed}:
        starred = next(
            (game for game in plan.games if game.game_id == star_game_id and game.eligible), None
        )
        if starred is None:
            star_moved = False
            star_game_id = ""
        else:
            changed = [*changed, starred]
    if not changed:
        return {"recorded": 0, "ledger_rows": len(existing)}

    reason_text = f"pick_refresh recompute ({note})" if note else "pick_refresh recompute"
    star_reason = (
        f"{BEST_PICK_RENOMINATION_REASON} ({note})" if note else BEST_PICK_RENOMINATION_REASON
    )
    handle_reason = f"{HANDLE_FOLLOW_REASON} ({note})" if note else HANDLE_FOLLOW_REASON
    veto_reason = f"{FOLLOW_NEWS_VETO_REASON} ({note})" if note else FOLLOW_NEWS_VETO_REASON
    big_spread_reason = f"{FOLLOW_BIG_SPREAD_REASON} ({note})" if note else FOLLOW_BIG_SPREAD_REASON
    observed_at = _utc(
        trigger_observed_at_utc if trigger_observed_at_utc is not None else plan.computed_at_utc
    )
    rows = pd.DataFrame(
        {
            "revision_recorded_at_utc": plan.computed_at_utc,
            "refresh_run_id": plan.refresh_run_id,
            "season": plan.season,
            "week": plan.week,
            "game_id": [game.game_id for game in changed],
            "home_team": [game.home_team for game in changed],
            "away_team": [game.away_team for game in changed],
            "kickoff": [game.kickoff for game in changed],
            "decision_home_spread": [game.decision_home_spread for game in changed],
            "original_recorded_at_utc": [game.original_recorded_at_utc for game in changed],
            "previous_pick_side": [game.previous_pick_side for game in changed],
            "previous_home_cover_probability": [
                game.previous_home_cover_probability for game in changed
            ],
            "new_pick_side": [game.new_pick_side for game in changed],
            "new_home_cover_probability": [game.new_home_cover_probability for game in changed],
            "decision_policy_id": [game.decision_policy_id for game in changed],
            "decision_policy_fingerprint": [game.decision_policy_fingerprint for game in changed],
            "coach_fade_flip": [game.coach_fade_flip for game in changed],
            "division_revenge_flip": [game.division_revenge_flip for game in changed],
            "player_arrests_flip": [game.player_arrests_flip for game in changed],
            "spread_gap_zone_flip": [game.spread_gap_zone_flip for game in changed],
            "composed_overlay_flip": [game.composed_overlay_flip for game in changed],
            "player_arrests_snapshot_id": [game.player_arrests_snapshot_id for game in changed],
            "player_arrests_safe_index_sha256": [
                game.player_arrests_safe_index_sha256 for game in changed
            ],
            "movement_policy": [game.movement_policy for game in changed],
            "movement_delta": [game.movement_delta for game in changed],
            "movement_pick_side": [game.movement_pick_side for game in changed],
            "model_only_pick_side": [game.model_only_pick_side for game in changed],
            "late_week_net_move": [game.late_week_net_move for game in changed],
            "late_week_pick_side": [game.late_week_pick_side for game in changed],
            "late_week_eligible_books": [game.late_week_eligible_books for game in changed],
            "late_week_threshold_applied": [game.late_week_threshold_applied for game in changed],
            "consensus_delta": [game.consensus_delta for game in changed],
            "consensus_pick_side": [game.consensus_pick_side for game in changed],
            "handle_pick_side": [game.handle_pick_side for game in changed],
            "handle_money_pct": [game.handle_money_pct for game in changed],
            "handle_ticket_pct": [game.handle_ticket_pct for game in changed],
            "handle_pre_rule_pick_side": [game.handle_pre_rule_pick_side for game in changed],
            "rookie_crew_flag": [game.rookie_crew_flag for game in changed],
            "rookie_crew_referee": [game.rookie_crew_referee for game in changed],
            "rookie_crew_pick_side": [game.rookie_crew_pick_side for game in changed],
            "follow_news_veto": [game.follow_news_veto for game in changed],
            "follow_news_source": [game.follow_news_source for game in changed],
            "follow_news_team": [game.follow_news_team for game in changed],
            "best_pick_before": (
                renomination.previous_game_id if star_moved and renomination is not None else ""
            ),
            "best_pick_after": star_game_id,
            "model_id": plan.model_id,
            "feature_table_sha256": plan.feature_table_sha256,
            "reason": [
                star_reason
                if game.game_id == star_game_id and not game.changed
                else handle_reason
                if game.movement_policy == HANDLE_FOLLOW_POLICY
                else veto_reason
                if game.movement_policy == LATE_WEEK_FOLLOW_NEWS_VETO_POLICY
                else big_spread_reason
                if game.movement_policy == LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY
                and game.late_week_threshold_applied < LATE_WEEK_FOLLOW_THRESHOLD
                else reason_text
                for game in changed
            ],
            "trigger_type": trigger_type,
            "trigger_source": trigger_source,
            "trigger_observed_at_utc": observed_at,
        }
    )
    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(combined[list(PICK_REVISION_COLUMNS)], pick_revision_ledger_path(artifacts_root))
    return {
        "recorded": len(rows),
        "ledger_rows": len(combined),
        "best_pick_after": star_game_id,
    }


def record_refresh(
    artifacts_root: Path,
    data_root: Path,
    *,
    season: int,
    week: int,
    features_path: Path | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    now: datetime | None = None,
    note: str = "",
    record_decisions: bool = False,
    trigger_type: str = TRIGGER_CLOCK_DISPATCH,
    trigger_source: str = "",
    trigger_observed_at_utc: datetime | None = None,
) -> dict[str, Any]:

    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=season,
        week=week,
        features_path=features_path,
        min_train_games=min_train_games,
        now=now,
    )
    summary = refresh_summary(plan, record_decisions=record_decisions)
    from nfl_ats.best_pick_refresh_prospective import record_best_pick_refresh
    from nfl_ats.best_pick_renomination import (
        plan_best_pick_renomination,
        renomination_summary,
    )

    renomination = plan_best_pick_renomination(artifacts_root, data_root, plan)
    summary["best_pick_renomination"] = renomination_summary(renomination)
    summary["best_pick_refresh_ledger"] = record_best_pick_refresh(
        artifacts_root, data_root, plan, record_decisions=record_decisions
    )
    summary["ledger"] = record_plan(
        artifacts_root,
        plan,
        note=note,
        record_decisions=record_decisions,
        trigger_type=trigger_type,
        trigger_source=trigger_source,
        trigger_observed_at_utc=trigger_observed_at_utc,
        renomination=renomination,
    )
    return summary


def final_pick_per_game(artifacts_root: Path, *, season: int, week: int) -> pd.DataFrame:

    original = original_card(artifacts_root, season=season, week=week)
    if original.empty:
        return original
    revisions = load_pick_revisions(artifacts_root)
    week_revisions = revisions.loc[
        revisions["season"].astype(int).eq(season) & revisions["week"].astype(int).eq(week)
    ]
    latest = (
        week_revisions.sort_values("revision_recorded_at_utc")
        .groupby("game_id", as_index=False)
        .tail(1)
        if not week_revisions.empty
        else week_revisions
    )
    result = original[
        [
            "game_id",
            "home_team",
            "away_team",
            "kickoff",
            "decision_home_spread",
            "pick_side",
            "recorded_at_utc",
        ]
    ].rename(
        columns={"pick_side": "tuesday_pick_side", "recorded_at_utc": "tuesday_recorded_at_utc"}
    )
    result["final_pick_side"] = result["tuesday_pick_side"]
    result["final_recorded_at_utc"] = result["tuesday_recorded_at_utc"]
    result["revised"] = False
    if not latest.empty:
        latest_indexed = latest.set_index("game_id")
        for game_id in latest_indexed.index:
            mask = result["game_id"].eq(game_id)
            result.loc[mask, "final_pick_side"] = latest_indexed.loc[game_id, "new_pick_side"]
            result.loc[mask, "final_recorded_at_utc"] = latest_indexed.loc[
                game_id, "revision_recorded_at_utc"
            ]
            result.loc[mask, "revised"] = True
    return result


LATE_WEEK_REFRESH_START = "<!-- LATE_WEEK_REFRESH:START -->"
LATE_WEEK_REFRESH_END = "<!-- LATE_WEEK_REFRESH:END -->"


def _renomination_sentence(renomination: SundayRenomination | None) -> str:
    if renomination is None or not renomination.moved:
        return ""
    return (
        f"The Best Pick moved to {renomination.matchup} on the Sunday refresh: of the games that "
        "have not kicked off yet, it is the one the model is now most sure about at the spread "
        f"the pool locked on Tuesday. Tuesday's Best Pick was {renomination.previous_matchup}.\n\n"
    )


def _refresh_section_markdown(
    result: RefreshResult, note: str, renomination: SundayRenomination | None = None
) -> str:
    changed = result.card_changed_games
    heading = f"## Late-week refresh (as of {result.computed_at_utc.isoformat()})\n\n"
    label = f" ({note})" if note else ""
    star = _renomination_sentence(renomination)
    if not changed:
        return heading + star + f"No pick changes since the Tuesday card{label}.\n"

    rows = []
    for game in changed:
        estimate = (
            game.new_home_cover_probability
            if game.new_pick_side == "HOME"
            else 1.0 - game.new_home_cover_probability
        )
        market_move = "n/a" if game.movement_delta is None else f"{game.movement_delta:+.2f}"
        rows.append(
            {
                "Matchup": f"{game.away_team} at {game.home_team}",
                "Previous pick": game.published_pick_side,
                "New pick": game.new_pick_side,
                "Model estimate": f"{estimate:.1%}",
                "Policy": game.movement_policy,
                "Market move": market_move,
            }
        )
    table = pd.DataFrame(rows).to_markdown(index=False)
    plural = "s" if len(changed) != 1 else ""
    intro = (
        f"{len(changed)} pick{plural} changed since the Tuesday card{label}, recomputed with "
        "current data but scored at the frozen Tuesday grading line. Only games whose "
        "deadline (their own kickoff, or that week's Sunday 4:00 PM ET if earlier) had not "
        'yet passed were eligible. "Policy" is '
        f"`{LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY}` when the "
        "three leading books moved the line at least a full point since Tuesday -- or half a "
        "point on the biggest spreads, 10.5 or more -- and the pick "
        f"followed them, `{LATE_WEEK_FOLLOW_NEWS_VETO_POLICY}` when they moved that far "
        "but the injury report points the other way, so Tuesday's pick stands, "
        "`handle_follow_0_70` when that rule did not fire and at least 70% of the money "
        "bet on the game sat on the other side, `rookie_crew_underdog_v1` when it did not "
        "fire and the officiating crew for that game is new this season, or `model_only` "
        "when nothing above fired (or no market evidence was available). The 1.0-point "
        "`movement_ge_1.0` consensus rule was retired from the served chain on 2026-09-10 "
        "and is recorded as the paired challenger `consensus_movement_1_0_off_incumbent` "
        "-- see docs/late_week_refresh.md's movement-policy sections. Where this table "
        "and the picks table above disagree, the side here is the one being played.\n\n"
    )
    return heading + star + intro + table + "\n"


def append_refresh_to_card(
    destination: Path,
    result: RefreshResult,
    *,
    note: str = "",
    renomination: SundayRenomination | None = None,
) -> None:

    if not destination.is_file():
        raise ValueError(
            f"Cannot append a late-week refresh section: no published card at {destination}; "
            "run `nfl-ats publish-predictions` first."
        )
    text = destination.read_text(encoding="utf-8")
    section = _refresh_section_markdown(result, note, renomination)
    block = f"{LATE_WEEK_REFRESH_START}\n{section.rstrip()}\n{LATE_WEEK_REFRESH_END}"
    if LATE_WEEK_REFRESH_START in text or LATE_WEEK_REFRESH_END in text:
        if text.count(LATE_WEEK_REFRESH_START) != 1 or text.count(LATE_WEEK_REFRESH_END) != 1:
            raise ValueError("Late-week refresh markers must appear exactly once as a pair")
        before, remainder = text.split(LATE_WEEK_REFRESH_START, maxsplit=1)
        _, after = remainder.split(LATE_WEEK_REFRESH_END, maxsplit=1)
        new_text = before.rstrip() + "\n\n" + block + "\n" + after.lstrip("\n")
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"
    atomic_text(new_text, destination)


def _served_home_side_center_offset(
    artifacts_root: Path, active: Mapping[str, Any], frame: pd.DataFrame
) -> np.ndarray | None:

    forecast = active_artifact_path(artifacts_root, dict(active), "weekly_forecast")
    if forecast is None:
        return None
    sidecar = load_forecast_home_side_offsets(forecast)
    if sidecar is None or not sidecar.get("served"):
        return None
    games = sidecar.get("games")
    if not isinstance(games, list):
        return None
    by_game = {
        str(row.get("game_id")): float(row.get("home_side_offset", 0.0) or 0.0)
        for row in games
        if isinstance(row, dict)
    }
    ids = frame["game_id"].astype(str)
    return np.asarray([by_game.get(game_id, 0.0) for game_id in ids], dtype=float)


def _served_lattice_reads(
    artifacts_root: Path,
    active: Mapping[str, Any],
    features: pd.DataFrame,
    frame: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    residuals: np.ndarray,
    probability_method: str,
    season: int,
    week: int,
) -> pd.DataFrame:

    forecast = active_artifact_path(artifacts_root, dict(active), "weekly_forecast")
    if forecast is None:
        return forecasts
    push_sidecar = load_forecast_discrete_push_read(forecast)
    key_sidecar = load_forecast_key_line_pick_read(forecast)
    push_served = push_sidecar is not None and bool(push_sidecar.get("served"))
    key_served = key_sidecar is not None and bool(key_sidecar.get("served"))
    if not push_served and not key_served:
        return forecasts
    recorded_atoms = key_sidecar.get("atoms") if key_sidecar is not None else None
    atoms = (
        tuple(float(atom) for atom in recorded_atoms)
        if isinstance(recorded_atoms, list) and recorded_atoms
        else KEY_LINE_ATOMS
    )
    try:
        production = mass_preserving_lattice.fit_production_discrete_push_reader(
            features, artifacts_root, dict(active), season=season, week=week
        )
        if production.reader is None:
            raise ValueError("no discrete lattice for the target week")
        result = serve_discrete_three_way(
            forecasts,
            frame,
            production.reader,
            residuals=residuals,
            probability_method=probability_method,
        )
        if key_served:
            result = apply_key_line_pick_read(
                result,
                frame,
                KeyLinePickRead(reader=production.reader, atoms=atoms),
                residuals=residuals,
                probability_method=probability_method,
            )
        return result
    except Exception:
        return _restore_served_lattice_reads(
            forecasts,
            frame,
            push_sidecar=push_sidecar if push_served else None,
            key_sidecar=key_sidecar if key_served else None,
            pick_overrides=served_pick_overrides(forecast) if key_served else None,
        )


def _sidecar_split(row: Mapping[str, Any], source: Mapping[str, Any]) -> tuple[float, ...] | None:

    split: list[float] = []
    for key in ("cover", "push", "loss"):
        value = source.get(key)
        if value is None or isinstance(value, bool):
            return None
        try:
            split.append(float(value))
        except (TypeError, ValueError):
            return None
    if not np.isfinite(split).all():
        return None
    return tuple(split)


def _restore_served_lattice_reads(
    forecasts: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    push_sidecar: Mapping[str, Any] | None,
    key_sidecar: Mapping[str, Any] | None,
    pick_overrides: Mapping[str, float] | None,
) -> pd.DataFrame:

    splits: dict[str, tuple[float, ...]] = {}
    if push_sidecar is not None:
        games = push_sidecar.get("games")
        for row in games if isinstance(games, list) else []:
            if not isinstance(row, Mapping):
                continue
            served = row.get("served")
            split = _sidecar_split(row, served) if isinstance(served, Mapping) else None
            if split is not None:
                splits[str(row.get("game_id"))] = split
    if key_sidecar is not None:
        games = key_sidecar.get("games")
        for row in games if isinstance(games, list) else []:
            if not isinstance(row, Mapping) or not row.get("touched"):
                continue
            split = _sidecar_split(row, row)
            if split is not None:
                splits[str(row.get("game_id"))] = split
    result = forecasts.copy()
    ids = frame["game_id"].astype(str).to_list()
    for position, column in enumerate(THREE_WAY_COLUMNS):
        if column not in result.columns:
            continue
        result[column] = apply_pick_overrides(
            result[column], ids, {game_id: split[position] for game_id, split in splits.items()}
        )
    result["home_cover_probability"] = apply_pick_overrides(
        result["home_cover_probability"], ids, pick_overrides
    )
    return result
