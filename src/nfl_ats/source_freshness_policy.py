from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nfl_ats.capture_freshness import (
    newest_json_field_instant,
    newest_snapshot_instant,
)
from nfl_ats.player_arrests_back_side_overlay import MAX_SNAPSHOT_AGE
from nfl_ats.public_board import humanize_identifier

COMPLETE = "complete"
DEGRADED = "degraded"
BLOCKED = "blocked"
STATES = (COMPLETE, DEGRADED, BLOCKED)
NOT_DUE = "not_due"
NOT_CONFIGURED = "not_configured"
SOURCE_STATES = (*STATES, NOT_DUE, NOT_CONFIGURED)

BREACH_BEHAVIOURS = (DEGRADED, BLOCKED)

_MINUTES_PER_WEEK = 7 * 24 * 60
_DAY_OFFSETS = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

POLICY_DOC = "docs/source_freshness_policy.md"


class SourceFreshnessError(RuntimeError):
    pass


def _cycle_minutes(day: str, at: str) -> int:

    hour, minute = (int(part) for part in at.split(":", maxsplit=1))
    return _DAY_OFFSETS[day] * 24 * 60 + hour * 60 + minute


def _derive_budget(jobs: tuple[tuple[str, str, int], ...]) -> tuple[int, int, int, str]:

    if not jobs:
        raise ValueError("a freshness budget needs at least one scheduled job")
    ordered = sorted(((_cycle_minutes(day, at), grace) for day, at, grace in jobs))
    worst_gap = 0
    worst_grace = ordered[0][1]
    for index, (start, _grace) in enumerate(ordered):
        closer_index = (index + 1) % len(ordered)
        end, closing_grace = ordered[closer_index]
        gap = (end - start) % _MINUTES_PER_WEEK
        if len(ordered) == 1:
            gap = _MINUTES_PER_WEEK
        if gap > worst_gap:
            worst_gap = gap
            worst_grace = closing_grace
    derivation = (
        f"longest scheduled gap {worst_gap} min over {len(jobs)} SCHEDULE job(s) "
        f"+ {worst_grace} min grace of the job that closes it"
    )
    return worst_gap, worst_grace, worst_gap + worst_grace, derivation


@dataclass(frozen=True)
class SourceLocation:
    kind: str
    root: str
    relative_path: str
    json_key: str = ""


@dataclass(frozen=True)
class SourceFreshnessPolicy:
    source_id: str
    label: str
    schedule_jobs: tuple[tuple[str, str, int], ...]
    schedule_job_names: tuple[str, ...]
    location: SourceLocation
    on_absent: str
    on_stale: str
    on_future_dated: str
    fallback: str
    enforced_by: str
    budget_override_minutes: int | None = None
    override_reason: str = ""

    @property
    def _derived(self) -> tuple[int, int, int, str]:
        return _derive_budget(self.schedule_jobs)

    @property
    def recurrence_minutes(self) -> int:
        return self._derived[0]

    @property
    def grace_minutes(self) -> int:
        return self._derived[1]

    @property
    def budget_minutes(self) -> int:
        if self.budget_override_minutes is not None:
            return self.budget_override_minutes
        return self._derived[2]

    @property
    def budget_derivation(self) -> str:
        derived = self._derived
        if self.budget_override_minutes is None:
            return derived[3]
        return (
            f"{derived[3]} = {derived[2]} min, tightened to "
            f"{self.budget_override_minutes} min: {self.override_reason}"
        )

    @property
    def fail_closed(self) -> bool:

        return BLOCKED in (self.on_absent, self.on_stale, self.on_future_dated)


def _policy(
    source_id: str,
    label: str,
    schedule_jobs: tuple[tuple[str, str, int], ...],
    schedule_job_names: tuple[str, ...],
    location: SourceLocation,
    *,
    on_absent: str,
    on_stale: str,
    on_future_dated: str,
    fallback: str,
    enforced_by: str,
    budget_override_minutes: int | None = None,
    override_reason: str = "",
) -> SourceFreshnessPolicy:
    for behaviour in (on_absent, on_stale, on_future_dated):
        if behaviour not in BREACH_BEHAVIOURS:
            raise ValueError(f"{source_id}: breach behaviour must be one of {BREACH_BEHAVIOURS}")
    return SourceFreshnessPolicy(
        source_id=source_id,
        label=label,
        schedule_jobs=schedule_jobs,
        schedule_job_names=schedule_job_names,
        location=location,
        on_absent=on_absent,
        on_stale=on_stale,
        on_future_dated=on_future_dated,
        fallback=fallback,
        enforced_by=enforced_by,
        budget_override_minutes=budget_override_minutes,
        override_reason=override_reason,
    )


SOURCE_FRESHNESS_POLICIES: dict[str, SourceFreshnessPolicy] = {
    policy.source_id: policy
    for policy in (
        _policy(
            "odds_opener",
            "The Odds API Tuesday opener (the grade the pool settles on)",
            (("tue", "09:00", 180),),
            ("odds_tue_open",),
            SourceLocation("snapshot_dir", "data", "market/raw"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=BLOCKED,
            fallback=(
                "publish on the newest opener snapshot on disk and disclose that line "
                "freshness is unverified (prediction_safety warning path)"
            ),
            enforced_by=(
                "nfl_ats.prediction_safety._prospective_checks (market_timing); "
                "scripts/capture_scheduler.py weekly_lock requires=('odds_tue_open',)"
            ),
        ),
        _policy(
            "odds_refresh",
            "The Odds API mid/late-week and closing captures",
            (
                ("tue", "09:00", 180),
                ("thu", "18:00", 90),
                ("sat", "12:00", 180),
                ("sun", "12:30", 25),
                ("sun", "16:15", 60),
                ("mon", "19:00", 90),
            ),
            (
                "odds_tue_open",
                "odds_thu_tnf",
                "odds_sat",
                "odds_sun_close",
                "odds_sun_late",
                "odds_mon_mnf",
            ),
            SourceLocation("snapshot_dir", "data", "market/raw"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=BLOCKED,
            fallback=(
                "the frozen Tuesday opener quote stands; no late-week line refresh is "
                "applied and CLV for the week is not computed"
            ),
            enforced_by="nfl_ats.clv / nfl_ats.pick_refresh (refresh passes are optional)",
        ),
        _policy(
            "injuries_nflverse",
            "nflverse weekly injury/schedule snapshot (weekly-run step 1 ingest)",
            (("tue", "09:15", 120),),
            ("weekly_lock",),
            SourceLocation("snapshot_dir", "data", "raw/nflverse_injuries"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback=(
                "the previous weekly snapshot is reused; availability features carry "
                "last week's report rather than an invented neutral value"
            ),
            enforced_by="nfl_ats.weekly (step 1 ingest) / nfl_ats.players",
        ),
        _policy(
            "injuries_nflverse_timestamps",
            "Real (non-proxy) date_modified coverage in the CONSUMED player "
            "snapshot (ENG-39, docs/injury_timestamp_fallback.md)",
            (("tue", "09:15", 120),),
            ("weekly_lock",),
            SourceLocation("snapshot_dir", "data", "players/raw"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback=(
                "the affected season's home_/away_/diff_injury_* feature block is "
                "exactly null/zero for every row -- nflverse's 2025 release drops "
                "date_modified entirely and the default canonicalization response "
                "is to drop every undated row (M1/M3, docs/injury_timestamp_fallback.md); "
                "the opt-in week_proxy fallback (nfl_ats.players.canonicalize_injuries) "
                "restores a leakage-safe timestamp, and prediction_safety's "
                "injury_feature_presence check (ENG-39) is the release gate that fails a "
                "prospective card on this exact failure mode instead of publishing it quietly"
            ),
            enforced_by=(
                "nfl_ats.players.canonicalize_injuries / "
                "nfl_ats.prediction_safety._injury_feature_checks"
            ),
        ),
        _policy(
            "injuries_sportradar",
            "Sportradar weekly injuries (credential-gated, PER-03)",
            (
                ("wed", "17:30", 240),
                ("thu", "17:30", 240),
                ("fri", "17:30", 240),
                ("sat", "10:00", 240),
            ),
            (
                "sportradar_injuries_wed",
                "sportradar_injuries_thu",
                "sportradar_injuries_fri",
                "sportradar_injuries_sat",
            ),
            SourceLocation("snapshot_dir", "data", "raw/sportradar_injuries"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback=(
                "dormant without SPORTRADAR_API_KEY; the nflverse report stands and no "
                "late-week injury revision is applied"
            ),
            enforced_by="scripts/capture_sportradar_injuries.py (credential-gated jobs)",
        ),
        _policy(
            "inactives",
            "Official game-day inactives (WP17, T-90 captures)",
            (
                ("thu", "11:35", 15),
                ("thu", "15:05", 15),
                ("thu", "18:50", 20),
                ("sat", "15:30", 15),
                ("sat", "18:50", 20),
                ("sun", "11:35", 15),
                ("sun", "14:40", 15),
            ),
            (
                "inactives_thu_afternoon_early",
                "inactives_thu_afternoon_late",
                "inactives_thu_primetime",
                "inactives_sat_early",
                "inactives_sat_late",
                "inactives_sun_early",
                "inactives_sun_late",
            ),
            SourceLocation("snapshot_dir", "data", "players/inactives"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback=(
                "SOURCE_NO_SNAPSHOT: the Tuesday pick stands, the row is tagged, and a "
                "zero-row snapshot counts as 'no report yet', never as 'nobody is out'"
            ),
            enforced_by="nfl_ats.inactives_refresh_overlay",
        ),
        _policy(
            "projected_lineups",
            "Projected depth-chart lineups artifact",
            tuple((day, "12:00", 180) for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")),
            tuple(f"lineups_{day}" for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")),
            SourceLocation(
                "json_timestamp", "artifacts", "lineups/current/lineups.json", "generated_at"
            ),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback=(
                "the This Week lineup panel is omitted; a missing injury feed or play "
                "probability is displayed as unavailable rather than estimated"
            ),
            enforced_by="nfl_ats.lineup_view (docs/projected_lineups.md)",
        ),
        _policy(
            "referee_assignments",
            "Weekly officiating-crew assignments (WP22)",
            (("wed", "15:00", 240),),
            ("referee_assignments_wed",),
            SourceLocation("snapshot_dir", "data", "players/referee_assignments"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback=(
                "DOCUMENTED NO-OP: zero crew tilt, the incumbent Tuesday pick stands, "
                "the row is tagged"
            ),
            enforced_by="nfl_ats.crew_tilt_refresh_overlay",
        ),
        _policy(
            "player_arrests",
            "USA Today player-arrests snapshot (promoted production policy member)",
            (("tue", "07:00", 90),),
            ("player_arrests_tue",),
            SourceLocation("snapshot_dir", "data", "raw/player_arrests"),
            on_absent=BLOCKED,
            on_stale=BLOCKED,
            on_future_dated=BLOCKED,
            fallback="none -- fail-closed; there is no public fail-open switch",
            enforced_by=(
                "nfl_ats.player_arrests_back_side_overlay."
                "load_latest_complete_arrest_snapshot; "
                "nfl_ats.card_view.resolve_player_arrests_overlay(require_fresh=True); "
                "nfl_ats.weekly step 7 ingest-player-arrests"
            ),
            budget_override_minutes=int(MAX_SNAPSHOT_AGE.total_seconds() // 60),
            override_reason=(
                "nfl_ats.player_arrests_back_side_overlay.MAX_SNAPSHOT_AGE is already "
                "enforced fail-closed at publish; the policy layer must never be looser "
                "than a gate production already applies"
            ),
        ),
        _policy(
            "pfr_transactions",
            "Pro Football Rumors transaction wire",
            (("wed", "07:00", 120), ("sat", "07:00", 120)),
            ("pfr_transactions_wed", "pfr_transactions_sat"),
            SourceLocation("snapshot_dir", "data", "raw/pfr_transactions"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback="transaction-wire features fall back to their neutral (no-news) value",
            enforced_by="nfl_ats.transaction_wire_features",
        ),
        _policy(
            "airnow_weather",
            "AirNow hourly AQI checkpoint (environmental exposure join)",
            (("tue", "11:40", 15),),
            ("airnow_tue_checkpoint",),
            SourceLocation("snapshot_dir", "data", "raw/airnow_hourly"),
            on_absent=DEGRADED,
            on_stale=DEGRADED,
            on_future_dated=DEGRADED,
            fallback="environmental features fall back to their neutral value",
            enforced_by="nfl_ats.forecast_weather_features",
        ),
    )
}


@dataclass(frozen=True)
class SourceObservation:
    source_id: str
    observed_at: datetime | None
    detail: str = ""


@dataclass(frozen=True)
class SourceState:
    source_id: str
    state: str
    reason: str
    age_minutes: float | None
    budget_minutes: int
    fallback: str
    detail: str = ""
    due_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "age_minutes": self.age_minutes,
            "budget_minutes": self.budget_minutes,
            "fallback": self.fallback,
            "detail": self.detail,
            "due_at_utc": self.due_at_utc,
        }


@dataclass(frozen=True)
class SourcePolicyReport:
    state: str
    evaluated_at_utc: str
    sources: tuple[SourceState, ...]
    unobserved: tuple[str, ...]

    def _ids(self, state: str) -> tuple[str, ...]:
        return tuple(row.source_id for row in self.sources if row.state == state)

    @property
    def complete(self) -> tuple[str, ...]:
        return self._ids(COMPLETE)

    @property
    def degraded(self) -> tuple[str, ...]:
        return self._ids(DEGRADED)

    @property
    def blocked(self) -> tuple[str, ...]:
        return self._ids(BLOCKED)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:

        reasons = []
        for row in self.sources:
            if row.state != BLOCKED:
                continue
            policy = SOURCE_FRESHNESS_POLICIES[row.source_id]
            reasons.append(
                f"{row.source_id}: {row.reason} "
                f"(rule: budget {policy.budget_minutes} min, fail-closed, "
                f"enforced by {policy.enforced_by})"
            )
        return tuple(reasons)

    def block_message(self) -> str:
        return "publication refused by source policy -- " + "; ".join(self.blocking_reasons)

    def summary_line(self) -> str:

        def _names(ids: tuple[str, ...]) -> str:
            return ", ".join(humanize_identifier(i) for i in ids) if ids else "none"

        return (
            f"**Source freshness: {self.state.upper()}.** "
            f"Complete: {_names(self.complete)}. "
            f"Degraded (allowed fallback): {_names(self.degraded)}. "
            f"Blocked: {_names(self.blocked)}. "
            f"Not due yet: {_names(self._ids(NOT_DUE))}. "
            f"Not set up: {_names(self._ids(NOT_CONFIGURED))}. "
            f"Budgets, fallbacks and source states: `{POLICY_DOC}`."
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "evaluated_at_utc": self.evaluated_at_utc,
            "complete": list(self.complete),
            "degraded": list(self.degraded),
            "blocked": list(self.blocked),
            "not_due": list(self._ids(NOT_DUE)),
            "not_configured": list(self._ids(NOT_CONFIGURED)),
            "unobserved": list(self.unobserved),
            "blocking_reasons": list(self.blocking_reasons),
            "sources": {row.source_id: row.to_dict() for row in self.sources},
        }


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _evaluate_one(
    policy: SourceFreshnessPolicy,
    observation: SourceObservation,
    now: datetime,
    first_kickoff: datetime | None,
) -> SourceState:
    budget = policy.budget_minutes
    future = observation.observed_at is not None and _as_utc(observation.observed_at) > now
    if not policy.fail_closed and not future:
        if policy.source_id == "injuries_sportradar" and not os.environ.get("SPORTRADAR_API_KEY"):
            return SourceState(
                policy.source_id,
                NOT_CONFIGURED,
                "credential-gated capture is not configured",
                None,
                budget,
                policy.fallback,
                observation.detail,
            )
        if policy.source_id == "inactives" and first_kickoff is not None:
            due = _as_utc(first_kickoff) - timedelta(minutes=90)
            if observation.observed_at is None and now < due:
                return SourceState(
                    policy.source_id,
                    NOT_DUE,
                    "the week's first inactive report is not due yet",
                    None,
                    budget,
                    policy.fallback,
                    observation.detail,
                    due.isoformat(),
                )
    if observation.observed_at is None:
        return SourceState(
            source_id=policy.source_id,
            state=policy.on_absent,
            reason=f"no snapshot present (budget {budget} min)",
            age_minutes=None,
            budget_minutes=budget,
            fallback=policy.fallback,
            detail=observation.detail,
        )
    age = (now - _as_utc(observation.observed_at)).total_seconds() / 60.0
    if age < 0.0:
        return SourceState(
            source_id=policy.source_id,
            state=policy.on_future_dated,
            reason=f"snapshot is future-dated by {-age:.1f} min",
            age_minutes=age,
            budget_minutes=budget,
            fallback=policy.fallback,
            detail=observation.detail,
        )
    if age > budget:
        return SourceState(
            source_id=policy.source_id,
            state=policy.on_stale,
            reason=f"snapshot is {age:.1f} min old, over the {budget} min budget",
            age_minutes=age,
            budget_minutes=budget,
            fallback=policy.fallback,
            detail=observation.detail,
        )
    return SourceState(
        source_id=policy.source_id,
        state=COMPLETE,
        reason=f"snapshot is {age:.1f} min old, inside the {budget} min budget",
        age_minutes=age,
        budget_minutes=budget,
        fallback=policy.fallback,
        detail=observation.detail,
    )


def evaluate_sources(
    observations: Iterable[SourceObservation] | Mapping[str, datetime | None],
    now: datetime,
    *,
    first_kickoff: datetime | None = None,
) -> SourcePolicyReport:

    now_utc = _as_utc(now)
    if isinstance(observations, Mapping):
        resolved = [
            SourceObservation(source_id, instant) for source_id, instant in observations.items()
        ]
    else:
        resolved = list(observations)

    seen: dict[str, SourceObservation] = {}
    for observation in resolved:
        if observation.source_id not in SOURCE_FRESHNESS_POLICIES:
            raise KeyError(f"Unknown source policy id: {observation.source_id!r}")
        seen[observation.source_id] = observation

    rows = tuple(
        _evaluate_one(policy, seen[source_id], now_utc, first_kickoff)
        for source_id, policy in SOURCE_FRESHNESS_POLICIES.items()
        if source_id in seen
    )
    unobserved = tuple(
        source_id for source_id in SOURCE_FRESHNESS_POLICIES if source_id not in seen
    )
    states = {row.state for row in rows}
    if BLOCKED in states:
        overall = BLOCKED
    elif DEGRADED in states or not rows:
        overall = DEGRADED
    else:
        overall = COMPLETE
    return SourcePolicyReport(
        state=overall,
        evaluated_at_utc=now_utc.isoformat(),
        sources=rows,
        unobserved=unobserved,
    )


def observe_from_disk(
    *,
    data_root: Path | None,
    artifacts_root: Path | None,
    source_ids: Iterable[str] | None = None,
    overrides: Mapping[str, SourceObservation] | None = None,
) -> tuple[SourceObservation, ...]:

    wanted = tuple(source_ids) if source_ids is not None else tuple(SOURCE_FRESHNESS_POLICIES)
    supplied = dict(overrides or {})
    observations: list[SourceObservation] = []
    for source_id in wanted:
        policy = SOURCE_FRESHNESS_POLICIES[source_id]
        if source_id in supplied:
            observations.append(supplied[source_id])
            continue
        root = data_root if policy.location.root == "data" else artifacts_root
        if root is None:
            continue
        target = root / policy.location.relative_path
        if policy.location.kind == "snapshot_dir":
            instant = newest_snapshot_instant(target)
            detail = f"newest snapshot dir under {policy.location.relative_path}"
        else:
            instant = newest_json_field_instant(target, policy.location.json_key)
            detail = (
                f"{policy.location.json_key} in {policy.location.relative_path}"
                if policy.location.json_key
                else policy.location.relative_path
            )
        observations.append(SourceObservation(source_id, instant, detail))
    return tuple(observations)


def player_snapshot_injury_timestamp_observation(
    player_snapshot_root: Path,
    *,
    season: int,
    source_id: str = "injuries_nflverse_timestamps",
) -> SourceObservation:

    manifest_path = player_snapshot_root / "manifest.json"
    injuries_path = player_snapshot_root / "injuries.parquet"
    if not manifest_path.is_file() or not injuries_path.is_file():
        return SourceObservation(
            source_id, None, f"no player snapshot found at {player_snapshot_root}"
        )
    import json

    import pandas as pd

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot_id = str(manifest.get("snapshot_id", player_snapshot_root.name))
        injuries = pd.read_parquet(injuries_path)
    except Exception as error:
        return SourceObservation(
            source_id, None, f"could not read player snapshot {player_snapshot_root}: {error}"
        )
    if "season" not in injuries.columns:
        return SourceObservation(
            source_id, None, f"player snapshot {snapshot_id} injuries has no season column"
        )
    season_rows = injuries.loc[pd.to_numeric(injuries["season"], errors="coerce").eq(season)]
    if "observed_at_basis" in season_rows.columns:
        real_rows = season_rows.loc[season_rows["observed_at_basis"].eq("date_modified")]
    else:
        real_rows = season_rows
    if "date_modified" in real_rows.columns:
        raw_dates = real_rows["date_modified"]
    else:
        raw_dates = pd.Series([], dtype="object")
    real_dates = pd.to_datetime(raw_dates, errors="coerce", utc=True).dropna()
    if real_dates.empty:
        return SourceObservation(
            source_id,
            None,
            f"player snapshot {snapshot_id} has zero real date_modified revisions for "
            f"season {season} ({len(season_rows)} rows total for that season) -- "
            "see docs/injury_timestamp_fallback.md M1/M3",
        )
    newest = pd.Timestamp(real_dates.max())
    return SourceObservation(
        source_id,
        newest.to_pydatetime(),
        f"newest real date_modified in player snapshot {snapshot_id} for season {season} "
        f"({len(real_dates)}/{len(season_rows)} season rows have one)",
    )


def report_for_publication(
    *,
    data_root: Path | None,
    artifacts_root: Path | None,
    now: datetime,
    arrest_snapshot_at: Any = None,
    arrest_snapshot_id: str | None = None,
    player_snapshot_root: Path | None = None,
    player_snapshot_season: int | None = None,
) -> SourcePolicyReport:

    overrides: dict[str, SourceObservation] = {}
    source_ids = list(SOURCE_FRESHNESS_POLICIES)
    instant: datetime | None = None
    if arrest_snapshot_at is not None:
        converter = getattr(arrest_snapshot_at, "to_pydatetime", None)
        candidate = converter() if callable(converter) else arrest_snapshot_at
        if isinstance(candidate, datetime):
            instant = _as_utc(candidate)
    if instant is None:
        source_ids = [source_id for source_id in source_ids if source_id != "player_arrests"]
    else:
        overrides["player_arrests"] = SourceObservation(
            "player_arrests",
            instant,
            f"hash-verified snapshot {arrest_snapshot_id or 'unknown'}",
        )
    if player_snapshot_root is not None and player_snapshot_season is not None:
        overrides["injuries_nflverse_timestamps"] = player_snapshot_injury_timestamp_observation(
            player_snapshot_root, season=player_snapshot_season
        )
    return evaluate_sources(
        observe_from_disk(
            data_root=data_root,
            artifacts_root=artifacts_root,
            source_ids=source_ids,
            overrides=overrides,
        ),
        now,
        first_kickoff=_first_week_kickoff(data_root, now),
    )


def _first_week_kickoff(data_root: Path | None, now: datetime) -> datetime | None:
    if data_root is None:
        return None
    import pandas as pd

    from nfl_ats.market_data_halves import current_week_kickoff_window

    now_utc = _as_utc(now)
    candidates = []
    for path in sorted((data_root / "raw").glob("*/schedules.parquet")):
        try:
            captured = datetime.strptime(path.parent.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if captured <= now_utc:
            candidates.append(path)
    if not candidates:
        return None
    try:
        schedule = pd.read_parquet(candidates[-1])
        schedule = schedule.loc[
            schedule["game_type"].isin({"REG", "WC", "DIV", "CON", "SB"})
        ].copy()
        local = pd.to_datetime(
            schedule["gameday"].astype(str).str[:10] + " " + schedule["gametime"].astype(str),
            errors="coerce",
        )
        schedule["kickoff"] = local.dt.tz_localize(
            ZoneInfo("America/New_York"), ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")
        start, end = current_week_kickoff_window(now_utc)
        current = schedule.loc[schedule["kickoff"].ge(start) & schedule["kickoff"].lt(end)]
        if current.empty:
            current = schedule.loc[schedule["kickoff"].ge(end)].sort_values("kickoff")
        if current.empty:
            return None
        first = current.sort_values("kickoff").iloc[0]
        slate = schedule.loc[
            schedule["season"].eq(first["season"])
            & schedule["week"].eq(first["week"])
            & schedule["game_type"].eq(first["game_type"])
        ]
        return pd.Timestamp(slate["kickoff"].min()).to_pydatetime()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def policy_table() -> tuple[dict[str, Any], ...]:

    return tuple(
        {
            "source_id": policy.source_id,
            "label": policy.label,
            "schedule_jobs": list(policy.schedule_job_names),
            "recurrence_minutes": policy.recurrence_minutes,
            "grace_minutes": policy.grace_minutes,
            "budget_minutes": policy.budget_minutes,
            "budget_derivation": policy.budget_derivation,
            "on_absent": policy.on_absent,
            "on_stale": policy.on_stale,
            "on_future_dated": policy.on_future_dated,
            "fail_closed": policy.fail_closed,
            "fallback": policy.fallback,
            "enforced_by": policy.enforced_by,
        }
        for policy in SOURCE_FRESHNESS_POLICIES.values()
    )


__all__ = [
    "BLOCKED",
    "COMPLETE",
    "DEGRADED",
    "NOT_CONFIGURED",
    "NOT_DUE",
    "POLICY_DOC",
    "SOURCE_FRESHNESS_POLICIES",
    "SOURCE_STATES",
    "STATES",
    "SourceFreshnessError",
    "SourceFreshnessPolicy",
    "SourceLocation",
    "SourceObservation",
    "SourcePolicyReport",
    "SourceState",
    "evaluate_sources",
    "observe_from_disk",
    "player_snapshot_injury_timestamp_observation",
    "policy_table",
    "report_for_publication",
]
