"""Per-source freshness budgets and degraded-mode policy for the played card (ENG-14).

Why this module is named ``source_freshness_policy`` and not ``source_policy``
-----------------------------------------------------------------------------
``nfl_ats.source_policy`` already exists and answers a different question: may
we ACQUIRE, retain, and republish this source at all (terms, risk colour,
quota -- MKT-09, backed by ``config/source_policies.json``). This module answers
the operational question that sits on top of it: given that a source is allowed,
is the snapshot we actually hold FRESH ENOUGH to publish a card from, and if it
is not, does the card degrade onto a documented fallback or refuse to publish?
Two different registries, two different failure modes, deliberately not merged.

What this layer does NOT do
---------------------------
It invents no new fail-closed behaviour. Every source below already has a
documented behaviour on missing/stale data, scattered across the module that
consumes it; this table names that behaviour in one place, gives it a budget
derived from the capture cadence, and makes the resulting state visible on the
card. A source whose current consumer degrades keeps degrading here
(``degraded`` is the strongest state it can reach), and only a source whose
consumer ALREADY refuses -- player arrests, and any anti-backdating gate --
can reach ``blocked``. Turning a permitted publish path into a blocked one is
a policy decision for the owner, not a side effect of writing this file.

Where the budgets come from
---------------------------
Every budget is DERIVED, not chosen. Each source declares the
``scripts/capture_scheduler.py`` ``SCHEDULE`` jobs that feed it, as
``(day, "HH:MM", grace_minutes)`` triples copied from that file, and
:func:`_derive_budget` computes

    budget = (longest gap between consecutive scheduled captures, over one
              weekly cycle including the Sunday->Monday wrap)
             + (the grace of the job that CLOSES that longest gap)

which is exactly the oldest a HEALTHY source can be at an arbitrary evaluation
instant. Anything older means a scheduled capture did not land. A single weekly
job therefore gets a 7-day budget plus its own grace -- correct, and loose on
purpose: a tighter number would false-alarm at the end of every cycle, and a
false ``degraded`` on the card is exactly as corrosive as a missed real one.

One source overrides the derived number DOWNWARD, and only because a stricter
gate is already enforced in production code:
:data:`nfl_ats.player_arrests_back_side_overlay.MAX_SNAPSHOT_AGE` (36 hours).
That constant is imported, never re-declared, so the two can never drift.

Prior art this extends rather than duplicates
---------------------------------------------
* ``nfl_ats.player_arrests_back_side_overlay.load_latest_complete_arrest_snapshot``
  -- manifest completeness, hash verification, future-dated rejection, and the
  36-hour staleness ceiling. Fail-closed at publish
  (``nfl_ats.card_view.resolve_player_arrests_overlay(require_fresh=True)``).
* ``nfl_ats.prediction_safety._prospective_checks`` -- ``market_timing`` FAILS
  on a market observation after freeze time or kickoff, and only WARNS when the
  timestamp is absent. That asymmetry is reproduced exactly below: odds are
  ``blocked`` when future-dated, ``degraded`` when merely stale or absent.
* ``nfl_ats.nflcom_refresh_overlay`` / ``nfl_ats.inactives_refresh_overlay`` /
  ``nfl_ats.crew_tilt_refresh_overlay`` -- an absent, stale, or post-deadline
  snapshot is a DOCUMENTED NO-OP: the Tuesday pick stands, the row is tagged.
* ``nfl_ats.capture_freshness`` (ENG-03) -- ``newest_snapshot_instant`` and
  ``newest_json_field_instant``, both read from the UTC-stamped directory NAME
  or payload field, never filesystem mtime, matching
  ``scripts/capture_scheduler.py.newest_snapshot_age_minutes``. This module
  IMPORTS them rather than carrying a second copy; see the join point below.

Join point with ENG-03
----------------------
``nfl_ats.capture_freshness`` answers "is each capture SOURCE producing data on
schedule", grouping jobs by their ``SCHEDULE`` ``dedupe_dir``. This module
answers "may this CARD publish", and its rows are deliberately finer-grained
than a directory: ``odds_opener`` and ``odds_refresh`` share
``data/market/raw`` but carry different budgets, and ``injuries_nflverse`` is
fed by ``weekly-run`` step 1 rather than by a capture job at all. So the
POLICY table stays here and only the two on-disk locators are shared -- ENG-03
made them public for exactly this. A future consolidation would move
:attr:`SourceFreshnessPolicy.location` onto that module's locator registry;
the policy table, the state machine and every caller stay unchanged either way.
"""

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

#: The three card states. ``complete`` = every observed source inside budget.
#: ``degraded`` = at least one source fell back to its documented fallback.
#: ``blocked`` = at least one fail-closed source breached and the card must
#: not be published.
COMPLETE = "complete"
DEGRADED = "degraded"
BLOCKED = "blocked"
STATES = (COMPLETE, DEGRADED, BLOCKED)
NOT_DUE = "not_due"
NOT_CONFIGURED = "not_configured"
SOURCE_STATES = (*STATES, NOT_DUE, NOT_CONFIGURED)

#: Breach behaviours a source may declare. Only ``BLOCKED`` refuses a publish.
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

#: Where the module doc lives, quoted on the card so a degraded line is
#: readable without reading this file.
POLICY_DOC = "docs/source_freshness_policy.md"


class SourceFreshnessError(RuntimeError):
    """A fail-closed source breached its budget; publication is refused."""


def _cycle_minutes(day: str, at: str) -> int:
    """Minutes from Monday 00:00 local, for one ``SCHEDULE`` job clock time."""

    hour, minute = (int(part) for part in at.split(":", maxsplit=1))
    return _DAY_OFFSETS[day] * 24 * 60 + hour * 60 + minute


def _derive_budget(jobs: tuple[tuple[str, str, int], ...]) -> tuple[int, int, int, str]:
    """``(recurrence_minutes, grace_minutes, budget_minutes, derivation)``.

    ``jobs`` are ``(day, "HH:MM", grace_minutes)`` triples read from
    ``scripts/capture_scheduler.py``'s ``SCHEDULE``. The longest gap over one
    weekly cycle (wrapping Sunday -> Monday) is the recurrence; the grace of
    the job that closes that gap is added, because a capture is not late until
    its own grace window has also closed.
    """

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
    """Where the newest capture instant for a source is read from.

    ``kind`` is ``"snapshot_dir"`` (newest ``YYYYMMDDTHHMMSSZ`` subdirectory
    name) or ``"json_timestamp"`` (a UTC stamp under ``json_key`` in a JSON
    file). ``root`` is ``"data"`` or ``"artifacts"``.
    """

    kind: str
    root: str
    relative_path: str
    json_key: str = ""


@dataclass(frozen=True)
class SourceFreshnessPolicy:
    """One row of the declarative table. Every field is auditable against
    ``scripts/capture_scheduler.py`` or the named consumer module."""

    source_id: str
    label: str
    #: ``(day, "HH:MM", grace_minutes)`` copied from ``SCHEDULE``.
    schedule_jobs: tuple[tuple[str, str, int], ...]
    #: Named ``SCHEDULE`` entries, so the cadence can be re-derived by hand.
    schedule_job_names: tuple[str, ...]
    location: SourceLocation
    #: State when no snapshot exists at all.
    on_absent: str
    #: State when the newest snapshot is older than the budget.
    on_stale: str
    #: State when the newest snapshot is dated after the evaluation instant.
    on_future_dated: str
    #: What the card does instead. Plain English; shown in the metadata block.
    fallback: str
    #: The module that already implements this behaviour.
    enforced_by: str
    #: Set only to tighten a derived budget to an already-enforced constant.
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
        """True when any breach of this source refuses a publish."""

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


#: The declarative table. Ordered as the card reads: market first, then the
#: availability feeds, then context.
SOURCE_FRESHNESS_POLICIES: dict[str, SourceFreshnessPolicy] = {
    policy.source_id: policy
    for policy in (
        _policy(
            "odds_opener",
            "The Odds API Tuesday opener (the grade the pool settles on)",
            (("tue", "09:00", 180),),
            ("odds_tue_open",),
            SourceLocation("snapshot_dir", "data", "market/raw"),
            # prediction_safety only WARNS on an absent market timestamp and
            # the manual publish path has never required one, so absent/stale
            # stay degraded. A quote dated after the decision instant already
            # FAILS prediction_safety's market_timing check -- naming it
            # blocked here changes nothing about what is permitted.
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
            # Documents where the feature build actually reads injuries from
            # -- data/players/raw -- not the raw capture directory the
            # "injuries_nflverse" row above watches (that row and this one
            # answer different questions: "did a capture land" vs "did the
            # snapshot production reads have a real revision timestamp for
            # the season being served"). See
            # player_snapshot_injury_timestamp_observation, the override this
            # row is meant to be evaluated with; the generic snapshot-dir
            # scan below is only the fallback when no override is supplied.
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
            # The ONLY source whose breach refuses a publish, and it already
            # did before this module existed: card_view passes
            # require_fresh=True and weekly-run step 7 is fatal.
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
    """What we actually hold for one source at evaluation time.

    ``observed_at=None`` means "we looked and there is nothing there" (absent).
    A source with NO observation at all is *unobserved*: it never appears in
    :attr:`SourcePolicyReport.sources` and never contributes to the roll-up,
    because "we did not look" is not evidence about the source.
    """

    source_id: str
    observed_at: datetime | None
    detail: str = ""


@dataclass(frozen=True)
class SourceState:
    """One source's adjudicated state."""

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
    """Per-source states plus the overall card state."""

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
        """One sentence per blocking source: which source, and which rule."""

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
        """The single line the published Markdown card carries."""

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
    # Never neutralize a future-dated observation or a fail-closed source.
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
    """Adjudicate every observed source against its budget and roll the card up.

    ``observations`` is either :class:`SourceObservation` values or a plain
    ``{source_id: instant_or_None}`` mapping. An unknown ``source_id`` raises:
    a typo must never silently become "no evidence about a source".

    The roll-up is worst-wins: ``blocked`` if any source is blocked,
    ``degraded`` if any is degraded, ``complete`` otherwise. An EMPTY report
    is ``degraded``, not ``complete`` -- nothing was looked at, so nothing can
    be claimed.
    """

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


# ---------------------------------------------------------------------------
# On-disk observation (locators shared with ENG-03 -- see the module docstring)
# ---------------------------------------------------------------------------


def observe_from_disk(
    *,
    data_root: Path | None,
    artifacts_root: Path | None,
    source_ids: Iterable[str] | None = None,
    overrides: Mapping[str, SourceObservation] | None = None,
) -> tuple[SourceObservation, ...]:
    """Observe each source's newest capture instant from the local tree.

    A source whose root is unavailable (``data_root=None`` for a data source)
    is left UNOBSERVED rather than reported absent -- "we could not look" is
    not "there is nothing there", and conflating them is how a fail-closed
    source would start blocking rendering paths that never required it.

    ``overrides`` lets a caller supply an observation it already computed
    through the source's own verified loader; that always wins over the naive
    directory scan. The publish path does exactly this for ``player_arrests``,
    whose real freshness comes from ``load_latest_complete_arrest_snapshot``'s
    manifest (complete + hash-verified), not from a directory name.
    """

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
    """ENG-39: does the player snapshot actually CONSUMED by feature-building
    have a real (non-proxy) injury revision timestamp for ``season``?

    ``injuries_nflverse`` watches the raw capture directory
    (``data/raw/nflverse_injuries``), which ``nfl_ats.players`` does not
    read (M7, ``docs/injury_timestamp_fallback.md``): the source actually
    consumed is the pinned snapshot under ``data/players/raw/<id>``. This
    reads that snapshot's own manifest and ``injuries.parquet`` directly and
    reports ``observed_at=None`` (absent -- this policy's ``on_absent`` is
    ``DEGRADED``, never ``BLOCKED``) when ``season`` has zero rows with a
    real ``date_modified``, exactly the nflverse 2025 release dropping that
    column entirely. A snapshot canonicalized with
    ``timestamp_fallback="week_proxy"`` carries an ``observed_at_basis``
    column; only rows basis-tagged ``"date_modified"`` count as real here,
    so a fully-proxied season still reports absent rather than borrowing the
    proxy's own manufactured credibility. Never raises on a missing/corrupt
    snapshot -- reports absent instead, consistent with every other
    observation in this module.
    """

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
    except Exception as error:  # report absent, never raise here
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
    """The report the publish path attaches to the card, in one call.

    ``player_arrests`` is the one source NOT read by directory scan. Its real
    freshness is whatever
    ``load_latest_complete_arrest_snapshot`` accepted -- a manifest that is
    complete, hash-verified, and neither future-dated nor over
    :data:`~nfl_ats.player_arrests_back_side_overlay.MAX_SNAPSHOT_AGE`. The
    caller passes that instant through (``ArrestOverlayResult.
    snapshot_fetched_at_utc``) so this layer can never disagree with the gate
    that already ran. When the caller has no verified instant (a rendering
    path that deliberately disabled the overlay), the source is left
    UNOBSERVED rather than absent -- a fail-closed source must not start
    blocking a path that never required it.

    ``arrest_snapshot_at`` is typed ``Any`` because callers pass a
    ``pandas.Timestamp``; anything with ``to_pydatetime`` or a ``datetime`` is
    accepted, so this module keeps no pandas import of its own.

    ``player_snapshot_root`` / ``player_snapshot_season`` (ENG-39, both
    optional and additive -- omitting either leaves
    ``injuries_nflverse_timestamps`` on the generic snapshot-dir scan, so no
    existing caller's report changes): when both are given, this overrides
    that source with :func:`player_snapshot_injury_timestamp_observation`
    read from the player snapshot actually consumed for the card, rather
    than the generic newest-directory scan.
    """

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
    """Read the current (or preseason's next) slate using the scheduler's ET clock.

    Keep the entire NFL week, including games already kicked off: choosing only
    remaining games would incorrectly reset a missed window to not-due. A missing
    or unreadable schedule supplies no exemption. Future snapshots are excluded.
    """
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
    """The table as plain rows, for docs/tests to assert against."""

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
