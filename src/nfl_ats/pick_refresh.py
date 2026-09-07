"""Late-week pick-refresh flow (POL-11): editable picks, frozen grading lines.

The pool's rule, confirmed by the owner 2026-08-20, is simpler than the
project had been assuming: **pool picks are editable up to each game's own
kickoff; only the grading LINES freeze at the Tuesday lock.** That is a real,
previously-unused edge -- a Friday injury designation or a kickoff-nearest
weather forecast can inform a pick that the frozen Tuesday line never had a
chance to price -- and this module is the second, opt-in step that spends it.

Two invariants make this safe rather than a way to quietly rewrite history:

1. **Grading is always against the frozen Tuesday line.** ``refresh-picks``
   recomputes the active model's probability with CURRENT features (current
   injury designations, current weather, current everything upstream of the
   spread), but always scores that recompute at the ORIGINAL card's spread
   line -- never at whatever the line has since become. The "original card"
   is read from the paper-decision ledger
   (:func:`nfl_ats.clv.load_paper_decisions`), not from a re-read of the
   linked weekly-forecast artifact, because the ledger is the one place in
   this codebase already proven to survive a same-week republish without its
   anchor moving (``nfl_ats.clv.record_paper_decisions``: "a republished card
   with a moved line never rewrites the CLV anchor"). A game with no recorded
   original line can never be refreshed -- fail closed for that game, not a
   silent fallback to whatever line the current feature table happens to
   carry.
2. **Every revision is an append-only, timestamped, kickoff-guarded row.**
   Nothing already written -- the Tuesday card, an earlier revision -- is
   ever rewritten in place. A game whose deadline has passed (see below) is
   never revised, enforced here in code, not left to caller discipline.

Per-game deadline, not one weekly pass
---------------------------------------
Thursday games exist, so "the week's refresh" cannot be a single Tuesday-to-
Sunday pass with one shared cutoff. Two owner directives (2026-08-20) pin the
exact rule:

* A pick may change up until its OWN game's kickoff.
* But nothing may change after **Sunday 4:00 PM ET** of that week, even for
  games that kick off later (SNF, MNF) -- so Sunday/Monday-night picks lock
  early, at the same moment as the rest of the week, not at their own
  kickoff.

Both hold at once, so the real per-game deadline is
``min(game_kickoff, sunday_16_00_et_of_that_week)`` (:func:`pick_deadline`,
:func:`sunday_pick_lock`). One consequence worth remembering whenever this
channel's evidence gets read later: a Thursday-night pick gets at most a
Tuesday-to-Thursday information window, while a Sunday or Monday pick can use
everything through Sunday afternoon -- the channel's information depth is not
uniform across a week's games.

Overlays
--------
Tuesday's paper ledger stores the final played side and the four production
members' frozen flags. A refresh refits the raw model at the frozen Tuesday
line, complements it once when any frozen member fired, and only then applies
the observed-movement policy. It never reloads or recomputes coach, division
revenge, player-arrest, or spread-gap inputs, so a later source revision cannot
retroactively change the Tuesday information set. See
``docs/late_week_refresh.md`` for the reasoning.

Observed-movement pick policy (POL-11 addendum, 2026-08-20)
-------------------------------------------------------------
One market-based decision rule IS applied to the played pick, distinct from
every pick-level overlay above (those stay challenger-tracked only). Once the
model's own recompute (post coach-fade) is in hand, :func:`plan_refresh` reads
whatever line the scheduled ``odds-ingest`` capture has already landed
(:func:`current_captured_home_spread` -- read-only, never a live fetch) and
compares it against the frozen Tuesday line. A move of >=1.0 point overrides
the pick to the side the market moved toward; below that, or with no fresh
capture (fail-open), the model's own recompute stands. Both the played pick
and the model-only counterfactual are recorded on every ledger row
(``movement_policy``, ``movement_delta``, ``movement_pick_side``,
``model_only_pick_side``) -- see ``docs/late_week_refresh.md``'s "Observed-
movement pick policy" section for the full predeclaration and the evidence
this is an EV play, not a resolved finding.

Promoted late-week follow (MKT-15/CX18, owner order 2026-09-05)
--------------------------------------------------------------
A second, separately predeclared market arm now takes precedence over the
1.0-point rule above: the equal-book Wednesday-to-deadline net move over the
frozen twelve-book universe follows the market at >=0.5 points
(``LATE_WEEK_MOVE_FOLLOW_POLICY``). It runs on the live intraday archive
only (read-only, fail-open), shares its exact computation with the paired
``late_week_move_follow_refresh_v1`` challenger ledger (which keeps
recording Tuesday-vs-movement sides on every pass), and every ledger row
keeps both arms' evidence (``late_week_*`` and ``consensus_*``) beside the
governing ``movement_policy`` and the ``model_only_pick_side``
counterfactual. See ``docs/late_week_refresh.md``'s promotion section.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.calibration import ResidualSmoothingMethod
from nfl_ats.clv import (
    LIVE_CAPTURE_KIND,
    load_decision_quotes,
    load_paper_decisions,
    refuse_if_outside_recording_lock_window,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet, atomic_text, run_id
from nfl_ats.lines import apply_external_lines
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, MarginFeatureProfile
from nfl_ats.market_data import load_quote_history, spread_consensus
from nfl_ats.nfl_week import week_cycle_sunday
from nfl_ats.outcomes import MARGIN_DISTRIBUTION_METHODS, fit_margin_models_for_week
from nfl_ats.prediction_safety import validate_three_way_split
from nfl_ats.provenance import sha256_file
from nfl_ats.sharp_book_movement_features import (
    THRESHOLD as LATE_WEEK_FOLLOW_THRESHOLD,
)
from nfl_ats.sharp_book_movement_features import (
    late_week_follow_frame,
)
from nfl_ats.weekly import CARD_PATH_TABLES

# ---------------------------------------------------------------------------
# Per-game deadline: min(own kickoff, that week's Sunday 4:00 PM ET)
# ---------------------------------------------------------------------------

#: Owner directive, 2026-08-20. Every DECISION_TIMES-style Eastern-anchored
#: timestamp elsewhere in this project (``nfl_ats.odds_backfill``) uses the
#: same zone; this is a separate constant (not that module's own
#: ``sun_late_close`` at 16:15 ET) because the pick-lock rule is a distinct,
#: explicitly-stated 4:00 PM cutoff, not a market-close proxy.
PICK_LOCK_TIMEZONE = ZoneInfo("America/New_York")
SUNDAY_PICK_LOCK_LOCAL_TIME = time(16, 0)


def sunday_pick_lock(kickoffs: pd.Series) -> pd.Timestamp:
    """The week-wide Sunday 4:00 PM ET pick-lock instant, in UTC.

    Anchored on the MODE Tue..Mon cycle Sunday among the supplied kickoffs
    (mirrors ``nfl_ats.odds_backfill.plan_backfill``'s own anchor selection),
    so one isolated Tuesday/Wednesday reschedule cannot shift the week's
    lock instant.
    """

    valid = pd.to_datetime(kickoffs, utc=True, errors="coerce").dropna()
    if valid.empty:
        raise ValueError("Cannot anchor a Sunday pick lock with no kickoffs")
    local_dates = valid.dt.tz_convert(PICK_LOCK_TIMEZONE).dt.date
    cycle_sundays: Counter[date] = Counter(week_cycle_sunday(day) for day in local_dates)
    anchor = min(cycle_sundays.items(), key=lambda item: (-item[1], item[0]))[0]
    local = datetime.combine(anchor, SUNDAY_PICK_LOCK_LOCAL_TIME, tzinfo=PICK_LOCK_TIMEZONE)
    return pd.Timestamp(local).tz_convert("UTC")


#: Every composition policy id a Tuesday paper ledger may carry. Ordered
#: oldest first; the last entry is the one currently played
#: (``nfl_ats.four_overlay_composition.POLICY_ID``).
PRODUCTION_COMPOSITION_POLICY_IDS: tuple[str, ...] = (
    "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1",
    "overlay_union_coach_division_revenge_player_arrests_v2",
)


def pick_deadline(kickoff: pd.Timestamp, sunday_lock: pd.Timestamp) -> pd.Timestamp:
    """One game's real pick deadline: the earlier of its own kickoff and the
    week-wide Sunday 4:00 PM ET lock -- so SNF/MNF picks lock early, and a
    Thursday game's own kickoff (always earlier than that Sunday) is
    untouched by the Sunday rule."""

    return min(kickoff, sunday_lock)


# ---------------------------------------------------------------------------
# Observed-movement pick policy (POL-11 addendum, 2026-08-20)
# ---------------------------------------------------------------------------

#: Frozen from the predeclared 0.5/1.0 measurement grid in
#: ``docs/observed_movement_channel.md``: 1.0 was the STRONGER arm at BOTH
#: gradings measured 2026-08-20 [read, registry/weak_signals.json] --
#: Tuesday-to-close full-slate (``observed_movement_threshold_1_0`` +1.863
#: accuracy points, probability_positive 0.935, n=1503, vs
#: ``observed_movement_threshold_0_5`` +1.663 points, P+ 0.873, n=1503) and
#: the Sunday-morning-realism grading (``observed_movement_threshold_1_0_sunday_am_realism``
#: +3.254 points, P+ 0.981, n=799, interval [+0.251, +6.266], vs
#: ``observed_movement_threshold_0_5_sunday_am_realism`` +1.627 points, P+
#: 0.764, n=799). Every one of those entries is classified
#: ``unresolved_below_power`` -- an interval crossing zero is never grounds to
#: reject a signal (AGENTS.md) -- so playing this threshold is an EV decision
#: under the project's forced-pick standing order (probability_positive far
#: above 0.5 on both gradings), not a claim that the channel is a resolved
#: finding. This constant is FROZEN by that predeclaration and is not
#: re-tuned here.
MOVEMENT_POLICY_THRESHOLD = 1.0
MOVEMENT_POLICY_MOVEMENT = "movement_ge_1.0"
MOVEMENT_POLICY_MODEL_ONLY = "model_only"

#: The promoted MKT-15/CX18 late-week follow, wired into the served refresh
#: pick by owner order 2026-09-05 (first live fire: the Thursday 2026-09-10
#: refresh): when the equal-book Wednesday-to-deadline net move reaches the
#: frozen 0.5-point threshold, the served pick follows the market. It takes
#: precedence over the 1.0-point latest-consensus rule below; both arms stay
#: recorded on every row so a later settlement pass can compare them.
LATE_WEEK_MOVE_FOLLOW_POLICY = "late_week_move_follow_0_5"

#: Every ``movement_policy`` value that means a market arm (rather than the
#: model's own recompute) governed ``new_pick_side`` this pass.
MOVEMENT_GOVERNED_POLICIES = (MOVEMENT_POLICY_MOVEMENT, LATE_WEEK_MOVE_FOLLOW_POLICY)


def _movement_side(delta: float) -> str:
    """The side the market moved toward: HOME if the home spread rose, else AWAY.

    Reuses ``scripts/observed_movement_channel.py``'s ``_threshold_pick`` sign
    logic verbatim: ``delta > 0`` (the home-oriented spread number increased,
    i.e. the market moved toward home) picks HOME, everything else (including
    an exact tie) picks AWAY. Only ever consulted by :func:`plan_refresh` when
    ``abs(delta) >= MOVEMENT_POLICY_THRESHOLD``, so the tie behavior is never
    actually selected -- it exists only so this helper totally orders every
    possible delta the same way the measurement script does.
    """

    return "HOME" if delta > 0.0 else "AWAY"


def current_captured_home_spread(
    data_root: Path, *, now: pd.Timestamp
) -> tuple[dict[str, float], dict[str, Any]]:
    """The latest LOCALLY CAPTURED market home spread per game, read-only.

    Reuses the exact adapter this project's scheduled capture already
    populates -- ``scripts/odds_capture.ps1`` (Task Scheduler) runs
    ``nfl-ats odds-ingest`` several times each morning
    (``docs/ops_runbook.md``: "the scheduled live odds captures land"
    ~06:00-09:00 ET), which writes through
    ``nfl_ats.market_data.write_market_snapshot`` into
    ``data_root/market/raw`` -- the SAME directory the historical
    ``odds-backfill`` executor also writes into. This function never
    triggers a live fetch itself: it only reads what is already committed
    there, via ``nfl_ats.market_data.load_quote_history`` /
    ``spread_consensus`` -- the IDENTICAL "current line" read
    ``nfl_ats.best_pick_nomination.week_dispersion_pool`` already uses for
    exactly the reason stated there ("never calls the odds API"), and the
    same one the ``odds-summary`` CLI command surfaces to a human. This
    mirrors ``nfl_ats.clv.predict_close_for_week``'s own read-only design
    (it raises ``ClosePredictionUnavailable`` rather than fetching on
    demand) -- ``refresh-picks`` is a scheduled/on-demand pipeline step, not
    a place that should spend API quota or make a network call on every
    invocation.

    Sign convention: the returned home spread is read from the SAME
    ``home_spread_line`` column (median across books, latest pre-kickoff
    quote per book) that ``tue_open_home_spread`` / ``close_home_spread``
    are built from throughout ``nfl_ats.clv`` -- the identical home-favorite-
    negative convention ``nfl_ats.clv.opener_pick_evaluation`` already
    relies on when it compares those columns directly against nflverse-
    sourced results. Subtracting this value from ``decision_home_spread``
    (also that same convention -- see :func:`original_card`) reuses that
    established pairing rather than a new one.

    "Fresh" means the newest quote's ``observed_at_utc`` across the WHOLE
    local store falls on the same America/New_York calendar date as ``now``.
    Historical-backfill snapshots (written under the same
    ``data_root/market/raw`` tree) carry the REQUESTED historical timestamp
    as ``observed_at_utc``, not the time they were actually fetched
    (``nfl_ats.odds_backfill``), so they can never masquerade as "today" --
    this simple, store-wide check is equivalent in practice to filtering for
    the scheduled live captures specifically, without a second read path.

    Returns ``({}, metadata)`` -- an EMPTY mapping, never a partial or stale
    one -- whenever the store is empty or its newest quote is not from today;
    ``metadata["fresh"]`` is ``False`` and ``metadata["reason"]`` names why.
    Callers MUST fail open on an empty mapping (recompute with the model
    only), never raise.
    """

    market_root = data_root / "market" / "raw"
    quotes = load_quote_history(market_root)
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


# ---------------------------------------------------------------------------
# The append-only pick-revision ledger
# ---------------------------------------------------------------------------

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
    "consensus_delta",
    "consensus_pick_side",
    "model_id",
    "feature_table_sha256",
    "reason",
    "trigger_type",
    "trigger_source",
    "trigger_observed_at_utc",
)


#: Refresh trigger vocabulary (MKT-08). Scheduled passes are clock-dispatched;
#: a future news-driven pass records "news_event" with the feed as source.
TRIGGER_CLOCK_DISPATCH = "clock_dispatch"
TRIGGER_NEWS_EVENT = "news_event"
TRIGGER_UNKNOWN = "unknown"


def pick_revision_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "pick_revisions.parquet"


def load_pick_revisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only pick-revision ledger (empty frame when none exists)."""

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
        "consensus_delta": None,
        "consensus_pick_side": "",
    }
    for column, default in legacy_defaults.items():
        if column not in ledger.columns:
            ledger[column] = default
    missing = sorted(set(PICK_REVISION_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(f"Pick-revision ledger is missing columns: {', '.join(missing)}")
    return ledger[list(PICK_REVISION_COLUMNS)]


def describe_week_revisions(
    revisions: pd.DataFrame,
    games: tuple[tuple[str, str, str], ...],
    *,
    season: int | None,
    week: int | None,
) -> tuple[str, ...]:
    """One plain sentence per late-week-refreshed game (UI-17).

    ``games`` is ``(game_id, away_team, home_team)`` triples for the
    published card. Only the latest revision per game in THIS
    season/week is reported; revisions for other weeks or unknown games
    are skipped, never interpolated. Every number below comes straight
    off the ledger row, so the assistant's numeric guard holds by
    construction.
    """

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
        if new_side == previous_side:
            lines.append(
                f"{away} at {home} refresh ({run_id}){trigger_text}: "
                f"refresh confirmed {new_side}, no change from Tuesday; {spread_text}."
            )
        else:
            delta = revision.get("movement_delta")
            movement_text = f"; line moved {float(delta):+g} points" if pd.notna(delta) else ""
            lines.append(
                f"{away} at {home} refresh ({run_id}){trigger_text}: "
                f"pick now {new_side} (Tuesday card: {previous_side}); "
                f"{spread_text}{movement_text}."
            )
    return tuple(lines)


def original_card(artifacts_root: Path, *, season: int, week: int) -> pd.DataFrame:
    """The frozen Tuesday card for one week: recorded lines, picks, kickoffs.

    Sourced from :func:`nfl_ats.clv.load_paper_decisions` -- the append-only
    paper-decision ledger written by ``publish-predictions
    --record-decisions`` -- rather than the active model's linked weekly-
    forecast artifact, because the ledger is what already survives a
    same-week republish without its ``decision_home_spread`` anchor moving.
    Empty when this week was never recorded (no ``--record-decisions`` run
    happened yet); callers must treat that as "nothing to refresh," not fill
    in a substitute line.
    """

    ledger = load_paper_decisions(artifacts_root)
    if ledger.empty:
        return ledger
    rows = ledger.loc[ledger["season"].astype(int).eq(season) & ledger["week"].astype(int).eq(week)]
    return rows.reset_index(drop=True)


def _utc(instant: datetime | None) -> pd.Timestamp:
    value = pd.Timestamp(instant if instant is not None else datetime.now(UTC))
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _published_pick_side(original: pd.DataFrame) -> pd.Series:
    """The final Tuesday-published side already frozen in the paper ledger."""

    return original.set_index("game_id")["pick_side"].astype(str)


# ---------------------------------------------------------------------------
# The refresh plan: pure computation, no ledger writes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefreshedGame:
    """One game's refreshed read, whether or not it ends up changing."""

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
    #: Which arm governed ``new_pick_side`` this pass:
    #: ``LATE_WEEK_MOVE_FOLLOW_POLICY`` (the equal-book Wednesday-to-deadline
    #: net move reached 0.5 points and the pick followed the market),
    #: ``MOVEMENT_POLICY_MOVEMENT`` (the latest captured consensus line moved
    #: >=1.0 point and the pick followed it), or ``MOVEMENT_POLICY_MODEL_ONLY``
    #: (below both thresholds, or no market evidence -- the model's own
    #: recompute stands, fail-open). The late-week arm takes precedence when
    #: both fire.
    movement_policy: str
    #: The governing arm's signed move in home-oriented points: the late-week
    #: equal-book net move when the late-week arm governs, else the
    #: current_captured_home_spread - decision_home_spread consensus delta
    #: (same sign convention as open_move elsewhere in this project) when a
    #: fresh captured line exists, else the late-week net move when only that
    #: arm has evidence. ``None`` when no market arm has evidence this pass.
    movement_delta: float | None
    #: The side the governing (or counterfactual) market arm points at,
    #: computed whenever ``movement_delta`` is not ``None`` (blank string
    #: otherwise) -- the candidate side even on passes where the policy did
    #: not select it.
    movement_pick_side: str
    #: The model's own recomputed pick (post coach-fade, pre movement-policy
    #: override) -- always present, the counterfactual arm the
    #: ``model_only_refresh_incumbent`` challenger tracks. Equals
    #: ``new_pick_side`` whenever ``movement_policy`` is
    #: ``MOVEMENT_POLICY_MODEL_ONLY``. NOTE: ``new_home_cover_probability`` is
    #: always the model's own probability estimate, never altered by the
    #: movement policy (only the discrete side can be overridden) -- when a
    #: movement arm governs, ``new_pick_side`` may therefore differ from
    #: the usual >=0.5-on-``new_home_cover_probability`` rule; that is the one
    #: deliberate, disclosed exception to that invariant in this codebase,
    #: fully recoverable from these columns.
    model_only_pick_side: str
    eligible: bool
    ineligible_reason: str
    changed: bool
    #: The MKT-15/CX18 late-week arm's own evidence, always recorded
    #: alongside the governor above: equal-book Wednesday-to-deadline net
    #: move (``None`` when the live intraday archive has no usable quotes
    #: for this game this pass), the market side at the frozen 0.5-point
    #: threshold (``""`` when unavailable), and how many of the twelve
    #: frozen-universe books contributed (0 when unavailable).
    late_week_net_move: float | None = None
    late_week_pick_side: str = ""
    late_week_eligible_books: int = 0
    #: The 1.0-point latest-consensus arm's own evidence, same shape:
    #: current_captured_home_spread - decision_home_spread (``None`` when no
    #: fresh captured line exists for this game) and the side the market
    #: moved toward (``""`` when unavailable).
    consensus_delta: float | None = None
    consensus_pick_side: str = ""


@dataclass(frozen=True)
class RefreshResult:
    """Everything one ``refresh-picks`` computation produced, unwritten."""

    season: int
    week: int
    refresh_run_id: str
    computed_at_utc: pd.Timestamp
    model_id: str
    feature_table_path: str
    feature_table_sha256: str
    games: tuple[RefreshedGame, ...]
    #: On the current week's feature table but with no recorded original
    #: line -- cannot be refreshed, fail closed for these.
    unrefreshable_game_ids: tuple[str, ...]
    #: In the recorded original card but absent from the current feature
    #: table (e.g. a stale/mismatched build) -- also cannot be refreshed.
    missing_from_features_game_ids: tuple[str, ...]
    #: :func:`current_captured_home_spread`'s metadata dict for this pass
    #: (``fresh``/``reason``/``latest_observed_at_utc``/``games_with_current_line``).
    #: Empty when there were no refreshable games to look a line up for.
    current_line_metadata: dict[str, Any] = field(default_factory=dict)
    #: The promoted late-week follow arm's metadata dict for this pass
    #: (``available``/``reason``/``games_with_exposure``/``games_followed``/
    #: ``refused_quote_rows``). ``available`` is ``False`` (fail-open) whenever
    #: the live intraday archive is absent, unusable, or covers no game.
    late_week_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_games(self) -> tuple[RefreshedGame, ...]:
        return tuple(game for game in self.games if game.changed)

    @property
    def ineligible_games(self) -> tuple[RefreshedGame, ...]:
        return tuple(game for game in self.games if not game.eligible)


def _active_model_config(
    artifacts_root: Path,
) -> tuple[dict[str, Any], str, MarginFeatureProfile, str, float, ResidualSmoothingMethod]:
    """Load and validate the active model identity, or fail closed.

    Mirrors ``nfl_ats.weekly.assert_synchronized``'s spirit: refresh-picks
    must recompute under the EXACT model identity the Tuesday card (and the
    lines it froze) were produced under, never a silently different one.
    Validates ``feature_profile`` against the full
    ``nfl_ats.margin.MARGIN_FEATURE_PROFILES`` set (what
    ``fit_margin_models_for_week`` actually accepts), NOT against
    ``nfl_ats.weekly.CARD_PATH_TABLES``'s narrower "player"/"weak_stack"
    allowlist -- that allowlist exists so weekly-run knows which BUILD
    command produces a profile's table, which is irrelevant here: this
    module only ever READS an already-built table, and consults
    ``CARD_PATH_TABLES`` in :func:`plan_refresh` purely as a default-path
    convenience when ``--features`` is not given explicitly.
    """

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
    """The promoted MKT-15/CX18 follow arm's per-game evidence, fail-open.

    Runs the exact frozen rule the paired ``late_week_move_follow_refresh_v1``
    challenger records (:func:`late_week_follow_frame`: equal-book
    Wednesday-to-deadline net increments over the twelve-book universe,
    0.5-point follow, Tuesday-anchored, Sunday evidence excluded) against the
    same Tuesday card, so the served pick and the challenger ledger agree by
    construction. Anything missing or unusable -- no live intraday archive,
    no pre-deadline book changes, an unreadable store -- returns an empty
    lookup (the arm is unavailable and the existing consensus/model-only
    logic stands), never raises into the refresh pass.
    """

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
        net_move = float(cast(Any, row.equal_net_move))
        eligible_books = int(cast(Any, row.eligible_books))
        lookup[str(row.game_id)] = {
            "net_move": net_move,
            "pick_side": str(row.movement_would_be_pick_side),
            "eligible_books": eligible_books,
        }
    followed = sum(
        1
        for value in lookup.values()
        if cast(int, value["eligible_books"]) > 0
        and abs(cast(float, value["net_move"])) >= LATE_WEEK_FOLLOW_THRESHOLD
    )
    return lookup, {
        "available": True,
        "reason": "",
        "games_with_exposure": int(exposure.eligible_books.gt(0).sum()),
        "games_followed": followed,
        "refused_quote_rows": refused,
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
    """Recompute this week's picks with current data at the frozen Tuesday lines.

    Read-only: never touches ``active_ats_model.json``, the linked weekly-
    forecast artifact, or any ledger. ``record_refresh`` is the write path.
    """

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
        raise ValueError(
            "The active model has changed since this week's original card was recorded "
            f"(recorded model_id(s) {mismatched_model}, active model_id {model_id!r}); "
            "refresh-picks refuses to recompute picks under a different model identity "
            "than the one the pool's grading lines were locked against."
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
    if not refreshable.empty:
        lines = original[["game_id", "decision_home_spread"]].rename(
            columns={"decision_home_spread": "home_spread"}
        )
        overridden = apply_external_lines(refreshable, lines)
        forecasts = model.predict(overridden, probability_method=probability_method)
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
        policy_ids = set(original["decision_policy_id"].astype(str))
        # One frozen production composition per week. Both the retired
        # four-member union (spread-gap flip included, through 2026-09-07) and
        # the three-member union that replaced it (owner order, 2026-09-07:
        # no unexplained threshold flips) carry the composed flag this
        # refresh re-applies, so either is acceptable -- never a mix.
        if len(policy_ids) != 1 or not policy_ids <= set(PRODUCTION_COMPOSITION_POLICY_IDS):
            raise DataContractError(
                "Refresh requires one frozen production composition policy for the week"
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

        sunday_lock = sunday_pick_lock(original["kickoff"])
        published_side = _published_pick_side(original)
        current_lines, line_metadata = current_captured_home_spread(data_root, now=computed_at)
        late_week_lookup, late_week_metadata = _late_week_follow_lookup(
            original_indexed,
            overlaid,
            data_root,
            sunday_lock=sunday_lock,
            now=computed_at,
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
            late_week = late_week_lookup.get(game_id)
            if late_week is None:
                late_week_net: float | None = None
                late_week_side = ""
                late_week_books = 0
                late_week_fires = False
            else:
                late_week_net = late_week["net_move"]
                late_week_side = late_week["pick_side"]
                late_week_books = late_week["eligible_books"]
                late_week_fires = (
                    late_week_books > 0
                    and late_week_net is not None
                    and abs(late_week_net) >= LATE_WEEK_FOLLOW_THRESHOLD
                )
            if late_week_fires:
                policy = LATE_WEEK_MOVE_FOLLOW_POLICY
                new_side = late_week_side
                movement_delta = late_week_net
                movement_pick_side = late_week_side
            elif consensus_fires:
                assert consensus_delta is not None and consensus_side
                policy = MOVEMENT_POLICY_MOVEMENT
                new_side = consensus_side
                movement_delta = consensus_delta
                movement_pick_side = consensus_side
            else:
                policy = MOVEMENT_POLICY_MODEL_ONLY
                new_side = model_only_side
                if consensus_delta is not None:
                    movement_delta = consensus_delta
                    movement_pick_side = consensus_side
                elif late_week_net is not None:
                    movement_delta = late_week_net
                    movement_pick_side = late_week_side
                else:
                    movement_delta = None
                    movement_pick_side = ""
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
                    consensus_delta=consensus_delta,
                    consensus_pick_side=consensus_side,
                    eligible=eligible,
                    ineligible_reason=reason,
                    changed=changed,
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
    )


# ---------------------------------------------------------------------------
# The write path: opt-in, kickoff-guarded, append-only
# ---------------------------------------------------------------------------


def refresh_summary(plan: RefreshResult, *, record_decisions: bool) -> dict[str, Any]:
    """The JSON-printable summary of an already-computed plan, minus the
    ``"ledger"`` key -- callers that need the ``RefreshResult`` itself for a
    SECOND purpose (e.g. ``--publish-card``'s section render) should call
    :func:`plan_refresh` once and build both from the same object, rather
    than calling :func:`record_refresh` and re-planning (which would refit
    the model twice and risk two calls disagreeing on "now")."""

    return {
        "season": plan.season,
        "week": plan.week,
        "refresh_run_id": plan.refresh_run_id,
        "computed_at_utc": plan.computed_at_utc.isoformat(),
        "model_id": plan.model_id,
        "games_considered": len(plan.games),
        "changed_game_ids": [game.game_id for game in plan.changed_games],
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
            "games_consensus_applied": [
                game.game_id
                for game in plan.games
                if game.movement_policy == MOVEMENT_POLICY_MOVEMENT
            ],
            "late_week_follow": {
                "threshold": LATE_WEEK_FOLLOW_THRESHOLD,
                "available": bool(plan.late_week_metadata.get("available", False)),
                "reason": plan.late_week_metadata.get("reason", ""),
                "games_with_exposure": plan.late_week_metadata.get("games_with_exposure", 0),
                "games_followed": plan.late_week_metadata.get("games_followed", 0),
                "refused_quote_rows": plan.late_week_metadata.get("refused_quote_rows", 0),
                "games_late_week_follow_applied": [
                    game.game_id
                    for game in plan.games
                    if game.movement_policy == LATE_WEEK_MOVE_FOLLOW_POLICY
                ],
            },
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
) -> dict[str, Any]:
    """Append ``plan``'s changed, eligible picks to the ledger, or not.

    ``record_decisions`` defaults to ``False`` -- mirrors
    ``publish-predictions --record-decisions``: an ordinary or rehearsal call
    never reaches the ledger. When true, this additionally reuses
    ``nfl_ats.clv.refuse_if_outside_recording_lock_window`` unchanged against
    the week's ORIGINAL kickoffs, so a refresh invoked weeks before a real
    lock week still cannot backdate anything; the per-game kickoff/Sunday-
    lock guard already computed inside :func:`plan_refresh` is what actually
    decided which games may be revised at all -- only ``changed`` (already
    eligibility-filtered) games are ever appended here.

    Every appended row carries MKT-08 trigger provenance: ``trigger_type`` is
    ``clock_dispatch`` for the scheduled passes (a future news-driven pass
    records ``news_event``), ``trigger_source`` names the scheduler job or
    invoking context, and ``trigger_observed_at_utc`` defaults to the plan's
    own computation time.
    """

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
    if not changed:
        return {"recorded": 0, "ledger_rows": len(existing)}

    reason_text = f"pick_refresh recompute ({note})" if note else "pick_refresh recompute"
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
            "consensus_delta": [game.consensus_delta for game in changed],
            "consensus_pick_side": [game.consensus_pick_side for game in changed],
            "model_id": plan.model_id,
            "feature_table_sha256": plan.feature_table_sha256,
            "reason": reason_text,
            "trigger_type": trigger_type,
            "trigger_source": trigger_source,
            "trigger_observed_at_utc": observed_at,
        }
    )
    combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    atomic_parquet(combined[list(PICK_REVISION_COLUMNS)], pick_revision_ledger_path(artifacts_root))
    return {"recorded": len(rows), "ledger_rows": len(combined)}


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
    """Plan a refresh and, when ``record_decisions``, append changed picks.

    A convenience one-shot wrapper around :func:`plan_refresh`,
    :func:`refresh_summary` and :func:`record_plan` for callers (tests, a
    one-shot CLI invocation) that do not need the ``RefreshResult`` for a
    second purpose. ``--publish-card`` on the ``refresh-picks`` CLI command
    needs the plan object itself, so it calls the three pieces directly
    instead of this wrapper -- see ``nfl_ats.cli._cmd_refresh_picks``.
    """

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
    )
    return summary


def final_pick_per_game(artifacts_root: Path, *, season: int, week: int) -> pd.DataFrame:
    """The FINAL pre-kickoff pick per game: the latest revision if any, else
    the Tuesday-published pick. Recovers both the Tuesday and final pick for
    scoring -- callers that want the Tuesday pick alone should read
    :func:`original_card` directly.
    """

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


# ---------------------------------------------------------------------------
# Opt-in CURRENT_PREDICTIONS.md append (never rewrites the Tuesday section)
# ---------------------------------------------------------------------------

LATE_WEEK_REFRESH_START = "<!-- LATE_WEEK_REFRESH:START -->"
LATE_WEEK_REFRESH_END = "<!-- LATE_WEEK_REFRESH:END -->"


def _refresh_section_markdown(result: RefreshResult, note: str) -> str:
    changed = result.changed_games
    heading = f"## Late-week refresh (as of {result.computed_at_utc.isoformat()})\n\n"
    label = f" ({note})" if note else ""
    if not changed:
        return heading + f"No pick changes since the Tuesday card{label}.\n"

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
                "Previous pick": game.previous_pick_side,
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
        'yet passed were eligible. "Policy" is `late_week_move_follow_0_5` when late-week '
        "lines moved at least half a point since Tuesday and the pick followed the market, "
        "`movement_ge_1.0` when the pool's own captured line instead moved >=1.0 point and "
        "the pick followed it, or `model_only` when neither market arm fired (or no market "
        "evidence was available) -- see docs/late_week_refresh.md's movement-policy "
        "sections.\n\n"
    )
    return heading + intro + table + "\n"


def append_refresh_to_card(destination: Path, result: RefreshResult, *, note: str = "") -> None:
    """Additively label a "Late-week refresh" section onto a published card.

    Never rewrites anything above the marker pair -- the Tuesday section
    publish-predictions wrote stays exactly as published. Re-running this
    against the same card replaces only its own section (idempotent), so
    repeated refresh passes across a week never pile up duplicate blocks.
    """

    if not destination.is_file():
        raise ValueError(
            f"Cannot append a late-week refresh section: no published card at {destination}; "
            "run `nfl-ats publish-predictions` first."
        )
    text = destination.read_text(encoding="utf-8")
    section = _refresh_section_markdown(result, note)
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
