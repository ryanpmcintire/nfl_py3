"""Shared view models for the ATS Terminal site's pages: The Model
(``model.html``), History (``history.html``), and What We've Learned
(``findings.html``). ``index.html``
(This Week) is :mod:`nfl_ats.board_content`'s own ``BoardContent``.

2026-09-02: the site has four pages, dedup'd so every
fact appears on exactly one page (the This Week headline strip is the one
declared exception -- see :class:`ModelPageContent`'s docstring). This
replaced an earlier six-extra-page draft (Models, Team Trends, Findings,
Track Record, Pool Workbench, Signal Ledger) that the owner found
"partially duplicating things": Models and Track Record both carried
overlapping headline/accuracy stats and the SAME tracked-challenger cards
the model ledger already carries with richer detail; Findings and the
Signal Ledger both existed to describe the weak-signal registry. Team
Trends and Pool Workbench were cut entirely -- their content is inventoried
in the session handoff for the owner to recall, not silently deleted from
the codebase: the underlying library modules (``nfl_ats.team_explorer``,
``nfl_ats.pool_workbench``) and their own tests are untouched, only this
site's use of them is gone.

Exactly like ``board_content.py``, this module is the ONLY place that
touches an artifact, a loader, or a piece of prose for these two pages; the
Terminal renderer (:mod:`nfl_ats.board_terminal`) must read fields off the
dataclasses below and never format a number or invent a sentence itself.

Every mandatory data-integrity guard the corresponding page in
``public_board.py`` ran is reproduced here too: ``model_ledger
.validate_ledger`` (The Model's ledger section) and ``findings_registry
.validate_curation`` (Findings, against BOTH ``FINDINGS`` and
``LEAD_BLURBS``) -- a stale or drifted claim must fail this loader the same
way it fails ``build_public_site``, never render quietly.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.board_content import (
    BoardContent,
    HeadlineStats,
    LinkPreview,
    NumberProvenanceError,
    TickerChrome,
    _load_game_outcomes,
    load_board_content,
    verify_number_provenance,
)
from nfl_ats.clv import live_close_reference
from nfl_ats.dashboard import findings_content as fc
from nfl_ats.data import DataContractError
from nfl_ats.findings_registry import (
    RecentActivityEntry,
    RecentRegistryActivity,
    RegistryEntry,
    WatchingLead,
    load_all_entries,
    load_rotation_registry,
    load_weak_signal_registry,
    recent_registry_activity,
    top_open_leads,
    validate_curation,
)
from nfl_ats.model_explanation import FamilyExplanation, load_model_explanation
from nfl_ats.model_ledger import (
    STATUS_BADGE_PROMOTED,
    LedgerRow,
    build_model_ledger,
    validate_ledger,
)
from nfl_ats.model_weak_spots import (
    HomeCorrection,
    WeakSpots,
    build_home_correction,
    build_weak_spots,
)
from nfl_ats.prospective_scoring import (
    CLOSE_GRADE,
    DECISION_GRADE,
    load_challenger_decisions,
    settle_prospective_picks,
)
from nfl_ats.public_board import (
    OpenerEvaluationArtifacts,
    find_matching_opener_evaluation,
    humanize_identifier,
    load_opener_evaluation_artifacts,
    load_prospective_challengers,
    load_public_board_artifacts,
)
from nfl_ats.published_picks import FrozenPick, frozen_picks
from nfl_ats.reporting import artifact_directories, read_json
from nfl_ats.signal_ledger import build_ledger_rows
from nfl_ats.weak_signals import default_registry_path


def _default_data_root() -> Path:
    """Duplicate of ``board_content._default_data_root`` -- see that
    function's docstring for why this small env-var default is duplicated
    rather than imported (the name is private)."""

    import os

    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _generated_at_text(generated_at: datetime) -> str:
    return generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")


_STAMP_SUFFIX_RE = re.compile(r"\s*\d{8}T\d{6}(?:\d{6})?Z\s*$")


_EMBEDDED_STAMP_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T\d{6}(?:\d{6})?Z")


def _artifact_directory_date_text(directory_name: str) -> str:
    """A UTC-stamped artifact directory's own name reduced to a bare
    reader-facing date (``"2026-09-05"``) -- never the full stamp, which
    reads as machine notation, not a date (owner mandate, 2026-09-05).
    Falls back to the raw name when it doesn't parse -- never hides real
    data behind a formatting bug."""

    match = _EMBEDDED_STAMP_RE.search(directory_name)
    if match is not None:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return directory_name


def _number_provenance_rows(
    artifacts_root: Path, active: Mapping[str, Any]
) -> tuple[tuple[NumberProvenanceRow, ...], str | None]:
    """The model page's "where these numbers come from" fine print --
    :func:`nfl_ats.board_content.verify_number_provenance`, translated to
    reader-facing rows (a date, not a directory stamp; the model's own
    method label, not its id -- owner mandate, 2026-09-05: "please do not
    let those percentages get out of date anymore"). Fails OPEN here (a
    caught :class:`NumberProvenanceError` becomes a plain note, never a
    raised exception) because building this page's content must never be
    the enforcement point -- ``publish-board``/``publish-predictions``
    already fail CLOSED on this exact check before a page is written; a
    content-model rebuild for a test fixture or an off-cycle rehearsal must
    still render a page, just one that says verification did not run."""

    model_text = (
        f"{humanize_identifier(str(active.get('feature_profile') or 'unknown'))} "
        f"({humanize_identifier(str(active.get('method') or 'unknown'))})"
    )
    try:
        entries = verify_number_provenance(artifacts_root)
    except NumberProvenanceError:
        return (), "The current model's archive scores have not all been verified yet."
    rows = tuple(
        NumberProvenanceRow(
            label=entry.label,
            artifact_kind=_STAMP_SUFFIX_RE.sub("", entry.detail).strip(),
            date_text=_artifact_directory_date_text(Path(entry.source).name),
            model_text=model_text,
        )
        for entry in entries
    )
    return rows, None


@dataclass(frozen=True)
class LedgerEvidenceItem:
    registry_key: str
    probability_positive: float | None
    classification: str | None


@dataclass(frozen=True)
class ModelLedgerRowView:
    arm_id: str
    display_name: str
    status_badge: str
    is_promoted: bool
    games: int | None
    accuracy: float | None
    interval_low: float | None
    interval_high: float | None
    interval_unit: str
    grade: str
    summary_sentence: str
    own_probability_positive: float | None
    evidence: tuple[LedgerEvidenceItem, ...]
    agreement_text: str | None
    artifact_ref: str | None


@dataclass(frozen=True)
class FamilyWeightRow:
    label: str
    margin_share: float
    spread_share: float
    weight_in_spread: float
    stability_word: str
    stability_detail: str
    classification: str
    caption: str


@dataclass(frozen=True)
class GradingRuleView:
    protocol_opener: float | None
    protocol_close: float | None
    production_opener: float | None
    production_close: float | None


@dataclass(frozen=True)
class SeasonRowView:
    season: str
    games: int | None
    opener_accuracy: float
    close_accuracy: float | None


@dataclass(frozen=True)
class ModelPageContent:
    """The Model page's content, told as one story: what we play, how it's
    done, what's challenging it.

    ``headline`` is the SAME :class:`~nfl_ats.board_content.HeadlineStats`
    object the This Week page's headline strip renders -- this is the one
    deliberate cross-page dedup exception (see
    ``tests/test_board_content_coverage.py``): a reader needs the played-
    policy numbers in both places, and reading the SAME object rather than
    recomputing it is what keeps them from ever disagreeing. Everything
    else below is unique to this page: the season-blocked long-run
    interval, the protocol-vs-production grading-rule comparison, the
    season-by-season table, and the model ledger (dropped entirely from
    the old six-page draft: ``played_chain_text``/``expectation_text``/
    the opener-close-games-seasons KPI row and the standalone tracked-
    challenger cards all restated a headline or ledger fact under a
    different name).
    """

    generated_at_text: str

    headline: HeadlineStats
    ceiling_text: str
    ladder_rungs: tuple[str, ...]
    grading: GradingRuleView
    long_run_games: int | None
    long_run_correct: int | None
    long_run_range: tuple[float, float] | None

    seasons: tuple[SeasonRowView, ...]
    seasons_above_coin_flip: int
    seasons_even: tuple[str, ...]
    seasons_below: tuple[tuple[str, float], ...]

    ledger_available: bool
    ledger_error: str | None
    rows: tuple[ModelLedgerRowView, ...]
    graded_rows: tuple[ModelLedgerRowView, ...]
    waiting_rows: tuple[ModelLedgerRowView, ...]
    explanation_available: bool
    run_directory: str | None
    feature_profile: str | None
    matches_active_feature_table: bool | None
    families: tuple[FamilyWeightRow, ...]
    ticker_chrome: TickerChrome
    link_preview: LinkPreview
    number_provenance: tuple[NumberProvenanceRow, ...]
    number_provenance_note: str | None
    weak_spots: WeakSpots = field(default_factory=WeakSpots)


@dataclass(frozen=True)
class NumberProvenanceRow:
    """One reader-facing row of the model page's provenance fine print --
    :class:`nfl_ats.board_content.NumberProvenance`, translated for display:
    a human date instead of a directory timestamp, the model's own method
    label instead of its id. See :func:`_number_provenance_rows`."""

    label: str
    artifact_kind: str
    date_text: str
    model_text: str


@dataclass(frozen=True)
class HistoryPickRow:
    """One immutable pick from the primary paper-decision ledger.

    Confidence is optional because the ledger records the chosen side and
    frozen line, while the probability lives in the linked forecast artifact.
    A missing linked forecast is therefore displayed as ``--`` rather than a
    confidence value guessed from edge or market price.
    """

    game_id: str
    season: int | None
    week: int | None
    away_team: str
    home_team: str
    pick_side: str
    decision_home_spread: float | None
    confidence: float | None
    model_id: str | None
    best_pick: bool
    status: str
    correct: bool | None
    score_text: str | None

    @property
    def pick_team(self) -> str:
        if self.pick_side == "HOME":
            return self.home_team
        if self.pick_side == "AWAY":
            return self.away_team
        return self.pick_side

    @property
    def pick_line_text(self) -> str:
        if self.decision_home_spread is None:
            return "--"
        line = -self.decision_home_spread if self.pick_side == "HOME" else self.decision_home_spread
        return "PK" if line == 0.0 else f"{line:+g}"


@dataclass(frozen=True)
class ChallengerAssessment:
    """Settled prospective comparison for one challenger.

    ``paired_games`` is the number of games with non-push decision-line
    outcomes available for both arms. ``delta_accuracy_points`` is measured
    on that same paired set and remains ``None`` until both arms have settled
    games. Registry evidence is shown separately as P+ or its recorded
    interval; neither is used to decide which card is played.
    """

    challenger_id: str
    display_name: str
    paired_games: int
    wins: int
    losses: int
    pushes: int
    pending: int
    accuracy: float | None
    delta_accuracy_points: float | None
    probability_positive: float | None
    interval_low: float | None
    interval_high: float | None
    grading_basis: str


HISTORY_GRADE_CAPTION = (
    "Opener is the pool's own decision line, locked before kickoff; close is the market "
    "at its sharpest, right before kickoff. They differ because the market keeps moving "
    "after the pool's card locks. The pool settles picks at the OPENER -- that is this "
    "project's primary number; the close is shown alongside for context and never decides "
    "what gets played."
)

HISTORY_WEEK_REPLAY_CAPTION = (
    "Week by week: the weeks of finished seasons run the model again on the opening "
    "numbers those weeks really had, using only what was known before each week "
    "kicked off; weeks of the current season are the picks exactly as they were "
    "written down before kickoff."
)

ARCHIVE_OPENER_CORRECT_COLUMN = "correct_at_open_probability_rule"
ARCHIVE_CLOSE_CORRECT_COLUMN = "correct_at_close_probability_rule"

NO_OPENER_LINE_ARCHIVED_WEEK_NOTE = "No opener line archived for this week."
NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE = "No close line archived for this week."
HISTORY_WEEK_NOT_SETTLED_NOTE = "Not yet settled -- these games have not finished."
NO_OPENER_LINE_ARCHIVED_SEASON_NOTE = (
    "No opener line archived for this season -- only the aggregate close-graded "
    "evaluation on The Model page covers it."
)


@dataclass(frozen=True)
class SeasonGradeRow:
    """One season's opener-vs-close grading, side by side (UI-20(h)).

    Reuses the SAME per-season pair the Model page's :class:`SeasonRowView`
    renders (:func:`_season_rows`, sourced from
    ``artifacts/opener_evaluation/<newest>/season_summary.csv``) -- never a
    second computation, so the two pages can never disagree. ``note`` is
    non-empty only for the one synthetic row (see
    :func:`_season_grade_rows`) covering seasons this evaluation's archived
    population does not reach at all; every real season it returns already
    carries both grades by construction (its population requires a paired
    opener/close line), so ``opener_accuracy``/``close_accuracy`` are only
    ever both-present or both-``None`` here.
    """

    season_label: str
    games: int | None
    opener_accuracy: float | None
    close_accuracy: float | None
    note: str = ""

    @property
    def delta(self) -> float | None:
        if self.opener_accuracy is None or self.close_accuracy is None:
            return None
        return self.opener_accuracy - self.close_accuracy

    @property
    def opener_text(self) -> str:
        return f"{self.opener_accuracy:.1%}" if self.opener_accuracy is not None else "--"

    @property
    def close_text(self) -> str:
        return f"{self.close_accuracy:.1%}" if self.close_accuracy is not None else "--"

    @property
    def delta_text(self) -> str:
        delta = self.delta
        return f"{delta:+.1%}" if delta is not None else "--"


@dataclass(frozen=True)
class HistoryWeekGrade:
    """One recorded week's opener-vs-close grading, side by side (UI-20(h)).

    Settled through the SAME :func:`nfl_ats.prospective_scoring
    .settle_prospective_picks` the per-game :class:`HistoryPickRow` rows use
    -- never a second settlement implementation -- now also supplied a
    close-line reference (:func:`nfl_ats.clv.live_close_reference`) so the
    close grade is real whenever a resolvable close exists, not merely
    absent by construction. ``note`` is non-empty exactly when one grade,
    or (a week not yet played) neither, could not be computed; the page
    renders that sentence instead of a blank cell.
    """

    season: int
    week: int
    picks: int
    opener_settled: int
    opener_wins: int
    opener_accuracy: float | None
    close_settled: int
    close_wins: int
    close_accuracy: float | None
    note: str = ""

    @property
    def delta(self) -> float | None:
        if self.opener_accuracy is None or self.close_accuracy is None:
            return None
        return self.opener_accuracy - self.close_accuracy

    @property
    def opener_record_text(self) -> str:
        if self.opener_accuracy is None:
            return "--"
        losses = self.opener_settled - self.opener_wins
        return f"{self.opener_wins}-{losses} ({self.opener_accuracy:.1%})"

    @property
    def close_record_text(self) -> str:
        if self.close_accuracy is None:
            return "--"
        losses = self.close_settled - self.close_wins
        return f"{self.close_wins}-{losses} ({self.close_accuracy:.1%})"

    @property
    def delta_text(self) -> str:
        delta = self.delta
        return f"{delta:+.1%}" if delta is not None else "--"


@dataclass(frozen=True)
class HistoryPageContent:
    generated_at_text: str
    picks: tuple[HistoryPickRow, ...]
    primary_available: bool
    primary_error: str | None
    challenger_assessments: tuple[ChallengerAssessment, ...]
    ticker_chrome: TickerChrome
    link_preview: LinkPreview
    season_grades: tuple[SeasonGradeRow, ...] = ()
    week_grades: tuple[HistoryWeekGrade, ...] = ()
    grade_caption: str = ""
    headline: HeadlineStats | None = None


@dataclass(frozen=True)
class SeasonRecordHeadline(HeadlineStats):
    """Shared headline with the current History record, assembled without new I/O."""

    season_record_text: str = ""


def headline_with_season_record(
    headline: HeadlineStats, history: HistoryPageContent
) -> SeasonRecordHeadline:
    """Summarize History's decision-line rows; pushes never enter the win rate."""

    season = history.ticker_chrome.season
    rows = tuple(row for row in history.picks if row.season == season)
    resolved = tuple(
        row
        for row in rows
        if row.status == "push" or (row.status == "settled" and row.correct is not None)
    )
    if history.primary_error:
        text = "This season's settled record is unavailable right now."
    elif not resolved:
        text = f"No picks have settled in {season} yet."
    else:
        wins = sum(row.status == "settled" and row.correct is True for row in resolved)
        losses = sum(row.status == "settled" and row.correct is False for row in resolved)
        pushes = sum(row.status == "push" for row in resolved)
        weeks = {row.week for row in rows if row.week is not None}
        complete = sum(all(row in resolved for row in rows if row.week == week) for week in weeks)
        text = (
            f"In {season}, the played card is {wins}-{losses}-{pushes} (wins-losses-pushes). "
            f"{complete} {'week' if complete == 1 else 'weeks'} fully settled. "
        )
        if wins + losses:
            text += f"The played card's rate is {wins / (wins + losses):.1%}, excluding pushes."
        else:
            text += "There is no win rate yet because every settled pick was a push."
        if len(resolved) < len(rows):
            text += " Unfinished picks are not counted."
    return SeasonRecordHeadline(
        **{item.name: getattr(headline, item.name) for item in fields(HeadlineStats)},
        season_record_text=text,
    )


def _ledger_row_view(row: LedgerRow) -> ModelLedgerRowView:
    track = row.track_record
    agreement_text = None
    if row.agreement is not None:
        agreement_text = (
            f"Agrees with the promoted card on {row.agreement.agree} of "
            f"{row.agreement.vs_promoted_games} games this week "
            f"({row.agreement.disagree} disagree)"
        )
    return ModelLedgerRowView(
        arm_id=row.arm_id,
        display_name=row.display_name,
        status_badge=row.status_badge,
        is_promoted=row.status_badge == STATUS_BADGE_PROMOTED,
        games=track.games if track else None,
        accuracy=track.accuracy if track else None,
        interval_low=track.interval_low if track else None,
        interval_high=track.interval_high if track else None,
        interval_unit=track.interval_unit if track else "accuracy_points",
        grade=track.grade if track else "",
        summary_sentence=row.summary_sentence,
        own_probability_positive=row.own_probability_positive,
        evidence=tuple(
            LedgerEvidenceItem(ref.registry_key, ref.probability_positive, ref.classification)
            for ref in row.evidence
        ),
        agreement_text=agreement_text,
        artifact_ref=track.artifact_ref if track else None,
    )


def _family_row_view(family: FamilyExplanation) -> FamilyWeightRow:
    return FamilyWeightRow(
        label=family.label,
        margin_share=family.margin_share,
        spread_share=family.spread_share,
        weight_in_spread=family.weight_in_spread,
        stability_word=family.stability_word,
        stability_detail=family.stability_detail,
        classification=family.classification,
        caption=family.caption,
    )


def _grading_rule_view(opener_metadata: Mapping[str, Any]) -> GradingRuleView:
    metrics = opener_metadata.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    return GradingRuleView(
        protocol_opener=_number(metrics.get("opener_accuracy")),
        protocol_close=_number(metrics.get("close_accuracy")),
        production_opener=_number(metrics.get("opener_accuracy_probability_rule")),
        production_close=_number(metrics.get("close_accuracy_probability_rule")),
    )


def _season_rows(seasons: pd.DataFrame) -> tuple[SeasonRowView, ...]:
    if seasons.empty or not {"season", "opener_accuracy"}.issubset(seasons.columns):
        return ()
    use_production = "opener_accuracy_probability_rule" in seasons.columns
    opener_column = "opener_accuracy_probability_rule" if use_production else "opener_accuracy"
    close_column = "close_accuracy_probability_rule" if use_production else "close_accuracy"

    rows = []
    for _, row in seasons.iterrows():
        opener_value = _number(row.get(opener_column))
        if opener_value is None:
            continue
        games_value = _number(row.get("games"))
        rows.append(
            SeasonRowView(
                season=str(row["season"]).split(".")[0],
                games=int(games_value) if games_value is not None else None,
                opener_accuracy=opener_value,
                close_accuracy=_number(row.get(close_column)),
            )
        )
    return tuple(rows)


def _evidence_strength(probability_positive: float | None) -> float:
    """How far a P+ sits from a coin flip, either direction -- mirrors
    ``public_board._challenger_evidence_strength`` exactly (a P+ of 0.05 is
    exactly as strong a signal as 0.95, just pointed the other way).
    Unmeasured rows sort last, matching that function's own tie-break."""

    return -1.0 if probability_positive is None else abs(probability_positive - 0.5)


def _grouped_ledger_rows(
    rows: tuple[ModelLedgerRowView, ...],
) -> tuple[tuple[ModelLedgerRowView, ...], tuple[ModelLedgerRowView, ...]]:
    """Split ``rows`` into (graded, waiting) -- owner-approved improvement
    batch, item 9. "Graded" means the arm has a prospective/graded record
    (``track_record.games is not None``, i.e. its own row shows a real
    games/accuracy figure rather than "--"); everything else is "Waiting on
    the season". Each group is sorted by :func:`_evidence_strength`,
    descending, with the promoted row pinned first within "graded"."""

    graded = tuple(
        sorted(
            (row for row in rows if row.games is not None),
            key=lambda row: (
                0 if row.is_promoted else 1,
                -_evidence_strength(row.own_probability_positive),
            ),
        )
    )
    waiting = tuple(
        sorted(
            (row for row in rows if row.games is None),
            key=lambda row: -_evidence_strength(row.own_probability_positive),
        )
    )
    return graded, waiting


def load_model_weak_spots(artifacts_root: Path, active: Mapping[str, Any]) -> WeakSpots:
    """Never fall back to another model's opener record."""
    match = find_matching_opener_evaluation(artifacts_root, active)
    if match is None:
        return WeakSpots()
    try:
        per_game = pd.read_parquet(match[1] / "per_game.parquet")
        spots = build_weak_spots(per_game)
    except (OSError, ValueError, KeyError):
        return WeakSpots()
    return replace(spots, home_correction=_load_home_correction(artifacts_root, active, per_game))


def _load_home_correction(
    artifacts_root: Path, active: Mapping[str, Any], per_game: pd.DataFrame
) -> HomeCorrection | None:
    """This week's served push (from the linked forecast's sidecar, when the
    card was built with it) beside the push's record on the matching opener
    evaluation. ``None`` when that evaluation predates the alignment."""

    from nfl_ats.active_model import active_artifact_path
    from nfl_ats.home_side_location import (
        HOME_SIDE_OFFSET_POLICY,
        load_forecast_home_side_offsets,
    )

    offsets: dict[str, float] | None = None
    prior_games: dict[str, int] | None = None
    try:
        forecast_dir = active_artifact_path(artifacts_root, dict(active), "weekly_forecast")
        sidecar = load_forecast_home_side_offsets(forecast_dir) if forecast_dir else None
    except (OSError, ValueError):
        sidecar = None
    if sidecar and sidecar.get("served"):
        fit = sidecar.get("fit")
        if isinstance(fit, Mapping) and fit.get("policy") != HOME_SIDE_OFFSET_POLICY:
            fit = None
        if isinstance(fit, Mapping):
            raw_offsets = fit.get("offsets")
            raw_prior = fit.get("prior_games")
            if isinstance(raw_offsets, Mapping):
                offsets = {str(k): float(v) for k, v in raw_offsets.items()}
            if isinstance(raw_prior, Mapping):
                prior_games = {str(k): int(v) for k, v in raw_prior.items()}
    try:
        return build_home_correction(per_game, offsets, prior_games)
    except (ValueError, KeyError, TypeError):
        return None


def _load_model_page_content(
    artifacts_root: Path,
    *,
    registry_root: Path | None,
    board: BoardContent,
    opener: OpenerEvaluationArtifacts,
    active: Mapping[str, Any],
    generated_at: datetime,
) -> ModelPageContent:
    """Mirrors the old ``load_model_ledger_html`` / track-record loaders'
    fail-open contracts exactly (see each field's origin below); a registry
    that exists but fails :func:`nfl_ats.model_ledger.validate_ledger` is
    drift the reader should see (``ledger_error`` set, no rows rendered),
    never a raised exception and never rows rendered without validation.
    """

    challengers_path = artifacts_root / "prospective" / "challengers.json"
    active_manifest_path = artifacts_root / "active_ats_model.json"
    rows: tuple[ModelLedgerRowView, ...] = ()
    ledger_available = False
    ledger_error: str | None = None
    if challengers_path.is_file():
        try:
            ledger = build_model_ledger(
                challengers_path, default_registry_path(registry_root), active_manifest_path
            )
            validate_ledger(ledger)
        except (ValueError, OSError) as error:
            ledger_error = str(error) or "unknown error"
        else:
            ledger_available = True
            rows = tuple(_ledger_row_view(row) for row in ledger.rows)

    explanation = load_model_explanation(artifacts_root)
    families = (
        tuple(_family_row_view(family) for family in explanation.families) if explanation else ()
    )

    grading = _grading_rule_view(opener.metadata)
    historical = active.get("historical_evaluation")
    historical = historical if isinstance(historical, Mapping) else {}
    long_run_games_raw = _number(historical.get("games"))
    long_run_correct_raw = _number(historical.get("correct"))
    intervals = historical.get("intervals")
    season_range = intervals.get("season") if isinstance(intervals, Mapping) else None
    long_run_range = None
    if isinstance(season_range, Mapping):
        lower, upper = _number(season_range.get("lower")), _number(season_range.get("upper"))
        if lower is not None and upper is not None:
            long_run_range = (lower, upper)

    season_rows = _season_rows(opener.seasons)
    above = sum(1 for row in season_rows if row.opener_accuracy > 0.5)
    even = tuple(row.season for row in season_rows if row.opener_accuracy == 0.5)
    below = tuple(
        (row.season, row.opener_accuracy) for row in season_rows if row.opener_accuracy < 0.5
    )

    played_chain_accuracy = (
        board.headline.prior_chain_pct / 100.0
        if board.headline.prior_chain_pct is not None
        else None
    )

    graded_rows, waiting_rows = _grouped_ledger_rows(rows) if ledger_available else ((), ())
    number_provenance, number_provenance_note = _number_provenance_rows(artifacts_root, active)

    return ModelPageContent(
        generated_at_text=_generated_at_text(generated_at),
        headline=board.headline,
        ceiling_text=fc.HERO_TILES[1].value,
        ladder_rungs=fc.ladder_rungs(played_chain_accuracy),
        grading=grading,
        long_run_games=int(long_run_games_raw) if long_run_games_raw is not None else None,
        long_run_correct=int(long_run_correct_raw) if long_run_correct_raw is not None else None,
        long_run_range=long_run_range,
        seasons=season_rows,
        seasons_above_coin_flip=above,
        seasons_even=even,
        seasons_below=below,
        ledger_available=ledger_available,
        ledger_error=ledger_error,
        rows=rows,
        graded_rows=graded_rows,
        waiting_rows=waiting_rows,
        explanation_available=explanation is not None,
        run_directory=explanation.run_directory if explanation else None,
        feature_profile=explanation.feature_profile if explanation else None,
        matches_active_feature_table=(
            explanation.matches_active_feature_table if explanation else None
        ),
        families=families,
        ticker_chrome=replace(board.ticker_chrome, page_command_suffix="--page model"),
        link_preview=LinkPreview(
            title="ATS Terminal — The Model",
            description=(
                f"{above} of {len(season_rows)} seasons finished above the coin flip -- the "
                "played policy, its measured record, and every arm tracked against it."
            ),
        ),
        number_provenance=number_provenance,
        number_provenance_note=number_provenance_note,
        weak_spots=load_model_weak_spots(artifacts_root, active),
    )


@dataclass(frozen=True)
class FindingItemView:
    question: str
    verdict: str
    plain_answer: str
    detail: str
    trace_signal_name: str | None = None
    trace_probability_positive: float | None = None


@dataclass(frozen=True)
class VerdictGroupView:
    verdict: str
    kicker: str
    title: str
    blurb: str
    chip_label: str
    findings: tuple[FindingItemView, ...]


@dataclass(frozen=True)
class HeroTileView:
    kicker: str
    value: str
    context: str


@dataclass(frozen=True)
class WatchingLeadView:
    name: str
    description: str
    effect_text: str
    probability_positive: float
    seasons_text: str
    league: str


@dataclass(frozen=True)
class HonestyRuleView:
    title: str
    body: str


@dataclass(frozen=True)
class SignalNotableRow:
    name: str
    idea: str
    effect_text: str
    probability_positive: float
    status: str


@dataclass(frozen=True)
class SignalLedgerSummary:
    """A dense, secondary summary of the weak-signal registry -- NOT the
    full per-signal table the old standalone Signal Ledger page showed
    (610+ rows was its own page's worth of content, and this project's
    binding rule is that an unresolved-below-power signal is never treated
    as settled just because it is not individually surfaced here; the full
    registry stays queryable via ``nfl-ats weak-signals ...``, this is a
    pointer, not a replacement)."""

    total_signals: int
    counts_by_status: Mapping[str, int]
    counts_by_category: Mapping[str, int]
    notable: tuple[SignalNotableRow, ...]


_NOTABLE_SIGNAL_LIMIT = 8


@dataclass(frozen=True)
class RecentActivityEntryView:
    """One line of "Research this week" (dashboard queue, ROADMAP.md
    UI-20(b)): a formatted, ready-to-render row over
    :class:`nfl_ats.findings_registry.RecentActivityEntry`."""

    plain_summary: str
    effect_text: str
    direction_sentence: str
    closed_label: str | None


@dataclass(frozen=True)
class RecentActivityCategoryView:
    category: str
    entries: tuple[RecentActivityEntryView, ...]


@dataclass(frozen=True)
class RecentActivityView:
    """:class:`FindingsPageContent`'s "Research this week" section --
    everything recorded or screened in the activity window, grouped by
    category. Renders correctly when ``categories`` is empty: the header
    counts are still ``0``/``0`` and the page shows "no new screens
    recorded this week" rather than an empty section."""

    window_days: int
    screened_count: int
    resolved_count: int
    categories: tuple[RecentActivityCategoryView, ...]

    @property
    def is_empty(self) -> bool:
        return self.screened_count == 0


PLAIN_SUMMARY_PENDING = "Plain-English summary pending."

_RECENT_ACTIVITY_EFFECT_UNIT_WORDS: dict[str, str] = {
    "accuracy_points": "accuracy points",
    "ats_points": "line points",
    "brier": "Brier-score points",
    "brier_improvement": "Brier-score points of improvement",
    "log_loss": "log-loss points",
    "log_loss_improvement": "log-loss points of improvement",
    "mae": "points of average error",
    "mae_improvement": "points of average-error improvement",
    "correlation": "correlation",
}


def _recent_activity_entry_view(entry: RecentActivityEntry) -> RecentActivityEntryView:
    unit_words = _RECENT_ACTIVITY_EFFECT_UNIT_WORDS.get(
        entry.effect_units or "", entry.effect_units
    )
    effect_text = (
        f"{entry.effect:+.2f} {unit_words}".strip()
        if entry.effect is not None
        else "not yet measured"
    )
    return RecentActivityEntryView(
        plain_summary=entry.plain_summary or PLAIN_SUMMARY_PENDING,
        effect_text=effect_text,
        direction_sentence=entry.direction_sentence or "No confidence figure recorded yet.",
        closed_label=entry.closed_label,
    )


def _recent_activity_view(activity: RecentRegistryActivity) -> RecentActivityView:
    return RecentActivityView(
        window_days=activity.window_days,
        screened_count=activity.screened_count,
        resolved_count=activity.resolved_count,
        categories=tuple(
            RecentActivityCategoryView(
                category=category,
                entries=tuple(_recent_activity_entry_view(entry) for entry in entries),
            )
            for category, entries in activity.entries_by_category
        ),
    )


@dataclass(frozen=True)
class FindingsPageContent:
    generated_at_text: str
    hero_tiles: tuple[HeroTileView, ...]
    groups: tuple[VerdictGroupView, ...]
    watching_leads: tuple[WatchingLeadView, ...]
    recent_activity: RecentActivityView
    honesty_rules: tuple[HonestyRuleView, ...]
    ledger_summary: SignalLedgerSummary
    ticker_chrome: TickerChrome
    link_preview: LinkPreview


def _history_forecast_probability(artifacts_root: Path, row: Mapping[Any, Any]) -> float | None:
    """Read chosen-side probability from the row's linked forecast, if any.

    ``decisions.parquet`` intentionally has no probability column.  Following
    ``forecast_artifact`` keeps History honest for old and new ledger schemas;
    edge, odds, and the active model's historical accuracy are not substitutes
    for a game-specific probability.
    """

    direct = _number(row.get("pick_probability"))
    if direct is not None and 0.0 <= direct <= 1.0:
        return direct
    home_direct = _number(row.get("home_cover_probability"))
    pick_side = str(row.get("pick_side") or "").upper()
    if home_direct is not None and 0.0 <= home_direct <= 1.0:
        return home_direct if pick_side == "HOME" else 1.0 - home_direct
    artifact = row.get("forecast_artifact")
    if artifact is None:
        return None
    raw = Path(str(artifact))
    candidates = [raw if raw.is_absolute() else artifacts_root / raw]
    if raw.name == "predictions.csv":
        candidates.append(artifacts_root / "margin_predictions" / raw.parent.name / raw.name)
    else:
        candidates.extend(
            [
                artifacts_root / "margin_predictions" / raw / "predictions.csv",
                artifacts_root / "margin_predictions" / raw / "recommendations.csv",
            ]
        )
    for candidate in candidates:
        path = candidate if candidate.suffix == ".csv" else candidate / "predictions.csv"
        if not path.is_file():
            continue
        card_path = path.parent / "pool_card.csv"
        if card_path.is_file():
            try:
                card = pd.read_csv(card_path, usecols=["game_id", "pool_side", "pick_probability"])
            except (OSError, ValueError):
                card = pd.DataFrame()
            if not card.empty:
                hit = card.loc[card["game_id"].astype(str).eq(str(row.get("game_id")))]
                if not hit.empty and str(hit.iloc[0]["pool_side"]) == pick_side:
                    picked = _number(hit.iloc[0]["pick_probability"])
                    if picked is not None and 0.0 <= picked <= 1.0:
                        return picked
        try:
            table = pd.read_csv(
                path,
                usecols=lambda column: (
                    column in {"game_id", "home_cover_probability", "pick_probability"}
                ),
            )
        except (OSError, ValueError):
            continue
        if "game_id" not in table.columns:
            continue
        matches = table.loc[table["game_id"].astype(str).eq(str(row.get("game_id")))]
        if matches.empty:
            continue
        found = matches.iloc[0]
        picked = _number(found.get("pick_probability"))
        if picked is not None and 0.0 <= picked <= 1.0:
            return picked
        home = _number(found.get("home_cover_probability"))
        if home is not None and 0.0 <= home <= 1.0:
            return home if pick_side == "HOME" else 1.0 - home
    return None


def _history_score_text(outcomes: pd.DataFrame, game_id: str) -> str | None:
    if outcomes.empty or "game_id" not in outcomes.columns:
        return None
    rows = outcomes.loc[outcomes["game_id"].astype(str).eq(game_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    home_score, away_score = _number(row.get("home_score")), _number(row.get("away_score"))
    if home_score is None or away_score is None:
        return None
    return f"{int(away_score)} at {int(home_score)}"


def _frozen_confidence(frozen: Mapping[str, FrozenPick], row: pd.Series) -> float | None:
    pick = frozen.get(str(row.get("game_id")))
    if pick is None:
        return None
    team = row.get("home_team") if str(row.get("pick_side")) == "HOME" else row.get("away_team")
    return pick.displayed_score if pick.pick_team == str(team) else None


def _history_pick_rows(
    artifacts_root: Path,
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> tuple[HistoryPickRow, ...]:
    if decisions.empty:
        return ()
    settled = settle_prospective_picks(decisions, outcomes)
    frozen = frozen_picks(artifacts_root, now=now or datetime.now(UTC), include_open=True)
    rows: list[HistoryPickRow] = []
    for _, row in settled.iterrows():
        status = str(row.get(f"status_at_{DECISION_GRADE}") or "pending")
        season_value = _number(row.get("season"))
        week_value = _number(row.get("week"))
        if status == "settled":
            correct_value = _number(row.get(f"correct_at_{DECISION_GRADE}"))
            correct = bool(correct_value == 1.0) if correct_value is not None else None
        else:
            correct = None
        rows.append(
            HistoryPickRow(
                game_id=str(row.get("game_id")),
                season=int(season_value) if season_value is not None else None,
                week=int(week_value) if week_value is not None else None,
                away_team=str(row.get("away_team") or "--"),
                home_team=str(row.get("home_team") or "--"),
                pick_side=str(row.get("pick_side") or "--"),
                decision_home_spread=_number(row.get("decision_home_spread")),
                confidence=(
                    frozen_score
                    if (frozen_score := _frozen_confidence(frozen, row)) is not None
                    else _history_forecast_probability(artifacts_root, row.to_dict())
                ),
                model_id=(str(row.get("model_id")) if row.get("model_id") is not None else None),
                best_pick=bool(row.get("is_best_pick", False)),
                status=status,
                correct=correct,
                score_text=(
                    _history_score_text(outcomes, str(row.get("game_id")))
                    if status in {"settled", "push"}
                    else None
                ),
            )
        )
    return tuple(rows)


_CLOSE_SCHEDULE_COLUMNS: tuple[str, ...] = ("game_id", "spread_line", "result")


def _load_close_schedule(data_root: Path) -> pd.DataFrame:
    """UI-20(h): read-only source for the History page's close-line
    reference, from the SAME feature table :func:`nfl_ats.board_content
    ._load_game_outcomes` already reads (``data/processed/
    game_features.parquet``), whose ``spread_line`` this repo already
    treats as the closing number (:func:`nfl_ats.clv.live_close_reference`'s
    own fallback names it ``schedule_close``; see also
    ``nfl_ats.tiebreaker``'s module docstring on the same convention).

    Kept as its own small reader rather than widening
    ``_load_game_outcomes``'s shared column list, so this page's addition
    cannot change any other page's behaviour. Fail-open to an empty frame
    with the right columns -- matching every other optional artifact on
    this page -- never raises.
    """

    path = data_root / "processed" / "game_features.parquet"
    try:
        table = pd.read_parquet(path, columns=list(_CLOSE_SCHEDULE_COLUMNS))
    except (OSError, ValueError):
        return pd.DataFrame(columns=_CLOSE_SCHEDULE_COLUMNS)
    if not set(_CLOSE_SCHEDULE_COLUMNS).issubset(table.columns):
        return pd.DataFrame(columns=_CLOSE_SCHEDULE_COLUMNS)
    return table


def _season_grade_rows(
    seasons: tuple[SeasonRowView, ...], active: Mapping[str, Any]
) -> tuple[SeasonGradeRow, ...]:
    """UI-20(h): the Model page's own per-season opener/close pair
    (:func:`_season_rows`), reused verbatim, plus one explicit, dynamically
    computed row for the seasons the archive itself does not reach.

    ``opener_evaluation``'s population requires a paired opener+close line
    (``docs/opener_evaluation.md``: "2020-2025 historical snapshot
    archive"), so every season it returns already carries both grades; a
    season with no archived opener line is not represented by a row with a
    blank cell -- it is simply ABSENT from ``seasons``. Measured 2026-09-05:
    the archive's population sums to 1,537 games while
    ``active["historical_evaluation"]["games"]`` (the model's own long-run
    close-graded evaluation, ``docs/opener_evaluation.md``'s wider
    chronological population) covers 2,075 -- a 538-game gap with no
    opener-vs-close row of its own. Rather than lose that gap silently, one
    synthetic row is computed here (games = the live difference between the
    two counts, never a hardcoded figure, since both totals grow over
    seasons) and shown first, with an explicit not-archived note instead of
    a blank. Omitted entirely when the two counts already agree (a future
    session that closes the gap) or no historical total is available.
    """

    rows = [
        SeasonGradeRow(
            season_label=row.season,
            games=row.games,
            opener_accuracy=row.opener_accuracy,
            close_accuracy=row.close_accuracy,
        )
        for row in seasons
    ]
    historical = active.get("historical_evaluation")
    historical = historical if isinstance(historical, Mapping) else {}
    total_games = _number(historical.get("games"))
    archived_games = sum(row.games or 0 for row in seasons)
    if total_games is not None and total_games > archived_games:
        rows.insert(
            0,
            SeasonGradeRow(
                season_label="Before the opener archive",
                games=int(total_games) - archived_games,
                opener_accuracy=None,
                close_accuracy=None,
                note=NO_OPENER_LINE_ARCHIVED_SEASON_NOTE,
            ),
        )
    return tuple(rows)


def _history_week_grades(
    decisions: pd.DataFrame, outcomes: pd.DataFrame, close_reference: pd.DataFrame
) -> tuple[HistoryWeekGrade, ...]:
    """UI-20(h): per-week opener-vs-close grading for the primary
    paper-decision ledger.

    Settled through the SAME :func:`nfl_ats.prospective_scoring
    .settle_prospective_picks` :func:`_history_pick_rows` already uses --
    never a second settlement implementation -- with a close-line
    reference now also supplied, so a week's close record is real whenever
    a resolvable close exists for its games rather than absent by
    construction. Empty until the first Tuesday lock records a pick, same
    as :func:`_history_pick_rows`.
    """

    if decisions.empty:
        return ()
    settled = settle_prospective_picks(decisions, outcomes, close_reference=close_reference)
    return _week_grades_from_graded_games(
        settled,
        opener_column=f"correct_at_{DECISION_GRADE}",
        close_column=f"correct_at_{CLOSE_GRADE}",
    )


def _week_grades_from_graded_games(
    graded: pd.DataFrame, *, opener_column: str, close_column: str
) -> tuple[HistoryWeekGrade, ...]:
    """One :class:`HistoryWeekGrade` per ``(season, week)`` of ``graded``.

    The single place a week's opener/close record is counted, shared by the
    recorded paper ledger (:func:`_history_week_grades`) and the archived
    replay (:func:`_archive_week_grades`) so the two can never count a week
    differently.  Both callers hand it a frame of already-graded games whose
    two correctness columns are 1 (right), 0 (wrong) or missing (a push, or
    a line that does not exist for that game).
    """

    if graded.empty or not {"season", "week"}.issubset(graded.columns):
        return ()
    if not {opener_column, close_column}.issubset(graded.columns):
        return ()
    rows: list[HistoryWeekGrade] = []
    for (season, week), group in graded.groupby(["season", "week"], sort=True):
        opener_correct = pd.to_numeric(group[opener_column], errors="coerce")
        close_correct = pd.to_numeric(group[close_column], errors="coerce")
        opener_resolved = opener_correct.dropna()
        close_resolved = close_correct.dropna()
        opener_settled, close_settled = len(opener_resolved), len(close_resolved)
        opener_wins, close_wins = int(opener_resolved.sum()), int(close_resolved.sum())
        opener_accuracy = float(opener_resolved.mean()) if opener_settled else None
        close_accuracy = float(close_resolved.mean()) if close_settled else None
        if opener_settled == 0 and close_settled == 0:
            note = HISTORY_WEEK_NOT_SETTLED_NOTE
        elif opener_settled == 0:
            note = NO_OPENER_LINE_ARCHIVED_WEEK_NOTE
        elif close_settled == 0:
            note = NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE
        else:
            note = ""
        rows.append(
            HistoryWeekGrade(
                season=int(str(season)),
                week=int(str(week)),
                picks=len(group),
                opener_settled=opener_settled,
                opener_wins=opener_wins,
                opener_accuracy=opener_accuracy,
                close_settled=close_settled,
                close_wins=close_wins,
                close_accuracy=close_accuracy,
                note=note,
            )
        )
    return tuple(rows)


def _archive_week_grades(
    artifacts_root: Path, active: Mapping[str, Any]
) -> tuple[HistoryWeekGrade, ...]:
    """Every finished week of the opener archive, graded at both lines.

    Why this exists (UI-20(h), second pass): the recorded paper ledger only
    starts at the first Tuesday lock, so the per-week table showed exactly
    one unsettled row while the season table above it carried six seasons
    of real opener-vs-close pairs.  A reader could see the difference the
    pool's own line makes over a season but never week to week -- which is
    the whole point of the section.

    The rows come from the SAME evaluation run the season table is built
    from -- :func:`find_matching_opener_evaluation` keyed to the ACTIVE
    model, exactly as :func:`load_model_weak_spots` does -- reading its
    per-game grades and counting them by week.  Summing a season's weeks
    reproduces that season's row above it exactly, by construction.  The
    served columns are used (the ones behind every accuracy the site
    already shows), never the unadjusted twins.  No matching run means no
    rows at all: a week is never graded with another model's picks.
    """

    match = find_matching_opener_evaluation(artifacts_root, active)
    if match is None:
        return ()
    try:
        per_game = pd.read_parquet(match[1] / "per_game.parquet")
    except (OSError, ValueError):
        return ()
    return _week_grades_from_graded_games(
        per_game,
        opener_column=ARCHIVE_OPENER_CORRECT_COLUMN,
        close_column=ARCHIVE_CLOSE_CORRECT_COLUMN,
    )


def _combined_week_grades(
    recorded: tuple[HistoryWeekGrade, ...], archived: tuple[HistoryWeekGrade, ...]
) -> tuple[HistoryWeekGrade, ...]:
    """Recorded weeks and archived weeks in one newest-first table.

    A week recorded in the paper ledger always wins its ``(season, week)``
    slot: those picks were locked in before kickoff, so they are the honest
    record, while the archive's row for the same week is a replay of it.
    """

    merged = {(row.season, row.week): row for row in archived}
    merged.update({(row.season, row.week): row for row in recorded})
    return tuple(sorted(merged.values(), key=lambda row: (row.season, row.week), reverse=True))


def _evidence_values(entry: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    evidence = entry.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    probability = _number(evidence.get("probability_positive"))
    interval = evidence.get("week_blocked_interval_points")
    if (
        not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) < 2
    ):
        interval = evidence.get("interval_points")
    if (
        not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) < 2
    ):
        return probability, None, None
    return probability, _number(interval[0]), _number(interval[1])


def _latest_prospective_reports(artifacts_root: Path) -> dict[str, Mapping[str, Any]]:
    """Return entrant reports from the newest prospective-score artifact."""

    directories = artifact_directories(artifacts_root / "prospective_scoring", "metadata.json")
    for directory in directories:
        try:
            payload = read_json(directory / "metadata.json")
        except (ValueError, OSError):
            continue
        entrants = payload.get("entrants")
        if not isinstance(entrants, list):
            continue
        return {
            str(entry.get("entrant")): entry
            for entry in entrants
            if isinstance(entry, Mapping) and entry.get("entrant") is not None
        }
    return {}


def _prospective_report_grade(
    report: Mapping[str, Any], *, paired_games: int
) -> tuple[float | None, float | None, float | None] | None:
    """Read uncertainty only from a report matching the settled paired grade.

    Prospective-score metadata can contain pending-only rows and uncertainty
    for several metrics.  Registration evidence is historical context, not a
    running score, so a report is eligible only when its decision-line game
    count matches the settled paired comparison and its uncertainty is for
    that decision-line accuracy metric.
    """

    forced = report.get("forced_picks")
    decision = forced.get(DECISION_GRADE) if isinstance(forced, Mapping) else None
    games = _number(decision.get("games")) if isinstance(decision, Mapping) else None
    if paired_games <= 0 or games is None or int(games) != paired_games:
        return None
    report_probability = _number(report.get("probability_positive"))
    uncertainty = report.get("uncertainty")
    if not isinstance(uncertainty, list):
        return (report_probability, None, None) if report_probability is not None else None
    candidates = [
        entry
        for entry in uncertainty
        if isinstance(entry, Mapping) and entry.get("metric") == "decision_line_accuracy"
    ]
    if not candidates:
        unlabeled = [
            entry for entry in uncertainty if isinstance(entry, Mapping) and "metric" not in entry
        ]
        if len(unlabeled) != 1:
            return (report_probability, None, None) if report_probability is not None else None
        candidates = unlabeled
    week_rows = [entry for entry in candidates if entry.get("block") == "week"]
    selected = week_rows[-1] if week_rows else candidates[-1]
    return (
        _number(selected.get("probability_positive"))
        if _number(selected.get("probability_positive")) is not None
        else report_probability,
        _number(selected.get("lower")),
        _number(selected.get("upper")),
    )


def _history_challenger_assessments(
    challengers: Sequence[Mapping[str, Any]],
    challenger_decisions: pd.DataFrame,
    active_decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[ChallengerAssessment, ...]:
    if challenger_decisions.empty:
        return ()
    try:
        active_settled = settle_prospective_picks(active_decisions, outcomes)
    except (ValueError, OSError):
        active_settled = pd.DataFrame()
    active_by_game = (
        active_settled.set_index("game_id") if not active_settled.empty else pd.DataFrame()
    )
    registry = {
        str(item.get("challenger_id")): item
        for item in challengers
        if item.get("challenger_id") is not None
    }
    rows: list[ChallengerAssessment] = []
    for challenger_id, group in challenger_decisions.groupby("challenger_id", sort=True):
        try:
            settled = settle_prospective_picks(group.reset_index(drop=True), outcomes)
        except (ValueError, OSError):
            continue
        status = settled[f"status_at_{DECISION_GRADE}"].astype(str)
        settled_nonpush = settled.loc[status == "settled"]
        correct = pd.to_numeric(
            settled_nonpush[f"correct_at_{DECISION_GRADE}"], errors="coerce"
        ).dropna()
        wins = int((correct == 1.0).sum())
        losses = int((correct == 0.0).sum())
        pushes = int((status == "push").sum())
        pending = int((status == "pending").sum())
        accuracy = float(correct.mean()) if len(correct) else None
        delta: float | None = None
        paired_games = 0
        if not active_settled.empty:
            challenger_index = settled.set_index("game_id")
            common = challenger_index.index.intersection(active_by_game.index)
            candidate = challenger_index.loc[common]
            incumbent = active_by_game.loc[common]
            candidate_status = candidate[f"status_at_{DECISION_GRADE}"].astype(str)
            incumbent_status = incumbent[f"status_at_{DECISION_GRADE}"].astype(str)
            paired = (candidate_status == "settled") & (incumbent_status == "settled")
            candidate_correct = pd.to_numeric(
                candidate.loc[paired, f"correct_at_{DECISION_GRADE}"], errors="coerce"
            )
            incumbent_correct = pd.to_numeric(
                incumbent.loc[paired, f"correct_at_{DECISION_GRADE}"], errors="coerce"
            )
            valid = candidate_correct.notna() & incumbent_correct.notna()
            paired_games = int(valid.sum())
            if paired_games:
                delta = float((candidate_correct[valid] - incumbent_correct[valid]).mean() * 100)
        report_grade = _prospective_report_grade(
            reports.get(str(challenger_id), {}), paired_games=paired_games
        )
        if report_grade is not None:
            probability, low, high = report_grade
        else:
            probability = low = high = None
        if report_grade is None:
            probability, low, high = _evidence_values(registry.get(str(challenger_id), {}))
            if probability is not None or low is not None or high is not None:
                grading_basis = (
                    "Settled prospectively at the frozen decision/opener line; "
                    "pre-registration/historical evidence shown because no matching "
                    "prospective-score report exists."
                )
            else:
                grading_basis = (
                    "Settled prospectively at the frozen decision/opener line; "
                    "no matching prospective-score uncertainty is recorded."
                )
        else:
            grading_basis = (
                "Settled prospectively at the frozen decision/opener line; "
                "paired with active model and sourced from the latest prospective-score "
                "report."
            )
        rows.append(
            ChallengerAssessment(
                challenger_id=str(challenger_id),
                display_name=fc.CHALLENGER_DISPLAY_NAMES.get(
                    str(challenger_id), humanize_identifier(str(challenger_id))
                ),
                paired_games=paired_games,
                wins=wins,
                losses=losses,
                pushes=pushes,
                pending=pending,
                accuracy=accuracy,
                delta_accuracy_points=delta,
                probability_positive=probability,
                interval_low=low,
                interval_high=high,
                grading_basis=grading_basis,
            )
        )
    return tuple(rows)


def _load_history_page_content(
    artifacts_root: Path,
    *,
    data_root: Path,
    challengers: Sequence[Mapping[str, Any]],
    board: BoardContent,
    opener: OpenerEvaluationArtifacts,
    active: Mapping[str, Any],
    generated_at: datetime,
) -> HistoryPageContent:
    primary_error: str | None = None
    primary_available = False
    try:
        from nfl_ats.clv import load_paper_decisions

        primary = load_paper_decisions(artifacts_root)
    except (ValueError, OSError) as error:
        primary = pd.DataFrame()
        primary_error = str(error) or "primary paper ledger unavailable"
    else:
        primary_available = not primary.empty
    outcomes = _load_game_outcomes(data_root, artifacts_root)
    try:
        picks = _history_pick_rows(artifacts_root, primary, outcomes, now=generated_at)
    except (ValueError, OSError) as error:
        picks = ()
        primary_error = primary_error or str(error) or "primary ledger could not be settled"
    try:
        challengers_ledger = load_challenger_decisions(artifacts_root)
    except (ValueError, OSError) as error:
        challengers_ledger = pd.DataFrame()
        primary_error = primary_error or str(error) or "challenger ledger unavailable"
    assessments = _history_challenger_assessments(
        challengers,
        challengers_ledger,
        primary,
        outcomes,
        _latest_prospective_reports(artifacts_root),
    )
    close_schedule = _load_close_schedule(data_root)
    close_reference = pd.DataFrame()
    if not close_schedule.empty:
        try:
            close_reference = live_close_reference(data_root, close_schedule, as_of=generated_at)
        except (DataContractError, ValueError, OSError):
            close_reference = pd.DataFrame()
    try:
        recorded_week_grades = _history_week_grades(primary, outcomes, close_reference)
    except (ValueError, OSError) as error:
        recorded_week_grades = ()
        primary_error = primary_error or str(error) or "primary ledger could not be settled"
    archived_week_grades = _archive_week_grades(artifacts_root, active)
    week_grades = _combined_week_grades(recorded_week_grades, archived_week_grades)
    season_grades = _season_grade_rows(_season_rows(opener.seasons), active)
    grade_caption = HISTORY_GRADE_CAPTION if (season_grades or week_grades) else ""
    if archived_week_grades:
        grade_caption = f"{grade_caption} {HISTORY_WEEK_REPLAY_CAPTION}"
    return HistoryPageContent(
        generated_at_text=_generated_at_text(generated_at),
        picks=picks,
        primary_available=primary_available,
        primary_error=primary_error,
        challenger_assessments=assessments,
        ticker_chrome=replace(board.ticker_chrome, page_command_suffix="--page history"),
        link_preview=LinkPreview(
            title="ATS Terminal — History",
            description=(
                "Recorded model picks at their frozen decision line, with settled "
                "outcomes and prospective challenger assessments."
            ),
        ),
        season_grades=season_grades,
        week_grades=week_grades,
        grade_caption=grade_caption,
        headline=board.headline,
    )


def _finding_trace(
    finding: fc.Finding, entries: Mapping[str, RegistryEntry]
) -> tuple[str | None, float | None]:
    """The first of ``finding.registry_keys`` that resolves to a registry
    entry carrying a measured ``probability_positive`` -- see
    :class:`FindingItemView`'s docstring. Curation already guarantees every
    non-evergreen finding's ``registry_keys`` resolve
    (:func:`nfl_ats.findings_registry.validate_curation`, run before this is
    called), so a missing key here just means "no trace chip", never a
    build failure."""

    for key in finding.registry_keys:
        entry = entries.get(key)
        if entry is not None and entry.probability_positive is not None:
            return entry.name, entry.probability_positive
    return None, None


def _group_view(group: fc.VerdictGroup, entries: Mapping[str, RegistryEntry]) -> VerdictGroupView:
    findings = tuple(
        FindingItemView(
            f.question, f.verdict, f.plain_answer, f.detail, *_finding_trace(f, entries)
        )
        for f in fc.findings_for(group.verdict)
    )
    return VerdictGroupView(
        verdict=group.verdict,
        kicker=group.kicker,
        title=group.title,
        blurb=group.blurb,
        chip_label=group.chip_label,
        findings=findings,
    )


def _watching_lead_view(
    lead: WatchingLead, blurbs_by_signal: Mapping[str, fc.LeadBlurb]
) -> WatchingLeadView:
    blurb = blurbs_by_signal.get(lead.name)
    if blurb is not None:
        description = blurb.text
    elif lead.plain_summary:
        description = lead.plain_summary
    else:
        description = PLAIN_SUMMARY_PENDING
    unit_words = _RECENT_ACTIVITY_EFFECT_UNIT_WORDS.get(lead.effect_units, lead.effect_units)
    return WatchingLeadView(
        name=lead.name,
        description=description,
        effect_text=f"{lead.effect:+.2f} {unit_words}",
        probability_positive=lead.probability_positive,
        seasons_text=f"{lead.seasons[0]}-{lead.seasons[1]}",
        league=lead.league,
    )


def _load_signal_ledger_summary(registry_root: Path | None) -> SignalLedgerSummary:
    registry = load_weak_signal_registry(registry_root)
    raw_rows = build_ledger_rows(registry)
    counts_status: dict[str, int] = {}
    counts_category: dict[str, int] = {}
    candidates: list[tuple[float, SignalNotableRow]] = []
    for row in raw_rows:
        status = str(row.get("status", ""))
        category = str(row.get("category", ""))
        counts_status[status] = counts_status.get(status, 0) + 1
        counts_category[category] = counts_category.get(category, 0) + 1
        probability_positive = _number(row.get("pp"))
        if probability_positive is None:
            continue
        digits = int(row.get("digits") or 2)
        raw_unit_words = str(row.get("unit_words") or "")
        unit_words = _RECENT_ACTIVITY_EFFECT_UNIT_WORDS.get(raw_unit_words, raw_unit_words)
        effect = row.get("effect")
        effect_text = (
            f"{effect:+.{digits}f} {unit_words}".strip()
            if isinstance(effect, int | float)
            else "--"
        )
        idea = (
            PLAIN_SUMMARY_PENDING
            if row.get("fallback")
            else html.unescape(str(row.get("idea", "")))
        )
        candidates.append(
            (
                probability_positive,
                SignalNotableRow(
                    name=html.unescape(str(row.get("name", ""))),
                    idea=idea,
                    effect_text=effect_text,
                    probability_positive=probability_positive,
                    status=status,
                ),
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    notable = tuple(row for _, row in candidates[:_NOTABLE_SIGNAL_LIMIT])
    return SignalLedgerSummary(
        total_signals=len(registry.signals),
        counts_by_status=counts_status,
        counts_by_category=counts_category,
        notable=notable,
    )


def _load_findings_content(
    challengers: Sequence[Mapping[str, Any]],
    *,
    registry_root: Path | None,
    generated_at: datetime,
    board: BoardContent,
) -> FindingsPageContent:
    """Mirrors ``render_findings_page`` item-for-item: the SAME curated
    ``FINDINGS``/``LEAD_BLURBS`` validated against the SAME live registries
    (:func:`nfl_ats.findings_registry.validate_curation`, a REQUIRED guard --
    a stale or drifted claim must raise here exactly as it would in the
    original page, never render quietly), then the SAME auto-rendered
    "what we're watching" leads (:func:`nfl_ats.findings_registry.top_open_leads`).

    ``challengers`` is still needed here even though this page no longer
    renders its own tracked-challenger cards (the model ledger on The Model
    page already shows them, with richer per-arm evidence): it is an input
    to :func:`nfl_ats.findings_registry.load_all_entries`, which
    ``validate_curation`` checks the curated prose against.
    """

    registry = load_weak_signal_registry(registry_root)
    entries = load_all_entries(
        registry_root=registry_root, weak_signal_registry=registry, challengers=challengers
    )
    validate_curation(fc.FINDINGS, entries)
    validate_curation(fc.LEAD_BLURBS, entries)

    leads = top_open_leads(registry)
    blurbs_by_signal = {blurb.weak_signal_name: blurb for blurb in fc.LEAD_BLURBS}
    ledger_summary = _load_signal_ledger_summary(registry_root)
    rotation_registry = load_rotation_registry(registry_root)
    recent_activity = _recent_activity_view(
        recent_registry_activity(registry, rotation_registry, generated_at)
    )
    return FindingsPageContent(
        generated_at_text=_generated_at_text(generated_at),
        hero_tiles=tuple(
            HeroTileView(t.kicker, t.value, t.context)
            for t in fc.baseline_hero_tiles(
                board.headline.raw_model_value_text, board.headline.raw_model_caption
            )
        ),
        groups=tuple(_group_view(group, entries) for group in fc.GROUPS),
        watching_leads=tuple(_watching_lead_view(lead, blurbs_by_signal) for lead in leads),
        recent_activity=recent_activity,
        honesty_rules=tuple(HonestyRuleView(r.title, r.body) for r in fc.HONESTY_RULES),
        ledger_summary=ledger_summary,
        ticker_chrome=replace(board.ticker_chrome, page_command_suffix="--page findings"),
        link_preview=LinkPreview(
            title="ATS Terminal — What We've Learned",
            description=(
                f"{ledger_summary.total_signals} recorded signals, grouped by verdict -- "
                "curated findings, open leads, and the honesty rules that keep them straight."
            ),
        ),
    )


@dataclass(frozen=True)
class SiteContent:
    """Every page's content, loaded once. ``board`` is the SAME
    :class:`~nfl_ats.board_content.BoardContent` :mod:`nfl_ats.board_content`
    already builds for the This Week page; ``model``/``findings`` are built
    here, including the ledger-backed History page."""

    board: BoardContent
    model: ModelPageContent
    history: HistoryPageContent
    findings: FindingsPageContent


def load_site_content(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    registry_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> SiteContent:
    """Load every page's content for the four-page site.

    ``require_fresh_arrest_overlay`` defaults to ``True`` here (unlike
    ``board_content.load_board_content``'s rehearsal-friendly ``False``)
    because this is the function the REAL publish path
    (``cli._write_public_site`` -> ``board_site.build_site``) calls --
    matching ``public_board.build_public_site``'s own default. A scratch or
    rehearsal build should pass ``False`` explicitly.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    resolved_data_root = data_root if data_root is not None else _default_data_root()

    board = load_board_content(
        artifacts_root,
        data_root=resolved_data_root,
        generated_at=generated,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
    )

    artifacts = load_public_board_artifacts(artifacts_root)
    opener = load_opener_evaluation_artifacts(
        artifacts_root, active_feature_profile=artifacts.active.get("feature_profile")
    )
    challengers = load_prospective_challengers(artifacts_root)

    model = _load_model_page_content(
        artifacts_root,
        registry_root=registry_root,
        board=board,
        opener=opener,
        active=artifacts.active,
        generated_at=generated,
    )
    history = _load_history_page_content(
        artifacts_root,
        data_root=resolved_data_root,
        challengers=challengers,
        board=board,
        opener=opener,
        active=artifacts.active,
        generated_at=generated,
    )
    findings = _load_findings_content(
        challengers, registry_root=registry_root, generated_at=generated, board=board
    )

    headline = headline_with_season_record(board.headline, history)
    board = replace(board, headline=headline)
    model = replace(model, headline=headline)
    history = replace(history, headline=headline)
    return SiteContent(board=board, model=model, history=history, findings=findings)


__all__ = [
    "ARCHIVE_CLOSE_CORRECT_COLUMN",
    "ARCHIVE_OPENER_CORRECT_COLUMN",
    "HISTORY_GRADE_CAPTION",
    "HISTORY_WEEK_NOT_SETTLED_NOTE",
    "HISTORY_WEEK_REPLAY_CAPTION",
    "NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE",
    "NO_OPENER_LINE_ARCHIVED_SEASON_NOTE",
    "NO_OPENER_LINE_ARCHIVED_WEEK_NOTE",
    "PLAIN_SUMMARY_PENDING",
    "ChallengerAssessment",
    "FamilyWeightRow",
    "FindingItemView",
    "FindingsPageContent",
    "GradingRuleView",
    "HeroTileView",
    "HistoryPageContent",
    "HistoryPickRow",
    "HistoryWeekGrade",
    "HonestyRuleView",
    "LedgerEvidenceItem",
    "ModelLedgerRowView",
    "ModelPageContent",
    "SeasonGradeRow",
    "SeasonRowView",
    "SignalLedgerSummary",
    "SignalNotableRow",
    "SiteContent",
    "VerdictGroupView",
    "WatchingLeadView",
    "load_site_content",
]
