"""Shared view models for the ATS Terminal site's other two pages: The Model
(``model.html``) and What We've Learned (``findings.html``). ``index.html``
(This Week) is :mod:`nfl_ats.board_content`'s own ``BoardContent``.

2026-08-31 owner redirect: the site is exactly THREE pages, dedup'd so every
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.board_content import (
    BoardContent,
    HeadlineStats,
    LinkPreview,
    TickerChrome,
    load_board_content,
)
from nfl_ats.dashboard import findings_content as fc
from nfl_ats.findings_registry import (
    RegistryEntry,
    WatchingLead,
    load_all_entries,
    load_weak_signal_registry,
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
from nfl_ats.public_board import (
    OpenerEvaluationArtifacts,
    load_opener_evaluation_artifacts,
    load_prospective_challengers,
    load_public_board_artifacts,
)
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


# ---------------------------------------------------------------------------
# The Model page -- merges what used to be the Models page (the model
# ledger: promoted card + every registered challenger, plus the family-
# weight explanation) and the Track Record page (season-by-season history,
# grading-rule comparison, the active model's own season-blocked evaluation
# sample), minus every fact that duplicated the This Week headline strip or
# the ledger's own richer challenger rows -- see the module docstring.
# ---------------------------------------------------------------------------


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
    #: ``"accuracy_rate"`` (a proportion, percent-formatted) or
    #: ``"accuracy_points"`` (a points-effect delta, signed-points-formatted)
    #: -- copied verbatim from :class:`nfl_ats.model_ledger.TrackRecord`'s
    #: own field so the renderer never has to guess a unit from magnitude.
    #: See that field's docstring for the 2026-08-31 browser-QA bug this
    #: guards: a points interval hitting a percent formatter renders e.g.
    #: ``[0.29, 2.038]`` as ``[29.0%, 203.8%]``.
    interval_unit: str
    grade: str
    summary_sentence: str
    own_probability_positive: float | None
    evidence: tuple[LedgerEvidenceItem, ...]
    agreement_text: str | None
    #: The promoted row's own historical-evaluation artifact path, when one
    #: is recorded. ``model_ledger._promoted_row`` never links a promoted
    #: row to a registry key (it is not a prospective challenger citing
    #: outside evidence -- its own track record above IS its evidence), so
    #: its ``evidence`` tuple is always empty; the renderer uses THIS field
    #: to give that row a real provenance line instead of the bare "no
    #: cited evidence" a challenger row would show for a genuine gap.
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

    # "What we play"
    headline: HeadlineStats
    ceiling_text: str
    ladder_rungs: tuple[str, ...]
    grading: GradingRuleView
    long_run_games: int | None
    long_run_correct: int | None
    long_run_range: tuple[float, float] | None

    # "How it's done"
    seasons: tuple[SeasonRowView, ...]
    seasons_above_coin_flip: int
    seasons_even: tuple[str, ...]
    seasons_below: tuple[tuple[str, float], ...]

    # "What's challenging it"
    ledger_available: bool
    ledger_error: str | None
    rows: tuple[ModelLedgerRowView, ...]
    #: ``rows`` split and re-sorted for the grouped ledger (owner-approved
    #: improvement batch, item 9): arms with a prospective/graded record
    #: (``track_record.games is not None``) vs. arms still "Waiting on the
    #: season". Within each group, sorted by evidence strength -- see
    #: :func:`_evidence_strength`, which mirrors
    #: ``public_board._challenger_evidence_strength`` -- with the promoted
    #: row pinned first in ``graded_rows``. Only populated when
    #: ``ledger_available``; empty otherwise.
    graded_rows: tuple[ModelLedgerRowView, ...]
    waiting_rows: tuple[ModelLedgerRowView, ...]
    explanation_available: bool
    run_directory: str | None
    feature_profile: str | None
    matches_active_feature_table: bool | None
    families: tuple[FamilyWeightRow, ...]
    #: Shared ticker + command-row content, reused verbatim from This Week
    #: (owner-approved improvement batch, item 7).
    ticker_chrome: TickerChrome
    link_preview: LinkPreview


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
        # Only meaningful when interval_low/high are set (see above); the
        # default is unused when there is no interval to render.
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

    # "What's challenging it" -- the model ledger.
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

    # "How it's done" -- season-by-season history and the grading-rule
    # comparison.
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

    # "What we play" -- reuse the This Week page's OWN headline object
    # rather than recompute it (see the class docstring); the ladder rungs
    # need the underlying fraction, derived from that same object's own
    # prior-chain percentage rather than a second artifact read.
    played_chain_accuracy = (
        board.headline.prior_chain_pct / 100.0
        if board.headline.prior_chain_pct is not None
        else None
    )

    graded_rows, waiting_rows = _grouped_ledger_rows(rows) if ledger_available else ((), ())

    return ModelPageContent(
        generated_at_text=_generated_at_text(generated_at),
        headline=board.headline,
        ceiling_text=fc.HEADLINE.ceiling,
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
    )


# ---------------------------------------------------------------------------
# Findings page -- plain-English findings by verdict, open leads, honesty
# rules, plus ONE dense secondary section summarizing the weak-signal
# registry (2026-08-31: replaces the old standalone Signal Ledger page --
# every recorded signal is still visible via ``nfl-ats weak-signals``; this
# section is a pointer into that registry, not a restatement of it).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FindingItemView:
    question: str
    verdict: str
    plain_answer: str
    detail: str
    #: Findings trace chip (owner-approved improvement batch, item 2 --
    #: "the owner hasn't approved this sight-unseen"): the registry signal
    #: this finding traces to, plus its recorded ``probability_positive``.
    #: Both ``None`` when the finding cites no registry key, or when none of
    #: its cited keys carry a measured P+ (e.g. an evergreen finding, or a
    #: rotation family with no scored window yet). Deliberately ISOLATED to
    #: these two fields plus one renderer helper
    #: (``board_terminal._trace_chip_html``) and one delimited CSS block
    #: (``.trace-chip``) so the whole feature can be removed in one commit
    #: if the owner vetoes it after seeing it.
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
    #: The registry's own highest-``probability_positive`` entries, capped
    #: at :data:`_NOTABLE_SIGNAL_LIMIT` -- "notable" means highest recorded
    #: confidence, nothing more; every entry excluded here is still a real,
    #: recorded row in the full registry.
    notable: tuple[SignalNotableRow, ...]


#: Cap on the compact "notable signals" list -- see :class:`SignalLedgerSummary`.
_NOTABLE_SIGNAL_LIMIT = 8


@dataclass(frozen=True)
class FindingsPageContent:
    generated_at_text: str
    hero_tiles: tuple[HeroTileView, ...]
    groups: tuple[VerdictGroupView, ...]
    watching_leads: tuple[WatchingLeadView, ...]
    honesty_rules: tuple[HonestyRuleView, ...]
    ledger_summary: SignalLedgerSummary
    #: Shared ticker + command-row content, reused verbatim from This Week
    #: (owner-approved improvement batch, item 7).
    ticker_chrome: TickerChrome
    link_preview: LinkPreview


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
    description = blurb.text if blurb is not None else lead.description
    return WatchingLeadView(
        name=lead.name,
        description=description,
        effect_text=f"{lead.effect:+.2f} {lead.effect_units}",
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
        unit_words = str(row.get("unit_words") or "")
        effect = row.get("effect")
        effect_text = (
            f"{effect:+.{digits}f} {unit_words}".strip()
            if isinstance(effect, int | float)
            else "--"
        )
        candidates.append(
            (
                probability_positive,
                SignalNotableRow(
                    name=html.unescape(str(row.get("name", ""))),
                    idea=html.unescape(str(row.get("idea", ""))),
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
    return FindingsPageContent(
        generated_at_text=_generated_at_text(generated_at),
        hero_tiles=tuple(HeroTileView(t.kicker, t.value, t.context) for t in fc.HERO_TILES),
        groups=tuple(_group_view(group, entries) for group in fc.GROUPS),
        watching_leads=tuple(_watching_lead_view(lead, blurbs_by_signal) for lead in leads),
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


# ---------------------------------------------------------------------------
# The full-site bundle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteContent:
    """Every page's content, loaded once. ``board`` is the SAME
    :class:`~nfl_ats.board_content.BoardContent` :mod:`nfl_ats.board_content`
    already builds for the This Week page; ``model``/``findings`` are built
    here."""

    board: BoardContent
    model: ModelPageContent
    findings: FindingsPageContent


def load_site_content(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    registry_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> SiteContent:
    """Load every page's content for the three-page site.

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
    findings = _load_findings_content(
        challengers, registry_root=registry_root, generated_at=generated, board=board
    )

    return SiteContent(board=board, model=model, findings=findings)


__all__ = [
    "FamilyWeightRow",
    "FindingItemView",
    "FindingsPageContent",
    "GradingRuleView",
    "HeroTileView",
    "HonestyRuleView",
    "LedgerEvidenceItem",
    "ModelLedgerRowView",
    "ModelPageContent",
    "SeasonRowView",
    "SignalLedgerSummary",
    "SignalNotableRow",
    "SiteContent",
    "VerdictGroupView",
    "WatchingLeadView",
    "load_site_content",
]
