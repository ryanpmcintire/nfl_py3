"""The one view model behind the ATS Terminal site's This Week page.

An owner-approved mockup renders this week's forced-pick board as an "ATS
Terminal" dark dashboard (:mod:`nfl_ats.board_terminal`). This module is the
ONLY place that touches an artifact, a loader, or a piece of prose: every
number, caption, id, and finding the page renders is a field on
:class:`BoardContent`, built ONCE here.

``board_terminal.py`` is a pure renderer -- ``render(content: BoardContent) ->
str`` -- and must never contain a content literal (a number, a sentence, a
policy id). That split exists so a future number change (a new experiment, a
refreshed interval, updated findings) touches this module exactly once and
the page picks it up automatically; a content-coverage test
(``tests/test_board_content_coverage.py``) asserts every fact this module
produces actually appears on the rendered page.

Every field traces to an artifact this repo already writes, using the SAME
loaders and overlay-resolution path
:func:`nfl_ats.public_board.build_public_site` uses for the real site, so
this page's numbers can never disagree with the actually-published card.
Fail-open discipline matches ``public_board.py`` throughout: an optional
artifact that is absent degrades the corresponding field to a designed
"unavailable" state rather than raising or inventing a figure.

Season-mode and prospective data sources (owner-approved improvement batch,
2026-08-31, items 3-4)
--------------------------------------------------------------------------
Two features here read data that does not exist until games are actually
played, and both are designed to be DORMANT (fail open to "nothing to show
yet") rather than raise, exactly like every other optional artifact on this
site:

* **In-season finals** (:func:`_load_game_outcomes`, :func:`_game_final_state`)
  -- read from ``data/processed/game_features.parquet``'s own
  ``home_score``/``away_score``/``result`` columns. Measured directly this
  session: this table already carries all 272 rows of the 2026 regular
  season with every score/result column ``NaN`` for a game that has not
  been played, and is the SAME home-minus-away margin convention (FND-04,
  ``docs/modeling.md``) every settlement in this repo already scores
  against. It is refreshed by the repo's own weekly data pipeline as games
  are played -- no separate "in-season" artifact exists or was found;
  weekly-forecast/predictions artifacts are pregame-only by construction,
  and the CLV/prospective ledgers record PICKS, not results. A missing or
  unreadable file degrades every game to "not final yet", never an
  exception.
* **The prospective scoreboard** (:func:`_build_prospective_scoreboard`) --
  pairs the played four-member policy's own paper-decision ledger
  (``nfl_ats.clv.load_paper_decisions``, filtered to
  ``four_overlay_composition.POLICY_ID``) against its immediate incumbent's
  challenger-ledger rows (``nfl_ats.prospective_scoring
  .load_challenger_decisions``, filtered to
  ``four_overlay_composition.INCUMBENT_CHALLENGER_ID``), settling both
  against the SAME in-season outcomes table above via
  ``nfl_ats.clv.pick_correct`` -- never a second push/win/loss
  implementation. Both ledgers are empty by design until the first Tuesday
  lock (``docs/prospective_evidence.md``); the scoreboard renders a
  designed dormant state ("prospective tracking begins Week 1") until then.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.card_view import resolve_card_view
from nfl_ats.clv import load_paper_decisions, pick_correct
from nfl_ats.dashboard.findings_content import (
    HEADLINE,
    OVERLAY_UNION_ARCHIVE_SCORE_FRACTION,
    OVERLAY_UNION_SUBSET_COUNT,
    PLAYED_CARD_EXPECTATION_PERCENT,
)
from nfl_ats.four_overlay_composition import (
    COACH_FADE,
    DIVISION_REVENGE_TILT,
    INCUMBENT_CHALLENGER_ID,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    POLICY_ID,
    SPREAD_GAP_ZONE_FADE,
)
from nfl_ats.lineup_view import TeamLineup, load_lineups, validate_lineup_model_sync
from nfl_ats.market_decomposition import FAMILY_PHRASES
from nfl_ats.pick_refresh import MOVEMENT_POLICY_THRESHOLD
from nfl_ats.prospective_scoring import load_challenger_decisions
from nfl_ats.public_board import (
    DISCLAIMER_FULL,
    DISCLAIMER_SHORT,
    SWEEP_HALF_WIDTH,
    assert_spread_explorer_matches_card,
    confidence_word,
    load_played_chain_accuracy,
    load_public_board_artifacts,
    load_waterfall_feed,
    pick_side,
    spread_words,
)
from nfl_ats.reporting import artifact_directories, read_json
from nfl_ats.spread_explorer import (
    SPREAD_EXPLORER_MAX_LINE,
    SPREAD_EXPLORER_MIN_LINE,
    SPREAD_EXPLORER_STEP,
    SpreadExplorerGameParams,
    compute_spread_explorer_params,
    load_feature_table_for_forecast,
    widget_home_cover_probability,
)
from nfl_ats.spread_gap_zone_fade_overlay import (
    SPREAD_GAP_LOWER_BOUND,
    SPREAD_GAP_UPPER_BOUND,
)

#: Filled-segment count per confidence band, for a 3-segment strength meter.
#: Mirrors ``nfl_ats.public_board._CONFIDENCE_FILL`` -- duplicated rather
#: than imported (that name is private), the same discipline
#: ``public_board.py`` itself uses for its own ``_default_data_root``
#: duplication.
_CONFIDENCE_FILL: dict[str, int] = {"slight": 1, "lean": 2, "strong": 3}

#: Plain-English words for the four production overlay members, keyed by
#: their real member id (imported from ``four_overlay_composition``, never
#: retyped) -- used only when the rich policy narrative is available.
_MEMBER_LABELS: dict[str, str] = {
    COACH_FADE: "coach fade",
    DIVISION_REVENGE_TILT: "division revenge",
    PLAYER_ARRESTS_BACK_SIDE_POLICY: "player arrests",
    SPREAD_GAP_ZONE_FADE: "spread-gap zone",
}

_WEEK_LABELS = {
    "WC": "Wild Card round",
    "DIV": "Divisional round",
    "CON": "Conference championships",
    "SB": "Super Bowl",
}


def _default_data_root() -> Path:
    """Same env var and default as ``cli._data_root`` / ``public_board``'s
    own duplicate -- see that function's docstring for why this is
    duplicated rather than imported."""

    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _sentence_case(label: str) -> str:
    stripped = label.strip()
    return stripped[:1].upper() + stripped[1:] if stripped else stripped


def _parse_gameday(value: Any, fallback: date) -> date:
    if isinstance(value, str) and value.strip():
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return fallback
    return fallback


@dataclass(frozen=True)
class GameRow:
    """One board row: exactly the fields the public card already publishes
    (see the licensing note in ``public_board.py`` -- no raw market field
    beyond the one consensus ``spread_line`` ever reaches either page).

    Every text field a skin needs to print is precomputed here as a
    property, so neither skin module formats a number itself -- it only
    reads one.
    """

    game_id: str
    gameday: date
    weekday_name: str
    home: str
    away: str
    market_spread: float  # home-oriented spread_line
    pick_team: str
    pick_probability: float  # 0..1, the PICK side's own cover probability
    confidence_word: str  # "slight" | "lean" | "strong"
    is_best: bool
    is_flipped: bool
    #: Plain-English policy-member label(s) that flipped this pick this week
    #: (e.g. ``("coach fade",)``), empty when ``is_flipped`` is ``False`` --
    #: see :func:`_flip_member_labels`. A tuple rather than a single string
    #: because an overlap (two members both flipping the same game) is a
    #: real, if rare, case the joint-OR policy allows.
    flip_member_labels: tuple[str, ...] = ()
    #: Season mode (owner-approved improvement batch, item 4): ``True`` once
    #: this game has a real result -- see the module docstring for the data
    #: source. ``cover_result``/``final_score_text`` are only meaningful
    #: when this is ``True``.
    final: bool = False
    #: ``"win"`` / ``"loss"`` / ``"push"`` for the PLAYED pick (never the
    #: raw model's side), or ``None`` when ``final`` is ``False``.
    cover_result: str | None = None
    #: ``"AWAY score at HOME score"``, or ``None`` when ``final`` is
    #: ``False``.
    final_score_text: str | None = None
    #: "Flips at" (owner request, 2026-09-01): the first half-point line at
    #: which the displayed pick would switch to the other team, computed from
    #: the SAME guarded Gaussian read the on-page spread adjuster uses, so
    #: this column can never disagree with the slider on the same page (real
    #: ``line_sweep`` rows are the fallback when the adjuster is degraded --
    #: see :func:`_flip_line`). Home-oriented like ``market_spread``.
    #: ``None`` when no switch exists inside the adjuster's explored range
    #: (``flip_held`` True) or when no source exists (``flip_held`` False).
    flip_line: float | None = None
    #: ``True`` when a source existed, the full ±``SWEEP_HALF_WIDTH`` span
    #: the adjuster explores was scanned, and nothing in it switches the
    #: played pick. Deliberately a BOUNDED claim (owner catch, 2026-09-01:
    #: an earlier "at any line" wording asserted the pick at absurd
    #: hypothetical spreads where the fix-up rules have no evidence and
    #: would mechanically produce nonsense like IND laying 20).
    flip_held: bool = False

    @property
    def flip_line_text(self) -> str:
        """``'NYJ +2.5 → TEN'`` -- the CURRENT pick's own handicap at the
        first half-point line that changes the mind, then the team it
        switches to. Stated in the pick's orientation on purpose (owner
        feedback, 2026-09-01: a first draft printed the flipped-to team's
        handicap, which sat in the opposite orientation from the Pick column
        one cell over and read as ambiguous): the number here is directly
        comparable to the Pick column's, so ``NYJ +3`` flipping at
        ``NYJ +2.5 → TEN`` visibly means "if NYJ's points drop to +2.5,
        take TEN", and ``SEA -3.5`` flipping at ``SEA -4.5 → NE`` visibly
        means "if SEA has to lay 4.5, take NE". Empty when no flip line is
        known (policy-pinned pick, or degraded artifacts)."""

        if self.flip_line is None:
            if self.flip_held:
                return f"{self.pick_team} within ±{SWEEP_HALF_WIDTH:g}"
            return ""
        flip_team = self.home if self.pick_team == self.away else self.away
        sign = -1.0 if self.pick_team == self.home else 1.0
        value = self.flip_line * sign
        text = "pick'em" if value == 0 else f"{value:+g}"
        return f"{self.pick_team} {text} → {flip_team}"

    @property
    def flip_pill_text(self) -> str:
        """``"⇄ coach fade"`` -- the swap glyph (never the word "FLIPPED",
        per the owner's explicit instruction) plus the member name(s)."""

        return "⇄ " + " + ".join(self.flip_member_labels)

    @property
    def cover_result_label(self) -> str:
        return {"win": "Covered", "loss": "No cover", "push": "Push"}.get(
            self.cover_result or "", ""
        )

    @property
    def spread_text(self) -> str:
        """``'SEA -3.5'`` style, home-oriented."""

        return spread_words(self.home, self.away, self.market_spread)

    @property
    def pick_spread_text(self) -> str:
        """The market line restated as the PICK's own handicap."""

        sign = -1.0 if self.pick_team == self.home else 1.0
        value = self.market_spread * sign
        return "pick'em" if value == 0 else f"{value:+g}"

    @property
    def probability_text(self) -> str:
        return f"{self.pick_probability:.1%}"

    @property
    def spread_magnitude(self) -> float:
        return abs(self.market_spread)

    @property
    def confidence_fill(self) -> int:
        return _CONFIDENCE_FILL.get(self.confidence_word, 0)

    @property
    def confidence_label(self) -> str:
        """Title-cased real confidence word (``Slight``/``Lean``/``Strong``)
        -- the site's own three-band vocabulary, not a skin-invented one."""

        return self.confidence_word.capitalize()

    @property
    def kickoff_group_label(self) -> str:
        """``'Wed Sep 09'`` -- the day-group header both skins render."""

        return f"{self.weekday_name[:3]} {self.gameday.strftime('%b %d')}"

    @property
    def kickoff_short_label(self) -> str:
        """``'WED 09/09'`` -- the compact per-row kickoff label."""

        return f"{self.weekday_name[:3].upper()} {self.gameday.strftime('%m/%d')}"

    @property
    def ticker_text(self) -> str:
        return f"{self.away}@{self.home}"


@dataclass(frozen=True)
class AttributionRow:
    label: str
    delta_points: float | None
    cumulative_points: float | None
    bar_width_pct: float  # 0-50, center-anchored bar width both skins share

    @property
    def is_positive(self) -> bool:
        return (self.delta_points or 0.0) >= 0.0

    @property
    def delta_text(self) -> str:
        return "--" if self.delta_points is None else f"{self.delta_points:+.1f} pts"


@dataclass(frozen=True)
class AttributionPanel:
    """One game's "why this pick" breakdown. ``available`` is the content
    decision the renderer styles but never makes: ``False`` means the
    waterfall feed had no usable rows for this game, and the page renders
    its own designed degraded state (an em-dash plus ``unavailable_note``),
    never a guess."""

    available: bool
    game_id: str | None = None
    matchup_label: str | None = None
    probability_text: str | None = None
    rows: tuple[AttributionRow, ...] = ()
    net_points: float | None = None
    #: Explicit label naming the orientation of every signed value above --
    #: e.g. "Net toward MIA +3.5" -- so a positive number always reads as
    #: "supports the pick" without the reader having to guess a sign
    #: convention. Required whenever ``available`` and ``net_points`` are
    #: both set; the two skins render it verbatim rather than each guessing
    #: their own phrasing.
    net_label: str | None = None
    unavailable_note: str = "Attribution not published."


@dataclass(frozen=True)
class CoverCurvePoint:
    """One real swept point for a game's cover-probability curve: ``offset``
    is line offset from the card's own quoted line, ``probability`` is the
    PICK side's probability at that hypothetical line -- both straight from
    ``line_sweep.parquet``, never a fitted approximation."""

    offset: float
    probability: float  # 0..1, pick-oriented


@dataclass(frozen=True)
class SpreadAdjusterParams:
    """The published-fields-only Gaussian read behind a game's interactive
    line-offset adjuster (the restored "spread explorer" widget, folded into
    each game's deep dive rather than living on its own page -- 2026-08-31
    owner redirect).

    ``center``/``residual_mean``/``residual_std``/``card_line`` are the SAME
    rounded values :func:`nfl_ats.spread_explorer.spread_explorer_payload`
    embeds, already proven (by :func:`nfl_ats.public_board
    .assert_spread_explorer_matches_card`, a REQUIRED build-time guard run
    for every game before this object is ever built -- see
    :func:`_load_spread_explorer_params`) to reproduce this game's own
    published ``home_cover_probability`` at its own quoted line. Only
    present when the active model's probability method has a closed-form
    read (``gaussian`` -- see that module's docstring); absent otherwise,
    so a game's dive degrades to its static real-sweep chart with no
    adjuster rather than ever inventing a formula.
    """

    center: float
    residual_mean: float
    residual_std: float
    card_line: float  # home-oriented, published
    pick_is_home: bool


@dataclass(frozen=True)
class GameDive:
    """One game's full deep-dive: attribution, cover curve, and the
    line-offset adjuster -- everything the This Week page's game selector
    can show for that game. Built for EVERY game on the board (not only the
    Best Pick), so a reader can pick any of the week's games from the
    selector; ``BoardContent.best_pick_game_id`` names which one the
    selector defaults to."""

    game_id: str
    matchup_label: str
    pick_team: str
    pick_spread_text: str
    home: str
    kickoff_group_label: str
    probability_text: str
    is_best: bool
    attribution: AttributionPanel
    cover_curve: tuple[CoverCurvePoint, ...]
    cover_curve_offset_zero_note: str | None
    adjuster: SpreadAdjusterParams | None
    #: One sentence naming the raw model's own side vs. the side actually
    #: played, only set when this game was flipped by the policy overlay --
    #: see :func:`_flip_note`. ``None`` for every unflipped game.
    flip_note: str | None = None
    home_lineup: TeamLineup | None = None
    away_lineup: TeamLineup | None = None


@dataclass(frozen=True)
class PolicyNote:
    """The week's overlay-policy disclosure. ``rich_narrative`` is the full
    four-member story (only true when the production overlay composition is
    actually fresh this run); ``composition_text`` is always safe and never
    claims a member fired that did not."""

    composition_text: str
    rich_narrative: str | None
    policy_id: str | None
    policy_fingerprint: str | None
    members_text: str | None


@dataclass(frozen=True)
class HeadlineStats:
    """The four accuracy statistics the headline block shows, each traced to
    a real artifact -- see the module docstring. A caption is ``None`` only
    when its source figure is itself unavailable.

    Rendered on BOTH the This Week page and The Model page (a deliberate,
    single dedup exception -- see ``tests/test_board_content_coverage.py``):
    it is the one set of facts a reader needs at a glance in both places,
    always read off this SAME object rather than recomputed twice."""

    model_id: str | None
    model_method_label: str
    synced_at_text: str | None

    played_card_pct: float
    played_card_value_text: str
    #: A full-sentence caption.
    played_card_caption: str
    #: Mockup-scale short caption (a few words) for tight-flex layouts --
    #: Terminal's ``.headline-main .foot``. NEVER put the long sentence in a
    #: ``flex:0 0 auto`` box: its intrinsic width has nothing to shrink
    #: against and crushes its sibling (2026-08 coordinator finding).
    played_card_foot_text: str
    selection_caveat_text: str

    prior_chain_pct: float | None
    prior_chain_value_text: str
    prior_chain_caption: str

    raw_model_pct: float
    raw_model_value_text: str
    raw_model_ci: tuple[float, float] | None
    raw_model_caption: str
    raw_model_season_count: int
    raw_model_seasons_above_coin_flip: int | None
    raw_model_season_note: str | None

    close_grade_pct: float | None
    close_grade_value_text: str
    close_grade_caption: str

    #: The paired prospective record beside the "tracked prospectively"
    #: caveat above -- see :func:`_build_prospective_scoreboard`.
    prospective_scoreboard: ProspectiveScoreboard


@dataclass(frozen=True)
class ProspectiveScoreboard:
    """The paired prospective record: the played four-member policy vs. its
    immediate incumbent (the former coach+arrest chain), settled against
    real results. ``dormant`` is ``True`` -- and ``detail_text`` is
    ``None`` -- until either ledger holds a row, which does not happen
    until the first Tuesday lock (``docs/prospective_evidence.md``); see
    the module docstring for exactly which ledgers and outcomes table this
    reads."""

    dormant: bool
    headline_text: str
    detail_text: str | None


@dataclass(frozen=True)
class SeasonRecordStrip:
    """The hero's running record: this week, season to date, and the Best
    Pick tracked separately (the pool scores it separately) -- see
    :func:`_build_season_record`. Absent from :class:`BoardContent`
    (``None``) until at least one game has a real result, matching today's
    all-upcoming rendering exactly until then."""

    week_record_text: str
    season_record_text: str
    best_pick_record_text: str | None


@dataclass(frozen=True)
class TickerChrome:
    """Shared ticker + command-row content, rendered identically on all
    three pages (2026-08-31 owner: "the terminal header/feed thing needs to
    appear on every page"). Built ONCE from the This Week board's own games
    and reused (via ``dataclasses.replace`` for the one differing field) on
    every other page, so no page's ticker can ever show a different pick
    than This Week itself renders."""

    games: tuple[GameRow, ...]
    best_pick_game_id: str | None
    season: int | None
    week: int | None
    model_method_label: str
    #: The one field that legitimately differs per page, e.g. ``""`` on
    #: This Week, ``"--page model"`` on The Model.
    page_command_suffix: str


@dataclass(frozen=True)
class LinkPreview:
    """Per-page ``og:title``/``og:description`` text -- see
    :func:`_page_link_preview`."""

    title: str
    description: str


@dataclass(frozen=True)
class Finding:
    tag: str
    text: str


@dataclass(frozen=True)
class Disclaimer:
    short: str
    full: str


#: Site-wide cadence policy, imported by ``board_terminal.py`` and appended
#: to every page's footer "generated" line (owner-approved improvement
#: batch, item 10) -- a fact about how the pool locks, not per-week data, so
#: it lives here as one constant rather than being recomputed per page. See
#: the "Picks lock at kickoff" project memory: picks are editable up to each
#: game's own kickoff; only the pool's LINES freeze Tuesday.
CADENCE_NOTE = (
    "Picks can be updated until each game's own kickoff; the pool's lines freeze Tuesday."
)


@dataclass(frozen=True)
class BoardContent:
    """Everything the This Week page renders. Built once by
    :func:`load_board_content`; ``board_terminal.py`` only reads fields off
    this object."""

    season: int | None
    week: int | None
    game_type: str
    week_label: str
    generated_at: datetime
    generated_at_text: str

    games: tuple[GameRow, ...]
    best_pick_game_id: str | None
    best_pick_note: str
    flip_count: int
    strong_count: int

    headline: HeadlineStats
    policy: PolicyNote
    #: One :class:`GameDive` per game in ``games`` (same order), each with
    #: its own attribution panel, cover curve, and (when available) line-
    #: offset adjuster -- the This Week page's game selector shows one at a
    #: time, defaulting to ``best_pick_game_id``.
    dives: tuple[GameDive, ...]
    findings: tuple[Finding, ...]
    disclaimer: Disclaimer
    #: Shared ticker + command-row content -- see :class:`TickerChrome`.
    ticker_chrome: TickerChrome
    #: ``og:title``/``og:description`` text for this page.
    link_preview: LinkPreview
    #: The hero's running record strip (season mode, item 4) -- ``None``
    #: until at least one game this season has a real result.
    season_record: SeasonRecordStrip | None = None


# ---------------------------------------------------------------------------
# Loading -- the only place an artifact is opened.
# ---------------------------------------------------------------------------


def _load_overlay_subset_composition_summary(artifacts_root: Path) -> dict[str, Any] | None:
    directories = artifact_directories(artifacts_root / "overlay_subset_composition", "result.json")
    for directory in directories:
        try:
            return read_json(directory / "result.json")
        except (ValueError, OSError):
            continue
    return None


def _opener_season_ci(
    artifacts_root: Path, feature_profile: str | None
) -> tuple[float, float] | None:
    """Season-blocked 95% interval for ``opener_accuracy_probability_rule``
    from the newest ``opener_evaluation`` run for the active feature profile
    -- mirrors :func:`nfl_ats.public_board.load_opener_evaluation_artifacts`'s
    own feature-profile pin (2026-08-18 incident it guards against: a later,
    unrelated research run silently overriding the active model's own
    figures)."""

    directories = artifact_directories(artifacts_root / "opener_evaluation", "metadata.json")
    for directory in directories:
        try:
            metadata = read_json(directory / "metadata.json")
        except (ValueError, OSError):
            continue
        if feature_profile is not None:
            config = metadata.get("active_model_config") or {}
            if config.get("feature_profile") != feature_profile:
                continue
        for row in metadata.get("uncertainty") or []:
            if not isinstance(row, dict):
                continue
            if (
                row.get("metric") == "opener_accuracy_probability_rule"
                and row.get("block") == "season"
            ):
                lower, upper = _number(row.get("lower")), _number(row.get("upper"))
                if lower is not None and upper is not None:
                    return lower * 100, upper * 100
        return None
    return None


def _build_headline_stats(
    artifacts_root: Path,
    active: Mapping[str, Any],
    *,
    prospective_scoreboard: ProspectiveScoreboard,
) -> HeadlineStats:
    method = str(active.get("method") or "")
    feature_profile = active.get("feature_profile")
    regressor = str(active.get("regressor") or "")
    model_id = active.get("model_id")
    model_method_label = f"{feature_profile} ({method} {regressor})".strip()

    synced_at_text: str | None = None
    synced_raw = active.get("activated_at_utc")
    if isinstance(synced_raw, str):
        try:
            synced_at_text = datetime.fromisoformat(synced_raw).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            synced_at_text = None

    prior_chain_fraction = load_played_chain_accuracy(artifacts_root)
    prior_chain_pct = prior_chain_fraction * 100 if prior_chain_fraction is not None else None

    summary = _load_overlay_subset_composition_summary(artifacts_root)
    scored_games = _number((summary or {}).get("n_scored_games"))
    scored_games_int = int(scored_games) if scored_games is not None else None

    raw_model_ci = _opener_season_ci(
        artifacts_root, str(feature_profile) if feature_profile else None
    )

    historical = active.get("historical_evaluation")
    historical = historical if isinstance(historical, Mapping) else {}
    close_accuracy = _number(historical.get("accuracy"))
    close_correct = _number(historical.get("correct"))
    close_games = _number(historical.get("games"))
    intervals = historical.get("intervals")
    week_interval = intervals.get("week") if isinstance(intervals, Mapping) else None
    close_ci: tuple[float, float] | None = None
    if isinstance(week_interval, Mapping):
        lower, upper = _number(week_interval.get("lower")), _number(week_interval.get("upper"))
        if lower is not None and upper is not None:
            close_ci = (lower * 100, upper * 100)
    close_grade_pct = close_accuracy * 100 if close_accuracy is not None else None

    played_card_pct = OVERLAY_UNION_ARCHIVE_SCORE_FRACTION * 100
    games_text = (
        f"{scored_games_int:,}" if scored_games_int is not None else "an unpublished count of"
    )
    played_card_caption = (
        f"Opener-graded accuracy across {games_text} paired games -- the four-member overlay "
        "union that is actually on the board this week, not a hypothetical."
    )
    # Mockup-scale short form -- see the field docstring: this is the ONLY
    # form allowed inside a ``flex:0 0 auto`` box (Terminal's headline-main
    # foot; Cover Desk's "played" rung caption).
    played_card_foot_text = f"{games_text} opener-graded games · four-member overlay union"
    prior_chain_text = (
        f"{prior_chain_pct:.1f}%" if prior_chain_pct is not None else "not yet measured"
    )
    if prior_chain_pct is not None:
        expectation_points = round(PLAYED_CARD_EXPECTATION_PERCENT - prior_chain_pct)
        expectation_text = (
            f"roughly +{expectation_points} accuracy point{'s' if expectation_points != 1 else ''}"
        )
    else:
        expectation_text = f"roughly the pinned ≈{PLAYED_CARD_EXPECTATION_PERCENT}% planning figure"
    selection_caveat_text = (
        f"This {played_card_pct:.1f}% was selected from {OVERLAY_UNION_SUBSET_COUNT} correlated "
        "subsets of the same overlay members -- it is not a prospective expectation. The "
        f"operating expectation going forward is {expectation_text} over the prior "
        f"coach-to-arrests chain ({prior_chain_text}), and is being tracked prospectively in "
        "fresh paired games against that prior chain, not restated as this archive figure."
    )
    prior_chain_caption = f"{games_text} paired games · reference point, no interval attached."
    season_count = HEADLINE.last_season - HEADLINE.first_season + 1
    seasons_above_coin_flip = season_count if HEADLINE.season_low > 50.0 else None
    raw_model_season_note = (
        f"{seasons_above_coin_flip} of {season_count} seasons finished above the coin flip"
        if seasons_above_coin_flip is not None
        else None
    )
    if raw_model_ci is not None:
        season_suffix = f" -- {raw_model_season_note}" if raw_model_season_note else ""
        raw_model_caption = (
            f"95% CI [{raw_model_ci[0]:.2f}%, {raw_model_ci[1]:.2f}%], "
            f"season-blocked{season_suffix}."
        )
    else:
        raw_model_caption = f"{HEADLINE.games} paired games, {HEADLINE.seasons}."
    if close_ci is not None and close_correct is not None and close_games is not None:
        close_grade_caption = (
            f"{int(close_correct):,} / {int(close_games):,} non-push · week-blocked 95% CI "
            f"[{close_ci[0]:.2f}%, {close_ci[1]:.2f}%]."
        )
    else:
        close_grade_caption = "Close-graded evaluation not yet available."

    return HeadlineStats(
        model_id=str(model_id) if model_id else None,
        model_method_label=model_method_label,
        synced_at_text=synced_at_text,
        played_card_pct=played_card_pct,
        played_card_value_text=f"{played_card_pct:.1f}%",
        played_card_caption=played_card_caption,
        played_card_foot_text=played_card_foot_text,
        selection_caveat_text=selection_caveat_text,
        prior_chain_pct=prior_chain_pct,
        prior_chain_value_text=prior_chain_text if prior_chain_pct is not None else "--",
        prior_chain_caption=prior_chain_caption,
        raw_model_pct=HEADLINE.opener_accuracy,
        raw_model_value_text=HEADLINE.opener,
        raw_model_ci=raw_model_ci,
        raw_model_caption=raw_model_caption,
        raw_model_season_count=season_count,
        raw_model_seasons_above_coin_flip=seasons_above_coin_flip,
        raw_model_season_note=raw_model_season_note,
        close_grade_pct=close_grade_pct,
        close_grade_value_text=f"{close_grade_pct:.2f}%" if close_grade_pct is not None else "--",
        close_grade_caption=close_grade_caption,
        prospective_scoreboard=prospective_scoreboard,
    )


# ---------------------------------------------------------------------------
# Season mode -- in-season finals, outcomes, and the running record strip.
# See the module docstring for the data-source decision.
# ---------------------------------------------------------------------------

_OUTCOME_COLUMNS: tuple[str, ...] = ("game_id", "result", "home_score", "away_score")


def _load_game_outcomes(data_root: Path) -> pd.DataFrame:
    """The in-season finals table -- see the module docstring. Fail-open: a
    missing/unreadable file or a table missing an expected column degrades
    to "nothing is final yet", never an exception."""

    path = data_root / "processed" / "game_features.parquet"
    try:
        table = pd.read_parquet(path, columns=list(_OUTCOME_COLUMNS))
    except (OSError, ValueError):
        return pd.DataFrame(columns=_OUTCOME_COLUMNS)
    if not set(_OUTCOME_COLUMNS).issubset(table.columns):
        return pd.DataFrame(columns=_OUTCOME_COLUMNS)
    return table


def _game_final_state(
    *,
    home: str,
    away: str,
    pick_team: str,
    market_spread: float,
    result: Any,
    home_score: Any,
    away_score: Any,
) -> tuple[bool, str | None, str | None]:
    """Whether this game is final, the PLAYED pick's own cover result
    (``"win"``/``"loss"``/``"push"``), and a final-score sentence fragment
    -- or ``(False, None, None)`` for an upcoming game (no result yet).
    Uses the repo's one margin convention (FND-04, ``docs/modeling.md``):
    ``result`` is home minus away; the pick covers when the signed margin
    agrees with the pick's own side."""

    result_value = _number(result)
    if result_value is None:
        return False, None, None
    margin = result_value - market_spread
    if margin == 0.0:
        cover = "push"
    else:
        covered_home = margin > 0.0
        cover = "win" if covered_home == (pick_team == home) else "loss"
    home_score_value = _number(home_score)
    away_score_value = _number(away_score)
    score_text = (
        f"{away} {int(away_score_value)} at {home} {int(home_score_value)}"
        if home_score_value is not None and away_score_value is not None
        else None
    )
    return True, cover, score_text


def _record_text(wins: int, losses: int, pushes: int) -> str:
    return f"{wins}-{losses}-{pushes}" if pushes else f"{wins}-{losses}"


def _grade_decisions(decisions: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[int, int, int, int]:
    """``(wins, losses, pushes, pending)`` for one set of recorded picks
    against ``outcomes`` -- the SAME home-minus-away margin convention and
    the SAME ``nfl_ats.clv.pick_correct`` push/win/loss rule every other
    settlement in this repo already uses (FND-04), never a second,
    independently-drifting implementation."""

    if decisions.empty or outcomes.empty:
        return 0, 0, 0, len(decisions)
    merged = decisions.merge(
        outcomes[["game_id", "result"]], on="game_id", how="left", suffixes=("", "_outcome")
    )
    result = pd.to_numeric(merged["result"], errors="coerce")
    line = pd.to_numeric(merged["decision_home_spread"], errors="coerce")
    margin = result - line
    pick_home = merged["pick_side"].astype(str).eq("HOME")
    settled = margin.notna()
    pushed = settled & margin.eq(0.0)
    decided = settled & ~pushed
    correct = pick_correct(pick_home, margin)
    wins = int((decided & correct.eq(1.0)).sum())
    losses = int((decided & correct.eq(0.0)).sum())
    pushes = int(pushed.sum())
    pending = int((~settled).sum())
    return wins, losses, pushes, pending


def _build_prospective_scoreboard(
    paper_decisions: pd.DataFrame,
    challenger_decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> ProspectiveScoreboard:
    """The paired prospective record beside the "tracked prospectively"
    caveat -- see the module docstring for exactly which ledgers and
    outcomes table this reads. Dormant until either ledger holds a row,
    which does not happen until the first Tuesday lock
    (``docs/prospective_evidence.md``)."""

    played = (
        paper_decisions.loc[paper_decisions["decision_policy_id"].astype(str).eq(POLICY_ID)]
        if "decision_policy_id" in paper_decisions.columns
        else paper_decisions.iloc[0:0]
    )
    prior = (
        challenger_decisions.loc[
            challenger_decisions["challenger_id"].astype(str).eq(INCUMBENT_CHALLENGER_ID)
        ]
        if "challenger_id" in challenger_decisions.columns
        else challenger_decisions.iloc[0:0]
    )
    if played.empty and prior.empty:
        return ProspectiveScoreboard(
            dormant=True,
            headline_text="Prospective tracking begins Week 1.",
            detail_text=None,
        )
    played_wins, played_losses, played_pushes, played_pending = _grade_decisions(played, outcomes)
    prior_wins, prior_losses, prior_pushes, _prior_pending = _grade_decisions(prior, outcomes)
    settled = played_wins + played_losses + played_pushes
    headline_text = (
        "Prospective record at the decision line: played policy "
        f"{_record_text(played_wins, played_losses, played_pushes)} vs. prior chain "
        f"{_record_text(prior_wins, prior_losses, prior_pushes)} -- "
        f"{settled} of {len(played)} recorded games settled."
    )
    detail_text = (
        f"{played_pending} recorded game{'s' if played_pending != 1 else ''} not yet kicked off."
        if played_pending
        else None
    )
    return ProspectiveScoreboard(
        dormant=False, headline_text=headline_text, detail_text=detail_text
    )


def _build_season_record(
    paper_decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    season: int | None,
    week: int | None,
) -> SeasonRecordStrip | None:
    """The hero's running record strip -- ``None`` (dormant) until at least
    one game recorded this season has a real result, so the hero renders
    exactly as it does today for an all-upcoming season."""

    if paper_decisions.empty or season is None or "season" not in paper_decisions.columns:
        return None
    season_rows = paper_decisions.loc[
        pd.to_numeric(paper_decisions["season"], errors="coerce").eq(float(season))
    ]
    if season_rows.empty:
        return None
    season_wins, season_losses, season_pushes, _season_pending = _grade_decisions(
        season_rows, outcomes
    )
    if season_wins + season_losses + season_pushes == 0:
        return None  # nothing graded yet this season -- stay dormant

    week_rows = (
        season_rows.loc[pd.to_numeric(season_rows["week"], errors="coerce").eq(float(week))]
        if week is not None and "week" in season_rows.columns
        else season_rows.iloc[0:0]
    )
    week_wins, week_losses, week_pushes, _week_pending = _grade_decisions(week_rows, outcomes)
    week_record_text = f"This week: {_record_text(week_wins, week_losses, week_pushes)} so far"
    season_record_text = (
        f"Season to date: {_record_text(season_wins, season_losses, season_pushes)}"
    )

    best_pick_rows = (
        season_rows.loc[season_rows["is_best_pick"].fillna(False).astype(bool)]
        if "is_best_pick" in season_rows.columns
        else season_rows.iloc[0:0]
    )
    best_wins, best_losses, best_pushes, _best_pending = _grade_decisions(best_pick_rows, outcomes)
    best_pick_record_text = (
        f"Best Pick: {_record_text(best_wins, best_losses, best_pushes)}"
        if best_wins + best_losses + best_pushes > 0
        else None
    )
    return SeasonRecordStrip(
        week_record_text=week_record_text,
        season_record_text=season_record_text,
        best_pick_record_text=best_pick_record_text,
    )


#: Individually-named channels, highest |contribution| first, before the
#: remainder collapses into one "everything else" row -- the mockup showed
#: four illustrative channels; this keeps the real panel the same visual
#: density instead of a 12-row jargon dump.
_MAX_ATTRIBUTION_CHANNELS = 4

#: The waterfall feed's own "probability rule" label is already
#: plain-English site-wide (public_board's own why-this-pick panel and the
#: rationale sentences both call it exactly this), so it is kept verbatim
#: rather than re-derived.
_PROBABILITY_RULE_LABEL = "Residual-sample calibration shift"


def _plain_family_label(family: str) -> str:
    """A reader-facing label for a model feature family, via the SAME
    ``FAMILY_PHRASES`` mapping the site's own market-decomposition prose
    uses -- never the raw registry id ("player_qb", "weekly_context")."""

    phrase = FAMILY_PHRASES.get(family)
    return _sentence_case(phrase) if phrase else _sentence_case(family.replace("_", " "))


def _build_attribution(entry: Mapping[str, Any] | None, game: GameRow | None) -> AttributionPanel:
    """Real waterfall rows for ``game``, curated and pick-oriented, or the
    designed unavailable state -- fail-open, exactly like every other
    optional artifact this site reads (see
    ``public_board.load_waterfall_feed``'s own docstring). Called once per
    game (see :func:`_build_dive`), not only for the Best Pick.

    The feed's ``delta_points``/``cumulative_points`` are HOME-oriented (the
    same sign convention ``predicted_residual`` uses); this function
    re-orients every value to the PICK side using the feed's own
    ``picked_side`` field (mirrors the sign the feed's own ``direction``
    field already encodes per step -- see
    ``nfl_ats.attribution_waterfall.build_game_waterfall``), so a positive
    number always means "supports the pick", never a mix of two
    conventions on one panel. Only feature-FAMILY steps and the probability
    rule are shown (never the market/final marker steps): the top
    :data:`_MAX_ATTRIBUTION_CHANNELS` by |oriented contribution|, individually
    labeled via ``FAMILY_PHRASES``, plus one aggregated "everything else" row
    that captures the exact remainder (so the displayed rows always sum to
    the true net -- nothing is dropped, only grouped).
    """

    if game is None or not isinstance(entry, Mapping):
        return AttributionPanel(available=False)
    picked_side = str(entry.get("picked_side") or "").upper()
    if picked_side not in ("HOME", "AWAY"):
        return AttributionPanel(available=False)
    sign = 1.0 if picked_side == "HOME" else -1.0
    steps_raw = entry.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        return AttributionPanel(available=False)

    family_oriented: list[tuple[str, float]] = []
    for step in steps_raw:
        if not isinstance(step, Mapping) or step.get("kind") != "family":
            continue
        raw_delta = _number(step.get("delta_points"))
        family = step.get("family")
        if raw_delta is None or not isinstance(family, str) or not family:
            continue
        family_oriented.append((family, raw_delta * sign))
    if not family_oriented:
        return AttributionPanel(available=False)

    family_oriented.sort(key=lambda item: abs(item[1]), reverse=True)
    shown, rest = (
        family_oriented[:_MAX_ATTRIBUTION_CHANNELS],
        family_oriented[_MAX_ATTRIBUTION_CHANNELS:],
    )
    max_abs = max((abs(value) for _, value in family_oriented), default=0.0) or 1.0

    def _row(label: str, value: float) -> AttributionRow:
        return AttributionRow(
            label=label,
            delta_points=value,
            cumulative_points=None,
            bar_width_pct=min(50.0, abs(value) / max_abs * 50.0),
        )

    rows = [_row(_plain_family_label(family), value) for family, value in shown]
    if rest:
        rows.append(_row(f"Everything else ({len(rest)} more factors)", sum(v for _, v in rest)))

    probability_rule_step = next(
        (
            step
            for step in steps_raw
            if isinstance(step, Mapping) and step.get("kind") == "probability_rule"
        ),
        None,
    )
    if probability_rule_step is not None:
        raw_delta = _number(probability_rule_step.get("delta_points"))
        if raw_delta is not None and raw_delta != 0.0:
            rows.append(_row(_PROBABILITY_RULE_LABEL, raw_delta * sign))

    net_points = sum(row.delta_points for row in rows if row.delta_points is not None)
    matchup_label = f"{game.pick_team} {game.pick_spread_text} at {game.home}"
    return AttributionPanel(
        available=True,
        game_id=game.game_id,
        matchup_label=matchup_label,
        probability_text=game.probability_text,
        rows=tuple(rows),
        net_points=net_points,
        net_label=f"Net toward {game.pick_team} {game.pick_spread_text}",
    )


#: The cover curve's fallback offset grid, for a game whose Gaussian read
#: exists but whose real ``line_sweep`` row does not. Mirrors
#: ``public_board._COVER_CURVE_FALLBACK_OFFSETS`` exactly (same span/step,
#: both public constants -- ``SWEEP_HALF_WIDTH``/``SPREAD_EXPLORER_STEP`` --
#: so a chart built from this fallback looks exactly like one built from
#: real swept rows); duplicated rather than imported since that name is
#: private, the same discipline this module already follows elsewhere (see
#: ``_CONFIDENCE_FILL`` above).
_COVER_CURVE_FALLBACK_OFFSETS: tuple[float, ...] = tuple(
    round(-SWEEP_HALF_WIDTH + step_index * SPREAD_EXPLORER_STEP, 1)
    for step_index in range(round(2 * SWEEP_HALF_WIDTH / SPREAD_EXPLORER_STEP) + 1)
)


def _build_cover_curve(
    sweep: pd.DataFrame,
    game: GameRow | None,
    spread_explorer_params: Mapping[str, SpreadExplorerGameParams] | None = None,
) -> tuple[CoverCurvePoint, ...]:
    """Real swept model output for ``game``, oriented to the pick side --
    mirrors ``public_board._game_deep_dive``'s own preference for REAL
    swept rows over any fitted approximation. Called once per game (see
    :func:`_build_dive`), not only for the Best Pick.

    ``spread_explorer_params`` (2026-08, full-site conversion item 4 -- "keep
    the spread explorer's exact published-card math and guard") is the SAME
    fallback production's own merged cover-curve/spread-explorer widget uses:
    when the game has no real ``line_sweep`` row (an older/rolled-back
    artifact tree), the curve is instead synthesized from the closed-form
    Gaussian read -- ``spread_explorer.widget_home_cover_probability``, the
    exact browser-mirrored erf approximation -- over the SAME standard offset
    grid (:data:`_COVER_CURVE_FALLBACK_OFFSETS`) production uses. Never a
    fitted approximation invented here: ``spread_explorer_params`` is only
    ever populated by :func:`load_board_content` after
    ``public_board.assert_spread_explorer_matches_card`` has already proven
    the SAME formula reproduces the published card's own quoted-line
    probability. Empty only when the game has neither a real sweep row nor a
    Gaussian read; the Terminal skin renders its degraded chart state then.
    """

    if game is None:
        return ()
    if not sweep.empty and {"game_id", "line_offset", "home_cover_probability"}.issubset(
        sweep.columns
    ):
        rows = sweep.loc[
            sweep["game_id"].astype(str).eq(game.game_id)
            & sweep["line_offset"].abs().le(SWEEP_HALF_WIDTH)
        ].sort_values("line_offset")
        if not rows.empty:
            pick_is_home = game.pick_team == game.home
            return tuple(
                CoverCurvePoint(
                    offset=float(offset),
                    probability=float(probability) if pick_is_home else 1.0 - float(probability),
                )
                for offset, probability in zip(
                    rows["line_offset"], rows["home_cover_probability"], strict=True
                )
            )
    params = (spread_explorer_params or {}).get(game.game_id)
    if params is None:
        return ()
    pick_is_home = game.pick_team == game.home
    points = []
    for offset in _COVER_CURVE_FALLBACK_OFFSETS:
        home_probability = widget_home_cover_probability(
            params.card_line + offset, params.center, params.residual_mean, params.residual_std
        )
        points.append(
            CoverCurvePoint(
                offset=offset,
                probability=home_probability if pick_is_home else 1.0 - home_probability,
            )
        )
    return tuple(points)


def _in_spread_gap_zone(line: float) -> bool:
    return SPREAD_GAP_LOWER_BOUND <= abs(line) <= SPREAD_GAP_UPPER_BOUND


def _flip_line(
    game_id: str,
    home: str,
    pick_team: str,
    quoted_line: float,
    flip_member_ids: tuple[str, ...],
    sweep: pd.DataFrame,
    spread_explorer_params: Mapping[str, SpreadExplorerGameParams],
) -> tuple[float | None, bool]:
    """The first half-point line at which the PLAYED pick would switch sides,
    and whether the pick is pinned (a fired pick-conditioned member, no
    switch found anywhere in range).

    The played pick is the raw model plus the four-member policy, so the
    hypothetical "what if the Tuesday line had been L" is answered with the
    policy re-evaluated at L, not the raw curve alone (owner catch,
    2026-09-01: the first draft dashed out policy-flipped games, hiding that
    CLE @ JAX -- flipped by the 7.5-10 spread-gap zone -- reverts to the raw
    pick on a HALF-POINT move out of the zone):

    * ``played(L) = raw(L)``, complemented once if any member fires at L
      (the composition's own ``complement_once`` semantics).
    * The spread-gap zone member fires on
      ``SPREAD_GAP_LOWER_BOUND <= |L| <= SPREAD_GAP_UPPER_BOUND``, but it is
      re-evaluated ONLY within ``MOVEMENT_POLICY_THRESHOLD`` (1.0 point --
      production's own measured definition of a decision-relevant line
      difference) of the quoted line; beyond that band its state is FROZEN
      at what really happened. Owner catch #3, 2026-09-01: the zone's
      evidence comes from games the market actually priced at 7.5-10, so
      re-firing it on a 3.5-point game hypothetically repriced to 7.5
      produced the absurd "IND +7.5 -> BAL" (give the pick MORE points and
      lose it). Within a point of the real line the counterfactual is a
      line the pool could genuinely quote (CLE +7.5 exits the zone at +7,
      DET -7 enters it at -7.5); four points away it is out of the rule's
      evidence entirely. Freezing (rather than disabling) beyond the band
      keeps the domain boundary itself from fabricating flips.
    * A fired pick-conditioned member (coach fade, division revenge,
      arrests -- none of their conditions reads the spread; verified in
      their modules 2026-09-01) keeps firing exactly when the raw side is
      the side it faded. Known approximation, stated once: a pick-
      conditioned member that did NOT fire on the real card is assumed
      never to fire at hypothetical lines either -- re-evaluating (say)
      whether the coach fade would trigger once the raw model crossed sides
      needs the member's own data, not the card, and is not worth the build
      dependency for a board column.

    The scan is DELIBERATELY bounded to the ±``SWEEP_HALF_WIDTH`` span the
    on-page chart and slider explore (owner catch, 2026-09-01, second
    round: an unbounded scan produced "at any line" -- an assertion about
    absurd hypothetical spreads, e.g. a spread-blind coach fade mechanically
    holding IND while laying 20, where no rule here has evidence). Inside
    that span the answer is (line, False); a full scan with no switch is
    (None, True) -- "holds within the explored range" and nothing stronger.

    The raw side comes from the guarded Gaussian read (the spread adjuster's
    own math, proven against the published card) with real ``line_sweep``
    rows as the degraded-artifact fallback; scans outward from the quoted
    line on the slider's own half-point grid.
    """

    pick_is_home = pick_team == home
    pin_fired = any(member != SPREAD_GAP_ZONE_FADE for member in flip_member_ids)
    zone_fired = SPREAD_GAP_ZONE_FADE in flip_member_ids
    is_flipped = bool(flip_member_ids)
    # The raw side at the quoted line: flipped games play the complement.
    raw_home_at_card = pick_is_home != is_flipped

    def zone_active(line: float) -> bool:
        if abs(line - quoted_line) <= MOVEMENT_POLICY_THRESHOLD:
            return _in_spread_gap_zone(line)
        return zone_fired

    def played_is_home(raw_is_home: bool, line: float) -> bool:
        fires = zone_active(line) or (pin_fired and raw_is_home == raw_home_at_card)
        return raw_is_home != fires

    params = spread_explorer_params.get(game_id)
    if params is not None:
        if played_is_home(raw_home_at_card, params.card_line) != pick_is_home:
            return None, False  # stale artifacts disagree with the card; show nothing
        steps = round(SWEEP_HALF_WIDTH / SPREAD_EXPLORER_STEP)
        for step_index in range(1, steps + 1):
            for direction in (-1.0, 1.0):
                line = params.card_line + direction * step_index * SPREAD_EXPLORER_STEP
                if not SPREAD_EXPLORER_MIN_LINE <= line <= SPREAD_EXPLORER_MAX_LINE:
                    continue
                raw_is_home = (
                    widget_home_cover_probability(
                        line, params.center, params.residual_mean, params.residual_std
                    )
                    >= 0.5
                )
                if played_is_home(raw_is_home, line) != pick_is_home:
                    return round(line, 1), False
        return None, True
    required = {"game_id", "line_offset", "alternative_line", "home_cover_probability"}
    if sweep.empty or not required.issubset(sweep.columns):
        return None, False
    rows = sweep.loc[
        sweep["game_id"].astype(str).eq(game_id) & sweep["line_offset"].abs().le(SWEEP_HALF_WIDTH)
    ].sort_values("line_offset", key=lambda offsets: offsets.abs(), kind="stable")
    for _, row in rows.iterrows():
        if float(row["line_offset"]) == 0.0:
            continue
        line = float(row["alternative_line"])
        raw_is_home = float(row["home_cover_probability"]) >= 0.5
        if played_is_home(raw_is_home, line) != pick_is_home:
            return round(line, 1), False
    return None, not rows.empty


#: Below this many probability POINTS of disagreement between the sweep's
#: own line-0 point and the live pick probability, treat it as rounding
#: noise -- never a genuine artifact-staleness disclosure.
_COVER_CURVE_DISAGREEMENT_THRESHOLD_POINTS = 0.5


def _cover_curve_offset_zero_note(
    cover_curve: tuple[CoverCurvePoint, ...], game: GameRow | None
) -> str | None:
    """A designed disclosure, never a silent contradiction: when the swept
    curve's own line-0 point reads a different probability than the live
    card (the sweep artifact can predate the latest weekly refresh -- a
    real, pre-existing characteristic of this site, not introduced by
    either skin), name both numbers explicitly rather than let a reader
    notice the mismatch unaided."""

    if game is None:
        return None
    current = next((point for point in cover_curve if point.offset == 0.0), None)
    if current is None:
        return None
    gap_points = (current.probability - game.pick_probability) * 100
    if abs(gap_points) < _COVER_CURVE_DISAGREEMENT_THRESHOLD_POINTS:
        return None
    return (
        f"This chart's own swept line reads {current.probability:.1%} at the card's quoted "
        f"line -- the live cover probability shown elsewhere on this page is "
        f"{game.probability_text}. The swept curve can predate the latest weekly refresh; "
        "the live number is the one actually played."
    )


def _build_adjuster(
    game: GameRow, spread_explorer_params: Mapping[str, SpreadExplorerGameParams]
) -> SpreadAdjusterParams | None:
    """This game's line-offset adjuster params, or ``None`` when this
    build's active model has no closed-form (Gaussian) probability read --
    see :func:`_load_spread_explorer_params`, which has ALREADY run the
    REQUIRED ``assert_spread_explorer_matches_card`` guard for every game in
    ``spread_explorer_params`` before this function ever sees it, so every
    :class:`SpreadAdjusterParams` this returns is guard-proven against the
    published card, never a formula invented here."""

    params = spread_explorer_params.get(game.game_id)
    if params is None:
        return None
    return SpreadAdjusterParams(
        center=params.center,
        residual_mean=params.residual_mean,
        residual_std=params.residual_std,
        card_line=params.card_line,
        pick_is_home=game.pick_team == game.home,
    )


def _flip_member_labels(view: Any, game_id: str) -> tuple[str, ...]:
    """Which policy member(s) flipped ``game_id`` this week, as plain-
    English labels (owner-approved improvement batch, item 1) -- traced from
    the real four-member production result when it is available (the live
    production path, :func:`nfl_ats.four_overlay_composition
    .apply_four_overlay_composition`'s own per-game provenance), falling
    back to the legacy coach-fade/player-arrests overlays' own flip lists
    for a rehearsal read with no ``production_overlay`` (``view.overlay``/
    ``view.arrest_overlay``, the pre-four-overlay code path). A tuple
    because the joint-OR policy allows more than one member to flip the
    SAME game (an overlap), which must show every member that fired, not
    just one."""

    if view is None:
        return ()
    if view.production_overlay is not None:
        provenance = next(
            (game for game in view.production_overlay.games if game.game_id == game_id), None
        )
        if provenance is None:
            return ()
        return tuple(
            _MEMBER_LABELS.get(member, member.replace("_", " ")) for member in provenance.member_ids
        )
    labels: list[str] = []
    if any(flip.game_id == game_id for flip in view.overlay.flips):
        labels.append(_MEMBER_LABELS[COACH_FADE])
    if view.arrest_overlay.enabled and any(
        flip.game_id == game_id for flip in view.arrest_overlay.flips
    ):
        labels.append(_MEMBER_LABELS[PLAYER_ARRESTS_BACK_SIDE_POLICY])
    return tuple(labels)


def _flip_note(game: GameRow, raw_home_cover_probability: float | None) -> str | None:
    """One sentence naming the raw model's own side vs. the side actually
    played, for a game's deep-dive header (owner-approved improvement
    batch, item 1) -- ``None`` when the game was not flipped, or when the
    raw (pre-overlay) probability this week's card started from is
    unavailable for some reason."""

    if not game.flip_member_labels or raw_home_cover_probability is None:
        return None
    raw_pick_team = game.home if raw_home_cover_probability >= 0.5 else game.away
    if raw_pick_team == game.pick_team:
        return None  # the flip toggled a probability but not the final side -- nothing to say
    members = " + ".join(game.flip_member_labels)
    return (
        f"The raw model favored {raw_pick_team} to cover this line; the played card flips to "
        f"{game.pick_team} via {members}."
    )


def _build_dive(
    game: GameRow,
    *,
    sweep: pd.DataFrame,
    waterfall_feed: Mapping[str, Mapping[str, Any]],
    spread_explorer_params: Mapping[str, SpreadExplorerGameParams],
    raw_home_cover_probability: float | None,
    lineups: Mapping[str, tuple[TeamLineup, TeamLineup]],
) -> GameDive:
    """One game's full deep dive -- attribution, cover curve, and adjuster --
    built the same way regardless of whether ``game`` is the Best Pick: the
    This Week page's game selector must be able to show any of them."""

    attribution = _build_attribution(waterfall_feed.get(game.game_id), game)
    cover_curve = _build_cover_curve(sweep, game, spread_explorer_params)
    cover_curve_offset_zero_note = _cover_curve_offset_zero_note(cover_curve, game)
    adjuster = _build_adjuster(game, spread_explorer_params)
    game_lineups = lineups.get(game.game_id)
    home_lineup = away_lineup = None
    if game_lineups is not None:
        home_lineup, away_lineup = game_lineups
        qb_family_points = next(
            (
                row.delta_points
                for row in attribution.rows
                if "quarterback" in row.label.lower() or "qb" in row.label.lower()
            ),
            None,
        )
        # The artifact stores the model's QB identity in the lineup payload.
        # ``with_model_impact`` intentionally leaves a mismatch visible.
        home_lineup = home_lineup.with_model_impact(
            family_points=qb_family_points
            if game.pick_team == game.home
            else -qb_family_points
            if qb_family_points is not None
            else None,
            model_qb_id=next(
                (p.gsis_id for p in home_lineup.players if p.model_role == "base_model"), None
            ),
        )
        away_lineup = away_lineup.with_model_impact(
            family_points=qb_family_points
            if game.pick_team == game.away
            else -qb_family_points
            if qb_family_points is not None
            else None,
            model_qb_id=next(
                (p.gsis_id for p in away_lineup.players if p.model_role == "base_model"), None
            ),
        )
    return GameDive(
        game_id=game.game_id,
        matchup_label=f"{game.pick_team} {game.pick_spread_text} at {game.home}",
        pick_team=game.pick_team,
        pick_spread_text=game.pick_spread_text,
        home=game.home,
        kickoff_group_label=game.kickoff_group_label,
        probability_text=game.probability_text,
        is_best=game.is_best,
        attribution=attribution,
        cover_curve=cover_curve,
        cover_curve_offset_zero_note=cover_curve_offset_zero_note,
        adjuster=adjuster,
        flip_note=_flip_note(game, raw_home_cover_probability),
        home_lineup=home_lineup,
        away_lineup=away_lineup,
    )


def _best_pick_note(nomination: Any) -> str:
    """Verbatim reuse of ``render_picks_page``'s own Best Pick disclosure
    sentence (2026-08-23 tie-break audit wording) so neither design page can
    ever disagree with the production picks page about why a game is the
    Best Pick."""

    if nomination is None or nomination.active_game_id is None:
        return ""
    if nomination.active_rule == "v2":
        return f"This pick was {nomination.method_note}"
    tie = f" {nomination.active_tie_note}" if nomination.active_tie_note else ""
    return (
        "This is the pick whose edge survives the widest range of line movement -- the "
        "best-measured lever among forced picks, budgeted at roughly +0.9 points, not the "
        f"+8.7 once recorded before a tie-break audit.{tie}"
    )


def _build_policy_note(view: Any, strong_count: int, n_games: int) -> tuple[PolicyNote, int]:
    """Returns the note plus the flip count it describes."""

    if view is None:
        return (
            PolicyNote(
                composition_text="Synchronized with the active model.",
                rich_narrative=None,
                policy_id=None,
                policy_fingerprint=None,
                members_text=None,
            ),
            0,
        )
    composition = ["Synchronized with the active model"]
    if strong_count:
        composition.append(f"{strong_count} strong lean{'s' if strong_count != 1 else ''}")
    production_overlay = view.production_overlay
    if production_overlay is not None:
        flip_count = production_overlay.flip_count
        composition.append(
            f"{flip_count} pick{'s' if flip_count != 1 else ''} flipped by the fix-up rules"
        )
        members_text = ", ".join(
            _MEMBER_LABELS.get(member, member.replace("_", " "))
            for member in production_overlay.composition_order
        )
        rich_narrative = (
            f"Policy overlay active -- {members_text}. Flipped {flip_count} pick"
            f"{'s' if flip_count != 1 else ''} vs. the raw model this week."
        )
        return (
            PolicyNote(
                composition_text=" · ".join(composition),
                rich_narrative=rich_narrative,
                policy_id=production_overlay.policy_id,
                policy_fingerprint=production_overlay.policy_fingerprint,
                members_text=members_text,
            ),
            flip_count,
        )
    flip_count = view.overlay.flip_count + (
        view.arrest_overlay.flip_count if view.arrest_overlay.enabled else 0
    )
    if view.overlay.flip_count:
        composition.append(
            f"{view.overlay.flip_count} pick{'s' if view.overlay.flip_count != 1 else ''} "
            "flipped by the coach-fade overlay"
        )
    if view.arrest_overlay.enabled:
        composition.append(
            f"player-arrest policy active · {view.arrest_overlay.flip_count} pick"
            f"{'s' if view.arrest_overlay.flip_count != 1 else ''} flipped this week"
        )
    return (
        PolicyNote(
            composition_text=" · ".join(composition),
            rich_narrative=None,
            policy_id=None,
            policy_fingerprint=None,
            members_text=None,
        ),
        flip_count,
    )


def _build_findings(games: tuple[GameRow, ...], flip_count: int) -> tuple[Finding, ...]:
    """Two findings, both computed from THIS week's real board -- never the
    mockups' illustrative, hand-typed prose."""

    if not games:
        return ()
    most_confident = max(games, key=lambda g: (g.pick_probability, g.game_id))
    shape_finding = Finding(
        tag="NOTE // CONFIDENCE SHAPE",
        text=(
            f"This week's most confident call is {most_confident.pick_team} "
            f"{most_confident.pick_spread_text} at {most_confident.probability_text} cover "
            "probability -- the strongest lean on the board. Most other games cluster just "
            "above the coin flip, which is the honest, expected shape for a market this "
            "efficient."
        ),
    )
    overlay_finding = Finding(
        tag="NOTE // OVERLAY REACH",
        text=(
            "The overlay nudges picks, it doesn't rebuild them. This week's active overlay "
            f"policy touched {flip_count} of this week's {len(games)} raw-model picks -- small, "
            "deliberate, and reported separately from the raw model so each layer's "
            "contribution stays visible."
        ),
    )
    return (shape_finding, overlay_finding)


def _load_spread_explorer_params(
    metadata: Mapping[str, Any], predictions: pd.DataFrame, data_root: Path
) -> dict[str, SpreadExplorerGameParams]:
    """The active forecast's spread-explorer Gaussian read, verified against
    the published card -- the SAME recipe ``public_board.build_public_site``
    runs, item-for-item: only the ``gaussian`` probability method has a
    closed-form mean/sd the widget's erf formula can read (an
    older/rolled-back ``ecdf`` active model does not), so this quietly
    returns ``{}`` for any other configuration -- the SAME graceful
    degradation contract every optional artifact on this site follows.
    ``public_board.assert_spread_explorer_matches_card`` is a REQUIRED guard,
    not a courtesy: it re-proves, at build time, that the exact formula
    shipped to the browser reproduces the published card's own quoted-line
    probability, and raises rather than let a chart silently disagree with
    the number already on the page. See :func:`_build_cover_curve`'s
    docstring for how the result is used (a fallback only -- real
    ``line_sweep`` rows are always preferred).
    """

    if str(metadata.get("probability_method")) != "gaussian" or predictions.empty:
        return {}
    features = load_feature_table_for_forecast(metadata, data_root)
    params = compute_spread_explorer_params(
        predictions,
        features,
        regressor=str(metadata.get("regressor")),
        ridge_alpha=float(metadata.get("ridge_alpha", 10.0)),
        feature_profile=str(metadata.get("feature_profile")),
        min_train_games=int(metadata.get("min_train_games", 500)),
    )
    assert_spread_explorer_matches_card(params, predictions)
    return params


def load_board_content(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = False,
) -> BoardContent:
    """Assemble the one view model the This Week page renders.

    Uses the SAME artifact loaders and overlay-resolution path
    (:func:`nfl_ats.card_view.resolve_card_view`) the real site's
    ``build_public_site`` uses, so this page can never show a different
    pick, probability, or Best Pick than the currently-published card.
    ``require_fresh_arrest_overlay`` defaults to ``False`` (a rehearsal read,
    matching how the test suite exercises this same path); the real publish
    path (``cli._write_public_site`` -> ``board_site.build_site`` ->
    :func:`nfl_ats.board_site_content.load_site_content`) passes ``True``.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    resolved_data_root = data_root if data_root is not None else _default_data_root()
    artifacts = load_public_board_artifacts(artifacts_root)

    game_type = (
        str(artifacts.predictions["game_type"].iloc[0])
        if "game_type" in artifacts.predictions and not artifacts.predictions.empty
        else "REG"
    )
    week_label = _WEEK_LABELS.get(game_type, f"Week {artifacts.metadata.get('week')}")

    view = (
        resolve_card_view(
            artifacts.predictions,
            artifacts.sweep,
            artifacts.metadata,
            data_root=resolved_data_root,
            now=generated,
            require_fresh_arrest_overlay=require_fresh_arrest_overlay,
        )
        if game_type == "REG" and not artifacts.predictions.empty
        else None
    )

    final = view.predictions if view is not None else artifacts.predictions
    sort_columns = [column for column in ("kickoff", "game_id") if column in final]
    ordered = final.sort_values(sort_columns, na_position="last") if sort_columns else final

    # ``flip_member_ids_by_game`` also feeds the "Flips at" column, which
    # needs member IDENTITY (the spread-gap zone member re-evaluates with a
    # hypothetical line; the pick-conditioned members do not), not only the
    # flipped-or-not fact.
    flip_member_ids_by_game: dict[str, tuple[str, ...]] = {}
    if view is not None and view.production_overlay is not None:
        flip_member_ids_by_game = {
            game.game_id: tuple(game.member_ids) for game in view.production_overlay.games
        }
    elif view is not None:
        for flip in view.overlay.flips:
            flip_member_ids_by_game[str(flip.game_id)] = (COACH_FADE,)
        for arrest_flip in view.arrest_overlay.flips:
            existing = flip_member_ids_by_game.get(str(arrest_flip.game_id), ())
            flip_member_ids_by_game[str(arrest_flip.game_id)] = (
                *existing,
                PLAYER_ARRESTS_BACK_SIDE_POLICY,
            )
    flipped_game_ids = set(flip_member_ids_by_game)

    best_pick_id = view.nomination.active_game_id if view is not None else None
    best_pick_note = _best_pick_note(view.nomination) if view is not None else ""

    # The RAW (pre-overlay) card, for the flip note's "raw model favored X"
    # sentence (item 1) -- ``artifacts.predictions`` is exactly the frame
    # passed into ``resolve_card_view`` above, never the overlaid one.
    raw_probability_by_game: dict[str, float] = {}
    if {"game_id", "home_cover_probability"}.issubset(artifacts.predictions.columns):
        for _, raw_row in artifacts.predictions.iterrows():
            raw_value = _number(raw_row.get("home_cover_probability"))
            if raw_value is not None:
                raw_probability_by_game[str(raw_row["game_id"])] = raw_value

    # In-season finals (item 4) -- see the module docstring for the source.
    outcomes = _load_game_outcomes(resolved_data_root)
    outcome_by_game_id: dict[str, tuple[Any, Any, Any]] = {}
    if not outcomes.empty:
        for _, outcome_row in outcomes.iterrows():
            outcome_by_game_id[str(outcome_row["game_id"])] = (
                outcome_row.get("result"),
                outcome_row.get("home_score"),
                outcome_row.get("away_score"),
            )

    # Loaded BEFORE the game rows (not only for the dives below) since
    # 2026-09-01: the board's "Flips at" column reads the same guarded
    # Gaussian params the adjuster uses -- see _load_spread_explorer_params /
    # assert_spread_explorer_matches_card for the guard, which still runs
    # exactly once per build.
    spread_explorer_params = _load_spread_explorer_params(
        artifacts.metadata, artifacts.predictions, resolved_data_root
    )

    games: list[GameRow] = []
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        team, probability = pick_side(row)
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        market_spread = float(row["spread_line"])
        result, home_score, away_score = outcome_by_game_id.get(game_id, (None, None, None))
        flip_line_value, flip_held_value = _flip_line(
            game_id,
            home_team,
            team,
            market_spread,
            flip_member_ids_by_game.get(game_id, ()),
            artifacts.sweep,
            spread_explorer_params,
        )
        # ``is_game_final`` -- never ``final`` -- the outer scope already
        # binds that name to the (overlaid) predictions DataFrame above.
        is_game_final, cover_result, final_score_text = _game_final_state(
            home=home_team,
            away=away_team,
            pick_team=team,
            market_spread=market_spread,
            result=result,
            home_score=home_score,
            away_score=away_score,
        )
        games.append(
            GameRow(
                game_id=game_id,
                gameday=_parse_gameday(row.get("gameday"), date(1970, 1, 1)),
                weekday_name=str(row.get("weekday") or ""),
                home=home_team,
                away=away_team,
                market_spread=market_spread,
                pick_team=team,
                pick_probability=probability,
                confidence_word=confidence_word(probability),
                is_best=best_pick_id is not None and game_id == best_pick_id,
                is_flipped=game_id in flipped_game_ids,
                flip_member_labels=_flip_member_labels(view, game_id),
                final=is_game_final,
                cover_result=cover_result,
                final_score_text=final_score_text,
                flip_line=flip_line_value,
                flip_held=flip_held_value,
            )
        )

    strong_count = sum(1 for game in games if game.confidence_word == "strong")
    policy, flip_count = _build_policy_note(view, strong_count, len(games))

    paper_decisions = load_paper_decisions(artifacts_root)
    challenger_decisions = load_challenger_decisions(artifacts_root)
    prospective_scoreboard = _build_prospective_scoreboard(
        paper_decisions, challenger_decisions, outcomes
    )
    headline = _build_headline_stats(
        artifacts_root, artifacts.active, prospective_scoreboard=prospective_scoreboard
    )
    season_record = _build_season_record(
        paper_decisions,
        outcomes,
        season=artifacts.metadata.get("season"),
        week=artifacts.metadata.get("week"),
    )

    waterfall_feed = load_waterfall_feed(artifacts_root)
    lineups = load_lineups(artifacts_root)
    validate_lineup_model_sync(lineups, artifacts.predictions)
    dives = tuple(
        _build_dive(
            game,
            sweep=artifacts.sweep,
            waterfall_feed=waterfall_feed,
            spread_explorer_params=spread_explorer_params,
            raw_home_cover_probability=raw_probability_by_game.get(game.game_id),
            lineups=lineups,
        )
        for game in games
    )
    findings = _build_findings(tuple(games), flip_count)

    ticker_chrome = TickerChrome(
        games=tuple(games),
        best_pick_game_id=best_pick_id,
        season=artifacts.metadata.get("season"),
        week=artifacts.metadata.get("week"),
        model_method_label=headline.model_method_label,
        page_command_suffix="",
    )
    # A short, mockup-scale sentence -- NEVER headline.played_card_caption,
    # the long prose sentence the ".headline-main" regression guard forbids
    # anywhere on the page (see tests/test_board_terminal.py's
    # ``test_terminal_headline_main_foot_text_stays_mockup_scale``).
    link_preview = LinkPreview(
        title=f"ATS Terminal — {week_label}",
        description=f"This week's forced-pick board: {headline.played_card_foot_text}.",
    )

    return BoardContent(
        season=artifacts.metadata.get("season"),
        week=artifacts.metadata.get("week"),
        game_type=game_type,
        week_label=week_label,
        generated_at=generated,
        generated_at_text=generated.strftime("%Y-%m-%d %H:%M:%S UTC"),
        games=tuple(games),
        best_pick_game_id=best_pick_id,
        best_pick_note=best_pick_note,
        flip_count=flip_count,
        strong_count=strong_count,
        headline=headline,
        policy=policy,
        dives=dives,
        findings=findings,
        disclaimer=Disclaimer(short=DISCLAIMER_SHORT, full=DISCLAIMER_FULL),
        ticker_chrome=ticker_chrome,
        link_preview=link_preview,
        season_record=season_record,
    )


__all__ = [
    "CADENCE_NOTE",
    "AttributionPanel",
    "AttributionRow",
    "BoardContent",
    "CoverCurvePoint",
    "Disclaimer",
    "Finding",
    "GameDive",
    "GameRow",
    "HeadlineStats",
    "LinkPreview",
    "PolicyNote",
    "ProspectiveScoreboard",
    "SeasonRecordStrip",
    "SpreadAdjusterParams",
    "TickerChrome",
    "load_board_content",
]
