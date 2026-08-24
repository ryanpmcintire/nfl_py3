"""The content model behind "What we've learned" -- every finding, in plain words.

This module is the *text*; :mod:`nfl_ats.public_board` is the layout. Nothing
here imports a web framework and nothing here formats HTML, so the wording can
be reviewed, diffed, and argued about on its own.

The rules the wording follows, because the owner asked for them explicitly:

- No jargon survives contact with this page. "EPA" becomes "how many points a
  play was worth on average"; "Brier score" becomes "how often the picks
  actually landed"; "walk-forward" becomes "scored only on games it had never
  seen". A term is allowed only when the sentence that uses it also explains it
  in football terms.
- Every claim traces to a committed record (``source``). No number is invented,
  softened, or rounded in our favour.
- Negative results are stated proudly. They are the most trustworthy part of
  the record, and there are more of them than positives.
- Untested leads are labelled untested, every time, even when they are
  exciting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nfl_ats.player_arrests_back_side_overlay import (
    POLICY_EFFECT_ACCURACY_POINTS,
    POLICY_GRADED_GAMES,
    POLICY_PROBABILITY_POSITIVE,
)

Verdict = Literal["helps", "no-edge", "unproven", "context"]
ChipKind = Literal["good", "warning", "muted", "plain"]


@dataclass(frozen=True)
class HeadlineNumbers:
    """The active model's grades, in ONE place.

    These used to be typed directly into the prose below, which is exactly how
    they went stale: promoting ``weak_stack`` over ``player`` on 2026-08-18
    changed every one of them and the page kept quoting the old model for
    several commits. Anything that would have to change when the active model
    changes belongs here and nowhere else.

    Update these together with the active model, from the artifact named in
    ``source``.

    Guard, stated accurately (corrected 2026-08-18): the real test is
    ``tests/test_findings_headline.py::test_active_model_grades_are_never_typed_into_the_prose``.
    It asserts that the two literals ``HEADLINE.opener_accuracy`` and
    ``HEADLINE.close_accuracy`` never appear as typed strings in the prose --
    it does NOT check "any bare percentage", and dozens of other bare
    percentages in ``FINDINGS`` carry only a document-level ``source``. This
    docstring previously cited ``tests/test_findings_content.py``, which has
    never existed; do not restore that claim.

    Every field here must come from the run whose
    ``active_model_config.feature_profile`` matches the active model. Mixing a
    point estimate from one run with an interval from another is how
    ``season_low``/``season_high`` went wrong below.
    """

    opener_accuracy: float
    close_accuracy: float
    #: The original protocol grading (sign of the residual). Kept for
    #: provenance: production has always played the probability rule, so the
    #: headline fields above carry the production-rule grades (owner decision,
    #: 2026-08-19; docs/opener_evaluation.md addendum), and these two say what
    #: the frozen sign-rule instrument measured on the same games.
    protocol_opener_accuracy: float
    protocol_close_accuracy: float
    paired_games: int
    first_season: int
    last_season: int
    season_low: float
    season_high: float
    ceiling_low: float
    ceiling_high: float
    source: str

    @property
    def opener(self) -> str:
        return f"{self.opener_accuracy:.1f}%"

    @property
    def close(self) -> str:
        return f"{self.close_accuracy:.1f}%"

    @property
    def protocol_opener(self) -> str:
        return f"{self.protocol_opener_accuracy:.1f}%"

    @property
    def protocol_close(self) -> str:
        return f"{self.protocol_close_accuracy:.1f}%"

    @property
    def edge_points(self) -> str:
        return f"{self.opener_accuracy - 50.0:.1f}"

    @property
    def games(self) -> str:
        return f"{self.paired_games:,}"

    @property
    def seasons(self) -> str:
        return f"{self.first_season}-{self.last_season}"

    @property
    def season_band(self) -> str:
        return f"{self.season_low:.1f}% and {self.season_high:.1f}%"

    @property
    def ceiling(self) -> str:
        return f"{self.ceiling_low:.0f}-{self.ceiling_high:.0f}%"

    @property
    def extra_correct_per_season(self) -> int:
        """Extra correct picks over a coin flip across a 285-game pool season."""

        return round((self.opener_accuracy - 50.0) / 100.0 * 285)


# ---------------------------------------------------------------------------
# Ceiling bands -- the ONE home of every ceiling figure on the site. All
# measured (from doc): docs/pool_edge_plan.md, "The ceiling, and why".
# Every module composes its ceiling prose from these; hand-typing a band
# into prose fails tests/test_number_variables.py.
# ---------------------------------------------------------------------------

#: measured (from doc): pool_edge_plan.md "Practical excellence band for us:
#: 54-55% vs the frozen opener." Also HEADLINE's own ceiling interval below,
#: so the findings hero tile and every repeat render the same band.
PRACTICAL_CEILING_LOW_PCT = 54.0
PRACTICAL_CEILING_HIGH_PCT = 55.0

#: measured (from doc): pool_edge_plan.md omniscient-pregame oracle vs the
#: close: "~55-56% ceiling. Matches the best documented career bettors."
BETTORS_VS_CLOSE_BAND = "55-56"

#: measured (from doc): docs/leak_ceiling_control.md total-leak positive
#: control -- pregame-feature arms B/B2 scored 55.57%/56.05%, quoted in the
#: ladder as "about 56%".
MEASURED_CEILING_PCT = 56

#: measured (from doc): pool_edge_plan.md "Same oracle vs a frozen Tuesday
#: pool line: ~57-58%" -- superseded as a guess by the leak control above,
#: which is why the ladder labels it the pre-measurement guess.
PREMEASUREMENT_GUESS_BAND = "57-58"

#: measured (from doc): floor of pool_edge_plan.md's ~57-58% frozen-Tuesday
#: oracle band; the ceiling card quotes the band's floor ("around 57%").
ORACLE_FROZEN_LINE_PCT = 57

#: measured (from doc): pool_edge_plan.md honesty guardrail -- "any backtest
#: showing 60%+ is a leak, not a breakthrough."
CEILING_BUG_MARK_PCT = 60

#: Active model ``3083f6cbc5e45acb`` (market_residual / weak_stack / ridge /
#: alpha 10.0), promoted 2026-08-18. Opener and close both from the single
#: opener-evaluation run over the paired Tuesday-opener archive
#: (artifacts/opener_evaluation/20260819T174244Z, the first run whose
#: evaluator grades BOTH model rules). Headline = the raw model probability
#: rule (home_cover_probability >= 0.5), now the baseline beneath the coach
#: and player-arrest pick-level policies; protocol_* = the original sign-rule
#: instrument's grades. The promoted policy constants are imported above from
#: the production implementation so public prose cannot drift from its card.
HEADLINE = HeadlineNumbers(
    opener_accuracy=53.4,
    close_accuracy=52.1,
    protocol_opener_accuracy=52.8,
    protocol_close_accuracy=51.6,
    paired_games=1537,
    first_season=2020,
    last_season=2025,
    # Corrected 2026-08-18: 50.2/54.3 was the OLD `player` baseline run's
    # season-blocked interval (artifacts/opener_evaluation/20260817T135624Z,
    # estimate 52.50%), left behind when the point estimates were promoted to
    # the active `weak_stack` run. Pairing this model's estimate with a
    # different model's interval is a provenance error, not a rounding one.
    # Updated 2026-08-19 with the headline-rule switch: these are the
    # PRODUCTION-rule season-blocked bounds from the same run as the point
    # estimate (…20260819T174244Z: opener_accuracy_probability_rule
    # 0.51974/0.54557) -- never pair a probability-rule estimate with the
    # sign rule's interval (…20260818T013115Z's 0.50976/0.54834 was sign-rule).
    season_low=52.0,
    season_high=54.6,
    ceiling_low=PRACTICAL_CEILING_LOW_PCT,
    ceiling_high=PRACTICAL_CEILING_HIGH_PCT,
    source="docs/opener_evaluation.md",
)


# ---------------------------------------------------------------------------
# Played-card expectation (2026-08-23, owner question: "what edge am I
# playing"). The played card is the four-member overlay union (coach fade +
# division revenge + player arrests + spread gap; policy fingerprint
# bbdd60a171238654) plus the POL-11 market-follow refresh rule. Its honest
# forward expectation is a PLANNING SYNTHESIS -- pinned here as a constant and
# never computed from an artifact, so it can never masquerade as a measured
# figure (AGENTS.md: measured and inferred must be distinguishable at a
# glance). Every number below names its provenance:
#
# - PLAYED_CARD_EXPECTATION_PERCENT: de-inflated planning synthesis from
#   docs/overlay_subset_composition.md, "[Inferred] Decision expectation"
#   section (cross-half shrinkage 0.6356 + both holdout directions => roughly
#   +1 accuracy point over the coach->arrest chain).
# - OVERLAY_UNION_PAIRED_*: docs/overlay_subset_composition.md (measured):
#   paired +1.2641383899 accuracy points over the same paired opener archive,
#   probability_positive = 0.85715.
# - OVERLAY_UNION_ARCHIVE_SCORE_FRACTION / OVERLAY_UNION_SUBSET_COUNT:
#   docs/overlay_subset_composition.md (measured): candidate accuracy
#   55.4225% selected as the maximum over 127 correlated subsets -- i.e. a
#   SELECTION-INFLATED archive score, never quotable as an expectation.
# - OVERLAY_SELECTION_RECHECK_*: registry entry ``redteam_overlay_subset_loso_cv``
#   (docs/edge_audit_redteam.md): leave-one-season-out CV of the selection
#   step itself measured 0.0000 pts, P+ 0.4930 -- the out-of-sample re-check
#   already discounted inside the expectation above.
# ---------------------------------------------------------------------------

PLAYED_CARD_EXPECTATION_PERCENT = 55

OVERLAY_UNION_PAIRED_EFFECT_POINTS = 1.2641
OVERLAY_UNION_PAIRED_PROBABILITY_POSITIVE = 0.85715

OVERLAY_UNION_ARCHIVE_SCORE_FRACTION = 0.554225
OVERLAY_UNION_SUBSET_COUNT = 127

OVERLAY_SELECTION_RECHECK_POINTS = 0.0
OVERLAY_SELECTION_RECHECK_P_PLUS = 0.4930

# - MOVEMENT_COMPOSED_*: docs/movement_composition_eval.md results table
#   (registry entry ``movement_rule_composed_chain``, measured): paired
#   +1.5303 accuracy points over the coach->arrest chain on the same 1,503
#   reused games, week-blocked P+ 0.8942 / season-blocked P+ 0.9297 -- an
#   attribution upper bound on already-looked-at data.
# ---------------------------------------------------------------------------

MOVEMENT_COMPOSED_EFFECT_POINTS = 1.5303
MOVEMENT_COMPOSED_WEEK_P_PLUS = 0.8942
MOVEMENT_COMPOSED_SEASON_P_PLUS = 0.9297

#: The page's single crowned hero figure ("≈55%" rendered): an
#: approximation sign on purpose -- planning estimate, not measurement.
PLAYED_CARD_EXPECTATION_HERO = f"\u2248{PLAYED_CARD_EXPECTATION_PERCENT}%"

PLAYED_CARD_EXPECTATION_DEK = "Planning estimate for the played card."

LEDGER_PROMOTED_CAVEAT = (
    "Archive score was selection-inflated; "
    f"played-card expectation {PLAYED_CARD_EXPECTATION_HERO} \u2014 "
    "full ladder on the track-record page."
)

#: Human names for every arm id that can render on the public ledger or the
#: candidate-rules panel (2026-08-24: owner directive -- no internal id may
#: render as a display name). Shared by model_ledger and public_board so the
#: two surfaces can never disagree about what an arm is called. Any id not
#: listed falls back to a humanized form of the id, and the map-coverage
#: test pins that every REGISTERED challenger id has a curated entry.
CHALLENGER_DISPLAY_NAMES: dict[str, str] = {
    "mod07_weak_signal_stack": "Model + seven-rule stat stack",
    "hc_year_one_fade_overlay": "Year-one coach fade",
    "best_pick_nomination_v2": "Best Pick by calibrated probability",
    "best_pick_nomination_v3": "Best Pick v3 ranker",
    "best_pick_big_spread_eligibility": "Best-Pick big-spread eligibility",
    "injury_value_lost_tilt_overlay": "Injury value-lost tilt",
    "division_revenge_tilt_overlay": "Division-revenge tilt",
    "backup_qb_fade_overlay": "Backup-quarterback fade",
    "surface_switch_tilt_overlay": "Turf-surface switch",
    "spread_gap_zone_fade_overlay": "Mid-spread zone fade",
    "smooth_cdf_mapping": "Smooth CDF probability mapping",
    "ecdf_mapping_incumbent": "ECDF probability mapping",
    "era_weighted_half_life_8": "Era-weighted refit (half-life 8)",
    "forecast_cold_visitor_tilt": "Cold-visitor weather tilt",
    "forecast_weather_kn_warm_team_cold_late_tilt": "Warm-team cold-late weather tilt",
    "forecast_weather_kn_precip_high_total_tilt": "Precip + high-total weather tilt",
    "model_only_refresh_incumbent": "Follow line moves \u22651pt",
    "model_only_fresh_incumbent": "Model only \u2014 no fix-up rules",
    "interim_hc_first_game_tilt_overlay": "Interim-coach first game tilt",
    "injury_signal_refresh_tilt": "Injury-news refresh flip",
    "player_arrests_recent_14d_back_side_overlay": "Player-arrests policy",
    "player_arrests_recent_14d_no_overlay_incumbent": "Model without the arrest rule",
    "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1": (
        "Former fix-up-rules card"
    ),
    "overlay_production_chain_coach_arrest_incumbent": "Former coach + arrest chain",
    "movement_rule_composed_v1": "Follow line moves \u22651pt",
    "nflcom_friday_refresh_out2_starters_v1": "Fade 2+ Out designations",
    "player_qb_continuity|ridge_alpha=1|calibration=none": "QB-continuity alpha probe",
}

# ---------------------------------------------------------------------------
# Per-card study figures that would otherwise collide, as typed literals,
# with the canonical headline grades (the full-player backtest below is a
# DIFFERENT measurement from the active close grade; it merely rounds to
# the same one decimal). All measured (from doc); each constant names its
# study so prose composes figures instead of retyping them.
# ---------------------------------------------------------------------------

#: measured (from doc): docs/modeling.md "Player availability and value" --
#: base (market-and-team-form) profile scored 51.08% on the fixed
#: 2018-2025 screen; renders rounded to one decimal.
MARKET_TEAM_FORM_MODEL_PCT = 51.1

#: measured (from doc): docs/modeling.md -- the value-extended full player
#: profile scored 52.14% on the same screen (also ROADMAP.md PER-05 row and
#: docs/data_feasibility.md). NOT the active model's close grade.
FULL_PLAYER_LAYER_PCT = 52.1

#: measured (from doc): ROADMAP.md first player-family ablation --
#: "injury-only reached 51.28%" on the same 2,075 games.
INJURY_ONLY_MODEL_PCT = 51.3

#: measured (from doc): docs/modeling.md learned-availability ATS
#: replacement -- "the candidate reached 52.24% (1,084/2,075) versus 52.14%
#: (1,082/2,075)"; also docs/data_feasibility.md.
LEARNED_AVAILABILITY_BEFORE_PCT = 52.14
LEARNED_AVAILABILITY_AFTER_PCT = 52.24

#: measured (from doc): docs/data_feasibility.md participation-rating screen
#: -- adding the plus/minus contrasts moved accuracy "from 52.14% to
#: 51.71%" (docs/modeling.md: 1,073 of 2,075 non-push games).
PARTICIPATION_RAPM_MODEL_PCT = 51.7


# ---------------------------------------------------------------------------
# End of the pinned-number region. Below this line no canonical accuracy
# figure may appear as a source literal in any dashboard module -- compose
# prose from the named constants above. Guard: tests/test_number_variables.py
# ---------------------------------------------------------------------------


def ladder_rungs(played_chain_accuracy: float | None) -> tuple[str, ...]:
    """The picks-page ceiling-ladder rungs, in FIXED order, as plain text.

    2026-08-23 consolidation law (owner): the index page's DEFAULT view
    carries exactly two accuracy statistics -- the ``≈55%`` planning hero
    and the measured chain history -- so every OTHER accuracy percentage
    lives inside the ONE collapsed ladder ``<details>``. These rungs ARE
    that ladder's entire content, one sentence per rung, in this order and
    no others; :mod:`nfl_ats.public_board` only wraps them in ``<p>`` tags.

    Every number is composed from the pinned constants above or
    :data:`HEADLINE`, never retyped, and the played-chain rung appears only
    when a chain artifact was actually read (``played_chain_accuracy`` is
    not ``None``) -- a missing artifact omits the rung rather than guessing.
    """

    played = f"{played_chain_accuracy:.1%}" if played_chain_accuracy is not None else None
    rungs = [
        # 1. The raw baseline, honestly placed beneath a coin flip.
        f"Coin flip: 50%. Raw model before policy overlays: {HEADLINE.opener} at the "
        f"opener ({HEADLINE.games} games, {HEADLINE.seasons}); {HEADLINE.close} at the "
        "sharper close.",
    ]
    if played is not None:
        # 2. The measured history the crowned hero sits on top of.
        rungs.append(
            f"Played chain (raw \u2192 coach fade \u2192 arrests): {played} measured on "
            f"{POLICY_GRADED_GAMES:,} paired games \u2014 the measured history under the "
            "crowned expectation."
        )
    # 3. The union actually played: paired evidence, selection inflation,
    #    and the out-of-sample re-check that already discounted it.
    rungs.append(
        f"Fix-up rules: paired +{OVERLAY_UNION_PAIRED_EFFECT_POINTS:.2f} points on "
        f"reused data (P+ {OVERLAY_UNION_PAIRED_PROBABILITY_POSITIVE:.3f}); its "
        f"{OVERLAY_UNION_ARCHIVE_SCORE_FRACTION:.1%} archive score is selection-inflated "
        f"(best of {OVERLAY_UNION_SUBSET_COUNT} subsets), and the out-of-sample re-check "
        f"of that selection measured {OVERLAY_SELECTION_RECHECK_POINTS:.2f} pts "
        f"(P+ {OVERLAY_SELECTION_RECHECK_P_PLUS:.2f}) \u2014 already discounted in the "
        f"{PLAYED_CARD_EXPECTATION_HERO} expectation."
    )
    # 4. The refresh rule, as an attribution upper bound only.
    rungs.append(
        "Movement rule (market-follow on >=1pt moves via refresh): composed "
        f"+{MOVEMENT_COMPOSED_EFFECT_POINTS:.2f} points "
        f"(P+ {MOVEMENT_COMPOSED_WEEK_P_PLUS:.2f}/{MOVEMENT_COMPOSED_SEASON_P_PLUS:.2f}) "
        "\u2014 an attribution upper bound on already-looked-at data."
    )
    # 5. The measured ceiling, replacing the pre-measurement guess.
    rungs.append(
        f"Best documented long-run bettors: roughly {BETTORS_VS_CLOSE_BAND}% against "
        f"the close. Measured pregame ceiling: about {MEASURED_CEILING_PCT}% (total-leak "
        f"control, docs/leak_ceiling_control.md); the older {PREMEASUREMENT_GUESS_BAND}% "
        "band was the pre-measurement guess."
    )
    # The binding closing hedge (AGENTS.md research framing).
    rungs.append(
        "A small step above a coin flip is not proof of a stable, profitable edge "
        "\u2014 sportsbook vig alone would likely erase it. These are forced "
        "paper picks \u2014 not a game-level probability, not a profit claim."
    )
    return tuple(rungs)


@dataclass(frozen=True)
class Finding:
    """One question a person might ask, and the honest answer to it.

    Curation metadata, added so the page can never silently go stale again
    (see :mod:`nfl_ats.findings_registry`): every finding either names the
    live registry entries its numbers were verified against
    (``registry_keys``, with a parallel ``registry_fingerprints`` snapshot
    taken on ``curated_as_of``) or declares itself ``evergreen`` -- a
    methodology explainer with no single number that could go stale. A build
    fails loudly, naming the finding and the key, the moment either drifts:
    a key that stops existing, or one whose recorded content moves out from
    under the prose. Fingerprints are opaque on purpose -- they are never
    hand-computed; see ``scratchpad`` tooling notes in
    ``docs/findings_generation.md`` for how to regenerate them after a real
    correction.
    """

    question: str
    verdict: Verdict
    plain_answer: str
    detail: str
    source: str
    registry_keys: tuple[str, ...] = ()
    registry_fingerprints: tuple[str, ...] = ()
    curated_as_of: str | None = None
    evergreen: bool = False


@dataclass(frozen=True)
class LeadBlurb:
    """A hand-picked plain-English one-liner for one of the highest-ranked
    entries in findings.html's "What we're watching" section (the open,
    ``unresolved_below_power`` leads auto-rendered straight from
    ``registry/weak_signals.json`` -- see ``nfl_ats.findings_registry.top_open_leads``).

    Most leads need no curation at all: the registry's own ``description``
    field is already a written sentence, if a research-toned one, and
    rendering it verbatim is the whole point of that section (zero prose to
    write, zero key to wire -- see ``docs/site_content_pipeline.md``). This
    exists only for the small number of leads worth a hand-written, plainer
    sentence instead.

    It carries the EXACT SAME freshness contract as a curated ``Finding``,
    and is validated by the SAME function
    (``nfl_ats.findings_registry.validate_curation``, which only reads
    ``question``/``evergreen``/``registry_keys``/``registry_fingerprints`` off
    whatever it is given -- seeing this dataclass as `Finding`-shaped is the
    intended reuse, not a workaround). Never hand-curate a lead without going
    through that same validation call in ``public_board.render_findings_page``.
    """

    weak_signal_name: str
    text: str
    curated_as_of: str
    registry_fingerprints: tuple[str, ...]
    evergreen: bool = False

    @property
    def registry_keys(self) -> tuple[str, ...]:
        return (f"weak_signal:{self.weak_signal_name}",)

    @property
    def question(self) -> str:
        """Read only by ``validate_curation``'s error messages -- never rendered."""

        return f"lead blurb for weak_signal:{self.weak_signal_name}"


@dataclass(frozen=True)
class VerdictGroup:
    """A section of the page: one verdict, its framing, and its chip."""

    verdict: Verdict
    kicker: str
    title: str
    blurb: str
    chip_label: str
    chip_kind: ChipKind
    legend: str


@dataclass(frozen=True)
class HeadlineTile:
    """One hero statistic."""

    kicker: str
    value: str
    context: str
    delta_text: str | None = None
    delta_good: bool | None = None


@dataclass(frozen=True)
class HonestyRule:
    """One rule we hold ourselves to when reporting a number."""

    title: str
    body: str


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

HERO_KICKER = "What we've learned"
HERO_TITLE = "Everything the research has settled, in plain English"
HERO_SUB = (
    "Every finding states its evidence and how confident we are. Every answer below is "
    "either something we measured on games the model had never seen, or something we tell "
    "you outright that we have not measured yet."
)

HERO_TILES: tuple[HeadlineTile, ...] = (
    HeadlineTile(
        kicker="Model baseline at the pool's line",
        value=HEADLINE.opener,
        context=(
            f"{HEADLINE.games} games, {HEADLINE.seasons}, every one scored by a model "
            "that never saw the result. This is the raw probability-rule baseline beneath "
            "the live coach and player-arrest policies. The "
            "original protocol grading (sign of the residual, a rule no published pick "
            f"ever used) scores {HEADLINE.protocol_opener} on the same games."
        ),
        delta_text=f"{HEADLINE.edge_points} points better than a coin flip",
        delta_good=True,
    ),
    # The closing-line grade is NOT tiled here anymore (owner law, 2026-08-23:
    # each canonical stat renders as a figure only on its home page -- the
    # close grade lives on track_record.html). The drift story it told is
    # kept in words by the second finding below.
    HeadlineTile(
        kicker="A realistic ceiling",
        value=HEADLINE.ceiling,
        context=(
            "What an excellent NFL model can hope for against a frozen line. Anything "
            f"near {CEILING_BUG_MARK_PCT}% is a bug in the test, not a breakthrough."
        ),
    ),
)

HERO_PARAGRAPHS: tuple[str, ...] = (
    "The pool locks every pick Tuesday at noon against a frozen early-week spread, and "
    f"everyone picks every game. Graded exactly that way -- {HEADLINE.games} games, "
    f"{HEADLINE.first_season}-{HEADLINE.last_season}, all scored by a model that never saw "
    f"the result -- the raw model baseline took the right side {HEADLINE.opener} of the "
    f"time. Its honest season-blocked range ({HEADLINE.season_band}) sits entirely "
    f"above the coin flip, worth roughly {HEADLINE.extra_correct_per_season} more correct "
    "picks than a coin flip across a 285-game season.",
    f"The card now adds the player-arrest policy after the coach policy. In its frozen "
    f"opener evaluation it finished above the model baseline on "
    f"{POLICY_GRADED_GAMES:,} graded games (+{POLICY_EFFECT_ACCURACY_POINTS:.3f} accuracy "
    f"points, P+ {POLICY_PROBABILITY_POSITIVE:.2f}); the paired "
    "accuracy figures themselves are home on the track-record page. That is the "
    "higher-expected-value side of a forced decision, not a resolved-effect claim; the "
    "former coach-only card remains a paired prospective control.",
    "That edge is smaller than it sounds and bigger than it looks. Smaller, because "
    "even the arrest evaluation's grade (track-record page) still loses a lot of "
    "Sundays and always will. Bigger, because the "
    f"practical ceiling here is around {HEADLINE.ceiling_high:.0f}%, so we are already about "
    "halfway from a coin flip to the limit of what anyone does. Most of what follows is the "
    "things that did not work on the way here -- not failures we are hiding, but the reason "
    "the number above is believable.",
)

LEGEND_KICKER = "How to read the labels"


# ---------------------------------------------------------------------------
# Groups, in page order
# ---------------------------------------------------------------------------

GROUPS: tuple[VerdictGroup, ...] = (
    VerdictGroup(
        verdict="helps",
        kicker="What actually works",
        title="The findings we act on",
        blurb=(
            "Each of these was tested on games the model had never seen, with the test "
            "written down before it ran, and each earns its place in what we actually "
            "play. That is not the same as proven -- see the note below on why we "
            "separate the two."
        ),
        chip_label="we act on this",
        chip_kind="good",
        legend="Tested out of sample and used in what we play. Strength varies; read each one.",
    ),
    VerdictGroup(
        verdict="unproven",
        kicker="Promising, not proven",
        title="Leads we have not validated yet",
        blurb=(
            "Each of these points the right way, or has support in published research. "
            "None has cleared the bar we set for CLAIMING a result. That is a separate "
            "question from whether we play it: the pool makes us submit 285 picks either "
            "way, so we back the better side of an uncertain bet and say plainly that it "
            "is uncertain. One of these is in the model we run today for exactly that "
            "reason. We list them here so nobody -- us included -- quietly promotes one "
            "to a finding by talking about it enough."
        ),
        chip_label="untested lead",
        chip_kind="warning",
        legend="Points the right way; not proven. Some we play anyway -- each says which.",
    ),
    VerdictGroup(
        verdict="no-edge",
        kicker="Tested, and no",
        title="Good ideas that turned out not to help",
        blurb=(
            "The biggest section on the page, deliberately. Every one of these was built "
            "and measured, and most fail for the same interesting reason: the betting "
            "market already knew. A caveat we added on 2026-08-18, after re-auditing our "
            "own measuring tools: for several of these the honest answer is 'too small "
            "for us to detect', not 'proven not to work'. We now say which is which "
            "rather than filing both as closed."
        ),
        chip_label="no edge found",
        chip_kind="muted",
        legend=(
            "Built it, measured it, it did not help -- or was too small to tell. "
            "Recorded, not deleted."
        ),
    ),
    VerdictGroup(
        verdict="context",
        kicker="How this actually works",
        title="Structural truths worth knowing before you read a percentage",
        blurb=(
            "Not results -- the shape of the problem. These are the facts that decide what any "
            "number on this dashboard is allowed to mean."
        ),
        chip_label="how it works",
        chip_kind="plain",
        legend="A fact about the problem rather than a result.",
    ),
)


# ---------------------------------------------------------------------------
# The findings
# ---------------------------------------------------------------------------

DETAIL_SUMMARY_LABEL = "How we know"
SOURCE_LABEL = "Source"

#: The day every ``registry_keys``/``registry_fingerprints`` pair below was
#: verified against the live registries (2026-08-19 audit: wired keys onto
#: every non-evergreen finding, corrected six stale facts found in the
#: process -- see ``docs/findings_generation.md``). A future correction only
#: needs to update the SPECIFIC finding it touches, not this constant.
_CURATED_AS_OF = "2026-08-19"

FINDINGS: tuple[Finding, ...] = (
    # -- helps ---------------------------------------------------------------
    Finding(
        question="Do our picks beat a coin flip against the line the pool actually uses?",
        verdict="helps",
        plain_answer=(
            f"The raw model baseline did, by about {HEADLINE.edge_points} points. On "
            f"{HEADLINE.games} games from "
            f"{HEADLINE.first_season} through {HEADLINE.last_season} -- "
            "every one of them scored by a model that had never seen the result -- it "
            "landed at the opener baseline shown at the top of this page, where a coin "
            "flip gets 50%. The promoted player-arrest policy separately finished above "
            "that same baseline in its frozen evaluation (the arrest evaluation is home "
            "on the track-record page). The "
            "baseline finished above 50% in all six seasons; the composed live "
            "policy continues to be tracked prospectively."
        ),
        detail=(
            "The season table reports every year rather than hiding the weakest one. We ran "
            "the baseline measurement exactly once, with the model frozen and the scoring "
            "rules written down beforehand, so nothing was adjusted after the number came "
            "back. The arrest-policy comparison remains unresolved and prospectively paired "
            "against the former coach-only card."
        ),
        source="docs/opener_evaluation.md; docs/player_arrests_back_side_overlay.md",
        evergreen=True,  # driven by HEADLINE, not a registry key; see test_findings_headline.py
    ),
    Finding(
        question="Does it matter which line we are graded against?",
        verdict="helps",
        plain_answer=(
            "More than anything else we have found. The same picks on the same games "
            "score at the opener baseline shown at the top of this page against "
            "Tuesday's opening line, and only at the close grade -- home on the "
            "track-record page -- against the line the market "
            "settles on by Sunday. The market spends the week drifting toward our number, and "
            "because the pool freezes its spread on Tuesday and never moves it, that drift "
            "gets handed straight back to us as accuracy."
        ),
        detail=(
            "The gap is 1.35 points, and it survives re-scoring the games in whole-week and "
            "whole-season chunks: about a 99.9% chance it is real. It also fixed a mistake we "
            "had been making for a year -- grading ourselves at the closing line, the way "
            "serious bettors do, was quietly understating the model against the only line we "
            "are actually judged on. For scale: someone who could perfectly foresee "
            "Wednesday-to-Sunday line movement, with zero football knowledge, would score "
            "55.1% against that frozen Tuesday number. We capture roughly half of that."
        ),
        source="docs/opener_evaluation.md",
        evergreen=True,  # driven by HEADLINE, not a registry key; see test_findings_headline.py
    ),
    Finding(
        question="Is it better to correct the market's number than to predict the game ourselves?",
        verdict="helps",
        plain_answer=(
            "Much better, and we tried both. Predicting the final margin from football data "
            "alone and then comparing it to the spread is worse than starting from the spread "
            "and predicting only how wrong it is. The correcting version misses the real "
            "margin by about 9.9 points on average; the from-scratch version misses by 10.1. "
            "That gap sounds trivial and is not: a fifth of a point of line accuracy is most "
            "of the distance between a working model and a useless one."
        ),
        detail=(
            "Everything we ship is built this way -- the market line is the starting "
            "assumption and the model estimates only the correction to it. We confirmed the "
            "same thing on college football, where we have almost three times as many games: "
            "the correcting model picked 51.6% of 8,933 games against 49.6% for a control "
            "that just follows the market. That college benchmark exists precisely because it "
            "can resolve effects of around one accuracy point that our 2,000-game NFL sample "
            "cannot."
        ),
        source="docs/modeling.md, ROADMAP.md (XLG-03)",
        evergreen=True,  # foundational modeling comparison, predates the weak-signal registry
    ),
    Finding(
        question="Does knowing who is hurt and who is starting at quarterback help?",
        verdict="helps",
        plain_answer=(
            "Modestly, and less than we first believed. The player layer -- quarterback form, "
            "the injury report weighted by how much each player actually plays, and how much "
            "the lineup has churned -- is in the model we run, and on matched games it is "
            "worth about a point of accuracy over the same model without it. But when we "
            "re-tested its most promising slice on seasons it had never been scored on, it "
            "added exactly nothing. So we keep it, and we do not brag about it."
        ),
        detail=(
            f"The numbers: {MARKET_TEAM_FORM_MODEL_PCT:.1f}% for a market-and-team-form "
            f"model, {FULL_PLAYER_LAYER_PCT:.1f}% with the full player layer, on the same "
            f"2,075 games; injury information on its own reached {INJURY_ONLY_MODEL_PCT:.1f}%. "
            "The "
            "'quarterback plus lineup continuity' variant looked like the best thing in the "
            "entire search at 52.3-52.6%, then scored +0.00 points on 997 untouched 2014-2017 "
            "games, splitting 88-88 on the games where it disagreed with the simpler model. "
            "Injury values are read from the report as it stood 24 hours before kickoff, "
            "never from what we learned afterwards, and they are split by unit -- offensive "
            "line, skill positions, front seven, secondary -- so each can be judged alone."
        ),
        source="docs/modeling.md, ROADMAP.md (PER-02/03/05)",
        registry_keys=("rotation:player_qb_continuity",),
        registry_fingerprints=("3a719416f790fb7e",),
        curated_as_of=_CURATED_AS_OF,
    ),
    # -- unproven ------------------------------------------------------------
    Finding(
        question="Can we learn who will actually play, instead of trusting the injury label?",
        verdict="unproven",
        plain_answer=(
            "This is our most promising unfinished idea. Rather than hand-assigning "
            "'questionable means a 35% chance of missing', we learned the real rates from "
            "sixteen years of injury reports and practice attendance. They are very "
            "different: a questionable player who practised fully misses about 20% of the "
            "time, one who did not practise at all misses about 60%. Predicting availability "
            "got clearly better. Whether that turns into better picks is still open."
        ),
        detail=(
            "Measured on 57,294 player-games from seasons the model was not fitted on, our "
            "error in predicting whether a player would take a snap fell by about 5%. Carried "
            "through into picks it moved accuracy from "
            f"{LEARNED_AVAILABILITY_BEFORE_PCT:.2f}% to {LEARNED_AVAILABILITY_AFTER_PCT:.2f}% "
            "-- two extra correct "
            "games out of 2,075, with a range around that change of -0.6 to +0.8 points. The "
            "best summary we have is roughly a 61% chance it helps at all, which is why it "
            "sits here as a weak positive to be combined with other weak positives rather "
            "than promoted on its own. Doubtful players who do not practise miss about 98% of "
            "the time, not the 85% our old hand-written table assumed."
        ),
        source="docs/modeling.md, docs/pool_edge_plan.md",
        registry_keys=("weak_signal:learned_availability_ats_2018_2025",),
        registry_fingerprints=("d289cb86e8c6b98f",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Do bookmakers make predictable mistakes when they first post a line?",
        verdict="unproven",
        plain_answer=(
            "Published research says they might, and a pool that locks on Tuesday is exactly "
            "where that would pay. Four leads: teams coming off a playoff run look overrated "
            "in Week 1 (one study has them covering only 35.6% of the time), Week 2 lines stay "
            "anchored to Week 1's, last week's result gets over-weighted, and games nobody is "
            "watching move the most once money arrives. All four have been built and "
            "measured. Jointly, "
            "the first three moved accuracy by +0.22 points, a coin-flip's worth of confidence "
            "(P+ 0.505); the playoff-holdover claim specifically does not "
            "reproduce in our data on its own. None of the four is proven. None is refuted "
            "either -- these are recorded, watched leads, not a closed question."
        ),
        detail=(
            "The 2026-08-18 test ablated the three built features (playoff holdover, "
            "prior-week ATS, week-2 anchoring) jointly inside the promoted weak-signal stack, "
            "on the same 456-game opener window MOD-07 used: +0.2193 accuracy points, "
            "P+ 0.505, interval [-2.66, +3.24] -- indistinguishable from "
            "noise at this sample size. The playoff-holdover claim was also tested directly, "
            "on its own: our replication of the published 35.6%-cover claim came back -3.6 "
            "points with a very wide interval, no usable direction. The fourth lead -- "
            "low-attention games moving most -- now has its own instrument, a Wikipedia-"
            "pageview attention-proxy battery; its closest cell (both teams cold) leans the "
            "hypothesized way (P+ 0.86) but the interval still crosses zero. "
            "Every one of these stays recorded and open rather than closed, per the project's "
            "own rule that a crossing-zero interval is the expected shape for a real small "
            "signal, not a verdict."
        ),
        source="docs/pool_edge_plan.md, ROADMAP.md, docs/mod07_stack.md",
        registry_keys=(
            "weak_signal:mod07_opener_bias_ablation",
            "weak_signal:mod07_holdover_bias_replication",
            "weak_signal:attention_battery_both_cold",
        ),
        registry_fingerprints=(
            "ff1713d32863f7c2",
            "eb03693fbd393a06",
            "054b9409261e93c5",
        ),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Can a pile of weak signals add up to one strong one?",
        verdict="unproven",
        plain_answer=(
            "That is the plan, and it has not been tested with one predeclared, scored look "
            "yet. We have several faintly-positive ideas that cannot be proven alone -- "
            "learned availability, playing-time-weighted injuries, the opening-line biases "
            "above, and roughly fifty more patterns from a recent overnight sweep. Combining "
            "them into one candidate and judging that once is legitimate (averaging noisy "
            "signals cancels some noise); doing it on already-mined seasons would prove "
            "nothing."
        ),
        detail=(
            "The rule we have committed to: each new family of ideas draws a window of "
            "seasons from 2009-2025 that it has never touched, trains only on earlier games, "
            "and gets exactly one scored look. That registry now holds 143 recorded results "
            "(measured 2026-08-19, `nfl-ats weak-signals status`), each with its direction, "
            "its uncertainty, and whether it is even eligible to be pooled. Running the "
            "pooling tool across the 84 NFL, accuracy-points-scaled entries eligible today -- "
            "a diagnostic reading, not the predeclared combined test -- comes back close to a "
            "coin flip: -0.003 accuracy points, 95% [-0.035, +0.028], sign test 40-of-84 "
            "favouring the candidate direction (p=0.74, 'consistent with a coin flip'). That "
            "is not the letdown it looks like. Every screening sweep floods the registry with "
            "dozens of exploratory, individually-unremarkable results, which pulls a pooled "
            "average toward zero even as it hands us more good raw material to rank and "
            "choose from than we had before -- the pile got diluted, not weaker. The value in "
            "the registry is the ranked leads worth a second look (see 'What we're watching' "
            "below, generated fresh from this same file), not this pooled average, which was "
            "never a finding on its own and is not one now. This number moves every time a "
            "new result is recorded, so treat it as a live reading -- re-run `nfl-ats "
            "weak-signals pool --league nfl --effect-units accuracy_points` for the current "
            "figure rather than trusting a number quoted here. What still has to happen is "
            "the one predeclared, scored pooling of a CHOSEN subset of that pile before any "
            "combined candidate can be judged honestly."
        ),
        source="docs/pool_edge_plan.md (MOD-07), registry/weak_signals.json",
        evergreen=True,  # a live command's output, not a single registry entry to fingerprint
    ),
    # -- no-edge -------------------------------------------------------------
    Finding(
        question="Does grading every single play tell us who is really better?",
        verdict="no-edge",
        plain_answer=(
            "It tells us plenty about football and nothing the line does not already know. We "
            "built the full play-by-play layer -- how many points each play was worth on "
            "average given down, distance and field position, how often teams stayed ahead of "
            "schedule, explosive plays, pressure allowed -- 48 columns of it. Added to the "
            "model it made the picks slightly worse. Adjusting it for opponent strength made "
            "them worse again."
        ),
        detail=(
            "This is the finding people find hardest to accept, so here is its strongest "
            "version. Re-tested on 1,247 games from 2013-2017 that it had never been "
            "scored on, with the test declared first, it came back at -0.08 accuracy points "
            "against the simpler model. That is noise, not a refutation: this evaluator "
            "resolves about 3.40 points at this sample size (paired standard error 1.21), so "
            "-0.08 sits far inside the margin of pure chance (P+ 0.474, "
            "essentially a coin flip). The original write-up called the margin error "
            "'resolvably worse' -- that leaned on a secondary, direction-only endpoint the "
            "test's own predeclaration had explicitly ruled out as a pass/fail gate, so the "
            "verdict is now recorded as unresolved_below_power, not a closed negative. An "
            "earlier look at more recent seasons had shown +1.69 points -- that number is now "
            "on the record as an example of what happens when you compare enough versions on "
            "the same years. The layer stays in the codebase for future work on how games are "
            "shaped; it is not, so far, a source of edge, and it is not proven never to be one."
        ),
        source="ROADMAP.md, docs/modeling.md",
        registry_keys=("rotation:pbp_drive_bundle",),
        registry_fingerprints=("7505bed89085a09d",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="What about drive-level stats -- points per possession, field position?",
        verdict="no-edge",
        plain_answer=(
            "Same answer, one level up. We built a possession table -- points per drive, "
            "yards, plays, time of possession, scoring rate, turnover rate, and the same "
            "figures allowed to opponents -- and added it to a model that already had the "
            "play-level data. The picks got no better and the probability estimates got very "
            "slightly worse, with the range around that change straddling zero. Not a loss; "
            "just nothing there."
        ),
        detail=(
            "The null makes mechanical sense: drive statistics are mostly a re-summary of "
            "plays we had already counted, so the information was in the model twice. The "
            "layer is kept and reproducible because a future model of how games are shaped -- "
            "scores, pace, totals -- will want possessions as a building block. It is closed "
            "as an accuracy idea, and it will not be re-tuned on the same seasons."
        ),
        source="ROADMAP.md (PBP-03), docs/modeling.md",
        evergreen=True,  # not tracked as its own registry entry
    ),
    Finding(
        question="Does a team that keeps its lineup together beat the number?",
        verdict="no-edge",
        plain_answer=(
            "It looked that way, and then it did not. Lineup continuity -- the share of last "
            "week's starters who are back -- was the strongest player-side signal in our first "
            "search, apparently worth about a point of accuracy. We froze that exact recipe "
            "and scored it on four seasons it had never been tested on. On the 176 games where "
            "it disagreed with the simpler model it went 88-88. Exactly break-even."
        ),
        detail=(
            "That is the cleanest lesson this project has learned about itself. Run enough "
            "versions against the same eight seasons and one of them will look excellent "
            "whether or not anything is there. The fix was not a better continuity feature, "
            "it was a rule: every new family now earns its verdict on seasons it has never "
            "touched. The 2014-2017 window is marked spent for the player family and cannot "
            "be reused, which is a real cost we accepted in exchange for one trustworthy "
            "answer. Recorded verdict: unresolved (P+ 0.50, exactly a coin "
            "flip) -- an honest null, not a refutation; nothing here claims the mechanism is "
            "wrong, only that this specific recipe added nothing on these games."
        ),
        source="ROADMAP.md, docs/modeling.md",
        registry_keys=("rotation:player_qb_continuity",),
        registry_fingerprints=("3a719416f790fb7e",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Does it hurt a team when the players who normally carry the load are missing?",
        verdict="no-edge",
        plain_answer=(
            "Probably, and the line probably already knows -- but this is softer than we once "
            "said. We tested it on college football, where we have 8,933 clean games instead "
            "of 2,000, by tracking whether the players who normally take a team's snaps at "
            "quarterback and running back had actually played in the most recent game. The "
            "screen found teams missing their usual load-carriers played worse, and "
            "telling the model about it made picks worse by about two thirds of a point. "
            "That result sits below what this instrument can actually resolve at that sample "
            "size, so the honest reading is not-yet-confirmed, not disproven."
        ),
        detail=(
            "Re-measured against the frozen CFB benchmark, the construct (participation-"
            "continuity at the skill positions, season-scoped) now reads -0.101 accuracy "
            "points, P+ 0.35, interval [-0.63, +0.44] -- crossing zero, "
            "recorded unresolved_below_power, not closed. The market-pricing story is still "
            "the leading explanation: college spreads move on quarterback and lead-back news "
            "exactly like NFL ones, so by the time we see a line the disruption is likely "
            "already inside it. We had to answer a prerequisite first -- telling a temporary "
            "absence from a permanent departure -- and that study found only about 16-19% of "
            "college players holding a real role at the end of a season ever appear for that "
            "team again, which is why the feature counts only players who have already played "
            "this season. No NFL window has been spent on this mechanism either way; it is "
            "open, not closed, and nothing here recommends building it next."
        ),
        source="docs/cfb_role_features.md",
        registry_keys=("weak_signal:cfb_role_continuity", "rotation:cfb_role_continuity"),
        registry_fingerprints=("f922f22061c1c508", "282f6629c405ef76"),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Can we rate players by what happens while they are on the field?",
        verdict="no-edge",
        plain_answer=(
            "We tried the basketball trick -- rate every player by how the team does with him "
            "out there, adjusted for who else is on the field -- using ten seasons of "
            "play-by-play participation data. It made the picks measurably worse than the "
            "model without it, with the probability estimates degrading too. Football "
            "gives you sixty snaps a game with twenty-two players on the field, and the "
            "plus-minus maths that works in basketball starves on that."
        ),
        detail=(
            f"Head to head on the same games: {PARTICIPATION_RAPM_MODEL_PCT:.1f}% against "
            f"{FULL_PLAYER_LAYER_PCT:.1f}% for the model without it. The fit used "
            "competitive eleven-on-eleven plays only, three-season rolling "
            "windows, heavy shrinkage toward the average, and an extra discount for players "
            "with few snaps. It still had to rate between 1,758 and 2,872 players a season "
            "from as few as 24,000 plays. Narrower versions -- position groups, formations, "
            "specific unit matchups -- remain possible; this particular formulation is closed "
            "and will not be re-tuned on the same seasons."
        ),
        source="docs/data_feasibility.md, ROADMAP.md (PER-09)",
        registry_keys=("weak_signal:participation_offense_defense_rapm",),
        registry_fingerprints=("33e8e6280ba1c728",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Can we tell in advance which games will be blowouts and which will be close?",
        verdict="no-edge",
        plain_answer=(
            "No better than assuming every game has roughly the usual spread of outcomes. We "
            "built a model to predict how far a game would land from its expected margin -- "
            "using mismatch size, the total, how fast the teams play, and how early in the "
            "season it is -- and used it to widen or narrow each game's range of likely "
            "results. The predictions got measurably worse. Our plain one-size-fits-all range "
            "was already close to right."
        ),
        detail=(
            "We ran this on college football first, deliberately, because 8,933 games can "
            "resolve a difference this small and 2,000 cannot. The scale model did produce "
            "real variation -- the games it called widest came out roughly 20% wider than the "
            "ones it called narrowest -- and it still lost. That is a compliment to the "
            "simple version: the scatter "
            "of results around a good prediction is remarkably stable in football, and our "
            "50% and 80% ranges already land at almost exactly the rates they claim."
        ),
        source="docs/margin_variance.md",
        evergreen=True,  # not tracked as its own registry entry
    ),
    Finding(
        question="Does a web of who-beat-whom rank teams better than a plain rating?",
        verdict="no-edge",
        plain_answer=(
            "No. We built the season as a network -- beat a team that beat a good team and "
            "credit flows back to you, the same maths that once ranked web pages -- and "
            "compared it against an ordinary strength rating built from the identical games. "
            "Across eight test seasons the network version was chosen zero times. The plain "
            "rating was chosen three times, and the model with neither was chosen five."
        ),
        detail=(
            "The network version also made probability estimates and margin errors worse in "
            "most seasons, which is the tell that it was not merely unlucky. It is a "
            "genuinely elegant idea that turns out to measure roughly what an ordinary rating "
            "measures, with many more moving parts. Both are kept as research tools; neither "
            "is a default in anything we run."
        ),
        source="docs/modeling.md (MOD-15)",
        registry_keys=("weak_signal:graph_schedule_rating_brier",),
        registry_fingerprints=("4010dd0e54c471a8",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Do players in a contract year try harder?",
        verdict="no-edge",
        plain_answer=(
            "**Essentially no** -- at least not in a way that reaches the scoreboard. "
            "It is exactly the sort of human factor a "
            "market of numbers might miss, but the research on an NFL contract-year effect is "
            "close to nothing, and our own screen agrees. It has no place in the model and "
            "will not get one as a "
            "standalone idea."
        ),
        detail=(
            "It survives only as one weak input among several in a future combined candidate, "
            "where nothing has to be individually convincing to be worth carrying. This is "
            "what an honest 'no' looks like when someone else has already done the work: we "
            "did not spend one of our own limited untouched test windows re-discovering it."
        ),
        source="docs/pool_edge_plan.md",
        evergreen=True,  # published literature, no internal registry entry
    ),
    # -- context -------------------------------------------------------------
    Finding(
        question="How much of the answer is already in the betting line?",
        verdict="context",
        plain_answer=(
            "Almost all of it. The spread is the output of a market full of people who are "
            "very good at this and who move money the moment they disagree, and it is by far "
            "the strongest single input we have. Everything else in the model exists to nudge "
            "it, usually by a fraction of a point. If you take one idea from this page, take "
            "this one: we are not trying to predict football games. We are trying to find the "
            "few games a week where a very good number is slightly wrong."
        ),
        detail=(
            "In practice the model treats the line as its starting assumption and estimates "
            "only the correction, and those corrections are small -- half a point of "
            "disagreement is a big disagreement by our standards. Alongside the line it "
            "carries team form (a weighted average of recent scoring, turnovers, sacks and "
            "past cover margins that leans on the last few games), a team-strength rating, "
            "quarterback state, the injury report, rest, and basic schedule context."
        ),
        source="docs/modeling.md",
        evergreen=True,  # methodology explainer, no single registry entry
    ),
    Finding(
        question="What is the best we could possibly do?",
        verdict="context",
        plain_answer=(
            f"About {HEADLINE.ceiling}, and {CEILING_BUG_MARK_PCT}% would mean we have a "
            "bug. Final scores scatter around even "
            "a perfect pregame prediction by about 13 points -- turnovers bounce, kickers "
            "miss, one-score games turn on one call. That noise sets the limit. The exchange "
            "rate is worth memorising: one point of line accuracy is worth about three points "
            "of pick accuracy, which is why tiny modelling gains matter and why huge results "
            "are impossible."
        ),
        detail=(
            f"A perfect pregame oracle graded against a frozen Tuesday line would land around "
            f"{ORACLE_FROZEN_LINE_PCT}%; against the sharper closing line, about "
            f"{BETTORS_VS_CLOSE_BAND}%. Documented career bettors "
            "live in the mid-50s. So the ceiling is not physical, it is adversarial -- it "
            "measures how much better than the line-setter we can be -- and the pool's "
            "line-setter is handicapped on purpose by freezing its number midweek. That "
            "handicap is the entire opportunity. The corollary is a rule we actually "
            "enforce: any backtest showing "
            f"{CEILING_BUG_MARK_PCT}% is a leak, and we have thrown out results for being too "
            "good."
        ),
        source="docs/pool_edge_plan.md",
        evergreen=True,  # ceiling arithmetic, no single registry entry
    ),
    Finding(
        question="Are the picks we feel best about more likely to win?",
        verdict="context",
        plain_answer=(
            "No, and this one stings. Our single most confident pick of the week -- the game "
            "where the model disagrees most with the line -- has won 48.6% of the time across "
            "107 weeks. Sorting every pick into four confidence groups gives 53.2%, 47.3%, "
            "55.7%, 53.7%: no ladder, just noise. The model's answer to 'which side' carries "
            "the signal. Its answer to 'how sure are you' does not."
        ),
        detail=(
            "So until that changes, every pick is weighted equally, and the pool's one Best "
            "Pick per week costs nothing whichever game we assign it to -- they are all "
            "worth about the same as the opener baseline at the top of this page. Finding a "
            "confidence measure that genuinely "
            "ranks pick "
            "quality would be free points and it is high on the queue. The candidates: using "
            "the calibrated chance of covering rather than raw disagreement with the line, "
            "accounting for the key numbers 3 and 7 where NFL margins pile up, and calibrating "
            "separately by era. None of them are tested. This measurement was read-only -- no "
            "pick was selected on it -- which is why we can quote it at all."
        ),
        source="docs/opener_evaluation.md, docs/pool_edge_plan.md",
        evergreen=True,  # a read-only historical measurement, not a registry entry
    ),
    Finding(
        question="What can't we see?",
        verdict="context",
        plain_answer=(
            "The things people who beat this market for a living actually use. We have no film "
            "-- we cannot see a guard's footwork or a blown coverage. We have no locker room, "
            "no beat-writer whispers, no coach quietly telling someone his starter is a decoy. "
            "We cannot see where the informed money is going, only where the line ended up. "
            "And we cannot see the weather forecast that existed when the line was posted, "
            "only the weather that happened."
        ),
        detail=(
            "We budget for this rather than pretend otherwise. Of the three to four accuracy "
            "points between us and a perfect pregame prediction, we concede about two to "
            "information we structurally cannot get. The reachable part -- maybe half a point "
            "to a point -- is better handling of thin evidence: what a backup quarterback is "
            "worth when we have seen twenty snaps of him, what a team is worth in Week 2, and "
            "borrowing college football as a starting guess where our NFL evidence runs out. "
            "That is the honest map of where the remaining room is."
        ),
        source="docs/pool_edge_plan.md",
        evergreen=True,  # methodology explainer, no single registry entry
    ),
    Finding(
        question="Do rest, travel and weather matter?",
        verdict="context",
        plain_answer=(
            "They matter to football and they ride along in the model as background, but none "
            "of them is an edge. Rest difference, divisional games, neutral sites, time of "
            "year, temperature and wind are all in there. They are small, and the market "
            "prices the obvious ones instantly -- everybody knows who is on a short week."
        ),
        detail=(
            "One honest limitation: the temperature and wind we hold are what the weather "
            "turned out to be, not the forecast that existed when the line was posted, so we "
            "treat those two columns as background rather than something to lean on. A proper "
            "decision-time weather feature needs an archived forecast source and has not been "
            "built. Travel distance, time-zone changes and body-clock effects are on the list "
            "and unbuilt as well."
        ),
        source="docs/data_feasibility.md, ROADMAP.md (ENV-01 to ENV-06)",
        evergreen=True,  # background context features, not their own registry entry
    ),
    Finding(
        question="Can we even make picks for the playoffs?",
        verdict="context",
        plain_answer=(
            "Now, yes. Playoff rows exist, and they were built so that every regular-season "
            "number "
            "is byte-for-byte what it was before, because changing them would have quietly "
            "turned our measured model into a different, unmeasured one."
        ),
        detail=(
            "Playoff games see everything that came before them, including earlier rounds -- a "
            "Super Bowl row knows what both teams did in the divisional round -- while "
            "regular-season rows never see a playoff result, so a January game cannot leak "
            "backwards into the following September. Training still uses regular-season games "
            "only. One deliberate gap remains: we have not yet graded the model on historical "
            "playoff games. That is a fresh test with about 65 games available, and it gets "
            "declared before it is computed, not after."
        ),
        source="docs/postseason_support.md",
        evergreen=True,  # infrastructure note, no registry entry
    ),
    Finding(
        question="Why not just keep testing ideas until something works?",
        verdict="context",
        plain_answer=(
            "Because we already did that, and we can measure what it cost. Roughly 130 to 150 "
            "versions of this model have been scored against the same eight seasons of "
            "results. Test that many things and the best of them looks good whether or not "
            "anything is there -- which is exactly why our prettiest numbers from those "
            "seasons (52.5-52.8%) are not evidence of anything, and why two of the three most "
            "promising leads from that era evaporated when re-tested on untouched seasons."
        ),
        detail=(
            "So we changed the rules. New ideas are scored on seasons that family has never "
            "used, with the method and the pass/fail line written down first; a used window is "
            "marked spent and never reused; and negative results are recorded permanently so "
            "nobody rediscovers them in two years and gets excited. The three untouched-window "
            "re-tests we have run came back -0.08 points, +0.00 points, and a failed "
            "prerequisite. That is what the discipline buys: not more wins, fewer fake ones."
        ),
        source="ROADMAP.md, docs/modeling.md",
        registry_keys=(
            "rotation:pbp_drive_bundle",
            "rotation:player_qb_continuity",
            "rotation:cfb_role_continuity",
        ),
        registry_fingerprints=(
            "7505bed89085a09d",
            "3a719416f790fb7e",
            "282f6629c405ef76",
        ),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="The pool scores one Best Pick a week. Can we tell which of our picks is best?",
        verdict="unproven",
        plain_answer=(
            "Partly. The Best Pick is worth **about +0.9 points** a week -- real but "
            "small. Today's rule ranks picks by calibrated distance from a coin flip, but "
            "only among games where bookmakers agree with each other "
            "(disagreement games run wilder and make a worse bonus-pick foundation). It is "
            "a same-night, one-look result on "
            "well-mined seasons, so not proven, but the strongest lean we have; every "
            "alternative has measured flat or negative."
        ),
        detail=(
            "The idea behind the first signal: re-score every pick at lines half a point "
            "either side of the real one, and measure how far the line can move before the "
            "pick stops being favoured. A pick that only works at exactly one number is "
            "fragile; one that survives four points in both directions is not. The catch, "
            "found on 2026-08-18: the robustness score hits its ceiling often, so it tied "
            "in 24 of 35 weeks and the recorded 60% top-pick accuracy was mostly "
            "alphabetical luck -- it sits at the 95th percentile of what random tie-breaking "
            "alone would produce, and flipping just three of the 35 weeks erases the whole "
            "result. Tie-agnostic, its honest edge is about +0.9 points, with roughly a coin "
            "flip's worth of confidence behind it. The replacement was found by directly "
            "testing whether market agreement matters, and it scored the clearest lean of "
            "anything measured for this decision -- though it reuses the same well-mined "
            "games a third time, a real cost we are stating rather than hiding, and the "
            "exact rule that ships (the agreement filter plus a tie-break rule inside it) "
            "was never itself tested as one combined thing, only as its two separate "
            "pieces. We use it anyway, for the same reason as before: the pool forces a "
            "Best Pick every week regardless of how confident we are in the method, and "
            "every alternative tried -- including the first signal on its own honest "
            "numbers -- is flat or worse."
        ),
        source="docs/best_pick_ranker.md",
        registry_keys=(
            "rotation:best_pick_ranker",
            "rotation:best_pick_ranker_opener",
            "weak_signal:best_pick_opener_ranker_candidate_prob_distance_vs_status_quo",
            "weak_signal:best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered",
        ),
        # Deliberately no challenger:* key here (or in any finding below): the
        # tracked registry/ files are always present in a real checkout, but
        # artifacts/prospective/challengers.json is a runtime ledger that many
        # legitimate contexts (a fresh checkout, most test fixtures) build
        # without -- gating curation on it would fail the whole page over an
        # optional store. The "currently tracked" challenger list is instead
        # covered by its own always-fresh, no-fingerprint section (see
        # findings_generation.md), which needs no curation at all.
        registry_fingerprints=(
            "00f87012f174f5d6",
            "006ad32121b1f645",
            "e3b7815f0557853f",
            "3067839eaba46a50",
        ),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Isn't our most confident pick the obvious Best Pick?",
        verdict="no-edge",
        plain_answer=(
            "No, and this is one of the more useful things we know. Taking the pick we are "
            "most confident in made the Best Pick materially WORSE: 41% against 49% for our "
            "picks overall. Ranking by how far our number sits from the market's did badly "
            "too, at 43%. Both were tested once on untouched seasons and both are closed."
        ),
        detail=(
            "This is the same result the track record shows from a different angle -- our "
            "confidence ordering carries no information, even after recalibrating the "
            "probabilities properly. The model's call on WHICH side to take is worth "
            "something; its opinion about HOW SURE it is, is not. That is why no pick on the "
            "picks page gets extra weight, and why the Best Pick is chosen by a completely "
            "different property of the pick rather than by confidence."
        ),
        source="docs/best_pick_ranker.md",
        registry_keys=(
            "weak_signal:best_pick_calibrated_probability_top1",
            "weak_signal:best_pick_key_number_distance_top1",
        ),
        registry_fingerprints=(
            "3f0f948b761f2701",
            "0b008e85182b2095",
        ),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Does stacking our leftover weak signals together beat the model we run?",
        verdict="unproven",
        plain_answer=(
            "Close, but below the bar for CLAIMING it works: **87% likely better**, "
            "against a 90% claim bar. The stack (learned injury availability, player-value "
            "weighting, three "
            "published line biases) scored **53.3%** against the pool's line on 456 games, "
            "versus 51.3% for the prior model. It runs today anyway: "
            "the pool forces 285 picks either way, so declining a candidate that is 87% "
            "likely better is just taking the other side of an 87/13 bet. The pass mark "
            "governs what we CLAIM, never what we PLAY."
        ),
        detail=(
            "Both versions agreed on 407 of the 456 picks. The whole difference comes from "
            "the 49 they disagreed on, where the new version went 29-20 -- a net of nine "
            "picks, which is well inside what luck produces. The seasons involved are also "
            "ones we have already searched heavily, so the bar was rightly set high. The next "
            "honest test is to run it alongside the real model through 2026 and see what it "
            "does on games nobody has looked at yet. We since took the stack apart to see "
            "which ingredient was doing the work, and it was not the one we assumed -- see "
            "the next answer."
        ),
        source="docs/mod07_stack.md",
        registry_keys=("rotation:mod07_weak_signal_stack",),
        registry_fingerprints=("78f848b1c5873155",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Which part of that stack was actually doing the work?",
        verdict="context",
        plain_answer=(
            "Not the published line biases, which were the reason we built it. Taking them "
            "out changes almost nothing: injury availability and player-value weighting on "
            "their own score 53.1% against the model's 51.3%, and adding the three bias "
            "features on top moves that to 53.3%. So the biases are worth about a fifth of "
            "a point, with a coin-flip chance of being worth anything at all. Everything "
            "with a real lean came from knowing who is playing."
        ),
        detail=(
            "This matters because it changes what to build next. We had been treating the "
            "near-miss as evidence that published early-season line biases are worth "
            "chasing; it is not. On the 39 picks the bias features alone flipped, they went "
            "19-39 one way and 20-39 the other -- a one-pick difference. Separately, the "
            "biggest of those biases does not reproduce here at all: the published claim is "
            "that last year's playoff teams cover only about 36% as Week 1 favourites, and "
            "in our data they cover 52.5% over 120 such games. We are not building more of "
            "them. Availability is where the signal was, and that is the thread to pull."
        ),
        source="docs/mod07_stack.md, docs/rotation_registry.md",
        registry_keys=(
            "weak_signal:mod07_opener_bias_ablation",
            "weak_signal:mod07_holdover_bias_replication",
        ),
        registry_fingerprints=("ff1713d32863f7c2", "eb03693fbd393a06"),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Plays outnumber games 166 to one. Shouldn't we learn from plays instead?",
        verdict="no-edge",
        plain_answer=(
            "No, and the reason is worth understanding because the number is seductive. "
            "There are about 780,000 plays behind 4,700 games. But we are predicting the "
            "game, and there are still only 4,700 of those to learn from. More plays make "
            "each team's average sharper; they do not create more things to learn from. "
            "Play data is also already most of what the model reads -- about 44 of its 79 "
            "inputs are built from plays."
        ),
        detail=(
            "We measured how much sharper. Knowing a team's per-play efficiency instead of "
            "just its points scored and allowed helps in weeks 1 to 3 and is worth nothing "
            "from week 4 onward, once a few games exist to average. Two specific ideas died "
            "here on measurement: forecasting the pace of a game (play counts barely vary, "
            "and games with more plays are LOWER-scoring blowouts, not higher -- teams kill "
            "the clock when ahead), and measuring how erratic a team is (a team's "
            "game-to-game spread turns out not to be a property of the team at all). One "
            "real gap survived: we measure defences about half as reliably as offences, and "
            "the standard fix for that has never been tested properly."
        ),
        source="docs/play_level_audit.md",
        evergreen=True,  # methodology explainer, no single registry entry
    ),
    Finding(
        question="How much history does the model need before its picks are trustworthy?",
        verdict="context",
        plain_answer=(
            "Less than we assumed, and there is no clean cut-off. We had a rule requiring "
            "500 finished games before the model would predict at all. Nobody had ever "
            "tested it. When we did, forced-pick accuracy turned out to be flat from 50 "
            "games all the way to 4,000 -- the rule was protecting against something that "
            "does not happen."
        ),
        detail=(
            "What DOES go wrong with little history is not being wrong, it is being loud: "
            "with 50 games the model moves the line by nine points on average, against under "
            "two points when fully trained. It is overconfident rather than misdirected, and "
            "the step that converts a prediction into a probability already fixes that. The "
            "practical consequence is about honesty rather than accuracy -- that unjustified "
            "500 was quietly deciding which seasons we are allowed to use to test future "
            "ideas, which is a permanent decision. It no longer does."
        ),
        source="docs/rotation_registry.md",
        evergreen=True,  # a code constant audit, not a registry entry
    ),
    Finding(
        question="If a signal is real but tiny, why not just include it?",
        verdict="context",
        plain_answer=(
            "You should -- and we nearly made the opposite mistake. When a test comes back "
            "showing 'no effect', that only means something if the test was sharp enough to "
            "have SEEN the effect. Ours usually is not. Our measurements can reliably detect "
            "about two percentage points of accuracy, and most individual football signals "
            "are worth a fraction of that. So 'we found nothing' is the expected result even "
            "for signals that are genuinely there."
        ),
        detail=(
            "The example that caught us: how aggressive a coach is on fourth down. It is a "
            "real, persistent trait, and our test of whether it beats the market came back "
            "with a range running from minus four tenths of a point to plus four tenths. The "
            "effect we were looking for -- the market ignoring it completely -- is under two "
            "tenths, which sits inside that range. The test could not tell the two apart. "
            "Confirming it directly would take about ninety NFL seasons. So the honest label "
            "is 'too small to measure alone', not 'doesn't work', and the right response is "
            "to fold it in with other small signals rather than throw it away. We now sort "
            "every negative result into three kinds: the mechanism is genuinely wrong, we "
            "proved our test could see it and it wasn't there, or we simply could not tell. "
            "Only the first two close anything."
        ),
        source="docs/pool_edge_plan.md",
        registry_keys=("weak_signal:fourth_down_aggressiveness",),
        registry_fingerprints=("4eb797d7530d0156",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Do football scores follow a bell curve?",
        verdict="context",
        plain_answer=(
            "Emphatically not. Almost 15% of games are decided by exactly three points -- "
            "nearly three times what a bell curve allows -- and seven is the next spike. "
            "The middle is stranger still: 346 games have ended +3 and 300 have ended -3, "
            "but only 13 in seventeen seasons have ended level, because overtime almost "
            "always breaks the tie. So the very centre really does have two humps with a "
            "hole between them."
        ),
        detail=(
            "Points arrive in threes and sevens, so some final margins are reachable many "
            "ways and others barely at all. Formally the test for a single peak is rejected "
            "outright. But here is the twist that matters: we do not predict the margin, we "
            "predict whether a team beats the spread -- and because the spread is a different "
            "number every week, those spikes get smeared out. Measured on the margin minus "
            "the spread, the lumpiness almost entirely vanishes and the shape comes back "
            "close to a bell curve. One thing survives: games pile up right ON the line about "
            "twice as often as a bell curve predicts, because the spread is deliberately set "
            "where the lump is. That is why a half-point matters enormously at a line of "
            "three and barely at all at a line of nine."
        ),
        source="docs/pool_edge_plan.md",
        evergreen=True,  # a fact about score distributions, not a registry entry
    ),
    Finding(
        question="Would a more careful statistical model squeeze more out of thin data?",
        verdict="no-edge",
        plain_answer=(
            "We built it to find out, and no. On 12,000 college games -- free to experiment "
            "on -- turning the caution dial across five orders of magnitude moves accuracy "
            "by less than a point, and turning it up, which is what the theory recommends, "
            "makes the thin-data games worse rather than better."
        ),
        detail=(
            "The reason is a structural fact worth remembering whenever someone proposes "
            "this kind of fix. Our pick is just which side of the line we land on. Being "
            "more cautious shrinks how far our number sits from the market's, but it cannot "
            "move it to the other side -- halving a number never changes its sign. So any "
            "method whose whole effect is to be more conservative cannot change a single "
            "pick, no matter how principled it is. It can improve how well-calibrated our "
            "stated confidence is, which is worth something, but not the thing we are "
            "actually scored on."
        ),
        source="docs/rotation_registry.md, ROADMAP.md",
        registry_keys=("weak_signal:ridge_alpha_global",),
        registry_fingerprints=("b8af0176fce754e5",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Did we set the model's season-to-season carryover number too high?",
        verdict="no-edge",
        plain_answer=(
            "No -- **the inherited number survives a direct test**. The model carries "
            "forward 67% of a team's rating between seasons -- a flat rate that was never "
            "derived, just inherited, while measured fade rates run much lower (as low as "
            "17%). But a fix using eight metric-specific "
            "fade rates lost head-to-head to the flat 67% by about three quarters of a point "
            "on 8,933 college games, with the whole range on the losing side."
        ),
        detail=(
            "The two questions -- how fast does a team's real form fade, and what carryover "
            "number makes the best picks -- turn out to have different answers, and only "
            "the second one is the model's actual job. The state features do better "
            "carrying MORE of last season forward than the raw fade rates say they should, "
            "most likely because a game depends on more than the handful of metrics we "
            "track, and leaning on last season's fuller picture beats trusting this "
            "season's still-thin sample. 67% survives its audit and stays. What is still "
            "genuinely open: this closes the college-football construction only. Nobody has "
            "run the equivalent per-metric test on NFL data, and that would need its own "
            "fresh, separately predeclared look rather than being answered by this result."
        ),
        source="ROADMAP.md (RWB-01), registry/weak_signals.json",
        registry_keys=(
            "weak_signal:offseason_retention_per_metric_cfb",
            "weak_signal:offseason_retention_075_cfb",
            "weak_signal:offseason_retention_050_cfb",
        ),
        registry_fingerprints=(
            "404fb7195f2280dc",
            "be9579b251d8cc1d",
            "d52e33053006f4fe",
        ),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question=(
            "Is the model's caution dial -- how hard it second-guesses its own numbers -- "
            "set correctly?"
        ),
        verdict="no-edge",
        plain_answer=(
            "Yes, in the sense that it was never chosen on purpose -- just a leftover "
            "example value. But it does not matter for accuracy: testing seven orders of "
            "magnitude on 12,500 college games, forced-pick accuracy stays flat throughout. "
            "A far more cautious setting (about 200x today's) does sharpen calibration, but "
            "swapping the live model to it tested about 95% likely to make picks WORSE, not "
            "better. So the dial stays where it is."
        ),
        detail=(
            "So the production dial stays exactly where it was, and the calibration gain "
            "goes somewhere it actually pays instead: the tool that chooses which single "
            "pick gets the week's Best Pick bonus needs a well-calibrated confidence number "
            "far more than it needs a differently tuned pick, and that swap costs nothing "
            "extra to make (see the next answer for how it is used today). The underlying "
            "number is still undefended in the sense that nobody chose it deliberately, but "
            "it is no longer unexamined: every value worth testing says it does not move "
            "the thing the pool actually grades us on."
        ),
        source="docs/ridge_alpha.md",
        registry_keys=(
            "weak_signal:ridge_alpha_2000_nfl_opener_confirmation",
            "weak_signal:ridge_alpha_global",
        ),
        registry_fingerprints=("c9c99b54f4faf955", "b8af0176fce754e5"),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question=(
            "Should we bet against a team early in the season just because it has a "
            "brand-new head coach?"
        ),
        verdict="unproven",
        plain_answer=(
            "It looks that way, and we are now playing it: teams under a brand-new head "
            "coach (weeks 1-8, opponent's coach not also new) have covered only about 47% "
            "against the market's price -- a real gap that replicates on college football at "
            "3.5x the sample size. So this season we flip the pick against the new-coach team "
            "in that window, publish both versions of every pick it touches, and let 2026 "
            "prove or kill it -- free, since the pool forces a pick either way and a "
            "coin-flip flip loses nothing on average. It already changed one pick on this "
            "season's first card."
        ),
        detail=(
            "The honest catch, and the reason this has never been formally confirmed: the "
            "effect only shows up from 2018 onward, the exact years the pattern was "
            "discovered in, and it is flat and unremarkable across the sixteen seasons "
            "before that. Every rotation window we have left to spend on a formal "
            "confirmation sits inside those same discovered years, so no confirmation test "
            "we could run would really be independent of the data that found the pattern "
            "-- that is circular, not confirmatory, and we said so rather than spend a "
            "window pretending otherwise. We are fielding it anyway, because this is a "
            "forced-pick pool: preferring the side with real historical direction over an "
            "unweighted coin flip costs nothing when a pick is required regardless, and the "
            "genuinely independent test -- how it does on 2026 games nobody has looked at "
            "yet -- starts for free the moment the season kicks off."
        ),
        source="docs/coach_fade_overlay.md, ROADMAP.md (PER-07)",
        registry_keys=("weak_signal:hc_year_one_fade",),
        registry_fingerprints=("c381308f869a066e",),
        curated_as_of=_CURATED_AS_OF,
    ),
    Finding(
        question="Can a new idea be tested without weeks of custom code for each one?",
        verdict="unproven",
        plain_answer=(
            "Yes -- about fifty ideas tested in one night. A standard "
            "template for simple yes/no situations (revenge game, rivalry finale, bookmaker "
            "disagreement) plus one command that runs the test correctly and records an "
            "honest result, whether good, bad, or too small to tell -- days of hand-built "
            "code per idea, collapsed to a spec and a command."
        ),
        detail=(
            "Almost all of it comes back exactly the way you would expect from screening "
            "fifty exploratory ideas at once: too small to tell from noise, which is the "
            "correct and expected outcome, not a failure of the tool. Two are worth naming "
            "as standouts. A team getting a second shot at a division rival it lost to "
            "earlier in the season covers about **0.19 points** better than the rest of the "
            "slate, with roughly an **88%** chance that lean is real -- a small, real-looking "
            "edge, not yet strong enough to call proven. And in college football, a team's "
            "last game of the regular season -- rivalry games, bowl-eligibility deciders, "
            "the games with the most on the line and the least practice time to prepare -- "
            "covers measurably worse than the rest of the schedule, one of the only results "
            "out of fifty whose range stays entirely on one side rather than straddling "
            "zero. Neither is played on the real card, but the division-"
            "revenge lean is dual-tracked for free as a prospective challenger (a "
            "post-prediction pick flip, scored against the active model's own picks once "
            "2026 games accrue) alongside four more mined leads built the same way -- see "
            "'What we're watching' below for the full, always-current list "
            "rather than a hand-typed one here that would only go stale again."
        ),
        source="docs/experiment_pipeline.md, registry/weak_signals.json",
        registry_keys=(
            "weak_signal:bias_battery_division_revenge_game",
            "weak_signal:cfb_bias_battery_rivalry_finale_proxy",
        ),
        registry_fingerprints=(
            "c2260e6fdb9f76fd",
            "9a1992d36fc40702",
        ),
        curated_as_of=_CURATED_AS_OF,
    ),
)


# ---------------------------------------------------------------------------
# Closing section: the honesty rules
# ---------------------------------------------------------------------------

HONESTY_KICKER = "Before you trust a percentage"
HONESTY_TITLE = "How to read any number on this dashboard"
HONESTY_SUB = (
    "Four rules we hold ourselves to. They are the difference between a number that means "
    "something and a number that merely looks good."
)

HONESTY_RULES: tuple[HonestyRule, ...] = (
    HonestyRule(
        title="Every number comes from games the model had never seen",
        body=(
            "Nothing here is scored on games the model learned from. For each week, the model "
            "is rebuilt using only games that had already finished before that week's first "
            "kickoff, and then it picks once. It is slower, and the numbers come out lower "
            "than the alternative. That is the trade: a number you can believe instead of a "
            "number you can enjoy."
        ),
    ),
    HonestyRule(
        title="A point estimate is not the whole answer",
        body=(
            "The opener baseline at the top of this page is a point estimate; its "
            "season-blocked range, quoted beside it in the hero above, is the honest "
            "answer. The arrest-policy component's evaluation (home on the "
            "track-record page) reports its P+ alongside its point "
            "estimate for the same reason. Those uncertainty "
            "summaries come from re-scoring the same games "
            "in whole-week and whole-season chunks, because games in the same week are not "
            "independent of each other. We report the estimate and P+ "
            "without turning uncertainty into a binary play-or-reject gate."
        ),
    ),
    HonestyRule(
        title="One look, decided in advance",
        body=(
            "Before a test runs we write down what is being measured, on which games, and "
            "what would count as passing. Then we run it once and record whatever comes out. "
            "No quiet re-runs with a different setting until it wins. When a result surprises "
            "us, the surprise goes into the record too, unedited -- the predeclaration is "
            "never rewritten after the fact."
        ),
    ),
    HonestyRule(
        title="Nothing tested gets deleted",
        body=(
            "Every measured result stays in the record, including wrong-sign, bounded, and "
            "unresolved work. Quietly deleting an inconvenient result lets it be rediscovered "
            "and over-fit later. If the research record ever gets shorter, something has gone "
            "wrong."
        ),
    ),
)

CLOSING_NOTE = (
    "And if a number on this dashboard ever reads "
    f"{CEILING_BUG_MARK_PCT}%, treat it as a bug rather than a "
    "breakthrough. That is not modesty -- the sport's own randomness makes it arithmetically "
    "out of reach, and we have thrown out results before for being too good."
)


# ---------------------------------------------------------------------------
# "What we're watching" -- hand-curated one-liners for the top few leads
# ---------------------------------------------------------------------------

#: Curated 2026-08-19 against the three most extreme entries in the LIVE
#: ``top_open_leads`` ranking at the time of writing (see
#: ``nfl_ats.findings_registry.top_open_leads`` -- ranked by
#: ``|probability_positive - 0.5|``, so a P+ near 0 is exactly as strong a
#: lead as one near 1, just pointed the other way). Every other rendered
#: lead falls back to the registry's own ``description`` with no curation
#: needed at all -- this tuple only needs to grow when a NEW entry earns a
#: plainer sentence than its own description already is, never on a
#: schedule.
#: Reader-safety curation (2026-08-23): the movement-agreement registry
#: entries carry internal audit prose (script paths, scratchpad references,
#: "NOT deleted per AGENTS.md" annotations) in their ``description`` fields.
#: Those descriptions stay untouched in the registry -- they are research
#: records -- but these blurbs replace what a reader of findings.html sees,
#: with the full record one link away. Fingerprints pin them to today's live
#: entries; any re-recording fails the build loudly and forces a re-read.
_MOVEMENT_AGREEMENT_BLURB_TEXT = (
    "A line-movement signal was re-checked independently; its measured edge did not "
    "survive the re-check as more than noise \u2014 tracked as unresolved."
)

LEAD_BLURBS: tuple[LeadBlurb, ...] = (
    LeadBlurb(
        weak_signal_name="weather_battery_surface_switch_grass_to_turf",
        text=(
            "When a team that normally plays home games on grass visits a turf stadium, "
            "the turf home team has covered more than expected -- right now the single "
            "strongest open lead on this page."
        ),
        curated_as_of=_CURATED_AS_OF,
        registry_fingerprints=("7ed27dc3f22c48d0",),
    ),
    LeadBlurb(
        weak_signal_name="cfb_bias_battery_neutral_site_designated_home",
        text=(
            "In neutral-site college games where the schedule still assigns one team a "
            "'home' label, that designated home team has covered LESS than expected -- a "
            "lead for fading the assigned home side, not backing it."
        ),
        curated_as_of=_CURATED_AS_OF,
        registry_fingerprints=("60c9af992eb2bb37",),
    ),
    LeadBlurb(
        weak_signal_name="best_pick_tiebreak_cfb_stage0_ecdf_gaussian",
        text=(
            "A statistical tie-break rule tested for choosing college Best Picks did "
            "markedly worse than the simple alphabetical tie-break already in use -- a "
            "lead AGAINST switching to it, not for it."
        ),
        curated_as_of=_CURATED_AS_OF,
        registry_fingerprints=("f8b7955edf83ed58",),
    ),
    LeadBlurb(
        weak_signal_name="opener_error_mining_movement_agreement_agrees",
        text=_MOVEMENT_AGREEMENT_BLURB_TEXT,
        curated_as_of="2026-08-23",
        registry_fingerprints=("84344d180d99fc5d",),
    ),
    LeadBlurb(
        weak_signal_name="opener_error_mining_movement_agreement_disagrees_corrected",
        text=_MOVEMENT_AGREEMENT_BLURB_TEXT,
        curated_as_of="2026-08-23",
        registry_fingerprints=("b39320e1d465ed61",),
    ),
)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def findings_for(verdict: Verdict) -> tuple[Finding, ...]:
    """Every finding carrying ``verdict``, in declaration order."""

    return tuple(finding for finding in FINDINGS if finding.verdict == verdict)
