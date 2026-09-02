"""Pool workbench: pool rules, a browser-local entry, and ownership scenarios.

ROADMAP item UI-09. The workbench reuses the existing forced-pick card from
:mod:`nfl_ats.pool` and the active model's ``recommendations.csv`` forecast
format; it invents nothing about future games. Entry edits persist only in the
operator's browser, scoped to one season/week, and never mutate the published
forecast. Ownership views are explicitly hypothetical favorite-side assumptions
because no entry-popularity feed is integrated.

The pool's format is the one confirmed in ``docs/pool_edge_plan.md``
(owner-corrected 2026-08-20): forced ATS sides for all 272 regular-season
games plus all 13 playoff games (285 cards, exactly the forced-pick metric
this project evaluates), one "Best Pick" per regular-season week, no passes,
and a line that locks Tuesday (revised once Wednesday, then frozen for the
week) while our picks stay editable up to each game's own per-game deadline
(SNF/MNF lock early at Sunday 16:00 ET).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from html import escape
from typing import Any, ClassVar

import pandas as pd

from nfl_ats.dashboard import viz
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock
from nfl_ats.pool import build_ats_pool_card

PAGE_FILENAME = "pool.html"
PAGE_TITLE = "Pool workbench"
ENTRY_STORAGE_VERSION = 1

# Confirmed Splash-style format (docs/pool_edge_plan.md, owner-corrected 2026-08-20).
REGULAR_SEASON_GAMES = 272
PLAYOFF_GAMES = 13
# SNF/MNF lock early at Sunday 16:00 ET, i.e. min(kickoff, Sunday 16:00 ET).
SUNDAY_EARLY_LOCK_ET = "Sunday 16:00 ET"

# Columns the active model's recommendations.csv must carry for the workbench.
_REQUIRED_COLUMNS = frozenset(
    {"game_id", "gameday", "away_team", "home_team", "spread_line", "home_cover_probability"}
)


# ---------------------------------------------------------------------------
# Pool rules input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolRules:
    """The pool's format and scoring rules -- the workbench's rules INPUT.

    Defaults encode the confirmed Splash-style format. :meth:`from_dict`
    accepts a partial override map so a future operator can tune the rules
    without code edits; any field absent from the map keeps its safe default.
    """

    regular_season_games: int = REGULAR_SEASON_GAMES
    playoff_games: int = PLAYOFF_GAMES
    pick_type: str = "ats"
    pool_type: str = "standard"
    scoring_method: str = "correct_picks"
    entry_count: int = 1
    correct_pick_points: float = 1.0
    incorrect_pick_points: float = 0.0
    push_points: float = 0.5
    confidence_assignment: str = "none"
    team_use_limit: int | None = None
    survivor_lives: int = 1
    best_pick_per_regular_season_week: int = 1
    best_pick_bonus: float = 1.0
    best_pick_penalty: float = 0.0
    forced_picks: bool = True
    passes_allowed: bool = False
    line_locks_tuesday: bool = True
    picks_due_per_game_kickoff: bool = True
    sunday_early_lock: str = SUNDAY_EARLY_LOCK_ET
    # The grading target is the frozen Tuesday OPENING line, not the sharp
    # market close (docs/pool_edge_plan.md:5, "beat the OPENING line the
    # user's Splash Sports pool grades against"; AGENTS.md's "Grade the
    # decision at the OPENER" section). "opener" here names the SAME line
    # `line_locks_tuesday` already describes as frozen -- this field exists
    # so a consumer can name *which* line without re-deriving it from the
    # boolean.
    grading_line: str = "opener"
    # The pool breaks ties on the final score of the week's LAST game
    # (src/nfl_ats/tiebreaker.py module docstring: "The pool breaks ties on
    # the final score of the week's LAST game (owner, 2026-09-01...)").
    # `nfl_ats.tiebreaker` implements the guess itself; this field only
    # names the rule for the board/report, read-only reference, no logic
    # duplicated here.
    tiebreak: str = "final_score_last_game"

    #: The owner's per-game pick deadline rule, reused verbatim rather than
    #: reimplemented: ``min(this game's own kickoff, that week's Sunday
    #: 16:00 ET lock)`` (owner rule, 2026-08-20, re-confirmed 2026-09-01;
    #: ``docs/late_week_refresh.md`` "Per-game deadline, not one weekly
    #: cutoff"). ``ClassVar`` so it is a plain function reference shared by
    #: every instance -- NOT a dataclass field, so it never appears in
    #: ``__init__``/``repr``/``==`` and cannot be overridden per instance via
    #: ``from_dict``. Wrapped in ``staticmethod`` so accessing it through an
    #: instance (``rules.deadline_rule``) returns the plain two-argument
    #: function rather than auto-binding ``self`` as its first argument. Use
    #: :meth:`deadline_for` for the ergonomic per-game call; this attribute
    #: exists so callers who already have both timestamps in hand (a
    #: kickoff and a precomputed Sunday lock) can call the exact same
    #: underlying function directly, e.g.
    #: ``PoolRules.deadline_rule(kickoff, sunday_lock)``.
    deadline_rule: ClassVar[Callable[[pd.Timestamp, pd.Timestamp], pd.Timestamp]] = staticmethod(
        pick_deadline
    )

    _PICK_TYPES: ClassVar[frozenset[str]] = frozenset({"ats", "straight_up"})
    _POOL_TYPES: ClassVar[frozenset[str]] = frozenset({"standard", "confidence", "survivor"})
    _GRADING_LINES: ClassVar[frozenset[str]] = frozenset({"opener", "close", "result"})
    _SCORING_BY_POOL_TYPE: ClassVar[dict[str, str]] = {
        "standard": "correct_picks",
        "confidence": "confidence_points",
        "survivor": "survival",
    }

    def __post_init__(self) -> None:
        if self.pick_type not in self._PICK_TYPES:
            raise ValueError(f"pick_type must be one of {sorted(self._PICK_TYPES)}")
        if self.pool_type not in self._POOL_TYPES:
            raise ValueError(f"pool_type must be one of {sorted(self._POOL_TYPES)}")
        if self.grading_line not in self._GRADING_LINES:
            raise ValueError(f"grading_line must be one of {sorted(self._GRADING_LINES)}")
        if self.pick_type == "straight_up" and self.grading_line != "result":
            raise ValueError("straight_up picks require grading_line='result'")
        if self.pick_type == "straight_up" and self.line_locks_tuesday:
            raise ValueError("straight_up picks cannot use an ATS Tuesday line lock")
        if self.pick_type == "ats" and self.grading_line == "result":
            raise ValueError("ATS picks require an opener or close grading line")
        expected_scoring = self._SCORING_BY_POOL_TYPE[self.pool_type]
        if self.scoring_method != expected_scoring:
            raise ValueError(
                f"pool_type {self.pool_type!r} requires scoring_method {expected_scoring!r}"
            )
        for name, value in (
            ("regular_season_games", self.regular_season_games),
            ("playoff_games", self.playoff_games),
            ("best_pick_per_regular_season_week", self.best_pick_per_regular_season_week),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.entry_count, bool) or not isinstance(self.entry_count, int):
            raise ValueError("entry_count must be a positive integer")
        if self.entry_count < 1:
            raise ValueError("entry_count must be a positive integer")
        if self.pool_type != "survivor" and self.survivor_lives != 1:
            raise ValueError("survivor_lives is only configurable for survivor pools")
        for name, point_value in (
            ("correct_pick_points", self.correct_pick_points),
            ("incorrect_pick_points", self.incorrect_pick_points),
            ("push_points", self.push_points),
            ("best_pick_bonus", self.best_pick_bonus),
            ("best_pick_penalty", self.best_pick_penalty),
        ):
            if not math.isfinite(point_value):
                raise ValueError(f"{name} must be finite")
        if self.forced_picks and self.passes_allowed:
            raise ValueError("forced_picks and passes_allowed cannot both be true")
        if self.pool_type == "confidence":
            if self.confidence_assignment != "unique_1_to_game_count":
                raise ValueError(
                    "confidence pools require confidence_assignment 'unique_1_to_game_count'"
                )
            if not self.forced_picks or self.passes_allowed:
                raise ValueError("confidence pools require forced picks with no passes")
        elif self.confidence_assignment != "none":
            raise ValueError("confidence_assignment is only valid for confidence pools")
        if self.pool_type == "survivor":
            if self.pick_type != "straight_up":
                raise ValueError("survivor pools require straight_up picks")
            if self.team_use_limit != 1:
                raise ValueError("survivor pools require team_use_limit=1")
            if (
                isinstance(self.survivor_lives, bool)
                or not isinstance(self.survivor_lives, int)
                or self.survivor_lives < 1
            ):
                raise ValueError("survivor_lives must be a positive integer")
            if self.best_pick_per_regular_season_week != 0:
                raise ValueError("survivor pools cannot also award a Best Pick")
            if not self.forced_picks or self.passes_allowed:
                raise ValueError("survivor pools require one forced selection with no passes")
        elif self.team_use_limit is not None:
            raise ValueError("team_use_limit is only valid for survivor pools")

    @property
    def total_games(self) -> int:
        """Forced picks across the whole season: 272 + 13 = 285 (measured)."""

        return self.regular_season_games + self.playoff_games

    @property
    def cards_per_season(self) -> int:
        """Alias for :attr:`total_games` in the owner's own vocabulary:
        "The pool is FORCED PICKS: 285 cards must be submitted either way."
        (AGENTS.md, "A promotion bar is not a decision bar"; also
        docs/pool_edge_plan.md:76-77, 272 regular-season + 13 playoff games).
        Derived from :attr:`total_games` rather than a second hardcoded 285,
        so the two can never drift out of sync (this project's "derive
        constants, do not duplicate them" discipline)."""

        return self.total_games

    @property
    def submissions_per_season(self) -> int | None:
        """Total pick submissions when the format selects every listed game.

        Survivor pools select one team per week, so their total cannot be
        derived from the game counts stored here and is deliberately ``None``.
        """

        if self.pool_type == "survivor":
            return None
        return self.cards_per_season * self.entry_count

    def deadline_for(
        self, kickoff: pd.Timestamp, week_kickoffs: Sequence[pd.Timestamp] | pd.Series
    ) -> pd.Timestamp:
        """This game's real pick deadline: ``min(its own kickoff, that
        week's Sunday 16:00 ET lock)``.

        Delegates to :attr:`deadline_rule`
        (``nfl_ats.pick_refresh.pick_deadline``) and
        ``nfl_ats.pick_refresh.sunday_pick_lock`` verbatim -- the rule is
        never reimplemented here. ``week_kickoffs`` should be every kickoff
        in that game's week; it anchors the Sunday lock on the mode
        Tue..Mon-cycle Sunday among them (owner rule, 2026-08-20,
        re-confirmed 2026-09-01; ``docs/late_week_refresh.md``: "The Sunday
        anchor is computed from the week's own games, not a calendar
        guess"), so one isolated Tuesday/Wednesday reschedule cannot shift
        the whole week's lock instant. A Thursday game's own kickoff is
        always earlier than that Sunday lock, so this reduces to "picks due
        at kickoff" for TNF; SNF and MNF are the only games this ever
        returns earlier than their own kickoff for.
        """

        sunday_lock = sunday_pick_lock(pd.Series(list(week_kickoffs)))
        return pick_deadline(pd.Timestamp(kickoff), sunday_lock)

    def describe(self) -> list[str]:
        """Plain-English rendering of these rules for the board/report --
        one sentence per rule, no HTML, safe to drop into a doc or console
        (the "Label how you know it" plain-English discipline)."""

        if self.pool_type == "survivor":
            format_description = (
                f"Survivor pool with {self.entry_count} "
                f"{'entries' if self.entry_count != 1 else 'entry'}: "
                "one straight-up team per week, "
                f"each team usable {self.team_use_limit} time; {self.survivor_lives} "
                f"{'lives' if self.survivor_lives != 1 else 'life'}."
            )
        else:
            format_name = (
                f"{self.pick_type.replace('_', '-')} confidence"
                if self.pool_type == "confidence"
                else self.pick_type
            )
            format_description = (
                f"{self.total_games} forced {format_name.replace('_', '-')} picks per entry "
                f"({self.regular_season_games} regular season + {self.playoff_games} playoff), "
                f"{self.entry_count} {'entries' if self.entry_count != 1 else 'entry'}; "
                + ("no passes allowed." if not self.passes_allowed else "passes allowed.")
            )
        scoring = (
            "Scoring: assign every confidence value from 1 through the week's game count once."
            if self.pool_type == "confidence"
            else (
                "Scoring: survive each week; a loss consumes one life."
                if self.pool_type == "survivor"
                else f"Scoring: {self.correct_pick_points:g} per correct pick, "
                f"{self.incorrect_pick_points:g} per incorrect pick, {self.push_points:g} per push."
            )
        )
        best_pick = (
            f"One Best Pick per regular-season week (+{self.best_pick_bonus:.1f} bonus, "
            f"-{self.best_pick_penalty:.1f} penalty)."
            if self.best_pick_per_regular_season_week
            else "No separate Best Pick award."
        )
        grading = (
            f"Graded against the {self.grading_line} line"
            + (", frozen Tuesday and never rewritten." if self.line_locks_tuesday else ".")
            if self.pick_type == "ats"
            else "Graded by the straight-up game result; no spread grades the pick."
        )
        return [
            format_description,
            scoring,
            best_pick,
            grading,
            "Picks are due by each game's own kickoff, or that week's Sunday "
            f"16:00 ET lock, whichever is earlier ({self.sunday_early_lock} for SNF/MNF).",
            f"Tiebreaker: {self.tiebreak.replace('_', ' ')}.",
        ]

    @classmethod
    def from_defaults(cls) -> PoolRules:
        """The confirmed Splash-style format from docs/pool_edge_plan.md."""

        return cls()

    @classmethod
    def straight_up(cls, **overrides: Any) -> PoolRules:
        """A standard straight-up pool, retaining the real deadline rule."""

        base: dict[str, Any] = {
            "pick_type": "straight_up",
            "grading_line": "result",
            "line_locks_tuesday": False,
            "best_pick_per_regular_season_week": 0,
        }
        base.update(overrides)
        return cls(**base)

    @classmethod
    def confidence(cls, *, pick_type: str = "ats", **overrides: Any) -> PoolRules:
        """A forced-pick confidence pool with unique weekly point values."""

        base: dict[str, Any] = {
            "pick_type": pick_type,
            "pool_type": "confidence",
            "scoring_method": "confidence_points",
            "confidence_assignment": "unique_1_to_game_count",
            "best_pick_per_regular_season_week": 0,
        }
        if pick_type == "straight_up":
            base.update({"grading_line": "result", "line_locks_tuesday": False})
        base.update(overrides)
        return cls(**base)

    @classmethod
    def survivor(cls, **overrides: Any) -> PoolRules:
        """A straight-up survivor pool with one use per team."""

        base: dict[str, Any] = {
            "pick_type": "straight_up",
            "pool_type": "survivor",
            "scoring_method": "survival",
            "grading_line": "result",
            "line_locks_tuesday": False,
            "best_pick_per_regular_season_week": 0,
            "team_use_limit": 1,
        }
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PoolRules:
        """Load pool rules from a partial override map (the rules INPUT)."""

        known = {field.name for field in fields(cls)}
        overrides = {key: value for key, value in dict(data).items() if key in known}
        pool_type = str(overrides.get("pool_type", "standard"))
        pick_type = str(overrides.get("pick_type", "ats"))
        if pool_type == "confidence":
            overrides.setdefault("scoring_method", "confidence_points")
            overrides.setdefault("confidence_assignment", "unique_1_to_game_count")
            overrides.setdefault("best_pick_per_regular_season_week", 0)
        elif pool_type == "survivor":
            overrides.setdefault("pick_type", "straight_up")
            pick_type = str(overrides["pick_type"])
            overrides.setdefault("scoring_method", "survival")
            overrides.setdefault("team_use_limit", 1)
            overrides.setdefault("best_pick_per_regular_season_week", 0)
        if pick_type == "straight_up":
            overrides.setdefault("grading_line", "result")
            overrides.setdefault("line_locks_tuesday", False)
        return cls(**overrides)


# ---------------------------------------------------------------------------
# Entry list (reuse the existing forced-pick card)
# ---------------------------------------------------------------------------


def _safe_pool_card(predictions: pd.DataFrame) -> pd.DataFrame:
    """``build_ats_pool_card`` or an empty frame when not buildable."""

    if predictions is None or predictions.empty:
        return pd.DataFrame()
    if _REQUIRED_COLUMNS.difference(predictions.columns):
        return pd.DataFrame()
    return build_ats_pool_card(predictions)


def build_entry_list(predictions: pd.DataFrame) -> pd.DataFrame:
    """The pool entry list: one forced ATS side per game, ranked by confidence.

    Reuses :func:`nfl_ats.pool.build_ats_pool_card` -- the existing
    pick/probability output -- and degrades to an empty frame when the
    forecast is missing the columns it needs (e.g. no active forecast yet).
    """

    return _safe_pool_card(predictions)


def derive_confidence_ranks(predictions: pd.DataFrame) -> pd.DataFrame:
    """Confidence ranks derived from the active model forecast format.

    Rank 1 is the model's highest-confidence pick; confidence is
    ``|pick_probability - 0.5|`` computed from ``home_cover_probability``
    (the forecast's calibrated cover probability). Same source and same
    ordering as :func:`build_entry_list` -- kept as a standalone view for
    callers that want the ranking without the entry list's pick/star
    rendering. The pool workbench page itself renders one merged table
    (the entry list) rather than this plus a duplicate.
    """

    card = _safe_pool_card(predictions)
    if card.empty:
        return card
    return card[
        [
            "confidence_rank",
            "gameday",
            "away_team",
            "home_team",
            "pool_pick",
            "pool_side",
            "pick_probability",
            "confidence",
            "game_id",
        ]
    ].copy()


# ---------------------------------------------------------------------------
# Ownership scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnershipScenario:
    """One disclosed, hypothetical favorite-side ownership assumption.

    ``favorite_share`` is not an estimate of this pool. It is the assumed
    fraction of a hypothetical field taking the favorite in each game. The
    defaults match the sensitivity points already used by the POL-05 simulator.
    """

    key: str
    label: str
    favorite_share: float
    note: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("ownership scenario key must not be empty")
        if not 0.0 <= self.favorite_share <= 1.0:
            raise ValueError("favorite_share must be between 0 and 1")


DEFAULT_OWNERSHIP_SCENARIOS: tuple[OwnershipScenario, ...] = (
    OwnershipScenario(
        key="even",
        label="Even field",
        favorite_share=0.50,
        note="Control: each ATS side is equally common.",
    ),
    OwnershipScenario(
        key="moderate_favorite",
        label="Moderate favorite crowding",
        favorite_share=0.65,
        note="Sensitivity case: 65% of entries take each game's favorite.",
    ),
    OwnershipScenario(
        key="strong_favorite",
        label="Strong favorite crowding",
        favorite_share=0.85,
        note="Sensitivity case: 85% of entries take each game's favorite.",
    ),
)


def build_ownership_scenarios(
    entry_list: pd.DataFrame,
    scenarios: Sequence[OwnershipScenario] = DEFAULT_OWNERSHIP_SCENARIOS,
) -> pd.DataFrame:
    """Summarise entry overlap under disclosed favorite-side assumptions.

    A negative ATS line marks the entry pick as the favorite and a positive
    line marks it as the underdog. Pick'em rows have no favorite and therefore
    contribute 50% overlap in every scenario. No row is observed ownership.
    """

    columns = [
        "key",
        "label",
        "favorite_share",
        "entry_side_share",
        "disagreements_per_100",
        "note",
        "observed",
    ]
    if entry_list.empty or "pick_line" not in entry_list:
        return pd.DataFrame(columns=columns)

    lines = pd.to_numeric(entry_list["pick_line"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        same_side = pd.Series(0.5, index=lines.index, dtype=float)
        same_side.loc[lines.lt(0)] = scenario.favorite_share
        same_side.loc[lines.gt(0)] = 1.0 - scenario.favorite_share
        share = float(same_side.mean())
        rows.append(
            {
                "key": scenario.key,
                "label": scenario.label,
                "favorite_share": scenario.favorite_share,
                "entry_side_share": share,
                "disagreements_per_100": (1.0 - share) * 100.0,
                "note": scenario.note,
                "observed": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Rendering (body fragments; public_board.py wraps them in the page shell)
# ---------------------------------------------------------------------------


def _section(kicker: str, title: str, inner: str) -> str:
    # <h2>: section headers nest directly under the page's single <h1>
    # (WCAG 1.3.1 Info and Relationships).
    return (
        f'<section style="margin-top:24px;">'
        f'<p class="kicker">{escape(kicker)}</p>'
        f'<h2 class="title" style="margin:2px 0 12px;">{escape(title)}</h2>'
        f"{inner}</section>"
    )


def _format_line(value: float) -> str:
    """Compact signed ATS line for an entry choice."""

    if value > 0:
        return f"+{value:g}"
    return f"{value:g}"


def _data_table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(f"<tr>{row}</tr>" for row in rows)
    return (
        '<table class="data" style="margin-top:4px;">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _rules_section(rules: PoolRules) -> str:
    tiles = [
        viz.stat_tile(
            "Forced picks",
            str(rules.total_games),
            f"{rules.regular_season_games} regular-season + {rules.playoff_games} playoff",
        ),
        viz.stat_tile(
            "Best Pick",
            f"{rules.best_pick_per_regular_season_week}/wk",
            f"+{rules.best_pick_bonus:.1f} bonus, -{rules.best_pick_penalty:.1f} penalty",
        ),
        viz.stat_tile(
            "Passes",
            "Not allowed" if not rules.passes_allowed else "Allowed",
            "Every game must be picked",
        ),
        viz.stat_tile(
            "Entries",
            str(rules.entry_count),
            (
                f"{rules.submissions_per_season} pick submissions per season"
                if rules.submissions_per_season is not None
                else "One weekly survivor selection per active entry"
            ),
        ),
        viz.stat_tile(
            "Line lock",
            "Tuesday" if rules.line_locks_tuesday else "—",
            "Revised once Wednesday, then frozen for the week",
        ),
    ]
    tile_grid = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));'
        f'gap:14px;">{"".join(tiles)}</div>'
    )
    bullets = [
        (
            "Picks are due by each game's own kickoff"
            + (
                f" (SNF/MNF lock early, {rules.sunday_early_lock})"
                if rules.picks_due_per_game_kickoff
                else ""
            )
        ),
        "The grading target is the frozen Tuesday line, not the sharp market close.",
        "One Best Pick per regular-season week; playoff weeks award none.",
    ]
    bullet_list = (
        '<ul class="prose" style="margin:12px 0 0;padding-left:18px;">'
        + "".join(f"<li>{escape(bullet)}</li>" for bullet in bullets)
        + "</ul>"
    )
    provenance = (
        '<p class="fine" style="margin-top:10px;">Format confirmed in '
        "docs/pool_edge_plan.md (owner-corrected 2026-08-20).</p>"
    )
    return _section(
        "Pool rules input", "How the pool is scored", tile_grid + bullet_list + provenance
    )


def _entry_list_section(
    entry_list: pd.DataFrame,
    *,
    best_pick_game_id: str | None = None,
    storage_key: str | None = None,
) -> str:
    """The pool's one entry-list table.

    This used to be two sections -- "Entry list" and "Confidence ranks" --
    rendering the same 16 rows of the same forced-pick card twice with
    different formatting (owner, 2026-08-26: "displaying identical/duplicated
    data"). Merged into one table: editable pick rendering and the best-pick
    star come from the former entry list; the cover-probability column renders with
    :func:`viz.probability_meter`, taken from the former confidence-ranks
    table, since a bar anchored on the coin flip reads better for a
    probability than a bare signed delta.
    """

    if entry_list.empty:
        inner = viz.empty_state(
            "No pick card yet",
            "Once the week's opening line is captured and a forecast card is built, "
            "the entry list fills in by itself.",
        )
        return _section("Entry list", "This week's forced picks", inner)

    # Deferred import: nfl_ats.public_board imports THIS module, so a
    # top-level import back would cycle. Same lazy-import pattern
    # nfl_ats.weekly._cli_runner already uses. Sharing the helper rather than
    # re-deriving the bands here keeps the pool page's Confidence column
    # visually identical to the week board's Strength column -- the reader
    # should not have to learn two encodings for one quantity.
    from nfl_ats.public_board import confidence_meter, confidence_word

    rows: list[str] = []
    for _, row in entry_list.iterrows():
        is_best = str(row["game_id"]) == str(best_pick_game_id)
        word = confidence_word(float(row["pick_probability"]))
        meter = viz.probability_meter(float(row["pick_probability"]), label="cover")
        model_side = str(row["pool_side"])
        model_line = float(row["pick_line"])
        choices: list[str] = []
        for side, team in (("AWAY", row["away_team"]), ("HOME", row["home_team"])):
            line = model_line if side == model_side else -model_line
            favorite = "unknown" if line == 0 else str(line < 0).lower()
            checked = " checked" if side == model_side else ""
            model_flag = " (model)" if side == model_side else ""
            choices.append(
                '<label style="display:block;white-space:nowrap;">'
                f'<input type="radio" class="entry-pick" '
                f'name="entry-{escape(str(row["game_id"]))}" '
                f'data-game-id="{escape(str(row["game_id"]))}" data-side="{side}" '
                f'data-favorite="{favorite}"{checked}> '
                f"{escape(str(team))} {_format_line(line)}{model_flag}</label>"
            )
        best_checked = " checked" if is_best else ""
        rows.append(
            "<td>"
            f"{int(row['confidence_rank'])}</td>"
            f"<td>{escape(str(row['away_team']))} at {escape(str(row['home_team']))}</td>"
            f"<td>{''.join(choices)}</td>"
            f"<td>{meter}</td>"
            f"<td>{confidence_meter(word)}{row['confidence']:.1%}</td>"
            '<td style="text-align:center;">'
            f'<label><input type="radio" class="entry-best" name="entry-best" '
            f'data-game-id="{escape(str(row["game_id"]))}" '
            f'aria-label="Best Pick for {escape(str(row["away_team"]))} at '
            f'{escape(str(row["home_team"]))}"{best_checked}> '
            '<span aria-hidden="true">&#9733;</span>'
            "</label></td>"
        )
    table = _data_table(
        ["#", "Game", "Entry pick", "Model cover probability", "Model confidence", "Best Pick"],
        rows,
    )
    scoped = storage_key is not None
    disabled = "" if scoped else " disabled"
    controls = (
        '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px;">'
        f'<button type="button" id="pool-entry-save"{disabled}>Save in this browser</button>'
        f'<button type="button" id="pool-entry-reset"{disabled}>Reset to model</button>'
        '<span id="pool-entry-status" class="fine" role="status" aria-live="polite">'
        + (
            "Model card loaded; local edits are not saved yet."
            if scoped
            else "A season and week are required before an entry can be saved."
        )
        + "</span></div>"
    )
    note = (
        "Choose one ATS side for every game and one Best Pick. Save stores this "
        "week's entry only in this browser; it does not change or publish the "
        "model forecast. Model probability and rank remain model aids only -- "
        "residual magnitude has not proven to rank pick quality (see the findings page)."
    )
    return _section(
        "Entry list",
        "Forced picks, editable entry",
        controls + table + f'<p class="fine" style="margin-top:8px;">{escape(note)}</p>',
    )


def _ownership_section(summaries: pd.DataFrame) -> str:
    if summaries.empty:
        inner = viz.empty_state(
            "No entry to compare",
            "Ownership sensitivity appears once a forced-pick card is available.",
        )
        return _section("Ownership scenarios", "Hypothetical field overlap", inner)

    rows: list[str] = []
    for _, row in summaries.iterrows():
        rows.append(
            f'<td data-ownership-scenario="{escape(str(row["key"]))}" '
            f'data-favorite-share="{float(row["favorite_share"]):.6f}">'
            f"{escape(str(row['label']))}</td>"
            f"<td>{float(row['favorite_share']):.0%}</td>"
            f"<td data-entry-share>{float(row['entry_side_share']):.1%}</td>"
            f"<td data-disagreements>{float(row['disagreements_per_100']):.1f}</td>"
            f"<td>{escape(str(row['note']))}</td>"
        )
    table = _data_table(
        [
            "Hypothetical field",
            "Favorite share assumed",
            "Field on this entry's sides",
            "Disagreements / 100 games",
            "Meaning",
        ],
        rows,
    )
    warning = viz.status_line("warning", "Sensitivity only — no ownership feed")
    note = (
        "These are arithmetic scenarios, not measured popularity and not a reason "
        "to flip a higher-EV forced pick. The favorite is only an observable proxy "
        "for the public side; 50%, 65%, and 85% are disclosed sensitivity inputs "
        "already used by the pool-format simulator. Pick'em contributes 50%."
    )
    return _section(
        "Ownership scenarios",
        "How chalky could this entry look?",
        warning + table + f'<p class="fine" style="margin-top:8px;">{escape(note)}</p>',
    )


def _entry_persistence_script() -> str:
    """Browser-local entry persistence and live ownership recomputation."""

    return f"""<script>
(function () {{
  "use strict";
  var root = document.getElementById("pool-entry-workbench");
  if (!root) return;
  var key = root.getAttribute("data-storage-key") || "";
  var status = document.getElementById("pool-entry-status");
  var save = document.getElementById("pool-entry-save");
  var reset = document.getElementById("pool-entry-reset");
  var pickInputs = Array.prototype.slice.call(root.querySelectorAll(".entry-pick"));
  var bestInputs = Array.prototype.slice.call(root.querySelectorAll(".entry-best"));

  function announce(message) {{ if (status) status.textContent = message; }}
  function selectedState() {{
    var picks = {{}};
    pickInputs.forEach(function (input) {{
      if (input.checked) {{
        picks[input.getAttribute("data-game-id")] = input.getAttribute("data-side");
      }}
    }});
    var best = bestInputs.filter(function (input) {{ return input.checked; }})[0];
    return {{ version: {ENTRY_STORAGE_VERSION}, picks: picks,
      bestPickGameId: best ? best.getAttribute("data-game-id") : null }};
  }}
  var modelState = selectedState();

  function applyState(state) {{
    if (!state || state.version !== {ENTRY_STORAGE_VERSION} || !state.picks ||
        typeof state.picks !== "object") return false;
    var known = {{}};
    pickInputs.forEach(function (input) {{
      var game = input.getAttribute("data-game-id");
      var side = input.getAttribute("data-side");
      known[game] = true;
      if ((state.picks[game] === "HOME" || state.picks[game] === "AWAY") &&
          state.picks[game] === side) input.checked = true;
    }});
    if (state.bestPickGameId && known[state.bestPickGameId]) {{
      bestInputs.forEach(function (input) {{
        input.checked = input.getAttribute("data-game-id") === state.bestPickGameId;
      }});
    }}
    return true;
  }}

  function updateOwnership() {{
    var selected = pickInputs.filter(function (input) {{ return input.checked; }});
    root.querySelectorAll("[data-ownership-scenario]").forEach(function (marker) {{
      var row = marker.parentElement;
      var favoriteShare = Number(marker.getAttribute("data-favorite-share"));
      var total = 0;
      selected.forEach(function (input) {{
        var favorite = input.getAttribute("data-favorite");
        total += favorite === "true" ? favoriteShare :
          (favorite === "false" ? 1 - favoriteShare : 0.5);
      }});
      var share = selected.length ? total / selected.length : 0;
      var shareCell = row.querySelector("[data-entry-share]");
      var gapCell = row.querySelector("[data-disagreements]");
      if (shareCell) shareCell.textContent = (share * 100).toFixed(1) + "%";
      if (gapCell) gapCell.textContent = ((1 - share) * 100).toFixed(1);
    }});
  }}

  if (key) {{
    try {{
      var raw = window.localStorage.getItem(key);
      if (raw && applyState(JSON.parse(raw))) announce("Saved entry restored from this browser.");
      else if (raw) announce("Saved entry was incompatible and was ignored; model card loaded.");
    }} catch (error) {{
      announce("Browser storage is unavailable; the model card remains usable without saving.");
    }}
  }}
  updateOwnership();

  root.addEventListener("change", function (event) {{
    if (event.target && (event.target.classList.contains("entry-pick") ||
        event.target.classList.contains("entry-best"))) {{
      updateOwnership();
      announce("Entry changed; save it in this browser when ready.");
    }}
  }});
  if (save) save.addEventListener("click", function () {{
    var state = selectedState();
    if (Object.keys(state.picks).length !== bestInputs.length) {{
      announce("Choose one ATS side for every game before saving."); return;
    }}
    if (bestInputs.length && !state.bestPickGameId) {{
      announce("Choose one Best Pick before saving."); return;
    }}
    try {{
      window.localStorage.setItem(key, JSON.stringify(state));
      announce("Entry saved in this browser for this season and week.");
    }} catch (error) {{
      announce("Browser storage is unavailable; this entry was not saved.");
    }}
  }});
  if (reset) reset.addEventListener("click", function () {{
    applyState(modelState); updateOwnership();
    try {{ window.localStorage.removeItem(key); }} catch (error) {{ /* fail open */ }}
    announce("Local entry cleared; model card restored.");
  }});
}}());
</script>"""


def build_pool_workbench_body(
    predictions: pd.DataFrame,
    pool_rules: PoolRules | None = None,
    *,
    best_pick_game_id: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> str:
    """Compose the pool-workbench body (rules, entry list, ownership)."""

    rules = pool_rules or PoolRules.from_defaults()
    entry_list = build_entry_list(predictions)
    ownership = build_ownership_scenarios(entry_list)

    if season is not None and week is not None:
        scope_line = f"Editing the {season} week {week} card"
    else:
        scope_line = "Active model forecast"

    header = viz.page_header(
        "Operations",
        PAGE_TITLE,
        f"{scope_line}. Forced picks against the pool's frozen Tuesday line; "
        "confidence ranks are a model-ordering aid, not a profitable-edge claim.",
    )
    storage_key = None
    if season is not None and week is not None:
        storage_key = f"nfl-ats:pool-entry:v{ENTRY_STORAGE_VERSION}:{season}:{week}"
    storage_attr = f' data-storage-key="{escape(storage_key)}"' if storage_key else ""
    content = "\n".join(
        [
            header,
            _rules_section(rules),
            _entry_list_section(
                entry_list,
                best_pick_game_id=best_pick_game_id,
                storage_key=storage_key,
            ),
            _ownership_section(ownership),
        ]
    )
    return f'<div id="pool-entry-workbench"{storage_attr}>{content}</div>' + (
        _entry_persistence_script() if not entry_list.empty else ""
    )


__all__ = [
    "DEFAULT_OWNERSHIP_SCENARIOS",
    "ENTRY_STORAGE_VERSION",
    "PAGE_FILENAME",
    "PAGE_TITLE",
    "PLAYOFF_GAMES",
    "REGULAR_SEASON_GAMES",
    "SUNDAY_EARLY_LOCK_ET",
    "OwnershipScenario",
    "PoolRules",
    "build_entry_list",
    "build_ownership_scenarios",
    "build_pool_workbench_body",
    "derive_confidence_ranks",
]
