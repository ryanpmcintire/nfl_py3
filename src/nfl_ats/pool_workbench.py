"""Minimal pool workbench: pool rules input, entry list, confidence ranks,
and an ownership-scenario placeholder.

ROADMAP item UI-09. The workbench reuses the existing forced-pick card from
:mod:`nfl_ats.pool` and the active model's ``recommendations.csv`` forecast
format; it invents nothing about future games. The ownership scenario is a
placeholder because no entry-popularity feed is integrated yet.

The pool's format is the one confirmed in ``docs/pool_edge_plan.md``
(owner-corrected 2026-08-20): forced ATS sides for all 272 regular-season
games plus all 13 playoff games (285 cards, exactly the forced-pick metric
this project evaluates), one "Best Pick" per regular-season week, no passes,
and a line that locks Tuesday (revised once Wednesday, then frozen for the
week) while our picks stay editable up to each game's own per-game deadline
(SNF/MNF lock early at Sunday 16:00 ET).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from html import escape
from typing import Any

import pandas as pd

from nfl_ats.dashboard import viz
from nfl_ats.pool import build_ats_pool_card

PAGE_FILENAME = "pool.html"
PAGE_TITLE = "Pool workbench"

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
    best_pick_per_regular_season_week: int = 1
    best_pick_bonus: float = 1.0
    best_pick_penalty: float = 0.0
    forced_picks: bool = True
    passes_allowed: bool = False
    line_locks_tuesday: bool = True
    picks_due_per_game_kickoff: bool = True
    sunday_early_lock: str = SUNDAY_EARLY_LOCK_ET

    @property
    def total_games(self) -> int:
        """Forced picks across the whole season: 272 + 13 = 285 (measured)."""

        return self.regular_season_games + self.playoff_games

    @classmethod
    def from_defaults(cls) -> PoolRules:
        """The confirmed Splash-style format from docs/pool_edge_plan.md."""

        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PoolRules:
        """Load pool rules from a partial override map (the rules INPUT)."""

        known = {field.name for field in fields(cls)}
        overrides = {key: value for key, value in dict(data).items() if key in known}
        return cls(**overrides)


# ---------------------------------------------------------------------------
# Entry list + confidence ranks (reuse the existing forced-pick card)
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
    (the forecast's calibrated cover probability). Same source as the entry
    list, surfaced as the ranking view the workbench sorts by.
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
# Ownership-scenario placeholder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnershipScenario:
    """Placeholder ownership scenario: no entry-popularity feed is wired yet."""

    available: bool = False
    note: str = (
        "Ownership scenarios are a placeholder: no entry-popularity feed is "
        "integrated, so the workbench cannot yet show how many entrants hold "
        "each pick or simulate contrarian Best-Pick leverage."
    )
    best_pick_game_id: str | None = None
    entrants_placeholder: int | None = None


def placeholder_ownership_scenarios(
    *,
    best_pick_game_id: str | None = None,
    entrants: int | None = None,
) -> OwnershipScenario:
    """Return the ownership-scenario placeholder for the workbench."""

    return OwnershipScenario(
        available=False,
        best_pick_game_id=best_pick_game_id,
        entrants_placeholder=entrants,
    )


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


def _pick_words(row: pd.Series) -> str:
    side = "home" if str(row["pool_side"]) == "HOME" else "away"
    return f"{escape(str(row['pool_pick']))} ({side})"


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


def _entry_list_section(entry_list: pd.DataFrame, *, best_pick_game_id: str | None = None) -> str:
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
        star = ' <span class="best-flag">&#9733;</span>' if is_best else ""
        word = confidence_word(float(row["pick_probability"]))
        rows.append(
            "<td>"
            f"{int(row['confidence_rank'])}</td>"
            f"<td>{escape(str(row['away_team']))} at {escape(str(row['home_team']))}</td>"
            f"<td>{_pick_words(row)}{star}</td>"
            f"<td>{row['pick_probability']:.1%}</td>"
            f"<td>{confidence_meter(word)}{row['confidence']:.1%}</td>"
        )
    table = _data_table(["#", "Game", "Pick", "Cover prob", "Confidence"], rows)
    note = (
        "★ marks the nominated Best Pick (one per regular-season week). "
        "Confidence = |cover probability minus 50%|."
    )
    return _section(
        "Entry list",
        "Forced picks, model order",
        table + f'<p class="fine" style="margin-top:8px;">{escape(note)}</p>',
    )


def _confidence_ranks_section(ranks: pd.DataFrame) -> str:
    if ranks.empty:
        inner = viz.empty_state(
            "No forecast yet",
            "Confidence ranks derive from the active model's calibrated cover "
            "probabilities; they appear once a forecast card exists.",
        )
        return _section("Confidence ranks", "Derived from the active forecast", inner)

    rows: list[str] = []
    for _, row in ranks.iterrows():
        meter = viz.probability_meter(float(row["pick_probability"]), label="cover")
        rows.append(
            "<td>"
            f"{int(row['confidence_rank'])}</td>"
            f"<td>{escape(str(row['away_team']))} at {escape(str(row['home_team']))}</td>"
            f"<td>{escape(str(row['pool_pick']))}</td>"
            f"<td>{meter}</td>"
            f"<td>{row['confidence']:.1%}</td>"
        )
    table = _data_table(["#", "Game", "Pick", "Cover probability", "Confidence"], rows)
    note = (
        "Rank 1 is the model's highest-confidence pick. The confidence ordering is a "
        "model aid only: the model's residual magnitude has not proven to rank pick "
        "quality (see the findings page)."
    )
    return _section(
        "Confidence ranks",
        "Derived from the active forecast",
        table + f'<p class="fine" style="margin-top:8px;">{escape(note)}</p>',
    )


def _ownership_section(ownership: OwnershipScenario) -> str:
    status = viz.status_line("warning", "Placeholder — no feed")
    body = f'<p class="prose" style="margin:8px 0 0;">{escape(ownership.note)}</p>'
    if ownership.best_pick_game_id:
        body += (
            f'<p class="fine" style="margin-top:6px;">Best Pick candidate: '
            f"{escape(str(ownership.best_pick_game_id))}</p>"
        )
    return _section("Ownership scenarios", "Contrarian leverage (placeholder)", status + body)


def build_pool_workbench_body(
    predictions: pd.DataFrame,
    pool_rules: PoolRules | None = None,
    *,
    best_pick_game_id: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> str:
    """Compose the pool-workbench body (rules, entry list, ranks, ownership)."""

    rules = pool_rules or PoolRules.from_defaults()
    entry_list = build_entry_list(predictions)
    ranks = derive_confidence_ranks(predictions)
    ownership = placeholder_ownership_scenarios(best_pick_game_id=best_pick_game_id)

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
    return "\n".join(
        [
            header,
            _rules_section(rules),
            _entry_list_section(entry_list, best_pick_game_id=best_pick_game_id),
            _confidence_ranks_section(ranks),
            _ownership_section(ownership),
        ]
    )


__all__ = [
    "PAGE_FILENAME",
    "PAGE_TITLE",
    "PLAYOFF_GAMES",
    "REGULAR_SEASON_GAMES",
    "SUNDAY_EARLY_LOCK_ET",
    "OwnershipScenario",
    "PoolRules",
    "build_entry_list",
    "build_pool_workbench_body",
    "derive_confidence_ranks",
    "placeholder_ownership_scenarios",
]
