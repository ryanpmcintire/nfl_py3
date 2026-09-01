"""Shared ``BoardContent`` fixture for the ATS Terminal renderer tests.

``board_terminal.render`` is a pure function over
:class:`nfl_ats.board_content.BoardContent`, so its tests never need a real
artifact tree -- a hand-built content object exercises the renderer exactly
as thoroughly as one built from real artifacts. Shared here so
``tests/test_board_terminal.py`` and ``tests/test_board_content_coverage.py``
(the content-coverage guarantee that replaced the old cross-skin parity
check once the Cover Desk skin was dropped) both render from the SAME
fixture.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

from nfl_ats.board_content import (
    AttributionPanel,
    AttributionRow,
    BoardContent,
    CoverCurvePoint,
    Disclaimer,
    Finding,
    GameDive,
    GameRow,
    HeadlineStats,
    LinkPreview,
    PolicyNote,
    ProspectiveScoreboard,
    SpreadAdjusterParams,
    TickerChrome,
)

#: 16 (away, home, spread_line, home_cover_probability) tuples -- enough
#: variety to exercise every confidence band and a multi-day board.
_FIXTURE_GAMES: tuple[tuple[str, str, float, float, str, date], ...] = (
    ("NE", "SEA", 3.5, 0.517, "Wednesday", date(2026, 9, 9)),
    ("SF", "LA", 3.5, 0.464, "Thursday", date(2026, 9, 10)),
    ("ARI", "LAC", 10.5, 0.362, "Sunday", date(2026, 9, 13)),
    ("ATL", "PIT", 3.0, 0.465, "Sunday", date(2026, 9, 13)),
    ("BAL", "IND", -3.5, 0.516, "Sunday", date(2026, 9, 13)),
    ("BUF", "HOU", -1.5, 0.561, "Sunday", date(2026, 9, 13)),
    ("CHI", "CAR", -2.5, 0.534, "Sunday", date(2026, 9, 13)),
    ("CLE", "JAX", 7.5, 0.521, "Sunday", date(2026, 9, 13)),
    ("DAL", "NYG", -2.5, 0.495, "Sunday", date(2026, 9, 13)),
    ("GB", "MIN", 1.5, 0.536, "Sunday", date(2026, 9, 13)),
    ("MIA", "LV", 3.5, 0.456, "Sunday", date(2026, 9, 13)),
    ("NO", "DET", 7.0, 0.538, "Sunday", date(2026, 9, 13)),
    ("NYJ", "TEN", 3.0, 0.495, "Sunday", date(2026, 9, 13)),
    ("TB", "CIN", 3.5, 0.526, "Sunday", date(2026, 9, 13)),
    ("WAS", "PHI", 4.5, 0.383, "Sunday", date(2026, 9, 13)),
    ("DEN", "KC", 3.0, 0.571, "Monday", date(2026, 9, 14)),
)

#: The Best Pick game -- matches the real 2026 Week 1 card's MIA @ LV pick,
#: so the fixture's shape mirrors the pilot's real data.
BEST_PICK_GAME_ID = "2026_01_MIA_LV"


def _confidence_word(probability: float) -> str:
    if probability > 0.56:
        return "strong"
    if probability >= 0.53:
        return "lean"
    return "slight"


def build_fixture_games() -> tuple[GameRow, ...]:
    games = []
    for away, home, home_spread, home_cover_probability, weekday, gameday in _FIXTURE_GAMES:
        game_id = f"2026_01_{away}_{home}"
        pick_team = home if home_cover_probability >= 0.5 else away
        pick_probability = (
            home_cover_probability
            if home_cover_probability >= 0.5
            else 1.0 - home_cover_probability
        )
        games.append(
            GameRow(
                game_id=game_id,
                gameday=gameday,
                weekday_name=weekday,
                home=home,
                away=away,
                market_spread=home_spread,
                pick_team=pick_team,
                pick_probability=pick_probability,
                confidence_word=_confidence_word(pick_probability),
                is_best=game_id == BEST_PICK_GAME_ID,
                is_flipped=game_id == "2026_01_BAL_IND",
                flip_member_labels=("coach fade",) if game_id == "2026_01_BAL_IND" else (),
            )
        )
    return tuple(games)


def _best_pick_attribution(game: GameRow) -> AttributionPanel:
    return AttributionPanel(
        available=True,
        game_id=game.game_id,
        matchup_label=f"{game.pick_team} {game.pick_spread_text} at {game.home}",
        probability_text=game.probability_text,
        rows=(
            AttributionRow(
                label="Market residual",
                delta_points=3.1,
                cumulative_points=3.1,
                bar_width_pct=31.0,
            ),
            AttributionRow(
                label="Injury channel",
                delta_points=1.4,
                cumulative_points=4.5,
                bar_width_pct=14.0,
            ),
            AttributionRow(
                label="Coach policy overlay",
                delta_points=-0.3,
                cumulative_points=4.2,
                bar_width_pct=3.0,
            ),
            AttributionRow(
                label="Arrest-policy overlay",
                delta_points=0.2,
                cumulative_points=4.4,
                bar_width_pct=2.0,
            ),
        ),
        net_points=4.4,
        net_label=f"Net toward {game.pick_team} {game.pick_spread_text}",
    )


_BEST_PICK_COVER_CURVE = tuple(
    CoverCurvePoint(offset=offset, probability=0.5 + offset * 0.02)
    for offset in (
        -4.0,
        -3.5,
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    )
)


def build_fixture_dives(games: tuple[GameRow, ...]) -> tuple[GameDive, ...]:
    """One dive per game: the Best Pick gets a full real attribution panel,
    real cover curve (deliberately mismatched at offset 0 -- see the note
    text below -- exercising the "designed disclosure" path), and a
    line-offset adjuster; every other game exercises the degraded
    "attribution not published" / "cover curve not published" / "adjuster
    unavailable" paths, which the renderer must handle without raising."""

    dives = []
    for game in games:
        if game.is_best:
            attribution = _best_pick_attribution(game)
            cover_curve = _BEST_PICK_COVER_CURVE
            note = (
                "This chart's own swept line reads 50.0% at the card's quoted line -- the "
                f"live cover probability shown elsewhere on this page is {game.probability_text}."
                " The swept curve can predate the latest weekly refresh; the live number is "
                "the one actually played."
            )
            adjuster = SpreadAdjusterParams(
                center=1.2,
                residual_mean=0.4,
                residual_std=6.5,
                card_line=game.market_spread,
                pick_is_home=game.pick_team == game.home,
            )
        else:
            attribution = AttributionPanel(available=False)
            cover_curve = ()
            note = None
            adjuster = None
        flip_note = (
            f"The raw model favored {game.away} to cover this line; the played card flips to "
            f"{game.pick_team} via coach fade."
            if game.flip_member_labels
            else None
        )
        dives.append(
            GameDive(
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
                cover_curve_offset_zero_note=note,
                adjuster=adjuster,
                flip_note=flip_note,
            )
        )
    return tuple(dives)


def build_fixture_content() -> BoardContent:
    """A complete, deterministic 16-game ``BoardContent`` fixture."""

    games = build_fixture_games()
    headline = HeadlineStats(
        model_id="d1f07d773475dc58",
        model_method_label="weak_stack (market_residual ridge)",
        synced_at_text="2026-08-24 12:07 UTC",
        played_card_pct=55.4225,
        played_card_value_text="55.4%",
        played_card_caption=(
            "Opener-graded accuracy across 1,503 paired games -- the four-member overlay "
            "union that is actually on the board this week, not a hypothetical."
        ),
        played_card_foot_text="1,503 opener-graded games · four-member overlay union",
        selection_caveat_text=(
            "This 55.4% was selected from 127 correlated subsets of the same overlay "
            "members -- it is not a prospective expectation. The operating expectation "
            "going forward is roughly +1 accuracy point over the prior coach-to-arrests "
            "chain (54.2%), and is being tracked prospectively in fresh paired games "
            "against that prior chain, not restated as this archive figure."
        ),
        prior_chain_pct=54.1583,
        prior_chain_value_text="54.2%",
        prior_chain_caption=(
            "1,503 paired opener-graded games -- the prior policy this week's chain is "
            "tracked against. No interval attached; reported as a reference point."
        ),
        raw_model_pct=53.4,
        raw_model_value_text="53.4%",
        raw_model_ci=(51.97, 54.56),
        raw_model_caption=(
            "95% CI [51.97%, 54.56%], season-blocked -- 6 of 6 seasons finished above the "
            "coin flip."
        ),
        raw_model_season_count=6,
        raw_model_seasons_above_coin_flip=6,
        raw_model_season_note="6 of 6 seasons finished above the coin flip",
        close_grade_pct=52.10,
        close_grade_value_text="52.10%",
        close_grade_caption=(
            "1,081 of 2,075 non-push games correctly classified -- week-blocked 95% "
            "interval [50.12%, 54.24%]."
        ),
        prospective_scoreboard=ProspectiveScoreboard(
            dormant=True,
            headline_text="Prospective tracking begins Week 1.",
            detail_text=None,
        ),
    )
    policy = PolicyNote(
        composition_text=(
            "Synchronized with the active model · 4 strong leans · 1 pick flipped "
            "by the coach-fade overlay"
        ),
        rich_narrative=(
            "Policy overlay active -- coach fade, division revenge, player arrests, "
            "spread-gap zone. Flipped 2 picks vs. the raw model this week."
        ),
        policy_id="overlay_union_coach_division_revenge_player_arrests_spread_gap_v1",
        policy_fingerprint="bbdd60a171238654",
        members_text="coach fade, division revenge, player arrests, spread-gap zone",
    )
    findings = (
        Finding(
            tag="NOTE // CONFIDENCE SHAPE",
            text=(
                "This week's most confident call is ARI +10.5 at 63.8% cover probability -- "
                "the strongest lean on the board. Most other games cluster just above the "
                "coin flip, which is the honest, expected shape for a market this efficient."
            ),
        ),
        Finding(
            tag="NOTE // OVERLAY REACH",
            text=(
                "The overlay nudges picks, it doesn't rebuild them. This week's active "
                "overlay policy touched 1 of this week's 16 raw-model picks -- small, "
                "deliberate, and reported separately from the raw model so each layer's "
                "contribution stays visible."
            ),
        ),
    )
    disclaimer = Disclaimer(
        short=(
            "Research project — simulated, paper picks only. Not betting advice. A small "
            "historical edge is not proof of a profitable one."
        ),
        full=(
            "This page is the output of a personal research project. Every pick shown is a "
            "simulated, paper pick made to evaluate a forecasting model."
        ),
    )
    return BoardContent(
        season=2026,
        week=1,
        game_type="REG",
        week_label="Week 1",
        generated_at=datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC),
        generated_at_text="2026-08-31 12:00:00 UTC",
        games=games,
        best_pick_game_id=BEST_PICK_GAME_ID,
        best_pick_note=(
            "This pick was nominated by calibrated probability among low-disagreement games."
        ),
        flip_count=1,
        strong_count=sum(1 for game in games if game.confidence_word == "strong"),
        headline=headline,
        policy=policy,
        dives=build_fixture_dives(games),
        findings=findings,
        disclaimer=disclaimer,
        ticker_chrome=TickerChrome(
            games=games,
            best_pick_game_id=BEST_PICK_GAME_ID,
            season=2026,
            week=1,
            model_method_label=headline.model_method_label,
            page_command_suffix="",
        ),
        link_preview=LinkPreview(
            title="ATS Terminal — Week 1",
            description=f"This week's forced-pick board: {headline.played_card_foot_text}.",
        ),
    )


def build_fixture_content_with_degraded_states() -> BoardContent:
    """Same fixture, but with EVERY game's dive in its designed
    "unavailable" state -- exercises the degraded paths the renderer must
    handle without raising, for the Best Pick's own panel too (not just the
    other 15, which the default fixture already exercises)."""

    content = build_fixture_content()
    degraded_dives = tuple(
        replace(
            dive,
            attribution=AttributionPanel(available=False),
            cover_curve=(),
            cover_curve_offset_zero_note=None,
            adjuster=None,
        )
        for dive in content.dives
    )
    return replace(content, dives=degraded_dives)
