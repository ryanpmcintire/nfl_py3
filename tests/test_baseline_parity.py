"""ENG-17/ENG-28: baseline-parity regression suite.

Proves the frozen fixture (``tests/fixtures/parity/games.csv``) grades
identically -- same games, same chronological cutoffs, same push handling --
across the market-only, simple-model, active-model, and overlay comparison
paths in ``nfl_ats.parity``. This is a PARITY suite, not a verdict: it never
asserts one path is more accurate than another, and it never touches
``registry/``. Per ``AGENTS.md``, an interval or accuracy gap crossing zero
is not evaluated here at all -- there is no interval anywhere in this file.

ENG-28 adds ``active_model_production``/``overlay_production``: the same
comparison, but through the REAL opener-snapshot machinery
(``nfl_ats.clv.opener_pick_evaluation`` at the real ``weak_stack``/
ridge-alpha-10 configuration, and
``nfl_ats.four_overlay_composition.apply_four_overlay_composition``) reading
the miniature, commit-safe market-snapshot store under
``tests/fixtures/parity/opener_store/``. These are intentionally NOT folded
into ``PATHS``/``all_results`` above: unlike the four original paths, whose
populations match by construction, the production paths' population is
allowed to differ from ``market_only`` by exactly the games the store lacks a
paired Tuesday-opener snapshot for -- one deliberately, in this fixture
(``CLOSE_ONLY_GAME_ID``). Folding them into the all-paths-identical
parametrization below would make that expected, documented divergence look
like a bug; see the dedicated ``PRODUCTION_PATHS`` section instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.parity import (
    MIN_TRAIN_GAMES,
    PUSH_RULE,
    PathResult,
    grade_games,
    load_fixture,
    paired_delta,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "games.csv"
OPENER_STORE_PATH = Path(__file__).parent / "fixtures" / "parity" / "opener_store"
PATHS = ("market_only", "simple_model", "active_model", "overlay")
PRODUCTION_PATHS = ("active_model_production", "overlay_production")
LINES = ("opener", "close")

#: The one game in the fixture with a close-snapshot but deliberately NO
#: tue_open snapshot in ``opener_store`` (see
#: ``.agent_tmp/generate_parity_opener_store.py``'s generator docstring) --
#: exercises the real 1,537-vs-1,552 snapshot-pair requirement documented in
#: ``docs/opener_evaluation.md``. Present in every walk-forward path's
#: population; absent from both production paths, at both lines.
CLOSE_ONLY_GAME_ID = "2022_05_T7_T8"


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    return load_fixture(FIXTURE_PATH)


@pytest.fixture(scope="module")
def all_results(fixture_frame: pd.DataFrame) -> dict[tuple[str, str], PathResult]:
    return {
        (line, path): grade_games(fixture_frame, line, path)  # type: ignore[arg-type]
        for line in LINES
        for path in PATHS
    }


@pytest.fixture(scope="module")
def production_results(fixture_frame: pd.DataFrame) -> dict[tuple[str, str], PathResult]:
    return {
        (line, path): grade_games(  # type: ignore[arg-type]
            fixture_frame, line, path, opener_store=OPENER_STORE_PATH
        )
        for line in LINES
        for path in PRODUCTION_PATHS
    }


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------


def test_fixture_is_small_and_spans_three_seasons(fixture_frame: pd.DataFrame) -> None:
    assert 40 <= len(fixture_frame) <= 65
    assert fixture_frame["season"].nunique() == 3
    assert fixture_frame["week"].nunique() >= 4
    assert fixture_frame["game_id"].is_unique


def test_fixture_has_a_true_opener_close_split(fixture_frame: pd.DataFrame) -> None:
    # Otherwise "opener" and "close" would trivially agree on every claim below.
    assert not fixture_frame["spread_line_open"].equals(fixture_frame["spread_line_close"])


# ---------------------------------------------------------------------------
# 1. Identical game-ID population, per line, across all four paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", LINES)
def test_all_four_paths_grade_the_identical_game_id_set(
    line: str, all_results: dict[tuple[str, str], PathResult]
) -> None:
    populations = {path: all_results[(line, path)].scored_game_ids for path in PATHS}
    reference = populations["market_only"]
    assert reference, "fixture produced no graded games at all"
    for path in PATHS:
        assert populations[path] == reference, (
            f"{path!r} scored a different game-ID set at the {line} line: "
            f"symmetric difference = {populations[path] ^ reference}"
        )


def test_populations_are_nontrivial_not_the_whole_fixture(
    fixture_frame: pd.DataFrame, all_results: dict[tuple[str, str], PathResult]
) -> None:
    # The MIN_FITTABLE_TRAIN_GAMES=50 floor must actually bite: most of the
    # fixture is warm-up, only the tail is graded. If this ever equals the
    # full fixture, the cutoff stopped doing anything and the test below
    # (chronological cutoffs) would be vacuous.
    scored = all_results[("opener", "active_model")].scored_game_ids
    assert 0 < len(scored) < len(fixture_frame)


# ---------------------------------------------------------------------------
# 2. Chronological cutoffs identical across paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", LINES)
def test_chronological_cutoffs_are_identical(
    line: str, all_results: dict[tuple[str, str], PathResult]
) -> None:
    skipped = {path: all_results[(line, path)].skipped_weeks for path in PATHS}
    reference = skipped["market_only"]
    assert reference, "no week was ever skipped -- the cutoff fixture design is broken"
    for path in PATHS:
        assert skipped[path] == reference, f"{path!r} skipped different weeks: {skipped[path]}"
    for result in (all_results[(line, path)] for path in PATHS):
        assert result.min_train_games == MIN_TRAIN_GAMES


def test_market_only_cutoff_agrees_with_walk_forward_backtest_own_cutoff(
    all_results: dict[tuple[str, str], PathResult],
) -> None:
    """Cross-check the ONE inlined cutoff helper against the real harness.

    ``market_only`` computes its eligible weeks with a small local helper
    (documented in ``nfl_ats.parity`` as mirroring, not reimplementing,
    ``backtest.walk_forward_backtest``'s own inlined cutoff). This asserts
    the two independent expressions of "which weeks are gradable" agree, for
    both real model-fitting paths.
    """

    market_skipped = all_results[("opener", "market_only")].skipped_weeks
    for path in ("simple_model", "active_model"):
        assert all_results[("opener", path)].skipped_weeks == market_skipped


# ---------------------------------------------------------------------------
# 3. Push handling identical across paths, and the rule is named
# ---------------------------------------------------------------------------


def test_push_rule_is_named() -> None:
    assert PUSH_RULE
    assert "push" in PUSH_RULE.lower()
    assert "excluded" in PUSH_RULE.lower()


@pytest.mark.parametrize("line", LINES)
def test_push_handling_is_identical_across_paths(
    line: str, all_results: dict[tuple[str, str], PathResult]
) -> None:
    pushed = {path: all_results[(line, path)].pushed_game_ids for path in PATHS}
    reference = pushed["market_only"]
    assert reference, f"fixture has no push at the {line} line; push parity is untested"
    for path in PATHS:
        assert pushed[path] == reference, f"{path!r} disagreed on which games pushed"
    # Excluded, never scored as a loss or half-win: every pushed game_id is
    # absent from correct_by_game (not present as False/0.5).
    for path in PATHS:
        result = all_results[(line, path)]
        assert reference.isdisjoint(result.correct_by_game)


def test_push_populations_differ_by_line_the_1537_vs_1503_pattern(
    all_results: dict[tuple[str, str], PathResult],
) -> None:
    """Same population, different line -> different push set -> different
    push-excluded count. This is exactly the mechanism documented in
    ``docs/opener_evaluation.md``/``HANDOFF.md``: the real 1,537-game paired
    archive has 34 pushes at the opener and 30 at the close (measured,
    ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``),
    giving 1,503 vs 1,507 evaluated games from the SAME 1,537-game
    population -- a population divergence entirely explained by the push
    rule being applied at two different lines, not a bug.
    """

    opener = all_results[("opener", "active_model")]
    close = all_results[("close", "active_model")]
    assert opener.scored_game_ids == close.scored_game_ids  # same population
    assert opener.pushed_game_ids != close.pushed_game_ids  # different push set
    assert len(opener.evaluated_game_ids) != len(close.evaluated_game_ids)


# ---------------------------------------------------------------------------
# 4. Paired delta on the intersection, with intersection size reported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", LINES)
def test_paired_delta_reports_intersection_size(
    line: str, all_results: dict[tuple[str, str], PathResult]
) -> None:
    a = all_results[(line, "market_only")]
    b = all_results[(line, "active_model")]
    delta = paired_delta(a, b)
    assert delta["intersection_size"] == len(a.evaluated_game_ids & b.evaluated_game_ids)
    assert delta["intersection_size"] > 0
    assert delta["delta_a_minus_b"] == pytest.approx(delta["a_accuracy"] - delta["b_accuracy"])


def test_paired_delta_on_disjoint_populations_reports_zero_intersection() -> None:
    a = PathResult(
        path="a",
        line="opener",
        min_train_games=50,
        push_rule=PUSH_RULE,
        scored_game_ids=frozenset({"g1"}),
        pushed_game_ids=frozenset(),
        evaluated_game_ids=frozenset({"g1"}),
        accuracy=1.0,
        skipped_weeks=(),
        correct_by_game={"g1": True},
    )
    b = PathResult(
        path="b",
        line="opener",
        min_train_games=50,
        push_rule=PUSH_RULE,
        scored_game_ids=frozenset({"g2"}),
        pushed_game_ids=frozenset(),
        evaluated_game_ids=frozenset({"g2"}),
        accuracy=0.0,
        skipped_weeks=(),
        correct_by_game={"g2": False},
    )
    delta = paired_delta(a, b)
    assert delta["intersection_size"] == 0
    assert pd.isna(delta["delta_a_minus_b"])


# ---------------------------------------------------------------------------
# 5. The suite catches a deliberately divergent population
#    (the 1,537-vs-1,503 SHAPE of bug: one path silently drops games)
# ---------------------------------------------------------------------------


def test_suite_catches_a_deliberately_divergent_population(fixture_frame: pd.DataFrame) -> None:
    """A path that silently drops graded games must show up as a symmetric
    difference, not be masked by comparing accuracy alone.

    Simulates the exact shape of the real divergence this item is named
    for: HANDOFF.md records the active model at 53.36% on 1,503 games and a
    sibling policy at 53.76% on 1,503 games -- both computed off the SAME
    1,537-game archive but with a couple dozen fewer games than 1,537
    because of how each read handled pushes/coverage. A parity suite that
    only compared the two accuracy PERCENTAGES would never have caught that;
    this test asserts the game-ID SET comparison does.
    """

    corrupted = fixture_frame.copy()
    graded = grade_games(fixture_frame, "opener", "active_model").scored_game_ids
    # Drop two graded games' rows entirely, as if a candidate path lost them
    # (e.g. a stricter join, a coverage gap) -- exactly the shape of a
    # population bug, not an accuracy difference.
    drop_ids = sorted(graded)[:2]
    corrupted = corrupted.loc[~corrupted["game_id"].isin(drop_ids)].reset_index(drop=True)

    healthy = grade_games(fixture_frame, "opener", "market_only")
    broken = grade_games(corrupted, "opener", "market_only")

    assert healthy.scored_game_ids != broken.scored_game_ids
    symmetric_difference = healthy.scored_game_ids ^ broken.scored_game_ids
    assert symmetric_difference == set(drop_ids)
    assert len(symmetric_difference) == 2


# ---------------------------------------------------------------------------
# 6. Overlay-specific: population preserved, at least one real trigger fires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", LINES)
def test_overlay_preserves_the_active_model_population(
    line: str, all_results: dict[tuple[str, str], PathResult]
) -> None:
    overlay = all_results[(line, "overlay")]
    active = all_results[(line, "active_model")]
    assert overlay.scored_game_ids == active.scored_game_ids
    assert overlay.pushed_game_ids == active.pushed_game_ids


def test_overlay_has_at_least_one_real_trigger(
    all_results: dict[tuple[str, str], PathResult],
) -> None:
    # The fixture includes one clean-case year-1-coach game (T1 in 2022,
    # coach differs from 2021; its opponent's coach does not) inside
    # coach_fade_overlay's weeks-1-8 window -- the "couple of overlay
    # triggers" the fixture is required to carry (ENG-17 item 1). Both lines
    # fit the same weekly-refit model and are checked, since either grading
    # is a legitimate real path.
    flips = {line: all_results[(line, "overlay")].flipped_game_ids for line in LINES}
    assert any(flips.values()), "fixture's year-1-coach trigger never fired the overlay"


def test_grade_games_rejects_unknown_path(fixture_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        grade_games(fixture_frame, "opener", "not_a_real_path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. ENG-28: production paths through the REAL opener-snapshot machinery
# ---------------------------------------------------------------------------


def test_production_paths_require_opener_store(fixture_frame: pd.DataFrame) -> None:
    for path in PRODUCTION_PATHS:
        with pytest.raises(ValueError):
            grade_games(fixture_frame, "opener", path)  # type: ignore[arg-type]


@pytest.mark.parametrize("line", LINES)
@pytest.mark.parametrize("path", PRODUCTION_PATHS)
def test_production_populations_are_nontrivial(
    line: str, path: str, production_results: dict[tuple[str, str], PathResult]
) -> None:
    # 8 graded games (2022 wk4 + wk5) minus the one deliberately missing its
    # tue_open snapshot -- see CLOSE_ONLY_GAME_ID.
    result = production_results[(line, path)]
    assert len(result.scored_game_ids) == 7


def test_production_paths_match_market_only_population_where_opener_snapshot_exists(
    all_results: dict[tuple[str, str], PathResult],
    production_results: dict[tuple[str, str], PathResult],
) -> None:
    """The core ENG-28 claim: same games, at the opener, once snapshot coverage agrees.

    ``market_only`` (and every walk-forward path) grades ``CLOSE_ONLY_GAME_ID``
    because it never consults the opener store; the production paths cannot,
    because ``opener_pick_evaluation``'s pairing table requires BOTH a
    tue_open and a close snapshot. Restricting ``market_only``'s population to
    the games that DO have a paired opener snapshot is exactly the 1,537 side
    of the real 1,537-vs-1,552 pattern (``docs/opener_evaluation.md``), now
    reproduced mechanically rather than asserted.
    """

    market_only_opener = all_results[("opener", "market_only")].scored_game_ids
    expected = market_only_opener - {CLOSE_ONLY_GAME_ID}
    assert expected, "fixture's paired-snapshot population collapsed to nothing"
    for path in PRODUCTION_PATHS:
        result = production_results[("opener", path)]
        assert result.scored_game_ids == expected, (
            f"{path!r} scored a different game-ID set than market_only's "
            f"opener-snapshot-paired population: "
            f"symmetric difference = {result.scored_game_ids ^ expected}"
        )


@pytest.mark.parametrize("line", LINES)
def test_missing_opener_snapshot_excludes_a_game_from_production_paths_only(
    line: str,
    all_results: dict[tuple[str, str], PathResult],
    production_results: dict[tuple[str, str], PathResult],
) -> None:
    """The 1,537-vs-1,552 rule, now encoded: a snapshot gap is NOT a data bug.

    ``CLOSE_ONLY_GAME_ID`` is graded by every path that never touches the
    opener store (market_only, simple_model, active_model, overlay) and by
    NEITHER production path, at EITHER line -- both production paths agree
    with each other about the exclusion, which is what "excluded from ALL
    paths consistently" means here (consistently between the two paths that
    share the same snapshot-pairing requirement, not with the four paths that
    have no such requirement to share).
    """

    for path in PATHS:
        assert CLOSE_ONLY_GAME_ID in all_results[(line, path)].scored_game_ids

    for path in PRODUCTION_PATHS:
        result = production_results[(line, path)]
        assert CLOSE_ONLY_GAME_ID not in result.scored_game_ids


@pytest.mark.parametrize("line", LINES)
def test_production_paths_push_handling_matches_market_only(
    line: str,
    all_results: dict[tuple[str, str], PathResult],
    production_results: dict[tuple[str, str], PathResult],
) -> None:
    market_pushed = all_results[(line, "market_only")].pushed_game_ids - {CLOSE_ONLY_GAME_ID}
    assert market_pushed, f"fixture has no push at the {line} line in the paired population"
    for path in PRODUCTION_PATHS:
        result = production_results[(line, path)]
        assert result.pushed_game_ids == market_pushed, (
            f"{path!r} disagreed with market_only on which paired games pushed at {line}"
        )
        # Excluded, never scored as a loss or half-win -- same contract as
        # section 3 above, now checked on the real production output.
        assert market_pushed.isdisjoint(result.correct_by_game)


def test_production_paths_skipped_weeks_match_market_only(
    all_results: dict[tuple[str, str], PathResult],
    production_results: dict[tuple[str, str], PathResult],
) -> None:
    """Chronological cutoff parity (section 2's claim) extended to the real path.

    ``active_model_production``'s ``skipped_weeks`` is computed by the SAME
    ``_eligible_weeks`` helper ``market_only`` uses (see
    ``nfl_ats.parity._grade_active_model_production``'s docstring) rather than
    read off ``opener_pick_evaluation``, which reports scored games only. This
    checks that independent computation still agrees with ``market_only``'s.
    """

    for line in LINES:
        reference = all_results[(line, "market_only")].skipped_weeks
        assert reference, "no week was ever skipped -- the cutoff fixture design is broken"
        for path in PRODUCTION_PATHS:
            result = production_results[(line, path)]
            assert result.skipped_weeks == reference, (
                f"{path!r} skipped different weeks at {line}: {result.skipped_weeks}"
            )
            assert result.min_train_games == MIN_TRAIN_GAMES


@pytest.mark.parametrize("line", LINES)
def test_overlay_production_preserves_active_model_production_population(
    line: str, production_results: dict[tuple[str, str], PathResult]
) -> None:
    overlay = production_results[(line, "overlay_production")]
    active = production_results[(line, "active_model_production")]
    assert overlay.scored_game_ids == active.scored_game_ids
    assert overlay.pushed_game_ids == active.pushed_game_ids


def test_production_paired_delta_reports_intersection_size(
    production_results: dict[tuple[str, str], PathResult],
) -> None:
    a = production_results[("opener", "active_model_production")]
    b = production_results[("opener", "overlay_production")]
    delta = paired_delta(a, b)
    assert delta["intersection_size"] == len(a.evaluated_game_ids & b.evaluated_game_ids)
    assert delta["intersection_size"] > 0
    assert delta["delta_a_minus_b"] == pytest.approx(delta["a_accuracy"] - delta["b_accuracy"])


def test_suite_catches_a_divergent_production_population(
    fixture_frame: pd.DataFrame,
) -> None:
    """Section 5's divergent-population detector, exercised on the real production path.

    Corrupting the fixture the same way ``test_suite_catches_a_deliberately_
    divergent_population`` does (dropping two graded rows) must show up as a
    symmetric difference on ``active_model_production`` too -- the detector
    is a property of comparing game-ID SETS, not specific to which path
    produced them.
    """

    healthy = grade_games(
        fixture_frame, "opener", "active_model_production", opener_store=OPENER_STORE_PATH
    )
    graded = healthy.scored_game_ids
    drop_ids = sorted(graded)[:2]
    corrupted = fixture_frame.loc[~fixture_frame["game_id"].isin(drop_ids)].reset_index(drop=True)
    broken = grade_games(
        corrupted, "opener", "active_model_production", opener_store=OPENER_STORE_PATH
    )

    assert healthy.scored_game_ids != broken.scored_game_ids
    symmetric_difference = healthy.scored_game_ids ^ broken.scored_game_ids
    assert symmetric_difference == set(drop_ids)
    assert len(symmetric_difference) == 2
