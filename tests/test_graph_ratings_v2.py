from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.graph_ratings_v2 import (
    AWAY_INJURY_VALUE_LOST_COLUMNS,
    HOME_INJURY_VALUE_LOST_COLUMNS,
    GraphRatingV2Config,
    _constrain_row_linf,
    _injury_discount,
    _katz_fixed_point,
    add_graph_ratings_v2_features,
    cfb_structural_coherence,
    katz_feature_columns,
    select_structural_config_on_cfb,
    signed_katz_centrality,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(
    week: int,
    home: str,
    away: str,
    result: float,
    spread_line: float,
    *,
    season: int = 2022,
    home_score: float | None = None,
    away_score: float | None = None,
) -> dict[str, object]:
    if home_score is None:
        home_score = 24.0 + max(0.0, result)
    if away_score is None:
        away_score = home_score - result
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "week": week,
        "gameday": pd.Timestamp("2022-09-01") + pd.Timedelta(days=7 * week),
        "away_team": away,
        "home_team": home,
        "away_score": away_score,
        "home_score": home_score,
        "result": result,
        "spread_line": spread_line,
    }


def _config(**overrides: object) -> GraphRatingV2Config:
    values: dict[str, object] = {"min_games": 2, "half_life_weeks": 6.0}
    values.update(overrides)
    return GraphRatingV2Config(**values)  # type: ignore[arg-type]


def _no_market_games() -> pd.DataFrame:
    """Four teams, spread_line == 0 everywhere: residual == raw margin."""

    matchups = (
        (1, "B", "A", 10.0),
        (1, "D", "C", 3.0),
        (2, "C", "A", 7.0),
        (2, "D", "B", 6.0),
        (3, "D", "A", 14.0),
        (3, "C", "B", 4.0),
        (4, "A", "C", -3.0),
        (4, "B", "D", -7.0),
        (5, "B", "A", 8.0),
        (5, "D", "C", 1.0),
    )
    return pd.DataFrame(
        [_row(week, home, away, margin, 0.0) for week, away, home, margin in matchups]
    )


def _chalk_and_dog_games(weeks: int = 6) -> pd.DataFrame:
    """CHALK always wins on the scoreboard but fails to cover; DOG always
    loses on the scoreboard but always covers. A clean, known-answer
    inversion between the raw-margin control arm and the residual arm.
    """

    rows = []
    for week in range(1, weeks + 1):
        # CHALK (home) beats MID by 14 but was favored by 24 -> ats_margin = -10.
        rows.append(_row(week, "CHALK", "MID1", 14.0, 24.0))
        # MID (home) beats DOG (away) by 14 while favored by 24 -> ats_margin = -10
        # for MID, i.e. DOG (the away underdog) covers decisively.
        rows.append(_row(week, "MID2", "DOG", 14.0, 24.0))
    return pd.DataFrame(rows)


def _push_games(weeks: int = 6) -> pd.DataFrame:
    """Every game lands exactly on the number: ats_margin == 0 always."""

    rows = [
        _row(week, "A" if week % 2 else "B", "B" if week % 2 else "A", 3.0, 3.0)
        for week in range(1, weeks + 1)
    ]
    return pd.DataFrame(rows)


def _cfb_like_games(weeks: int = 20, teams: int = 8, seed: int = 20260826) -> pd.DataFrame:
    """A larger synthetic multi-team round robin with genuine game-to-game
    variance in both `result` and `spread_line` (unlike the deterministic
    CHALK/DOG fixture, which is intentionally constant and would make any
    Pearson correlation undefined). Used only for the CFB structural-fitting
    tests, which need a non-degenerate signal column to correlate against.
    """

    rng = np.random.default_rng(seed)
    team_names = [f"T{i}" for i in range(teams)]
    strengths = {team: rng.normal(scale=6.0) for team in team_names}
    rows = []
    for week in range(1, weeks + 1):
        shuffled = list(team_names)
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            home, away = shuffled[i], shuffled[i + 1]
            true_gap = strengths[home] - strengths[away] + 2.0
            spread_line = round(float(true_gap + rng.normal(scale=1.0)), 1)
            result = round(float(true_gap + rng.normal(scale=10.0)), 1)
            rows.append(_row(week, home, away, result, spread_line, season=2022))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Numeric primitives: convergence, the row-Linf constraint, Katz correctness.
# ---------------------------------------------------------------------------


def test_row_linf_constraint_rescales_only_rows_over_the_bound() -> None:
    matrix = np.array([[0.0, 3.0, -4.0], [0.1, 0.0, 0.1], [0.0, 0.0, 0.0]])
    constrained = _constrain_row_linf(matrix, 1.0)
    row_l1 = np.abs(constrained).sum(axis=1)
    assert row_l1[0] == pytest.approx(1.0)
    # Direction/ratio preserved within the rescaled row.
    assert constrained[0, 1] / constrained[0, 2] == pytest.approx(matrix[0, 1] / matrix[0, 2])
    # Row already under the bound is untouched.
    np.testing.assert_array_equal(constrained[1], matrix[1])
    np.testing.assert_array_equal(constrained[2], matrix[2])


def test_katz_fixed_point_converges_when_spectral_radius_bound_holds() -> None:
    rng = np.random.default_rng(20260826)
    raw = rng.normal(size=(6, 6))
    np.fill_diagonal(raw, 0.0)
    constrained = _constrain_row_linf(raw, 1.0)
    alpha = 0.85  # alpha * max_row_l1(<=1) < 1: the module's own validated guarantee.
    base = np.ones(6)
    x, converged = _katz_fixed_point(
        constrained, base, alpha=alpha, iterations=500, tolerance=1e-10
    )
    assert converged
    assert np.all(np.isfinite(x))
    # The fixed point must actually satisfy x = base + alpha * W @ x.
    residual = x - (base + alpha * (constrained @ x))
    assert float(np.abs(residual).max()) < 1e-6


def test_katz_fixed_point_diverges_when_spectral_radius_bound_is_violated() -> None:
    """The convergence test the design doc requires: feed the ITERATOR a
    matrix that violates ||alpha*W||_inf < 1 (skipping the row constraint on
    purpose) and confirm it does NOT converge. This is the direct evidence
    that the row-Linf constraint is load-bearing, not decorative.
    """

    size = 5
    # Every row's absolute sum is exactly 5 -- alpha * 5 = 4.25 with alpha=0.85,
    # far past the spectral-radius bound the module relies on.
    raw = np.full((size, size), 1.0)
    np.fill_diagonal(raw, 0.0)
    raw[:, 0] = 1.25  # keep row sums at 5 after zeroing the diagonal on row 0
    base = np.ones(size)
    _, converged = _katz_fixed_point(raw, base, alpha=0.85, iterations=500, tolerance=1e-10)
    assert not converged

    # The SAME matrix, constrained first, converges -- isolating the
    # constraint as the difference that makes convergence possible.
    constrained = _constrain_row_linf(raw, 1.0)
    _, converged_after_constraint = _katz_fixed_point(
        constrained, base, alpha=0.85, iterations=500, tolerance=1e-10
    )
    assert converged_after_constraint


def test_signed_katz_matches_the_closed_form_linear_solve() -> None:
    """Proves the iteration correctly implements x = (I - alpha*W)^-1 v."""

    rng = np.random.default_rng(7)
    n = 8
    raw = rng.normal(scale=2.0, size=(n, n))
    np.fill_diagonal(raw, 0.0)
    alpha = 0.7
    max_row_l1 = 1.0
    iterative = signed_katz_centrality(
        raw, alpha=alpha, max_row_l1=max_row_l1, tolerance=1e-13, iterations=2000
    )
    constrained = _constrain_row_linf(raw, max_row_l1)
    closed_form = np.linalg.solve(np.eye(n) - alpha * constrained, np.ones(n))
    np.testing.assert_allclose(iterative, closed_form, atol=1e-6)


def test_signed_katz_empty_graph_is_the_base_vector() -> None:
    x = signed_katz_centrality(np.zeros((4, 4)), alpha=0.85, max_row_l1=1.0)
    np.testing.assert_allclose(x, np.ones(4))


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("half_life_weeks", 0.0, "half_life"),
        ("offseason_retention", 1.1, "offseason_retention"),
        ("offseason_age_weeks", -1, "offseason_age_weeks"),
        ("alpha", 1.0, "alpha"),
        ("max_row_l1", 0.0, "max_row_l1"),
        ("max_row_l1", -0.5, "max_row_l1"),
        ("prior_weight", -1.0, "prior_weight"),
        ("min_games", 0, "min_games"),
        ("injury_beta", -0.1, "injury_beta"),
        ("iterations", 0, "iterations"),
        ("tolerance", 0.0, "tolerance"),
        ("edge_signal", "bogus", "edge_signal"),
        ("propagation", "bogus", "propagation"),
    ],
)
def test_configuration_guards(setting: str, value: object, message: str) -> None:
    values = _config().__dict__ | {setting: value}
    with pytest.raises(ValueError, match=message):
        add_graph_ratings_v2_features(_no_market_games(), GraphRatingV2Config(**values))  # type: ignore[arg-type]


def test_alpha_times_max_row_l1_must_be_strictly_below_one() -> None:
    """This is the exact spectral-radius guarantee stated in the design doc:
    alpha * max_row_l1 bounds the spectral radius of the propagation matrix.
    """

    with pytest.raises(ValueError, match="spectral radius"):
        GraphRatingV2Config(alpha=0.6, max_row_l1=2.0).validate()
    # A configuration that keeps the product below 1 is accepted.
    GraphRatingV2Config(alpha=0.5, max_row_l1=0.9).validate()


def test_schema_guard_requires_spread_line_only_for_residual_arm() -> None:
    games = _no_market_games().drop(columns=["spread_line"])
    with pytest.raises(ValueError, match="graph_ratings_v2 requires columns"):
        add_graph_ratings_v2_features(games, _config(edge_signal="residual"))
    # The control arm never touches spread_line.
    add_graph_ratings_v2_features(games, _config(edge_signal="raw_margin"))


def test_schema_guard_requires_injury_columns_only_when_the_modifier_is_on() -> None:
    games = _no_market_games()
    add_graph_ratings_v2_features(games, _config(injury_beta=0.0))
    with pytest.raises(ValueError, match="graph_ratings_v2 requires columns"):
        add_graph_ratings_v2_features(games, _config(injury_beta=0.05))


# ---------------------------------------------------------------------------
# Synthetic validation with known answers.
# ---------------------------------------------------------------------------


def test_residual_arm_ranks_who_beats_the_number_not_who_wins() -> None:
    """CHALK always wins the scoreboard but never covers; DOG always loses
    the scoreboard but always covers. The raw-margin control arm and the
    residual arm must DISAGREE on which team ranks higher -- this is the
    direct, known-answer demonstration that residual edges measure
    something other than team quality.
    """

    games = _chalk_and_dog_games()

    residual = add_graph_ratings_v2_features(games, _config(edge_signal="residual"))
    control = add_graph_ratings_v2_features(games, _config(edge_signal="raw_margin"))

    residual_columns = katz_feature_columns(_config(edge_signal="residual"))
    control_columns = katz_feature_columns(_config(edge_signal="raw_margin"))

    last_week = games["week"].max()
    chalk_residual = residual.loc[
        (residual["week"] == last_week) & (residual["home_team"] == "CHALK"), residual_columns[0]
    ].iloc[0]
    dog_residual = residual.loc[
        (residual["week"] == last_week) & (residual["away_team"] == "DOG"), residual_columns[1]
    ].iloc[0]
    assert (
        dog_residual > chalk_residual
    )  # DOG (covers every game) outranks CHALK under residual edges.

    chalk_control = control.loc[
        (control["week"] == last_week) & (control["home_team"] == "CHALK"), control_columns[0]
    ].iloc[0]
    dog_control = control.loc[
        (control["week"] == last_week) & (control["away_team"] == "DOG"), control_columns[1]
    ].iloc[0]
    assert (
        chalk_control > dog_control
    )  # CHALK (wins every game) outranks DOG under raw-margin edges.


def test_residual_and_raw_margin_arms_agree_when_the_market_is_uninformative() -> None:
    """With spread_line == 0 everywhere, ats_margin == result identically,
    so the two arms must produce byte-identical ratings -- a consistency
    check tying the "treatment" and "control" arms to the same mechanism.
    """

    games = _no_market_games()
    residual = add_graph_ratings_v2_features(games, _config(edge_signal="residual"))
    control = add_graph_ratings_v2_features(games, _config(edge_signal="raw_margin"))
    residual_columns = katz_feature_columns(_config(edge_signal="residual"))
    control_columns = katz_feature_columns(_config(edge_signal="raw_margin"))
    np.testing.assert_allclose(
        residual[list(residual_columns)].to_numpy(dtype=float),
        control[list(control_columns)].to_numpy(dtype=float),
    )


def test_uncompressed_magnitude_scales_the_rating_gap() -> None:
    """A one-point win and a forty-point win must NOT produce nearly the
    same edge -- the defect the original module had (``1.0 + min(margin,
    28.0) / 14.0`` squeezed every edge into ``[1.0, 3.0]``). Tested on the
    Katz primitive directly, not through the full standardized pipeline:
    with only two teams, z-scoring always yields an exact +-2.0 gap
    regardless of the underlying magnitude (population std with n=2 is
    exactly half the raw gap), which would mask this property rather than
    demonstrate it. The RAW (unstandardized) centrality gap is where the
    uncompressed weighting actually shows up, and it must scale with the
    edge magnitude the way the original's compressed ``[1.0, 3.0]`` band
    never could.
    """

    def _raw_gap(margin: float) -> float:
        matrix = np.zeros((2, 2))
        matrix[0, 1] = margin  # team 0 beat team 1 by `margin`, uncompressed.
        matrix[1, 0] = -margin
        # A large max_row_l1 keeps the row-sum constraint (tested on its own
        # in test_row_linf_constraint_rescales_only_rows_over_the_bound and
        # test_adversarial_blowout_density_stays_finite_under_the_row_constraint)
        # from binding on a single isolated edge, isolating the property
        # under test here: whether raw magnitude survives uncompressed.
        x = signed_katz_centrality(matrix, alpha=0.01, max_row_l1=50.0)
        return float(x[0] - x[1])

    small_gap = _raw_gap(1.0)
    large_gap = _raw_gap(40.0)
    assert large_gap > small_gap
    # Not just larger -- meaningfully larger, not "nearly the same edge".
    assert large_gap > small_gap * 1.5


def test_exact_pushes_carry_no_signal() -> None:
    """Every game lands exactly on the number: ats_margin == 0 always, so
    no edges are ever added and every team's rating stays at the neutral
    baseline (standardized diff of exactly zero).
    """

    games = _push_games()
    rated = add_graph_ratings_v2_features(games, _config())
    columns = katz_feature_columns(_config())
    diffs = rated[columns[2]].dropna()
    assert len(diffs) > 0
    np.testing.assert_allclose(diffs.to_numpy(dtype=float), 0.0, atol=1e-12)


def test_adversarial_blowout_density_stays_finite_under_the_row_constraint() -> None:
    """Many uncapped blowouts in a short, low-decay window would blow the
    row-Linf sum well past 1 without the constraint. End to end, ratings
    must stay finite regardless.
    """

    rows = []
    teams = ["A", "B", "C", "D", "E"]
    for week in range(1, 12):
        for i in range(0, len(teams) - 1, 2):
            rows.append(_row(week, teams[i], teams[i + 1], 63.0, 0.0))
    games = pd.DataFrame(rows)
    rated = add_graph_ratings_v2_features(games, _config(half_life_weeks=52.0, min_games=2))
    columns = katz_feature_columns(_config())
    values = rated[list(columns)].to_numpy(dtype=float)
    finite = values[~np.isnan(values)]
    assert finite.size > 0
    assert np.all(np.isfinite(finite))


def test_nonneg_arm_offense_defense_reward_outscoring_opponents() -> None:
    games = _no_market_games()
    rated = add_graph_ratings_v2_features(games, _config(propagation="nonneg_pagerank"))
    columns = katz_feature_columns(_config(propagation="nonneg_pagerank"))
    late = rated.loc[rated["week"] == rated["week"].max()].iloc[0]
    assert set(columns).issubset(rated.columns)
    assert late[columns[2]] == pytest.approx(late[columns[0]] - late[columns[1]])
    assert late[columns[7]] == pytest.approx(
        (late[columns[3]] - late[columns[6]]) - (late[columns[4]] - late[columns[5]])
    )


# ---------------------------------------------------------------------------
# Injury modifier: off-by-default contract, and its measurable effect when on.
# ---------------------------------------------------------------------------


def test_injury_discount_is_a_pure_function_of_the_underperformer() -> None:
    assert _injury_discount(0.0, 0.0, 5.0, beta=0.1) == pytest.approx(1.0 / (1.0 + 0.1 * 0.0))
    # Home over-performed (signal > 0): discount reads AWAY's injury loss.
    assert _injury_discount(home_lost=10.0, away_lost=4.0, signal=5.0, beta=0.1) == pytest.approx(
        1.0 / (1.0 + 0.1 * 4.0)
    )
    # Away over-performed (signal < 0): discount reads HOME's injury loss.
    assert _injury_discount(home_lost=10.0, away_lost=4.0, signal=-5.0, beta=0.1) == pytest.approx(
        1.0 / (1.0 + 0.1 * 10.0)
    )
    # beta == 0 is a hard off-switch.
    assert _injury_discount(home_lost=99.0, away_lost=99.0, signal=5.0, beta=0.0) == 1.0
    # A push (signal == 0) carries no signal regardless of beta.
    assert _injury_discount(home_lost=99.0, away_lost=99.0, signal=0.0, beta=0.5) == 1.0


def _games_with_injury_columns(heavy: bool) -> pd.DataFrame:
    games = _chalk_and_dog_games(weeks=6)
    for column in HOME_INJURY_VALUE_LOST_COLUMNS + AWAY_INJURY_VALUE_LOST_COLUMNS:
        games[column] = 0.0
    if heavy:
        # Both game types in _chalk_and_dog_games have ats_margin = -10, i.e.
        # the HOME team (CHALK, or MID2) is always the side that fell short
        # of the spread -- the injury discount reads the UNDERPERFORMER's
        # loss, which is home's here, not away's. CHALK's games specifically
        # carry the heavy injury load.
        games.loc[games["home_team"] == "CHALK", "home_injury_skill_epa_value_lost"] = 8.0
    return games


def test_injury_modifier_is_a_true_no_op_at_beta_zero() -> None:
    plain = add_graph_ratings_v2_features(_chalk_and_dog_games(), _config(injury_beta=0.0))
    with_columns = add_graph_ratings_v2_features(
        _games_with_injury_columns(heavy=True), _config(injury_beta=0.0)
    )
    columns = katz_feature_columns(_config(injury_beta=0.0))
    pd.testing.assert_frame_equal(
        plain[list(columns)].reset_index(drop=True),
        with_columns[list(columns)].reset_index(drop=True),
    )


def test_injury_modifier_discounts_the_underperformers_edge_when_enabled() -> None:
    """A large max_row_l1 (paired with a small alpha to keep the spectral-
    radius guard satisfied) keeps the row-Linf constraint from saturating
    both arms to the same capped value -- CHALK's uncapped accumulated
    magnitude across 6 blowout-sized games is well past a max_row_l1 of 1.0,
    so at the module's default row cap the injury discount's effect would be
    invisible until it were strong enough to pull the row back under the
    cap. That interaction (the row constraint can mask a weak discount) is
    real and expected, not this test's concern; here the cap is relaxed so
    the discount's own effect is isolated and visible at a realistic beta.
    """

    injmod_config = _config(injury_beta=0.2, alpha=0.01, max_row_l1=50.0)
    light = add_graph_ratings_v2_features(_games_with_injury_columns(heavy=False), injmod_config)
    heavy = add_graph_ratings_v2_features(_games_with_injury_columns(heavy=True), injmod_config)
    columns = katz_feature_columns(injmod_config)
    last_week = light["week"].max()
    light_gap = light.loc[light["week"] == last_week, columns[2]].to_numpy(dtype=float)
    heavy_gap = heavy.loc[heavy["week"] == last_week, columns[2]].to_numpy(dtype=float)
    assert not np.allclose(light_gap, heavy_gap)


# ---------------------------------------------------------------------------
# Leakage regression tests.
# ---------------------------------------------------------------------------


def test_current_week_outcomes_cannot_change_current_ratings() -> None:
    games = _chalk_and_dog_games()
    baseline = add_graph_ratings_v2_features(games, _config())
    changed = games.copy()
    current = changed["week"].eq(3)
    changed.loc[current, "result"] *= -5.0
    changed.loc[current, "spread_line"] *= -5.0
    rerun = add_graph_ratings_v2_features(changed, _config())

    columns = list(katz_feature_columns(_config()))
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["week"].eq(3), columns].reset_index(drop=True),
        rerun.loc[rerun["week"].eq(3), columns].reset_index(drop=True),
    )
    assert not np.allclose(
        baseline.loc[baseline["week"].eq(4), columns].to_numpy(dtype=float),
        rerun.loc[rerun["week"].eq(4), columns].to_numpy(dtype=float),
    )


def test_future_outcomes_spread_and_input_order_cannot_change_prior_ratings() -> None:
    games = _chalk_and_dog_games()
    baseline = add_graph_ratings_v2_features(games, _config())
    changed = games.sample(frac=1.0, random_state=91).copy()
    future = changed["week"].eq(changed["week"].max())
    changed.loc[future, ["result", "spread_line", "home_score", "away_score"]] = np.nan
    rerun = add_graph_ratings_v2_features(changed, _config())

    columns = ["game_id", *katz_feature_columns(_config())]
    cutoff = games["week"].max() - 1
    expected = baseline.loc[baseline["week"].le(cutoff), columns].sort_values("game_id")
    actual = rerun.loc[rerun["week"].le(cutoff), columns].sort_values("game_id")
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


def test_future_injury_values_cannot_change_prior_ratings() -> None:
    games = _games_with_injury_columns(heavy=False)
    config = _config(injury_beta=0.2)
    baseline = add_graph_ratings_v2_features(games, config)
    changed = games.copy()
    future = changed["week"].eq(changed["week"].max())
    changed.loc[future & (changed["home_team"] == "CHALK"), "home_injury_skill_epa_value_lost"] = (
        25.0
    )
    rerun = add_graph_ratings_v2_features(changed, config)

    columns = ["game_id", *katz_feature_columns(config)]
    cutoff = games["week"].max() - 1
    expected = baseline.loc[baseline["week"].le(cutoff), columns].sort_values("game_id")
    actual = rerun.loc[rerun["week"].le(cutoff), columns].sort_values("game_id")
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


# ---------------------------------------------------------------------------
# CFB structural fitting (non-ATS coherence diagnostic).
# ---------------------------------------------------------------------------


def test_cfb_structural_coherence_is_finite_and_bounded() -> None:
    games = _cfb_like_games(weeks=12)
    coherence = cfb_structural_coherence(games, _config(edge_signal="residual"))
    assert np.isfinite(coherence)
    assert -1.0 <= coherence <= 1.0


def test_select_structural_config_on_cfb_returns_the_best_scoring_candidate() -> None:
    games = _cfb_like_games(weeks=12)
    candidates = [
        _config(edge_signal="residual", alpha=0.5),
        _config(edge_signal="residual", alpha=0.85),
        _config(edge_signal="residual", half_life_weeks=1.0),
    ]
    best, table = select_structural_config_on_cfb(games, candidates)
    assert len(table) == len(candidates)
    assert best in candidates
    best_row = table.loc[
        (table["alpha"] == best.alpha) & (table["half_life_weeks"] == best.half_life_weeks)
    ].iloc[0]
    assert abs(best_row["coherence"]) == pytest.approx(table["coherence"].abs().max())


def test_select_structural_config_on_cfb_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_structural_config_on_cfb(_cfb_like_games(), [])


# ---------------------------------------------------------------------------
# The team_stat arm: one graph per SCREENED statistic (docs/graph_input_screen.md).
# ---------------------------------------------------------------------------


def _with_team_stat(
    games: pd.DataFrame,
    family: str,
    home_values: dict[str, float],
    default: float = 0.0,
) -> pd.DataFrame:
    """Attach a pregame ``home_<family>``/``away_<family>`` pair keyed by team.

    The screened families are PREGAME rolling team values (measured:
    ``home_def_takeaway_rate`` is populated in week 1), so a per-team constant
    is a faithful, degenerate case of the real column shape.
    """

    frame = games.copy()
    frame[f"home_{family}"] = [home_values.get(str(team), default) for team in frame["home_team"]]
    frame[f"away_{family}"] = [home_values.get(str(team), default) for team in frame["away_team"]]
    return frame


def test_team_stat_arm_requires_a_signal_column() -> None:
    with pytest.raises(ValueError, match="requires signal_column"):
        _config(edge_signal="team_stat").validate()


def test_signal_column_is_rejected_on_the_outcome_arms() -> None:
    with pytest.raises(ValueError, match="only meaningful with edge_signal='team_stat'"):
        _config(edge_signal="residual", signal_column="def_takeaway_rate").validate()


def test_signal_column_rejects_a_side_prefixed_name() -> None:
    """The config takes the FAMILY name; passing one side of the pair would
    silently look for ``home_home_def_takeaway_rate`` and fail far away from
    the mistake.
    """

    with pytest.raises(ValueError, match="FAMILY name"):
        _config(edge_signal="team_stat", signal_column="home_def_takeaway_rate").validate()


def test_schema_guard_names_the_missing_team_stat_pair() -> None:
    config = _config(edge_signal="team_stat", signal_column="def_takeaway_rate")
    with pytest.raises(DataContractError) as excinfo:
        add_graph_ratings_v2_features(_no_market_games(), config)
    message = str(excinfo.value)
    assert "home_def_takeaway_rate" in message
    assert "away_def_takeaway_rate" in message


def test_team_stat_arm_matches_the_control_when_the_pair_encodes_the_margin() -> None:
    """Known-answer: if ``home_x - away_x`` is constructed to equal ``result``
    game by game, the team_stat arm and the raw_margin control must produce
    byte-identical ratings. Any divergence between the arms on real data then
    comes from the statistic, not from a hidden implementation difference --
    the same discipline
    ``test_residual_and_raw_margin_arms_agree_when_the_market_is_uninformative``
    applies to the residual arm.
    """

    games = _no_market_games()
    staged = games.copy()
    staged["home_stub"] = pd.to_numeric(staged["result"], errors="coerce")
    staged["away_stub"] = 0.0

    stat_config = _config(edge_signal="team_stat", signal_column="stub")
    control_config = _config(edge_signal="raw_margin")
    stat = add_graph_ratings_v2_features(staged, stat_config)
    control = add_graph_ratings_v2_features(games, control_config)

    np.testing.assert_allclose(
        stat[list(katz_feature_columns(stat_config))].to_numpy(dtype=float),
        control[list(katz_feature_columns(control_config))].to_numpy(dtype=float),
    )


def test_team_stat_arm_ranks_the_team_the_statistic_favours() -> None:
    """A team whose screened statistic exceeds every opponent's must rate
    above one whose statistic trails every opponent's, and swapping the pair
    must invert the ordering -- proving the arm reads the pair rather than
    any outcome column.
    """

    family = "def_takeaway_rate"
    values = {"A": 0.030, "B": 0.020, "C": 0.015, "D": 0.010}
    games = _with_team_stat(_no_market_games(), family, values)
    config = _config(edge_signal="team_stat", signal_column=family)
    rated = add_graph_ratings_v2_features(games, config)
    home_column, away_column, _ = katz_feature_columns(config)

    def _mean_rating(frame: pd.DataFrame, team: str) -> float:
        home = frame.loc[frame["home_team"] == team, home_column]
        away = frame.loc[frame["away_team"] == team, away_column]
        return float(pd.concat([home, away]).dropna().mean())

    assert _mean_rating(rated, "A") > _mean_rating(rated, "D")

    swapped = games.copy()
    swapped[f"home_{family}"], swapped[f"away_{family}"] = (
        games[f"away_{family}"],
        games[f"home_{family}"],
    )
    inverted = add_graph_ratings_v2_features(swapped, config)
    assert _mean_rating(inverted, "A") < _mean_rating(inverted, "D")


def test_team_stat_columns_are_namespaced_per_family() -> None:
    """Two screened families must be computable side by side on one frame
    without colliding -- the whole point of the namespacing, since the screen
    hands over 38 cluster representatives, not one.
    """

    first = _config(edge_signal="team_stat", signal_column="def_takeaway_rate")
    second = _config(edge_signal="team_stat", signal_column="off_cpoe")
    assert not set(katz_feature_columns(first)) & set(katz_feature_columns(second))
    assert not set(katz_feature_columns(first)) & set(
        katz_feature_columns(_config(edge_signal="raw_margin"))
    )

    games = _with_team_stat(
        _with_team_stat(_no_market_games(), "def_takeaway_rate", {"A": 0.03, "D": 0.01}),
        "off_cpoe",
        {"A": -0.02, "D": 0.04},
    )
    merged = add_graph_ratings_v2_features(add_graph_ratings_v2_features(games, first), second)
    for column in (*katz_feature_columns(first), *katz_feature_columns(second)):
        assert column in merged.columns
    assert (
        merged[katz_feature_columns(first)[2]].dropna().to_numpy()
        != merged[katz_feature_columns(second)[2]].dropna().to_numpy()
    ).any()


def test_future_team_stat_values_cannot_change_prior_ratings() -> None:
    """Leakage regression for the new input family, mirroring the outcome and
    injury arms: blanking and violently perturbing a future week's statistic
    leaves every prior week's ratings unchanged.
    """

    family = "def_takeaway_rate"
    games = _with_team_stat(
        _no_market_games(), family, {"A": 0.030, "B": 0.020, "C": 0.015, "D": 0.010}
    )
    config = _config(edge_signal="team_stat", signal_column=family)
    baseline = add_graph_ratings_v2_features(games, config)

    changed = games.copy()
    future = changed["week"].eq(changed["week"].max())
    changed.loc[future, f"home_{family}"] = 99.0
    changed.loc[future, f"away_{family}"] = -99.0
    rerun = add_graph_ratings_v2_features(changed, config)

    columns = ["game_id", *katz_feature_columns(config)]
    cutoff = games["week"].max() - 1
    expected = baseline.loc[baseline["week"].le(cutoff), columns].sort_values("game_id")
    actual = rerun.loc[rerun["week"].le(cutoff), columns].sort_values("game_id")
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


def test_cfb_structural_coherence_refuses_the_team_stat_arm() -> None:
    """Correlating the rating diff against the very quantity the edges were
    built from is a self-correlation, not a structural diagnostic. The
    team_stat arm inherits the frozen structural hyperparameters instead.
    """

    family = "def_takeaway_rate"
    games = _with_team_stat(_cfb_like_games(weeks=12), family, {"T0": 0.03, "T1": 0.01})
    with pytest.raises(ValueError, match="self-correlation"):
        cfb_structural_coherence(games, _config(edge_signal="team_stat", signal_column=family))


def test_team_stat_arm_reads_the_suffix_naming_convention() -> None:
    """The feature table carries both conventions (measured:
    ``home_def_takeaway_rate`` is prefix-form, ``gap_division_revenge_home``
    is suffix-form). A suffix-form family must be expressible, not silently
    unreachable -- five of the screen's 38 cluster representatives are
    suffix-form.
    """

    games = _no_market_games()
    values = {"A": 0.030, "B": 0.020, "C": 0.015, "D": 0.010}
    suffix = games.copy()
    suffix["revenge_home"] = [values[str(t)] for t in suffix["home_team"]]
    suffix["revenge_away"] = [values[str(t)] for t in suffix["away_team"]]

    prefix = _with_team_stat(games, "revenge", values)

    pair_config = _config(
        edge_signal="team_stat",
        signal_column="revenge",
        signal_column_pair=("revenge_home", "revenge_away"),
    )
    prefix_config = _config(edge_signal="team_stat", signal_column="revenge")

    by_pair = add_graph_ratings_v2_features(suffix, pair_config)
    by_prefix = add_graph_ratings_v2_features(prefix, prefix_config)
    np.testing.assert_allclose(
        by_pair[list(katz_feature_columns(pair_config))].to_numpy(dtype=float),
        by_prefix[list(katz_feature_columns(prefix_config))].to_numpy(dtype=float),
    )


def test_signal_column_pair_is_rejected_on_the_outcome_arms() -> None:
    with pytest.raises(ValueError, match="only meaningful with"):
        _config(edge_signal="raw_margin", signal_column_pair=("a", "b")).validate()
