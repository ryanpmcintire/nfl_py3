from __future__ import annotations

import itertools

import pandas as pd
import pytest

from nfl_ats.survivor import build_survivor_plan, survivor_plan_markdown


def _predictions(probabilities: list[tuple[float, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week, (alpha_probability, beta_probability) in enumerate(probabilities, start=1):
        rows.extend(
            [
                {
                    "season": 2026,
                    "week": week,
                    "game_id": f"2026_{week:02d}_A_X{week}",
                    "gameday": f"2026-09-{week + 1:02d}",
                    "away_team": f"X{week}",
                    "home_team": "A",
                    "method": "straight_up",
                    "home_win_probability": alpha_probability,
                },
                {
                    "season": 2026,
                    "week": week,
                    "game_id": f"2026_{week:02d}_B_Y{week}",
                    "gameday": f"2026-09-{week + 1:02d}",
                    "away_team": f"Y{week}",
                    "home_team": "B",
                    "method": "straight_up",
                    "home_win_probability": beta_probability,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_planner_trades_current_probability_for_future_team_value() -> None:
    predictions = _predictions([(0.80, 0.75), (0.95, 0.55)])

    plan = build_survivor_plan(predictions)

    assert plan["team"].tolist() == ["B", "A"]
    assert plan["pick_probability"].tolist() == pytest.approx([0.75, 0.95])
    assert plan["current_probability_sacrifice"].tolist() == pytest.approx([0.05, 0.0])
    assert plan["horizon_survival_probability"].unique().tolist() == pytest.approx([0.7125])
    assert plan.loc[0, "future_team_opportunity_cost"] == pytest.approx(0.0)
    assert plan.loc[0, "survival_probability_from_week"] == pytest.approx(0.7125)
    assert plan["cumulative_survival_probability"].tolist() == pytest.approx([0.75, 0.7125])


def test_future_opportunity_cost_values_a_team_needed_later() -> None:
    predictions = _predictions([(0.80, 0.79), (0.95, 0.55)])

    plan = build_survivor_plan(predictions, locked_picks={1: "A"})

    assert plan["team"].tolist() == ["A", "B"]
    assert plan.loc[0, "is_locked"]
    assert plan.loc[0, "future_survival_probability"] == pytest.approx(0.55)
    assert plan.loc[0, "future_team_opportunity_cost"] == pytest.approx(0.40)


def test_optimizer_matches_brute_force_and_never_reuses_a_team() -> None:
    predictions = _predictions([(0.61, 0.72), (0.84, 0.73), (0.76, 0.91)])

    plan = build_survivor_plan(predictions)
    options: list[list[tuple[str, float]]] = []
    for _, week_rows in predictions.groupby("week", sort=True):
        week_options: list[tuple[str, float]] = []
        for row in week_rows.itertuples(index=False):
            week_options.extend(
                [
                    (str(row.home_team), float(row.home_win_probability)),
                    (str(row.away_team), 1.0 - float(row.home_win_probability)),
                ]
            )
        options.append(week_options)
    best = max(
        first[1] * second[1] * third[1]
        for first, second, third in itertools.product(*options)
        if len({first[0], second[0], third[0]}) == 3
    )

    assert plan["team"].is_unique
    assert plan.loc[0, "horizon_survival_probability"] == pytest.approx(best)


def test_used_and_locked_teams_are_audited_and_cannot_be_reused() -> None:
    predictions = _predictions([(0.80, 0.75), (0.95, 0.55)])

    plan = build_survivor_plan(predictions, used_teams=["A"], locked_picks={1: "B"})

    assert plan["team"].tolist() == ["B", "Y2"]
    assert plan.loc[0, "used_teams_before"] == "A"
    assert plan.loc[1, "used_teams_before"] == "A,B"
    with pytest.raises(ValueError, match="cannot be reused"):
        build_survivor_plan(predictions, used_teams=["A"], locked_picks={1: "A"})
    with pytest.raises(ValueError, match="duplicate team"):
        build_survivor_plan(predictions, used_teams=["A", "a"])


def test_plan_is_stable_under_input_order_and_exact_ties() -> None:
    predictions = _predictions([(0.70, 0.70), (0.70, 0.70)])

    first = build_survivor_plan(predictions)
    second = build_survivor_plan(predictions.sample(frac=1.0, random_state=9))

    assert first["team"].tolist() == second["team"].tolist()
    assert first["game_id"].tolist() == second["game_id"].tolist()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="week"), "missing survivor columns"),
        (lambda frame: frame.assign(home_win_probability=1.1), "must lie in"),
        (lambda frame: frame.assign(home_win_probability=float("nan")), "finite numbers"),
        (lambda frame: frame.assign(season=[2026, 2026, 2027, 2027]), "exactly one season"),
    ],
)
def test_prediction_contract_fails_closed(mutate: object, message: str) -> None:
    predictions = _predictions([(0.80, 0.75), (0.95, 0.55)])

    with pytest.raises(ValueError, match=message):
        build_survivor_plan(mutate(predictions))  # type: ignore[operator]


def test_schedule_contract_rejects_gaps_duplicate_teams_and_missing_weeks() -> None:
    predictions = _predictions([(0.80, 0.75), (0.95, 0.55), (0.70, 0.65)])
    duplicate_team = predictions.copy()
    duplicate_team.loc[duplicate_team.index[1], "home_team"] = "A"

    with pytest.raises(ValueError, match="must be consecutive"):
        build_survivor_plan(predictions, weeks=[1, 3])
    with pytest.raises(ValueError, match="missing weeks"):
        build_survivor_plan(predictions, weeks=[1, 2, 4])
    with pytest.raises(ValueError, match="schedules team A more than once"):
        build_survivor_plan(duplicate_team)


def test_infeasible_assignment_fails_instead_of_reusing_team() -> None:
    predictions = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": week,
                "game_id": f"g{week}",
                "gameday": f"2026-09-{week + 1:02d}",
                "away_team": "B",
                "home_team": "A",
                "method": "straight_up",
                "home_win_probability": 0.6,
            }
            for week in range(1, 4)
        ]
    )

    with pytest.raises(ValueError, match="fewer available teams"):
        build_survivor_plan(predictions)


def test_method_filter_and_markdown_keep_provenance_visible() -> None:
    predictions = _predictions([(0.80, 0.75), (0.95, 0.55)])
    other = predictions.assign(method="market_residual", home_win_probability=0.99)

    plan = build_survivor_plan(pd.concat([other, predictions], ignore_index=True))
    markdown = survivor_plan_markdown(plan)

    assert plan["method"].eq("straight_up").all()
    assert "Straight-up method: `straight_up`" in markdown
    assert "Each team is used at most once" in markdown
    assert "model estimates, not guarantees" in markdown
