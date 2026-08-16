"""Research: team & matchup explorer -- pregame state trends and one-matchup explanations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.active_model import active_artifact_path
from nfl_ats.dashboard.data import (
    artifacts_root,
    data_root,
    load_active_model,
    load_weekly_ats_forecast,
)
from nfl_ats.dashboard.ui import favorite_label

st.title("Team & matchup explorer")
st.caption("Inspect what the model knew about a team, or why it leans a certain way in one game.")


def _team_game_rows(features: pd.DataFrame, team: str, season: int) -> pd.DataFrame:
    games = features.loc[
        features["season"].eq(season)
        & (features["home_team"].eq(team) | features["away_team"].eq(team))
    ].copy()
    rows = []
    for _, game in games.iterrows():
        side = "home" if game["home_team"] == team else "away"
        opponent = game["away_team"] if side == "home" else game["home_team"]
        rows.append(
            {
                "week": int(game["week"]),
                "gameday": game["gameday"],
                "opponent": opponent,
                "venue": "Home" if side == "home" else "Away",
                "off_epa": game[f"{side}_off_epa_per_play"],
                "def_epa_allowed": game[f"{side}_def_epa_per_play"],
                "point_diff_state": game[f"{side}_point_diff"],
                "ats_state": game[f"{side}_ats_residual"],
            }
        )
    return pd.DataFrame(rows).sort_values("week") if rows else pd.DataFrame()


def _team_explorer() -> None:
    feature_path = data_root() / "processed" / "game_features.parquet"
    if not feature_path.is_file():
        st.warning("No feature table. Run `nfl-ats build-features`.")
        return
    features = pd.read_parquet(feature_path)
    seasons = sorted(features["season"].dropna().astype(int).unique(), reverse=True)
    col1, col2 = st.columns(2)
    with col1:
        season = st.selectbox("Season", seasons, key="explorer_season")
    season_games = features.loc[features["season"].eq(season)]
    teams = sorted(set(season_games["home_team"]).union(season_games["away_team"]))
    with col2:
        team = st.selectbox("Team", teams, key="explorer_team")
    trends = _team_game_rows(features, team, int(season))
    if trends.empty:
        st.info("No games were found for that team and season.")
        return
    st.line_chart(trends, x="week", y=["off_epa", "def_epa_allowed"], height=280)
    st.dataframe(
        trends,
        hide_index=True,
        width="stretch",
        column_config={
            "off_epa": st.column_config.NumberColumn("Off EPA/play", format="%.3f"),
            "def_epa_allowed": st.column_config.NumberColumn("Def EPA allowed", format="%.3f"),
            "point_diff_state": st.column_config.NumberColumn("Point-diff state", format="%.2f"),
            "ats_state": st.column_config.NumberColumn("ATS state", format="%.2f"),
        },
    )


def _matchup_signal_table(game: pd.Series) -> pd.DataFrame:
    signals = (
        ("Pregame Elo difference", "elo_diff", True),
        ("Expected QB EPA/dropback difference", "diff_qb_expected_epa_per_dropback", True),
        ("Recent offensive EPA/play difference", "diff_off_epa_per_play", True),
        ("Defensive EPA allowed difference", "diff_def_epa_per_play", False),
        ("Offensive injury unavailability difference", "diff_injury_offense_unavailability", False),
        ("Offensive-line continuity difference", "diff_offensive_line_continuity", True),
    )
    rows = []
    home, away = str(game["home_team"]), str(game["away_team"])
    for label, column, higher_favors_home in signals:
        value = pd.to_numeric(pd.Series([game.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        numeric = float(value)
        if abs(numeric) < 1e-12:
            direction = "Roughly even"
        else:
            positive_favors = home if higher_favors_home else away
            negative_favors = away if higher_favors_home else home
            direction = f"Leans {positive_favors if numeric > 0 else negative_favors}"
        rows.append({"Pregame signal": label, "Home minus away": numeric, "Direction": direction})
    return pd.DataFrame(rows)


def _why_this_pick() -> None:
    root = artifacts_root()
    active = load_active_model(root)
    if active is None:
        st.warning("No synchronized active model is available yet.")
        return
    forecast_directory = active_artifact_path(root, active, "weekly_forecast")
    if forecast_directory is None:
        st.warning("The active model has no linked weekly forecast.")
        return
    try:
        recommendations, metadata = load_weekly_ats_forecast(forecast_directory)
    except ValueError as error:
        st.error(str(error))
        return
    confidence = recommendations["home_cover_probability"].where(
        recommendations["home_cover_probability"].ge(0.5),
        1.0 - recommendations["home_cover_probability"],
    )
    order = pd.DataFrame(
        {"confidence": confidence, "game_id": recommendations["game_id"].astype(str)}
    ).sort_values(["confidence", "game_id"], ascending=[False, True])
    recommendations = recommendations.loc[order.index]
    labels = {
        str(row["game_id"]): f"{row['away_team']} at {row['home_team']}"
        for _, row in recommendations.iterrows()
    }
    selected_game = st.selectbox(
        "Upcoming matchup",
        recommendations["game_id"].astype(str).tolist(),
        format_func=lambda gid: labels[str(gid)],
    )
    game = recommendations.loc[recommendations["game_id"].astype(str).eq(selected_game)].iloc[0]
    home_probability = float(game["home_cover_probability"])
    home_pick = home_probability >= 0.5
    pick = str(game["home_team"] if home_pick else game["away_team"])
    pick_line = -float(game["spread_line"]) if home_pick else float(game["spread_line"])
    pick_line_label = "PK" if pick_line == 0 else f"{pick_line:+g}"
    pick_probability = max(home_probability, 1.0 - home_probability)
    st.success(
        f"**ATS pick: {pick} {pick_line_label}** at {pick_probability:.1%} estimated confidence."
    )
    st.caption(f"Market line: {favorite_label(game)}")
    residual = game.get("predicted_market_residual")
    if pd.notna(residual):
        residual = float(residual)
        correction_team = str(game["home_team"] if residual > 0 else game["away_team"])
        st.write(
            f"The model expects **{correction_team}** to outperform the market baseline by "
            f"about {abs(residual):.1f} points."
        )
    signals = _matchup_signal_table(game)
    st.subheader("Recognizable pregame inputs")
    if signals.empty:
        st.info("No readable matchup signals are available for this card.")
    else:
        st.dataframe(
            signals,
            hide_index=True,
            width="stretch",
            column_config={"Home minus away": st.column_config.NumberColumn(format="%.3f")},
        )
    st.caption(f"Active model `{active.get('model_id')}` · `{metadata.get('ats_method')}` method.")


tab_team, tab_matchup = st.tabs(["Team trends", "Why this pick"])
with tab_team:
    _team_explorer()
with tab_matchup:
    _why_this_pick()
