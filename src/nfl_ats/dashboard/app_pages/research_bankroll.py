"""Research: bankroll simulation -- what quarter-Kelly paper staking would have looked like."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nfl_ats.dashboard.data import (
    artifact_time,
    artifacts_root,
    describe_artifact_source,
    read_json_safe,
)
from nfl_ats.dashboard.ui import metric_with_context
from nfl_ats.reporting import artifact_directories, bankroll_curve

st.title("Bankroll simulation")
st.caption(
    "A simulated, non-real paper bankroll using quarter-Kelly stakes on the model's forced "
    "picks. This is a sizing exercise, not evidence the underlying picks are profitable."
)

root = artifacts_root()
directories = artifact_directories(root / "backtests", "paper_ledger.parquet")
if not directories:
    st.info("No saved backtest has a paper-bankroll ledger yet. Run `nfl-ats backtest`.")
    st.stop()

selected = st.selectbox("Saved backtest run", directories, format_func=artifact_time)
st.caption(describe_artifact_source(selected, root))
ledger = pd.read_parquet(selected / "paper_ledger.parquet")
portfolio = read_json_safe(selected / "portfolio_metrics.json") or {}

with st.container(horizontal=True):
    metric_with_context(
        "Starting bankroll", portfolio.get("initial_bankroll"), digits=2, border=True
    )
    metric_with_context("Ending bankroll", portfolio.get("final_bankroll"), digits=2, border=True)
    metric_with_context("Paper return", portfolio.get("return"), percent=True, border=True)
    metric_with_context("Worst drawdown", portfolio.get("max_drawdown"), percent=True, border=True)

curve = bankroll_curve(ledger)
st.line_chart(curve, x="season_week", y="bankroll", height=330)
st.caption(
    "Quarter-Kelly stakes are capped at 2% per game and 10% per NFL week. Sizing discipline "
    "cannot rescue inaccurate probabilities."
)

simulation_path = selected / "bankroll_simulation.json"
paths_path = selected / "bankroll_paths.parquet"
if simulation_path.is_file():
    simulation = read_json_safe(simulation_path) or {}
    st.subheader("Conditional bankroll risk")
    st.caption(
        "These paths assume the model's stated probabilities are correct. They show sizing "
        "risk, not proof the probabilities are accurate."
    )
    with st.container(horizontal=True):
        metric_with_context(
            "Median ending bankroll",
            simulation.get("terminal_bankroll_median"),
            digits=2,
            border=True,
        )
        metric_with_context(
            "5th-percentile ending bankroll",
            simulation.get("terminal_bankroll_p05"),
            digits=2,
            border=True,
        )
        metric_with_context(
            "Chance of finishing below start",
            simulation.get("probability_of_loss"),
            percent=True,
            border=True,
        )
        metric_with_context(
            "Median worst drawdown",
            simulation.get("median_max_drawdown"),
            percent=True,
            border=True,
        )
    if paths_path.is_file():
        paths = pd.read_parquet(paths_path)
        long_paths = paths.melt(
            var_name="season_week", value_name="bankroll", ignore_index=False
        ).reset_index(names="path")
        bands = (
            long_paths.groupby("season_week", sort=False)["bankroll"]
            .quantile(np.asarray([0.05, 0.50, 0.95]))
            .unstack()
            .rename(columns={0.05: "5th percentile", 0.50: "Median", 0.95: "95th percentile"})
            .reset_index()
        )
        st.line_chart(
            bands, x="season_week", y=["5th percentile", "Median", "95th percentile"], height=300
        )

with st.expander("Every paper decision"):
    columns = [
        "gameday",
        "away_team",
        "home_team",
        "bet_side",
        "bet_probability",
        "stake",
        "profit",
        "bankroll_after_week",
    ]
    st.dataframe(
        ledger.loc[:, columns],
        hide_index=True,
        width="stretch",
        column_config={
            "gameday": "Game date",
            "away_team": "Away",
            "home_team": "Home",
            "bet_side": "Side",
            "bet_probability": st.column_config.NumberColumn("Model probability", format="percent"),
            "stake": st.column_config.NumberColumn("Stake", format="%.2f"),
            "profit": st.column_config.NumberColumn("Profit", format="%.2f"),
            "bankroll_after_week": st.column_config.NumberColumn("Bankroll", format="%.2f"),
        },
    )
