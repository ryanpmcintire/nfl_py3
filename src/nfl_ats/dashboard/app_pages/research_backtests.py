"""Research: backtests -- winner/margin/ATS lab, plus the archived direct classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nfl_ats.active_model import active_artifact_path
from nfl_ats.dashboard.data import (
    artifact_time,
    artifacts_root,
    describe_artifact_source,
    load_active_model,
    read_json_safe,
    select_research_artifact,
)
from nfl_ats.dashboard.ui import metric_with_context, render_research_run_picker
from nfl_ats.reporting import (
    artifact_directories,
    block_bootstrap_intervals,
    calibration_table,
    season_scorecard,
)

st.title("Backtests")
st.caption(
    "Deeper mechanics behind the track-record page: multiple saved runs, uncertainty "
    "methodology, and the older direct classifier kept for comparison."
)

METHOD_LABELS = {
    "market": "Closing market",
    "fair_margin": "Independent fair margin",
    "market_residual": "Market + predicted correction",
    "straight_up": "Straight-up classifier",
    "direct_ats": "Direct ATS classifier",
}


@st.cache_data(show_spinner=False, ttl="10m")
def _compute_uncertainty(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            block_bootstrap_intervals(predictions, samples=500, block="week"),
            block_bootstrap_intervals(predictions, samples=500, block="season"),
        ],
        ignore_index=True,
    )


def _render_outcome_run(selected: Path, root: Path) -> None:
    st.caption(describe_artifact_source(selected, root))
    summary = pd.read_csv(selected / "summary.csv")
    metadata = read_json_safe(selected / "metadata.json") or {}
    st.caption(
        f"{metadata.get('regressor', 'unknown')} margin regressor · "
        f"{metadata.get('feature_profile', 'base')} feature profile · testing starts "
        f"{metadata.get('start_season', '?')}"
    )

    market_rows = summary.loc[summary["method"].eq("market")]
    market_row = market_rows.iloc[0] if not market_rows.empty else pd.Series(dtype=float)
    fair = summary.loc[summary["method"].eq("fair_margin")]

    candidates = summary.loc[
        summary["method"].isin(["direct_ats", "fair_margin", "market_residual"])
        & summary["cover_accuracy"].notna()
    ]
    if not candidates.empty:
        best = candidates.sort_values("cover_accuracy", ascending=False).iloc[0]
        accuracy = float(best["cover_accuracy"])
        games = int(best["cover_games"])
        approach = METHOD_LABELS.get(str(best["method"]), str(best["method"]))
        st.write(
            f"**{approach}** classified {round(accuracy * games):,} of {games:,} games "
            f"correctly ({accuracy:.1%}), which is {accuracy - 0.5:+.1%} versus a coin flip."
        )

    with st.container(horizontal=True):
        metric_with_context(
            "Market winner accuracy", market_row.get("win_accuracy"), percent=True, border=True
        )
        metric_with_context(
            "Fair-model winner accuracy",
            fair.iloc[0].get("win_accuracy") if not fair.empty else np.nan,
            percent=True,
            border=True,
        )
        metric_with_context(
            "Market margin error", market_row.get("margin_mae"), digits=2, border=True
        )
        metric_with_context(
            "Fair-model margin error",
            fair.iloc[0].get("margin_mae") if not fair.empty else np.nan,
            digits=2,
            border=True,
        )

    with st.expander("Every forecasting approach, side by side"):
        readable = summary.copy()
        readable["Approach"] = readable["method"].map(METHOD_LABELS).fillna(readable["method"])
        columns = [
            "Approach",
            "win_accuracy",
            "win_brier_score",
            "margin_mae",
            "cover_accuracy",
            "cover_brier_score",
            "roi",
        ]
        for column in columns:
            if column not in readable:
                readable[column] = np.nan
        st.dataframe(
            readable[columns].rename(
                columns={
                    "win_accuracy": "Winner accuracy",
                    "win_brier_score": "Winner Brier",
                    "margin_mae": "Margin MAE",
                    "cover_accuracy": "ATS accuracy",
                    "cover_brier_score": "ATS Brier",
                    "roi": "Flat paper ROI",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Winner accuracy": st.column_config.NumberColumn(format="percent"),
                "ATS accuracy": st.column_config.NumberColumn(format="percent"),
                "Flat paper ROI": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "Winner accuracy is not ATS accuracy -- a favorite can win but not cover. Brier "
            "scores diagnose probability quality; paper ROI is an optional side simulation."
        )

    uncertainty_path = selected / "uncertainty.csv"
    if uncertainty_path.is_file():
        with st.expander("How uncertain are these differences?"):
            uncertainty = pd.read_csv(uncertainty_path)
            block = st.radio(
                "Resampling unit",
                ("week", "season"),
                horizontal=True,
                key=f"outcome_block_{selected.name}",
            )
            useful = uncertainty.loc[
                uncertainty["block"].eq(block)
                & uncertainty["method"].isin(["fair_margin", "market_residual", "straight_up"])
                & uncertainty["metric"].isin(
                    ["cover_accuracy", "win_accuracy", "margin_mae", "cover_brier_score"]
                )
            ].copy()
            useful["Approach"] = useful["method"].map(METHOD_LABELS)
            useful["Measure"] = useful["metric"].replace(
                {
                    "win_accuracy": "Winner accuracy",
                    "margin_mae": "Margin MAE",
                    "cover_accuracy": "ATS accuracy",
                    "cover_brier_score": "ATS Brier",
                }
            )
            useful["95% interval"] = useful.apply(
                lambda row: f"{row['lower']:.4f} to {row['upper']:.4f}", axis=1
            )
            st.dataframe(
                useful[["Approach", "Measure", "estimate", "95% interval"]].rename(
                    columns={"estimate": "Estimate"}
                ),
                hide_index=True,
                width="stretch",
            )


def _outcome_lab() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "margins", "summary.csv")
    active = load_active_model(root)
    active_declared = (
        active_artifact_path(root, active, "historical_evaluation") if active else None
    )
    selection = select_research_artifact(directories, active_declared)

    if selection.active_declared_but_missing and active_declared is not None:
        try:
            expected = active_declared.relative_to(root).as_posix()
        except ValueError:
            expected = str(active_declared)
        st.error(
            "The active model's evaluation artifact is missing locally -- regenerate with "
            f"`nfl-ats margin-backtest` (expected at `{expected}`)."
        )
        if not directories:
            return
        st.caption("Showing other saved runs instead; none of them is linked to the active model.")
        chosen = st.selectbox(
            "Saved outcome-model test",
            directories,
            format_func=artifact_time,
            key="outcome_lab_run",
        )
        _render_outcome_run(chosen, root)
        return

    if selection.featured is None:
        st.warning("No saved outcome-model test. Run `nfl-ats margin-backtest` first.")
        return

    render_research_run_picker(
        selection, lambda path: _render_outcome_run(path, root), key="outcome_lab"
    )


def _direct_classifier() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "backtests", "metrics.json")
    selection = select_research_artifact(directories)
    render_research_run_picker(
        selection, lambda path: _render_direct_classifier_run(path, root), key="direct_backtest"
    )
    if selection.featured is None:
        st.info("No saved direct-classifier test. Run `nfl-ats backtest`.")


def _render_direct_classifier_run(selected: Path, root: Path) -> None:
    st.caption(describe_artifact_source(selected, root))
    metrics = read_json_safe(selected / "metrics.json") or {}
    predictions = pd.read_parquet(selected / "predictions.parquet")
    accuracy = float(metrics.get("accuracy", np.nan))
    if accuracy > 0.5:
        st.success(f"Selected the covering team in {accuracy:.1%} of evaluated games.")
    else:
        st.warning("No classification lead was observed in this run.")

    with st.container(horizontal=True):
        metric_with_context("Games tested", metrics.get("games_evaluated"), border=True)
        metric_with_context("Accuracy", metrics.get("accuracy"), percent=True, border=True)
        metric_with_context("Brier error", metrics.get("brier_score"), border=True)
        metric_with_context("Paper ROI", metrics.get("roi"), percent=True, border=True)

    with st.expander("Season by season"):
        scorecard = season_scorecard(predictions)
        chart = scorecard[["season", "accuracy"]].copy()
        chart["coin flip"] = 0.5
        st.line_chart(chart, x="season", y=["accuracy", "coin flip"], height=280)
        st.dataframe(scorecard, hide_index=True, width="stretch")

    with st.expander("Calibration"):
        calibration = calibration_table(predictions)
        if calibration.empty:
            st.write("No completed non-push games to calibrate.")
        else:
            chart = calibration[["mean_probability", "observed_cover_rate"]].copy()
            chart["ideal"] = chart["mean_probability"]
            st.line_chart(
                chart, x="mean_probability", y=["observed_cover_rate", "ideal"], height=300
            )

    with st.expander("Uncertainty"):
        uncertainty_path = selected / "uncertainty.csv"
        intervals = (
            pd.read_csv(uncertainty_path)
            if uncertainty_path.is_file()
            else _compute_uncertainty(predictions)
        )
        st.dataframe(intervals, hide_index=True, width="stretch")

    with st.expander("Reproducibility record"):
        run_path = selected / "run.json"
        if run_path.is_file():
            st.json(read_json_safe(run_path) or {}, expanded=False)
        st.caption(f"Artifact directory: {selected}")


tab_outcome, tab_direct = st.tabs(["Winner, margin & ATS lab", "Archived direct-classifier test"])
with tab_outcome:
    _outcome_lab()
with tab_direct:
    _direct_classifier()
