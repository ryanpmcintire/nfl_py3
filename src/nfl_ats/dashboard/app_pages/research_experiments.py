"""Research: feature & player experiments -- does adding information help?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.constants import FEATURE_SETS
from nfl_ats.dashboard.data import artifact_time, artifacts_root
from nfl_ats.reporting import artifact_directories

st.title("Feature & player experiments")
st.caption(
    "Every candidate is tested on the exact same historical weeks. Accuracy is the headline; "
    "Brier and paired intervals diagnose whether a gain is real or noise."
)

FEATURE_SET_DESCRIPTIONS = {
    "market": "Spread + total only",
    "market_context": "Market + rest, venue, weather, division, and season timing",
    "market_elo": "Market + Elo strength",
    "football": "Football state without the betting market",
    "full_without_ats": "All current inputs except prior ATS residuals",
    "full": "All current inputs",
    "graph": "Temporal PageRank/HITS only",
    "schedule_rating": "Regularized opponent-adjusted scoring ratings only",
}


def _feature_lab() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "experiments", "summary.csv")
    if not directories:
        st.info("No comparison is saved yet. Run `nfl-ats experiment`.")
        rows = [
            {
                "Feature set": name,
                "Inputs": len(columns),
                "Meaning": FEATURE_SET_DESCRIPTIONS.get(name, name),
            }
            for name, columns in FEATURE_SETS.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        return
    selected = st.selectbox(
        "Saved comparison", directories, format_func=artifact_time, key="feature_lab_run"
    )
    summary = (
        pd.read_csv(selected / "summary.csv")
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    best = summary.iloc[0]
    best_probability = summary.loc[summary["brier_score"].idxmin()]
    st.write(
        f"**Best ATS classification:** `{best['feature_set']}` selected the covering team in "
        f"{best['accuracy']:.1%} of {int(best['games_evaluated']):,} games. Lowest probability "
        f"error: `{best_probability['feature_set']}` (Brier {best_probability['brier_score']:.4f})."
    )
    st.caption("Best region: up for higher accuracy, left for lower probability error.")
    st.scatter_chart(
        summary,
        x="brier_score",
        y="accuracy",
        color="feature_set",
        size="games_evaluated",
        height=360,
    )
    with st.expander("Every feature set, side by side"):
        display = summary[
            [
                "feature_set",
                "feature_count",
                "games_evaluated",
                "accuracy",
                "brier_score",
                "expected_calibration_error",
                "roi",
            ]
        ].copy()
        display["description"] = display["feature_set"].map(
            lambda name: FEATURE_SET_DESCRIPTIONS.get(name, name)
        )
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "feature_set": "Feature set",
                "description": "What it contains",
                "feature_count": "Inputs",
                "games_evaluated": "Games",
                "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                "brier_score": st.column_config.NumberColumn("Brier", format="%.5f"),
                "expected_calibration_error": st.column_config.NumberColumn(
                    "Calibration gap", format="percent"
                ),
                "roi": st.column_config.NumberColumn("Paper ROI", format="percent"),
            },
        )
    paired_path = selected / "paired_comparisons.csv"
    if paired_path.is_file():
        with st.expander("Paired improvement over the baseline"):
            paired = pd.read_csv(paired_path)
            block = st.radio(
                "Uncertainty block", ["season", "week"], horizontal=True, key="feature_paired_block"
            )
            paired_display = paired.loc[paired["block"].eq(block)].copy()
            paired_display["Estimate [95% interval]"] = paired_display.apply(
                lambda row: f"{row['estimate']:.5f} [{row['lower']:.5f}, {row['upper']:.5f}]",
                axis=1,
            )
            st.dataframe(
                paired_display[
                    ["candidate_feature_set", "metric", "Estimate [95% interval]", "paired_games"]
                ].rename(
                    columns={
                        "candidate_feature_set": "Candidate",
                        "metric": "Improvement metric",
                        "paired_games": "Paired games",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


def _player_family_ablation() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "player_experiments", "summary.csv")
    if not directories:
        st.info("No player-family comparison is saved yet. Run `nfl-ats player-ablation`.")
        return
    selected = st.selectbox(
        "Saved player comparison", directories, format_func=artifact_time, key="player_lab_run"
    )
    summary = pd.read_csv(selected / "summary.csv").sort_values(
        ["cover_accuracy", "cover_brier_score"], ascending=[False, True]
    )
    best = summary.iloc[0]
    base = summary.loc[summary["feature_profile"].eq("base")]
    base_accuracy = float(base.iloc[0]["cover_accuracy"]) if not base.empty else float("nan")
    st.write(
        f"Strongest player-information family: `{best['feature_profile']}` at "
        f"{float(best['cover_accuracy']):.1%}, versus a base accuracy of {base_accuracy:.1%}."
    )
    st.dataframe(
        summary[
            ["feature_profile", "cover_games", "cover_accuracy", "cover_brier_score", "margin_mae"]
        ].rename(
            columns={
                "feature_profile": "Player profile",
                "cover_games": "Games",
                "cover_accuracy": "ATS accuracy",
                "cover_brier_score": "Brier error",
                "margin_mae": "Margin MAE",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"ATS accuracy": st.column_config.NumberColumn(format="percent")},
    )
    st.warning(
        "Player-value features have not replaced the active weekly model; these remain "
        "research leads, not promoted improvements. See Validation for the formal gates."
    )


tab_features, tab_players = st.tabs(["Feature-set comparison", "Player-family comparison"])
with tab_features:
    _feature_lab()
with tab_players:
    _player_family_ablation()
