"""Research: model selection & validation -- nested evaluation and promotion gates."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.dashboard.data import artifact_time, artifacts_root, read_json_safe
from nfl_ats.dashboard.ui import metric_with_context
from nfl_ats.reporting import artifact_directories

st.title("Model selection & validation")
st.caption(
    "The stricter test: was a candidate chosen using only earlier seasons before it was "
    "scored, or did it just look good after the fact?"
)


def _nested_evaluation() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "nested_evaluations", "metrics.json")
    if not directories:
        st.info("No nested evaluation is saved yet. Run `nfl-ats nested-evaluate`.")
        return
    selected = st.selectbox("Saved nested evaluation", directories, format_func=artifact_time)
    metrics = read_json_safe(selected / "metrics.json") or {}
    folds = pd.read_csv(selected / "fold_summary.csv")
    candidates = pd.read_csv(selected / "candidate_validation.csv")

    with st.container(horizontal=True):
        metric_with_context("Outer seasons", metrics.get("outer_folds"), border=True)
        metric_with_context(
            "Outer-test accuracy", metrics.get("accuracy"), percent=True, border=True
        )
        metric_with_context("Outer-test Brier", metrics.get("brier_score"), border=True)
        metric_with_context("Outer-test paper ROI", metrics.get("roi"), percent=True, border=True)
    st.caption(
        f"{metrics.get('candidate_count', '?')} frozen candidates · "
        f"selection by {metrics.get('selection_metric', '?')}."
    )

    st.subheader("What was selected before each outer season?")
    fold_columns = [
        column
        for column in (
            "outer_test_season",
            "selected_model_name",
            "selected_feature_set",
            "validation_selection_score",
            "test_accuracy",
            "test_brier_score",
        )
        if column in folds
    ]
    st.dataframe(
        folds.loc[:, fold_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "outer_test_season": "Outer season",
            "selected_model_name": "Selected model",
            "selected_feature_set": "Selected inputs",
            "validation_selection_score": st.column_config.NumberColumn(
                "Validation score", format="%.4f"
            ),
            "test_accuracy": st.column_config.NumberColumn("Outer accuracy", format="percent"),
            "test_brier_score": st.column_config.NumberColumn("Outer Brier", format="%.4f"),
        },
    )
    with st.expander("Every candidate ranking"):
        display_columns = [
            column
            for column in (
                "outer_test_season",
                "candidate_id",
                "selection_rank",
                "selected",
                "brier_score",
                "accuracy",
            )
            if column in candidates
        ]
        st.dataframe(candidates.loc[:, display_columns], hide_index=True, width="stretch")
    st.info(
        "These outer seasons are historically out-of-time under the declared algorithm, but "
        "2018-2025 results have already influenced the broader project. Treat this as a "
        "corrected repeatable benchmark, not proof those years were never observed."
    )


def _player_promotion_gate() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "player_model_selection", "candidate_summary.csv")
    if not directories:
        st.info(
            "No frozen player-model selection is saved yet. Run `nfl-ats player-model-selection`."
        )
        return
    selected = st.selectbox(
        "Saved regularization/calibration gate", directories, format_func=artifact_time
    )
    summary = pd.read_csv(selected / "candidate_summary.csv").sort_values(
        ["cover_accuracy", "cover_brier_score"], ascending=[False, True]
    )
    nested = pd.read_csv(selected / "nested_summary.csv").iloc[0]
    paired = pd.read_csv(selected / "nested_paired_comparisons.csv")
    week_accuracy = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")
    ].iloc[0]
    nested_accuracy = float(nested["cover_accuracy"])
    improvement = float(week_accuracy["estimate"])
    baseline_outer_accuracy = nested_accuracy - improvement
    best = summary.iloc[0]

    with st.container(horizontal=True):
        metric_with_context("Nested selector", nested_accuracy, percent=True, border=True)
        metric_with_context(
            "Fixed base, same games", baseline_outer_accuracy, percent=True, border=True
        )
        metric_with_context("Selector minus base", improvement, percent=True, border=True)
        metric_with_context("Best pooled row", best["cover_accuracy"], percent=True, border=True)
    st.warning(
        "**Promotion verdict: NO.** The historical tuning policy did not beat the fixed base "
        f"configuration: {nested_accuracy:.1%} versus {baseline_outer_accuracy:.1%} (week-block "
        f"95% interval for the difference: {float(week_accuracy['lower']):.1%} to "
        f"{float(week_accuracy['upper']):.1%})."
    )
    with st.expander("Every candidate row"):
        st.dataframe(
            summary.head(12)[
                [
                    "feature_profile",
                    "ridge_alpha",
                    "calibration_method",
                    "cover_games",
                    "cover_accuracy",
                    "cover_brier_score",
                    "cover_ece",
                ]
            ].rename(
                columns={
                    "feature_profile": "Player profile",
                    "ridge_alpha": "Ridge alpha",
                    "calibration_method": "Calibration",
                    "cover_games": "Games",
                    "cover_accuracy": "ATS accuracy",
                    "cover_brier_score": "Brier error",
                    "cover_ece": "Calibration error",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"ATS accuracy": st.column_config.NumberColumn(format="percent")},
        )


def _participation_gate() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "participation_experiments", "summary.csv")
    if not directories:
        st.info(
            "No participation-rating comparison is saved yet. Run `nfl-ats participation-ablation`."
        )
        return
    selected = st.selectbox(
        "Saved participation-rating comparison", directories, format_func=artifact_time
    )
    summary = pd.read_csv(selected / "summary.csv")
    paired = pd.read_csv(selected / "paired_comparisons.csv")
    baseline_rows = summary.loc[summary["feature_profile"].eq("player_value")]
    candidate_rows = summary.loc[summary["feature_profile"].eq("player_participation")]
    if baseline_rows.empty or candidate_rows.empty:
        st.error("The saved participation artifact is missing its declared matched profiles.")
        return
    baseline, candidate = baseline_rows.iloc[0], candidate_rows.iloc[0]
    week_accuracy_rows = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")
    ]
    if week_accuracy_rows.empty:
        st.error("The saved participation artifact has no week-blocked accuracy comparison.")
        return
    week_accuracy = week_accuracy_rows.iloc[0]
    with st.container(horizontal=True):
        metric_with_context(
            "Prior player value", baseline["cover_accuracy"], percent=True, border=True
        )
        metric_with_context(
            "With participation", candidate["cover_accuracy"], percent=True, border=True
        )
        metric_with_context("Accuracy change", week_accuracy["estimate"], percent=True, border=True)
    estimate = float(week_accuracy["estimate"])
    interval = f"{float(week_accuracy['lower']):.1%} to {float(week_accuracy['upper']):.1%}"
    if estimate <= 0:
        st.warning(
            f"**Participation verdict: NO.** Historical ATS classification fell from "
            f"{float(baseline['cover_accuracy']):.1%} to {float(candidate['cover_accuracy']):.1%}."
        )
    elif float(week_accuracy["lower"]) <= 0:
        st.warning(
            f"**Participation verdict: unresolved.** Week-blocked 95% interval ({interval}) "
            "still includes no improvement."
        )
    else:
        st.success(
            "The participation candidate cleared its paired accuracy interval, pending a full "
            "promotion audit."
        )


def _availability_gate() -> None:
    root = artifacts_root()
    directories = artifact_directories(root / "availability_experiments", "summary.csv")
    if not directories:
        st.info(
            "No learned-availability comparison is saved yet. Run `nfl-ats availability-ablation`."
        )
        return
    selected = st.selectbox(
        "Saved learned-availability comparison", directories, format_func=artifact_time
    )
    summary = pd.read_csv(selected / "summary.csv")
    paired = pd.read_csv(selected / "paired_comparisons.csv")
    metadata = read_json_safe(selected / "metadata.json") or {}
    fixed_rows = summary.loc[summary["availability_method"].eq("fixed")]
    learned_rows = summary.loc[summary["availability_method"].eq("learned")]
    accuracy_rows = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")
    ]
    if fixed_rows.empty or learned_rows.empty or accuracy_rows.empty:
        st.error("The saved availability artifact is missing its declared matched comparison.")
        return
    fixed, learned, accuracy = fixed_rows.iloc[0], learned_rows.iloc[0], accuracy_rows.iloc[0]
    learned_manifest = (
        metadata.get("learned_provenance", {}).get("feature_table", {}).get("manifest", {})
    )
    player_brier_improvement = learned_manifest.get("availability_brier_improvement")
    with st.container(horizontal=True):
        metric_with_context("Hand weights ATS", fixed["cover_accuracy"], percent=True, border=True)
        metric_with_context(
            "Learned rates ATS", learned["cover_accuracy"], percent=True, border=True
        )
        metric_with_context("ATS accuracy change", accuracy["estimate"], percent=True, border=True)
        metric_with_context("Player Brier improvement", player_brier_improvement, border=True)
    st.warning(
        "**Availability verdict: PROMISING BUT UNRESOLVED.** Learned rates improved player-level "
        f"availability Brier by {float(player_brier_improvement):.4f} and moved ATS classification "
        f"from {float(fixed['cover_accuracy']):.1%} to {float(learned['cover_accuracy']):.1%}, but "
        f"the week-blocked 95% interval ({float(accuracy['lower']):.1%} to "
        f"{float(accuracy['upper']):.1%}) means the active model is unchanged."
    )


sections = {
    "Nested evaluation (candidate selection)": _nested_evaluation,
    "Player regularization & calibration gate": _player_promotion_gate,
    "Participation-based injury value gate": _participation_gate,
    "Learned availability-rate gate": _availability_gate,
}
choice = st.selectbox("Validation tool", tuple(sections))
st.divider()
sections[choice]()
