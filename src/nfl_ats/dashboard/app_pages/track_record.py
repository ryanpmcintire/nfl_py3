"""Is this thing actually good? -- an honest track record in plain words."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.active_model import active_artifact_path
from nfl_ats.dashboard.data import (
    artifacts_root,
    describe_artifact_source,
    find_latest_clv_ledger,
    load_active_model,
    load_clv_ledger,
    load_evaluation_predictions,
)
from nfl_ats.dashboard.ui import honesty_note, metric_with_context
from nfl_ats.reporting import calibration_table, season_scorecard

st.title("Is this thing actually good?")
st.caption(
    "The honest answer, not the hopeful one: how the model has done historically, how much "
    "that estimate could plausibly move, and whether its stated confidence can be trusted."
)

active = load_active_model(artifacts_root())

if active is None:
    st.info(
        "No synchronized active model is available yet, so there is no linked track record "
        "to show. Run an evaluation and a weekly forecast to activate one.",
        icon=":material/info:",
    )
    st.stop()

historical = active.get("historical_evaluation", {})
if not isinstance(historical, dict) or historical.get("accuracy") is None:
    st.error("The active model's evaluation record is malformed.")
    st.stop()

accuracy = float(historical["accuracy"])
games = int(historical["games"])
correct = int(historical["correct"])
intervals = historical.get("intervals", {})
season_interval = intervals.get("season") if isinstance(intervals, dict) else None
week_interval = intervals.get("week") if isinstance(intervals, dict) else None

# --- Headline -----------------------------------------------------------------
st.subheader("Historical accuracy")
st.write(
    f"Tested only on games it never trained on, the model has picked the side that covered "
    f"the spread **{accuracy:.1%} of the time** ({correct:,} correct out of {games:,} games)."
)
if isinstance(season_interval, dict):
    st.write(
        f"Because that is one sample of seasons, not a guarantee, the honest range is: **we'd "
        f"expect somewhere between {season_interval['lower']:.1%} and "
        f"{season_interval['upper']:.1%} on new seasons**, with {accuracy:.1%} as the single "
        "best estimate."
    )

with st.container(horizontal=True):
    metric_with_context(
        "Accuracy",
        accuracy,
        percent=True,
        delta=f"{accuracy - 0.5:+.1%} vs. coin flip",
        border=True,
    )
    metric_with_context("Games tested", games, border=True)
    if isinstance(season_interval, dict):
        with st.container(border=True):
            st.caption(
                "95% range (season-blocked)",
                help="A range from resampling whole seasons, a more conservative check than "
                "treating every game as independent.",
            )
            st.markdown(f"**{season_interval['lower']:.1%} - {season_interval['upper']:.1%}**")

honesty_note()

# --- This season so far ---------------------------------------------------------
st.subheader("This season so far")
evaluation_directory = active_artifact_path(artifacts_root(), active, "historical_evaluation")
if evaluation_directory is None or not (evaluation_directory / "predictions.parquet").is_file():
    st.error(
        "The active model's evaluation artifact is missing locally -- regenerate with "
        "`nfl-ats margin-backtest`. The headline accuracy above still reflects the active "
        "model's last synchronized run, but season-by-season and calibration detail below "
        "cannot be recomputed without it."
    )
    predictions = pd.DataFrame()
else:
    st.caption(describe_artifact_source(evaluation_directory, artifacts_root()))
    predictions = load_evaluation_predictions(evaluation_directory)
method = str(active.get("method", "market_residual"))
current_season = active.get("weekly_forecast", {}).get("season")
method_predictions = (
    predictions.loc[predictions["method"].eq(method)] if not predictions.empty else predictions
)

season_row = None
if not method_predictions.empty and current_season is not None:
    scorecard = season_scorecard(method_predictions)
    matches = scorecard.loc[scorecard["season"].eq(current_season)]
    if not matches.empty and int(matches.iloc[0]["games"]) > 0:
        season_row = matches.iloc[0]

if season_row is None:
    st.write(
        f"No games from {current_season if current_season is not None else 'this season'} have "
        "settled yet, so there is nothing to score. Check back once games start finishing."
    )
else:
    season_accuracy = float(season_row["accuracy"])
    season_games = int(season_row["games"])
    season_wins = round(season_accuracy * season_games)
    st.write(
        f"So far in {int(current_season)}, the model has gone "
        f"**{season_wins}-{season_games - season_wins}** against the spread on settled games "
        f"({season_accuracy:.1%})."
    )
    metric_with_context(
        f"{int(current_season)} accuracy",
        season_accuracy,
        percent=True,
        delta=f"{season_accuracy - 0.5:+.1%} vs. coin flip",
        border=True,
    )

# --- Closing-line value ------------------------------------------------------
st.subheader("Does the market move toward our picks?")
clv_directory = find_latest_clv_ledger(artifacts_root())
if clv_directory is None:
    st.write(
        "No paper-decision ledger has been scored yet. Every published weekly card's picks "
        "are recorded at their published line; once games close, `nfl-ats clv-ledger` "
        "measures whether the closing line moved toward or away from each pick."
    )
else:
    ledger = load_clv_ledger(clv_directory)
    scored_rows = ledger.loc[ledger["clv_status"].eq("scored")] if not ledger.empty else ledger
    pending = int(len(ledger) - len(scored_rows))
    if scored_rows.empty:
        st.write(
            f"**{len(ledger)}** published pick{'s' if len(ledger) != 1 else ''} are recorded "
            "and waiting for their games to close. Closing-line value appears here once the "
            "first recorded week finishes."
        )
    else:
        st.caption(describe_artifact_source(clv_directory, artifacts_root()))
        mean_clv = float(scored_rows["clv_points"].mean())
        positive_rate = float(scored_rows["clv_points"].gt(0.0).mean())
        st.write(
            "Closing-line value asks a humbler question than wins: after we published a "
            "pick, did the market's closing line move toward it (positive) or away from it "
            "(negative)? Beating the close consistently is generally considered a stronger "
            "signal of real edge than a short stretch of wins."
        )
        with st.container(horizontal=True):
            metric_with_context(
                "Average CLV",
                round(mean_clv, 2),
                delta=f"{mean_clv:+.2f} points vs the close",
                border=True,
            )
            metric_with_context("Picks beating the close", positive_rate, percent=True, border=True)
            metric_with_context("Picks scored", len(scored_rows), border=True)
        if pending:
            st.caption(
                f"{pending} more recorded pick{'s' if pending != 1 else ''} will score once "
                "their games close."
            )

# --- Calibration ------------------------------------------------------------
st.subheader("Can you trust the stated confidence?")
if method_predictions.empty:
    st.write("No historical predictions are available to check calibration.")
else:
    calibration = calibration_table(method_predictions)
    if calibration.empty:
        st.write("No completed, non-push games are available to check calibration yet.")
    else:
        worst_gap = calibration["gap"].abs().max()
        typical_gap = calibration["gap"].abs().mean()
        if typical_gap < 0.03:
            st.write(
                "**Yes, roughly.** When the model states a probability, the observed cover rate "
                f"in that bucket tracks it closely -- off by about {typical_gap:.1%} on average, "
                f"and {worst_gap:.1%} at the worst bucket."
            )
        else:
            st.write(
                "**Only loosely.** The observed cover rate drifts from the stated probability by "
                f"about {typical_gap:.1%} on average across buckets, and {worst_gap:.1%} at the "
                "worst bucket, so treat individual percentages as directional, not exact."
            )
        with st.expander("See the calibration chart and bucket table"):
            chart = calibration[["mean_probability", "observed_cover_rate"]].copy()
            chart["ideal (perfectly calibrated)"] = chart["mean_probability"]
            st.line_chart(
                chart,
                x="mean_probability",
                y=["observed_cover_rate", "ideal (perfectly calibrated)"],
                x_label="Model-stated probability",
                y_label="Observed cover rate",
                height=320,
            )
            st.dataframe(
                calibration,
                hide_index=True,
                width="stretch",
                column_config={
                    "bin": "Probability bucket",
                    "games": "Games",
                    "mean_probability": st.column_config.NumberColumn(
                        "Average prediction", format="percent"
                    ),
                    "observed_cover_rate": st.column_config.NumberColumn(
                        "Observed cover rate", format="percent"
                    ),
                    "gap": st.column_config.NumberColumn("Gap", format="percent"),
                },
            )

# --- Season by season --------------------------------------------------------
if not method_predictions.empty:
    with st.expander("See every season, not just the pooled number"):
        scorecard = season_scorecard(method_predictions)
        chart = scorecard[["season", "accuracy"]].copy()
        chart["coin flip"] = 0.5
        st.line_chart(chart, x="season", y=["accuracy", "coin flip"], height=280)
        above = int(scorecard["accuracy"].gt(0.5).sum())
        st.caption(f"Finished above 50% in {above} of {len(scorecard)} tested seasons.")
        st.dataframe(
            scorecard[["season", "games", "accuracy", "brier_score"]].rename(
                columns={
                    "season": "Season",
                    "games": "Games",
                    "accuracy": "Accuracy",
                    "brier_score": "Brier (probability error)",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"Accuracy": st.column_config.NumberColumn(format="percent")},
        )

st.caption(
    "Deeper diagnostics -- validation methodology, dependence checks, feature experiments -- "
    "live in Research if you want to audit how this number was produced."
)
