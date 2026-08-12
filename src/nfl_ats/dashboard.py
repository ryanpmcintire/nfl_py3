"""Human-readable, local Streamlit dashboard for research artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.constants import FEATURE_FAMILIES, FEATURE_SETS, MODEL_FEATURE_COLUMNS
from nfl_ats.pool import build_ats_pool_card
from nfl_ats.prospective import verify_frozen_forecast
from nfl_ats.reporting import (
    artifact_directories,
    bankroll_curve,
    block_bootstrap_intervals,
    calibration_table,
    feature_coverage,
    read_json,
    season_scorecard,
)
from nfl_ats.snapshots import describe_snapshot, latest_snapshot

FAMILY_LABELS = {
    "market": "Betting market",
    "context": "Game context",
    "elo": "Team strength (Elo)",
    "experience": "Games played this season",
    "offense": "Recent offense",
    "results": "Recent results",
    "defense": "Recent defense",
    "pbp": "Play-by-play efficiency",
    "pbp_opponent_adjusted": "Opponent-adjusted PBP matchups",
    "drive": "Drive and possession efficiency",
    "quarterback": "Quarterback state",
    "player_qb": "Projected quarterback difference",
    "player_injuries": "Player injury availability",
    "player_continuity": "Roster and lineup continuity",
    "graph": "Temporal schedule graph",
    "schedule_rating": "Regularized schedule rating",
}

FAMILY_DESCRIPTIONS = {
    "market": "Closing point spread and game total. This is the baseline the model must beat.",
    "context": "Rest, venue, division, weather, and time-of-season context.",
    "elo": "Pregame team-strength ratings updated only after completed games.",
    "experience": "How many current-season games inform each team's state.",
    "offense": "Exponentially weighted efficiency, passing, rushing, turnovers, and sacks.",
    "graph": "Continuous PageRank/HITS ratings computed before each week from prior opponents.",
    "schedule_rating": "Time-weighted ridge/SRS comparator fit to prior scoring margins.",
    "results": "Recent point differential and performance relative to the spread.",
    "defense": "Opponent production allowed, takeaways, and sacks.",
    "pbp": (
        "Versioned early-down EPA, success, explosive plays, pressure, pass rate, "
        "PROE, drives, and field position from prior games."
    ),
    "pbp_opponent_adjusted": (
        "Weekly ridge estimates of expected offensive efficiency against the actual opposing "
        "defense, fit only to earlier games with time decay and shrinkage."
    ),
    "drive": (
        "Pregame points, yards, plays, duration, scoring, and turnover rates per drive, "
        "including opponent-allowed counterparts."
    ),
    "quarterback": "Lagged quarterback efficiency and projected starter state.",
    "player_qb": "Home-away projected quarterback efficiency, experience, and availability.",
    "player_injuries": (
        "Timestamp-filtered position-group unavailability known before the decision cutoff."
    ),
    "player_continuity": (
        "Prior-week snap and roster continuity for offense, defense, and special teams."
    ),
}

METRIC_LABELS = {
    "accuracy": "Side-pick accuracy",
    "brier_score": "Probability error (Brier)",
    "log_loss": "Probability error (log loss)",
    "expected_calibration_error": "Calibration gap (ECE)",
    "roi": "Flat-stake paper ROI",
}


def launch_dashboard(address: str = "127.0.0.1", port: int = 8501, headless: bool = False) -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.headless",
        str(headless).lower(),
    ]
    return subprocess.run(command, check=False).returncode


def _render_value(value: Any, *, percent: bool = False, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if percent:
        return f"{float(value):.1%}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _metric(
    label: str,
    value: Any,
    *,
    percent: bool = False,
    digits: int = 4,
    delta: str | None = None,
    help_text: str | None = None,
) -> None:
    st.metric(
        label,
        _render_value(value, percent=percent, digits=digits),
        delta=delta,
        delta_color="off",
        help=help_text,
    )


def _artifact_time(path: Path) -> str:
    token = path.name.rsplit("-", maxsplit=1)[-1]
    try:
        instant = datetime.strptime(token, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return path.name
    return instant.strftime("%Y-%m-%d %H:%M UTC")


def _show_research_verdict(metrics: dict[str, Any]) -> None:
    accuracy = float(metrics.get("accuracy", np.nan))
    brier = float(metrics.get("brier_score", np.nan))
    roi = float(metrics.get("roi", np.nan))
    if accuracy > 0.5:
        st.success(
            "**ATS classification lead observed.** This model selected the team that covered "
            f"in {accuracy:.2%} of evaluated games, above the 50% coin-flip reference. Check "
            "the blocked interval and season results to judge how stable that lead is."
        )
    else:
        st.warning(
            "**No ATS classification lead in this run.** The model did not select the team "
            "that covered more often than the 50% coin-flip reference."
        )
    st.caption(
        f"Probability calibration (Brier {brier:.4f}) and optional paper return "
        f"({roi:.1%}) answer different questions; neither changes the observed side accuracy."
    )


def _data_summary(data_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    snapshot_metadata: dict[str, Any] | None = None
    feature_metadata: dict[str, Any] | None = None
    try:
        snapshot = latest_snapshot(data_root / "raw")
        snapshot_metadata = describe_snapshot(snapshot)
    except FileNotFoundError:
        pass
    feature_manifest = data_root / "processed" / "game_features.manifest.json"
    if feature_manifest.is_file():
        feature_metadata = read_json(feature_manifest)
    return snapshot_metadata, feature_metadata


def _show_active_ats_summary(active: dict[str, Any]) -> None:
    historical = active.get("historical_evaluation", {})
    forecast = active.get("weekly_forecast", {})
    if not isinstance(historical, dict) or not isinstance(forecast, dict):
        st.error("The active-model manifest is malformed.")
        return
    accuracy = float(historical.get("accuracy", np.nan))
    games = int(historical.get("games", 0))
    correct = int(historical.get("correct", 0))
    st.subheader("Active synchronized ATS model")
    st.success(
        f"**Synchronization: PASS.** Historical evaluation and weekly forecast both use "
        f"`{active.get('method', 'unknown')}` with `{active.get('feature_profile', 'unknown')}` "
        f"features (model `{active.get('model_id', 'unknown')}`)."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Historical ATS accuracy", f"{accuracy:.2%}")
    with c2:
        _metric("Correct classifications", f"{correct:,} / {games:,}")
    with c3:
        _metric("Above coin flip", f"{accuracy - 0.5:.2%}")
    with c4:
        forecast_label = f"{forecast.get('season', '?')} Week {forecast.get('week', '?')}"
        _metric("Current forecast", forecast_label)
    intervals = historical.get("intervals")
    week = intervals.get("week") if isinstance(intervals, dict) else None
    season = intervals.get("season") if isinstance(intervals, dict) else None
    interval_parts = []
    if isinstance(week, dict):
        interval_parts.append(f"week-blocked {week['lower']:.2%}-{week['upper']:.2%}")
    if isinstance(season, dict):
        interval_parts.append(f"season-blocked {season['lower']:.2%}-{season['upper']:.2%}")
    if interval_parts:
        st.caption("Historical 95% accuracy intervals: " + "; ".join(interval_parts) + ".")


def _start_here(data_root: Path, artifacts_root: Path) -> None:
    st.header("Start here")
    st.write(
        "This project asks one hard question: **can a pregame model classify which NFL team "
        "will cover the spread more accurately than a coin flip?** Probability calibration "
        "and paper betting are useful secondary analyses, not the definition of success."
    )

    active = load_active_ats_model(artifacts_root)
    directories = artifact_directories(artifacts_root / "backtests", "metrics.json")
    if active is not None:
        _show_active_ats_summary(active)
    elif not directories:
        st.warning("There is no synchronized ATS model yet. Run an evaluation and weekly card.")
    else:
        latest = directories[0]
        metrics = read_json(latest / "metrics.json")
        _show_research_verdict(metrics)
        st.subheader("Latest historical result")
        st.caption(
            f"Saved {_artifact_time(latest)} · {metrics.get('model_name', 'unknown')} model · "
            f"{metrics.get('feature_set', 'unknown')} feature set · "
            f"{metrics.get('first_test_gameday', '?')} through "
            f"{metrics.get('last_test_gameday', '?')}"
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _metric(
                "All-game side accuracy",
                metrics.get("accuracy"),
                percent=True,
                delta=f"{float(metrics.get('accuracy', 0.0)) - 0.5:+.1%} vs 50/50",
                help_text="The model's more-likely ATS side on every non-push game.",
            )
        with c2:
            _metric(
                "Probability error",
                metrics.get("brier_score"),
                delta=f"{float(metrics.get('brier_score', 0.25)) - 0.25:+.4f} vs 0.2500",
                help_text="Brier score. Lower is better; 0.2500 is the 50/50 reference.",
            )
        with c3:
            _metric(
                "Selected-bet paper ROI",
                metrics.get("roi"),
                percent=True,
                delta=f"{int(metrics.get('bets', 0))} paper bets",
                help_text="Flat one-unit result after the stored prices; no real wagers.",
            )
        portfolio = metrics.get("paper_portfolio", {})
        with c4:
            _metric(
                "Kelly paper bankroll",
                portfolio.get("final_bankroll") if isinstance(portfolio, dict) else None,
                digits=2,
                delta=(
                    f"{float(portfolio.get('return', 0.0)):+.1%} return"
                    if isinstance(portfolio, dict)
                    else None
                ),
                help_text="A simulated bankroll using quarter Kelly and exposure caps.",
            )
        st.caption(
            "This is a legacy direct-model benchmark because no synchronized active outcome "
            "model is available yet."
        )

    st.subheader("Where to go next")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Historical test**")
        st.write(
            "See whether performance is stable by season and whether probabilities are honest."
        )
    with c2:
        st.markdown("**2. Feature lab**")
        st.write("Compare small and large feature sets on exactly the same historical weeks.")
    with c3:
        st.markdown("**3. Weekly forecast**")
        st.write("Inspect a saved card, including passes and the separate forced-pick pool card.")

    snapshot, feature_manifest = _data_summary(data_root)
    st.subheader("What is loaded")
    if snapshot is None:
        st.warning("No complete raw data snapshot was found.")
    else:
        manifest = snapshot["manifest"]
        c1, c2, c3 = st.columns(3)
        with c1:
            _metric("NFL games in source", manifest["files"]["schedules.parquet"]["rows"])
        with c2:
            _metric("Team-game stat rows", manifest["files"]["team_stats.parquet"]["rows"])
        with c3:
            _metric(
                "Processed game rows",
                feature_manifest.get("rows") if feature_manifest is not None else None,
            )
        st.caption(
            f"Raw snapshot {snapshot['snapshot_id']} fetched {manifest['fetched_at_utc']}. "
            "Use Data health for provenance and missingness."
        )

    st.info(
        "**Important limitation:** historical nflverse spreads are closing lines. Until the "
        "project archives earlier quotes with timestamps, a backtest cannot answer whether a "
        "specific Monday or Thursday decision would have beaten the market available then."
    )


@st.cache_data(show_spinner=False)
def _compute_uncertainty(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            block_bootstrap_intervals(predictions, samples=500, block="week"),
            block_bootstrap_intervals(predictions, samples=500, block="season"),
        ],
        ignore_index=True,
    )


def _uncertainty_table(intervals: pd.DataFrame, block: str) -> pd.DataFrame:
    selected = intervals.loc[
        intervals["block"].eq(block) & intervals["metric"].isin(["accuracy", "brier_score", "roi"])
    ].copy()
    references = {"accuracy": 0.5, "brier_score": 0.25, "roi": 0.0}
    percent_metrics = {"accuracy", "roi"}
    rows = []
    for _, row in selected.iterrows():
        metric = str(row["metric"])
        percent = metric in percent_metrics
        rows.append(
            {
                "Measure": METRIC_LABELS[metric],
                "Estimate": _render_value(row["estimate"], percent=percent),
                "95% interval": (
                    f"{_render_value(row['lower'], percent=percent)} to "
                    f"{_render_value(row['upper'], percent=percent)}"
                ),
                "Simple reference": _render_value(references[metric], percent=percent),
            }
        )
    return pd.DataFrame(rows)


def _historical_test(artifacts_root: Path) -> None:
    st.header("Historical test")
    st.write(
        "A walk-forward test recreates each NFL week using only games from earlier dates. The "
        "model is refit, scores that week once, and then moves forward. This is the primary "
        "screen for deciding whether an idea deserves another experiment."
    )
    active = load_active_ats_model(artifacts_root)
    if active is not None:
        _show_active_ats_summary(active)
        st.divider()
        st.subheader("Archived direct-classifier diagnostics")
        st.caption(
            "The runs below are retained research artifacts. They do not replace the active "
            "model summarized above or the matching evaluation in Outcome lab."
        )
    directories = artifact_directories(artifacts_root / "backtests", "metrics.json")
    if not directories:
        st.warning("No saved historical tests. Run `nfl-ats backtest`.")
        return
    selected = st.selectbox(
        "Saved test run",
        directories,
        format_func=lambda path: f"{_artifact_time(path)} ({path.name})",
    )
    metrics = read_json(selected / "metrics.json")
    predictions = pd.read_parquet(selected / "predictions.parquet")
    _show_research_verdict(metrics)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric("Games tested", metrics.get("games_evaluated"))
    with c2:
        _metric("Side accuracy", metrics.get("accuracy"), percent=True)
    with c3:
        _metric("Brier error", metrics.get("brier_score"))
    with c4:
        _metric("Calibration gap", metrics.get("expected_calibration_error"), percent=True)
    with c5:
        _metric("Flat paper ROI", metrics.get("roi"), percent=True)
    st.caption(
        "Classification reference: 50% side accuracy. Brier error is a probability diagnostic; "
        "paper ROI is a separate optional simulation."
    )

    tabs = st.tabs(
        [
            "What it means",
            "Season by season",
            "Probability honesty",
            "Paper bankroll",
            "Audit details",
        ]
    )
    with tabs[0]:
        st.subheader("How uncertain is this result?")
        uncertainty_path = selected / "uncertainty.csv"
        intervals = (
            pd.read_csv(uncertainty_path)
            if uncertainty_path.is_file()
            else _compute_uncertainty(predictions)
        )
        block_label = st.radio(
            "Keep correlated games together by",
            ("NFL week", "Season"),
            horizontal=True,
            help=(
                "A block bootstrap resamples whole groups instead of pretending every game is "
                "independent. Season blocks are more conservative but there are fewer of them."
            ),
        )
        block = "week" if block_label == "NFL week" else "season"
        st.dataframe(_uncertainty_table(intervals, block), hide_index=True, width="stretch")
        if not uncertainty_path.is_file():
            st.caption(
                "Preview intervals use 500 deterministic resamples. New backtest artifacts save "
                "2,000-resample week and season intervals for reproducibility."
            )
        st.markdown("**What would count as progress?**")
        st.write(
            "ATS accuracy above 50% that repeats across seasons, with week- and season-blocked "
            "intervals supporting the improvement. Brier error diagnoses probability "
            "calibration, while ROI belongs only to the optional paper-betting analysis."
        )
        with st.expander("How leakage is prevented"):
            st.write(
                "For every test week, the training cutoff is the earliest game date in that "
                "week. All training games are strictly earlier. Preprocessing, imputation, model "
                "fitting, and calibration are learned inside that historical training window."
            )

    with tabs[1]:
        scorecard = season_scorecard(predictions)
        st.subheader("Does the result repeat?")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.caption("Side accuracy — above 50% is directionally better")
            accuracy_chart = scorecard[["season", "accuracy"]].copy()
            accuracy_chart["50/50 reference"] = 0.5
            st.line_chart(
                accuracy_chart,
                x="season",
                y=["accuracy", "50/50 reference"],
                height=280,
            )
        with chart_right:
            st.caption("Brier error — below 0.2500 is directionally better")
            brier_chart = scorecard[["season", "brier_score"]].copy()
            brier_chart["50/50 reference"] = 0.25
            st.line_chart(
                brier_chart,
                x="season",
                y=["brier_score", "50/50 reference"],
                height=280,
            )
        display_columns = [
            column
            for column in (
                "season",
                "games",
                "accuracy",
                "brier_score",
                "expected_calibration_error",
                "bets",
                "wins",
                "losses",
                "bet_pushes",
                "roi",
            )
            if column in scorecard
        ]
        st.dataframe(
            scorecard.loc[:, display_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "season": "Season",
                "games": "Games",
                "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                "brier_score": st.column_config.NumberColumn("Brier", format="%.4f"),
                "expected_calibration_error": st.column_config.NumberColumn(
                    "Calibration gap", format="percent"
                ),
                "bets": "Paper bets",
                "wins": "Wins",
                "losses": "Losses",
                "bet_pushes": "Pushes",
                "roi": st.column_config.NumberColumn("ROI", format="percent"),
            },
        )

    with tabs[2]:
        st.subheader("Do stated probabilities match reality?")
        st.write(
            "Predictions are grouped into probability buckets. If games called 55% actually "
            "cover about 55% of the time, the model is calibrated."
        )
        calibration = calibration_table(predictions)
        if calibration.empty:
            st.info("There are no completed non-push predictions to calibrate.")
        else:
            chart = calibration[["mean_probability", "observed_cover_rate"]].copy()
            chart["ideal_calibration"] = chart["mean_probability"]
            st.line_chart(
                chart,
                x="mean_probability",
                y=["observed_cover_rate", "ideal_calibration"],
                x_label="Model probability",
                y_label="Observed home-cover rate",
                height=330,
            )
            st.caption("The observed line should track the ideal calibration line.")
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
                        "Observed home covers", format="percent"
                    ),
                    "gap": st.column_config.NumberColumn("Gap", format="percent"),
                },
            )

    with tabs[3]:
        ledger_path = selected / "paper_ledger.parquet"
        if not ledger_path.is_file():
            st.info("This test predates the paper-bankroll ledger.")
        else:
            ledger = pd.read_parquet(ledger_path)
            portfolio_path = selected / "portfolio_metrics.json"
            portfolio = read_json(portfolio_path) if portfolio_path.is_file() else {}
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                _metric("Starting bankroll", portfolio.get("initial_bankroll"), digits=2)
            with c2:
                _metric("Ending bankroll", portfolio.get("final_bankroll"), digits=2)
            with c3:
                _metric("Paper return", portfolio.get("return"), percent=True)
            with c4:
                _metric("Worst drawdown", portfolio.get("max_drawdown"), percent=True)
            curve = bankroll_curve(ledger)
            st.line_chart(curve, x="season_week", y="bankroll", height=330)
            st.caption(
                "Quarter Kelly sizes stakes from model probabilities, capped at 2% per game and "
                "10% per NFL week. Sizing cannot rescue inaccurate probabilities."
            )
            simulation_path = selected / "bankroll_simulation.json"
            paths_path = selected / "bankroll_paths.parquet"
            if simulation_path.is_file():
                simulation = read_json(simulation_path)
                st.subheader("Conditional bankroll risk")
                st.caption(
                    "These paths assume the model probabilities are correct. They show sizing "
                    "risk, not proof that the probabilities are accurate."
                )
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    _metric(
                        "Median ending bankroll",
                        simulation.get("terminal_bankroll_median"),
                        digits=2,
                    )
                with s2:
                    _metric(
                        "5th-percentile ending bankroll",
                        simulation.get("terminal_bankroll_p05"),
                        digits=2,
                    )
                with s3:
                    _metric(
                        "Chance of finishing below start",
                        simulation.get("probability_of_loss"),
                        percent=True,
                    )
                with s4:
                    _metric(
                        "Median worst drawdown",
                        simulation.get("median_max_drawdown"),
                        percent=True,
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
                        .rename(
                            columns={
                                0.05: "5th percentile",
                                0.50: "Median",
                                0.95: "95th percentile",
                            }
                        )
                        .reset_index()
                    )
                    st.line_chart(
                        bands,
                        x="season_week",
                        y=["5th percentile", "Median", "95th percentile"],
                        height=300,
                    )
            with st.expander("Show every paper decision"):
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
                        "bet_probability": st.column_config.NumberColumn(
                            "Model probability", format="percent"
                        ),
                        "stake": st.column_config.NumberColumn("Stake", format="%.2f"),
                        "profit": st.column_config.NumberColumn("Profit", format="%.2f"),
                        "bankroll_after_week": st.column_config.NumberColumn(
                            "Bankroll", format="%.2f"
                        ),
                    },
                )

    with tabs[4]:
        st.subheader("Reproducibility record")
        st.write(
            "These details identify the exact configuration, source snapshot, feature table, "
            "dependency lock, and code revision used for this run."
        )
        run_path = selected / "run.json"
        if run_path.is_file():
            run = read_json(run_path)
            configuration = run.get("configuration", {})
            st.markdown(
                f"**Training cutoff rule:** minimum {configuration.get('min_train_games', '?')} "
                f"prior games · **Model:** {configuration.get('model', '?')} · "
                f"**Features:** {configuration.get('feature_set', '?')} · "
                f"**Paper edge threshold:** "
                f"{float(configuration.get('min_edge', 0.0)):.1%}"
            )
            st.json(run, expanded=False)
        card_path = selected / "model_card.md"
        if card_path.is_file():
            with st.expander("Read this run's model card"):
                card_text = card_path.read_text(encoding="utf-8")
                st.markdown(card_text)
                st.download_button(
                    "Download model card",
                    data=card_text,
                    file_name=f"{selected.name}-model-card.md",
                    mime="text/markdown",
                )
        st.caption(f"Artifact directory: {selected}")
        st.warning(
            "Closing-line value is not shown yet because this project does not yet archive "
            "timestamped decision lines and same-book closing quotes."
        )


def _feature_set_description(name: str) -> str:
    descriptions = {
        "market": "Spread + total only",
        "market_context": "Market + rest, venue, weather, division, and season timing",
        "market_elo": "Market + Elo strength",
        "football": "Football state without the betting market",
        "full_without_ats": "All current inputs except prior ATS residuals",
        "full": "All 58 current inputs",
        "graph": "Temporal PageRank/HITS only",
        "schedule_rating": "Regularized opponent-adjusted scoring ratings only",
        "market_graph": "Market/context + temporal PageRank/HITS",
        "market_schedule": "Market/context + regularized schedule ratings",
        "market_graph_schedule": "Market/context + both opponent-adjustment methods",
        "football_graph": "Football state + temporal PageRank/HITS",
        "football_schedule": "Football state + regularized schedule ratings",
        "football_graph_schedule": "Football state + both opponent-adjustment methods",
        "full_graph": "Full baseline + temporal PageRank/HITS",
        "full_schedule": "Full baseline + regularized schedule ratings",
        "full_graph_schedule": "Full baseline + both opponent-adjustment methods",
    }
    return descriptions.get(name, name)


def _validation_lab(artifacts_root: Path) -> None:
    st.header("Validation and outer-test lab")
    st.write(
        "This is the model-selection test. For each outer season, candidate configurations "
        "are ranked using only earlier validation seasons. The winner is fixed before the "
        "outer season is scored. Validation is intentional; outer-test reuse is what this "
        "protocol prevents."
    )
    directories = artifact_directories(artifacts_root / "nested_evaluations", "metrics.json")
    if not directories:
        st.info("No nested evaluation is saved yet. Run `nfl-ats nested-evaluate`.")
        return
    selected = st.selectbox(
        "Saved nested evaluation",
        directories,
        format_func=lambda path: f"{_artifact_time(path)} ({path.name})",
    )
    metrics = read_json(selected / "metrics.json")
    folds = pd.read_csv(selected / "fold_summary.csv")
    candidates = pd.read_csv(selected / "candidate_validation.csv")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Outer seasons", metrics.get("outer_folds"))
    with c2:
        _metric("Outer-test accuracy", metrics.get("accuracy"), percent=True)
    with c3:
        _metric("Outer-test Brier", metrics.get("brier_score"))
    with c4:
        _metric("Outer-test paper ROI", metrics.get("roi"), percent=True)
    st.caption(
        f"{metrics.get('candidate_count', '?')} frozen candidates · "
        f"{metrics.get('validation_seasons', '?')} validation seasons per fold · selection by "
        f"{metrics.get('selection_metric', '?')}."
    )

    st.subheader("What was selected before each outer season?")
    fold_columns = [
        column
        for column in (
            "outer_test_season",
            "validation_start_season",
            "validation_end_season",
            "selected_model_name",
            "selected_feature_set",
            "validation_selection_score",
            "test_accuracy",
            "test_brier_score",
            "test_log_loss",
            "test_roi",
        )
        if column in folds
    ]
    st.dataframe(
        folds.loc[:, fold_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "outer_test_season": "Outer season",
            "validation_start_season": "Validation starts",
            "validation_end_season": "Validation ends",
            "selected_model_name": "Selected model",
            "selected_feature_set": "Selected inputs",
            "validation_selection_score": st.column_config.NumberColumn(
                "Validation score", format="%.4f"
            ),
            "test_accuracy": st.column_config.NumberColumn("Outer accuracy", format="percent"),
            "test_brier_score": st.column_config.NumberColumn("Outer Brier", format="%.4f"),
            "test_log_loss": st.column_config.NumberColumn("Outer log loss", format="%.4f"),
            "test_roi": st.column_config.NumberColumn("Outer ROI", format="percent"),
        },
    )
    if {"outer_test_season", "validation_selection_score", "test_brier_score"}.issubset(
        folds.columns
    ):
        comparison = folds[
            ["outer_test_season", "validation_selection_score", "test_brier_score"]
        ].rename(
            columns={
                "validation_selection_score": "Prior validation Brier",
                "test_brier_score": "Outer-test Brier",
            }
        )
        st.line_chart(
            comparison,
            x="outer_test_season",
            y=["Prior validation Brier", "Outer-test Brier"],
            height=300,
        )

    with st.expander("See every candidate ranking"):
        display_columns = [
            column
            for column in (
                "outer_test_season",
                "candidate_id",
                "feature_count",
                "selection_rank",
                "selected",
                "brier_score",
                "log_loss",
                "accuracy",
                "roi",
            )
            if column in candidates
        ]
        st.dataframe(
            candidates.loc[:, display_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "outer_test_season": "Outer season",
                "candidate_id": "Candidate",
                "feature_count": "Inputs",
                "selection_rank": "Rank on validation",
                "selected": "Selected",
                "brier_score": st.column_config.NumberColumn("Validation Brier", format="%.4f"),
                "log_loss": st.column_config.NumberColumn("Validation log loss", format="%.4f"),
                "accuracy": st.column_config.NumberColumn("Validation accuracy", format="percent"),
                "roi": st.column_config.NumberColumn("Validation ROI", format="percent"),
            },
        )

    dependence_path = selected / "dependence_summary.json"
    if not dependence_path.is_file():
        dependence_runs = artifact_directories(artifacts_root / "dependence", "summary.json")
        if dependence_runs:
            candidate_path = dependence_runs[0] / "summary.json"
            candidate_summary = read_json(candidate_path)
            expected = (selected / "predictions.parquet").resolve()
            if candidate_summary.get("source_predictions") == str(expected):
                dependence_path = candidate_path
    if dependence_path.is_file():
        dependence = read_json(dependence_path)
        st.subheader("Do team-specific errors persist?")
        d1, d2, d3 = st.columns(3)
        with d1:
            _metric(
                "Pooled lag-1 correlation",
                dependence.get("pooled_team_lag1_error_correlation"),
            )
        with d2:
            _metric(
                "Season-shuffle p-value",
                dependence.get("season_shuffle_two_sided_p_value"),
            )
        with d3:
            _metric("Team lag pairs", dependence.get("team_lag_pairs"))
        st.caption(
            "The null range comes from shuffling forecast errors within seasons while keeping "
            "the schedule structure. Dependence changes uncertainty estimates; it does not "
            "make chronological validation invalid."
        )
    st.info(
        "These outer seasons are historically out of time under the declared algorithm, but "
        "2018-2025 results have already influenced the broader research project. Treat this as "
        "a corrected repeatable benchmark, not a claim that those years were never observed."
    )


def _feature_lab(artifacts_root: Path) -> None:
    st.header("Feature lab")
    st.write(
        "This page asks whether adding information improves out-of-time ATS classification. "
        "Every feature set is tested on the same NFL weeks. Accuracy is the headline result; "
        "Brier and log loss diagnose the quality of the associated probabilities."
    )
    directories = artifact_directories(artifacts_root / "experiments", "summary.csv")
    if not directories:
        st.info("No comparison is saved yet. Run `nfl-ats experiment`.")
        rows = [
            {
                "Feature set": name,
                "Inputs": len(columns),
                "Meaning": _feature_set_description(name),
            }
            for name, columns in FEATURE_SETS.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        return
    _feature_lab_saved(directories)


def _outcome_lab(artifacts_root: Path) -> None:
    st.header("Winner, fair spread, and ATS lab")
    st.write(
        "Instead of asking only which side covers, this page separates three questions: "
        "**who wins**, **what should the margin be**, and **how that fair margin differs "
        "from the market spread**. ATS accuracy measures the primary binary classification "
        "goal; probability and margin errors are separate diagnostics."
    )
    directories = artifact_directories(artifacts_root / "margins", "summary.csv")
    if not directories:
        st.warning("No saved outcome-model test. Run `nfl-ats margin-backtest` first.")
        return
    active = load_active_ats_model(artifacts_root)
    active_evaluation = (
        active_artifact_path(artifacts_root, active, "historical_evaluation")
        if active is not None
        else None
    )
    if active_evaluation is not None:
        directories = sorted(
            directories,
            key=lambda path: path.resolve() != active_evaluation.resolve(),
        )
    selected = st.selectbox(
        "Saved outcome-model test",
        directories,
        format_func=lambda path: _artifact_time(path),
    )
    if (
        active is not None
        and active_evaluation is not None
        and selected.resolve() == active_evaluation.resolve()
    ):
        st.success(
            f"Synchronization: PASS · active model `{active.get('model_id')}` · this is the "
            "historical evaluation linked to the current weekly forecast."
        )
    elif active_evaluation is not None:
        st.warning("Archived evaluation selected; it is not the model on the weekly page.")
    summary = pd.read_csv(selected / "summary.csv")
    metadata = (
        read_json(selected / "metadata.json") if (selected / "metadata.json").is_file() else {}
    )
    profile = metadata.get("feature_profile", "base")
    st.caption(
        f"{metadata.get('regressor', 'unknown')} margin regressor · {profile} feature profile · "
        f"testing starts in {metadata.get('start_season', '?')} · expanding weekly refits"
    )
    method_labels = {
        "market": "Closing market",
        "fair_margin": "Independent fair margin",
        "market_residual": "Market + predicted correction",
        "straight_up": "Straight-up classifier",
        "direct_ats": "Direct ATS classifier",
    }
    market = summary.loc[summary["method"].eq("market")]
    market_row = market.iloc[0] if not market.empty else pd.Series(dtype=float)
    fair = summary.loc[summary["method"].eq("fair_margin")]
    residual = summary.loc[summary["method"].eq("market_residual")]

    ats_candidates = summary.loc[
        summary["method"].isin(["direct_ats", "fair_margin", "market_residual"])
        & summary["cover_accuracy"].notna()
    ]
    if not ats_candidates.empty:
        best_ats = ats_candidates.sort_values("cover_accuracy", ascending=False).iloc[0]
        accuracy = float(best_ats["cover_accuracy"])
        games = int(best_ats["cover_games"])
        correct = round(accuracy * games)
        approach = method_labels.get(str(best_ats["method"]), str(best_ats["method"]))
        if accuracy > 0.5:
            st.success(
                f"**Observed ATS classification lead:** {approach} selected the covering team "
                f"in **{correct:,} of {games:,} games ({accuracy:.2%})**, which is "
                f"{accuracy - 0.5:.2%} above the coin-flip reference."
            )
        else:
            st.warning(
                f"The best learned ATS classifier in this run reached {accuracy:.2%}, below "
                "the 50% coin-flip reference."
            )
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            _metric("Correct ATS classifications", correct)
        with a2:
            _metric("Non-push games", games)
        with a3:
            _metric("ATS accuracy", f"{accuracy:.2%}")
        with a4:
            _metric("Above coin flip", f"{accuracy - 0.5:.2%}")

    if not fair.empty and not market.empty:
        fair_row = fair.iloc[0]
        if float(fair_row["margin_mae"]) < float(market_row["margin_mae"]):
            st.success(
                "The independent fair-spread model beat the closing spread on absolute margin "
                "error in this test. Check uncertainty and season stability before promoting it."
            )
        else:
            st.warning(
                "The closing market remained the better margin forecast. The fair-spread model "
                "can predict winners while still adding no ATS edge."
            )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Market winner accuracy", market_row.get("win_accuracy"), percent=True)
    with c2:
        _metric(
            "Fair-model winner accuracy",
            fair.iloc[0].get("win_accuracy") if not fair.empty else np.nan,
            percent=True,
        )
    with c3:
        _metric("Market margin MAE", market_row.get("margin_mae"), digits=2)
    with c4:
        _metric(
            "Fair-model margin MAE",
            fair.iloc[0].get("margin_mae") if not fair.empty else np.nan,
            digits=2,
        )

    readable = summary.copy()
    readable["Approach"] = readable["method"].map(method_labels).fillna(readable["method"])
    display_columns = [
        "Approach",
        "win_accuracy",
        "win_brier_score",
        "margin_mae",
        "cover_accuracy",
        "cover_brier_score",
        "roi",
    ]
    for column in display_columns:
        if column not in readable:
            readable[column] = np.nan
    readable = readable[display_columns].rename(
        columns={
            "win_accuracy": "Winner accuracy",
            "win_brier_score": "Winner Brier",
            "margin_mae": "Margin MAE",
            "cover_accuracy": "ATS accuracy",
            "cover_brier_score": "ATS Brier",
            "roi": "Flat paper ROI",
        }
    )
    st.subheader("Same games, different forecasting questions")
    st.dataframe(
        readable,
        hide_index=True,
        width="stretch",
        column_config={
            "Winner accuracy": st.column_config.NumberColumn(format="percent"),
            "Winner Brier": st.column_config.NumberColumn(format="%.4f"),
            "Margin MAE": st.column_config.NumberColumn(format="%.2f points"),
            "ATS accuracy": st.column_config.NumberColumn(format="percent"),
            "ATS Brier": st.column_config.NumberColumn(format="%.4f"),
            "Flat paper ROI": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Winner accuracy is not ATS accuracy. A favorite can be correctly forecast to win but "
        "still fail to cover. ATS accuracy answers the binary classification question. Brier "
        "scores only evaluate whether the probability magnitudes are well calibrated."
    )

    if not residual.empty and not market.empty:
        residual_row = residual.iloc[0]
        st.subheader("Did correcting the market help?")
        d1, d2, d3 = st.columns(3)
        with d1:
            _metric(
                "Winner-accuracy change",
                float(residual_row["win_accuracy"]) - float(market_row["win_accuracy"]),
                percent=True,
            )
        with d2:
            _metric(
                "Margin-MAE change",
                float(residual_row["margin_mae"]) - float(market_row["margin_mae"]),
                digits=3,
                help_text="Negative is better.",
            )
        with d3:
            _metric(
                "ATS accuracy above chance",
                float(residual_row["cover_accuracy"]) - 0.5,
                percent=True,
                help_text="Positive means more correct ATS classifications than a coin flip.",
            )

    uncertainty_path = selected / "uncertainty.csv"
    if uncertainty_path.is_file():
        uncertainty = pd.read_csv(uncertainty_path)
        st.subheader("How uncertain are those differences?")
        block = st.radio(
            "Resampling unit",
            ("week", "season"),
            horizontal=True,
            key="outcome_uncertainty_block",
        )
        useful = uncertainty.loc[
            uncertainty["block"].eq(block)
            & uncertainty["method"].isin(["fair_margin", "market_residual", "straight_up"])
            & uncertainty["metric"].isin(
                ["cover_accuracy", "win_accuracy", "margin_mae", "cover_brier_score"]
            )
        ].copy()
        useful["Approach"] = useful["method"].map(method_labels)
        useful["Measure"] = useful["metric"].replace(
            {
                "win_accuracy": "Winner accuracy",
                "margin_mae": "Margin MAE",
                "cover_accuracy": "ATS accuracy",
                "cover_brier_score": "ATS Brier",
            }
        )
        useful["Estimate"] = useful["estimate"]
        useful["95% interval"] = useful.apply(
            lambda row: f"{row['lower']:.4f} to {row['upper']:.4f}", axis=1
        )
        useful["Change vs market"] = useful.get("delta_vs_market", np.nan)
        useful["Change interval"] = useful.apply(
            lambda row: (
                f"{row['delta_lower']:.4f} to {row['delta_upper']:.4f}"
                if pd.notna(row.get("delta_lower"))
                else "—"
            ),
            axis=1,
        )
        st.dataframe(
            useful[
                [
                    "Approach",
                    "Measure",
                    "Estimate",
                    "95% interval",
                    "Change vs market",
                    "Change interval",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

    predictions_path = selected / "predictions.parquet"
    if predictions_path.is_file():
        predictions = pd.read_parquet(predictions_path)
        example = predictions.loc[
            predictions["method"].eq("fair_margin") & predictions["result"].notna()
        ].tail(12)
        if not example.empty:
            st.subheader("Recent fair-line examples from this historical test")
            example = example[
                [
                    "gameday",
                    "away_team",
                    "home_team",
                    "market_spread",
                    "fair_spread",
                    "predicted_market_residual",
                    "result",
                    "home_win_probability",
                ]
            ].rename(
                columns={
                    "gameday": "Date",
                    "away_team": "Away",
                    "home_team": "Home",
                    "market_spread": "Market spread",
                    "fair_spread": "Model fair margin",
                    "predicted_market_residual": "Model-market difference",
                    "result": "Actual home margin",
                    "home_win_probability": "Home win probability",
                }
            )
            st.dataframe(example, hide_index=True, width="stretch")
        return


def _feature_lab_saved(directories: list[Path]) -> None:
    selected = st.selectbox(
        "Saved comparison",
        directories,
        format_func=lambda path: f"{_artifact_time(path)} ({path.name})",
    )
    summary = (
        pd.read_csv(selected / "summary.csv").sort_values("accuracy", ascending=False).reset_index(
            drop=True
        )
    )
    best = summary.iloc[0]
    best_probability = summary.loc[summary["brier_score"].idxmin()]
    st.info(
        f"**Best ATS classification in this run:** `{best['feature_set']}` selected the "
        f"covering team in {best['accuracy']:.2%} of {int(best['games_evaluated']):,} games. "
        "That is the primary result; blocked intervals determine how stable it is."
    )
    st.caption(
        f"Probability diagnostic: `{best_probability['feature_set']}` had the lowest Brier "
        f"score at {best_probability['brier_score']:.5f}."
    )

    st.subheader("Overall comparison")
    st.caption("Best region: up for higher ATS accuracy and left for lower probability error.")
    st.scatter_chart(
        summary,
        x="brier_score",
        y="accuracy",
        color="feature_set",
        size="games_evaluated",
        height=380,
    )
    display = summary[
        [
            "feature_set",
            "feature_count",
            "games_evaluated",
            "accuracy",
            "brier_score",
            "expected_calibration_error",
            "bets",
            "roi",
        ]
    ].copy()
    display["description"] = display["feature_set"].map(_feature_set_description)
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
            "bets": "Paper bets",
            "roi": st.column_config.NumberColumn("Paper ROI", format="percent"),
        },
    )
    st.warning(
        "Small accuracy differences can be sampling noise. Use the paired week/season intervals "
        "below before promoting a feature set. The paper ROI column is optional and is not the "
        "classification objective."
    )

    paired_path = selected / "paired_comparisons.csv"
    if paired_path.is_file():
        st.subheader("Paired improvement over the baseline")
        st.write(
            "Each candidate is compared on the exact same games. Positive values favor the "
            "candidate; an interval crossing zero means the improvement is unresolved."
        )
        paired = pd.read_csv(paired_path)
        block = st.radio(
            "Uncertainty block",
            ["season", "week"],
            horizontal=True,
            key="feature_paired_block",
        )
        paired_display = paired.loc[paired["block"].eq(block)].copy()
        paired_display["interval"] = paired_display.apply(
            lambda row: f"{row['estimate']:.5f} [{row['lower']:.5f}, {row['upper']:.5f}]",
            axis=1,
        )
        paired_display["conclusion"] = np.select(
            [paired_display["lower"].gt(0), paired_display["upper"].lt(0)],
            ["Candidate improved", "Candidate worsened"],
            default="Unresolved",
        )
        st.dataframe(
            paired_display[
                [
                    "candidate_feature_set",
                    "metric",
                    "interval",
                    "conclusion",
                    "paired_games",
                ]
            ].rename(
                columns={
                    "candidate_feature_set": "Candidate",
                    "metric": "Improvement metric",
                    "interval": "Estimate [95% interval]",
                    "conclusion": "Conclusion",
                    "paired_games": "Paired games",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    season_path = selected / "season_summary.csv"
    if season_path.is_file():
        st.subheader("Does the ranking survive each season?")
        season_summary = pd.read_csv(season_path)
        feature_set = st.selectbox(
            "Feature set to inspect",
            sorted(season_summary["feature_set"].unique()),
        )
        selected_seasons = season_summary.loc[season_summary["feature_set"].eq(feature_set)].copy()
        c1, c2 = st.columns(2)
        with c1:
            accuracy = selected_seasons[["season", "accuracy"]].copy()
            accuracy["50/50 reference"] = 0.5
            st.line_chart(accuracy, x="season", y=["accuracy", "50/50 reference"], height=280)
        with c2:
            brier = selected_seasons[["season", "brier_score"]].copy()
            brier["50/50 reference"] = 0.25
            st.line_chart(
                brier,
                x="season",
                y=["brier_score", "50/50 reference"],
                height=280,
            )
        st.dataframe(selected_seasons, hide_index=True, width="stretch")
    with st.expander("Exact experiment configuration"):
        metadata = selected / "metadata.json"
        if metadata.is_file():
            st.json(read_json(metadata), expanded=False)


def _favorite_label(row: pd.Series) -> str:
    spread = float(row["spread_line"])
    if spread > 0:
        return f"{row['home_team']} -{spread:g}"
    if spread < 0:
        return f"{row['away_team']} -{abs(spread):g}"
    return "Pick'em"


def _readable_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    table = recommendations.copy()
    table["Matchup"] = table["away_team"] + " at " + table["home_team"]
    table["Market line"] = table.apply(_favorite_label, axis=1)
    home_lean = table["home_cover_probability"].ge(0.5)
    table["ATS pick"] = table["home_team"].where(home_lean, table["away_team"])
    pick_line = (-table["spread_line"]).where(home_lean, table["spread_line"])
    table["ATS prediction"] = table["ATS pick"] + " " + pick_line.map(
        lambda value: "PK" if value == 0 else f"{value:+g}"
    )
    table["Estimated cover chance"] = table["home_cover_probability"].where(
        home_lean, 1.0 - table["home_cover_probability"]
    )
    table["Separation from coin flip"] = table["Estimated cover chance"] - 0.5
    table["Signal"] = np.select(
        [
            table["Separation from coin flip"].lt(0.01),
            table["Separation from coin flip"].lt(0.025),
            table["Separation from coin flip"].lt(0.05),
        ],
        ["No separation", "Weak lean", "Moderate lean"],
        default="Larger lean (unvalidated)",
    )
    selected_team = table["home_team"].where(table["bet_side"].eq("HOME"), table["away_team"])
    table["Paper action"] = np.where(
        table["bet_side"].eq("PASS"), "PASS", "Paper pick: " + selected_team
    )
    table["Model edge"] = table["edge"]
    return table[
        [
            "gameday",
            "Matchup",
            "Market line",
            "ATS prediction",
            "Estimated cover chance",
            "Separation from coin flip",
            "Signal",
            "Paper action",
        ]
    ]


def _forecast_label(path: Path) -> str:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return f"Saved {_artifact_time(path)}"
    metadata = read_json(metadata_path)
    created_value = metadata.get("created_at_utc")
    created = (
        pd.to_datetime(str(created_value), errors="coerce")
        if created_value is not None
        else pd.NaT
    )
    created_label = created.strftime("%b %d, %Y %I:%M %p UTC") if pd.notna(created) else "unknown"
    feature_name = metadata.get("feature_profile", metadata.get("feature_set", "unknown"))
    ats_method = metadata.get("ats_method")
    if ats_method is None and metadata.get("command") == "margin-predict":
        ats_method = "market_residual"
    method_label = f" · {ats_method}" if ats_method is not None else ""
    return (
        f"{metadata.get('season', '?')} Week {metadata.get('week', '?')} · "
        f"{feature_name}{method_label} · generated {created_label}"
    )


def _load_weekly_ats_forecast(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the named ATS method from either outcome or legacy direct cards."""

    metadata_path = path / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    recommendations_path = path / "recommendations.csv"
    if recommendations_path.is_file():
        recommendations = pd.read_csv(recommendations_path)
        metadata.setdefault("ats_method", "direct_ats")
        return recommendations, metadata

    predictions_path = path / "predictions.csv"
    if not predictions_path.is_file():
        raise ValueError(f"Forecast {path.name} has no ATS prediction table")
    predictions = pd.read_csv(predictions_path)
    ats_method = str(metadata.get("ats_method", "market_residual"))
    recommendations = predictions.loc[predictions["method"].eq(ats_method)].copy()
    if recommendations.empty:
        raise ValueError(f"Forecast {path.name} has no rows for ATS method {ats_method!r}")
    metadata["ats_method"] = ats_method
    return recommendations, metadata


def _matching_historical_ats_benchmark(
    artifacts_root: Path, metadata: dict[str, Any]
) -> dict[str, Any] | None:
    """Find the latest walk-forward result matching a weekly outcome card."""

    linked = metadata.get("historical_evaluation")
    if isinstance(linked, dict) and linked.get("accuracy") is not None:
        linked_result: dict[str, Any] = {
            "accuracy": float(linked["accuracy"]),
            "games": int(linked["games"]),
            "artifact": linked.get("artifact"),
        }
        intervals = linked.get("intervals")
        week = intervals.get("week") if isinstance(intervals, dict) else None
        if isinstance(week, dict):
            linked_result["lower"] = float(week["lower"])
            linked_result["upper"] = float(week["upper"])
        return linked_result

    feature_profile = metadata.get("feature_profile")
    regressor = metadata.get("regressor")
    ats_method = str(metadata.get("ats_method", "market_residual"))
    if feature_profile is None:
        return None
    for directory in artifact_directories(artifacts_root / "margins", "summary.csv"):
        candidate_path = directory / "metadata.json"
        if not candidate_path.is_file():
            continue
        candidate = read_json(candidate_path)
        if candidate.get("feature_profile") != feature_profile:
            continue
        if regressor is not None and candidate.get("regressor") != regressor:
            continue
        summary = pd.read_csv(directory / "summary.csv")
        rows = summary.loc[summary["method"].eq(ats_method)]
        if rows.empty or pd.isna(rows.iloc[0].get("cover_accuracy")):
            continue
        row = rows.iloc[0]
        result: dict[str, Any] = {
            "accuracy": float(row["cover_accuracy"]),
            "games": int(row["cover_games"]),
            "artifact": directory.name,
        }
        uncertainty_path = directory / "uncertainty.csv"
        if uncertainty_path.is_file():
            uncertainty = pd.read_csv(uncertainty_path)
            interval = uncertainty.loc[
                uncertainty["method"].eq(ats_method)
                & uncertainty["metric"].eq("cover_accuracy")
                & uncertainty["block"].eq("week")
            ]
            if not interval.empty:
                result["lower"] = float(interval.iloc[0]["lower"])
                result["upper"] = float(interval.iloc[0]["upper"])
        return result
    return None


def _prospective_summary(artifacts_root: Path) -> None:
    frozen = artifact_directories(artifacts_root / "prospective", "manifest.json")
    if not frozen:
        st.info(
            "No forecasts are frozen. That is intentional while the model and inputs are still "
            "being developed. A prospective evaluation window should be declared explicitly "
            "before immutable records begin."
        )
        return
    rows = []
    for directory in frozen:
        try:
            manifest = verify_frozen_forecast(directory)
            verification = "Verified"
        except ValueError as error:
            st.error(str(error))
            manifest = read_json(directory / "manifest.json")
            verification = "FAILED"
        model = manifest.get("model", {})
        safety = manifest.get("prediction_safety", {})
        rows.append(
            {
                "Forecast": manifest.get("forecast_id"),
                "Model": model.get("model_name") if isinstance(model, dict) else None,
                "Features": model.get("feature_set") if isinstance(model, dict) else None,
                "Frozen at": manifest.get("frozen_at_utc"),
                "Season": manifest.get("season"),
                "Week": manifest.get("week"),
                "Games": manifest.get("games"),
                "First kickoff": manifest.get("earliest_kickoff_utc"),
                "Safety": safety.get("status", "Legacy / uncertified")
                if isinstance(safety, dict)
                else "Invalid",
                "Integrity": verification,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Frozen records are retained as separate directories and are never overwritten.")


def _weekly_forecast(artifacts_root: Path) -> None:
    st.header("Upcoming ATS predictions")
    st.write(
        "This is the model's current estimate of which team will cover the displayed spread. "
        "Forecasts here are mutable research snapshots: rerunning the command as lines and "
        "inputs change creates an updated card."
    )
    directories = artifact_directories(artifacts_root / "margin_predictions", "predictions.csv")
    if not directories:
        directories = artifact_directories(artifacts_root / "predictions", "recommendations.csv")
    if not directories:
        st.warning(
            "No saved weekly forecasts. Run `nfl-ats margin-predict --season YEAR --week WEEK`."
        )
        with st.expander("Prospective forecast archive", expanded=True):
            _prospective_summary(artifacts_root)
        return
    active = load_active_ats_model(artifacts_root)
    active_forecast = (
        active_artifact_path(artifacts_root, active, "weekly_forecast")
        if active is not None
        else None
    )
    if active_forecast is not None:
        directories = sorted(
            directories,
            key=lambda path: path.resolve() != active_forecast.resolve(),
        )
    selected = st.selectbox(
        "Forecast version",
        directories,
        format_func=_forecast_label,
    )
    try:
        recommendations, metadata = _load_weekly_ats_forecast(selected)
    except ValueError as error:
        st.error(str(error))
        return
    if (
        active is not None
        and active_forecast is not None
        and selected.resolve() == active_forecast.resolve()
    ):
        st.success(
            f"Synchronization: PASS · active model `{active.get('model_id')}` · this forecast "
            "is linked to the active historical evaluation."
        )
    elif active_forecast is not None:
        st.warning("Archived forecast selected; its model may differ from the active evaluation.")
    season = metadata.get("season", recommendations.get("season", pd.Series(["?"])).iloc[0])
    week = metadata.get("week", recommendations.get("week", pd.Series(["?"])).iloc[0])
    last_game = pd.to_datetime(recommendations["gameday"], errors="coerce").max()
    first_game = pd.to_datetime(recommendations["gameday"], errors="coerce").min()
    today = datetime.now(UTC).date()
    if pd.notna(last_game) and last_game.date() < today:
        st.warning(
            f"**Saved historical card: {season} Week {week}.** Its games are already over. "
            "This is useful for inspecting output format, not as a current recommendation."
        )
    else:
        days_until = (first_game.date() - today).days if pd.notna(first_game) else None
        timing = f" The first game is {days_until} days away." if days_until is not None else ""
        st.warning(
            f"**Early, unfrozen estimate: {season} Week {week}.**{timing} Lines, injuries, and "
            "other inputs can change; regenerate the card closer to kickoff."
        )
    model_name = metadata.get(
        "model_name", recommendations.get("model_name", pd.Series(["Unknown"])).iloc[0]
    )
    feature_name = metadata.get("feature_profile", metadata.get("feature_set", "unknown"))
    training_max = metadata.get(
        "training_max_gameday",
        recommendations.get("train_max_gameday", pd.Series(["?"])).max(),
    )
    ats_method = str(metadata.get("ats_method", "direct_ats"))
    st.caption(
        f"{model_name} model · {feature_name} features · ATS method `{ats_method}` · "
        f"trained through {training_max} · created "
        f"{metadata.get('created_at_utc', _artifact_time(selected))}"
    )
    safety = metadata.get("prediction_safety")
    if not isinstance(safety, dict):
        st.warning(
            "Legacy card: it predates the independent prediction-safety audit. Regenerate it "
            "before treating it as a current forecast."
        )
    elif safety.get("status") == "PASS":
        st.success("Prediction safety: PASS. Schema, market math, decisions, and cutoff verified.")
    elif safety.get("status") == "PASS_WITH_WARNINGS":
        warnings = safety.get("warnings", [])
        details = "; ".join(str(value) for value in warnings)
        st.warning(f"Prediction safety: PASS WITH WARNINGS. {details}")
    else:
        st.error("Prediction safety certification is invalid; do not use this card.")

    readable = _readable_recommendations(recommendations)
    strongest = float(readable["Estimated cover chance"].max())
    separated = int(readable["Estimated cover chance"].ge(0.525).sum())
    benchmark = _matching_historical_ats_benchmark(artifacts_root, metadata)
    if benchmark is not None:
        interval = ""
        if "lower" in benchmark and "upper" in benchmark:
            interval = (
                f"; week-blocked 95% interval {benchmark['lower']:.2%}-"
                f"{benchmark['upper']:.2%}"
            )
        st.info(
            f"**Model identity check:** this weekly card uses `{ats_method}` with "
            f"`{feature_name}` features. The matching historical walk-forward result was "
            f"**{benchmark['accuracy']:.2%} ATS accuracy across {benchmark['games']:,} games**"
            f"{interval}. The percentages below are estimates for individual upcoming games, "
            "not the historical hit rate."
        )
    if strongest < 0.51:
        st.error(
            "**Bottom line: the model currently has no meaningful separation.** Every forced "
            "pick is below 51%, so read the teams below as coin-flip leans, not confident calls."
        )
    elif strongest < 0.525:
        st.warning(
            "**Bottom line: only weak leans are present.** No game reaches a 52.5% estimated "
            "cover probability."
        )

    c1, c2, c3, c4 = st.columns(4)
    suggested = int(recommendations["bet_side"].ne("PASS").sum())
    with c1:
        _metric("Games on card", len(recommendations))
    with c2:
        _metric("Strongest forced pick", strongest, percent=True)
    with c3:
        _metric("Picks at least 52.5%", separated)
    with c4:
        if benchmark is None:
            _metric("Optional paper picks", suggested)
        else:
            _metric("Historical ATS accuracy", f"{benchmark['accuracy']:.2%}")

    tabs = st.tabs(["Predictions", "Confidence ranking", "Why the model leans"])
    with tabs[0]:
        st.subheader(f"{season} Week {week}: predicted ATS winners")
        st.write(
            "The ATS prediction is the forced side the model estimates is more likely to cover. "
            "The separate paper action remains PASS unless the estimated edge clears the "
            "configured threshold."
        )
        st.dataframe(
            readable,
            hide_index=True,
            width="stretch",
            height=630,
            column_config={
                "gameday": "Game date",
                "Estimated cover chance": st.column_config.ProgressColumn(
                    "Estimated cover chance", min_value=0.5, max_value=1.0, format="percent"
                ),
                "Separation from coin flip": st.column_config.NumberColumn(
                    "Above 50%", format="percent"
                ),
            },
        )
        st.download_button(
            "Download this card as CSV",
            data=readable.to_csv(index=False),
            file_name=f"nfl-ats-{season}-week-{week}.csv",
            mime="text/csv",
        )
        st.caption(
            "A forced prediction is always shown. Its estimated cover chance is a game-specific "
            "model output; it is not the model's historical classification accuracy."
        )

    with tabs[1]:
        pool_path = selected / "pool_card.csv"
        pool = (
            pd.read_csv(pool_path)
            if pool_path.is_file()
            else build_ats_pool_card(recommendations)
        )
        if not pool.empty:
            display = pool.copy()
            display["Matchup"] = display["away_team"] + " at " + display["home_team"]
            display["Pick"] = display["pool_pick"]
            display["Confidence"] = display["pick_probability"]
            display["Rank"] = display["confidence_rank"]
            st.write(
                "Pools often require a pick in every game, so this card cannot PASS. Rank 1 is "
                "the model's strongest lean; it is not necessarily a profitable wager."
            )
            st.dataframe(
                display[["Rank", "Matchup", "Pick", "pick_line", "Confidence"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "pick_line": "Pick spread",
                    "Confidence": st.column_config.ProgressColumn(
                        "Model confidence", min_value=0.5, max_value=1.0, format="percent"
                    ),
                },
            )

    with tabs[2]:
        coefficients = selected / "coefficients.csv"
        if not coefficients.is_file():
            st.info("Standardized coefficients are available only for logistic models.")
        else:
            table = pd.read_csv(coefficients)
            st.write(
                "A positive coefficient pushes the estimate toward the home team covering; a "
                "negative coefficient pushes it toward the away team. Magnitude reflects this "
                "fitted model, not causation or standalone predictive value."
            )
            st.bar_chart(table.head(25), x="feature", y="coefficient", height=420)
            st.dataframe(table, hide_index=True, width="stretch")
            st.warning(
                "Correlated inputs can split or reverse coefficients. Use the Feature lab's "
                "walk-forward comparisons—not coefficient size—to judge whether a family helps."
            )

    with st.expander("Prospective archive (not active)", expanded=False):
        _prospective_summary(artifacts_root)


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
                "games_in_state": game[f"{side}_team_games"],
                "off_epa": game[f"{side}_off_epa_per_play"],
                "pass_epa": game[f"{side}_off_pass_epa_per_play"],
                "rush_epa": game[f"{side}_off_rush_epa_per_play"],
                "yards_per_play": game[f"{side}_off_yards_per_play"],
                "turnover_rate": game[f"{side}_off_turnover_rate"],
                "def_epa_allowed": game[f"{side}_def_epa_per_play"],
                "point_diff_state": game[f"{side}_point_diff"],
                "ats_state": game[f"{side}_ats_residual"],
            }
        )
    return pd.DataFrame(rows).sort_values("week") if rows else pd.DataFrame()


def _matchup_comparison(game: pd.Series) -> pd.DataFrame:
    measures = (
        ("Games informing state", "team_games", "More means less early-season uncertainty"),
        ("Offensive EPA per play", "off_epa_per_play", "Higher is better"),
        ("Passing EPA per play", "off_pass_epa_per_play", "Higher is better"),
        ("Rushing EPA per play", "off_rush_epa_per_play", "Higher is better"),
        ("Offensive yards per play", "off_yards_per_play", "Higher is better"),
        ("Offensive turnover rate", "off_turnover_rate", "Lower is better"),
        ("Defensive EPA allowed/play", "def_epa_per_play", "Lower is better"),
        ("Recent point differential", "point_diff", "Higher is better"),
        ("Recent ATS residual", "ats_residual", "Higher is better"),
    )
    return pd.DataFrame(
        [
            {
                "Measure": label,
                str(game["away_team"]): game[f"away_{suffix}"],
                str(game["home_team"]): game[f"home_{suffix}"],
                "Direction": direction,
            }
            for label, suffix, direction in measures
        ]
    )


def _team_explorer(data_root: Path) -> None:
    st.header("Team explorer")
    st.write(
        "Inspect exactly what the model knew about each team before a game. These are smoothed "
        "pregame states—not the box score from the game on that row."
    )
    feature_path = data_root / "processed" / "game_features.parquet"
    if not feature_path.is_file():
        st.warning("No feature table. Run `nfl-ats build-features`.")
        return
    features = pd.read_parquet(feature_path)
    seasons = sorted(features["season"].dropna().astype(int).unique(), reverse=True)
    c1, c2 = st.columns(2)
    with c1:
        season = st.selectbox("Season", seasons)
    season_games = features.loc[features["season"].eq(season)]
    teams = sorted(set(season_games["home_team"]).union(season_games["away_team"]))
    with c2:
        team = st.selectbox("Team", teams)
    trends = _team_game_rows(features, team, int(season))
    if trends.empty:
        st.info("No games were found for that team and season.")
        return

    st.subheader(f"{team} pregame trend")
    st.caption(
        "The default state uses an eight-game exponential span. It retains 67% of its "
        "league-relative value through the offseason and then updates after each game."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Offensive EPA per play — higher is better")
        st.line_chart(trends, x="week", y=["off_epa", "pass_epa", "rush_epa"], height=290)
    with c2:
        st.caption("Defensive EPA allowed per play — lower is better")
        st.line_chart(trends, x="week", y="def_epa_allowed", height=290)
    st.dataframe(
        trends,
        hide_index=True,
        width="stretch",
        column_config={
            "week": "Week",
            "gameday": "Game date",
            "opponent": "Opponent",
            "venue": "Venue",
            "games_in_state": "Current-season games",
            "off_epa": st.column_config.NumberColumn("Off EPA/play", format="%.3f"),
            "pass_epa": st.column_config.NumberColumn("Pass EPA/play", format="%.3f"),
            "rush_epa": st.column_config.NumberColumn("Rush EPA/play", format="%.3f"),
            "yards_per_play": st.column_config.NumberColumn("Yards/play", format="%.2f"),
            "turnover_rate": st.column_config.NumberColumn("Turnover rate", format="percent"),
            "def_epa_allowed": st.column_config.NumberColumn("Def EPA allowed", format="%.3f"),
            "point_diff_state": st.column_config.NumberColumn("Point-diff state", format="%.2f"),
            "ats_state": st.column_config.NumberColumn("ATS state", format="%.2f"),
        },
    )

    st.subheader("Pregame matchup comparison")
    team_games = season_games.loc[
        season_games["home_team"].eq(team) | season_games["away_team"].eq(team)
    ].sort_values(["week", "gameday"])
    game_options = team_games["game_id"].astype(str).tolist()
    game_labels = {
        str(row["game_id"]): (f"Week {int(row['week'])}: {row['away_team']} at {row['home_team']}")
        for _, row in team_games.iterrows()
    }
    selected_game_id = st.selectbox(
        "Matchup",
        game_options,
        format_func=lambda game_id: game_labels[str(game_id)],
    )
    game = team_games.loc[team_games["game_id"].astype(str).eq(selected_game_id)].iloc[0]
    st.markdown(
        f"**Market:** {_favorite_label(game)} · **Total:** "
        f"{_render_value(game.get('total_line'), digits=1)} · **Rest advantage (home):** "
        f"{_render_value(game.get('rest_diff'), digits=0)} days"
    )
    st.dataframe(_matchup_comparison(game), hide_index=True, width="stretch")
    st.caption(
        "Week 1 values are regressed priors carried from the previous season; Week 2 combines "
        "that prior with Week 1 rather than relying on one game alone."
    )


def _data_health(data_root: Path, artifacts_root: Path | None = None) -> None:
    st.header("Is the data and system healthy?")
    st.write(
        "**Purpose:** verify that the active forecast is linked to its evaluation, passed the "
        "prediction-safety checks, and came from identifiable source data. This page diagnoses "
        "pipeline integrity; it does not judge whether the model is predictive."
    )
    if artifacts_root is not None:
        active = load_active_ats_model(artifacts_root)
        forecast = (
            active_artifact_path(artifacts_root, active, "weekly_forecast")
            if active is not None
            else None
        )
        if active is None or forecast is None:
            st.warning("No synchronized active evaluation/forecast pair is available.")
        else:
            metadata_path = forecast / "metadata.json"
            metadata = read_json(metadata_path) if metadata_path.is_file() else {}
            safety = metadata.get("prediction_safety")
            safety_status = safety.get("status") if isinstance(safety, dict) else "MISSING"
            if safety_status in {"PASS", "PASS_WITH_WARNINGS"}:
                st.success(
                    f"**Active-model synchronization: PASS.** Model "
                    f"`{active.get('model_id')}` links one evaluation and weekly forecast. "
                    f"Prediction safety: **{safety_status}**."
                )
            else:
                st.error(
                    f"Prediction safety is not valid for the active forecast: {safety_status}"
                )
    snapshot, feature_manifest = _data_summary(data_root)
    if snapshot is None:
        st.warning("No complete raw snapshot. Run `nfl-ats ingest`.")
    else:
        manifest = snapshot["manifest"]
        st.success(f"Raw snapshot {snapshot['snapshot_id']} is complete.")
        c1, c2, c3 = st.columns(3)
        with c1:
            _metric("Schedule rows", manifest["files"]["schedules.parquet"]["rows"])
        with c2:
            _metric("Team-stat rows", manifest["files"]["team_stats.parquet"]["rows"])
        with c3:
            _metric("Seasons requested", len(manifest.get("seasons", [])))
        st.caption(f"Fetched {manifest['fetched_at_utc']} from nflverse-backed datasets.")
        with st.expander("Raw snapshot provenance"):
            st.json(snapshot, expanded=False)

    feature_path = data_root / "processed" / "game_features.parquet"
    if not feature_path.is_file():
        st.warning("No processed feature table. Run `nfl-ats build-features`.")
        return
    features = pd.read_parquet(feature_path)
    st.subheader("Processed research table")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Games", len(features))
    with c2:
        _metric("Seasons", features["season"].nunique())
    with c3:
        _metric("Named model inputs", len(MODEL_FEATURE_COLUMNS))
    with c4:
        _metric("Completed labels", int(features["home_cover"].notna().sum()))
    if feature_manifest is not None:
        st.caption(
            f"Built {feature_manifest.get('built_at_utc', '?')} from raw snapshot "
            f"{feature_manifest.get('source_snapshot', '?')} · EWM span "
            f"{feature_manifest.get('ewm_span', '?')} · offseason retention "
            f"{float(feature_manifest.get('offseason_retention', 0.0)):.0%}."
        )

    pbp_manifest_path = data_root / "processed" / "game_features_pbp.manifest.json"
    if pbp_manifest_path.is_file():
        pbp_manifest = read_json(pbp_manifest_path)
        st.success(
            f"Play-by-play enrichment is available: {pbp_manifest.get('pbp_rows', 0):,} "
            f"plays from snapshot {pbp_manifest.get('source_pbp_snapshot', '?')}."
        )

    market_directories = sorted(
        (data_root / "market" / "historical" / "raw").glob("*/manifest.json"),
        reverse=True,
    )
    if market_directories:
        market_manifest = read_json(market_directories[0])
        audit = market_manifest.get("audit") or {}
        source = market_manifest.get("source") or {}
        st.subheader("Historical closing-line cross-check")
        st.success(
            f"Public archive version {source.get('version', '?')} is preserved with "
            f"{market_manifest.get('rows', {}).get('normalized_market', 0):,} games."
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _metric("Matched nflverse games", audit.get("matched_nflverse_games"))
        with m2:
            _metric("Exact spread matches", audit.get("exact_agreement_rate"), percent=True)
        with m3:
            _metric("Within half a point", audit.get("within_half_point_rate"), percent=True)
        with m4:
            _metric("Mean absolute difference", audit.get("mean_absolute_difference"), digits=2)
        st.caption(
            "This source reports one closing line per game and has no quote timestamp or "
            "sportsbook identifier. Differences are retained rather than silently overwritten."
        )

    open_close_manifests = sorted(
        (data_root / "market" / "historical" / "open_close" / "raw").glob("*/manifest.json"),
        reverse=True,
    )
    if open_close_manifests:
        open_close = read_json(open_close_manifests[0])
        audit = open_close.get("audit") or {}
        st.subheader("2025 opening and closing market sample")
        st.success(
            f"The free multi-book sample is preserved with "
            f"{audit.get('games', 0):,} games and {audit.get('quote_rows', 0):,} quote rows."
        )
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            _metric("Regular-season matches", audit.get("matched_nflverse_games"))
        with o2:
            _metric("Closing books", audit.get("closing_books"))
        with o3:
            _metric("Spreads that moved", audit.get("spread_changed_rate"), percent=True)
        with o4:
            _metric(
                "Mean absolute movement",
                audit.get("mean_absolute_open_to_close_movement"),
                digits=2,
            )
        st.caption(
            "This one-season source labels market openers and book closes but does not provide "
            "their timestamps. It supports open-to-close research, not arbitrary decision-time "
            f"backtests. The movement audit excludes {audit.get('inconsistent_opener_games', 0)} "
            "opener rows and "
            f"{audit.get('inconsistent_closing_book_games', 0)} book/game rows whose spread "
            "favorite contradicts the same source's moneyline; raw rows remain preserved."
        )

    counts = features.groupby("season", as_index=False).agg(
        games=("game_id", "size"), completed=("home_cover", "count")
    )
    st.subheader("Season coverage")
    st.bar_chart(counts, x="season", y=["games", "completed"], height=280)

    coverage = feature_coverage(features, MODEL_FEATURE_COLUMNS)
    low_coverage = coverage.loc[coverage["coverage"].lt(0.95)]
    st.subheader("Missing inputs")
    if low_coverage.empty:
        st.success("Every named input is available for at least 95% of game rows.")
    else:
        st.warning(
            f"{len(low_coverage)} of {len(coverage)} inputs are present for less than 95% of rows. "
            "The fitted pipeline imputes missing values using its historical training window."
        )
    st.dataframe(
        coverage,
        hide_index=True,
        width="stretch",
        column_config={
            "feature": "Input",
            "available": "Available rows",
            "missing": "Missing rows",
            "coverage": st.column_config.ProgressColumn("Coverage", format="percent"),
        },
    )

    st.subheader("What the input families mean")
    family_rows = [
        {
            "Family": FAMILY_LABELS[family],
            "Inputs": len(columns),
            "Meaning": FAMILY_DESCRIPTIONS[family],
        }
        for family, columns in FEATURE_FAMILIES.items()
    ]
    st.dataframe(pd.DataFrame(family_rows), hide_index=True, width="stretch")
    with st.expander("Exact columns in each family"):
        rows = [
            {
                "Family": FAMILY_LABELS[family],
                "Columns": ", ".join(columns),
            }
            for family, columns in FEATURE_FAMILIES.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _model_performance(artifacts_root: Path) -> None:
    st.header("Does the model actually work?")
    st.write(
        "**Purpose:** answer one question—has the active model classified ATS winners more "
        "accurately than a coin flip on games it had not trained on? This is where the 52.05% "
        "number comes from."
    )
    active = load_active_ats_model(artifacts_root)
    if active is None:
        st.warning("No synchronized active model is available yet.")
        return
    _show_active_ats_summary(active)
    historical = active.get("historical_evaluation", {})
    if not isinstance(historical, dict):
        st.error("The active historical evaluation is malformed.")
        return
    correct = int(historical["correct"])
    games = int(historical["games"])
    excess_correct = correct - games / 2.0
    intervals = historical.get("intervals", {})
    week = intervals.get("week") if isinstance(intervals, dict) else None
    season = intervals.get("season") if isinstance(intervals, dict) else None
    week_supports = isinstance(week, dict) and float(week["lower"]) > 0.5
    season_supports = isinstance(season, dict) and float(season["lower"]) > 0.5
    if week_supports and season_supports:
        st.success(
            "**Current interpretation:** the observed classification advantage remains above "
            "50% under both week- and season-blocked uncertainty checks."
        )
    elif week_supports or season_supports:
        supported_by = "week" if week_supports else "season"
        st.info(
            f"**Current interpretation:** meaningful and promising, but not fully resolved. "
            f"The {supported_by}-blocked interval stays above 50%; the other blocking choice "
            "still includes a coin flip."
        )
    else:
        st.info(
            "**Current interpretation:** the historical result is above 50%, but both blocked "
            "intervals still include a coin flip."
        )
    st.write(
        f"The model made **{correct:,} correct classifications**, approximately "
        f"**{excess_correct:.1f} more than the {games / 2:,.1f} expected from random picks**. "
        "The uncertainty checks ask whether that difference could plausibly be sampling noise."
    )

    evaluation = active_artifact_path(artifacts_root, active, "historical_evaluation")
    if evaluation is None:
        return
    season_path = evaluation / "season_summary.csv"
    if season_path.is_file():
        season_summary = pd.read_csv(season_path)
        method = str(active.get("method", "market_residual"))
        seasons = season_summary.loc[
            season_summary["method"].eq(method)
            & season_summary["cover_accuracy"].notna(),
            ["season", "cover_games", "cover_accuracy"],
        ].copy()
        if not seasons.empty:
            st.subheader("Did it repeat each season?")
            above = int(seasons["cover_accuracy"].gt(0.5).sum())
            st.write(
                f"The active classifier finished above 50% in **{above} of {len(seasons)} "
                "seasons**. The chart makes instability visible instead of hiding it inside "
                "the pooled 52.05%."
            )
            chart = seasons[["season", "cover_accuracy"]].copy()
            chart["Coin flip"] = 0.5
            st.line_chart(chart, x="season", y=["cover_accuracy", "Coin flip"], height=320)
            st.dataframe(
                seasons.rename(
                    columns={
                        "season": "Season",
                        "cover_games": "Non-push games",
                        "cover_accuracy": "ATS accuracy",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "ATS accuracy": st.column_config.NumberColumn(format="percent"),
                },
            )
    st.caption(
        "This page evaluates classification performance. It does not tell you this week's "
        "picks, explain an individual matchup, or claim that the probabilities are calibrated."
    )


def _matchup_signal_table(game: pd.Series) -> pd.DataFrame:
    signals = (
        ("Pregame Elo difference", "elo_diff", True),
        ("Expected QB EPA/dropback difference", "diff_qb_expected_epa_per_dropback", True),
        ("Recent offensive EPA/play difference", "diff_off_epa_per_play", True),
        ("Defensive EPA allowed difference", "diff_def_epa_per_play", False),
        ("Offensive injury unavailability difference", "diff_injury_offense_unavailability", False),
        ("Defensive injury unavailability difference", "diff_injury_defense_unavailability", False),
        ("Offensive-line continuity difference", "diff_offensive_line_continuity", True),
        ("Secondary continuity difference", "diff_secondary_lineup_continuity", True),
        ("Active-roster continuity difference", "diff_active_roster_continuity", True),
    )
    rows = []
    home = str(game["home_team"])
    away = str(game["away_team"])
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


def _why_these_picks(artifacts_root: Path) -> None:
    st.header("Why did the model make this pick?")
    st.write(
        "**Purpose:** inspect one upcoming matchup at a time. This page shows the market line, "
        "the active model's ATS side, its proposed correction to the market, and the most "
        "recognizable pregame differences supplied to the model."
    )
    active = load_active_ats_model(artifacts_root)
    if active is None:
        st.warning("No synchronized active model is available yet.")
        return
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        st.warning("The active model has no linked weekly forecast.")
        return
    recommendations, metadata = _load_weekly_ats_forecast(forecast)
    confidence = recommendations["home_cover_probability"].where(
        recommendations["home_cover_probability"].ge(0.5),
        1.0 - recommendations["home_cover_probability"],
    )
    order = pd.DataFrame(
        {"confidence": confidence, "game_id": recommendations["game_id"].astype(str)}
    ).sort_values(["confidence", "game_id"], ascending=[False, True])
    recommendations = recommendations.loc[order.index]
    game_ids = recommendations["game_id"].astype(str).tolist()
    labels = {
        str(row["game_id"]): f"{row['away_team']} at {row['home_team']}"
        for _, row in recommendations.iterrows()
    }
    selected_game = st.selectbox(
        "Upcoming matchup",
        game_ids,
        format_func=lambda game_id: labels[str(game_id)],
    )
    game = recommendations.loc[recommendations["game_id"].astype(str).eq(selected_game)].iloc[0]
    home_probability = float(game["home_cover_probability"])
    home_pick = home_probability >= 0.5
    pick = str(game["home_team"] if home_pick else game["away_team"])
    pick_line = -float(game["spread_line"]) if home_pick else float(game["spread_line"])
    pick_line_label = "PK" if pick_line == 0 else f"{pick_line:+g}"
    pick_probability = max(home_probability, 1.0 - home_probability)
    historical = active["historical_evaluation"]
    st.success(
        f"**ATS pick: {pick} {pick_line_label}.** The current game-specific model estimate is "
        f"{pick_probability:.1%}; the same model's historical walk-forward accuracy is "
        f"{float(historical['accuracy']):.2%}."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Market line", _favorite_label(game))
    with c2:
        _metric("Model ATS pick", f"{pick} {pick_line_label}")
    with c3:
        _metric("Game-specific estimate", pick_probability, percent=True)
    with c4:
        _metric("Historical model accuracy", f"{float(historical['accuracy']):.2%}")

    residual = float(game.get("predicted_market_residual", np.nan))
    if np.isfinite(residual):
        correction_team = str(game["home_team"] if residual > 0 else game["away_team"])
        st.subheader("The model's disagreement with the market")
        st.write(
            f"The model expects **{correction_team} to outperform the market baseline by "
            f"about {abs(residual):.1f} points**. That predicted correction is what determines "
            "the ATS side."
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
            column_config={
                "Home minus away": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.caption(
        f"Active model `{active.get('model_id')}` · `{metadata.get('ats_method')}` method. "
        "These rows describe model inputs, not isolated causal effects. Exact per-feature "
        "prediction contributions are a separate explanation artifact still to build."
    )


def _advanced_research(data_root: Path, artifacts_root: Path) -> None:
    st.header("Advanced research")
    st.write(
        "**Purpose:** inspect how candidate models were selected, which feature ideas failed or "
        "helped, and older archived tests. You do not need this section to read the weekly picks."
    )
    tools = {
        "Model-selection validation": (
            "Checks whether a candidate was chosen using earlier seasons before testing it.",
            lambda: _validation_lab(artifacts_root),
        ),
        "Feature experiments": (
            "Compares information families on identical historical games.",
            lambda: _feature_lab(artifacts_root),
        ),
        "Winner and margin diagnostics": (
            "Separates straight-up winners, predicted margins, and ATS classification.",
            lambda: _outcome_lab(artifacts_root),
        ),
        "Archived direct-model test": (
            "Retains the older direct ATS classifier and its diagnostics.",
            lambda: _historical_test(artifacts_root),
        ),
        "Historical team states": (
            "Shows what the older team-state table contained before each game.",
            lambda: _team_explorer(data_root),
        ),
    }
    selected = st.selectbox("Research tool", tuple(tools))
    description, renderer = tools[selected]
    st.info(description)
    st.divider()
    renderer()


def _sidebar_guide(data_root: Path, artifacts_root: Path) -> None:
    st.sidebar.markdown(
        "**Predictions** = the upcoming card  \n"
        "**Does it work?** = historical accuracy  \n"
        "**Why?** = one matchup's inputs  \n"
        "**Health** = sync, safety, and data  \n"
        "**Advanced** = researcher diagnostics"
    )
    with st.sidebar.expander("Technical paths", expanded=False):
        st.caption(f"Data: {data_root.resolve()}")
        st.caption(f"Artifacts: {artifacts_root.resolve()}")
    with st.sidebar.expander("Research terms", expanded=False):
        st.markdown(
            "- **Accuracy:** 50% is the simple ATS reference.\n"
            "- **Blocked interval:** uncertainty that keeps weeks or seasons together.\n"
            "- **Brier:** whether probability magnitudes are honest; lower is better.\n"
            "- **Calibration:** whether stated probabilities match observed rates."
        )
    st.sidebar.caption("Research project; no wagering integration.")


def render_dashboard() -> None:
    st.set_page_config(page_title="NFL ATS Research Lab", page_icon="🏈", layout="wide")
    st.title("NFL ATS Research Lab")
    st.caption("A guided workbench for honest NFL against-the-spread research")
    data_root = Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))
    artifacts_root = Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))
    page = st.sidebar.radio(
        "What do you want to know?",
        (
            "This week's predictions",
            "Does the model work?",
            "Why these picks?",
            "Data & system health",
            "Advanced research",
        ),
        index=0,
    )
    _sidebar_guide(data_root, artifacts_root)
    if page == "This week's predictions":
        _weekly_forecast(artifacts_root)
    elif page == "Does the model work?":
        _model_performance(artifacts_root)
    elif page == "Why these picks?":
        _why_these_picks(artifacts_root)
    elif page == "Data & system health":
        _data_health(data_root, artifacts_root)
    else:
        _advanced_research(data_root, artifacts_root)


if __name__ == "__main__":
    render_dashboard()
