"""This week's picks -- the home page.

Answers the one question the owner actually asks every week: who do I pick,
and how confident are we? One card per game, headline first, details behind
an expander.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.dashboard.data import (
    artifact_time,
    artifacts_root,
    data_root,
    detect_cover_push_loss,
    find_latest_close_predictions,
    find_line_sweep_file,
    game_opening_lines,
    latest_weekly_forecast,
    list_recent_market_snapshots,
    list_weekly_forecasts,
    load_close_predictions,
    load_line_sweep,
    load_live_quotes,
    load_named_weekly_forecast,
)
from nfl_ats.dashboard.ui import (
    FairLine,
    confidence_meter,
    favorite_label,
    format_kickoff,
    format_line_journey,
    implied_fair_line,
    line_sweep_ribbon,
    next_tuesday_capture_label,
    pick_side_and_line,
    render_confidence_ribbon,
    spread_label,
)
from nfl_ats.market_data import spread_consensus

st.title("This week's picks")

forecast = latest_weekly_forecast(artifacts_root())

if forecast is None:
    st.info(
        "**No pick card yet for this week.** Nothing has been generated since the last "
        "kickoff, which is expected outside of the regular capture cadence (for example, "
        "in the preseason).",
        icon=":material/schedule:",
    )
    st.write(
        f"The next Tuesday opening-line capture is expected **"
        f"{next_tuesday_capture_label()}**. Once it runs and a forecast card is generated "
        "(`nfl-ats margin-predict`), this page will fill in automatically."
    )
    st.page_link(
        "app_pages/track_record.py",
        label="See the model's track record while you wait",
        icon=":material/query_stats:",
    )
    st.stop()

recommendations = forecast.recommendations
metadata = forecast.metadata
season = metadata.get("season", recommendations.get("season", pd.Series(["?"])).iloc[0])
week = metadata.get("week", recommendations.get("week", pd.Series(["?"])).iloc[0])

st.caption(f"{season} · Week {week}")

if forecast.is_active:
    st.badge("Synchronized with the active model", icon=":material/verified:", color="green")
else:
    st.badge("Archived card (not the active model)", icon=":material/info:", color="gray")

# --- Timing banner ----------------------------------------------------------
if "kickoff" in recommendations.columns:
    kickoffs = pd.to_datetime(recommendations["kickoff"], errors="coerce", utc=True)
else:
    kickoffs = pd.Series(pd.NaT, index=recommendations.index, dtype="datetime64[ns, UTC]")
if kickoffs.notna().any():
    first_kickoff = kickoffs.min()
    last_kickoff = kickoffs.max()
else:
    first_kickoff = pd.to_datetime(recommendations["gameday"], errors="coerce").min()
    last_kickoff = pd.to_datetime(recommendations["gameday"], errors="coerce").max()
now = pd.Timestamp.now(tz="UTC")
all_final = pd.notna(last_kickoff) and last_kickoff < now
if all_final:
    st.caption("This week's games are complete; showing the pregame card for reference.")
elif pd.notna(first_kickoff):
    days_until = (first_kickoff.tz_convert("UTC").normalize() - now.normalize()).days
    when = "today" if days_until <= 0 else f"in {days_until} day{'s' if days_until != 1 else ''}"
    st.caption(
        f"Early estimate -- kickoff starts {when}. Lines and inputs can still change before "
        "kickoff; regenerate the card closer to game time for the freshest numbers."
    )

# --- Safety status -----------------------------------------------------------
safety = metadata.get("prediction_safety")
if not isinstance(safety, dict):
    st.warning("This is a legacy card that predates the safety audit; treat it as reference only.")
elif safety.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
    st.error("This card failed its safety checks. Do not treat it as a current forecast.")
elif safety.get("status") == "PASS_WITH_WARNINGS":
    warnings_text = "; ".join(str(item) for item in safety.get("warnings", []))
    st.warning(f"Passed with warnings: {warnings_text}")

# --- Headline summary ---------------------------------------------------------
pick_probabilities = recommendations["home_cover_probability"].map(
    lambda value: max(float(value), 1.0 - float(value))
)
strong_count = int((pick_probabilities >= 0.57).sum())
lean_count = int(((pick_probabilities >= 0.52) & (pick_probabilities < 0.57)).sum())
coinflip_count = int((pick_probabilities < 0.52).sum())

with st.container(horizontal=True):
    st.metric("Games this week", len(recommendations), border=True)
    st.metric("Strong leans", strong_count, border=True, help="Picked-side confidence at 57%+")
    st.metric("Leans", lean_count, border=True, help="Picked-side confidence 52-57%")
    st.metric("Coin flips", coinflip_count, border=True, help="Picked-side confidence under 52%")

historical = metadata.get("historical_evaluation")
if isinstance(historical, dict) and historical.get("accuracy") is not None:
    st.caption(
        f"This model's own long-run accuracy is about {float(historical['accuracy']):.1%} -- "
        "see “Is this thing good?” for the full picture before trusting any single pick."
    )

st.divider()

# --- One card per game --------------------------------------------------------
ordering = (
    kickoffs
    if kickoffs.notna().any()
    else pd.to_datetime(recommendations["gameday"], errors="coerce")
)
ordered = recommendations.iloc[ordering.to_numpy().argsort(kind="stable")]

split_columns = detect_cover_push_loss(recommendations)
sweep_path = find_line_sweep_file(forecast.directory)
sweep_frame = load_line_sweep(sweep_path) if sweep_path is not None else pd.DataFrame()
if "method" in sweep_frame.columns:
    # score_outcome_week_line_sweep() (nfl-ats margin-predict --line-sweep)
    # stacks every margin-distribution method's sweep into one file with no
    # per-game uniqueness otherwise -- keep only the method backing this
    # card's picks, or every game_id would carry 2-3x duplicate rows.
    sweep_frame = sweep_frame.loc[
        sweep_frame["method"].eq(metadata.get("ats_method", "market_residual"))
    ]
sweep_game_key = "game_id" if "game_id" in sweep_frame.columns else None

# --- Line journey inputs: live quotes + predicted close (both optional) -----
snapshots = list_recent_market_snapshots(data_root() / "market" / "raw", since_days=10)
quotes = (
    load_live_quotes(tuple(info.directory for info in snapshots)) if snapshots else pd.DataFrame()
)
opener_table = game_opening_lines(quotes) if not quotes.empty else pd.DataFrame()
opener_by_game = (
    opener_table.set_index("nflverse_game_id")["opener_home_spread"].to_dict()
    if not opener_table.empty
    else {}
)
latest_table = spread_consensus(quotes) if not quotes.empty else pd.DataFrame()
latest_by_game = (
    latest_table.set_index("nflverse_game_id")["consensus_home_spread"].to_dict()
    if not latest_table.empty
    else {}
)

close_predictions_path = find_latest_close_predictions(artifacts_root())
close_predictions = (
    load_close_predictions(close_predictions_path)
    if close_predictions_path is not None
    else pd.DataFrame()
)
predicted_close_by_game = (
    close_predictions.set_index("game_id")["predicted_close_home_spread"].to_dict()
    if not close_predictions.empty
    and {"game_id", "predicted_close_home_spread"}.issubset(close_predictions.columns)
    else {}
)
if close_predictions_path is None:
    st.caption(
        "Predicted close isn't wired up yet -- the MKT-06 pilot model trains after the "
        "2020-2022 line archive re-fetch (mid-September). Every card shows an em dash "
        "there until then."
    )
if not opener_by_game:
    st.caption(
        f"No live line captures yet this week -- first capture expected "
        f"{next_tuesday_capture_label()}."
    )

cards_per_row = 2
rows = [
    ordered.iloc[index : index + cards_per_row] for index in range(0, len(ordered), cards_per_row)
]

for row_frame in rows:
    columns = st.columns(cards_per_row)
    for column, (_, game) in zip(columns, row_frame.iterrows(), strict=False):
        with column, st.container(border=True):
            st.markdown(f"**{game['away_team']} @ {game['home_team']}**")
            st.caption(format_kickoff(game))

            team, line, pick_probability = pick_side_and_line(game)
            st.markdown(f"### Pick: {team} {spread_label(line)}")
            st.caption(f"Line used: {favorite_label(game)}")

            ats_margin = game.get("ats_margin")
            if pd.notna(ats_margin):
                if float(ats_margin) == 0.0:
                    st.badge("Push", color="gray", icon=":material/remove:")
                else:
                    home_covered = bool(game["home_cover"])
                    picked_home = team == game["home_team"]
                    won = home_covered if picked_home else not home_covered
                    if won:
                        st.badge("Final: covered", color="green", icon=":material/check_circle:")
                    else:
                        st.badge("Final: missed", color="red", icon=":material/cancel:")

            game_sweep = (
                sweep_frame.loc[sweep_frame[sweep_game_key].eq(game["game_id"])]
                if sweep_game_key is not None
                else pd.DataFrame()
            )
            has_sweep = not game_sweep.empty

            if has_sweep:
                pick_ribbon = line_sweep_ribbon(game, game_sweep, perspective="pick")
                render_confidence_ribbon(pick_ribbon)
                fair = implied_fair_line(pick_ribbon)
                if fair.value is not None:
                    gap = fair.value - line
                    direction = "tougher" if gap < 0 else "more generous"
                    st.caption(
                        f"Model fair line for {team}: {fair.label} (market {spread_label(line)}, "
                        f"{abs(gap):.1f} pt {direction} than market)"
                    )
                else:
                    st.caption(
                        f"Model fair line for {team}: {fair.label} (beyond the swept window; "
                        f"market {spread_label(line)})"
                    )
            else:
                confidence_meter(pick_probability)

            home_fair = (
                implied_fair_line(line_sweep_ribbon(game, game_sweep, perspective="home"))
                if has_sweep
                else FairLine(None, "—")
            )
            st.caption(
                format_line_journey(
                    opener_by_game.get(game["game_id"]),
                    latest_by_game.get(game["game_id"]),
                    predicted_close_by_game.get(game["game_id"]),
                    home_fair,
                )
            )

            residual = game.get("predicted_market_residual")
            if residual is not None and pd.notna(residual):
                correction_team = game["home_team"] if float(residual) > 0 else game["away_team"]
                st.caption(
                    f"Model leans {abs(float(residual)):.1f} pt toward {correction_team} "
                    "versus the market."
                )

            with st.expander("Details"):
                detail_rows = {
                    "Market spread": favorite_label(game),
                    "Total": (
                        f"{float(game['total_line']):g}"
                        if pd.notna(game.get("total_line"))
                        else "—"
                    ),
                    "Model estimate for this side": f"{pick_probability:.1%}",
                }
                edge = game.get("edge")
                bet_side = game.get("bet_side")
                if edge is not None and pd.notna(edge) and bet_side is not None:
                    detail_rows["Paper action"] = (
                        "PASS" if bet_side == "PASS" else f"Paper pick: {bet_side}"
                    )
                st.table(pd.DataFrame(detail_rows.items(), columns=["", " "]).set_index(""))

                if split_columns:
                    st.caption("Cover / push / loss split (newer schema, when available):")
                    split_values = {
                        kind.capitalize(): game.get(column)
                        for kind, column in split_columns.items()
                        if pd.notna(game.get(column))
                    }
                    if split_values:
                        st.bar_chart(pd.Series(split_values), height=180)

                if has_sweep:
                    st.caption("Full line sweep (home-oriented, cover / push / loss):")
                    detail_table = (
                        game_sweep.sort_values("alternative_line")[
                            [
                                "alternative_line",
                                "home_cover_probability",
                                "push_probability",
                                "home_loss_probability",
                            ]
                        ]
                        .rename(
                            columns={
                                "alternative_line": "Home spread",
                                "home_cover_probability": "Home cover",
                                "push_probability": "Push",
                                "home_loss_probability": "Home loss",
                            }
                        )
                        .reset_index(drop=True)
                    )
                    st.dataframe(
                        detail_table,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Home spread": st.column_config.NumberColumn(format="%.1f"),
                            "Home cover": st.column_config.NumberColumn(format="%.4f"),
                            "Push": st.column_config.NumberColumn(format="%.4f"),
                            "Home loss": st.column_config.NumberColumn(format="%.4f"),
                        },
                    )

# --- Browse other weeks -------------------------------------------------------
other_weeks = [
    path for path in list_weekly_forecasts(artifacts_root()) if path != forecast.directory
]
if other_weeks:
    with st.expander("Browse a different saved week"):
        chosen_path = st.selectbox(
            "Saved forecast card", other_weeks, format_func=artifact_time, key="home_other_week"
        )
        other = load_named_weekly_forecast(chosen_path)
        if other is None:
            st.info("That saved card could not be read.")
        else:
            preview = other.recommendations.copy()
            preview["Matchup"] = preview["away_team"] + " @ " + preview["home_team"]
            preview["Pick"] = preview.apply(
                lambda row: (
                    f"{pick_side_and_line(row)[0]} {spread_label(pick_side_and_line(row)[1])}"
                ),
                axis=1,
            )
            st.dataframe(
                preview[["Matchup", "Pick", "home_cover_probability"]].rename(
                    columns={"home_cover_probability": "Model home-cover probability"}
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Model home-cover probability": st.column_config.NumberColumn(format="percent")
                },
            )
