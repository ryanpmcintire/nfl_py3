"""This week's picks -- the home page.

Renders the owner-approved picks board: a compact header (week, sync status,
one-line honesty note), the tier-count chips, the legend (written once), then
one self-contained HTML board (:mod:`nfl_ats.dashboard.board`) with one row per
game -- kickoff, matchup, a tier-colored pick chip, confidence, edge vs. the
model's fair line, and a numberless confidence strip. Tap a row for the numbered
ribbon, a plain-English explanation, and the line journey.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.dashboard.board import render_picks_board, tier_counts
from nfl_ats.dashboard.data import (
    artifact_time,
    artifacts_root,
    data_root,
    explanations_by_game,
    find_latest_attribution_file,
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
    load_parquet_safe,
)
from nfl_ats.dashboard.ui import next_tuesday_capture_label, pick_side_and_line, spread_label
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

# --- Compact header: week + games + sync status ------------------------------
with st.container(horizontal=True, vertical_alignment="center"):
    st.caption(f"{season} · Week {week} · {len(recommendations)} games")
    if forecast.is_active:
        st.badge("Synchronized with the active model", icon=":material/verified:", color="green")
    else:
        st.badge("Archived card (not the active model)", icon=":material/info:", color="gray")

# --- Safety status: correctness-critical, so it stays even though the rest of
# the header is compact -- it is silent (no widget at all) in the common PASS
# case, so it does not add clutter.
safety = metadata.get("prediction_safety")
if not isinstance(safety, dict):
    st.warning("This is a legacy card that predates the safety audit; treat it as reference only.")
elif safety.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
    st.error("This card failed its safety checks. Do not treat it as a current forecast.")
elif safety.get("status") == "PASS_WITH_WARNINGS":
    warnings_text = "; ".join(str(item) for item in safety.get("warnings", []))
    st.warning(f"Passed with warnings: {warnings_text}")

# --- Honesty note, one line ---------------------------------------------------
historical = metadata.get("historical_evaluation")
if isinstance(historical, dict) and historical.get("accuracy") is not None:
    st.caption(
        f"Long-run accuracy is about {float(historical['accuracy']):.1%} -- not proof of a "
        'profitable edge. See "Is this thing good?" for the full picture before trusting any '
        "single pick."
    )
else:
    st.caption(
        "Long-run accuracy isn't available for this card -- treat every pick below as a lean, "
        "not proof of a profitable edge."
    )

# --- Tier-count chips ----------------------------------------------------------
strong_count, lean_count, coinflip_count = tier_counts(recommendations)
with st.container(horizontal=True):
    st.badge(f"{strong_count} strong", color="green")
    st.badge(f"{lean_count} leans", color="blue")
    st.badge(f"{coinflip_count} coin flips", color="gray")
    st.badge("Sorted by confidence", color="gray")

# --- Legend, written once above the board -------------------------------------
st.caption(
    "Conf = chance the pick covers at the line shown. Edge = points better (+) or worse (-) "
    "than the model's fair line. The strip is that confidence at every half-point within ±4 "
    "of the market line (amber box = market); tap a row for numbers and the why."
)

# --- Line sweep (optional: rows render without a strip if absent) ------------
sweep_path = find_line_sweep_file(forecast.directory)
sweep_frame = load_line_sweep(sweep_path) if sweep_path is not None else pd.DataFrame()
if "method" in sweep_frame.columns:
    # score_outcome_week_line_sweep() (nfl-ats margin-predict --line-sweep)
    # stacks every margin-distribution method's sweep into one file with no
    # per-game uniqueness otherwise -- keep only the method backing this card's
    # picks, or every game_id would carry 2-3x duplicate rows.
    sweep_frame = sweep_frame.loc[
        sweep_frame["method"].eq(metadata.get("ats_method", "market_residual"))
    ]

# --- Plain-English explanations (optional, feature-detected) -----------------
attribution_path = find_latest_attribution_file(artifacts_root())
explanations = (
    explanations_by_game(load_parquet_safe(attribution_path))
    if attribution_path is not None
    else {}
)

# --- Line journey inputs: live quotes + predicted close (both optional) ------
snapshots = list_recent_market_snapshots(data_root() / "market" / "raw", since_days=10)
quotes = (
    load_live_quotes(tuple(info.directory for info in snapshots)) if snapshots else pd.DataFrame()
)
opener_table = game_opening_lines(quotes) if not quotes.empty else pd.DataFrame()
opener_by_game = (
    {
        str(game_id): value
        for game_id, value in opener_table.set_index("nflverse_game_id")[
            "opener_home_spread"
        ].items()
    }
    if not opener_table.empty
    else {}
)
latest_table = spread_consensus(quotes) if not quotes.empty else pd.DataFrame()
latest_by_game = (
    {
        str(game_id): value
        for game_id, value in latest_table.set_index("nflverse_game_id")[
            "consensus_home_spread"
        ].items()
    }
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
    {
        str(game_id): value
        for game_id, value in close_predictions.set_index("game_id")[
            "predicted_close_home_spread"
        ].items()
    }
    if not close_predictions.empty
    and {"game_id", "predicted_close_home_spread"}.issubset(close_predictions.columns)
    else {}
)
if close_predictions_path is None:
    st.caption(
        "Predicted close isn't wired up yet -- the MKT-06 pilot model trains after the "
        "2020-2022 line archive re-fetch (mid-September). Every row shows an em dash there "
        "until then."
    )
if not opener_by_game:
    st.caption(
        f"No live line captures yet this week -- first capture expected "
        f"{next_tuesday_capture_label()}."
    )

# --- The board -----------------------------------------------------------------
board_html = render_picks_board(
    recommendations,
    sweep_frame,
    explanations,
    openers=opener_by_game,
    latest_lines=latest_by_game,
    predicted_close=predicted_close_by_game,
    metadata=metadata,
    theme=st.context.theme.type,
)
st.html(board_html, unsafe_allow_javascript=True)

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
