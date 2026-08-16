"""What's happening with the lines? -- this week's live capture archive."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.dashboard.data import data_root, list_recent_market_snapshots, load_live_quotes
from nfl_ats.dashboard.ui import next_tuesday_capture_label, spread_label
from nfl_ats.market_data import latest_book_quotes, spread_consensus

st.title("What's happening with the lines?")
st.caption(
    "Live sportsbook captures for the current week: how much each line has moved since the "
    "Tuesday open, and how much books disagree right now."
)

snapshots = list_recent_market_snapshots(data_root() / "market" / "raw", since_days=10)

if not snapshots:
    st.info(
        "**No live captures yet this week.** The Tuesday opening-line capture is expected "
        f"**{next_tuesday_capture_label()}**. This page fills in automatically once captures "
        "start landing.",
        icon=":material/schedule:",
    )
    st.stop()

quotes = load_live_quotes(tuple(info.directory for info in snapshots))

if quotes.empty:
    st.info("Snapshot files exist for this window, but none contain any readable quotes yet.")
    st.stop()

matched = quotes.loc[quotes["nflverse_game_id"].notna()].copy()
unmatched_events = quotes.loc[quotes["nflverse_game_id"].isna(), "provider_event_id"].nunique()

# Books post lines for the whole season well in advance, so narrow "this week"
# down to one NFL week: the soonest week with an upcoming kickoff, or the
# earliest week captured if every kickoff has already passed.
week_tokens = matched["nflverse_game_id"].str.extract(r"^(\d{4}_\d{2})_")[0]
if week_tokens.notna().any():
    matched = matched.assign(_week_token=week_tokens)
    now_ts = pd.Timestamp.now(tz="UTC")
    commence = pd.to_datetime(matched["commence_time_utc"], errors="coerce", utc=True)
    upcoming_tokens = matched.loc[commence.ge(now_ts), "_week_token"]
    target_week = (
        upcoming_tokens.sort_values().iloc[0]
        if not upcoming_tokens.empty
        else matched["_week_token"].sort_values().iloc[0]
    )
    matched = matched.loc[matched["_week_token"].eq(target_week)].drop(columns="_week_token")

with st.container(horizontal=True):
    st.metric("Snapshots captured", len(snapshots), border=True)
    st.metric("Games tracked", int(matched["nflverse_game_id"].nunique()), border=True)
    st.metric(
        "Books seen",
        int(quotes["bookmaker_key"].nunique()),
        border=True,
    )

first_capture = min(info.observed_at for info in snapshots)
last_capture = max(info.observed_at for info in snapshots)
st.caption(
    f"Captures from {first_capture.strftime('%a %b %d, %H:%M UTC')} through "
    f"{last_capture.strftime('%a %b %d, %H:%M UTC')}"
    + (
        f" · {unmatched_events} event(s) not yet matched to a scheduled game"
        if unmatched_events
        else ""
    )
)

if matched.empty:
    st.info("No captured quotes have been matched to a scheduled game yet.")
    st.stop()

# --- Book dispersion right now -------------------------------------------------
st.subheader("Book dispersion right now")
st.write("How much sportsbooks currently disagree on each spread -- wider means less consensus.")
consensus = spread_consensus(matched)
if consensus.empty:
    st.write("No current spread quotes are available yet.")
else:
    game_names = (
        matched.loc[matched["market"].eq("spreads") & matched["outcome_side"].eq("HOME")]
        .drop_duplicates("nflverse_game_id")
        .set_index("nflverse_game_id")[["home_team", "away_team"]]
    )
    display = consensus.join(game_names, on="nflverse_game_id")
    display["Matchup"] = display["away_team"].fillna("?") + " @ " + display["home_team"].fillna("?")
    display["Spread"] = display["spread_max"] - display["spread_min"]
    st.dataframe(
        display[
            ["Matchup", "bookmakers", "consensus_home_spread", "spread_min", "spread_max", "Spread"]
        ]
        .rename(
            columns={
                "bookmakers": "Books",
                "consensus_home_spread": "Consensus (home)",
                "spread_min": "Low",
                "spread_max": "High",
                "Spread": "Range",
            }
        )
        .sort_values("Range", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "Consensus (home)": st.column_config.NumberColumn(format="%.1f"),
            "Low": st.column_config.NumberColumn(format="%.1f"),
            "High": st.column_config.NumberColumn(format="%.1f"),
            "Range": st.column_config.NumberColumn(format="%.1f"),
        },
    )

st.divider()

# --- Movement since Tuesday open -----------------------------------------------
st.subheader("Line movement since Tuesday open")
home_spreads = matched.loc[
    matched["market"].eq("spreads") & matched["outcome_side"].eq("HOME")
].copy()
if home_spreads.empty:
    st.write("No spread history is available yet for this week's games.")
else:
    home_spreads["observed_at_utc"] = pd.to_datetime(home_spreads["observed_at_utc"], utc=True)
    per_game = (
        home_spreads.groupby(["nflverse_game_id", "observed_at_utc"], as_index=False)
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            median_home_spread=("home_spread_line", "median"),
        )
        .sort_values("observed_at_utc")
    )
    games = per_game.drop_duplicates("nflverse_game_id").sort_values(["away_team", "home_team"])
    columns_per_row = 3
    game_ids = games["nflverse_game_id"].tolist()
    for start in range(0, len(game_ids), columns_per_row):
        row_ids = game_ids[start : start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column, game_id in zip(columns, row_ids, strict=False):
            with column:
                game_rows = per_game.loc[per_game["nflverse_game_id"].eq(game_id)]
                label = f"{game_rows['away_team'].iloc[0]} @ {game_rows['home_team'].iloc[0]}"
                with st.container(border=True):
                    st.caption(label)
                    if len(game_rows) < 2:
                        st.caption("Only one capture so far -- movement will show once more land.")
                    st.line_chart(
                        game_rows.set_index("observed_at_utc")["median_home_spread"],
                        height=140,
                    )
                    latest = latest_book_quotes(
                        matched.loc[matched["nflverse_game_id"].eq(game_id)], before_kickoff=False
                    )
                    latest_home = latest.loc[
                        latest["market"].eq("spreads") & latest["outcome_side"].eq("HOME")
                    ]
                    if not latest_home.empty:
                        latest_line = float(latest_home["home_spread_line"].median())
                        st.caption(f"Latest: {label.split(' @ ')[1]} {spread_label(latest_line)}")

with st.expander("What counts as this week?"):
    st.write(
        "This page only shows *live* captures -- the ones taken automatically each week. It "
        "excludes the 2020-2025 historical backfill archive used for research, which is "
        "labeled separately in storage and never mixed into this view."
    )
