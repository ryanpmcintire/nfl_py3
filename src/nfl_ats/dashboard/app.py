"""Entry point: `streamlit run app.py` (launched via `nfl-ats dashboard`).

Navigation follows the owner's priority order: this week's picks first, then
the honest track record, then the live line archive, then a collapsed
Research section for everything else.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from nfl_ats.dashboard.data import artifacts_root, data_root

PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

st.set_page_config(
    page_title="NFL ATS picks",
    page_icon=":material/sports_football:",
    layout="wide",
)

page = st.navigation(
    {
        "": [
            st.Page(
                PAGES_DIR / "home.py",
                title="This week's picks",
                icon=":material/sports_football:",
                default=True,
            ),
            st.Page(
                PAGES_DIR / "track_record.py",
                title="Is this thing good?",
                icon=":material/query_stats:",
            ),
            st.Page(
                PAGES_DIR / "market.py",
                title="What's happening with the lines?",
                icon=":material/show_chart:",
            ),
        ],
        "Research": [
            st.Page(
                PAGES_DIR / "research_backtests.py",
                title="Backtests",
                icon=":material/history:",
            ),
            st.Page(
                PAGES_DIR / "research_validation.py",
                title="Model selection & validation",
                icon=":material/rule:",
            ),
            st.Page(
                PAGES_DIR / "research_experiments.py",
                title="Feature & player experiments",
                icon=":material/science:",
            ),
            st.Page(
                PAGES_DIR / "research_bankroll.py",
                title="Bankroll simulation",
                icon=":material/payments:",
            ),
            st.Page(
                PAGES_DIR / "research_data_health.py",
                title="Data & system health",
                icon=":material/monitor_heart:",
            ),
            st.Page(
                PAGES_DIR / "research_explorer.py",
                title="Team & matchup explorer",
                icon=":material/search:",
            ),
            st.Page(
                PAGES_DIR / "research_glossary.py",
                title="Glossary",
                icon=":material/menu_book:",
            ),
        ],
    },
    position="sidebar",
)

with st.sidebar:
    st.caption("Research project; no wagering integration.")
    with st.expander("Technical paths", icon=":material/folder:"):
        st.caption(f"Data: {data_root().resolve()}")
        st.caption(f"Artifacts: {artifacts_root().resolve()}")

page.run()
