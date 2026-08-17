"""Entry point: `streamlit run app.py` (launched via `nfl-ats dashboard`).

Four pages, organized around what the owner actually asks the project:
this week's picks (the three prediction views), what we've learned (every
finding in plain English), the honest track record, and the engine room
(which artifacts are current, and what to run when they aren't). Every page
anchors its numbers through :mod:`nfl_ats.dashboard.state` -- one definition
of "current," staleness said out loud instead of silently disagreeing.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from nfl_ats.dashboard.data import artifacts_root, data_root

PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

st.set_page_config(
    page_title="NFL ATS",
    page_icon=":material/sports_football:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            PAGES_DIR / "picks.py",
            title="This week",
            icon=":material/sports_football:",
            default=True,
        ),
        st.Page(
            PAGES_DIR / "findings.py",
            title="What we've learned",
            icon=":material/school:",
        ),
        st.Page(
            PAGES_DIR / "track_record.py",
            title="Track record",
            icon=":material/query_stats:",
        ),
        st.Page(
            PAGES_DIR / "engine_room.py",
            title="Engine room",
            icon=":material/settings:",
        ),
    ],
    position="sidebar",
)

with st.sidebar:
    st.caption("Research project; no wagering integration.")
    with st.expander("Technical paths", icon=":material/folder:"):
        st.caption(f"Data: {data_root().resolve()}")
        st.caption(f"Artifacts: {artifacts_root().resolve()}")

page.run()
