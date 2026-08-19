"""Entry point: `streamlit run app.py` (launched via `nfl-ats dashboard`).

This is NOT the dashboard. The public GitHub Pages site (`docs/`, built by
`nfl-ats publish-board`) is the one dashboard the owner reads and shares --
this week's picks, every finding in plain English, and the honest track
record all live there, and only there. Duplicating that content here was
the exact confusion this app used to cause (two places to check, silently
able to disagree), so those three views were retired from this app rather
than kept as a second copy.

What is left is an internal research console: three pages of artifact
diagnostics that have no public-site equivalent because they are not meant
for the owner to read casually -- how the model decides (the family-weight
and per-game market-decomposition view), the engine room (which artifacts
are current, and what to run when they aren't), and the pool workbench
(submission status, rules, entries ranked by confidence, the Best Pick
nomination, and what the pool's own format is and isn't worth). Every page
anchors its numbers through :mod:`nfl_ats.dashboard.state` -- one definition
of "current," staleness said out loud instead of silently disagreeing.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from nfl_ats.dashboard.data import artifacts_root, data_root

PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

st.set_page_config(
    page_title="NFL ATS -- research console",
    page_icon=":material/settings:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            PAGES_DIR / "engine_room.py",
            title="Engine room",
            icon=":material/settings:",
            default=True,
        ),
        st.Page(
            PAGES_DIR / "model_explanation.py",
            title="How the model decides",
            icon=":material/insights:",
        ),
        st.Page(
            PAGES_DIR / "workbench.py",
            title="Pool workbench",
            icon=":material/checklist:",
        ),
    ],
    position="sidebar",
)

with st.sidebar:
    st.caption("Internal research console -- not the dashboard.")
    st.caption(
        "The dashboard is the public site: docs/ on GitHub Pages, built by "
        "`nfl-ats publish-board` (serve `docs/` locally with any static file "
        "server, e.g. `python -m http.server`). This app is diagnostics only "
        "-- no wagering integration."
    )
    with st.expander("Technical paths", icon=":material/folder:"):
        st.caption(f"Data: {data_root().resolve()}")
        st.caption(f"Artifacts: {artifacts_root().resolve()}")

page.run()
