"""Research: glossary -- plain-language definitions for terms used across the dashboard."""

from __future__ import annotations

import streamlit as st

from nfl_ats.dashboard.glossary import GLOSSARY

st.title("Glossary")
st.caption("Plain-language definitions for the terms used across this dashboard.")

for term, definition in GLOSSARY:
    with st.container(border=True):
        st.markdown(f"**{term}**")
        st.write(definition)
