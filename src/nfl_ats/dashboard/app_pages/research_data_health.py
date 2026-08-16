"""Research: data & system health -- provenance, sync status, coverage, missingness."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_ats.active_model import active_artifact_path
from nfl_ats.constants import FEATURE_FAMILIES, MODEL_FEATURE_COLUMNS
from nfl_ats.dashboard.data import (
    artifacts_root,
    data_root,
    data_summary,
    describe_artifact_source,
    load_active_model,
    read_json_safe,
)
from nfl_ats.dashboard.ui import metric_with_context
from nfl_ats.reporting import feature_coverage

st.title("Data & system health")
st.caption(
    "Pipeline integrity, not model quality: is the active forecast linked to its evaluation, "
    "did it pass its safety checks, and does the source data look complete?"
)

artifacts = artifacts_root()
data = data_root()

active = load_active_model(artifacts)
forecast = (
    active_artifact_path(artifacts, active, "weekly_forecast") if active is not None else None
)
if active is None or forecast is None:
    st.warning("No synchronized active evaluation/forecast pair is available.")
elif not (forecast / "metadata.json").is_file():
    st.error(
        "The active model's weekly-forecast artifact is missing locally -- regenerate with "
        "`nfl-ats margin-predict`."
    )
else:
    st.caption(describe_artifact_source(forecast, artifacts))
    metadata = read_json_safe(forecast / "metadata.json") or {}
    safety = metadata.get("prediction_safety")
    status = safety.get("status") if isinstance(safety, dict) else "MISSING"
    if status in {"PASS", "PASS_WITH_WARNINGS"}:
        st.success(
            f"Active-model synchronization: PASS · model `{active.get('model_id')}` · "
            f"safety: {status}"
        )
    else:
        st.error(f"Prediction safety is not valid for the active forecast: {status}")

snapshot, feature_manifest = data_summary(data)
if snapshot is None:
    st.warning("No complete raw snapshot. Run `nfl-ats ingest`.")
else:
    manifest = snapshot["manifest"]
    st.success(f"Raw snapshot {snapshot['snapshot_id']} is complete.")
    with st.container(horizontal=True):
        metric_with_context(
            "Schedule rows", manifest["files"]["schedules.parquet"]["rows"], border=True
        )
        metric_with_context(
            "Team-stat rows", manifest["files"]["team_stats.parquet"]["rows"], border=True
        )
        metric_with_context("Seasons requested", len(manifest.get("seasons", [])), border=True)
    st.caption(f"Fetched {manifest['fetched_at_utc']} from nflverse-backed datasets.")
    with st.expander("Raw snapshot provenance"):
        st.json(snapshot, expanded=False)

feature_path = data / "processed" / "game_features.parquet"
if not feature_path.is_file():
    st.warning("No processed feature table. Run `nfl-ats build-features`.")
    st.stop()

features = pd.read_parquet(feature_path)
st.subheader("Processed research table")
with st.container(horizontal=True):
    metric_with_context("Games", len(features), border=True)
    metric_with_context("Seasons", features["season"].nunique(), border=True)
    metric_with_context("Named model inputs", len(MODEL_FEATURE_COLUMNS), border=True)
    metric_with_context("Completed labels", int(features["home_cover"].notna().sum()), border=True)
if feature_manifest is not None:
    st.caption(
        f"Built {feature_manifest.get('built_at_utc', '?')} · EWM span "
        f"{feature_manifest.get('ewm_span', '?')} · offseason retention "
        f"{float(feature_manifest.get('offseason_retention', 0.0)):.0%}."
    )

counts = features.groupby("season", as_index=False).agg(
    games=("game_id", "size"), completed=("home_cover", "count")
)
st.subheader("Season coverage")
st.bar_chart(counts, x="season", y=["games", "completed"], height=260)

coverage = feature_coverage(features, MODEL_FEATURE_COLUMNS)
low_coverage = coverage.loc[coverage["coverage"].lt(0.95)]
st.subheader("Missing inputs")
if low_coverage.empty:
    st.success("Every named input is available for at least 95% of game rows.")
else:
    st.warning(
        f"{len(low_coverage)} of {len(coverage)} inputs are present for less than 95% of rows."
    )
with st.expander("Coverage by input"):
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

with st.expander("What the input families mean"):
    st.dataframe(
        pd.DataFrame(
            [
                {"Family": family, "Inputs": len(columns)}
                for family, columns in FEATURE_FAMILIES.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
