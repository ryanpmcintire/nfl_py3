"""Engine room: the sync-status board that makes desync visible instead of silent.

The dashboard's historical failure mode was pages quietly anchored to
different pipeline states. This page is the antidote: it renders
:func:`nfl_ats.dashboard.state.project_state` directly -- the headline sync
verdict, every staleness statement it reports with the exact command that
fixes it, the active model's identity in human terms, how fresh the data
underneath is, which artifact directory each page is actually reading, and
the weekly running order.

Operator-focused by design: restrained type, no charts beyond stat tiles, and
every runnable command set in monospace so it can be copied without squinting.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.data import artifact_time, artifacts_root, data_root, data_summary
from nfl_ats.dashboard.state import project_state

# ---------------------------------------------------------------------------
# Formatting helpers (presentation only -- every fact comes from ProjectState)
# ---------------------------------------------------------------------------


def _html(inner: str) -> None:
    """Render one composed fragment inside the scoped ``.ats`` root."""

    st.html(f'<div class="ats">{inner}</div>', unsafe_allow_javascript=True)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(source: Any, key: str) -> dict[str, Any]:
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, dict) else {}


def _mono(text: str) -> str:
    return f'<code style="font-family:monospace;font-size:12.5px;">{escape(text)}</code>'


def _commands_as_code(text: str) -> str:
    """Escape a plain-English status string, rendering its `backticked` bits as code."""

    parts = text.split("`")
    return "".join(
        f'<code style="font-family:monospace;">{escape(part)}</code>' if index % 2 else escape(part)
        for index, part in enumerate(parts)
    )


def _commands_in(text: str) -> list[str]:
    """The `backticked` commands inside one of ProjectState's issue strings."""

    parts = text.split("`")
    return [part for index, part in enumerate(parts) if index % 2 and part.strip()]


def _relative(directory: Path, base: Path) -> str:
    try:
        return directory.resolve().relative_to(base.resolve()).as_posix()
    except (OSError, ValueError):
        return str(directory)


def _timestamp(value: Any) -> str:
    """A UTC wall-clock stamp from an ISO-8601 artifact field."""

    text = str(value) if value is not None else ""
    if not text:
        return "unknown"
    try:
        instant = datetime.fromisoformat(text)
    except ValueError:
        return text
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _detail_rows(rows: Sequence[tuple[str, str]]) -> str:
    """A label/value grid; values are pre-rendered HTML, labels are escaped here."""

    cells = "".join(
        f'<p class="fine" style="margin:0;">{escape(label)}</p>'
        f'<p class="sub" style="margin:0;">{value}</p>'
        for label, value in rows
    )
    return (
        '<div style="display:grid;grid-template-columns:minmax(120px,190px) 1fr;'
        f'gap:9px 18px;align-items:baseline;margin-top:12px;">{cells}</div>'
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        '<div style="overflow-x:auto;margin-top:12px;">'
        f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _chip(label: str, satisfied: bool) -> str:
    color = "var(--good-text)" if satisfied else "var(--critical)"
    mark = "&#10003;" if satisfied else "&#10007;"
    return (
        f'<span class="chip" style="color:{color};border-color:currentColor;">'
        f"{mark} {escape(label)}</span>"
    )


PROFILE_NOTES: dict[str, str] = {
    "base": "team form, ratings, rest and the market line",
    "pbp": "the base inputs plus play-by-play states",
    "pbp_adjusted": "opponent-adjusted play-by-play states on top of the base inputs",
    "graph": "the base inputs plus schedule-graph strength ratings",
    "player": "base and play-by-play inputs plus expected lineup, injury and quarterback states",
}

METHOD_NOTES: dict[str, str] = {
    "market_residual": "how far the model's own number sits from the market's line",
    "fair_margin": "the model's own point margin, compared against the market line",
    "direct_ats": "which side covers, predicted directly",
    "straight_up": "which team wins outright",
    "market": "the market line taken at face value (the baseline to beat)",
}

# The weekly running order. Sources: README.md's reproduce-and-publish block
# (the authoritative command list, including the feature flags the active
# `player` profile needs) and HANDOFF.md's end-of-session contract.
WEEKLY_STEPS: tuple[tuple[str, str], ...] = (
    (
        "nfl-ats ingest",
        "Download a fresh, dated nflverse snapshot of schedules and team stats. Nothing "
        "downstream sees new results until this runs.",
    ),
    (
        "nfl-ats build-features",
        "Turn that snapshot into the canonical game table -- team form, ratings, rest and the "
        "market line -- including postseason rows.",
    ),
    (
        "nfl-ats build-pbp-features",
        "Layer the leak-safe play-by-play states onto that table. Needs a current "
        "`nfl-ats pbp-ingest` snapshot first.",
    ),
    (
        "nfl-ats build-player-features",
        "Layer on expected lineup, injury and quarterback states. This is the table the active "
        "model actually reads. Needs `nfl-ats player-ingest` and `nfl-ats player-value-ingest` "
        "first.",
    ),
    (
        "nfl-ats margin-backtest",
        "Re-score the model across history and write the evaluation the track record quotes. "
        "Run with `--features data/processed/game_features_player.parquet --feature-profile "
        "player`.",
    ),
    (
        "nfl-ats margin-predict",
        "Build this week's card and activate it. Same feature flags, plus `--season` and "
        "`--week`. It fails closed rather than publish a card that misses a safety check.",
    ),
    (
        "nfl-ats publish-predictions",
        "Write the public Markdown card and record every pick in the paper-picks ledger at the "
        "line it was published at.",
    ),
)

# ---------------------------------------------------------------------------
# The anchor
# ---------------------------------------------------------------------------

state = project_state()
root = artifacts_root()
data = data_root()
active = state.active
forecast = state.forecast

historical = _mapping(active, "historical_evaluation")
evaluation_artifact = str(historical.get("artifact") or "")
evaluation_directory = (root / evaluation_artifact) if evaluation_artifact else None
forecast_directory = forecast.directory if forecast is not None else None

st.html(theme.stylesheet())

_html(
    viz.page_header(
        "Engine room",
        "What is current, and what to run when it is not",
        "One anchor for the whole dashboard. If anything here is stale, the fix is on this "
        "page next to the thing that went stale.",
    )
)

# ---------------------------------------------------------------------------
# 1. The headline verdict
# ---------------------------------------------------------------------------

outstanding = len(state.issues)

if state.synchronized and not state.issues:
    hero_word = "Synchronized"
    marker = viz.status_line("good", "Every page is reading the same pipeline state")
    plain = (
        "The card on the picks page was produced by the active model, and the track record "
        "quotes that same model's evaluation. Nothing on the dashboard is quietly describing "
        "a different build."
    )
elif state.synchronized:
    # The picks/model pair is sound, but a supporting artifact is stale. Saying
    # "synchronized" alone here would contradict the warning cards underneath.
    hero_word = "Synchronized"
    marker = viz.status_line(
        "warning",
        f"{outstanding} supporting artifact{'s' if outstanding != 1 else ''} still needs a re-run",
    )
    plain = (
        "The picks and the active model line up, so this week's card can be trusted. What is "
        "behind is a supporting measurement: it was taken on an earlier build of the inputs, "
        "so it describes a slightly older version of the pipeline until it is re-run."
    )
else:
    hero_word = "Not synchronized"
    marker = viz.status_line("critical", "Pages may be describing different builds")
    plain = (
        "Something the dashboard reads does not belong to the active model. The numbers are "
        "still real measurements, but they may have come from different pipeline states, so "
        "two pages can disagree without either being wrong. Every fix is listed below."
    )

checks = (
    ("An active model is on record", active is not None),
    ("A pick card exists", forecast is not None),
    ("The pick card is the active model's", forecast is not None and forecast.is_active),
    (
        "The opening-line grade matches it",
        state.opener_evaluation.consistent_with_active is not False,
    ),
)

_html(
    viz.card(
        '<p class="kicker">Pipeline status</p>'
        f'<div class="hero">{escape(hero_word)}</div>'
        f'<div style="margin:8px 0 10px;">{marker}</div>'
        f'<p class="sub">{escape(plain)}</p>'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;">'
        + "".join(_chip(label, satisfied) for label, satisfied in checks)
        + "</div>",
        accent=True,
    )
)

# ---------------------------------------------------------------------------
# 2. Every staleness statement, with the command that clears it
# ---------------------------------------------------------------------------

if state.issues:
    issue_cards: list[str] = []
    for issue in state.issues:
        commands = _commands_in(issue)
        run_block = ""
        if commands:
            listed = "".join(
                '<div style="font-family:monospace;font-size:13px;font-weight:600;'
                f'margin-top:4px;">{escape(command)}</div>'
                for command in commands
            )
            run_block = f'<p class="kicker" style="margin:14px 0 0;">Run this</p>{listed}'
        issue_cards.append(
            viz.card(
                viz.status_line("warning", "Out of sync")
                + f'<p class="sub" style="margin-top:8px;">{_commands_as_code(issue)}</p>'
                + run_block
            )
        )
    _html("".join(issue_cards))
else:
    _html(
        viz.card(
            viz.status_line("good", "Nothing to re-run")
            + '<p class="sub" style="margin-top:8px;">Every artifact the dashboard reads '
            "belongs to the active model.</p>"
        )
    )

# ---------------------------------------------------------------------------
# 3. The active model, in human terms
# ---------------------------------------------------------------------------

if active is None:
    _html(
        viz.empty_state(
            "No active model",
            "Nothing has been activated yet, so the dashboard has no anchor to describe. Run "
            "nfl-ats margin-backtest and then nfl-ats margin-predict to activate one.",
        )
    )
else:
    profile = str(active.get("feature_profile", "?"))
    method = str(active.get("method", "?"))
    regressor = str(active.get("regressor", "?"))
    alpha = _number(active.get("ridge_alpha"))
    calibration = str(active.get("calibration_method", "none"))
    weekly = _mapping(active, "weekly_forecast")

    profile_note = PROFILE_NOTES.get(profile, "a named combination of the project's inputs")
    method_note = METHOD_NOTES.get(method, "a named prediction target")
    recipe = f"{regressor} regression"
    if alpha is not None:
        recipe += f", strength {alpha:g}"
    recipe += f"; probability calibration: {calibration}"

    if evaluation_artifact:
        made = escape(artifact_time(Path(evaluation_artifact)))
        evaluation_value = f"{_mono(evaluation_artifact)} &middot; made {made}"
    else:
        evaluation_value = (
            '<span style="color:var(--muted);">the manifest names no evaluation artifact</span>'
        )

    season = weekly.get("season")
    week = weekly.get("week")
    card_label = f"{season} Week {week}" if season is not None and week is not None else "unknown"
    if forecast_directory is not None:
        card_value = (
            f"{escape(card_label)} &middot; {_mono(_relative(forecast_directory, root))} "
            f"&middot; made {escape(artifact_time(forecast_directory))}"
        )
    else:
        card_value = '<span style="color:var(--muted);">no card is saved on disk</span>'

    rows = (
        ("Model id", _mono(str(active.get("model_id", "?")))),
        ("Activated", escape(_timestamp(active.get("activated_at_utc")))),
        ("What it predicts", f"{escape(method)} &mdash; {escape(method_note)}"),
        ("Inputs it reads", f"{escape(profile)} profile &mdash; {escape(profile_note)}"),
        ("How it is fitted", escape(recipe)),
        ("Evaluation it quotes", evaluation_value),
        ("Card it is serving", card_value),
    )
    serving_status = (
        viz.status_line("good", "The saved card is this model's card")
        if forecast is not None and forecast.is_active
        else viz.status_line("warning", "The saved card was not produced by this model")
    )
    _html(
        viz.card(
            '<p class="kicker">The active model</p>'
            '<p class="title">Who is actually making the picks</p>'
            f'<div style="margin-top:8px;">{serving_status}</div>' + _detail_rows(rows)
        )
    )

# ---------------------------------------------------------------------------
# 4. Data freshness
# ---------------------------------------------------------------------------

snapshot, feature_manifest = data_summary(data)

if snapshot is None:
    _html(
        viz.card(
            viz.status_line("warning", "No complete raw snapshot")
            + '<p class="sub" style="margin-top:8px;">Nothing has been downloaded yet. Run '
            + _mono("nfl-ats ingest")
            + " to pull schedules and team stats.</p>"
        )
    )
elif feature_manifest is None:
    _html(
        viz.card(
            viz.status_line("warning", "No feature table has been built")
            + '<p class="sub" style="margin-top:8px;">A raw snapshot exists but nothing has '
            "been built from it. Run " + _mono("nfl-ats build-features") + ".</p>"
        )
    )
else:
    manifest = _mapping(snapshot, "manifest")
    files = _mapping(manifest, "files")
    snapshot_id = str(snapshot.get("snapshot_id", "?"))
    source_snapshot = str(feature_manifest.get("source_snapshot", ""))

    feature_rows = _number(feature_manifest.get("rows"))
    postseason_rows = _number(feature_manifest.get("postseason_rows"))
    settled_rows = _number(feature_manifest.get("completed_non_push_rows"))
    includes_playoffs = postseason_rows is not None and postseason_rows > 0

    tiles = [
        viz.stat_tile(
            "Games in the table",
            f"{int(feature_rows):,}" if feature_rows is not None else "--",
            "Every game row the model can see, across all covered seasons.",
        ),
        viz.stat_tile(
            "Playoff rows",
            f"{int(postseason_rows):,}" if postseason_rows is not None else "0",
            "Wild card through Super Bowl. The pool needs 13 playoff picks in January.",
        ),
        viz.stat_tile(
            "Games with a settled result",
            f"{int(settled_rows):,}" if settled_rows is not None else "--",
            "Finished, non-push games -- the only ones a grade can be measured on.",
        ),
    ]

    playoff_status = (
        viz.status_line("good", "Includes playoff rows: yes")
        if includes_playoffs
        else viz.status_line("warning", "Includes playoff rows: no")
    )
    playoff_note = (
        "January serving depends on this: the pool requires a pick on every playoff game, and "
        "the model can only serve weeks whose rows exist in the table. The two-pass build that "
        "adds them without disturbing any regular-season number is written up in "
        "docs/postseason_support.md."
    )

    freshness_status = ""
    if source_snapshot and snapshot_id and source_snapshot != snapshot_id:
        freshness_status = (
            '<div style="margin-top:12px;">'
            + viz.status_line("warning", "Built from an older snapshot than the newest on disk")
            + '</div><p class="fine" style="margin-top:6px;">The feature table was built from '
            + _mono(source_snapshot)
            + " but "
            + _mono(snapshot_id)
            + " has since been downloaded. Re-run "
            + _mono("nfl-ats build-features")
            + " (and the two enriched builds) to pick it up.</p>"
        )

    detail = (
        ("nflverse snapshot", _mono(snapshot_id)),
        ("Downloaded", escape(_timestamp(manifest.get("fetched_at_utc")))),
        (
            "Rows downloaded",
            f"{int(_number(_mapping(files, 'schedules.parquet').get('rows')) or 0):,} schedule "
            f"&middot; {int(_number(_mapping(files, 'team_stats.parquet').get('rows')) or 0):,} "
            "team-stat",
        ),
        ("Feature table built", escape(_timestamp(feature_manifest.get("built_at_utc")))),
        ("Built from snapshot", _mono(source_snapshot) if source_snapshot else "unknown"),
        (
            "Seasons covered",
            f"{escape(str(feature_manifest.get('first_season', '?')))}"
            f"&ndash;{escape(str(feature_manifest.get('last_season', '?')))}",
        ),
    )

    _html(
        viz.card(
            '<p class="kicker">Data freshness</p>'
            '<p class="title" style="margin-bottom:12px;">What the model is looking at</p>'
            f'<div class="row">{"".join(tiles)}</div>'
            + _detail_rows(detail)
            + f'<div style="margin-top:14px;">{playoff_status}</div>'
            + f'<p class="fine" style="margin-top:6px;">{escape(playoff_note)}</p>'
            + freshness_status
        )
    )

# ---------------------------------------------------------------------------
# 5. Artifact currency -- which directory each page is actually reading
# ---------------------------------------------------------------------------


def _artifact_cells(directory: Path | None) -> tuple[str, str]:
    if directory is None:
        return '<span style="color:var(--muted);">not built yet</span>', "&mdash;"
    return _mono(_relative(directory, root)), escape(artifact_time(directory))


ARTIFACTS: tuple[tuple[str, str, Path | None, str], ...] = (
    (
        "This week's pick card",
        "The board on the picks page",
        forecast_directory,
        "nfl-ats margin-predict",
    ),
    (
        "Model evaluation",
        "The long-run record on the track record page",
        evaluation_directory,
        "nfl-ats margin-backtest",
    ),
    (
        "Opening-line grade",
        "The pool-relevant accuracy number",
        state.opener_evaluation.directory,
        "nfl-ats opener-evaluation",
    ),
    (
        "Paper-picks ledger",
        "How the line moved after each published pick",
        state.clv_ledger.directory,
        "nfl-ats clv-ledger",
    ),
    (
        "Predicted close",
        "The close guess shown on each pick row",
        state.close_predictions.directory,
        "nfl-ats predict-close",
    ),
)

artifact_cells: list[list[str]] = []
for name, purpose, directory, command in ARTIFACTS:
    shown, created = _artifact_cells(directory)
    artifact_cells.append(
        [
            f'<b>{escape(name)}</b><br><span class="fine">{escape(purpose)}</span>',
            shown,
            created,
            _mono(command),
        ]
    )

_html(
    viz.card(
        '<p class="kicker">Artifact currency</p>'
        '<p class="title">Exactly which directory each page is reading</p>'
        '<p class="sub" style="margin-top:6px;">These are the paths the dashboard resolved this '
        "rerun, not the newest directories on disk. If one of them is older than you expect, "
        "the refresh command is in the last column.</p>"
        + _table(
            ["Artifact", "Directory the dashboard is reading", "Created", "Refresh with"],
            artifact_cells,
        )
    )
)

# ---------------------------------------------------------------------------
# 6. Ops crib sheet -- the weekly running order
# ---------------------------------------------------------------------------

steps = "".join(
    '<li style="display:grid;grid-template-columns:24px 1fr;gap:12px;padding:10px 0;'
    + ("" if index == 0 else "border-top:1px solid var(--grid);")
    + '">'
    + f'<span class="fine num" style="margin:0;font-weight:700;">{index + 1}</span>'
    + "<div>"
    + f'<div style="font-family:monospace;font-size:13px;font-weight:650;">{escape(command)}</div>'
    + f'<p class="fine" style="margin:5px 0 0;">{_commands_as_code(description)}</p>'
    + "</div></li>"
    for index, (command, description) in enumerate(WEEKLY_STEPS)
)

_html(
    viz.card(
        '<p class="kicker">Ops crib sheet</p>'
        '<p class="title">The weekly running order</p>'
        '<p class="sub" style="margin-top:6px;">Run it early Tuesday: the pool locks picks at '
        "Tuesday noon, so the card has to exist and be published before then. Each step feeds "
        "the next, so a skipped step leaves the dashboard reporting a desync on this page.</p>"
        f'<ol style="margin:14px 0 0;padding:0;list-style:none;">{steps}</ol>'
        '<p class="fine" style="margin-top:14px;">The scoring passes are separate and can run '
        "any time after a week's games close: "
        + _mono("nfl-ats opener-evaluation")
        + " re-measures the pool-relevant grade, "
        + _mono("nfl-ats clv-ledger")
        + " scores published picks against the close, and "
        + _mono("nfl-ats predict-close")
        + " needs the week's Tuesday line capture before it will write anything.</p>"
    )
)

st.html(theme.theme_sync_script(), unsafe_allow_javascript=True)
