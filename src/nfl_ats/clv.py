"""Closing-line value (CLV) evaluation harness and the predeclared MKT-06 pilot.

This module turns the append-only point-in-time market archive under
``data/market/raw/`` (see ``nfl_ats.odds_backfill`` and ``nfl_ats.market_data``)
into:

1. a long snapshot **pairing table**: one row per (game, decision label) with a
   cross-book consensus home spread (plus book count and dispersion), totals,
   and moneyline where present, joined to the schedule's closing spread;
2. a generic **CLV metric harness** that scores any pick stream against the
   store's own closing snapshot (falling back to the nflverse schedule close
   with a visible source column) and reports week-blocked bootstrap intervals;
3. the **predeclared MKT-06 close-prediction pilot**: a frozen ridge model that
   predicts ``close_home_spread - tue_open_home_spread`` from a frozen,
   five-feature list, trained/validated/tested on a frozen season split;
4. the **sign test** (report Pilot B): does the active market-residual model's
   fair-margin correction from the opener predict the direction the close
   actually moves?

Store contract
---------------
Every snapshot directory under ``data/market/raw/<id>/`` is committed
atomically with ``manifest.json`` written last, so only directories that
contain a manifest are read (an in-progress backfill's newest directory is
silently skipped, not treated as corrupt). Each manifest's ``capture_kind`` is
either ``"historical_backfill"`` (from ``nfl_ats.odds_backfill``) or absent,
which this module normalizes to ``"live"`` for parity with
``nfl_ats.market_data``'s live capture path. Every quote row returned by this
module's loaders carries an explicit ``capture_kind`` column -- live and
historical-backfill rows are therefore never silently mixed without a column
that distinguishes them, even though both live under the same store root.

Decision labels come from ``nfl_ats.odds_backfill.DECISION_LABELS`` and are
already chronologically ordered within an NFL week
(``tue_open < thu_pre_tnf < sat_midday < sun_early_close < sun_late_close <
mon_pre_mnf``); :func:`assert_monotone_decision_timeline` checks that every
game's retained pregame snapshots respect that order.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.best_pick import select_best_pick
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import (
    MarginFeatureProfile,
    fit_margin_model,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.market_data import tuesday_opener_quotes
from nfl_ats.modeling import regular_season_rows
from nfl_ats.odds_backfill import DECISION_LABELS, HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import sha256_file
from nfl_ats.snapshots import latest_snapshot

LIVE_CAPTURE_KIND = "live"
BootstrapBlock = Literal["week", "season"]

DECISION_LABEL_ORDER: dict[str, int] = {label: index for index, label in enumerate(DECISION_LABELS)}

# The store's own two Sunday-close labels are the project's "close" proxy, as
# specified in the market-data report (section 2/5): prefer the later Sunday
# checkpoint, fall back to the earlier one, and only then fall back to the
# nflverse schedule's recorded closing spread. ``mon_pre_mnf`` is deliberately
# excluded even though it postdates both Sunday labels, matching the literal
# instruction this module was built against.
CLOSE_LABEL_PRIORITY: tuple[str, ...] = ("sun_late_close", "sun_early_close")

KEY_NUMBERS: tuple[float, ...] = (3.0, 7.0)

# The currently active market-residual model's configuration, read from
# ``artifacts/active_ats_model.json`` on 2026-08-16 (feature_profile="player",
# regressor="ridge", ridge_alpha defaults to 10.0 when absent from the
# manifest, target="market_residual"). Used only as a fallback when a live
# manifest is not present in this worktree's own artifacts root (it is a
# generated, gitignored artifact and may legitimately be absent -- see
# AGENTS.md); :func:`resolve_active_model_config` prefers the live manifest.
_ACTIVE_MODEL_FALLBACK_CONFIG: dict[str, Any] = {
    "feature_profile": "player",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "target": "market_residual",
}

# The pilot's frozen, five-feature list (BUILD item 3). Order is fixed so a
# trained estimator's coefficients stay meaningfully positioned.
FROZEN_PILOT_FEATURES: tuple[str, ...] = (
    "tue_open_home_spread",
    "tue_open_key_number_distance",
    "active_model_residual_at_opener",
    "week",
    "rest_diff",
)
FROZEN_PILOT_RIDGE_ALPHA = 10.0


class PilotProtocolBlocked(ValueError):
    """Raised when the frozen train/validate/test season split has no data.

    Per the task's constraints, an impossible piece of the frozen protocol is
    reported as a blocker, not silently satisfied with a substitute season
    split.
    """


@dataclass(frozen=True)
class PilotSplit:
    train_start_season: int
    train_end_season: int
    validate_season: int
    test_season: int


# Predeclared before any 2024/2025 pairing was inspected (BUILD item 3).
FROZEN_PILOT_PROTOCOL = PilotSplit(
    train_start_season=2020, train_end_season=2023, validate_season=2024, test_season=2025
)


# ---------------------------------------------------------------------------
# 1. Snapshot manifest index and decision-label quote loading
# ---------------------------------------------------------------------------


def load_snapshot_manifest_index(root: Path) -> pd.DataFrame:
    """Enumerate committed snapshot directories under a market ``raw/`` root.

    Only directories containing ``manifest.json`` are read -- the store is
    append-only and each snapshot is committed atomically with the manifest
    written last, so a backfill executor's in-progress newest directory is
    tolerated (silently skipped) rather than treated as a broken read.
    """

    columns = [
        "dir",
        "snapshot_id",
        "capture_kind",
        "season",
        "week",
        "decision_label",
        "snapshot_timestamp_utc",
        "requested_at_utc",
    ]
    if not root.is_dir():
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request = manifest.get("request") or {}
        rows.append(
            {
                "dir": str(entry),
                "snapshot_id": manifest.get("snapshot_id", entry.name),
                "capture_kind": manifest.get("capture_kind", LIVE_CAPTURE_KIND),
                "season": request.get("season"),
                "week": request.get("week"),
                "decision_label": request.get("decision_label"),
                "snapshot_timestamp_utc": (
                    manifest.get("snapshot_timestamp_utc") or manifest.get("observed_at_utc")
                ),
                "requested_at_utc": manifest.get("requested_at_utc"),
            }
        )
    index = pd.DataFrame(rows, columns=columns)
    if index.empty:
        return index
    index["season"] = pd.to_numeric(index["season"], errors="coerce").astype("Int64")
    index["week"] = pd.to_numeric(index["week"], errors="coerce").astype("Int64")
    # Live manifests carry microsecond-precision timestamps while historical-
    # backfill manifests do not; ISO8601 parsing tolerates both formats in one
    # column instead of pandas inferring (and locking to) just one of them.
    index["snapshot_timestamp_utc"] = pd.to_datetime(
        index["snapshot_timestamp_utc"], utc=True, format="ISO8601"
    )
    index["requested_at_utc"] = pd.to_datetime(
        index["requested_at_utc"], utc=True, format="ISO8601"
    )
    return index.sort_values("snapshot_timestamp_utc", kind="stable").reset_index(drop=True)


def load_decision_quotes(
    root: Path,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    labels: Iterable[str] | None = None,
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load and tag quotes for one capture kind's committed snapshots.

    Every returned row carries an explicit ``capture_kind`` column (backfilled
    from the manifest for live rows, whose ``quotes.parquet`` predates that
    column) plus ``season``/``week``/``decision_label``/
    ``snapshot_timestamp_utc`` from the owning manifest, since those are
    request metadata rather than per-row quote fields.
    """

    tag_columns = [
        "capture_kind",
        "season",
        "week",
        "decision_label",
        "snapshot_timestamp_utc",
    ]
    index = load_snapshot_manifest_index(root)
    selected = index.loc[index["capture_kind"].eq(capture_kind)]
    if labels is not None:
        selected = selected.loc[selected["decision_label"].isin(set(labels))]
    if seasons is not None:
        selected = selected.loc[selected["season"].isin(set(seasons))]
    if selected.empty:
        return pd.DataFrame(columns=[*tag_columns])
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        quotes_path = Path(str(row.dir)) / "quotes.parquet"
        if not quotes_path.is_file():
            continue
        quotes = pd.read_parquet(quotes_path)
        if "capture_kind" not in quotes.columns:
            quotes["capture_kind"] = row.capture_kind
        else:
            quotes["capture_kind"] = quotes["capture_kind"].fillna(row.capture_kind)
        quotes["season"] = row.season
        quotes["week"] = row.week
        quotes["decision_label"] = row.decision_label
        quotes["snapshot_timestamp_utc"] = row.snapshot_timestamp_utc
        frames.append(quotes)
    if not frames:
        return pd.DataFrame(columns=[*tag_columns])
    combined = pd.concat(frames, ignore_index=True)
    combined["commence_time_utc"] = pd.to_datetime(combined["commence_time_utc"], utc=True)
    combined["observed_at_utc"] = pd.to_datetime(combined["observed_at_utc"], utc=True)
    return combined


# ---------------------------------------------------------------------------
# 2. Consensus, pairing table, and the monotone-timeline contract
# ---------------------------------------------------------------------------

_CONSENSUS_REQUIRED_COLUMNS = (
    "nflverse_game_id",
    "season",
    "week",
    "decision_label",
    "capture_kind",
    "market",
    "outcome_side",
    "line",
    "price",
    "home_spread_line",
    "bookmaker_key",
    "observed_at_utc",
    "commence_time_utc",
    "snapshot_timestamp_utc",
)


def decision_market_consensus(quotes: pd.DataFrame) -> pd.DataFrame:
    """Per-(game, decision label, market, side) cross-book consensus.

    Restricted to matched games (a resolved ``nflverse_game_id``) and to
    quotes observed strictly before that specific game's kickoff -- a
    decision-label snapshot captured for an upcoming week can still contain a
    later game's board, and a Thursday game's post-kickoff Saturday/Sunday
    snapshots must never count as one of its pregame quotes even though they
    are pregame for the rest of that week's slate.
    """

    missing = sorted(set(_CONSENSUS_REQUIRED_COLUMNS).difference(quotes.columns))
    if missing:
        raise DataContractError(f"Decision quotes are missing columns: {', '.join(missing)}")
    if quotes.empty:
        return pd.DataFrame(
            columns=[
                "nflverse_game_id",
                "season",
                "week",
                "decision_label",
                "market",
                "outcome_side",
                "books",
                "consensus_line",
                "line_min",
                "line_max",
                "line_std",
                "consensus_price",
                "snapshot_timestamp_utc",
            ]
        )
    working = quotes.copy()
    working = working.loc[working["nflverse_game_id"].notna()]
    working = working.loc[working["observed_at_utc"].lt(working["commence_time_utc"])]
    if working.empty:
        return decision_market_consensus(quotes.iloc[0:0])
    working["line_value"] = np.where(
        working["market"].eq("spreads"),
        working["home_spread_line"],
        working["line"],
    )
    # A book should contribute one quote per (game, label, market, side); keep
    # its latest if a capture ever contained more than one (defensive only --
    # the backfill executor's own collision guard already prevents duplicate
    # (season, week, label) snapshots).
    deduped = (
        working.sort_values("observed_at_utc")
        .groupby(
            ["nflverse_game_id", "decision_label", "market", "outcome_side", "bookmaker_key"],
            as_index=False,
            dropna=False,
        )
        .tail(1)
    )
    grouped = deduped.groupby(
        ["nflverse_game_id", "season", "week", "decision_label", "market", "outcome_side"],
        as_index=False,
        dropna=False,
    ).agg(
        books=("bookmaker_key", "nunique"),
        consensus_line=("line_value", "median"),
        line_min=("line_value", "min"),
        line_max=("line_value", "max"),
        line_std=("line_value", "std"),
        consensus_price=("price", "median"),
        snapshot_timestamp_utc=("snapshot_timestamp_utc", "max"),
    )
    # Sportsbooks routinely post a game's board more than a week ahead of
    # kickoff, so a backfill request for an EARLIER week can also capture a
    # LATER week's game (an "early sighting"): the same game and decision
    # label can then appear twice, tagged under two different (season, week)
    # requests. A game's true (season, week) request is always the larger one
    # (NFL weeks only look forward), so keep only that one per (game, label,
    # market, side) -- this also keeps every group a single row, which is what
    # both the pairing table and the monotone-timeline contract assume.
    grouped = (
        grouped.sort_values(["season", "week"])
        .groupby(
            ["nflverse_game_id", "decision_label", "market", "outcome_side"],
            as_index=False,
            dropna=False,
        )
        .tail(1)
    )
    return grouped


def assert_monotone_decision_timeline(consensus: pd.DataFrame) -> None:
    """Verify each game's retained pregame snapshots are chronologically ordered.

    Decision labels are fixed offsets before an NFL week's anchor Sunday, so
    once quotes are filtered to strictly pregame (see
    :func:`decision_market_consensus`), the snapshot timestamps a game
    actually has must be non-decreasing in label order. A violation would
    indicate a corrupted manifest or a reordered append to the store.
    """

    if consensus.empty:
        return
    unknown = sorted(set(consensus["decision_label"].dropna()) - set(DECISION_LABEL_ORDER))
    if unknown:
        raise DataContractError(f"Unknown decision labels in consensus table: {', '.join(unknown)}")
    working = consensus[["nflverse_game_id", "decision_label", "snapshot_timestamp_utc"]].copy()
    working["_order"] = working["decision_label"].map(DECISION_LABEL_ORDER)
    bad_games: list[str] = []
    for game_id, group in working.groupby("nflverse_game_id", dropna=True):
        ordered = group.sort_values("_order")
        timestamps = ordered["snapshot_timestamp_utc"].to_numpy()
        if len(timestamps) > 1 and not pd.Series(timestamps).is_monotonic_increasing:
            bad_games.append(str(game_id))
    if bad_games:
        raise DataContractError(
            f"Non-monotone decision-label timeline for games: {', '.join(bad_games[:5])}"
        )


def build_pairing_table(
    root: Path,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    labels: Iterable[str] | None = None,
    seasons: Iterable[int] | None = None,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long per-(game, decision label) pairing table (BUILD item 1).

    One row per game and decision label with the consensus home spread (plus
    book count and dispersion), the total line, and moneyline prices where
    present. Every row carries ``capture_kind`` explicitly.

    Sportsbooks routinely post a game's board more than a week ahead of
    kickoff, so a backfill request planned for an EARLIER week can also
    capture a LATER week's still-pregame game (an "early sighting"): the same
    game and decision label can then be tagged under two different (season,
    week) requests, or -- when the game's own true-week checkpoint for that
    label falls after its kickoff and is correctly excluded (see
    :func:`decision_market_consensus`) -- under only the wrong, earlier week.
    When ``schedule`` (needs ``game_id``/``season``/``week``) is supplied,
    quotes are restricted to each game's own scheduled (season, week) before
    aggregation, which removes these mislabeled sightings; without it, this
    correction is skipped (useful for small fixtures that never span weeks).
    """

    quotes = load_decision_quotes(root, capture_kind=capture_kind, labels=labels, seasons=seasons)
    if schedule is not None and not quotes.empty:
        required = {"game_id", "season", "week"}
        missing = sorted(required.difference(schedule.columns))
        if missing:
            raise DataContractError(f"Pairing schedule is missing columns: {', '.join(missing)}")
        true_week = (
            schedule[["game_id", "season", "week"]]
            .drop_duplicates("game_id")
            .rename(
                columns={
                    "game_id": "nflverse_game_id",
                    "season": "_true_season",
                    "week": "_true_week",
                }
            )
        )
        quotes = quotes.merge(true_week, on="nflverse_game_id", how="left")
        quotes = quotes.loc[
            quotes["season"].eq(quotes["_true_season"]) & quotes["week"].eq(quotes["_true_week"])
        ].drop(columns=["_true_season", "_true_week"])
    empty_columns = [
        "game_id",
        "season",
        "week",
        "decision_label",
        "capture_kind",
        "snapshot_timestamp_utc",
        "home_spread",
        "spread_books",
        "spread_min",
        "spread_max",
        "spread_std",
        "total_line",
        "total_books",
        "home_moneyline",
        "away_moneyline",
        "moneyline_books",
    ]
    if quotes.empty:
        return pd.DataFrame(columns=empty_columns)
    consensus = decision_market_consensus(quotes)
    assert_monotone_decision_timeline(consensus)

    spreads = consensus.loc[
        consensus["market"].eq("spreads") & consensus["outcome_side"].eq("HOME")
    ][
        [
            "nflverse_game_id",
            "season",
            "week",
            "decision_label",
            "books",
            "consensus_line",
            "line_min",
            "line_max",
            "line_std",
            "snapshot_timestamp_utc",
        ]
    ].rename(
        columns={
            "books": "spread_books",
            "consensus_line": "home_spread",
            "line_min": "spread_min",
            "line_max": "spread_max",
            "line_std": "spread_std",
        }
    )
    keys = ["nflverse_game_id", "season", "week", "decision_label"]
    totals = consensus.loc[consensus["market"].eq("totals") & consensus["outcome_side"].eq("OVER")][
        [*keys, "books", "consensus_line"]
    ].rename(columns={"books": "total_books", "consensus_line": "total_line"})
    home_ml = consensus.loc[consensus["market"].eq("h2h") & consensus["outcome_side"].eq("HOME")][
        [*keys, "books", "consensus_price"]
    ].rename(columns={"books": "moneyline_books", "consensus_price": "home_moneyline"})
    away_ml = consensus.loc[consensus["market"].eq("h2h") & consensus["outcome_side"].eq("AWAY")][
        [*keys, "consensus_price"]
    ].rename(columns={"consensus_price": "away_moneyline"})

    pairing = spreads.merge(totals, on=keys, how="left")
    pairing = pairing.merge(home_ml, on=keys, how="left")
    pairing = pairing.merge(away_ml, on=keys, how="left")
    pairing = pairing.rename(columns={"nflverse_game_id": "game_id"})
    pairing["capture_kind"] = capture_kind
    return (
        pairing[empty_columns]
        .sort_values(
            ["game_id", "decision_label"],
            key=lambda s: s.map(DECISION_LABEL_ORDER) if s.name == "decision_label" else s,
        )
        .reset_index(drop=True)
    )


def close_reference_table(pairing: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Per-game closing spread: store close (preferred) or schedule close (fallback).

    Prefers ``sun_late_close`` over ``sun_early_close`` from the store; when
    neither is present for a game, falls back to the nflverse schedule's
    ``spread_line`` with an explicit ``close_source`` column so a caller can
    always tell which close a CLV figure was measured against.
    """

    required = {"game_id", "spread_line"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Schedule close is missing columns: {', '.join(missing)}")
    candidates = pairing.loc[pairing["decision_label"].isin(CLOSE_LABEL_PRIORITY)][
        ["game_id", "decision_label", "home_spread", "spread_books"]
    ].copy()
    candidates["_priority"] = candidates["decision_label"].map(
        {label: index for index, label in enumerate(CLOSE_LABEL_PRIORITY)}
    )
    best = (
        candidates.sort_values("_priority")
        .groupby("game_id", as_index=False)
        .first()
        .rename(
            columns={
                "home_spread": "close_home_spread",
                "decision_label": "close_source",
                "spread_books": "close_books",
            }
        )
    )[["game_id", "close_home_spread", "close_source", "close_books"]]

    schedule_close = schedule[["game_id", "spread_line"]].drop_duplicates("game_id")
    merged = schedule_close.merge(best, on="game_id", how="left")
    missing_store = merged["close_home_spread"].isna()
    merged.loc[missing_store, "close_home_spread"] = merged.loc[missing_store, "spread_line"]
    merged.loc[missing_store, "close_source"] = "schedule_close"
    merged.loc[missing_store, "close_books"] = 0
    merged["close_books"] = merged["close_books"].astype(int)
    return merged[["game_id", "close_home_spread", "close_source", "close_books"]]


_SPREAD_PRICE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "decision_label",
    "capture_kind",
    "snapshot_timestamp_utc",
    "home_spread_price",
    "home_spread_price_books",
    "away_spread_price",
    "away_spread_price_books",
)


def spread_price_consensus_table(
    root: Path,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    labels: Iterable[str] | None = None,
    seasons: Iterable[int] | None = None,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-(game, decision label) consensus spread PRICE, home and away sides.

    Additive sibling to :func:`build_pairing_table`, built for MKT-03
    (no-vig diagnostics, ``docs/novig_diagnostics.md``).
    :func:`decision_market_consensus` already computes a per-(game, label,
    market, side) ``consensus_price`` for the ``spreads`` market -- exactly
    what a no-vig ATS probability needs -- but :func:`build_pairing_table`
    only carries the spread's LINE through (``home_spread``/
    ``spread_min``/``spread_max``/``spread_std``), never the PRICE, so this
    value is computed inside ``decision_market_consensus`` and then silently
    dropped every time a pairing table is built.

    This function surfaces it without touching :func:`build_pairing_table`'s
    code or output. The loading, true-week correction, and monotone-timeline
    steps below deliberately duplicate that function's own steps rather than
    factor them into a shared helper, so ``build_pairing_table`` has zero
    code-path overlap with this addition -- its frozen output needs no
    re-verification beyond a plain diff of this file (which shows no lines
    of :func:`build_pairing_table` changed).

    Same join keys as :func:`build_pairing_table`
    (``["game_id", "season", "week", "decision_label", "capture_kind"]``), so
    the two can be merged directly to attach price alongside line.
    """

    quotes = load_decision_quotes(root, capture_kind=capture_kind, labels=labels, seasons=seasons)
    if schedule is not None and not quotes.empty:
        required = {"game_id", "season", "week"}
        missing = sorted(required.difference(schedule.columns))
        if missing:
            raise DataContractError(f"Pairing schedule is missing columns: {', '.join(missing)}")
        true_week = (
            schedule[["game_id", "season", "week"]]
            .drop_duplicates("game_id")
            .rename(
                columns={
                    "game_id": "nflverse_game_id",
                    "season": "_true_season",
                    "week": "_true_week",
                }
            )
        )
        quotes = quotes.merge(true_week, on="nflverse_game_id", how="left")
        quotes = quotes.loc[
            quotes["season"].eq(quotes["_true_season"]) & quotes["week"].eq(quotes["_true_week"])
        ].drop(columns=["_true_season", "_true_week"])
    if quotes.empty:
        return pd.DataFrame(columns=list(_SPREAD_PRICE_COLUMNS))
    consensus = decision_market_consensus(quotes)
    assert_monotone_decision_timeline(consensus)

    keys = ["nflverse_game_id", "season", "week", "decision_label"]
    home = consensus.loc[consensus["market"].eq("spreads") & consensus["outcome_side"].eq("HOME")][
        [*keys, "books", "consensus_price", "snapshot_timestamp_utc"]
    ].rename(columns={"books": "home_spread_price_books", "consensus_price": "home_spread_price"})
    away = consensus.loc[consensus["market"].eq("spreads") & consensus["outcome_side"].eq("AWAY")][
        [*keys, "books", "consensus_price"]
    ].rename(columns={"books": "away_spread_price_books", "consensus_price": "away_spread_price"})

    table = home.merge(away, on=keys, how="outer")
    table["capture_kind"] = capture_kind
    table = table.rename(columns={"nflverse_game_id": "game_id"})
    return (
        table[list(_SPREAD_PRICE_COLUMNS)]
        .sort_values(
            ["game_id", "decision_label"],
            key=lambda s: s.map(DECISION_LABEL_ORDER) if s.name == "decision_label" else s,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 3. CLV metric harness
# ---------------------------------------------------------------------------

_PICK_REQUIRED_COLUMNS = ("game_id", "side", "decision_label")


def score_clv(
    picks: pd.DataFrame,
    pairing: pd.DataFrame,
    close_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Signed CLV in points for a pick stream (BUILD item 2).

    ``picks`` needs ``game_id``, ``side`` (``HOME``/``AWAY``), and
    ``decision_label`` (the moment the decision line is read from). CLV is the
    line move from that moment's consensus toward (positive) or away from
    (negative) the picked side, measured against the close from
    :func:`close_reference_table`, whose ``close_source`` column is carried
    through unchanged.
    """

    missing = sorted(set(_PICK_REQUIRED_COLUMNS).difference(picks.columns))
    if missing:
        raise DataContractError(f"CLV picks are missing columns: {', '.join(missing)}")
    unknown_sides = sorted(set(picks["side"]) - {"HOME", "AWAY"})
    if unknown_sides:
        raise ValueError(f"CLV picks contain unsupported sides: {', '.join(unknown_sides)}")

    decision_spreads = pairing[["game_id", "decision_label", "home_spread", "spread_books"]].rename(
        columns={"home_spread": "decision_home_spread", "spread_books": "decision_books"}
    )
    scored = picks.merge(decision_spreads, on=["game_id", "decision_label"], how="left")
    scored = scored.merge(close_reference, on="game_id", how="left")
    direction = scored["side"].map({"HOME": 1.0, "AWAY": -1.0})
    scored["clv_points"] = direction * (
        scored["close_home_spread"] - scored["decision_home_spread"]
    )
    if not {"season", "week"}.issubset(scored.columns):
        week_lookup = pairing[["game_id", "season", "week"]].drop_duplicates("game_id")
        scored = scored.merge(week_lookup, on="game_id", how="left")
    return scored


def clv_summary(scored: pd.DataFrame) -> dict[str, float]:
    """Sufficient-statistic-free summary of a scored CLV frame (metric_fn shape)."""

    values = pd.to_numeric(scored["clv_points"], errors="coerce").dropna()
    n = float(len(values))
    return {
        "n": n,
        "mean_clv_points": float(values.mean()) if n else float("nan"),
        "median_clv_points": float(values.median()) if n else float("nan"),
        "positive_clv_rate": float((values > 0.0).mean()) if n else float("nan"),
    }


def week_blocked_bootstrap(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict[str, float]],
    *,
    block: BootstrapBlock = "week",
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260816,
) -> pd.DataFrame:
    """Resample whole NFL weeks or seasons and recompute an arbitrary metric.

    Neither ``nfl_ats.evaluation`` nor ``nfl_ats.reporting``/``nfl_ats.outcomes``
    exposes a block-bootstrap utility for an arbitrary metric function:
    ``nfl_ats.evaluation`` has no bootstrap code at all, and
    ``reporting.block_bootstrap_intervals`` /
    ``outcomes.outcome_bootstrap_intervals`` are both hard-wired to the ATS
    classification prediction schema (``home_cover`` /
    ``home_cover_probability`` / ``method`` columns), which CLV and pilot
    metrics do not have. This generalizes the same whole-block resampling
    algorithm those functions use (group indices, draw block indices with a
    seeded RNG, recompute the metric on the concatenated rows) rather than
    inventing a different resampling design.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    group_columns = ["season", "week"] if block == "week" else ["season"]
    missing = sorted(set(group_columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"Bootstrap frame is missing columns: {', '.join(missing)}")
    valid = frame.dropna(subset=["clv_points"]) if "clv_points" in frame.columns else frame
    if valid.empty:
        raise ValueError("No rows available to bootstrap")
    grouped_indices = list(valid.groupby(group_columns, sort=False, dropna=False).indices.values())
    if not grouped_indices:
        raise ValueError("Bootstrap frame contains no blocks")

    estimate = metric_fn(valid)
    metric_names = list(estimate)
    draws = np.empty((samples, len(metric_names)), dtype=float)
    generator = np.random.default_rng(seed)
    for sample_index in range(samples):
        selected = generator.integers(0, len(grouped_indices), size=len(grouped_indices))
        positions = np.concatenate([grouped_indices[index] for index in selected])
        sampled = metric_fn(valid.iloc[positions])
        draws[sample_index] = [sampled[name] for name in metric_names]

    tail = (1.0 - confidence) / 2.0
    lower = np.quantile(draws, tail, axis=0)
    upper = np.quantile(draws, 1.0 - tail, axis=0)
    return pd.DataFrame(
        {
            "metric": metric_names,
            "estimate": [estimate[name] for name in metric_names],
            "lower": lower,
            "upper": upper,
            # Continuous evidence alongside the interval endpoints: the
            # fraction of blocked resamples in which the metric is positive.
            "probability_positive": np.mean(draws > 0.0, axis=0),
            "confidence": confidence,
            "block": block,
            "samples": samples,
        }
    )


# ---------------------------------------------------------------------------
# 4. Predeclared MKT-06 pilot: frozen feature construction
# ---------------------------------------------------------------------------


def key_number_distance(
    spread: pd.Series, key_numbers: tuple[float, ...] = KEY_NUMBERS
) -> pd.Series:
    """Distance from a home spread to the nearest key number, folded by magnitude.

    Defined as ``min(|abs(spread) - k| for k in key_numbers)``: a 3-point key
    number matters the same way whichever side is favored, so distance is
    measured from the spread's magnitude rather than its signed value. This
    exact definition is not dictated by the frozen feature name
    ("distance of tue_open to nearest key number (3, 7)"); it is recorded here
    and in the pilot report as part of freezing the spec.
    """

    magnitude = spread.abs()
    distances = pd.concat([(magnitude - k).abs() for k in key_numbers], axis=1)
    return distances.min(axis=1)


def resolve_active_model_config(artifacts_root: Path) -> dict[str, Any]:
    """Load the active market-residual model's configuration, or a documented fallback.

    Prefers ``artifacts/active_ats_model.json`` when present (it is a
    generated, gitignored artifact and may legitimately be absent -- see
    AGENTS.md); falls back to the configuration recorded from the main
    checkout on 2026-08-16 (feature_profile="player", ridge, alpha 10).
    """

    manifest = load_active_ats_model(artifacts_root)
    if manifest is None or manifest.get("method") != "market_residual":
        return dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    ridge_alpha = manifest.get("ridge_alpha")
    return {
        "feature_profile": manifest.get(
            "feature_profile", _ACTIVE_MODEL_FALLBACK_CONFIG["feature_profile"]
        ),
        "regressor": manifest.get("regressor", _ACTIVE_MODEL_FALLBACK_CONFIG["regressor"]),
        "ridge_alpha": float(ridge_alpha) if ridge_alpha is not None else 10.0,
        "target": "market_residual",
    }


def active_model_residual_at_opener(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    feature_profile: MarginFeatureProfile = "player",
    model_name: str = "ridge",
    ridge_alpha: float = 10.0,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """Weekly-refit active-model-equivalent residual, evaluated at the opener.

    For each (season, week) present in ``targets``, fits a fresh
    ``target="market_residual"`` model (``nfl_ats.margin.fit_margin_model``,
    the same estimator-fitting machinery ``nfl_ats.outcomes.walk_forward_outcomes``
    uses to produce the active model's historical evaluation) on completed
    games strictly before that week's first kickoff, then scores that week's
    target games with ``spread_line`` overridden to the paired opener value.

    Known, documented limitation (task-specified, not fixed here): every
    other input to this model -- including market-derived features built from
    closing prices/odds -- was fit, and is evaluated for every *other* row,
    relative to the CLOSE. Only the ``spread_line`` input driving this
    model's base/ridge term is substituted with the opener; the model was
    never refit against opener-time market state. The resulting residual is
    therefore an approximation of "what would the active model say if the
    market had stopped moving at the opener", not a faithful reconstruction of
    an opener-time model.

    Games in weeks with fewer than ``min_train_games`` completed training
    rows are silently dropped (insufficient chronological history); the
    caller can detect this by comparing row counts against ``targets``.
    """

    feature_columns = margin_feature_columns("market_residual", feature_profile)
    required = {"game_id", "gameday", "result", "ats_margin", "spread_line", *feature_columns}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Active-model features are missing columns: {', '.join(missing)}")
    target_required = {"game_id", "season", "week", "tue_open_home_spread"}
    missing_targets = sorted(target_required.difference(targets.columns))
    if missing_targets:
        raise DataContractError(
            f"Active-model targets are missing columns: {', '.join(missing_targets)}"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    rows: list[pd.DataFrame] = []
    for (season, week), group in targets.groupby(["season", "week"], sort=True):
        week_game_ids = set(group["game_id"])
        week_rows = frame.loc[frame["game_id"].isin(week_game_ids)]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=model_name,
            feature_profile=feature_profile,
            ridge_alpha=ridge_alpha,
        )
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread"]], on="game_id", how="inner"
        ).copy()
        scoring["spread_line"] = scoring["tue_open_home_spread"]
        predicted = model.predict(scoring)
        rows.append(
            pd.DataFrame(
                {
                    "game_id": scoring["game_id"].to_numpy(),
                    "season": season,
                    "week": week,
                    "active_model_residual_at_opener": predicted[
                        "predicted_market_residual"
                    ].to_numpy(),
                }
            )
        )
    if not rows:
        raise ValueError(
            "No target week had at least min_train_games completed training rows before it"
        )
    return pd.concat(rows, ignore_index=True)


def build_pilot_frame(
    root: Path,
    features: pd.DataFrame,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """Assemble the frozen target and five frozen features for every paired game.

    A game is included only when it has both a ``tue_open`` consensus spread
    and a resolvable close (store or schedule fallback) and enough prior
    training history for :func:`active_model_residual_at_opener`. Missing
    weeks in the archive simply shrink this frame; callers should compare its
    season/week coverage against the frozen protocol's requirements and
    report gaps rather than filling them in.
    """

    required = {"game_id", "season", "week", "rest_diff", "spread_line", "gameday"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Pilot feature table is missing columns: {', '.join(missing)}")

    pairing = build_pairing_table(
        root,
        capture_kind=capture_kind,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    if pairing.empty:
        return pd.DataFrame(
            columns=[*FROZEN_PILOT_FEATURES, "game_id", "season", "target_close_minus_open"]
        )
    close = close_reference_table(pairing, features)

    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})
    frame = tue_open.merge(close, on="game_id", how="inner")
    frame["target_close_minus_open"] = frame["close_home_spread"] - frame["tue_open_home_spread"]
    frame["tue_open_key_number_distance"] = key_number_distance(frame["tue_open_home_spread"])

    rest = features[["game_id", "rest_diff"]].drop_duplicates("game_id")
    frame = frame.merge(rest, on="game_id", how="left")

    config = active_model_config or dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    residual = active_model_residual_at_opener(
        features,
        frame[["game_id", "season", "week", "tue_open_home_spread"]],
        feature_profile=config["feature_profile"],
        model_name=config["regressor"],
        ridge_alpha=config["ridge_alpha"],
        min_train_games=min_train_games,
    )
    frame = frame.merge(
        residual[["game_id", "active_model_residual_at_opener"]], on="game_id", how="inner"
    )
    return frame.reset_index(drop=True)


def fit_pilot_model(train_frame: pd.DataFrame) -> Pipeline:
    """Fit the frozen ridge(alpha=10, no tuning) close-prediction pilot model."""

    missing = sorted(
        set(FROZEN_PILOT_FEATURES)
        .union({"target_close_minus_open"})
        .difference(train_frame.columns)
    )
    if missing:
        raise DataContractError(f"Pilot training frame is missing columns: {', '.join(missing)}")
    estimator = make_margin_estimator("ridge", ridge_alpha=FROZEN_PILOT_RIDGE_ALPHA)
    estimator.fit(
        train_frame.loc[:, list(FROZEN_PILOT_FEATURES)],
        train_frame["target_close_minus_open"],
    )
    return estimator


def evaluate_pilot(estimator: Pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    """Direction accuracy, MAE, and their baselines for one season slice.

    "No movement" baseline: always predicts zero close-vs-open movement. Its
    MAE is therefore the mean absolute actual movement; its "direction
    accuracy" is the base rate of games whose close exactly equalled the
    opener (the only games a constant-zero prediction gets right).
    """

    if frame.empty:
        raise ValueError("Cannot evaluate the pilot model on an empty frame")
    actual = frame["target_close_minus_open"].to_numpy(dtype=float)
    predicted = np.asarray(
        estimator.predict(frame.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float
    )

    movers = actual != 0.0
    direction_matches = np.sign(predicted[movers]) == np.sign(actual[movers])
    direction_accuracy = float(direction_matches.mean()) if movers.any() else float("nan")
    direction_n = int(movers.sum())

    baseline_predicted = np.zeros_like(actual)
    mae_model = float(np.mean(np.abs(actual - predicted)))
    mae_baseline = float(np.mean(np.abs(actual - baseline_predicted)))
    no_movement_direction_accuracy = float((~movers).mean())

    return {
        "n_games": len(frame),
        "direction_accuracy": direction_accuracy,
        "direction_n_movers": direction_n,
        "direction_accuracy_vs_50pct": direction_accuracy - 0.5 if movers.any() else float("nan"),
        "no_movement_baseline_direction_accuracy": no_movement_direction_accuracy,
        "mae_model": mae_model,
        "mae_no_movement_baseline": mae_baseline,
        "mae_improvement_vs_baseline": mae_baseline - mae_model,
    }


def threshold_policy_clv(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-game realized CLV of "bet the opener side the model favors, |pred| >= threshold".

    Realized CLV equals ``target_close_minus_open`` (HOME side) or its
    negation (AWAY side), because the decision spread is the opener and the
    close is the same close used to build the target -- this is exactly what
    :func:`score_clv` would produce for these picks, computed directly here to
    avoid re-deriving the pairing/close-reference join for a quantity already
    on hand.
    """

    if len(predicted) != len(frame):
        raise ValueError("predicted must have one value per row of frame")
    side = np.where(
        predicted >= threshold, "HOME", np.where(predicted <= -threshold, "AWAY", "PASS")
    )
    actual = frame["target_close_minus_open"].to_numpy(dtype=float)
    clv_points = np.where(side == "HOME", actual, np.where(side == "AWAY", -actual, np.nan))
    result = frame[["game_id", "season", "week"]].copy()
    result["side"] = side
    result["predicted_move"] = predicted
    result["clv_points"] = clv_points
    return result.loc[result["side"].ne("PASS")].reset_index(drop=True)


def run_predeclared_pilot(
    root: Path,
    features: pd.DataFrame,
    *,
    protocol: PilotSplit = FROZEN_PILOT_PROTOCOL,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260816,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the frozen train/validate/test protocol; raise a labeled blocker if any split is empty.

    Never substitutes a different season split when the frozen one is
    unavailable -- callers must catch :class:`PilotProtocolBlocked` and report
    it, per this task's constraints.
    """

    pilot_frame = build_pilot_frame(
        root,
        features,
        capture_kind=capture_kind,
        active_model_config=active_model_config,
        min_train_games=min_train_games,
    )
    available_seasons = (
        sorted(pilot_frame["season"].unique().tolist()) if not pilot_frame.empty else []
    )
    coverage = pilot_frame.groupby("season").size().to_dict() if not pilot_frame.empty else {}

    train = pilot_frame.loc[
        pilot_frame["season"].between(protocol.train_start_season, protocol.train_end_season)
    ]
    validate = pilot_frame.loc[pilot_frame["season"].eq(protocol.validate_season)]
    test = pilot_frame.loc[pilot_frame["season"].eq(protocol.test_season)]

    if train.empty or validate.empty or test.empty:
        raise PilotProtocolBlocked(
            "Frozen protocol requires paired tue_open+close data for train seasons "
            f"{protocol.train_start_season}-{protocol.train_end_season}, validate season "
            f"{protocol.validate_season}, and test season {protocol.test_season}; the archive "
            f"currently has paired coverage for seasons {available_seasons} "
            f"(games per season: {coverage}). "
            f"train_games={len(train)} validate_games={len(validate)} test_games={len(test)}."
        )

    estimator = fit_pilot_model(train)
    validate_metrics = evaluate_pilot(estimator, validate)
    test_metrics = evaluate_pilot(estimator, test)

    test_predicted = np.asarray(
        estimator.predict(test.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float
    )
    policy = threshold_policy_clv(test, test_predicted, threshold=threshold)
    if policy.empty:
        clv_bootstrap = pd.DataFrame(
            columns=["metric", "estimate", "lower", "upper", "confidence", "block", "samples"]
        )
    else:
        clv_bootstrap = week_blocked_bootstrap(
            policy, clv_summary, block="week", samples=bootstrap_samples, seed=bootstrap_seed
        )

    return {
        "protocol": {
            "train_start_season": protocol.train_start_season,
            "train_end_season": protocol.train_end_season,
            "validate_season": protocol.validate_season,
            "test_season": protocol.test_season,
            "frozen_features": list(FROZEN_PILOT_FEATURES),
            "ridge_alpha": FROZEN_PILOT_RIDGE_ALPHA,
            "threshold_points": threshold,
        },
        "coverage": {"available_seasons": available_seasons, "games_per_season": coverage},
        "train_games": len(train),
        "validate": validate_metrics,
        "test": test_metrics,
        "test_threshold_policy_bets": len(policy),
        "test_threshold_policy_clv_bootstrap": clv_bootstrap.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 5. The sign test (report Pilot B)
# ---------------------------------------------------------------------------


def sign_test_pilot_b(
    root: Path,
    features: pd.DataFrame,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Does sign(active-model fair margin - opener) predict sign(close - opener)?

    ``fair_spread - opener == predicted_market_residual`` when the residual
    model is evaluated with the opener supplied as ``spread_line`` (see
    ``nfl_ats.margin.MarginModel.predict``), so this reuses exactly the same
    ``active_model_residual_at_opener`` feature the pilot model consumes.
    Uses every available season (no fixed split); reported once, not iterated.
    """

    pilot_frame = build_pilot_frame(
        root,
        features,
        capture_kind=capture_kind,
        active_model_config=active_model_config,
        min_train_games=min_train_games,
    )
    if pilot_frame.empty:
        raise ValueError("No paired games with a resolvable active-model residual are available")

    working = pilot_frame.loc[
        pilot_frame["active_model_residual_at_opener"].ne(0.0)
        & pilot_frame["target_close_minus_open"].ne(0.0)
    ].copy()
    excluded = len(pilot_frame) - len(working)
    working["correct"] = np.sign(working["active_model_residual_at_opener"]) == np.sign(
        working["target_close_minus_open"]
    )

    def _season_result(rows: pd.DataFrame) -> dict[str, Any]:
        successes = int(rows["correct"].sum())
        trials = len(rows)
        test = binomtest(successes, trials, p=0.5) if trials else None
        interval = test.proportion_ci(confidence_level=confidence) if test is not None else None
        return {
            "games": trials,
            "correct": successes,
            "accuracy": successes / trials if trials else float("nan"),
            "p_value_vs_50pct": test.pvalue if test is not None else float("nan"),
            "confidence_interval": (
                {"lower": interval.low, "upper": interval.high} if interval is not None else None
            ),
        }

    overall = _season_result(working)
    season_values = sorted(int(value) for value in working["season"].unique())
    per_season = {
        season: _season_result(working.loc[working["season"].eq(season)])
        for season in season_values
    }
    return {
        "games_excluded_zero_move_or_zero_signal": excluded,
        "overall": overall,
        "per_season": per_season,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 6. Production wiring: predicted close for an upcoming week (MKT-06)
# ---------------------------------------------------------------------------


class ClosePredictionUnavailable(ValueError):
    """Raised when the target week has no usable Tuesday-opener consensus yet.

    This is the expected weekly resting state until the live Tuesday capture
    lands (and, before the season, for every week): callers report it and
    write no artifact, so the Week Board keeps showing its em-dash column
    rather than a stale or fabricated predicted close.
    """


def upcoming_week(features: pd.DataFrame) -> tuple[int, int]:
    """The earliest (season, week) that still has an unplayed game."""

    required = {"season", "week", "result"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Feature table is missing columns: {', '.join(missing)}")
    unplayed = regular_season_rows(features)
    unplayed = unplayed.loc[unplayed["result"].isna()]
    if unplayed.empty:
        raise ValueError("Feature table has no unplayed games to predict a close for")
    first = unplayed.sort_values(["season", "week"]).iloc[0]
    return int(first["season"]), int(first["week"])


def live_tuesday_openers(root: Path) -> pd.DataFrame:
    """Per-game Tuesday-opener consensus from the store's live captures.

    Live capture manifests carry no ``decision_label`` (labels are a
    historical-backfill request concept), so the ``tue_open`` equivalent is
    derived from observation times instead: pregame quotes observed on each
    game's own-week Tuesday (the most recent UTC Tuesday on or before
    kickoff -- NFL games fall on Thu-Mon, so that is always the Tuesday the
    game's week opened), reduced by
    :func:`nfl_ats.market_data.tuesday_opener_quotes` to the cross-book
    median of each book's earliest such quote.
    """

    columns = ["game_id", "tue_open_home_spread", "opener_books", "opener_observed_at_utc"]
    quotes = load_decision_quotes(root, capture_kind=LIVE_CAPTURE_KIND)
    if quotes.empty:
        return pd.DataFrame(columns=columns)
    spreads = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("HOME")
        & quotes["nflverse_game_id"].notna()
        & quotes["observed_at_utc"].lt(quotes["commence_time_utc"])
    ].copy()
    if spreads.empty:
        return pd.DataFrame(columns=columns)
    days_since_tuesday = (spreads["commence_time_utc"].dt.weekday - 1) % 7
    own_week_tuesday = spreads["commence_time_utc"].dt.normalize() - pd.to_timedelta(
        days_since_tuesday, unit="D"
    )
    own_week = spreads.loc[spreads["observed_at_utc"].dt.normalize().eq(own_week_tuesday)]
    if own_week.empty:
        return pd.DataFrame(columns=columns)
    opener = tuesday_opener_quotes(own_week)
    return opener.rename(
        columns={
            "nflverse_game_id": "game_id",
            "opener_home_spread": "tue_open_home_spread",
            "bookmakers": "opener_books",
            "observed_at_utc": "opener_observed_at_utc",
        }
    )[columns]


def _validate_close_predictions(predictions: pd.DataFrame) -> None:
    """Fail-closed output contract for the Week Board's predicted-close artifact."""

    if predictions["game_id"].duplicated().any():
        raise DataContractError("Close predictions contain duplicate game_id rows")
    values = predictions[
        ["tue_open_home_spread", "predicted_close_minus_open", "predicted_close_home_spread"]
    ].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError("Close predictions contain non-finite values")
    if float(np.abs(predictions["predicted_close_home_spread"]).max()) > 30.0:
        raise DataContractError("Predicted close outside the plausible NFL spread range (|x| > 30)")


def predict_close_for_week(
    root: Path,
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    protocol: PilotSplit = FROZEN_PILOT_PROTOCOL,
    train_capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> dict[str, Any]:
    """Train the frozen pilot on its frozen window and predict one week's closes.

    The estimator is exactly the predeclared MKT-06 pilot: the frozen
    five-feature list, ridge alpha 10, trained on the protocol's train
    seasons only (never extended into the validate/test seasons, so the
    production model stays the audited one). The target week's opener comes
    from live Tuesday captures via :func:`live_tuesday_openers`; a week
    without one raises :class:`ClosePredictionUnavailable`, the expected
    resting state rather than an error to retry.
    """

    required = {"game_id", "season", "week", "rest_diff"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Feature table is missing columns: {', '.join(missing)}")

    # The opener check comes first: "no Tuesday capture yet" is the expected
    # resting state most of every week, and it must not pay for the training
    # rebuild (seasons of weekly active-model refits) just to find that out.
    schedule = features[["game_id", "season", "week", "rest_diff"]].drop_duplicates("game_id")
    target_games = schedule.loc[schedule["season"].eq(season) & schedule["week"].eq(week)]
    target = target_games.merge(live_tuesday_openers(root), on="game_id", how="inner")
    if target.empty:
        raise ClosePredictionUnavailable(
            f"No live Tuesday-opener quotes for season {season} week {week} are in the store; "
            "the predicted close becomes available after that week's Tuesday capture."
        )
    target["tue_open_key_number_distance"] = key_number_distance(target["tue_open_home_spread"])

    train_frame = build_pilot_frame(
        root,
        features,
        capture_kind=train_capture_kind,
        active_model_config=active_model_config,
        min_train_games=min_train_games,
    )
    train = (
        train_frame.loc[
            train_frame["season"].between(protocol.train_start_season, protocol.train_end_season)
        ]
        if not train_frame.empty
        else train_frame
    )
    if train.empty:
        raise PilotProtocolBlocked(
            "Frozen protocol requires paired tue_open+close training data for seasons "
            f"{protocol.train_start_season}-{protocol.train_end_season}; none is under {root}"
        )
    estimator = fit_pilot_model(train)

    config = active_model_config or dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    try:
        residual = active_model_residual_at_opener(
            features,
            target[["game_id", "season", "week", "tue_open_home_spread"]],
            feature_profile=config["feature_profile"],
            model_name=config["regressor"],
            ridge_alpha=config["ridge_alpha"],
            min_train_games=min_train_games,
        )
    except ValueError as error:
        raise ClosePredictionUnavailable(
            f"Cannot rebuild the opener-time residual for season {season} week {week}: {error}"
        ) from error
    target = target.merge(
        residual[["game_id", "active_model_residual_at_opener"]], on="game_id", how="inner"
    )

    predicted = np.asarray(
        estimator.predict(target.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float
    )
    predictions = target[
        [
            "game_id",
            "season",
            "week",
            "tue_open_home_spread",
            "opener_books",
            "opener_observed_at_utc",
        ]
    ].copy()
    predictions["predicted_close_minus_open"] = predicted
    predictions["predicted_close_home_spread"] = (
        predictions["tue_open_home_spread"] + predictions["predicted_close_minus_open"]
    )
    _validate_close_predictions(predictions)
    return {
        "predictions": predictions.reset_index(drop=True),
        "train_games": len(train),
        "train_start_season": protocol.train_start_season,
        "train_end_season": protocol.train_end_season,
    }


# ---------------------------------------------------------------------------
# 7. Routine paper-decision CLV ledger (MKT-04)
# ---------------------------------------------------------------------------

# One row per published, pre-kickoff paper decision. ``decision_home_spread``
# is the spread the published card was priced against; CLV is later measured
# from exactly that number, so a republished card never rewrites an anchor --
# the ledger keeps the FIRST recorded decision per game.
#
# ``is_best_pick`` (POL-10) marks the ONE game per regular-season week the pool
# scores as the Best Pick. It lives here rather than only on the rendered card
# because ``docs/index.html`` is a single current-week page that every publish
# overwrites: without a durable row, week 1's Best Pick stops existing the
# moment week 2 publishes, and the ``sweep_robustness`` ranker -- unresolved,
# not confirmed; its naive +8.68-point delta was mostly tie-break luck and
# the honest edge is ~+0.9 (86 top-1 picks, `docs/best_pick_ranker.md`) --
# never accumulates prospective evidence. See :func:`record_paper_decisions`
# for the write rules.
PAPER_DECISION_COLUMNS: tuple[str, ...] = (
    "recorded_at_utc",
    "forecast_artifact",
    "forecast_created_at_utc",
    "model_id",
    "method",
    "decision_policy_id",
    "decision_policy_fingerprint",
    "game_id",
    "season",
    "week",
    "kickoff",
    "away_team",
    "home_team",
    "model_pick_side",
    "pre_arrest_pick_side",
    "former_policy_pick_side",
    "pick_side",
    "coach_fade_flip",
    "division_revenge_flip",
    "player_arrests_flip",
    "spread_gap_zone_flip",
    "composed_overlay_flip",
    "player_arrests_home_flag",
    "player_arrests_away_flag",
    "player_arrests_snapshot_id",
    "player_arrests_snapshot_fetched_at_utc",
    "player_arrests_safe_index_sha256",
    "schedule_snapshot_id",
    "schedule_parquet_sha256",
    "bet_side",
    "decision_home_spread",
    "edge",
    "is_best_pick",
)

# Ledgers written before POL-10 have every column above except ``is_best_pick``.
# They are read back with the flag defaulted to False, which is also its true
# value: no Best Pick had been recorded when they were written.
_LEGACY_PAPER_DECISION_DEFAULTS: dict[str, Any] = {
    "is_best_pick": False,
    "decision_policy_id": "legacy_model_only",
    "decision_policy_fingerprint": "",
    "coach_fade_flip": False,
    "division_revenge_flip": False,
    "player_arrests_flip": False,
    "spread_gap_zone_flip": False,
    "composed_overlay_flip": False,
    "player_arrests_home_flag": False,
    "player_arrests_away_flag": False,
    "player_arrests_snapshot_id": "",
    "player_arrests_snapshot_fetched_at_utc": pd.NaT,
    "player_arrests_safe_index_sha256": "",
    "schedule_snapshot_id": "",
    "schedule_parquet_sha256": "",
}

_FOUR_OVERLAY_POLICY_ID = "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"

_CLOSE_REFERENCE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "close_home_spread",
    "close_source",
    "close_books",
    "close_observed_at_utc",
)

VALID_PICK_SIDES = frozenset({"HOME", "AWAY"})
VALID_BET_SIDES = frozenset({"HOME", "AWAY", "PASS"})


def paper_decision_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "clv_ledger" / "decisions.parquet"


def load_paper_decisions(artifacts_root: Path) -> pd.DataFrame:
    """The append-only paper-decision ledger (empty frame when none exists)."""

    path = paper_decision_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(PAPER_DECISION_COLUMNS))
    ledger = pd.read_parquet(path)
    for column, default in _LEGACY_PAPER_DECISION_DEFAULTS.items():
        if column not in ledger.columns:
            ledger[column] = default
    if "model_pick_side" not in ledger.columns and "pick_side" in ledger.columns:
        ledger["model_pick_side"] = ledger["pick_side"]
    if "pre_arrest_pick_side" not in ledger.columns and "pick_side" in ledger.columns:
        ledger["pre_arrest_pick_side"] = ledger["pick_side"]
    if "former_policy_pick_side" not in ledger.columns and "pick_side" in ledger.columns:
        ledger["former_policy_pick_side"] = ledger["pick_side"]
    missing = sorted(set(PAPER_DECISION_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(f"Paper-decision ledger is missing columns: {', '.join(missing)}")
    if ledger["game_id"].duplicated().any():
        raise DataContractError(f"Paper-decision ledger contains duplicate game rows: {path}")
    ledger["is_best_pick"] = ledger["is_best_pick"].fillna(False).astype(bool)
    flagged = ledger.loc[ledger["is_best_pick"]]
    if flagged.duplicated(subset=["season", "week"]).any():
        raise DataContractError(
            f"Paper-decision ledger marks more than one Best Pick in a week: {path}"
        )
    composed = ledger["decision_policy_id"].astype(str).eq(_FOUR_OVERLAY_POLICY_ID)
    if composed.any():
        member_flip = (
            ledger.loc[
                composed,
                [
                    "coach_fade_flip",
                    "division_revenge_flip",
                    "player_arrests_flip",
                    "spread_gap_zone_flip",
                ],
            ]
            .astype(bool)
            .any(axis=1)
        )
        declared_flip = ledger.loc[composed, "composed_overlay_flip"].astype(bool)
        observed_flip = (
            ledger.loc[composed, "model_pick_side"]
            .astype(str)
            .ne(ledger.loc[composed, "pick_side"].astype(str))
        )
        if not member_flip.equals(declared_flip) or not observed_flip.equals(declared_flip):
            raise DataContractError(
                "Four-overlay paper-decision rows violate the raw-card OR-union invariant"
            )
    return ledger[list(PAPER_DECISION_COLUMNS)]


def _weekly_best_pick(forecast_directory: Path, card: pd.DataFrame) -> str | None:
    """The week's Best Pick game_id from the forecast's own line sweep, or None.

    Regular season only -- the pool awards a Best Pick per regular-season week
    and no postseason evidence for the ranker exists. Reads the FULL sweep the
    forecast wrote, exactly as the picks pages do; ranking a narrowed sweep
    would score a different, unconfirmed signal (``nfl_ats.best_pick``).
    """

    if "game_type" in card.columns and not card["game_type"].astype(str).eq("REG").all():
        return None
    sweep_path = forecast_directory / "line_sweep.parquet"
    if not sweep_path.is_file():
        return None
    sweep = pd.read_parquet(sweep_path)
    if "method" in sweep.columns and "method" in card.columns:
        sweep = sweep.loc[sweep["method"].isin(set(card["method"].astype(str)))]
    return select_best_pick(card, sweep)


#: How close to a week's earliest kickoff a recording call is allowed to be.
#: The pool locks Tuesday noon ET and the earliest game of a week kicks off
#: Thursday night -- at most a few days later -- so 7 days comfortably covers
#: every real weekly recording (including a Monday catch-up run) while
#: excluding anything that looks like a rehearsal weeks in advance. This is
#: the guard for the 2026-08-18 incident: a rehearsal/test run of the
#: ordinary ``publish-predictions`` command recorded 16 real Week 1 rows on
#: 2026-08-18T01:24:56Z, three weeks before the games it recorded picks for
#: even kick off (docs/prospective_evidence.md, "Known divergence"). Deleting
#: those rows was not a fix -- nothing stopped the exact same command from
#: repopulating the ledger the same way before the real 2026-09-08 lock. This
#: constant is that fix: it is checked inside the recording functions
#: themselves (not only at the CLI layer), so it protects every caller --
#: ``publish-predictions --record-decisions``, ``clv-ledger``, and
#: ``prospective-record`` alike -- regardless of which flag did or did not
#: gate the call. There is deliberately no override; if a legitimate
#: recording somehow needs the window widened, that is a considered change to
#: this constant, not a per-call bypass.
RECORDING_LOCK_WINDOW = timedelta(days=7)


def refuse_if_outside_recording_lock_window(
    kickoffs: pd.Series, recorded_at: pd.Timestamp, *, ledger: str
) -> None:
    """Fail closed when a recording call is far earlier than any real lock.

    ``kickoffs`` is the full card's kickoff timestamps; the check uses the
    earliest one, matching the existing "whole-week pre-kickoff" framing the
    Best Pick rule already uses. A negative gap (kickoff already at or before
    ``recorded_at``) is not a rehearsal -- it is a normal, or even a late,
    recording -- so only a gap LARGER than the window is refused.
    """

    earliest_kickoff = kickoffs.min()
    if pd.isna(earliest_kickoff):
        return
    gap = earliest_kickoff - recorded_at
    if gap > RECORDING_LOCK_WINDOW:
        raise ValueError(
            f"Refusing to record to the {ledger} ledger: this week's earliest kickoff "
            f"({earliest_kickoff.isoformat()}) is {gap.days} days after the recording "
            f"instant ({recorded_at.isoformat()}), more than RECORDING_LOCK_WINDOW "
            f"({RECORDING_LOCK_WINDOW.days} days). This looks like a rehearsal or a "
            "test run made weeks before the week's real Tuesday lock, not the "
            "deliberate recording itself -- the exact shape of the 2026-08-18 incident "
            "(docs/prospective_evidence.md, 'Known divergence'). If this really is the "
            "week's deliberate lock-day recording, the schedule/kickoff data is wrong; "
            "fix that, don't widen this window for one call."
        )


def record_paper_decisions(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> dict[str, Any]:
    """Append the active published card's pre-kickoff picks to the decision ledger.

    Reads the same synchronized weekly forecast that ``publish-predictions``
    publishes (the active manifest's linked artifact), takes its forced ATS
    pick and paper ``bet_side`` for every game whose kickoff is still in the
    future, and appends any game not already in the ledger. Games already
    recorded keep their original decision line -- republishing a card with a
    moved line never rewrites the CLV anchor. Games at or past kickoff are
    counted and skipped, mirroring the frozen-forecast pre-kickoff rule.

    The week's Best Pick (POL-10) is written at the same moment, under three
    rules that together make a backdated Best Pick impossible:

    1. **Whole-week pre-kickoff.** The flag is written only while EVERY game on
       the card is still in the future. The pool locks all picks on Tuesday
       before any game; nominating a Best Pick once Thursday night has been
       played would be choosing with results in hand, so once any game has
       started the week simply gets no Best Pick.
    2. **First write wins.** A week that already carries a flagged row is never
       re-flagged, so republishing cannot move the nomination onto a game that
       has since looked better.
    3. **Exactly one per week.** Enforced on read as well
       (:func:`load_paper_decisions`), so a hand-edited ledger fails loudly.

    Rule 1 is why the flag may land on a row appended by an EARLIER run: the
    decision rows are append-only, but the Best Pick is a separate, one-time,
    still-pre-kickoff write about that week.
    """

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to record decisions from")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    recommendations_path = forecast / "recommendations.csv"
    metadata_path = forecast / "metadata.json"
    if not recommendations_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    card = pd.read_csv(recommendations_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
        "bet_side",
        "edge",
        "method",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Weekly forecast card is missing columns: {', '.join(missing)}")
    method = str(active.get("method"))
    if not card["method"].eq(method).all():
        raise DataContractError("Weekly forecast card contains a method other than the active one")
    if card["game_id"].duplicated().any():
        raise DataContractError("Weekly forecast card contains duplicate games")
    unknown_bets = sorted(set(card["bet_side"].astype(str)) - VALID_BET_SIDES)
    if unknown_bets:
        raise DataContractError(
            f"Weekly card contains invalid bet sides: {', '.join(unknown_bets)}"
        )
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Weekly forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Weekly forecast card has games without a kickoff timestamp")

    recorded_at = _record_instant(now)
    sweep_path = forecast / "line_sweep.parquet"
    sweep = pd.read_parquet(sweep_path) if sweep_path.is_file() else pd.DataFrame()
    # Local import avoids clv -> card_view -> player_arrests -> clv at import time.
    from nfl_ats.card_view import resolve_card_view

    view = resolve_card_view(
        card,
        sweep,
        metadata,
        data_root=data_root,
        now=recorded_at.to_pydatetime(),
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
    )
    raw_card = card.reset_index(drop=True)
    coach_card = view.overlay.overlaid_predictions.reset_index(drop=True)
    former_policy_card = view.arrest_overlay.overlaid_predictions.reset_index(drop=True)
    played_card = view.predictions.reset_index(drop=True)
    model_pick_side = pd.Series(
        np.where(raw_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    pre_arrest_pick_side = pd.Series(
        np.where(coach_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    former_policy_pick_side = pd.Series(
        np.where(former_policy_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    final_pick_side = pd.Series(
        np.where(played_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    composition = view.production_overlay
    if require_fresh_arrest_overlay and composition is None:
        raise DataContractError("Four-overlay production policy was not resolved")
    member_flip_ids: dict[str, set[str]] = {}
    if composition is not None:
        member_flip_ids = {
            member.member_id: set(member.flipped_game_ids) for member in composition.members
        }
    coach_flip_ids = member_flip_ids.get(
        "coach_fade", {flip.game_id for flip in view.overlay.flips}
    )
    division_flip_ids = member_flip_ids.get("division_revenge_tilt", set())
    arrest_flip_ids = member_flip_ids.get("player_arrests_back_side_policy", set())
    spread_gap_flip_ids = member_flip_ids.get("spread_gap_zone_fade", set())
    composed_flip_ids = (
        set(composition.union_flipped_game_ids) if composition is not None else arrest_flip_ids
    )
    decision_policy_id = (
        composition.policy_id if composition is not None else "coach_fade_then_player_arrests_v1"
    )
    decision_policy_fingerprint = composition.policy_fingerprint if composition is not None else ""
    schedule_snapshot_id = ""
    schedule_parquet_sha256 = ""
    if data_root is not None:
        schedule_snapshot = latest_snapshot(data_root / "raw")
        schedule_path = schedule_snapshot.schedules_path
        schedule_snapshot_id = schedule_snapshot.snapshot_id
        schedule_parquet_sha256 = sha256_file(schedule_path)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="paper-decision")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_paper_decisions(artifacts_root)
    already = card["game_id"].isin(set(existing["game_id"].astype(str)))
    fresh = card.loc[pre_kickoff & ~already]

    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    week_already_flagged = bool(
        existing.loc[
            existing["season"].astype(int).eq(season)
            & existing["week"].astype(int).eq(week)
            & existing["is_best_pick"].astype(bool)
        ].shape[0]
    )
    best_pick_id: str | None = None
    if not week_already_flagged and bool(pre_kickoff.all()):
        best_pick_id = view.nomination.active_game_id

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "forecast_artifact": str(active["weekly_forecast"]["artifact"]),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "model_id": str(active.get("model_id")),
            "method": method,
            "decision_policy_id": decision_policy_id,
            "decision_policy_fingerprint": decision_policy_fingerprint,
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "model_pick_side": model_pick_side.loc[fresh.index].astype(str),
            "pre_arrest_pick_side": pre_arrest_pick_side.loc[fresh.index].astype(str),
            "former_policy_pick_side": former_policy_pick_side.loc[fresh.index].astype(str),
            "pick_side": final_pick_side.loc[fresh.index].astype(str),
            "coach_fade_flip": fresh["game_id"].astype(str).isin(coach_flip_ids),
            "division_revenge_flip": fresh["game_id"].astype(str).isin(division_flip_ids),
            "player_arrests_flip": fresh["game_id"].astype(str).isin(arrest_flip_ids),
            "spread_gap_zone_flip": fresh["game_id"].astype(str).isin(spread_gap_flip_ids),
            "composed_overlay_flip": fresh["game_id"].astype(str).isin(composed_flip_ids),
            "player_arrests_home_flag": view.arrest_overlay.home_flags.loc[fresh.index].astype(
                bool
            ),
            "player_arrests_away_flag": view.arrest_overlay.away_flags.loc[fresh.index].astype(
                bool
            ),
            "player_arrests_snapshot_id": str(view.arrest_overlay.snapshot_id or ""),
            "player_arrests_snapshot_fetched_at_utc": view.arrest_overlay.snapshot_fetched_at_utc,
            "player_arrests_safe_index_sha256": str(view.arrest_overlay.safe_index_sha256 or ""),
            "schedule_snapshot_id": schedule_snapshot_id,
            "schedule_parquet_sha256": schedule_parquet_sha256,
            "bet_side": np.where(
                final_pick_side.loc[fresh.index].eq(model_pick_side.loc[fresh.index]),
                fresh["bet_side"].astype(str),
                "PASS",
            ),
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": pd.to_numeric(fresh["edge"], errors="coerce").where(
                final_pick_side.loc[fresh.index].eq(model_pick_side.loc[fresh.index])
            ),
            "is_best_pick": fresh["game_id"].astype(str).eq(str(best_pick_id)),
        }
    )
    if decisions.empty:
        combined = existing.copy()
    elif existing.empty:
        combined = decisions
    else:
        combined = pd.concat([existing, decisions], ignore_index=True)
    # Rule 1's other half: the nomination can land on a row an earlier run
    # appended, which is still a pre-kickoff write because every game on this
    # card is verified to be in the future above.
    best_pick_recorded = False
    flag_written = False
    if best_pick_id is not None and not combined.empty:
        target = combined["game_id"].astype(str).eq(best_pick_id)
        if target.any():
            best_pick_recorded = True
            if not combined.loc[target, "is_best_pick"].fillna(False).astype(bool).any():
                combined.loc[target, "is_best_pick"] = True
                flag_written = True
    if not decisions.empty or flag_written:
        combined["is_best_pick"] = combined["is_best_pick"].fillna(False).astype(bool)
        atomic_parquet(
            combined[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)
    return {
        "season": season,
        "week": week,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "best_pick_game_id": best_pick_id if best_pick_recorded else None,
        "best_pick_recorded": best_pick_recorded,
        "best_pick_already_recorded": week_already_flagged,
        "decision_policy_id": decision_policy_id,
        "decision_policy_fingerprint": decision_policy_fingerprint,
        "coach_flip_count": view.overlay.flip_count,
        "player_arrests_flip_count": len(arrest_flip_ids),
        "division_revenge_flip_count": len(division_flip_ids),
        "spread_gap_zone_flip_count": len(spread_gap_flip_ids),
        "composed_overlay_flip_count": len(composed_flip_ids),
        "player_arrests_snapshot_id": view.arrest_overlay.snapshot_id,
        "player_arrests_safe_index_sha256": view.arrest_overlay.safe_index_sha256,
    }


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def live_close_reference(root: Path, schedule: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    """Per-game closing spread for ledger scoring, from live captures first.

    A live capture's last pre-kickoff cross-book median only becomes a
    *closing* line once the game has kicked off, so live closes are reported
    only for games whose kickoff is at or before ``as_of``. Completed games
    with no live capture fall back to the nflverse schedule close (source
    ``schedule_close``), matching :func:`close_reference_table`'s fallback.
    Games with neither are simply absent -- their ledger rows stay pending.
    """

    required = {"game_id", "spread_line", "result"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Close schedule is missing columns: {', '.join(missing)}")
    cutoff = _record_instant(as_of)

    live = pd.DataFrame(columns=list(_CLOSE_REFERENCE_COLUMNS))
    quotes = load_decision_quotes(root, capture_kind=LIVE_CAPTURE_KIND)
    if not quotes.empty:
        spreads = quotes.loc[
            quotes["market"].eq("spreads")
            & quotes["outcome_side"].eq("HOME")
            & quotes["nflverse_game_id"].notna()
            & quotes["observed_at_utc"].lt(quotes["commence_time_utc"])
            & quotes["commence_time_utc"].le(cutoff)
        ]
        if not spreads.empty:
            last_per_book = (
                spreads.sort_values("observed_at_utc")
                .groupby(["nflverse_game_id", "bookmaker_key"], as_index=False)
                .tail(1)
            )
            live = (
                last_per_book.groupby("nflverse_game_id", as_index=False)
                .agg(
                    close_home_spread=("home_spread_line", "median"),
                    close_books=("bookmaker_key", "nunique"),
                    close_observed_at_utc=("observed_at_utc", "max"),
                )
                .rename(columns={"nflverse_game_id": "game_id"})
            )
            live["close_source"] = "live_store_close"
            live = live[list(_CLOSE_REFERENCE_COLUMNS)]

    completed = schedule.loc[
        schedule["result"].notna(), ["game_id", "spread_line"]
    ].drop_duplicates("game_id")
    fallback_rows = completed.loc[~completed["game_id"].isin(set(live["game_id"].astype(str)))]
    fallback = pd.DataFrame(
        {
            "game_id": fallback_rows["game_id"].astype(str),
            "close_home_spread": pd.to_numeric(fallback_rows["spread_line"], errors="coerce"),
            "close_source": "schedule_close",
            "close_books": 0,
            "close_observed_at_utc": pd.NaT,
        }
    )
    frames = [frame for frame in (live, fallback) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(_CLOSE_REFERENCE_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    return combined[list(_CLOSE_REFERENCE_COLUMNS)].reset_index(drop=True)


def score_paper_ledger(decisions: pd.DataFrame, close_reference: pd.DataFrame) -> pd.DataFrame:
    """Score every ledger decision that has a close; the rest stay pending.

    ``clv_points`` scores the forced pick side for every game;
    ``bet_clv_points`` scores only rows the paper policy actually bet
    (``bet_side`` is ``HOME``/``AWAY``). Both use the ledger's own frozen
    ``decision_home_spread`` as the anchor, never a re-read current line.
    """

    required = {"game_id", "season", "week", "pick_side", "bet_side", "decision_home_spread"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise DataContractError(f"Ledger decisions are missing columns: {', '.join(missing)}")
    if decisions["game_id"].duplicated().any():
        raise DataContractError("Ledger decisions contain duplicate game rows")
    unknown_picks = sorted(set(decisions["pick_side"].astype(str)) - VALID_PICK_SIDES)
    if unknown_picks:
        raise ValueError(f"Ledger contains unsupported pick sides: {', '.join(unknown_picks)}")
    unknown_bets = sorted(set(decisions["bet_side"].astype(str)) - VALID_BET_SIDES)
    if unknown_bets:
        raise ValueError(f"Ledger contains unsupported bet sides: {', '.join(unknown_bets)}")

    reference = (
        close_reference
        if not close_reference.empty
        else pd.DataFrame(columns=list(_CLOSE_REFERENCE_COLUMNS))
    )
    scored = decisions.merge(reference, on="game_id", how="left")
    move = scored["close_home_spread"] - scored["decision_home_spread"]
    pick_direction = scored["pick_side"].map({"HOME": 1.0, "AWAY": -1.0})
    scored["clv_points"] = pick_direction * move
    bet_direction = scored["bet_side"].map({"HOME": 1.0, "AWAY": -1.0})
    scored["bet_clv_points"] = bet_direction * move
    scored["clv_status"] = np.where(scored["clv_points"].notna(), "scored", "pending")
    return scored


# ---------------------------------------------------------------------------
# 8. Opener-graded evaluation of the frozen active model (the pool goal)
# ---------------------------------------------------------------------------

# The primary project goal is beating the OPENING line a pool grades against,
# not the close (see docs/opener_evaluation.md). This section measures the
# frozen active model at both lines on every archived game with a paired
# Tuesday opener and close: one weekly-refit model per week, its residual
# evaluated at each line, each forced pick settled against its own line.


def pick_correct(pick_home: pd.Series, settle_margin: pd.Series) -> pd.Series:
    """1.0 correct / 0.0 wrong / NaN push, for a HOME-pick flag vs a settle margin.

    The repo's ATS convention (FND-04, ``docs/modeling.md``): the settle margin
    is ``result - line``, home covers when it is strictly positive, and a zero
    margin is a push that is excluded from accuracy rather than scored as a
    loss. Public because prospective settlement must use exactly this function
    -- a second implementation is a second chance to get pushes wrong.

    A NaN settle margin (no result, no line) is NOT a push; it is unsettled,
    and this function returns 0.0/1.0 for it because ``NaN > 0`` is False.
    Callers with possibly-unsettled rows must mask them themselves.
    """

    covered_home = settle_margin.gt(0.0)
    correct = np.where(pick_home.astype(bool), covered_home, ~covered_home).astype(float)
    return pd.Series(np.where(settle_margin.eq(0.0), np.nan, correct), index=settle_margin.index)


def opener_pick_evaluation(
    root: Path,
    features: pd.DataFrame,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """Per-game opener/close picks and settlements for the frozen active model.

    For each archived game with both a ``tue_open`` consensus and a
    resolvable close, ONE weekly-refit market-residual model (trained on
    completed games strictly before the week's first kickoff, exactly the
    active model's recipe) is evaluated twice -- once with ``spread_line``
    overridden to the opener, once to the close -- and each resulting forced
    pick is settled against the line it was formed at. A ``movement oracle``
    diagnostic (pick the side the close eventually moved toward, settle at
    the opener) bounds how much accuracy pure line-movement capture is worth.

    The inherited approximation from :func:`active_model_residual_at_opener`
    applies: only ``spread_line`` is swapped to the opener; every other
    feature (including ``total_line``) is close-era. Declared, not fixed.
    """

    config = active_model_config or dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    profile: MarginFeatureProfile = config["feature_profile"]
    feature_columns = margin_feature_columns("market_residual", profile)
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "result",
        "ats_margin",
        "spread_line",
        *feature_columns,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Opener evaluation is missing columns: {', '.join(missing)}")
    # The predeclared opener metric is regular-season only; postseason rows in
    # the feature table must never widen its game set or its training data.
    features = regular_season_rows(features)

    pairing = build_pairing_table(
        root,
        capture_kind=capture_kind,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    if pairing.empty:
        raise ValueError(f"No {capture_kind!r} snapshots with decision quotes under {root}")
    close = close_reference_table(pairing, features)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread", "spread_books"]
    ].rename(columns={"home_spread": "tue_open_home_spread", "spread_books": "opener_books"})
    paired = tue_open.merge(close, on="game_id", how="inner")

    outcomes = features[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()
    if paired.empty:
        raise ValueError("No completed games have both a Tuesday opener and a close")

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    scored_weeks: list[pd.DataFrame] = []
    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=config["regressor"],
            feature_profile=profile,
            ridge_alpha=config["ridge_alpha"],
        )
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread", "close_home_spread"]],
            on="game_id",
            how="inner",
        ).copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        at_close = scoring.copy()
        at_close["spread_line"] = at_close["close_home_spread"]
        scored = scoring[["game_id"]].copy()
        scored["season"] = int(str(season))
        scored["week"] = int(str(week))
        predicted_at_open = model.predict(at_open)
        predicted_at_close = model.predict(at_close)
        scored["residual_at_open"] = predicted_at_open["predicted_market_residual"].to_numpy()
        scored["residual_at_close"] = predicted_at_close["predicted_market_residual"].to_numpy()
        # Production (``pool.py``/``backtest.py``) grades picks with the
        # PROBABILITY rule (``home_cover_probability >= 0.5``), not the sign
        # rule above (``residual > 0``). They usually agree but can diverge:
        # ``home_cover_probability`` is the fraction of the model's empirical
        # out-of-time residual distribution landing above the line, so a rule
        # keyed on its 0.5 crossing is really keyed on that distribution's
        # MEDIAN sitting at the line, while the sign rule is keyed on its MEAN
        # (via the point prediction) doing so -- the two coincide only when
        # the residual distribution is symmetric. Captured here purely as an
        # additional, non-destructive read; the sign-rule columns above are
        # unchanged (see docs/opener_evaluation.md, dated addendum).
        scored["home_cover_probability_at_open"] = predicted_at_open[
            "home_cover_probability"
        ].to_numpy()
        scored["home_cover_probability_at_close"] = predicted_at_close[
            "home_cover_probability"
        ].to_numpy()
        scored_weeks.append(scored)
    if not scored_weeks:
        raise ValueError("No paired week had at least min_train_games completed training rows")
    residuals = pd.concat(scored_weeks, ignore_index=True)

    result = paired.merge(residuals.drop(columns=["season", "week"]), on="game_id", how="inner")
    result["margin_vs_open"] = result["result"] - result["tue_open_home_spread"]
    result["margin_vs_close"] = result["result"] - result["close_home_spread"]
    result["open_move"] = result["close_home_spread"] - result["tue_open_home_spread"]

    result["pick_home_at_open"] = result["residual_at_open"].gt(0.0)
    result["pick_home_at_close"] = result["residual_at_close"].gt(0.0)
    result["correct_at_open"] = pick_correct(result["pick_home_at_open"], result["margin_vs_open"])
    result["correct_at_close"] = pick_correct(
        result["pick_home_at_close"], result["margin_vs_close"]
    )
    # Additive: production's actual pick rule (``pool.py``/``backtest.py``),
    # graded alongside the predeclared sign rule above -- see the comment on
    # ``home_cover_probability_at_open``/``_at_close`` above. Never replaces
    # or renumbers the sign-rule fields; the sign rule remains the
    # predeclared historical record (docs/opener_evaluation.md).
    result["pick_home_at_open_probability_rule"] = result["home_cover_probability_at_open"].ge(0.5)
    result["pick_home_at_close_probability_rule"] = result["home_cover_probability_at_close"].ge(
        0.5
    )
    result["correct_at_open_probability_rule"] = pick_correct(
        result["pick_home_at_open_probability_rule"], result["margin_vs_open"]
    )
    result["correct_at_close_probability_rule"] = pick_correct(
        result["pick_home_at_close_probability_rule"], result["margin_vs_close"]
    )
    oracle_correct = pick_correct(result["open_move"].gt(0.0), result["margin_vs_open"])
    result["oracle_correct_at_open"] = oracle_correct.where(result["open_move"].ne(0.0))
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def opener_evaluation_metrics(scored: pd.DataFrame) -> dict[str, float]:
    """Accuracy at each line plus the paired opener-minus-close delta (metric_fn shape).

    Sign-rule fields (``opener_accuracy``, ``close_accuracy``,
    ``opener_minus_close``, ``opener_vs_coin_flip``) are the predeclared
    historical record (``residual > 0``, docs/opener_evaluation.md) and are
    computed unconditionally, unchanged from before this docstring's
    addition. The ``*_probability_rule`` fields alongside them are an
    additive read of what production (``pool.py``/``backtest.py``) actually
    plays -- ``home_cover_probability >= 0.5`` -- computed only when
    ``scored`` carries the probability-rule columns (i.e. it came from the
    current :func:`opener_pick_evaluation`); older or hand-rolled scored
    frames without them (e.g. ``scripts/ridge_alpha_promotion_eval.py``'s
    line-for-line copy) simply do not get these keys, rather than raising.
    """

    at_open = pd.to_numeric(scored["correct_at_open"], errors="coerce").dropna()
    at_close = pd.to_numeric(scored["correct_at_close"], errors="coerce").dropna()
    both = scored.dropna(subset=["correct_at_open", "correct_at_close"])
    oracle = pd.to_numeric(scored["oracle_correct_at_open"], errors="coerce").dropna()
    metrics = {
        "opener_accuracy": float(at_open.mean()) if len(at_open) else float("nan"),
        "close_accuracy": float(at_close.mean()) if len(at_close) else float("nan"),
        "opener_minus_close": (
            float(both["correct_at_open"].mean() - both["correct_at_close"].mean())
            if len(both)
            else float("nan")
        ),
        "opener_vs_coin_flip": (float(at_open.mean()) - 0.5) if len(at_open) else float("nan"),
        "movement_oracle_accuracy": float(oracle.mean()) if len(oracle) else float("nan"),
    }
    probability_rule_columns = {
        "correct_at_open_probability_rule",
        "correct_at_close_probability_rule",
    }
    if probability_rule_columns.issubset(scored.columns):
        at_open_pr = pd.to_numeric(scored["correct_at_open_probability_rule"], errors="coerce")
        at_open_pr = at_open_pr.dropna()
        at_close_pr = pd.to_numeric(scored["correct_at_close_probability_rule"], errors="coerce")
        at_close_pr = at_close_pr.dropna()
        both_pr = scored.dropna(
            subset=["correct_at_open_probability_rule", "correct_at_close_probability_rule"]
        )
        metrics.update(
            {
                "opener_accuracy_probability_rule": (
                    float(at_open_pr.mean()) if len(at_open_pr) else float("nan")
                ),
                "close_accuracy_probability_rule": (
                    float(at_close_pr.mean()) if len(at_close_pr) else float("nan")
                ),
                "opener_minus_close_probability_rule": (
                    float(
                        both_pr["correct_at_open_probability_rule"].mean()
                        - both_pr["correct_at_close_probability_rule"].mean()
                    )
                    if len(both_pr)
                    else float("nan")
                ),
                "opener_vs_coin_flip_probability_rule": (
                    (float(at_open_pr.mean()) - 0.5) if len(at_open_pr) else float("nan")
                ),
            }
        )
    return metrics
