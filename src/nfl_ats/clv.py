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
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.data import DataContractError
from nfl_ats.margin import (
    MarginFeatureProfile,
    fit_margin_model,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.odds_backfill import DECISION_LABELS, HISTORICAL_CAPTURE_KIND

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
    min_train_games: int = 500,
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
    min_train_games: int = 500,
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
    min_train_games: int = 500,
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
    min_train_games: int = 500,
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
