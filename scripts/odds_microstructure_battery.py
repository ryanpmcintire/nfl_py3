"""Odds microstructure lead-generation battery (measure-only, no registry writes).

Companion to `docs/novig_diagnostics.md`/`src/nfl_ats/novig.py` (MKT-03), which
established two facts this script builds on: (1) 55.50% of archived spread
quote rows are NOT exactly -110 -- price genuinely carries information a
half-point line cannot -- and (2) the pairing pipeline (`build_pairing_table`)
computes and then silently DROPS the spread's consensus PRICE on every call.
Books shade juice where they fear the outcome; that is market *behavior*, the
class where `docs/pool_edge_plan.md`'s ceiling finding says edge lives, as
opposed to team-quality measurement, which is bounded near zero
(`docs/pool_edge_plan.md`, "The standing lesson").

**This is a DIAGNOSTIC / lead-generation screen, not a confirmation look.**
Per the same rule `docs/novig_diagnostics.md` and rule 8 of
`docs/rotation_registry.md` rely on ("CFB and non-reserved seasons stay
free... the registry governs NFL confirmation looks only"), purely
descriptive market measurement that declares no family, draws no
confirmation window, and produces no accept/reject verdict is not itself an
NFL confirmation look. Nothing here calls `nfl_ats.rotation.assign_window`/
`record_look`, and no `registry/*.json` file is written by this script.
Reads only committed local snapshots under `data/market/raw/` via
`nfl_ats.clv`'s loaders -- the paid Odds API is never called. Nothing here
reaches `docs/*.html` (the public-pages guardrail).

## Predeclaration (all cells, before any scoring)

Population base for H1/H2/H4/H5/H6: every archived game with a `tue_open`
consensus spread PRICE 2020-2025 (the same 1,537-ish-game population
`docs/novig_diagnostics.md` measured; exact n reported at run time), built
from `nfl_ats.clv.build_pairing_table` + `nfl_ats.clv.spread_price_consensus_table`
+ `nfl_ats.novig.spread_novig_probabilities` -- identical construction to
`scripts/novig_diagnostics_screen.py`, reused rather than re-derived. Pushes
(`ats_margin == 0`) are excluded from every accuracy cell; a probability of
exactly 0.5 (no lean) is excluded from cells that require a directional
pick. H3 uses a separate, narrower population: seasons 2023-2025 only, where
the archive's `intraday_hourly` capture (measured: 6,966 manifests, only
2023/2024/2025) gives a Wednesday checkpoint the standard 6 decision labels
do not carry (there is no `wed_*` label in `nfl_ats.odds_backfill.DECISION_TIMES`).

1. **H1 -- asymmetric opener juice predicts cover** (task hypothesis 1).
   Cell 1.1: forced-pick accuracy of "predict HOME covers iff
   `no_vig_home_cover_probability > 0.5`" against `home_cover_at_label`,
   week- and season-blocked bootstrap, `P+` vs 50%.
   Cell 1.2: dose-response -- split into fixed terciles of
   `|no_vig_home_cover_probability - 0.5|` (edges from the full sample);
   per-tercile accuracy, and a week-blocked bootstrap of the
   high-minus-low-tercile accuracy difference.
2. **H2 -- cross-book dispersion as an uncertainty / game-selection signal**
   (task hypothesis 2). Two dispersion proxies: `spread_std` (line
   dispersion, already computed by `build_pairing_table`) and a new
   `price_std` (raw HOME-side spread price std across books at `tue_open`,
   computed directly from raw quotes -- `decision_market_consensus` computes
   this for the LINE but not the PRICE; `docs/novig_diagnostics.md` flagged
   this exact gap and left it undone). For each proxy: (a) mean `|ats_margin|`
   by tercile, week-blocked bootstrap of high-minus-low difference; (b) H1's
   juice-lean-rule accuracy by tercile, same bootstrap -- directly answers
   "is the juice signal less reliable in high-disagreement games", the
   Best-Pick-relevant framing.
3. **H3 -- the Tuesday-to-Wednesday drift echo** (task hypothesis 3).
   Cell 3.1: reproduce the project's own movement-oracle formula exactly
   (`nfl_ats.clv.opener_pick_evaluation`, lines ~1963-1974: pick the side the
   line moves toward, settle at the frozen opener) with the terminal point
   swapped from the close to the nearest-to-Wednesday-noon-ET
   `intraday_hourly` snapshot, on the 2023-2025 subsample. Reported against
   two baselines measured on the SAME subsample for an apples-to-apples read:
   the full 2020-2025 population's oracle (sanity check against the
   documented 55.1% read) and the 2023-2025-only full-week oracle. Playability
   is read, not inferred, from `docs/pool_edge_plan.md`'s confirmed pool
   format section (Tuesday-noon lock) before any number is quoted.
4. **H4 -- off-market single-book openers** (task hypothesis 4). Per game
   with >=3 books quoting both spread sides at `tue_open`, find the single
   book whose no-vig home-cover probability diverges most from the
   consensus. Restrict to the subset where the divergent book's LEAN
   (>/< 0.5) disagrees with the consensus's lean (both requiring a defined
   lean, excluding exact 0.5) -- by construction the two picks are exact
   opposites on this subset, so "does the divergent book win" and "does the
   consensus win" are complementary reads of the same bootstrap. Both
   reported, week- and season-blocked, `P+` vs 50%.
5. **H5 (own) -- totals-market juice asymmetry**, the same H1 mechanism
   applied to the `totals` market: forced-pick accuracy of "predict OVER
   iff no-vig over-probability > 0.5" at `tue_open`, graded against that
   label's own `total_line` (same care `docs/novig_diagnostics.md` section
   "which line the outcome is graded against" applies to spreads).
6. **H6 (own) -- market hold (vig level) as an uncertainty signal**, a
   proxy distinct from cross-book dispersion (same mechanism class: a
   cautious/fearful book widens its own margin, not just disagreeing with
   its peers). Cell 6.1: mean `|ats_margin|` by tercile of `spread_hold`
   (already computed by `nfl_ats.novig.spread_novig_probabilities`), week-
   blocked bootstrap of high-minus-low difference. Cell 6.2: H1's rule
   accuracy by hold tercile, same bootstrap.

Bootstrap: `nfl_ats.clv.week_blocked_bootstrap`, `samples=20_000`, seed fixed
below, both `block="week"` (primary) and `block="season"` (secondary, wide
at n=6 or n=3 seasons -- reported for honesty, not leaned on). Per the
project's binding AGENTS.md rule, an interval containing zero is reported as
`probability_positive`, never treated as a rejection; every cell is reported
regardless of what it shows. Coverage/era caveats are stated inline wherever
a cell's population differs from the base 2020-2025 set.

Writes `*.parquet`/`cells_summary.csv`/`metadata.json` to
`artifacts/odds_microstructure/<run_id>/`. Proposes (never executes)
`nfl-ats weak-signals record` commands for any cell whose lean is worth
carrying forward -- printed and written to `metadata.json`, not run.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.clv import (
    build_pairing_table,
    close_reference_table,
    decision_market_consensus,
    load_decision_quotes,
    load_snapshot_manifest_index,
    pick_correct,
    spread_price_consensus_table,
    week_blocked_bootstrap,
)
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_json, atomic_parquet, run_id
from nfl_ats.modeling import regular_season_rows
from nfl_ats.novig import spread_novig_probabilities
from nfl_ats.odds import market_hold, no_vig_probabilities
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data/market/raw"
DEFAULT_FEATURES = REPO / "data/processed/game_features_player.parquet"
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/odds_microstructure"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818

CLOSE_LABEL_PRIORITY: tuple[str, ...] = ("sun_late_close", "sun_early_close")
INTRADAY_LABEL = "intraday_hourly"
INTRADAY_SEASONS: tuple[int, ...] = (2023, 2024, 2025)
EASTERN = ZoneInfo("America/New_York")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _no_vig_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Row-by-row no-vig probability/hold for an arbitrary two-sided price pair.

    Same NaN-guarded pattern as `nfl_ats.novig._apply_no_vig_row_by_row`
    (private to that module, not imported here so this script never reaches
    into another module's private API) -- reuses the SAME audited,
    unit-tested primitives (`nfl_ats.odds.no_vig_probabilities`/
    `market_hold`) rather than re-deriving the math, for column pairs that
    module's public functions don't cover (book-level spread price, totals
    OVER/UNDER price).
    """

    a_values = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b_values = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a_values) & np.isfinite(b_values) & (a_values != 0.0) & (b_values != 0.0)
    probability = np.full(len(a_values), np.nan, dtype=float)
    hold = np.full(len(a_values), np.nan, dtype=float)
    for position in np.flatnonzero(valid):
        a_prob, _ = no_vig_probabilities(a_values[position], b_values[position])
        probability[position] = a_prob
        hold[position] = market_hold(a_values[position], b_values[position])
    return probability, hold


def _true_week_correct(quotes: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Restrict quotes to each game's own scheduled (season, week).

    Same "early sighting" correction `build_pairing_table`/
    `spread_price_consensus_table` apply, duplicated here (not imported) for
    the raw-quote uses (book-level prices, totals, Wednesday intraday) those
    two functions don't cover.
    """

    if quotes.empty:
        return quotes
    required = {"game_id", "season", "week"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Correction schedule is missing columns: {', '.join(missing)}")
    true_week = (
        schedule[["game_id", "season", "week"]]
        .drop_duplicates("game_id")
        .rename(
            columns={"game_id": "nflverse_game_id", "season": "_true_season", "week": "_true_week"}
        )
    )
    merged = quotes.merge(true_week, on="nflverse_game_id", how="left")
    return merged.loc[
        merged["season"].eq(merged["_true_season"]) & merged["week"].eq(merged["_true_week"])
    ].drop(columns=["_true_season", "_true_week"])


def _tercile_edges(values: pd.Series) -> np.ndarray:
    """Fixed tercile edges computed once, reused for point estimate + bootstrap.

    Same pattern as `nfl_ats.novig.calibration_bucket_edges`: recomputing
    edges per bootstrap resample would make the buckets themselves a moving
    target.
    """

    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        raise ValueError("No values to compute tercile edges from")
    edges = np.unique(np.quantile(clean, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
    if len(edges) < 2:
        raise ValueError("Too little variation to form terciles")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _assign_tercile(values: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"), bins=list(edges), include_lowest=True, labels=False
    )


def _book_level_spread_prices(quotes: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, book) with tue_open HOME/AWAY spread price.

    Mirrors `nfl_ats.clv.decision_market_consensus`'s own pregame filter and
    per-book dedup (keep latest `observed_at_utc`) so a book that posted more
    than one quote in a snapshot contributes once -- duplicated rather than
    imported because that function collapses across books (median) and this
    needs the book-level rows the median collapses away.
    """

    working = quotes.loc[quotes["market"].eq("spreads") & quotes["nflverse_game_id"].notna()].copy()
    working = working.loc[working["observed_at_utc"].lt(working["commence_time_utc"])]
    if working.empty:
        return pd.DataFrame(
            columns=["nflverse_game_id", "bookmaker_key", "home_price", "away_price"]
        )
    deduped = (
        working.sort_values("observed_at_utc")
        .groupby(
            ["nflverse_game_id", "outcome_side", "bookmaker_key"], as_index=False, dropna=False
        )
        .tail(1)
    )
    home = deduped.loc[deduped["outcome_side"].eq("HOME")][
        ["nflverse_game_id", "bookmaker_key", "price"]
    ].rename(columns={"price": "home_price"})
    away = deduped.loc[deduped["outcome_side"].eq("AWAY")][
        ["nflverse_game_id", "bookmaker_key", "price"]
    ].rename(columns={"price": "away_price"})
    return home.merge(away, on=["nflverse_game_id", "bookmaker_key"], how="inner")


def _totals_price_consensus(consensus: pd.DataFrame) -> pd.DataFrame:
    """Per-game consensus OVER/UNDER price at whatever label ``consensus`` was built at."""

    keys = ["nflverse_game_id", "season", "week", "decision_label"]
    over = consensus.loc[consensus["market"].eq("totals") & consensus["outcome_side"].eq("OVER")][
        [*keys, "books", "consensus_price"]
    ].rename(columns={"books": "over_price_books", "consensus_price": "over_price"})
    under = consensus.loc[consensus["market"].eq("totals") & consensus["outcome_side"].eq("UNDER")][
        [*keys, "consensus_price"]
    ].rename(columns={"consensus_price": "under_price"})
    return over.merge(under, on=keys, how="inner").rename(columns={"nflverse_game_id": "game_id"})


def _wednesday_noon_snapshots(root: Path) -> pd.DataFrame:
    """One manifest row per (season, week): the intraday_hourly snapshot nearest Wed 12:00 ET.

    [measured] the archive's ``intraday_hourly`` capture only exists for
    2023-2025 and runs hourly across the whole NFL week; there is no
    dedicated Wednesday decision label (``nfl_ats.odds_backfill.DECISION_TIMES``
    has none). Selection is self-contained: within each (season, week)
    group of intraday manifests, filter to local America/New_York weekday
    Wednesday and take the row closest to local noon. Weeks with zero
    Wednesday-local rows are dropped (reported as a coverage count, not
    silently backfilled).
    """

    index = load_snapshot_manifest_index(root)
    intraday = index.loc[
        index["capture_kind"].eq(HISTORICAL_CAPTURE_KIND)
        & index["decision_label"].eq(INTRADAY_LABEL)
        & index["season"].isin(INTRADAY_SEASONS)
    ].copy()
    if intraday.empty:
        return intraday
    local = intraday["snapshot_timestamp_utc"].dt.tz_convert(EASTERN)
    intraday["_local_weekday"] = local.dt.weekday  # Monday=0 .. Wednesday=2
    intraday["_hour_distance_from_noon"] = (local.dt.hour + local.dt.minute / 60.0 - 12.0).abs()
    wednesday = intraday.loc[intraday["_local_weekday"].eq(2)]
    if wednesday.empty:
        return wednesday
    selected = (
        wednesday.sort_values(
            ["season", "week", "_hour_distance_from_noon", "snapshot_timestamp_utc"]
        )
        .groupby(["season", "week"], as_index=False)
        .head(1)
    )
    return selected.drop(columns=["_local_weekday", "_hour_distance_from_noon"])


def _load_selected_snapshots(rows: pd.DataFrame) -> pd.DataFrame:
    """Load only the quotes.parquet files for a pre-filtered manifest subset.

    Same per-row tagging loop `nfl_ats.clv.load_decision_quotes` runs
    internally, applied to a caller-filtered subset instead of a
    capture_kind/label/season filter -- keeps this to the ~50-90 selected
    Wednesday snapshot files instead of loading every one of the archive's
    6,966 `intraday_hourly` manifests.
    """

    tag_columns = ["capture_kind", "season", "week", "decision_label", "snapshot_timestamp_utc"]
    if rows.empty:
        return pd.DataFrame(columns=tag_columns)
    frames: list[pd.DataFrame] = []
    for row in rows.itertuples(index=False):
        quotes_path = Path(str(row.dir)) / "quotes.parquet"
        if not quotes_path.is_file():
            continue
        quotes = pd.read_parquet(quotes_path)
        quotes["capture_kind"] = row.capture_kind
        quotes["season"] = row.season
        quotes["week"] = row.week
        quotes["decision_label"] = row.decision_label
        quotes["snapshot_timestamp_utc"] = row.snapshot_timestamp_utc
        frames.append(quotes)
    if not frames:
        return pd.DataFrame(columns=tag_columns)
    combined = pd.concat(frames, ignore_index=True)
    combined["commence_time_utc"] = pd.to_datetime(combined["commence_time_utc"], utc=True)
    combined["observed_at_utc"] = pd.to_datetime(combined["observed_at_utc"], utc=True)
    return combined


# ---------------------------------------------------------------------------
# Bootstrap metric builders
# ---------------------------------------------------------------------------


def _accuracy_metric_fn(correct_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        values = pd.to_numeric(rows[correct_col], errors="coerce").dropna()
        accuracy = float(values.mean()) if len(values) else float("nan")
        return {"accuracy": accuracy, "accuracy_minus_half": accuracy - 0.5}

    return _metric


def _tercile_diff_metric_fn(value_col: str, tercile_col: str, high: int, low: int):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        values = pd.to_numeric(rows[value_col], errors="coerce")
        tercile = rows[tercile_col]
        high_values = values.loc[tercile.eq(high)].dropna()
        low_values = values.loc[tercile.eq(low)].dropna()
        high_mean = float(high_values.mean()) if len(high_values) else float("nan")
        low_mean = float(low_values.mean()) if len(low_values) else float("nan")
        return {
            "high_tercile_mean": high_mean,
            "low_tercile_mean": low_mean,
            "high_minus_low": high_mean - low_mean,
        }

    return _metric


def _bootstrap(frame: pd.DataFrame, metric_fn, *, block: str) -> pd.DataFrame:
    return week_blocked_bootstrap(
        frame, metric_fn, block=block, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )


def _both_blocks(frame: pd.DataFrame, metric_fn) -> dict[str, pd.DataFrame]:
    return {
        "week": _bootstrap(frame, metric_fn, block="week"),
        "season": _bootstrap(frame, metric_fn, block="season"),
    }


def _row(ci: dict[str, pd.DataFrame], metric: str) -> dict[str, float]:
    week_row = ci["week"].loc[ci["week"]["metric"].eq(metric)].iloc[0]
    season_row = ci["season"].loc[ci["season"]["metric"].eq(metric)].iloc[0]
    return {
        "estimate": float(week_row["estimate"]),
        "week_lower": float(week_row["lower"]),
        "week_upper": float(week_row["upper"]),
        "week_probability_positive": float(week_row["probability_positive"]),
        "season_lower": float(season_row["lower"]),
        "season_upper": float(season_row["upper"]),
        "season_probability_positive": float(season_row["probability_positive"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    cells: list[dict[str, Any]] = []
    proposed_records: list[dict[str, Any]] = []

    print(f"Loading features: {args.features}")
    features = pd.read_parquet(args.features)
    regular = regular_season_rows(features)
    schedule = regular[["game_id", "season", "week", "spread_line"]].drop_duplicates("game_id")
    outcomes = regular[["game_id", "result", "home_score", "away_score"]].drop_duplicates("game_id")

    # ------------------------------------------------------------------
    # Base tue_open construction (H1, H2, H4, H5, H6): reuse the audited
    # build_pairing_table / spread_price_consensus_table / novig functions
    # exactly like scripts/novig_diagnostics_screen.py.
    # ------------------------------------------------------------------
    print("Building tue_open pairing + spread price consensus (H1/H2/H6 base)")
    pairing_tue = build_pairing_table(
        args.root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",), schedule=schedule
    )
    prices_tue = spread_price_consensus_table(
        args.root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",), schedule=schedule
    ).drop(columns=["snapshot_timestamp_utc"])
    join_keys = ["game_id", "season", "week", "decision_label", "capture_kind"]
    spread_source = pairing_tue.merge(prices_tue, on=join_keys, how="inner")
    spread_novig = spread_novig_probabilities(spread_source)
    spread_novig = spread_novig.merge(outcomes[["game_id", "result"]], on="game_id", how="inner")
    ats_margin = pd.to_numeric(spread_novig["result"], errors="coerce") - pd.to_numeric(
        spread_novig["home_spread"], errors="coerce"
    )
    spread_novig["ats_margin"] = ats_margin
    spread_novig["home_cover_at_label"] = np.select(
        [ats_margin.gt(0.0), ats_margin.lt(0.0)], [1.0, 0.0], default=np.nan
    )
    n_base = len(spread_novig)
    n_pushes = int(ats_margin.eq(0.0).sum())
    print(
        f"tue_open base population: {n_base} games, {n_pushes} pushes excluded from accuracy cells"
    )

    # ------------------------------------------------------------------
    # Raw tue_open quotes, once, reused for H2's price dispersion, H4's
    # book-level prices, and H5's totals price consensus.
    # ------------------------------------------------------------------
    print("Loading raw tue_open quotes for book-level and totals work")
    tue_quotes_raw = load_decision_quotes(
        args.root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",)
    )
    tue_quotes = _true_week_correct(tue_quotes_raw, schedule)
    tue_consensus = decision_market_consensus(tue_quotes)

    # ==================================================================
    # H1: asymmetric opener juice predicts cover
    # ==================================================================
    print("\n=== H1: asymmetric opener juice ===")
    prob = spread_novig["no_vig_home_cover_probability"]
    h1_pop = spread_novig.loc[prob.notna() & prob.ne(0.5) & ats_margin.ne(0.0)].copy()
    h1_pop["pick_home"] = h1_pop["no_vig_home_cover_probability"].gt(0.5)
    h1_pop["correct"] = pick_correct(h1_pop["pick_home"], h1_pop["ats_margin"])
    h1_pop = h1_pop.dropna(subset=["correct"])
    print(f"H1 cell 1.1 population: {len(h1_pop)} games (excludes pushes and exact-0.5 juice)")
    h1_ci = _both_blocks(h1_pop, _accuracy_metric_fn("correct"))
    h1_1 = _row(h1_ci, "accuracy_minus_half")
    h1_1["n"] = len(h1_pop)
    h1_1["point_accuracy"] = float(h1_pop["correct"].mean())
    cells.append({"hypothesis": "H1", "cell": "1.1_juice_lean_accuracy", **h1_1})
    wp = h1_1["week_probability_positive"]
    print(f"H1.1 accuracy={h1_1['point_accuracy']:.4f} week_P+={wp:.3f}")

    edges_juice = _tercile_edges((h1_pop["no_vig_home_cover_probability"] - 0.5).abs())
    h1_pop["juice_tercile"] = _assign_tercile(
        (h1_pop["no_vig_home_cover_probability"] - 0.5).abs(), edges_juice
    )
    h1_diff_ci = _both_blocks(
        h1_pop, _tercile_diff_metric_fn("correct", "juice_tercile", high=2, low=0)
    )
    h1_2 = _row(h1_diff_ci, "high_minus_low")
    for tercile in (0, 1, 2):
        sub = h1_pop.loc[h1_pop["juice_tercile"].eq(tercile)]
        print(
            f"  juice tercile {tercile}: n={len(sub)} accuracy={sub['correct'].mean():.4f} "
            f"mean|prob-0.5|={(sub['no_vig_home_cover_probability'] - 0.5).abs().mean():.4f}"
        )
    h1_2["n"] = len(h1_pop)
    cells.append({"hypothesis": "H1", "cell": "1.2_dose_response_high_minus_low", **h1_2})

    # ==================================================================
    # H2: cross-book dispersion as an uncertainty / game-selection signal
    # ==================================================================
    print("\n=== H2: cross-book dispersion ===")
    book_prices_raw = _book_level_spread_prices(tue_quotes)
    price_dispersion = (
        book_prices_raw.groupby("nflverse_game_id")["home_price"]
        .agg(price_std="std", price_books="count")
        .reset_index()
        .rename(columns={"nflverse_game_id": "game_id"})
    )
    h2_pop = h1_pop.merge(price_dispersion, on="game_id", how="left")
    h2_pop["abs_ats_margin"] = h2_pop["ats_margin"].abs()

    for proxy, min_books in (("spread_std", None), ("price_std", 3)):
        working = h2_pop.copy()
        if min_books is not None:
            working = working.loc[working["price_books"].fillna(0).ge(min_books)]
        working = working.loc[pd.to_numeric(working[proxy], errors="coerce").notna()].copy()
        edges = _tercile_edges(working[proxy])
        working["dispersion_tercile"] = _assign_tercile(working[proxy], edges)
        print(f"H2 proxy={proxy}: n={len(working)}")

        margin_ci = _both_blocks(
            working, _tercile_diff_metric_fn("abs_ats_margin", "dispersion_tercile", high=2, low=0)
        )
        margin_row = _row(margin_ci, "high_minus_low")
        margin_row["n"] = len(working)
        cells.append(
            {"hypothesis": "H2", "cell": f"2a_{proxy}_abs_ats_margin_high_minus_low", **margin_row}
        )
        print(
            f"  |ats_margin| high-low tercile diff = {margin_row['estimate']:.4f} "
            f"week_P+={margin_row['week_probability_positive']:.3f}"
        )

        accuracy_ci = _both_blocks(
            working, _tercile_diff_metric_fn("correct", "dispersion_tercile", high=0, low=2)
        )
        accuracy_row = _row(accuracy_ci, "high_minus_low")
        accuracy_row["n"] = len(working)
        cells.append(
            {
                "hypothesis": "H2",
                "cell": f"2b_{proxy}_juice_accuracy_low_minus_high_dispersion",
                **accuracy_row,
            }
        )
        print(
            f"  juice accuracy low-minus-high dispersion diff = {accuracy_row['estimate']:.4f} "
            f"week_P+={accuracy_row['week_probability_positive']:.3f}"
        )

    # ==================================================================
    # H3: Tuesday-to-Wednesday drift echo
    # ==================================================================
    print("\n=== H3: Tuesday-to-Wednesday drift echo ===")
    pairing_full = build_pairing_table(
        args.root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=schedule,
    )
    close_full = close_reference_table(pairing_full, schedule)
    tue_open_full = pairing_full.loc[pairing_full["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})
    paired_full = tue_open_full.merge(close_full, on="game_id", how="inner").merge(
        outcomes[["game_id", "result"]], on="game_id", how="inner"
    )
    paired_full["margin_vs_open"] = paired_full["result"] - paired_full["tue_open_home_spread"]
    paired_full["open_move"] = (
        paired_full["close_home_spread"] - paired_full["tue_open_home_spread"]
    )
    paired_full["pick_home"] = paired_full["open_move"].gt(0.0)
    paired_full["oracle_correct"] = pick_correct(
        paired_full["pick_home"], paired_full["margin_vs_open"]
    ).where(paired_full["open_move"].ne(0.0))

    def _oracle_cell(frame: pd.DataFrame, label: str) -> None:
        pop = frame.dropna(subset=["oracle_correct"]).copy()
        print(f"  {label}: n={len(pop)}")
        if pop.empty:
            return
        ci = _both_blocks(pop, _accuracy_metric_fn("oracle_correct"))
        row = _row(ci, "accuracy_minus_half")
        row["n"] = len(pop)
        row["point_accuracy"] = float(pop["oracle_correct"].mean())
        cells.append({"hypothesis": "H3", "cell": label, **row})
        wp = row["week_probability_positive"]
        print(f"    accuracy={row['point_accuracy']:.4f} week_P+={wp:.3f}")

    _oracle_cell(paired_full, "3.0a_full_week_oracle_2020_2025_sanity_check")
    paired_2023_2025 = paired_full.loc[paired_full["season"].isin(INTRADAY_SEASONS)]
    _oracle_cell(paired_2023_2025, "3.0b_full_week_oracle_2023_2025_baseline")

    wed_manifests = _wednesday_noon_snapshots(args.root)
    n_weeks_with_intraday = (
        load_snapshot_manifest_index(args.root)
        .loc[
            lambda d: (
                d["capture_kind"].eq(HISTORICAL_CAPTURE_KIND)
                & d["decision_label"].eq(INTRADAY_LABEL)
            )
        ]
        .drop_duplicates(["season", "week"])
        .shape[0]
    )
    print(
        f"Wednesday-noon-ET snapshots selected: {len(wed_manifests)} of "
        f"{n_weeks_with_intraday} weeks with intraday_hourly coverage"
    )
    wed_quotes_raw = _load_selected_snapshots(wed_manifests)
    wed_quotes = (
        _true_week_correct(wed_quotes_raw, schedule) if not wed_quotes_raw.empty else wed_quotes_raw
    )
    if not wed_quotes.empty:
        wed_consensus = decision_market_consensus(wed_quotes)
        wed_spread = wed_consensus.loc[
            wed_consensus["market"].eq("spreads") & wed_consensus["outcome_side"].eq("HOME")
        ][["nflverse_game_id", "consensus_line"]].rename(
            columns={"nflverse_game_id": "game_id", "consensus_line": "wed_home_spread"}
        )
    else:
        wed_spread = pd.DataFrame(columns=["game_id", "wed_home_spread"])

    paired_wed = tue_open_full.merge(wed_spread, on="game_id", how="inner").merge(
        outcomes[["game_id", "result"]], on="game_id", how="inner"
    )
    paired_wed["margin_vs_open"] = paired_wed["result"] - paired_wed["tue_open_home_spread"]
    paired_wed["open_move"] = paired_wed["wed_home_spread"] - paired_wed["tue_open_home_spread"]
    paired_wed["pick_home"] = paired_wed["open_move"].gt(0.0)
    paired_wed["oracle_correct"] = pick_correct(
        paired_wed["pick_home"], paired_wed["margin_vs_open"]
    ).where(paired_wed["open_move"].ne(0.0))
    _oracle_cell(paired_wed, "3.1_tue_to_wed_oracle_2023_2025")

    # ==================================================================
    # H4: off-market single-book openers
    # ==================================================================
    print("\n=== H4: off-market single-book openers ===")
    book_prices = book_prices_raw.copy()
    book_prob, book_hold = _no_vig_pair(book_prices["home_price"], book_prices["away_price"])
    book_prices = book_prices.assign(book_home_cover_probability=book_prob, book_hold=book_hold)
    book_prices = book_prices.dropna(subset=["book_home_cover_probability"])
    book_prices = book_prices.rename(columns={"nflverse_game_id": "game_id"})

    consensus_lookup = spread_novig[["game_id", "no_vig_home_cover_probability"]].rename(
        columns={"no_vig_home_cover_probability": "consensus_probability"}
    )
    book_prices = book_prices.merge(consensus_lookup, on="game_id", how="inner")
    book_prices["divergence"] = (
        book_prices["book_home_cover_probability"] - book_prices["consensus_probability"]
    ).abs()
    book_counts = book_prices.groupby("game_id").size().rename("n_books")
    eligible_games = book_counts.loc[book_counts.ge(3)].index
    print(f"Games with >=3 books both-sided at tue_open: {len(eligible_games)}")
    eligible = book_prices.loc[book_prices["game_id"].isin(eligible_games)]
    divergent = (
        eligible.sort_values("divergence", ascending=False)
        .groupby("game_id", as_index=False)
        .head(1)
    )
    divergent = divergent.merge(
        h1_pop[["game_id", "ats_margin", "home_cover_at_label"]], on="game_id", how="inner"
    )
    lean_book = divergent["book_home_cover_probability"].gt(0.5)
    lean_consensus = divergent["consensus_probability"].gt(0.5)
    has_lean = divergent["book_home_cover_probability"].ne(0.5) & divergent[
        "consensus_probability"
    ].ne(0.5)
    disagreement = divergent.loc[has_lean & lean_book.ne(lean_consensus)].copy()
    print(f"Disagreement subset (divergent book vs consensus lean differs): {len(disagreement)}")
    if not disagreement.empty:
        disagreement["pick_home_book"] = disagreement["book_home_cover_probability"].gt(0.5)
        disagreement["pick_home_consensus"] = disagreement["consensus_probability"].gt(0.5)
        disagreement["correct_book"] = pick_correct(
            disagreement["pick_home_book"], disagreement["ats_margin"]
        )
        disagreement["correct_consensus"] = pick_correct(
            disagreement["pick_home_consensus"], disagreement["ats_margin"]
        )
        disagreement = disagreement.merge(
            spread_novig[["game_id", "season", "week"]], on="game_id", how="left"
        )
        for col, tag in (
            ("correct_book", "4.1a_divergent_book_accuracy"),
            ("correct_consensus", "4.1b_consensus_accuracy"),
        ):
            pop = disagreement.dropna(subset=[col])
            ci = _both_blocks(pop, _accuracy_metric_fn(col))
            row = _row(ci, "accuracy_minus_half")
            row["n"] = len(pop)
            row["point_accuracy"] = float(pop[col].mean())
            cells.append({"hypothesis": "H4", "cell": tag, **row})
            wp = row["week_probability_positive"]
            print(f"  {tag}: n={len(pop)} accuracy={row['point_accuracy']:.4f} week_P+={wp:.3f}")
    else:
        print("  No disagreement games at this threshold; cell reported as zero-n below.")

    # ==================================================================
    # H5 (own): totals-market juice asymmetry
    # ==================================================================
    print("\n=== H5: totals-market juice asymmetry ===")
    totals_prices = _totals_price_consensus(tue_consensus)
    totals_source = pairing_tue[["game_id", "season", "week", "total_line"]].merge(
        totals_prices, on=["game_id", "season", "week"], how="inner"
    )
    over_prob, over_hold = _no_vig_pair(totals_source["over_price"], totals_source["under_price"])
    totals_source = totals_source.assign(no_vig_over_probability=over_prob, totals_hold=over_hold)
    totals_source = totals_source.merge(
        outcomes[["game_id", "home_score", "away_score"]], on="game_id", how="inner"
    )
    total_points = pd.to_numeric(totals_source["home_score"], errors="coerce") + pd.to_numeric(
        totals_source["away_score"], errors="coerce"
    )
    total_margin = total_points - pd.to_numeric(totals_source["total_line"], errors="coerce")
    totals_source["over_covers"] = np.select(
        [total_margin.gt(0.0), total_margin.lt(0.0)], [1.0, 0.0], default=np.nan
    )
    h5_pop = totals_source.loc[
        totals_source["no_vig_over_probability"].notna()
        & totals_source["no_vig_over_probability"].ne(0.5)
        & total_margin.ne(0.0)
    ].copy()
    h5_pop["pick_over"] = h5_pop["no_vig_over_probability"].gt(0.5)
    h5_pop["over_margin"] = total_margin.loc[h5_pop.index]
    h5_pop["correct"] = pick_correct(h5_pop["pick_over"], h5_pop["over_margin"])
    h5_pop = h5_pop.dropna(subset=["correct"])
    print(f"H5 population: {len(h5_pop)} games (excludes total pushes and exact-0.5 juice)")
    h5_ci = _both_blocks(h5_pop, _accuracy_metric_fn("correct"))
    h5_row = _row(h5_ci, "accuracy_minus_half")
    h5_row["n"] = len(h5_pop)
    h5_row["point_accuracy"] = float(h5_pop["correct"].mean())
    cells.append({"hypothesis": "H5", "cell": "5.1_totals_juice_lean_accuracy", **h5_row})
    wp = h5_row["week_probability_positive"]
    print(f"H5.1 accuracy={h5_row['point_accuracy']:.4f} week_P+={wp:.3f}")

    # ==================================================================
    # H6 (own): market hold (vig level) as an uncertainty signal
    # ==================================================================
    print("\n=== H6: market hold as an uncertainty signal ===")
    h6_pop = h1_pop.loc[pd.to_numeric(h1_pop["spread_hold"], errors="coerce").notna()].copy()
    h6_pop["abs_ats_margin"] = h6_pop["ats_margin"].abs()
    edges_hold = _tercile_edges(h6_pop["spread_hold"])
    h6_pop["hold_tercile"] = _assign_tercile(h6_pop["spread_hold"], edges_hold)
    print(f"H6 population: {len(h6_pop)}")

    hold_margin_ci = _both_blocks(
        h6_pop, _tercile_diff_metric_fn("abs_ats_margin", "hold_tercile", high=2, low=0)
    )
    hold_margin_row = _row(hold_margin_ci, "high_minus_low")
    hold_margin_row["n"] = len(h6_pop)
    cells.append(
        {"hypothesis": "H6", "cell": "6.1_abs_ats_margin_high_minus_low_hold", **hold_margin_row}
    )
    print(
        f"  |ats_margin| high-low hold tercile diff = {hold_margin_row['estimate']:.4f} "
        f"week_P+={hold_margin_row['week_probability_positive']:.3f}"
    )

    hold_accuracy_ci = _both_blocks(
        h6_pop, _tercile_diff_metric_fn("correct", "hold_tercile", high=2, low=0)
    )
    hold_accuracy_row = _row(hold_accuracy_ci, "high_minus_low")
    hold_accuracy_row["n"] = len(h6_pop)
    cells.append(
        {"hypothesis": "H6", "cell": "6.2_juice_accuracy_high_minus_low_hold", **hold_accuracy_row}
    )
    print(
        f"  juice accuracy high-minus-low hold tercile diff = {hold_accuracy_row['estimate']:.4f} "
        f"week_P+={hold_accuracy_row['week_probability_positive']:.3f}"
    )

    # ------------------------------------------------------------------
    # Propose (never execute) weak-signals record commands for leads
    # ------------------------------------------------------------------
    for c in cells:
        p_plus = c.get("week_probability_positive")
        if p_plus is None:
            continue
        lean = max(p_plus, 1.0 - p_plus)
        if lean < 0.75:
            continue
        direction = "positive" if p_plus >= 0.5 else "negative"
        name = f"odds_microstructure_{c['hypothesis']}_{c['cell']}".replace(".", "_")
        cell_ref = f"{c['hypothesis']}/{c['cell']}"
        description = f"odds microstructure lead-gen battery, cell {cell_ref}"
        notes = "lead-generation only, not independently confirmed on a rotated window"
        proposed_records.append(
            {
                "cell": cell_ref,
                "week_probability_positive": p_plus,
                "direction": direction,
                "proposed_command": (
                    "nfl-ats weak-signals record "
                    f"--name {name} "
                    f'--description "{description}" '
                    "--source artifacts/odds_microstructure/<run_id>/cells_summary.csv "
                    f"--effect {c['estimate']:.6f} --effect-units accuracy_points "
                    "--classification unresolved_below_power --league nfl "
                    "--season-start 2020 --season-end 2025 "
                    f"--interval-low {c['week_lower']:.6f} --interval-high {c['week_upper']:.6f} "
                    f"--probability-positive {p_plus:.6f} --sample-games {c['n']} "
                    f'--notes "{notes}"'
                ),
            }
        )

    print(f"\n{len(proposed_records)} cells cross the 0.75/0.25 lean threshold; proposed commands:")
    for record in proposed_records:
        wp = record["week_probability_positive"]
        print(f"  [{record['direction']}, P+={wp:.3f}] {record['cell']}")
        print(f"    {record['proposed_command']}")

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------
    output_dir = args.output_root / run_id()
    atomic_parquet(spread_novig, output_dir / "spread_novig_tue_open.parquet")
    atomic_parquet(book_prices, output_dir / "book_level_novig_tue_open.parquet")
    atomic_parquet(divergent, output_dir / "divergent_book_games.parquet")
    atomic_parquet(totals_source, output_dir / "totals_novig_tue_open.parquet")
    atomic_parquet(paired_wed, output_dir / "wed_movement.parquet")
    cells_frame = pd.DataFrame(cells)
    atomic_parquet(cells_frame, output_dir / "cells_summary.parquet")
    (output_dir / "cells_summary.csv").parent.mkdir(parents=True, exist_ok=True)
    cells_frame.to_csv(output_dir / "cells_summary.csv", index=False)

    metadata: dict[str, Any] = {
        "predeclaration": "scripts/odds_microstructure_battery.py module docstring",
        "diagnostic_not_confirmation": True,
        "rotation_registry_touched": False,
        "registry_files_written": [],
        "git_revision": _git_revision(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "coverage": {
            "tue_open_base_games": n_base,
            "tue_open_pushes_excluded": n_pushes,
            "h3_weeks_with_intraday_coverage": int(n_weeks_with_intraday),
            "h3_weeks_with_wed_snapshot": len(wed_manifests),
            "h3_seasons": list(INTRADAY_SEASONS),
            "h4_games_ge3_books": len(eligible_games),
            "h4_disagreement_games": len(disagreement) if not disagreement.empty else 0,
            "h5_totals_games": len(h5_pop),
        },
        "cells": cells,
        "proposed_weak_signal_records": proposed_records,
    }
    atomic_json(metadata, output_dir / "metadata.json")
    print(f"\nartifacts: {output_dir}")


if __name__ == "__main__":
    main()
