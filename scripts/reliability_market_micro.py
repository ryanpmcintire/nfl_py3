"""Split-half reliability for the 26 ``market_micro`` registry cells (ORCH-D).

**What these cells are.** Six screens, one construct family each, all of them
market microstructure rather than team quality:

``odds_microstructure_*`` (7)
    ``scripts/odds_microstructure_battery.py``. Parent quantities: the no-vig
    juice lean off the ``tue_open`` spread PRICE, cross-book price dispersion,
    market hold, the totals-market juice lean, and the line MOVEMENT the
    ``*_oracle_*`` cells peek at. The three ``H3`` cells are DELIBERATELY
    LEAKED oracle controls -- they settle at the frozen opener but pick the
    side the line later moved toward, so their effect numbers are ceilings by
    construction and no reliability read from them is evidence about a
    playable rule. Their inputs' reliability is still worth having: it says
    whether line movement carries stable team structure at all.
``public_betting_battery_*`` (5)
    ``scripts/public_betting_battery_screen.py``. Parent: the public bet (and
    money) percentage on a side, from a deliberately sparse archive.
``sagarin_battery_*`` (7)
    ``scripts/sagarin_divergence_battery.py``. Parent: Sagarin-rating-implied
    spread minus the market line -- a genuinely continuous per-team-week
    trait, so ``METHOD_TRAIT`` applies cleanly.
``sbr_opener_*`` (4) + ``proxy_opener_production_rule_2009_2019``
    ``scripts/sbr_era_opener_eval.py`` / ``scripts/proxy_opener_replication.py``.
    Parent: the SBR-substituted opening line, and its difference from the
    production (purchased-archive) Tuesday opener on the 2020-2021 overlap.
``mod08_smooth_cdf_mapping``/``_opener`` (2)
    ``scripts/smooth_cdf_mapping_measurement.py`` and
    ``scripts/smooth_cdf_mapping_opener_measurement.py``. A probability
    CALIBRATION construct, not a team feature: both arms score the SAME
    fitted model off the SAME out-of-time residual sample and differ only in
    the mapping from predicted margin to cover probability. There is no
    team-week trait underneath, so these are reported
    ``not_applicable: no underlying trait to be reliable``. That is a
    legitimate, informative outcome -- NOT a closure, and it says nothing
    negative about the signal.

**Sign convention, stated once.** A quantity with a team side is signed toward
the team on whose row it sits: the home row carries the builder's own
home-positive value, the away row carries its negation (juice lean, line
movement, Sagarin divergence, the SBR opener line). The public bet and money
percentages already arrive per side, so each row carries its OWN side's
percentage rather than a negation. A quantity with no team side at all --
market hold, cross-book price dispersion, the totals over/under juice -- is
NOT signed; it is measured once per game at the HOME-TEAM-season unit as a
venue-season proxy, tagged ``METHOD_VENUE`` and reported as such. It is not a
team trait, it is not comparable to a team-season reliability, and a low value
there is not a closing ground on its own.

**Method.** Everything runs through ``scripts/reliability_lib.py``: unit =
team-season (home-team-season for the venue-tagged rows), halves = odd/even
weeks, Spearman-Brown corrected, block bootstrap over units, seed 20260901,
4000 draws, restricted to each cell's OWN registry season window. The
estimator is ``nfl_ats.cfb_qb_dependence.split_half_reliability``, imported
through that harness and never reimplemented. Every parent quantity is
produced by importing the screen that built the cell, never re-derived here.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. Only
two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED
wrong sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an effect
that size. Everything else is ``unresolved_below_power``; report
``probability_positive``, never "contains zero". This script CLOSES NOTHING
and RECLASSIFIES NOTHING: it measures, and a low number is a candidate for the
reliability ground, never the closure itself. Within-week correlation is ZERO.

A construct with too few usable units is reported as UNMEASURED, never as
reliability 0 -- writing a NaN through as a number would manufacture the
appearance of a closing ground out of nothing. A near-CONSTANT column is
reported ``not_informative_near_constant`` with its numbers shown rather than
recorded, because a column with almost no cross-unit spread can return a large
|correlation| of either sign that flips with the season window.

Writes ``artifacts/reliability_sweep/market_micro/<stamp>/results.json`` and
prints the ``set-reliability`` commands it would run (this script runs none;
recording goes through the locked CLI).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import odds_microstructure_battery as omb  # noqa: E402
import proxy_opener_replication as pxy  # noqa: E402
import public_betting_battery_screen as pbb  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import sagarin_divergence_battery as sag  # noqa: E402

from nfl_ats.clv import (  # noqa: E402
    build_pairing_table,
    close_reference_table,
    decision_market_consensus,
    load_decision_quotes,
    pick_correct,
    spread_price_consensus_table,
)
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.novig import spread_novig_probabilities  # noqa: E402
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

MARKET_ROOT = REPO / "data/market/raw"
PLAYER_FEATURES = REPO / "data/processed/game_features_player.parquet"
WEAK_STACK_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
SBR_ODDS = REPO / "data/processed/sbr_odds.parquet"
SBR_SCORED = REPO / "artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet"
PROXY_SCORED = REPO / "artifacts/proxy_opener_replication/20260819T194330Z/main_scored.parquet"

#: A game-level quantity with no team side is still measured with the shared
#: estimator, but at the HOME-TEAM-season unit (the stadium/venue proxy). The
#: tag says so out loud so nobody reads it as a team-trait number.
METHOD_VENUE_HOME_UNIT = rlib.METHOD_VENUE + (
    ". VENUE KEY = the home team (one stadium per franchise); the quantity is "
    "symmetric across the two sides of the game and has no team split, so it is "
    "measured once per game on the home side rather than signed onto both teams"
)

ORACLE_CAVEAT = (
    "DELIBERATELY-LEAKED ORACLE CONTROL: the cell picks the side the line later moved "
    "toward and settles at the frozen opener, so its EFFECT number is a ceiling by "
    "construction. The reliability here is the reliability of the movement INPUT; it is "
    "not evidence about any playable rule."
)

NOT_APPLICABLE = "not_applicable: no underlying trait to be reliable"
NEAR_CONSTANT = "not_informative_near_constant"
NO_ROWS = "data_not_present_locally"

MOD08_NOTE = (
    "Both MOD-08 arms score the SAME fitted model off the SAME out-of-time residual "
    "sample and differ only in the margin-to-probability mapping "
    "(smooth_cdf_mapping_measurement.py:122-131; "
    "smooth_cdf_mapping_opener_measurement.py:176-183), so there is no team-week trait "
    "whose split-half reliability could be measured. This is informative, NOT a closure "
    "and NOT a negative about the signal: the entry keeps its classification and its "
    "reliability field stays null rather than being filled with a number belonging to a "
    "different construct."
)

#: Near-constant guard. A column with (almost) no cross-unit spread in its
#: half-means can return a large |r| of either sign that flips with the season
#: window; that is an artifact, not a trait, and must not be recorded.
MIN_DISTINCT_VALUES = 3
MIN_SPREAD = 1e-9


# ---------------------------------------------------------------------------
# Long-frame shapes (every VALUE comes from a screen's own builder)
# ---------------------------------------------------------------------------


def signed_team_week(
    games: pd.DataFrame,
    value_col: str,
    *,
    metric: str,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """Explode a HOME-POSITIVE game quantity into a signed team-week frame.

    Each game contributes two rows: the home team's row carries the builder's
    own home-positive value, the away team's row carries its negation, so both
    rows read "how far this quantity leans toward the team on this row".
    """

    values = pd.to_numeric(games[value_col], errors="coerce").to_numpy(dtype=float)
    pieces = []
    for team_col, sign in ((home_col, 1.0), (away_col, -1.0)):
        piece = games.loc[:, ["season", "week", team_col]].rename(columns={team_col: "team_id"})
        piece[metric] = sign * values
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def per_side_team_week(
    games: pd.DataFrame,
    home_value_col: str,
    away_value_col: str,
    *,
    metric: str,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """Explode a quantity that already exists per side into a team-week frame.

    Used for the public bet/money percentages, which the archive reports for
    the home and the away side separately -- so a row carries its own side's
    number, never a negation of the other side's.
    """

    pieces = []
    for team_col, value_col in ((home_col, home_value_col), (away_col, away_value_col)):
        piece = games.loc[:, ["season", "week", team_col]].rename(columns={team_col: "team_id"})
        piece[metric] = pd.to_numeric(games[value_col], errors="coerce").to_numpy(dtype=float)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def home_unit_frame(
    games: pd.DataFrame, value_col: str, *, metric: str, home_col: str = "home_team"
) -> pd.DataFrame:
    """One row per game at the HOME-TEAM (venue-proxy) unit, for symmetric quantities."""

    frame = games.loc[:, ["season", "week", home_col]].rename(columns={home_col: "team_id"})
    frame[metric] = pd.to_numeric(games[value_col], errors="coerce").to_numpy(dtype=float)
    return frame


def price_dispersion(tue_quotes: pd.DataFrame) -> pd.DataFrame:
    """Per-game cross-book HOME-side spread-price dispersion.

    Routes through the screen's own ``_book_level_spread_prices`` (which
    applies its pregame filter and per-book latest-quote dedup) and then the
    same ``groupby(...).agg(price_std="std", price_books="count")`` the screen
    runs at ``scripts/odds_microstructure_battery.py:507-513``.
    """

    book_prices = omb._book_level_spread_prices(tue_quotes)
    return (
        book_prices.groupby("nflverse_game_id")["home_price"]
        .agg(price_std="std", price_books="count")
        .reset_index()
        .rename(columns={"nflverse_game_id": "game_id"})
    )


def _home_cover_from_margin(margin: pd.Series) -> np.ndarray:
    return np.select([margin.gt(0.0), margin.lt(0.0)], [1.0, 0.0], default=np.nan)


def near_constant_report(
    long: pd.DataFrame, metric: str, *, seasons: tuple[int, int]
) -> dict[str, Any]:
    """Numbers behind the near-constant guard: always shown, never inferred."""

    frame = long.loc[:, ["team_id", "season", "week"]].copy()
    frame[metric] = pd.to_numeric(long[metric], errors="coerce").to_numpy(dtype=float)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=[metric, "season", "week"])
    frame = frame.loc[frame["season"].between(seasons[0], seasons[1])]
    if frame.empty:
        return {"n_observations": 0, "near_constant": False}
    frame["half"] = np.where(frame["week"].astype(int) % 2 == 0, "even", "odd")
    half_means = frame.groupby(["team_id", "season", "half"])[metric].mean().unstack("half")
    spreads: dict[str, float] = {}
    for half in ("odd", "even"):
        column = half_means[half].dropna() if half in half_means else pd.Series(dtype=float)
        spreads[half] = float(column.std()) if len(column) > 1 else math.nan
    values = frame[metric]
    observation_std = float(values.std()) if len(values) > 1 else math.nan
    flagged = (
        int(values.nunique()) < MIN_DISTINCT_VALUES
        or (np.isfinite(observation_std) and observation_std <= MIN_SPREAD)
        or (np.isfinite(spreads["odd"]) and spreads["odd"] <= MIN_SPREAD)
        or (np.isfinite(spreads["even"]) and spreads["even"] <= MIN_SPREAD)
    )
    return {
        "n_observations": len(values),
        "n_distinct_values": int(values.nunique()),
        "observation_std": observation_std,
        "half_mean_cross_unit_std_odd": spreads["odd"],
        "half_mean_cross_unit_std_even": spreads["even"],
        "near_constant": bool(flagged),
    }


def window_sign_stability(
    long: pd.DataFrame, metric: str, *, unit_col: str, seasons: tuple[int, int], n_boot: int
) -> dict[str, Any]:
    """Re-measure on a shortened window: does the correlation keep its sign?

    The orchestrator's own hazard is that a near-constant column returns a
    large |r| of EITHER sign that flips when the season window moves. This
    diagnostic sits beside every measurement so a reader can see stability
    instead of taking a point estimate on trust. Reported, never recorded.
    """

    low, high = int(seasons[0]), int(seasons[1])
    if high <= low:
        return {"status": "single_season_window", "shortened_seasons": None}
    shortened = (low + 1, high)
    measured = rlib.measure_reliability(
        long,
        metric,
        method="window-stability diagnostic",
        unit_col=unit_col,
        seasons=shortened,
        n_boot=n_boot,
    )
    return {
        "status": measured["status"],
        "shortened_seasons": list(shortened),
        "n_units": measured["n_units"],
        "pearson_r": measured["pearson_r"],
        "reliability": measured["reliability"],
    }


# ---------------------------------------------------------------------------
# Screen-by-screen population construction
# ---------------------------------------------------------------------------


def build_odds_micro(market_root: Path, features_path: Path) -> dict[str, Any]:
    """Rebuild the odds-microstructure battery's own populations.

    ``scripts/odds_microstructure_battery.py`` factors none of its ``main``
    into functions, so the audited ``nfl_ats`` builders it calls are called
    here in the same order with the same arguments (base construction lines
    434-450, price dispersion 507-513, movement oracle 560-638, totals
    707-731, hold 745-748), and every one of its own private helpers is
    IMPORTED rather than copied.
    """

    features = pd.read_parquet(features_path)
    regular = regular_season_rows(features)
    schedule = regular[["game_id", "season", "week", "spread_line"]].drop_duplicates("game_id")
    outcomes = regular[["game_id", "result", "home_score", "away_score"]].drop_duplicates("game_id")
    teams = regular[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")

    pairing_full = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *omb.CLOSE_LABEL_PRIORITY),
        schedule=schedule,
    )
    prices_tue = spread_price_consensus_table(
        market_root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",), schedule=schedule
    ).drop(columns=["snapshot_timestamp_utc"])
    join_keys = ["game_id", "season", "week", "decision_label", "capture_kind"]
    pairing_tue = pairing_full.loc[pairing_full["decision_label"].eq("tue_open")].copy()
    spread_source = pairing_tue.merge(prices_tue, on=join_keys, how="inner")
    spread_novig = spread_novig_probabilities(spread_source)
    spread_novig = spread_novig.merge(outcomes[["game_id", "result"]], on="game_id", how="inner")
    ats_margin = pd.to_numeric(spread_novig["result"], errors="coerce") - pd.to_numeric(
        spread_novig["home_spread"], errors="coerce"
    )
    spread_novig["ats_margin"] = ats_margin
    spread_novig = spread_novig.merge(teams, on="game_id", how="left")

    prob = spread_novig["no_vig_home_cover_probability"]
    h1 = spread_novig.loc[prob.notna() & prob.ne(0.5) & ats_margin.ne(0.0)].copy()
    h1["pick_home"] = h1["no_vig_home_cover_probability"].gt(0.5)
    h1["correct"] = pick_correct(h1["pick_home"], h1["ats_margin"])
    h1 = h1.dropna(subset=["correct"]).reset_index(drop=True)
    h1["juice_lean_home"] = h1["no_vig_home_cover_probability"] - 0.5
    h1["juice_magnitude"] = h1["juice_lean_home"].abs()
    h1["home_cover_at_label"] = _home_cover_from_margin(h1["ats_margin"])

    tue_quotes = omb._true_week_correct(
        load_decision_quotes(
            market_root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",)
        ),
        schedule,
    )
    dispersion = price_dispersion(tue_quotes)
    h2 = h1.merge(dispersion, on="game_id", how="left")
    h2 = h2.loc[h2["price_books"].fillna(0).ge(3) & h2["price_std"].notna()].reset_index(drop=True)

    h6 = h1.loc[pd.to_numeric(h1["spread_hold"], errors="coerce").notna()].reset_index(drop=True)

    tue_consensus = decision_market_consensus(tue_quotes)
    totals_prices = omb._totals_price_consensus(tue_consensus)
    totals = pairing_tue[["game_id", "season", "week", "total_line"]].merge(
        totals_prices, on=["game_id", "season", "week"], how="inner"
    )
    over_prob, over_hold = omb._no_vig_pair(totals["over_price"], totals["under_price"])
    totals = totals.assign(no_vig_over_probability=over_prob, totals_hold=over_hold)
    totals = totals.merge(
        outcomes[["game_id", "home_score", "away_score"]], on="game_id", how="inner"
    ).merge(teams, on="game_id", how="left")
    points = pd.to_numeric(totals["home_score"], errors="coerce") + pd.to_numeric(
        totals["away_score"], errors="coerce"
    )
    total_margin = points - pd.to_numeric(totals["total_line"], errors="coerce")
    h5 = totals.loc[
        totals["no_vig_over_probability"].notna()
        & totals["no_vig_over_probability"].ne(0.5)
        & total_margin.ne(0.0)
    ].copy()
    h5["over_covers"] = (total_margin.loc[h5.index] > 0.0).astype(float)
    h5["pick_over"] = h5["no_vig_over_probability"].gt(0.5)
    h5["totals_juice_lean_over"] = h5["no_vig_over_probability"] - 0.5
    h5 = h5.reset_index(drop=True)

    close_full = close_reference_table(pairing_full, schedule)
    tue_open_full = pairing_tue[["game_id", "season", "week", "home_spread"]].rename(
        columns={"home_spread": "tue_open_home_spread"}
    )

    def _movement(terminal: pd.DataFrame, terminal_col: str) -> pd.DataFrame:
        paired = (
            tue_open_full.merge(terminal, on="game_id", how="inner")
            .merge(outcomes[["game_id", "result"]], on="game_id", how="inner")
            .merge(teams, on="game_id", how="left")
        )
        if paired.empty:
            return paired
        paired["margin_vs_open"] = paired["result"] - paired["tue_open_home_spread"]
        paired["line_move_home"] = paired[terminal_col] - paired["tue_open_home_spread"]
        paired["pick_home"] = paired["line_move_home"].gt(0.0)
        paired["home_cover_at_open"] = _home_cover_from_margin(paired["margin_vs_open"])
        paired = paired.loc[paired["line_move_home"].ne(0.0)]
        return paired.dropna(subset=["home_cover_at_open"]).reset_index(drop=True)

    paired_full = _movement(close_full, "close_home_spread")

    wed_manifests = omb._wednesday_noon_snapshots(market_root)
    wed_quotes_raw = omb._load_selected_snapshots(wed_manifests)
    wed_quotes = (
        omb._true_week_correct(wed_quotes_raw, schedule)
        if not wed_quotes_raw.empty
        else wed_quotes_raw
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
    paired_wed = _movement(wed_spread, "wed_home_spread")

    return {
        "h1": h1,
        "h2": h2,
        "h5": h5,
        "h6": h6,
        "paired_full": paired_full,
        "paired_wed": paired_wed,
        "tue_open_lines": tue_open_full[["game_id", "tue_open_home_spread"]],
        "tue_open_games": len(spread_novig),
        "wed_weeks_selected": len(wed_manifests),
        "close_source_counts": close_full["close_source"].value_counts().to_dict(),
    }


def build_public_betting(
    archive_path: Path, features_path: Path, tue_open_lines: pd.DataFrame
) -> dict[str, Any]:
    """Rebuild the public-betting battery's base population via its own helpers."""

    archive = pd.read_parquet(archive_path)
    features = pd.read_parquet(features_path)
    regular = regular_season_rows(features)
    matched = pbb._match_to_schedule(archive, regular)
    latest = pbb._latest_pregame_capture(matched)
    latest = latest.loc[
        latest["season"].between(pbb.BASE_SEASON_START, pbb.BASE_SEASON_END)
        & latest["result"].notna()
        & latest["spread_line"].notna()
    ].reset_index(drop=True)

    bet = latest.loc[
        latest["spread_home_bet_pct"].notna() & latest["spread_away_bet_pct"].notna()
    ].copy()
    threshold = pbb.FADE_THRESHOLD
    bet["heavy"] = bet["spread_home_bet_pct"].ge(threshold) | bet["spread_away_bet_pct"].ge(
        threshold
    )
    bet["public_side_home"] = bet["spread_home_bet_pct"].gt(bet["spread_away_bet_pct"])
    bet["fade_side_home"] = ~bet["public_side_home"]
    bet["home_cover_close"] = pd.to_numeric(bet["home_cover"], errors="coerce")
    bet = bet.merge(tue_open_lines, on="game_id", how="left")
    margin_vs_open = pd.to_numeric(bet["result"], errors="coerce") - pd.to_numeric(
        bet["tue_open_home_spread"], errors="coerce"
    )
    bet["home_cover_at_open"] = _home_cover_from_margin(margin_vs_open)
    bet = bet.reset_index(drop=True)

    era2 = latest.loc[latest["era"].eq("era2_scoreboard_response")].copy()
    era2 = era2.loc[
        era2["spread_home_bet_pct"].notna()
        & era2["spread_away_bet_pct"].notna()
        & era2["spread_home_money_pct"].notna()
        & era2["spread_away_money_pct"].notna()
    ].copy()
    era2["gap_home"] = era2["spread_home_money_pct"] - era2["spread_home_bet_pct"]
    era2["gap_away"] = era2["spread_away_money_pct"] - era2["spread_away_bet_pct"]
    era2["money_side_home"] = np.select(
        [
            era2["gap_home"].ge(pbb.DIVERGENCE_THRESHOLD),
            era2["gap_away"].ge(pbb.DIVERGENCE_THRESHOLD),
        ],
        [True, False],
        default=np.nan,
    )
    era2["home_cover_close"] = pd.to_numeric(era2["home_cover"], errors="coerce")
    era2 = era2.reset_index(drop=True)

    return {
        "latest": latest,
        "bet": bet,
        "era2": era2,
        "base_games": len(latest),
        "bet_games": len(bet),
        "era2_games": len(era2),
        "opener_matched_games": int(bet["tue_open_home_spread"].notna().sum()),
    }


def build_sagarin(artifacts_root: Path) -> dict[str, Any]:
    """Rebuild the Sagarin battery's three populations via its own builders."""

    sagarin_root = REPO / "data" / "raw" / "sagarin" / sag.DEFAULT_SAGARIN_SNAPSHOT
    schedules = sag.default_schedules()
    close_pop, close_coverage = sag.build_close_population(schedules, sagarin_root)
    open_pop, _coverage, open_note = sag.build_open_population(
        schedules, sagarin_root, repo_root=REPO
    )
    agreement_pop, agreement_note = sag.build_model_agreement_population(
        close_pop, artifacts_root=artifacts_root
    )
    return {
        "close_pop": close_pop.reset_index(drop=True),
        "open_pop": open_pop.reset_index(drop=True),
        "agreement_pop": agreement_pop.reset_index(drop=True),
        "close_coverage": close_coverage.to_dict(orient="records"),
        "open_note": open_note,
        "agreement_note": agreement_note,
        "sagarin_snapshot": str(sagarin_root),
    }


def build_sbr(tue_open_lines: pd.DataFrame) -> dict[str, Any]:
    """SBR-opener population from the proxy screen's own ``build_population``."""

    features = pd.read_parquet(WEAK_STACK_FEATURES)
    sbr_odds = pd.read_parquet(SBR_ODDS)
    population = pxy.build_population(features, sbr_odds, 2009, 2021)
    regular = regular_season_rows(features)
    teams = regular[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    population = population.merge(teams, on="game_id", how="left").reset_index(drop=True)

    overlap = population.merge(tue_open_lines, on="game_id", how="inner").copy()
    overlap["proxy_minus_true_open"] = (
        overlap["proxy_open_home_spread"] - overlap["tue_open_home_spread"]
    )
    overlap = overlap.reset_index(drop=True)

    def _load_scored(path: Path) -> pd.DataFrame:
        if not path.is_file():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        frame["home_cover_at_open_proxy"] = _home_cover_from_margin(frame["margin_vs_open_proxy"])
        return frame.dropna(subset=["home_cover_at_open_proxy"]).reset_index(drop=True)

    return {
        "population": population,
        "overlap": overlap,
        "scored": _load_scored(SBR_SCORED),
        "proxy_scored": _load_scored(PROXY_SCORED),
        "population_games": len(population),
        "overlap_games": len(overlap),
        "overlap_seasons": sorted(int(s) for s in overlap["season"].unique())
        if len(overlap)
        else [],
    }


# ---------------------------------------------------------------------------
# Entry -> parent quantity mapping
# ---------------------------------------------------------------------------

TUE_OPEN_BASIS = (
    "purchased point-in-time odds archive, capture_kind='historical_backfill', "
    "decision_label='tue_open' (data/market/raw)"
)
PUBLIC_BASIS = (
    "ActionNetwork public-betting archive 20260820T111148Z, latest pregame capture per game"
)
SAGARIN_BASIS = "Sagarin asof-Tuesday snapshot 20260820T112501Z x nflverse schedule spread_line"
SAGARIN_OPEN_BASIS = (
    "Sagarin snapshot 20260820T112501Z x the 1,537-game paired tue_open archive "
    "(nfl_ats.experiment_runner._opener_graded_features)"
)
SBR_BASIS = "data/processed/sbr_odds.parquet, via proxy_opener_replication.build_population"

#: frame key -> (market quantity, snapshot basis, unit label, method, method tag)
QUANTITIES: dict[str, tuple[str, str, str, str, str]] = {
    "juice_lean": (
        "no-vig juice lean off the tue_open spread PRICE, signed toward the team on the "
        "row (home row = no_vig_home_cover_probability - 0.5)",
        TUE_OPEN_BASIS,
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "juice_magnitude": (
        "|no_vig_home_cover_probability - 0.5|, the cell's own tercile variable",
        TUE_OPEN_BASIS,
        "home-team-season (venue proxy)",
        METHOD_VENUE_HOME_UNIT,
        "VENUE",
    ),
    "price_std": (
        "cross-book standard deviation of the HOME-side tue_open spread price, on games "
        "with >=3 both-sided books",
        TUE_OPEN_BASIS + ", raw book-level quotes",
        "home-team-season (venue proxy)",
        METHOD_VENUE_HOME_UNIT,
        "VENUE",
    ),
    "spread_hold": (
        "market HOLD on the tue_open spread (spread_hold from "
        "nfl_ats.novig.spread_novig_probabilities)",
        TUE_OPEN_BASIS,
        "home-team-season (venue proxy)",
        METHOD_VENUE_HOME_UNIT,
        "VENUE",
    ),
    "totals_juice_lean": (
        "totals-market no-vig juice lean toward OVER (no_vig_over_probability - 0.5) at tue_open",
        TUE_OPEN_BASIS + ", totals consensus OVER/UNDER prices",
        "home-team-season (venue proxy)",
        METHOD_VENUE_HOME_UNIT,
        "VENUE",
    ),
    "line_move": (
        "tue_open-to-close line MOVEMENT, signed toward the team on the row "
        "(home row = close_home_spread - tue_open_home_spread)",
        TUE_OPEN_BASIS + " paired with close_reference_table",
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "line_move_wed": (
        "tue_open-to-Wednesday-noon-ET line MOVEMENT, signed toward the team on the row",
        "purchased odds archive, tue_open paired with the intraday_hourly snapshot nearest "
        "Wed 12:00 ET (2023-2025 only; there is no wed_* decision label)",
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "public_bet_pct": (
        "public BET percentage on the team on the row (spread_home_bet_pct / spread_away_bet_pct)",
        PUBLIC_BASIS,
        "team-season (per side)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "money_minus_bet_gap": (
        "money%-minus-bet% gap on the team on the row (the cell's own sharp-divergence "
        "variable), era2 eligibility",
        PUBLIC_BASIS + ", era2_scoreboard_response rows carrying all four spread percentages",
        "team-season (per side)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "sagarin_divergence_close": (
        "Sagarin-implied spread minus the market CLOSE line, signed toward the team on the row",
        SAGARIN_BASIS,
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "sagarin_divergence_open": (
        "Sagarin-implied spread minus the tue_open OPENER line, signed toward the team on the row",
        SAGARIN_OPEN_BASIS,
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "sagarin_divergence_agreement": (
        "Sagarin-implied spread minus the market CLOSE line, signed toward the team on the "
        "row -- the quantity whose SIGN defines this cell's agree/disagree split",
        SAGARIN_BASIS + " x the active model's walk-forward predictions (2018-2025)",
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "sbr_open_line": (
        "SBR sportsbookreviewsonline 'Open' spread substituted as the settlement line, "
        "signed toward the team on the row",
        SBR_BASIS,
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
    "sbr_minus_true_open": (
        "SBR 'Open' MINUS the production tue_open line -- the proxy discount itself",
        SBR_BASIS + " paired with nfl_ats.clv.build_pairing_table's tue_open consensus",
        "team-season (signed)",
        rlib.METHOD_TRAIT,
        "TRAIT",
    ),
}

#: entry -> (battery, frame key, oracle control?, secondary frame keys)
CELL_TABLE: dict[str, tuple[str, str | None, bool, tuple[str, ...]]] = {
    "odds_microstructure_H1_1_2_dose_response_high_minus_low": (
        "odds_microstructure",
        "juice_lean",
        False,
        ("juice_magnitude",),
    ),
    "odds_microstructure_H2_2b_price_std_juice_accuracy_low_minus_high_dispersion": (
        "odds_microstructure",
        "price_std",
        False,
        (),
    ),
    "odds_microstructure_H3_3_0a_full_week_oracle_2020_2025_sanity_check": (
        "odds_microstructure",
        "line_move",
        True,
        (),
    ),
    "odds_microstructure_H3_3_0b_full_week_oracle_2023_2025_baseline": (
        "odds_microstructure",
        "line_move",
        True,
        (),
    ),
    "odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025": (
        "odds_microstructure",
        "line_move_wed",
        True,
        (),
    ),
    "odds_microstructure_H5_5_1_totals_juice_lean_accuracy": (
        "odds_microstructure",
        "totals_juice_lean",
        False,
        (),
    ),
    "odds_microstructure_H6_6_2_juice_accuracy_high_minus_low_hold": (
        "odds_microstructure",
        "spread_hold",
        False,
        (),
    ),
    "public_betting_battery_fade_heavy_public_close": (
        "public_betting_battery",
        "public_bet_pct",
        False,
        (),
    ),
    "public_betting_battery_fade_heavy_public_opener": (
        "public_betting_battery",
        "public_bet_pct",
        False,
        (),
    ),
    "public_betting_battery_model_interaction_against": (
        "public_betting_battery",
        "public_bet_pct",
        False,
        (),
    ),
    "public_betting_battery_model_interaction_diff": (
        "public_betting_battery",
        "public_bet_pct",
        False,
        (),
    ),
    "public_betting_battery_sharp_divergence_close": (
        "public_betting_battery",
        "money_minus_bet_gap",
        False,
        (),
    ),
    "sagarin_battery_large_divergence_close": (
        "sagarin_battery",
        "sagarin_divergence_close",
        False,
        (),
    ),
    "sagarin_battery_large_divergence_era_2010_2016": (
        "sagarin_battery",
        "sagarin_divergence_close",
        False,
        (),
    ),
    "sagarin_battery_large_divergence_era_2017_2025": (
        "sagarin_battery",
        "sagarin_divergence_close",
        False,
        (),
    ),
    "sagarin_battery_large_divergence_open": (
        "sagarin_battery",
        "sagarin_divergence_open",
        False,
        (),
    ),
    "sagarin_battery_model_agreement_close": (
        "sagarin_battery",
        "sagarin_divergence_agreement",
        False,
        (),
    ),
    "sagarin_battery_top_decile_close": (
        "sagarin_battery",
        "sagarin_divergence_close",
        False,
        (),
    ),
    "sagarin_battery_top_decile_open": (
        "sagarin_battery",
        "sagarin_divergence_open",
        False,
        (),
    ),
    "proxy_opener_production_rule_2009_2019": ("sbr_opener", "sbr_open_line", False, ()),
    "sbr_opener_era_2011_2014": ("sbr_opener", "sbr_open_line", False, ()),
    "sbr_opener_era_2015_2019": ("sbr_opener", "sbr_open_line", False, ()),
    "sbr_opener_era_2020_2021": (
        "sbr_opener",
        "sbr_open_line",
        False,
        ("sbr_minus_true_open",),
    ),
    "sbr_opener_pooled_2011_2021": ("sbr_opener", "sbr_open_line", False, ()),
    "mod08_smooth_cdf_mapping": ("mod08_cdf_mapping", None, False, ()),
    "mod08_smooth_cdf_mapping_opener": ("mod08_cdf_mapping", None, False, ()),
}


def build_long_frames(
    odds: dict[str, Any],
    public: dict[str, Any],
    sagarin: dict[str, Any],
    sbr: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Every parent quantity, exploded onto its measurement unit."""

    empty = pd.DataFrame(columns=["team_id", "season", "week"])

    def signed(frame: pd.DataFrame, column: str, metric: str) -> pd.DataFrame:
        if not len(frame):
            return empty.assign(**{metric: pd.Series(dtype=float)})
        return signed_team_week(frame, column, metric=metric)

    return {
        "juice_lean": signed(odds["h1"], "juice_lean_home", "juice_lean"),
        "juice_magnitude": home_unit_frame(odds["h1"], "juice_magnitude", metric="juice_magnitude"),
        "price_std": home_unit_frame(odds["h2"], "price_std", metric="price_std"),
        "spread_hold": home_unit_frame(odds["h6"], "spread_hold", metric="spread_hold"),
        "totals_juice_lean": home_unit_frame(
            odds["h5"], "totals_juice_lean_over", metric="totals_juice_lean"
        ),
        "line_move": signed(odds["paired_full"], "line_move_home", "line_move"),
        "line_move_wed": signed(odds["paired_wed"], "line_move_home", "line_move_wed"),
        "public_bet_pct": per_side_team_week(
            public["bet"], "spread_home_bet_pct", "spread_away_bet_pct", metric="public_bet_pct"
        ),
        "money_minus_bet_gap": per_side_team_week(
            public["era2"], "gap_home", "gap_away", metric="money_minus_bet_gap"
        )
        if len(public["era2"])
        else empty.assign(money_minus_bet_gap=pd.Series(dtype=float)),
        "sagarin_divergence_close": signed(
            sagarin["close_pop"], "divergence_close", "sagarin_divergence_close"
        ),
        "sagarin_divergence_open": signed(
            sagarin["open_pop"], "divergence_open", "sagarin_divergence_open"
        ),
        "sagarin_divergence_agreement": signed(
            sagarin["agreement_pop"], "divergence_close", "sagarin_divergence_agreement"
        ),
        "sbr_open_line": signed(sbr["population"], "proxy_open_home_spread", "sbr_open_line"),
        "sbr_minus_true_open": signed(
            sbr["overlap"], "proxy_minus_true_open", "sbr_minus_true_open"
        ),
    }


def build_replications(
    odds: dict[str, Any],
    public: dict[str, Any],
    sagarin: dict[str, Any],
    sbr: dict[str, Any],
) -> dict[str, tuple[pd.DataFrame, pd.Series, str]]:
    """Per-cell effect-replication inputs: (games, the rule's pick-home flag, outcome).

    Uniform framing, stated once: for a forced-pick cell the flag is the
    RULE'S OWN pick-home indicator and the outcome is home cover at that
    cell's grading line, so the odd/even-season gap reads as the rule's
    directional edge in each season half. Two-group cells (the model
    interaction and the Sagarin agreement cell) keep their own flag and their
    own accuracy column. Reported, never recorded.
    """

    out: dict[str, tuple[pd.DataFrame, pd.Series, str]] = {}

    def put(name: str, frame: pd.DataFrame, flag_col: str, outcome: str) -> None:
        if not len(frame) or flag_col not in frame or outcome not in frame:
            return
        frame = frame.reset_index(drop=True)
        out[name] = (frame, frame[flag_col].fillna(False).astype(bool), outcome)

    h1, h2, h5, h6 = odds["h1"], odds["h2"], odds["h5"], odds["h6"]
    full, wed = odds["paired_full"], odds["paired_wed"]
    put(
        "odds_microstructure_H1_1_2_dose_response_high_minus_low",
        h1,
        "pick_home",
        "home_cover_at_label",
    )
    put(
        "odds_microstructure_H2_2b_price_std_juice_accuracy_low_minus_high_dispersion",
        h2,
        "pick_home",
        "home_cover_at_label",
    )
    put(
        "odds_microstructure_H3_3_0a_full_week_oracle_2020_2025_sanity_check",
        full,
        "pick_home",
        "home_cover_at_open",
    )
    put(
        "odds_microstructure_H3_3_0b_full_week_oracle_2023_2025_baseline",
        full.loc[full["season"].between(2023, 2025)] if len(full) else full,
        "pick_home",
        "home_cover_at_open",
    )
    put(
        "odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025",
        wed,
        "pick_home",
        "home_cover_at_open",
    )
    put("odds_microstructure_H5_5_1_totals_juice_lean_accuracy", h5, "pick_over", "over_covers")
    put(
        "odds_microstructure_H6_6_2_juice_accuracy_high_minus_low_hold",
        h6,
        "pick_home",
        "home_cover_at_label",
    )

    bet = public["bet"]
    heavy = bet.loc[bet["heavy"]] if len(bet) else bet
    put(
        "public_betting_battery_fade_heavy_public_close",
        heavy,
        "fade_side_home",
        "home_cover_close",
    )
    put(
        "public_betting_battery_fade_heavy_public_opener",
        heavy.loc[heavy["season"].between(2020, 2025)] if len(heavy) else heavy,
        "fade_side_home",
        "home_cover_at_open",
    )
    put(
        "public_betting_battery_model_interaction_against",
        heavy,
        "public_side_home",
        "home_cover_close",
    )
    put(
        "public_betting_battery_model_interaction_diff",
        heavy,
        "public_side_home",
        "home_cover_close",
    )
    era2 = public["era2"]
    divergent = era2.loc[era2["money_side_home"].notna()] if len(era2) else era2
    if len(divergent):
        divergent = divergent.copy()
        divergent["money_side_home"] = divergent["money_side_home"].astype(bool)
        put(
            "public_betting_battery_sharp_divergence_close",
            divergent,
            "money_side_home",
            "home_cover_close",
        )

    close_pop, open_pop = sagarin["close_pop"], sagarin["open_pop"]
    for frame, column, prefix in (
        (close_pop, "divergence_close", "close"),
        (open_pop, "divergence_open", "open"),
    ):
        if not len(frame):
            continue
        work = frame.copy()
        work["sagarin_side_home"] = work[column].gt(0.0)
        work["home_cover_graded"] = pd.to_numeric(work["home_cover"], errors="coerce")
        big = work[column].abs().ge(sag.LARGE_DIVERGENCE_THRESHOLD)
        decile = work[column].abs().ge(float(work[column].abs().quantile(0.90)))
        names = {
            "close": (
                ("sagarin_battery_large_divergence_close", big),
                (
                    "sagarin_battery_large_divergence_era_2010_2016",
                    big & work["season"].between(2010, 2016),
                ),
                (
                    "sagarin_battery_large_divergence_era_2017_2025",
                    big & work["season"].between(2017, 2025),
                ),
                ("sagarin_battery_top_decile_close", decile),
            ),
            "open": (
                ("sagarin_battery_large_divergence_open", big),
                ("sagarin_battery_top_decile_open", decile),
            ),
        }[prefix]
        for name, mask in names:
            put(name, work.loc[mask], "sagarin_side_home", "home_cover_graded")

    agreement = sagarin["agreement_pop"]
    if len(agreement):
        agree = agreement.copy()
        agree["model_correct"] = pd.to_numeric(agree["model_correct"], errors="coerce")
        put("sagarin_battery_model_agreement_close", agree, "agree", "model_correct")

    scored, proxy_scored = sbr["scored"], sbr["proxy_scored"]
    era_windows = {
        "sbr_opener_era_2011_2014": (2011, 2014),
        "sbr_opener_era_2015_2019": (2015, 2019),
        "sbr_opener_era_2020_2021": (2020, 2021),
        "sbr_opener_pooled_2011_2021": (2011, 2021),
    }
    for name, (low, high) in era_windows.items():
        if not len(scored):
            continue
        put(
            name,
            scored.loc[scored["season"].between(low, high)],
            "pick_home_at_open_proxy",
            "home_cover_at_open_proxy",
        )
    if len(proxy_scored):
        put(
            "proxy_opener_production_rule_2009_2019",
            proxy_scored.loc[proxy_scored["season"].between(2011, 2019)],
            "pick_home_at_open_proxy",
            "home_cover_at_open_proxy",
        )
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def registry_seasons() -> dict[str, tuple[int, int]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    return {
        name: (int(signal.seasons[0]), int(signal.seasons[1]))
        for name, signal in registry.signals.items()
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    return value


def _measure_entry(
    entry: str,
    battery: str,
    frame_key: str,
    oracle: bool,
    secondaries: tuple[str, ...],
    *,
    frames: dict[str, pd.DataFrame],
    window: tuple[int, int],
    replication: tuple[pd.DataFrame, pd.Series, str] | None,
    n_boot: int,
    stability_boot: int,
) -> dict[str, Any]:
    quantity, basis, unit, method, tag = QUANTITIES[frame_key]
    long = frames[frame_key]
    metric = frame_key
    row: dict[str, Any] = {
        "entry": entry,
        "battery": battery,
        "market_quantity": quantity,
        "snapshot_basis": basis,
        "metric": metric,
        "unit": unit,
        "seasons": list(window),
        "method_tag": tag,
        "oracle_control": oracle,
        "reliability": None,
        "reliability_low": None,
        "reliability_high": None,
    }
    if oracle:
        row["oracle_caveat"] = ORACLE_CAVEAT
    if not len(long):
        row["status"] = NO_ROWS
        row["n_units"] = 0
        return row

    constant = near_constant_report(long, metric, seasons=window)
    measured = rlib.measure_reliability(long, metric, method=method, seasons=window, n_boot=n_boot)
    status = measured["status"]
    if status == rlib.STATUS_MEASURED and constant["near_constant"]:
        status = NEAR_CONSTANT
    recordable = status == rlib.STATUS_MEASURED
    seasons_seen = pd.to_numeric(long["season"], errors="coerce").dropna().astype(int)
    seasons_seen = seasons_seen.loc[seasons_seen.between(window[0], window[1])]
    row.update(
        {
            "n_units": measured["n_units"],
            "n_observations_in_window": int(constant.get("n_observations", 0)),
            "seasons_present": sorted(int(s) for s in seasons_seen.unique()),
            "pearson_r": measured["pearson_r"],
            "pearson_r_ci95": measured["pearson_r_ci95"],
            "spearman_rho": measured["spearman_rho"],
            "spearman_brown_full_length_reliability": measured[
                "spearman_brown_full_length_reliability"
            ],
            "probability_positive": measured["probability_positive"],
            "reliability": measured["reliability"] if recordable else None,
            "reliability_low": measured["reliability_low"] if recordable else None,
            "reliability_high": measured["reliability_high"] if recordable else None,
            "status": status,
            "method": measured["method"],
            "near_constant_check": constant,
            "window_sign_stability": window_sign_stability(
                long, metric, unit_col="team_id", seasons=window, n_boot=stability_boot
            ),
        }
    )

    if replication is not None:
        games, flag, outcome = replication
        row["half_season_replication"] = rlib.half_season_replication(
            games, flag, outcome_col=outcome
        )
    else:
        row["half_season_replication"] = {
            "status": "no_directional_flag_available",
            "note": (
                "This cell's own pick indicator is not reconstructable without re-running "
                "the screen's scoring pass; the sibling cells sharing its population carry "
                "the replication read."
            ),
        }

    extras: list[dict[str, Any]] = []
    for key in secondaries:
        extra_long = frames[key]
        extra_quantity, _basis, extra_unit, _method, extra_tag = QUANTITIES[key]
        if not len(extra_long):
            extras.append({"metric": key, "status": NO_ROWS, "quantity": extra_quantity})
            continue
        extra = rlib.measure_reliability(
            extra_long,
            key,
            method="secondary read, reported not recorded",
            seasons=window,
            n_boot=stability_boot,
        )
        extras.append(
            {
                "metric": key,
                "quantity": extra_quantity,
                "unit": extra_unit,
                "method_tag": extra_tag,
                "seasons": list(window),
                "n_units": extra["n_units"],
                "pearson_r": extra["pearson_r"],
                "reliability": extra["reliability"],
                "reliability_low": extra["reliability_low"],
                "reliability_high": extra["reliability_high"],
                "status": extra["status"],
            }
        )
    row["secondary_reads"] = extras
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    parser.add_argument("--control-boot", type=int, default=1000)
    parser.add_argument("--stability-boot", type=int, default=500)
    args = parser.parse_args()

    started = time.time()
    windows = registry_seasons()

    print("=== rebuilding the market-microstructure populations from the screens' builders ===")
    odds = build_odds_micro(MARKET_ROOT, PLAYER_FEATURES)
    print(
        f"  odds micro: tue_open base {odds['tue_open_games']} games | H1 {len(odds['h1'])} | "
        f"H2 price_std(>=3 books) {len(odds['h2'])} | H5 totals {len(odds['h5'])} | "
        f"H6 hold {len(odds['h6'])} | full-week movement {len(odds['paired_full'])} | "
        f"tue->wed movement {len(odds['paired_wed'])} ({odds['wed_weeks_selected']} snapshots)"
    )
    public = build_public_betting(pbb.DEFAULT_ARCHIVE, pbb.DEFAULT_FEATURES, odds["tue_open_lines"])
    print(
        f"  public betting: base {public['base_games']} | with bet% {public['bet_games']} | "
        f"era2 sharp-divergence eligible {public['era2_games']} | "
        f"opener-matched {public['opener_matched_games']}"
    )
    sagarin = build_sagarin(REPO / "artifacts")
    print(
        f"  sagarin: close {len(sagarin['close_pop'])} | open {len(sagarin['open_pop'])} | "
        f"model-agreement {len(sagarin['agreement_pop'])}"
    )
    sbr = build_sbr(odds["tue_open_lines"])
    print(
        f"  sbr: population {sbr['population_games']} | tue_open overlap "
        f"{sbr['overlap_games']} (seasons {sbr['overlap_seasons']})"
    )

    frames = build_long_frames(odds, public, sagarin, sbr)
    replications = build_replications(odds, public, sagarin, sbr)

    print("\n=== per-entry measurements ===")
    rows: list[dict[str, Any]] = []
    control_keys: dict[str, tuple[pd.DataFrame, tuple[int, int]]] = {}
    for entry, (battery, frame_key, oracle, secondaries) in CELL_TABLE.items():
        window = windows.get(entry)
        if frame_key is None:
            rows.append(
                {
                    "entry": entry,
                    "battery": battery,
                    "market_quantity": (
                        "probability-CALIBRATION mapping (Gaussian CDF vs raw ECDF) applied to "
                        "one fitted model's out-of-time residual sample"
                    ),
                    "snapshot_basis": (
                        "no market snapshot enters as a feature; the opener variant uses the "
                        "1,537-paired-game tue_open archive only to choose the settlement line"
                    ),
                    "metric": None,
                    "unit": "n/a",
                    "seasons": list(window) if window else None,
                    "method_tag": "N/A",
                    "method": None,
                    "oracle_control": False,
                    "n_units": 0,
                    "reliability": None,
                    "reliability_low": None,
                    "reliability_high": None,
                    "status": NOT_APPLICABLE,
                    "note": MOD08_NOTE,
                }
            )
            print(f"  {entry:<62} {NOT_APPLICABLE}")
            continue
        if window is None:
            rows.append({"entry": entry, "battery": battery, "status": "not_in_registry"})
            continue
        row = _measure_entry(
            entry,
            battery,
            frame_key,
            oracle,
            secondaries,
            frames=frames,
            window=window,
            replication=replications.get(entry),
            n_boot=args.n_boot,
            stability_boot=args.stability_boot,
        )
        rows.append(row)
        shown = f"{row['reliability']:+.4f}" if row["reliability"] is not None else "   n/a "
        print(
            f"  {entry:<62} {frame_key:<28} n={row.get('n_units', 0):>4} "
            f"rel={shown} {row['status']}"
        )
        if row["status"] in (rlib.STATUS_MEASURED, NEAR_CONSTANT):
            control_keys.setdefault(
                f"{frame_key}|{window[0]}-{window[1]}", (frames[frame_key], window)
            )

    print("\n=== positive controls (planted traits on this group's own unit structure) ===")
    controls: dict[str, list[dict[str, Any]]] = {}
    for key, (long, window) in sorted(control_keys.items()):
        seasons = pd.to_numeric(long["season"], errors="coerce")
        restricted = long.loc[seasons.between(window[0], window[1])]
        controls[key] = rlib.positive_control(restricted, n_boot=args.control_boot)
        recovered = ", ".join(
            f"{item['planted_unit_variance_share']:.1f}->"
            + (
                "n/a"
                if item["recovered_reliability"] is None
                else f"{item['recovered_reliability']:+.3f}"
            )
            for item in controls[key]
        )
        print(f"  {key:<48} n_units={controls[key][0]['n_units']:>4}  {recovered}")

    batteries: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        replication = row.get("half_season_replication") or {}
        if "odd_seasons" in replication:
            batteries.setdefault(row["battery"], {})[row["entry"]] = replication
    battery_correlations = {
        name: rlib.battery_replication_correlation(cells) for name, cells in batteries.items()
    }

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "market_micro" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "reliability-market-micro",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "entries": sorted(CELL_TABLE),
    }
    payload: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "control_n_boot": args.control_boot,
        "stability_n_boot": args.stability_boot,
        "min_units": rlib.MIN_UNITS,
        "sign_convention": (
            "A quantity with a team side is signed toward the team on its row: the home row "
            "carries the builder's own home-positive value, the away row its negation. The "
            "public bet/money percentages already exist per side, so each row carries its own "
            "side's number. A quantity with NO team side (market hold, cross-book price "
            "dispersion, totals over/under juice) is measured once per game at the home-team "
            "(venue-proxy) unit and tagged METHOD_VENUE."
        ),
        "builder_provenance": BUILDER_PROVENANCE,
        "entry_to_builder": {
            entry: BUILDER_FOR_BATTERY[battery]
            for entry, (battery, _key, _oracle, _extra) in CELL_TABLE.items()
        },
        "population_summary": {
            "odds_tue_open_games": odds["tue_open_games"],
            "odds_close_source_counts": odds["close_source_counts"],
            "odds_wed_weeks_selected": odds["wed_weeks_selected"],
            "public_betting_base_games": public["base_games"],
            "public_betting_games_with_bet_pct": public["bet_games"],
            "public_betting_era2_eligible_games": public["era2_games"],
            "public_betting_opener_matched_games": public["opener_matched_games"],
            "sagarin_close_games": len(sagarin["close_pop"]),
            "sagarin_open_games": len(sagarin["open_pop"]),
            "sagarin_agreement_games": len(sagarin["agreement_pop"]),
            "sagarin_open_note": sagarin["open_note"],
            "sagarin_agreement_note": sagarin["agreement_note"],
            "sagarin_snapshot": sagarin["sagarin_snapshot"],
            "sagarin_close_coverage_by_season": sagarin["close_coverage"],
            "sbr_population_games": sbr["population_games"],
            "sbr_tue_open_overlap_games": sbr["overlap_games"],
            "sbr_tue_open_overlap_seasons": sbr["overlap_seasons"],
        },
        "positive_control": controls,
        "battery_replication_correlation": battery_correlations,
        "results": rows,
        "closes_nothing": (
            "This artifact records reliabilities and NOTHING else. No entry is closed, "
            "reclassified, or assigned a closing_ground here; an interval containing zero is "
            "never a rejection."
        ),
        "provenance": artifact_provenance(configuration, PLAYER_FEATURES, project_root=REPO),
    }
    measured_rows = [row for row in rows if row.get("status") == rlib.STATUS_MEASURED]
    write_experiment_artifact(
        output_dir,
        "results.json",
        _jsonable(payload),
        command="reliability-market-micro",
        metrics={
            "n_entries": len(rows),
            "n_measured": len(measured_rows),
            "n_unmeasured": len(rows) - len(measured_rows),
        },
        notes=(
            "Measure-only split-half reliability for the 26 market_micro registry cells; every "
            "cell measured regardless of sign or interval shape, and nothing is closed or "
            "reclassified, per AGENTS.md's binding closing-grounds taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    print(f"\n{len(measured_rows)} of {len(rows)} measured; {len(rows) - len(measured_rows)} not")
    for label, predicate in (
        ("<= 0.10", lambda value: value <= 0.10),
        (">= 0.80", lambda value: value >= 0.80),
    ):
        hits = [row for row in measured_rows if predicate(row["reliability"])]
        print(f"  {label}: {len(hits)}")
        for row in sorted(hits, key=lambda item: item["reliability"]):
            flags = [] if row["method_tag"] == "TRAIT" else [row["method_tag"]]
            if row["oracle_control"]:
                flags.append("ORACLE CONTROL")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(
                f"    {row['entry']:<62} {row['reliability']:+.4f} "
                f"[{row['reliability_low']:+.4f}, {row['reliability_high']:+.4f}]{suffix}"
            )

    print("\n=== set-reliability commands (this script runs none) ===")
    source = str((output_dir / "results.json").relative_to(REPO)).replace("\\", "/")
    for row in measured_rows:
        reason = (
            f"ORCH-D reliability sweep, market_micro group: {row['market_quantity']}; measured "
            f"on the entry's own {row['seasons'][0]}-{row['seasons'][1]} window at the "
            f"{row['unit']} unit."
        )
        print(
            "nfl-ats weak-signals set-reliability "
            f"--name {row['entry']} "
            f"--reliability {row['reliability']:.6f} "
            f"--reliability-low {row['reliability_low']:.6f} "
            f"--reliability-high {row['reliability_high']:.6f} "
            f'--method "{row["method"]}" '
            f"--source {source} "
            f'--reason "{reason}"'
        )
    return 0


BUILDER_FOR_BATTERY: dict[str, str] = {
    "odds_microstructure": "scripts/odds_microstructure_battery.py",
    "public_betting_battery": "scripts/public_betting_battery_screen.py",
    "sagarin_battery": "scripts/sagarin_divergence_battery.py",
    "sbr_opener": "scripts/sbr_era_opener_eval.py + scripts/proxy_opener_replication.py",
    "mod08_cdf_mapping": (
        "scripts/smooth_cdf_mapping_measurement.py + "
        "scripts/smooth_cdf_mapping_opener_measurement.py"
    ),
}

BUILDER_PROVENANCE: dict[str, str] = {
    "odds_microstructure_*": (
        "scripts/odds_microstructure_battery.py -- base tue_open no-vig construction lines "
        "434-450, price dispersion 507-513, movement oracle 560-638, totals 707-731, hold "
        "745-748; its private helpers (_book_level_spread_prices, _true_week_correct, "
        "_totals_price_consensus, _no_vig_pair, _wednesday_noon_snapshots, "
        "_load_selected_snapshots) are IMPORTED, not copied. Read 2026-09-01."
    ),
    "public_betting_battery_*": (
        "scripts/public_betting_battery_screen.py -- _match_to_schedule (89-130) and "
        "_latest_pregame_capture (133-148) imported; the parent trait is "
        "spread_home_bet_pct/spread_away_bet_pct (314-320) and, for the sharp-divergence cell, "
        "spread_*_money_pct minus spread_*_bet_pct (392-393). Read 2026-09-01."
    ),
    "sagarin_battery_*": (
        "scripts/sagarin_divergence_battery.py -- build_close_population (250-264), "
        "build_open_population (267-290) and build_model_agreement_population (293-336) "
        "imported; the parent trait is add_divergence's divergence_close / divergence_open "
        "(224-230). Read 2026-09-01."
    ),
    "sbr_opener_* / proxy_opener_production_rule_2009_2019": (
        "scripts/proxy_opener_replication.py build_population (160-178) imported, which "
        "scripts/sbr_era_opener_eval.py itself imports unmodified (35-42); the parent quantity "
        "is sbr_odds.open_home_spread renamed proxy_open_home_spread. The construct's other "
        "half -- SBR Open minus the production tue_open line -- uses "
        "nfl_ats.clv.build_pairing_table, the same builder proxy_opener_replication's own "
        "calibration arm grades against. Read 2026-09-01."
    ),
    "mod08_smooth_cdf_mapping*": (
        "scripts/smooth_cdf_mapping_measurement.py:122-131 and "
        "scripts/smooth_cdf_mapping_opener_measurement.py:176-183 -- both arms share one fitted "
        "model, one residual sample and one predicted margin, differing only in "
        "nfl_ats.calibration.smoothed_home_cover_probability's method argument. No team-week "
        "trait exists to be reliable. Read 2026-09-01."
    ),
}


if __name__ == "__main__":
    raise SystemExit(main())
