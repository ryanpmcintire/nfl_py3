"""Pool tiebreaker: a defensible final-score guess for one game.

The pool breaks ties on the final score of the week's LAST game (owner,
2026-09-01; Week 1 that is DEN @ KC on Monday night). This module produces
the guess and, just as importantly, states how accurate such a guess can be.

Method, and why it is deliberately modest
-----------------------------------------
The market's own spread and total are the strong baseline this project holds
every model to, so the guess starts from the market-implied score::

    home = (total + home_expected_margin) / 2
    away = (total - home_expected_margin) / 2

and then calibrates it against every completed game with a recorded spread
and total (4,630 games, 2009-2025): the *neighborhood* of historically
similar market shapes supplies the median actual total, median actual home
margin, and the most common exact final scores. When the active weekly
forecast prices the game, the model's margin disagreement is blended in at
weight :data:`MODEL_RESIDUAL_WEIGHT` (0.2) -- see that constant's docstring
for the measurement showing why the model does NOT simply override the
market here the way it picks sides against it. Median-based numbers are the
right guess when the tiebreak metric is closest-total (median minimizes
absolute error); the exact-score modes are the right guess when the metric
is exact-score matching. Both are reported because the pool's metric is not
recorded anywhere in this repository.

Measured accuracy of the baseline itself (2009-2025, 4,630 games): the
market total misses the actual total by ~10.5 points on average (median 9.0,
bias +0.5 -- actuals run half a point OVER the line); each implied team
score misses by ~7.4 points. A tiebreaker guess is a coin toss weighted a
few points in your favour, not a prediction -- any write-up quoting this
module must keep that framing.

A dedicated over/under training regime (beating the market total
chronologically, the same discipline as the ATS work) is future work, queued
in ROADMAP.md; until something beats this baseline on held-out seasons, this
IS the tool.

Spread-sign conventions, stated once
------------------------------------
Two sources, two conventions, converted at the edge and nowhere else:

- ``schedules.parquet`` ``spread_line``: POSITIVE = home favored by that
  many (verified empirically: mean(actual home margin - spread_line) = +0.06
  over 4,630 games). This module's ``home_expected_margin`` equals it.
- Odds-snapshot ``quotes.parquet`` HOME outcome ``line``: NEGATIVE = home
  favored (a home side at -2.5 gives 2.5). ``home_expected_margin`` is its
  negation.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

#: Widening (margin, total) windows for the historical neighborhood, walked
#: in order until one holds ``_MIN_NEIGHBORHOOD`` games. The final ``None``
#: entry means "all of history" -- with 4,630 games the earlier windows
#: essentially always satisfy the floor first.
_NEIGHBORHOOD_WINDOWS: tuple[tuple[float, float] | None, ...] = (
    (1.0, 1.5),
    (1.5, 2.5),
    (2.5, 3.5),
    (3.5, 5.0),
    None,
)
_MIN_NEIGHBORHOOD = 150

#: Weight on the active model's margin disagreement when blending it into the
#: market margin for SCORE-GUESSING (not side-picking). Measured 2026-09-01 on
#: ``artifacts/opener_evaluation/20260819T174244Z`` (1,537 opener-graded
#: games, chronological prediction-level output): home-margin MAE is 9.912 at
#: k=0 (market alone), 9.906 at k=0.2 (the optimum, better in 5 of 6
#: seasons), and 10.003 at k=1 -- the RAW MODEL IS WORSE THAN THE MARKET as a
#: point estimate, even though it beats the market on sides. On the 487 games
#: where the model disagrees by >=2 points, no weight helps at all (9.800 at
#: k=0.3 vs 9.793 at k=0). Side-picking skill is not point-estimate skill:
#: the pick needs only P(margin > line) tilted past 50%, the tiebreaker needs
#: E[margin], and the market wins the latter. This constant is derived from
#: that sweep, not chosen (AGENTS.md: underived constants are defects).
MODEL_RESIDUAL_WEIGHT = 0.2


@dataclass(frozen=True)
class MarketConsensus:
    """One game's freshest market read: median across books in the newest
    local odds snapshot that quotes it, or the schedules row as fallback."""

    game_id: str
    home_expected_margin: float  # positive = home favored
    total_line: float
    source: str  # "snapshot <stamp> (<n> books)" or "schedules (fallback)"


@dataclass(frozen=True)
class ModelView:
    """The active model's margin opinion for the game, read from the newest
    weekly forecast that prices it -- the same numbers behind the played
    pick, shown so the guess can acknowledge a disagreement (e.g. Week 1
    DEN @ KC: market KC by 2.5, model KC by ~4.3) instead of silently
    ignoring it."""

    predicted_margin: float  # model's expected home margin
    forecast_line: float  # the spread_line the residual was measured against
    residual: float  # predicted_margin - forecast_line
    source: str


@dataclass(frozen=True)
class TiebreakerReport:
    game_id: str
    home: str
    away: str
    consensus: MarketConsensus
    #: The model's view when a forecast prices this game, else ``None``; the
    #: guess margin is then market + MODEL_RESIDUAL_WEIGHT * residual.
    model_view: ModelView | None
    #: The margin the guess is actually built from (blended when a model
    #: view exists, the market consensus alone otherwise).
    guess_margin: float
    implied_home: float
    implied_away: float
    neighborhood_games: int
    neighborhood_window: str
    median_total: float
    median_home_margin: float
    #: Integer guess: closest-total-optimal (median-based), margin-consistent.
    guess_home: int
    guess_away: int
    #: Most common exact ``(home_score, away_score)`` finals in the
    #: neighborhood, with counts -- the exact-score-metric guess.
    common_scores: tuple[tuple[int, int, int], ...]
    #: Whole-history honest error bars for the baseline.
    total_mae: float
    total_median_ae: float
    total_bias: float
    implied_score_mae: float


def newest_schedules_path(data_root: Path) -> Path:
    hits = sorted((data_root / "raw").glob("*/schedules.parquet"))
    if not hits:
        raise FileNotFoundError(f"no schedules.parquet under {data_root / 'raw'}")
    return hits[-1]


def lined_finals(schedules: pd.DataFrame) -> pd.DataFrame:
    """Completed games with a recorded spread and total."""

    mask = (
        schedules["home_score"].notna()
        & schedules["away_score"].notna()
        & schedules["spread_line"].notna()
        & schedules["total_line"].notna()
    )
    return schedules.loc[mask]


def last_game_of_week(schedules: pd.DataFrame, season: int, week: int) -> pd.Series:
    """The week's last kickoff -- the pool's tiebreaker game -- by
    ``(gameday, gametime)``."""

    games = schedules.loc[
        (schedules["season"] == season)
        & (schedules["week"] == week)
        & (schedules["game_type"].astype(str) == "REG")
    ]
    if games.empty:
        raise ValueError(f"no REG games for season {season} week {week}")
    keys = games["gameday"].astype(str)
    if "gametime" in games.columns:
        keys = keys + " " + games["gametime"].astype(str).fillna("")
    last: pd.Series = games.loc[keys.sort_values().index[-1]]
    return last


def upcoming_week(schedules: pd.DataFrame, today: date) -> tuple[int, int]:
    """The (season, week) of the next REG game on or after ``today`` --
    the week whose card is currently in play."""

    regular = schedules.loc[schedules["game_type"].astype(str) == "REG"].copy()
    days = pd.to_datetime(regular["gameday"], errors="coerce")
    ahead = regular.loc[days.dt.date >= today]
    if ahead.empty:
        raise ValueError(f"no REG games on or after {today.isoformat()}")
    first = ahead.loc[pd.to_datetime(ahead["gameday"]).sort_values().index[0]]
    return int(first["season"]), int(first["week"])


def market_implied_scores(home_expected_margin: float, total_line: float) -> tuple[float, float]:
    home = (total_line + home_expected_margin) / 2.0
    away = (total_line - home_expected_margin) / 2.0
    return home, away


def snapshot_consensus(game_id: str, data_root: Path) -> MarketConsensus | None:
    """Median spread/total across books in the NEWEST snapshot quoting the
    game. Walks snapshots newest-first so one capture missing the game (an
    early-week partial board) falls back to the one before it."""

    snapshots = sorted((data_root / "market" / "raw").glob("*/quotes.parquet"), reverse=True)
    for quotes_path in snapshots:
        quotes = pd.read_parquet(
            quotes_path,
            columns=["nflverse_game_id", "market", "outcome_side", "line", "bookmaker_key"],
        )
        rows = quotes.loc[quotes["nflverse_game_id"].astype(str).eq(game_id)]
        if rows.empty:
            continue
        spreads = rows.loc[(rows["market"] == "spreads") & (rows["outcome_side"] == "HOME")]
        totals = rows.loc[(rows["market"] == "totals") & (rows["outcome_side"] == "OVER")]
        if spreads.empty or totals.empty:
            continue
        # One line per book (a book quotes each market once per snapshot,
        # but groupby-first makes that an invariant rather than a hope).
        spread_by_book = spreads.groupby("bookmaker_key")["line"].first()
        total_by_book = totals.groupby("bookmaker_key")["line"].first()
        return MarketConsensus(
            game_id=game_id,
            home_expected_margin=-float(spread_by_book.median()),
            total_line=float(total_by_book.median()),
            source=f"snapshot {quotes_path.parent.name} ({len(spread_by_book)} books)",
        )
    return None


def active_model_view(game_id: str, artifacts_root: Path) -> ModelView | None:
    """The active method's ``predicted_market_residual`` for the game, from
    the newest weekly forecast that prices it. ``None`` when no forecast
    covers the game (a historical query) or the artifact tree is absent (a
    fresh clone) -- the guess then simply uses the market alone."""

    active_path = artifacts_root / "active_ats_model.json"
    if not active_path.is_file():
        return None
    method = str(json.loads(active_path.read_text(encoding="utf-8")).get("method", ""))
    if not method:
        return None
    forecasts = sorted(
        (artifacts_root / "margin_predictions").glob("*/predictions.csv"), reverse=True
    )
    for predictions_path in forecasts:
        frame = pd.read_csv(predictions_path)
        required = {"game_id", "method", "predicted_margin", "predicted_market_residual"}
        if not required.issubset(frame.columns):
            continue
        rows = frame.loc[
            frame["game_id"].astype(str).eq(game_id)
            & frame["method"].astype(str).eq(method)
            & frame["predicted_market_residual"].notna()
        ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        return ModelView(
            predicted_margin=float(row["predicted_margin"]),
            forecast_line=float(row["spread_line"]),
            residual=float(row["predicted_market_residual"]),
            source=f"forecast {predictions_path.parent.name} ({method})",
        )
    return None


def _neighborhood(
    finals: pd.DataFrame, home_expected_margin: float, total_line: float
) -> tuple[pd.DataFrame, str]:
    for window in _NEIGHBORHOOD_WINDOWS:
        if window is None:
            return finals, "all history"
        margin_width, total_width = window
        rows = finals.loc[
            (finals["spread_line"] - home_expected_margin).abs().le(margin_width)
            & (finals["total_line"] - total_line).abs().le(total_width)
        ]
        if len(rows) >= _MIN_NEIGHBORHOOD:
            return rows, f"±{margin_width:g} margin, ±{total_width:g} total"
    return finals, "all history"  # pragma: no cover - the None entry returns first


def build_report(
    game: pd.Series,
    consensus: MarketConsensus,
    finals: pd.DataFrame,
    model_view: ModelView | None = None,
) -> TiebreakerReport:
    guess_margin = consensus.home_expected_margin
    if model_view is not None:
        guess_margin += MODEL_RESIDUAL_WEIGHT * model_view.residual
    implied_home, implied_away = market_implied_scores(guess_margin, consensus.total_line)
    neighborhood, window_label = _neighborhood(finals, guess_margin, consensus.total_line)
    actual_totals = neighborhood["home_score"] + neighborhood["away_score"]
    actual_margins = neighborhood["home_score"] - neighborhood["away_score"]
    median_total = float(actual_totals.median())
    median_margin = float(actual_margins.median())
    guess_total = round(median_total)
    guess_home = round((guess_total + median_margin) / 2.0)
    guess_away = guess_total - guess_home
    score_counts = Counter(
        (int(home_score), int(away_score))
        for home_score, away_score in zip(
            neighborhood["home_score"], neighborhood["away_score"], strict=True
        )
    )
    common = tuple(
        (home_score, away_score, count)
        for (home_score, away_score), count in score_counts.most_common(3)
    )

    total_error = (finals["home_score"] + finals["away_score"]) - finals["total_line"]
    implied_home_all = (finals["total_line"] + finals["spread_line"]) / 2.0
    implied_away_all = (finals["total_line"] - finals["spread_line"]) / 2.0
    implied_mae = float(
        pd.concat(
            [
                (finals["home_score"] - implied_home_all).abs(),
                (finals["away_score"] - implied_away_all).abs(),
            ]
        ).mean()
    )
    return TiebreakerReport(
        game_id=str(game["game_id"]),
        home=str(game["home_team"]),
        away=str(game["away_team"]),
        consensus=consensus,
        model_view=model_view,
        guess_margin=guess_margin,
        implied_home=implied_home,
        implied_away=implied_away,
        neighborhood_games=len(neighborhood),
        neighborhood_window=window_label,
        median_total=median_total,
        median_home_margin=median_margin,
        guess_home=guess_home,
        guess_away=guess_away,
        common_scores=common,
        total_mae=float(total_error.abs().mean()),
        total_median_ae=float(total_error.abs().median()),
        total_bias=float(total_error.mean()),
        implied_score_mae=implied_mae,
    )


def tiebreaker_report(
    data_root: Path,
    *,
    artifacts_root: Path | None = None,
    season: int | None = None,
    week: int | None = None,
    game_id: str | None = None,
    today: date | None = None,
) -> TiebreakerReport:
    """The full pipeline: resolve the game, read the freshest market, blend
    in the active model's view (weight :data:`MODEL_RESIDUAL_WEIGHT`), build
    the calibrated guess. ``game_id`` overrides ``season``/``week``; with
    neither, the week of the next upcoming game is used."""

    schedules = pd.read_parquet(newest_schedules_path(data_root))
    if game_id is not None:
        rows = schedules.loc[schedules["game_id"].astype(str).eq(game_id)]
        if rows.empty:
            raise ValueError(f"game_id {game_id!r} not in schedules")
        game = rows.iloc[0]
    else:
        if season is None or week is None:
            season, week = upcoming_week(schedules, today or date.today())
        game = last_game_of_week(schedules, season, week)

    consensus = snapshot_consensus(str(game["game_id"]), data_root)
    if consensus is None:
        if pd.isna(game.get("spread_line")) or pd.isna(game.get("total_line")):
            raise ValueError(
                f"no odds snapshot quotes {game['game_id']} and schedules has no line for it"
            )
        consensus = MarketConsensus(
            game_id=str(game["game_id"]),
            home_expected_margin=float(game["spread_line"]),
            total_line=float(game["total_line"]),
            source="schedules (fallback -- possibly stale)",
        )
    model_view = (
        active_model_view(str(game["game_id"]), artifacts_root)
        if artifacts_root is not None
        else None
    )
    return build_report(game, consensus, lined_finals(schedules), model_view)


def format_report(report: TiebreakerReport) -> str:
    lines = [
        f"tiebreaker guess -- {report.away} at {report.home} ({report.game_id})",
        f"market: {report.home} by {report.consensus.home_expected_margin:g}, "
        f"total {report.consensus.total_line:g}  [{report.consensus.source}]",
    ]
    if report.model_view is not None:
        lines += [
            f"model view: expects {report.home} by {report.model_view.predicted_margin:.2f} "
            f"vs the {report.model_view.forecast_line:g} forecast line "
            f"(disagreement {report.model_view.residual:+.2f})  "
            f"[{report.model_view.source}]",
            f"guess margin blends it at weight {MODEL_RESIDUAL_WEIGHT:g} (measured optimum "
            f"-- the raw model is WORSE than the market as a point estimate): "
            f"{report.home} by {report.guess_margin:.2f}",
        ]
    lines += [
        f"implied score at the guess margin: {report.home} {report.implied_home:.2f}, "
        f"{report.away} {report.implied_away:.2f}",
        "",
        f"calibration neighborhood: {report.neighborhood_games} similar games "
        f"({report.neighborhood_window})",
        f"  median actual total {report.median_total:g}, "
        f"median home margin {report.median_home_margin:+g}",
        "",
        f"GUESS (closest-total metric): {report.home} {report.guess_home}, "
        f"{report.away} {report.guess_away}"
        f"  (total {report.guess_home + report.guess_away})",
        "most common exact finals in the neighborhood (exact-score metric):",
    ]
    for home_score, away_score, count in report.common_scores:
        lines.append(f"  {report.home} {home_score} - {report.away} {away_score}  ({count}x)")
    lines += [
        "",
        "honest error bars (all 2009-2025 lined finals): the market total "
        f"misses by {report.total_mae:.1f} on average (median {report.total_median_ae:.1f}, "
        f"bias {report.total_bias:+.1f}); each implied team score misses by "
        f"{report.implied_score_mae:.1f}.",
    ]
    return "\n".join(lines)
