from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.lineage import TiebreakerSource, parse_snapshot_capture
from nfl_ats.served_total import (
    SERVED_TOTAL_METHOD,
    ServedTotalMethod,
    joint_residual_total_view,
    served_total,
    served_total_blend_k01,
)
from nfl_ats.totals import TotalsDataError, TotalsView, model_total_view
from nfl_ats.totals_wave2 import model_total_view_wave2

BUILDER_VERSION = "v1"

_NEIGHBORHOOD_WINDOWS: tuple[tuple[float, float] | None, ...] = (
    (1.0, 1.5),
    (1.5, 2.5),
    (2.5, 3.5),
    (3.5, 5.0),
    None,
)

_BANDWIDTH_SCHEDULE: tuple[tuple[float, float], ...] = tuple(
    window for window in _NEIGHBORHOOD_WINDOWS if window is not None
)

_MIN_NEIGHBORHOOD = 150

_BANDWIDTH_BISECTION_STEPS = 40

_MEDIAN_TIE_TOLERANCE = 1e-12

MODEL_RESIDUAL_WEIGHT = 0.2

TOTALS_RESIDUAL_WEIGHT = 0.1


@dataclass(frozen=True)
class MarketConsensus:
    game_id: str
    home_expected_margin: float
    total_line: float
    source: str


@dataclass(frozen=True)
class ModelView:
    predicted_margin: float
    forecast_line: float
    residual: float
    source: str


@dataclass(frozen=True)
class TiebreakerReport:
    game_id: str
    home: str
    away: str
    consensus: MarketConsensus
    model_view: ModelView | None
    totals_view: TotalsView | None
    guess_margin: float
    guess_total_line: float
    served_total_method: ServedTotalMethod
    comparison_total_blend_k01: float
    implied_home: float
    implied_away: float
    neighborhood_games: int
    neighborhood_window: str
    median_total: float
    median_home_margin: float
    guess_home: int
    guess_away: int
    common_scores: tuple[tuple[int, int, float], ...]
    total_mae: float
    total_median_ae: float
    total_bias: float
    implied_score_mae: float
    pick_side: str | None = None
    pick_spread_line: float | None = None
    pick_cover_probability: float | None = None
    pick_push_probability: float | None = None
    consistency_note: str = ""

    @property
    def served_total(self) -> float:

        return self.guess_total_line


def newest_schedules_path(data_root: Path) -> Path:
    hits = sorted((data_root / "raw").glob("*/schedules.parquet"))
    if not hits:
        raise FileNotFoundError(f"no schedules.parquet under {data_root / 'raw'}")
    return hits[-1]


def lined_finals(schedules: pd.DataFrame) -> pd.DataFrame:

    mask = (
        schedules["home_score"].notna()
        & schedules["away_score"].notna()
        & schedules["spread_line"].notna()
        & schedules["total_line"].notna()
    )
    return schedules.loc[mask]


def last_game_of_week(schedules: pd.DataFrame, season: int, week: int) -> pd.Series:

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

    active_path = artifacts_root / "active_ats_model.json"
    if not active_path.is_file():
        return None
    active = json.loads(active_path.read_text(encoding="utf-8"))
    method = str(active.get("method", ""))
    if not method:
        return None
    linked = active.get("weekly_forecast", {}).get("artifact")
    if not linked:
        return None
    forecast_dir = artifacts_root / linked
    metadata_path = forecast_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not active.get("model_id") or metadata.get("active_model_id") != active["model_id"]:
        raise TiebreakerConsistencyError("Linked forecast model ID does not match the active model")
    forecasts = [forecast_dir / "predictions.csv"]
    if not forecasts[0].is_file():
        return None
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


@dataclass(frozen=True)
class Neighborhood:
    frame: pd.DataFrame
    weights: npt.NDArray[np.float64]
    label: str
    effective_size: float


def kernel_weights(
    finals: pd.DataFrame,
    home_expected_margin: float,
    total_line: float,
    margin_bandwidth: float,
    total_bandwidth: float,
) -> npt.NDArray[np.float64]:

    margins = finals["spread_line"].to_numpy(dtype=float)
    totals = finals["total_line"].to_numpy(dtype=float)
    distance = np.sqrt(
        ((margins - home_expected_margin) / margin_bandwidth) ** 2
        + ((totals - total_line) / total_bandwidth) ** 2
    )
    weights: npt.NDArray[np.float64] = np.clip(1.0 - distance, 0.0, None)
    return weights


def effective_sample_size(weights: npt.NDArray[np.float64]) -> float:

    total = float(weights.sum())
    squared = float((weights**2).sum())
    if squared <= 0.0:
        return 0.0
    return total * total / squared


def _bandwidths_at(scale: float) -> tuple[float, float]:

    last = len(_BANDWIDTH_SCHEDULE) - 1
    lower_index = min(max(math.floor(scale), 0), last)
    upper_index = min(lower_index + 1, last)
    fraction = min(max(scale - lower_index, 0.0), 1.0)
    lower_margin, lower_total = _BANDWIDTH_SCHEDULE[lower_index]
    upper_margin, upper_total = _BANDWIDTH_SCHEDULE[upper_index]
    return (
        lower_margin + fraction * (upper_margin - lower_margin),
        lower_total + fraction * (upper_total - lower_total),
    )


def _label(margin_bandwidth: float, total_bandwidth: float) -> str:
    return f"±{margin_bandwidth:.2f} margin, ±{total_bandwidth:.2f} total"


def _neighborhood(
    finals: pd.DataFrame, home_expected_margin: float, total_line: float
) -> Neighborhood:

    def weights_at(scale: float) -> npt.NDArray[np.float64]:
        return kernel_weights(finals, home_expected_margin, total_line, *_bandwidths_at(scale))

    def pack(scale: float, weights: npt.NDArray[np.float64]) -> Neighborhood:
        positive = weights > 0.0
        return Neighborhood(
            frame=finals.loc[positive],
            weights=weights[positive],
            label=_label(*_bandwidths_at(scale)),
            effective_size=effective_sample_size(weights),
        )

    base = weights_at(0.0)
    if effective_sample_size(base) >= _MIN_NEIGHBORHOOD:
        return pack(0.0, base)

    upper: float | None = None
    for index in range(1, len(_BANDWIDTH_SCHEDULE)):
        if effective_sample_size(weights_at(float(index))) >= _MIN_NEIGHBORHOOD:
            upper = float(index)
            break
    if upper is None:
        return Neighborhood(
            frame=finals,
            weights=np.ones(len(finals), dtype=float),
            label="all history",
            effective_size=float(len(finals)),
        )

    lower = upper - 1.0
    for _ in range(_BANDWIDTH_BISECTION_STEPS):
        middle = (lower + upper) / 2.0
        if effective_sample_size(weights_at(middle)) >= _MIN_NEIGHBORHOOD:
            upper = middle
        else:
            lower = middle
    return pack(upper, weights_at(upper))


def weighted_median(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:

    if len(values) == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1])
    if total <= 0.0:
        return float("nan")
    half = total / 2.0
    index = min(int(np.searchsorted(cumulative, half, side="left")), len(ordered_values) - 1)
    on_the_half_point = math.isclose(
        float(cumulative[index]), half, rel_tol=_MEDIAN_TIE_TOLERANCE, abs_tol=0.0
    )
    if on_the_half_point and index + 1 < len(ordered_values):
        return float((ordered_values[index] + ordered_values[index + 1]) / 2.0)
    return float(ordered_values[index])


def weighted_score_counts(
    frame: pd.DataFrame, weights: npt.NDArray[np.float64]
) -> dict[tuple[int, int], float]:

    counts: dict[tuple[int, int], float] = {}
    for home_score, away_score, weight in zip(
        frame["home_score"], frame["away_score"], weights, strict=True
    ):
        key = (int(home_score), int(away_score))
        counts[key] = counts.get(key, 0.0) + float(weight)
    return counts


class TiebreakerConsistencyError(ValueError):
    pass


def build_report(
    game: pd.Series,
    consensus: MarketConsensus,
    finals: pd.DataFrame,
    model_view: ModelView | None = None,
    totals_view: TotalsView | None = None,
    joint_totals_view: TotalsView | None = None,
    *,
    published_pick_side: str | None = None,
) -> TiebreakerReport:
    guess_margin = consensus.home_expected_margin
    if model_view is not None:
        guess_margin += MODEL_RESIDUAL_WEIGHT * model_view.residual
    comparison_total_blend_k01 = served_total_blend_k01(
        consensus.total_line, totals_view, weight=TOTALS_RESIDUAL_WEIGHT
    )
    guess_total_line, served_total_method = served_total(
        SERVED_TOTAL_METHOD,
        market_total=consensus.total_line,
        blend_view=totals_view,
        joint_view=joint_totals_view,
        blend_weight=TOTALS_RESIDUAL_WEIGHT,
    )
    implied_home, implied_away = market_implied_scores(guess_margin, guess_total_line)
    neighborhood = _neighborhood(finals, guess_margin, guess_total_line)
    rows, weights = neighborhood.frame, neighborhood.weights
    actual_totals = (rows["home_score"] + rows["away_score"]).to_numpy(dtype=float)
    actual_margins = (rows["home_score"] - rows["away_score"]).to_numpy(dtype=float)
    median_total = weighted_median(actual_totals, weights)
    median_margin = weighted_median(actual_margins, weights)

    pick_side: str | None = None
    pick_spread_line: float | None = None
    pick_cover_probability: float | None = None
    pick_push_probability: float | None = None
    consistency_note = ""
    if model_view is not None and (published_pick_side is not None or model_view.residual != 0.0):
        import nfl_ats.score_lattice as score_lattice_module

        pick_side = published_pick_side or ("HOME" if model_view.residual > 0.0 else "AWAY")
        if pick_side not in {"HOME", "AWAY"}:
            raise TiebreakerConsistencyError("Invalid published pick side")
        pick_spread_line = model_view.forecast_line
        try:
            lattice = score_lattice_module.score_lattice(
                finals, model_view.predicted_margin, guess_total_line
            )
        except ValueError as error:
            raise TiebreakerConsistencyError(
                f"{game['game_id']}: could not build a score lattice ({error}) -- refusing "
                "to publish an inconsistent tiebreaker guess"
            ) from error
        chosen = score_lattice_module.pick_consistent_top_score(
            lattice,
            pick_side=pick_side,
            spread_line=pick_spread_line,
            served_total=guess_total_line,
            centre_margin=model_view.predicted_margin,
        )
        if chosen is None:
            raise TiebreakerConsistencyError(
                f"{game['game_id']}: no final on the {pick_side} side of "
                f"{pick_spread_line:g} both sits within a total-proximity tolerance of "
                f"the served total {guess_total_line:.2f} AND lands within the "
                "score-lattice hard guard (3 points of the centre "
                f"({model_view.predicted_margin:.2f}, {guess_total_line:.2f}) on both axes) "
                "-- refusing to publish a tail-score tiebreaker guess"
            )
        guess_home, guess_away, _cell_probability, total_tolerance = chosen
        rounded_total = guess_home + guess_away
        if abs(rounded_total - guess_total_line) > total_tolerance + 1e-9:
            raise TiebreakerConsistencyError(
                f"{game['game_id']}: the lattice-consistent score totals {rounded_total}, "
                f"more than {total_tolerance:g} point(s) from the served total "
                f"{guess_total_line:.2f} -- refusing to publish an inconsistent tiebreaker "
                "guess"
            )
        pick_cover_probability = score_lattice_module.pick_cover_probability(
            lattice, pick_side=pick_side, spread_line=pick_spread_line
        )
        pick_push_probability = lattice.push_probability(pick_spread_line)
        pick_team = str(game["home_team"]) if pick_side == "HOME" else str(game["away_team"])
        team_line = -pick_spread_line if pick_side == "HOME" else pick_spread_line
        pick_line_text = "pick'em" if team_line == 0 else f"{team_line:+g}"
        widened_note = (
            f"; total tolerance widened to {total_tolerance:g} points (no candidate within 1)"
            if total_tolerance > 1.0
            else ""
        )
        consistency_note = f"consistent with the {pick_team} {pick_line_text} pick{widened_note}"
    else:
        guess_total = round(median_total)
        guess_home = round((guess_total + median_margin) / 2.0)
        guess_away = guess_total - guess_home
    score_counts = weighted_score_counts(rows, weights)
    ranked = sorted(score_counts.items(), key=lambda item: (-item[1], item[0]))
    common = tuple(
        (home_score, away_score, count) for (home_score, away_score), count in ranked[:3]
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
        totals_view=totals_view,
        guess_margin=guess_margin,
        guess_total_line=guess_total_line,
        served_total_method=served_total_method,
        comparison_total_blend_k01=comparison_total_blend_k01,
        implied_home=implied_home,
        implied_away=implied_away,
        neighborhood_games=round(neighborhood.effective_size),
        neighborhood_window=neighborhood.label,
        median_total=median_total,
        median_home_margin=median_margin,
        guess_home=guess_home,
        guess_away=guess_away,
        common_scores=common,
        pick_side=pick_side,
        pick_spread_line=pick_spread_line,
        pick_cover_probability=pick_cover_probability,
        pick_push_probability=pick_push_probability,
        consistency_note=consistency_note,
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
    features_path: Path | None = None,
    wave2_features_path: Path | None = None,
    joint_features_path: Path | None = None,
    forecast_row: pd.Series | None = None,
    forecast_model_id: str | None = None,
    forecast_artifact: str | None = None,
    model_id: str | None = None,
    published_pick_side: str | None = None,
    frozen_spread: float | None = None,
) -> TiebreakerReport:

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
    model_view: ModelView | None
    if forecast_row is not None:
        if not model_id or model_id != forecast_model_id:
            raise TiebreakerConsistencyError("Forecast model ID does not match the verified model")
        if str(forecast_row.get("game_id")) != str(game["game_id"]):
            raise TiebreakerConsistencyError("Forecast row does not match the tiebreaker game")
        if published_pick_side not in {"HOME", "AWAY"} or frozen_spread is None:
            raise TiebreakerConsistencyError("Published pick side and frozen spread are required")
        if float(forecast_row["spread_line"]) != frozen_spread:
            raise TiebreakerConsistencyError("Frozen spread does not match the forecast row")
        model_view = ModelView(
            predicted_margin=float(forecast_row["predicted_margin"]),
            forecast_line=frozen_spread,
            residual=float(forecast_row["predicted_market_residual"]),
            source=f"forecast {forecast_artifact or 'published row'} ({model_id})",
        )
    else:
        if any(
            value is not None
            for value in (
                model_id,
                forecast_model_id,
                forecast_artifact,
                published_pick_side,
                frozen_spread,
            )
        ):
            raise TiebreakerConsistencyError("Published forecast row is required")
        model_view = (
            active_model_view(str(game["game_id"]), artifacts_root)
            if artifacts_root is not None
            else None
        )
    wave1_features = (
        features_path
        if features_path is not None
        else data_root / "processed" / "game_features.parquet"
    )
    wave2_features = (
        wave2_features_path
        if wave2_features_path is not None
        else data_root / "processed" / "game_features_pbp.parquet"
    )
    if wave2_features.is_file():
        try:
            totals_view = model_total_view_wave2(str(game["game_id"]), data_root, wave2_features)
        except (TotalsDataError, KeyError, OSError, TypeError, ValueError):
            totals_view = None
    else:
        totals_view = model_total_view(str(game["game_id"]), data_root, wave1_features)
        if totals_view is not None:
            totals_view = replace(
                totals_view,
                source=f"wave 1 fallback (PBP table absent) -- {totals_view.source}",
            )
    joint_features = (
        joint_features_path
        if joint_features_path is not None
        else data_root / "processed" / "game_features_weak_stack.parquet"
    )
    try:
        joint_totals_view = joint_residual_total_view(
            str(game["game_id"]), data_root, features_path=joint_features
        )
    except (TotalsDataError, KeyError, OSError, TypeError, ValueError):
        joint_totals_view = None
    return build_report(
        game,
        consensus,
        lined_finals(schedules),
        model_view,
        totals_view,
        joint_totals_view,
        published_pick_side=published_pick_side,
    )


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
    if report.totals_view is not None:
        lines += [
            f"model total view: expects {report.totals_view.predicted_total:.2f} "
            f"vs the {report.totals_view.market_total:g} market total "
            f"(disagreement {report.totals_view.residual:+.2f})  "
            f"[{report.totals_view.source}]",
            f"comparison arm blend_k01 blends it at weight {TOTALS_RESIDUAL_WEIGHT:g}: "
            f"{report.comparison_total_blend_k01:.2f}",
        ]
    lines += [
        f"served total ({report.served_total_method}, MOD-17 nfl_ats.served_total): "
        f"{report.served_total:.2f}",
    ]
    lines += [
        f"implied score at the guess margin: {report.home} {report.implied_home:.2f}, "
        f"{report.away} {report.implied_away:.2f}",
        "",
        f"calibration neighborhood: effective {report.neighborhood_games} similar games "
        f"(kernel-weighted; {report.neighborhood_window})",
        f"  median actual total {report.median_total:g}, "
        f"median home margin {report.median_home_margin:+g}"
        + (" (reference only -- see the lattice guess below)" if report.pick_side else ""),
        "",
    ]
    if report.pick_side is not None:
        lines += [
            "GUESS (one lattice, one margin, one total -- see docs/tiebreaker.md): "
            f"{report.home} {report.guess_home}, {report.away} {report.guess_away}  "
            f"(total {report.guess_home + report.guess_away})",
            f"  {report.consistency_note}",
        ]
        if report.pick_push_probability is not None:
            lines.append(
                f"  P(push against the pick's own line) {report.pick_push_probability:.0%}"
            )
    else:
        lines.append(
            f"GUESS (closest-total metric): {report.home} {report.guess_home}, "
            f"{report.away} {report.guess_away}"
            f"  (total {report.guess_home + report.guess_away})"
        )
    lines.append("most common exact finals in the neighborhood (exact-score metric, weighted):")
    for home_score, away_score, count in report.common_scores:
        lines.append(f"  {report.home} {home_score} - {report.away} {away_score}  ({count:.1f}x)")
    lines += [
        "",
        "honest error bars (all 2009-2025 lined finals): the market total "
        f"misses by {report.total_mae:.1f} on average (median {report.total_median_ae:.1f}, "
        f"bias {report.total_bias:+.1f}); each implied team score misses by "
        f"{report.implied_score_mae:.1f}.",
    ]
    return "\n".join(lines)


_SNAPSHOT_ID_PATTERN = re.compile(r"\d{8}T\d{6}Z")


def _embedded_snapshot_id(source: str) -> str | None:

    match = _SNAPSHOT_ID_PATTERN.search(source)
    return match.group(0) if match else None


def tiebreaker_lineage_sources(
    report: TiebreakerReport, *, fallback_effective_timestamp: str
) -> tuple[TiebreakerSource, ...]:

    sources: list[TiebreakerSource] = []

    consensus_snapshot = _embedded_snapshot_id(report.consensus.source)
    consensus_captured = parse_snapshot_capture(consensus_snapshot)
    sources.append(
        TiebreakerSource(
            input_name="market_consensus",
            builder_module="nfl_ats.tiebreaker",
            builder_version=BUILDER_VERSION,
            effective_timestamp=consensus_captured or fallback_effective_timestamp,
            source_snapshot=consensus_snapshot,
            source_captured_at=consensus_captured,
            effective_timestamp_basis=(
                "source_capture" if consensus_captured else "feature_table_build"
            ),
            unknown_source_reason=(
                None
                if consensus_snapshot is not None
                else (
                    f"market consensus source {report.consensus.source!r} names no "
                    "recoverable snapshot id"
                )
            ),
        )
    )

    if report.model_view is not None:
        margin_snapshot = _embedded_snapshot_id(report.model_view.source)
        margin_captured = parse_snapshot_capture(margin_snapshot)
        sources.append(
            TiebreakerSource(
                input_name="model_margin_view",
                builder_module="nfl_ats.tiebreaker",
                builder_version=BUILDER_VERSION,
                effective_timestamp=margin_captured or fallback_effective_timestamp,
                source_snapshot=margin_snapshot,
                source_captured_at=margin_captured,
                effective_timestamp_basis=(
                    "source_capture" if margin_captured else "feature_table_build"
                ),
                unknown_source_reason=(
                    None
                    if margin_snapshot is not None
                    else (
                        f"model margin view source {report.model_view.source!r} names no "
                        "recoverable snapshot id"
                    )
                ),
            )
        )

    if report.totals_view is not None:
        sources.append(
            TiebreakerSource(
                input_name="model_total_view",
                builder_module="nfl_ats.totals",
                builder_version=BUILDER_VERSION,
                effective_timestamp=fallback_effective_timestamp,
                source_snapshot=None,
                source_captured_at=None,
                effective_timestamp_basis="feature_table_build",
                unknown_source_reason=(
                    f"totals model view source {report.totals_view.source!r} names a "
                    "walk-forward training window, not a capture instant; the feature "
                    "table it trained on is already covered by the model_input records"
                ),
            )
        )

    return tuple(sources)
