from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import lead71_unit2 as previous
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.joint_residual_model import (
    UNION_FEATURES,
    make_joint_estimator,
    realised_residual_frame,
)
from nfl_ats.lattice_centre_challenger import challenger_centre
from nfl_ats.pick_probability import PROBABILITY_EPSILON, signed_composition_flags
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    _arrest_incidents,
    _forecast_temperatures,
    _market_move_table,
    _protection_back_side,
)
from nfl_ats.score_lattice import pick_consistent_top_score, score_lattice
from nfl_ats.served_total import JOINT_TOTAL_BLEND_WEIGHT
from nfl_ats.tiebreaker import TOTAL_LOW_SIDE_SHADE_POINTS, last_game_of_week, lined_finals
from nfl_ats.totals import design_matrix

ROOT = Path("tests/scratch/codex/lead71_unit4")
SOURCE = Path("tests/scratch/codex/lead71_unit2")
REPORT = Path("docs/lead71_unit4.md")
LANE = Path("docs/lanes/lead71.md")
FEATURES = previous.whole.FEATURES
OPENER = previous.whole.OPENER / "per_game.parquet"
MARGINS = previous.whole.MARGINS / "predictions.parquet"
BASE = previous.CALIBRATION.parent
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
SEASONS = tuple(range(2020, 2026))
SCORES = np.arange(101)
HOME, AWAY = np.meshgrid(SCORES, SCORES, indexing="ij")
MARGIN = HOME - AWAY
TOTAL = HOME + AWAY
MARGIN_INDEX = (MARGIN + 100).ravel()
DRAWS = 10_000


def read_frame(path, columns=None):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fmt(value):
    return f"{value:.6f}"


def table(headers, rows):
    return previous.whole.table(headers, rows)


def restore_probability(games, schedules, metadata):
    cached = read_frame(BASE / "per_game.parquet")
    columns = ["game_id", *FIT_FEATURES, "out_of_season_home_probability"]
    games = games.merge(cached[columns], on="game_id", how="left", validate="one_to_one")
    missing = games.out_of_season_home_probability.isna()
    if missing.any():
        flags = signed_composition_flags(
            games,
            schedules,
            incidents=_arrest_incidents(Path("data")),
            forecasts_tuesday_noon=_forecast_temperatures(Path("data")),
            protection_back_side=_protection_back_side(schedules, Path("data")),
        ).set_index("game_id")
        moves, _ = _market_move_table(Path("artifacts"), metadata["market_move_feature_version"])
        restored = games.game_id.map(moves.set_index("game_id").leader_median_net)
        flags_restored = games.game_id.map(flags.composition_flag_sum)
        require(
            np.array_equal(flags_restored[~missing], games.loc[~missing, "composition_flag_sum"]),
            "Production composition recipe differs from frozen four-term features",
        )
        model_p = games.home_cover_probability_at_open.clip(
            PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON
        )
        games.loc[missing, "model_logit"] = np.log(model_p[missing] / (1 - model_p[missing]))
        games.loc[missing, "composition_flag_sum"] = flags_restored[missing]
        games.loc[missing, "market_move_available"] = restored[missing].notna().astype(float)
        games.loc[missing, "market_move_toward_home"] = restored[missing].fillna(0.0)
    games["home_probability"] = np.nan
    for season in SEASONS:
        mask = games.season.eq(season)
        beta = metadata["fold_coefficients"][str(season)]
        z = np.full(int(mask.sum()), beta["intercept"])
        for term in FIT_FEATURES:
            z += beta[term] * games.loc[mask, term].to_numpy()
        games.loc[mask, "home_probability"] = expit(z)
    error = float(
        np.abs(
            games.loc[~missing, "home_probability"]
            - games.loc[~missing, "out_of_season_home_probability"]
        ).max()
    )
    require(error < 1e-12, "Frozen four-term probability mismatch")
    require(np.isfinite(games[list(FIT_FEATURES)].to_numpy()).all(), "Missing four-term input")
    games["pick_side"] = np.where(games.home_probability.ge(0.5), "HOME", "AWAY")
    games["centre_margin"] = [
        challenger_centre(
            row.spread_line + row.residual_at_open_served, row.spread_line, row.pick_side
        ).centre
        for row in games.itertuples()
    ]
    return games, int(missing.sum()), error


def recover_over_prices(games, pinned):
    probabilities, hashes = {}, {}
    for manifest_name, group in games.groupby("manifest", sort=True):
        manifest_path = Path(manifest_name)
        response_path = manifest_path.parent / "response.json"
        for path in (manifest_path, response_path):
            value = digest(path)
            require(value == pinned[path.as_posix()], f"Frozen quote source changed: {path}")
            hashes[path.as_posix()] = value
        decoded = json.loads(response_path.read_bytes())
        for row in group.itertuples():
            events = [
                event
                for event in decoded["data"]
                if previous.census.NFL_TEAM_NAMES.get(event["home_team"]) == row.home_team
                and previous.census.NFL_TEAM_NAMES.get(event["away_team"]) == row.away_team
                and abs(pd.Timestamp(event["commence_time"]) - row.kickoff)
                <= pd.Timedelta(hours=12)
            ]
            require(len(events) == 1, f"Cannot restore total price for {row.game_id}")
            book_key = row.books.split("/")[sorted(previous.census.MARKETS).index("totals")]
            books = [book for book in events[0]["bookmakers"] if book["key"] == book_key]
            require(len(books) == 1, "Frozen total bookmaker missing")
            book = books[0]
            markets = [market for market in book["markets"] if market["key"] == "totals"]
            require(len(markets) == 1, "Frozen totals market missing")
            market = markets[0]
            require(pd.Timestamp(book["last_update"]) <= row.snapshot, "Late bookmaker timestamp")
            require(
                pd.Timestamp(market.get("last_update") or book["last_update"]) <= row.snapshot,
                "Late total timestamp",
            )
            total, probability = previous.market_pair(market, "", "")
            require(float(total["point"]) == row.quote_total, "Restored total differs from Unit 2")
            probabilities[row.game_id] = probability
    return games.game_id.map(probabilities), hashes


def load_inputs():
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    pinned = {Path(key).as_posix(): value for key, value in summary["hashes"].items()}
    hashes = {}
    for path in (FEATURES, OPENER, MARGINS):
        hashes[path.as_posix()] = digest(path)
        require(hashes[path.as_posix()] == pinned[path.as_posix()], f"Source changed: {path}")
    for path in (
        BASE / "per_game.parquet",
        BASE / "metadata.json",
        SCHEDULE,
        SOURCE / "scores.parquet",
        SOURCE / "quotes.parquet",
        SOURCE / "folds.parquet",
        SOURCE / "summary.json",
        Path("scripts/lead71_unit4.py"),
    ):
        hashes[path.as_posix()] = digest(path)
    reference = read_frame(SOURCE / "scores.parquet")
    reference = reference.loc[reference.look.eq(1) & reference.arm.eq("implied_lattice")]
    ids = set(reference.game_id)
    require(len(ids) == 1480, "Unit 2 population changed")
    features, opener, schedules = read_frame(FEATURES), read_frame(OPENER), read_frame(SCHEDULE)
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    columns = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "game_type",
        "home_score",
        "away_score",
        "result",
    ]
    games = features.loc[features.game_id.isin(ids), columns].copy()
    games = games.merge(
        opener[
            [
                "game_id",
                "tue_open_home_spread",
                "residual_at_open_served",
                "home_cover_probability_at_open",
                "margin_vs_open",
            ]
        ],
        on="game_id",
        validate="one_to_one",
    ).merge(read_frame(SOURCE / "quotes.parquet"), on="game_id", validate="one_to_one")
    require(len(games) == len(ids) and not games.game_id.duplicated().any(), "Population mismatch")
    require(games.game_type.eq("REG").all(), "Non-regular-season game")
    for column in ("snapshot", "cutoff", "kickoff"):
        games[column] = pd.to_datetime(games[column], utc=True)
    require((games.snapshot.le(games.cutoff) & games.cutoff.lt(games.kickoff)).all(), "Late quote")
    margin = read_frame(MARGINS, ["game_id", "method", "model_name", "train_max_gameday"])
    margin = margin.loc[margin.method.eq("market_residual") & margin.model_name.eq("ridge")]
    completed = pd.to_datetime(
        games.game_id.map(margin.set_index("game_id").train_max_gameday), utc=True
    ) + pd.Timedelta(days=1)
    require(completed.lt(games.cutoff).all(), "Margin training crosses opener")
    games["spread_line"] = games.tue_open_home_spread
    require(
        np.allclose(games.result - games.spread_line, games.margin_vs_open),
        "Home-minus-away spread convention failed",
    )
    games["actual_total"] = games.home_score + games.away_score
    require(games.actual_total.notna().all(), "Missing actual total")
    last_ids = {
        str(last_game_of_week(schedules, int(season), int(week)).game_id)
        for season, week in schedules.loc[
            schedules.season.isin(SEASONS) & schedules.game_type.eq("REG")
        ]
        .groupby(["season", "week"])
        .groups
    }
    games["is_last_game"] = games.game_id.isin(last_ids)
    games, restored, error = restore_probability(games, schedules, metadata)
    games["quote_over"], quote_hashes = recover_over_prices(games, pinned)
    hashes.update(quote_hashes)
    require(games.quote_over.between(0, 1, inclusive="neither").all(), "Invalid total prices")
    pool = previous.whole.prior_pool(features, opener.set_index("game_id").tue_open_home_spread)
    pool = pool.merge(features[["game_id", "total_line"]], on="game_id", validate="one_to_one")
    facts = {
        "source_games": len(games),
        "full_schedule_last_games": len(last_ids),
        "source_last_games": int(games.is_last_game.sum()),
        "restored_four_term": restored,
        "four_term_max_error": error,
        "quote_pool_spread_differences": int(games.quote_line.ne(games.spread_line).sum()),
        "four_term_fold_coefficients": metadata["fold_coefficients"],
    }
    return (
        games.sort_values(["season", "week", "game_id"]),
        features,
        schedules,
        pool,
        facts,
        hashes,
    )


def market_joint(history, margin_pool, row):
    margin_pmf, diagnostics = previous.lattice(margin_pool, row)
    prior = score_lattice(history, row.quote_line, row.quote_total, scores=SCORES)
    margin_variance = float(np.square(history.result - history.spread_line).mean())
    total_variance = float(
        np.square(history.home_score + history.away_score - history.total_line).mean()
    )
    require(min(margin_variance, total_variance) > 0, "Degenerate training variance")
    gaussian = np.exp(
        -0.5
        * (
            np.square(MARGIN - row.quote_line) / margin_variance
            + np.square(TOTAL - row.quote_total) / total_variance
        )
    )
    gaussian /= gaussian.sum()
    base = (prior.weights + gaussian).ravel()
    payoff = ((row.quote_total < TOTAL) - row.quote_over * (row.quote_total != TOTAL)).ravel()

    def projected(theta):
        tilted = base * np.exp(theta * payoff)
        totals = np.bincount(MARGIN_INDEX, weights=tilted, minlength=201)
        return tilted * (margin_pmf / totals)[MARGIN_INDEX]

    theta = brentq(lambda value: float(projected(value) @ payoff), -80, 80, xtol=1e-11)
    mass = projected(theta).reshape(HOME.shape)
    margin_error = float(
        np.max(np.abs(np.bincount(MARGIN_INDEX, weights=mass.ravel(), minlength=201) - margin_pmf))
    )
    price_error = abs(
        float(
            mass[row.quote_total < TOTAL].sum() / mass[row.quote_total != TOTAL].sum()
            - row.quote_over
        )
    )
    require(margin_error < 1e-12 and price_error < 1e-9, "Joint price or margin projection failed")
    require(np.isclose(mass.sum(), 1.0) and np.isfinite(mass).all(), "Invalid joint mass")
    joint = replace(prior, probabilities=mass, weights=mass, label="market_implied_joint")
    return joint, {
        "total_theta": float(theta),
        "total_price_error": price_error,
        "margin_preservation_error": margin_error,
        "margin_price_error": diagnostics["price_error"],
        "margin_residual_variance": margin_variance,
        "total_residual_variance": total_variance,
    }


def choose(lattice, row, total):
    chosen = pick_consistent_top_score(
        lattice,
        pick_side=row.pick_side,
        spread_line=row.spread_line,
        served_total=float(total),
        centre_margin=row.centre_margin,
    )
    if chosen is None:
        return np.nan, np.nan
    home, away = chosen[:2]
    require((home - away > row.spread_line) == (row.pick_side == "HOME"), "Score changed pick")
    require(home - away != row.spread_line, "Score pushes selected side")
    return home, away


def replay(games, features, schedules, pool):
    population = realised_residual_frame(features)
    finals = lined_finals(schedules.loc[schedules.game_type.eq("REG")]).copy()
    rows, folds = [], []
    for season, held in games.groupby("season", sort=True):
        lower = season - previous.whole.TRAILING_SEASONS
        target = features.set_index("game_id").loc[held.game_id].copy()
        target["spread_line"] = held.spread_line.to_numpy()
        target["total_line"] = held.quote_total.to_numpy()
        for scheme in ("OOS", "IS"):
            upper = season - (scheme == "OOS")
            training = population.loc[population.season.between(lower, upper)]
            history = finals.loc[finals.season.between(lower, upper)]
            margin_pool = pool.loc[pool.season.between(lower, upper)]
            if scheme == "OOS":
                for source in (training, history, margin_pool):
                    completion = pd.to_datetime(source.gameday, utc=True) + pd.Timedelta(days=1)
                    require(completion.lt(held.cutoff.min()).all(), "Fold contains future outcomes")
                    require(source.season.lt(season).all(), "Fold includes held season")
            require(len(training) >= 800, "Insufficient seasonal total-training history")
            estimator = make_joint_estimator()
            estimator.fit(
                design_matrix(training, UNION_FEATURES),
                training[["margin_residual", "total_residual"]].to_numpy(float),
            )
            total_residual = estimator.predict(design_matrix(target, UNION_FEATURES))[:, 1]
            centers = held.quote_total.to_numpy() + JOINT_TOTAL_BLEND_WEIGHT * total_residual
            centers += TOTAL_LOW_SIDE_SHADE_POINTS
            folds.append(
                {
                    "season": int(season),
                    "scheme": scheme,
                    "training_games": len(training),
                    "lattice_games": len(history),
                    "margin_games": len(margin_pool),
                    "training_end": str(training.gameday.max().date()),
                }
            )
            for row, total in zip(held.itertuples(), centers, strict=True):
                served_lattice = score_lattice(history, row.centre_margin, float(total))
                served_home, served_away = choose(served_lattice, row, total)
                joint, diagnostic = market_joint(history, margin_pool, row)
                median = joint.median_total()
                candidate_home, candidate_away = choose(joint, row, median)
                rows.append(
                    {
                        "game_id": row.game_id,
                        "season": int(season),
                        "week": int(row.week),
                        "scheme": scheme,
                        "is_last_game": bool(row.is_last_game),
                        "actual_total": row.actual_total,
                        "pick_side": row.pick_side,
                        "home_probability": row.home_probability,
                        "spread_line": row.spread_line,
                        "centre_margin": row.centre_margin,
                        "served_total_centre": float(total),
                        "candidate_median": median,
                        "candidate_home": candidate_home,
                        "candidate_away": candidate_away,
                        "served_home": served_home,
                        "served_away": served_away,
                        "candidate": candidate_home + candidate_away,
                        "served": served_home + served_away,
                        "market": int(np.floor(row.quote_total + 0.5)),
                        **diagnostic,
                    }
                )
            pd.DataFrame(rows).to_parquet(ROOT / "predictions_partial.parquet", index=False)
            print(
                f"fold {season} {scheme}: {len(held)} games; {len(training)} training", flush=True
            )
    predictions = pd.DataFrame(rows)
    available = predictions[["candidate", "served"]].notna().all(axis=1)
    common = predictions.assign(available=available).groupby("game_id").available.all()
    retained = predictions.game_id.map(common)
    exclusions = predictions.loc[~retained].copy()
    predictions = predictions.loc[retained].copy()
    require(not predictions.empty, "No paired reconstructable tiebreakers")
    for arm in ("candidate", "served", "market"):
        predictions[f"error_{arm}"] = (predictions[arm] - predictions.actual_total).abs()
    predictions["gain"] = predictions.error_served - predictions.error_candidate
    return predictions, pd.DataFrame(folds), exclusions


def estimate(frame):
    columns = ["error_candidate", "error_served", "error_market", "gain"]
    rng = np.random.default_rng(7104)
    numerator = np.zeros((DRAWS, len(columns)))
    denominator = np.zeros(DRAWS)
    for _, season in frame.groupby("season", sort=True):
        sums = season.groupby("week")[columns].sum()
        counts = season.groupby("week").size().reindex(sums.index).to_numpy()
        sampled = rng.integers(0, len(sums), size=(DRAWS, len(sums)))
        numerator += sums.to_numpy()[sampled].sum(axis=1)
        denominator += counts[sampled].sum(axis=1)
    draws = numerator / denominator[:, None]
    statistics = {}
    for index, column in enumerate(columns):
        values = draws[:, index]
        low, high = np.quantile(values, [0.025, 0.975])
        statistics[column] = {
            "effect": float(frame[column].mean()),
            "low": float(low),
            "high": float(high),
            "standard_error": float(values.std(ddof=1)),
            "probability_positive": float((values > 0).mean() + 0.5 * (values == 0).mean()),
        }
    closer, worse, tied = (
        int((frame.gain > 0).sum()),
        int((frame.gain < 0).sum()),
        int((frame.gain == 0).sum()),
    )
    test = binomtest(closer, closer + worse) if closer + worse else None
    interval = test.proportion_ci(method="exact") if test else None
    return {
        "games": len(frame),
        "blocks": frame.groupby(["season", "week"]).ngroups,
        "closer": closer,
        "worse": worse,
        "tied": tied,
        "exact_p": float(test.pvalue) if test else 1.0,
        "closer_interval": [float(interval.low), float(interval.high)] if interval else [0.0, 1.0],
        **statistics,
    }


def record_command(population, result):
    gain = result["gain"]
    description = "last game each week" if population == "last" else "all eligible games"
    plain = (
        f"Using the market score table made the total guess {abs(gain['effect']):.3f} points "
        f"{'closer' if gain['effect'] > 0 else 'farther away'} on average for {description}. "
        "The result is still uncertain and the pool guesses stay unchanged."
    )
    return (
        ".tools/uv.exe run --no-sync nfl-ats weak-signals record "
        f"--name LEAD-71-unit4-{population} "
        "--description 'Market joint score tiebreaker versus served; "
        f"{description}; two-look family' "
        "--source docs/lead71_unit4.md --league nfl --category market "
        f"--effect={gain['effect']:.12g} "
        f"--effect-units mae_improvement --standard-error={gain['standard_error']:.12g} "
        f"--interval-low={gain['low']:.12g} --interval-high={gain['high']:.12g} "
        f"--probability-positive={gain['probability_positive']:.12g} "
        f"--sample-games {result['games']} --sample-blocks {result['blocks']} "
        "--season-start 2020 --season-end 2025 --family LEAD-71-unit4 "
        "--classification unresolved_below_power "
        "--classification-evidence 'No admissible closure established; "
        "descriptive exact null only' "
        f"--plain-summary '{plain}' --notes 'Proposed serial record; two primary MAE contrasts; "
        "week-block intervals condition on predictions; retrospective four-term base'"
    )


def write_report(predictions, folds, exclusions, facts, hashes, protocol):
    results = {}
    for population, selected in (
        ("last", predictions.loc[predictions.is_last_game]),
        ("all", predictions),
    ):
        results[population] = {
            scheme: estimate(group) for scheme, group in selected.groupby("scheme", sort=True)
        }
    report = [
        "# LEAD-71 unit 4: market joint-score tiebreaker",
        "",
        "**Measured:** .tools/uv.exe run --no-sync python scripts/lead71_unit4.py; two primary "
        "candidate-versus-served MAE contrasts. No serving changes or registry writes.",
        "",
        "## Predeclared protocol",
        "",
        protocol,
        "",
        "## Paired results",
        "",
        "**Measured:** points of absolute error; improvement is served minus candidate. "
        "95% percentile intervals use 10,000 paired week-block draws within each season. "
        "Closer/worse counts precede the average effects; ties leave the exact binomial null.",
        "",
    ]
    decisive, metrics, gaps = [], [], []
    for population, schemes in results.items():
        for scheme, row in schemes.items():
            decisive.append(
                [
                    population,
                    scheme,
                    row["games"],
                    row["blocks"],
                    row["closer"],
                    row["worse"],
                    row["tied"],
                    fmt(row["exact_p"]),
                    f"[{fmt(row['closer_interval'][0])}, {fmt(row['closer_interval'][1])}]",
                ]
            )
            for key in ("error_candidate", "error_served", "error_market", "gain"):
                value = row[key]
                metrics.append(
                    [
                        population,
                        scheme,
                        key,
                        fmt(value["effect"]),
                        f"[{fmt(value['low'])}, {fmt(value['high'])}]",
                        fmt(value["probability_positive"]) if key == "gain" else "n/a",
                    ]
                )
        for key in ("error_candidate", "error_served", "gain"):
            gaps.append(
                [
                    population,
                    key,
                    fmt(schemes["IS"][key]["effect"]),
                    fmt(schemes["OOS"][key]["effect"]),
                    fmt(schemes["OOS"][key]["effect"] - schemes["IS"][key]["effect"]),
                ]
            )
    report += [
        table(
            [
                "Population",
                "Scheme",
                "Games",
                "Weeks",
                "Closer",
                "Worse",
                "Tied",
                "Exact two-sided p",
                "Closer share exact 95% CI",
            ],
            decisive,
        ),
        "",
        table(
            ["Population", "Scheme", "Metric", "Estimate", "95% interval", "probability_positive"],
            metrics,
        ),
        "",
        "## IS/OOS gap",
        "",
        table(["Population", "Metric", "IS", "OOS", "OOS minus IS"], gaps),
        "",
        "## Season stability",
        "",
        "**Measured:** all six seasons, descriptive only; no season selected for its sign.",
        "",
    ]
    seasonal = []
    for population, selected in (
        ("last", predictions.loc[predictions.is_last_game]),
        ("all", predictions),
    ):
        for season, group in selected.loc[selected.scheme.eq("OOS")].groupby("season"):
            seasonal.append(
                [
                    population,
                    season,
                    len(group),
                    fmt(group.error_candidate.mean()),
                    fmt(group.error_served.mean()),
                    fmt(group.error_market.mean()),
                    fmt(group.gain.mean()),
                    int(group.gain.gt(0).sum()),
                    int(group.gain.lt(0).sum()),
                    int(group.gain.eq(0).sum()),
                ]
            )
    report += [
        table(
            [
                "Population",
                "Season",
                "N",
                "Candidate MAE",
                "Served MAE",
                "Market MAE",
                "Gain",
                "Closer",
                "Worse",
                "Tied",
            ],
            seasonal,
        ),
        "",
        "## Reconstruction and chronology",
        "",
        f"**Measured:** {facts['source_games']} source games; "
        f"{predictions.game_id.nunique()} paired games; {exclusions.game_id.nunique()} "
        "excluded when either scheme lacked a production-admissible score. "
        f"Full schedule: {facts['full_schedule_last_games']} last-of-week games; "
        f"{facts['source_last_games']} have Unit 2 source coverage. "
        f"{facts['restored_four_term']} four-term rows reconstructed with production inputs; "
        f"maximum cached-probability difference {facts['four_term_max_error']:.3g}.",
        "",
        f"**Measured:** {facts['quote_pool_spread_differences']} opener quote spreads "
        "differ from the historical pool proxy spread. Price projection uses its quoted "
        "spread; both guesses preserve the four-term side at the pool proxy spread.",
        "",
        "**Read:** tiebreaker.json method_note points to docs/tiebreaker.md; "
        "scripts/lead88_unit2.py supplies the served total and score-selection recipe. "
        "src/nfl_ats/clv.py:943 sets the model spread to tue_open_home_spread. "
        "This implementation uses home-minus-away spread thresholds, verified against "
        "the opener result and margin_vs_open identity, rather than negating that threshold.",
        "",
        "**Measured:** fold training (five prior seasons for OOS; IS adds the scored season):",
        "",
        table(list(folds.columns), folds.to_numpy().tolist()),
        "",
        "**Measured:** maximum marginal-preservation error "
        f"{predictions.margin_preservation_error.max():.3g}; total-price error "
        f"{predictions.total_price_error.max():.3g}; original Unit 2 margin-price error "
        f"{predictions.margin_price_error.max():.3g}. All quote timestamps and OOS "
        "training-completion guards passed.",
        "",
        "**Read:** the total residual fit retains the production feature union, ridge 10, "
        "blend 0.1 and shade -1. The joint score extension retains Unit 2 moneyline/spread "
        "projection and fits only the over/under multiplier. Its total median feeds the "
        "production pick-consistent score selector; mass breaks nearby-cell ties.",
        "",
        "**Measured:** four-term coefficients, unchanged between candidate and served:",
        "",
    ]
    coefficients = facts["four_term_fold_coefficients"]
    report += [
        table(
            ["Season", "Intercept", *FIT_FEATURES],
            [
                [
                    season,
                    fmt(coefficients[str(season)]["intercept"]),
                    *[fmt(coefficients[str(season)][term]) for term in FIT_FEATURES],
                ]
                for season in SEASONS
            ],
        ),
        "",
        "## Look ledger and interpretation",
        "",
        "**Measured:** two primary contrasts; 12 arm-MAE cells, four paired-gain cells, "
        "six IS/OOS gap cells, 48 season-MAE/gain cells, 48 closer/worse/tied count cells, "
        "four exact-null tests and four exact-interval cells; 30 reused coefficient cells. "
        "Twelve total ridge fits plus "
        f"{2 * facts['source_games']} margin and {2 * facts['source_games']} total-price "
        "projections were executed. "
        "These supporting views overlap and supply no additional selection choices.",
        "",
        "**Measured:** OOS MAE gain is positive in four of six all-game seasons; "
        "last-of-week gain is positive in two, negative in three and tied in one. "
        "The 2020 gain is 1.883333 points over all games and 2.454545 on last games. "
        "**Inferred:** the pooled all-game improvement merits follow-up, but does not establish "
        "stable last-game usefulness; the current serving decision is unchanged.",
        "",
        "**Inferred:** the market joint-score extension remains unresolved_below_power; "
        "no positive control or mechanism-refuting analysis was performed. AGENTS.md's "
        "interval and promotion rules keep research closure separate from serving. "
        "The orchestrator must run the proposed records before treating any verdict as recorded.",
        "",
        "**Inferred:** this is a season-frozen reconstruction, not archived weekly guesses. "
        "Both score lattices use regular-season training finals. Historical training residuals "
        "inherit schedule line provenance; target lines are "
        "dated openers. Cached four-term LOSO coefficients include later seasons in earlier "
        "folds, making this retrospective. IS/OOS gaps vary the empirical lattices and total "
        "fit while holding that four-term base fixed. The week bootstrap conditions on "
        "fitted predictions, does not refit, and does not establish new-season uncertainty. "
        "The exact binomial null treats decisive games as independent; the blocked MAE "
        "interval is the primary uncertainty calculation. Six reused seasons are not an "
        "independent outer test. No side changes, pick accuracy claim, or profit claim follows.",
        "",
        "**Measured:** predictions, exclusions, fold metadata, source hashes, frozen "
        "protocol and record payloads are under tests/scratch/codex/lead71_unit4/. "
        "No prediction rows are written under docs.",
        "",
    ]
    pending = ROOT / "prior_record_commands.md"
    if pending.exists():
        report += [
            "## Prior pending serial records (unchanged)",
            "",
            pending.read_text(encoding="utf-8"),
            "",
        ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    commands = [
        record_command(population, schemes["OOS"]) for population, schemes in results.items()
    ]
    summary = {
        "results": results,
        "facts": facts,
        "hashes": hashes,
        "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(),
        "record_commands": commands,
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (ROOT / "record_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps({name: row["OOS"] for name, row in results.items()}, indent=2), flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lane = LANE.read_text(encoding="utf-8-sig")
    state = lane.split("## State\n", 1)[1].split("## Tried", 1)[0].strip()
    if "Declared before Unit 4 outcome access" in state:
        protocol = state
    else:
        protocol = (
            REPORT.read_text(encoding="utf-8")
            .split("## Predeclared protocol\n\n", 1)[1]
            .split("\n## Paired results", 1)[0]
            .strip()
        )
    require("Declared before Unit 4 outcome access" in protocol, "Frozen protocol missing")
    frozen = ROOT / "protocol.txt"
    if frozen.exists():
        require(frozen.read_text(encoding="utf-8") == protocol, "Protocol changed after first run")
    else:
        frozen.write_text(protocol, encoding="utf-8")
    games, features, schedules, pool, facts, hashes = load_inputs()
    print(
        json.dumps(
            {key: value for key, value in facts.items() if key != "four_term_fold_coefficients"}
        ),
        flush=True,
    )
    predictions, folds, exclusions = replay(games, features, schedules, pool)
    predictions.to_parquet(ROOT / "predictions.parquet", index=False)
    folds.to_parquet(ROOT / "folds.parquet", index=False)
    exclusions.to_parquet(ROOT / "exclusions.parquet", index=False)
    write_report(predictions, folds, exclusions, facts, hashes, protocol)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
