from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import lead66_unit1 as whole
import lead71_unit1 as census
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

REPORT = Path("docs/lead71_unit2.md")
LANE = Path("docs/lanes/lead71.md")
OUTPUT = Path("tests/scratch/codex/lead71_unit2")
CALIBRATION = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
GRID = np.arange(-100, 101, dtype=float)
SEED = 20260929
REPLICATES = 10_000
CONTRASTS = (
    (1, "served_lattice", "implied_lattice", "rps"),
    (2, "four_term", "four_plus_implied", "log_loss"),
)


def implied_price(price):
    return -price / (100.0 - price) if price < 0 else 100.0 / (100.0 + price)


def market_pair(market, home, away):
    outcomes = {row["name"]: row for row in market["outcomes"]}
    first, second = ("Over", "Under") if market["key"] == "totals" else (home, away)
    left = implied_price(float(outcomes[first]["price"]))
    right = implied_price(float(outcomes[second]["price"]))
    return outcomes[first], left / (left + right)


def load_quotes():
    schedule = census.load_schedule(Path("data/processed/game_features.parquet"))
    targets = {
        (target.season, target.week): pd.Timestamp(target.requested_at_utc)
        for target in census.plan_backfill(schedule, 2020, 2025, labels=["tue_open"])
    }
    pairs = defaultdict(list)
    for row in schedule.itertuples(index=False):
        pairs[(row.home_team, row.away_team)].append(row)
    candidates, hashes = [], {}
    for path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        if (
            manifest.get("provider") != census.ODDS_API_PROVIDER
            or request.get("sport") != census.ODDS_API_SPORT
            or request.get("decision_label") != "tue_open"
            or int(request["season"]) not in census.SEASONS
        ):
            continue
        key = (int(request["season"]), int(request["week"]))
        if key not in targets:
            continue
        cutoff = targets[key]
        snapshot = census.instant(manifest["snapshot_timestamp_utc"])
        if (
            census.instant(manifest["requested_at_utc"]) != cutoff
            or snapshot > cutoff
            or census.instant(manifest["observed_at_utc"]) != snapshot
            or manifest.get("capture_kind") != "historical_backfill"
        ):
            continue
        if request.get("odds_format") != "american":
            raise ValueError("Unsupported quote format")
        response_path = path.parent / "response.json"
        if whole.digest(response_path) != manifest["files"]["response.json"]["sha256"]:
            raise ValueError(f"Manifest integrity mismatch: {response_path}")
        decoded = json.loads(response_path.read_bytes())
        if census.instant(decoded["timestamp"]) != snapshot:
            raise ValueError("Response timestamp differs from manifest")
        hashes[path.as_posix()] = whole.digest(path)
        hashes[response_path.as_posix()] = whole.digest(response_path)
        for event in decoded["data"]:
            if event.get("sport_key") != census.ODDS_API_SPORT:
                continue
            home, away = str(event.get("home_team")), str(event.get("away_team"))
            event_time = census.instant(event["commence_time"])
            team_pair = census.NFL_TEAM_NAMES.get(home), census.NFL_TEAM_NAMES.get(away)
            matching = [
                game
                for game in pairs.get(team_pair, [])
                if abs(game.kickoff - event_time) <= pd.Timedelta(hours=12)
            ]
            if len(matching) != 1:
                continue
            game = matching[0]
            if (int(game.season), int(game.week)) != key:
                continue
            if cutoff >= min(game.kickoff, event_time):
                continue
            books = {}
            for book in event.get("bookmakers", []):
                if not book.get("key"):
                    continue
                book_time = census.instant(book["last_update"])
                if book_time > snapshot:
                    continue
                markets = {}
                for market in book.get("markets", []):
                    if market.get("key") not in census.MARKETS:
                        continue
                    if census.instant(market.get("last_update") or book_time) > snapshot:
                        continue
                    if census.paired_market(market, home, away):
                        markets[market["key"]] = market
                books[book["key"]] = markets
            complete = sorted(
                book for book, markets in books.items() if set(markets) >= census.MARKETS
            )
            selected = {}
            for market in sorted(census.MARKETS):
                eligible = complete or sorted(
                    book for book, markets in books.items() if market in markets
                )
                if eligible:
                    selected[market] = eligible[0], books[eligible[0]][market]
            if set(selected) != census.MARKETS:
                continue
            _, win_probability = market_pair(selected["h2h"][1], home, away)
            spread, spread_probability = market_pair(selected["spreads"][1], home, away)
            total, _ = market_pair(selected["totals"][1], home, away)
            candidates.append(
                {
                    "game_id": str(game.game_id),
                    "quote_season": int(game.season),
                    "cutoff": cutoff,
                    "snapshot": snapshot,
                    "kickoff": game.kickoff,
                    "quote_line": -float(spread["point"]),
                    "quote_total": float(total["point"]),
                    "quote_win": win_probability,
                    "quote_cover": spread_probability,
                    "same_book": bool(complete),
                    "books": "/".join(selected[market][0] for market in sorted(census.MARKETS)),
                    "manifest": path.as_posix(),
                }
            )
    quotes = (
        pd.DataFrame(candidates)
        .sort_values(["game_id", "snapshot", "manifest"], ascending=[True, False, True])
        .drop_duplicates("game_id", keep="first")
    )
    if len(quotes) != 1602:
        raise ValueError(f"Quote population differs from declared census: {len(quotes)}")
    if not (quotes.snapshot.le(quotes.cutoff) & quotes.cutoff.lt(quotes.kickoff)).all():
        raise ValueError("A quote fails the frozen prediction timestamp")
    return quotes, hashes


def dense(atoms, mass):
    if not np.allclose(atoms, np.rint(atoms)) or np.max(np.abs(atoms)) > 100:
        raise ValueError("Margin support exceeds the frozen integer grid")
    result = np.zeros(len(GRID))
    np.add.at(result, np.rint(atoms).astype(int) + 100, mass)
    if not np.isclose(result.sum(), 1.0) or np.any(result < 0):
        raise ValueError("Invalid PMF")
    return result


def lattice(history, game):
    selected, band = whole.select_band(history, float(game.quote_line))
    total = selected.total_line.to_numpy(float)
    if not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError("Historical pregame total is missing or invalid")
    residual = selected.result.to_numpy(float) - selected.line.to_numpy(float)
    variance_scale = float(np.square(residual).sum() / total.sum())
    variance_reference = variance_scale * float(total.mean())
    variance_game = variance_scale * float(game.quote_total)
    if min(variance_game, variance_reference) <= 0:
        raise ValueError("Nonpositive margin variance")
    squared_distance = np.square(GRID - float(game.quote_line))
    normal = np.exp(-squared_distance / (2 * variance_reference))
    normal /= normal.sum()
    atoms, counts = whole.atomic_counts(selected)
    base = dense(atoms, counts / counts.sum()) * counts.sum() + normal
    log_base = np.log(base) - 0.5 * squared_distance * (1 / variance_game - 1 / variance_reference)
    payoff = np.column_stack(
        [
            (GRID > 0).astype(float) - float(game.quote_win) * (GRID != 0),
            (game.quote_line < GRID).astype(float)
            - float(game.quote_cover) * (game.quote_line != GRID),
        ]
    )

    def objective(theta):
        logits = log_base + payoff @ theta
        normalization = logsumexp(logits)
        mass = np.exp(logits - normalization)
        return (
            normalization + 0.5e-6 * theta @ theta,
            payoff.T @ mass + 1e-6 * theta,
        )

    fit = minimize(
        objective,
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-40, 40)] * 2,
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 500},
    )
    _, gradient = objective(fit.x)
    stationarity = float(np.max(np.abs(fit.x - np.clip(fit.x - gradient, -40, 40))))
    fallback = stationarity > 1e-6
    if fallback:
        fit = minimize(
            objective,
            fit.x,
            jac=True,
            method="SLSQP",
            bounds=[(-40, 40)] * 2,
            options={"ftol": 1e-13, "maxiter": 500},
        )
        _, gradient = objective(fit.x)
        stationarity = float(np.max(np.abs(fit.x - np.clip(fit.x - gradient, -40, 40))))
    if not np.isfinite(stationarity) or stationarity > 1e-6:
        raise ValueError(
            f"Price projection failed for {game.game_id}: {fit.message}; KKT={stationarity}"
        )
    logits = log_base + payoff @ fit.x
    mass = np.exp(logits - logsumexp(logits))
    errors = [
        float(mass[line < GRID].sum() / mass[line != GRID].sum() - price)
        for line, price in ((0.0, game.quote_win), (game.quote_line, game.quote_cover))
    ]
    return mass, {
        "theta_moneyline": float(fit.x[0]),
        "theta_spread": float(fit.x[1]),
        "variance_scale": variance_scale,
        "band": band,
        "band_games": len(selected),
        "price_error": max(abs(value) for value in errors),
        "stationarity": stationarity,
        "solver_fallback": fallback,
    }


def pmf_score(mass, game, look, arm, scheme):
    actual, line = float(game.result), float(game.tue_open_home_spread)
    if actual != round(actual) or abs(actual) > 100:
        raise ValueError("Outcome is outside the declared integer support")
    home, away = mass[line < GRID].sum(), mass[line > GRID].sum()
    probability = float(home / (home + away))
    decisive, y = actual != line, float(actual > line)
    bounded = float(np.clip(probability, 1e-12, 1 - 1e-12))
    realized = mass[int(actual) + 100]
    return {
        "game_id": game.game_id,
        "season": int(game.season),
        "look": look,
        "arm": arm,
        "scheme": scheme,
        "actual_home_cover": y,
        "actual_push": not decisive,
        "probability": probability,
        "push_probability": float(mass[line == GRID].sum()),
        "rps": float(np.square(np.cumsum(mass) - (actual <= GRID)).sum()),
        "margin_log_score": float(-np.log(realized)) if realized > 0 else float("inf"),
        "log_loss": -(y * np.log(bounded) + (1 - y) * np.log1p(-bounded)) if decisive else np.nan,
        "brier": (probability - y) ** 2 if decisive else np.nan,
        "accuracy": float((probability >= 0.5) == bool(y)) if decisive else np.nan,
    }


def evaluate_lattices(games, pool):
    rows, diagnostics, folds, served = [], [], [], {}
    for season, group in games.groupby("season", sort=True):
        completed = (
            pd.to_datetime(pool.gameday) + pd.Timedelta(days=whole.COMPLETION_ALLOWANCE_DAYS)
        ).lt(group.gameday.min())
        training = pool.loc[
            pool.season.ge(season - whole.TRAILING_SEASONS) & pool.season.lt(season) & completed
        ]
        if training.empty or training.season.ge(season).any():
            raise ValueError("Invalid chronological LOSO pool")
        optimistic = pd.concat([training, pool.loc[pool.season.eq(season)]], ignore_index=True)
        folds.append(
            {
                "season": int(season),
                "games": len(group),
                "training_games": len(training),
                "training_last_day": str(training.gameday.max().date()),
                "in_sample_games": len(optimistic),
            }
        )
        for scheme, history in (("OOS", training), ("IS", optimistic)):
            for game in group.itertuples(index=False):
                selected, _ = whole.select_band(history, game.tue_open_home_spread)
                atoms, counts = whole.atomic_counts(selected)
                masses, _ = whole.tilted_atoms(
                    atoms, counts, game.tue_open_home_spread, game.model_center
                )
                served_mass = dense(atoms, masses)
                market_atoms, market_mass, _ = whole.residual_pmf(
                    selected, game.tue_open_home_spread
                )
                implied_mass, diagnostics_row = lattice(history, game)
                served[(scheme, game.game_id)] = served_mass
                for arm, mass in (
                    ("market_residual", dense(market_atoms, market_mass)),
                    ("served_lattice", served_mass),
                    ("implied_lattice", implied_mass),
                ):
                    rows.append(pmf_score(mass, game, 1, arm, scheme))
                diagnostics.append(
                    dict(
                        game_id=game.game_id,
                        season=int(season),
                        scheme=scheme,
                        **diagnostics_row,
                    )
                )
        print(f"lattice fold {season} complete: {len(group)} games", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics), pd.DataFrame(folds), served


def fit_probability(training, scored, terms):
    values = training[list(terms)].to_numpy(float)
    means, stds = values.mean(axis=0), values.std(axis=0)
    stds[stds == 0] = 1.0
    design = np.column_stack([np.ones(len(training)), (values - means) / stds])
    beta = _fit_logit(design, training.actual_home_cover.to_numpy(float), FIT_RIDGE)
    target = np.column_stack(
        [
            np.ones(len(scored)),
            (scored[list(terms)].to_numpy(float) - means) / stds,
        ]
    )
    coefficients = dict(zip(terms, beta[1:] / stds, strict=True))
    coefficients["intercept"] = float(beta[0] - np.sum(beta[1:] * means / stds))
    return expit(np.clip(target @ beta, -35, 35)), coefficients


def calibrated_mass(mass, line, probability):
    result = mass.copy()
    home, away = line < GRID, line > GRID
    nonpush = mass[home | away].sum()
    if min(mass[home].sum(), mass[away].sum()) <= 0:
        raise ValueError("Cannot calibrate an empty non-push tail")
    result[home] *= nonpush * probability / mass[home].sum()
    result[away] *= nonpush * (1 - probability) / mass[away].sum()
    if not np.isclose(result.sum(), 1.0):
        raise ValueError("Calibrated PMF lost probability mass")
    return result


def evaluate_combined(games, lattice_rows, served):
    columns = ["game_id", "model_probability", *FIT_FEATURES]
    cached = pd.read_parquet(CALIBRATION, columns=columns)
    if cached.game_id.duplicated().any():
        raise ValueError("Duplicate four-term input rows")
    inputs = games.merge(cached, on="game_id", how="inner", validate="one_to_one")
    decisive = games.result.ne(games.tue_open_home_spread)
    if set(inputs.game_id) != set(games.loc[decisive, "game_id"]):
        raise ValueError("Frozen four-term population fails decisive-game coverage")
    if not np.allclose(inputs.model_probability, inputs.home_cover_probability_at_open, atol=1e-12):
        raise ValueError("Four-term inputs differ from the pinned model probabilities")
    implied = lattice_rows.loc[
        lattice_rows.arm.eq("implied_lattice") & lattice_rows.scheme.eq("OOS"),
        ["game_id", "probability"],
    ]
    inputs = inputs.merge(implied, on="game_id", validate="one_to_one")
    p = inputs.probability.clip(1e-6, 1 - 1e-6)
    inputs["implied_logit"] = np.log(p / (1 - p))
    inputs["actual_home_cover"] = inputs.result.gt(inputs.tue_open_home_spread).astype(float)
    if not np.isfinite(inputs[[*FIT_FEATURES, "implied_logit"]].to_numpy(float)).all():
        raise ValueError("Nonfinite combination feature")
    rows, coefficients = [], []
    terms_by_arm = {
        "simple_logit": ("model_logit",),
        "four_term": FIT_FEATURES,
        "four_plus_implied": (*FIT_FEATURES, "implied_logit"),
    }
    for season in range(2021, 2026):
        group = inputs.loc[inputs.season.eq(season)]
        for scheme in ("OOS", "IS"):
            training = inputs.loc[
                inputs.season.lt(season) if scheme == "OOS" else inputs.season.le(season)
            ]
            if training.empty or (scheme == "OOS" and training.season.ge(season).any()):
                raise ValueError("Invalid chronological calibration fold")
            for arm, terms in terms_by_arm.items():
                probabilities, beta = fit_probability(training, group, terms)
                coefficients.append(
                    dict(
                        season=season,
                        scheme=scheme,
                        arm=arm,
                        training_games=len(training),
                        **beta,
                    )
                )
                for game, probability in zip(
                    group.itertuples(index=False), probabilities, strict=True
                ):
                    mass = calibrated_mass(
                        served[(scheme, game.game_id)],
                        game.tue_open_home_spread,
                        probability,
                    )
                    rows.append(pmf_score(mass, game, 2, arm, scheme))
            for arm in ("market_residual", "served_lattice"):
                copied = lattice_rows.loc[
                    lattice_rows.game_id.isin(group.game_id)
                    & lattice_rows.scheme.eq(scheme)
                    & lattice_rows.arm.eq(arm)
                ].copy()
                copied["look"] = 2
                rows.extend(copied.to_dict("records"))
        print(f"combination fold {season} complete: {len(group)} decisive games", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(coefficients)


def bootstrap(frame, metric):
    values = frame[["season", metric]].dropna()
    if not np.isfinite(values[metric]).all():
        raise ValueError("Nonfinite primary or conditional-cover metric")
    blocks = values.groupby("season")[metric].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, len(blocks), size=(REPLICATES, len(blocks)))
    replicates = blocks["sum"].to_numpy()[draw].sum(axis=1) / blocks["count"].to_numpy()[draw].sum(
        axis=1
    )
    low, high = np.quantile(replicates, [0.025, 0.975])
    return {
        "effect": float(values[metric].mean()),
        "low": float(low),
        "high": float(high),
        "standard_error": float(replicates.std(ddof=1)),
        "probability_positive": float((replicates > 0).mean() + 0.5 * (replicates == 0).mean()),
        "games": len(values),
        "blocks": len(blocks),
        "season_start": int(blocks.index.min()),
        "season_end": int(blocks.index.max()),
    }


def contrasts(scores):
    results = []
    for look, baseline, challenger, primary in CONTRASTS:
        selected = scores.loc[scores.look.eq(look)]
        for scheme in ("OOS", "IS"):
            left = selected.loc[selected.arm.eq(baseline) & selected.scheme.eq(scheme)]
            right = selected.loc[selected.arm.eq(challenger) & selected.scheme.eq(scheme)]
            pairs = left.merge(
                right,
                on=["game_id", "season"],
                validate="one_to_one",
                suffixes=("_base", "_candidate"),
            )
            for metric in ("rps", "log_loss", "brier", "accuracy"):
                direction = -1 if metric == "accuracy" else 1
                pairs[metric] = direction * (pairs[f"{metric}_base"] - pairs[f"{metric}_candidate"])
                estimate = bootstrap(pairs, metric)
                results.append(
                    dict(
                        look=look,
                        scheme=scheme,
                        metric=metric,
                        primary=metric == primary,
                        **estimate,
                    )
                )
    return results


def summary_row(group):
    decisive = group.loc[~group.actual_push]
    wins, count = int(decisive.accuracy.sum()), len(decisive)
    return [
        len(group),
        f"{wins}-{count - wins}",
        f"{100 * wins / count:.2f}% {whole.wilson(wins, count)}",
        *[
            whole.fmt(group[metric].mean())
            for metric in (
                "log_loss",
                "brier",
                "rps",
                "margin_log_score",
            )
        ],
        int(np.isinf(group.margin_log_score).sum()),
    ]


def record_command(result):
    look = result["look"]
    units = "rps_improvement" if look == 1 else "log_loss_improvement"
    resolved = look == 2 and result["high"] < 0
    classification = "refuted_mechanism" if resolved else "unresolved_below_power"
    closing = "--closing-ground wrong_sign_resolved " if resolved else ""
    evidence = (
        "Fixed fifth-term extension only: log-loss improvement interval entirely negative; "
        "the margin-lattice family remains open"
        if resolved
        else "The margin-lattice mechanism remains open; no admissible closure established"
    )
    plain = (
        "Moneyline, spread and total prices gave a slightly better picture of final margins. "
        "That result leaves this idea open; the pool picks are unchanged."
        if look == 1
        else "Adding the new market chance to the pick formula made held-out chances worse. "
        "This finding concerns that extra term; the separate margin table idea remains open."
    )
    return (
        f".tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit2-look{look} "
        f"--description 'Reserved market-implied lattice look {look}; two-look family' "
        "--source docs/lead71_unit2.md --league nfl --category market "
        f"--effect={result['effect']:.12g} --effect-units {units} "
        f"--standard-error={result['standard_error']:.12g} "
        f"--interval-low={result['low']:.12g} --interval-high={result['high']:.12g} "
        f"--probability-positive={result['probability_positive']:.12g} "
        f"--sample-games {result['games']} --sample-blocks {result['blocks']} "
        f"--season-start {result['season_start']} --season-end {result['season_end']} "
        f"--family LEAD-71-unit2 --classification {classification} {closing}"
        f"--classification-evidence '{evidence}' --plain-summary '{plain}' "
        "--notes 'Proposed serial record; 2 primary contrasts; diagnostic looks listed in source'"
    )


def write_report(scores, diagnostics, folds, coefficients, estimates, inventory, hashes, protocol):
    report = [
        "# LEAD-71 unit 2: moneyline, spread and total margin lattice",
        "",
        "**Measured:** two reserved contrasts executed; research only; served card unchanged.",
        "**Inferred:** records remain proposed until the orchestrator runs the lane commands.",
        "",
        "## Decisive opener records and scores",
        "",
        "**Measured:** cover scores exclude pushes; look-1 whole-margin scores include pushes.",
        "Accuracy intervals are Wilson 95%; improvement intervals use paired season blocks.",
        "",
    ]
    score_headers = [
        "Games",
        "W-L",
        "Accuracy [95% CI]",
        "Cover log loss",
        "Brier",
        "RPS",
        "Margin log score",
        "Zero-mass outcomes",
    ]
    report.append(
        whole.table(
            ["Look", "Scheme", "Arm", *score_headers],
            [
                [look, scheme, arm, *summary_row(group)]
                for (look, scheme, arm), group in scores.groupby(["look", "scheme", "arm"])
            ],
        )
    )
    report.extend(
        [
            "",
            "## Paired improvements",
            "",
            "**Measured:** positive means the challenger improves on its reserved comparator.",
            "",
        ]
    )
    report.append(
        whole.table(
            [
                "Look",
                "Scheme",
                "Metric",
                "Primary",
                "Improvement",
                "95% season-block CI",
                "probability_positive",
            ],
            [
                [
                    row["look"],
                    row["scheme"],
                    row["metric"],
                    row["primary"],
                    whole.fmt(row["effect"]),
                    f"[{whole.fmt(row['low'])}, {whole.fmt(row['high'])}]",
                    whole.fmt(row["probability_positive"]),
                ]
                for row in estimates
            ],
        )
    )
    report.extend(
        ["", "## OOS-minus-IS gaps", "", "**Measured:** same-game score differences.", ""]
    )
    gaps = []
    for (look, arm), group in scores.groupby(["look", "arm"]):
        means = group.groupby("scheme")[["log_loss", "brier", "rps", "accuracy"]].mean()
        gaps.append(
            [
                look,
                arm,
                *[
                    whole.fmt(means.loc["OOS", metric] - means.loc["IS", metric])
                    for metric in means.columns
                ],
            ]
        )
    report.append(
        whole.table(["Look", "Arm", "Log loss gap", "Brier gap", "RPS gap", "Accuracy gap"], gaps)
    )
    report.extend(
        [
            "",
            "## Season stability",
            "",
            "**Measured:** descriptive season summaries; no season selected for its result.",
            "",
        ]
    )
    report.append(
        whole.table(
            ["Look", "Season", "Arm", *score_headers],
            [
                [look, season, arm, *summary_row(group)]
                for (look, season, arm), group in scores.loc[scores.scheme.eq("OOS")].groupby(
                    ["look", "season", "arm"]
                )
            ],
        )
    )
    report.extend(
        [
            "",
            "## Cover reliability",
            "",
            "**Measured:** five fixed bands per arm, including empty bands; descriptive only.",
            "",
        ]
    )
    reliability = []
    for (look, arm), group in scores.loc[scores.scheme.eq("OOS") & ~scores.actual_push].groupby(
        ["look", "arm"]
    ):
        band = np.minimum((group.probability.to_numpy() * 5).astype(int), 4)
        for index in range(5):
            selected = group.loc[band == index]
            reliability.append(
                [
                    look,
                    arm,
                    f"{index / 5:.1f}-{(index + 1) / 5:.1f}",
                    len(selected),
                    whole.fmt(selected.probability.mean()),
                    whole.fmt(selected.actual_home_cover.mean()),
                ]
            )
    report.append(
        whole.table(["Look", "Arm", "Band", "Games", "Predicted", "Observed"], reliability)
    )
    report.extend(
        [
            "",
            "## Fitting and price diagnostics",
            "",
            "**Measured:** chronological training pools:",
            "",
            whole.table(list(folds.columns), folds.to_numpy().tolist()),
            "",
            "**Measured:** natural-scale coefficients (constant inputs have zero weight):",
            "",
        ]
    )
    coefficient_columns = [
        "season",
        "scheme",
        "arm",
        "training_games",
        "intercept",
        *FIT_FEATURES,
        "implied_logit",
    ]
    report.append(
        whole.table(
            coefficient_columns,
            [
                [
                    row.get(key, "")
                    if key in ("season", "scheme", "arm", "training_games")
                    else whole.fmt(float(row.get(key, np.nan)))
                    for key in coefficient_columns
                ]
                for row in coefficients.to_dict("records")
            ],
        )
    )
    report.extend(
        ["", "**Measured:** coefficient stability across the five OOS combination folds:", ""]
    )
    stability = []
    for arm, group in coefficients.loc[coefficients.scheme.eq("OOS")].groupby("arm"):
        for term in ("intercept", *FIT_FEATURES, "implied_logit"):
            values = group[term].dropna()
            if len(values):
                stability.append(
                    [
                        arm,
                        term,
                        whole.fmt(values.min()),
                        whole.fmt(values.max()),
                        int(values.gt(0).sum()),
                        int(values.lt(0).sum()),
                    ]
                )
    report.append(
        whole.table(
            ["Arm", "Term", "Minimum", "Maximum", "Positive folds", "Negative folds"], stability
        )
    )
    report.extend(["", "**Measured:** per-game variance scale and price-tilt ranges:", ""])
    report.append(
        whole.table(
            [
                "Season",
                "Scheme",
                "Variance scale min/max",
                "Moneyline tilt min/max",
                "Spread tilt min/max",
                "Max price error",
                "Price error > 0.01",
            ],
            [
                [
                    season,
                    scheme,
                    *[
                        f"{whole.fmt(group[column].min())}/{whole.fmt(group[column].max())}"
                        for column in ("variance_scale", "theta_moneyline", "theta_spread")
                    ],
                    whole.fmt(group.price_error.max()),
                    int(group.price_error.gt(0.01).sum()),
                ]
                for (season, scheme), group in diagnostics.groupby(["season", "scheme"])
            ],
        )
    )
    report.extend(
        [
            "",
            "## Interpretation and scope",
            "",
            "**Read:** AGENTS.md research rules prohibit closure from a zero-crossing interval "
            "and require one fitted probability to select a side.",
            "**Inferred:** these are two fixed-family measurements, not a serving decision. "
            "No positive control or split-half reliability adjudication was performed.",
            "**Inferred:** leave the margin-lattice mechanism open. The fixed extra-term "
            "extension has a proposed wrong_sign_resolved record when its whole improvement "
            "interval is adverse, limited to that extension. AGENTS.md and "
            "weak_signals.validate_closure impose that condition; the table above supplies "
            "the interval. Registry execution remains with the orchestrator.",
            "**Measured:** look 1 has primary RPS; look 2 has primary cover log loss. "
            "The other metrics are descriptive views of the same two contrasts.",
            "Look ledger for LEAD-71-unit2: 2 reserved primary contrasts; 16 arm/scheme views; "
            "40 fixed reliability-band cells; 43 OOS season/arm cells; 16 paired metric readouts "
            "(including IS); 30 logistic fits; and 2,960 per-game price projections. "
            "These overlap and are not independent votes. All were predetermined views; "
            "no best cell changed the family.",
            "IS is an explicitly optimistic same-game diagnostic: the held season enters the "
            "lattice pool or logistic fit. Logistic IS retains the frozen OOS implied feature.",
            "Look 2 uses archived prekick situational flags and movement. Only the new market "
            "feature freezes Tuesday 09:00 Eastern; the combined model is not Tuesday-only.",
            "Model centres remain frozen. Rows whose training day plus one day is not before "
            "the Tuesday cutoff are excluded by the source-timing contract before scoring. "
            "Their game-level training chronology and reconstructed "
            "probabilities are checked; archived base models are not refitted here.",
            "Historical totals and pre-2020 line proxies enter only completed prior-season OOS "
            "pools; held games are graded at the frozen pool opener.",
            "The total market supplies its quoted scoring line. Moneyline and spread prices "
            "are proportionally de-vigged; no external retrieval occurs.",
            "Exact margin log scores are infinite for empirical baselines assigning zero mass "
            "to an observed margin. Those baselines are not silently smoothed.",
            "Look 2 excludes pushes because its archived input rows do. Look 1 retains pushes "
            "for distribution scoring. 2020 initializes chronological calibration.",
            "",
            "## Source census and reproduction",
            "",
            "Command: .tools/uv.exe run --no-sync python scripts/lead71_unit2.py",
            "Thread limits: 2; one research process; no full-history rebuild, registry write, "
            "publication or Git mutation.",
            "",
        ]
    )
    report.extend(f"- **Measured:** {key}: {value}." for key, value in inventory.items())
    report.extend(
        [
            "",
            "**Measured:** source hashes, prediction-level scores, selected quotes, coefficients "
            "and price residuals are saved under tests/scratch/codex/lead71_unit2/; "
            "no prediction-row dump is stored in docs.",
            "",
            "## Predeclared protocol",
            "",
            protocol,
            "",
        ]
    )
    REPORT.write_text("\n".join(report), encoding="utf-8")
    primary = [row for row in estimates if row["primary"] and row["scheme"] == "OOS"]
    summary = {
        "inventory": inventory,
        "primary": primary,
        "estimates": estimates,
        "hashes": hashes,
        "record_commands": [record_command(row) for row in primary],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary": primary, "inventory": inventory}, indent=2), flush=True)


def main():
    lane = LANE.read_text(encoding="utf-8-sig")
    if "### Frozen protocol" in lane:
        protocol = lane.split("### Frozen protocol", 1)[1].split("## Tried", 1)[0].strip()
    else:
        protocol = (
            REPORT.read_text(encoding="utf-8").split("## Predeclared protocol\n\n", 1)[1].strip()
        )
    quotes, hashes = load_quotes()
    games, pool, input_hashes, inventory = whole.load_inputs()
    hashes.update(input_hashes)
    hashes[CALIBRATION.as_posix()] = whole.digest(CALIBRATION)
    original_games = len(games)
    games = games.merge(quotes, on="game_id", how="inner", validate="one_to_one")
    total = pd.read_parquet(whole.FEATURES, columns=["game_id", "total_line"])
    pool = pool.merge(total, on="game_id", validate="one_to_one", how="left")
    if len(games) == 0 or set(games.season) != set(census.SEASONS):
        raise ValueError("Missing evaluation seasons")
    model_training_end = pd.to_datetime(games.train_max_gameday, utc=True) + pd.Timedelta(days=1)
    timing_valid = model_training_end.lt(games.cutoff)
    rejected = games.loc[
        ~timing_valid, ["game_id", "season", "train_max_gameday", "cutoff", "kickoff"]
    ].copy()
    timing_eligible = len(games)
    games = games.loc[timing_valid].copy()
    if games.empty or not model_training_end.loc[timing_valid].lt(games.cutoff).all():
        raise ValueError("No timing-valid frozen-opener comparison")
    inventory["model_timestamp_rejections"] = len(rejected)
    inventory["opener_quote_matches_before_timing"] = timing_eligible
    inventory.update(
        quote_games=len(quotes),
        same_book_games=int(quotes.same_book.sum()),
        source_opener_games=original_games,
        matched_opener_games=len(games),
        unavailable_opener_games=original_games - timing_eligible,
        look1_pushes=int(games.result.eq(games.tue_open_home_spread).sum()),
        quote_pool_line_disagreements=int(games.quote_line.ne(games.tue_open_home_spread).sum()),
    )
    lattice_rows, diagnostics, folds, served = evaluate_lattices(games, pool)
    combined_rows, coefficients = evaluate_combined(games, lattice_rows, served)
    scores = pd.concat([lattice_rows, combined_rows], ignore_index=True)
    estimates = contrasts(scores)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("timestamp_rejections", rejected),
        ("scores", scores),
        ("quotes", quotes),
        ("price_diagnostics", diagnostics),
        ("folds", folds),
        ("coefficients", coefficients),
    ):
        frame.to_parquet(OUTPUT / f"{name}.parquet", index=False)
    write_report(scores, diagnostics, folds, coefficients, estimates, inventory, hashes, protocol)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
