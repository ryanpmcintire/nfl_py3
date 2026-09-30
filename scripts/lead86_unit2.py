from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import lead86_unit1 as previous
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

BASE = Path("artifacts/loso_base/20260929T232711656489Z")
OUTPUT = Path("tests/scratch/codex/lead86_unit2")
REPORT = Path("docs/lead86_unit2.md")
LANE = Path("docs/lanes/lead86.md")
SEASONS = tuple(range(2020, 2026))
ARMS = ("candidate", "served_loso", "matched_zero", "model_only", "market")
METRICS = previous.METRICS
TOLERANCE = 0.0051
REPLICATES = 10_000
SEED = 20260929
READ = previous.read


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def price_filter(frame, path, audits):
    rejected = set()
    for (game_id, book), group in frame.groupby(["nflverse_game_id", "bookmaker_key"]):
        products = {}
        for name in ("h2h", "spreads"):
            part = group.loc[group.market.eq(name)].drop_duplicates(
                ["outcome_name", "line", "price"]
            )
            if len(part) != 2 or set(part.outcome_side) != {"HOME", "AWAY"}:
                break
            part = part.set_index("outcome_side")
            prices = part.loc[["HOME", "AWAY"], "price"].to_numpy(float)
            if not np.isfinite(prices).all() or np.any(np.abs(prices) < 100):
                break
            implied = np.array([previous.market.implied_price(price) for price in prices])
            products[name] = (float(implied[0] / implied.sum()), part.loc["HOME", "line"])
        if len(products) != 2:
            continue
        win, cover = products["h2h"][0], products["spreads"][0]
        threshold = -float(products["spreads"][1])
        if not np.isfinite(threshold):
            continue
        violation = (
            abs(win - cover)
            if threshold == 0
            else max(0.0, cover - win)
            if threshold > 0
            else max(0.0, win - cover)
        )
        drop = violation > TOLERANCE
        audits.append(
            {
                "game_id": str(game_id),
                "book": str(book),
                "capture": str(path.parent),
                "home_margin_threshold": threshold,
                "moneyline_probability": win,
                "spread_probability": cover,
                "ordering_violation": violation,
                "drop": bool(drop),
            }
        )
        if drop:
            rejected.add((game_id, book))
    keys = pd.MultiIndex.from_frame(frame[["nflverse_game_id", "bookmaker_key"]])
    return frame.loc[~keys.isin(rejected)].copy()


def inventory():
    audits, raw_counts = [], {}

    def filtered_read(path, columns=None):
        frame = READ(path, columns)
        if Path(path).name == "quotes.parquet":
            raw_counts[str(path)] = len(frame)
            return price_filter(frame, Path(path), audits)
        return frame

    old_output, old_read = previous.OUTPUT, previous.read
    previous.OUTPUT, previous.read = OUTPUT, filtered_read
    try:
        panel, eligible, summary = previous.inventory()
    finally:
        previous.OUTPUT, previous.read = old_output, old_read
    audit = pd.DataFrame(audits)
    audit.to_parquet(OUTPUT / "price_filter.parquet", index=False)
    summary["raw_quote_rows"] = sum(raw_counts.values())
    summary["pricing_checked_panels"] = len(audit)
    summary["pricing_dropped_panels"] = int(audit["drop"].sum())
    summary["pricing_dropped_games"] = int(audit.loc[audit["drop"], "game_id"].nunique())
    summary["maximum_ordering_violation"] = float(audit.ordering_violation.max())
    summary["maximum_retained_violation"] = float(
        audit.loc[~audit["drop"], "ordering_violation"].max()
    )
    summary["tolerance"] = TOLERANCE
    summary["protocol_sha256"] = previous.digest(OUTPUT / "protocol.md")
    dump("inventory.json", summary)
    print(
        f"inventory: captures={summary['files']} raw_quotes={summary['raw_quote_rows']} "
        f"dropped_panels={summary['pricing_dropped_panels']} eligible={len(eligible)}",
        flush=True,
    )
    return panel, eligible, summary


def load_games(eligible):
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    require(
        previous.digest(BASE / "predictions.parquet") == metadata["predictions_sha256"],
        "Pinned base hash mismatch",
    )
    base = READ(BASE / "predictions.parquet")
    require(not base.duplicated(["game_id", "season"]).any(), "Duplicate base keys")
    require(set(base.season) == set(SEASONS), "Unexpected base seasons")
    require(base.same_season_training_rows.eq(0).all(), "Base fit overlaps its held season")
    require(base.held_out_season.eq(base.season).all(), "Base held-season labels differ")
    for season in SEASONS:
        fold = metadata["folds"][str(season)]
        expected = set(SEASONS) - {season}
        require(set(fold["training_seasons"]) == expected, "Wrong base training seasons")
        subset = base.loc[base.season.eq(season)]
        require(
            subset.base_training_seasons.map(
                lambda value, expected=expected: set(map(int, value.split(","))) == expected
            ).all(),
            "Base row provenance differs from metadata",
        )
        reproduced = previous.predict(subset, fold["coefficients"])
        require(
            np.allclose(reproduced, subset.base_home_probability, atol=1e-12, rtol=0),
            "Base coefficients fail reproduction",
        )
    panel = eligible.rename(
        columns={
            "week": "panel_week",
            "market_move_toward_home": "panel_market_move_toward_home",
            "market_move_available": "panel_market_move_available",
        }
    )
    games = panel.merge(base, on=["game_id", "season"], how="left", validate="one_to_one")
    require(games.base_home_probability.notna().all(), "Missing eligible base predictions")
    require(games.week.eq(games.panel_week).all(), "Panel/base week mismatch")
    require(games.game_type.eq("REG").all(), "Non-regular-season scored row")
    require(games.margin_vs_open.ne(0).all(), "Push in conditional scoring population")
    games["result"] = games.margin_vs_open + games.tue_open_home_spread
    require(
        games.home_covered.eq(games.result.gt(games.tue_open_home_spread)).all(),
        "Opener target mismatch",
    )
    require(
        np.isfinite(games[[*previous.TERMS, "result"]].to_numpy(float)).all(),
        "Nonfinite fitting input",
    )
    metadata["base_population"] = len(base)
    return games.sort_values(["season", "week", "game_id"]).reset_index(drop=True), metadata


def combine(fitted, calibration):
    slope = calibration["logit"]
    return {
        name: value * slope + (calibration["intercept"] if name == "intercept" else 0.0)
        for name, value in fitted.items()
    }


def replay(panel, eligible, inventory_summary):
    games, metadata = load_games(eligible)
    opener = READ(
        previous.OPENER / "per_game.parquet", ["game_id", "result", "tue_open_home_spread"]
    )
    history = panel.merge(opener, on="game_id", validate="one_to_one")
    history["line"], history["total_line"] = history.tuesday_line, history.tuesday_total
    target = games.home_covered.to_numpy(float)
    coefficients, predictions, diagnostics = [], [], []
    inventory_summary["base_metadata_sha256"] = previous.digest(BASE / "metadata.json")
    inventory_summary["base_predictions_sha256"] = metadata["predictions_sha256"]
    inventory_summary["eligible_base_coverage"] = len(games)
    inventory_summary["base_population"] = metadata["base_population"]
    dump("inventory.json", inventory_summary)
    for position, season in enumerate(SEASONS):
        calibration_season = SEASONS[(position - 1) % len(SEASONS)]
        selection_season = SEASONS[(position - 2) % len(SEASONS)]
        fit_seasons = sorted(set(SEASONS) - {season, calibration_season, selection_season})
        fit_rows = games.season.isin(fit_seasons).to_numpy()
        select_rows = games.season.eq(selection_season).to_numpy()
        calibrate_rows = games.season.eq(calibration_season).to_numpy()
        outer_rows = games.season.eq(season).to_numpy()
        require(
            np.all(np.sum([fit_rows, select_rows, calibrate_rows, outer_rows], axis=0) == 1),
            "Fold stages overlap",
        )
        require(
            all(mask.any() for mask in (fit_rows, select_rows, calibrate_rows, outer_rows)),
            "Empty fold stage",
        )
        prior = history.loc[history.season.isin(fit_seasons)].copy()
        require(set(prior.season).issubset(fit_seasons), "Prior leaks a held-out season")
        masses = np.empty((len(games), len(previous.market.GRID)))
        for index, game in enumerate(games.itertuples(index=False)):
            mass, diagnostic = previous.market.lattice(prior, SimpleNamespace(**game._asdict()))
            masses[index] = mass
            diagnostics.append({"fold": season, "game_id": game.game_id, **diagnostic})
        market_probability = previous.probability(
            masses, games.tue_open_home_spread.to_numpy(float)
        )
        fits, selection = {}, {}
        for weight in previous.WEIGHTS:
            fits[weight] = previous.fit_logistic(
                games.loc[fit_rows],
                previous.TERMS,
                target[fit_rows],
                market_probability[fit_rows],
                weight,
            )
            probability = previous.predict(games.loc[select_rows], fits[weight])
            selection[weight] = float(np.square(probability - target[select_rows]).mean())
        chosen = min(previous.WEIGHTS, key=lambda weight: (selection[weight], weight))
        raw = {
            "served_loso": previous.predict(games, metadata["folds"][str(season)]["coefficients"]),
            "model_only": games.model_probability.to_numpy(float),
            "market": market_probability,
        }
        raw["served_loso"][outer_rows] = games.loc[outer_rows, "base_home_probability"].to_numpy()
        calibrators, combined = {}, {}
        for arm, weight in (("candidate", chosen), ("matched_zero", 0.0)):
            uncalibrated = previous.predict(games, fits[weight])
            calibration_input = pd.DataFrame(
                {"logit": logit(np.clip(uncalibrated, 1e-9, 1 - 1e-9))}, index=games.index
            )
            calibrators[arm] = previous.fit_logistic(
                calibration_input.loc[calibrate_rows],
                ("logit",),
                target[calibrate_rows],
                positive_slope=True,
            )
            raw[arm] = previous.predict(calibration_input, calibrators[arm])
            combined[arm] = combine(fits[weight], calibrators[arm])
        coefficients.append(
            {
                "fold": season,
                "fit_seasons": fit_seasons,
                "selection_season": selection_season,
                "calibration_season": calibration_season,
                "chosen_weight": chosen,
                "selection_brier": selection,
                "fits": fits,
                "calibration": calibrators,
                "combined": combined,
                "base_coefficients": metadata["folds"][str(season)]["coefficients"],
                "base_training_seasons": metadata["folds"][str(season)]["training_seasons"],
                "fit_games": int(fit_rows.sum()),
                "outer_games": int(outer_rows.sum()),
                "prior_games": len(prior),
                "prior_pushes": int(prior.result.eq(prior.tue_open_home_spread).sum()),
            }
        )
        for scheme, mask in (("IS", fit_rows), ("OOS", outer_rows)):
            for arm in ARMS:
                predictions.append(
                    previous.score(
                        games.loc[mask], masses[mask], raw[arm][mask], season, scheme, arm
                    )
                )
        pd.DataFrame(
            {
                "game_id": games.game_id,
                "season": games.season,
                "market_probability": market_probability,
            }
        ).to_parquet(OUTPUT / f"market_targets_{season}.parquet", index=False)
        np.savez_compressed(OUTPUT / f"market_masses_{season}.npz", masses=masses)
        print(
            f"fold={season} fit={fit_seasons} select={selection_season} "
            f"calibrate={calibration_season} outer={int(outer_rows.sum())} penalty={chosen}",
            flush=True,
        )
    scores = pd.concat(predictions, ignore_index=True)
    scores.to_parquet(OUTPUT / "predictions.parquet", index=False)
    pd.DataFrame(diagnostics).to_parquet(OUTPUT / "lattice_diagnostics.parquet", index=False)
    dump("coefficients.json", coefficients)
    return scores, coefficients


def estimate(value, draws):
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return {
            "value": value,
            "interval": None,
            "probability_positive": 0.5,
            "standard_error": None,
        }
    return {
        "value": float(value),
        "interval": [float(number) for number in np.quantile(draws, [0.025, 0.975])],
        "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
        "standard_error": float(np.std(draws, ddof=1)),
    }


def bootstrap_weights(keys, rng):
    weights = np.zeros((REPLICATES, len(keys)), dtype=np.int32)
    seasons = keys.get_level_values("season").to_numpy()
    for season in np.unique(seasons):
        indices = np.flatnonzero(seasons == season)
        sampled = indices[rng.integers(0, len(indices), size=(REPLICATES, len(indices)))]
        np.add.at(weights, (np.arange(REPLICATES)[:, None], sampled), 1)
    return weights


def grouped_draws(frame, keys, weights):
    result, draws = {}, {}
    for arm in ARMS:
        rows = frame.loc[frame.arm.eq(arm)]
        grouped = rows.groupby(["season", "week"])
        sums = grouped[list(METRICS)].sum().reindex(keys, fill_value=0).to_numpy()
        counts = grouped.size().reindex(keys, fill_value=0).to_numpy()
        result[arm] = rows[list(METRICS)].mean().to_numpy()
        draws[arm] = (weights @ sums) / (weights @ counts)[:, None]
    return result, draws


def summarize(scores, coefficients, inventory_summary):
    rng = np.random.default_rng(SEED)
    metrics, contrasts, decisive, records = [], [], [], []
    for fold in (*SEASONS, "pooled"):
        scope = scores if fold == "pooled" else scores.loc[scores.fold.eq(fold)]
        keys = pd.MultiIndex.from_frame(
            scope[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        )
        weights = bootstrap_weights(keys, rng)
        means, draws = {}, {}
        for scheme in ("IS", "OOS"):
            means[scheme], draws[scheme] = grouped_draws(
                scope.loc[scope.scheme.eq(scheme)], keys, weights
            )
        means["gap"] = {arm: means["OOS"][arm] - means["IS"][arm] for arm in ARMS}
        draws["gap"] = {arm: draws["OOS"][arm] - draws["IS"][arm] for arm in ARMS}
        for scheme in ("IS", "OOS", "gap"):
            for arm in ARMS:
                for index, metric in enumerate(METRICS):
                    metrics.append(
                        {
                            "fold": fold,
                            "scheme": scheme,
                            "arm": arm,
                            "metric": metric,
                            **estimate(means[scheme][arm][index], draws[scheme][arm][:, index]),
                        }
                    )
            for arm in ARMS[1:]:
                for index, metric in enumerate(METRICS):
                    sign = 1.0 if metric == "accuracy" else -1.0
                    value = sign * (means[scheme]["candidate"][index] - means[scheme][arm][index])
                    distribution = sign * (
                        draws[scheme]["candidate"][:, index] - draws[scheme][arm][:, index]
                    )
                    contrasts.append(
                        {
                            "fold": fold,
                            "scheme": scheme,
                            "baseline": arm,
                            "metric": metric,
                            **estimate(value, distribution),
                        }
                    )
        outer = scope.loc[scope.scheme.eq("OOS")]
        for arm in ARMS:
            rows = outer.loc[outer.arm.eq(arm)]
            wins = int(rows.accuracy.sum())
            records.append({"fold": fold, "arm": arm, "wins": wins, "losses": len(rows) - wins})
        candidate = outer.loc[outer.arm.eq("candidate")].set_index(["fold", "game_id"])
        for arm in ARMS[1:]:
            baseline = (
                outer.loc[outer.arm.eq(arm)].set_index(["fold", "game_id"]).reindex(candidate.index)
            )
            mask = candidate.probability.ge(0.5).ne(baseline.probability.ge(0.5))
            rows = candidate.loc[mask]
            wins, n = int(rows.accuracy.sum()), len(rows)
            grouped = rows.groupby(["season", "week"])
            successes = grouped.accuracy.sum().reindex(keys, fill_value=0).to_numpy()
            counts = grouped.size().reindex(keys, fill_value=0).to_numpy()
            denominator = weights @ counts
            rates = np.divide(
                weights @ successes,
                denominator,
                out=np.full(REPLICATES, np.nan),
                where=denominator > 0,
            )
            decisive.append(
                {
                    "fold": fold,
                    "baseline": arm,
                    "wins": wins,
                    "losses": n - wins,
                    "games": n,
                    "win_rate": estimate(wins / n if n else None, rates),
                    "probability_positive": estimate(wins / n - 0.5 if n else None, rates - 0.5)[
                        "probability_positive"
                    ],
                    "exact_null_p": float(binomtest(wins, n).pvalue) if n else None,
                }
            )
    reliability = []
    outer = scores.loc[scores.scheme.eq("OOS")]
    for arm in ARMS:
        rows = outer.loc[outer.arm.eq(arm)]
        bins = pd.cut(rows.probability, np.linspace(0, 1, 6), include_lowest=True, labels=False)
        for number in range(5):
            sample = rows.loc[bins.eq(number)]
            reliability.append(
                {
                    "arm": arm,
                    "band": f"{number / 5:.1f}-{(number + 1) / 5:.1f}",
                    "games": len(sample),
                    "predicted": float(sample.probability.mean()) if len(sample) else None,
                    "observed": float(sample.target.mean()) if len(sample) else None,
                }
            )
    lattice = READ(OUTPUT / "lattice_diagnostics.parquet")
    result = {
        "metrics": metrics,
        "contrasts": contrasts,
        "decisive": decisive,
        "records": records,
        "reliability": reliability,
        "inventory": inventory_summary,
        "coefficients": coefficients,
        "decision_looks": 2,
        "diagnostic_looks": 851,
        "lattice_fits": len(lattice),
        "maximum_price_projection_error": float(lattice.price_error.max()),
        "projection_errors_over_tolerance": int(lattice.price_error.gt(TOLERANCE).sum()),
        "source_hashes": {
            str(path): previous.digest(path)
            for path in (Path(__file__), Path(previous.__file__), Path(previous.market.__file__))
        },
    }
    dump("summary.json", result)
    return result


def cell(row, metric=None, signed=False):
    if row["value"] is None:
        return "unavailable"
    scale = 100 if metric == "accuracy" else 1
    spec = "+.6f" if signed else ".6f"
    value = format(row["value"] * scale, spec)
    interval = row["interval"]
    if interval is None:
        return value
    return f"{value} [{format(interval[0] * scale, spec)}, {format(interval[1] * scale, spec)}]"


def write_report(summary):
    table = previous.table
    inventory_summary = summary["inventory"]
    protocol = (OUTPUT / "protocol.md").read_text(encoding="utf-8-sig")
    protocol = protocol.split("## Protocol (frozen before scoring)\n", 1)[1].split("\n## Tried", 1)[
        0
    ]
    lines = [
        "# LEAD-86 unit 2: price-consistent market regularization",
        "",
        "**Measured:** one fixed-protocol six-season replay. All intervals below are 95% paired,",
        "season-stratified week-block percentile intervals (10,000 draws). Positive improvement",
        "favors the candidate. Accuracy is percent; differences are percentage points.",
        "RPS is the unnormalized sum over integer margins -100 through 100.",
        "",
        "## Decisive games first",
        "",
        "**Measured:** decisive games are non-push opener games where candidate and comparator",
        "choose different sides. Their intervals resample whole weeks; exact-null p-values",
        "are two-sided binomial diagnostics.",
        "",
    ]
    lines += table(
        [
            "Fold",
            "Against",
            "Candidate W-L",
            "Win %, interval",
            "probability_positive (above half)",
            "Exact-null p",
        ],
        [
            [
                row["fold"],
                row["baseline"],
                f"{row['wins']}-{row['losses']}",
                cell(row["win_rate"], "accuracy"),
                f"{row['probability_positive']:.4f}",
                f"{row['exact_null_p']:.6f}" if row["exact_null_p"] is not None else "unavailable",
            ]
            for row in summary["decisive"]
        ],
    )
    lines += [
        "",
        "## Candidate versus served LOSO",
        "",
        "**Measured:** Brier and accuracy are the two declared decision comparisons.",
        "Log loss and RPS are descriptive; none supplies an additional selected variant.",
        "",
    ]
    primary = [
        row
        for row in summary["contrasts"]
        if row["fold"] == "pooled" and row["scheme"] == "OOS" and row["baseline"] == "served_loso"
    ]
    lines += table(
        ["Metric", "Improvement, interval", "probability_positive"],
        [
            [row["metric"], cell(row, row["metric"], True), f"{row['probability_positive']:.4f}"]
            for row in primary
        ],
    )
    lines += [
        "",
        "**Inferred:** proposed registry dispositions await the orchestrator: primary Brier",
        "remains unresolved_below_power. This specified training workflow has resolved",
        "negative accuracy versus served; its accuracy cell can use refuted_mechanism with",
        "closing ground wrong_sign_resolved under AGENTS.md:65-85. That endpoint does not",
        "close the broader market-regularization mechanism: its matched-zero comparison",
        "isolates the penalty, while the served comparison also changes training design.",
        "No serving change is authorized. One calibrated probability selects each candidate",
        "side (AGENTS.md:87-100).",
        "",
        "## Frozen protocol",
        "",
        protocol,
        "",
        "## Source and population checks",
        "",
        f"**Measured:** {inventory_summary['files']} capture files;",
        f"{inventory_summary['raw_quote_rows']:,} raw quotes;",
        f"{inventory_summary['pricing_checked_panels']:,} price pairs checked;",
        f"{inventory_summary['pricing_dropped_panels']} rejected book/capture panels",
        f"across {inventory_summary['pricing_dropped_games']} games.",
        "Exclusions precede book selection and use prices only.",
        f"{inventory_summary['complete_fixed_games']} of",
        f"{inventory_summary['base_population']} base games have eligible complete",
        "panels; every eligible game joined once to the pinned base.",
        "",
        "**Measured:** maximum pre-filter ordering violation",
        f"{inventory_summary['maximum_ordering_violation'] * 100:.6f} percentage points;",
        f"retained maximum {inventory_summary['maximum_retained_violation'] * 100:.6f}.",
        f"Across {summary['lattice_fits']:,} lattice fits, maximum quoted-price",
        f"projection error {summary['maximum_price_projection_error'] * 100:.6f} points;",
        f"{summary['projection_errors_over_tolerance']} exceed the fixed 0.51-point tolerance.",
        "Projection diagnostics do not change the population after outcomes.",
        "**Read:** docs/loso_base_artifact.md:26-41 states that the base holds out the",
        "four-term fitting season while upstream model inputs retain retrospective training",
        "provenance; Sunday features cannot represent Tuesday or Thursday decisions.",
        "**Inferred:** this is retrospective season-separated recalibration, not end-to-end",
        "season-held-out forecasting or an untouched chronological test. Historical opener",
        "is the grading line, with Sunday archived pre-kick information.",
        "",
        "**Inferred:** candidate and matched-zero fit three seasons, select on a fourth,",
        "calibrate on a fifth, and score the sixth. The served comparator fits all five",
        "other seasons on its larger eligible population and retains its existing fit",
        "convention. Candidate-versus-served therefore combines training-design differences",
        "with regularization; matched-zero isolates the penalty inside the candidate design.",
        "Old-versus-filtered differences cannot be attributed solely to the filter because",
        "the comparator and folds also changed.",
        "",
        "**Measured:** artifact hash, base coefficient reproduction, complete eligible coverage,",
        "one-to-one game/season joins, disjoint stages, NFL identity, capture/update/kickoff",
        "clocks and conditional non-push targets passed runtime checks. Exact cached served",
        "terms retain their missing-move policy; the reconstructed filtered move is not",
        "substituted into that design.",
        "",
        "## Fold choices and calibrated coefficients",
        "",
    ]
    choices = []
    for row in summary["coefficients"]:
        selection = row["selection_brier"]
        losses = " / ".join(
            f"{selection[str(weight) if str(weight) in selection else weight]:.6f}"
            for weight in previous.WEIGHTS
        )
        choices.append(
            [
                row["fold"],
                ", ".join(map(str, row["fit_seasons"])),
                row["selection_season"],
                row["calibration_season"],
                row["fit_games"],
                row["outer_games"],
                row["chosen_weight"],
                losses,
                f"{row['prior_games']} / {row['prior_pushes']}",
            ]
        )
    lines += table(
        [
            "Outer",
            "Fit seasons",
            "Select",
            "Calibrate",
            "Fit games",
            "OOS games",
            "Penalty",
            "Selection Brier 0 / 0.1 / 1",
            "Prior games / pushes",
        ],
        choices,
    )
    names = ("intercept", *previous.TERMS)
    coefficient_rows = []
    for fold in summary["coefficients"]:
        for arm in ("candidate", "matched_zero", "served_loso"):
            values = fold["base_coefficients"] if arm == "served_loso" else fold["combined"][arm]
            coefficient_rows.append(
                [fold["fold"], arm, *[f"{values[name]:+.6f}" for name in names]]
            )
    lines += ["", *table(["Fold", "Arm", *names], coefficient_rows)]
    lines += ["", "**Measured:** coefficient stability across the six candidate folds:", ""]
    stability = []
    for name in names:
        values = [fold["combined"]["candidate"][name] for fold in summary["coefficients"]]
        counts = " / ".join(
            str(sum(np.sign(value) == sign for value in values)) for sign in (1, 0, -1)
        )
        stability.append([name, f"{min(values):+.6f}", f"{max(values):+.6f}", counts])
    lines += table(["Term", "Minimum", "Maximum", "Positive / zero / negative folds"], stability)
    lines += [
        "",
        "## IS, OOS and gaps",
        "",
        "**Measured:** gaps are OOS minus IS, with the same sampled season/week weights applied",
        "to both. Pooled IS repeats each fit game across three outer folds; intervals preserve",
        "those shared weeks. IS is optimistic. Fixed-prediction intervals omit refit and",
        "feature-selection uncertainty. Served IS uses its corresponding outer-fold",
        "coefficients on candidate fit rows, never another row's LOSO prediction.",
        "",
    ]
    lines += table(
        ["Fold", "Scheme", "Arm", "Metric", "Estimate, interval"],
        [
            [
                row["fold"],
                row["scheme"],
                row["arm"],
                row["metric"],
                cell(row, row["metric"], row["scheme"] == "gap"),
            ]
            for row in summary["metrics"]
        ],
    )
    lines += [
        "",
        "## Paired candidate improvements",
        "",
        "**Measured:** losses use comparator minus candidate; accuracy uses candidate minus",
        "comparator. Improvement gaps are OOS improvement minus IS improvement. Baselines",
        "share each scored population and discrete lattice; conditional cover mass differs.",
        "",
    ]
    lines += table(
        ["Fold", "Scheme", "Against", "Metric", "Improvement, interval", "probability_positive"],
        [
            [
                row["fold"],
                row["scheme"],
                row["baseline"],
                row["metric"],
                cell(row, row["metric"], True),
                f"{row['probability_positive']:.4f}",
            ]
            for row in summary["contrasts"]
        ],
    )
    lines += [
        "",
        "## Opener records",
        "",
        "**Measured:** forced-pick records are diagnostics, not evidence of a profitable or",
        "stable edge, and not individual game probabilities.",
        "",
    ]
    lines += table(
        ["Fold", "Arm", "W-L"],
        [[row["fold"], row["arm"], f"{row['wins']}-{row['losses']}"] for row in summary["records"]],
    )
    lines += [
        "",
        "## Reliability",
        "",
        "**Measured:** five predeclared equal-width bands; empty cells remain visible and counted.",
        "",
    ]
    lines += table(
        ["Arm", "Probability band", "Games", "Mean probability", "Observed home-cover rate"],
        [
            [
                row["arm"],
                row["band"],
                row["games"],
                f"{row['predicted']:.6f}" if row["predicted"] is not None else "unavailable",
                f"{row['observed']:.6f}" if row["observed"] is not None else "unavailable",
            ]
            for row in summary["reliability"]
        ],
    )
    lines += [
        "",
        "## Look accounting and reproducibility",
        "",
        "**Measured:** one candidate specification and one replay; two decision comparisons.",
        "Mandatory diagnostics are not counted as only two looks: 420 absolute metric cells",
        "+ 336 contrast cells + 28 decisive cells + 25 reliability cells + 42 fit/coefficient",
        "summaries = 851 declared diagnostic looks. Actual per-game lattice optimizer calls",
        "are separately enumerated above. The 35 forced-pick records restate accuracy cells;",
        "coefficient ranges restate the six fixed fits. No best diagnostic is promoted.",
        "",
        "**Measured:** rows, input checksums, exclusions, market masses/targets, coefficients,",
        "fixed-prediction bootstrap summaries and the pre-run protocol snapshot are local:",
        "tests/scratch/codex/lead86_unit2/. No prediction rows are written under docs.",
        "Registry commands are prepared only in docs/lanes/lead86.md, candidate-versus-served",
        "only, for serial orchestrator execution.",
        "",
        chr(96) * 3 + "bash",
        "export UV_CACHE_DIR=tests/scratch/codex/lead86_unit2_uv_cache",
        "export RUFF_CACHE_DIR=tests/scratch/codex/lead86_unit2_ruff_cache",
        "export PYTHONDONTWRITEBYTECODE=1",
        ".tools/uv.exe run --no-sync python scripts/lead86_unit2.py",
        ".tools/uv.exe run --no-sync ruff check scripts/lead86_unit2.py",
        ".tools/uv.exe run --no-sync ruff format --check scripts/lead86_unit2.py",
        chr(96) * 3,
        "",
        "Replay refuses to overwrite existing predictions. --summarize-only regenerates",
        "this report from saved summaries without fitting or scoring.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the fixed LEAD-86 price-consistency replay.")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        write_report(summary)
        print(f"Report regenerated from cached summary: {REPORT}")
        return
    require(not (OUTPUT / "predictions.parquet").exists(), "Replay already exists; do not refit")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_text = LANE.read_text(encoding="utf-8-sig")
    if "## Protocol (frozen before scoring)" not in protocol_text:
        archived = REPORT.read_text(encoding="utf-8")
        protocol = (
            archived.split("## Frozen protocol\n", 1)[1]
            .split("\n## Source and population checks", 1)[0]
            .strip()
        )
        protocol_text = "## Protocol (frozen before scoring)\n" + protocol + "\n## Tried\n"
    (OUTPUT / "protocol.md").write_text(protocol_text, encoding="utf-8")
    with threadpool_limits(limits=2):
        panel, eligible, inventory_summary = inventory()
        scores, coefficients = replay(panel, eligible, inventory_summary)
        summary = summarize(scores, coefficients, inventory_summary)
        write_report(summary)
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "penalties": {row["fold"]: row["chosen_weight"] for row in coefficients},
                "candidate_vs_served": [
                    row
                    for row in summary["contrasts"]
                    if row["fold"] == "pooled"
                    and row["scheme"] == "OOS"
                    and row["baseline"] == "served_loso"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
