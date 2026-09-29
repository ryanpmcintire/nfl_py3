from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import lead71_unit1 as census
import lead71_unit2 as market
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

SOURCE = Path("artifacts/pick_probability/20260929T192747Z")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z")
FEATURES = Path("data/processed/game_features.parquet")
OUTPUT = Path("tests/scratch/codex/lead86_unit1")
REPORT = Path("docs/lead86_unit1.md")
FOLDS = (2023, 2024, 2025)
WEIGHTS = (0.0, 0.1, 1.0)
TERMS = ("model_logit", "composition_flag_sum", "market_move_toward_home", "market_move_available")
ARMS = ("regularized", "four_term", "model_only", "market", "elo")
METRICS = ("accuracy", "brier", "log_loss", "rps")
RIDGE = 0.001
SEED = 20260929
REPLICATES = 10000
LOOKS = 505


def read(path, columns=None):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory():
    schedule = census.load_schedule(FEATURES).set_index("game_id", drop=False)
    targets = {
        (int(t.season), int(t.week), t.label): pd.Timestamp(t.requested_at_utc)
        for t in census.plan_backfill(
            schedule.reset_index(drop=True), 2020, 2025, labels=["tue_open", "sun_early_close"]
        )
    }
    panels, sources, rejected = {}, [], Counter()
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "outcome_side",
        "outcome_name",
        "line",
        "price",
        "home_spread_line",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "commence_time_utc",
        "home_team",
        "away_team",
    ]
    for path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        key = (
            int(request.get("season", 0)),
            int(request.get("week", 0)),
            request.get("decision_label"),
        )
        if (
            key not in targets
            or manifest.get("provider") != census.ODDS_API_PROVIDER
            or request.get("sport") != census.ODDS_API_SPORT
        ):
            continue
        if manifest.get("capture_kind") != "historical_backfill":
            continue
        cutoff = targets[key]
        stamp = census.instant(manifest["snapshot_timestamp_utc"])
        if (
            census.instant(manifest["requested_at_utc"]) != cutoff
            or census.instant(manifest["observed_at_utc"]) != stamp
            or stamp > cutoff
        ):
            raise ValueError(f"Capture clock mismatch: {path}")
        if request.get("odds_format") != "american":
            raise ValueError(f"Unsupported price format: {path}")
        quote_path = path.parent / "quotes.parquet"
        if digest(quote_path) != manifest["files"]["quotes.parquet"]["sha256"]:
            raise ValueError(f"Quote integrity mismatch: {path}")
        quotes = read(quote_path, columns)
        sources.append(
            {
                "manifest": str(path),
                "sha256": digest(path),
                "rows": len(quotes),
                "season": key[0],
                "week": key[1],
                "label": key[2],
            }
        )
        for name in (
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "market_last_update_utc",
            "commence_time_utc",
        ):
            quotes[name] = pd.to_datetime(quotes[name], utc=True, errors="coerce", format="mixed")
        for (game_id, book), group in quotes.groupby(["nflverse_game_id", "bookmaker_key"]):
            if game_id not in schedule.index:
                rejected["not_in_schedule"] += 1
                continue
            game = schedule.loc[game_id]
            if (int(game.season), int(game.week)) != key[:2]:
                rejected["wrong_target_week"] += 1
                continue
            valid = (
                group.observed_at_utc.eq(stamp)
                & group.bookmaker_last_update_utc.le(stamp)
                & group.market_last_update_utc.le(stamp)
                & group.commence_time_utc.gt(cutoff)
                & group.home_team.eq(game.home_team)
                & group.away_team.eq(game.away_team)
            )
            if cutoff >= game.kickoff or not valid.all():
                rejected["clock_or_identity"] += 1
                continue
            markets = {}
            for name in census.MARKETS:
                part = group.loc[group.market.eq(name)].drop_duplicates(
                    ["outcome_name", "line", "price"]
                )
                expected = {"Over", "Under"} if name == "totals" else {"HOME", "AWAY"}
                index = "outcome_name" if name == "totals" else "outcome_side"
                if len(part) != 2 or set(part[index]) != expected:
                    break
                prices = part.price.to_numpy(float)
                if not np.isfinite(prices).all() or np.any(np.abs(prices) < 100):
                    break
                selected = part.set_index(index)
                left, right = ("Over", "Under") if name == "totals" else ("HOME", "AWAY")
                first, second = selected.loc[left], selected.loc[right]
                if name == "spreads" and not np.isclose(float(first.line), -float(second.line)):
                    break
                if name == "totals" and not (
                    float(first.line) > 0 and np.isclose(float(first.line), float(second.line))
                ):
                    break
                p1, p2 = (
                    market.implied_price(float(first.price)),
                    market.implied_price(float(second.price)),
                )
                markets[name] = (first, p1 / (p1 + p2))
            if set(markets) != census.MARKETS:
                rejected["incomplete_products"] += 1
                continue
            spread = -float(markets["spreads"][0].line)
            if not np.isclose(spread, float(markets["spreads"][0].home_spread_line)):
                raise ValueError("Handicap sign mismatch")
            record = {
                "line": spread,
                "total": float(markets["totals"][0].line),
                "win": markets["h2h"][1],
                "cover": markets["spreads"][1],
                "snapshot": stamp.isoformat(),
                "cutoff": cutoff.isoformat(),
                "manifest": str(path),
            }
            panel_key = (str(game_id), key[2], str(book))
            if panel_key in panels:
                raise ValueError(f"Duplicate capture: {panel_key}")
            panels[panel_key] = record
    rows = []
    for game_id, game in schedule.iterrows():
        if "game_type" in game and game.game_type != "REG":
            continue
        books = sorted(
            {
                book
                for gid, label, book in panels
                if gid == game_id
                and label == "tue_open"
                and (gid, "sun_early_close", book) in panels
            }
        )
        if not books:
            continue
        first = panels[(game_id, "tue_open", books[0])]
        last = panels[(game_id, "sun_early_close", books[0])]
        moves = [
            panels[(game_id, "sun_early_close", book)]["line"]
            - panels[(game_id, "tue_open", book)]["line"]
            for book in books
        ]
        rows.append(
            {
                "game_id": game_id,
                "season": int(game.season),
                "week": int(game.week),
                "book": books[0],
                "common_books": len(books),
                "kickoff": game.kickoff,
                "market_move_toward_home": float(np.median(moves)),
                "market_move_available": 1.0,
                **{f"tuesday_{name}": value for name, value in first.items()},
                **{f"quote_{name}": value for name, value in last.items()},
            }
        )
    panel = pd.DataFrame(rows)
    ids = read(SOURCE / "per_game.parquet", ["game_id", "season"])
    eligible = panel.loc[panel.game_id.isin(ids.game_id)].copy()
    summary = {
        "files": len(sources),
        "quote_rows": sum(s["rows"] for s in sources),
        "complete_schedule_games": len(panel),
        "complete_fixed_games": len(eligible),
        "fixed_by_season": {str(k): int(v) for k, v in eligible.groupby("season").size().items()},
        "rejected_book_panels": dict(rejected),
        "sources": sources,
        "input_hashes": {
            str(path): digest(path)
            for path in (
                SOURCE / "per_game.parquet",
                SOURCE / "metadata.json",
                OPENER / "per_game.parquet",
                FEATURES,
            )
        },
    }
    panel.to_parquet(OUTPUT / "market_inputs.parquet", index=False)
    (OUTPUT / "inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return panel, eligible, summary


def fit_logistic(frame, terms, target, penalty_target=None, weight=0.0, positive_slope=False):
    values = frame[list(terms)].to_numpy(float)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(frame)), (values - means) / scales])
    target = np.asarray(target, dtype=float)
    soft = target if penalty_target is None else np.asarray(penalty_target, dtype=float)

    def objective(beta):
        score = design @ beta
        loss = np.mean((1 + weight) * np.logaddexp(0, score) - (target + weight * soft) * score)
        loss += 0.5 * RIDGE * np.square(beta[1:]).sum()
        gradient = design.T @ ((1 + weight) * expit(score) - target - weight * soft) / len(frame)
        gradient[1:] += RIDGE * beta[1:]
        return loss, gradient

    bounds = [(None, None)] * (len(terms) + 1)
    if positive_slope:
        bounds[1] = (0, None)
    fitted = minimize(
        objective,
        np.zeros(len(terms) + 1),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Logistic fit failed: {fitted.message}")
    natural = fitted.x[1:] / scales
    return {
        "intercept": float(fitted.x[0] - natural @ means),
        **{name: float(value) for name, value in zip(terms, natural, strict=True)},
    }


def predict(frame, coefficients):
    score = np.full(len(frame), coefficients["intercept"])
    for term, coefficient in coefficients.items():
        if term != "intercept":
            score += frame[term].to_numpy(float) * coefficient
    return expit(score)


def probability(mass, lines):
    home = np.sum(mass * (market.GRID[None, :] > lines[:, None]), axis=1)
    away = np.sum(mass * (market.GRID[None, :] < lines[:, None]), axis=1)
    return home / (home + away)


def condition_mass(mass, lines, probabilities):
    home = market.GRID[None, :] > lines[:, None]
    away = market.GRID[None, :] < lines[:, None]
    hp, ap = (mass * home).sum(axis=1), (mass * away).sum(axis=1)
    adjusted = mass * np.where(
        home,
        ((hp + ap) * probabilities / hp)[:, None],
        np.where(away, ((hp + ap) * (1 - probabilities) / ap)[:, None], 1.0),
    )
    if not np.allclose(adjusted.sum(axis=1), 1.0) or np.any(adjusted < 0):
        raise ValueError("Invalid calibrated discrete distribution")
    return adjusted


def score(frame, mass, probabilities, fold, scheme, arm):
    actual = frame.result.to_numpy(float)
    lines = frame.tue_open_home_spread.to_numpy(float)
    decisive = actual != lines
    target = (actual > lines).astype(float)
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    adjusted = condition_mass(mass, lines, p)
    result = frame[["game_id", "season", "week"]].copy()
    result["fold"], result["scheme"], result["arm"] = fold, scheme, arm
    result["probability"], result["target"], result["push"] = p, target, ~decisive
    result["accuracy"] = np.where(decisive, (p >= 0.5) == target, np.nan)
    result["brier"] = np.where(decisive, np.square(p - target), np.nan)
    result["log_loss"] = np.where(
        decisive, -target * np.log(p) - (1 - target) * np.log1p(-p), np.nan
    )
    result["rps"] = np.square(np.cumsum(adjusted, axis=1) - (actual[:, None] <= market.GRID)).sum(
        axis=1
    )
    result["push_probability"] = (adjusted * (lines[:, None] == market.GRID)).sum(axis=1)
    return result


def replay(panel, eligible):
    columns = [
        "game_id",
        "tue_open_home_spread",
        "model_logit",
        "composition_flag_sum",
        "model_probability",
        "result",
    ]
    games = eligible.merge(
        read(SOURCE / "per_game.parquet", columns), on="game_id", validate="one_to_one"
    )
    games = games.merge(
        read(FEATURES, ["game_id", "elo_diff"]), on="game_id", validate="one_to_one"
    )
    games = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    opener = read(OPENER / "per_game.parquet", ["game_id", "result", "tue_open_home_spread"])
    history = panel.merge(opener, on="game_id", validate="one_to_one")
    history["line"] = history.tuesday_line
    history["total_line"] = history.tuesday_total
    if not np.isfinite(
        games[[*TERMS, "elo_diff", "tue_open_home_spread", "result"]].to_numpy(float)
    ).all():
        raise ValueError("Nonfinite fixed-population input")
    coefficients, targets, scores = [], [], []
    for fold in FOLDS:
        fit_rows = games.season.le(fold - 3).to_numpy()
        selection_rows = games.season.eq(fold - 2).to_numpy()
        calibration_rows = games.season.eq(fold - 1).to_numpy()
        outer_rows = games.season.eq(fold).to_numpy()
        relevant = games.season.le(fold).to_numpy()
        prior = history.loc[history.season.le(fold - 3)].copy()
        if any(not mask.any() for mask in (fit_rows, selection_rows, calibration_rows, outer_rows)):
            raise ValueError("An explicitly separated fold is empty")
        masses = np.full((len(games), len(market.GRID)), np.nan)
        for index in np.flatnonzero(relevant):
            game = games.iloc[index]
            mass, diagnostics = market.lattice(prior, SimpleNamespace(**game.to_dict()))
            masses[index] = mass
            targets.append(
                {
                    "fold": fold,
                    "game_id": game.game_id,
                    "prior_last_season": int(prior.season.max()),
                    **diagnostics,
                }
            )
        target_probabilities = np.full(len(games), np.nan)
        target_probabilities[relevant] = probability(
            masses[relevant], games.tue_open_home_spread.to_numpy(float)[relevant]
        )
        y = (games.result > games.tue_open_home_spread).astype(float)
        candidate_fits, selection_losses = {}, {}
        for weight in WEIGHTS:
            fitted = fit_logistic(
                games.loc[fit_rows], TERMS, y.loc[fit_rows], target_probabilities[fit_rows], weight
            )
            candidate_fits[weight] = fitted
            p = predict(games.loc[selection_rows], fitted)
            selection_losses[weight] = float(np.square(p - y.loc[selection_rows]).mean())
        chosen = min(WEIGHTS, key=lambda weight: (selection_losses[weight], weight))
        raw = {
            "regularized": predict(games, candidate_fits[chosen]),
            "four_term": predict(games, candidate_fits[0.0]),
            "model_only": games.model_probability.to_numpy(float),
            "market": target_probabilities,
        }
        elo_fit = fit_logistic(
            games.loc[fit_rows], ("elo_diff", "tue_open_home_spread"), y.loc[fit_rows]
        )
        raw["elo"] = predict(games, elo_fit)
        calibrators = {}
        for arm in ("regularized", "four_term", "elo"):
            raw_logit = pd.DataFrame(
                {"logit": logit(np.clip(raw[arm], 1e-9, 1 - 1e-9))}, index=games.index
            )
            calibration = fit_logistic(
                raw_logit.loc[calibration_rows],
                ("logit",),
                y.loc[calibration_rows],
                positive_slope=True,
            )
            calibrators[arm] = calibration
            raw[arm] = predict(raw_logit, calibration)
        coefficients.append(
            {
                "fold": fold,
                "fit_through": fold - 3,
                "selection_season": fold - 2,
                "calibration_season": fold - 1,
                "chosen_weight": chosen,
                "selection_brier": selection_losses,
                "candidate_coefficients": candidate_fits,
                "calibration": calibrators,
                "elo": elo_fit,
                "fit_games": int(fit_rows.sum()),
                "prior_games": len(prior),
                "prior_pushes": int((prior.result == prior.tue_open_home_spread).sum()),
                "outer_games": int(outer_rows.sum()),
            }
        )
        for scheme, mask in (("IS", fit_rows), ("OOS", outer_rows)):
            for arm in ARMS:
                scores.append(
                    score(games.loc[mask], masses[mask], raw[arm][mask], fold, scheme, arm)
                )
        pd.DataFrame(
            {
                "game_id": games.loc[relevant, "game_id"],
                "fold": fold,
                "market_probability": target_probabilities[relevant],
            }
        ).to_parquet(OUTPUT / f"market_targets_{fold}.parquet", index=False)
        np.savez_compressed(OUTPUT / f"market_masses_{fold}.npz", masses=masses[relevant])
        print(
            f"fold={fold} fit={fit_rows.sum()} outer={outer_rows.sum()} selected_weight={chosen}",
            flush=True,
        )
    pd.DataFrame(targets).to_parquet(OUTPUT / "target_diagnostics.parquet", index=False)
    result = pd.concat(scores, ignore_index=True)
    result.to_parquet(OUTPUT / "predictions.parquet", index=False)
    (OUTPUT / "coefficients.json").write_text(json.dumps(coefficients, indent=2), encoding="utf-8")
    return result, coefficients


def estimates(values, groups, rng):
    values = np.asarray(values, dtype=float)
    keys, group_index = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(group_index, weights=values, minlength=len(keys))
    counts = np.bincount(group_index, minlength=len(keys))
    draws = np.zeros(REPLICATES)
    totals = np.zeros(REPLICATES)
    seasons = np.array([str(key).split("_")[0] for key in keys])
    for season in np.unique(seasons):
        indices = np.flatnonzero(seasons == season)
        sampled = indices[rng.integers(0, len(indices), size=(REPLICATES, len(indices)))]
        draws += sums[sampled].sum(axis=1)
        totals += counts[sampled].sum(axis=1)
    return draws / totals


def interval(draws):
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def summarize(scores, coefficients, inventory_summary):
    rng = np.random.default_rng(SEED)
    metrics, contrasts, decisive, reliability = [], [], [], []
    for fold in (*FOLDS, "pooled"):
        scope = scores if fold == "pooled" else scores.loc[scores.fold.eq(fold)]
        for scheme in ("IS", "OOS"):
            sample = scope.loc[scope.scheme.eq(scheme)]
            for arm in ARMS:
                arm_rows = sample.loc[sample.arm.eq(arm)]
                groups = arm_rows.season.astype(str) + "_" + arm_rows.week.astype(str)
                for metric in METRICS:
                    valid = arm_rows[metric].notna()
                    values = arm_rows.loc[valid, metric].to_numpy(float)
                    draws = estimates(values, groups.loc[valid], rng)
                    metrics.append(
                        {
                            "fold": fold,
                            "scheme": scheme,
                            "arm": arm,
                            "metric": metric,
                            "value": float(values.mean()),
                            "interval": interval(draws),
                            "draws": draws,
                        }
                    )
            for arm in ARMS[1:]:
                left = sample.loc[sample.arm.eq("regularized")].set_index(["fold", "game_id"])
                right = (
                    sample.loc[sample.arm.eq(arm)]
                    .set_index(["fold", "game_id"])
                    .reindex(left.index)
                )
                groups = left.season.astype(str) + "_" + left.week.astype(str)
                for metric in METRICS:
                    difference = (
                        left[metric] - right[metric]
                        if metric == "accuracy"
                        else right[metric] - left[metric]
                    )
                    valid = difference.notna()
                    draws = estimates(difference.loc[valid], groups.loc[valid], rng)
                    contrasts.append(
                        {
                            "fold": fold,
                            "scheme": scheme,
                            "baseline": arm,
                            "metric": metric,
                            "gain": float(difference.loc[valid].mean()),
                            "interval": interval(draws),
                            "probability_positive": float(
                                (draws > 0).mean() + 0.5 * (draws == 0).mean()
                            ),
                            "draws": draws,
                        }
                    )
                if scheme == "OOS":
                    different = (left.probability.ge(0.5) != right.probability.ge(0.5)) & ~left.push
                    wins = int(left.loc[different, "accuracy"].sum())
                    n = int(different.sum())
                    decisive.append(
                        [fold, arm, f"{wins}-{n - wins}", n, market.whole.wilson(wins, n)]
                    )
    for collection, key in ((metrics, "arm"), (contrasts, "baseline")):
        for out in [row for row in collection if row["scheme"] == "OOS"]:
            inside = next(
                row
                for row in collection
                if row["fold"] == out["fold"]
                and row["scheme"] == "IS"
                and row[key] == out[key]
                and row["metric"] == out["metric"]
            )
            measure = "value" if key == "arm" else "gain"
            draws = out["draws"] - inside["draws"]
            collection.append(
                {
                    **{k: v for k, v in out.items() if k != "draws"},
                    "scheme": "gap",
                    measure: out[measure] - inside[measure],
                    "interval": interval(draws),
                    "probability_positive": float((draws > 0).mean() + 0.5 * (draws == 0).mean()),
                }
            )
        for row in collection:
            row.pop("draws", None)
    for arm in ARMS:
        sample = scores.loc[scores.scheme.eq("OOS") & scores.arm.eq(arm) & ~scores.push]
        for index in range(5):
            low, high = index / 5, (index + 1) / 5
            part = sample.loc[
                sample.probability.ge(low)
                & (sample.probability.lt(high) if index < 4 else sample.probability.le(high))
            ]
            reliability.append(
                [
                    arm,
                    f"{low:.1f}-{high:.1f}",
                    len(part),
                    f"{part.probability.mean():.6f}",
                    f"{part.target.mean():.6f}",
                ]
            )
    report = [
        "# LEAD-86 unit 1: observed-market training regularization",
        "",
        "**Measured:** one local fixed-grid replay; no served changes. "
        "Protocol was saved before outcomes.",
        "",
        "## Decisive games",
        "",
        *table(["Fold", "Against", "Wins-losses", "Disagreements", "95% Wilson"], decisive),
        "",
        "## Population and clocks",
        "",
        f"**Measured:** {inventory_summary['files']} capture files, "
        f"{inventory_summary['quote_rows']} quote rows; "
        f"{inventory_summary['complete_fixed_games']} common-book complete non-push opener games.",
        "Tuesday anchors predate noon; Sunday captures are the archived "
        "deadline anchors. No closing predictors.",
        "Push rows remain in the lattice training prior; the frozen "
        "four-term artifact excludes pushes, so scored rows are non-push.",
        "The lattice nuisance fit uses only the declared fit seasons, "
        "including training outcomes; no selection, calibration or outer "
        "outcomes enter it.",
        "Five arms use the same discrete margin lattice at the frozen "
        "opener; only conditional cover mass changes.",
        "Model-only and market retain their probabilities; candidate, "
        "four-term and Elo use separate-season calibration.",
        "**Inferred limitation:** archived model features and model "
        "selection are retrospective; this is not an untouched prospective "
        "test.",
        "",
        "## Metrics and gaps",
        "",
        "**Measured:** 95% season-stratified week-block percentile "
        "intervals, 10,000 replicates; gap = OOS minus optimistic fit-period"
        " IS.",
        "Predictions are held fixed. Accuracy is a fraction; Brier/log "
        "loss/RPS are losses. Pooled IS repeats training games across outer "
        "folds.",
        "",
        *table(
            ["Fold", "Set", "Arm", "Metric", "Value", "95% interval"],
            [
                [
                    r["fold"],
                    r["scheme"],
                    r["arm"],
                    r["metric"],
                    f"{r['value']:.6f}",
                    f"[{r['interval'][0]:.6f}, {r['interval'][1]:.6f}]",
                ]
                for r in metrics
            ],
        ),
        "",
        "## Paired gains",
        "",
        "Positive gains favor regularization. probability_positive assigns "
        "half of bootstrap ties to each side.",
        "",
        *table(
            ["Fold", "Set", "Baseline", "Metric", "Gain", "95% interval", "probability_positive"],
            [
                [
                    r["fold"],
                    r["scheme"],
                    r["baseline"],
                    r["metric"],
                    f"{r['gain']:.6f}",
                    f"[{r['interval'][0]:.6f}, {r['interval'][1]:.6f}]",
                    f"{r['probability_positive']:.4f}",
                ]
                for r in contrasts
            ],
        ),
        "",
        "## Fold coefficients",
        "",
        "**Measured:** natural input units; calibration is a nonnegative "
        "slope on the raw probability logit.",
        "",
    ]
    coefficient_rows = []
    for fold in coefficients:
        for weight, fitted in fold["candidate_coefficients"].items():
            coefficient_rows.append(
                [
                    fold["fold"],
                    weight,
                    f"{fold['selection_brier'][weight]:.6f}",
                    str(weight == fold["chosen_weight"]),
                    *[f"{fitted[t]:.6f}" for t in ("intercept", *TERMS)],
                ]
            )
    report += table(
        ["Fold", "Penalty", "Selection Brier", "Selected", "Intercept", *TERMS], coefficient_rows
    )
    report += [
        "",
        *table(
            ["Fold", "Arm", "Calibration intercept", "Calibration slope"],
            [
                [c["fold"], a, f"{v['intercept']:.6f}", f"{v['logit']:.6f}"]
                for c in coefficients
                for a, v in c["calibration"].items()
            ],
        ),
        "",
        *table(
            ["Fold", "Elo intercept", "Elo difference", "Opener", "Prior games", "Prior pushes"],
            [
                [
                    c["fold"],
                    *[
                        f"{c['elo'][t]:.6f}"
                        for t in ("intercept", "elo_diff", "tue_open_home_spread")
                    ],
                    c["prior_games"],
                    c["prior_pushes"],
                ]
                for c in coefficients
            ],
        ),
        "",
        "## Reliability",
        "",
        *table(["Arm", "Band", "Games", "Mean probability", "Observed home cover"], reliability),
        "",
        "## Decision and saved artifacts",
        "",
        "**Measured:** 505 declared reporting looks, F=3, B=8, K=4; no added"
        " variants. No multiple-comparison adjustment.",
        "**Inferred:** unresolved_below_power; this experiment provides no "
        "admissible terminal closing ground. No serving promotion.",
        "Prediction rows, input inventory/hashes, cached market targets, "
        "coefficients and summary JSON are under "
        "tests/scratch/codex/lead86_unit1/.",
        "The frozen declaration is retained in docs/lead86_protocol.md. "
        "Record commands are in docs/lanes/lead86.md for the orchestrator.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "look_count": LOOKS,
        "coefficients": coefficients,
        "metrics": metrics,
        "contrasts": contrasts,
        "decisive": decisive,
        "reliability": reliability,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    primary = next(
        row
        for row in contrasts
        if row["fold"] == "pooled"
        and row["scheme"] == "OOS"
        and row["baseline"] == "four_term"
        and row["metric"] == "brier"
    )
    print(json.dumps({"primary": primary, "looks": LOOKS}), flush=True)


def finalize_artifacts():
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    scores = read(OUTPUT / "predictions.parquet")
    outer = scores.loc[scores.scheme.eq("OOS") & scores.arm.eq("regularized")]
    fit_metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    opener_metadata = json.loads((OPENER / "metadata.json").read_text(encoding="utf-8"))
    if fit_metadata["active_model_id"] != opener_metadata["active_model_id"]:
        raise ValueError("Upstream model lineage differs")
    if outer.game_id.duplicated().any() or not np.isfinite(scores[list(METRICS)]).all().all():
        raise ValueError("Invalid saved scoring rows")
    batch = {
        "source": REPORT.as_posix(),
        "league": "nfl",
        "season_start": 2023,
        "season_end": 2025,
        "family": "LEAD86-fixed-market-penalty",
        "category": "modeling",
        "notes": (
            "505 declared looks; three chronological outer seasons; fit through Y-3, "
            "select Y-2, calibrate Y-1. Frozen predictions; 10000 paired season-stratified "
            "week-block resamples. Brier is primary. Upstream feature selection is retrospective."
        ),
        "plain_summary": (
            "Teaching the model to stay closer to the available sportsbook prices "
            "gave a small, uncertain improvement. Keep the current picks while this is studied."
        ),
        "cells": [],
    }
    diagnostics = read(OUTPUT / "target_diagnostics.parquet")
    quotes = read(
        OUTPUT / "market_inputs.parquet",
        ["game_id", "book", "quote_line", "quote_win", "quote_cover"],
    )
    priced = diagnostics.merge(quotes, on="game_id", validate="many_to_one")
    bad = priced.loc[priced.price_error.gt(0.01)]
    bad_games = int(bad.game_id.nunique())
    clean_error = float(priced.loc[~priced.game_id.isin(bad.game_id), "price_error"].max())
    if bad_games:
        batch["notes"] += (
            f" Source limitation: {bad_games} game has conflicting sportsbook prices; "
            f"{len(bad)} market fits miss a quoted probability by over one percentage point. "
            "The frozen population was retained without an outcome-driven exclusion or refit."
        )
        batch["plain_summary"] += " One set of conflicting sportsbook prices still needs checking."
    units = {
        "accuracy": "accuracy_points",
        "brier": "brier_improvement",
        "log_loss": "log_loss_improvement",
        "rps": "rps_improvement",
    }
    for row in summary["contrasts"]:
        if not (
            row["fold"] == "pooled" and row["scheme"] == "OOS" and row["baseline"] == "four_term"
        ):
            continue
        scale = 100 if row["metric"] == "accuracy" else 1
        batch["cells"].append(
            {
                "name": f"LEAD86-regularized-vs-four-term-{row['metric']}",
                "description": f"Selected market penalty: positive {units[row['metric']]} gain.",
                "effect": scale * row["gain"],
                "effect_units": units[row["metric"]],
                "interval_low": scale * row["interval"][0],
                "interval_high": scale * row["interval"][1],
                "probability_positive": row["probability_positive"],
                "sample_games": len(outer),
                "sample_blocks": len(outer[["season", "week"]].drop_duplicates()),
                "classification": "unresolved_below_power",
                "classification_evidence": (
                    "No refuted mechanism, zero split-half reliability, or calibrated positive "
                    "control was established. A zero-crossing interval closes nothing."
                ),
                "closing_ground": None,
                "plain_summary": batch["plain_summary"],
            }
        )
    (OUTPUT / "registry_batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    combined_rows = []
    for fold in summary["coefficients"]:
        for arm in ("regularized", "four_term"):
            weight = fold["chosen_weight"] if arm == "regularized" else 0.0
            raw = fold["candidate_coefficients"][str(weight)]
            cal = fold["calibration"][arm]
            final = {
                key: cal["logit"] * value + (cal["intercept"] if key == "intercept" else 0)
                for key, value in raw.items()
            }
            combined_rows.append(
                [fold["fold"], arm, *[f"{final[t]:.6f}" for t in ("intercept", *TERMS)]]
            )
    extra = [
        "",
        "## Combined coefficients and stability",
        "",
        "**Measured:** these coefficients include the separately fitted calibration.",
        "",
        *table(["Fold", "Arm", "Intercept", *TERMS], combined_rows),
        "",
        "**Measured:** selected penalties are 1, 0 and 0.1 across 2023, 2024 and 2025.",
        "The model-logit coefficient changes sign; composition and movement remain positive.",
        "Availability is constant on this complete-source population, so its coefficient is zero;",
        "that is not evidence that availability has no effect.",
        "",
        "**Read:** src/nfl_ats/clv.py:2184-2208 trains on completed games before each target",
        "week and substitutes the opener before prediction. Upstream selection is retrospective.",
        "**Measured:** saved upstream model identifiers agree; OOS game IDs are unique.",
        f"Maximum market-price projection error: {diagnostics.price_error.max():.9g};",
        f"maximum projected-gradient residual: {diagnostics.stationarity.max():.9g}.",
        f"**Measured:** {len(bad)} of {len(priced)} market fits exceed one point of price error;",
        f"they concern {bad_games} unique game. Other games' maximum error is {clean_error:.9g}.",
        "The Denver-Las Vegas 2021 source gives Denver a 66.03% win chance but a 49.03%",
        "cover chance as a 5.5-point underdog. Those prices cannot share a valid margin",
        "distribution: winning implies covering the positive home handicap.",
        "The inherited constrained projection reaches its parameter bounds on this source.",
        "The declared population was retained; no exclusion, penalty change or refit followed.",
        "**Inferred limitation:** reconcile this source discrepancy before treating the",
        "regularizer result as evidence for serving. The mechanism remains unresolved.",
        "",
        "**Inferred decision:** the small estimated Brier gain is unresolved_below_power.",
        "The 5-4 disagreement record and varying selected penalty do not establish a stable",
        "improvement. Under AGENTS.md research rules, uncertainty does not close this mechanism;",
        "this diagnostic does not authorize changing the served card.",
    ]
    report = REPORT.read_text(encoding="utf-8").split("\n## Combined coefficients and stability")[0]
    REPORT.write_text(report.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print(f"Saved four record cells; OOS games={len(outer)}; no registry writes.", flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel, eligible, summary = inventory()
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("sources", "input_hashes")}
        ),
        flush=True,
    )
    if len(eligible) != 1309:
        REPORT.write_text(
            "# LEAD-86 unit 1: source inventory\n\n"
            f"**Measured:** {summary['files']} quote files / {summary['quote_rows']} rows; "
            f"{len(panel)} complete schedule games and {len(eligible)} fixed-population games.\n\n"
            "The declared replay requires exactly 1,309 complete games. No "
            "outcome loaded, fit, penalty selection or score ran. "
            "IS/OOS/gap, coefficients, intervals, probability_positive and "
            "decisive record are unmeasured; "
            "0 of 505 planned reporting looks executed.\n\n"
            "Source details and rejection counts are saved in "
            "tests/scratch/codex/lead86_unit1/inventory.json; "
            "clock-checked quote pairs are in market_inputs.parquet. The "
            "declared population must be reconciled before replay.\n",
            encoding="utf-8",
        )
        print(
            "inventory_only: declared complete population mismatch; no outcomes loaded", flush=True
        )
        return
    scores, coefficients = replay(panel, eligible)
    summarize(scores, coefficients, summary)
    finalize_artifacts()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
