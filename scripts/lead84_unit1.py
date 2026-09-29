from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import lead66_unit1 as discrete
import lead71_unit1 as census
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import logit, logsumexp
from threadpoolctl import threadpool_limits

OUTPUT = Path("tests/scratch/codex/lead84_unit1")
REPORT = Path("docs/lead84_unit1.md")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z/per_game.parquet")
BASE = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
ARCHIVED = Path("artifacts/margins/20260929T192312Z/predictions.parquet")
SCHEDULE = Path("data/processed/game_features.parquet")
GRID = np.arange(-100, 101, dtype=float)
FOLDS = (2023, 2024, 2025)
CONSTRAINTS = ("quote_line", "quote_total", "quote_win", "quote_cover")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_frame(path, columns):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def market_probability(first, second):
    prices = np.asarray([first, second], dtype=float)
    if not np.isfinite(prices).all() or np.any(np.abs(prices) < 100):
        raise ValueError("Invalid American odds")
    implied = np.array([-p / (100 - p) if p < 0 else 100 / (100 + p) for p in prices])
    return float(implied[0] / implied.sum())


def book_constraints(group):
    markets = {}
    for market, rows in group.groupby("market", sort=True):
        expected = {"OVER", "UNDER"} if market == "totals" else {"HOME", "AWAY"}
        if len(rows) != 2 or set(rows.outcome_side) != expected:
            return None
        markets[market] = rows.set_index("outcome_side")
    if set(markets) != {"h2h", "spreads", "totals"}:
        return None
    spread, total = markets["spreads"], markets["totals"]
    points = [spread.at[side, "line"] for side in ("HOME", "AWAY")]
    totals = [total.at[side, "line"] for side in ("OVER", "UNDER")]
    if not np.isfinite([*points, *totals]).all():
        return None
    if not np.isclose(sum(points), 0) or not np.isclose(totals[0], totals[1]) or totals[0] <= 0:
        return None
    if not np.allclose(spread.home_spread_line.to_numpy(float), -float(points[0])):
        raise ValueError("Bookmaker handicap disagrees with the home margin threshold")
    try:
        win = market_probability(
            markets["h2h"].at["HOME", "price"], markets["h2h"].at["AWAY", "price"]
        )
        cover = market_probability(spread.at["HOME", "price"], spread.at["AWAY", "price"])
        market_probability(total.at["OVER", "price"], total.at["UNDER", "price"])
    except ValueError:
        return None
    return {
        "quote_line": -float(points[0]),
        "quote_total": float(totals[0]),
        "quote_win": win,
        "quote_cover": cover,
    }


def load_panels():
    schedule = census.load_schedule(SCHEDULE)
    targets = {
        (item.season, item.week): pd.Timestamp(item.requested_at_utc)
        for item in census.plan_backfill(schedule, 2020, 2025, labels=["sun_early_close"])
    }
    columns = [
        "nflverse_game_id",
        "home_team",
        "away_team",
        "commence_time_utc",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "bookmaker_key",
        "market",
        "outcome_side",
        "line",
        "price",
        "home_spread_line",
    ]
    panels, provenance, counts = [], [], Counter()
    for manifest_path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        if (
            request.get("decision_label") != "sun_early_close"
            or request.get("season") not in census.SEASONS
            or manifest.get("provider") != census.ODDS_API_PROVIDER
            or request.get("sport") != census.ODDS_API_SPORT
            or manifest.get("capture_kind") != "historical_backfill"
        ):
            continue
        key = int(request["season"]), int(request["week"])
        if key not in targets:
            continue
        snapshot = census.instant(manifest["snapshot_timestamp_utc"])
        cutoff = targets[key]
        if (
            census.instant(manifest["requested_at_utc"]) != cutoff
            or snapshot > cutoff
            or census.instant(manifest["observed_at_utc"]) != snapshot
            or request.get("odds_format") != "american"
        ):
            counts["rejected_manifest_clocks"] += 1
            continue
        path = manifest_path.parent / "quotes.parquet"
        expected = manifest.get("files", {}).get("quotes.parquet", {}).get("sha256")
        if not expected or digest(path) != expected:
            raise ValueError(f"Quote integrity failure: {path}")
        quotes = read_frame(path, columns)
        counts["quote_files"] += 1
        counts["quote_rows"] += len(quotes)
        target = schedule.loc[schedule.season.eq(key[0]) & schedule.week.eq(key[1])]
        quotes = quotes.merge(
            target[["game_id", "season", "week", "home_team", "away_team", "kickoff"]],
            left_on="nflverse_game_id",
            right_on="game_id",
            validate="many_to_one",
            suffixes=("", "_schedule"),
        )
        for name in (
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "market_last_update_utc",
            "commence_time_utc",
        ):
            quotes[name] = pd.to_datetime(quotes[name], utc=True, errors="coerce", format="mixed")
        if not (
            quotes.home_team.eq(quotes.home_team_schedule)
            & quotes.away_team.eq(quotes.away_team_schedule)
        ).all():
            raise ValueError("Quote teams disagree with schedule")
        earliest_kickoff = quotes[["kickoff", "commence_time_utc"]].min(axis=1)
        valid = (
            quotes.observed_at_utc.eq(snapshot)
            & quotes.bookmaker_last_update_utc.le(snapshot)
            & quotes.market_last_update_utc.le(snapshot)
            & quotes.commence_time_utc.notna()
            & (quotes.commence_time_utc - quotes.kickoff).abs().le(pd.Timedelta(hours=12))
            & earliest_kickoff.gt(cutoff)
            & quotes.bookmaker_key.notna()
            & quotes.market.isin(census.MARKETS)
        )
        counts["excluded_rows_clock_or_market"] += int((~valid).sum())
        quotes = quotes.loc[valid]
        provenance.append(
            {"path": path.as_posix(), "sha256": expected, "manifest_sha256": digest(manifest_path)}
        )
        for (game_id, book), group in quotes.groupby(["game_id", "bookmaker_key"], sort=True):
            constraints = book_constraints(group)
            if constraints is None:
                counts["incomplete_book_panels"] += 1
                continue
            panels.append(
                dict(
                    game_id=game_id,
                    book=book,
                    season=key[0],
                    week=key[1],
                    snapshot=snapshot,
                    cutoff=cutoff,
                    kickoff=group.kickoff.iloc[0],
                    book_update=group.bookmaker_last_update_utc.max(),
                    market_update=group.market_last_update_utc.max(),
                    source=path.as_posix(),
                    **constraints,
                )
            )
    if not panels:
        return pd.DataFrame(), dict(counts), provenance
    panels = pd.DataFrame(panels)
    latest = panels.groupby("game_id").snapshot.transform("max")
    panels = panels.loc[panels.snapshot.eq(latest)].sort_values(["game_id", "book", "source"])
    panels = panels.drop_duplicates(["game_id", "book"])
    panels = panels.loc[panels.groupby("game_id").book.transform("size").ge(2)].copy()
    if not (panels.snapshot.le(panels.cutoff) & panels.cutoff.lt(panels.kickoff)).all():
        raise ValueError("Selected panels violate the prediction clock")
    return panels.reset_index(drop=True), dict(counts), provenance


def load_population(panels):
    games = read_frame(OPENER, ["game_id", "season", "week", "tue_open_home_spread", "result"])
    games = games.loc[games.season.between(2020, 2025)].copy()
    base = read_frame(BASE, ["game_id", "season"])
    if games.game_id.duplicated().any() or base.game_id.duplicated().any():
        raise ValueError("Duplicate upstream game keys")
    nonpush_ids = set(games.loc[games.result.ne(games.tue_open_home_spread), "game_id"])
    if nonpush_ids != set(base.game_id):
        raise ValueError("The opener population differs from the four-term base population")
    averages = panels.groupby("game_id", as_index=False)[list(CONSTRAINTS)].mean()
    clocks = panels.groupby("game_id", as_index=False).agg(
        cutoff=("cutoff", "first"),
        snapshot=("snapshot", "first"),
        kickoff=("kickoff", "first"),
        books=("book", "size"),
        quote_season=("season", "first"),
        quote_week=("week", "first"),
    )
    games = games.merge(averages, on="game_id", validate="one_to_one")
    games = games.merge(clocks, on="game_id", validate="one_to_one")
    if not (games.season.eq(games.quote_season) & games.week.eq(games.quote_week)).all():
        raise ValueError("Quote target week disagrees with the frozen opener")
    archived = read_frame(ARCHIVED, ["game_id", "method", "model_name", "train_max_gameday"])
    archived = archived.loc[archived.method.eq("market_residual") & archived.model_name.eq("ridge")]
    games = games.merge(
        archived[["game_id", "train_max_gameday"]], on="game_id", how="left", validate="one_to_one"
    )
    trained = pd.to_datetime(games.train_max_gameday, utc=True, errors="coerce")
    if trained.isna().any() or not (trained + pd.Timedelta(days=2)).lt(games.cutoff).all():
        raise ValueError("Upstream model training is not completed before the quote cutoff")
    games["is_push"] = games.result.eq(games.tue_open_home_spread)
    if (
        not np.isfinite(games.result).all()
        or not np.equal(games.result, np.rint(games.result)).all()
    ):
        raise ValueError("The prior requires observed integer margins")
    if games.result.abs().gt(100).any():
        raise ValueError("Observed margin exceeds the fixed grid")
    return games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def shared_prior(history, anchor):
    selected, band = discrete.select_band(history, anchor)
    result = selected.result.to_numpy(float)
    line = selected.line.to_numpy(float)
    total = selected.quote_total.to_numpy(float)
    variance_scale = float(np.square(result - line).sum() / total.sum())
    reference_variance = float(variance_scale * total.mean())
    if reference_variance <= 0 or not np.isfinite(reference_variance):
        raise ValueError("The prior has invalid variance")
    normal = np.exp(-np.square(GRID - anchor) / (2 * reference_variance))
    normal /= normal.sum()
    counts = np.bincount(result.astype(int) + 100, minlength=len(GRID)).astype(float)
    return np.log(counts + normal), {
        "variance_scale": variance_scale,
        "reference_variance": reference_variance,
        "prior_games": len(selected),
        "band": band,
    }


def project(log_prior, prior, constraint):
    variance = prior["variance_scale"] * constraint.quote_total
    log_base = log_prior - 0.5 * np.square(GRID - constraint.quote_line) * (
        1 / variance - 1 / prior["reference_variance"]
    )
    payoff = np.column_stack(
        [
            (GRID > 0) - constraint.quote_win * (GRID != 0),
            (constraint.quote_line < GRID)
            - constraint.quote_cover * (constraint.quote_line != GRID),
        ]
    )

    def objective(theta):
        logits = log_base + payoff @ theta
        normalization = logsumexp(logits)
        mass = np.exp(logits - normalization)
        return normalization + 0.5e-6 * theta @ theta, payoff.T @ mass + 1e-6 * theta

    fit = minimize(
        objective,
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-40, 40)] * 2,
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 500},
    )
    gradient = objective(fit.x)[1]
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
        gradient = objective(fit.x)[1]
        stationarity = float(np.max(np.abs(fit.x - np.clip(fit.x - gradient, -40, 40))))
    if not np.isfinite(stationarity) or stationarity > 1e-6:
        raise ValueError(f"Projection failed: {fit.message}; KKT={stationarity}")
    logits = log_base + payoff @ fit.x
    mass = np.exp(logits - logsumexp(logits))
    if not np.isfinite(mass).all() or np.any(mass < 0) or not np.isclose(mass.sum(), 1):
        raise ValueError("Projection did not produce a normalized PMF")
    price_error = max(
        abs(float(mass[line < GRID].sum() / mass[line != GRID].sum()) - probability)
        for line, probability in (
            (0, constraint.quote_win),
            (constraint.quote_line, constraint.quote_cover),
        )
    )
    return mass, {
        "theta_moneyline": float(fit.x[0]),
        "theta_spread": float(fit.x[1]),
        "price_error": price_error,
        "stationarity": stationarity,
        "solver_fallback": fallback,
    }


def frozen_probability(mass, line):
    home, away, push = mass[line < GRID].sum(), mass[line > GRID].sum(), mass[line == GRID].sum()
    if min(home, away) <= 0 or not np.isclose(home + away + push, 1):
        raise ValueError("Invalid frozen-line probability partition")
    return float(home / (home + away)), float(push)


def construct(games, panels):
    rows, diagnostics, fold_rows = [], [], []
    mixture_arrays, average_arrays = [], []
    books_by_game = {str(key): group for key, group in panels.groupby("game_id", sort=False)}
    for outer in FOLDS:
        history = games.loc[games.season.le(outer - 3)].rename(columns={"quote_line": "line"})
        scored = games.loc[games.season.le(outer)]
        future = scored.loc[scored.season.gt(outer - 3)]
        if history.empty or history.season.max() != outer - 3 or future.empty:
            raise ValueError(f"Missing source-complete prior or held-out rows for {outer}")
        if history.kickoff.max() + pd.Timedelta(days=2) >= future.cutoff.min():
            raise ValueError("A fold prior includes games unresolved at a later-stage cutoff")
        fold_rows.append(
            {
                "outer": outer,
                "fit_through": outer - 3,
                "tune": outer - 2,
                "calibrate": outer - 1,
                "prior_games": len(history),
                "rows": len(scored),
                "outer_games": int(scored.season.eq(outer).sum()),
                "prior_last_kickoff": history.kickoff.max().isoformat(),
            }
        )
        for game in scored.itertuples(index=False):
            stage = (
                "fit_IS"
                if game.season <= outer - 3
                else {
                    outer - 2: "tune",
                    outer - 1: "calibrate",
                    outer: "outer_OOS",
                }[game.season]
            )
            prior_log, prior = shared_prior(history, game.quote_line)
            per_book = []
            for book in books_by_game[game.game_id].itertuples(index=False):
                mass, diagnostic = project(prior_log, prior, book)
                per_book.append(mass)
                diagnostics.append(
                    dict(
                        outer=outer,
                        game_id=game.game_id,
                        stage=stage,
                        construction="book",
                        book=book.book,
                        **prior,
                        **diagnostic,
                    )
                )
            average, diagnostic = project(
                prior_log,
                prior,
                SimpleNamespace(**{name: getattr(game, name) for name in CONSTRAINTS}),
            )
            diagnostics.append(
                dict(
                    outer=outer,
                    game_id=game.game_id,
                    stage=stage,
                    construction="averaged_constraints",
                    book="",
                    **prior,
                    **diagnostic,
                )
            )
            mixture = np.mean(per_book, axis=0)
            mixture_p, mixture_push = frozen_probability(mixture, game.tue_open_home_spread)
            average_p, average_push = frozen_probability(average, game.tue_open_home_spread)
            rows.append(
                dict(
                    array_row=len(rows),
                    outer=outer,
                    game_id=game.game_id,
                    season=int(game.season),
                    week=int(game.week),
                    stage=stage,
                    books=int(game.books),
                    frozen_opener=float(game.tue_open_home_spread),
                    cutoff=game.cutoff,
                    snapshot=game.snapshot,
                    mixture_probability=mixture_p,
                    averaged_probability=average_p,
                    mixture_push_probability=mixture_push,
                    averaged_push_probability=average_push,
                    jensen_gap=float(logit(mixture_p) - logit(average_p)),
                    **prior,
                )
            )
            mixture_arrays.append(mixture)
            average_arrays.append(average)
        print(f"fold {outer}: prior={len(history)} distribution_pairs={len(scored)}", flush=True)
    return (
        pd.DataFrame(rows),
        pd.DataFrame(diagnostics),
        pd.DataFrame(fold_rows),
        np.stack(mixture_arrays),
        np.stack(average_arrays),
    )


def write_report(games, rows, diagnostics, folds, inventory):
    outer = rows.loc[rows.stage.eq("outer_OOS")]
    lines = [
        "# LEAD-84 unit 1: mix book lattices before the frozen line",
        "",
        "**Measured:** .tools/uv.exe run --no-sync python scripts/lead84_unit1.py",
        "",
        "**Read:** ROADMAP.md:867 assigns the two constructions to unit 1 and",
        "the paired combined-probability replay to unit 2. Protocol C and",
        "numerical conventions were saved in the lane before fitting.",
        "",
        "## Decisive games and requested performance statistics",
        "",
        "No side was selected or scored. Decisive records, IS/OOS performance",
        "and their gap, combined per-fold coefficients, effect intervals and",
        "probability_positive are **unmeasured**, pending the unit-2 replay.",
        "The Jensen-gap feature must enter the fitted four-term calibrated",
        "probability; it cannot select a side alone. No closure or promotion.",
        "",
        "## Source and chronology",
        "",
        f"**Measured:** {inventory['quote_files']} local Sunday files;",
        f"{inventory['quote_rows']:,} quote rows; {len(games):,} opener games;",
        f"{int(games.is_push.sum())} pushes retained and",
        f"{int((~games.is_push).sum()):,} games in the four-term nonpush population.",
        "Each game has at least two complete books at the same snapshot.",
        "Quote rows are not independent games.",
        "",
        "**Measured:** source hashes, target week, observation, bookmaker,",
        "market and kickoff clocks passed. Snapshots are at or before the fixed",
        "Sunday cutoff and before scheduled and quoted kickoff. Upstream model",
        "training completed at least two days before that cutoff.",
        "No closing columns or pre-2026 pool captures were read. The historical",
        "opener is the frozen pool-line proxy. Archives remain retrospective.",
        "",
        "| Season | Games | Pushes | Nonpush games |",
        "|---|---:|---:|---:|",
    ]
    for season, group in games.groupby("season"):
        push = int(group.is_push.sum())
        lines.append(f"| {season} | {len(group)} | {push} | {len(group) - push} |")
    lines += [
        "",
        "## Fold construction",
        "",
        "**Measured:** the common prior fits only seasons through Y-3.",
        "Training rows are explicitly optimistic IS; tune, calibration and",
        "outer rows are separate. All stages are saved for the replay.",
        "",
        "| Outer | Prior through | Prior games | Tune | Calibrate | Outer games |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fold in folds.itertuples(index=False):
        lines.append(
            f"| {fold.outer} | {fold.fit_through} | {fold.prior_games} |"
            f" {fold.tune} | {fold.calibrate} | {fold.outer_games} |"
        )
    lines += [
        "",
        "**Read:** the fixed integer support is -100..100. Both constructions",
        "share the earlier-season empirical line-band prior and variance",
        "coefficient. Total points adjust variance; entropic projection matches",
        "no-vig moneyline and spread constraints. Two-sided total prices",
        "validate the source; the total point sets variance, as in LEAD-71.",
        "The mixture averages book PMFs equally before conditioning on nonpush",
        "mass at the opener. The comparator averages the same books' spread",
        "points, total points and de-vigged probabilities before one projection.",
        "",
        f"**Measured:** {len(rows):,} paired distributions;",
        f"{len(diagnostics):,} numerical projections; {len(outer):,} outer rows.",
        f"Maximum PMF normalization error: {inventory['max_normalization_error']:.3g}.",
        f"Maximum projection stationarity: {diagnostics.stationarity.max():.3g}.",
        f"Maximum constraint price error: {diagnostics.price_error.max():.6g}.",
        "Price multipliers and variance parameters are saved per fold/game/book.",
        "They are not combined-probability coefficients selecting a side.",
        "",
        "**Read:** F=3, B=8, K=4; **505 planned looks** for the full family.",
        "**Measured:** zero performance looks in this construction unit.",
        "",
        "## Saved handoff",
        "",
        "Under tests/scratch/codex/lead84_unit1/: features.parquet indexes",
        "pmfs.npz; projection_diagnostics.parquet, book_panels.parquet,",
        "folds.json and manifest.json retain numerical checks and provenance.",
        "Prediction rows remain outside docs/.",
        "",
        "Next: reconstruct timestamp-matched moves in every year, fit the",
        "four-term probability plus the sole Jensen-gap term, and perform the",
        "declared five-arm replay using the reserved tune/calibration years.",
        "Report season/week-block intervals, reliability, IS/OOS/gap, per-fold",
        "coefficients, decisive records and all 505 looks. The orchestrator",
        "alone runs record commands; none is warranted before scoring.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panels, inventory, sources = load_panels()
    if panels.empty:
        REPORT.write_text(
            "# LEAD-84 unit 1\n\n**Measured:** no admissible local two-book panels. "
            "Construction and scoring did not run. No effect, interval or verdict is available.\n",
            encoding="utf-8",
        )
        (OUTPUT / "manifest.json").write_text(
            json.dumps(
                {"inventory": inventory, "sources": sources, "state": "source_absent"}, indent=2
            ),
            encoding="utf-8",
        )
        print("source_absent: inventory only", flush=True)
        return
    games = load_population(panels)
    panels = panels.loc[panels.game_id.isin(games.game_id)].copy()
    print(f"inventory: {len(games)} games, {len(panels)} complete book panels", flush=True)
    rows, diagnostics, folds, mixture, average = construct(games, panels)
    inventory["max_normalization_error"] = float(
        max(np.abs(mixture.sum(axis=1) - 1).max(), np.abs(average.sum(axis=1) - 1).max())
    )
    rows.to_parquet(OUTPUT / "features.parquet", index=False)
    diagnostics.to_parquet(OUTPUT / "projection_diagnostics.parquet", index=False)
    panels.to_parquet(OUTPUT / "book_panels.parquet", index=False)
    np.savez_compressed(OUTPUT / "pmfs.npz", grid=GRID, mixture=mixture, averaged=average)
    (OUTPUT / "folds.json").write_text(folds.to_json(orient="records", indent=2), encoding="utf-8")
    manifest = {
        "state": "construction_complete_replay_pending",
        "planned_looks": 505,
        "scored_looks": 0,
        "inventory": inventory,
        "sources": sources,
        "inputs": {
            str(path): digest(path) for path in (OPENER, BASE, ARCHIVED, SCHEDULE, Path(__file__))
        },
        "outputs": {
            name: digest(OUTPUT / name)
            for name in (
                "features.parquet",
                "projection_diagnostics.parquet",
                "book_panels.parquet",
                "pmfs.npz",
                "folds.json",
            )
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(games, rows, diagnostics, folds, inventory)
    print(
        f"complete: {len(rows)} distribution pairs; 0 scored / 505 planned looks; report={REPORT}",
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
