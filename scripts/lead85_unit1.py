from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lead66_unit1 as lattice
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lead73_unit2 import recover_moves
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logit
from sunday_market_probability_eval import QUOTE_CACHE, sunday_move
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import signed_composition_flags
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    SCHEDULE_COLUMNS,
    _arrest_incidents,
    _design,
    _fit_logit,
    _forecast_temperatures,
    _natural_coefficients,
    _protection_back_side,
    _standardisers,
)

OUTPUT = Path("tests/scratch/codex/lead85_unit1")
REPORT = Path("docs/lead85_unit1.md")
LANE = Path("docs/lanes/lead85.md")
FIT = Path("artifacts/pick_probability/20260929T192747Z")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
ARMS = ("four_term", "integrated")
METRICS = ("accuracy_points", "brier", "log_loss", "rps")
GRID = np.arange(-100, 101, dtype=float)
FOLDS = (2023, 2024, 2025)
BOOTSTRAPS = 10000
SEED = 85
LOOKS = 713


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frame(path, columns=None):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def population():
    print("Checking frozen PMFs and upstream training dates", flush=True)
    games, pool, hashes, inventory = lattice.load_inputs()
    fitted = read_frame(FIT / "per_game.parquet", ["game_id", *FIT_FEATURES])
    metadata = json.loads((FIT / "metadata.json").read_text(encoding="utf-8"))
    if metadata["active_model_id"] != inventory["source_model_id"]:
        raise ValueError("Combined and discrete model identities differ")
    schedules = read_frame(SCHEDULE)
    joined = games.merge(
        schedules[[name for name in SCHEDULE_COLUMNS if name in schedules]],
        on=["game_id", "season"],
        suffixes=("", "_schedule"),
        validate="one_to_one",
    )
    flags = signed_composition_flags(
        joined,
        schedules,
        incidents=_arrest_incidents(Path("data")),
        forecasts_tuesday_noon=_forecast_temperatures(Path("data")),
        protection_back_side=_protection_back_side(schedules, Path("data")),
    )
    games = games.merge(flags[["game_id", "composition_flag_sum"]], validate="one_to_one")
    parity = games.merge(fitted, on="game_id", suffixes=("", "_frozen"), validate="one_to_one")
    if not parity.composition_flag_sum.eq(parity.composition_flag_sum_frozen).all():
        raise ValueError("Reconstructed push-preserving flags fail frozen parity")
    games["model_logit"] = logit(games.home_cover_probability_at_open.to_numpy(float))
    if not np.allclose(
        games.set_index("game_id").loc[fitted.game_id, "model_logit"], fitted.model_logit
    ):
        raise ValueError("Discrete model logit fails frozen parity")
    clock = schedules.loc[
        schedules.game_id.isin(games.game_id), ["game_id", "gameday", "gametime"]
    ].copy()
    local = pd.to_datetime(clock.gameday.astype(str).str[:10] + " " + clock.gametime.astype(str))
    clock["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    games = games.merge(clock[["game_id", "kickoff"]], validate="one_to_one")
    day = games.kickoff.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    first = day.groupby([games.season, games.week]).transform("min")
    sunday = first + pd.to_timedelta((6 - first.dt.dayofweek) % 7, unit="D")
    games["nomination_cutoff"] = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    games["nomination_eligible"] = games.kickoff.gt(games.nomination_cutoff)
    print("Reconstructing dated historical moves for both arms", flush=True)
    early, audit = recover_moves()
    early = early[["game_id", "recovered_move", "sunday_books"]].rename(
        columns={"recovered_move": "sunday_move"}
    )
    quotes = read_frame(QUOTE_CACHE)
    quotes = quotes.loc[
        quotes.decision_label.eq("intraday_hourly")
        & quotes.archive_season.isin(FOLDS)
        & quotes.nflverse_game_id.isin(games.game_id)
    ].copy()
    for name in ("observed_at_utc", "bookmaker_last_update_utc"):
        quotes[name] = pd.to_datetime(quotes[name], utc=True)
    bounds = games.loc[games.season.isin(FOLDS), ["game_id", "kickoff", "season", "week"]].copy()
    bounds["commence_time_utc"] = bounds.kickoff
    bounds["week_first_commence_utc"] = bounds.groupby(["season", "week"]).kickoff.transform("min")
    check = quotes.merge(
        bounds[["game_id", "season", "week"]],
        left_on="nflverse_game_id",
        right_on="game_id",
        validate="many_to_one",
    )
    if not check.archive_season.eq(check.season).all():
        raise ValueError("Archive season differs from target game")
    if "archive_week" in check and not check.archive_week.eq(check.week).all():
        raise ValueError("Archive week differs from target game")
    moves = pd.concat([early, sunday_move(quotes, bounds)], ignore_index=True)
    games = games.merge(moves, how="left", on="game_id", validate="one_to_one")
    excluded = games.loc[games.sunday_move.isna(), ["game_id", "season", "week", "margin_vs_open"]]
    games = games.loc[games.sunday_move.notna()].copy().reset_index(drop=True)
    games["market_move_toward_home"] = games.sunday_move
    games["market_move_available"] = 1.0
    if not np.isfinite(games[list(FIT_FEATURES)].to_numpy(float)).all():
        raise ValueError("Incomplete source-complete feature matrix")
    parity = games.loc[games.season.isin(FOLDS)].merge(
        fitted, on="game_id", suffixes=("", "_frozen")
    )
    available = parity.market_move_available_frozen.eq(1)
    error = float(
        np.max(
            np.abs(
                parity.loc[available, "market_move_toward_home"]
                - parity.loc[available, "market_move_toward_home_frozen"]
            )
        )
    )
    if error > 1e-9:
        raise ValueError(f"Late-move reproduction error: {error}")
    mass = np.zeros((len(games), len(GRID)))
    print("Reconstructing integer margin masses, retaining pushes", flush=True)
    for (season, week), group in games.groupby(["season", "week"]):
        full_week = schedules.loc[schedules.season.eq(season) & schedules.week.eq(week)]
        prior = lattice.prior_pool_for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=pd.to_datetime(full_week.gameday).min(),
            exclude_game_ids=group.game_id,
        )
        for index, game in group.iterrows():
            selected, _ = lattice.select_band(prior, float(game.tue_open_home_spread))
            atoms, counts = lattice.atomic_counts(selected)
            values, _ = lattice.tilted_atoms(
                atoms, counts, float(game.tue_open_home_spread), float(game.model_center)
            )
            if not np.equal(atoms, np.round(atoms)).all() or np.abs(atoms).max() > 100:
                raise ValueError("Invalid integer margin support")
            mass[index, (atoms + 100).astype(int)] = values
    line = games.tue_open_home_spread.to_numpy(float)
    push = (mass * (line[:, None] == GRID)).sum(axis=1)
    home = (mass * (line[:, None] < GRID)).sum(axis=1)
    pmf_error = float(
        max(
            np.max(np.abs(push - games.push_probability_at_open)),
            np.max(np.abs(home / (1 - push) - games.home_cover_probability_at_open)),
            np.max(np.abs(mass.sum(axis=1) - 1)),
        )
    )
    if pmf_error > 2e-7:
        raise ValueError(f"PMF reconstruction error: {pmf_error}")
    inventory.update(
        source_complete_games=len(games),
        source_complete_pushes=int(games.margin_vs_open.eq(0).sum()),
        excluded_missing_move=len(excluded),
        late_move_max_error=error,
        pmf_max_error=pmf_error,
        early_quote_sources=len(audit["sources"]),
        early_quote_rows=audit["admissible_quote_rows"],
        by_season=games.groupby("season").size().to_dict(),
    )
    hashes.update(
        {
            str(path): digest(path)
            for path in (
                FIT / "per_game.parquet",
                FIT / "metadata.json",
                SCHEDULE,
                QUOTE_CACHE,
                Path("scripts/lead66_unit1.py"),
                Path("scripts/lead73_unit2.py"),
            )
        }
    )
    audit.pop("evidence").to_parquet(OUTPUT / "early_quote_evidence.parquet", index=False)
    excluded.to_parquet(OUTPUT / "excluded_missing_move.parquet", index=False)
    return games, mass, {"hashes": hashes, "inventory": inventory, "early_quotes": audit}


def integrated_probability(design, beta, covariance):
    nodes, weights = hermgauss(20)
    weights /= np.sqrt(np.pi)
    variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    if np.min(variance) < -1e-9:
        raise ValueError("Negative predictive coefficient variance")
    node_p = expit((design @ beta)[:, None] + np.sqrt(2 * np.maximum(variance, 0))[:, None] * nodes)
    return node_p @ weights, node_p, weights, variance


def reweight_mass(mass, line, probability):
    home, away = GRID[None, :] > line[:, None], GRID[None, :] < line[:, None]
    h, a = (mass * home).sum(axis=1), (mass * away).sum(axis=1)
    if np.any(h <= 0) or np.any(a <= 0):
        raise ValueError("A discrete PMF lacks a side")
    return mass * (
        home * ((h + a) * probability / h)[:, None]
        + away * ((h + a) * (1 - probability) / a)[:, None]
        + ~(home | away)
    )


def temperature(probability, outcome):
    logits = logit(np.clip(probability, 1e-12, 1 - 1e-12))

    def objective(value):
        z = logits * value[0]
        return float(np.mean(np.logaddexp(0, z) - outcome * z)), np.array(
            [np.mean(logits * (expit(z) - outcome))]
        )

    fit = minimize(objective, np.ones(1), jac=True, bounds=[(1e-6, None)], method="L-BFGS-B")
    if not fit.success:
        raise ValueError(f"Temperature fit failed: {fit.message}")
    return float(fit.x[0])


def fit_adapter(games, mass):
    predictions, coefficients = [], []
    for fold in FOLDS:
        train = games.loc[games.season.le(fold - 3) & games.margin_vs_open.ne(0)]
        cal_mask = games.season.eq(fold - 1) & games.margin_vs_open.ne(0)
        calibration = games.loc[cal_mask]
        if train.empty or calibration.empty or games.loc[games.season.eq(fold - 2)].empty:
            raise ValueError(f"Missing chronological role in fold {fold}")
        means, stds = _standardisers(train)
        x = _design(train, means, stds)
        beta = _fit_logit(x, train.margin_vs_open.gt(0).to_numpy(float), FIT_RIDGE)
        p = expit(x @ beta)
        hessian = (x.T * (p * (1 - p))) @ x + FIT_RIDGE * np.eye(x.shape[1])
        covariance = np.linalg.inv(hessian)
        design = _design(games, means, stds)
        integrated, nodes, weights, variance = integrated_probability(design, beta, covariance)
        raw = {"four_term": expit(design @ beta), "integrated": integrated}
        slopes = {
            arm: temperature(values[cal_mask], calibration.margin_vs_open.gt(0).to_numpy(float))
            for arm, values in raw.items()
        }
        transform = np.zeros_like(covariance)
        transform[0, 0] = 1
        for index, name in enumerate(FIT_FEATURES, 1):
            transform[index, index] = 1 / stds[name]
            transform[0, index] = -means[name] / stds[name]
        natural_covariance = transform @ covariance @ transform.T
        natural = _natural_coefficients(beta, means, stds)
        intervals = {
            name: [
                natural[name] - 1.96 * np.sqrt(natural_covariance[i, i]),
                natural[name] + 1.96 * np.sqrt(natural_covariance[i, i]),
            ]
            for i, name in enumerate(["intercept", *FIT_FEATURES])
        }
        coefficients.append(
            {
                "fold": fold,
                "train_seasons": sorted(train.season.unique().tolist()),
                "train_nonpush": len(train),
                "tuning_season_reserved": fold - 2,
                "calibration_season": fold - 1,
                "calibration_nonpush": len(calibration),
                "ridge": FIT_RIDGE,
                "coefficients": natural,
                "natural_covariance": natural_covariance.tolist(),
                "coefficient_intervals": intervals,
                "inverse_temperatures": slopes,
                "hessian_condition": float(np.linalg.cond(hessian)),
                "means": means,
                "stds": stds,
            }
        )
        print(f"Fold {fold}: train={len(train)}, calibration={len(calibration)}", flush=True)
        for split, mask in (("IS", games.season.le(fold - 3)), ("OOS", games.season.eq(fold))):
            selected = games.loc[mask].copy()
            row_mass = mass[mask]
            line = selected.tue_open_home_spread.to_numpy(float)
            selected["fold"], selected["split"] = fold, split
            selected["coefficient_logit_variance"] = variance[mask]
            for arm in ARMS:
                mixed = reweight_mass(row_mass, line, raw[arm][mask])
                if arm == "integrated":
                    explicit = sum(
                        weights[j] * reweight_mass(row_mass, line, nodes[mask, j])
                        for j in range(20)
                    )
                    if np.max(np.abs(explicit - mixed)) > 1e-12:
                        raise ValueError("Quadrature probability and PMF mixture disagree")
                probability = expit(slopes[arm] * logit(np.clip(raw[arm][mask], 1e-12, 1 - 1e-12)))
                calibrated = reweight_mass(mixed, line, probability)
                push = (calibrated * (line[:, None] == GRID)).sum(axis=1)
                home = (calibrated * (line[:, None] < GRID)).sum(axis=1)
                if not np.allclose(home / (1 - push), probability, atol=1e-12, rtol=0):
                    raise ValueError("Calibrated PMF and conditional probability disagree")
                if not np.allclose(push, selected.push_probability_at_open, atol=2e-7, rtol=0):
                    raise ValueError("Calibration changed archived push mass")
                selected[f"{arm}_probability"], selected[f"{arm}_push"] = probability, push
                y, active = (
                    selected.margin_vs_open.gt(0).to_numpy(float),
                    selected.margin_vs_open.ne(0).to_numpy(),
                )
                bounded = np.clip(probability, 1e-12, 1 - 1e-12)
                selected[f"{arm}_accuracy_points"] = np.where(
                    active, 100 * ((probability >= 0.5) == y), np.nan
                )
                selected[f"{arm}_brier"] = np.where(active, (probability - y) ** 2, np.nan)
                selected[f"{arm}_log_loss"] = np.where(
                    active, -(y * np.log(bounded) + (1 - y) * np.log1p(-bounded)), np.nan
                )
                selected[f"{arm}_rps"] = np.square(
                    np.cumsum(calibrated, axis=1) - (selected.result.to_numpy()[:, None] <= GRID)
                ).sum(axis=1)
                if split == "OOS":
                    np.savez_compressed(
                        OUTPUT / f"pmf_{fold}_{arm}.npz",
                        game_ids=selected.game_id.to_numpy().astype(str),
                        grid=GRID,
                        mass=calibrated,
                    )
            predictions.append(selected)
    return pd.concat(predictions, ignore_index=True), coefficients


def interval(values):
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "probability_positive": float(
            np.mean(values > 1e-12) + 0.5 * np.mean(np.abs(values) <= 1e-12)
        ),
    }


def bootstrap(frame, seed):
    columns = [f"{arm}_{metric}" for arm in ARMS for metric in METRICS]
    values = frame[columns].to_numpy(float)
    aggregate = (
        pd.DataFrame(
            np.column_stack([np.nan_to_num(values), np.isfinite(values).astype(float)]),
            index=pd.MultiIndex.from_frame(frame[["season", "week"]]),
        )
        .groupby(level=[0, 1])
        .sum()
    )
    seasons = aggregate.index.get_level_values(0).unique().to_numpy()
    blocks = {season: aggregate.loc[season].to_numpy() for season in seasons}
    total, count = aggregate.to_numpy().sum(axis=0), len(columns)
    draws, rng = np.empty((BOOTSTRAPS, count)), np.random.default_rng(seed)
    for draw in range(BOOTSTRAPS):
        sampled = np.zeros(count * 2)
        for season in rng.choice(seasons, len(seasons), replace=True):
            weeks = blocks[season]
            sampled += weeks[rng.integers(0, len(weeks), len(weeks))].sum(axis=0)
        draws[draw] = sampled[:count] / sampled[count:]
    return columns, total[:count] / total[count:], draws


def evaluate(predictions):
    rows, decisive, reliability = [], [], []
    for panel in (*FOLDS, "pooled"):
        group = predictions if panel == "pooled" else predictions.loc[predictions.fold.eq(panel)]
        estimates = {
            split: bootstrap(
                group.loc[group.split.eq(split)],
                SEED + (0 if panel == "pooled" else panel) + (100 if split == "IS" else 0),
            )
            for split in ("IS", "OOS")
        }
        for split in ("IS", "OOS", "gap"):
            if split == "gap":
                point, draws = (
                    estimates["OOS"][1] - estimates["IS"][1],
                    estimates["OOS"][2] - estimates["IS"][2],
                )
            else:
                _, point, draws = estimates[split]
            for index, name in enumerate(estimates["IS"][0]):
                arm = next(arm for arm in ARMS if name.startswith(arm + "_"))
                rows.append(
                    dict(
                        panel=panel,
                        split=split,
                        arm=arm,
                        metric=name.removeprefix(arm + "_"),
                        value=float(point[index]),
                        **interval(draws[:, index]),
                    )
                )
            for index, metric in enumerate(METRICS):
                sign = 1 if metric == "accuracy_points" else -1
                difference = sign * (draws[:, index + len(METRICS)] - draws[:, index])
                rows.append(
                    dict(
                        panel=panel,
                        split=split,
                        arm="improvement",
                        metric=metric,
                        value=float(sign * (point[index + len(METRICS)] - point[index])),
                        **interval(difference),
                    )
                )
        frame = group.loc[group.split.eq("OOS") & group.margin_vs_open.ne(0)]
        changed = frame.four_term_probability.ge(0.5) != frame.integrated_probability.ge(0.5)
        wins = frame.loc[changed, "integrated_probability"].ge(0.5) == frame.loc[
            changed, "margin_vs_open"
        ].gt(0)
        decisive.append(
            {
                "panel": panel,
                "n": int(changed.sum()),
                "candidate_wins": int(wins.sum()),
                "candidate_losses": int((~wins).sum()),
                "nonpush_games": len(frame),
                "four_term_wins": int(frame.four_term_accuracy_points.eq(100).sum()),
                "integrated_wins": int(frame.integrated_accuracy_points.eq(100).sum()),
            }
        )
    frame = predictions.loc[predictions.split.eq("OOS") & predictions.margin_vs_open.ne(0)]
    for arm in ARMS:
        bands = np.minimum((frame[f"{arm}_probability"] * 5).astype(int), 4)
        for band in range(5):
            cell = frame.loc[bands.eq(band)]
            reliability.append(
                {
                    "arm": arm,
                    "low": band / 5,
                    "high": (band + 1) / 5,
                    "n": len(cell),
                    "mean_probability": float(cell[f"{arm}_probability"].mean())
                    if len(cell)
                    else None,
                    "home_cover_rate": float(cell.margin_vs_open.gt(0).mean())
                    if len(cell)
                    else None,
                }
            )
    return rows, decisive, reliability


def markdown_table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def render(summary):
    inventory = summary["lineage"]["inventory"]
    result = summary["metrics"]
    pooled = [row for row in result if row["panel"] == "pooled"]
    lines = [
        "# LEAD-85 unit 1: integrated coefficient probability adapter",
        "",
        (
            "**Measured:** `.tools/uv.exe run --no-sync python scripts/lead85_unit1.py`; "
            "protocol saved before outcomes in `docs/lanes/lead85.md`."
        ),
        "",
        "## Decisive games first",
        "",
        markdown_table(
            [
                "Outer season",
                "Decisive n",
                "Candidate W-L",
                "Four-term / candidate wins",
                "Nonpush n",
            ],
            [
                [
                    r["panel"],
                    r["n"],
                    f"{r['candidate_wins']}-{r['candidate_losses']}",
                    f"{r['four_term_wins']} / {r['integrated_wins']}",
                    r["nonpush_games"],
                ]
                for r in summary["decisive"]
            ],
        ),
        "",
        (
            "**Inferred:** symmetric Gaussian logit integration and positive "
            "temperature retain each point-fit side; heterogeneous uncertainty "
            "can still change weekly ranking. Zero decisive games provide "
            "no accuracy mechanism test."
        ),
        "",
        "## Population and chronology",
        "",
        (
            f"**Measured:** {inventory['source_rows']} frozen opener rows; "
            f"{inventory['source_complete_games']} source-complete games, including "
            f"{inventory['source_complete_pushes']} pushes; {inventory['excluded_missing_move']} "
            f"excluded for missing dated moves. No historical pool captures "
            f"required."
        ),
        (
            f"**Measured:** counts {inventory['by_season']}; {inventory['early_quote_sources']} "
            f"verified early quote files, {inventory['early_quote_rows']} admissible "
            f"early rows. Max late-move error {inventory['late_move_max_error']:.3g}; "
            f"archived PMF error {inventory['pmf_max_error']:.3g}."
        ),
        (
            "**Read:** frozen upstream per-game training dates must precede "
            "their game; archived PMFs are reconstructed from completed prior "
            "games. These previously examined archives and upstream feature "
            "choices are retrospective, not untouched prospective evidence."
        ),
        (
            "**Measured:** outer 2023/2024/2025; combined fit through Y-3, "
            "reserve Y-2, calibrate on Y-1. Existing ridge 0.001 is fixed "
            "with no search. Both arms use identical source-complete rows. "
            "Pushes remain in PMFs/RPS and are excluded from conditional fitting/scoring."
        ),
        "",
        "## Adapter and uncertainty",
        "",
        (
            "**Measured:** inverse penalized Hessian, fixed 20-node Gauss-Hermite "
            "integration, same PMF reweighted within each home/away region "
            "at every node with unchanged push mass. Explicit node mixtures "
            "match integrated probabilities. Separate positive temperatures "
            "use the same calibration season and coherently reweight each "
            "final PMF."
        ),
        (
            "**Measured:** 10,000 paired hierarchical season/week bootstrap "
            "draws, seed 85; 95% percentile intervals, half-weight zero effects. "
            "Fixed predictions are not refitted. Gap is OOS minus optimistic "
            "IS; training games repeated across folds stay in their original "
            "week blocks."
        ),
        (
            "**Read:** 713 declared study looks: (27*6+6+4)*(3+1)+25. Unit "
            "1 reports two arms and four ordinary endpoints; nominee-Brier "
            "primary, weekly reward and all five arms belong to the row's "
            "unit 2. No variant is selected from these results."
        ),
        "",
        "## In-sample, held-out and gap",
        "",
        markdown_table(
            ["Arm", "Metric", "IS [95% interval]", "OOS [95% interval]", "OOS-IS [95% interval]"],
            [
                [
                    arm,
                    metric,
                    *[
                        next(
                            f"{r['value']:.6f} [{r['low']:.6f}, {r['high']:.6f}]"
                            for r in pooled
                            if r["arm"] == arm and r["metric"] == metric and r["split"] == split
                        )
                        for split in ("IS", "OOS", "gap")
                    ],
                ]
                for arm in ARMS
                for metric in METRICS
            ],
        ),
        "",
        "## Paired held-out improvement over the fitted four-term recipe",
        "",
        markdown_table(
            ["Panel", "Metric", "Improvement [95% interval]", "probability_positive"],
            [
                [
                    r["panel"],
                    r["metric"],
                    f"{r['value']:.6f} [{r['low']:.6f}, {r['high']:.6f}]",
                    f"{r['probability_positive']:.6f}",
                ]
                for r in result
                if r["arm"] == "improvement" and r["split"] == "OOS"
            ],
        ),
        "",
        "## Fold coefficients and calibration",
        "",
        (
            "**Measured:** natural-scale point coefficients and Laplace 95% "
            "intervals. Both arms share the combined fit. Availability is "
            "constant in this source-complete population, so its coefficient "
            "is unidentified by the likelihood; ridge supplies precision. "
            "Full covariance and standardizers are saved in the scratch summary."
        ),
        "",
        markdown_table(
            ["Fold", "Train n", "Coefficient", "Estimate [95% interval]"],
            [
                [
                    f["fold"],
                    f["train_nonpush"],
                    name,
                    (
                        f"{value:.6f} [{f['coefficient_intervals'][name][0]:.6f}, "
                        f"{f['coefficient_intervals'][name][1]:.6f}]"
                    ),
                ]
                for f in summary["coefficients"]
                for name, value in f["coefficients"].items()
            ],
        ),
        "",
        markdown_table(
            [
                "Fold",
                "Cal n",
                "Four-term inverse temperature",
                "Integrated inverse temperature",
                "Hessian condition",
            ],
            [
                [
                    f["fold"],
                    f["calibration_nonpush"],
                    f"{f['inverse_temperatures']['four_term']:.6f}",
                    f"{f['inverse_temperatures']['integrated']:.6f}",
                    f"{f['hessian_condition']:.6g}",
                ]
                for f in summary["coefficients"]
            ],
        ),
        "",
        "## Five equal-width reliability bands",
        "",
        markdown_table(
            ["Arm", "Band", "n", "Mean probability", "Observed home cover"],
            [
                [
                    r["arm"],
                    f"{r['low']:.1f}-{r['high']:.1f}",
                    r["n"],
                    "n/a" if r["mean_probability"] is None else f"{r['mean_probability']:.6f}",
                    "n/a" if r["home_cover_rate"] is None else f"{r['home_cover_rate']:.6f}",
                ]
                for r in summary["reliability"]
            ],
        ),
        "",
        "## State and next unit",
        "",
        (
            "**Inferred:** unresolved_below_power; the predeclared nominee-Brier "
            "primary endpoint is unmeasured in unit 1. Zero crossing closes "
            "nothing. No promotion, rejection, served change or wagering. "
            "The orchestrator records the diagnostic serially; no registry "
            "command was executed here."
        ),
        (
            "Prediction rows, OOS discrete PMFs, source hashes, coefficients "
            "and all fold/pooled IS/OOS/gap intervals are in `tests/scratch/codex/lead85_unit1/`. "
            "Unit 2 replays fixed weekly contenders against four-term, model-only, "
            "timestamp-matched market and Elo, with the existing push/tie/reward "
            "rules and the declared 713-look family."
        ),
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    declaration = LANE.read_text(encoding="utf-8")
    if not all(text in declaration for text in ("20-node", "713", "before outcomes")):
        raise ValueError("Frozen declaration missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "protocol_declaration.md").write_text(declaration, encoding="utf-8")
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with threadpool_limits(limits=2):
        games, mass, lineage = population()
        predictions, coefficients = fit_adapter(games, mass)
        print("Computing paired season/week intervals", flush=True)
        metrics, decisive, reliability = evaluate(predictions)
    predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
    summary = {
        "lineage": lineage,
        "coefficients": coefficients,
        "metrics": metrics,
        "decisive": decisive,
        "reliability": reliability,
        "declared_looks": LOOKS,
        "status": "unresolved_below_power",
        "script_sha256": digest(__file__),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render(summary)
    for row in metrics:
        if row["panel"] == "pooled" and row["split"] == "OOS" and row["arm"] == "improvement":
            print(json.dumps(row), flush=True)
    print(f"Saved {REPORT}; declared looks={LOOKS}; nomination replay remains unit 2", flush=True)


if __name__ == "__main__":
    main()
