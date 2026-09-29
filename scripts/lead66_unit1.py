from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from nfl_ats.home_side_location import COMPLETION_ALLOWANCE_DAYS, TRAILING_SEASONS
from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    BAND_STEP,
    MAX_BAND,
    MIN_BAND_GAMES,
    prior_pool,
    prior_pool_for_week,
    tilted_atoms,
)

MARGINS = Path("artifacts/margins/20260929T192312Z")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
FEATURE_HASH = "a3eedb0323818f95b029047c5b80de3ce8d458a48b82ae4489c1fb70d16d5685"
REPORT = Path("docs/lead66_unit1.md")
PREDICTIONS = Path("docs/lead66_prediction_scores.md")
SEED = 20260929
REPLICATES = 10_000
ARMS = ("market_residual", "served_lattice", "model_centred_lattice")
CONTRASTS = ((0, 1), (0, 2), (1, 2))
THRESHOLDS = np.arange(-70, 71, dtype=float)
TAIL_EDGES = np.linspace(0.0, 1.0, 6)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fmt(value, digits=6):
    if np.isnan(value):
        return "undefined"
    if np.isinf(value):
        return "infinity" if value > 0 else "-infinity"
    return f"{value:.{digits}f}"


def interval(values):
    ordered = np.sort(values)
    return tuple(float(ordered[max(0, int(np.ceil(p * len(ordered))) - 1)]) for p in (0.025, 0.975))


def interval_text(values):
    low, high = interval(values)
    return f"[{fmt(low)}, {fmt(high)}]"


def wilson(wins, total):
    if total == 0:
        return "unavailable"
    p, z = wins / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return f"[{100 * (center - radius):.2f}%, {100 * (center + radius):.2f}%]"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def select_band(pool, anchor):
    band = BAND_HALF_WIDTH
    distances = np.abs(pool["line"].to_numpy(float) - anchor)
    while True:
        selected = distances <= band
        if selected.sum() >= MIN_BAND_GAMES or band >= MAX_BAND:
            break
        band = min(band + BAND_STEP, MAX_BAND)
    if not selected.any():
        raise ValueError(f"Empty historical line band at {anchor}")
    return pool.loc[selected], band


def atomic_counts(selected):
    atoms, counts = np.unique(selected["result"].to_numpy(float), return_counts=True)
    return atoms, counts.astype(float)


def mass_at_theta(atoms, counts, theta):
    logits = theta * atoms
    unnormalized = counts * np.exp(logits - logits.max())
    return unnormalized / unnormalized.sum()


def recover_center(selected, row):
    atoms, counts = atomic_counts(selected)
    line, target = float(row.tue_open_home_spread), float(row.home_cover_probability_at_open)
    home, away = atoms > line, atoms < line
    if not home.any() or not away.any() or not 0 < target < 1:
        raise ValueError(f"Archived centre is not identifiable for {row.game_id}")

    def difference(theta):
        mass = mass_at_theta(atoms, counts, theta)
        return float(mass[home].sum() / mass[home | away].sum() - target)

    theta = brentq(difference, -1.0, 1.0, xtol=1e-13)
    mass = mass_at_theta(atoms, counts, theta)
    expected = np.array(
        [
            row.home_cover_probability_excluding_push_at_open,
            row.push_probability_at_open,
            row.home_loss_probability_at_open,
        ],
        dtype=float,
    )
    observed = np.array([mass[home].sum(), mass[atoms == line].sum(), mass[away].sum()])
    error = float(np.max(np.abs(expected - observed)))
    if error > 2e-7:
        raise ValueError(f"Archived PMF reconstruction mismatch for {row.game_id}: {error}")
    return float(atoms @ mass), error


def residual_pmf(selected, line):
    residuals = selected["result"].to_numpy(float) - selected["line"].to_numpy(float)
    shift = float(-residuals.mean())
    positions = line + residuals + shift
    low = np.floor(positions)
    fraction = positions - low
    atoms, inverse = np.unique(np.concatenate([low, low + 1]), return_inverse=True)
    mass = np.bincount(inverse, weights=np.concatenate([1 - fraction, fraction]))
    mass /= mass.sum()
    return atoms[mass > 0], mass[mass > 0], shift


def pmfs(pool, line, center):
    selected, band = select_band(pool, line)
    atoms, counts = atomic_counts(selected)
    base_atoms, base_mass, shift = residual_pmf(selected, line)
    mass, theta = tilted_atoms(atoms, counts, line, center)
    centered, centered_band = select_band(pool, center)
    c_atoms, c_counts = atomic_counts(centered)
    c_mass, c_theta = tilted_atoms(c_atoms, c_counts, center, center)
    return [
        (base_atoms, base_mass, shift, band, len(selected)),
        (atoms, mass, theta, band, len(selected)),
        (c_atoms, c_mass, c_theta, centered_band, len(centered)),
    ]


def score(pmf, actual, line):
    atoms, mass, coefficient, band, band_rows = pmf
    if not np.allclose(atoms, np.rint(atoms)) or not np.isclose(mass.sum(), 1.0):
        raise ValueError("PMF must have integer support and unit mass")
    grid = np.arange(min(atoms.min(), actual), max(atoms.max(), actual) + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(mass)])
    cdf = cumulative[np.searchsorted(atoms, grid, side="right")]
    rps = float(np.square(cdf - (actual <= grid)).sum())
    realized = float(mass[atoms == actual].sum())
    log_score = float(-np.log(realized)) if realized > 0 else float("inf")
    push = float(mass[atoms == line].sum())
    home, away = float(mass[atoms > line].sum()), float(mass[atoms < line].sum())
    conditional = home / (home + away) if home + away > 0 else 0.5
    tail = 1 - cumulative[np.searchsorted(atoms, THRESHOLDS, side="right")]
    return {
        "rps": rps,
        "log_score": log_score,
        "home_probability": conditional,
        "push_probability": push,
        "coefficient": coefficient,
        "band": band,
        "band_rows": band_rows,
        "tail": np.clip(tail, 0.0, 1.0),
    }


def load_inputs():
    paths = [
        MARGINS / "metadata.json",
        MARGINS / "predictions.parquet",
        OPENER / "metadata.json",
        OPENER / "per_game.parquet",
        FEATURES,
    ]
    hashes = {str(path): digest(path) for path in paths}
    if hashes[str(FEATURES)] != FEATURE_HASH:
        raise ValueError("Pinned feature table changed; do not silently substitute a source")
    margin_metadata = json.loads((MARGINS / "metadata.json").read_text(encoding="utf-8"))
    opener_metadata = json.loads((OPENER / "metadata.json").read_text(encoding="utf-8"))
    for metadata in (margin_metadata, opener_metadata):
        if metadata["provenance"]["feature_table"]["sha256"] != FEATURE_HASH:
            raise ValueError("Source feature hashes differ")
        if metadata["feature_profile"] != "weak_stack" or metadata["regressor"] != "ridge":
            raise ValueError("Pinned model configuration differs")
    features = pd.read_parquet(
        FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "gameday",
            "game_type",
            "spread_line",
            "result",
        ],
    )
    opener = pd.read_parquet(OPENER / "per_game.parquet")
    archived = pd.read_parquet(
        MARGINS / "predictions.parquet",
        columns=[
            "game_id",
            "season",
            "method",
            "model_name",
            "train_max_gameday",
            "gameday",
        ],
    )
    archived = archived.loc[archived.method.eq("market_residual") & archived.model_name.eq("ridge")]
    if opener.game_id.duplicated().any() or archived.game_id.duplicated().any():
        raise ValueError("Source game IDs must be unique within the selected model")
    if not opener.base_probability_policy.eq("discrete_conditional_non_push_v1").all():
        raise ValueError("Opener source does not use the served discrete reader")
    if not opener.probability_method.eq("gaussian_median").all():
        raise ValueError("Opener probability method differs")
    pool = prior_pool(features, opener.set_index("game_id").tue_open_home_spread)
    games = opener.loc[opener.season.between(2020, 2025)].drop(columns=["result"]).copy()
    games = games.merge(
        archived[["game_id", "train_max_gameday"]],
        on="game_id",
        validate="one_to_one",
        how="left",
    )
    games = games.merge(
        features[["game_id", "gameday", "result", "game_type"]],
        on="game_id",
        validate="one_to_one",
        how="left",
    )
    if games[["train_max_gameday", "gameday", "result"]].isna().any().any():
        raise ValueError("Primary games lack completed outcomes or chronological predictions")
    games["gameday"] = pd.to_datetime(games.gameday)
    if not pd.to_datetime(games.train_max_gameday).lt(games.gameday).all():
        raise ValueError("A frozen model forecast uses training outcomes after its prediction date")
    if not np.allclose(games.margin_vs_open + games.tue_open_home_spread, games.result):
        raise ValueError("Outcome sign or source mismatch")
    if not games.game_type.eq("REG").all() or set(games.season) != set(range(2020, 2026)):
        raise ValueError("Unexpected primary population")
    games = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    games["model_center"] = np.nan
    errors = []
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        target = features.loc[features.season.eq(season) & features.week.eq(week)]
        prior = prior_pool_for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=pd.to_datetime(target.gameday).min(),
            exclude_game_ids=group.game_id,
        )
        locations = []
        for index, row in group.iterrows():
            selected, _ = select_band(prior, float(row.tue_open_home_spread))
            center, error = recover_center(selected, row)
            games.loc[index, "model_center"] = center
            errors.append(error)
            locations.append(center - row.tue_open_home_spread - row.residual_at_open_served)
        if np.ptp(locations) > 2e-5:
            raise ValueError(f"Recovered residual location differs within {season} week {week}")
    inventory = {
        "source_rows": len(opener),
        "archived_first_season": int(archived.season.min()),
        "missing_secondary_seasons": sorted(set(range(2009, 2020)) - set(archived.season)),
        "reconstruction_max_error": max(errors),
        "source_model_id": opener_metadata["active_model_id"],
    }
    return games, pool, hashes, inventory


def evaluate(games, pool):
    rows, fold_rows = [], []
    calibration = np.zeros((6, 3, 5, 3), dtype=float)
    for fold, (season, group) in enumerate(games.groupby("season", sort=True)):
        past = pool.season.ge(season - TRAILING_SEASONS) & pool.season.lt(season)
        completed = (
            pd.to_datetime(pool.gameday) + pd.Timedelta(days=COMPLETION_ALLOWANCE_DAYS)
        ).lt(group.gameday.min())
        training = pool.loc[past & completed]
        in_sample = pd.concat([training, pool.loc[pool.season.eq(season)]], ignore_index=True)
        if training.empty or training.season.eq(season).any():
            raise ValueError("Invalid chronological LOSO training population")
        fold_rows.append(
            [
                season,
                len(group),
                len(training),
                len(in_sample),
                str(training.gameday.max().date()),
            ]
        )
        for _, game in group.iterrows():
            actual, line, center = (
                float(game.result),
                float(game.tue_open_home_spread),
                game.model_center,
            )
            for scheme, history in (("OOS", training), ("IS", in_sample)):
                for arm_index, pmf in enumerate(pmfs(history, line, center)):
                    result = score(pmf, actual, line)
                    tail = result.pop("tail")
                    rows.append(
                        dict(
                            game_id=game.game_id,
                            season=int(season),
                            arm=ARMS[arm_index],
                            scheme=scheme,
                            actual_home_cover=actual > line,
                            actual_push=actual == line,
                            **result,
                        )
                    )
                    if scheme == "OOS":
                        bands = np.minimum(np.searchsorted(TAIL_EDGES, tail, side="right") - 1, 4)
                        outcome = actual > THRESHOLDS
                        for band in range(5):
                            selected = bands == band
                            calibration[fold, arm_index, band] += np.array(
                                [
                                    selected.sum(),
                                    tail[selected].sum(),
                                    outcome[selected].sum(),
                                ]
                            ) / len(THRESHOLDS)
        print(f"Completed chronological fold {season}: {len(group)} held-out games", flush=True)
    return pd.DataFrame(rows), fold_rows, calibration


def resample_metric(frame, column, draws):
    groups = frame.groupby("season", sort=True)[column]
    sums = groups.agg(lambda series: series.to_numpy().sum()).to_numpy()
    counts = groups.size().to_numpy()
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def write_predictions(scored):
    headers = [
        "Game",
        "Arm",
        "Read",
        "RPS",
        "Log score",
        "P home conditional",
        "P push",
        "Coefficient",
    ]
    rows = [
        [
            row.game_id,
            row.arm,
            row.scheme,
            fmt(row.rps),
            fmt(row.log_score),
            fmt(row.home_probability),
            fmt(row.push_probability),
            fmt(row.coefficient),
        ]
        for row in scored.itertuples(index=False)
    ]
    PREDICTIONS.write_text(
        "# LEAD-66 prediction-level scores\n\n"
        "**Measured:** generated by scripts/lead66_unit1.py; see lead66_unit1.md for protocol, "
        "source hashes and limitations. Coefficient is residual recentering for market_residual "
        "and exponential tilt theta for lattice arms. IS deliberately includes target outcomes.\n\n"
        + table(headers, rows)
        + "\n",
        encoding="utf-8",
    )


def paired_results(scored, draws):
    oos = scored.loc[scored.scheme.eq("OOS")]
    records, contrasts = [], []
    for baseline, candidate in CONTRASTS:
        left = oos.loc[oos.arm.eq(ARMS[baseline])].set_index("game_id")
        right = oos.loc[oos.arm.eq(ARMS[candidate])].set_index("game_id").loc[left.index]
        difference = left.rps - right.rps
        wins, losses = int(difference.gt(1e-12).sum()), int(difference.lt(-1e-12).sum())
        disagreement = (
            left.home_probability.ge(0.5).ne(right.home_probability.ge(0.5)) & ~left.actual_push
        )
        correct = right.home_probability.ge(0.5).eq(right.actual_home_cover)
        cover_wins, cover_total = int((disagreement & correct).sum()), int(disagreement.sum())
        name = f"{ARMS[candidate]} vs {ARMS[baseline]}"
        records.append(
            [
                name,
                f"{wins}-{losses}-{len(left) - wins - losses}",
                wilson(wins, wins + losses),
                f"{cover_wins}-{cover_total - cover_wins}",
                wilson(cover_wins, cover_total),
            ]
        )
        delta = pd.DataFrame({"season": left.season, "delta": difference})
        bootstrap = resample_metric(delta, "delta", draws)
        p_positive = float((bootstrap > 0).mean() + 0.5 * (bootstrap == 0).mean())
        is_left = scored.loc[scored.scheme.eq("IS") & scored.arm.eq(ARMS[baseline])].set_index(
            "game_id"
        )
        is_right = scored.loc[scored.scheme.eq("IS") & scored.arm.eq(ARMS[candidate])].set_index(
            "game_id"
        )
        is_delta = float((is_left.rps - is_right.rps).mean())
        contrasts.append(
            {
                "name": name,
                "effect": float(difference.mean()),
                "interval": interval(bootstrap),
                "p": p_positive,
                "in_sample": is_delta,
                "gap": float(difference.mean()) - is_delta,
            }
        )
    return records, contrasts


def write_report(games, scored, fold_rows, calibration, hashes, inventory):
    draws = np.random.default_rng(SEED).integers(0, 6, size=(REPLICATES, 6))
    records, contrasts = paired_results(scored, draws)
    oos = scored.loc[scored.scheme.eq("OOS")]
    text = [
        "# LEAD-66 whole-margin PMF unit 1",
        "",
        "**Measured:** .tools/uv.exe run --no-sync python scripts/lead66_unit1.py.",
        "Protocol was frozen in docs/lanes/lead66.md before outcome scoring.",
        "",
        "## Population and decisive records",
        "",
        f"**Measured:** {len(games)} opener games, "
        f"{int(games.margin_vs_open.ne(0).sum())} non-push "
        f"and {int(games.margin_vs_open.eq(0).sum())} pushes, 2020-2025; "
        f"{inventory['source_rows']} source rows before filtering.",
        "All PMF scores include pushes. No zero-mass outcome was excluded. The roadmap's 1,503 "
        "non-push count is a declaration, not an imposed sample-size filter.",
        "",
        "**Measured:** RPS records are candidate lower-score wins/losses/ties. Cover records use "
        "PMF-implied side disagreements on non-push games: diagnostics only, not served picks. "
        "Wilson intervals describe game counts; inferential RPS intervals resample seasons.",
        "",
        table(
            ["Candidate vs baseline", "RPS W-L-T", "Win fraction 95%", "Cover W-L", "Cover 95%"],
            records,
        ),
        "",
        "## Paired RPS improvements",
        "",
        "**Measured:** positive means lower candidate RPS. RPS sums squared CDF error across every "
        "integer threshold (margin-point units). Bootstrap: 10,000 paired season-block draws, "
        "seed 20260929, game-weighted aggregation, percentile 95% intervals; six blocks.",
        "",
        table(
            ["Contrast", "IS gain", "OOS gain", "OOS 95%", "OOS-IS gap", "probability_positive"],
            [
                [
                    row["name"],
                    fmt(row["in_sample"]),
                    fmt(row["effect"]),
                    f"[{fmt(row['interval'][0])}, {fmt(row['interval'][1])}]",
                    fmt(row["gap"]),
                    fmt(row["p"]),
                ]
                for row in contrasts
            ],
        ),
        "",
        "## Arm metrics and IS-OOS gap",
        "",
        "**Measured:** exact log scores retain infinity for zero realized mass. Infinite IS/OOS "
        "log scores give an undefined gap; no clipping or smoothing is applied. Cover Brier/log "
        "loss exclude pushes; full-margin scores include them.",
        "",
    ]
    arm_rows = []
    for arm in ARMS:
        heldout = oos.loc[oos.arm.eq(arm)]
        fitted = scored.loc[scored.arm.eq(arm) & scored.scheme.eq("IS")]
        nonpush = heldout.loc[~heldout.actual_push]
        probability = nonpush.home_probability.to_numpy()
        observed = nonpush.actual_home_cover.to_numpy(float)
        with np.errstate(divide="ignore"):
            cover_log = -np.mean(np.log(np.where(observed > 0, probability, 1 - probability)))
        arm_rows.append(
            [
                arm,
                fmt(fitted.rps.mean()),
                fmt(heldout.rps.mean()),
                interval_text(resample_metric(heldout, "rps", draws)),
                fmt(heldout.rps.mean() - fitted.rps.mean()),
                fmt(fitted.log_score.mean()),
                fmt(heldout.log_score.mean()),
                interval_text(resample_metric(heldout, "log_score", draws)),
                int(np.isinf(heldout.log_score).sum()),
                fmt(np.mean(np.square(probability - observed))),
                fmt(cover_log),
            ]
        )
    text += [
        table(
            [
                "Arm",
                "IS RPS",
                "OOS RPS",
                "OOS RPS 95%",
                "RPS gap",
                "IS log",
                "OOS log",
                "OOS log 95%",
                "OOS zero mass",
                "Cover Brier",
                "Cover log loss",
            ],
            arm_rows,
        ),
        "",
        "## Fold populations and metrics",
        "",
        "**Measured:** chronological LOSO mass fitting uses preceding seasons in the production "
        "five-season window. No held-out-season outcomes enter OOS mass fits. IS adds the whole "
        "held-out season to that same pool and is deliberately optimistic. Base model forecasts "
        "remain frozen walk-forward forecasts; base models were not refitted LOSO.",
        "",
        table(["Held-out season", "Games", "OOS pool", "IS pool", "Latest OOS outcome"], fold_rows),
        "",
    ]
    season_rows, coefficient_rows = [], []
    for (season, arm), group in scored.groupby(["season", "arm"], sort=True):
        heldout, fitted = group.loc[group.scheme.eq("OOS")], group.loc[group.scheme.eq("IS")]
        season_rows.append(
            [
                season,
                arm,
                len(heldout),
                fmt(fitted.rps.mean()),
                fmt(heldout.rps.mean()),
                fmt(heldout.rps.mean() - fitted.rps.mean()),
                fmt(fitted.log_score.mean()),
                fmt(heldout.log_score.mean()),
                int(np.isinf(heldout.log_score).sum()),
            ]
        )
        for scheme, sample in (("OOS", heldout), ("IS", fitted)):
            coefficient_rows.append(
                [
                    season,
                    arm,
                    scheme,
                    fmt(sample.coefficient.min()),
                    fmt(sample.coefficient.median()),
                    fmt(sample.coefficient.max()),
                    fmt(sample.band.max(), 1),
                    int(sample.band_rows.min()),
                ]
            )
    text += [
        table(
            [
                "Season",
                "Arm",
                "Games",
                "IS RPS",
                "OOS RPS",
                "Gap",
                "IS log",
                "OOS log",
                "Zero mass",
            ],
            season_rows,
        ),
        "",
        "## Coefficients and their stability",
        "",
        "**Measured:** market_residual coefficient is minus the selected training residual mean, "
        "recentering its PMF at the market line. Fractional shifts split mass between adjacent "
        "integers. Lattice coefficient is exponential tilt theta using production tilted_atoms "
        "and the frozen model centre. These coefficients vary by game; the table gives each fold's "
        "full range and median. Individual values are in the prediction appendix. No logistic or "
        "combined-pick coefficients apply because this unit fits no side-selection model.",
        "",
        table(
            ["Season", "Arm", "Read", "Min", "Median", "Max", "Max band", "Min band games"],
            coefficient_rows,
        ),
        "",
        "## Predicted-tail calibration",
        "",
        "**Measured:** five fixed bands per arm, integer thresholds -70..70, equal total weight "
        "per game. Counts are game-equivalent weights, not independent observations. Gap intervals "
        "resample the same season blocks. No band is selected as a finding.",
        "",
    ]
    calibration_rows = []
    for arm_index, arm in enumerate(ARMS):
        for band in range(5):
            blocks = calibration[:, arm_index, band]
            total, samples = blocks.sum(axis=0), blocks[draws].sum(axis=1)
            differences = (samples[:, 2] - samples[:, 1]) / samples[:, 0]
            calibration_rows.append(
                [
                    arm,
                    f"{TAIL_EDGES[band]:.1f}-{TAIL_EDGES[band + 1]:.1f}",
                    fmt(total[0], 2),
                    fmt(total[1] / total[0]),
                    fmt(total[2] / total[0]),
                    interval_text(differences),
                ]
            )
    text += [
        table(
            [
                "Arm",
                "Predicted tail band",
                "Game-equivalent count",
                "Predicted",
                "Observed",
                "Gap 95%",
            ],
            calibration_rows,
        ),
        "",
        "## Scope, provenance and next action",
        "",
        "**Read:** ROADMAP.md LEAD-66 declares three primary looks. **Measured:** three arms and "
        "three paired contrasts plus 15 fixed descriptive calibration cells: 18 reported looks "
        "including diagnostics. No parameter search or additional arm was run.",
        "",
        f"**Measured:** secondary 2009-2019 inventory lacks seasons "
        f"{', '.join(map(str, inventory['missing_secondary_seasons']))}; "
        "matching predictions begin "
        f"in {inventory['archived_first_season']}. The secondary metric is unavailable, not zero. "
        "No partial 2018-2019 substitute or full-history rebuild was run.",
        "",
        f"**Measured:** maximum archived cover/push/loss reconstruction discrepancy "
        f"{inventory['reconstruction_max_error']:.3g}; recovered residual location agrees within "
        "each week. Source feature hashes match; "
        "each frozen model training date precedes its game. "
        "Inputs were hashed before and after evaluation.",
        "",
        "**Read:** served_lattice retains production market-line conditioning; "
        "model_centred_lattice "
        "uses the existing conditioning_line mechanism at the same recovered model centre. "
        "OOS pools "
        "omit the whole target season, so this is a chronological LOSO mapping reconstruction, not "
        "a replay of the production in-season reference pool. "
        "Historical reference lines use archived openers when available and feature-table "
        "spreads otherwise, as production prior_pool does; all target grades use the opener.",
        "",
        "**Inferred:** results concern PMF mapping conditional on frozen forecasts. They establish "
        "no new fitted pick model, profitable edge or serving change. "
        "AGENTS.md requires one fitted "
        "calibrated probability to choose the served side; no serving decision changes here. "
        "The served-versus-market contrast remains unresolved_below_power. "
        "Both model-centred contrasts have wholly negative 95% intervals; their proposed "
        "registry classification is refuted_mechanism with closing ground wrong_sign_resolved. "
        "These are pending commands for the orchestrator, not recorded closures. "
        "The LEAD-66 research programme itself remains open.",
        "",
        "**Read:** weak_signals.py EFFECT_UNITS does not include rps_improvement, required by the "
        "roadmap. Exact commands in the lane remain unexecuted and parser-blocked until the "
        "orchestrator resolves that schema. "
        "Do not substitute accuracy, Brier, MAE or log-loss units.",
        "",
        f"Prediction-level output: {PREDICTIONS.as_posix()}. "
        f"Source model: {inventory['source_model_id']}. "
        f"Fixed band {BAND_HALF_WIDTH}; minimum {MIN_BAND_GAMES}; "
        f"step {BAND_STEP}; cap {MAX_BAND}.",
        "",
        table(["Input", "SHA256"], [[path, value] for path, value in hashes.items()]),
        "",
        "## Pending record payloads",
        "",
        table(
            ["Signal", "Estimate", "CI lower", "CI upper", "probability_positive", "Blocks"],
            [
                [
                    f"lead66_{ARMS[candidate]}_vs_{ARMS[baseline]}",
                    fmt(row["effect"], 10),
                    fmt(row["interval"][0], 10),
                    fmt(row["interval"][1], 10),
                    fmt(row["p"], 10),
                    6,
                ]
                for (baseline, candidate), row in zip(CONTRASTS, contrasts, strict=True)
            ],
        ),
        "",
    ]
    REPORT.write_text("\n".join(text), encoding="utf-8")
    for row in contrasts:
        print(
            f"{row['name']}: RPS gain={fmt(row['effect'])}, "
            f"95%=[{fmt(row['interval'][0])}, {fmt(row['interval'][1])}], "
            f"probability_positive={fmt(row['p'])}",
            flush=True,
        )


def main():
    games, pool, hashes, inventory = load_inputs()
    scored, fold_rows, calibration = evaluate(games, pool)
    if any(digest(Path(path)) != expected for path, expected in hashes.items()):
        raise ValueError("Pinned source changed during evaluation; discard the run")
    write_predictions(scored)
    write_report(games, scored, fold_rows, calibration, hashes, inventory)
    print(f"Saved {REPORT} and {PREDICTIONS}; no registry or served files changed", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
