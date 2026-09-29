from __future__ import annotations

import hashlib
import json
from pathlib import Path

import four_term_probability_eval as four
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

SOURCE = Path("artifacts/four_term_probability/20260914T222345Z/per_game.csv")
REPORT = Path("docs/lead68_unit1.md")
PREDICTIONS = Path("docs/lead68_predictions.md")
PHASES = (("weeks_1_4", 1, 4), ("weeks_5_9", 5, 9), ("weeks_10_18", 10, 18))
BASE_FEATURES = four.M5_FEATURES
PHASE_FEATURES = [name for name, _, _ in PHASES] + BASE_FEATURES[1:]
SEASONS = tuple(range(2020, 2026))


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        ]
    )


def losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.column_stack(
        (
            -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)),
            (p - y) ** 2,
            100 * ((p >= 0.5) == y),
        )
    )


def interval(values: np.ndarray) -> str:
    low, high = np.quantile(values, [0.025, 0.975])
    return f"[{low:.6f}, {high:.6f}]"


def fit_arm(df: pd.DataFrame, features: list[str]) -> dict:
    held_predictions = np.full(len(df), np.nan)
    train_losses = []
    folds = []
    coefficients = []
    for held in SEASONS:
        train = df.loc[df.season.ne(held)]
        test = df.loc[df.season.eq(held)]
        beta, means, stds, hessian = four.jpm.standardize_fit(train, features, "home_covered", 1e-3)
        train_p = four.jpm.predict_p(train, features, beta, means, stds)
        test_p = four.jpm.predict_p(test, features, beta, means, stds)
        held_predictions[test.index] = test_p
        train_loss = losses(train.home_covered.to_numpy(dtype=float), train_p)
        test_loss = losses(test.home_covered.to_numpy(dtype=float), test_p)
        train_losses.append(train_loss)
        transform = np.eye(len(features) + 1)
        for index, feature in enumerate(features, start=1):
            transform[index, index] = 1 / stds[feature]
            transform[0, index] = -means[feature] / stds[feature]
        natural = transform @ beta
        covariance = transform @ np.linalg.inv(hessian) @ transform.T
        errors = np.sqrt(np.maximum(np.diag(covariance), 0))
        for feature, value, error in zip(["intercept", *features], natural, errors, strict=True):
            coefficients.append(
                [
                    held,
                    feature,
                    f"{value:.6f}",
                    f"[{value - 1.959963985 * error:.6f}, {value + 1.959963985 * error:.6f}]",
                    f"{norm.cdf(value / error):.6f}",
                ]
            )
        folds.append((held, len(train), len(test), train_loss.mean(0), test_loss.mean(0)))
    if not np.isfinite(held_predictions).all():
        raise ValueError("Every game must have exactly one held-out prediction")
    return {
        "p": held_predictions,
        "losses": losses(df.home_covered.to_numpy(dtype=float), held_predictions),
        "train": np.concatenate(train_losses).mean(0),
        "folds": folds,
        "coefficients": coefficients,
    }


def run() -> None:
    df = pd.read_csv(SOURCE)
    required = ["game_id", "season", "week", "home_covered", "x_p_raw", *BASE_FEATURES[1:]]
    if df[required].isna().any().any():
        raise ValueError("Required population fields have missing values")
    if len(df) != 1503 or df.game_id.duplicated().any():
        raise ValueError("The predeclared unique 1,503-game population is required")
    if tuple(sorted(df.season.unique())) != SEASONS or not df.week.between(1, 18).all():
        raise ValueError("The predeclared 2020-2025 regular-season phases are required")
    if not df.home_covered.isin([0, 1]).all() or not df.x_p_raw.between(0, 1).all():
        raise ValueError("Invalid opener target or raw home-cover probability")
    clipped = df.x_p_raw.clip(1e-6, 1 - 1e-6)
    df["x0_model_logit"] = np.log(clipped / (1 - clipped))
    for name, first, last in PHASES:
        df[name] = df.x0_model_logit * df.week.between(first, last)
    if not np.isfinite(df[[*BASE_FEATURES, *PHASE_FEATURES]].to_numpy()).all():
        raise ValueError("All model terms must be finite")
    baseline = fit_arm(df, BASE_FEATURES)
    candidate = fit_arm(df, PHASE_FEATURES)
    reproduction_error = float(np.max(np.abs(baseline["p"] - df.m5_oos_p)))
    if reproduction_error > 1e-10:
        raise ValueError(f"Four-term cached probabilities did not reproduce: {reproduction_error}")
    y = df.home_covered.to_numpy(dtype=float)
    season_index = df.season.to_numpy(dtype=int) - SEASONS[0]
    counts = np.bincount(season_index, minlength=6)
    sampled = np.random.default_rng(68).integers(0, 6, size=(2000, 6))
    denominator = counts[sampled].sum(1)
    predictions = {
        "four_term": baseline["p"],
        "phase_slopes": candidate["p"],
        "raw_model": df.x_p_raw.to_numpy(dtype=float),
        "market_even": np.full(len(df), 0.5),
    }
    arm_losses = {name: losses(y, p) for name, p in predictions.items()}
    bootstrap = {}
    for name, values in arm_losses.items():
        sums = np.vstack([values[season_index == index].sum(0) for index in range(6)])
        bootstrap[name] = sums[sampled].sum(1) / denominator[:, None]
    signs = np.array([1.0, 1.0, -1.0])
    effect = (baseline["losses"] - candidate["losses"]).mean(0) * signs
    draws = (bootstrap["four_term"] - bootstrap["phase_slopes"]) * signs
    probability_positive = (draws > 0).mean(0) + 0.5 * (draws == 0).mean(0)
    different = (baseline["p"] >= 0.5) != (candidate["p"] >= 0.5)
    decisive_n = int(different.sum())
    decisive_wins = int(((candidate["p"][different] >= 0.5) == y[different]).sum())
    wins = int(((candidate["p"] >= 0.5) == y).sum())
    baseline_wins = int(((baseline["p"] >= 0.5) == y).sum())
    rows = []
    for name, values in arm_losses.items():
        average = values.mean(0)
        record, accuracy = "n/a (no directional pick)", "n/a"
        if name != "market_even":
            arm_wins = int(((predictions[name] >= 0.5) == y).sum())
            record = f"{arm_wins}-{len(df) - arm_wins}"
            accuracy = f"{average[2]:.6f} {interval(bootstrap[name][:, 2])}"
        rows.append(
            [
                name,
                f"{average[0]:.6f} {interval(bootstrap[name][:, 0])}",
                f"{average[1]:.6f} {interval(bootstrap[name][:, 1])}",
                accuracy,
                record,
            ]
        )
    sections = [
        "# LEAD-68 unit 1: model slope by weeks of information",
        "**Measured:** `.tools/uv.exe run --no-sync python scripts/lead68_unit1.py`.",
        "## Population and protocol",
        f"**Read:** frozen source `{SOURCE.as_posix()}`, SHA-256 "
        f"`{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}`. "
        "**Measured:** 1,503 unique non-push opener games, seasons 2020-2025. "
        "The baseline has model logit, flag_sum, move, move_available and an intercept. "
        "The candidate replaces only model logit with logit multiplied by the fixed "
        "weeks 1-4, 5-9 and 10-18 indicators. All coefficients and standardization fit "
        "on the other five seasons, fixed L2=0.001, 50 Newton iterations; no tuning. "
        "Four declared looks: three phase slopes and the joint fit. Reference "
        "baselines and descriptive reliability cells do not select new candidates.",
        "**Read:** protocol frozen before outcome access in `docs/lanes/lead68.md`. "
        "**Inferred:** requested retrospective LOSO includes later seasons in "
        "earlier holdout training; this is not a prospective rolling evaluation. "
        "Cached inputs inherit the original evaluator's timing and model limitations; "
        "this unit does not re-audit feature provenance or change served output.",
        f"**Measured:** four-term reproduction maximum absolute probability error "
        f"{reproduction_error:.12g}.",
        "## Decisive record and uncertainty",
        f"**Measured:** {decisive_n} games change side: candidate "
        f"{decisive_wins}-{decisive_n - decisive_wins}, baseline "
        f"{decisive_n - decisive_wins}-{decisive_wins}. All games: candidate "
        f"{wins}-{len(df) - wins}, baseline {baseline_wins}-{len(df) - baseline_wins}. "
        "One fitted probability selects each side at 0.5; no isolated flip rule.",
        "**Measured:** 2,000 paired whole-season bootstrap draws, seed 68, six seasons "
        "sampled with replacement, 95% percentile intervals. probability_positive "
        "assigns half weight to exact zero draws. Fixed LOSO predictions are not "
        "refitted in the bootstrap; six clusters limit precision. Positive effects "
        "favor the candidate. No best phase or reliability bin is selected.",
        table(
            ["Metric", "Candidate improvement", "95% interval", "probability_positive"],
            [
                [
                    label,
                    f"{effect[index]:.6f}",
                    interval(draws[:, index]),
                    f"{probability_positive[index]:.6f}",
                ]
                for index, label in enumerate(["log loss", "Brier", "accuracy points"])
            ],
        ),
        "## Held-out metrics",
        "**Measured:** metric estimates with season-bootstrap 95% intervals. "
        "Market-even is the explicit uninformative 0.5 opener-cover baseline with "
        "no directional pick; raw_model is the unchanged model-only reference.",
        table(["Arm", "Log loss", "Brier", "Accuracy %", "W-L"], rows),
        "## Training and held-out gap",
        "**Measured:** training pools 7,515 fit/game evaluations (five appearances "
        "per game); descriptive only. Gap = held-out minus training; accuracy in points.",
        table(
            ["Arm", "Metric", "Training", "Held out", "Gap"],
            [
                [
                    name,
                    metric,
                    f"{fit['train'][index]:.6f}",
                    f"{fit['losses'].mean(0)[index]:.6f}",
                    f"{fit['losses'].mean(0)[index] - fit['train'][index]:.6f}",
                ]
                for name, fit in [("four_term", baseline), ("phase_slopes", candidate)]
                for index, metric in enumerate(["log loss", "Brier", "accuracy %"])
            ],
        ),
        "## Season stability",
        "**Measured:** each metric is training / held-out / gap; accuracy in percent.",
        table(
            ["Arm", "Holdout", "n train/test", "Log loss", "Brier", "Accuracy %"],
            [
                [
                    name,
                    held,
                    f"{n_train}/{n_test}",
                    *(
                        f"{train[i]:.6f} / {test[i]:.6f} / {test[i] - train[i]:.6f}"
                        for i in range(3)
                    ),
                ]
                for name, fit in [("four_term", baseline), ("phase_slopes", candidate)]
                for held, n_train, n_test, train, test in fit["folds"]
            ],
        ),
    ]
    reliability_rows = []
    for name, p in predictions.items():
        bins = np.minimum((p * 5).astype(int), 4)
        for index in range(5):
            mask = bins == index
            reliability_rows.append(
                [
                    name,
                    f"{index / 5:.1f}-{(index + 1) / 5:.1f}",
                    int(mask.sum()),
                    f"{p[mask].mean():.6f}" if mask.any() else "n/a",
                    f"{y[mask].mean():.6f}" if mask.any() else "n/a",
                ]
            )
    sections.extend(
        [
            "## Reliability",
            "**Measured:** fixed evaluator bins, left closed/right open except 1.0. "
            "Five descriptive bins per arm; no cell-based side selection.",
            table(
                ["Arm", "Probability bin", "n", "Mean p(home cover)", "Home-cover rate"],
                reliability_rows,
            ),
            "## Per-fold coefficients",
            "**Measured:** natural units, conditional penalized-Hessian normal 95% "
            "working intervals and normal-approximation probability_positive. These "
            "are not season-cluster intervals or multiplicity-adjusted evidence. Folds "
            "overlap in training. Intercept variance includes the centering transformation.",
        ]
    )
    for name, fit in [("four_term", baseline), ("phase_slopes", candidate)]:
        sections.extend(
            [
                f"### {name}",
                table(
                    [
                        "Holdout",
                        "Term",
                        "Coefficient",
                        "95% working interval",
                        "probability_positive",
                    ],
                    fit["coefficients"],
                ),
            ]
        )
    sections.extend(
        [
            "## Decision and limitations",
            "**Inferred:** retain unresolved_below_power pending the orchestrator's "
            "serial registry recording; no terminal closure or serving change. "
            "AGENTS.md:65-84 requires admissible closing evidence; AGENTS.md:115-120 "
            "separates closure from serving. Three phase slopes form one joint candidate. "
            "Historical accuracy is not a game probability or profitability evidence.",
            "**Read:** this is the historic four-term population, not a newly materialized "
            "active-card forecast. **Measured:** prediction rows preserved in "
            f"`{PREDICTIONS.as_posix()}`, a generated local research artifact; do not "
            "commit that processed-data file. No registry command was run.",
        ]
    )
    REPORT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    prediction_rows = [
        [
            row.game_id,
            row.season,
            row.week,
            int(row.home_covered),
            f"{baseline['p'][index]:.12f}",
            f"{candidate['p'][index]:.12f}",
        ]
        for index, row in enumerate(df.itertuples(index=False))
    ]
    PREDICTIONS.write_text(
        "# LEAD-68 local generated predictions (do not commit)\n\n"
        "**Measured:** each row is scored with its entire season excluded.\n\n"
        + table(
            [
                "game_id",
                "held_out_season",
                "week",
                "home_covered",
                "four_term_home_cover_probability",
                "phase_home_cover_probability",
            ],
            prediction_rows,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"report={REPORT} predictions={PREDICTIONS} games={len(df)} looks=4")
    print(f"candidate={wins}-{len(df) - wins} baseline={baseline_wins}-{len(df) - baseline_wins}")
    print(f"decisive={decisive_wins}-{decisive_n - decisive_wins} n={decisive_n}")
    for index, metric in enumerate(["log_loss", "brier", "accuracy_points"]):
        print(
            json.dumps(
                {
                    "metric": metric,
                    "effect": float(effect[index]),
                    "ci_low": float(np.quantile(draws[:, index], 0.025)),
                    "ci_high": float(np.quantile(draws[:, index], 0.975)),
                    "probability_positive": float(probability_positive[index]),
                }
            )
        )
    print(f"baseline_reproduction_max_abs_error={reproduction_error:.12g}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
