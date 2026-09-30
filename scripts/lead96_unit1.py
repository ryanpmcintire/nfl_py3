from __future__ import annotations

import json
from pathlib import Path

import lead83_unit2 as shared
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit, logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability_fit import FIT_ITERATIONS, FIT_RIDGE

OUTPUT = Path("tests/scratch/codex/lead96_unit1")
REPORT = Path("docs/lead96_unit1.md")
LANE = Path("docs/lanes/lead96.md")
POPULATION = Path("artifacts/extended_fit_population/20260923T205910Z/population.parquet")
OLD_OPENERS = Path("artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet")
ARMS = ("transport", "four_term", "model_only", "market", "elo")
METRICS = shared.METRICS
LABELS = (*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:]))
CAP = 10.0
LOOKS = 497
CLASSIFIER_TERMS = ("model_logit", "composition_flag_sum", "abs_opener")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    previous = json.loads(
        Path("tests/scratch/codex/lead83_unit2/summary.json").read_text(encoding="utf-8")
    )
    for filename, expected in previous["lineage"]["inputs"].items():
        if shared.digest(Path(filename)) != expected:
            raise ValueError(f"Certified upstream input changed: {filename}")
    shared.LANE = LANE
    frame, cached, lineage = shared.load()
    population = pd.read_parquet(POPULATION)
    old = population.loc[population.season.lt(2020)].copy()
    openers = pd.read_parquet(OLD_OPENERS, columns=["game_id", "proxy_open_home_spread"])
    old = old.merge(openers, on="game_id", how="left", validate="one_to_one")
    if len(old) != 2231 or old.proxy_open_home_spread.isna().any():
        raise ValueError("Older rows or their opener values are missing")
    old["abs_opener"] = old.proxy_open_home_spread.abs()
    old["original_move"] = old.market_move_toward_home
    old["available"] = old.market_move_available
    if old.original_move.ne(0).any() or old.available.ne(0).any():
        raise ValueError("Older rows must carry missing-move semantics")
    lineage["old_rows"] = len(old)
    lineage["implementation"] = {
        str(path): shared.digest(path) for path in (Path(__file__), Path("scripts/lead83_unit2.py"))
    }
    lineage["inputs"][str(POPULATION)] = shared.digest(POPULATION)
    lineage["inputs"][str(OLD_OPENERS)] = shared.digest(OLD_OPENERS)
    print(
        f"inventory passed: {len(frame)} games including pushes; {len(old)} older rows",
        flush=True,
    )
    return frame, cached, old, lineage


def weighted_logit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(FIT_ITERATIONS):
        p = expit(np.clip(x @ beta, -35.0, 35.0))
        curvature = weights * np.clip(p * (1.0 - p), 1e-6, None)
        gradient = x.T @ (weights * (y - p)) - FIT_RIDGE * beta
        hessian = (x.T * curvature) @ x + FIT_RIDGE * np.eye(x.shape[1])
        beta = beta + np.linalg.solve(hessian, gradient)
    return beta


def transport_weights(new: pd.DataFrame, old: pd.DataFrame) -> tuple[np.ndarray, dict]:
    x = np.vstack(
        [
            old.loc[:, list(CLASSIFIER_TERMS)].to_numpy(float),
            new.loc[:, list(CLASSIFIER_TERMS)].to_numpy(float),
        ]
    )
    y = np.r_[np.ones(len(old)), np.zeros(len(new))]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta = weighted_logit(design, y, np.ones(len(y)))
    score = expit(np.clip(design @ beta, -35.0, 35.0))
    p = score[: len(old)]
    raw = (1.0 - p) / p * (len(old) / len(new))
    weights = np.minimum(raw, CAP)
    ranks = pd.Series(score).rank().to_numpy()[: len(old)]
    auc = float((ranks.sum() - len(old) * (len(old) + 1) / 2) / (len(old) * len(new)))
    everyone = np.r_[weights, np.ones(len(new))]
    detail = {
        "classifier_standardized_coefficients": dict(
            zip(("intercept", *CLASSIFIER_TERMS), beta.tolist(), strict=True)
        ),
        "classifier_auc_in_sample": auc,
        "old_rows": len(old),
        "new_rows": len(new),
        "weight_mean": float(weights.mean()),
        "weight_median": float(np.median(weights)),
        "weight_max_uncapped": float(raw.max()),
        "rows_at_cap": int((raw >= CAP).sum()),
        "weight_sum_old": float(weights.sum()),
        "effective_independent_old": float(weights.sum() ** 2 / (weights**2).sum()),
        "effective_independent_all": float(everyone.sum() ** 2 / (everyone**2).sum()),
    }
    return weights, detail


def fit_transport(
    panel: pd.DataFrame, training: np.ndarray, old: pd.DataFrame
) -> tuple[np.ndarray, dict, dict]:
    new = panel.loc[training].copy()
    new["abs_opener"] = new.spread_line.abs()
    weights, detail = transport_weights(new, old)
    terms = list(shared.TERMS)
    train_x = np.vstack([old.loc[:, terms].to_numpy(float), new.loc[:, terms].to_numpy(float)])
    train_y = np.r_[old.home_covered.to_numpy(float), new.home_covered.to_numpy(float)]
    train_w = np.r_[weights, np.ones(len(new))]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack(
        [np.ones(len(panel)), (panel.loc[:, terms].to_numpy(float) - mean) / scale]
    )
    beta = weighted_logit(
        np.column_stack([np.ones(len(train_x)), (train_x - mean) / scale]), train_y, train_w
    )
    natural = beta[1:] / scale
    coefficients = dict(zip(shared.TERMS, natural.tolist(), strict=True))
    coefficients["intercept"] = float(beta[0] - mean @ natural)
    return expit(np.clip(design @ beta, -30, 30)), coefficients, detail


def replay(
    frame: pd.DataFrame, cached: pd.DataFrame, old: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    details = []
    for outer in (2023, 2024, 2025):
        roles = cached.loc[cached.outer_season.eq(outer), ["game_id", "role"]]
        panel = (
            frame.merge(roles, on="game_id", validate="one_to_one")
            .sort_values(["season", "week", "game_id"])
            .reset_index(drop=True)
        )
        training = panel.role.eq("fit").to_numpy()
        train = panel.loc[training].copy()
        if set(train.season) != set(range(2020, outer - 2)):
            raise ValueError("Upstream training cutoff mismatch")
        if old.season.max() >= train.season.min():
            raise ValueError("Older rows do not precede the fit seasons")
        for role, year in (("tune", outer - 2), ("calibrate", outer - 1), ("outer", outer)):
            if set(panel.loc[panel.role.eq(role), "season"]) != {year}:
                raise ValueError("Selection/calibration/test roles overlap or are missing")
        model = shared.fit_margin_model(
            train,
            target="market_residual",
            model_name="ridge",
            feature_profile="weak_stack",
            ridge_alpha=10.0,
        )
        forecast = model.predict(panel, probability_method="gaussian_median")
        point = forecast.predicted_margin.to_numpy() + shared.residual_location(
            model.residuals, "gaussian_median"
        )
        offset_rows = train.copy()
        offset_rows["point_incumbent"] = point[training]
        offsets = shared.fit_home_side_offsets(offset_rows)
        point += offsets.offset_for(panel.spread_line).to_numpy(float)
        elo_x = np.column_stack([np.ones(len(panel)), panel.elo_diff.to_numpy(float)])
        elo_beta = np.linalg.lstsq(
            elo_x[training], panel.loc[training, "result"].to_numpy(float), rcond=None
        )[0]
        points = {
            "model_only": point,
            "market": panel.sunday_line.to_numpy(float),
            "elo": elo_x @ elo_beta,
        }
        masses = {arm: shared.lattice(train, panel, value) for arm, value in points.items()}
        raw = {
            arm: shared.probability(mass, panel.spread_line.to_numpy(float))
            for arm, mass in masses.items()
        }
        panel["model_logit"] = logit(raw["model_only"])
        upstream = panel[["game_id", "season", "week", "role", "spread_line", "model_logit"]].copy()
        upstream["fit_through"] = outer - 3
        upstream["point"] = point
        upstream["training_max_gameday"] = str(pd.to_datetime(train.gameday).max().date())
        pq.write_table(
            pa.Table.from_pandas(upstream, preserve_index=False),
            OUTPUT / f"upstream_{outer}.parquet",
        )
        kept = panel.declared_fit_population.to_numpy()
        panel = panel.loc[kept].reset_index(drop=True)
        panel["home_covered"] = panel.ats_margin.gt(0).astype(float)
        training = panel.role.eq("fit").to_numpy()
        raw = {arm: values[kept] for arm, values in raw.items()}
        masses = {arm: values[kept] for arm, values in masses.items()}
        coefficients = {}
        raw["four_term"], coefficients["four_term"] = shared.fit_probability(
            panel, training, shared.TERMS
        )
        raw["transport"], coefficients["transport"], weighting = fit_transport(panel, training, old)
        masses["four_term"] = masses["model_only"]
        masses["transport"] = masses["model_only"]
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "upstream_games_with_pushes": len(train),
            "training_max_gameday": str(pd.to_datetime(train.gameday).max().date()),
            "roles": panel.role.value_counts().to_dict(),
            "home_side_offsets": offsets.offsets,
            "elo_margin_coefficients": elo_beta.tolist(),
            "weighting": weighting,
            "probability_coefficients": {},
            "calibration": {},
        }
        for arm in ARMS:
            p, calibration = shared.calibrate(raw[arm], panel)
            scores, push = shared.metrics(panel, p, masses[arm])
            detail["calibration"][arm] = calibration
            if arm in coefficients:
                effective = {
                    name: value * calibration["slope"] for name, value in coefficients[arm].items()
                }
                effective["intercept"] += calibration["intercept"]
                detail["probability_coefficients"][arm] = {
                    "fit": coefficients[arm],
                    "final": effective,
                }
            for phase, role in (("IS", "fit"), ("OOS", "outer")):
                selected = panel.role.eq(role).to_numpy()
                result = panel.loc[
                    selected, ["game_id", "season", "week", "result", "spread_line", "home_covered"]
                ].copy()
                result["outer"] = outer
                result["phase"] = phase
                result["arm"] = arm
                result["probability"] = p[selected]
                result["push_probability"] = push[selected]
                for index, metric in enumerate(METRICS):
                    result[metric] = scores[selected, index]
                rows.append(result)
        details.append(detail)
        print(
            (
                f"fold={outer} complete; fit through {outer - 3}; "
                f"outer n={int(panel.role.eq('outer').sum())}; "
                f"old effective n={weighting['effective_independent_old']:.1f}"
            ),
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), details


def endpoints(raw: np.ndarray) -> np.ndarray:
    arms = raw.reshape(5, 4)
    gains = (arms[0] - arms[1:]) * np.array([1.0, -1.0, -1.0, -1.0])
    return np.vstack([arms, gains]).ravel()


def bootstrap(
    frame: pd.DataFrame, arrays: list[np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.column_stack(arrays)
    blocks = pd.DataFrame(values).assign(season=frame.season.to_numpy(), week=frame.week.to_numpy())
    grouped = blocks.groupby(["season", "week"])
    sums = grouped.sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    years = grouped.size().index.get_level_values(0).to_numpy()
    indexes = [np.flatnonzero(years == year) for year in np.unique(years)]
    means = endpoints(values.mean(axis=0))
    samples = np.empty((shared.BOOTSTRAPS, len(means)))
    for draw in range(shared.BOOTSTRAPS):
        sampled_sums = []
        sampled_counts = []
        for season in rng.integers(0, len(indexes), len(indexes)):
            ix = indexes[season]
            chosen = rng.choice(ix, len(ix), replace=True)
            sampled_sums.append(sums[chosen].sum(axis=0))
            sampled_counts.append(counts[chosen].sum())
        samples[draw] = endpoints(np.sum(sampled_sums, axis=0) / np.sum(sampled_counts))
    return means, samples


def cells(means: np.ndarray, draws: np.ndarray) -> dict:
    return {
        label: {
            metric: shared.estimate(means[i * 4 + j], draws[:, i * 4 + j])
            for j, metric in enumerate(METRICS)
        }
        for i, label in enumerate(LABELS)
    }


def summarize(predictions: pd.DataFrame) -> dict:
    panels = {}
    for outer in (2023, 2024, 2025, "pooled"):
        selected = (
            predictions if outer == "pooled" else predictions.loc[predictions.outer.eq(outer)]
        )
        panel = {"phases": {}, "decisive": {}}
        phase_means = {}
        phase_draws = {}
        for phase in ("IS", "OOS"):
            frame = selected.loc[selected.phase.eq(phase)]
            groups = {
                arm: frame.loc[frame.arm.eq(arm)]
                .sort_values(["outer", "game_id"])
                .reset_index(drop=True)
                for arm in ARMS
            }
            reference = groups["transport"]
            if any(
                not g[["outer", "game_id"]].equals(reference[["outer", "game_id"]])
                for g in groups.values()
            ):
                raise ValueError("Unpaired arm populations")
            means, draws = bootstrap(
                reference,
                [groups[arm].loc[:, list(METRICS)].to_numpy(float) for arm in ARMS],
                shared.SEED
                + (0 if outer == "pooled" else int(outer))
                + (100 if phase == "OOS" else 0),
            )
            phase_means[phase], phase_draws[phase] = means, draws
            panel["phases"][phase] = cells(means, draws)
            for arm in ARMS:
                wins = int(groups[arm].accuracy_points.eq(100).sum())
                panel["phases"][phase][arm]["record"] = f"{wins}-{len(reference) - wins}"
                panel["phases"][phase][arm]["games"] = len(reference)
            if phase == "OOS":
                panel["blocks"] = len(reference[["season", "week"]].drop_duplicates())
                for baseline in ARMS[1:]:
                    decisive = reference.probability.ge(0.5).ne(
                        groups[baseline].probability.ge(0.5)
                    )
                    wins = int(reference.loc[decisive, "accuracy_points"].eq(100).sum())
                    n = int(decisive.sum())
                    panel["decisive"][baseline] = {
                        "games": n,
                        "wins": wins,
                        "losses": n - wins,
                        "exact_p": float(binomtest(wins, n).pvalue) if n else 1.0,
                    }
        panel["gaps_OOS_minus_IS"] = cells(
            phase_means["OOS"] - phase_means["IS"], phase_draws["OOS"] - phase_draws["IS"]
        )
        panels[str(outer)] = panel
        print(f"intervals complete: {outer}", flush=True)
    reliability = []
    oos = predictions.loc[predictions.phase.eq("OOS")]
    for arm in ARMS:
        group = oos.loc[oos.arm.eq(arm)]
        bands = np.minimum((group.probability * 5).astype(int), 4)
        for index in range(5):
            cell = group.loc[bands.eq(index)]
            reliability.append(
                {
                    "arm": arm,
                    "band": f"{index / 5:.1f}-{(index + 1) / 5:.1f}",
                    "n": len(cell),
                    "predicted": float(cell.probability.mean()) if len(cell) else None,
                    "observed": float(cell.home_covered.mean()) if len(cell) else None,
                }
            )
    return {"panels": panels, "reliability": reliability}


def record_commands(summary: dict) -> str:
    units = ("accuracy_points", "log_loss_improvement", "brier_improvement", "rps_improvement")
    lines = [
        chr(96) * 3 + "bash",
        "while read -r panel first last games blocks units effect low high pp; do",
        (
            ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record "
            '--name "lead96_unit1_${panel}_${units}" --league nfl \\'
        ),
        (
            '  --description "Older seasons reweighted to look like recent ones versus the'
            ' usual four-term calculation: ${panel}" --source docs/lead96_unit1.md --family '
            "lead96_unit1_497_looks \\"
        ),
        (
            '  --season-start "$first" --season-end "$last" --sample-games "$games"'
            ' --sample-blocks "$blocks" --effect-units "$units" \\'
        ),
        (
            '  --effect="$effect" --interval-low="$low" --interval-high="$high" '
            '--probability-positive "$pp" --classification unresolved_below_power \\'
        ),
        (
            '  --classification-evidence "Three held-out seasons; no mechanism '
            'closure or serving claim. All 497 declared looks reported." \\'
        ),
        (
            '  --plain-summary "This lets older seasons that resemble today count in the pick '
            "calculation and older ones that do not count less. The held-out comparison "
            'remains a research result; pool picks have not changed."'
        ),
        "done <<'CELLS'",
    ]
    for label, panel in summary["panels"].items():
        first, last = (2023, 2025) if label == "pooled" else (int(label), int(label))
        oos = panel["phases"]["OOS"]
        for metric, unit in zip(METRICS, units, strict=True):
            cell = oos["gain_vs_four_term"][metric]
            values = " ".join(
                f"{cell[key]:.12g}" for key in ("estimate", "low", "high", "probability_positive")
            )
            lines.append(
                f"{label} {first} {last} {oos['transport']['games']} {panel['blocks']} "
                f"{unit} {values}"
            )
    return "\n".join([*lines, "CELLS", chr(96) * 3])


def report(summary: dict) -> None:
    lines = [
        "# LEAD-96 unit 1: transport-weighted older training games",
        "",
        "## Decisive games first",
        "",
        (
            "**Measured:** candidate record only where its selected side differs "
            "from each comparator; exact two-sided fair-coin null."
        ),
        "",
        shared.table(
            ["Outer", "Comparator", "Decisive games", "Candidate W-L", "Exact p"],
            [
                [
                    outer,
                    arm,
                    cell["games"],
                    f"{cell['wins']}-{cell['losses']}",
                    f"{cell['exact_p']:.6f}",
                ]
                for outer, panel in summary["panels"].items()
                for arm, cell in panel["decisive"].items()
            ],
        ),
        "",
        "## Fixed protocol and interpretation",
        "",
        (
            "The declaration below was saved before outcomes; its immutable copy is"
            " tests/scratch/codex/lead96_unit1/protocol.md."
        ),
        "",
        summary["protocol"],
        "",
        (
            "**Measured:** all upstream margin/offset/Elo models and discrete "
            "lattices refit through Y-3 as in LEAD-91. Older rows are the 2011-2019 "
            "walk-forward discrete-read population rows with the SBR-proxy opener; "
            "every one precedes each fit season. IS is optimistic, with repeated "
            "training rows across pooled folds. All five arms receive the same "
            "separate slope selection and intercept calibration. Calibrated "
            "cover/loss mass retains the discrete lattice's push mass and alone "
            "selects the side."
        ),
        "",
        (
            f"**Measured:** {summary['lineage']['population']} source-complete "
            f"games including pushes; {summary['lineage']['nonpush']} conditional-cover "
            f"rows; {summary['lineage']['old_rows']} older rows; {LOOKS} looks, 10,000 "
            f"paired season/week-block resamples. Gap intervals subtract independently "
            f"resampled IS from OOS. Intervals condition on fitted models and omit "
            f"refit uncertainty."
        ),
        "",
        (
            "**Inferred:** unresolved_below_power pending orchestrator recording. "
            "Three outer seasons and retrospective source reuse limit "
            "generalization. Per AGENTS.md Margins/Promotion, a diagnostic gain "
            "cannot promote ATS sides; per its weak-signal rules, zero crossing "
            "closes nothing. No card change or closure."
        ),
        "",
        "## Weights and effective independent games",
        "",
        (
            "**Measured:** training-only domain classifier (old = 1) on model "
            "logit, flag sum and absolute opener; weight = odds of new times the "
            f"old-to-new row ratio, capped at {CAP:g}. Effective games use "
            "(sum w)^2 / sum w^2."
        ),
        "",
        shared.table(
            [
                "Outer",
                "New rows",
                "Old rows",
                "Classifier AUC (in-sample)",
                "Mean weight",
                "Median weight",
                "Max uncapped",
                "At cap",
                "Sum weight",
                "Effective old",
                "Effective all",
            ],
            [
                [
                    fold["outer"],
                    fold["weighting"]["new_rows"],
                    fold["weighting"]["old_rows"],
                    f"{fold['weighting']['classifier_auc_in_sample']:.4f}",
                    f"{fold['weighting']['weight_mean']:.4f}",
                    f"{fold['weighting']['weight_median']:.4f}",
                    f"{fold['weighting']['weight_max_uncapped']:.3f}",
                    fold["weighting"]["rows_at_cap"],
                    f"{fold['weighting']['weight_sum_old']:.1f}",
                    f"{fold['weighting']['effective_independent_old']:.1f}",
                    f"{fold['weighting']['effective_independent_all']:.1f}",
                ]
                for fold in summary["folds"]
            ],
        ),
        "",
        shared.table(
            ["Outer", "Intercept", *CLASSIFIER_TERMS],
            [
                [
                    fold["outer"],
                    *[
                        f"{fold['weighting']['classifier_standardized_coefficients'][k]:.5f}"
                        for k in ("intercept", *CLASSIFIER_TERMS)
                    ],
                ]
                for fold in summary["folds"]
            ],
        ),
        "",
        "## Fold coefficients and stability",
        "",
        (
            "**Measured:** natural coefficients below, before and after "
            "slope/intercept calibration. The base fit has a constant availability "
            "term (zero coefficient); the transport fit identifies it from the "
            "older rows."
        ),
        "",
    ]
    coefficient_rows = []
    for fold in summary["folds"]:
        for arm, phases in fold["probability_coefficients"].items():
            for phase, coef in phases.items():
                coefficient_rows.append(
                    [
                        fold["outer"],
                        arm,
                        phase,
                        *[f"{coef[term]:.8f}" for term in ("intercept", *shared.TERMS)],
                    ]
                )
    lines += [
        shared.table(["Outer", "Arm", "Stage", "Intercept", *shared.TERMS], coefficient_rows),
        "",
        shared.table(
            ["Outer", "Arm", "Selection slope", "Calibration intercept"],
            [
                [fold["outer"], arm, f"{cal['slope']:.8f}", f"{cal['intercept']:.8f}"]
                for fold in summary["folds"]
                for arm, cal in fold["calibration"].items()
            ],
        ),
        "",
        "## IS, OOS and gaps",
        "",
        (
            "**Measured:** each cell is estimate [95% interval]; "
            "probability_positive. Accuracy is percentage points; other endpoints "
            "are losses. Gain rows orient all endpoints so positive favors the "
            "candidate. Gaps are OOS minus IS, also for gain rows. Raw loss "
            "probability_positive means loss above zero, not improvement."
        ),
    ]
    for outer, panel in summary["panels"].items():
        rows = []
        for phase, cells_ in (*panel["phases"].items(), ("gap", panel["gaps_OOS_minus_IS"])):
            for label, measures in cells_.items():
                rows.append(
                    [
                        phase,
                        label,
                        measures.get("record", "—"),
                        *[
                            (
                                f"{shared.number(measures[metric])}; "
                                f"{measures[metric]['probability_positive']:.5f}"
                            )
                            for metric in METRICS
                        ],
                    ]
                )
        lines += [
            "",
            f"### {outer}",
            "",
            shared.table(["Phase", "Arm/contrast", "W-L", *METRICS], rows),
        ]
    lines += [
        "",
        "## Five equal-width reliability bands",
        "",
        (
            "**Measured:** home-cover probabilities conditional on no push, outer "
            "rows only; empty cells remain empty."
        ),
        "",
        shared.table(
            ["Arm", "Band", "Games", "Mean probability", "Home cover rate"],
            [
                [
                    row["arm"],
                    row["band"],
                    row["n"],
                    "—" if row["predicted"] is None else f"{row['predicted']:.6f}",
                    "—" if row["observed"] is None else f"{row['observed']:.6f}",
                ]
                for row in summary["reliability"]
            ],
        ),
        "",
        "## Reproduction and saved rows",
        "",
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead96_unit1.py",
        "",
        (
            "Prediction-level rows, upstream cutoffs, complete coefficients, source"
            " hashes and all intervals: tests/scratch/codex/lead96_unit1/. Serial "
            "registry commands are in docs/lanes/lead96.md and were not run by this"
            " worker."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    pooled = summary["panels"]["pooled"]
    oos = pooled["phases"]["OOS"]
    brier = oos["gain_vs_four_term"]["brier"]
    decisive = pooled["decisive"]["four_term"]
    text = LANE.read_text(encoding="utf-8")
    goal_and_protocol = text.split("## State\n", 1)[0] + "## State\n"
    protocol = (
        "## Protocol (fixed before outcomes)\n"
        + text.split("## Protocol (fixed before outcomes)\n", 1)[1].split("\n## Tried", 1)[0]
    )
    lane = [
        goal_and_protocol.rstrip("\n").rsplit("## State", 1)[0].rstrip("\n"),
        "",
        "## State",
        (
            f"**Measured:** 497-look replay complete: {oos['transport']['games']} "
            f"outer games; candidate {oos['transport']['record']}, "
            f"base {oos['four_term']['record']}; "
            f"decisive {decisive['wins']}-{decisive['losses']}. Brier improvement "
            f"{shared.number(brier)}, probability_positive={brier['probability_positive']:.5f}. "
            "**Inferred:** unresolved_below_power; no serving change."
        ),
        "",
        protocol.rstrip("\n"),
        "",
        "## Tried",
        (
            "Protocol saved before outcomes (copy in tests/scratch/codex/"
            "lead96_unit1/protocol.md). Ran `.tools/uv.exe run --no-sync --no-cache "
            "python scripts/lead96_unit1.py` once. Report docs/lead96_unit1.md; rows "
            "in tests/scratch/codex/lead96_unit1/."
        ),
        "",
        "## Record commands",
        (
            "Orchestrator runs these 12 candidate-versus-four-term OOS cells "
            "serially; worker did not run them."
        ),
        "",
        record_commands(summary),
        "",
        "## Next",
        "Orchestrator reviews the report and runs the records serially.",
        "",
        "## Open",
        (
            "Three outer seasons; optimistic IS; retrospective archive reuse; no "
            "pool-noon fidelity claim. No closure or promotion."
        ),
        "",
    ]
    LANE.write_text("\n".join(lane), encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise ValueError("A scored replay already exists; do not silently rescore")
    protocol = LANE.read_text(encoding="utf-8")
    (OUTPUT / "protocol.md").write_text(protocol, encoding="utf-8")
    with threadpool_limits(limits=1):
        frame, cached, old, lineage = load()
        predictions, folds = replay(frame, cached, old)
        pq.write_table(
            pa.Table.from_pandas(predictions, preserve_index=False), OUTPUT / "predictions.parquet"
        )
        summary = summarize(predictions)
        declaration = protocol.split("## Protocol (fixed before outcomes)\n", 1)[1].split(
            "\n## Tried", 1
        )[0]
        summary.update(
            {
                "protocol": declaration,
                "lineage": lineage,
                "folds": folds,
                "looks": LOOKS,
                "bootstrap_samples": shared.BOOTSTRAPS,
                "seed": shared.SEED,
            }
        )
        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        report(summary)
    pooled = summary["panels"]["pooled"]["phases"]["OOS"]
    print(
        json.dumps(
            {
                "status": "complete",
                "games": pooled["transport"]["games"],
                "record": pooled["transport"]["record"],
                "candidate_vs_four_term": pooled["gain_vs_four_term"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
