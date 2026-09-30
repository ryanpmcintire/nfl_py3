from __future__ import annotations

import json
from pathlib import Path

import lead83_unit2 as shared
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

OUTPUT = Path("tests/scratch/codex/lead91_unit1")
REPORT = Path("docs/lead91_unit1.md")
LANE = Path("docs/lanes/lead91.md")
ARMS = ("minimax", "four_term", "model_only", "market", "elo")
METRICS = (*shared.METRICS, "worst_season_log_loss")
LABELS = (*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:]))
RIDGE = 0.001
LOOKS = 601


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    previous = json.loads(
        Path("tests/scratch/codex/lead83_unit2/summary.json").read_text(encoding="utf-8")
    )
    for filename, expected in previous["lineage"]["inputs"].items():
        if shared.digest(Path(filename)) != expected:
            raise ValueError(f"Certified upstream input changed: {filename}")
    source = json.loads((shared.ROOT / "summary.json").read_text(encoding="utf-8"))
    for entry in source["source_manifests"]:
        path = Path(entry["manifest"])
        if (
            shared.digest(path) != entry["manifest_sha256"]
            or shared.digest(path.parent / "quotes.parquet") != entry["quotes_sha256"]
        ):
            raise ValueError(f"Certified quote archive changed: {path}")
    shared.LANE = LANE
    frame, cached, lineage = shared.load()
    lineage["verified_quote_manifests"] = len(source["source_manifests"])
    lineage["implementation"] = {
        str(path): shared.digest(path) for path in (Path(__file__), Path("scripts/lead83_unit2.py"))
    }
    print(
        f"inventory passed: {len(frame)} games including pushes; "
        f"{int(frame.declared_fit_population.sum())} conditional-cover rows",
        flush=True,
    )
    return frame, cached, lineage


def fit_minimax(frame: pd.DataFrame, training: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    x = frame.loc[:, list(shared.TERMS)].to_numpy(float)
    mean = x[training].mean(axis=0)
    scale = x[training].std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(frame)), (x - mean) / scale])
    train_x = design[training]
    y = frame.loc[training, "home_covered"].to_numpy(float)
    seasons = frame.loc[training, "season"].to_numpy(int)
    years = np.unique(seasons)
    masks = [seasons == year for year in years]
    penalty = RIDGE / len(y)

    def losses(beta: np.ndarray) -> np.ndarray:
        z = train_x @ beta
        loss = np.logaddexp(0, z) - y * z
        return np.array([loss[mask].mean() for mask in masks])

    def constraint_jacobian(value: np.ndarray) -> np.ndarray:
        error = expit(train_x @ value[:-1]) - y
        return np.array([np.r_[-train_x[mask].T @ error[mask] / mask.sum(), 1.0] for mask in masks])

    fitted = minimize(
        lambda value: value[-1] + 0.5 * penalty * (value[:-1] @ value[:-1]),
        np.r_[np.zeros(train_x.shape[1]), np.log(2.0)],
        jac=lambda value: np.r_[penalty * value[:-1], 1.0],
        constraints={
            "type": "ineq",
            "fun": lambda value: value[-1] - losses(value[:-1]),
            "jac": constraint_jacobian,
        },
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    beta = fitted.x[:-1]
    season_losses = losses(beta)
    violation = float(max(0, season_losses.max() - fitted.x[-1]))
    if not fitted.success or violation > 1e-8 or not np.isfinite(fitted.x).all():
        raise ValueError(f"Minimax optimization failed: {fitted.message}; {violation=}")
    natural = beta[1:] / scale
    coefficients = dict(zip(shared.TERMS, natural.tolist(), strict=True))
    coefficients["intercept"] = float(beta[0] - mean @ natural)
    details = {
        "success": bool(fitted.success),
        "iterations": int(fitted.nit),
        "constraint_violation": violation,
        "ridge": RIDGE,
        "ridge_mean_loss_multiplier": penalty,
        "maximum_training_season_log_loss": float(season_losses.max()),
        "training_season_log_losses": dict(
            zip(map(str, years), season_losses.tolist(), strict=True)
        ),
    }
    return expit(np.clip(design @ beta, -30, 30)), coefficients, details


def replay(frame: pd.DataFrame, cached: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
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
        raw["minimax"], coefficients["minimax"], optimization = fit_minimax(panel, training)
        masses["four_term"] = masses["model_only"]
        masses["minimax"] = masses["model_only"]
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "upstream_games_with_pushes": len(train),
            "training_max_gameday": str(pd.to_datetime(train.gameday).max().date()),
            "roles": panel.role.value_counts().to_dict(),
            "home_side_offsets": offsets.offsets,
            "elo_margin_coefficients": elo_beta.tolist(),
            "optimization": optimization,
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
                for index, metric in enumerate(shared.METRICS):
                    result[metric] = scores[selected, index]
                rows.append(result)
        details.append(detail)
        print(
            (
                f"fold={outer} complete; fit through {outer - 3}; "
                f"outer n={int(panel.role.eq('outer').sum())}"
            ),
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), details


def endpoints(raw: np.ndarray, worst: np.ndarray) -> np.ndarray:
    arms = np.column_stack([raw.reshape(5, 4), worst])
    gains = (arms[0] - arms[1:]) * np.array([1.0, -1.0, -1.0, -1.0, -1.0])
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
    loss_columns = np.arange(5) * 4 + 1
    worst = np.max(
        [sums[ix].sum(axis=0)[loss_columns] / counts[ix].sum() for ix in indexes], axis=0
    )
    means = endpoints(values.mean(axis=0), worst)
    samples = np.empty((shared.BOOTSTRAPS, len(means)))
    for draw in range(shared.BOOTSTRAPS):
        sampled_sums = []
        sampled_counts = []
        for season in rng.integers(0, len(indexes), len(indexes)):
            ix = indexes[season]
            chosen = rng.choice(ix, len(ix), replace=True)
            sampled_sums.append(sums[chosen].sum(axis=0))
            sampled_counts.append(counts[chosen].sum())
        total = np.sum(sampled_sums, axis=0) / np.sum(sampled_counts)
        worst = (np.array(sampled_sums)[:, loss_columns] / np.array(sampled_counts)[:, None]).max(
            axis=0
        )
        samples[draw] = endpoints(total, worst)
    return means, samples


def cells(means: np.ndarray, draws: np.ndarray) -> dict:
    return {
        label: {
            metric: shared.estimate(means[i * 5 + j], draws[:, i * 5 + j])
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
            reference = groups["minimax"]
            if any(
                not g[["outer", "game_id"]].equals(reference[["outer", "game_id"]])
                for g in groups.values()
            ):
                raise ValueError("Unpaired arm populations")
            means, draws = bootstrap(
                reference,
                [groups[arm].loc[:, list(shared.METRICS)].to_numpy(float) for arm in ARMS],
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
    units = (
        "accuracy_points",
        "log_loss_improvement",
        "brier_improvement",
        "rps_improvement",
        "worst_season_log_loss_improvement",
    )
    lines = [
        chr(96) * 3 + "bash",
        "while read -r panel first last games blocks units effect low high pp; do",
        (
            ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record "
            '--name "lead91_unit1_$panel-$units" --league nfl \\'
        ),
        (
            '  --description "Fit for the hardest season versus the usual four-term'
            ' calculation: $panel" --source docs/lead91_unit1.md --family '
            "lead91_unit1_601_looks \\"
        ),
        (
            '  --season-start "$first" --season-end "$last" --sample-games "$games"'
            ' --sample-blocks "$blocks" --effect-units "$units" \\'
        ),
        (
            '  --effect "$effect" --interval-low "$low" --interval-high "$high" '
            '--probability-positive "$pp" --classification unresolved_below_power \\'
        ),
        (
            '  --classification-evidence "Three held-out seasons; no mechanism '
            'closure or serving claim. All 601 declared looks reported." \\'
        ),
        (
            '  --plain-summary "This gives the hardest training season more say in '
            "the pick calculation. The held-out comparison remains a research "
            'result; pool picks have not changed."'
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
                f"{label} {first} {last} {oos['minimax']['games']} {panel['blocks']} "
                f"{unit} {values}"
            )
    return "\n".join([*lines, "CELLS", chr(96) * 3])


def report(summary: dict) -> None:
    lines = [
        "# LEAD-91 unit 1: fit for season stability",
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
            " tests/scratch/codex/lead91_unit1/protocol.md."
        ),
        "",
        summary["protocol"],
        "",
        (
            "**Measured:** all upstream margin/offset/Elo models and discrete "
            "lattices refit through Y-3; no cached weekly model probabilities used."
            " Historical opener replaces closing spread, archived Tuesday totals "
            "replace closing totals. Hashes verify the LEAD-83 inputs and both "
            "quote archives. IS is optimistic, with repeated training rows across "
            "pooled folds. All five arms receive the same separate slope selection "
            "and intercept calibration. Calibrated cover/loss mass retains the "
            "discrete lattice's push mass and alone selects the side."
        ),
        "",
        (
            f"**Measured:** {summary['lineage']['population']} source-complete "
            f"games including pushes; {summary['lineage']['nonpush']} conditional-cover "
            f"rows; {summary['lineage']['missing_tuesday_totals']} missing "
            f"Tuesday totals retained as missing; {LOOKS} looks, 10,000 "
            f"paired season/week-block resamples. Worst-season loss is "
            f"recomputed within each draw; pooled worst-season contrast "
            f"subtracts the candidate maximum from each comparator maximum. "
            f"Gap intervals subtract independently resampled IS from OOS. "
            f"Intervals condition on fitted models and omit full refit "
            f"uncertainty."
        ),
        "",
        (
            "**Inferred:** unresolved_below_power pending orchestrator recording. "
            "Three outer seasons and retrospective source reuse limit "
            "generalization. Per AGENTS.md Margins/Promotion, a diagnostic gain "
            "cannot promote ATS sides; per its weak-signal rules, zero crossing "
            "closes nothing. No card change or closure. The one-season 2023 "
            "training fold makes minimax and pooled objectives mathematically "
            "identical up to optimizer tolerance."
        ),
        "",
        "## Fold coefficients and stability",
        "",
        (
            "**Measured:** natural coefficients below, before and after "
            "slope/intercept calibration. Availability is constant in this "
            "source-complete population, so its standardized fitted coefficient is "
            "zero."
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
        shared.table(
            [
                "Outer",
                "Fit through",
                "Roles",
                "Optimizer iterations",
                "Constraint violation",
                "Training-season losses",
            ],
            [
                [
                    fold["outer"],
                    fold["fit_through"],
                    json.dumps(fold["roles"]),
                    fold["optimization"]["iterations"],
                    f"{fold['optimization']['constraint_violation']:.3g}",
                    json.dumps(fold["optimization"]["training_season_log_losses"]),
                ]
                for fold in summary["folds"]
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
            "probability_positive means loss above zero, not improvement. "
            "Worst-season loss duplicates ordinary log loss for single-season "
            "panels but remains a declared look."
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
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead91_unit1.py",
        "",
        (
            "Prediction-level rows, upstream cutoffs, complete coefficients, source"
            " hashes and all intervals: tests/scratch/codex/lead91_unit1/. Serial "
            "registry commands are in docs/lanes/lead91.md and were not run by this"
            " worker."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    pooled = summary["panels"]["pooled"]
    oos = pooled["phases"]["OOS"]
    brier = oos["gain_vs_four_term"]["brier"]
    decisive = pooled["decisive"]["four_term"]
    lane = [
        "# LEAD-91 — season-stable fitting",
        "",
        "## Goal",
        (
            "Execute the fixed four-term minimax-season challenger; historical "
            "OPENER is the frozen pool-line proxy. Research only."
        ),
        "",
        "## State",
        (
            f"**Measured:** 601-look replay complete: {oos['minimax']['games']} "
            f"outer games; candidate {oos['minimax']['record']}, "
            f"base {oos['four_term']['record']}; "
            f"decisive {decisive['wins']}-{decisive['losses']}. Brier improvement "
            f"{shared.number(brier)}, probability_positive={brier['probability_positive']:.5f}. "
            f"**Inferred:** unresolved_below_power; no serving change."
        ),
        "",
        "## Tried",
        (
            "Protocol saved before outcomes (verbatim in docs/lead91_unit1.md and "
            "scratch protocol.md); fixed ridge .001, four terms, minimax season "
            "mean loss; outer 2023/24/25, fit Y-3, select Y-2, calibrate Y-1. Refit"
            " upstream models, preserve pushes in discrete distributions. Full "
            "IS/OOS/gaps, five arms, coefficients and 601 looks saved in report. "
            "Single-thread real replay; verification pending."
        ),
        "",
        "## Record commands",
        (
            "Orchestrator runs these 20 exact candidate-versus-base OOS expansions "
            "serially; worker did not run them. Other contrasts and all IS/gap "
            "looks remain reported."
        ),
        "",
        record_commands(summary),
        "",
        "## Next",
        (
            "Orchestrator reviews report and executes serial records, then assigns "
            "the next bounded task. Rows/lineage: "
            "tests/scratch/codex/lead91_unit1/."
        ),
        "",
        "## Open",
        (
            "Three outer seasons; optimistic IS; retrospective archive reuse; "
            "missing Tuesday totals; no pool-noon fidelity claim. No closure or "
            "promotion. No unrelated edits, Git mutations or publication."
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
        frame, cached, lineage = load()
        predictions, folds = replay(frame, cached)
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
                "games": pooled["minimax"]["games"],
                "record": pooled["minimax"]["record"],
                "candidate_vs_four_term": pooled["gain_vs_four_term"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
