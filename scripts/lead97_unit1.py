from __future__ import annotations

import json
from pathlib import Path

import lead83_unit2 as shared
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

OUTPUT = Path("tests/scratch/codex/lead97_unit1")
REPORT = Path("docs/lead97_unit1.md")
LANE = Path("docs/lanes/lead97.md")
SERVED = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
ARMS = ("integrated", "four_term", "model_only", "market", "elo")
LABELS = (*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:]))
RIDGE = 0.001
LOOKS = 497
MOVE = list(shared.TERMS).index("original_move") + 1


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    previous = json.loads(
        Path("tests/scratch/codex/lead83_unit2/summary.json").read_text(encoding="utf-8")
    )
    for filename, expected in previous["lineage"]["inputs"].items():
        if shared.digest(Path(filename)) != expected:
            raise ValueError(f"Certified upstream input changed: {filename}")
    source = json.loads((shared.ROOT / "summary.json").read_text(encoding="utf-8-sig"))
    manifests = {Path(entry["manifest"]).parent.name for entry in source["source_manifests"]}
    auxiliary = {"20200908T125500Z", "20200913T162500Z"}
    if not auxiliary <= manifests:
        raise ValueError("Auxiliary 2020 archives are not part of the certified quote set")
    for entry in source["source_manifests"]:
        path = Path(entry["manifest"])
        if (
            shared.digest(path) != entry["manifest_sha256"]
            or shared.digest(path.parent / "quotes.parquet") != entry["quotes_sha256"]
        ):
            raise ValueError(f"Certified quote archive changed: {path}")
    shared.LANE = LANE
    frame, cached, lineage = shared.load()
    served = shared.read(
        SERVED, ["game_id", "market_move_available", "market_move_toward_home"]
    ).rename(
        columns={"market_move_available": "legacy_available", "market_move_toward_home": "legacy"}
    )
    frame = frame.merge(served, on="game_id", how="left", validate="one_to_one")
    scored = frame.declared_fit_population
    if frame.loc[scored, ["legacy_available", "legacy"]].isna().any().any():
        raise ValueError("A scored game lacks the served legacy move mask")
    frame["dated_move"] = frame.original_move
    frame["available"] = frame.legacy_available.fillna(0.0).astype(float)
    frame["original_move"] = frame.legacy.fillna(0.0).astype(float)
    if frame.loc[frame.available.eq(0), "original_move"].ne(0).any():
        raise ValueError("Legacy unavailable rows must carry a zero move")
    both = frame.loc[scored & frame.available.eq(1)]
    lineage["verified_quote_manifests"] = len(source["source_manifests"])
    lineage["auxiliary_archives_in_quote_set"] = sorted(auxiliary)
    lineage["legacy_availability_by_season"] = {
        str(season): [int(group.available.sum()), len(group)]
        for season, group in frame.loc[scored].groupby("season")
    }
    lineage["observed_legacy_vs_dated_move"] = {
        "games": len(both),
        "correlation": float(np.corrcoef(both.original_move, both.dated_move)[0, 1]),
        "legacy_mean": float(both.original_move.mean()),
        "dated_mean": float(both.dated_move.mean()),
        "legacy_std": float(both.original_move.std()),
        "dated_std": float(both.dated_move.std()),
    }
    lineage["implementation"] = {
        str(path): shared.digest(path)
        for path in (Path(__file__), Path("scripts/lead83_unit2.py"), SERVED)
    }
    print(
        f"inventory passed: {len(frame)} games including pushes; "
        f"{int(scored.sum())} conditional-cover rows; "
        f"{int(frame.loc[scored].available.eq(0).sum())} legacy-unavailable",
        flush=True,
    )
    return frame, cached, lineage


def fit_integrated(
    frame: pd.DataFrame, training: np.ndarray, atoms: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, dict, dict]:
    x = frame.loc[:, list(shared.TERMS)].to_numpy(float)
    mean = x[training].mean(axis=0)
    scale = x[training].std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(frame)), (x - mean) / scale])
    y_all = frame.home_covered.to_numpy(float)
    missing = frame.available.eq(0).to_numpy()
    standard = (atoms - mean[MOVE - 1]) / scale[MOVE - 1]
    log_weights = np.log(weights)
    seen = training & ~missing
    hidden = training & missing
    x_seen, y_seen = design[seen], y_all[seen]
    x_hidden, y_hidden = design[hidden], y_all[hidden]
    hidden_move = x_hidden[:, MOVE]

    def atom_logits(beta: np.ndarray) -> np.ndarray:
        return (x_hidden @ beta)[:, None] + beta[MOVE] * (standard[None, :] - hidden_move[:, None])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = x_seen @ beta
        value = float(np.sum(np.logaddexp(0, z) - y_seen * z))
        gradient = -x_seen.T @ (y_seen - expit(z))
        if len(y_hidden):
            zk = atom_logits(beta)
            loglik = -(np.logaddexp(0, zk) - y_hidden[:, None] * zk) + log_weights[None, :]
            total = logsumexp(loglik, axis=1)
            value -= float(total.sum())
            share = np.exp(loglik - total[:, None])
            coefficient = share * (y_hidden[:, None] - expit(zk))
            hidden_gradient = x_hidden.T @ coefficient.sum(axis=1)
            hidden_gradient[MOVE] += (
                coefficient * (standard[None, :] - hidden_move[:, None])
            ).sum()
            gradient = gradient - hidden_gradient
        return value + 0.5 * RIDGE * float(beta @ beta), gradient + RIDGE * beta

    fitted = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 5000},
    )
    beta = fitted.x
    gradient_max = float(np.abs(objective(beta)[1]).max())
    if not np.isfinite(beta).all() or gradient_max > 1e-5:
        raise ValueError(f"Integrated likelihood optimization failed: {fitted.message}")
    z = design @ beta
    probability = expit(np.clip(z, -35, 35))
    if missing.any():
        rows = np.flatnonzero(missing)
        zk = z[rows][:, None] + beta[MOVE] * (standard[None, :] - design[rows, MOVE][:, None])
        probability[rows] = expit(np.clip(zk, -35, 35)) @ weights
    natural = beta[1:] / scale
    coefficients = dict(zip(shared.TERMS, natural.tolist(), strict=True))
    coefficients["intercept"] = float(beta[0] - mean @ natural)
    details = {
        "iterations": int(fitted.nit),
        "success": bool(fitted.success),
        "gradient_max": gradient_max,
        "unavailable_training_rows": int(hidden.sum()),
        "observed_training_rows": int(seen.sum()),
        "law_atoms": len(atoms),
        "law_mean": float(weights @ atoms),
        "law_std": float(np.sqrt(weights @ (atoms - weights @ atoms) ** 2)),
    }
    return probability, coefficients, details


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
        law_moves = train.dated_move.to_numpy(float)
        if not np.isfinite(law_moves).all():
            raise ValueError("Move law contains a missing dated move")
        atoms, counts = np.unique(law_moves, return_counts=True)
        weights = counts / counts.sum()
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
        raw["integrated"], coefficients["integrated"], optimization = fit_integrated(
            panel, training, atoms, weights
        )
        degenerate, degenerate_coefficients, _ = fit_integrated(
            panel, training, np.zeros(1), np.ones(1)
        )
        parity = max(
            abs(degenerate_coefficients[name] - coefficients["four_term"][name])
            for name in coefficients["four_term"]
        )
        if parity > 1e-4 or float(np.abs(degenerate - raw["four_term"]).max()) > 1e-4:
            raise ValueError(f"Degenerate-law parity with the four-term fit failed: {parity}")
        optimization["degenerate_law_parity"] = float(parity)
        masses["four_term"] = masses["model_only"]
        masses["integrated"] = masses["model_only"]
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "upstream_games_with_pushes": len(train),
            "training_max_gameday": str(pd.to_datetime(train.gameday).max().date()),
            "roles": panel.role.value_counts().to_dict(),
            "home_side_offsets": offsets.offsets,
            "elo_margin_coefficients": elo_beta.tolist(),
            "optimization": optimization,
            "outer_unavailable": int(panel.loc[panel.role.eq("outer"), "available"].eq(0).sum()),
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
                    selected,
                    ["game_id", "season", "week", "result", "spread_line", "home_covered"],
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
            f"fold={outer} complete; fit through {outer - 3}; "
            f"outer n={int(panel.role.eq('outer').sum())}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), details


def summarize(predictions: pd.DataFrame) -> dict:
    panels = {}
    direction = np.array([1.0, -1.0, -1.0, -1.0])
    for outer in (2023, 2024, 2025, "pooled"):
        selected = (
            predictions if outer == "pooled" else predictions.loc[predictions.outer.eq(outer)]
        )
        panel = {"phases": {}, "decisive": {}}
        phase_draws = {}
        phase_means = {}
        for phase in ("IS", "OOS"):
            frame = selected.loc[selected.phase.eq(phase)]
            groups = {
                arm: frame.loc[frame.arm.eq(arm)]
                .sort_values(["outer", "game_id"])
                .reset_index(drop=True)
                for arm in ARMS
            }
            reference = groups["integrated"]
            if any(
                not g[["outer", "game_id"]].equals(reference[["outer", "game_id"]])
                for g in groups.values()
            ):
                raise ValueError("Unpaired arm populations")
            arrays = [groups[arm].loc[:, list(shared.METRICS)].to_numpy(float) for arm in ARMS]
            contrasts = [(arrays[0] - array) * direction for array in arrays[1:]]
            values = np.column_stack([*arrays, *contrasts])
            samples = shared.blocked(
                reference,
                values,
                shared.SEED
                + (0 if outer == "pooled" else int(outer))
                + (0 if phase == "IS" else 100),
            )
            means = values.mean(axis=0)
            phase_draws[phase] = samples
            phase_means[phase] = means
            cells = {}
            for index, label in enumerate(LABELS):
                cells[label] = {
                    metric: shared.estimate(means[index * 4 + m], samples[:, index * 4 + m])
                    for m, metric in enumerate(shared.METRICS)
                }
                cells[label]["games"] = len(reference)
                if label in ARMS:
                    wins = int(groups[label].accuracy_points.eq(100).sum())
                    cells[label]["record"] = f"{wins}-{len(reference) - wins}"
            panel["phases"][phase] = cells
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
        gap_means = phase_means["OOS"] - phase_means["IS"]
        gap_draws = phase_draws["OOS"] - phase_draws["IS"]
        panel["gaps_OOS_minus_IS"] = {
            label: {
                metric: shared.estimate(gap_means[index * 4 + m], gap_draws[:, index * 4 + m])
                for m, metric in enumerate(shared.METRICS)
            }
            for index, label in enumerate(LABELS)
        }
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


def record_cells(summary: dict) -> str:
    units = ("accuracy_points", "log_loss_improvement", "brier_improvement", "rps_improvement")
    lines = []
    for label, panel in summary["panels"].items():
        first, last = (2023, 2025) if label == "pooled" else (int(label), int(label))
        oos = panel["phases"]["OOS"]
        for metric, unit in zip(shared.METRICS, units, strict=True):
            cell = oos["gain_vs_four_term"][metric]
            values = " ".join(
                f"{cell[key]:.12g}" for key in ("estimate", "low", "high", "probability_positive")
            )
            lines.append(
                f"{label} {first} {last} {oos['integrated']['games']} {panel['blocks']} "
                f"{unit} {values}"
            )
    return "\n".join(lines)


def report(summary: dict, protocol: str) -> None:
    lineage = summary["lineage"]
    lines = [
        "# LEAD-97 unit 1: integrate unavailable moves",
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
            " tests/scratch/codex/lead97_unit1/protocol.md."
        ),
        "",
        protocol,
        "",
        (
            "**Measured:** upstream margin/offset/Elo models and discrete lattices "
            "refit through Y-3; no cached weekly model probabilities used. "
            "Historical opener replaces closing spread, archived Tuesday totals "
            "replace closing totals. Hashes verify the LEAD-83 inputs and every "
            "quote archive. IS is optimistic, with repeated training rows across "
            "pooled folds. All five arms share slope selection and intercept "
            "calibration; the calibrated discrete lattice alone selects the side."
        ),
        "",
        (
            f"**Measured:** {lineage['population']} source-complete games including "
            f"pushes; {lineage['nonpush']} conditional-cover rows; "
            f"{lineage['missing_tuesday_totals']} missing Tuesday totals retained as "
            f"missing; {LOOKS} looks, 10,000 paired season/week-block resamples. "
            f"Legacy [available, games] by season: "
            f"{json.dumps(lineage['legacy_availability_by_season'])}. "
            f"Observed legacy versus dated matched-book move: "
            f"{json.dumps(lineage['observed_legacy_vs_dated_move'])}. Intervals "
            "condition on fitted models and omit full refit uncertainty."
        ),
        "",
        (
            "**Inferred:** unresolved_below_power pending orchestrator recording. "
            "Legacy availability is 0 for every 2020-2022 game and 1 for every "
            "2023-2025 game, so each fold fits on unavailable rows only and every "
            "outer row carries an observed move; integration can change outer "
            "predictions only through the fitted coefficients, which is the "
            "information-availability estimand tested here. Per AGENTS.md "
            "Margins/Promotion, diagnostic gains cannot promote ATS sides and zero "
            "crossing closes nothing. No card change or closure."
        ),
        "",
        "## Fold coefficients and stability",
        "",
        "**Measured:** natural coefficients before and after slope/intercept calibration.",
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
            ["Outer", "Fit through", "Roles", "Outer unavailable", "Optimizer and law"],
            [
                [
                    fold["outer"],
                    fold["fit_through"],
                    json.dumps(fold["roles"]),
                    fold["outer_unavailable"],
                    json.dumps(fold["optimization"]),
                ]
                for fold in summary["folds"]
            ],
        ),
        "",
        "## IS, OOS and gaps",
        "",
        (
            "**Measured:** each cell is estimate [95% interval]; probability_positive. "
            "Accuracy is percentage points; other endpoints are losses. Gain rows "
            "orient all endpoints so positive favors the candidate. Gaps are OOS "
            "minus IS. Raw loss probability_positive means loss above zero, not "
            "improvement."
        ),
    ]
    for outer, panel in summary["panels"].items():
        rows = []
        for phase, cells in (*panel["phases"].items(), ("gap", panel["gaps_OOS_minus_IS"])):
            for label, measures in cells.items():
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
                            for metric in shared.METRICS
                        ],
                    ]
                )
        lines += [
            "",
            f"### {outer}",
            "",
            shared.table(["Phase", "Arm/contrast", "W-L", *shared.METRICS], rows),
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
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead97_unit1.py",
        "",
        (
            "Prediction-level rows, upstream cutoffs, coefficients and intervals: "
            "tests/scratch/codex/lead97_unit1/. Serial registry commands are in "
            "docs/lanes/lead97.md and were not run by this worker."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise ValueError("A scored replay already exists; do not silently rescore")
    protocol = LANE.read_text(encoding="utf-8")
    (OUTPUT / "protocol.md").write_text(protocol, encoding="utf-8")
    declaration = protocol.split("## Protocol (fixed before outcomes)\n", 1)[1].split(
        "\n## Tried", 1
    )[0]
    with threadpool_limits(limits=1):
        frame, cached, lineage = load()
        predictions, folds = replay(frame, cached)
        pq.write_table(
            pa.Table.from_pandas(predictions, preserve_index=False), OUTPUT / "predictions.parquet"
        )
        summary = summarize(predictions)
        summary.update(
            {
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
        (OUTPUT / "record_cells.txt").write_text(record_cells(summary), encoding="utf-8")
        report(summary, declaration)
    pooled = summary["panels"]["pooled"]["phases"]["OOS"]
    print(
        json.dumps(
            {
                "status": "complete",
                "games": pooled["integrated"]["games"],
                "record": pooled["integrated"]["record"],
                "candidate_vs_four_term": pooled["gain_vs_four_term"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
