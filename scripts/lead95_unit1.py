from __future__ import annotations

import json
from pathlib import Path

import lead83_unit2 as shared
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

OUTPUT = Path("tests/scratch/codex/lead95_unit1")
REPORT = Path("docs/lead95_unit1.md")
LANE = Path("docs/lanes/lead95.md")
PBP = Path("data/pbp/raw/20260925T202544Z")
ARMS = ("candidate", "four_term", "model_only", "market", "elo")
METRICS = (*shared.METRICS, "regulation_log_score")
LABELS = (*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:]))
CLASSES = (0, 1, 2, 3, 6, 7, 8)
FLOOR = 1e-4
LOOKS = 609
NM = len(METRICS)


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
    overtime = set()
    pbp_hashes = {}
    plays_total = 0
    for season in range(2020, 2026):
        path = PBP / f"season={season}" / "plays.parquet"
        plays = pd.read_parquet(path, columns=["game_id", "qtr"])
        plays_total += len(plays)
        overtime |= set(plays.loc[plays.qtr.eq(5), "game_id"])
        pbp_hashes[str(path)] = shared.digest(path)
    ids = set(frame.game_id)
    missing = overtime - ids
    scored = {game for game in overtime if game in ids}
    if not scored:
        raise ValueError("No overtime games matched the population")
    frame["overtime"] = frame.game_id.isin(overtime)
    frame["regulation_result"] = np.where(frame.overtime, 0.0, frame.result.astype(float))
    if frame.loc[frame.overtime, "result"].abs().isin(CLASSES).eq(False).any():
        raise ValueError("Overtime final margin outside the declared kernel support")
    lineage["pbp_plays"] = plays_total
    lineage["pbp_hashes"] = pbp_hashes
    lineage["overtime_games_in_population"] = len(scored)
    lineage["overtime_games_outside_population"] = len(missing)
    lineage["implementation"] = {
        str(path): shared.digest(path) for path in (Path(__file__), Path("scripts/lead83_unit2.py"))
    }
    print(
        f"inventory passed: {len(frame)} games; {len(scored)} overtime games in population; "
        f"{len(missing)} overtime games outside it; {plays_total} plays",
        flush=True,
    )
    return frame, cached, lineage


def kernel(train: pd.DataFrame) -> np.ndarray:
    results = np.abs(train.loc[train.overtime, "result"].to_numpy(float)).astype(int)
    counts = np.array([float((results == c).sum()) + 0.5 for c in CLASSES])
    probs = counts / counts.sum()
    output = np.zeros(len(shared.GRID))
    for cls, prob in zip(CLASSES, probs, strict=True):
        if cls == 0:
            output[100] += prob
        else:
            output[100 + cls] += prob / 2
            output[100 - cls] += prob / 2
    return output


def settle(mass: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = mass.copy()
    tie = output[:, 100].copy()
    output[:, 100] = 0.0
    output += tie[:, None] * weights[None, :]
    return output


def regulation_score(mass: np.ndarray, margin: np.ndarray) -> np.ndarray:
    picked = mass[np.arange(len(mass)), np.rint(margin).astype(int) + 100]
    return -np.log(np.maximum(picked, FLOOR))


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
        regulation_train = train.copy()
        regulation_train["result"] = regulation_train.regulation_result
        regulation_train["ats_margin"] = regulation_train.result - regulation_train.spread_line
        weights = kernel(train)
        elo_x = np.column_stack([np.ones(len(panel)), panel.elo_diff.to_numpy(float)])
        elo_beta = np.linalg.lstsq(
            elo_x[training], panel.loc[training, "result"].to_numpy(float), rcond=None
        )[0]
        points = {}
        offsets_report = {}
        for name, source in (("final", train), ("regulation", regulation_train)):
            model = shared.fit_margin_model(
                source,
                target="market_residual",
                model_name="ridge",
                feature_profile="weak_stack",
                ridge_alpha=10.0,
            )
            forecast = model.predict(panel, probability_method="gaussian_median")
            point = forecast.predicted_margin.to_numpy() + shared.residual_location(
                model.residuals, "gaussian_median"
            )
            offset_rows = source.copy()
            offset_rows["point_incumbent"] = point[training]
            offsets = shared.fit_home_side_offsets(offset_rows)
            points[name] = point + offsets.offset_for(panel.spread_line).to_numpy(float)
            offsets_report[name] = offsets.offsets
        lines = panel.spread_line.to_numpy(float)
        final_points = {
            "model_only": points["final"],
            "market": panel.sunday_line.to_numpy(float),
            "elo": elo_x @ elo_beta,
        }
        masses = {arm: shared.lattice(train, panel, value) for arm, value in final_points.items()}
        regulation_mass = shared.lattice(regulation_train, panel, points["regulation"])
        masses["candidate"] = settle(regulation_mass, weights)
        raw = {arm: shared.probability(mass, lines) for arm, mass in masses.items()}
        panel["model_logit"] = logit(raw["model_only"])
        candidate_logit = logit(raw["candidate"])
        scores_source = {
            "candidate": regulation_mass,
            "four_term": masses["model_only"],
            "model_only": masses["model_only"],
            "market": masses["market"],
            "elo": masses["elo"],
        }
        regulation_scores = {
            arm: regulation_score(mass, panel.regulation_result.to_numpy(float))
            for arm, mass in scores_source.items()
        }
        upstream = panel[["game_id", "season", "week", "role", "spread_line", "model_logit"]].copy()
        upstream["fit_through"] = outer - 3
        upstream["point_final"] = points["final"]
        upstream["point_regulation"] = points["regulation"]
        upstream["candidate_logit"] = candidate_logit
        upstream["training_max_gameday"] = str(pd.to_datetime(train.gameday).max().date())
        pq.write_table(
            pa.Table.from_pandas(upstream, preserve_index=False),
            OUTPUT / f"upstream_{outer}.parquet",
        )
        kept = panel.declared_fit_population.to_numpy()
        panel = panel.loc[kept].reset_index(drop=True)
        candidate_logit = candidate_logit[kept]
        panel["home_covered"] = panel.ats_margin.gt(0).astype(float)
        training = panel.role.eq("fit").to_numpy()
        raw = {arm: values[kept] for arm, values in raw.items()}
        masses = {arm: values[kept] for arm, values in masses.items()}
        regulation_scores = {arm: values[kept] for arm, values in regulation_scores.items()}
        coefficients = {}
        raw["four_term"], coefficients["four_term"] = shared.fit_probability(
            panel, training, shared.TERMS
        )
        candidate_panel = panel.copy()
        candidate_panel["model_logit"] = candidate_logit
        raw["candidate"], coefficients["candidate"] = shared.fit_probability(
            candidate_panel, training, shared.TERMS
        )
        masses["four_term"] = masses["model_only"]
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "upstream_games_with_pushes": len(train),
            "training_overtime_games": int(train.overtime.sum()),
            "overtime_kernel": {
                str(int(m)): float(w) for m, w in zip(shared.GRID, weights, strict=True) if w > 0
            },
            "training_max_gameday": str(pd.to_datetime(train.gameday).max().date()),
            "roles": panel.role.value_counts().to_dict(),
            "home_side_offsets": offsets_report,
            "elo_margin_coefficients": elo_beta.tolist(),
            "probability_coefficients": {},
            "calibration": {},
        }
        for arm in ARMS:
            p, calibration = shared.calibrate(raw[arm], panel)
            scores, push = shared.metrics(panel, p, masses[arm])
            scores = np.column_stack([scores, regulation_scores[arm]])
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
                    [
                        "game_id",
                        "season",
                        "week",
                        "result",
                        "spread_line",
                        "home_covered",
                        "overtime",
                    ],
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
                f"outer n={int(panel.role.eq('outer').sum())}"
            ),
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), details


def endpoints(raw: np.ndarray) -> np.ndarray:
    arms = raw.reshape(5, NM)
    signs = np.array([1.0, -1.0, -1.0, -1.0, -1.0])
    gains = (arms[0] - arms[1:]) * signs
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
            metric: shared.estimate(means[i * NM + j], draws[:, i * NM + j])
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
            reference = groups["candidate"]
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
                panel["overtime_games"] = int(reference.overtime.sum())
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
        ("accuracy_points", "accuracy_points"),
        ("log_loss", "log_loss_improvement"),
        ("brier", "brier_improvement"),
        ("rps", "rps_improvement"),
        ("regulation_log_score", "log_loss_improvement"),
    )
    fence = chr(96) * 3
    lines = [
        f"{fence}bash",
        "while read -r panel first last games blocks label units effect low high pp; do",
        (
            ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record "
            '--name "lead95_unit1_${panel}_${label}" --league nfl \\'
        ),
        (
            '  --description "Regulation-margin fit with overtime settlement versus the usual'
            ' four-term calculation: ${panel} ${label}" --source docs/lead95_unit1.md '
            "--family lead95_unit1_609_looks \\"
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
            'closure or serving claim. All 609 declared looks reported." \\'
        ),
        (
            '  --plain-summary "This trains the pick calculation on the score at the end of '
            "regulation and treats overtime as a separate coin flip on tied games. The "
            'held-out comparison stays a research result; pool picks have not changed."'
        ),
        "done <<'CELLS'",
    ]
    for label, panel in summary["panels"].items():
        first, last = (2023, 2025) if label == "pooled" else (int(label), int(label))
        oos = panel["phases"]["OOS"]
        for metric, (name, unit) in zip(METRICS, units, strict=True):
            cell = oos["gain_vs_four_term"][metric]
            values = " ".join(
                f"{cell[key]:.12g}" for key in ("estimate", "low", "high", "probability_positive")
            )
            lines.append(
                f"{label} {first} {last} {oos['candidate']['games']} {panel['blocks']} "
                f"{name} {unit} {values}"
            )
    return "\n".join([*lines, "CELLS", fence])


def report(summary: dict) -> None:
    lines = [
        "# LEAD-95 unit 1: regulation labels and overtime settlement",
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
            " tests/scratch/codex/lead95_unit1/protocol.md."
        ),
        "",
        summary["protocol"],
        "",
        (
            "**Measured:** all upstream margin/offset/Elo models and discrete "
            "lattices refit through Y-3; the candidate refits the margin ridge, "
            "offsets and lattice on regulation margins and settles regulation-tie "
            "mass with an earlier-season overtime kernel. Historical opener replaces "
            "the closing spread. Hashes verify the LEAD-83 inputs, both quote "
            "archives and the play files. IS is optimistic. All five arms receive "
            "the same slope selection and intercept calibration on final cover "
            "labels."
        ),
        "",
        (
            f"**Measured:** {summary['lineage']['population']} games including pushes; "
            f"{summary['lineage']['nonpush']} conditional-cover rows; "
            f"{summary['lineage']['overtime_games_in_population']} overtime games in the "
            f"population ({summary['lineage']['overtime_games_outside_population']} outside "
            f"it); {summary['lineage']['pbp_plays']} plays read; {LOOKS} looks, 10,000 "
            "paired season/week-block resamples. Gap intervals subtract independently "
            "resampled IS from OOS. Intervals condition on fitted models and omit "
            "refit uncertainty. The regulation log score compares a regulation lattice "
            "(candidate) with final-margin lattices (comparators), floored at 1e-4."
        ),
        "",
        (
            "**Inferred:** unresolved_below_power pending orchestrator recording. "
            "Three outer seasons and retrospective source reuse limit "
            "generalization; the 2023 kernel comes from the 2020 season alone and "
            "the 2025 overtime rule change is not modelled. Per AGENTS.md "
            "Margins/Promotion, a diagnostic gain cannot promote ATS sides; zero "
            "crossing closes nothing. No card change or closure."
        ),
        "",
        "## Fold coefficients, kernels and stability",
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
            ["Outer", "Fit through", "Roles", "Training overtime games", "Overtime kernel"],
            [
                [
                    fold["outer"],
                    fold["fit_through"],
                    json.dumps(fold["roles"]),
                    fold["training_overtime_games"],
                    json.dumps({k: round(v, 4) for k, v in fold["overtime_kernel"].items()}),
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
            f"Held-out overtime games: {panel['overtime_games']}.",
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
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead95_unit1.py",
        "",
        (
            "Prediction-level rows, upstream cutoffs, complete coefficients, source"
            " hashes and all intervals: tests/scratch/codex/lead95_unit1/. Serial "
            "registry commands are in docs/lanes/lead95.md and were not run by this"
            " worker."
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
        (OUTPUT / "record_commands.md").write_text(record_commands(summary), encoding="utf-8")
        report(summary)
    pooled = summary["panels"]["pooled"]["phases"]["OOS"]
    print(
        json.dumps(
            {
                "status": "complete",
                "games": pooled["candidate"]["games"],
                "record": pooled["candidate"]["record"],
                "candidate_vs_four_term": pooled["gain_vs_four_term"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
