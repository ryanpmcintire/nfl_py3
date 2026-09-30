from __future__ import annotations

import json
from pathlib import Path

import lead83_unit2 as shared
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import binomtest, t
from threadpoolctl import threadpool_limits

from nfl_ats.cfb_features import (
    _LINE_LOAD_COLUMNS,
    _SCHEDULE_LOAD_COLUMNS,
    _filtered_schedule,
    build_cfb_market_table,
    load_cfb_seasons,
)

OUTPUT = Path("tests/scratch/codex/lead93_unit1")
REPORT = Path("docs/lead93_unit1.md")
LANE = Path("docs/lanes/lead93.md")
CFB_ROOT = Path("data/cfb")
ARMS = ("shape", "four_term", "model_only", "market", "elo")
METRICS = (*shared.METRICS, "tail_rps")
LABELS = (*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:]))
SIGNS = np.array([1.0, -1.0, -1.0, -1.0, -1.0])
LOOKS = 609
TAIL_IQRS = 1.5
NU_BOUNDS = (1.0, 200.0)


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


def college_openers(last_season: int) -> tuple[pd.DataFrame, dict]:
    seasons = list(range(2006, last_season + 1))
    schedules = load_cfb_seasons(CFB_ROOT, "schedules", seasons, list(_SCHEDULE_LOAD_COLUMNS))
    lines = load_cfb_seasons(CFB_ROOT, "lines", seasons, list(_LINE_LOAD_COLUMNS))
    games, schedule_audit = _filtered_schedule(schedules, seasons[0], seasons[-1])
    market, market_audit = build_cfb_market_table(lines, games)
    merged = market.loc[market.spread_open.notna(), ["game_id", "spread_open"]].merge(
        games[["game_id", "season", "home_points", "away_points"]],
        on="game_id",
        validate="one_to_one",
    )
    merged["residual"] = (merged.home_points - merged.away_points) - merged.spread_open
    merged = merged.loc[:, ["game_id", "season", "residual"]]
    audit = {
        "seasons_loaded": len(seasons),
        "schedule": schedule_audit,
        "market": market_audit,
        "games_with_opener": len(merged),
        "games_with_opener_by_season": {
            str(k): int(v) for k, v in merged.groupby("season").size().items()
        },
    }
    print(f"college openers: {len(merged)} games through {last_season}", flush=True)
    return merged, audit


def centre_scale(residual: np.ndarray) -> tuple[float, float]:
    centre = float(np.median(residual))
    q1, q3 = np.quantile(residual, [0.25, 0.75])
    return centre, float(q3 - q1)


def bin_edges(centred: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(centred)
    return np.maximum(magnitude - 0.5, 0.0) / scale, (magnitude + 0.5) / scale


def fit_shape(edges: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    low = np.concatenate([e[0] for e in edges])
    high = np.concatenate([e[1] for e in edges])

    def loss(value: np.ndarray) -> float:
        nu = float(np.clip(np.exp(value[0]), *NU_BOUNDS))
        sigma = float(np.exp(value[1]))
        mass = 2 * (t.cdf(high / sigma, nu) - t.cdf(low / sigma, nu))
        return float(-np.log(np.clip(mass, 1e-300, None)).sum())

    fitted = minimize(
        loss,
        np.log([5.0, 1.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000},
    )
    if not fitted.success:
        raise ValueError(f"Student-t shape fit failed: {fitted.message}")
    fitted.x[0] = np.log(np.clip(np.exp(fitted.x[0]), *NU_BOUNDS))
    nu, sigma = np.exp(fitted.x)
    return {
        "nu": float(nu),
        "scale": float(sigma),
        "negative_log_likelihood": float(fitted.fun),
        "rows": len(low),
    }


def reshape_mass(mass: np.ndarray, lines: np.ndarray, centre: float, scale: float, shared_fit, own):
    grid = shared.GRID[None, :].astype(float)
    signed = grid - lines[:, None] - centre
    low, high = (signed - 0.5) / scale, (signed + 0.5) / scale

    def cell(fit: dict) -> np.ndarray:
        return t.cdf(high / fit["scale"], fit["nu"]) - t.cdf(low / fit["scale"], fit["nu"])

    weight = cell(shared_fit) / np.clip(cell(own), 1e-300, None)
    if not np.isfinite(weight).all():
        raise ValueError("Non-finite shape weights")
    out = mass * weight
    return out / out.sum(axis=1, keepdims=True)


def metrics(
    frame: pd.DataFrame, p: np.ndarray, raw_mass: np.ndarray, centre: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    lines = frame.spread_line.to_numpy(float)
    grid = shared.GRID[None, :]
    home = grid > lines[:, None]
    away = grid < lines[:, None]
    pushes = ~(home | away)
    cover_mass = (raw_mass * home).sum(axis=1)
    loss_mass = (raw_mass * away).sum(axis=1)
    push_mass = (raw_mass * pushes).sum(axis=1)
    mass = raw_mass * (
        home * ((1 - push_mass) * p / cover_mass)[:, None]
        + away * ((1 - push_mass) * (1 - p) / loss_mass)[:, None]
        + pushes
    )
    if not np.allclose(mass.sum(axis=1), 1) or not np.allclose(shared.probability(mass, lines), p):
        raise ValueError("Calibrated lattice failed mass/probability identity")
    y = frame.home_covered.to_numpy(float)
    squared = (np.cumsum(mass, axis=1) - (frame.result.to_numpy()[:, None] <= grid)) ** 2
    tail = np.abs(grid - lines[:, None] - centre) > TAIL_IQRS * scale
    result = np.column_stack(
        [
            100 * ((p >= 0.5) == y),
            -(y * np.log(p) + (1 - y) * np.log1p(-p)),
            (p - y) ** 2,
            squared.sum(axis=1),
            (squared * tail).sum(axis=1),
        ]
    )
    return result, push_mass


def replay(
    frame: pd.DataFrame, cached: pd.DataFrame, college: pd.DataFrame
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
        lines_all = panel.spread_line.to_numpy(float)
        raw = {arm: shared.probability(mass, lines_all) for arm, mass in masses.items()}
        nfl_residual = train.result.to_numpy(float) - train.spread_line.to_numpy(float)
        nfl_centre, nfl_scale = centre_scale(nfl_residual)
        cfb = college.loc[college.season.le(outer - 3)]
        cfb_centre, cfb_scale = centre_scale(cfb.residual.to_numpy(float))
        nfl_edges = bin_edges(nfl_residual - nfl_centre, nfl_scale)
        cfb_edges = bin_edges(cfb.residual.to_numpy(float) - cfb_centre, cfb_scale)
        shared_fit = fit_shape([nfl_edges, cfb_edges])
        own_fit = fit_shape([nfl_edges])
        masses["shape"] = reshape_mass(
            masses["model_only"], lines_all, nfl_centre, nfl_scale, shared_fit, own_fit
        )
        raw["shape"] = shared.probability(masses["shape"], lines_all)
        panel["model_logit"] = logit(raw["model_only"])
        panel["shape_logit"] = logit(raw["shape"])
        upstream = panel[["game_id", "season", "week", "role", "spread_line", "model_logit"]].copy()
        upstream["shape_logit"] = panel.shape_logit
        upstream["fit_through"] = outer - 3
        upstream["point"] = point
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
        swapped = panel.copy()
        swapped["model_logit"] = panel.shape_logit
        raw["shape"], coefficients["shape"] = shared.fit_probability(
            swapped, training, shared.TERMS
        )
        masses["four_term"] = masses["model_only"]
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "upstream_games_with_pushes": len(train),
            "college_training_games": len(cfb),
            "roles": panel.role.value_counts().to_dict(),
            "nfl_centre": nfl_centre,
            "nfl_iqr": nfl_scale,
            "college_centre": cfb_centre,
            "college_iqr": cfb_scale,
            "shared_shape": shared_fit,
            "nfl_only_shape": own_fit,
            "home_side_offsets": offsets.offsets,
            "elo_margin_coefficients": elo_beta.tolist(),
            "probability_coefficients": {},
            "calibration": {},
        }
        for arm in ARMS:
            p, calibration = shared.calibrate(raw[arm], panel)
            scores, push = metrics(panel, p, masses[arm], nfl_centre, nfl_scale)
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
                f"shared nu={shared_fit['nu']:.3f} own nu={own_fit['nu']:.3f}"
            ),
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), details


def endpoints(raw: np.ndarray) -> np.ndarray:
    arms = raw.reshape(-1, 5, len(METRICS))
    gains = (arms[:, :1] - arms[:, 1:]) * SIGNS[None, None, :]
    return np.concatenate([arms, gains], axis=1).reshape(len(arms), -1)


def cells(means: np.ndarray, draws: np.ndarray) -> dict:
    return {
        label: {
            metric: shared.estimate(means[i * len(METRICS) + j], draws[:, i * len(METRICS) + j])
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
            reference = groups["shape"]
            if any(
                not g[["outer", "game_id"]].equals(reference[["outer", "game_id"]])
                for g in groups.values()
            ):
                raise ValueError("Unpaired arm populations")
            values = np.column_stack(
                [groups[arm].loc[:, list(METRICS)].to_numpy(float) for arm in ARMS]
            )
            seed = (
                shared.SEED
                + (0 if outer == "pooled" else int(outer))
                + (100 if phase == "OOS" else 0)
            )
            means = endpoints(values.mean(axis=0)[None, :])[0]
            draws = endpoints(shared.blocked(reference, values, seed))
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


UNITS = {
    "accuracy_points": "accuracy_points",
    "log_loss": "log_loss_improvement",
    "brier": "brier_improvement",
    "rps": "rps_improvement",
    "tail_rps": "rps_improvement",
}


def record_commands(summary: dict) -> str:
    fence = chr(96) * 3
    lines = [
        f"{fence}bash",
        "while read -r panel first last games blocks metric units effect low high pp; do",
        (
            ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record "
            '--name "lead93_unit1_${panel}_${metric}" --league nfl \\'
        ),
        (
            '  --description "College-shaped uncertainty around the pool line versus the '
            'usual four-term calculation: ${panel} ${metric}" --source docs/lead93_unit1.md '
            "--family lead93_unit1_609_looks \\"
        ),
        (
            '  --season-start "$first" --season-end "$last" --sample-games "$games" '
            '--sample-blocks "$blocks" --effect-units "$units" \\'
        ),
        (
            '  --effect="$effect" --interval-low="$low" --interval-high="$high" '
            '--probability-positive "$pp" --classification unresolved_below_power \\'
        ),
        (
            '  --classification-evidence "Three held-out seasons; no mechanism closure or '
            'serving claim. All 609 declared looks reported." \\'
        ),
        (
            '  --plain-summary "This borrows how spread misses are shaped in college football '
            "to set how wide the pick calculation's uncertainty is, without using any college "
            'picks. The held-out comparison is a research result; pool picks have not changed."'
        ),
        "done <<'CELLS'",
    ]
    for label, panel in summary["panels"].items():
        first, last = (2023, 2025) if label == "pooled" else (int(label), int(label))
        oos = panel["phases"]["OOS"]
        for metric in METRICS:
            cell = oos["gain_vs_four_term"][metric]
            values = " ".join(
                f"{cell[key]:.12g}" for key in ("estimate", "low", "high", "probability_positive")
            )
            lines.append(
                f"{label} {first} {last} {oos['shape']['games']} {panel['blocks']} "
                f"{metric} {UNITS[metric]} {values}"
            )
    return "\n".join([*lines, "CELLS", fence])


def report(summary: dict) -> None:
    lines = [
        "# LEAD-93 unit 1: college noise shape for the NFL score spread",
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
            " tests/scratch/codex/lead93_unit1/protocol.md."
        ),
        "",
        summary["protocol"],
        "",
        (
            f"**Measured:** {summary['lineage']['population']} source-complete "
            f"games including pushes; {summary['lineage']['nonpush']} conditional-cover "
            f"rows; {LOOKS} looks, 10,000 paired season/week-block resamples. "
            "IS is optimistic. Gap intervals subtract independently resampled IS "
            "from OOS. Intervals condition on fitted models and omit full refit "
            "uncertainty. All five arms receive the same slope selection and "
            "intercept calibration; only the candidate changes its PMF shape."
        ),
        "",
        (
            "**Inferred:** unresolved_below_power pending orchestrator recording. "
            "Per AGENTS.md Margins/Promotion, RPS-type gains alone cannot promote "
            "ATS sides; zero crossing closes nothing. No card change or closure."
        ),
        "",
        "## College shape inputs per fold",
        "",
        shared.table(
            [
                "Outer",
                "College games",
                "NFL centre",
                "NFL IQR",
                "College centre",
                "College IQR",
                "Shared nu",
                "Shared scale",
                "NFL-only nu",
                "NFL-only scale",
            ],
            [
                [
                    fold["outer"],
                    fold["college_training_games"],
                    f"{fold['nfl_centre']:.3f}",
                    f"{fold['nfl_iqr']:.3f}",
                    f"{fold['college_centre']:.3f}",
                    f"{fold['college_iqr']:.3f}",
                    f"{fold['shared_shape']['nu']:.4f}",
                    f"{fold['shared_shape']['scale']:.4f}",
                    f"{fold['nfl_only_shape']['nu']:.4f}",
                    f"{fold['nfl_only_shape']['scale']:.4f}",
                ]
                for fold in summary["folds"]
            ],
        ),
        "",
        "## Fold coefficients and stability",
        "",
        (
            "**Measured:** natural coefficients before and after slope/intercept "
            "calibration. Availability is constant, so its coefficient is zero."
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
            "are losses. Gain rows orient every endpoint so positive favors the "
            "candidate. Gaps are OOS minus IS. Raw loss probability_positive means "
            "loss above zero, not improvement."
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
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead93_unit1.py",
        "",
        (
            "Prediction-level rows, upstream cutoffs, college audit, source hashes "
            "and all intervals: tests/scratch/codex/lead93_unit1/. Serial registry "
            "commands are in docs/lanes/lead93.md and were not run by this worker."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    pooled = summary["panels"]["pooled"]
    oos = pooled["phases"]["OOS"]
    brier = oos["gain_vs_four_term"]["brier"]
    tail = oos["gain_vs_four_term"]["tail_rps"]
    decisive = pooled["decisive"]["four_term"]
    text = LANE.read_text(encoding="utf-8")
    head = text.split("## State\n", 1)[0]
    protocol = "## Protocol (fixed before outcomes)\n" + summary["protocol"]
    state = (
        f"**Measured:** 609-look replay complete: {oos['shape']['games']} outer games; "
        f"candidate {oos['shape']['record']}, base {oos['four_term']['record']}; decisive "
        f"{decisive['wins']}-{decisive['losses']}. Brier improvement "
        f"{shared.number(brier)}, probability_positive={brier['probability_positive']:.5f}; "
        f"tail RPS improvement {shared.number(tail)}, "
        f"probability_positive={tail['probability_positive']:.5f}. "
        "**Inferred:** unresolved_below_power; no serving change."
    )
    lane = "\n".join(
        [
            head.rstrip(),
            "",
            "## State",
            state,
            "",
            protocol,
            "",
            "## Tried",
            (
                "Protocol saved before outcomes. Ran "
                "`.tools/uv.exe run --no-sync --no-cache python scripts/lead93_unit1.py` "
                "once, single thread. Report docs/lead93_unit1.md; rows in "
                "tests/scratch/codex/lead93_unit1/."
            ),
            "",
            "## Next",
            "Orchestrator reviews the report, runs the record commands serially, assigns next.",
            "",
            "## Open",
            (
                "Three outer seasons, optimistic IS, retrospective archives; college opener "
                "is the source opening_lines with sparse timestamps. No closure or promotion."
            ),
            "",
            "## Record commands",
            (
                "Orchestrator alone runs these 20 candidate-versus-four-term OOS cells "
                "serially. Other contrasts and IS/gap looks stay reported."
            ),
            "",
            record_commands(summary),
            "",
        ]
    )
    LANE.write_text(lane, encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise ValueError("A scored replay already exists; do not silently rescore")
    protocol = LANE.read_text(encoding="utf-8")
    (OUTPUT / "protocol.md").write_text(protocol, encoding="utf-8")
    with threadpool_limits(limits=1):
        frame, cached, lineage = load()
        college, college_audit = college_openers(2022)
        predictions, folds = replay(frame, cached, college)
        pq.write_table(
            pa.Table.from_pandas(predictions, preserve_index=False), OUTPUT / "predictions.parquet"
        )
        summary = summarize(predictions)
        declaration = protocol.split("## Protocol (fixed before outcomes)\n", 1)[1].split(
            "\n## Tried", 1
        )[0]
        lineage["college"] = college_audit
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
                "games": pooled["shape"]["games"],
                "record": pooled["shape"]["record"],
                "candidate_vs_four_term": pooled["gain_vs_four_term"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
