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

from nfl_ats.tiebreaker import weighted_median

OUTPUT = Path("tests/scratch/codex/lead94_unit1")
REPORT = Path("docs/lead94_unit1.md")
LANE = Path("docs/lanes/lead94.md")
PROTOCOL = Path("docs/lead94_protocol.md")
PAIRS = Path("tests/scratch/codex/lead88_unit1/paired_games.parquet")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z/per_game.parquet")
PREVIOUS = Path("tests/scratch/codex/lead83_unit2")
OUTERS = (2023, 2024, 2025)
RIDGE = 0.001
CAP = 10.0
SEED = 20260994
LOOKS = 717
ATS_ARMS = ("candidate", "four_term", "model_only", "market", "elo")
TOTAL_ARMS = ("candidate", "tuesday", "deadline", "adjusted_deadline", "median")
TOTAL_ENDPOINTS = ("last_mae", "all_mae")
ATS_SIGNS = np.array([1.0, -1.0, -1.0, -1.0])
TOTAL_SIGNS = np.array([-1.0, -1.0])


def load():
    previous = json.loads((PREVIOUS / "summary.json").read_text(encoding="utf-8"))
    for filename, expected in previous["lineage"]["inputs"].items():
        if shared.digest(Path(filename)) != expected:
            raise ValueError(f"Certified upstream input changed: {filename}")
    source = json.loads(
        (Path("tests/scratch/codex/lead83_unit1") / "summary.json").read_text(encoding="utf-8-sig")
    )
    for entry in source["source_manifests"]:
        path = Path(entry["manifest"])
        if (
            shared.digest(path) != entry["manifest_sha256"]
            or shared.digest(path.parent / "quotes.parquet") != entry["quotes_sha256"]
        ):
            raise ValueError(f"Certified quote archive changed: {path}")
    pairs = shared.read(PAIRS).sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    if len(pairs) != 1343 or pairs.game_id.duplicated().any() or set(pairs.game_type) != {"REG"}:
        raise ValueError("Frozen paired population changed")
    if not (
        pairs.tuesday_last_observed.le(pairs.freeze).all()
        and pairs.deadline_last_observed.le(pairs.deadline).all()
        and pairs.deadline.lt(pairs.kickoff).all()
        and pairs.total_move.sub(pairs.deadline_total - pairs.tuesday_total).abs().lt(1e-9).all()
    ):
        raise ValueError("Total quote clock or move violation")
    schedule = shared.read(SCHEDULE, ["game_id", "home_score", "away_score"])
    pairs = pairs.merge(schedule, on="game_id", how="left", validate="one_to_one")
    pairs["actual_total"] = pairs.home_score + pairs.away_score
    opener = shared.read(OPENER, ["game_id", "tue_open_home_spread"])
    pairs = pairs.merge(opener, on="game_id", how="left", validate="one_to_one")
    if pairs.actual_total.isna().any() or pairs.tue_open_home_spread.isna().any():
        raise ValueError("Missing final score or opener")
    pairs["abs_open"] = pairs.tue_open_home_spread.abs()
    pairs["slot"] = pd.to_datetime(pairs.gameday).dt.day_name().str[:3] + " " + pairs.gametime
    predictions = shared.read(PREVIOUS / "predictions.parquet")
    predictions = predictions.loc[
        predictions.arm.isin(("four_term", "model_only", "market", "elo"))
        & predictions.game_id.isin(pairs.game_id)
    ].copy()
    candidate = predictions.loc[predictions.arm.eq("four_term")].copy()
    candidate["arm"] = "candidate"
    predictions = pd.concat([predictions, candidate], ignore_index=True)
    for outer in OUTERS:
        upstream = shared.read(PREVIOUS / f"upstream_{outer}.parquet")
        cutoff = pd.to_datetime(upstream.training_max_gameday).max()
        if not (
            upstream.fit_through.eq(outer - 3).all() and cutoff < pd.Timestamp(f"{outer - 2}-03-01")
        ):
            raise ValueError("Upstream training cutoff mismatch")
    lineage = {
        "inputs": {
            str(path): shared.digest(path)
            for path in (PAIRS, SCHEDULE, OPENER, PREVIOUS / "predictions.parquet")
        },
        "verified_lead83_inputs": len(previous["lineage"]["inputs"]),
        "verified_quote_manifests": len(source["source_manifests"]),
        "total_population": len(pairs),
        "last_games": int(pairs.is_last_game.sum()),
        "ats_intersection_games": int(
            predictions.loc[predictions.arm.eq("four_term")].game_id.nunique()
        ),
        "implementation": {
            str(path): shared.digest(path)
            for path in (Path(__file__), Path("scripts/lead83_unit2.py"))
        },
        "protocol_sha256": shared.digest(PROTOCOL),
    }
    return pairs, predictions, lineage


def fit_membership(train):
    slots = sorted(train.slot.unique())
    numbers = train[["tuesday_total", "abs_open"]].to_numpy(float)
    mean, scale = numbers.mean(axis=0), numbers.std(axis=0)
    scale[scale == 0] = 1.0

    def design(frame):
        z = (frame[["tuesday_total", "abs_open"]].to_numpy(float) - mean) / scale
        dummies = [(frame.slot == slot).to_numpy(float) for slot in slots]
        return np.column_stack([np.ones(len(frame)), z, *dummies])

    x = design(train)
    y = train.is_last_game.to_numpy(float)
    n, positives = len(y), y.sum()
    balance = np.where(y == 1, n / (2 * positives), n / (2 * (n - positives)))
    penalty = np.r_[0.0, np.full(x.shape[1] - 1, RIDGE)]

    def objective(beta):
        z = x @ beta
        value = (balance * (np.logaddexp(0, z) - y * z)).sum() + 0.5 * (penalty * beta) @ beta
        gradient = x.T @ (balance * (expit(z) - y)) + penalty * beta
        return value, gradient

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fit.success:
        raise ValueError(f"Membership fit failed: {fit.message}")
    prior = positives / n
    ratio = expit(x @ fit.x + logit(prior)) / prior
    weights = np.minimum(ratio, CAP)
    names = ["intercept", "tuesday_total", "abs_open", *[f"slot {slot}" for slot in slots]]
    return weights, {
        "coefficients": dict(zip(names, fit.x.tolist(), strict=True)),
        "prior": float(prior),
        "capped": int((ratio > CAP).sum()),
        "max_ratio": float(ratio.max()),
        "slots": len(slots),
    }


def fit_response(train, weights):
    moving = train.total_move.to_numpy() != 0
    move = train.total_move.to_numpy(float)[moving]
    ratios = (
        train.actual_total.to_numpy(float)[moving] - train.tuesday_total.to_numpy(float)[moving]
    ) / move
    return weighted_median(ratios, weights[moving] * np.abs(move))


def ess(weights):
    return float(weights.sum() ** 2 / (weights**2).sum())


def total_arms(frame, coefficient, intercept, median):
    return {
        "candidate": frame.tuesday_total.to_numpy(float)
        + coefficient * frame.total_move.to_numpy(float),
        "tuesday": frame.tuesday_total.to_numpy(float),
        "deadline": frame.deadline_total.to_numpy(float),
        "adjusted_deadline": frame.deadline_total.to_numpy(float) + intercept,
        "median": np.full(len(frame), median),
    }


def replay_totals(pairs):
    rows, folds = [], []
    for outer in OUTERS:
        train = pairs.loc[pairs.season.le(outer - 3)]
        calibrate = pairs.loc[pairs.season.eq(outer - 1)]
        test = pairs.loc[pairs.season.eq(outer)]
        if train.season.max() != outer - 3 or train.season.min() != 2020:
            raise ValueError("Total training cutoff mismatch")
        weights, membership = fit_membership(train)
        moving = train.total_move.ne(0).to_numpy()
        loss_weights = weights[moving] * np.abs(train.total_move.to_numpy(float)[moving])
        weighted = fit_response(train, weights)
        plain = fit_response(train, np.ones(len(train)))
        intercept = float(np.median(train.actual_total - train.deadline_total))
        median = float(np.median(train.actual_total))
        calibration_centre = calibrate.tuesday_total.to_numpy(
            float
        ) + weighted * calibrate.total_move.to_numpy(float)
        atoms = calibrate.actual_total.to_numpy(float) - np.rint(calibration_centre)
        atoms = atoms - np.median(atoms)
        for phase, frame in (("IS", train), ("OOS", test)):
            arms = total_arms(frame, weighted, intercept, median)
            law_median = np.array([np.median(centre + atoms) for centre in arms["candidate"]])
            if np.abs(law_median - arms["candidate"]).max() > 1e-9:
                raise ValueError("Discrete total law is not centred at the equation")
            actual = frame.actual_total.to_numpy(float)
            result = frame[["game_id", "season", "week", "is_last_game", "actual_total"]].copy()
            result["outer"] = outer
            result["phase"] = phase
            for arm, centre in arms.items():
                point = np.rint(law_median) if arm == "candidate" else np.rint(centre)
                result[arm + "_total"] = point
                result[arm + "_error"] = np.abs(point - actual)
            rows.append(result)
        folds.append(
            {
                "outer": outer,
                "fit_through": outer - 3,
                "train_games": len(train),
                "train_last_games": int(train.is_last_game.sum()),
                "moving_games": int(moving.sum()),
                "calibration_atoms": len(atoms),
                "weighted_b": weighted,
                "unweighted_b": plain,
                "intercept_shift": intercept,
                "training_median": median,
                "ess_games": ess(weights),
                "ess_loss": ess(loss_weights),
                "weight_max": float(weights.max()),
                "weight_min": float(weights.min()),
                "membership": membership,
            }
        )
        print(f"total fold {outer} complete", flush=True)
    return pd.concat(rows, ignore_index=True), folds


def blocked(frame, values, weights, seed):
    rng = np.random.default_rng(seed)
    blocks = pd.DataFrame(np.column_stack([values, weights])).assign(
        season=frame.season.to_numpy(), week=frame.week.to_numpy()
    )
    grouped = blocks.groupby(["season", "week"])
    sums = grouped.sum().to_numpy(float)
    years = grouped.size().index.get_level_values(0).to_numpy()
    seasons = np.unique(years)
    indexes = {year: np.flatnonzero(years == year) for year in seasons}
    columns = values.shape[1]
    samples = np.empty((shared.BOOTSTRAPS, columns))
    for draw in range(shared.BOOTSTRAPS):
        chosen = np.concatenate(
            [
                rng.choice(indexes[year], len(indexes[year]), replace=True)
                for year in rng.choice(seasons, len(seasons), replace=True)
            ]
        )
        total = sums[chosen].sum(axis=0)
        samples[draw] = total[:columns] / total[columns:]
    means = values.sum(axis=0) / weights.sum(axis=0)
    return means, samples, len(grouped)


def cells(means, draws, arms, metrics, signs):
    width = len(metrics)
    output = {}
    for index, arm in enumerate(arms):
        output[arm] = {
            metric: shared.estimate(means[index * width + j], draws[:, index * width + j])
            for j, metric in enumerate(metrics)
        }
    for index, arm in enumerate(arms[1:], start=1):
        output[f"gain_vs_{arm}"] = {
            metric: shared.estimate(
                signs[j] * (means[j] - means[index * width + j]),
                signs[j] * (draws[:, j] - draws[:, index * width + j]),
            )
            for j, metric in enumerate(metrics)
        }
    return output


def phase_panel(frame, arms, metrics, signs, values_of, weights_of, seed):
    result = {"phases": {}}
    means, draws = {}, {}
    for phase, offset in (("IS", 0), ("OOS", 100)):
        selected = frame.loc[frame.phase.eq(phase)].reset_index(drop=True)
        values, weights = values_of(selected), weights_of(selected)
        means[phase], draws[phase], blocks = blocked(selected, values, weights, seed + offset)
        result["phases"][phase] = cells(means[phase], draws[phase], arms, metrics, signs)
        result["phases"][phase]["games"] = len(selected)
        result.setdefault("blocks", {})[phase] = blocks
        result["last_games"] = (
            int(selected.is_last_game.sum()) if "is_last_game" in selected else None
        )
    result["gap"] = cells(
        means["OOS"] - means["IS"], draws["OOS"] - draws["IS"], arms, metrics, signs
    )
    return result


def total_values(frame):
    columns = []
    for arm in TOTAL_ARMS:
        error = frame[arm + "_error"].to_numpy(float)
        columns += [error * frame.is_last_game.to_numpy(float), error]
    return np.column_stack(columns)


def total_weights(frame):
    last = frame.is_last_game.to_numpy(float)
    return np.column_stack([last, np.ones(len(frame))] * len(TOTAL_ARMS))


def ats_frames(predictions, outer):
    selected = predictions if outer == "pooled" else predictions.loc[predictions.outer.eq(outer)]
    return selected


def ats_panel(selected, seed):
    result = {"phases": {}}
    means, draws = {}, {}
    for phase, offset in (("IS", 0), ("OOS", 100)):
        frame = selected.loc[selected.phase.eq(phase)]
        keys = (
            frame.loc[frame.arm.eq("candidate"), ["outer", "game_id", "season", "week"]]
            .sort_values(["outer", "game_id"])
            .reset_index(drop=True)
        )
        matrices = []
        for arm in ATS_ARMS:
            group = frame.loc[frame.arm.eq(arm)].sort_values(["outer", "game_id"])
            if (
                not group[["outer", "game_id"]]
                .reset_index(drop=True)
                .equals(keys[["outer", "game_id"]])
            ):
                raise ValueError("Unpaired ATS arm populations")
            matrices.append(group[list(shared.METRICS)].to_numpy(float))
        values = np.column_stack(matrices)
        weights = np.ones_like(values)
        means[phase], draws[phase], blocks = blocked(keys, values, weights, seed + offset)
        result["phases"][phase] = cells(
            means[phase], draws[phase], ATS_ARMS, shared.METRICS, ATS_SIGNS
        )
        result["phases"][phase]["games"] = len(keys)
        for arm in ATS_ARMS:
            group = frame.loc[frame.arm.eq(arm)]
            wins = int(group.accuracy_points.eq(100).sum())
            result["phases"][phase][arm]["record"] = f"{wins}-{len(group) - wins}"
        result.setdefault("blocks", {})[phase] = blocks
    result["gap"] = cells(
        means["OOS"] - means["IS"], draws["OOS"] - draws["IS"], ATS_ARMS, shared.METRICS, ATS_SIGNS
    )
    return result


def decisive_ats(selected):
    oos = selected.loc[selected.phase.eq("OOS")]
    reference = (
        oos.loc[oos.arm.eq("candidate")].sort_values(["outer", "game_id"]).reset_index(drop=True)
    )
    output = {}
    for arm in ATS_ARMS[1:]:
        other = oos.loc[oos.arm.eq(arm)].sort_values(["outer", "game_id"]).reset_index(drop=True)
        different = reference.probability.ge(0.5).ne(other.probability.ge(0.5))
        wins = int(reference.loc[different, "accuracy_points"].eq(100).sum())
        n = int(different.sum())
        output[arm] = {
            "games": n,
            "wins": wins,
            "losses": n - wins,
            "exact_p": float(binomtest(wins, n).pvalue) if n else 1.0,
        }
    return output


def decisive_totals(selected):
    oos = selected.loc[selected.phase.eq("OOS")]
    output = {}
    for population, frame in (("last", oos.loc[oos.is_last_game]), ("all", oos)):
        for arm in TOTAL_ARMS[1:]:
            gain = frame[arm + "_error"] - frame.candidate_error
            closer, worse, tied = (
                int(gain.gt(0).sum()),
                int(gain.lt(0).sum()),
                int(gain.eq(0).sum()),
            )
            decisive = closer + worse
            output[f"{population}_{arm}"] = {
                "closer": closer,
                "worse": worse,
                "tied": tied,
                "exact_p": float(binomtest(closer, decisive).pvalue) if decisive else 1.0,
            }
    return output


def summarize(totals, predictions):
    panels = {}
    for outer in (*OUTERS, "pooled"):
        total_selected = totals if outer == "pooled" else totals.loc[totals.outer.eq(outer)]
        ats_selected = ats_frames(predictions, outer)
        seed = SEED + (0 if outer == "pooled" else outer)
        panels[str(outer)] = {
            "total": phase_panel(
                total_selected,
                TOTAL_ARMS,
                TOTAL_ENDPOINTS,
                TOTAL_SIGNS,
                total_values,
                total_weights,
                seed + 500,
            ),
            "total_decisive": decisive_totals(total_selected),
            "ats": ats_panel(ats_selected, seed),
            "ats_decisive": decisive_ats(ats_selected),
        }
        print(f"intervals complete: {outer}", flush=True)
    reliability = []
    oos = predictions.loc[predictions.phase.eq("OOS")]
    for arm in ATS_ARMS:
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


def record_commands(summary):
    lines = [
        "```bash",
        "while read -r panel first last games blocks units effect low high pp; do",
        ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record "
        '--name "lead94_unit1_$panel-$units" --league nfl \\',
        '  --description "Final-game total guess weighted toward final-game-like games versus'
        ' the comparator: $panel" --source docs/lead94_unit1.md --family '
        "lead94_unit1_717_looks \\",
        '  --season-start "$first" --season-end "$last" --sample-games "$games" '
        '--sample-blocks "$blocks" --effect-units "$units" \\',
        '  --effect "$effect" --interval-low "$low" --interval-high "$high" '
        '--probability-positive "$pp" --classification unresolved_below_power \\',
        '  --classification-evidence "Three held-out seasons; no refuted mechanism or powered '
        'control. All 717 declared looks reported." \\',
        '  --plain-summary "This lets the last game of the week count more when learning how '
        "totals move after Tuesday, then guesses its combined score. Picks against the spread "
        'are unchanged; this is a research result only."',
        "done <<'CELLS'",
    ]
    for label, panel in summary["panels"].items():
        first, last = (2023, 2025) if label == "pooled" else (int(label), int(label))
        oos = panel["total"]["phases"]["OOS"]
        last_games = panel["total"]["last_games"]
        for arm in TOTAL_ARMS[1:]:
            for endpoint, games in (("last_mae", last_games), ("all_mae", oos["games"])):
                cell = oos[f"gain_vs_{arm}"][endpoint]
                values = " ".join(
                    f"{cell[key]:.12g}"
                    for key in ("estimate", "low", "high", "probability_positive")
                )
                lines.append(
                    f"{label}-{arm}-{endpoint} {first} {last} {games} "
                    f"{panel['total']['blocks']['OOS']} "
                    f"mae_improvement {values}"
                )
    return "\n".join([*lines, "CELLS", "```"])


def cell_text(cell):
    return f"{shared.number(cell)}; pp={cell['probability_positive']:.5f}"


def report(summary, folds, lineage, declaration, supplement):
    panels = summary["panels"]
    pooled = panels["pooled"]
    oos = pooled["total"]["phases"]["OOS"]
    is_ = pooled["total"]["phases"]["IS"]
    gap = pooled["total"]["gap"]
    lines = [
        "# LEAD-94 unit 1: last-game tiebreaker total population weighting",
        "",
        "## Primary result: last-game total MAE",
        "",
        f"**Measured:** {oos['games']} outer games, {pooled['total']['last_games']} last games "
        "(pooled 2023-2025 held-out seasons). Positive gain means the candidate total is "
        "closer than the control. Intervals are 10,000 paired season/week-block draws "
        f"(seed {SEED}); probability_positive gives exact bootstrap ties half credit.",
        "",
        shared.table(
            ["Arm or contrast", "OOS last MAE", "IS last MAE", "OOS minus IS gap"],
            [
                [
                    label,
                    cell_text(oos[label]["last_mae"]),
                    cell_text(is_[label]["last_mae"]),
                    cell_text(gap[label]["last_mae"]),
                ]
                for label in (*TOTAL_ARMS, *(f"gain_vs_{a}" for a in TOTAL_ARMS[1:]))
            ],
        ),
        "",
        "## Decisive records first",
        "",
        "**Measured:** total-guess records count games where absolute errors differ "
        "(OOS pooled); exact fair-coin two-sided p. ATS sides are identical to the "
        "four-term recipe, so the candidate has zero decisive ATS games against it.",
        "",
        shared.table(
            ["Population and control", "Closer", "Worse", "Tied", "Exact p"],
            [
                [
                    key,
                    cell["closer"],
                    cell["worse"],
                    cell["tied"],
                    f"{cell['exact_p']:.6f}",
                ]
                for key, cell in pooled["total_decisive"].items()
            ],
        ),
        "",
        shared.table(
            ["ATS comparator", "Decisive games", "Candidate W-L", "Exact p"],
            [
                [arm, c["games"], f"{c['wins']}-{c['losses']}", f"{c['exact_p']:.6f}"]
                for arm, c in pooled["ats_decisive"].items()
            ],
        ),
        "",
        "## Effective sample size, weights and coefficients",
        "",
        "**Measured:** ESS = (sum w)^2 / sum w^2, over training games and over the "
        "moving-total loss weights w|move|. Weights are capped at 10 and use no outcome.",
        "",
        shared.table(
            [
                "Outer",
                "Train",
                "Train last",
                "ESS games",
                "ESS loss",
                "Weight range",
                "Capped",
                "Weighted b",
                "Unweighted b (diagnostic)",
                "Intercept shift",
                "Training median",
            ],
            [
                [
                    f["outer"],
                    f["train_games"],
                    f["train_last_games"],
                    f"{f['ess_games']:.2f}",
                    f"{f['ess_loss']:.2f}",
                    f"{f['weight_min']:.4f}-{f['weight_max']:.4f}",
                    f["membership"]["capped"],
                    f"{f['weighted_b']:.6f}",
                    f"{f['unweighted_b']:.6f}",
                    f"{f['intercept_shift']:.4f}",
                    f"{f['training_median']:.2f}",
                ]
                for f in folds
            ],
        ),
        "",
        shared.table(
            ["Outer", "Slots", "Class prior", "Max uncapped ratio", "Coefficients"],
            [
                [
                    f["outer"],
                    f["membership"]["slots"],
                    f"{f['membership']['prior']:.5f}",
                    f"{f['membership']['max_ratio']:.3f}",
                    "; ".join(f"{k}={v:.4f}" for k, v in f["membership"]["coefficients"].items()),
                ]
                for f in folds
            ],
        ),
        "",
        "**Inferred:** coefficient stability is judged by how far the weighted b sits "
        "from the unweighted b and from 1 in each fold; three folds cannot establish "
        "stability.",
        "",
        "## IS, OOS and gaps by panel",
        "",
        "**Measured:** each cell is estimate [95% interval]; probability_positive. MAE and "
        "loss rows are raw losses; gain rows orient so positive favors the candidate; "
        "gaps are OOS minus IS. IS is optimistic and repeated training games share blocks "
        "across pooled folds.",
    ]
    for outer, panel in panels.items():
        total = panel["total"]
        rows = []
        for phase, block in (
            ("IS", total["phases"]["IS"]),
            ("OOS", total["phases"]["OOS"]),
            ("gap", total["gap"]),
        ):
            for label, measures in block.items():
                if label == "games":
                    continue
                rows.append([phase, label, *[cell_text(measures[e]) for e in TOTAL_ENDPOINTS]])
        ats = panel["ats"]
        ats_rows = []
        for phase, block in (
            ("IS", ats["phases"]["IS"]),
            ("OOS", ats["phases"]["OOS"]),
            ("gap", ats["gap"]),
        ):
            for label, measures in block.items():
                if label == "games":
                    continue
                ats_rows.append(
                    [
                        phase,
                        label,
                        measures.get("record", "-"),
                        *[cell_text(measures[m]) for m in shared.METRICS],
                    ]
                )
        lines += [
            "",
            f"### {outer}: totals",
            "",
            shared.table(["Phase", "Arm/contrast", *TOTAL_ENDPOINTS], rows),
            "",
            f"### {outer}: ATS diagnostics (candidate sides equal four-term sides)",
            "",
            shared.table(["Phase", "Arm/contrast", "W-L", *shared.METRICS], ats_rows),
        ]
    lines += [
        "",
        "## Five equal-width reliability bands",
        "",
        "**Measured:** home-cover probabilities on non-push outer rows; empty cells stay empty.",
        "",
        shared.table(
            ["Arm", "Band", "Games", "Mean probability", "Home cover rate"],
            [
                [
                    r["arm"],
                    r["band"],
                    r["n"],
                    "-" if r["predicted"] is None else f"{r['predicted']:.6f}",
                    "-" if r["observed"] is None else f"{r['observed']:.6f}",
                ]
                for r in summary["reliability"]
            ],
        ),
        "",
        "## Inference, limits and looks",
        "",
        f"**Measured:** {LOOKS} declared looks: (27*6+7+4)*(3+1)+25. Total endpoints use "
        f"the {lineage['total_population']}-game paired population with pushes retained; "
        f"ATS diagnostics use the {lineage['ats_intersection_games']}-game non-push "
        "intersection with the certified LEAD-83 refits (hashes and upstream cutoffs "
        "verified in the run).",
        "**Inferred:** classification is unresolved_below_power pending serial recording. "
        "Zero crossing closes nothing; no refuted mechanism or powered control is "
        "established. A totals gain cannot promote ATS sides (AGENTS.md, Margins/Promotion), "
        "and this study changes no pick. Only three outer seasons, a 2020-only first "
        "training fold, retrospective archive reuse and intervals conditional on fitted "
        "weights limit inference. The discrete total law is centred at the equation, so its "
        "median equals the centre and the served rounding rule decides the integer.",
        f"Prediction rows: {OUTPUT}/totals.parquet, ATS rows reused from "
        f"{PREVIOUS}/predictions.parquet. Command: .tools/uv.exe run --no-sync --no-cache "
        "python scripts/lead94_unit1.py",
        "",
        "## Frozen protocol",
        "",
        declaration.strip(),
        "",
        supplement.strip(),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise ValueError("A scored replay already exists; do not silently rescore")
    lane = LANE.read_text(encoding="utf-8")
    (OUTPUT / "protocol.md").write_text(lane, encoding="utf-8")
    declaration = lane.split("## Protocol (verbatim declaration)\n", 1)[1].split("\n## Tried", 1)[0]
    supplement = PROTOCOL.read_text(encoding="utf-8")
    with threadpool_limits(limits=1):
        pairs, predictions, lineage = load()
        totals, folds = replay_totals(pairs)
        pq.write_table(
            pa.Table.from_pandas(totals, preserve_index=False), OUTPUT / "totals.parquet"
        )
        summary = summarize(totals, predictions)
        summary.update({"lineage": lineage, "folds": folds, "looks": LOOKS, "seed": SEED})
        commands = record_commands(summary)
        (OUTPUT / "record_commands.md").write_text(commands, encoding="utf-8")
        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        report(summary, folds, lineage, declaration, supplement)
    pooled = summary["panels"]["pooled"]["total"]["phases"]["OOS"]
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate": pooled["candidate"],
                "gain_vs_tuesday": pooled["gain_vs_tuesday"],
                "gain_vs_deadline": pooled["gain_vs_deadline"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
