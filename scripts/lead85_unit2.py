from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from best_pick_sunday_renomination_eval import select_with_tie_rule
from threadpoolctl import threadpool_limits

from nfl_ats.best_pick_nomination import dispersion_pool_from_frame
from nfl_ats.pool_workbench import PoolRules

SOURCE = Path("tests/scratch/codex/lead85_unit1")
OUTPUT = Path("tests/scratch/codex/lead85_unit2")
DISPERSION = Path("artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet")
LANE = Path("docs/lanes/lead85.md")
REPORT = Path("docs/lead85_unit2.md")
ARMS = ("four_term", "integrated")
FOLDS = (2023, 2024, 2025)
SAMPLES = 10000
SEED = 85
LOOKS = 741


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frame(path, columns=None):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def load():
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    if digest("scripts/lead85_unit1.py") != summary["script_sha256"]:
        raise ValueError("Unit 1 script no longer matches the frozen fit")
    columns = [
        "game_id",
        "season",
        "week",
        "fold",
        "split",
        "tue_open_home_spread",
        "margin_vs_open",
        "kickoff",
        "nomination_cutoff",
        "nomination_eligible",
        "train_max_gameday",
        "gameday",
        "four_term_probability",
        "integrated_probability",
        "four_term_push",
        "integrated_push",
    ]
    frame = read_frame(SOURCE / "predictions.parquet", columns)
    if frame.duplicated(["fold", "split", "game_id"]).any():
        raise ValueError("Duplicate frozen fold/game prediction")
    if set(frame.fold) != set(FOLDS) or set(frame.split) != {"IS", "OOS"}:
        raise ValueError("Unexpected frozen folds or splits")
    upstream = pd.to_datetime(frame.train_max_gameday, utc=True)
    dates = pd.to_datetime(frame.gameday, utc=True)
    if upstream.isna().any() or not upstream.lt(dates).all():
        raise ValueError("Upstream training cutoff reaches target date")
    frame["kickoff"] = pd.to_datetime(frame.kickoff, utc=True)
    frame["nomination_cutoff"] = pd.to_datetime(frame.nomination_cutoff, utc=True)
    if not frame.nomination_eligible.eq(frame.kickoff.gt(frame.nomination_cutoff)).all():
        raise ValueError("Frozen eligibility differs from the pre-kickoff rule")
    if not frame.season.between(2020, 2025).all():
        raise ValueError("Nondeclared season")
    oos, ins = frame.split.eq("OOS"), frame.split.eq("IS")
    if not frame.loc[oos, "season"].eq(frame.loc[oos, "fold"]).all():
        raise ValueError("Outer season is not held out")
    if not frame.loc[ins, "season"].le(frame.loc[ins, "fold"] - 3).all():
        raise ValueError("IS rows include reserved seasons")
    for coefficient in summary["coefficients"]:
        fold = coefficient["fold"]
        if (
            max(coefficient["train_seasons"]) > fold - 3
            or coefficient["tuning_season_reserved"] != fold - 2
            or coefficient["calibration_season"] != fold - 1
            or min(coefficient["inverse_temperatures"].values()) <= 0
        ):
            raise ValueError("Frozen coefficient/calibration chronology failed")
    names = [f"{arm}_{suffix}" for arm in ARMS for suffix in ("probability", "push")]
    values = frame[names].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid frozen discrete probabilities")
    if not np.allclose(frame.four_term_push, frame.integrated_push, atol=1e-12, rtol=0):
        raise ValueError("Candidate changed push mass")
    if not frame.four_term_probability.ge(0.5).eq(frame.integrated_probability.ge(0.5)).all():
        raise ValueError("Unexpected candidate side change")
    max_error = 0.0
    for fold in FOLDS:
        target = frame.loc[frame.fold.eq(fold) & oos].set_index("game_id")
        for arm in ARMS:
            with np.load(SOURCE / f"pmf_{fold}_{arm}.npz") as saved:
                ids, grid, mass = saved["game_ids"], saved["grid"], saved["mass"]
            if set(ids) != set(target.index) or not np.allclose(mass.sum(axis=1), 1):
                raise ValueError("Frozen PMF membership or normalization failed")
            rows = target.loc[ids]
            line = rows.tue_open_home_spread.to_numpy(float)[:, None]
            push = (mass * (line == grid)).sum(axis=1)
            conditional = (mass * (line < grid)).sum(axis=1) / (1 - push)
            error = max(
                np.max(np.abs(push - rows[f"{arm}_push"].to_numpy(float))),
                np.max(np.abs(conditional - rows[f"{arm}_probability"].to_numpy(float))),
            )
            max_error = max(max_error, float(error))
    if max_error > 1e-12:
        raise ValueError("Frozen rows differ from their discrete PMFs")
    dispersion = read_frame(DISPERSION, ["game_id", "spread_std"])
    frame = frame.merge(dispersion, on="game_id", how="left", validate="many_to_one")
    frame["tie_dispersion"] = frame.spread_std
    return frame, summary, max_error


def nominate(frame):
    rows, selected_rows = [], []
    weight = PoolRules().push_points
    if weight != 0.5:
        raise ValueError("Pool push rule changed after declaration")
    for (fold, split, season, week), group in frame.groupby(
        ["fold", "split", "season", "week"], sort=True
    ):
        available = group.loc[group.nomination_eligible].copy()
        if available.empty:
            raise ValueError("No eligible weekly contender")
        pool = dispersion_pool_from_frame(available[["game_id", "spread_std"]])
        passed = pool.frame.loc[pool.frame.pool_pass, "game_id"]
        eligible = available.loc[available.game_id.isin(passed)].copy()
        row = {
            "fold": int(fold),
            "split": split,
            "season": int(season),
            "week": int(week),
            "available": len(available),
            "eligible": len(eligible),
            "pool_fallback": pool.fallback,
            "pool_fallback_reason": pool.fallback_reason,
        }
        choices = {}
        for arm in ARMS:
            p, push = eligible[f"{arm}_probability"], eligible[f"{arm}_push"]
            eligible[f"{arm}_rank"] = np.maximum(p, 1 - p) * (1 - push) + weight * push
            ids, tie = select_with_tie_rule(eligible, f"{arm}_rank")
            choices[arm] = ids
            chosen = eligible.set_index("game_id").loc[list(ids)]
            p = chosen[f"{arm}_probability"].to_numpy(float)
            side = p >= 0.5
            margin = chosen.margin_vs_open.to_numpy(float)
            if not np.isfinite(margin).all():
                raise ValueError("Missing frozen opener outcome")
            signed = np.where(side, margin, -margin)
            win, push = signed > 0, signed == 0
            cover = np.maximum(p, 1 - p)
            brier = np.where(push, 0, (cover - win) ** 2)
            row.update(
                {
                    f"{arm}_ids": ",".join(ids),
                    f"{arm}_sides": ",".join(np.where(side, "home", "away")),
                    f"{arm}_tie": tie,
                    f"{arm}_win": float(win.mean()),
                    f"{arm}_push": float(push.mean()),
                    f"{arm}_loss": float((signed < 0).mean()),
                    f"{arm}_reward": float((win + weight * push).mean()),
                    f"{arm}_brier_sum": float(brier.mean()),
                    f"{arm}_nonpush": float((~push).mean()),
                }
            )
            for i, game_id in enumerate(ids):
                selected_rows.append(
                    {
                        "fold": int(fold),
                        "split": split,
                        "season": int(season),
                        "week": int(week),
                        "arm": arm,
                        "game_id": game_id,
                        "home": bool(side[i]),
                        "probability": float(cover[i]),
                        "win": bool(win[i]),
                        "push": bool(push[i]),
                        "weight": 1 / len(ids),
                        "rank": float(chosen.iloc[i][f"{arm}_rank"]),
                        "frozen_home_spread": float(chosen.iloc[i].tue_open_home_spread),
                    }
                )
        row["changed"] = choices[ARMS[0]] != choices[ARMS[1]]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def interval(draws):
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "se": float(np.std(draws, ddof=1)),
        "probability_positive": float(
            (draws > 1e-12).mean() + 0.5 * (np.abs(draws) <= 1e-12).mean()
        ),
    }


def totals_to_metrics(values):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            [
                values[..., 0] / values[..., 4],
                values[..., 1] / values[..., 5],
                100 * values[..., 2] / values[..., 6],
                100 * values[..., 3] / values[..., 6],
            ],
            axis=-1,
        )


def bootstrap(frame, seed, reward_only=False):
    columns = [
        "four_term_brier_sum",
        "integrated_brier_sum",
        "four_term_reward",
        "integrated_reward",
        "four_term_nonpush",
        "integrated_nonpush",
    ]
    values = frame[columns].copy()
    values["weeks"] = 1.0
    values["season"], values["week"] = frame.season, frame.week
    aggregate = values.groupby(["season", "week"]).sum()
    blocks = [group.to_numpy(float) for _, group in aggregate.groupby(level="season")]
    rng = np.random.default_rng(seed)
    selected_seasons = rng.integers(len(blocks), size=(SAMPLES, len(blocks)))
    totals = np.zeros((SAMPLES, 7))
    for index, block in enumerate(blocks):
        draw, _ = np.nonzero(selected_seasons == index)
        samples = rng.integers(len(block), size=(len(draw), len(block)))
        np.add.at(totals, draw, block[samples].sum(axis=1))
    result = totals_to_metrics(aggregate.to_numpy().sum(axis=0))
    draws = totals_to_metrics(totals)
    if not np.isfinite(draws[:, 2:] if reward_only else draws).all():
        raise ValueError("A bootstrap draw lacks scored nominees")
    return result, draws


def metric_cells(weekly):
    cells = []
    for panel_index, panel in enumerate((*FOLDS, "pooled")):
        subset = weekly if panel == "pooled" else weekly.loc[weekly.fold.eq(panel)]
        results = {}
        for split_index, split in enumerate(("IS", "OOS")):
            results[split] = bootstrap(
                subset.loc[subset.split.eq(split)], SEED + panel_index * 2 + split_index
            )
        results["gap"] = (
            results["OOS"][0] - results["IS"][0],
            results["OOS"][1] - results["IS"][1],
        )
        for split, (point, draws) in results.items():
            for metric, offset, sign in (("brier", 0, -1), ("reward_points", 2, 1)):
                for index, arm in enumerate(ARMS):
                    cells.append(
                        {
                            "panel": panel,
                            "split": split,
                            "metric": metric,
                            "arm": arm,
                            "effect": float(point[offset + index]),
                            **interval(draws[:, offset + index]),
                        }
                    )
                contrast = sign * (point[offset + 1] - point[offset])
                delta = sign * (draws[:, offset + 1] - draws[:, offset])
                cells.append(
                    {
                        "panel": panel,
                        "split": split,
                        "metric": metric,
                        "arm": "improvement",
                        "effect": float(contrast),
                        **interval(delta),
                    }
                )
    return cells


def wilson(wins, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p, denominator = wins / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"rate": 100 * p, "low": 100 * (center - radius), "high": 100 * (center + radius)}


def exact_null(frame):
    delta = frame.integrated_reward.to_numpy() - frame.four_term_reward.to_numpy()
    fractions = [Fraction(float(value)).limit_denominator(1000000) for value in delta]
    if any(abs(float(a) - b) > 1e-12 for a, b in zip(fractions, delta, strict=True)):
        raise ValueError("Cannot represent tied nominee rewards exactly")
    scale = math.lcm(*(value.denominator for value in fractions)) if fractions else 1
    values = [int(value * scale) for value in fractions if value]
    distribution = {0: 1}
    for value in values:
        following = defaultdict(int)
        for total, count in distribution.items():
            following[total + abs(value)] += count
            following[total - abs(value)] += count
        distribution = following
    threshold = abs(sum(values))
    extreme = sum(count for total, count in distribution.items() if abs(total) >= threshold)
    return {
        "p_two_sided": extreme / (2 ** len(values)),
        "discordant_rewards": len(values),
        "candidate_better": int((delta > 0).sum()),
        "candidate_worse": int((delta < 0).sum()),
        "equal": int((delta == 0).sum()),
    }


def records(weekly):
    rows = []
    oos = weekly.loc[weekly.split.eq("OOS")]
    for panel_index, panel in enumerate((*FOLDS, "pooled")):
        frame = oos if panel == "pooled" else oos.loc[oos.fold.eq(panel)]
        for population in ("all", "changed"):
            selected = frame if population == "all" else frame.loc[frame.changed]
            row = {
                "panel": panel,
                "population": population,
                "weeks": len(selected),
                **exact_null(selected),
            }
            for arm in ARMS:
                wins, pushes, losses = [
                    float(selected[f"{arm}_{field}"].sum()) for field in ("win", "push", "loss")
                ]
                row[arm] = {
                    "wins": wins,
                    "pushes": pushes,
                    "losses": losses,
                    "wilson": wilson(wins, wins + losses),
                }
            if len(selected):
                point, draws = bootstrap(selected, SEED + panel_index * 2 + 1, reward_only=True)
                row["reward_difference"] = {
                    "effect": float(point[3] - point[2]),
                    **interval(draws[:, 3] - draws[:, 2]),
                }
            else:
                row["reward_difference"] = None
            rows.append(row)
    return rows


def reliability(selected):
    rows = []
    for arm in ARMS:
        frame = selected.loc[selected.split.eq("OOS") & selected.arm.eq(arm) & ~selected.push]
        bins = np.minimum((frame.probability.to_numpy() * 5).astype(int), 4)
        for band in range(5):
            cell = frame.loc[bins == band]
            weight = float(cell.weight.sum())
            rows.append(
                {
                    "arm": arm,
                    "band": f"{band / 5:.1f}-{(band + 1) / 5:.1f}",
                    "n": weight,
                    "predicted": float(np.average(cell.probability, weights=cell.weight))
                    if weight
                    else None,
                    "observed": float(np.average(cell.win, weights=cell.weight))
                    if weight
                    else None,
                }
            )
    return rows


def number(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def estimate(row):
    if row is None:
        return "n/a"
    return (
        f"{row['effect']:+.8f} [{row['low']:+.8f},{row['high']:+.8f}]; "
        f"P+ {row['probability_positive']:.4f}"
    )


def record_text(row):
    return f"{row['wins']:g}-{row['losses']:g}-{row['pushes']:g}"


def wilson_text(row):
    interval = row["wilson"]
    if interval is None:
        return "n/a"
    return f"{interval['rate']:.2f}% [{interval['low']:.2f},{interval['high']:.2f}]"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    )


def record_commands(summary):
    commands = []
    for metric, units, label, player_text in (
        (
            "brier",
            "brier_improvement",
            "nominee_brier",
            "Allowing for uncertainty changed the probability score for the weekly Best Pick; "
            "the season replay is too small to settle the benefit.",
        ),
        (
            "reward_points",
            "accuracy_points",
            "weekly_reward",
            "Allowing for uncertainty can change which game receives the weekly Best Pick; wins "
            "count one point and ties count half a point.",
        ),
    ):
        row = next(
            r
            for r in summary["metrics"]
            if r["panel"] == "pooled"
            and r["split"] == "OOS"
            and r["metric"] == metric
            and r["arm"] == "improvement"
        )
        error_arg = f"--standard-error {row['se']:.15g} " if row["se"] > 0 else ""
        commands.append(
            ".tools/uv.exe run --no-sync nfl-ats weak-signals record "
            f"--name lead85_unit2_{label}_2023_2025 --description 'Uncertainty averaged versus "
            f"four-term weekly Best Pick' "
            f"--source docs/lead85_unit2.md --effect-units {units} --effect {row['effect']:.15g} "
            f"{error_arg}--interval-low {row['low']:.15g} --interval-high {row['high']:.15g} "
            f"--probability-positive {row['probability_positive']:.15g} --sample-blocks "
            f"{summary['oos_weeks']} "
            "--classification unresolved_below_power --league nfl --season-start 2023 "
            "--season-end 2025 "
            f"--family lead85_weekly_{label} --classification-evidence 'No admissible closing "
            f"ground established' "
            "--notes '741 charged study looks, 110 unit-2 cells; paired hierarchical season/week "
            "bootstrap; "
            "nominee study only, do not pool with ordinary game accuracy; exact weekly "
            "label-swap null in source' "
            f"--plain-summary '{player_text}'"
        )
    return commands


def render(summary, weekly):
    record_rows = summary["records"]
    changed = weekly.loc[weekly.split.eq("OOS") & weekly.changed]
    metric_rows = []
    for panel in (*FOLDS, "pooled"):
        for split in ("IS", "OOS", "gap"):
            for metric in ("brier", "reward_points"):
                cells = {
                    r["arm"]: r
                    for r in summary["metrics"]
                    if r["panel"] == panel and r["split"] == split and r["metric"] == metric
                }
                metric_rows.append(
                    [
                        panel,
                        split,
                        metric,
                        number(cells["four_term"]["effect"], 8),
                        number(cells["integrated"]["effect"], 8),
                        estimate(cells["improvement"]),
                    ]
                )
    coefficient_rows = []
    for c in summary["coefficients"]:
        coefficient_rows.append(
            [
                c["fold"],
                ",".join(map(str, c["train_seasons"])),
                *[
                    number(c["coefficients"][name])
                    for name in (
                        "model_logit",
                        "composition_flag_sum",
                        "market_move_toward_home",
                        "market_move_available",
                        "intercept",
                    )
                ],
                f"{c['inverse_temperatures']['four_term']:.6f}/{c['inverse_temperatures']['integrated']:.6f}",
            ]
        )
    lines = [
        "# LEAD-85 unit 2: weekly Best Pick replay",
        "",
        "**Measured:** Command: .tools/uv.exe run --no-sync python scripts/lead85_unit2.py. "
        "The unit-2 declaration was saved in the lane before nomination outcomes; its unchanged "
        "snapshot and hashes are in "
        "tests/scratch/codex/lead85_unit2/. All intervals below are 95%.",
        "",
        "## Changed nominations first",
        "",
        "**Measured:** records are wins-losses-pushes. Wilson intervals describe cover rate "
        "among nonpush nominees; "
        "pool reward credits pushes 0.5. Probability_positive is abbreviated P+ in tables.",
        "",
        table(
            [
                "Season",
                "Changed weeks",
                "Served W-L-P",
                "Candidate W-L-P",
                "Better/worse/equal",
                "Reward change pp [interval]; P+",
                "Exact p",
            ],
            [
                [
                    r["panel"],
                    r["weeks"],
                    record_text(r["four_term"]),
                    record_text(r["integrated"]),
                    f"{r['candidate_better']}/{r['candidate_worse']}/{r['equal']}",
                    estimate(r["reward_difference"]),
                    number(r["p_two_sided"]),
                ]
                for r in record_rows
                if r["population"] == "changed"
            ],
        ),
        "",
        "**Measured:** the following identifiers locate changed weeks; full "
        "nomination/prediction rows remain in scratch.",
        "",
        table(
            [
                "Season/week",
                "Served nominee (side)",
                "Candidate nominee (side)",
                "Served/candidate reward",
            ],
            [
                [
                    f"{r.season}/{r.week}",
                    f"{r.four_term_ids} ({r.four_term_sides})",
                    f"{r.integrated_ids} ({r.integrated_sides})",
                    f"{r.four_term_reward:g}/{r.integrated_reward:g}",
                ]
                for r in changed.itertuples()
            ],
        ),
        "",
        "## Nominee cover rates",
        "",
        "**Measured:** Wilson intervals treat weekly nonpush outcomes as Bernoulli observations; "
        "season dependence is handled "
        "separately by the paired hierarchical intervals. Unresolved selection ties have equal "
        "weights.",
        "",
        table(
            [
                "Season",
                "Population",
                "Weeks",
                "Served W-L-P",
                "Served Wilson",
                "Candidate W-L-P",
                "Candidate Wilson",
                "Exact p",
            ],
            [
                [
                    r["panel"],
                    r["population"],
                    r["weeks"],
                    record_text(r["four_term"]),
                    wilson_text(r["four_term"]),
                    record_text(r["integrated"]),
                    wilson_text(r["integrated"]),
                    number(r["p_two_sided"]),
                ]
                for r in record_rows
            ],
        ),
        "",
        "## Primary nominee Brier and weekly reward",
        "",
        "**Measured:** positive improvements favor the candidate: served minus candidate for "
        "Brier, candidate minus served for reward. "
        "Reward is in percentage points. Brier conditions on each arm's nonpush nominees; paired "
        "resampling retains every week. "
        "Gap means OOS minus optimistic IS. Pooled IS repeats earlier training weeks across "
        "folds and resamples them together.",
        "",
        table(
            ["Panel", "Split", "Metric", "Served", "Candidate", "Improvement [interval]; P+"],
            metric_rows,
        ),
        "",
        "## Population, calibration and audit",
        "",
        f"**Measured:** {summary['source_games']} source-complete historical opener games; "
        f"{summary['oos_games']} outer-season games; "
        f"{summary['eligible_oos_games']} post-cutoff games before dispersion filtering; "
        f"{summary['oos_weeks']} held-out weeks. "
        f"{summary['pool_fallback_weeks']} dispersion fallback weeks; "
        f"{summary['unresolved_ties']} unresolved arm/week ties. "
        f"Maximum row-versus-discrete-PMF error {summary['max_pmf_error']:.3g}; no game side "
        f"changes.",
        "",
        "**Read:** historical openers are the frozen pool-line proxy; no pre-2026 Splash "
        "captures are used. "
        "Eligibility is kickoff after Sunday 12:45 Eastern from unit 1, followed by the shared "
        "Tuesday dispersion filter. "
        "Ranking is unconditional calibrated cover + 0.5 push, as frozen in "
        "docs/lead85_protocol.md:14 and docs/lanes/lead72.md:29. "
        "Tie order is score rounded to 12 decimals, lowest dispersion, smallest absolute spread, "
        "earliest kickoff, then equal weights. "
        "The one calibrated conditional cover probability chooses each game's side.",
        "",
        "**Measured:** no fitting or parameter selection occurred here. Unit 1 coefficients and "
        "separate-year positive temperatures "
        "are reused unchanged. Fit through Y-3, reserve Y-2, calibrate Y-1, score Y. The "
        "archived training cutoff and saved discrete "
        "PMFs passed replay checks. Source hashes and coefficient intervals are copied to the "
        "scratch summary.",
        "",
        table(
            [
                "Outer",
                "Fit seasons",
                "Model logit",
                "Composition",
                "Market move",
                "Availability",
                "Intercept",
                "Served/candidate inverse temperature",
            ],
            coefficient_rows,
        ),
        "",
        "**Inferred:** coefficient variation is descriptive, not a new fit or stability test; "
        "the source-complete availability term is "
        "constant and unidentified by the likelihood. Its zero coefficient comes from ridge "
        "identification. Only three outer seasons exist.",
        "",
        "**Measured:** five equal-width reliability bands of the nominated side's conditional "
        "cover probability, OOS nonpush only:",
        "",
        table(
            ["Arm", "Band", "Nonpush weight", "Mean probability", "Observed cover"],
            [
                [r["arm"], r["band"], r["n"], number(r["predicted"]), number(r["observed"])]
                for r in summary["reliability"]
            ],
        ),
        "",
        "## Inference, look accounting and limits",
        "",
        "**Measured:** 10,000 paired hierarchical season/week bootstrap draws, base seed 85 with "
        "deterministic panel/split offsets, "
        "percentile intervals and half credit for zero draws. Bootstrap intervals condition on "
        "the saved fits; they do not refit models. "
        "The exact reward null swaps arm labels independently within each week; rational reward "
        "differences are enumerated by convolution, "
        "using the absolute total as the two-sided statistic. Equal-reward weeks contribute no "
        "randomization information. "
        "With no changed weeks, subset estimates are unavailable; a degenerate unchanged-reward "
        "bootstrap is not evidence of tight prospective precision.",
        "",
        "**Read:** 713 study looks were reserved in the original declaration. Unit 2 reports 72 "
        "metric cells "
        "(three arm/contrast cells x two metrics x IS/OOS/gap x four panels), 10 reliability "
        "cells, 16 Wilson cells, "
        "8 exact-null cells and 4 changed-reward contrast cells: 110. The latter 28 supplemental "
        "cells increase charged study looks to 741. "
        "No post-outcome variant or threshold was selected. These correlated cells are not "
        "independent discoveries.",
        "",
        "**Inferred:** unresolved_below_power, pending the orchestrator's serial registry entry. "
        "No admissible closing ground has been established "
        "under AGENTS.md:65-79; probability_positive, not an interval's zero crossing, informs "
        "the estimate. "
        "The original five-arm model-only/dated-market/Elo comparison remains open beyond this "
        "assigned two-arm replay. "
        "Reused historical archives and three seasons do not establish prospective improvement; "
        "no serving decision follows. "
        "Candidate-versus-served record commands are prepared only in the lane.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    declaration = LANE.read_text(encoding="utf-8-sig")
    if not all(
        text in declaration for text in ("Protocol fixed before outcomes", "741", "110", "10,000")
    ):
        raise ValueError("Missing predeclared unit-2 protocol")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "protocol_declaration.md").write_text(declaration, encoding="utf-8")
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with threadpool_limits(limits=2):
        frame, upstream, max_error = load()
        weekly, selected = nominate(frame)
        metrics = metric_cells(weekly)
        record_rows = records(weekly)
        calibration = reliability(selected)
    oos = weekly.loc[weekly.split.eq("OOS")]
    summary = {
        "source_games": int(frame.game_id.nunique()),
        "oos_games": int(frame.loc[frame.split.eq("OOS"), "game_id"].nunique()),
        "eligible_oos_games": int(
            frame.loc[frame.split.eq("OOS") & frame.nomination_eligible, "game_id"].nunique()
        ),
        "oos_weeks": len(oos),
        "changed_weeks": int(oos.changed.sum()),
        "pool_fallback_weeks": int(oos.pool_fallback.sum()),
        "unresolved_ties": int(sum(oos[f"{arm}_tie"].eq(4).sum() for arm in ARMS)),
        "max_pmf_error": max_error,
        "metrics": metrics,
        "records": record_rows,
        "reliability": calibration,
        "coefficients": upstream["coefficients"],
        "unit1_lineage": upstream["lineage"],
        "charged_looks": LOOKS,
        "unit2_looks": 110,
        "hashes": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                SOURCE / "predictions.parquet",
                SOURCE / "summary.json",
                DISPERSION,
                Path("scripts/best_pick_sunday_renomination_eval.py"),
                Path("src/nfl_ats/best_pick_nomination.py"),
                Path("src/nfl_ats/pool_workbench.py"),
                OUTPUT / "protocol_declaration.md",
            )
        },
        "registry_status": "prepared_not_recorded",
        "classification": "unresolved_below_power",
    }
    summary["record_commands"] = record_commands(summary)
    weekly.to_parquet(OUTPUT / "weekly.parquet", index=False)
    selected.to_parquet(OUTPUT / "nominees.parquet", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render(summary, weekly)
    for row in record_rows:
        if row["panel"] == "pooled":
            print(json.dumps(row), flush=True)
    for row in metrics:
        if row["panel"] == "pooled" and row["split"] == "OOS" and row["arm"] == "improvement":
            print(json.dumps(row), flush=True)
    print(
        f"Saved {REPORT}; outer weeks={len(oos)}; changed={summary['changed_weeks']}; charged "
        f"looks={LOOKS}",
        flush=True,
    )


if __name__ == "__main__":
    main()
