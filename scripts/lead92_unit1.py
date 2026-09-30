from __future__ import annotations

import json
from pathlib import Path

import lead83_unit2 as shared
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from best_pick_sunday_renomination_eval import select_with_tie_rule
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.best_pick_nomination import dispersion_pool_from_frame
from nfl_ats.pool_workbench import PoolRules

OUTPUT = Path("tests/scratch/codex/lead92_unit1")
REPORT = Path("docs/lead92_unit1.md")
LANE = Path("docs/lanes/lead92.md")
DISPERSION = Path("artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet")
ARMS = ("pairwise", "four_term", "model_only", "market", "elo")
ENDPOINTS = (*shared.METRICS, "nominee_reward", "nominee_brier")
DIRECTION = np.array([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
UNITS = (
    "accuracy_points",
    "log_loss_improvement",
    "brier_improvement",
    "rps_improvement",
    "nominee_reward_points",
    "nominee_brier_improvement",
)
LABELS = (*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:]))
RIDGE = 0.001
LOOKS = 713
FLAGS = Path("tests/scratch/codex/lead85_unit1/predictions.parquet")
FOLDS = (2023, 2024, 2025)


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
    frame["kickoff"] = pd.to_datetime(frame.kickoff, utc=True)
    day = frame.kickoff.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    first = day.groupby([frame.season, frame.week]).transform("min")
    sunday = first + pd.to_timedelta((6 - first.dt.dayofweek) % 7, unit="D")
    frame["nomination_cutoff"] = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    frame["nomination_eligible"] = frame.kickoff.gt(frame.nomination_cutoff)
    flags = shared.read(FLAGS, ["game_id", "composition_flag_sum"]).drop_duplicates("game_id")
    audited = frame[["game_id", "composition_flag_sum"]].merge(
        flags, on="game_id", how="left", suffixes=("", "_audited"), validate="one_to_one"
    )
    known = audited.composition_flag_sum.notna()
    if (
        not audited.loc[known, "composition_flag_sum"]
        .eq(audited.loc[known, "composition_flag_sum_audited"])
        .all()
    ):
        raise ValueError("Audited push-preserving flags disagree with cached flags")
    frame["composition_flag_sum"] = audited.composition_flag_sum.fillna(
        audited.composition_flag_sum_audited
    ).to_numpy()
    if frame.composition_flag_sum.isna().any():
        raise ValueError("A game lacks audited composition flags")
    lineage["flags_sha256"] = shared.digest(FLAGS)
    dispersion = shared.read(DISPERSION, ["game_id", "spread_std"])
    frame = frame.merge(dispersion, on="game_id", how="left", validate="many_to_one")
    frame["tie_dispersion"] = frame.spread_std
    lineage["dispersion_sha256"] = shared.digest(DISPERSION)
    lineage["missing_dispersion"] = int(frame.spread_std.isna().sum())
    lineage["eligible_games"] = int(frame.nomination_eligible.sum())
    print(
        f"inventory passed: {len(frame)} games including pushes; "
        f"{int(frame.declared_fit_population.sum())} conditional-cover rows; "
        f"{lineage['eligible_games']} nomination-eligible; "
        f"{lineage['missing_dispersion']} missing dispersion",
        flush=True,
    )
    return frame, cached, lineage


def pair_design(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    diffs, weights = [], []
    for value in np.unique(groups):
        rows = np.flatnonzero(groups == value)
        bets = np.vstack([x[rows], -x[rows]])
        outcome = np.r_[y[rows], 1 - y[rows]]
        game = np.r_[np.arange(len(rows)), np.arange(len(rows))]
        wins, losses = np.flatnonzero(outcome == 1), np.flatnonzero(outcome == 0)
        difference = bets[wins][:, None, :] - bets[losses][None, :, :]
        keep = game[wins][:, None] != game[losses][None, :]
        chosen = difference[keep]
        if len(chosen):
            diffs.append(chosen)
            weights.append(np.full(len(chosen), 1.0 / len(chosen)))
    return np.vstack(diffs), np.concatenate(weights), len(np.unique(groups))


def fit_pairwise(frame: pd.DataFrame, training: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    x = frame.loc[:, list(shared.TERMS)].to_numpy(float)
    mean = x[training].mean(axis=0)
    scale = x[training].std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    weeks = frame.loc[training, "season"].to_numpy(int) * 100 + frame.loc[
        training, "week"
    ].to_numpy(int)
    diffs, weights, week_count = pair_design(
        z[training], frame.loc[training, "home_covered"].to_numpy(float), weeks
    )

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        d = diffs @ beta
        loss = float(weights @ np.logaddexp(0, -d)) + 0.5 * RIDGE * float(beta @ beta)
        grad = -(diffs.T @ (weights * expit(-d))) + RIDGE * beta
        return loss, grad

    fitted = minimize(
        objective,
        np.zeros(z.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Pairwise optimization failed: {fitted.message}")
    natural = fitted.x / scale
    coefficients = dict(zip(shared.TERMS, natural.tolist(), strict=True))
    coefficients["intercept"] = float(-mean @ natural)
    details = {
        "success": bool(fitted.success),
        "iterations": int(fitted.nit),
        "objective": float(fitted.fun),
        "pairs": len(diffs),
        "weeks": week_count,
        "gradient_max": float(np.abs(fitted.jac).max()),
        "ridge": RIDGE,
        "standardized": dict(zip(shared.TERMS, fitted.x.tolist(), strict=True)),
    }
    return expit(np.clip(z @ fitted.x, -30, 30)), coefficients, details


def nominate(panel: pd.DataFrame, probabilities: dict, pushes: dict, outer: int) -> pd.DataFrame:
    weight = PoolRules().push_points
    if weight != 0.5:
        raise ValueError("Pool push rule changed after declaration")
    rows = []
    for phase, role in (("IS", "fit"), ("OOS", "outer")):
        selected = panel.loc[panel.role.eq(role)]
        for (season, week), group in selected.groupby(["season", "week"], sort=True):
            available = group.loc[group.nomination_eligible].copy()
            if available.empty:
                raise ValueError("No eligible weekly contender")
            pool = dispersion_pool_from_frame(available[["game_id", "spread_std"]])
            passed = pool.frame.loc[pool.frame.pool_pass, "game_id"]
            eligible = available.loc[available.game_id.isin(passed)].copy()
            row_base = {
                "outer": outer,
                "phase": phase,
                "season": int(season),
                "week": int(week),
                "available": len(available),
                "eligible": len(eligible),
                "pool_fallback": bool(pool.fallback),
            }
            for arm in ARMS:
                p = pd.Series(probabilities[arm][eligible.index], index=eligible.index)
                push = pd.Series(pushes[arm][eligible.index], index=eligible.index)
                eligible["rank"] = np.maximum(p, 1 - p) * (1 - push) + weight * push
                ids, tie = select_with_tie_rule(eligible, "rank")
                chosen = eligible.set_index("game_id").loc[list(ids)]
                cp = pd.Series(probabilities[arm][eligible.index], index=eligible.game_id).loc[
                    list(ids)
                ]
                home = cp.to_numpy(float) >= 0.5
                margin = chosen.ats_margin.to_numpy(float)
                signed = np.where(home, margin, -margin)
                win, tied = signed > 0, signed == 0
                cover = np.maximum(cp.to_numpy(float), 1 - cp.to_numpy(float))
                brier = np.where(tied, 0.0, (cover - win) ** 2)
                rows.append(
                    {
                        **row_base,
                        "arm": arm,
                        "ids": ",".join(ids),
                        "tie": tie,
                        "win": float(win.mean()),
                        "push": float(tied.mean()),
                        "loss": float((signed < 0).mean()),
                        "reward": float((win + weight * tied).mean()),
                        "brier_num": float(brier.mean()),
                        "nonpush": float((~tied).mean()),
                    }
                )
    return pd.DataFrame(rows)


def replay(
    frame: pd.DataFrame, cached: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows, nominees, details = [], [], []
    for outer in FOLDS:
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
        panel["home_covered"] = panel.ats_margin.gt(0).astype(float)
        upstream = panel[["game_id", "season", "week", "role", "spread_line", "model_logit"]].copy()
        upstream["fit_through"] = outer - 3
        upstream["point"] = point
        upstream["training_max_gameday"] = str(pd.to_datetime(train.gameday).max().date())
        pq.write_table(
            pa.Table.from_pandas(upstream, preserve_index=False),
            OUTPUT / f"upstream_{outer}.parquet",
        )
        kept = panel.declared_fit_population.to_numpy()
        fit_mask = training & kept
        coefficients = {}
        raw["four_term"], coefficients["four_term"] = shared.fit_probability(
            panel, fit_mask, shared.TERMS
        )
        raw["pairwise"], coefficients["pairwise"], optimization = fit_pairwise(panel, fit_mask)
        masses["four_term"] = masses["model_only"]
        masses["pairwise"] = masses["model_only"]
        scoring = panel.copy()
        scoring["role"] = scoring.role.where(kept, "push")
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "upstream_games_with_pushes": len(train),
            "training_max_gameday": str(pd.to_datetime(train.gameday).max().date()),
            "roles": panel.loc[kept, "role"].value_counts().to_dict(),
            "home_side_offsets": offsets.offsets,
            "elo_margin_coefficients": elo_beta.tolist(),
            "optimization": optimization,
            "probability_coefficients": {},
            "calibration": {},
        }
        calibrated, pushes = {}, {}
        for arm in ARMS:
            p, calibration = shared.calibrate(raw[arm], scoring)
            calibrated[arm] = p
            _, pushes[arm] = shared.metrics(panel, p, masses[arm])
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
                selected = panel.role.eq(role).to_numpy() & kept
                result = panel.loc[
                    selected, ["game_id", "season", "week", "result", "spread_line", "home_covered"]
                ].copy()
                scored, _ = shared.metrics(panel.loc[selected], p[selected], masses[arm][selected])
                result["outer"] = outer
                result["phase"] = phase
                result["arm"] = arm
                result["probability"] = p[selected]
                result["push_probability"] = pushes[arm][selected]
                for index, metric in enumerate(shared.METRICS):
                    result[metric] = scored[:, index]
                rows.append(result)
        nominees.append(nominate(panel, calibrated, pushes, outer))
        details.append(detail)
        print(
            f"fold={outer} complete; fit through {outer - 3}; "
            f"outer n={int((panel.role.eq('outer') & kept).sum())}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), pd.concat(nominees, ignore_index=True), details


def blocks(
    predictions: pd.DataFrame, nominees: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = ["outer", "season", "week"]
    index = (
        predictions[keys].drop_duplicates().sort_values(keys).reset_index(drop=True).set_index(keys)
    )
    numerator = np.zeros((len(index), len(ARMS), len(ENDPOINTS)))
    denominator = np.zeros_like(numerator)
    for a, arm in enumerate(ARMS):
        games = predictions.loc[predictions.arm.eq(arm)]
        summed = games.groupby(keys)[list(shared.METRICS)].sum().reindex(index.index)
        counts = games.groupby(keys).size().reindex(index.index)
        weekly = nominees.loc[nominees.arm.eq(arm)].set_index(keys).reindex(index.index)
        numerator[:, a, :4] = summed.fillna(0).to_numpy(float)
        denominator[:, a, :4] = counts.fillna(0).to_numpy(float)[:, None]
        numerator[:, a, 4] = weekly.reward.fillna(0).to_numpy(float)
        denominator[:, a, 4] = weekly.reward.notna().to_numpy(float)
        numerator[:, a, 5] = weekly.brier_num.fillna(0).to_numpy(float)
        denominator[:, a, 5] = weekly.nonpush.fillna(0).to_numpy(float)
    strata = (
        index.index.get_level_values(0).to_numpy() * 10000
        + index.index.get_level_values(1).to_numpy()
    )
    return numerator, denominator, strata


def vector(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = numerator.sum(axis=0) / denominator.sum(axis=0)
    gains = (ratio[0][None, :] - ratio[1:]) * DIRECTION
    return np.vstack([ratio, gains]).ravel()


def bootstrap(
    numerator: np.ndarray, denominator: np.ndarray, strata: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.unique(strata)
    indexes = [np.flatnonzero(strata == value) for value in values]
    means = vector(numerator, denominator)
    samples = np.empty((shared.BOOTSTRAPS, len(means)))
    for draw in range(shared.BOOTSTRAPS):
        chosen = np.concatenate(
            [
                rng.choice(indexes[s], len(indexes[s]), replace=True)
                for s in rng.integers(0, len(indexes), len(indexes))
            ]
        )
        samples[draw] = vector(numerator[chosen], denominator[chosen])
    return means, samples


def cells(means: np.ndarray, draws: np.ndarray) -> dict:
    width = len(ENDPOINTS)
    return {
        label: {
            metric: shared.estimate(means[i * width + j], draws[:, i * width + j])
            for j, metric in enumerate(ENDPOINTS)
        }
        for i, label in enumerate(LABELS)
    }


def summarize(predictions: pd.DataFrame, nominees: pd.DataFrame) -> dict:
    panels = {}
    for outer in (*FOLDS, "pooled"):
        panel = {"phases": {}, "decisive": {}, "nominees": {}}
        phase_means, phase_draws = {}, {}
        for phase in ("IS", "OOS"):
            selected = predictions.phase.eq(phase)
            weeks = nominees.phase.eq(phase)
            if outer != "pooled":
                selected &= predictions.outer.eq(outer)
                weeks &= nominees.outer.eq(outer)
            frame, weekly = predictions.loc[selected], nominees.loc[weeks]
            groups = {
                arm: frame.loc[frame.arm.eq(arm)]
                .sort_values(["outer", "game_id"])
                .reset_index(drop=True)
                for arm in ARMS
            }
            reference = groups["pairwise"]
            if any(
                not g[["outer", "game_id"]].equals(reference[["outer", "game_id"]])
                for g in groups.values()
            ):
                raise ValueError("Unpaired arm populations")
            numerator, denominator, strata = blocks(frame, weekly)
            means, draws = bootstrap(
                numerator,
                denominator,
                strata,
                shared.SEED
                + (0 if outer == "pooled" else int(outer))
                + (100 if phase == "OOS" else 0),
            )
            phase_means[phase], phase_draws[phase] = means, draws
            panel["phases"][phase] = cells(means, draws)
            for arm in ARMS:
                wins = int(groups[arm].accuracy_points.eq(100).sum())
                cell = panel["phases"][phase][arm]
                cell["record"] = f"{wins}-{len(reference) - wins}"
                cell["games"] = len(reference)
                mine = weekly.loc[weekly.arm.eq(arm)]
                cell["nominee_record"] = (
                    f"{mine.win.sum():.1f}-{mine.loss.sum():.1f}-{mine.push.sum():.1f}"
                )
            if phase == "OOS":
                panel["blocks"] = len(strata)
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
                    a = weekly.loc[weekly.arm.eq("pairwise")].set_index(["outer", "season", "week"])
                    b = weekly.loc[weekly.arm.eq(baseline)].set_index(["outer", "season", "week"])
                    changed = a.ids.ne(b.ids.reindex(a.index))
                    panel["nominees"][baseline] = {
                        "weeks": len(a),
                        "changed_weeks": int(changed.sum()),
                        "reward_difference_on_changed": float(
                            (a.reward[changed] - b.reward.reindex(a.index)[changed]).mean()
                        )
                        if changed.any()
                        else 0.0,
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
    lines = [
        chr(96) * 3 + "bash",
        "while read -r panel first last games blocks units effect low high pp; do",
        (
            ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record "
            '--name "lead92_unit1_${panel}_${units}" --league nfl \\'
        ),
        (
            '  --description "Best Pick ordering fit versus the usual four-term'
            ' calculation: ${panel}" --source docs/lead92_unit1.md --family '
            "lead92_unit1_713_looks \\"
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
            'closure or serving claim. All 713 declared looks reported." \\'
        ),
        (
            '  --plain-summary "This teaches the pick calculation to rank the '
            "week's games against each other, the way the bonus pick works. The "
            "held-out comparison stays a research result; pool picks have not "
            'changed."'
        ),
        "done <<'CELLS'",
    ]
    for label, panel in summary["panels"].items():
        first, last = (2023, 2025) if label == "pooled" else (int(label), int(label))
        oos = panel["phases"]["OOS"]
        for metric, unit in zip(ENDPOINTS, UNITS, strict=True):
            cell = oos["gain_vs_four_term"][metric]
            values = " ".join(
                f"{cell[key]:.12g}" for key in ("estimate", "low", "high", "probability_positive")
            )
            lines.append(
                f"{label} {first} {last} {oos['pairwise']['games']} {panel['blocks']} "
                f"{unit} {values}"
            )
    return "\n".join([*lines, "CELLS", chr(96) * 3])


def report(summary: dict) -> None:
    lineage = summary["lineage"]
    lines = [
        "# LEAD-92 unit 1: within-week Best Pick ordering",
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
        (
            "**Measured:** weekly nominee changes, held-out weeks (description only, "
            "not additional looks)."
        ),
        "",
        shared.table(
            ["Outer", "Comparator", "Weeks", "Changed nominee weeks", "Reward diff on changed"],
            [
                [
                    outer,
                    arm,
                    cell["weeks"],
                    cell["changed_weeks"],
                    f"{cell['reward_difference_on_changed']:+.4f}",
                ]
                for outer, panel in summary["panels"].items()
                for arm, cell in panel["nominees"].items()
            ],
        ),
        "",
        "## Fixed protocol and interpretation",
        "",
        (
            "The declaration below was saved before outcomes; its immutable copy is"
            " tests/scratch/codex/lead92_unit1/protocol.md and the original study"
            " text is docs/lead92_protocol.md."
        ),
        "",
        summary["protocol"],
        "",
        (
            "**Measured:** all upstream margin/offset/Elo models and discrete "
            "lattices refit through Y-3. Historical opener replaces closing spread, "
            "archived Tuesday totals replace closing totals. Hashes verify the "
            "LEAD-83 inputs and both quote archives. IS is optimistic, with "
            "repeated training rows across pooled folds. All five arms receive the "
            "same separate slope selection and intercept calibration; the pairwise "
            "raw logit has no intercept of its own, which pairs cannot identify, so "
            "calibration supplies it. Calibrated cover/loss mass retains the "
            "discrete lattice's push mass and alone selects the side and nominee."
        ),
        "",
        (
            f"**Measured:** {lineage['population']} source-complete games including "
            f"pushes; {lineage['nonpush']} conditional-cover rows; "
            f"{lineage['missing_tuesday_totals']} missing Tuesday totals retained "
            f"as missing; {lineage['eligible_games']} games eligible after the "
            f"Sunday 12:45 cutoff; {lineage['missing_dispersion']} games without "
            f"dispersion; {LOOKS} looks, 10,000 paired season/week-block "
            "resamples. Gap intervals subtract independently resampled IS from "
            "OOS. Intervals condition on fitted models and omit refit uncertainty."
        ),
        "",
        (
            "**Inferred:** unresolved_below_power pending orchestrator recording. "
            "Three outer seasons and retrospective source reuse limit "
            "generalization. Per AGENTS.md, zero crossing closes nothing and "
            "diagnostic or nominee gains alone cannot promote ATS sides. No card "
            "change or closure."
        ),
        "",
        "## Fold coefficients and stability",
        "",
        (
            "**Measured:** natural coefficients before (fit) and after (final) "
            "slope/intercept calibration."
        ),
        "",
    ]
    terms = ("intercept", *shared.TERMS)
    coefficient_rows = [
        [fold["outer"], arm, phase, *[f"{coef[term]:.8f}" for term in terms]]
        for fold in summary["folds"]
        for arm, phases in fold["probability_coefficients"].items()
        for phase, coef in phases.items()
    ]
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
            ["Outer", "Fit through", "Roles", "Pairs", "Weeks", "Objective", "Max gradient"],
            [
                [
                    fold["outer"],
                    fold["fit_through"],
                    json.dumps(fold["roles"]),
                    fold["optimization"]["pairs"],
                    fold["optimization"]["weeks"],
                    f"{fold['optimization']['objective']:.6f}",
                    f"{fold['optimization']['gradient_max']:.3g}",
                ]
                for fold in summary["folds"]
            ],
        ),
        "",
        "## IS, OOS and gaps",
        "",
        (
            "**Measured:** each cell is estimate [95% interval]; "
            "probability_positive. Accuracy is percentage points; other game "
            "endpoints are losses; nominee reward is points per week; nominee "
            "Brier is conditional on a non-push. Gain rows orient every endpoint "
            "so positive favors the candidate. Gaps are OOS minus IS. Raw loss "
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
                        measures.get("nominee_record", "—"),
                        *[
                            (
                                f"{shared.number(measures[metric])}; "
                                f"{measures[metric]['probability_positive']:.5f}"
                            )
                            for metric in ENDPOINTS
                        ],
                    ]
                )
        lines += [
            "",
            f"### {outer}",
            "",
            shared.table(["Phase", "Arm/contrast", "W-L", "Nominee W-L-P", *ENDPOINTS], rows),
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
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead92_unit1.py",
        "",
        (
            "Prediction-level rows, nominee rows, upstream cutoffs, complete "
            "coefficients, source hashes and all intervals: "
            "tests/scratch/codex/lead92_unit1/. Serial registry commands are in "
            "docs/lanes/lead92.md and were not run by this worker."
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
        predictions, nominees, folds = replay(frame, cached)
        pq.write_table(
            pa.Table.from_pandas(predictions, preserve_index=False), OUTPUT / "predictions.parquet"
        )
        pq.write_table(
            pa.Table.from_pandas(nominees, preserve_index=False), OUTPUT / "nominees.parquet"
        )
        summary = summarize(predictions, nominees)
        declaration = protocol.split("## Protocol declaration", 1)[1].split("\n## Tried", 1)[0]
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
                "games": pooled["pairwise"]["games"],
                "record": pooled["pairwise"]["record"],
                "candidate_vs_four_term": pooled["gain_vs_four_term"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
