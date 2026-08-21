"""Best Pick ranker follow-up: four PREDECLARED top-1 nomination signals on the
free CFB XLG-03 benchmark. No NFL rotation window is spent; no registry window
is touched.

Predeclaration: ``docs/best_pick_followup.md`` § "Frozen predeclaration",
written before this script's scoring pass ran. The candidate list, the
status-quo comparator, the population, the bootstrap design, and the 0.75
screen gate are all frozen there and mirrored in the constants below.

Question: MOD-12 routed the Brier-optimal ridge refit to the Best Pick ranker
(``docs/ridge_alpha.md``) and MOD-08 promoted a smooth-CDF probability mapping
(``docs/ecdf_smoothing.md``). Do any of four predeclared confidence orderings
built from those two improvements beat the deployed ``sweep_robustness`` +
alphabetical rule at choosing one Best Pick per week?

Population: the already-scored stage-0 CFB sweep artifact
(``artifacts/best_pick_tiebreak_cfb/20260818T212916Z/sweep_picks.parquet``,
280 weeks / 11,780 resolved picks, seasons 2006-2025), reused read-only. The
only fresh computation is a walk-forward pass emitting alpha=2000 cover
probabilities and each week's out-of-time residual-sample sd (the dispersion
input); the alpha=10 leg of that pass doubles as a reproduction check against
the stored artifact.

Run::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_ranker_followup.py

Record the four predeclared cells (after seeing results)::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_ranker_followup.py --record
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    fit_cfb_residual_model,
)
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"
STATUS_QUO_ARTIFACT = (
    REPO / "artifacts" / "best_pick_tiebreak_cfb" / "20260818T212916Z" / "sweep_picks.parquet"
)
SMOOTH_CDF_ARTIFACT = (
    REPO / "artifacts" / "ecdf_smoothing" / "20260818T000600Z" / "cfb_predictions.parquet"
)
OUTPUT_ROOT = REPO / "artifacts" / "best_pick_followup"

ALPHA_2000 = 2000.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
SCREEN_GATE = 0.75

CANDIDATES = (
    "best_pick_followup_smooth_cdf_distance",
    "best_pick_followup_alpha2000_distance",
    "best_pick_followup_dispersion_gated_smooth_distance",
    "best_pick_followup_ensemble_distance",
)


def walk_forward_alpha_pair(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """Walk forward the frozen CFB config at BOTH alphas, one fit per week each.

    Mirrors ``scripts/best_pick_tiebreak_cfb_screen.py::cfb_sweep_and_point``
    exactly (same cutoff rule, same min-train floor) minus the line sweep. The
    alpha=10 leg exists for the reproduction check and for the week-level
    out-of-time residual sd; the alpha=2000 leg is candidate 2's probability
    read. Neither fit ever sees the week being scored.
    """

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError("No completed CFB games in the requested window")

    rows: list[pd.DataFrame] = []
    for (season, week), weekly in test.groupby(["season", "week"], sort=True):
        cutoff = weekly["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model_alpha10 = fit_cfb_residual_model(training, ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA)
        model_alpha2000 = fit_cfb_residual_model(training, ridge_alpha=ALPHA_2000)
        rows.append(
            pd.DataFrame(
                {
                    "game_id": weekly["game_id"].to_numpy(),
                    "season": int(str(season)),
                    "week": int(str(week)),
                    "fresh_alpha10_probability": model_alpha10.predict(weekly)[
                        "home_cover_probability"
                    ].to_numpy(),
                    "fresh_alpha2000_probability": model_alpha2000.predict(weekly)[
                        "home_cover_probability"
                    ].to_numpy(),
                    "week_residual_sd": float(np.std(model_alpha10.residuals, ddof=1)),
                }
            )
        )

    if not rows:
        raise ValueError("No CFB week had enough prior training games")
    return pd.concat(rows, ignore_index=True)


def build_scores(picks: pd.DataFrame) -> pd.DataFrame:
    """Attach one score column per predeclared candidate to the picks frame.

    Scores are oriented so LARGER IS MORE CONFIDENT. Missing scores become
    -inf (rank last). The dispersion-gated composite is built per week in
    chronological order against the expanding median of PRIOR scored weeks'
    residual sds; the first scored week has no priors and keeps status quo.
    """

    work = picks.copy()
    work["game_id"] = work["game_id"].astype(str)

    pick_side_smooth = np.where(
        work["pick"].eq("HOME"),
        work["smooth_cdf_probability"],
        1.0 - work["smooth_cdf_probability"],
    )
    pick_side_alpha2000 = np.where(
        work["pick"].eq("HOME"),
        work["alpha2000_probability"],
        1.0 - work["alpha2000_probability"],
    )
    work["score_best_pick_followup_smooth_cdf_distance"] = np.abs(pick_side_smooth - 0.5)
    work["score_best_pick_followup_alpha2000_distance"] = np.abs(pick_side_alpha2000 - 0.5)
    work["score_best_pick_followup_ensemble_distance"] = 0.5 * (
        work["score_best_pick_followup_smooth_cdf_distance"]
        + work["score_best_pick_followup_alpha2000_distance"]
    )
    work["score_status_quo"] = work["sweep_robustness"]

    gate_by_week: dict[tuple[int, int], bool] = {}
    prior_sds: list[float] = []
    for (season, week), week_rows in work.groupby(["season", "week"], sort=True):
        sd = float(week_rows["week_residual_sd"].iloc[0])
        threshold = float(np.median(prior_sds)) if prior_sds else np.inf
        gate_by_week[(int(season), int(week))] = bool(sd < threshold)
        prior_sds.append(sd)

    keys = list(zip(work["season"].astype(int), work["week"].astype(int), strict=True))
    work["gated_low_dispersion"] = [gate_by_week[key] for key in keys]
    work["score_best_pick_followup_dispersion_gated_smooth_distance"] = np.where(
        work["gated_low_dispersion"],
        work["score_best_pick_followup_smooth_cdf_distance"],
        work["score_status_quo"],
    )

    for column in [c for c in work.columns if c.startswith("score_")]:
        work[column] = work[column].astype(float).fillna(-np.inf)
    return work


def nominate(work: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """One row per week: each rule's nominee, correctness, and divergence."""

    rows: list[dict[str, Any]] = []
    for (season, week), grp in work.groupby(["season", "week"], sort=True):
        ordered = grp.sort_values([score_column, "game_id"], ascending=[False, True])
        nominee = ordered.iloc[0]
        sq_ordered = grp.sort_values(["score_status_quo", "game_id"], ascending=[False, True])
        sq_nominee = sq_ordered.iloc[0]
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "evaluation_window": str(grp["evaluation_window"].iloc[0]),
                "n_picks": len(grp),
                f"{score_column}_nominee": str(nominee["game_id"]),
                f"{score_column}_correct": float(nominee["correct"]),
                "status_quo_nominee": str(sq_nominee["game_id"]),
                "status_quo_correct": float(sq_nominee["correct"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame[f"{score_column}_diverges"] = (
        frame[f"{score_column}_nominee"] != frame["status_quo_nominee"]
    )
    return frame


def paired_bootstrap(nominations: pd.DataFrame, score_column: str) -> pd.DataFrame:
    def metric(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "delta_accuracy_points": 100.0
            * (frame[f"{score_column}_correct"].mean() - frame["status_quo_correct"].mean()),
        }

    return week_blocked_bootstrap(
        nominations,
        metric,
        block="week",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )


def descriptive(work: pd.DataFrame, score_column: str) -> dict[str, Any]:
    scored = work.loc[work[score_column] > -np.inf]
    tau = kendalltau(scored[score_column], scored["correct"])
    return {
        "kendall_tau_all_picks": float(tau.statistic),
        "kendall_tau_p_value": float(tau.pvalue),
        "n_picks_scored": len(scored),
    }


def _cell_result(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return summary["candidates"][name]["bootstrap_full"][0]


def record_cells(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pop = summary["population"]
    source = (
        f"scripts/best_pick_ranker_followup.py; {summary_path.relative_to(REPO)}; "
        "docs/best_pick_followup.md (frozen predeclaration written before scoring)"
    )
    descriptions = {
        "best_pick_followup_smooth_cdf_distance": (
            "Weekly top-1 Best Pick nomination by |pick-side cover probability - 0.5| "
            "under MOD-08's PROMOTED smooth-CDF mapping (analytic Gaussian smoother, "
            "feature_set=='gaussian'; not the quantised ECDF, not gaussian_kde/"
            "skew_normal) vs the deployed sweep_robustness+alphabetical rule"
        ),
        "best_pick_followup_alpha2000_distance": (
            "Weekly top-1 Best Pick nomination by |pick-side cover probability - 0.5| "
            "from a walk-forward refit at ridge_alpha=2000 (the Brier-optimal shrinkage "
            "per docs/ridge_alpha.md) vs the deployed sweep_robustness+alphabetical rule"
        ),
        "best_pick_followup_dispersion_gated_smooth_distance": (
            "Composite: in weeks whose alpha=10 out-of-time residual-sample sd is below "
            "the expanding median of prior scored weeks' sds, nominate by the smooth-CDF "
            "distance; otherwise keep the status quo (low-dispersion weeks trusted more)"
        ),
        "best_pick_followup_ensemble_distance": (
            "Equal-weight mean of the smooth-CDF distance and the alpha=2000 distance "
            "(the two independently measured Brier-positive probability improvements) as "
            "the weekly top-1 nomination signal vs the deployed rule"
        ),
    }

    for name in CANDIDATES:
        row = _cell_result(summary, name)
        effect = float(row["estimate"])
        low, high = float(row["lower"]), float(row["upper"])
        classification = "unresolved_below_power"
        closing_ground = None
        evidence = (
            "Hypothesised direction is positive (candidate beats the deployed ranker). "
            "No positive control was run, so bounded_by_control is unavailable. "
        )
        if high < 0.0:
            classification = "refuted_mechanism"
            closing_ground = "wrong_sign_resolved"
            evidence += (
                "The whole week-blocked interval sits below zero, so wrong_sign_resolved "
                "is admissible: the ordering made the weekly top-1 pick WORSE than the "
                "deployed rule with the sign resolved."
            )
        else:
            evidence += (
                "The interval does not sit entirely below zero, so wrong_sign_resolved "
                "is inadmissible; there is no confirmed-positive classification in this "
                "registry's schema, so unresolved_below_power is the only admissible "
                "classification regardless of where the point estimate landed."
            )

        cmd = [
            sys.executable,
            "-m",
            "nfl_ats.cli",
            "weak-signals",
            "record",
            "--name",
            name,
            "--description",
            descriptions[name],
            "--source",
            source,
            "--effect",
            f"{effect:.10f}",
            "--effect-units",
            "accuracy_points",
            "--classification",
            classification,
            "--league",
            "cfb",
            "--season-start",
            str(pop["season_start"]),
            "--season-end",
            str(pop["season_end"]),
            "--interval-low",
            f"{low:.10f}",
            "--interval-high",
            f"{high:.10f}",
            "--probability-positive",
            f"{float(row['probability_positive']):.10f}",
            "--sample-games",
            str(pop["games"]),
            "--sample-blocks",
            str(pop["weeks"]),
            "--classification-evidence",
            evidence,
            "--notes",
            (
                f"Free CFB XLG-03 benchmark screen (rotation rule 8); NO NFL rotation "
                f"window spent. Frozen predeclaration: docs/best_pick_followup.md, "
                f"written before scoring. Status quo: sweep_robustness desc + ascending "
                f"game_id ties (deployed NFL rule computed on CFB by the stage-0 "
                f"harness). Week-blocked bootstrap, {summary['bootstrap_samples']:,} "
                f"samples, seed {summary['bootstrap_seed']}. Nominee diverged from the "
                f"status quo in {summary['candidates'][name]['n_weeks_diverge']} of "
                f"{pop['weeks']} weeks. Clean-core secondary delta: "
                f"{summary['candidates'][name]['clean_core_delta_accuracy_points']:.4f} "
                f"pts (descriptive, not gated). Screen gate P+ >= 0.75: "
                f"{'PASS' if float(row['probability_positive']) >= SCREEN_GATE else 'not passed'}; "
                f"a pass would only make the signal eligible for its own future NFL "
                f"predeclared look."
            ),
        ]
        if closing_ground is not None:
            cmd += ["--closing-ground", closing_ground]

        print(f"=== recording {name} ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for {name} (exit {result.returncode}); "
                "per AGENTS.md, if a record command errors the verdict is wrong, not "
                "the validator -- reclassify, do not force it through."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--status-quo-artifact", type=Path, default=STATUS_QUO_ARTIFACT)
    parser.add_argument("--smooth-cdf-artifact", type=Path, default=SMOOTH_CDF_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    if args.record:
        runs = sorted(p for p in args.output_root.iterdir() if p.is_dir())
        if not runs:
            raise SystemExit(f"No runs found under {args.output_root}; run the screen first")
        record_cells(runs[-1] / "summary.json")
        return

    print("[followup] walking forward the frozen CFB config at alpha=10 and alpha=2000 ...")
    fresh = walk_forward_alpha_pair(pd.read_parquet(args.features))
    fresh["game_id"] = fresh["game_id"].astype(str)

    picks = pd.read_parquet(args.status_quo_artifact)
    picks["game_id"] = picks["game_id"].astype(str)

    merged = picks.merge(fresh, on=["game_id", "season", "week"], how="inner", validate="1:1")
    if len(merged) != len(picks):
        raise SystemExit(f"Reproduction merge lost rows: stored {len(picks)}, merged {len(merged)}")

    repro = float(
        (merged["home_cover_probability"] - merged["fresh_alpha10_probability"]).abs().max()
    )
    print(f"[check a] stored vs fresh alpha=10 home_cover_probability max|diff| = {repro:.3e}")
    if repro > 1e-8:
        raise SystemExit("alpha=10 reproduction check failed; refusing to score")

    smooth = pd.read_parquet(args.smooth_cdf_artifact)
    smooth["game_id"] = smooth["game_id"].astype(str)
    gaussian = (
        smooth.loc[smooth["feature_set"].eq("gaussian")]
        .set_index("game_id")["home_cover_probability"]
        .rename("smooth_cdf_probability")
    )
    merged = merged.merge(gaussian, on="game_id", how="left", validate="1:1")
    merged = merged.rename(columns={"fresh_alpha2000_probability": "alpha2000_probability"})
    n_missing_smooth = int(merged["smooth_cdf_probability"].isna().sum())
    n_missing_sd = int(merged["week_residual_sd"].isna().sum())
    print(
        f"[population] games {len(merged)}, missing smooth-CDF probability "
        f"{n_missing_smooth}, missing week residual sd {n_missing_sd}"
    )

    work = build_scores(merged)

    candidates_out: dict[str, Any] = {}
    for name in CANDIDATES:
        score_column = f"score_{name}"
        nominations = nominate(work, score_column)
        bootstrap_full = paired_bootstrap(nominations, score_column)
        clean_core = nominations.loc[nominations["evaluation_window"].eq("clean_core")]
        bootstrap_clean = paired_bootstrap(clean_core, score_column)
        row = bootstrap_full.iloc[0]
        clean_row = bootstrap_clean.iloc[0]
        candidates_out[name] = {
            "top1_accuracy": float(nominations[f"{score_column}_correct"].mean()),
            "status_quo_top1_accuracy": float(nominations["status_quo_correct"].mean()),
            "n_weeks_diverge": int(nominations[f"{score_column}_diverges"].sum()),
            "bootstrap_full": bootstrap_full,
            "clean_core_delta_accuracy_points": float(clean_row["estimate"]),
            "clean_core_probability_positive": float(clean_row["probability_positive"]),
            "descriptive": descriptive(work, score_column),
        }
        print(
            f"[candidate] {name}: delta {row['estimate']:+.4f} pts, "
            f"95% [{row['lower']:+.4f}, {row['upper']:+.4f}], "
            f"P+ {row['probability_positive']:.4f}, diverges "
            f"{int(nominations[f'{score_column}_diverges'].sum())}/{len(nominations)} wk"
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    work.to_parquet(out_dir / "per_game.parquet", index=False)

    serialisable_candidates = {
        name: {
            key: (value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value)
            for key, value in cell.items()
        }
        for name, cell in candidates_out.items()
    }
    summary: dict[str, Any] = {
        "predeclaration": "docs/best_pick_followup.md (written before scoring)",
        "rotation_registry_touched": False,
        "nfl_window_spent": False,
        "population": {
            "games": len(work),
            "weeks": int(work.groupby(["season", "week"]).ngroups),
            "season_start": int(work["season"].min()),
            "season_end": int(work["season"].max()),
            "source_artifact": str(args.status_quo_artifact.relative_to(REPO)),
        },
        "reproduction_check_alpha10_max_abs_diff": repro,
        "missing_smooth_cdf_probability": n_missing_smooth,
        "gate": SCREEN_GATE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "candidates": serialisable_candidates,
        "gate_note": (
            "P+ >= 0.75 on the FULL population is the predeclared screen gate; a pass "
            "only makes a signal eligible for its own future NFL predeclared look. "
            "NFL activation needs its own predeclared look; none is spent or implied."
        ),
    }
    summary["provenance"] = artifact_provenance(
        {
            "command": "best-pick-ranker-followup",
            "features": str(args.features),
            "status_quo_artifact": str(args.status_quo_artifact),
            "smooth_cdf_artifact": str(args.smooth_cdf_artifact),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        args.features,
        project_root=REPO,
    )
    write_experiment_artifact(
        out_dir,
        "summary.json",
        summary,
        command="best-pick-ranker-followup",
        metrics={
            name: {
                "delta_accuracy_points": cell["bootstrap_full"].iloc[0]["estimate"],
                "interval_low": cell["bootstrap_full"].iloc[0]["lower"],
                "interval_high": cell["bootstrap_full"].iloc[0]["upper"],
                "probability_positive": cell["bootstrap_full"].iloc[0]["probability_positive"],
            }
            for name, cell in candidates_out.items()
        },
        notes=(
            "Four predeclared Best-Pick top-1 nomination signals vs the deployed "
            "sweep_robustness rule on the free CFB XLG-03 benchmark; frozen "
            "predeclaration in docs/best_pick_followup.md; no NFL window spent."
        ),
    )
    print(f"\nartifacts: {out_dir}")


if __name__ == "__main__":
    main()
