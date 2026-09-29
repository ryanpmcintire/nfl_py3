from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from best_pick_sunday_renomination_eval import select_with_tie_rule

from nfl_ats.best_pick_nomination import dispersion_pool_from_frame
from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pool_workbench import PoolRules

OPENER = Path("artifacts/opener_evaluation/20260920T135435Z/per_game.parquet")
PAIRED = Path("artifacts/ridge_alpha_promotion/20260818T221459Z/opener_paired.parquet")
DISPERSION = Path("artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
REPORT = Path("docs/lead72_results.md")
WEEKLY = Path("docs/lead72_weekly.md")
SEASONS = tuple(range(2020, 2026))
EXPECTED_WEEKS = 107
SAMPLES = 20_000
SEED = 20260929
ARMS = ("raw_cover", "push_adjusted")
COVER = "home_cover_probability_excluding_push_at_open"
PUSH = "push_probability_at_open"
LOSS = "home_loss_probability_at_open"
SIDE = "pick_home_at_open_probability_rule"
REQUIRED = {
    OPENER: [
        "game_id",
        "season",
        "week",
        "tue_open_home_spread",
        "base_probability_policy",
        "home_cover_probability_at_open",
        COVER,
        PUSH,
        LOSS,
        SIDE,
        "margin_vs_open",
    ],
    PAIRED: ["game_id", "season", "week", "tue_open_home_spread"],
    DISPERSION: ["game_id", "spread_std"],
    FEATURES: ["game_id", "kickoff"],
}


def inventory() -> tuple[list[str], bool]:
    lines = ["| Source | Rows | Required columns | SHA-256 |", "|---|---:|---|---|"]
    ready = True
    for path, columns in REQUIRED.items():
        if not path.is_file():
            lines.append(f"| `{path.as_posix()}` | absent | unavailable | unavailable |")
            ready = False
            continue
        metadata = pq.read_metadata(path)
        missing = sorted(set(columns) - set(pq.read_schema(path).names))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        status = ", ".join(missing) if missing else "present"
        lines.append(f"| `{path.as_posix()}` | {metadata.num_rows} | {status} | `{digest}` |")
        ready &= not missing
    return lines, ready


def load_candidates(weight: float) -> pd.DataFrame:
    paired = pd.read_parquet(PAIRED, columns=REQUIRED[PAIRED])
    frame = pd.read_parquet(OPENER, columns=REQUIRED[OPENER])
    paired = paired.loc[paired.season.isin(SEASONS)].copy()
    frame = frame.loc[frame.season.isin(SEASONS)].copy()
    for table in (paired, frame):
        if table.game_id.duplicated().any():
            raise ValueError("Repeated game in historical archive")
    if set(frame.game_id) != set(paired.game_id):
        raise ValueError("Discrete opener archive does not match the LEAD-53 population")
    keys = ["game_id", "season", "week", "tue_open_home_spread"]
    if len(frame.merge(paired, on=keys, validate="one_to_one")) != len(frame):
        raise ValueError("Historical season, week or frozen line differs")
    if len(frame[["season", "week"]].drop_duplicates()) != EXPECTED_WEEKS:
        raise ValueError("The declared 107-week population is unavailable")
    if set(frame.season) != set(SEASONS):
        raise ValueError("A declared season is absent")
    for path in (DISPERSION, FEATURES):
        other = pd.read_parquet(path, columns=REQUIRED[path])
        other = other.loc[other.game_id.isin(frame.game_id)].copy()
        frame = frame.merge(other, on="game_id", how="left", validate="one_to_one")
    frame["kickoff"] = pd.to_datetime(frame.kickoff, utc=True, errors="raise")
    if frame.kickoff.isna().any():
        raise ValueError("LEAD-53 tie rule needs kickoff for every candidate")
    mass = frame[[COVER, PUSH, LOSS]].to_numpy(dtype=float)
    if not np.isfinite(mass).all() or (mass < 0).any() or (mass > 1).any():
        raise ValueError("Invalid discrete probabilities")
    if not np.allclose(mass.sum(axis=1), 1.0, atol=1e-9, rtol=0):
        raise ValueError("Discrete cover, push and loss do not conserve mass")
    conditional = np.divide(
        mass[:, 0],
        mass[:, 0] + mass[:, 2],
        out=np.full(len(frame), 0.5),
        where=mass[:, 0] + mass[:, 2] > 0,
    )
    if not np.allclose(conditional, frame.home_cover_probability_at_open, atol=1e-9, rtol=0):
        raise ValueError("Saved selection probability differs from the discrete distribution")
    if frame[SIDE].isna().any() or not frame[SIDE].isin([True, False]).all():
        raise ValueError("Saved side is missing or nonboolean")
    if not np.array_equal(frame[SIDE].to_numpy(), conditional >= 0.5):
        raise ValueError("Saved side does not follow the archived calibrated probability")
    frame["raw_cover"] = np.where(frame[SIDE], frame[COVER], frame[LOSS])
    frame["push_adjusted"] = frame.raw_cover + weight * frame[PUSH]
    frame["tie_dispersion"] = frame.spread_std
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def weekly_replay(frame: pd.DataFrame, weight: float) -> pd.DataFrame:
    rows = []
    for (season, week), candidates in frame.groupby(["season", "week"], sort=True):
        pool = dispersion_pool_from_frame(candidates[["game_id", "spread_std"]])
        passed = pool.frame.loc[pool.frame.pool_pass, "game_id"]
        eligible = candidates.loc[candidates.game_id.isin(passed)].copy()
        choices = {arm: select_with_tie_rule(eligible, arm) for arm in ARMS}
        row = {"season": int(season), "week": int(week), "candidates": len(eligible)}
        row["pool_fallback"] = pool.fallback
        row["changed"] = choices[ARMS[0]][0] != choices[ARMS[1]][0]
        for arm, (game_ids, tie_level) in choices.items():
            chosen = eligible.set_index("game_id").loc[list(game_ids)]
            margins = chosen.margin_vs_open.to_numpy(dtype=float)
            if not np.isfinite(margins).all():
                raise ValueError("Selected nominee lacks an opener outcome")
            signed = np.where(chosen[SIDE], margins, -margins)
            push = np.isclose(signed, 0.0, atol=1e-9, rtol=0)
            win = (signed > 0) & ~push
            row[f"{arm}_ids"] = ",".join(game_ids)
            row[f"{arm}_sides"] = ",".join(np.where(chosen[SIDE], "home", "away"))
            row[f"{arm}_tie"] = tie_level
            row[f"{arm}_cover"] = float(chosen.raw_cover.mean())
            row[f"{arm}_push"] = float(chosen[PUSH].mean())
            row[f"{arm}_stat"] = float(chosen[arm].mean())
            row[f"{arm}_win"] = float(win.mean())
            row[f"{arm}_push_result"] = float(push.mean())
            row[f"{arm}_loss"] = float((~win & ~push).mean())
            row[f"{arm}_grade"] = float((win + weight * push).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(weekly: pd.DataFrame) -> dict:
    if weekly.empty:
        return {"weeks": 0, "metrics": None}
    values = weekly[[f"{arm}_grade" for arm in ARMS]].to_numpy(dtype=float)
    values = np.column_stack([values, values[:, 1] - values[:, 0]])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(SAMPLES, len(values)))
    draws = values[indices].mean(axis=1)
    estimates = values.mean(axis=0)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    metrics = {}
    for index, name in enumerate((*ARMS, "difference")):
        metrics[name] = {
            "estimate": float(estimates[index]),
            "low": float(low[index]),
            "high": float(high[index]),
            "se": float(draws[:, index].std(ddof=1)),
        }
    metrics["difference"]["probability_positive"] = probability_positive_from_draws(draws[:, 2])
    return {"weeks": len(weekly), "metrics": metrics}


def record(weekly: pd.DataFrame, arm: str) -> str:
    counts = [weekly[f"{arm}_{suffix}"].sum() for suffix in ("win", "loss", "push_result")]
    return "-".join(f"{number:g}" for number in counts)


def comparison(label: str, weekly: pd.DataFrame, result: dict) -> list[str]:
    if not result["weeks"]:
        return [f"**Measured: {label}:** 0 weeks; paired effect and interval not estimable."]
    delta = result["metrics"]["difference"]
    lines = [
        f"**Measured: {label}:** {len(weekly)} weeks; raw-cover W-L-P "
        f"{record(weekly, ARMS[0])}; push-adjusted W-L-P {record(weekly, ARMS[1])}.",
        f"Paired gain {100 * delta['estimate']:+.4f} grade percentage points "
        f"(95% week-bootstrap interval {100 * delta['low']:+.4f} to "
        f"{100 * delta['high']:+.4f}); probability_positive="
        f"{delta['probability_positive']:.6f}.",
        "",
        "| Ranking | Pool-grade accuracy | 95% interval |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        metric = result["metrics"][arm]
        lines.append(
            f"| {arm} | {100 * metric['estimate']:.4f}% | "
            f"{100 * metric['low']:.4f}% to {100 * metric['high']:.4f}% |"
        )
    raw = weekly.raw_cover_grade.to_numpy()
    adjusted = weekly.push_adjusted_grade.to_numpy()
    lines.extend(
        [
            "",
            f"Paired better/worse/equal weeks: {int((adjusted > raw).sum())}/"
            f"{int((adjusted < raw).sum())}/{int((adjusted == raw).sum())}.",
        ]
    )
    return lines


def write_weekly(weekly: pd.DataFrame) -> None:
    lines = [
        "# LEAD-72 paired weekly nominations",
        "",
        "**Measured:** opener outcomes; W-L-P ties retain equal selection weight.",
        "Each ranking cell is game ID(s); side(s); P(cover); P(push); "
        "rank statistic; grade; tie level.",
        "",
        "| Season | Week | Eligible | Changed | Raw cover | Push adjusted |",
        "|---:|---:|---:|---|---|---|",
    ]
    for row in weekly.to_dict("records"):
        cells = []
        for arm in ARMS:
            cells.append(
                "; ".join(
                    [
                        row[f"{arm}_ids"],
                        row[f"{arm}_sides"],
                        *(
                            f"{row[f'{arm}_{key}']:.9f}"
                            for key in ("cover", "push", "stat", "grade")
                        ),
                        str(row[f"{arm}_tie"]),
                    ]
                )
            )
        lines.append(
            f"| {row['season']} | {row['week']} | {row['candidates']} | "
            f"{row['changed']} | {cells[0]} | {cells[1]} |"
        )
    WEEKLY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sources, ready = inventory()
    if not ready:
        REPORT.write_text(
            "\n".join(
                [
                    "# LEAD-72 source inventory",
                    "",
                    "**Measured:** required local source absent; "
                    "inventory only, zero scoring looks executed. No outcome, interval, "
                    "probability_positive, fitted coefficient or IS/OOS gap is estimable.",
                    "",
                    *sources,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "source_gated", "report": REPORT.as_posix()}))
        return
    rules = PoolRules.from_defaults()
    if (rules.correct_pick_points, rules.incorrect_pick_points, rules.push_points) != (1, 0, 0.5):
        raise ValueError("Pool rules differ from the predeclared push weight")
    frame = load_candidates(rules.push_points)
    weekly = weekly_replay(frame, rules.push_points)
    decisive = weekly.loc[weekly.changed].copy()
    primary = bootstrap(weekly)
    changed = bootstrap(decisive)
    lines = [
        "# LEAD-72 push-adjusted Best Pick replay",
        "",
        *comparison("Decisive weeks (nominee changes)", decisive, changed),
        "",
        *comparison("All declared weeks", weekly, primary),
        "",
        "## Protocol and interpretation",
        "",
        "**Read:** declaration saved in `docs/lanes/lead72.md` before outcome access. "
        "Two declared ranking looks: raw P(served-side cover), and P(cover)+0.5*P(push). "
        "The decisive subset and all six season rows are required descriptive diagnostics; "
        "no subgroup, threshold, model, timing or additional ranking was selected.",
        "",
        "**Read:** `PoolRules` (src/nfl_ats/pool_workbench.py:37-45) fixes win/push/loss "
        "at 1/0.5/0. LEAD-53 Tuesday dispersion eligibility and its exact tie function "
        "are reused (scripts/best_pick_sunday_renomination_eval.py:54-77,228-243). "
        "Tie order: rank statistic rounded to 12 decimals, minimum dispersion, minimum "
        "absolute frozen spread, earliest kickoff, then equal weight across remaining ties.",
        "",
        "**Measured:** ranking leaves every archived probability-selected side unchanged. "
        "Raw cover plus push plus loss sums to one; the saved non-push probability "
        "matches cover/(cover+loss). No smooth residual or future line enters ranking. "
        "Pushes remain in the outcome denominator. The normalized accuracy target uses "
        "the pool 0.5 push grade; it is not total bonus points or wagering return.",
        "",
        f"**Measured:** {len(frame)} games, {len(weekly)} weeks, six seasons; "
        f"{weekly.candidates.sum()} eligible game-week rows; "
        f"{int(weekly.pool_fallback.sum())} dispersion fallback weeks. "
        f"Policies: {', '.join(sorted(frame.base_probability_policy.unique()))}.",
        "",
        f"Paired week-block bootstrap: {SAMPLES:,} draws, seed {SEED}, percentile 95% "
        "intervals; probability_positive credits an exactly zero draw by 0.5. "
        "All-week and decisive-week resampling preserve both arms within each sampled week.",
        "",
        "## Season stability and coefficients",
        "",
        "**Read:** the row explicitly requires deterministic arithmetic and no fold choice. "
        "No parameter was fitted or selected: in-sample fit scores and IS/OOS fitting "
        "gap are N/A. Fitted fold coefficients are N/A; fixed coefficients below apply "
        "in every season. OOS means the saved chronological weekly forecasts, not an "
        "untouched outer test of this research idea. No new LOSO model was fitted.",
        "",
        "| Season | Weeks | Changed | Raw W-L-P | Adjusted W-L-P | Raw OOS grade | "
        "Adjusted OOS grade | Difference (pp) | Raw (cover,push) | Adjusted (cover,push) |",
        "|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    for season, part in weekly.groupby("season", sort=True):
        raw, adjusted = part.raw_cover_grade.mean(), part.push_adjusted_grade.mean()
        lines.append(
            f"| {season} | {len(part)} | {int(part.changed.sum())} | "
            f"{record(part, ARMS[0])} | {record(part, ARMS[1])} | {100 * raw:.4f}% | "
            f"{100 * adjusted:.4f}% | {100 * (adjusted - raw):+.4f} | (1,0) | (1,0.5) |"
        )
    lines.extend(
        [
            "",
            "## Source inventory",
            "",
            "**Measured:** source hashes identify "
            "the exact local replay inputs; only the declared columns were loaded.",
            "",
            *sources,
            "",
            "Paired nomination-level output: `docs/lead72_weekly.md`.",
        ]
    )
    write_weekly(weekly)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "measured",
                "games": len(frame),
                "weeks": len(weekly),
                "decisive": changed,
                "all_weeks": primary,
                "looks": 2,
                "report": REPORT.as_posix(),
                "weekly": WEEKLY.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
