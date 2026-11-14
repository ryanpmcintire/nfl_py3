import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod25_wfj_forward as wfj

ROOT = wfj.REPO
PICK_ROOT = ROOT / "artifacts" / "pick_probability"
BOOT_DRAWS = 20000
BOOT_CHUNK = 500
BOOT_SEED = 20261114
NEWTON_STEPS = 100
NEWTON_TOLERANCE = 1e-10
COLS = ("served_logit", "student_ats_margin")
PER_GAME_COLS = [
    "game_id",
    "season",
    "tue_open_home_spread",
    "result",
    "home_covered",
    "out_of_season_home_probability",
]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return np.log(p / (1.0 - p))


def log_loss(p, y):
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(frame, cols):
    return np.column_stack([np.ones(len(frame)), *(frame[c].to_numpy(float) for c in cols)])


def fit_logit(x, y):
    beta = np.zeros(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = sigmoid(x @ beta)
        step = np.linalg.solve((x * (p * (1.0 - p))[:, None]).T @ x, x.T @ (y - p))
        beta = beta + step
        if np.abs(step).max() < NEWTON_TOLERANCE:
            return beta
    raise SystemExit("logistic fit did not converge")


def held_out(frame, cols):
    p = np.full(len(frame), np.nan)
    coefs = {}
    y = frame["y"].to_numpy(float)
    for season in sorted(frame["season"].unique()):
        train = frame["season"].ne(season).to_numpy()
        beta = fit_logit(design(frame.loc[train], cols), y[train])
        p[~train] = sigmoid(design(frame.loc[~train], cols) @ beta)
        coefs[str(int(season))] = [float(v) for v in beta]
    return p, coefs


def interval(delta, rng):
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    means = np.concatenate(
        [
            delta[rng.integers(0, n, size=(BOOT_CHUNK, n))].mean(axis=1)
            for _ in range(BOOT_DRAWS // BOOT_CHUNK)
        ]
    )
    return {
        "mean": float(delta.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "probability_positive": float((means > 0).mean() + 0.5 * (means == 0).mean()),
    }


def record(correct):
    return f"{int(correct.sum())}-{int(len(correct) - correct.sum())}"


def load():
    folder = sorted(p for p in PICK_ROOT.iterdir() if (p / "per_game.parquet").exists())[-1]
    path = folder / "per_game.parquet"
    games = pd.read_parquet(path, columns=PER_GAME_COLS)
    games = games.loc[games["season"].lt(wfj.SEASON)].copy()
    student = wfj.distilled_frame()[["game_id", wfj.ARM]]
    games = games.merge(student.rename(columns={wfj.ARM: "distilled_margin"}), on="game_id")
    games["served_logit"] = logit(games["out_of_season_home_probability"].to_numpy(float))
    games["student_ats_margin"] = games["distilled_margin"] - games["tue_open_home_spread"]
    games["y"] = games["home_covered"]
    usable = games[["y", *COLS]].notna().all(axis=1)
    return path, games, games.loc[usable].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the served-base version of the joint-engine distilled student term: a "
            "logistic fit of the opener cover on the served probability's logit and the "
            "student's margin against the opener, on seasons before the forward season. "
            "Research only; serves nothing."
        )
    )
    parser.parse_args()
    if wfj.SERVED_FROZEN.exists():
        raise SystemExit(f"already frozen: {wfj.SERVED_FROZEN}")
    path, games, frame = load()
    assert int(frame["season"].max()) < wfj.SEASON
    rng = np.random.default_rng(BOOT_SEED)
    y = frame["y"].to_numpy(float)
    served = frame["out_of_season_home_probability"].to_numpy(float)
    full = fit_logit(design(frame, COLS), y)
    control = fit_logit(design(frame, COLS[:1]), y)
    p_in = sigmoid(design(frame, COLS) @ full)
    p_control_in = sigmoid(design(frame, COLS[:1]) @ control)
    p_out, folds = held_out(frame, COLS)
    p_control_out, _ = held_out(frame, COLS[:1])
    changed = (p_out >= 0.5) != (served >= 0.5)
    right = ((p_out >= 0.5) == (y == 1)).astype(float)
    served_right = ((served >= 0.5) == (y == 1)).astype(float)
    by_season = {}
    for season, part in frame.groupby("season"):
        i = part.index.to_numpy()
        by_season[str(int(season))] = {
            "games": len(i),
            "log_loss_gain_v_served": float(
                (log_loss(served[i], y[i]) - log_loss(p_out[i], y[i])).mean()
            ),
            "changed_picks": int(changed[i].sum()),
            "changed_picks_won": int(right[i][changed[i]].sum()),
        }
    coefficients = [float(v) for v in full]
    out = {
        "term": "mod25_wfj_served",
        "term_definition": (
            "home cover probability = sigmoid(intercept + served_logit_coefficient * logit of "
            "the served home probability + b_logit_per_point * (distilled home margin of the "
            "joint-engine hgb_nomkt student minus the home spread)); the fit uses the opener "
            "and out_of_season_home_probability, the forward record uses the pool line and "
            "the published card probability"
        ),
        "intercept": coefficients[0],
        "served_logit_coefficient": coefficients[1],
        "b_logit_per_point": coefficients[2],
        "loso_coefficients_by_heldout_season": folds,
        "loso_b_logit_per_point_min": min(v[2] for v in folds.values()),
        "loso_b_logit_per_point_max": max(v[2] for v in folds.values()),
        "loso_served_logit_coefficient_min": min(v[1] for v in folds.values()),
        "loso_served_logit_coefficient_max": max(v[1] for v in folds.values()),
        "control_served_only_coefficients": [float(v) for v in control],
        "log_loss": {
            "constant_half": float(np.log(2.0)),
            "served_as_is": float(log_loss(served, y).mean()),
            "control_in_sample": float(log_loss(p_control_in, y).mean()),
            "control_out_of_season": float(log_loss(p_control_out, y).mean()),
            "term_in_sample": float(log_loss(p_in, y).mean()),
            "term_out_of_season": float(log_loss(p_out, y).mean()),
            "term_out_of_season_minus_in_sample": float(
                log_loss(p_out, y).mean() - log_loss(p_in, y).mean()
            ),
        },
        "out_of_season_log_loss_gain_v_served": interval(
            log_loss(served, y) - log_loss(p_out, y), rng
        ),
        "out_of_season_log_loss_gain_v_control": interval(
            log_loss(p_control_out, y) - log_loss(p_out, y), rng
        ),
        "in_sample_log_loss_gain_v_served": float((log_loss(served, y) - log_loss(p_in, y)).mean()),
        "out_of_season_accuracy_points_v_served": interval(100.0 * (right - served_right), rng),
        "served_record": record(served_right),
        "term_out_of_season_record": record(right),
        "changed_picks_v_served": int(changed.sum()),
        "changed_picks_won": int(right[changed].sum()),
        "by_season": by_season,
        "looks": 1,
        "student_margin_result_correlation": float(
            np.corrcoef(frame["distilled_margin"], frame["result"])[0, 1]
        ),
        "fit_seasons": [int(frame["season"].min()), int(frame["season"].max())],
        "fit_games": len(games),
        "fit_usable_games": len(frame),
        "fit_base": (
            "logit of out_of_season_home_probability from the pick_probability artifact; "
            "opener grade; opener pushes are absent from the artifact"
        ),
        "base_artifact": str(path.relative_to(ROOT)),
        "base_artifact_sha256": sha(path),
        "student_arm": wfj.ARM,
        "student_files_sha256": {
            f"T{t}": sha(wfj.PIPE / f"T{t}" / "real_distilled.parquet")
            for t in range(2020, wfj.STUDENT_T + 1)
        },
        "fit_script": "scripts/mod25_wfj_served_freeze.py",
        "target_season": wfj.SEASON,
        "declared_first_week": wfj.SERVED_FIRST_WEEK,
        "no_target_season_outcomes": True,
        "coefficient_hash": hashlib.sha256(json.dumps(coefficients).encode()).hexdigest(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    wfj.SERVED_DIR.mkdir(parents=True, exist_ok=True)
    with open(wfj.SERVED_FROZEN, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
