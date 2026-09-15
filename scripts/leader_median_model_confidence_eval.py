from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.sharp_book_movement_features import leader_follow_threshold

POPULATION_PATH = Path("artifacts/sharp_weighted_follow/20260909T233606Z/per_game.parquet")
SEED = 20260914
SAMPLES = 20000
CONFIDENCE_BANDS = [
    ("le_0_52", -np.inf, 0.52),
    ("0_52_to_0_55", 0.52, 0.55),
    ("0_55_to_0_58", 0.55, 0.58),
    ("gt_0_58", 0.58, np.inf),
]
MOVE_BINS = [
    ("0_5_to_1_0", 0.5, 1.0),
    ("1_0_plus", 1.0, np.inf),
]
LOSO_MIN_TRAIN_FLIPS = 15
CONFIDENCE_GATE_CANDIDATES = [0.52, 0.55, 0.58, float(np.inf)]
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def load_population() -> pd.DataFrame:
    raw = pd.read_parquet(POPULATION_PATH)
    graded = raw.loc[raw.correct_at_open_probability_rule.notna()].copy()
    graded = graded.reset_index(drop=True)
    p_home = graded.home_cover_probability_at_open.astype(float)
    card_pick_home = graded.pick_home_at_open_probability_rule.astype(bool)
    fired = graded.s1_leader_only_fired.astype(bool)
    rule_pick_home = graded.s1_leader_only_pick.astype(bool)
    flip = fired & rule_pick_home.ne(card_pick_home)
    served_p = np.where(rule_pick_home, p_home, 1.0 - p_home)
    confidence = np.maximum(p_home, 1.0 - p_home)
    move = graded.leader_median_net.astype(float).abs()
    graded["p_home"] = p_home
    graded["card_pick_home"] = card_pick_home
    graded["card_correct"] = graded.correct_at_open_probability_rule.astype(float)
    graded["fired"] = fired
    graded["rule_pick_home"] = rule_pick_home
    graded["rule_correct"] = graded.s1_leader_only_correct.astype(float)
    graded["flip"] = flip
    graded["served_p"] = served_p
    graded["confidence"] = confidence
    graded["move"] = move

    net = graded.leader_median_net.astype(float)
    line = graded.tue_open_home_spread.astype(float)
    threshold_arms = {
        "t05": pd.Series(0.5, index=graded.index),
        "t0": pd.Series(1.0, index=graded.index),
        "t3": pd.Series([leader_follow_threshold(value) for value in line], index=graded.index),
    }
    for arm_name, threshold in threshold_arms.items():
        arm_fired = net.abs().ge(threshold)
        arm_pick_home = net.gt(0.0)
        arm_flip = arm_fired & arm_pick_home.ne(card_pick_home)
        arm_correct = np.where(arm_flip, 1.0 - graded["card_correct"], graded["card_correct"])
        graded[f"{arm_name}_flip"] = arm_flip
        graded[f"{arm_name}_correct"] = arm_correct
    return graded


def run_sanity_checks(df: pd.DataFrame) -> dict[str, Any]:
    fired = df.fired
    flip = df.flip
    card_argmax_mismatch = int((df.p_home.ge(0.5) != df.card_pick_home).sum())
    flip_complement_ok = bool(
        np.isclose(df.loc[flip, "rule_correct"], 1.0 - df.loc[flip, "card_correct"]).all()
    )
    nonflip_fire_same_ok = bool(
        np.isclose(
            df.loc[fired & ~flip, "rule_correct"], df.loc[fired & ~flip, "card_correct"]
        ).all()
    )
    conf_flip_desc = df.loc[flip, "confidence"].describe().to_dict()
    checks = {
        "graded_games": len(df),
        "fires": int(fired.sum()),
        "flips": int(flip.sum()),
        "card_argmax_mismatch_count": card_argmax_mismatch,
        "flip_rule_correct_is_complement_of_card": flip_complement_ok,
        "nonflip_fire_rule_correct_equals_card_correct": nonflip_fire_same_ok,
        "confidence_among_flips_describe": {k: float(v) for k, v in conf_flip_desc.items()},
        "confidence_all_time_diff_vs_override_distance_max_abs": float(
            ((df.loc[flip, "confidence"] - 0.5) - (0.5 - df.loc[flip, "served_p"])).abs().max()
        ),
    }
    return checks


def paired_effect(
    df: pd.DataFrame, candidate_col: str, baseline_col: str, *, block: str
) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "effect": float(
                (sub[candidate_col].astype(float).mean() - sub[baseline_col].astype(float).mean())
                * 100.0
            )
        }

    result = week_blocked_bootstrap(df, metric_fn, block=block, samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "block": block,
        "blocks": int(df.groupby(["season", "week"] if block == "week" else ["season"]).ngroups),
    }


def correlation_bootstrap(df: pd.DataFrame, col_x: str, col_y: str) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        x = sub[col_x].astype(float).to_numpy()
        y = sub[col_y].astype(float).to_numpy()
        if x.std() == 0.0 or y.std() == 0.0:
            return {"corr": 0.0}
        return {"corr": float(np.corrcoef(x, y)[0, 1])}

    result = week_blocked_bootstrap(df, metric_fn, block="week", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "estimate": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def fit_logit(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-3, iters: int = 30) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        z = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = x.T @ (y - p) - l2 * beta
        hessian = (x.T * w) @ x + l2 * np.eye(x.shape[1])
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        beta = beta + step
    return beta


def logistic_bootstrap(
    df: pd.DataFrame, predictor_cols: list[str], target_col: str
) -> dict[str, Any]:
    means = {c: float(df[c].mean()) for c in predictor_cols}
    stds = {c: float(df[c].std(ddof=0)) or 1.0 for c in predictor_cols}

    def build_design(sub: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(sub))]
        for c in predictor_cols:
            cols.append(((sub[c].astype(float) - means[c]) / stds[c]).to_numpy())
        if len(predictor_cols) == 2:
            interaction = (
                (sub[predictor_cols[0]].astype(float) - means[predictor_cols[0]])
                / stds[predictor_cols[0]]
            ) * (
                (sub[predictor_cols[1]].astype(float) - means[predictor_cols[1]])
                / stds[predictor_cols[1]]
            )
            cols.append(interaction.to_numpy())
        return np.column_stack(cols)

    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        x = build_design(sub)
        y = sub[target_col].astype(float).to_numpy()
        beta = fit_logit(x, y)
        out = {"intercept": float(beta[0])}
        for i, c in enumerate(predictor_cols):
            out[f"slope_std_{c}"] = float(beta[1 + i])
            out[f"slope_per_unit_{c}"] = float(beta[1 + i] / stds[c])
        if len(predictor_cols) == 2:
            out["slope_std_interaction"] = float(beta[-1])
        return out

    result = week_blocked_bootstrap(df, metric_fn, block="week", samples=SAMPLES, seed=SEED)
    out: dict[str, Any] = {}
    for _, row in result.iterrows():
        out[row["metric"]] = {
            "estimate": float(row["estimate"]),
            "interval_low": float(row["lower"]),
            "interval_high": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
        }
    return out


def band_policy_column(df: pd.DataFrame, in_band: pd.Series) -> pd.Series:
    return np.where(df.flip & in_band, df.rule_correct, df.card_correct)


def games_record(correct: pd.Series) -> str:
    wins = int(correct.sum())
    losses = int(len(correct) - wins)
    return f"{wins}-{losses}"


def band_report(df: pd.DataFrame, name: str, in_band: pd.Series) -> dict[str, Any]:
    flips_in_band = df.flip & in_band
    n = int(flips_in_band.sum())
    card_rec = games_record(df.loc[flips_in_band, "card_correct"])
    follow_rec = games_record(df.loc[flips_in_band, "rule_correct"])
    policy_col = f"policy_band_{name}"
    df_local = df.copy()
    df_local[policy_col] = band_policy_column(df, in_band)
    effect = paired_effect(df_local, policy_col, "card_correct", block="week")
    record_look(f"band:{name}")
    foresight_col = f"foresight_band_{name}"
    df_local[foresight_col] = np.where(
        flips_in_band & df.card_correct.eq(0.0), 1.0, df.card_correct
    )
    foresight_effect = paired_effect(df_local, foresight_col, "card_correct", block="week")
    record_look(f"band_foresight:{name}")
    return {
        "band": name,
        "n_flips": n,
        "card_record": card_rec,
        "follow_record": follow_rec,
        "effect_vs_card": effect,
        "foresight_ceiling_effect_vs_card": foresight_effect,
    }


def gate_policy_column(df: pd.DataFrame, cut: float) -> pd.Series:
    gated = df.flip & df.confidence.le(cut)
    return np.where(gated, df.rule_correct, df.card_correct)


def loso_recommended_gate(df: pd.DataFrame) -> dict[str, Any]:
    flip_conf = df.loc[df.flip, "confidence"]
    deciles = sorted({float(flip_conf.quantile(q)) for q in np.arange(0.1, 1.0, 0.1)})
    candidates = [*deciles, float(np.inf)]
    record_look(f"loso_candidate_grid:{len(candidates)}_cuts")

    def training_score(sub: pd.DataFrame, cut: float) -> tuple[float, int]:
        train_flips = int((sub.flip & sub.confidence.le(cut)).sum())
        policy = gate_policy_column(sub, cut)
        effect = float((policy.mean() - sub.card_correct.mean()) * 100.0)
        return effect, train_flips

    seasons = sorted(df.season.unique().tolist())
    fold_choices = {}
    pooled_policy = pd.Series(index=df.index, dtype=float)
    for held_out in seasons:
        train = df.loc[df.season.ne(held_out)]
        best_cut = float(np.inf)
        best_effect = float("-inf")
        for cut in candidates:
            effect, n_train_flips = training_score(train, cut)
            if n_train_flips < LOSO_MIN_TRAIN_FLIPS and cut != float(np.inf):
                continue
            if effect > best_effect:
                best_effect = effect
                best_cut = cut
        fold_choices[str(held_out)] = {"chosen_cut": best_cut, "training_effect": best_effect}
        test = df.loc[df.season.eq(held_out)]
        test_policy = gate_policy_column(test, best_cut)
        pooled_policy.loc[test.index] = test_policy
    record_look("loso_pooled_out_of_sample")

    df_pooled = df.copy()
    df_pooled["loso_policy_correct"] = pooled_policy.astype(float)
    oos_effect = paired_effect(df_pooled, "loso_policy_correct", "card_correct", block="week")
    record_look("loso_pooled_out_of_sample_vs_full_rule")
    oos_vs_full_rule = paired_effect(df_pooled, "loso_policy_correct", "rule_correct", block="week")

    best_in_sample_cut = float(np.inf)
    best_in_sample_effect = float("-inf")
    for cut in candidates:
        effect, n_flips_at_cut = training_score(df, cut)
        if n_flips_at_cut < LOSO_MIN_TRAIN_FLIPS and cut != float(np.inf):
            continue
        if effect > best_in_sample_effect:
            best_in_sample_effect = effect
            best_in_sample_cut = cut
    record_look("in_sample_selected_gate")
    df_in_sample = df.copy()
    df_in_sample["in_sample_policy_correct"] = gate_policy_column(df, best_in_sample_cut).astype(
        float
    )
    in_sample_boot = paired_effect(
        df_in_sample, "in_sample_policy_correct", "card_correct", block="week"
    )

    return {
        "candidate_cuts": [c if np.isfinite(c) else "no_gate" for c in candidates],
        "fold_choices": fold_choices,
        "out_of_sample": oos_effect,
        "out_of_sample_vs_full_rule": oos_vs_full_rule,
        "in_sample_selected_cut": (
            best_in_sample_cut if np.isfinite(best_in_sample_cut) else "no_gate"
        ),
        "in_sample": in_sample_boot,
        "gap_in_sample_minus_out_of_sample": float(in_sample_boot["effect"] - oos_effect["effect"]),
        "pooled_policy_column": pooled_policy,
    }


THRESHOLD_ARMS = ("t05", "t0", "t3")


def combo_policy_column(df: pd.DataFrame, arm_name: str, cut: float) -> pd.Series:
    flip_col = df[f"{arm_name}_flip"]
    correct_col = df[f"{arm_name}_correct"]
    gated = flip_col & df.confidence.le(cut)
    return np.where(gated, correct_col, df.card_correct)


def nested_threshold_confidence_loso(df: pd.DataFrame, *, allow_gate: bool) -> dict[str, Any]:
    cuts = CONFIDENCE_GATE_CANDIDATES if allow_gate else [float(np.inf)]
    combos = [(arm, cut) for arm in THRESHOLD_ARMS for cut in cuts]
    label = "with_gate" if allow_gate else "no_gate"
    record_look(f"nested_loso_candidate_grid_{label}:{len(combos)}_combos")

    def training_score(sub: pd.DataFrame, arm: str, cut: float) -> tuple[float, int]:
        n_used = int((sub[f"{arm}_flip"] & sub.confidence.le(cut)).sum())
        policy = combo_policy_column(sub, arm, cut)
        effect = float((policy.mean() - sub.card_correct.mean()) * 100.0)
        return effect, n_used

    seasons = sorted(df.season.unique().tolist())
    fold_choices: dict[str, Any] = {}
    pooled_policy = pd.Series(index=df.index, dtype=float)
    for held_out in seasons:
        train = df.loc[df.season.ne(held_out)]
        best = None
        best_effect = float("-inf")
        for arm, cut in combos:
            effect, n_used = training_score(train, arm, cut)
            if np.isfinite(cut) and n_used < LOSO_MIN_TRAIN_FLIPS:
                continue
            if effect > best_effect:
                best_effect = effect
                best = (arm, cut)
        if best is None:
            best = ("t05", float(np.inf))
            best_effect, _ = training_score(train, *best)
        fold_choices[str(held_out)] = {
            "chosen_arm": best[0],
            "chosen_cut": best[1] if np.isfinite(best[1]) else "no_gate",
            "training_effect": best_effect,
        }
        test = df.loc[df.season.eq(held_out)]
        pooled_policy.loc[test.index] = combo_policy_column(test, *best)
    record_look(f"nested_loso_pooled_out_of_sample_{label}")

    df_pooled = df.copy()
    col = f"nested_loso_policy_correct_{label}"
    df_pooled[col] = pooled_policy.astype(float)
    oos_effect = paired_effect(df_pooled, col, "card_correct", block="week")

    best_in_sample = None
    best_in_sample_effect = float("-inf")
    for arm, cut in combos:
        effect, n_used = training_score(df, arm, cut)
        if np.isfinite(cut) and n_used < LOSO_MIN_TRAIN_FLIPS:
            continue
        if effect > best_in_sample_effect:
            best_in_sample_effect = effect
            best_in_sample = (arm, cut)
    record_look(f"nested_in_sample_selected_combo_{label}")
    df_in_sample = df.copy()
    in_col = f"nested_in_sample_policy_correct_{label}"
    df_in_sample[in_col] = combo_policy_column(df, *best_in_sample).astype(float)
    in_sample_boot = paired_effect(df_in_sample, in_col, "card_correct", block="week")

    return {
        "allow_confidence_gate": allow_gate,
        "candidate_combos": [
            {"arm": a, "cut": c if np.isfinite(c) else "no_gate"} for a, c in combos
        ],
        "fold_choices": fold_choices,
        "out_of_sample": oos_effect,
        "in_sample_selected_combo": {
            "arm": best_in_sample[0],
            "cut": best_in_sample[1] if np.isfinite(best_in_sample[1]) else "no_gate",
        },
        "in_sample": in_sample_boot,
        "gap_in_sample_minus_out_of_sample": float(in_sample_boot["effect"] - oos_effect["effect"]),
    }


def foresight_full_control(df: pd.DataFrame) -> dict[str, Any]:
    df_local = df.copy()
    df_local["foresight_correct"] = np.where(
        df.flip & df.card_correct.eq(0.0), 1.0, df.card_correct
    )
    effect = paired_effect(df_local, "foresight_correct", "card_correct", block="week")
    record_look("foresight_all_flips")
    n_fixed = int((df.flip & df.card_correct.eq(0.0)).sum())
    return {"n_flips_fixed": n_fixed, "effect_vs_card": effect}


def decile_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    flips = df.loc[df.flip].sort_values("confidence").reset_index(drop=True)
    flips["decile"] = pd.qcut(flips.index, 10, labels=False, duplicates="drop")
    rows = []
    for decile_id, sub in flips.groupby("decile"):
        wins = int(sub.rule_correct.sum())
        n = len(sub)
        rows.append(
            {
                "decile": int(decile_id),
                "n": n,
                "confidence_min": float(sub.confidence.min()),
                "confidence_max": float(sub.confidence.max()),
                "follow_record": f"{wins}-{n - wins}",
                "follow_win_pct": round(100.0 * wins / n, 1) if n else float("nan"),
            }
        )
    record_look("decile_table_descriptive")
    return rows


def two_d_split(df: pd.DataFrame, tercile_edges: tuple[float, float]) -> list[dict[str, Any]]:
    t1, t2 = tercile_edges
    conf_bins = [
        ("low_third", -np.inf, t1),
        ("mid_third", t1, t2),
        ("high_third", t2, np.inf),
    ]
    rows = []
    for conf_name, lo, hi in conf_bins:
        for move_name, mlo, mhi in MOVE_BINS:
            mask = (
                df.flip
                & df.confidence.gt(lo)
                & df.confidence.le(hi)
                & df.move.ge(mlo)
                & df.move.lt(mhi)
            )
            n = int(mask.sum())
            wins = int(df.loc[mask, "rule_correct"].sum()) if n else 0
            rows.append(
                {
                    "confidence_third": conf_name,
                    "move_bin": move_name,
                    "n": n,
                    "follow_record": f"{wins}-{n - wins}",
                    "follow_win_pct": round(100.0 * wins / n, 1) if n else float("nan"),
                }
            )
            record_look(f"2d_cell:{conf_name}x{move_name}")
    return rows


def season_split(df: pd.DataFrame, policy_col: str) -> list[dict[str, Any]]:
    rows = []
    for season, sub in df.groupby("season"):
        candidate_acc = float(sub[policy_col].mean())
        card_acc = float(sub.card_correct.mean())
        rows.append(
            {
                "season": int(season),
                "n_games": len(sub),
                "candidate_accuracy": candidate_acc,
                "card_accuracy": card_acc,
                "effect_points": round((candidate_acc - card_acc) * 100.0, 4),
            }
        )
    return rows


def main() -> None:
    df = load_population()
    sanity = run_sanity_checks(df)

    record_look("headline_full_rule_vs_card_week")
    headline_week = paired_effect(df, "rule_correct", "card_correct", block="week")
    record_look("headline_full_rule_vs_card_season")
    headline_season = paired_effect(df, "rule_correct", "card_correct", block="season")

    flip_conf = df.loc[df.flip, "confidence"]
    tercile_edges = (
        float(flip_conf.quantile(1.0 / 3.0)),
        float(flip_conf.quantile(2.0 / 3.0)),
    )
    tercile_bands = [
        ("low_third", -np.inf, tercile_edges[0]),
        ("mid_third", tercile_edges[0], tercile_edges[1]),
        ("high_third", tercile_edges[1], np.inf),
    ]

    tercile_reports = []
    for name, lo, hi in tercile_bands:
        in_band = df.confidence.gt(lo) & df.confidence.le(hi)
        tercile_reports.append(band_report(df, name, in_band))

    fixed_band_reports = []
    for name, lo, hi in CONFIDENCE_BANDS:
        in_band = df.confidence.gt(lo) & df.confidence.le(hi)
        fixed_band_reports.append(band_report(df, name, in_band))

    flips_df = df.loc[df.flip].copy()
    record_look("correlation_confidence_vs_follow_correct")
    corr_conf = correlation_bootstrap(flips_df, "confidence", "rule_correct")
    record_look("correlation_move_vs_follow_correct")
    corr_move = correlation_bootstrap(flips_df, "move", "rule_correct")

    record_look("logistic_confidence_only")
    logit_conf = logistic_bootstrap(flips_df, ["confidence"], "rule_correct")
    record_look("logistic_move_only")
    logit_move = logistic_bootstrap(flips_df, ["move"], "rule_correct")
    record_look("logistic_confidence_move_interaction")
    logit_interaction = logistic_bootstrap(flips_df, ["confidence", "move"], "rule_correct")

    two_d = two_d_split(df, tercile_edges)

    gated_arm_reports = []
    for name, _lo, hi in tercile_bands[:-1]:
        cut = hi
        df_local = df.copy()
        col = f"gate_{name}"
        df_local[col] = gate_policy_column(df, cut).astype(float)
        record_look(f"gated_arm_descriptive:{name}")
        vs_card = paired_effect(df_local, col, "card_correct", block="week")
        vs_full_rule = paired_effect(df_local, col, "rule_correct", block="week")
        n_fired_under_cut = int((df.flip & df.confidence.le(cut)).sum())
        gated_arm_reports.append(
            {
                "arm": f"follow_only_conf_le_{name}_edge",
                "cut": cut,
                "n_flips_followed": n_fired_under_cut,
                "vs_card": vs_card,
                "vs_full_rule": vs_full_rule,
            }
        )

    for name, _lo, hi in CONFIDENCE_BANDS[:-1]:
        cut = hi
        df_local = df.copy()
        col = f"gate_fixed_{name}"
        df_local[col] = gate_policy_column(df, cut).astype(float)
        record_look(f"gated_arm_descriptive_fixed:{name}")
        vs_card = paired_effect(df_local, col, "card_correct", block="week")
        vs_full_rule = paired_effect(df_local, col, "rule_correct", block="week")
        n_fired_under_cut = int((df.flip & df.confidence.le(cut)).sum())
        gated_arm_reports.append(
            {
                "arm": f"follow_only_conf_le_{cut}",
                "cut": cut,
                "n_flips_followed": n_fired_under_cut,
                "vs_card": vs_card,
                "vs_full_rule": vs_full_rule,
            }
        )

    foresight = foresight_full_control(df)
    loso = loso_recommended_gate(df)
    nested_no_gate = nested_threshold_confidence_loso(df, allow_gate=False)
    nested_with_gate = nested_threshold_confidence_loso(df, allow_gate=True)

    median_conf = float(flip_conf.median())
    shallow_mask = df.flip & df.confidence.le(median_conf)
    deep_mask = df.flip & df.confidence.gt(median_conf)
    median_split = {
        "median_confidence_among_flips": median_conf,
        "shallow_n": int(shallow_mask.sum()),
        "deep_n": int(deep_mask.sum()),
        "shallow_card_record": games_record(df.loc[shallow_mask, "card_correct"]),
        "shallow_follow_record": games_record(df.loc[shallow_mask, "rule_correct"]),
        "deep_card_record": games_record(df.loc[deep_mask, "card_correct"]),
        "deep_follow_record": games_record(df.loc[deep_mask, "rule_correct"]),
    }
    record_look("median_split_reproduction")
    shallow_report = band_report(df, "median_split_shallow", df.confidence.le(median_conf))
    deep_report = band_report(df, "median_split_deep", df.confidence.gt(median_conf))
    median_split["shallow_effect_vs_card"] = shallow_report["effect_vs_card"]
    median_split["deep_effect_vs_card"] = deep_report["effect_vs_card"]

    deciles = decile_table(df)

    loso_pooled_policy = loso.pop("pooled_policy_column")
    df_for_season = df.copy()
    df_for_season["loso_policy_correct"] = loso_pooled_policy.astype(float)
    loso_season_split = season_split(df_for_season, "loso_policy_correct")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("artifacts/leader_median_model_confidence") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "population": str(POPULATION_PATH),
        "graded_games": len(df),
        "seed": SEED,
        "samples": SAMPLES,
        "sanity_checks": sanity,
        "headline": {
            "week_blocked": headline_week,
            "season_blocked": headline_season,
            "stored_promotion_read_docs_sharp_weighted_follow": {
                "effect": 3.0037546933667083,
                "interval_low": -0.9925866559807653,
                "interval_high": 6.95322376738306,
                "probability_positive": 0.9304,
            },
        },
        "tercile_edges": tercile_edges,
        "tercile_bands": tercile_reports,
        "fixed_bands": fixed_band_reports,
        "correlation_confidence_vs_follow_correct": corr_conf,
        "correlation_move_vs_follow_correct": corr_move,
        "logistic_confidence_only": logit_conf,
        "logistic_move_only": logit_move,
        "logistic_confidence_move_interaction": logit_interaction,
        "two_d_split": two_d,
        "gated_arms_descriptive_in_sample": gated_arm_reports,
        "foresight_positive_control_all_flips": foresight,
        "loso_recommended_gate": loso,
        "loso_recommended_gate_season_split": loso_season_split,
        "nested_threshold_confidence_loso_no_gate": nested_no_gate,
        "nested_threshold_confidence_loso_with_gate": nested_with_gate,
        "median_split_reproduction": median_split,
        "decile_table": deciles,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    flip_rows = df.loc[df.flip].copy()
    flip_rows["decile"] = pd.qcut(
        flip_rows.confidence.rank(method="first"), 10, labels=False, duplicates="drop"
    )
    export_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "tue_open_home_spread",
        "card_pick_home",
        "p_home",
        "confidence",
        "move",
        "leader_books",
        "rule_pick_home",
        "card_correct",
        "rule_correct",
        "decile",
    ]
    flip_rows[export_cols].to_csv(out_dir / "flips.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print(f"fires={sanity['fires']} flips={sanity['flips']}")
    print(
        "headline_week effect={:.4f} [{:.4f}, {:.4f}] P+={:.4f}".format(
            headline_week["effect"],
            headline_week["interval_low"],
            headline_week["interval_high"],
            headline_week["probability_positive"],
        )
    )
    print("median_split", json.dumps(median_split))
    print("look_count", len(LOOKS))
    print("loso_out_of_sample", json.dumps(loso["out_of_sample"]))
    print("loso_in_sample", json.dumps(loso["in_sample"]))
    print("loso_gap", loso["gap_in_sample_minus_out_of_sample"])
    print(
        "nested_no_gate out_of_sample",
        json.dumps(nested_no_gate["out_of_sample"]),
        "in_sample",
        json.dumps(nested_no_gate["in_sample"]),
        "gap",
        nested_no_gate["gap_in_sample_minus_out_of_sample"],
    )
    print("nested_no_gate fold_choices", json.dumps(nested_no_gate["fold_choices"]))
    print(
        "nested_with_gate out_of_sample",
        json.dumps(nested_with_gate["out_of_sample"]),
        "in_sample",
        json.dumps(nested_with_gate["in_sample"]),
        "gap",
        nested_with_gate["gap_in_sample_minus_out_of_sample"],
    )
    print("nested_with_gate fold_choices", json.dumps(nested_with_gate["fold_choices"]))


if __name__ == "__main__":
    main()
