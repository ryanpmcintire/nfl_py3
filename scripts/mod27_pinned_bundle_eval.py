import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod27_fenced_repairs_replay as rp  # noqa: E402

import nfl_ats.pick_probability_fit as ppf  # noqa: E402
from nfl_ats.pick_probability import active_market_move_feature_version  # noqa: E402
from nfl_ats.pick_probability_fit import (  # noqa: E402
    _fit_offset_coefficient,
    build_fit_population,
)

PATCHED = REPO / "tests" / "scratch" / "mod27_bundle" / "b" / "src" / "nfl_ats" / "pool_hook.py"
OUT = REPO / "artifacts" / "mod27_pinned_bundle"
OPENER = REPO / "artifacts" / "opener_evaluation" / "20261101T151614Z"
STORED_METADATA = json.loads((rp.PICK / "metadata.json").read_text())
COUNTED_AT_STORED_FIT = STORED_METADATA["counted_flag_columns"]
FOUR = rp.FOUR
LIVE_HOOK_COEFFICIENT = 0.8052599733299968


def load_patched():
    spec = importlib.util.spec_from_file_location("pool_hook_patched", PATCHED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def logit(p):
    p = np.clip(np.asarray(p, float), rp.EPS, 1 - rp.EPS)
    return np.log(p / (1 - p))


def guarded(frame, structures, repro):
    f = frame.copy()
    for i, g in enumerate(f.game_id):
        s = structures.get(g)
        if s is not None and repro.get(g, True) and len(s["lb"]) < rp.MIN_LEADERS:
            f.loc[f.index[i], rp.MOVE_COLUMN] = 0.0
            f.loc[f.index[i], rp.MOVE_AVAILABLE_COLUMN] = 0.0
    return f


def stage_one_logits(frame, features, stack_rows, stack_frame):
    out = np.full(len(stack_frame), np.nan)
    coefs = {}
    for season in sorted(frame.season.unique()):
        fit = rp.Fit(features, frame[frame.season != season])
        held = (stack_frame.season == season).to_numpy()
        out[held] = fit.design(stack_frame[held]) @ fit.beta
        coefs[int(season)] = fit
    return out, coefs


def stacked_arm(stack, values, base_logit, folds, features, frame_rows):
    y = stack.home_covered.to_numpy(float)
    p_base = sigmoid(base_logit)
    p_term = np.full(len(stack), np.nan)
    fold_coef = {}
    for season in sorted(stack.season.unique()):
        fit = folds[int(season)]
        rest = (stack.season != season).to_numpy()
        held = ~rest
        rest_logit = fit.design(frame_rows[rest]) @ fit.beta
        c = _fit_offset_coefficient(values[rest], rest_logit, y[rest])
        fold_coef[int(season)] = round(float(c), 4)
        p_term[held] = sigmoid(base_logit[held] + c * values[held])
    return p_base, p_term, fold_coef


def summarize(stack, pa, pb):
    y = stack.home_covered.to_numpy(float)
    dis = (pa >= 0.5) != (pb >= 0.5)
    return {
        "changed_sides": int(dis.sum()),
        "decisive_record_old": rp.record(pa[dis], y[dis]) if dis.any() else "0-0",
        "decisive_record_new": rp.record(pb[dis], y[dis]) if dis.any() else "0-0",
        "record_old": rp.record(pa, y),
        "record_new": rp.record(pb, y),
        "paired_gain_new_over_old": rp.paired(stack, pa, pb, "season_week"),
        "log_loss_old": float(rp.log_loss(pa, y).mean()),
        "log_loss_new": float(rp.log_loss(pb, y).mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ph = load_patched()
    opener_metadata = json.loads((OPENER / "metadata.json").read_text())
    ppf.find_matching_opener_evaluation = lambda root, active: (opener_metadata, OPENER)
    artifacts = REPO / "artifacts"
    version = active_market_move_feature_version(artifacts)
    every, _ = build_fit_population(
        artifacts, REPO / "data", market_move_feature_version=version, keep_pushes=True
    )
    every = every.reset_index(drop=True)
    every[rp.FLAG_SUM_COLUMN] = every[COUNTED_AT_STORED_FIT].astype(float).sum(axis=1)
    if rp.FLAG_SUM_COLUMN != "composition_flag_sum" and "composition_flag_sum" in every:
        every = every.rename(columns={"composition_flag_sum": rp.FLAG_SUM_COLUMN})
    pop_mask = every["margin_vs_open"].ne(0.0).to_numpy()
    hist = rp.structs(pd.read_pickle(rp.INPUTS / "hist_books.pkl"))
    live = rp.structs(pd.read_pickle(rp.INPUTS / "live_books.pkl"))
    served_move = every.set_index("game_id")[rp.MOVE_COLUMN]
    repro = {
        g: abs(rp.move_of(s, "served")[0] - served_move[g]) < 1e-9
        for g, s in hist.items()
        if g in served_move.index
    }
    every_g = guarded(every, hist, repro)
    pop = every[pop_mask].reset_index(drop=True)
    pop_g = every_g[pop_mask].reset_index(drop=True)
    y = pop.home_covered.to_numpy(float)
    result = {"population": len(pop), "all_rows": len(every)}

    p_s = rp.loso(pop, FOUR)
    p_g = rp.loso(pop_g, FOUR)
    result["served_loso_record"] = rp.record(p_s, y)
    pop2 = pop.assign(season_week=pop.season * 100 + pop.week)
    dis = ((p_s >= 0.5) != (p_g >= 0.5)).to_numpy()
    full_s = rp.Fit(FOUR, pop)
    full_g = rp.Fit(FOUR, pop_g)
    result["guard_game_level"] = {
        "rows_changed": int(
            (
                (pop_g[rp.MOVE_COLUMN] - pop[rp.MOVE_COLUMN]).abs().gt(1e-9)
                | (pop_g[rp.MOVE_AVAILABLE_COLUMN] - pop[rp.MOVE_AVAILABLE_COLUMN]).abs().gt(1e-9)
            ).sum()
        ),
        "changed_sides": int(dis.sum()),
        "decisive_record_served": rp.record(p_s[dis], y[dis]) if dis.any() else "0-0",
        "decisive_record_guard": rp.record(p_g[dis], y[dis]) if dis.any() else "0-0",
        "record_served": rp.record(p_s, y),
        "record_guard": rp.record(p_g, y),
        "paired_gain_guard": rp.paired(pop2, p_s.to_numpy(), p_g.to_numpy(), "season_week"),
        "in_sample_ll": {
            "served": float(rp.log_loss(full_s.predict(pop), y).mean()),
            "guard": float(rp.log_loss(full_g.predict(pop_g), y).mean()),
        },
        "oos_ll": {
            "served": float(rp.log_loss(p_s, y).mean()),
            "guard": float(rp.log_loss(p_g, y).mean()),
        },
        "move_coef_per_fold_guard": {
            int(s): round(rp.Fit(FOUR, pop_g[pop_g.season != s]).natural()[rp.MOVE_COLUMN], 4)
            for s in sorted(pop.season.unique())
        },
        "move_coef_per_fold_served": {
            int(s): round(rp.Fit(FOUR, pop[pop.season != s]).natural()[rp.MOVE_COLUMN], 4)
            for s in sorted(pop.season.unique())
        },
    }

    stacked = ph.hook_rows(every)
    rows = stacked["row"].to_numpy()
    stack = stacked.reset_index(drop=True)
    stack["week"] = every["week"].to_numpy()[rows]
    stack["season_week"] = stack.season * 100 + stack.week
    frame_s = every.iloc[rows].reset_index(drop=True)
    frame_g = every_g.iloc[rows].reset_index(drop=True)
    values = stack[ph.HOOK_COLUMN].to_numpy(float)
    offsets = (stack.pool_line.to_numpy() - every["tue_open_home_spread"].to_numpy()[rows]).round(2)
    stack["offset"] = offsets
    logit_s, folds_s = stage_one_logits(pop, FOUR, rows, frame_s)
    logit_g, folds_g = stage_one_logits(pop_g, FOUR, rows, frame_g)
    result["stack"] = {
        "rows": len(stack),
        "rows_with_nonzero_value": int((values != 0).sum()),
        "per_offset_rows": {str(k): int((offsets == k).sum()) for k in sorted(set(offsets))},
    }
    ps_base, p_o, coef_o = stacked_arm(stack, values, logit_s, folds_s, FOUR, frame_s)
    pg_base, p_b, coef_b = stacked_arm(stack, values, logit_g, folds_g, FOUR, frame_g)
    result["arm_O_offset_on_served_stage_one"] = {
        "coef_per_fold": coef_o,
        **summarize(stack, ps_base, p_o),
    }
    result["arm_B_bundle_vs_served"] = {
        "coef_per_fold": coef_b,
        **summarize(stack, ps_base, p_b),
    }
    result["arm_B_bundle_vs_guard_only_stack"] = summarize(stack, pg_base, p_b)
    for name, mask in (
        ("abs_0.5", np.abs(offsets) == 0.5),
        ("abs_1.0", np.abs(offsets) == 1.0),
        ("whole_market", ph.whole_mask(every["tue_open_home_spread"])[rows]),
        ("half_market", ~ph.whole_mask(every["tue_open_home_spread"])[rows]),
    ):
        sub = stack[mask].reset_index(drop=True)
        result[f"arm_O_cell_{name}"] = summarize(sub, ps_base[mask], p_o[mask])

    table = ph.push_mass_table(every)
    c_full = _fit_offset_coefficient(
        values, full_s.design(frame_s) @ full_s.beta, stack.home_covered.to_numpy(float)
    )
    result["offset_coef_full_data"] = float(c_full)

    G = pd.read_parquet(rp.INPUTS / "g26.parquet").reset_index(drop=True)
    G["model_logit"] = logit(G.discrete)
    po = pd.read_csv(rp.INPUTS / "pool_offset2.csv").set_index("game_id")
    Gs = G
    Gg = guarded(G, live, {})
    base_s = full_s.design(Gs) @ full_s.beta
    base_g = full_g.design(Gg) @ full_g.beta
    old_value = np.zeros(len(G))
    new_value = np.zeros(len(G))
    for i, g in enumerate(G.game_id):
        if g not in po.index:
            continue
        pool, market = float(po.loc[g, "pool"]), float(po.loc[g, "tue_mkt"])
        new_value[i] = ph.offset_value(market, pool, table)
        if ph.is_whole(market) and np.isclose(abs(pool - market), 0.5):
            mass = table.get(market)
            if mass is not None:
                old_value[i] = ph.hook_value(np.array([np.sign(pool - market)]), np.array([mass]))[
                    0
                ]
    p_old = sigmoid(base_s + LIVE_HOOK_COEFFICIENT * old_value)
    p_new = sigmoid(base_g + c_full * new_value)
    p_g_only = sigmoid(base_g + LIVE_HOOK_COEFFICIENT * old_value)
    result["forward_2026_w2_w8"] = {
        "games": len(G),
        "games_with_nonzero_new_value": int((new_value != 0).sum()),
        "games_with_nonzero_old_hook": int((old_value != 0).sum()),
        "thin_leader_games": int(sum(1 for s in live.values() if len(s["lb"]) < rp.MIN_LEADERS)),
        "week_table_bundle_vs_served_plus_hook": rp.week_table(G, p_old, p_new),
        "changed_games_bundle": rp.changed_games(G, p_old, p_new),
        "changed_games_guard_only": rp.changed_games(G, p_old, p_g_only),
        "record_served_plus_hook": rp.record(p_old, G.home_covered.to_numpy(float)),
        "record_bundle": rp.record(p_new, G.home_covered.to_numpy(float)),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=float))
    print("done", OUT / "results.json")


if __name__ == "__main__":
    main()
