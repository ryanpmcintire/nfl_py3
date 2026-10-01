import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_SERVED,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import (
    _MARGIN_PROFILE_FEATURE_SETS,
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    fit_margin_model,
    make_margin_estimator,
    margin_feature_columns,
    resolve_feature_groups,
)
from nfl_ats.mass_preserving_lattice import (
    DiscretePushReader,
    prior_pool,
    serve_discrete_three_way,
)
from nfl_ats.modeling import regular_season_rows

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod24_u1"
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
SEL_TRUE = ROOT / "artifacts" / "mod23_unit1" / "selection.json"
SEL_PROXY = OUT / "selection_proxy.json"
TERMS5 = ROOT / "artifacts" / "mod23_unit5" / "terms.parquet"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
BASE_ALPHA = 10.0
ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
PROXY_SEASONS = tuple(range(2011, 2020))
TRUE_SEASONS = tuple(range(2020, 2026))
INNER_SEASONS = 4
FIRST_INNER = 2010
MIN_RELATIVE_GAIN = 0.0005
PINNED = "market"
PROB_METHOD = "gaussian_median"
BOOT = 20000
SEED = 20260817
FULL = tuple(margin_feature_columns("market_residual", PROFILE))
COLFAM = dict(zip(FULL, resolve_feature_groups(FULL), strict=True))
FAMILIES = sorted({f for f in COLFAM.values() if f != PINNED})
SUPP = _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ())


def cols_for(dropped):
    return tuple(c for c in FULL if COLFAM[c] not in dropped)


def inner_mse(frame, columns, alpha, seasons):
    total = 0.0
    count = 0
    for s in seasons:
        train = frame.loc[frame["season"].lt(s)]
        test = frame.loc[frame["season"].eq(s)]
        est = make_margin_estimator("ridge", 42, ridge_alpha=alpha, suppressed_indicator_columns=SUPP)
        est.fit(train.loc[:, list(columns)], train["ats_margin"])
        pred = np.asarray(est.predict(test.loc[:, list(columns)]), dtype=float)
        err = test["ats_margin"].to_numpy(dtype=float) - pred
        total += float(np.sum(err**2))
        count += len(err)
    return total / count


def best_alpha(frame, columns, inner):
    scores = {a: inner_mse(frame, columns, a, inner) for a in ALPHA_GRID}
    a = min(ALPHA_GRID, key=lambda x: (scores[x], abs(x - BASE_ALPHA)))
    return a, scores[a]


def select_alpha_only(frame, season):
    inner = list(range(max(FIRST_INNER, season - INNER_SEASONS), season))
    a, s = best_alpha(frame, FULL, inner)
    return {"alpha": a, "dropped": [], "inner_mse": s}


def select_greedy(frame, season):
    inner = list(range(max(FIRST_INNER, season - INNER_SEASONS), season))
    dropped = frozenset()
    alpha, current = best_alpha(frame, cols_for(dropped), inner)
    while True:
        cands = []
        for fam in FAMILIES:
            if fam in dropped:
                continue
            trial = dropped | {fam}
            if not cols_for(trial):
                continue
            a, s = best_alpha(frame, cols_for(trial), inner)
            cands.append((s, fam, a))
        if not cands:
            break
        s, fam, a = min(cands)
        if s < current * (1.0 - MIN_RELATIVE_GAIN):
            dropped = dropped | {fam}
            alpha, current = a, s
        else:
            break
    return {"alpha": alpha, "dropped": sorted(dropped), "inner_mse": current}


def patched(columns):
    class Ctx:
        def __enter__(self):
            self.orig = FEATURE_SETS[SET_NAME]
            FEATURE_SETS[SET_NAME] = tuple(columns)

        def __exit__(self, *a):
            FEATURE_SETS[SET_NAME] = self.orig

    return Ctx()


def proxy_population(features):
    reg = regular_season_rows(features)
    reg = reg.loc[reg["season"].between(PROXY_SEASONS[0], PROXY_SEASONS[-1])]
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"])
    sbr = sbr.drop_duplicates("game_id")
    pop = reg[["game_id", "season", "week", "result"]].merge(
        sbr[["game_id", "open_home_spread"]], on="game_id", how="inner"
    )
    pop = pop.loc[pd.to_numeric(pop["result"], errors="coerce").notna()].copy()
    pop["season"] = pop["season"].astype(int)
    pop["week"] = pop["week"].astype(int)
    return pop.rename(columns={"open_home_spread": "open_line"})


def proxy_eval(features, pop, spec):
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    completed = frame.loc[frame["result"].notna()].copy()
    pool = prior_pool(frame, pop.set_index("game_id")["open_line"])
    rows = []
    stream = pd.DataFrame()
    for (season, week), group in pop.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
        cutoff = target["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < DEFAULT_MIN_TRAIN_GAMES:
            continue
        columns, alpha = spec(season)
        with patched(columns):
            model = fit_margin_model(
                training,
                target="market_residual",
                model_name="ridge",
                feature_profile=PROFILE,
                ridge_alpha=alpha,
            )
            reader = DiscretePushReader.for_week(
                pool,
                season=int(season),
                week=int(week),
                cutoff=pd.Timestamp(cutoff),
                exclude_game_ids=set(week_rows["game_id"].astype(str)),
            )
            scoring = week_rows.merge(group[["game_id", "open_line"]], on="game_id").copy()
            at_open = scoring.copy()
            at_open["spread_line"] = at_open["open_line"]
            pred = model.predict(at_open, probability_method=PROB_METHOD)
            if HOME_SIDE_OFFSET_SERVED and not stream.empty:
                fitted = fit_home_side_offsets(prior_rows_before(stream, int(season), int(week)))
                off = fitted.offset_for(at_open["spread_line"]).fillna(0.0).to_numpy(dtype=float)
            else:
                off = np.zeros(len(at_open))
            served = (
                model.predict(at_open, probability_method=PROB_METHOD, center_offset=off)
                if np.any(off != 0.0)
                else pred
            )
            disc = serve_discrete_three_way(
                served, at_open, reader, residuals=model.residuals, probability_method=PROB_METHOD
            )
        res = pred["predicted_market_residual"].to_numpy()
        out = scoring[["game_id", "result", "open_line"]].copy()
        out["season"] = int(season)
        out["week"] = int(week)
        out["p"] = disc["home_cover_probability"].to_numpy()
        rows.append(out)
        ws = pd.DataFrame(
            {
                "game_id": scoring["game_id"].astype(str).to_numpy(),
                "season": int(season),
                "week": int(week),
                "spread_line": scoring["open_line"].to_numpy(dtype=float),
                "point_incumbent": scoring["open_line"].to_numpy(dtype=float) + res,
                "result": pd.to_numeric(scoring["result"], errors="coerce").to_numpy(),
            }
        )
        stream = ws if stream.empty else pd.concat([stream, ws], ignore_index=True)
    out = pd.concat(rows, ignore_index=True)
    out["margin_vs_open"] = out["result"] - out["open_line"]
    return finish(out, out["p"])


def finish(df, p):
    df = df.copy()
    df["pick_home"] = p.to_numpy() >= 0.5
    m = df["margin_vs_open"]
    df["correct"] = np.where(m.eq(0), np.nan, np.where(df["pick_home"], m.gt(0), ~m.gt(0)))
    df = df.loc[df["correct"].notna()].copy()
    df["y"] = (df["margin_vs_open"] > 0).astype(float)
    pc = df["p"].clip(1e-6, 1 - 1e-6)
    df["ll"] = -(df["y"] * np.log(pc) + (1 - df["y"]) * np.log(1 - pc))
    df["brier"] = (pc - df["y"]) ** 2
    return df.set_index("game_id")[["season", "week", "p", "pick_home", "correct", "ll", "brier"]]


def true_eval(features, columns, alpha):
    cfg = {
        "feature_profile": PROFILE,
        "regressor": "ridge",
        "target": "market_residual",
        "probability_method": PROB_METHOD,
        "calibration_method": "none",
        "ridge_alpha": alpha,
        "mod24_variant": json.dumps([list(columns), alpha]),
    }
    with patched(columns):
        scored = clv.opener_pick_evaluation(
            u3.MARKET_ROOT, features, active_model_config=cfg, min_train_games=DEFAULT_MIN_TRAIN_GAMES
        )
    scored = scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()
    scored["p"] = scored["home_cover_probability_at_open"]
    return finish(scored[["game_id", "season", "week", "p", "margin_vs_open"]], scored["p"])


def run_arm(features, pop, spec):
    proxy = proxy_eval(features, pop, spec)
    cache = {}
    parts = []
    for s in TRUE_SEASONS:
        columns, alpha = spec(s)
        key = (tuple(columns), alpha)
        if key not in cache:
            cache[key] = true_eval(features, columns, alpha)
        parts.append(cache[key].loc[cache[key]["season"].eq(s)])
    return pd.concat([proxy, *parts])


def boot(df, col, block, rng_seed=SEED):
    keys = df[block].astype(str).agg("-".join, axis=1) if len(block) > 1 else df[block[0]]
    g = df.groupby(keys)[col].agg(["sum", "count"])
    sums = g["sum"].to_numpy()
    sizes = g["count"].to_numpy()
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(sums), (BOOT, len(sums)))
    draws = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return {
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "pp": float((draws > 0).mean()),
        "blocks": len(sums),
    }


def compare(arm, base):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["d_ll"] = j["ll"] - j["ll_b"]
    j["d_brier"] = j["brier"] - j["brier_b"]
    eras = {
        "proxy_2011_2019": j["season"].le(2019),
        "true_2020_2025": j["season"].ge(2020),
        "pooled": j["season"].ge(0),
        "from_2018": j["season"].ge(2018),
    }
    out = {}
    for name, mask in eras.items():
        s = j.loc[mask]
        a = arm.loc[s.index]
        wins = int(a["correct"].sum())
        r = {
            "n": len(s),
            "record": f"{wins}-{len(s) - wins}",
            "flips": int((s["pick_home"] != s["pick_home_b"]).sum()),
            "acc_arm": float(a["correct"].mean()),
            "acc_base": float(s["correct_b"].mean()),
            "d_acc": float(s["d_acc"].mean()),
            "d_ll": float(s["d_ll"].mean()),
            "d_brier": float(s["d_brier"].mean()),
        }
        for blockname, block in (("season", ["season"]), ("week", ["season", "week"])):
            r[f"acc_{blockname}"] = boot(s, "d_acc", block)
            r[f"ll_{blockname}"] = boot(s, "d_ll", block)
            r[f"brier_{blockname}"] = boot(s, "d_brier", block)
        out[name] = r
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "grade"
    features = pd.read_parquet(u3.FEATURES)
    print("feature sha", hashlib.sha256(u3.FEATURES.read_bytes()).hexdigest(), flush=True)
    frame = regular_season_rows(features)
    frame = frame.loc[frame["ats_margin"].notna() & frame["result"].notna()].copy()
    if stage == "select":
        sel = {"greedy": {}, "alpha_only": {}}
        for s in PROXY_SEASONS:
            sel["greedy"][str(s)] = select_greedy(frame, s)
            sel["alpha_only"][str(s)] = select_alpha_only(frame, s)
            print(s, sel["greedy"][str(s)], sel["alpha_only"][str(s)], flush=True)
            SEL_PROXY.write_text(json.dumps(sel, indent=2))
        return
    sel_true = json.loads(SEL_TRUE.read_text())
    sel_proxy = json.loads(SEL_PROXY.read_text())
    pop = proxy_population(features)
    print("proxy population", len(pop), flush=True)

    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(u3.SNAPSHOT / "team_stats.parquet")
    terms3, _ = u3.adjusted_columns(features, schedules, team_stats)
    terms5 = pd.read_parquet(TERMS5)
    ext = features.merge(terms3, on="game_id", how="left", validate="one_to_one")
    ext = ext.merge(terms5, on="game_id", how="left", validate="one_to_one")
    net = u3.arm_columns()["compact_net"]
    sides = ("home", "away", "diff")
    mz = [f"{s}_cov_mz" for s in sides]

    def sel_spec(name):
        def spec(season):
            src = sel_true if season >= 2020 else sel_proxy
            c = src[name][str(season)]
            return cols_for(frozenset(c["dropped"])), c["alpha"]

        return spec

    arms = {
        "base": lambda s: (FULL, BASE_ALPHA),
        "u1_nested_families_alpha": sel_spec("greedy"),
        "u1_alpha_only": sel_spec("alpha_only"),
        "u3_compact_net": lambda s: (tuple(net), BASE_ALPHA),
        "u5a_man_zone": lambda s: (tuple(FULL) + tuple(mz), BASE_ALPHA),
    }
    results = {}
    base = None
    for name, spec in arms.items():
        arm = run_arm(ext, pop, spec)
        arm.to_parquet(OUT / f"{name}_per_game.parquet")
        if name == "base":
            base = arm
            tr = arm.loc[arm["season"].ge(2020)]
            print("base true-era reproduction", int(tr["correct"].sum()), len(tr), flush=True)
            px = arm.loc[arm["season"].le(2019)]
            print("base proxy", int(px["correct"].sum()), len(px), flush=True)
            continue
        results[name] = compare(arm, base)
        print(name, results[name]["pooled"]["record"], results[name]["pooled"]["d_acc"], flush=True)
        (OUT / "report.json").write_text(json.dumps(results, indent=2))


main()
