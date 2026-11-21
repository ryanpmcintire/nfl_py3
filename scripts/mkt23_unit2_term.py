import json
import subprocess
import sys

import mkt23_move_decomposition as D
import numpy as np
import pandas as pd
import sky04_lead_lag as L

OUT = D.OUT
FAMILY = "mkt23_unit2_term"
SEASONS = (2023, 2024, 2025)
ARMS = {
    "a_intercept": [],
    "b2_base_refit": ["logit"],
    "c_direction": ["logit", "direction"],
    "d_one_team_direction": ["logit", "one_team_direction"],
    "e_both": ["logit", "direction", "one_team_direction"],
}
VS_B = ("c_direction", "d_one_team_direction", "e_both", "b2_base_refit")
LOOKS = len(VS_B) * 2 + 1
NEWTON_STEPS = 50
NEWTON_TOL = 1e-10


def design(x, n):
    return np.column_stack([np.ones(n), x]) if x.size else np.ones((n, 1))


def fit(x, y):
    x = design(x, len(y))
    w = np.zeros(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (y - p)
        hess = (x * (p * (1 - p))[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return w


def predict(w, x):
    return 1 / (1 + np.exp(-design(x, len(x)) @ w))


def loss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def population():
    g = pd.read_parquet(L.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS) & (g.margin_vs_open != 0)]
    g = g.dropna(subset=["out_of_season_home_probability", "home_covered"])
    mv = pd.read_parquet(OUT / "move_table.parquet")[["game_id", "d_hs", "cat"]]
    g = g.merge(mv, on="game_id", how="left").reset_index(drop=True)
    g["d_hs"] = g.d_hs.fillna(0.0)
    p = g.out_of_season_home_probability.astype(float)
    g["logit"] = np.log(p / (1 - p))
    g["direction"] = np.sign(g.d_hs)
    g["one_team_direction"] = g.direction * (g.cat == "one_team")
    return g


def interval(d, rng):
    m = d[rng.integers(0, len(d), size=(L.BOOT, len(d)))].mean(axis=1)
    return {
        "effect": float(d.mean()),
        "interval": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
        "standard_error": float(m.std(ddof=1)),
        "probability_positive": float((m > 0).mean()),
    }


def run():
    g = population()
    y = g.home_covered.values.astype(float)
    base_p = g.out_of_season_home_probability.values.astype(float)
    season = g.season.values
    preds = {"b_served_base": base_p}
    coefs = {}
    ins = {}
    for arm, cols in ARMS.items():
        x = g[cols].values.astype(float) if cols else np.empty((len(g), 0))
        oos = np.zeros(len(g))
        coefs[arm] = {}
        for s in SEASONS:
            te = season == s
            w = fit(x[~te], y[~te])
            oos[te] = predict(w, x[te])
            coefs[arm][str(s)] = [float(v) for v in w]
        preds[arm] = oos
        wall = fit(x, y)
        ins[arm] = float(loss(predict(wall, x), y).mean())
        coefs[arm]["all"] = [float(v) for v in wall]
    rng = np.random.default_rng(L.SEED)
    res = {
        "family": FAMILY,
        "n": len(g),
        "n_moved_direction_nonzero": int((g.direction != 0).sum()),
        "n_one_team_direction_nonzero": int((g.one_team_direction != 0).sum()),
        "look_count": LOOKS,
        "arms": {},
        "coefficients": coefs,
    }
    base_side = base_p > 0.5
    for arm, p in preds.items():
        ll = loss(p, y)
        entry = {
            "log_loss": float(ll.mean()),
            "brier": float(((p - y) ** 2).mean()),
            "accuracy": float(((p > 0.5) == (y == 1)).mean()),
            "in_sample_log_loss": ins.get(arm),
        }
        if arm in ins:
            entry["oos_minus_in_sample_log_loss"] = entry["log_loss"] - ins[arm]
        if arm in VS_B:
            for ref in ("b_served_base", "b2_base_refit"):
                if ref == arm:
                    continue
                entry[f"log_loss_improvement_vs_{ref}"] = interval(loss(preds[ref], y) - ll, rng)
                entry[f"brier_improvement_vs_{ref}"] = interval(
                    ((preds[ref] - y) ** 2) - ((p - y) ** 2), rng
                )
            flip = (p > 0.5) != base_side
            entry["decisive_vs_served_base"] = {
                "flips": int(flip.sum()),
                "arm_right": int(((p > 0.5) == (y == 1))[flip].sum()),
                "base_right": int((base_side == (y == 1))[flip].sum()),
            }
            entry["by_season_log_loss_improvement_vs_b2"] = {
                str(s): float((loss(preds["b2_base_refit"], y) - ll)[season == s].mean())
                for s in SEASONS
            }
        res["arms"][arm] = entry
    (OUT / "unit2_term.json").write_text(json.dumps(res, indent=1))
    cells = []
    for arm in ("c_direction", "d_one_team_direction", "e_both"):
        e = res["arms"][arm]["log_loss_improvement_vs_b2_base_refit"]
        cells.append(
            {
                "name": f"mkt23_unit2_{arm}_vs_refit_base",
                "effect": e["effect"],
                "effect_units": "log_loss_improvement",
                "interval_low": e["interval"][0],
                "interval_high": e["interval"][1],
                "standard_error": e["standard_error"],
                "probability_positive": e["probability_positive"],
                "classification_evidence": (
                    f"LOSO 2023-25, n {len(g)}, {LOOKS} looks; no power control or "
                    "mechanism refutation; deadline-view term, unknown at the Tuesday pick"
                ),
            }
        )
    batch = {
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2023,
        "season_end": 2025,
        "family": FAMILY,
        "category": "market",
        "source": "artifacts/mkt23/unit2_term.json",
        "sample_games": len(g),
        "plain_summary": (
            "Whether the direction of the Tuesday-to-deadline spread move, alone or for "
            "one-team total moves, improves the out-of-season cover probability at the opener."
        ),
        "description": (
            f"LOSO logistic terms on base logit; log loss gain vs refit base; {LOOKS} looks"
        ),
        "cells": cells,
    }
    (OUT / "unit2_record_batch.json").write_text(json.dumps(batch, indent=1))
    print(json.dumps(res, indent=1))
    if "--record" in sys.argv:
        subprocess.run(
            [
                ".tools/uv.exe",
                "run",
                "--no-sync",
                "--no-cache",
                "nfl-ats",
                "weak-signals",
                "record",
                "--batch",
                str(OUT / "unit2_record_batch.json"),
            ],
            check=True,
            cwd=L.ROOT,
        )


ART = L.ROOT / "artifacts/pick_probability/20261120T171725Z"
LIVE = 2026
TERM_ARMS = ("c_direction", "d_one_team_direction", "e_both")
POWER_ARMS = {"c_direction": "direction", "d_one_team_direction": "one_team_direction"}
NULL_DRAWS = 200
POWER_DRAWS = 100
LADDER_RUNGS = 10
NULL_QUANTILE = 95.0
POWER_TARGET = 0.80
U3_FAMILY = FAMILY
U3_LOOKS = (len(TERM_ARMS) + 1) * 2 * 2
SCHEMES = {
    "history": "leave-one-season-out",
    "live_2026": "fit on 2023-25, scored on the played card",
}


def wfit(x, y, w):
    x = design(x, len(y))
    b = np.zeros(x.shape[1])
    hess = np.eye(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = 1 / (1 + np.exp(-x @ b))
        hess = (x * (w * p * (1 - p))[:, None]).T @ x
        step = np.linalg.solve(hess, x.T @ (w * (y - p)))
        b = b + step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return b, hess


def terms(df, path):
    mv = pd.read_parquet(OUT / path)[["game_id", "d_hs", "cat"]]
    df = df.merge(mv, on="game_id", how="left")
    df["direction"] = np.sign(df.d_hs.fillna(0.0)) * df.cat.notna()
    df["one_team_direction"] = df.direction * (df.cat == "one_team")
    df["logit"] = np.log(df.p / (1 - df.p))
    df["code"] = pd.factorize(df.game_id)[0]
    return df.reset_index(drop=True)


def history_pool():
    pg = pd.read_parquet(ART / "per_game.parquet")
    hk = pd.read_parquet(ART / "hook_rows.parquet")
    pg = pg[(pg.game_type == "REG") & pg.season.isin(SEASONS)]
    hk = hk[hk.season.isin(SEASONS)]
    half = pg[~pg.game_id.isin(hk.game_id)]
    a = pd.DataFrame(
        {
            "game_id": half.game_id,
            "season": half.season,
            "p": half.out_of_season_home_probability.astype(float),
            "y": half.home_covered.astype(float),
            "w": 1.0,
            "pool_line": half.tue_open_home_spread,
        }
    )
    b = pd.DataFrame(
        {
            "game_id": hk.game_id,
            "season": hk.season,
            "p": hk.two_stage_out_of_season_home_probability.astype(float),
            "y": hk.home_covered.astype(float),
            "w": 0.5,
            "pool_line": hk.pool_line,
        }
    )
    d = pd.concat([a, b], ignore_index=True)
    d["week"] = d.game_id.str[5:7].astype(int)
    return terms(d, "move_table_history_all.parquet")


def live_pool():
    pk = pd.read_parquet(L.ROOT / "artifacts/clv_ledger/published_picks.parquet")
    pk = pk[(pk.season == LIVE) & (pk.published_at_utc <= pk.pick_deadline_utc)]
    pk = pk.sort_values("published_at_utc").drop_duplicates("game_id", keep="last")
    pk["p"] = np.where(pk.pick_team == pk.home_team, pk.displayed_score, 1 - pk.displayed_score)
    gf = pd.read_parquet(L.ROOT / "data/processed/game_features.parquet")
    gf = gf[(gf.season == LIVE) & (gf.game_type == "REG") & gf.home_cover.notna()]
    d = gf[["game_id", "season", "week", "spread_line", "home_cover"]].merge(
        pk[["game_id", "p", "market_spread"]], on="game_id"
    )
    d = d.rename(columns={"home_cover": "y", "spread_line": "pool_line"}).assign(w=1.0)
    d["y"] = d.y.astype(float)
    return terms(d, "move_table_2026.parquet")


def oos(h, lv, cols, yh, yl):
    xh = h[cols].values.astype(float) if cols else np.empty((len(h), 0))
    xl = lv[cols].values.astype(float) if cols else np.empty((len(lv), 0))
    wh = h.w.values
    ph = np.zeros(len(h))
    folds = {}
    for s in SEASONS:
        te = h.season.values == s
        b, _ = wfit(xh[~te], yh[~te], wh[~te])
        ph[te] = predict(b, xh[te])
        folds[str(s)] = [float(v) for v in b]
    b, hess = wfit(xh, yh, wh)
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    return ph, predict(b, xl), folds, b, se, float((wh * loss(predict(b, xh), yh)).sum() / wh.sum())


def per_game(df, v):
    return np.bincount(df.code.values, weights=v * df.w.values)


def acc(p, y):
    return ((p > 0.5) == (y == 1)).astype(float)


def gain(df, ref, arm, y):
    return per_game(df, loss(ref, y) - loss(arm, y)).mean()


def shuffled(df, rng):
    gi = df.drop_duplicates("game_id").reset_index(drop=True)
    idx = np.arange(len(gi))
    for ix in gi.groupby(["season", "week"]).indices.values():
        idx[ix] = rng.permutation(ix)
    src = gi.iloc[idx].reset_index(drop=True)
    out = df.copy()
    for c in ("direction", "one_team_direction"):
        out[c] = df.game_id.map(dict(zip(gi.game_id, src[c])))  # noqa: B905
    return out


def summarize(df, preds, y, rng):
    res = {}
    base_side = preds["b_served_base"] > 0.5
    for arm, p in preds.items():
        e = {
            "log_loss": float(per_game(df, loss(p, y)).mean()),
            "brier": float(per_game(df, (p - y) ** 2).mean()),
            "accuracy": float(per_game(df, acc(p, y)).mean()),
        }
        if arm in TERM_ARMS or arm == "b2_base_refit":
            for ref in ("b_served_base", "b2_base_refit"):
                if ref == arm:
                    continue
                e[f"log_loss_improvement_vs_{ref}"] = interval(
                    per_game(df, loss(preds[ref], y) - loss(p, y)), rng
                )
                e[f"accuracy_points_vs_{ref}"] = interval(
                    per_game(df, (acc(p, y) - acc(preds[ref], y)) * 100), rng
                )
            flip = (p > 0.5) != base_side
            e["decisive_vs_served_base"] = {
                "flip_games": float(df.w.values[flip].sum()),
                "arm_right": float(df.w.values[flip & (acc(p, y) == 1)].sum()),
                "base_right": float(
                    df.w.values[flip & (acc(preds["b_served_base"], y) == 1)].sum()
                ),
            }
        res[arm] = e
    return res


def evaluate(h, lv, yh, yl, arms):
    bh, bl, *_ = oos(h, lv, ["logit"], yh, yl)
    out = {}
    for arm in arms:
        ph, pl, *_ = oos(h, lv, ARMS[arm], yh, yl)
        out[arm] = (gain(h, bh, ph, yh), gain(lv, bl, pl, yl))
    return out, bh, bl


def pool_line():
    h, lv = history_pool(), live_pool()
    yh, yl = h.y.values, lv.y.values
    rng = np.random.default_rng(L.SEED)
    hp = {"b_served_base": h.p.values}
    lp = {"b_served_base": lv.p.values}
    fits = {}
    for arm, cols in ARMS.items():
        ph, pl, folds, b, se, ins = oos(h, lv, cols, yh, yl)
        hp[arm], lp[arm] = ph, pl
        fits[arm] = {"folds": folds, "all": [float(v) for v in b], "se_all": [float(v) for v in se]}
        fits[arm]["history_oos_minus_in_sample_log_loss"] = float(
            per_game(h, loss(ph, yh)).mean() - ins
        )
    pair = h[h.w < 1].groupby("game_id").y.agg(["min", "max"])
    hook_games = h[h.w < 1].drop_duplicates("game_id")
    res = {
        "family": U3_FAMILY,
        "served_artifact": str(ART.relative_to(L.ROOT)),
        "look_count": U3_LOOKS,
        "history": {
            "games": int(h.game_id.nunique()),
            "rows": len(h),
            "whole_number_games_split_at_pool_lines": len(pair),
            "whole_number_pushes": int((pair["min"] != pair["max"]).sum()),
            "games_with_direction": int((h.drop_duplicates("game_id").direction != 0).sum()),
            "games_with_one_team_direction": int(
                (h.drop_duplicates("game_id").one_team_direction != 0).sum()
            ),
            "arms": summarize(h, hp, yh, rng),
            "by_season_log_loss_improvement_vs_refit": {
                arm: {
                    str(s): float(
                        per_game(
                            h,
                            (loss(hp["b2_base_refit"], yh) - loss(hp[arm], yh))
                            * (h.season.values == s),
                        ).sum()
                        / h[h.season == s].game_id.nunique()
                    )
                    for s in SEASONS
                }
                for arm in TERM_ARMS
            },
        },
        "live_2026": {
            "games": int(lv.game_id.nunique()),
            "weeks": sorted(int(w) for w in lv.week.unique()),
            "pool_line_equals_card_line": int((lv.pool_line == lv.market_spread).sum()),
            "games_with_direction": int((lv.direction != 0).sum()),
            "games_with_one_team_direction": int((lv.one_team_direction != 0).sum()),
            "arms": summarize(lv, lp, yl, rng),
        },
        "fits": fits,
        "hook_game_count": len(hook_games),
    }
    obs, _, _ = evaluate(h, lv, yh, yl, TERM_ARMS)
    null = {arm: [[], []] for arm in TERM_ARMS}
    for _ in range(NULL_DRAWS):
        g, _, _ = evaluate(shuffled(h, rng), shuffled(lv, rng), yh, yl, TERM_ARMS)
        for arm in TERM_ARMS:
            null[arm][0].append(g[arm][0])
            null[arm][1].append(g[arm][1])
    res["matched_null"] = {}
    cut = {}
    for arm in TERM_ARMS:
        cut[arm] = []
        ent = {}
        for i, pop in enumerate(("history", "live_2026")):
            v = np.array(null[arm][i])
            cut[arm].append(float(np.percentile(v, NULL_QUANTILE)))
            ent[pop] = {
                "observed": float(obs[arm][i]),
                "null_percentile_of_observed": float((v < obs[arm][i]).mean() * 100),
                "null_median": float(np.median(v)),
                "null_95th": cut[arm][i],
            }
        res["matched_null"][arm] = ent
    eta_h = h.logit.values * fits["b2_base_refit"]["all"][1] + fits["b2_base_refit"]["all"][0]
    eta_l = lv.logit.values * fits["b2_base_refit"]["all"][1] + fits["b2_base_refit"]["all"][0]
    res["power"] = {}
    for arm, col in POWER_ARMS.items():
        se = fits[arm]["se_all"][2]
        rungs = []
        for k in range(1, LADDER_RUNGS + 1):
            beta = k * se
            hit = [[], []]
            pts = [[], []]
            for _ in range(POWER_DRAWS):
                uh = rng.random(h.code.max() + 1)[h.code.values]
                ul = rng.random(lv.code.max() + 1)[lv.code.values]
                sh = (uh < 1 / (1 + np.exp(-(eta_h + beta * h[col].values)))).astype(float)
                sl = (ul < 1 / (1 + np.exp(-(eta_l + beta * lv[col].values)))).astype(float)
                bh, bl, *_ = oos(h, lv, ["logit"], sh, sl)
                ph, pl, *_ = oos(h, lv, ARMS[arm], sh, sl)
                for i, (df, ref, p, y) in enumerate(((h, bh, ph, sh), (lv, bl, pl, sl))):
                    hit[i].append(gain(df, ref, p, y) > cut[arm][i])
                    pts[i].append(per_game(df, (acc(p, y) - acc(ref, y)) * 100).mean())
            rungs.append(
                {
                    "beta_logit": float(beta),
                    "power_history": float(np.mean(hit[0])),
                    "power_live_2026": float(np.mean(hit[1])),
                    "mean_accuracy_points_history": float(np.mean(pts[0])),
                    "mean_accuracy_points_live_2026": float(np.mean(pts[1])),
                }
            )
        det = {}
        for pop in ("history", "live_2026"):
            ok = [r for r in rungs if r[f"power_{pop}"] >= POWER_TARGET]
            det[pop] = ok[0] if ok else None
        res["power"][arm] = {
            "rungs": rungs,
            "detectable": det,
            "observed_coef": fits[arm]["all"][2],
        }
    (OUT / "unit3_pool_line.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def write_batches():
    r = json.loads((OUT / "unit3_pool_line.json").read_text())
    spans = {"history": (SEASONS[0], SEASONS[-1]), "live_2026": (LIVE, LIVE)}
    for pop, (s0, s1) in spans.items():
        cells = []
        for arm in TERM_ARMS:
            e = r[pop]["arms"][arm]["log_loss_improvement_vs_b2_base_refit"]
            a = r[pop]["arms"][arm]["accuracy_points_vs_b2_base_refit"]
            dv = r[pop]["arms"][arm]["decisive_vs_served_base"]
            nl = r["matched_null"][arm][pop]
            det = r["power"].get(arm, {}).get("detectable", {}).get(pop)
            size = (
                f"detectable beta {det['beta_logit']:.3f} logit"
                if det
                else "no ladder rung reached power"
            )
            if arm not in r["power"]:
                size = "two-term arm, no ladder"
            cells.append(
                {
                    "name": f"mkt23_u3_pool_line_{pop}_{arm}_vs_refit_base",
                    "effect": e["effect"],
                    "effect_units": "log_loss_improvement",
                    "interval_low": e["interval"][0],
                    "interval_high": e["interval"][1],
                    "standard_error": e["standard_error"],
                    "probability_positive": e["probability_positive"],
                    "classification_evidence": (
                        f"pool line, {r[pop]['games']} games, matched-null percentile "
                        f"{nl['null_percentile_of_observed']:.1f}, {size}, accuracy "
                        f"{a['effect']:+.2f} pts, decisive "
                        f"{dv['arm_right']:g}-{dv['base_right']:g}; "
                        f"{U3_LOOKS} looks; no power control or mechanism refutation"
                    ),
                }
            )
        batch = {
            "classification": "unresolved_below_power",
            "league": "nfl",
            "season_start": s0,
            "season_end": s1,
            "family": U3_FAMILY,
            "category": "market",
            "source": "artifacts/mkt23/unit3_pool_line.json",
            "sample_games": r[pop]["games"],
            "plain_summary": (
                "Whether the direction of the Tuesday-to-deadline spread move, alone or for "
                "one-team total moves, improves the served cover probability at the pool line."
            ),
            "description": (
                f"Unit 2 terms re-tested at the pool line on the served probability "
                f"({SCHEMES[pop]}); {U3_LOOKS} looks"
            ),
            "cells": cells,
        }
        (OUT / f"unit3_record_batch_{pop}.json").write_text(json.dumps(batch, indent=1))
        print(json.dumps(batch, indent=1))


if __name__ == "__main__":
    if "--batches" in sys.argv:
        sys.exit(write_batches())
    sys.exit(pool_line() if "--pool-line" in sys.argv else run())
