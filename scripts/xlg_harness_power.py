import json
import os
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402
import mod24_pooled_levers as lev  # noqa: E402
import pooled_screen as ps  # noqa: E402
import xlg10_addone as x10  # noqa: E402
import xlg14_family_screen as x14  # noqa: E402

MOD36 = ROOT / "scripts" / "mod36_addone_noise_floor.py"
MOD36_SUMMARY = ROOT / "artifacts" / "mod36" / "summary.json"
XLG13 = ROOT / "artifacts" / "xlg13_screen_transfer" / "report_fixed.json"
XLG10_CHECK = ROOT / "artifacts" / "pooled_screen" / "xlg10_short_week_diff.json"
OUT = ROOT / "artifacts" / "xlg_harness_power"
LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "xlg_harness_power"
STREAM_ADDONE = 1
STREAM_SHARED = 2
LEAGUES = ("nfl", "cfb")
POOL_KEYS = ("accuracy", "log_loss", "brier")
ARMS = ("nfl", "cfb", "pooled")
TEMPLATES = [
    {
        "name": "short_week_side",
        "column": "short_week",
        "shared_column": "short_week_diff",
        "intercept": False,
        "what": "which games had one team on a short week and the other not",
    },
    {
        "name": "rest_difference",
        "column": "rest_diff",
        "shared_column": "rest_diff",
        "intercept": False,
        "what": "the gap in rest days between the two teams",
    },
]


def load_mod36():
    text = MOD36.read_text(encoding="utf-8")
    body, call, rest = text.rpartition("\nmain()")
    if not call or rest.strip():
        raise SystemExit("mod36 script no longer ends with a bare main() call")
    mod = types.ModuleType("mod36_addone_noise_floor")
    mod.__file__ = str(MOD36)
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        exec(compile(body, str(MOD36), "exec"), mod.__dict__)
    finally:
        sys.argv = argv
    return mod


M36 = load_mod36()
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else M36.DECLARED_DRAWS
OUT = OUT if DRAWS == M36.DECLARED_DRAWS else OUT / "smoke"
SEED = M36.SEED
LEVEL = f"{M36.NOTABLE[0]:.2f}"
LOW = f"{1 - M36.NOTABLE[0]:.2f}"


def mod36_reference():
    s = json.loads(MOD36_SUMMARY.read_text(encoding="utf-8"))
    out = {"reported": sorted(s["reported_sizes_logit_per_sd"].values()), "templates": {}}
    for tpl in TEMPLATES:
        b = s["templates"][tpl["name"]]
        ref = b["matched_reference"]
        null = b["null"][ref]
        out["templates"][tpl["name"]] = {
            "games": b["shape"]["n"],
            "null_coefficient_sd": b["null_coefficient_sd"],
            "null": {k: null[k] for k in (*M36.SHORT, "changed", "arm_right", "ref_right")},
            "record": null["record"],
            "detectable": {k: b["detectable"][ref][f"detect_{k}_{LEVEL}"] for k in ("acc", "ll")},
        }
    return out


def transfer():
    r = json.loads(XLG13.read_text(encoding="utf-8"))
    screen = np.array(r["screen"], dtype=float)
    served = np.array(r["served"], dtype=float)
    slope, intercept = np.polyfit(screen, served, 1)
    return {
        "pairs": len(screen),
        "screen": [float(v) for v in screen],
        "served": [float(v) for v in served],
        "slope_served_per_screen_point": float(slope),
        "intercept": float(intercept),
        "pearson": r["pearson"],
        "spearman": r["spearman"],
        "sign_agreement": r["sign_agreement"],
    }


def grid_for(ref, name, cross, ladders):
    found = ref["templates"][name]["detectable"].values()
    det = [d["beta_per_sd"] for d in found if d["reached"]]
    steps = [k * se for se in ladders for k in range(1, M36.LADDER_RUNGS + 1)]
    return sorted({0.0, *ref["reported"], *cross, *det, *steps})


def plant_tables(st, beta):
    tabs = []
    for k in st["groups"]:
        v = st["xs"][k]
        w = expit(st["sign"][k][:, None] * (st["z"][k][:, None] + beta * v[None, :]))
        tabs.append(np.cumsum(w, axis=1))
    return tabs


def planted_from(st, tabs, rng):
    out = np.empty(st["n"])
    for k, c in zip(st["groups"], tabs, strict=True):
        u = rng.random(len(k)) * c[:, -1]
        j = np.minimum((c < u[:, None]).sum(axis=1), len(k) - 1)
        out[k] = st["xs"][k][j]
    return out


def addone_frame():
    d = x14.cfb_frame()
    out = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            M36.SERVED_COLUMN: expit(d["logit_p"].to_numpy(float)),
        }
    )
    for tpl in TEMPLATES:
        out[tpl["column"]] = d[tpl["column"]].fillna(0.0).to_numpy(float)
    return out


def run_addone(frame, ref, started):
    rows, states, reals = [], [], []
    for t, tpl in enumerate(TEMPLATES):
        st = M36.setup(frame, tpl, t)
        states.append(st)
        reals.append(M36.real_read(st, tpl))
        rng = np.random.default_rng([SEED, STREAM_ADDONE, t, 0])
        for i in range(DRAWS):
            M36.draw_rows(st, tpl, "null", 0.0, i, M36.permuted(st, rng), rows)
        print("addone null", tpl["name"], st["shape"], f"{time.time() - started:.0f}s", flush=True)
    null = pd.DataFrame(rows)
    cross = []
    for real in reals:
        cross.append(abs(real["full_coefficient_per_sd"]))
        cross.append(max(abs(v) for v in real["fold_coefficients_per_sd"]))
    prow, grids = [], {}
    for t, (tpl, st) in enumerate(zip(TEMPLATES, states, strict=True)):
        se = float(null[null.template == tpl["name"]].coef.std(ddof=1))
        grid = grid_for(ref, tpl["name"], cross, [se])
        grids[tpl["name"]] = grid
        for c, beta in enumerate(grid):
            rng = np.random.default_rng([SEED, STREAM_ADDONE, t, 1 + c])
            tabs = plant_tables(st, beta)
            for i in range(DRAWS):
                x = planted_from(st, tabs, rng)
                M36.draw_rows(st, tpl, "planted", beta, i, x, prow)
        print("addone grid", tpl["name"], len(grid), f"{time.time() - started:.0f}s", flush=True)
    return null, pd.DataFrame(prow), states, reals, cross, grids


def nfl_mask(f):
    lo, hi = ps.NFL_OPENER_SEASONS
    return (
        f["game_type"].eq("REG")
        & f["season"].between(lo, hi)
        & f["open"].notna()
        & f["result"].notna()
    )


def cfb_mask(f):
    lo, hi = ps.CFB_SEASONS
    return (
        f["season"].between(lo, hi)
        & f["season"].ne(ps.CFB_EXCLUDED_SEASON)
        & f["open"].notna()
        & f["result"].notna()
        & f["source_regime"].eq(lev.OPENER_REGIME)
    )


def league_state(frame, mask, min_train, cols, index):
    frame = base.to_opener(frame)
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    keep = frame["ats_margin"].notna() & frame["game_type"].eq("REG")
    t = frame.loc[keep].reset_index(drop=True)
    x = t[cols].to_numpy(float)
    y = t["ats_margin"].to_numpy(float)
    p = len(cols)
    picks = []
    for _key, wk in t.loc[mask(t)].groupby(["season", "week"], sort=True):
        k = int(t["gameday"].lt(wk["gameday"].min()).sum())
        if k >= min_train:
            picks.append((k, wk.index.to_numpy()))
    w = len(picks)
    med, mu, sd, ay = (np.empty((w, p)) for _ in range(4))
    ybar, syy = np.empty(w), np.empty(w)
    gram = np.zeros((w, p + 1, p + 1))
    b, ti, tw = [], [], []
    for j, (k, rows) in enumerate(picks):
        xtr = x[:k]
        md = np.nanmedian(xtr, axis=0)
        xi = np.where(np.isnan(xtr), md, xtr)
        m = xi.mean(axis=0)
        s = xi.std(axis=0)
        s = np.where(s > 0, s, 1.0)
        a = (xi - m) / s
        yc = y[:k] - y[:k].mean()
        med[j], mu[j], sd[j] = md, m, s
        ybar[j], syy[j], ay[j] = y[:k].mean(), yc @ yc, a.T @ yc
        gram[j, :p, :p] = a.T @ a
        xte = x[rows]
        b.append((np.where(np.isnan(xte), md, xte) - m) / s)
        ti.append(rows)
        tw.append(np.full(len(rows), j))
    kk = np.array([k for k, _ in picks], dtype=float)
    ti, tw, b = np.concatenate(ti), np.concatenate(tw), np.vstack(b)
    ridge = lev.ALPHA * np.eye(p + 1)
    c0 = np.linalg.solve(gram[:, :p, :p] + ridge[:p, :p], ay[..., None])[..., 0]
    rss = syy - 2 * (c0 * ay).sum(axis=1) + np.einsum("wi,wij,wj->w", c0, gram[:, :p, :p], c0)
    sig0 = np.sqrt(rss / (kk - 1))[tw]
    pred0 = ybar[tw] + (b * c0[tw]).sum(axis=1)
    cm = y[ti]
    graded = cm != 0
    side = np.sign(cm[graded])
    y01 = (cm[graded] > 0).astype(float)
    p0 = np.clip(ndtr(pred0 / sig0), M36.EPS, 1 - M36.EPS)
    correct0 = (np.sign(pred0[graded]) == side).astype(float)
    pg = p0[graded]
    label = t["season"].astype(str) + "-" + t["week"].astype(str)
    names, block = np.unique(label.to_numpy()[ti][graded], return_inverse=True)
    nb = len(names)
    cnt = np.bincount(block, minlength=nb).astype(float)
    rng = np.random.default_rng(base.SEED)
    mult, den = [], []
    for _ in POOL_KEYS:
        idx = rng.integers(0, nb, size=(base.DRAWS, nb))
        mm = np.zeros((base.DRAWS, nb))
        np.add.at(mm, (np.arange(base.DRAWS)[:, None], idx), 1.0)
        mult.append(mm)
        den.append(mm @ cnt)
    season = t["season"].to_numpy()
    zfull = np.zeros(len(t))
    zfull[ti] = logit(p0)
    plant = {}
    for tpl in TEMPLATES:
        raw = t[tpl["shared_column"]].fillna(0.0).to_numpy(float)
        plant[tpl["name"]] = {
            "n": len(t),
            "xs": raw / float(raw.std(ddof=1)),
            "groups": [np.flatnonzero(season == s) for s in np.unique(season)],
            "sign": np.sign(y),
            "z": zfull,
            "nonzero_share_graded": float((raw[ti][graded] != 0).mean()),
            "distinct_values": len(np.unique(raw)),
        }
    return {
        "p": p,
        "y": y,
        "x0": np.nan_to_num(x),
        "nm": np.isnan(x).astype(float),
        "k": kk,
        "last": kk.astype(int) - 1,
        "med": med,
        "mu": mu,
        "sd": sd,
        "ybar": ybar,
        "syy": syy,
        "ay": ay,
        "gram": gram,
        "ridge": ridge,
        "b": b,
        "ti": ti,
        "tw": tw,
        "keep": graded,
        "gi": ti[graded],
        "side": side,
        "y01": y01,
        "correct0": correct0,
        "ll0": -(y01 * np.log(pg) + (1 - y01) * np.log(1 - pg)),
        "br0": (pg - y01) ** 2,
        "zg": logit(pg),
        "m0": M36.metric_matrix(pg, y01),
        "block": block,
        "nb": nb,
        "games": int(graded.sum()),
        "mult": mult,
        "den": den,
        "plant": plant,
        "train_rows": len(t),
        "weeks": w,
        "graded_seasons": sorted(int(s) for s in np.unique(season[ti][graded])),
        "base_wins": int(correct0.sum()),
        "base_log_loss": float(-(y01 * np.log(pg) + (1 - y01) * np.log(1 - pg)).mean()),
        "index": index,
    }


def shared_state():
    cols = ps.shared_columns()
    nfl = x10.schedule_features(pd.read_parquet(base.NFL_FEATURES))
    nfl["gameday"] = pd.to_datetime(nfl["gameday"])
    nfl = base.nfl_open(nfl)
    cfb = x10.schedule_features(pd.read_parquet(base.CFB_FEATURES))
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cfb["open"] = cfb["spread_open"]
    cfb["game_type"] = "REG"
    return {
        "cols": cols,
        "nfl": league_state(nfl, nfl_mask, base.NFL_MIN_TRAIN, cols, 0),
        "cfb": league_state(cfb, cfb_mask, base.CFB_MIN_TRAIN, cols, 1),
    }


def ridge_draw(L, x):
    k, last, p, tw = L["k"], L["last"], L["p"], L["tw"]
    mx = np.cumsum(x)[last] / k
    vx = np.maximum(np.cumsum(x * x)[last] / k - mx**2, 0.0)
    sx = np.sqrt(vx)
    sx = np.where(sx > 0, sx, 1.0)
    cross = np.cumsum(x[:, None] * L["x0"], axis=0)[last]
    cross = cross + L["med"] * np.cumsum(x[:, None] * L["nm"], axis=0)[last]
    g = (cross - (k * mx)[:, None] * L["mu"]) / (sx[:, None] * L["sd"])
    m = L["gram"].copy()
    m[:, :p, p] = g
    m[:, p, :p] = g
    m[:, p, p] = k * vx / sx**2
    xy = (np.cumsum(x * L["y"])[last] - k * mx * L["ybar"]) / sx
    rhs = np.concatenate([L["ay"], xy[:, None]], axis=1)
    c = np.linalg.solve(m + L["ridge"], rhs[..., None])[..., 0]
    rss = L["syy"] - 2 * (c * rhs).sum(axis=1) + np.einsum("wi,wij,wj->w", c, m, c)
    sigma = np.sqrt(rss / (k - 1))
    xt = (x[L["ti"]] - mx[tw]) / sx[tw]
    pred = L["ybar"][tw] + (L["b"] * c[tw, :p]).sum(axis=1) + xt * c[tw, p]
    return pred, sigma[tw]


def contrast(L, pred, sigma):
    g = L["keep"]
    correct = (np.sign(pred[g]) == L["side"]).astype(float)
    pr = np.clip(ndtr(pred[g] / sigma[g]), M36.EPS, 1 - M36.EPS)
    y = L["y01"]
    ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    diffs = (100 * (correct - L["correct0"]), L["ll0"] - ll, L["br0"] - (pr - y) ** 2)
    res = {}
    for i, key in enumerate(POOL_KEYS):
        s = np.bincount(L["block"], weights=diffs[i], minlength=L["nb"])
        est = (L["mult"][i] @ s) / L["den"][i]
        res[key] = {
            "effect": float(s.sum() / L["games"]),
            "se": np.float64(est.std(ddof=1)),
            "p_pos": float((est > 0).mean()),
        }
    flips = correct - L["correct0"]
    return res, int((flips > 0).sum()), int((flips < 0).sum())


def make_row(tpl, arm, kind, beta, draw, coef, x_sd, win, loss, eff, pp, oracle, het, adm):
    out = {
        "template": tpl["name"],
        "reference": arm,
        "kind": kind,
        "beta": beta,
        "draw": draw,
        "coef": coef,
        "x_sd": x_sd,
        "changed": win + loss,
        "arm_right": win,
        "ref_right": loss,
        "het_z": het,
        "admissible": adm,
    }
    for i, k in enumerate(M36.SHORT):
        out[k] = float(eff[i])
        out[f"pp_{k}"] = float(pp[i])
        out[f"oracle_{k}"] = float(oracle[i])
    return out


def shared_rows(S, tpl, kind, beta, draw, xs, rows):
    per = {}
    for lg in LEAGUES:
        L = S[lg]
        pred, sigma = ridge_draw(L, xs[lg])
        res, win, loss = contrast(L, pred, sigma)
        xg = xs[lg][L["gi"]]
        oracle = np.full(len(M36.SHORT), np.nan)
        if kind == "planted":
            m = M36.metric_matrix(expit(L["zg"] + beta * xg), L["y01"])
            oracle = ((m - L["m0"]) * M36.DIRECTION).mean(axis=0)
        per[lg] = {"res": res, "win": win, "loss": loss, "xg": xg, "oracle": oracle}
        coef = float(M36.fit_offset(xg[:, None], L["y01"], L["zg"])[0])
        eff = [res[key]["effect"] for key in POOL_KEYS]
        pp = [res[key]["p_pos"] for key in POOL_KEYS]
        sd = float(xg.std(ddof=1))
        rows.append(
            make_row(tpl, lg, kind, beta, draw, coef, sd, win, loss, eff, pp, oracle, 0.0, 1.0)
        )
    n, c = per["nfl"], per["cfb"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = lev.pool(n["res"], c["res"])
    gn, gc = S["nfl"]["games"], S["cfb"]["games"]
    eff, pp = [], []
    for key in POOL_KEYS:
        e, q = pooled[key]["effect"], pooled[key]["p_pos"]
        flat = (gn * n["res"][key]["effect"] + gc * c["res"][key]["effect"]) / (gn + gc)
        eff.append(e if np.isfinite(e) else flat)
        pp.append(q if np.isfinite(q) else 0.5)
    xg = np.concatenate([n["xg"], c["xg"]])
    y = np.concatenate([S["nfl"]["y01"], S["cfb"]["y01"]])
    z = np.concatenate([S["nfl"]["zg"], S["cfb"]["zg"]])
    coef = float(M36.fit_offset(xg[:, None], y, z)[0])
    oracle = (gn * n["oracle"] + gc * c["oracle"]) / (gn + gc)
    het = pooled["accuracy"]["het_z"]
    adm = float(pooled["accuracy"]["admissible"])
    win, loss = n["win"] + c["win"], n["loss"] + c["loss"]
    sd = float(xg.std(ddof=1))
    rows.append(
        make_row(tpl, "pooled", kind, beta, draw, coef, sd, win, loss, eff, pp, oracle, het, adm)
    )


def run_shared(S, ref, cross, started):
    null_rows, plant_rows, grids, reals = [], [], {}, {}
    for t, tpl in enumerate(TEMPLATES):
        sts = {lg: S[lg]["plant"][tpl["name"]] for lg in LEAGUES}
        real = []
        shared_rows(S, tpl, "real", 0.0, 0, {lg: sts[lg]["xs"] for lg in LEAGUES}, real)
        reals[tpl["name"]] = real
        rng = np.random.default_rng([SEED, STREAM_SHARED, t, 0])
        for i in range(DRAWS):
            xs = {lg: M36.permuted(sts[lg], rng) for lg in LEAGUES}
            shared_rows(S, tpl, "null", 0.0, i, xs, null_rows)
        print("shared null", tpl["name"], f"{time.time() - started:.0f}s", flush=True)
    null = pd.DataFrame(null_rows)
    for t, tpl in enumerate(TEMPLATES):
        own = null[null.template == tpl["name"]]
        ladders = [float(own[own.reference == a].coef.std(ddof=1)) for a in ("pooled", "nfl")]
        grid = grid_for(ref, tpl["name"], cross, ladders)
        grids[tpl["name"]] = grid
        sts = {lg: S[lg]["plant"][tpl["name"]] for lg in LEAGUES}
        for c, beta in enumerate(grid):
            rng = np.random.default_rng([SEED, STREAM_SHARED, t, 1 + c])
            tabs = {lg: plant_tables(sts[lg], beta) for lg in LEAGUES}
            for i in range(DRAWS):
                xs = {lg: planted_from(sts[lg], tabs[lg], rng) for lg in LEAGUES}
                shared_rows(S, tpl, "planted", beta, i, xs, plant_rows)
        print("shared grid", tpl["name"], len(grid), f"{time.time() - started:.0f}s", flush=True)
    return null, pd.DataFrame(plant_rows), grids, reals


def summarise(null, plant):
    out = {}
    for (name, arm), g in null.groupby(["template", "reference"], sort=False):
        s = M36.null_summary(g)
        for k in M36.SHORT:
            s[k]["share_positive"] = float((g[k] > 0).mean() + 0.5 * (g[k] == 0).mean())
        if "admissible" in g.columns:
            s["admissible_share"] = float(g["admissible"].mean())
        cells = []
        pg = plant[(plant.template == name) & (plant.reference == arm)]
        for _beta, x in pg.groupby("beta", sort=True):
            c = M36.planted_summary(x)
            if "admissible" in x.columns:
                c["admissible_share"] = float(x["admissible"].mean())
            cells.append(c)
        out.setdefault(name, {})[arm] = {
            "null": s,
            "planted": cells,
            "detectable": M36.smallest_detected(cells),
        }
    return out


def report(harness, games, block):
    lines = []
    for name, arms in block.items():
        for arm, b in arms.items():
            s = b["null"]
            a, ll, ch, pa, pl = s["acc"], s["ll"], s["changed"], s["pp_acc"], s["pp_ll"]
            lines.append(
                f"T|{harness}|{arm}|{name}|n={games[arm]}"
                f"|acc {a['mean']:+.3f} [{a['q05']:+.3f},{a['q95']:+.3f}]"
                f"|ll {ll['mean']:+.6f} [{ll['q05']:+.6f},{ll['q95']:+.6f}]"
                f"|changed {ch['q05']:.0f}/{ch['q50']:.0f}/{ch['q95']:.0f}"
                f" record {s['arm_right']['q50']:.0f}-{s['ref_right']['q50']:.0f}"
                f" losing {s['record']['losing_share']:.2f}"
                f"|P+<={LOW} acc {pa[f'share_at_or_below_{LOW}']:.3f}"
                f" ll {pl[f'share_at_or_below_{LOW}']:.3f}"
                f"|P+>={LEVEL} acc {pa[f'share_at_or_above_{LEVEL}']:.3f}"
                f" ll {pl[f'share_at_or_above_{LEVEL}']:.3f}"
                f"|coef sd {s['coef']['sd']:.4f}"
            )
            for k in ("ll", "acc"):
                d = b["detectable"][f"detect_{k}_{LEVEL}"]
                head = f"D|{harness}|{arm}|{name}|{k}"
                if d["reached"]:
                    lines.append(
                        f"{head}|beta {d['beta_per_sd']:.4f}|share {d['share']:.2f}"
                        f"|true acc {d['true_accuracy_points']:+.3f}"
                        f"|true ll {d['true_log_loss_improvement']:+.6f}"
                        f"|monotone {d['holds_at_every_larger_size']}"
                    )
                else:
                    lines.append(
                        f"{head}|not reached|largest {d['largest_beta_tried']:.4f}"
                        f"|share {d['share_at_largest']:.2f}"
                    )
            for c in b["planted"]:
                lines.append(
                    f"P|{harness}|{arm}|{name}|beta {c['beta']:.4f}"
                    f"|true acc {c['oracle_acc']['mean']:+.3f} ll {c['oracle_ll']['mean']:+.6f}"
                    f"|fit acc {c['fitted_acc']['mean']:+.3f} ll {c['fitted_ll']['mean']:+.6f}"
                    f"|det acc {c[f'detect_acc_{LEVEL}']:.2f} ll {c[f'detect_ll_{LEVEL}']:.2f}"
                    f"|adm {c.get('admissible_share', 1.0):.2f}"
                )
    return lines


def null_cells(harness, arm, league, seasons, games, blocks, tpl, column, s, how):
    cells = []
    where = "college" if league == "cfb" else "pro"
    for k, units in zip(M36.SHORT, M36.METRICS, strict=True):
        cells.append(
            {
                "name": f"xlghp_null_{harness}_{arm}_{tpl['name']}_{units}",
                "description": (
                    f"Matched-marginal null on {how}: the real {column} column shuffled among "
                    f"games within each season, {DRAWS} draws (seed {SEED}); effect is the mean "
                    "of draws, interval the 5th to 95th percentile, probability_positive the "
                    "share of draws above zero; median changed picks "
                    f"{s['changed']['q50']:.0f}, median record "
                    f"{s['arm_right']['q50']:.0f}-{s['ref_right']['q50']:.0f}."
                ),
                "source": LANE,
                "effect": round(s[k]["mean"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": league,
                "season_start": int(seasons[0]),
                "season_end": int(seasons[-1]),
                "family": FAMILY,
                "category": "control",
                "classification_evidence": (
                    "Negative-control null distribution of a term with no game link; it is "
                    "an instrument reading, closes nothing and is no closing ground."
                ),
                "notes": (
                    f"Family {FAMILY}: {len(TEMPLATES)} templates x {len(M36.METRICS)} metrics "
                    "x 3 recorded arms (college add-one, shared-harness college arm, "
                    "shared-harness pro arm); planted-effect cells, detection shares and the "
                    "cross-league pooled null stay in the lane."
                ),
                "interval_low": round(s[k]["q05"], 6),
                "interval_high": round(s[k]["q95"], 6),
                "probability_positive": round(float(s[k]["share_positive"]), 4),
                "sample_games": int(games),
                "sample_blocks": int(blocks),
                "plain_summary": (
                    f"A check that scrambles {tpl['what']} in {where} games, to show how much "
                    "a meaningless extra ingredient moves a test model's score."
                ),
            }
        )
    return cells


def reproduction(S, reals):
    want = json.loads(XLG10_CHECK.read_text(encoding="utf-8"))
    got = {r["reference"]: r for r in reals[TEMPLATES[0]["name"]]}
    out = {}
    for lg, key in (("nfl", "nfl_opener"), ("cfb", "cfb")):
        w = want["populations"][key]
        out[lg] = {
            "base_wins": [S[lg]["base_wins"], w["base"]["wins"]],
            "games": [S[lg]["games"], w["base"]["n"]],
            "flips_win": [got[lg]["arm_right"], w["flips_win"]],
            "flips_loss": [got[lg]["ref_right"], w["flips_loss"]],
            "accuracy_effect": [got[lg]["acc"], w["accuracy"]["effect"]],
            "accuracy_p_pos": [got[lg]["pp_acc"], w["accuracy"]["p_pos"]],
            "log_loss_effect": [got[lg]["ll"], w["log_loss"]["effect"]],
        }
    p = want["pooled"]["accuracy"]
    out["pooled"] = {
        "accuracy_effect": [got["pooled"]["acc"], p["effect"]],
        "accuracy_p_pos": [got["pooled"]["pp_acc"], p["p_pos"]],
        "het_z": [got["pooled"]["het_z"], p["het_z"]],
    }
    return out


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = mod36_reference()
    frame = addone_frame()
    print("addone frame", len(frame), sorted(frame.season.unique()), flush=True)
    a_null, a_plant, states, a_reals, cross, a_grids = run_addone(frame, ref, started)
    S = shared_state()
    for lg in LEAGUES:
        L = S[lg]
        print(
            "shared",
            lg,
            "train rows",
            L["train_rows"],
            "weeks",
            L["weeks"],
            "graded",
            L["games"],
            "base wins",
            L["base_wins"],
            "base ll",
            round(L["base_log_loss"], 5),
            flush=True,
        )
    b_null, b_plant, b_grids, b_reals = run_shared(S, ref, cross, started)
    repro = reproduction(S, b_reals)
    print("REPRO", json.dumps(repro), flush=True)
    a_null["harness"], a_plant["harness"] = "cfb_addone", "cfb_addone"
    b_null["harness"], b_plant["harness"] = "shared14", "shared14"
    pd.concat([a_null, a_plant, b_null, b_plant]).to_parquet(OUT / "draws.parquet", index=False)
    a_block = summarise(a_null, a_plant)
    b_block = summarise(b_null, b_plant)
    a_games = {"served": len(frame)}
    b_games = {lg: S[lg]["games"] for lg in LEAGUES}
    b_games["pooled"] = sum(b_games.values())
    lines = report("cfb_addone", a_games, a_block) + report("shared14", b_games, b_block)
    for tpl in TEMPLATES:
        m = ref["templates"][tpl["name"]]
        a, ll = m["null"]["acc"], m["null"]["ll"]
        lines.append(
            f"T|nfl_served_mod36|served|{tpl['name']}|n={m['games']}"
            f"|acc {a['mean']:+.3f} [{a['q05']:+.3f},{a['q95']:+.3f}]"
            f"|ll {ll['mean']:+.6f} [{ll['q05']:+.6f},{ll['q95']:+.6f}]"
            f"|coef sd {m['null_coefficient_sd']:.4f}"
        )
        for k, d in m["detectable"].items():
            lines.append(
                f"D|nfl_served_mod36|served|{tpl['name']}|{k}|beta {d['beta_per_sd']:.4f}"
                f"|true acc {d['true_accuracy_points']:+.3f}"
                f"|true ll {d['true_log_loss_improvement']:+.6f}"
            )
    for line in lines:
        print(line)
    cells = []
    a_seasons = sorted(int(s) for s in frame.season.unique())
    for tpl, st, real in zip(TEMPLATES, states, a_reals, strict=True):
        print(
            "REAL cfb_addone",
            tpl["name"],
            "coef/sd",
            round(real["full_coefficient_per_sd"], 4),
            "folds",
            [round(v, 3) for v in real["fold_coefficients_per_sd"]],
            {k: round(v["effect"], 5) for k, v in real["served"].items() if k in M36.SHORT},
            real["served"]["arm_right"],
            real["served"]["ref_right"],
        )
        cells.extend(
            null_cells(
                "addone",
                "cfb",
                "cfb",
                a_seasons,
                st["n"],
                len(a_seasons),
                tpl,
                tpl["column"],
                a_block[tpl["name"]]["served"]["null"],
                "the college add-one harness (one offset-logistic coefficient on a "
                "leave-one-season-out ridge cover probability at the opener, 13 seasons)",
            )
        )
        for lg in LEAGUES:
            L = S[lg]
            cells.extend(
                null_cells(
                    "shared14",
                    lg,
                    lg,
                    L["graded_seasons"],
                    L["games"],
                    len(L["graded_seasons"]),
                    tpl,
                    tpl["shared_column"],
                    b_block[tpl["name"]][lg]["null"],
                    f"the shared 14-input ridge harness, {lg} arm (the column refit as a "
                    "15th input, walk-forward by week, graded at the opener)",
                )
            )
    summary = {
        "family": FAMILY,
        "draws": DRAWS,
        "seed": SEED,
        "streams": {"cfb_addone": STREAM_ADDONE, "shared14": STREAM_SHARED},
        "notable_levels": list(M36.NOTABLE),
        "power_target": M36.POWER_TARGET,
        "mod36_reference": ref,
        "transfer": transfer(),
        "cross_league_fitted_sizes_logit_per_sd": cross,
        "harnesses": {
            "cfb_addone": {
                "games": a_games,
                "seasons": a_seasons,
                "shapes": {t["name"]: st["shape"] for t, st in zip(TEMPLATES, states, strict=True)},
                "real": {t["name"]: r for t, r in zip(TEMPLATES, a_reals, strict=True)},
                "grids": a_grids,
                "templates": a_block,
            },
            "shared14": {
                "games": b_games,
                "columns": S["cols"],
                "leagues": {
                    lg: {
                        "train_rows": S[lg]["train_rows"],
                        "weeks": S[lg]["weeks"],
                        "graded_seasons": S[lg]["graded_seasons"],
                        "base_wins": S[lg]["base_wins"],
                        "base_log_loss": S[lg]["base_log_loss"],
                        "shapes": {
                            n: {k: v for k, v in p.items() if not isinstance(v, np.ndarray | list)}
                            for n, p in S[lg]["plant"].items()
                        },
                    }
                    for lg in LEAGUES
                },
                "reproduction": repro,
                "real": b_reals,
                "grids": b_grids,
                "templates": b_block,
            },
        },
        "table": lines,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": cells}, indent=1), encoding="utf-8"
    )
    print("cells", len(cells), f"elapsed {time.time() - started:.0f}s", flush=True)


main()
