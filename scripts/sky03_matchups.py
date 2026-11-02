import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sky04_lead_lag import boot, fit_b, logit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sky03"
PERGAME = ROOT / "artifacts" / "pick_probability" / "20260929T192747Z" / "per_game.parquet"
PBP_GLOB = str(ROOT / "data" / "pbp" / "raw" / "*" / "season={s}" / "plays.parquet")
WARMUP_START = 2012
TRAIT_START = 2014
EVAL_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
SEED = 20261102
BOOT = 2000
TRAITS = {
    "oP": ("press", "n_db"), "dP": ("press", "n_db"),
    "oE": ("epa", "n_db"), "dE": ("epa", "n_db"),
    "oN": ("epa_ns", "n_ns"), "dN": ("epa_ns", "n_ns"),
}


def load_games():
    cache = OUT / "team_games.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for s in range(WARMUP_START, max(EVAL_SEASONS) + 1):
        f = sorted(glob.glob(PBP_GLOB.format(s=s)))[-1]
        d = pd.read_parquet(f, columns=["game_id", "season_type", "week", "posteam", "defteam", "qb_dropback", "qb_kneel", "qb_spike", "sack", "qb_hit", "epa"])
        d = d[(d.season_type == "REG") & (d.qb_dropback == 1) & (d.qb_kneel != 1) & (d.qb_spike != 1) & d.epa.notna() & d.posteam.notna()].copy()
        d["season"] = s
        d["press"] = ((d.sack == 1) | (d.qb_hit == 1)).astype(float)
        d["ns"] = (d.sack != 1).astype(float)
        d["epa_ns_v"] = d.epa * d.ns
        g = d.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False).agg(
            n_db=("epa", "size"), n_ns=("ns", "sum"), press_s=("press", "sum"), epa_s=("epa", "sum"), epa_ns_s=("epa_ns_v", "sum"))
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    t = t.rename(columns={"posteam": "off", "defteam": "deff"})
    t["press"] = t.press_s / t.n_db
    t["epa"] = t.epa_s / t.n_db
    t["epa_ns"] = t.epa_ns_s / t.n_ns.clip(lower=1)
    t["key"] = t.season * 100 + t.week
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(cache)
    return t


def grp_of(trait):
    return "off" if trait[0] == "o" else "deff"


def variance_components(t, season):
    h = t[t.season < season]
    res = {}
    for name, (val, nn) in TRAITS.items():
        grp = grp_of(name)
        pairs = []
        num = 0.0
        den = 0.0
        for (s, tm), g in h.groupby(["season", grp]):
            if len(g) < 4:
                continue
            m = np.average(g[val], weights=g[nn].clip(lower=1e-9))
            num += float((g[nn] * (g[val] - m) ** 2).sum())
            den += float(g[nn].sum() - len(g))
            a = g[g.week % 2 == 1]
            b = g[g.week % 2 == 0]
            if len(a) and len(b):
                pairs.append((np.average(a[val], weights=a[nn]), np.average(b[val], weights=b[nn])))
        oe = np.array(pairs)
        tau2 = max(float(np.cov(oe[:, 0], oe[:, 1])[0, 1]), 1e-12)
        res[name] = num / den / tau2
    return res


def as_of_traits(t, season, week, k):
    pool = t[((t.season == season) & (t.week < week)) | (t.season == season - 1)]
    assert pool.empty or pool.key.max() < season * 100 + week
    out = {}
    for name, (val, nn) in TRAITS.items():
        grp = grp_of(name)
        p = pool.assign(w=pool[nn], wv=pool[nn] * pool[val])
        mu = p.wv.sum() / p.w.sum()
        a = p.groupby(grp).agg(w=("w", "sum"), wv=("wv", "sum"))
        out[name] = ((a.wv + k[name] * mu) / (a.w + k[name]), mu)
    dbm = pool.groupby("off").n_db.mean()
    share = pool.n_ns.sum() / pool.n_db.sum()
    return out, dbm, share


def build_rows(t, seasons):
    parts = []
    for s in seasons:
        k = variance_components(t, s)
        for w in sorted(t[t.season == s].week.unique()):
            cur = t[(t.season == s) & (t.week == w)].copy()
            tr, dbm, share = as_of_traits(t, s, w, k)
            for name, (ser, mu) in tr.items():
                cur[name] = cur[grp_of(name)].map(ser).fillna(mu) - mu
            cur["muE"] = tr["oE"][1]
            cur["muN"] = tr["oN"][1]
            cur["db_exp"] = cur.off.map(dbm).fillna(dbm.mean())
            cur["ns_share"] = share
            parts.append(cur)
    r = pd.concat(parts, ignore_index=True)
    r["zA"] = r.oP * r.dP
    r["zB"] = r.oN * r.dN
    r["rA"] = r.epa - r.muE - r.oE - r.dE
    r["rB"] = r.epa_ns - r.muN - r.oN - r.dN
    return r


def fit_c(rr, zc, rc, wc):
    den = (rr[wc] * rr[zc] ** 2).sum()
    return float((rr[wc] * rr[zc] * rr[rc]).sum() / den) if den > 0 else 0.0


def reliability(rr, seasons):
    rng = np.random.default_rng(SEED)
    d = rr[rr.season.isin(seasons) & (rr.week > 1)]
    out = {}
    for lab, zc, rc, wc in (("A_protection_x_rush", "zA", "rA", "n_db"), ("B_receivers_x_coverage", "zB", "rB", "n_ns")):
        odd = d[d.week % 2 == 1]
        even = d[d.week % 2 == 0]
        res = {"pooled_c_odd": fit_c(odd, zc, rc, wc), "pooled_c_even": fit_c(even, zc, rc, wc)}
        for side, grp in (("offense", "off"), ("defense", "deff")):
            teams = sorted(d[grp].unique())
            so = np.array([fit_c(odd[odd[grp] == tm], zc, rc, wc) for tm in teams])
            se = np.array([fit_c(even[even[grp] == tm], zc, rc, wc) for tm in teams])
            r0 = float(np.corrcoef(so, se)[0, 1])
            bs = np.array([np.corrcoef(so[i], se[i])[0, 1] for i in (rng.integers(0, len(teams), len(teams)) for _ in range(BOOT))])
            res[f"team_slope_split_half_{side}"] = {"r": r0, "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
                                                    "p_positive": float(np.nanmean(bs > 0)), "teams": len(teams)}
        gm = d.game_id.unique()
        bs = []
        for _ in range(BOOT // 4):
            cnt = pd.Series(rng.choice(gm, len(gm))).value_counts()
            dd = d.assign(bw=d.game_id.map(cnt).fillna(0).values * d[wc])
            bs.append(fit_c(dd[dd.week % 2 == 1], zc, rc, "bw") - fit_c(dd[dd.week % 2 == 0], zc, rc, "bw"))
        res["pooled_c_odd_minus_even_boot"] = [float(np.mean(bs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res["rows"] = int(len(d))
        out[lab] = res
    return out


def game_x(rr, seasons):
    rows = []
    for s in seasons:
        for w in sorted(rr[rr.season == s].week.unique()):
            hist = rr[rr.key < s * 100 + w]
            assert hist.empty or hist.key.max() < s * 100 + w
            cur = rr[(rr.season == s) & (rr.week == w)]
            cA = fit_c(hist, "zA", "rA", "n_db")
            cB = fit_c(hist, "zB", "rB", "n_ns")
            cur = cur.assign(pts=cur.db_exp * (cA * cur.zA + cB * cur.zB * cur.ns_share), cA=cA, cB=cB)
            rows.append(cur[["game_id", "off", "deff", "pts", "cA", "cB"]])
    return pd.concat(rows, ignore_index=True)


def fitted_term(rr, seasons):
    g = pd.read_parquet(PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(seasons)]
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    gx = game_x(rr, seasons)
    hp = gx.merge(sub[["game_id", "home_team"]], on="game_id")
    hp["sp"] = np.where(hp.off == hp.home_team, 1.0, -1.0) * hp.pts
    sub["x"] = sub.game_id.map(hp.groupby("game_id").sp.sum())
    sub = sub[sub.x.notna()].reset_index(drop=True)
    y = sub.home_covered.astype(float).values
    pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
    off = logit(pr)
    xs = sub.x.values
    se = sub.season.values
    pooled = np.zeros(len(sub))
    fc = {}
    for s in seasons:
        te = se == s
        b = fit_b(off[~te], xs[~te], y[~te])
        fc[int(s)] = float(b)
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * xs[te])))
    b_all = fit_b(off, xs, y)
    ins = 1 / (1 + np.exp(-(off + b_all * xs)))
    res = {"games": int(len(sub)), "x_sd_points": float(xs.std()), "fold_coefficients": fc, "pooled_coefficient": float(b_all),
           "c_A_last": float(gx.cA.iloc[-1]), "c_B_last": float(gx.cB.iloc[-1])}
    rng = np.random.default_rng(SEED)
    for label, arr in (("oos", pooled), ("ins", ins)):
        d_brier = (pr - y) ** 2 - (arr - y) ** 2
        d_ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr)) + (y * np.log(arr) + (1 - y) * np.log(1 - arr))
        d_acc = (((arr > 0.5) == (y == 1)).astype(float) - ((pr > 0.5) == (y == 1)).astype(float)) * 100
        dec = (pr > 0.5) != (arr > 0.5)
        res[label] = {"brier_gain": boot(d_brier, rng), "logloss_gain": boot(d_ll, rng), "accuracy_gain_points": boot(d_acc, rng),
                      "decisive_games": int(dec.sum()), "decisive_new_correct": int(((arr > 0.5) == (y == 1))[dec].sum()),
                      "decisive_base_correct": int(((pr > 0.5) == (y == 1))[dec].sum())}
    res["gap_brier_gain_ins_minus_oos"] = res["ins"]["brier_gain"][0] - res["oos"]["brier_gain"][0]
    res["oos_brier_gain_by_season"] = {int(s): float(np.mean(((pr - y) ** 2 - (pooled - y) ** 2)[se == s])) for s in seasons}
    res["corr_x_with_home_cover_residual"] = float(np.corrcoef(xs, y - pr)[0, 1])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rel", "term", "all"], default="all")
    ap.add_argument("--seasons", default=",".join(map(str, EVAL_SEASONS)))
    a = ap.parse_args()
    seasons = tuple(int(x) for x in a.seasons.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_games()
    rr = build_rows(t, tuple(range(TRAIT_START + 1, max(seasons) + 1)))
    print("rows", len(rr), flush=True)
    if a.stage in ("rel", "all"):
        rel_seasons = tuple(range(TRAIT_START + 2, max(seasons) + 1))
        r = reliability(rr, rel_seasons)
        json.dump(r, open(OUT / "reliability.json", "w"), indent=1)
        print(json.dumps(r, indent=1), flush=True)
    if a.stage in ("term", "all"):
        r = fitted_term(rr, seasons)
        json.dump(r, open(OUT / "fitted_term.json", "w"), indent=1)
        print(json.dumps(r, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
