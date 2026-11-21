import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import binom, binomtest, norm
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "mkt24_slot_residual"
OPENER = ROOT / "artifacts" / "opener_evaluation" / "20261103T195331Z" / "per_game.parquet"
PROXY = ROOT / "artifacts" / "mod24_u1" / "base_per_game.parquet"
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
ERA_PROXY = (2011, 2019)
ERA_OPENER = (2020, 2025)
SPLIT_LABELS = {
    "era": ["2011-2019 SBR open", "2020-2025 Tuesday open"],
    "parity": ["even seasons 2011-2025", "odd seasons 2011-2025"],
    "parity_opener": ["even seasons 2020-2025", "odd seasons 2020-2025"],
}
PRIMARY_SPLITS = ("era", "parity")
THURSDAY = "thursday_night"
PLANT_GRID_POINTS = [1.0, 2.0, 3.0, 4.0, 6.0]
ALPHA_LEVEL = 0.05
INTERVAL_PCT = (2.5, 97.5)
EVEN = 0.5
FAMILY = "mkt24_slot_opener_residual"
UNIT2_LOOKS = 113
GAP_MINUTES = 60
PEN_SCALE = 100.0
C_GRID = [0.01, 0.1, 1.0, 10.0]
DRAWS = 5000
SEED = 1
POPS = {"opener_2020_2025": (2020, 2025), "sbr_proxy_2011_2019": (2011, 2019)}


def kickoff_cuts(sched):
    sun = sched[(sched.weekday == "Sunday") & (sched.location == "Home")]
    mins = np.sort(sun.minute.unique())
    return [
        (mins[i] + mins[i + 1]) / 2
        for i in range(len(mins) - 1)
        if mins[i + 1] - mins[i] > GAP_MINUTES
    ]


def holiday_mask(day):
    thanks = (day.dt.month == 11) & (day.dt.weekday == 3) & day.dt.day.between(22, 28)
    xmas = (day.dt.month == 12) & (day.dt.day == 25)
    return thanks | xmas


def build_slots():
    s = pd.read_parquet(FEATURES)
    s = s[s.game_type == "REG"].copy()
    s["minute"] = s.gametime.str.slice(0, 2).astype(int) * 60 + s.gametime.str.slice(3, 5).astype(
        int
    )
    cuts = kickoff_cuts(s)
    day = pd.to_datetime(s.gameday)
    slot = pd.Series("other_weekday", index=s.index)
    wd = s.weekday
    slot[wd == "Thursday"] = "thursday_night"
    slot[wd == "Monday"] = "monday_night"
    slot[wd == "Saturday"] = "saturday"
    sun = wd == "Sunday"
    slot[sun & (s.minute < cuts[0])] = "sunday_early"
    slot[sun & (s.minute >= cuts[0]) & (s.minute < cuts[1])] = "sunday_late"
    slot[sun & (s.minute >= cuts[1])] = "sunday_night"
    slot[holiday_mask(day)] = "holiday"
    slot[s.location == "Neutral"] = "international"
    s["slot"] = slot
    return s[["game_id", "slot"]], [float(c) for c in cuts]


def population(name, slots):
    lo, hi = POPS[name]
    if name.startswith("opener"):
        o = pd.read_parquet(OPENER)
        d = o[o.season.between(lo, hi) & o.margin_vs_open.ne(0)].copy()
        d["y"] = (d.margin_vs_open > 0).astype(int)
        d["p"] = d.home_cover_probability_at_open
    else:
        b = pd.read_parquet(PROXY).reset_index()
        d = b[b.season.between(lo, hi) & b.correct.isin([0.0, 1.0])].copy()
        d["y"] = np.where(d.pick_home, d.correct, 1 - d.correct).astype(int)
    d = d[["game_id", "season", "week", "y", "p"]]
    d = d.merge(slots, on="game_id", how="left").reset_index(drop=True)
    d = d[d.slot.notna() & d.slot.ne("other_weekday")].reset_index(drop=True)
    d["z"] = logit(d.p.clip(1e-6, 1 - 1e-6))
    d["r"] = d.y - d.p
    return d


def shrink(d):
    g = d.groupby("slot").r
    n = g.size()
    m = g.mean()
    se2 = (g.var(ddof=1) / n).fillna(0.0)
    w = 1 / se2
    mu = (w * m).sum() / w.sum()
    q = (w * (m - mu) ** 2).sum()
    k = len(n)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    b = tau2 / (tau2 + se2)
    post = b * m
    psd = np.sqrt(b * se2)
    raw_p = pd.Series(norm.cdf(m / np.sqrt(se2)), index=n.index)
    pp = pd.Series(norm.cdf(post / psd.replace(0, np.nan)), index=n.index).fillna(0.5)
    return pd.DataFrame(
        {
            "n": n,
            "raw_resid": m,
            "raw_se": np.sqrt(se2),
            "raw_Ppos": raw_p,
            "tau2": tau2,
            "B": b,
            "shrunk_resid": post,
            "shrunk_sd": psd,
            "shrunk_lo": post - 1.96 * psd,
            "shrunk_hi": post + 1.96 * psd,
            "shrunk_Ppos": pp,
        }
    )


def design(d, slots_list):
    x = pd.DataFrame({"zs": d.z * PEN_SCALE}, index=d.index)
    for s in slots_list:
        x[s] = (d.slot == s).astype(float)
    return x


def fit_pred(tr, te, slots_list, c):
    m = LogisticRegression(C=c, max_iter=2000).fit(design(tr, slots_list), tr.y)
    return m.predict_proba(design(te, slots_list))[:, 1], m


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def choose_c(tr, slots_list):
    best, bll = None, np.inf
    for c in C_GRID:
        tot = 0.0
        for s in sorted(tr.season.unique()):
            a, b = tr[tr.season != s], tr[tr.season == s]
            p, _ = fit_pred(a, b, slots_list, c)
            tot += ll(b.y.values, p).sum()
        if tot < bll:
            best, bll = c, tot
    return best


def loso(d):
    slots_list = sorted(d.slot.unique())
    P = {k: pd.Series(np.nan, index=d.index) for k in ["intercept", "served", "slot"]}
    coefs, cs = {}, {}
    for s in sorted(d.season.unique()):
        tr, te = d[d.season != s], d[d.season == s]
        P["intercept"][te.index] = tr.y.mean()
        mz = LogisticRegression(C=1e6, max_iter=2000).fit(tr[["z"]], tr.y)
        P["served"][te.index] = mz.predict_proba(te[["z"]])[:, 1]
        c = choose_c(tr, slots_list)
        p, m = fit_pred(tr, te, slots_list, c)
        P["slot"][te.index] = p
        cs[int(s)] = c
        coefs[int(s)] = {
            "z": float(m.coef_[0][0] * PEN_SCALE),
            **{k: float(v) for k, v in zip(slots_list, m.coef_[0][1:], strict=True)},
        }
    c_all = choose_c(d, slots_list)
    p_in, _ = fit_pred(d, d, slots_list, c_all)
    mz = LogisticRegression(C=1e6, max_iter=2000).fit(d[["z"]], d.y)
    ins = {"slot": p_in, "served": mz.predict_proba(d[["z"]])[:, 1]}
    return {k: v.clip(1e-6, 1 - 1e-6) for k, v in P.items()}, coefs, cs, ins


def compare(d, P, a, b, rng):
    y = d.y.values
    seas = d.season.values
    useas = np.unique(seas)
    dl = ll(y, P[b].values) - ll(y, P[a].values)
    dbr = (P[b].values - y) ** 2 - (P[a].values - y) ** 2
    acc = ((P[a].values > 0.5) == (y == 1)).astype(float) - (
        (P[b].values > 0.5) == (y == 1)
    ).astype(float)
    bg = np.array([dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(DRAWS)])
    bl, ba = [], []
    for _ in range(DRAWS):
        sel = np.concatenate([np.where(seas == s)[0] for s in rng.choice(useas, len(useas))])
        bl.append(dl[sel].mean())
        ba.append(acc[sel].mean() * 100)
    bl, ba = np.array(bl), np.array(ba)
    sb = pd.Series(dl).groupby(seas).mean()
    return {
        "dLL": float(dl.mean()),
        "dLL_game_CI": [float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))],
        "dLL_game_Ppos": float((bg > 0).mean()),
        "dLL_season_CI": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
        "dLL_season_Ppos": float((bl > 0).mean()),
        "dBrier": float(dbr.mean()),
        "dAccPts": float(acc.mean() * 100),
        "dAcc_season_CI": [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))],
        "dAcc_season_Ppos": float((ba > 0).mean()),
        "seasons_positive": f"{int((sb > 0).sum())}/{len(sb)}",
    }


def analyse(name, slots):
    d = population(name, slots)
    rng = np.random.default_rng(SEED)
    y = d.y.values
    out = {"n": len(d), "slot_counts": d.slot.value_counts().to_dict()}
    out["shrink"] = shrink(d).round(5).reset_index().to_dict("records")
    P, coefs, cs, ins = loso(d)
    arms = {}
    for a in P:
        w = int(((P[a] > 0.5).values == (y == 1)).sum())
        arms[a] = {
            "W": w,
            "L": len(d) - w,
            "LL": float(ll(y, P[a].values).mean()),
            "Brier": float(((P[a].values - y) ** 2).mean()),
        }
    out["arms"] = arms
    fl = (P["slot"] > 0.5).values != (P["served"] > 0.5).values
    fw = int(((P["slot"] > 0.5).values[fl] == (y[fl] == 1)).sum())
    out["decisive_flips"] = {
        "flips": int(fl.sum()),
        "slot_right": fw,
        "served_right": int(fl.sum()) - fw,
    }
    out["comparisons"] = {
        "slot_vs_served": compare(d, P, "slot", "served", rng),
        "served_vs_intercept": compare(d, P, "served", "intercept", rng),
    }
    out["insample"] = {
        "in_sample_dLL": float((ll(y, ins["served"]) - ll(y, ins["slot"])).mean()),
        "out_of_sample_dLL": out["comparisons"]["slot_vs_served"]["dLL"],
    }
    out["chosen_C"] = cs
    out["fold_coefs"] = coefs
    return out


def residual_frame(slots):
    f = pd.read_parquet(FEATURES)
    f = f[f.game_type == "REG"][["game_id", "season", "week", "result"]]
    o = pd.read_parquet(OPENER)[["game_id", "margin_vs_open"]]
    s = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"])
    s = s.drop_duplicates("game_id")[["game_id", "open_home_spread", "open_ambiguous"]]
    d = f.merge(o, on="game_id", how="left").merge(s, on="game_id", how="left")
    opener = d.season.between(*ERA_OPENER)
    proxy = d.season.between(*ERA_PROXY)
    d["resid"] = np.where(
        opener, d.margin_vs_open, np.where(proxy, d.result - d.open_home_spread, np.nan)
    )
    d = d.merge(slots, on="game_id", how="left")
    d = d[d.resid.notna() & d.slot.notna() & d.slot.ne("other_weekday")]
    return d.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def half_of(season, name):
    odd = np.asarray(season) % 2
    lo, hi = ERA_PROXY
    if name == "era":
        return np.where((season >= lo) & (season <= hi), 0, 1)
    if name == "parity":
        return odd
    lo, hi = ERA_OPENER
    return np.where((season >= lo) & (season <= hi), odd, -1)


def season_tables(d, slot_idx, k):
    seasons = np.sort(d.season.unique())
    idx = np.searchsorted(seasons, d.season.to_numpy()) * k + slot_idx
    shape = (len(seasons), k)
    r = d.resid.to_numpy(float)
    cols = {
        "n": np.ones(len(r)),
        "sm": r,
        "ss": r**2,
        "np": (r != 0) * 1.0,
        "win": (r > 0) * 1.0,
    }
    return seasons, {
        key: np.bincount(idx, weights=v, minlength=len(seasons) * k).reshape(shape)
        for key, v in cols.items()
    }


def boot_tables(tab, rows, rng):
    share = np.full(len(rows), 1 / len(rows))
    c = rng.multinomial(len(rows), share, size=DRAWS).astype(float)
    return {key: c @ v[rows] for key, v in tab.items()}


def permute(d, slot_idx, k, halves, rng):
    grp = (d.season * 100 + d.week).to_numpy()
    r = d.resid.to_numpy(float)
    out = {s: (np.zeros((DRAWS, 2, k)), np.zeros((DRAWS, 2, k))) for s in halves}
    for i in range(DRAWS):
        ps = slot_idx[np.lexsort((rng.random(len(r)), grp))]
        for s, h in halves.items():
            keep = h >= 0
            idx = h[keep] * k + ps[keep]
            out[s][0][i] = np.bincount(idx, minlength=2 * k).reshape(2, k)
            out[s][1][i] = np.bincount(idx, weights=r[keep], minlength=2 * k).reshape(2, k)
    return out


def pattern_stats(n, sm, thu):
    with np.errstate(divide="ignore", invalid="ignore"):
        m = sm / n
        a, b = m[..., 0, :], m[..., 1, :]
        na, nb = n[..., 0, :], n[..., 1, :]
        ok = (na > 0) & (nb > 0)
        a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
        out = {"product": a * b, "concordance": 2 * a * b / (a**2 + b**2)}
        weights = {
            "weighted": np.where(ok, na * nb / np.where(ok, na + nb, 1.0), 0.0),
            "unweighted": ok.astype(float),
        }
        for key, w in weights.items():
            tot = w.sum(-1)
            ma = np.asarray((w * a0).sum(-1) / tot)[..., None]
            mb = np.asarray((w * b0).sum(-1) / tot)[..., None]
            cov = (w * (a0 - ma) * (b0 - mb)).sum(-1) / tot
            va = (w * (a0 - ma) ** 2).sum(-1) / tot
            vb = (w * (b0 - mb) ** 2).sum(-1) / tot
            out[f"corr_{key}"] = cov / np.sqrt(va * vb)
            out[f"cov_{key}"] = cov
        out["agree"] = np.sign(a) == np.sign(b)
    out["thu_product"] = out["product"][..., thu]
    out["thu_concordance"] = out["concordance"][..., thu]
    return out


def summarise(value, draws):
    live = np.asarray(draws, float)
    live = live[~np.isnan(live)]
    lo, hi = np.percentile(live, INTERVAL_PCT)
    return {
        "value": float(value),
        "lo": float(lo),
        "hi": float(hi),
        "ppos": float((live > 0).mean()),
    }


def perm_p(value, null):
    live = null[~np.isnan(null)]
    return float((1 + (live >= value).sum()) / (1 + len(live)))


def slot_rows(obs, bt, slot_names, stat, null):
    rows = []
    for j, slot in enumerate(slot_names):
        row = {
            "slot": slot,
            "sign_agrees": bool(stat["agree"][j]),
            "product": float(stat["product"][j]),
            "product_perm_p": perm_p(stat["product"][j], null["product"][:, j]),
        }
        for h in (0, 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_draws = bt["sm"][:, h, j] / bt["n"][:, h, j]
                cover_draws = 100 * (bt["win"][:, h, j] / bt["np"][:, h, j] - EVEN)
            wins, live = int(obs["win"][h, j]), int(obs["np"][h, j])
            n = obs["n"][h, j]
            mean = obs["sm"][h, j] / n
            se = np.sqrt((obs["ss"][h, j] - n * mean**2) / (n - 1) / n)
            z = norm.ppf(INTERVAL_PCT[1] / 100)
            test = binomtest(wins, live, EVEN)
            ci = test.proportion_ci(
                confidence_level=(INTERVAL_PCT[1] - INTERVAL_PCT[0]) / 100, method="exact"
            )
            row[f"half{h}"] = {
                "n": int(n),
                "mean": {
                    **summarise(mean, mean_draws),
                    "game_lo": float(mean - z * se),
                    "game_hi": float(mean + z * se),
                    "game_ppos": float(norm.cdf(mean / se)),
                },
                "cover": {
                    **summarise(100 * (wins / live - EVEN), cover_draws),
                    "game_lo": float(100 * (ci.low - EVEN)),
                    "game_hi": float(100 * (ci.high - EVEN)),
                    "game_ppos": float(binom.sf(live // 2, live, wins / live)),
                },
                "cover_record": f"{wins}-{live - wins}",
                "binom_p": float(test.pvalue),
            }
        rows.append(row)
    return rows


def conservative(s):
    return {
        "value": s["value"],
        "lo": min(s["lo"], s["game_lo"]),
        "hi": max(s["hi"], s["game_hi"]),
        "ppos": min((s["ppos"], s["game_ppos"]), key=lambda p: abs(p - EVEN)),
    }


def power_text(rows, key):
    return ", ".join(f"{row['points']:.0f} pt {row[f'power_{key}']:.2f}" for row in rows)


def planted(pn, ps, thu, r, null):
    live = r != 0
    base = (r[live] > 0).mean()
    rows = []
    for pts in PLANT_GRID_POINTS:
        sp = ps.copy()
        sp[..., thu] += pts * pn[..., thu]
        st = pattern_stats(pn, sp, thu)
        moved = r + pts
        row = {
            "points": pts,
            "cover_shift_pp": float(100 * ((moved[moved != 0] > 0).mean() - base)),
        }
        for key in ("corr_weighted", "thu_product"):
            crit = np.nanquantile(null[key], 1 - ALPHA_LEVEL)
            row[f"power_{key}"] = float(np.nanmean(st[key] > crit))
        rows.append(row)
    return rows


def split_report(name, tab, seasons, perm_tables, thu, slot_names, r, rng):
    season_half = half_of(seasons, name)
    rows = [np.where(season_half == h)[0] for h in (0, 1)]
    obs = {key: np.stack([v[x].sum(0) for x in rows]) for key, v in tab.items()}
    boot = [boot_tables(tab, x, rng) for x in rows]
    bt = {key: np.stack([boot[0][key], boot[1][key]], axis=1) for key in tab}
    stat = pattern_stats(obs["n"], obs["sm"], thu)
    bstat = pattern_stats(bt["n"], bt["sm"], thu)
    null = pattern_stats(*perm_tables, thu)
    keys = ("corr_weighted", "corr_unweighted", "cov_weighted", "thu_product", "thu_concordance")
    pattern = {
        key: {
            **summarise(stat[key], bstat[key]),
            "perm_p": perm_p(stat[key], null[key]),
            "null_mean": float(np.nanmean(null[key])),
        }
        for key in keys
    }
    cov = pattern["cov_weighted"]["value"]
    out = {
        "halves": SPLIT_LABELS[name],
        "seasons": [[int(s) for s in seasons[x]] for x in rows],
        "games": [int(obs["n"][h].sum()) for h in (0, 1)],
        "all_games_mean": [float(obs["sm"][h].sum() / obs["n"][h].sum()) for h in (0, 1)],
        "slots": slot_rows(obs, bt, slot_names, stat, null),
        "pattern": pattern,
        "implied_slot_sd_points": float(np.sign(cov) * np.sqrt(abs(cov))),
        "planted_thursday": planted(*perm_tables, thu, r, null),
    }
    return out, stat["agree"], null["agree"]


def make_cell(name, description, units, summary, plain, seasons, games, blocks, rel, evidence):
    lo, hi = min(summary["lo"], summary["value"]), max(summary["hi"], summary["value"])
    return {
        "name": name,
        "description": description,
        "source": "artifacts/mkt24_slot_residual/reliability.json",
        "effect": summary["value"],
        "effect_units": units,
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": seasons[0],
        "season_end": seasons[1],
        "family": FAMILY,
        "category": "market",
        "classification_evidence": evidence,
        "notes": f"MKT-24 unit 2; {UNIT2_LOOKS} looks declared before computing; nothing served.",
        "interval_low": lo,
        "interval_high": hi,
        "probability_positive": summary["ppos"],
        "sample_games": games,
        "sample_blocks": blocks,
        "reliability": rel,
        "plain_summary": plain,
    }


SLOT_WORDS = {
    "thursday_night": "Thursday night",
    "sunday_early": "Sunday early-window",
    "sunday_late": "Sunday late-afternoon",
    "sunday_night": "Sunday night",
    "monday_night": "Monday night",
    "saturday": "Saturday",
    "holiday": "Thanksgiving and Christmas",
    "international": "international neutral-site",
}
ERA_TAGS = ["sbr_2011_2019", "opener_2020_2025"]
ERA_LINES = [
    "SBR archive opening number (a proxy, not the Tuesday opener)",
    "Tuesday multi-book opener",
]
ERA_WORDS = [
    "2011 to 2019, using an archived opening number",
    "2020 to 2025, using the Tuesday opening line",
]


def build_cells(res):
    cells = []
    primary = [res["splits"][s]["pattern"]["corr_weighted"] for s in PRIMARY_SPLITS]
    closed = all(p["hi"] <= 0 for p in primary)
    ground = (
        "whole interval at or below zero in both primary splits; closure left to the orchestrator"
        if closed
        else "interval reaches above zero, so zero reliability is not established"
    )
    rel = res["splits"]["era"]["pattern"]["corr_weighted"]
    evidence = (
        f"Split-half reliability of the slot pattern across eras {rel['value']:+.2f} "
        f"[{rel['lo']:+.2f}, {rel['hi']:+.2f}], probability_positive {rel['ppos']:.2f}, "
        f"within-week permutation p {rel['perm_p']:.2f}: {ground}. Planted control (a "
        "Thursday-only miss added to label-shuffled data), share of draws the pattern "
        "correlation flags: "
        f"{power_text(res['splits']['era']['planted_thursday'], 'corr_weighted')}. No resolved "
        "wrong sign and no control bound at the size in question, so no admissible closing ground."
    )
    era = res["splits"]["era"]
    for row in era["slots"]:
        for h in (0, 1):
            half = row[f"half{h}"]
            lo, hi = (ERA_PROXY, ERA_OPENER)[h]
            words = SLOT_WORDS[row["slot"]]
            for stat, units, label, what, plain in (
                (
                    "mean",
                    "ats_points",
                    "mean_residual_points",
                    "mean final home margin minus the line in points",
                    f"Asks by how many points home teams in {words} games beat or missed the "
                    f"opening line on average, {ERA_WORDS[h]}.",
                ),
                (
                    "cover",
                    "rate_pp",
                    "home_cover_rate",
                    "home cover rate minus 50 in percentage points, pushes out",
                    f"Asks how often the home team covered the opening line in {words} games, "
                    f"{ERA_WORDS[h]}, compared with a coin flip.",
                ),
            ):
                cells.append(
                    make_cell(
                        f"mkt24_u2_era_{ERA_TAGS[h]}_{row['slot']}_{label}",
                        f"{row['slot']} slot, {lo}-{hi}, line = {ERA_LINES[h]}: {what}"
                        f"; interval is the wider of the season-block bootstrap and the "
                        f"game-level interval; home record at the line {half['cover_record']}"
                        f", exact binomial p {half['binom_p']:.3f}; sign "
                        f"{'agrees' if row['sign_agrees'] else 'disagrees'} with the other era.",
                        units,
                        conservative(half[stat]),
                        plain,
                        (lo, hi),
                        half["n"],
                        hi - lo + 1,
                        rel["value"],
                        evidence,
                    )
                )
    for name, sp in res["splits"].items():
        seasons = [s for half in sp["seasons"] for s in half]
        span = (min(seasons), max(seasons))
        halves = " vs ".join(sp["halves"])
        pat = sp["pattern"]["corr_weighted"]
        cells.append(
            make_cell(
                f"mkt24_u2_{name}_slot_pattern_split_half_correlation",
                f"Correlation of the 8 kickoff-slot mean residuals (final home margin minus the "
                f"line) across halves ({halves}), weighted by n1*n2/(n1+n2); unweighted "
                f"{sp['pattern']['corr_unweighted']['value']:+.2f}; permutation p (slot labels "
                f"shuffled inside season-week) {pat['perm_p']:.3f}; implied slot spread "
                f"{sp['implied_slot_sd_points']:+.2f} points.",
                "correlation",
                pat,
                "Asks whether kickoff slots where home teams beat the opening line in one half "
                "of the seasons did so again in the other half.",
                span,
                sum(sp["games"]),
                len(seasons),
                pat["value"],
                evidence,
            )
        )
        thu = sp["pattern"]["thu_concordance"]
        prod = sp["pattern"]["thu_product"]
        cells.append(
            make_cell(
                f"mkt24_u2_{name}_thursday_two_half_concordance",
                f"Thursday night alone ({halves}): concordance 2ab/(a^2+b^2) of the two half "
                f"mean residuals; cross-half product {prod['value']:+.2f} points squared "
                f"[{prod['lo']:+.2f}, {prod['hi']:+.2f}], probability same sign "
                f"{prod['ppos']:.2f}, permutation p {prod['perm_p']:.3f}. Planted control, share "
                f"of draws the product test flags a constant Thursday miss: "
                f"{power_text(sp['planted_thursday'], 'thu_product')}.",
                "correlation",
                thu,
                "Asks whether the Thursday night home-team result against the opening line in "
                "one half of the seasons shows up again in the other half.",
                span,
                sum(
                    r["half0"]["n"] + r["half1"]["n"] for r in sp["slots"] if r["slot"] == THURSDAY
                ),
                len(seasons),
                thu["value"],
                f"Two-half reliability of the Thursday residual {thu['value']:+.2f} "
                f"[{thu['lo']:+.2f}, {thu['hi']:+.2f}] cannot be told from zero or from one. "
                "Planted control: the product test flags a constant Thursday miss in this share "
                f"of draws: {power_text(sp['planted_thursday'], 'thu_product')}; the observed "
                f"product is not flagged (permutation p {prod['perm_p']:.2f}), which bounds only "
                "the larger misses on that grid. Smaller misses stay below power, so no "
                "admissible closing ground for the mechanism.",
            )
        )
    return cells


def reliability(splits):
    OUT.mkdir(parents=True, exist_ok=True)
    slots, _ = build_slots()
    d = residual_frame(slots)
    slot_names = sorted(d.slot.unique())
    k = len(slot_names)
    thu = slot_names.index(THURSDAY)
    slot_idx = d.slot.map({s: i for i, s in enumerate(slot_names)}).to_numpy()
    r = d.resid.to_numpy(float)
    rng = np.random.default_rng(SEED)
    seasons, tab = season_tables(d, slot_idx, k)
    perm = permute(d, slot_idx, k, {s: half_of(d.season.to_numpy(), s) for s in splits}, rng)
    proxy = d.season.between(*ERA_PROXY)
    live = d.resid.ne(0)
    pooled = []
    for j, slot in enumerate(slot_names):
        g = d[slot_idx == j]
        cover = float((g.resid[g.resid.ne(0)] > 0).mean())
        pooled.append(
            {
                "slot": slot,
                "n": len(g),
                "mean": float(g.resid.mean()),
                "home_covers": int((g.resid > 0).sum()),
                "home_fails": int((g.resid < 0).sum()),
                "binom_p": float(
                    binomtest(int((g.resid > 0).sum()), int(g.resid.ne(0).sum()), EVEN).pvalue
                ),
                "cover_pp": 100 * (cover - EVEN),
                "card_share": len(g) / len(d),
                "card_accuracy_points": 100 * abs(cover - EVEN) * len(g) / len(d),
            }
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        thu_season = tab["sm"][:, thu] / tab["n"][:, thu]
    played = int((tab["n"][:, thu] > 0).sum())
    positive = int((thu_season > 0).sum())
    res = {
        "family": FAMILY,
        "looks": UNIT2_LOOKS,
        "draws": DRAWS,
        "line_sources": {
            "2011_2019": "sbr_odds.parquet open_home_spread (SBR archive open, proxy)",
            "2020_2025": "opener_evaluation per_game margin_vs_open (Tuesday multi-book opener)",
            "proxy_games": int(proxy.sum()),
            "opener_games": int((~proxy).sum()),
            "proxy_open_ambiguous": int(d.open_ambiguous[proxy].fillna(False).astype(bool).sum()),
            "pushes": int((~live).sum()),
        },
        "residual_sd_points": float(d.resid.std(ddof=1)),
        "pooled_slots": pooled,
        "thursday_seasons": {
            "positive": positive,
            "seasons": played,
            "binom_p": float(binomtest(positive, played, EVEN).pvalue),
            "by_season": {int(s): float(v) for s, v in zip(seasons, thu_season, strict=True)},
        },
        "splits": {},
    }
    agree, agree_null = {}, {}
    for s in splits:
        res["splits"][s], agree[s], agree_null[s] = split_report(
            s, tab, seasons, perm[s], thu, slot_names, r, rng
        )
    if all(s in splits for s in PRIMARY_SPLITS):
        both = np.logical_and.reduce([agree[s] for s in PRIMARY_SPLITS])
        null = np.logical_and.reduce([agree_null[s] for s in PRIMARY_SPLITS]).sum(-1)
        res["both_primary_agree"] = {
            "slots": [slot_names[j] for j in np.where(both)[0]],
            "count": int(both.sum()),
            "null_mean_count": float(null.mean()),
            "perm_p": float((1 + (null >= both.sum()).sum()) / (1 + len(null))),
        }
        (OUT / "weak_signals_batch_u2.json").write_text(
            json.dumps({"cells": build_cells(res)}, indent=1)
        )
    (OUT / "reliability.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["line_sources"]), "sd", round(res["residual_sd_points"], 2))
    for p in pooled:
        print("pooled", json.dumps({k2: round(v, 3) if k2 != "slot" else v for k2, v in p.items()}))
    print("thursday_seasons", positive, played, round(res["thursday_seasons"]["binom_p"], 3))
    for s, sp in res["splits"].items():
        print("==", s, sp["halves"], sp["games"], [round(v, 3) for v in sp["all_games_mean"]])
        for row in sp["slots"]:
            parts = [f"{row['slot']:15s}"]
            for h in (0, 1):
                x = row[f"half{h}"]
                m, c = x["mean"], x["cover"]
                parts.append(
                    f"n{x['n']:5d} {m['value']:+6.2f} [{m['lo']:+6.2f},{m['hi']:+6.2f}] "
                    f"P+{m['ppos']:.2f} cov{c['value']:+6.1f} [{c['lo']:+6.1f},{c['hi']:+6.1f}] "
                    f"P+{c['ppos']:.2f} {x['cover_record']:>8s} p{x['binom_p']:.3f}"
                )
            parts.append(f"agree={row['sign_agrees']} prod_p={row['product_perm_p']:.3f}")
            print(" | ".join(parts))
        for key, v in sp["pattern"].items():
            print(" ", key, json.dumps({k2: round(x, 4) for k2, x in v.items()}))
        print("  implied_slot_sd_points", round(sp["implied_slot_sd_points"], 3))
        for row in sp["slots"]:
            for h in (0, 1):
                for key in ("mean", "cover"):
                    x = row[f"half{h}"][key]
                    if x["lo"] > 0 or x["hi"] < 0:
                        print(
                            "  block_excludes_zero",
                            row["slot"],
                            h,
                            key,
                            f"game [{x['game_lo']:+.2f},{x['game_hi']:+.2f}]",
                            f"P+{x['game_ppos']:.3f}",
                        )
        for row in sp["planted_thursday"]:
            print("  plant", json.dumps({k2: round(x, 3) for k2, x in row.items()}))
    print("both_primary_agree", json.dumps(res.get("both_primary_agree")))


def unit1():
    OUT.mkdir(parents=True, exist_ok=True)
    slots, cuts = build_slots()
    res = {"minute_cuts": cuts, "populations": {n: analyse(n, slots) for n in POPS}}
    (OUT / "report.json").write_text(json.dumps(res, indent=1, default=str))
    cols = ["slot", "n", "raw_resid", "raw_Ppos", "tau2", "shrunk_resid", "shrunk_lo", "shrunk_hi"]
    for n, r in res["populations"].items():
        print(n, r["n"], r["slot_counts"])
        sh = pd.DataFrame(r["shrink"])
        print(sh[[*cols, "shrunk_Ppos"]].round(4).to_string())
        print("arms", json.dumps(r["arms"]))
        print("flips", r["decisive_flips"])
        print(json.dumps(r["comparisons"]))
        print(r["insample"], r["chosen_C"])


def main():
    parser = argparse.ArgumentParser(
        description="MKT-24 kickoff-slot opener residual: unit 1 by default, unit 2 with --split"
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=sorted(SPLIT_LABELS),
        help="run the split-half reliability unit on these splits instead of unit 1",
    )
    args = parser.parse_args()
    if args.split:
        reliability(args.split)
    else:
        unit1()


if __name__ == "__main__":
    main()
