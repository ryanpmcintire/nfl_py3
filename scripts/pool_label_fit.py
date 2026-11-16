import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.four_overlay_composition import protection_flags_for_card
from nfl_ats.pick_probability import (
    active_market_move_feature_version,
    signed_composition_flags_fail_open,
)
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    _design,
    _fit_logit,
    _fit_offset_coefficient,
    _natural_coefficients,
    _stage_one_logit,
    _standardisers,
    build_fit_population,
)
from nfl_ats.pool_hook import (
    HOOK_COLUMN,
    crossed_push_mass,
    hook_rows,
    hook_value,
    push_mass_table,
    tuesday_market_lines,
    whole_mask,
)
from nfl_ats.snapshots import latest_snapshot, load_verified_snapshot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "artifacts" / "pool_label_fit"
SCRATCH = ROOT / "tests" / "scratch"
ACTIVE = json.loads((ROOT / "artifacts" / "active_pick_probability.json").read_text())
ART = ROOT / "artifacts" / ACTIVE["artifact"]
POST_WINDOW_HOURS = 1.0
PRE_WINDOW_HOURS = 4.0
GRADE = (2023, 2025)
REPS = 4000
SEED = 20261116
EPS = 1e-6
H = "h"
L_FEATURES = list(FIT_FEATURES) + [H]


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def wfit(frame, features, y, w):
    mu = {f: float(np.average(frame[f], weights=w)) for f in features}
    sd = {
        f: float(np.sqrt(np.average((frame[f] - mu[f]) ** 2, weights=w))) or 1.0 for f in features
    }
    x = np.column_stack(
        [np.ones(len(frame))] + [((frame[f] - mu[f]) / sd[f]).to_numpy(float) for f in features]
    )
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        p = sig(x @ beta)
        g = x.T @ (w * (y - p)) - FIT_RIDGE * beta
        hs = (x.T * (w * np.clip(p * (1 - p), 1e-6, None))) @ x + FIT_RIDGE * np.eye(x.shape[1])
        step = np.linalg.solve(hs, g)
        beta = beta + step
        if np.abs(step).max() < 1e-12:
            break
    nat = {f: float(beta[i + 1] / sd[f]) for i, f in enumerate(features)}
    nat["intercept"] = float(
        beta[0] - sum(beta[i + 1] * mu[f] / sd[f] for i, f in enumerate(features))
    )
    return nat


def wlogit(frame, nat, features):
    return nat["intercept"] + sum(nat[f] * frame[f].to_numpy(float) for f in features)


def metrics(p, y, w):
    p = np.clip(p, EPS, 1 - EPS)
    ok = (p >= 0.5) == (y == 1)
    W = w.sum()
    return dict(
        games=float(W),
        wins=float((w * ok).sum()),
        losses=float(W - (w * ok).sum()),
        acc=float((w * ok).sum() / W),
        ll=float((w * -(y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / W),
        brier=float((w * (p - y) ** 2).sum() / W),
    )


def paired(pa, pb, y, w, blocks, rng):
    pa = np.clip(pa, EPS, 1 - EPS)
    pb = np.clip(pb, EPS, 1 - EPS)
    a = w * (((pa >= 0.5) == (y == 1)).astype(float) - ((pb >= 0.5) == (y == 1)).astype(float))
    l_ = w * (
        (-(y * np.log(pb) + (1 - y) * np.log(1 - pb)))
        - (-(y * np.log(pa) + (1 - y) * np.log(1 - pa)))
    )
    b = w * ((pb - y) ** 2 - (pa - y) ** 2)
    codes, uniq = pd.factorize(pd.Series(blocks))
    st = np.zeros((len(uniq), 4))
    np.add.at(st, codes, np.column_stack([a, l_, b, w]))
    pt = st[:, :3].sum(0) / st[:, 3].sum()
    draws = []
    for _ in range(REPS):
        ix = rng.integers(0, len(uniq), len(uniq))
        s = st[ix].sum(0)
        draws.append(s[:3] / s[3])
    draws = np.array(draws)
    scale = np.array([100.0, 1.0, 1.0])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return dict(
        blocks=len(uniq),
        effect=(pt * scale).tolist(),
        low=(lo * scale).tolist(),
        high=(hi * scale).tolist(),
        pp=(draws > 0).mean(0).tolist(),
    )


def fd_quotes():
    arch = pd.read_pickle(SCRATCH / "mod57" / "u1_arch.pkl")
    pre = pd.read_pickle(SCRATCH / "mod57" / "u1_pre.pkl")
    post = (
        arch.loc[
            (arch.season >= 2023)
            & (arch.bookmaker_key == "fanduel")
            & (arch.lag <= POST_WINDOW_HOURS)
        ]
        .sort_values("t")
        .groupby("nflverse_game_id")
        .head(1)
    )
    prior = (
        pre.loc[
            (pre.season <= 2022) & (pre.bookmaker_key == "fanduel") & (pre.lag >= -PRE_WINDOW_HOURS)
        ]
        .sort_values("t")
        .groupby("nflverse_game_id")
        .tail(1)
    )
    q = pd.concat([post, prior])
    return (
        q.set_index("nflverse_game_id")
        .home_spread_line.rename("fd")
        .rename_axis("game_id")
        .reset_index()
    )


def pool_rows(frame):
    d = frame.loc[frame.fd.notna()].reset_index(drop=True)
    line = d.fd.to_numpy(float)
    whole = whole_mask(d.fd)
    up = d.assign(pool=np.where(whole, line + 0.5, line), w=np.where(whole, 0.5, 1.0))
    dn = d.loc[whole].assign(pool=line[whole] - 0.5, w=0.5)
    s = pd.concat([up, dn], ignore_index=True)
    s = s.loc[s.margin != s.pool].reset_index(drop=True)
    s["y"] = (s.margin > s.pool).astype(float)
    return s


def hook_for(anchor, pool, table):
    mass = np.array([crossed_push_mass(float(a), float(p), table) for a, p in zip(anchor, pool)])
    return hook_value(np.where(mass > 0, np.sign(pool - anchor), 0.0), mass)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {}
    ver = active_market_move_feature_version(ROOT / "artifacts")
    er, _ = build_fit_population(
        ROOT / "artifacts", ROOT / "data", market_move_feature_version=ver, keep_pushes=True
    )
    er = er.reset_index(drop=True)
    er["margin"] = er.margin_vs_open + er.tue_open_home_spread
    pop = er.loc[er.margin_vs_open.ne(0.0)]
    seasons = sorted(int(s) for s in pop.season.unique())
    stacked = hook_rows(er)
    sframe = er.iloc[stacked["row"].to_numpy()].reset_index(drop=True)
    hv = stacked[HOOK_COLUMN].to_numpy(float)
    ht = stacked["home_covered"].to_numpy(float)
    folds = {}
    for held in seasons + [-1]:
        tr = pop.loc[pop.season.ne(held)]
        m, s = _standardisers(tr)
        beta = _fit_logit(_design(tr, m, s), tr.home_covered.astype(float).to_numpy(), FIT_RIDGE)
        rest = stacked.season.ne(held).to_numpy()
        c = _fit_offset_coefficient(
            hv[rest], _stage_one_logit(sframe.loc[rest], beta, m, s), ht[rest]
        )
        nat = _natural_coefficients(beta, m, s)
        folds[held] = dict(nat=nat, hook=float(c), logit=_stage_one_logit(er, beta, m, s))
    pg = pd.read_parquet(ART / "per_game.parquet")
    meta = json.loads((ART / "metadata.json").read_text())
    coefs = json.loads((ART / "coefficients.json").read_text())
    oos = np.full(len(er), np.nan)
    for held in seasons:
        te = er.season.eq(held).to_numpy()
        oos[te] = sig(folds[held]["logit"][te])
    chk = er.assign(mine=oos).merge(pg[["game_id", "out_of_season_home_probability"]], on="game_id")
    art_folds = meta.get("pool_hook_stage", {}).get("fold_coefficients", {})
    res["check"] = dict(
        population=len(er),
        pushes=int(er.margin_vs_open.eq(0).sum()),
        stage1_oos_max_abs_diff_vs_artifact=float(
            (chk.mine - chk.out_of_season_home_probability).abs().max()
        ),
        shared=len(chk),
        hook_full_mine=folds[-1]["hook"],
        hook_full_artifact=coefs["coefficients"]["pool_hook"],
        hook_folds_mine={str(k): v["hook"] for k, v in folds.items() if k != -1},
        hook_folds_artifact=art_folds,
        full_mine=folds[-1]["nat"],
        full_artifact=coefs["coefficients"],
    )
    print(json.dumps(res["check"], indent=1, default=float))
    er = er.merge(fd_quotes(), on="game_id", how="left")
    res["fd_coverage"] = (
        er.groupby("season").fd.apply(lambda v: f"{int(v.notna().sum())}/{len(v)}").to_dict()
    )
    print("fd coverage", res["fd_coverage"])
    rows = pool_rows(er)
    idx = pd.Series(np.arange(len(er)), index=er.game_id)
    gi = idx.loc[rows.game_id].to_numpy()
    tables = {h: push_mass_table(er.loc[er.season.ne(h)]) for h in seasons}
    tables[-1] = push_mass_table(er)
    S = rows.season.to_numpy()
    y = rows.y.to_numpy()
    w = rows.w.to_numpy()
    pS = np.full(len(rows), np.nan)
    pL = np.full(len(rows), np.nan)
    pL4 = np.full(len(rows), np.nan)
    fold_out = {}
    for held in seasons:
        rows[H] = hook_for(
            rows.tue_open_home_spread.to_numpy(float), rows.pool.to_numpy(float), tables[held]
        )
        te = held == S
        tr = ~te
        pS[te] = sig(folds[held]["logit"][gi[te]] + folds[held]["hook"] * rows[H].to_numpy()[te])
        nL = wfit(rows.loc[tr], L_FEATURES, y[tr], w[tr])
        nL4 = wfit(rows.loc[tr], list(FIT_FEATURES), y[tr], w[tr])
        pL[te] = sig(wlogit(rows.loc[te], nL, L_FEATURES))
        pL4[te] = sig(wlogit(rows.loc[te], nL4, list(FIT_FEATURES)))
        fold_out[held] = dict(L=nL, L4=nL4, S=dict(folds[held]["nat"], hook=folds[held]["hook"]))
    rows[H] = hook_for(
        rows.tue_open_home_spread.to_numpy(float), rows.pool.to_numpy(float), tables[-1]
    )
    fullL = wfit(rows, L_FEATURES, y, w)
    fullL4 = wfit(rows, list(FIT_FEATURES), y, w)
    inS = sig(folds[-1]["logit"][gi] + folds[-1]["hook"] * rows[H].to_numpy())
    inL = sig(wlogit(rows, fullL, L_FEATURES))
    inL4 = sig(wlogit(rows, fullL4, list(FIT_FEATURES)))
    res["fold_coefficients"] = {str(k): v for k, v in fold_out.items()}
    res["full_coefficients"] = dict(
        L=fullL, L4=fullL4, S=dict(folds[-1]["nat"], hook=folds[-1]["hook"])
    )
    g = (GRADE[0] <= S) & (GRADE[1] >= S)
    hist = {}
    for name, p, pin in (("S", pS, inS), ("L", pL, inL), ("L4", pL4, inL4)):
        mo = metrics(p[g], y[g], w[g])
        mi = metrics(pin[g], y[g], w[g])
        hist[name] = dict(
            oos=mo,
            ins=mi,
            gap_acc_pts=100 * (mi["acc"] - mo["acc"]),
            gap_ll=mo["ll"] - mi["ll"],
            by_season={
                int(s): metrics(p[s == S], y[s == S], w[s == S])
                for s in range(GRADE[0], GRADE[1] + 1)
            },
        )
    wk = list(zip(rows.season[g], rows.week[g]))
    for arm, p in (("L", pL), ("L4", pL4)):
        dec = g & ((p >= 0.5) != (pS >= 0.5))
        hist[arm]["decisive"] = dict(
            games=float(w[dec].sum()),
            arm_right=float((w * ((p >= 0.5) == (y == 1)))[dec].sum()),
            served_right=float((w * ((pS >= 0.5) == (y == 1)))[dec].sum()),
            whole_fd_rows=float(w[dec & (rows.w.to_numpy() < 1)].sum()),
        )
        hist[arm]["week_block"] = paired(p[g], pS[g], y[g], w[g], wk, rng)
        hist[arm]["season_block"] = paired(p[g], pS[g], y[g], w[g], list(rows.season[g]), rng)
    res["history_2023_25"] = hist
    res["history_rows"] = dict(
        rows=int(g.sum()),
        weighted_games=float(w[g].sum()),
        whole_fd_games=float(w[g & (rows.w.to_numpy() < 1)].sum()),
        nonzero_h_share=float(w[g & (rows[H].to_numpy() != 0)].sum() / w[g].sum()),
    )
    print(json.dumps(dict(rows=res["history_rows"], hist=hist), indent=1, default=float))

    import pool_relative_move_forward_log as fwd

    old = pd.read_csv(SCRATCH / "fixpicks" / "root_m2_games.csv")
    cols = [
        "game_id",
        "week",
        "pool_home_spread",
        "raw_model_home_prob_final",
        "served_move",
        "served_move_available",
    ]
    old = old[cols]
    w10 = fwd.build(10, True)
    base = {}
    for feed in w10["feed"].unique():
        path = ROOT / "artifacts" / "waterfall_feed" / str(feed) / "feed.json"
        if not path.is_file():
            path = Path(str(feed)) / "feed.json"
        for game in json.loads(path.read_text())["games"]:
            base[(str(feed), game["game_id"])] = game["discrete_base_home_cover_probability"]
    w10["raw_model_home_prob_final"] = [
        base.get((str(f), gid)) for f, gid in zip(w10["feed"], w10["game_id"])
    ]
    d = pd.concat([old, w10[cols]], ignore_index=True).drop_duplicates("game_id", keep="last")
    schedules, _ = load_verified_snapshot(latest_snapshot(ROOT / "data" / "raw"))
    sc = schedules.loc[schedules.game_id.isin(d.game_id)]
    pred = sc[["game_id", "season", "week", "gameday", "home_team", "away_team"]].copy()
    flags = signed_composition_flags_fail_open(
        pred, schedules, protection_back_side=protection_flags_for_card(pred, ROOT / "data")
    )
    counted = coefs["counted_flag_columns"]
    d = d.merge(flags[["game_id"] + counted], on="game_id", how="left").merge(
        sc[["game_id", "result"]], on="game_id", how="left"
    )
    anchor = {}
    for week, grp in d.groupby("week"):
        anchor.update(tuesday_market_lines(ROOT / "data", 2026, int(week), grp.game_id.tolist()))
    d["anchor"] = d.game_id.map(anchor)
    names = list(FIT_FEATURES)
    d[names[0]] = lg(d.raw_model_home_prob_final.astype(float).to_numpy())
    d[names[1]] = d[counted].fillna(0).sum(axis=1).astype(float)
    d[names[2]] = d.served_move.fillna(0.0).astype(float)
    d[names[3]] = d.served_move_available.fillna(0.0).astype(float)
    served_table = {float(a): float(b) for a, b in coefs["hook_push_mass"]}
    has = d.anchor.notna().to_numpy()
    hh = np.zeros(len(d))
    hh[has] = hook_for(
        d.anchor.to_numpy(float)[has], d.pool_home_spread.to_numpy(float)[has], served_table
    )
    d[H] = hh
    art = coefs["coefficients"]
    zS = (
        art["intercept"]
        + art["model_logit"] * d[names[0]]
        + art["flag_sum"] * d[names[1]]
        + art["move_toward_home"] * d[names[2]]
        + art["move_available"] * d[names[3]]
        + art["pool_hook"] * d[H]
    )
    d["pS"] = sig(zS.to_numpy(float))
    d["pL"] = sig(wlogit(d, fullL, L_FEATURES))
    d["pL4"] = sig(wlogit(d, fullL4, list(FIT_FEATURES)))
    d["margin"] = d.result - d.pool_home_spread
    ok = (d.margin.notna() & d.margin.ne(0)).to_numpy()
    yy = (d.margin > 0).to_numpy(float)
    ones = np.ones(len(d))
    fwd_res = dict(
        games=int(ok.sum()),
        missing_anchor=int((~has).sum()),
        nonzero_h=int((hh != 0).sum()),
        missing_model=int(d.raw_model_home_prob_final.isna().sum()),
    )
    for name in ("pS", "pL", "pL4"):
        p = d[name].to_numpy()
        fwd_res[name] = dict(
            metrics(p[ok], yy[ok], ones[ok]),
            by_week=d[ok]
            .assign(r=((p >= 0.5) == (yy == 1))[ok])
            .groupby("week")
            .r.agg(lambda v: f"{int(v.sum())}-{int((~v).sum())}")
            .to_dict(),
        )
    for name in ("pL", "pL4"):
        p = d[name].to_numpy()
        ch = (p >= 0.5) != (d.pS.to_numpy() >= 0.5)
        fwd_res[name]["changed"] = dict(
            all=int(ch.sum()),
            settled=int((ch & ok).sum()),
            arm_right=int((((p >= 0.5) == (yy == 1)) & ch & ok).sum()),
            games=d.loc[ch, ["week", "game_id", "pool_home_spread", "anchor", "result"]]
            .assign(pS=d.pS[ch].round(3), parm=p[ch].round(3))
            .to_dict("records"),
        )
        fwd_res[name]["week_block"] = paired(
            p[ok], d.pS.to_numpy()[ok], yy[ok], ones[ok], list(d.week[ok]), rng
        )
    res["forward_2026"] = fwd_res
    d.to_parquet(OUT / "forward_2026.parquet", index=False)
    rows.assign(pS=pS, pL=pL, pL4=pL4).to_parquet(OUT / "history_rows.parquet", index=False)
    print(json.dumps(fwd_res, indent=1, default=float))
    (OUT / "result.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
