import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import mod25_generator as gen  # noqa: E402
import mod25e_e96 as e96  # noqa: E402

E3 = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokgndecsmfwtjo2as2yp"
SEEDS = (11, 12, 13)
BURN = 2
REAL_SEASONS = tuple(range(2011, 2018))
PAT_SEASONS = tuple(range(2009, 2020))
TRAIN = tuple(range(2009, 2018))
RULE_YEAR = 2015
NB = 300
OUT = []
LOOKS = [0]
rng = np.random.default_rng(98)
SCRIM = ("run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike")
COLS = ["game_id", "season_type", "play_id", "qtr", "posteam", "down", "ydstogo", "yardline_100", "play_type", "play_type_nfl", "fixed_drive", "score_differential", "posteam_score", "posteam_score_post", "touchdown"]
BANDS = [(1, 10), (11, 20), (21, 30), (31, 40)]
DIST = [(0, 19), (20, 29), (30, 39), (40, 49), (50, 70)]
STATES = e96.STATES


def say(s=""):
    OUT.append(s)
    print(s)


def fmt(a, d=4):
    return f"{a[0]:.{d}f} [{np.quantile(a[1:], .025):.{d}f},{np.quantile(a[1:], .975):.{d}f}]"


def fmtd(a, d=4):
    return f"{a[0]:+.{d}f} [{np.quantile(a[1:], .025):+.{d}f},{np.quantile(a[1:], .975):+.{d}f}]"


def stat_line(label, r, s, d=4):
    diff = s - r
    LOOKS[0] += 1
    return f"{label:46s} real {fmt(r, d)} sim {fmt(s, d)} s-r {fmtd(diff, d)} P(s-r>0) {np.mean(diff[1:] > 0):.2f}"


def boot_w(G):
    return np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])


def per_game(W, v, G):
    return (W @ v) / G


def ratio(W, a, b):
    return (W @ a) / np.maximum(W @ b, 1e-12)


def real_plays(seasons):
    rows = []
    for s in seasons:
        p = pd.read_parquet(gen.PBP_DIR / f"season={s}" / "plays.parquet", columns=COLS)
        p = p[p.season_type == "REG"].copy()
        p["season"] = s
        rows.append(p)
    p = pd.concat(rows, ignore_index=True).sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
    p["d"] = (p.posteam_score_post - p.posteam_score).fillna(0.0)
    return p


def pat_by_season(p):
    out = []
    x = p[p.posteam.notna()]
    for s in PAT_SEASONS:
        q = x[x.season == s]
        kick = q[(q.play_type == "extra_point") & (q.play_type_nfl == "XP_KICK")]
        two = q[q.play_type_nfl == "PAT2"]
        td = q[(q.touchdown == 1) & q.play_type.isin(["run", "pass"]) & (q.d >= 6) & (q.qtr <= 4)]
        out.append({"season": s, "td": len(td), "kick": len(kick), "kick_made": int((kick.d == 1).sum()), "two": len(two), "two_made": int((two.d == 2).sum())})
    return pd.DataFrame(out)


def loso_kick(df, train):
    tr = df[df.season.isin(train)].reset_index(drop=True)
    cur_rate = tr[tr.season >= RULE_YEAR].kick_made.sum() / tr[tr.season >= RULE_YEAR].kick.sum()
    rows = []
    for s in train:
        o = tr[tr.season != s]
        reg = o[(o.season >= RULE_YEAR) == (s >= RULE_YEAR)]
        pr = reg.kick_made.sum() / reg.kick.sum()
        pool = o.kick_made.sum() / o.kick.sum()
        h = tr[tr.season == s].iloc[0]
        miss = h.kick - h.kick_made

        def ll(q):
            return -(h.kick_made * np.log(q) + miss * np.log(1 - q)) / h.kick

        rows.append((s, h.kick_made / h.kick, pr, pool, cur_rate, ll(pr), ll(pool), ll(cur_rate)))
    return pd.DataFrame(rows, columns=["season", "obs", "regime_loso", "pooled_loso", "served_2015plus", "ll_regime", "ll_pooled", "ll_served"])


def sim_play_files():
    for sd in SEEDS:
        sdir = E3 / f"e5_{LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            yield sd, w, s, f


def drive_table_sim():
    parts = []
    gk = {}
    for sd, w, s, f in sim_play_files():
        d = pd.read_parquet(f, columns=["g", "down", "yl", "sd", "qtr", "code", "po", "flip"])
        d = d.sort_values("g", kind="stable").reset_index(drop=True)
        d = d[d.qtr <= 4].reset_index(drop=True)
        key = f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str)
        for k in key.unique():
            gk.setdefault(k, len(gk))
        gid = key.map(gk).to_numpy()
        fl = d.flip.to_numpy()
        newd = np.r_[True, (gid[1:] != gid[:-1]) | (fl[:-1] == 1)]
        did = np.cumsum(newd)
        d["gi"] = gid
        d["did"] = did + (sd * 100 + w * 10 + s) * 10**7
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d[d.code != 6]
    d["fga"] = (d.code == 3)
    d["fgm"] = (d.code == 3) & (d.po == 3)
    d["td"] = (d.po >= 6) & (d.code != 3)
    return d, len(gk)


def drive_table_real(p):
    x = p[p.play_type.isin(SCRIM) & p.posteam.notna() & p.yardline_100.notna() & (p.qtr <= 4)].copy()
    gk = {g: i for i, g in enumerate(x.game_id.unique())}
    x["gi"] = x.game_id.map(gk)
    x["did"] = x.groupby(["game_id", "fixed_drive"], sort=False).ngroup()
    x["fga"] = x.play_type == "field_goal"
    x["fgm"] = x.fga & (x.d == 3)
    x["td"] = (x.touchdown == 1) & (x.d >= 6) & x.play_type.isin(["run", "pass"])
    x = x.rename(columns={"yardline_100": "yl", "score_differential": "sd"})
    x["sd"] = x.sd.fillna(0.0)
    return x, len(gk)


def drives(d):
    d = d.sort_values(["did"], kind="stable")
    g = d.groupby("did", sort=False)
    first = g.first()
    out = pd.DataFrame({"gi": first.gi, "sd0": first.sd})
    out["minyl"] = g.yl.min()
    out["td"] = g.td.any()
    out["fga"] = g.fga.any()
    out["fgm"] = g.fgm.any()
    fg = d[d.fga]
    fgrow = fg.groupby("did", sort=False).first()
    out["fgyl"] = fgrow.yl.reindex(out.index)
    out["fgdown"] = fgrow.down.reindex(out.index)
    out["fgsd"] = fgrow.sd.reindex(out.index)
    out["fgq"] = fgrow.qtr.reindex(out.index)
    return out.reset_index(drop=True)


def cells_for(dr, G):
    gi = dr.gi.to_numpy()
    reach = (dr.minyl <= 40).to_numpy()
    td = dr.td.to_numpy()
    fga = dr.fga.to_numpy()
    fgm = dr.fgm.to_numpy()

    def c(m):
        return np.bincount(gi[m], minlength=G).astype(float)

    C = {}
    C["reach"] = c(reach)
    C["td_reach"] = c(reach & td)
    C["fga_reach"] = c(reach & fga)
    C["fgm_reach"] = c(reach & fgm)
    C["fgm_all"] = c(fgm)
    C["fga_all"] = c(fga)
    C["td_all"] = c(td)
    for lo, hi in BANDS:
        b = reach & (dr.minyl.to_numpy() >= lo - 1e-9) & (dr.minyl.to_numpy() <= hi + 1e-9)
        if lo == 1:
            b = reach & (dr.minyl.to_numpy() <= hi + 1e-9)
        if lo > 1:
            b = reach & (dr.minyl.to_numpy() > lo - 1) & (dr.minyl.to_numpy() <= hi)
        C[f"b{lo}-{hi} reach"] = c(b)
        C[f"b{lo}-{hi} td"] = c(b & td)
        C[f"b{lo}-{hi} fga"] = c(b & fga)
        C[f"b{lo}-{hi} fgm"] = c(b & fgm)
    fd = dr.fgdown.to_numpy()
    for nm, m in (("4th", fd == 4), ("1-3rd", fd < 4)):
        C[f"down {nm} fga"] = c(fga & m)
        C[f"down {nm} fgm"] = c(fgm & m)
    fy = dr.fgyl.to_numpy()
    for lo, hi in DIST:
        m = fga & (fy + 17 >= lo) & (fy + 17 <= hi)
        C[f"dist {lo}-{hi} fga"] = c(m)
        C[f"dist {lo}-{hi} fgm"] = c(m & fgm)
    sd0 = dr.sd0.to_numpy()
    for nm, lo, hi in STATES:
        m = reach & (sd0 >= lo) & (sd0 <= hi)
        C[f"st {nm} reach"] = c(m)
        C[f"st {nm} fga"] = c(m & fga)
        C[f"st {nm} td"] = c(m & td)
    rz = reach & (dr.minyl.to_numpy() <= 20)
    C["rz reach"] = c(rz)
    C["rz td"] = c(rz & td)
    C["rz fga"] = c(rz & fga)
    C["rz fgm"] = c(rz & fgm)
    return C


def main():
    p = real_plays(PAT_SEASONS)
    say(f"E98 regulation kick mechanisms; label {LABEL} seeds {SEEDS} burn {BURN}; real REG {REAL_SEASONS[0]}-{REAL_SEASONS[-1]} (PAT history {PAT_SEASONS[0]}-{PAT_SEASONS[-1]}); game bootstrap {NB}; regulation (qtr<=4) only in section 2")
    say("")
    say("1. PAT")
    say("1a. engine path (read): scripts/mod25_mechanisms.py:299 fit_pat fits P(2pt) on (score diff+6, clock) pooled over TRAIN years, P(2pt good) pooled, P(kick good) from season>=2015 rows only (:322-324); scripts/mod25_mechanisms.py:500-511 overrides drawn TD points in regulation; OTY (mod25e_ot.py) touches only OT rules, never the try draw")
    pt = pat_by_season(p)
    say("1b. real tries by season: td, kick att, kick made rate, 2pt per TD, 2pt made rate")
    for r in pt.itertuples():
        say(f"  {r.season} td {r.td:4d} kick {r.kick:4d} made {r.kick_made / r.kick:.4f} two/td {r.two / r.td:.4f} two made {r.two_made / max(r.two, 1):.3f}")
    import mod25_mechanisms as m25

    pbp_tr = __import__("sim04_engine").load_reg_seasons(TRAIN)
    pg, s1, s2, n_td_fit, y_mean = m25.fit_pat(pbp_tr)
    say(f"1c. served fit (TRAIN {TRAIN[0]}-{TRAIN[-1]}) [measured]: P(kick good) {s1:.4f} P(2pt good) {s2:.4f} pooled 2pt share of tries {y_mean:.4f} n {n_td_fit}")
    ktr = pt[pt.season.isin(REAL_SEASONS)]
    say(f"1d. real gate years kick made {ktr.kick_made.sum() / ktr.kick.sum():.4f} (2011-14 {pt[pt.season.between(2011, 2014)].kick_made.sum() / pt[pt.season.between(2011, 2014)].kick.sum():.4f}, 2015-17 {pt[pt.season.between(2015, 2017)].kick_made.sum() / pt[pt.season.between(2015, 2017)].kick.sum():.4f}); served rate applied to every year {s1:.4f}")
    lo = loso_kick(pt, TRAIN)
    say("1e. leave-one-season-out kick-made forms on TRAIN (obs, regime-mean of other seasons in same rule regime, pooled other seasons, served 2015+ rate in-sample), log loss per kick")
    for r in lo.itertuples():
        say(f"  {r.season} obs {r.obs:.4f} regime {r.regime_loso:.4f} pooled {r.pooled_loso:.4f} served {r.served_2015plus:.4f} ll regime {r.ll_regime:.5f} pooled {r.ll_pooled:.5f} served {r.ll_served:.5f}")
    say(f"  mean ll regime {lo.ll_regime.mean():.5f} pooled {lo.ll_pooled.mean():.5f} served {lo.ll_served.mean():.5f}; pre-2015 seasons only: regime {lo[lo.season < RULE_YEAR].ll_regime.mean():.5f} served {lo[lo.season < RULE_YEAR].ll_served.mean():.5f}")
    say(f"  gap out-of-sample minus in-sample (2015+ seasons, regime form vs served): {lo[lo.season >= RULE_YEAR].ll_regime.mean() - lo[lo.season >= RULE_YEAR].ll_served.mean():+.5f}")
    LOOKS[0] += 3

    evr, gr = e96.real_events()
    evs, gs = e96.sim_events()
    evr, gr, _ = e96.build(e96.classify(evr), gr)
    evs, gs, _ = e96.build(e96.classify(evs), gs)
    Fr, _, _ = e96.features(evr, gr)
    Fs, _, _ = e96.features(evs, gs)
    Wr, Ws = boot_w(len(gr)), boot_w(len(gs))
    say("1f. score-based try outcome shares per TD (game bootstrap), real v sim")
    for nm, a in (("kick good (td7)", "n_td7"), ("2pt good (td8)", "n_td8"), ("no point after (td6)", "n_td6")):
        say(stat_line(nm, ratio(Wr, Fr[a], Fr["n_td"]), ratio(Ws, Fs[a], Fs["n_td"])))
    td_r, td_s = Wr @ Fr["n_td"] / len(gr), Ws @ Fs["n_td"] / len(gs)
    say(stat_line("TD per game", td_r, td_s))
    say("1g. counterfactual td6 share if served kick rate followed the sim's season year (years 2011-2016 equally) [inferred from 1b and 1c]")
    yrs = range(2011, 2017)
    mk = {r.season: r.kick_made / r.kick for r in pt.itertuples()}
    p2 = float(np.mean(Fs["n_td8"]) / np.mean(Fs["n_td"])) / s2
    td6_served = (1 - p2) * (1 - s1) + p2 * (1 - s2)
    td6_year = float(np.mean([(1 - p2) * (1 - mk[y]) + p2 * (1 - s2) for y in yrs]))
    say(f"  implied 2pt attempt share {p2:.4f}; td6 under served rate {td6_served:.4f} (sim observed {np.mean(Fs['n_td6']) / np.mean(Fs['n_td']):.4f}); td6 under per-year real kick rate {td6_year:.4f}; real gate td6 {np.mean(Fr['n_td6']) / np.mean(Fr['n_td']):.4f}")
    LOOKS[0] += 1
    say("")

    say("2. FG attempts and red-zone outcomes (regulation, drive level; reach = drive's minimum yardline_100 <= 40)")
    dr_r, Gr_ = drive_table_real(p[p.season.isin(REAL_SEASONS)])
    dr_s, Gs_ = drive_table_sim()
    Dr, Ds = drives(dr_r), drives(dr_s)
    Cr, Cs = cells_for(Dr, Gr_), cells_for(Ds, Gs_)
    Wr, Ws = boot_w(Gr_), boot_w(Gs_)
    say(f"games real {Gr_} sim {Gs_}; drives real {len(Dr)} sim {len(Ds)}")
    for nm, k in (("drives reaching yl<=40 per game", "reach"), ("TD per game on those drives", "td_reach"), ("FG attempts per game on those drives", "fga_reach"), ("FG made per game on those drives", "fgm_reach"), ("FG attempts per game all drives", "fga_all"), ("FG made per game all drives", "fgm_all"), ("TD per game all drives", "td_all")):
        say(stat_line(nm, per_game(Wr, Cr[k], Gr_), per_game(Ws, Cs[k], Gs_)))
    say(stat_line("P(FG attempt | reach)", ratio(Wr, Cr["fga_reach"], Cr["reach"]), ratio(Ws, Cs["fga_reach"], Cs["reach"])))
    say(stat_line("P(TD | reach)", ratio(Wr, Cr["td_reach"], Cr["reach"]), ratio(Ws, Cs["td_reach"], Cs["reach"])))
    say(stat_line("FG make rate | attempt (all)", ratio(Wr, Cr["fgm_all"], Cr["fga_all"]), ratio(Ws, Cs["fgm_all"], Cs["fga_all"])))
    say(stat_line("red zone (yl<=20) reach per game", per_game(Wr, Cr["rz reach"], Gr_), per_game(Ws, Cs["rz reach"], Gs_)))
    say(stat_line("red zone TD | reach", ratio(Wr, Cr["rz td"], Cr["rz reach"]), ratio(Ws, Cs["rz td"], Cs["rz reach"])))
    say(stat_line("red zone FG attempt | reach", ratio(Wr, Cr["rz fga"], Cr["rz reach"]), ratio(Ws, Cs["rz fga"], Cs["rz reach"])))
    say("2b. by deepest yardline reached")
    for lo, hi in BANDS:
        b = f"b{lo}-{hi}"
        say(stat_line(f"{b} reach/g", per_game(Wr, Cr[b + " reach"], Gr_), per_game(Ws, Cs[b + " reach"], Gs_)))
        say(stat_line(f"{b} P(TD|reach)", ratio(Wr, Cr[b + " td"], Cr[b + " reach"]), ratio(Ws, Cs[b + " td"], Cs[b + " reach"])))
        say(stat_line(f"{b} P(FGatt|reach)", ratio(Wr, Cr[b + " fga"], Cr[b + " reach"]), ratio(Ws, Cs[b + " fga"], Cs[b + " reach"])))
        say(stat_line(f"{b} FG made/g", per_game(Wr, Cr[b + " fgm"], Gr_), per_game(Ws, Cs[b + " fgm"], Gs_)))
    say("2c. FG attempts by down and by kick distance (yardline_100+17)")
    for nm in ("4th", "1-3rd"):
        say(stat_line(f"down {nm} FG att/g", per_game(Wr, Cr[f"down {nm} fga"], Gr_), per_game(Ws, Cs[f"down {nm} fga"], Gs_)))
    for lo, hi in DIST:
        k = f"dist {lo}-{hi}"
        say(stat_line(f"{k} att/g", per_game(Wr, Cr[k + " fga"], Gr_), per_game(Ws, Cs[k + " fga"], Gs_)))
        say(stat_line(f"{k} make rate", ratio(Wr, Cr[k + " fgm"], Cr[k + " fga"]), ratio(Ws, Cs[k + " fgm"], Cs[k + " fga"])))
    say("2d. by score state at drive start (drives reaching yl<=40)")
    for nm, _, _ in STATES:
        say(stat_line(f"{nm} reach/g", per_game(Wr, Cr[f"st {nm} reach"], Gr_), per_game(Ws, Cs[f"st {nm} reach"], Gs_)))
        say(stat_line(f"{nm} P(FGatt|reach)", ratio(Wr, Cr[f"st {nm} fga"], Cr[f"st {nm} reach"]), ratio(Ws, Cs[f"st {nm} fga"], Cs[f"st {nm} reach"])))
        say(stat_line(f"{nm} P(TD|reach)", ratio(Wr, Cr[f"st {nm} td"], Cr[f"st {nm} reach"]), ratio(Ws, Cs[f"st {nm} td"], Cs[f"st {nm} reach"])))

    say("")
    say("3. fourth-down FG decision by season and FG make form")
    pr = p[p.season.isin(TRAIN) & p.posteam.notna() & p.yardline_100.notna() & (p.qtr <= 4)]
    f4 = pr[(pr.down == 4) & pr.play_type.isin(["run", "pass", "punt", "field_goal"]) & (pr.yardline_100 <= 40)].copy()
    f4["fg"] = (f4.play_type == "field_goal").astype(float)
    f4["go"] = f4.play_type.isin(["run", "pass"]).astype(float)
    say("3a. real 4th downs at yl<=40: FG share and go share by season")
    for sn, q in f4.groupby("season"):
        say(f"  {sn} n {len(q)} FG {q.fg.mean():.4f} go {q.go.mean():.4f}")
    gi4 = f4.game_id.astype("category").cat.codes.to_numpy()
    G4 = gi4.max() + 1
    W4 = boot_w(G4)
    yr = f4.season.to_numpy(float) - 2013.0
    slope = np.zeros(NB + 1)
    for b in range(NB + 1):
        w = W4[b][gi4]
        m = np.average(yr, weights=w)
        slope[b] = np.sum(w * (yr - m) * (f4.fg.to_numpy() - np.average(f4.fg.to_numpy(), weights=w))) / np.sum(w * (yr - m) ** 2)
    LOOKS[0] += 1
    say(f"3b. slope of FG share per season 2009-2017 {fmtd(slope, 5)} probability_positive {np.mean(slope[1:] > 0):.2f}; fd4 era_at_sim is pinned to 2017 (artifacts/mod25e3/fd4/fit.json) while sim years run 2011-2016")
    gate4 = f4[f4.season.isin(REAL_SEASONS)]
    say(f"  FG share real gate years 2011-17 {gate4.fg.mean():.4f}; 2017 only {f4[f4.season == 2017].fg.mean():.4f}")
    s4 = dr_s[(dr_s.down == 4) & (dr_s.yl <= 40) & dr_s.code.isin([0, 1, 2, 3])]
    r4 = gate4
    gr4 = r4.game_id.astype("category").cat.codes.to_numpy()
    Wr4, Ws4 = boot_w(gr4.max() + 1), boot_w(Gs_)
    gs4 = s4.gi.to_numpy()
    for nm, rv, sv in (("FG", (r4.play_type == "field_goal").to_numpy(float), (s4.code == 3).to_numpy(float)), ("go", r4.play_type.isin(["run", "pass"]).to_numpy(float), s4.code.isin([0, 1]).to_numpy(float))):
        ra = np.bincount(gr4, weights=rv, minlength=gr4.max() + 1)
        rn = np.bincount(gr4, minlength=gr4.max() + 1).astype(float)
        sa = np.bincount(gs4, weights=sv, minlength=Gs_)
        sn_ = np.bincount(gs4, minlength=Gs_).astype(float)
        say(stat_line(f"3d. 4th down yl<=40 {nm} share real 2011-17 v sim", ratio(Wr4, ra, rn), ratio(Ws4, sa, sn_)))
    fg = pr[pr.play_type == "field_goal"].copy()
    fg["made"] = (fg.d == 3).astype(float)
    fg["kd"] = fg.yardline_100.to_numpy(float) + 17.0
    from sklearn.linear_model import LogisticRegression

    say("3c. leave-one-season-out FG make: logistic in kick distance vs constant, log loss per attempt; coefficients per fold")
    tot = {"logit": [], "const": []}
    for sn in TRAIN:
        a, b = fg[fg.season != sn], fg[fg.season == sn]
        m = LogisticRegression(C=1e6, max_iter=500).fit(a[["kd"]], a.made)
        pl = np.clip(m.predict_proba(b[["kd"]])[:, 1], 1e-6, 1 - 1e-6)
        pc = a.made.mean()
        yv = b.made.to_numpy()
        l1 = -np.mean(yv * np.log(pl) + (1 - yv) * np.log(1 - pl))
        l0 = -np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc))
        tot["logit"].append(l1)
        tot["const"].append(l0)
        say(f"  {sn} n {len(b)} coef {m.coef_[0][0]:+.4f} intercept {m.intercept_[0]:+.3f} ll logit {l1:.4f} const {l0:.4f}")
    say(f"  mean ll logit {np.mean(tot['logit']):.4f} const {np.mean(tot['const']):.4f}; seasons logit better {int(np.sum(np.array(tot['logit']) < np.array(tot['const'])))}/{len(TRAIN)}")
    LOOKS[0] += 2
    say(f"looks counted {LOOKS[0]} (families: PAT table, LOSO forms, drive cells by band/down/distance/state)")
    (E3 / "e98").mkdir(parents=True, exist_ok=True)
    (E3 / "e98" / "e98.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
