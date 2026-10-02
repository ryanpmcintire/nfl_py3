import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sim09_urgency import cluster_ols  # noqa: E402

NV = Path("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv")
OUT = ROOT / "artifacts" / "sim09"
PBP = sorted(glob.glob(str(ROOT / "data/pbp/raw/*")))[-1]
PART = ROOT / "data/players/participation/raw/20260813T131635Z"
FIT = (2016, 2021)
HELD = (2022, 2025)
ROOKIE_DB = 300
PRESNAP = ["False Start", "Delay of Game", "Illegal Formation", "Illegal Shift", "Illegal Motion", "Offensive Offside", "Illegal Substitution"]
TWO_HIGH = ["COVER_2", "COVER_4", "2_MAN"]
ONE_HIGH = ["COVER_1", "COVER_3", "COVER_0"]

SPECS = [
    dict(fam="A1", name="TM", S="tm", rs=["qe", "rookie"], pairs=[], ys=["epa", "sack", "intc", "pspen"], fit=FIT, held=HELD),
    dict(fam="A1", name="NH", S="nh", rs=["qe", "rookie"], pairs=[], ys=["epa", "sack", "intc", "pspen"], fit=(2013, 2015), held=(2016, 2017)),
    dict(fam="A3", name="OPD3", S="opd3", rs=["PP", "RUSH"], pairs=[("PP", "RUSH")], ys=["pressure", "sack", "intc", "epa"], fit=FIT, held=HELD),
    dict(fam="A3", name="TRL", S="trl", rs=["PP", "RUSH"], pairs=[("PP", "RUSH")], ys=["pressure", "sack", "intc", "epa"], fit=FIT, held=HELD),
    dict(fam="A4", name="DEEP", S="deep", rs=["PO", "SEP", "PD", "SEPd"], pairs=[], ys=["comp", "intc", "epa"], fit=FIT, held=(2022, 2024)),
    dict(fam="A4", name="DEEPSHELL", S="deep", rs=["SEP", "two"], pairs=[("SEP", "two")], ys=["comp", "intc", "epa"], fit=(2018, 2021), held=(2022, 2024)),
    dict(fam="A5", name="LEADRUN", S="lead", rs=["RB", "RD", "box"], pairs=[("RB", "box")], ys=["succ", "fd", "epa"], fit=FIT, held=HELD),
    dict(fam="A5", name="LEADDRIVE", S="lead", rs=["RB", "RD"], pairs=[], ys=["tno", "clock"], fit=FIT, held=HELD),
]


def term_names(sp):
    t = [f"{sp['S']}x{r}" for r in sp["rs"]] + [f"{sp['S']}x{a}x{b}" for a, b in sp["pairs"]]
    return t


def declare():
    d = {"declared_before_signs": True, "fit": list(FIT), "held": list(HELD), "rookie_dropbacks_lt": ROOKIE_DB,
         "pressure_two_high": TWO_HIGH, "one_high": ONE_HIGH, "families": {}}
    n = 0
    for sp in SPECS:
        ts = term_names(sp)
        d["families"][f"{sp['fam']}_{sp['name']}"] = {"situation": sp["S"], "terms": ts, "outcomes": sp["ys"], "fit": list(sp["fit"]), "held": list(sp["held"])}
        n += len(ts) * len(sp["ys"])
    d["looks"] = n
    d["unavailable"] = ["years with same head coach beyond 2017 (no coach field in 2018-25 pbp)", "no_huddle after 2017 (not in raw pbp)", "air yards 2025", "coverage shell before 2018"]
    d["orientation"] = "each rating oriented so higher = better for its own side, standardized with fit-window mean/sd; effects per 1 SD; rookie per flag"
    (OUT / "pers_declared.json").write_text(json.dumps(d, indent=1))
    return n


def load_raw():
    cols = ["game_id", "play_id", "season", "season_type", "week", "home_team", "posteam", "defteam", "down", "play_type", "yards_gained",
            "rush_attempt", "sack", "epa", "success", "qtr", "ydstogo", "yardline_100", "game_seconds_remaining", "qb_dropback", "qb_kneel", "qb_spike",
            "complete_pass", "interception", "first_down", "passer_player_id", "penalty_type", "penalty_team", "fixed_drive", "fixed_drive_result", "score_differential"]
    fr = []
    for s in range(2009, 2026):
        x = pd.read_parquet(f"{PBP}/season={s}/plays.parquet", columns=cols)
        fr.append(x[x.season_type == "REG"])
    return pd.concat(fr, ignore_index=True)


def experience(d):
    db = d[(d.qb_dropback == 1) & d.passer_player_id.notna() & (d.play_type == "pass")]
    g = db.groupby(["passer_player_id", "season", "week", "game_id"]).size().rename("n").reset_index().sort_values(["passer_player_id", "season", "week"])
    g["cum"] = g.groupby("passer_player_id").n.cumsum() - g.n
    assert (g.groupby("passer_player_id").cum.diff().dropna() > 0).all()
    first = db.sort_values(["game_id", "play_id"]).groupby(["game_id", "posteam"]).passer_player_id.first().rename("starter").reset_index()
    first = first.merge(g[["passer_player_id", "game_id", "cum"]], left_on=["starter", "game_id"], right_on=["passer_player_id", "game_id"], how="left")
    return first[["game_id", "posteam", "cum"]].rename(columns={"cum": "cum_db"})


def ratings_table():
    u = pd.read_parquet(ROOT / "data/processed/sim09_unit_ratings.parquet")
    assert not u.duplicated(["season", "week", "team"]).any()
    fit = u[u.season.between(*FIT)]
    spec = {"PO": ("epa_pass_off", "epa_pass_off", 1), "SEP": ("sep_off", "epa_pass_off", 1), "PP": ("press_net_off", "epa_pass_off", 1), "RB": ("run_net_off", "epa_run_off", 1),
            "PD": ("epa_pass_def", None, -1), "SEPd": ("sep_def", "epa_pass_def", -1), "RUSH": ("press_net_def", "epa_pass_def", -1), "RD": ("run_net_def", "epa_run_def", -1)}
    t = u[["season", "week", "team"]].copy()
    orient = {}
    for k, (c, ref, sg) in spec.items():
        if ref is None:
            s = sg
        else:
            cr = fit[[c, ref]].dropna().corr().iloc[0, 1]
            s = int(np.sign(cr)) * (1 if sg == 1 else -1)
            orient[k] = float(cr)
        q = s * u[c]
        t[k] = (q - q[u.season.between(*FIT)].mean()) / q[u.season.between(*FIT)].std()
    for k in ("epa_all_off", "epa_all_def"):
        t[k] = (u[k] - fit[k].mean()) / fit[k].std()
    return t, orient


def participation():
    fr = []
    for s in range(2016, 2026):
        x = pd.read_parquet(PART / f"season={s}" / "participation.parquet", columns=["game_id", "play_id", "defenders_in_box", "was_pressure", "ngs_air_yards", "defense_coverage_type"])
        fr.append(x)
    return pd.concat(fr, ignore_index=True)


def build():
    d = load_raw()
    d = d[d.posteam.notna() & d.play_type.isin(["pass", "run", "no_play"]) & (d.qb_kneel != 1) & (d.qb_spike != 1) & (d.qtr <= 4)].copy()
    ex = experience(load_raw_ids(d))
    d = d.merge(ex, on=["game_id", "posteam"], how="left")
    nv = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=["game_id", "play_id", "no_huddle"]) for s in range(2009, 2018)])
    nv["play_id"] = nv.play_id.astype(d.play_id.dtype)
    d = d.merge(nv.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
    pt = participation()
    pt["play_id"] = pt.play_id.astype(d.play_id.dtype)
    d = d.merge(pt.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
    t, orient = ratings_table()
    offk = ["PO", "SEP", "PP", "RB", "epa_all_off"]
    defk = ["PD", "SEPd", "RUSH", "RD", "epa_all_def"]
    n0 = len(d)
    d = d.merge(t[["season", "week", "team"] + offk].rename(columns={"team": "posteam", "epa_all_off": "eo"}), on=["season", "week", "posteam"], how="left")
    d = d.merge(t[["season", "week", "team"] + defk].rename(columns={"team": "defteam", "epa_all_def": "ed"}), on=["season", "week", "defteam"], how="left")
    assert len(d) == n0
    d["tsec"] = d.game_seconds_remaining.clip(lower=0)
    d["half_sec"] = pd.Series(np.where(d.qtr <= 2, d.tsec - 1800, d.tsec), index=d.index).clip(lower=0)
    d["sd"] = d.score_differential.astype(float)
    d["zs"] = d.sd / np.sqrt(d.tsec / 60.0 + 1.0)
    d["home"] = (d.posteam == d.home_team).astype(float)
    d["d2"] = (d.down == 2).astype(float)
    d["d3"] = (d.down == 3).astype(float)
    d["d4"] = (d.down == 4).astype(float)
    d["lyd"] = np.log(d.ydstogo.clip(lower=1))
    d["yl"] = d.yardline_100 / 100.0
    d["qe"] = np.log1p(d.cum_db)
    m, s = d.loc[d.season.between(*FIT) & d.qe.notna(), "qe"].agg(["mean", "std"])
    d["qe"] = (d.qe - m) / s
    d["rookie"] = np.where(d.cum_db.notna(), (d.cum_db < ROOKIE_DB).astype(float), np.nan)
    db = (d.play_type == "pass") & (d.qb_dropback == 1)
    d["db"] = db
    d["epa_db"] = np.where(db, d.epa, np.nan)
    d["sack_db"] = np.where(db, d.sack.astype(float), np.nan)
    d["intc_db"] = np.where(db & (d.sack != 1), d.interception.astype(float), np.nan)
    d["pspen"] = ((d.play_type == "no_play") & (d.penalty_team == d.posteam) & d.penalty_type.isin(PRESNAP)).astype(float)
    d["tm"] = (d.half_sec <= 120).astype(float)
    d["nh"] = d.no_huddle.astype(float)
    d["opd3"] = ((d.down == 3) & (d.ydstogo >= 7)).astype(float)
    d["trl"] = ((d.qtr == 4) & (d.tsec <= 480) & (d.sd <= -4) & (d.opd3 == 0)).astype(float)
    d["pressure_db"] = np.where(db & d.was_pressure.notna(), d.was_pressure.astype(float), np.nan)
    att = (d.play_type == "pass") & (d.sack != 1) & d.ngs_air_yards.notna() & (d.qb_dropback == 1)
    d["deep"] = np.where(att, (d.ngs_air_yards >= 20).astype(float), np.nan)
    d["comp_a"] = np.where(att, d.complete_pass.astype(float), np.nan)
    d["intc_a"] = np.where(att, d.interception.astype(float), np.nan)
    d["epa_a"] = np.where(att, d.epa, np.nan)
    cov = d.defense_coverage_type
    d["two"] = np.where(cov.isin(TWO_HIGH), 1.0, np.where(cov.isin(ONE_HIGH), 0.0, np.nan))
    run = (d.play_type == "run") & (d.rush_attempt == 1)
    d["run"] = run
    d["lead"] = ((d.qtr == 4) & (d.sd > 0)).astype(float)
    d["succ_r"] = np.where(run, d.success.astype(float), np.nan)
    d["fd_r"] = np.where(run, d.first_down.astype(float), np.nan)
    d["epa_r"] = np.where(run, d.epa, np.nan)
    bx = d.defenders_in_box
    mb, sb = bx[d.season.between(*FIT) & run].agg(["mean", "std"])
    d["box"] = np.where(run & bx.notna(), (bx - mb) / sb, np.nan)
    return d, orient


def load_raw_ids(d):
    return d


def drives(d):
    g = d.sort_values(["game_id", "play_id"])
    g = g[g.fixed_drive.notna() & g.play_type.isin(["pass", "run"])]
    a = g.groupby(["game_id", "fixed_drive"]).agg(season=("season", "first"), week=("week", "first"), posteam=("posteam", "first"), defteam=("defteam", "first"),
                                                  qtr=("qtr", "first"), t0=("tsec", "first"), sd0=("sd", "first"), n=("play_id", "size"), res=("fixed_drive_result", "last"),
                                                  home=("home", "first"), eo=("eo", "first"), ed=("ed", "first"), RB=("RB", "first"), RD=("RD", "first"),
                                                  d2=("d2", "first")).reset_index()
    a = a.sort_values(["game_id", "fixed_drive"])
    a["t1"] = a.groupby("game_id").t0.shift(-1)
    a["t1"] = a.t1.fillna(0)
    a["clock"] = (a.t0 - a.t1).clip(lower=0) / 60.0
    a["tno"] = ((a.n <= 3) & (a.res == "Punt")).astype(float)
    a["lead"] = ((a.qtr == 4) & (a.sd0 > 0)).astype(float)
    a = a[(a.qtr >= 3) & (a.qtr == a.qtr)].copy()
    a["zs"] = a.sd0 / np.sqrt(a.t0 / 60.0 + 1.0)
    a["d2"] = 0.0
    a["d3"] = 0.0
    a["d4"] = 0.0
    a["lyd"] = 0.0
    a["yl"] = 0.0
    return a


CTRL = ["home", "d2", "d3", "d4", "lyd", "yl", "zs", "eo", "ed"]


def fit_one(df, y, S, rs, pairs, ctrl):
    cols = [S] + rs + [y, "game_id"] + ctrl
    x = df[list(dict.fromkeys(cols))].dropna()
    if len(x) < 2000:
        return None
    X = pd.DataFrame({"const": 1.0}, index=x.index)
    X[S] = x[S]
    for r in rs:
        X[r] = x[r]
    for a, b in pairs:
        X[f"{a}x{b}"] = x[a] * x[b]
    names = []
    for r in rs:
        X[f"{S}x{r}"] = x[S] * x[r]
        names.append(f"{S}x{r}")
    for a, b in pairs:
        X[f"{S}x{a}x{b}"] = x[S] * x[a] * x[b]
        names.append(f"{S}x{a}x{b}")
    for c in ctrl:
        if c not in X:
            X[c] = x[c]
    X = X.loc[:, X.std().fillna(0).gt(0) | (X.columns == "const")]
    if (x[S] == 1).sum() < 200:
        return None
    b, se, u = cluster_ols(X, x[y], x["game_id"])
    sdy = float(x[y].std())
    return {n: (b[n], se[n]) for n in names if n in b.index}, len(x), int((x[S] == 1).sum()), sdy


def eb(rows):
    e = np.array([r["std_fit"] for r in rows])
    s = np.array([r["se_std_fit"] for r in rows])
    w = 1 / s**2
    mu0 = (w * e).sum() / w.sum()
    q = (w * (e - mu0) ** 2).sum()
    tau2 = max(0.0, (q - (len(e) - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    B = tau2 / (tau2 + s**2)
    mu = (e / s**2 + 0) .sum() * 0 + mu0
    sh = mu + B * (e - mu)
    ps = np.sqrt(B * s**2)
    for r, a, c in zip(rows, sh, ps):
        r["shrunk_std"] = float(a)
        r["pp_shrunk"] = float(norm.cdf(a / c)) if c > 0 else 0.5
    return tau2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nlook = declare()
    d, orient = build()
    dr = drives(d)
    yc = {"epa": {"A1": "epa_db", "A3": "epa_db", "A4": "epa_a", "A5": "epa_r"}, "sack": "sack_db", "intc": {"A1": "intc_db", "A3": "intc_db", "A4": "intc_a"},
          "pspen": "pspen", "pressure": "pressure_db", "comp": "comp_a", "succ": "succ_r", "fd": "fd_r", "tno": "tno", "clock": "clock"}
    rows = []
    for sp in SPECS:
        for y in sp["ys"]:
            ycol = yc[y]
            if isinstance(ycol, dict):
                ycol = ycol[sp["fam"]]
            src = dr if sp["name"] == "LEADDRIVE" else d
            ctrl = ["home", "eo", "ed", "zs"] if sp["name"] == "LEADDRIVE" else CTRL
            S = sp["S"]
            if sp["fam"] == "A4":
                S = "deep"
                df = src.assign(**{"deep": src.deep})
            else:
                df = src
            res = {}
            for tag, win in (("fit", sp["fit"]), ("held", sp["held"])):
                sub = df[df.season.between(*win)]
                r = fit_one(sub, ycol, S, sp["rs"], sp["pairs"], ctrl)
                res[tag] = r
            if res["fit"] is None:
                continue
            for tn in res["fit"][0]:
                b, se = res["fit"][0][tn]
                sdy = res["fit"][3]
                row = dict(fam=sp["fam"], model=sp["name"], outcome=y, term=tn, n_fit=res["fit"][1], n_sit_fit=res["fit"][2], est_fit=b, se_fit=se,
                           std_fit=b / sdy, se_std_fit=se / sdy, pp_fit=float(norm.cdf(b / se)))
                h = res["held"]
                if h is not None and tn in h[0]:
                    bh, seh = h[0][tn]
                    row.update(n_held=h[1], n_sit_held=h[2], est_held=bh, se_held=seh, pp_held=float(norm.cdf(bh / seh)), sign_match=bool(np.sign(bh) == np.sign(b)))
                rows.append(row)
    tau = {}
    for fam in sorted({r["fam"] for r in rows}):
        fr = [r for r in rows if r["fam"] == fam]
        tau[fam] = eb(fr)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pers_terms.csv", index=False)
    summ = {"declared_looks": nlook, "fitted_terms": len(df), "tau2_std_by_family": tau, "orientation_corr": orient,
            "leak_checks": ["experience cum_db strictly excludes the current game (asserted increasing per passer)", "ratings keyed to the play's own season-week from F1 (rolling pregame assert in sim09_units.py)",
                            "merge preserved row count"],
            "n_plays": int(len(d)), "n_drives": int(len(dr))}
    (OUT / "pers_summary.json").write_text(json.dumps(summ, indent=1))
    pd.set_option("display.width", 250)
    show = df[["fam", "model", "outcome", "term", "est_fit", "pp_fit", "shrunk_std", "pp_shrunk", "est_held", "pp_held", "sign_match", "n_sit_fit", "n_sit_held"]]
    print(show.round(4).to_string())
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
