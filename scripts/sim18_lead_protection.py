import glob
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

MIN_SPELL_GAMES = 16
MIN_SEASON_GAMES = 6
N_PERM = 2000
TIME_EDGES = [0, 300, 600, 900, 1200, 1500, 1800]
LEAD_EDGES = [1, 9, 17, 10**6]
LEAD_NAMES = ["lead_1_8", "lead_9_16", "lead_17_plus"]
rng = np.random.default_rng(1801)
OUT = "artifacts/sim18"


def tbin(s):
    return np.clip(np.searchsorted(TIME_EDGES, s, side="right") - 1, 0, len(TIME_EDGES) - 2)


def lbin(abs_lead):
    return np.where(abs_lead >= 1, np.searchsorted(LEAD_EDGES, abs_lead, side="right") - 1, -1)


d = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season",
        "season_type",
        "posteam",
        "fixed_drive",
        "play_type",
        "pass_attempt",
        "complete_pass",
        "sack",
        "qb_kneel",
        "qb_spike",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
        "play_id",
    ],
)
d = d[d.season_type.eq("REG") & d.qtr.isin([3, 4]) & d.posteam.notna()].copy()
d = d.dropna(subset=["game_seconds_remaining", "score_differential", "fixed_drive"])
d = d.sort_values(["game_id", "play_id"]).reset_index(drop=True)

dv = (
    d.groupby(["game_id", "fixed_drive"], sort=False)
    .agg(
        season=("season", "first"),
        posteam=("posteam", "first"),
        start=("game_seconds_remaining", "first"),
        lead=("score_differential", "first"),
    )
    .reset_index()
)
dv["nxt"] = dv.groupby("game_id").start.shift(-1).fillna(0.0)
dv["el"] = np.minimum(dv.start - dv.nxt, dv.start)
dv["tb"] = tbin(dv.start.values)


def remaining(frame):
    out_l, out_t = [], []
    for _, g in frame.groupby("game_id", sort=False):
        pos = g.posteam.values
        lead = g.lead.values
        teams = sorted(set(pos))
        for i in range(len(g)):
            lp = None
            if lead[i] > 0:
                lp = pos[i]
            elif lead[i] < 0:
                oth = [t for t in teams if t != pos[i]]
                lp = oth[0] if len(oth) == 1 else None
            if lp is None:
                out_l.append(np.nan)
                out_t.append(np.nan)
                continue
            fut = pos[i:]
            out_l.append(float((fut == lp).sum()))
            out_t.append(float((fut != lp).sum()))
    return out_l, out_t


dv["rem_leader"], dv["rem_trailer"] = remaining(dv)
dv["abs_lead"] = dv.lead.abs()
dv["lb"] = lbin(dv.abs_lead.values)
dv["leader_has_ball"] = dv.lead > 0


def drive_table(frame, ball_col, el_col):
    out = {}
    for (lb, tb), g in frame[frame.lb >= 0].groupby(["lb", "tb"]):
        out[f"{LEAD_NAMES[lb]}|t{TIME_EDGES[tb]}_{TIME_EDGES[tb + 1]}"] = {
            "n": len(g),
            "rem_leader": float(g.rem_leader.mean()),
            "rem_trailer": float(g.rem_trailer.mean()),
            "leader_ball_el": float(g[g[ball_col]][el_col].mean()),
            "trailer_ball_el": float(g[~g[ball_col]][el_col].mean()),
        }
    return out


real_drive = drive_table(dv, "leader_has_ball", "el")

sim = []
for f in sorted(glob.glob("artifacts/sim10/instrument/instr_s11/*.jsonl")):
    for line in Path(f).read_text().splitlines():
        sim.append(json.loads(line))
srows = []
for gm in sim:
    cum = [0.0, 0.0]
    recs = []
    for hf, pos, _yl, secs, po, pdf, el, _tr in gm["dr"]:
        recs.append((hf, pos, secs, cum[pos] - cum[1 - pos], el))
        cum[pos] += po
        cum[1 - pos] += pdf
    sec_half = [r for r in recs if r[0] == 2]
    posl = np.array([r[1] for r in sec_half])
    for i, r in enumerate(sec_half):
        lead = r[3]
        if lead == 0:
            continue
        lp = r[1] if lead > 0 else 1 - r[1]
        fut = posl[i:]
        srows.append(
            (abs(lead), r[2], float((fut == lp).sum()), float((fut != lp).sum()), lead > 0, r[4])
        )
sd = pd.DataFrame(
    srows, columns=["abs_lead", "secs", "rem_leader", "rem_trailer", "leader_has_ball", "el"]
)
sd["lb"] = lbin(sd.abs_lead.values)
sd["tb"] = tbin(sd.secs.values)
sim_drive = drive_table(sd, "leader_has_ball", "el")

p = d[d.play_type.isin(["run", "pass"]) | d.qb_kneel.fillna(0).astype(bool)].copy()
p["kneel"] = p.qb_kneel.fillna(0).astype(float)
p["run"] = (p.play_type.eq("run") & p.kneel.eq(0)).astype(float)
p = p[p.qb_spike.fillna(0).eq(0)].copy()
p["nxt_gsr"] = p.groupby(["game_id", "fixed_drive"]).game_seconds_remaining.shift(-1)
p["nxt_qtr"] = p.groupby(["game_id", "fixed_drive"]).qtr.shift(-1)
p["sec"] = np.where(p.nxt_qtr.eq(p.qtr), p.game_seconds_remaining - p.nxt_gsr, np.nan)
p["lead"] = p.score_differential
p["tb"] = tbin(p.game_seconds_remaining.values)
p["lb"] = lbin(p.lead.abs().values)
p["incomp"] = (p.pass_attempt.eq(1) & p.complete_pass.eq(0) & p.sack.fillna(0).eq(0)).astype(float)
p["is_pass"] = p.pass_attempt.eq(1).astype(float)
p["sgn"] = np.sign(p.lead)

curve = {}
cells = p[p.sgn != 0].copy()
cells["role"] = np.where(cells.sgn > 0, "leader", "trailer")
for (role, lb, tb), g in cells[cells.lb >= 0].groupby(["role", "lb", "tb"]):
    gs = g.dropna(subset=["sec"])
    curve[f"{role}|{LEAD_NAMES[lb]}|t{TIME_EDGES[tb]}_{TIME_EDGES[tb + 1]}"] = {
        "n_plays": len(g),
        "sec_per_play": float(gs.sec.mean()),
        "run_share": float(g.run.mean()),
        "kneel_share": float(g.kneel.mean()),
        "incomplete_per_pass": float(g[g.is_pass == 1].incomp.mean()),
    }
tied = p[p.sgn == 0]
curve["tied"] = {
    "n_plays": len(tied),
    "sec_per_play": float(tied.sec.mean()),
    "run_share": float(tied.run.mean()),
}

seasons = sorted(p.season.unique())


def design(x, deg):
    return np.column_stack([x**j for j in range(deg + 1)])


def loso_deg(frame, y):
    best = {}
    for tb in range(len(TIME_EDGES) - 1):
        f = frame[(frame.tb == tb) & frame[y].notna()]
        sc_ = f.x.std()
        xs = f.x.values / sc_
        yv = f[y].values
        ss = f.season.values
        score = {}
        for deg in range(0, 5):
            err = 0.0
            for s in seasons:
                m = ss == s
                if m.sum() == 0:
                    continue
                b = np.linalg.lstsq(design(xs[~m], deg), yv[~m], rcond=None)[0]
                err += float(((yv[m] - design(xs[m], deg) @ b) ** 2).sum())
            score[deg] = err / len(f)
        dbest = min(score, key=score.get)
        b = np.linalg.lstsq(design(xs, dbest), yv, rcond=None)[0]
        grid = {
            int(g_): float((design(np.array([g_ / sc_]), dbest) @ b)[0]) for g_ in (1, 4, 8, 16, 24)
        }
        best[f"t{TIME_EDGES[tb]}_{TIME_EDGES[tb + 1]}"] = {
            "loso_degree": int(dbest),
            "loso_mse_by_degree": {int(k): float(v) for k, v in score.items()},
            "fit_at_lead": grid,
        }
    return best


lp_ = p[p.sgn > 0].assign(x=lambda f: f.lead.astype(float))
smooth = {"sec_per_play": loso_deg(lp_, "sec"), "run": loso_deg(lp_, "run")}

cell = p[(p.sgn > 0) & (p.lb >= 0)].copy()
cell["cell"] = cell.lb * 10 + cell.tb
for col in ["sec", "run"]:
    base = np.full(len(cell), np.nan)
    for s in seasons:
        m = (cell.season == s).values
        mu = cell[~m & cell[col].notna().values].groupby("cell")[col].mean()
        base[m] = cell.loc[m, "cell"].map(mu).values
    cell["res_" + col] = cell[col] - base


def team_game(col):
    sub = cell[cell["res_" + col].notna()]
    return (
        sub.groupby(["game_id", "season", "posteam"])["res_" + col]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "res_" + col, "size": "n_" + col})
    )


tg = team_game("sec").merge(
    team_game("run")[["game_id", "posteam", "res_run", "n_run"]], on=["game_id", "posteam"]
)
sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[
    ["game_id", "home_team", "home_coach", "away_coach", "gameday"]
]
tg = tg.merge(sch, on="game_id").rename(columns={"posteam": "team"})
tg["coach"] = np.where(tg.team == tg.home_team, tg.home_coach, tg.away_coach)
tg["t"] = (pd.to_datetime(tg.gameday, utc=True) + pd.Timedelta(days=1)).astype(
    "datetime64[ns, UTC]"
)
tg["season"] = tg.season.astype("int64")
tg = tg.sort_values("t").reset_index(drop=True)
cdir = sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))[-1]
ch = pd.read_parquet(cdir)
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)
oc = ch[ch.role == "OC"].sort_values("effective_observed_at").astype({"season": "int64"})
m = pd.merge_asof(
    tg[["game_id", "team", "season", "t"]],
    oc[["team", "season", "effective_observed_at", "person"]],
    left_on="t",
    right_on="effective_observed_at",
    by=["team", "season"],
    direction="backward",
)
first = oc.drop_duplicates(["team", "season"])[["team", "season", "person"]].rename(
    columns={"person": "first_person"}
)
m = m.merge(first, on=["team", "season"], how="left")
tg["oc"] = (m.person.fillna(m.first_person)).values
tg = tg.sort_values(["season", "game_id"]).reset_index(drop=True)
for k, c in [("hc", "coach"), ("oc", "oc")]:
    tg["sp_" + k] = tg[c].astype("string") + "|" + tg.team
    tg.loc[tg[c].isna(), "sp_" + k] = pd.NA
    tg["par_" + k] = tg.groupby("sp_" + k).cumcount() % 2


def split_stat(frame, sk, pk, col, wcol, flips=None):
    v = frame[col].values * (flips if flips is not None else 1.0)
    fr = pd.DataFrame(
        {
            sk: frame[sk].values,
            pk: frame[pk].values,
            "vw": v * frame[wcol].values,
            "w": frame[wcol].values,
        }
    )
    g = fr.groupby([sk, pk]).agg(vw=("vw", "sum"), w=("w", "sum")).reset_index()
    g["mu"] = g.vw / g.w
    pv = g.pivot(index=sk, columns=pk, values="mu").dropna()
    r = float(np.corrcoef(pv[0].values, pv[1].values)[0, 1])
    return 2 * r / (1 + r), len(pv)


def season_stat(frame, sk, col, wcol, flips=None):
    v = frame[col].values * (flips if flips is not None else 1.0)
    fr = pd.DataFrame(
        {
            sk: frame[sk].values,
            "season": frame.season.values,
            "vw": v * frame[wcol].values,
            "w": frame[wcol].values,
        }
    )
    g = (
        fr.groupby([sk, "season"])
        .agg(vw=("vw", "sum"), w=("w", "sum"), ng=("w", "size"))
        .reset_index()
    )
    g = g[g.ng >= MIN_SEASON_GAMES]
    g["mu"] = g.vw / g.w
    nx = g[[sk, "season", "mu"]].copy()
    nx["season"] = nx.season - 1
    pr = g.merge(nx, on=[sk, "season"], suffixes=("_t", "_t1"))
    return float(np.corrcoef(pr.mu_t, pr.mu_t1)[0, 1]), len(pr)


rel = {}
for key in ["hc", "oc"]:
    sk, pk = "sp_" + key, "par_" + key
    gk = tg[tg[sk].notna()]
    sz = gk.groupby(sk).size()
    gk = gk[gk[sk].isin(sz[sz >= MIN_SPELL_GAMES].index)].copy()
    for col in ["sec", "run"]:
        c, w = "res_" + col, "n_" + col
        obs, nsp = split_stat(gk, sk, pk, c, w)
        obs_s, npairs = season_stat(gk, sk, c, w)
        pa, pb = [], []
        for _ in range(N_PERM):
            fl = rng.choice([-1.0, 1.0], len(gk))
            pa.append(split_stat(gk, sk, pk, c, w, fl)[0])
            pb.append(season_stat(gk, sk, c, w, fl)[0])
        pa, pb = np.array(pa), np.array(pb)
        rel[f"{key}|{c}"] = {
            "n_spells": nsp,
            "split_half_sb": obs,
            "split_half_p_signflip": float((pa >= obs).mean()),
            "split_half_null_q95": float(np.quantile(pa, 0.95)),
            "n_season_pairs": npairs,
            "season_t_t1_r": obs_s,
            "season_p_signflip": float((pb >= obs_s).mean()),
            "season_null_q95": float(np.quantile(pb, 0.95)),
        }

res = {
    "looks": {
        "curve_cells": len(curve),
        "reliability_tests": len(rel) * 2,
        "loso_smooth_fits": 12,
        "drive_cells_real": len(real_drive),
        "drive_cells_sim": len(sim_drive),
        "family": "lead-protection curve plus spell persistence, declared before any sign seen",
    },
    "time_edges": TIME_EDGES,
    "lead_edges": LEAD_EDGES[:-1],
    "n_plays": len(p),
    "n_real_drives": len(dv),
    "n_sim_games": len(sim),
    "n_sim_drives_2h": len(sd),
    "play_curve": curve,
    "smooth_loso": smooth,
    "reliability": rel,
    "real_drive": real_drive,
    "sim_drive": sim_drive,
}
Path(f"{OUT}/unit1.json").write_text(json.dumps(res, indent=1))
print("done", res["looks"], res["n_plays"], res["n_real_drives"], res["n_sim_drives_2h"])
