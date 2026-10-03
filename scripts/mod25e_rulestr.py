import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "artifacts" / "mod25e3" / "rulestr"
NV = Path("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv")
COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam", "home_team", "away_team", "home_score", "away_score", "down", "play_type", "qb_kneel", "qb_spike", "penalty", "penalty_type", "penalty_team", "penalty_yards", "timeout", "timeout_team", "epa", "game_seconds_remaining"]


def boot(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(x)
    b = np.empty(n)
    for i in range(n):
        j = rng.integers(0, k, k)
        b[i] = np.polyfit(x[j], y[j], 1)[0]
    s = float(np.polyfit(x, y, 1)[0])
    return s, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float((b > 0).mean())


def team_games():
    d = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=COLS) for s in range(2009, 2018)], ignore_index=True)
    d = d[d.week <= 17].drop_duplicates(["game_id", "play_id"])
    live = d.posteam.notna() & d.down.isin([1, 2, 3, 4]) & d.play_type.isin(["run", "pass", "no_play"]) & (d.qb_kneel != 1) & (d.qb_spike != 1)
    acc = (d.penalty == 1) & d.penalty_type.notna() & d.penalty_team.notna()
    rows = []
    g = d.groupby("game_id")
    meta = g.agg(season=("season", "first"), home=("home_team", "first"), away=("away_team", "first"), hs=("home_score", "max"), as_=("away_score", "max"))
    for side in ("home", "away"):
        tm = meta[side]
        o = d[live].assign(team=d.loc[live, "posteam"])
        sn = o.groupby(["game_id", "team"]).size().rename("snaps")
        ao = d[live & acc & (d.penalty_team == d.posteam)].groupby(["game_id", "posteam"]).agg(pen_off=("penalty", "size"), pen_off_yds=("penalty_yards", "sum"))
        ao.index.names = ["game_id", "team"]
        ad = d[live & acc & (d.penalty_team == d.defteam)].groupby(["game_id", "defteam"]).agg(pen_def=("penalty", "size"), pen_def_yds=("penalty_yards", "sum"))
        ad.index.names = ["game_id", "team"]
        nop = d[live & (d.play_type == "no_play") & acc].groupby(["game_id", "posteam"]).size().rename("noplay")
        nop.index.names = ["game_id", "team"]
        to = d[(d.timeout == 1) & d.timeout_team.notna()].groupby(["game_id", "timeout_team"]).size().rename("to_used")
        to.index.names = ["game_id", "team"]
        ep = d[live].groupby(["game_id", "posteam"]).epa.sum().rename("epa_sum")
        ep.index.names = ["game_id", "team"]
        epp = d[live & acc].groupby(["game_id", "posteam"]).epa.sum().rename("epa_pen")
        epp.index.names = ["game_id", "team"]
        base = pd.DataFrame({"team": tm, "season": meta.season, "margin": (meta.hs - meta.as_) * (1 if side == "home" else -1), "home": int(side == "home")}).reset_index().set_index(["game_id", "team"])
        rows.append(base.join([sn, ao, ad, nop, to, ep, epp]).fillna(0.0).reset_index())
    return pd.concat(rows, ignore_index=True)


def cmd_real(a):
    OUT.mkdir(parents=True, exist_ok=True)
    t = team_games()
    ts = t.groupby(["season", "team"]).agg(g=("margin", "size"), m=("margin", "sum"), snaps=("snaps", "sum"), pen_off=("pen_off", "sum"), pen_def=("pen_def", "sum"), py=("pen_off_yds", "sum"), pdy=("pen_def_yds", "sum"), noplay=("noplay", "sum"), to=("to_used", "sum"), epa=("epa_sum", "sum"), epp=("epa_pen", "sum")).reset_index()
    ts["str_lgo"] = ts.m / ts.g
    prev = ts[["season", "team", "str_lgo"]].copy()
    prev["season"] += 1
    ts = ts.merge(prev.rename(columns={"str_lgo": "str_prev"}), on=["season", "team"], how="left")
    sd = ts.str_lgo.std()
    lines = [f"team-seasons {len(ts)} strength sd (pts/game) {sd:.2f}"]
    meas = {
        "off pen accepted per 100 snaps": ts.pen_off / ts.snaps * 100,
        "opp pen accepted per 100 own snaps": ts.pen_def / ts.snaps * 100,
        "off pen yards/game": ts.py / ts.g,
        "no-play accepted per 100 snaps": ts.noplay / ts.snaps * 100,
        "timeouts used/game": ts.to / ts.g,
        "epa/game all snaps": ts.epa / ts.g,
        "epa/game on penalty snaps": ts.epp / ts.g,
    }
    for nm, y in meas.items():
        for sn, col in (("same-season", "str_lgo"), ("prior-season", "str_prev")):
            k = ts[col].notna()
            x = ((ts[col] - ts[col].mean()) / ts[col].std())[k].to_numpy()
            s, lo, hi, pp = boot(x, y[k].to_numpy())
            lines.append(f"{nm} vs {sn} strength: slope per SD {s:+.4f} [{lo:+.4f},{hi:+.4f}] pp {pp:.3f} mean {y[k].mean():.3f} n {int(k.sum())}")
    k = ts.str_lgo.notna()
    x = ((ts.str_lgo - ts.str_lgo.mean()) / ts.str_lgo.std()).to_numpy()
    sa = np.polyfit(x, (ts.epa / ts.g).to_numpy(), 1)[0]
    sp = np.polyfit(x, (ts.epp / ts.g).to_numpy(), 1)[0]
    lines.append(f"share of epa/game strength slope carried by penalty snaps {sp / sa:.3f} (slope all {sa:.3f}, penalty {sp:.3f}); penalty snap share {(ts.pen_off.sum() + ts.pen_def.sum()) / ts.snaps.sum():.4f}")
    (OUT / "real.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def dump(c):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cnt_{os.getpid()}.json").write_text(json.dumps(c))


C = {"calls": 0, "f2_changed": 0, "f2_cls_diff": 0, "ov_calls": 0, "ov_noplay": 0, "ov_counted": 0, "ov_declined": 0}


WPICK_SRC = """
def WPICK(nb, rng):
    fr = sys._getframe(1)
    while fr is not None and "off_sim" not in fr.f_locals:
        fr = fr.f_back
    L = fr.f_locals
    ns = dv._G["ns"]
    t = dv._G["tables"]
    a = t["arrays"]
    h = t["team_kernel_h"]
    d2 = (a["off_row"][nb] - L["off_sim"]) ** 2 + (a["def_row"][nb] - L["def_sim"]) ** 2
    w = np.exp(-d2 / (2.0 * h * h)) * np.where(a["is_home_off"][nb] == L["is_home_sim"], ns["TEAM_KERNEL_LAMBDA"], 1.0)
    if "IPW" in ns:
        w = w * np.asarray(ns["IPW"])[nb]
    if "PASS_W" in ns:
        w = w * ns["PASS_W"](None, nb, L["off_sim"], L["def_sim"])
    c = np.cumsum(w)
    if not np.isfinite(c[-1]) or c[-1] <= 0.0:
        return int(nb[rng.integers(len(nb))])
    return int(nb[min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(nb) - 1)])
"""


def weighted_install():
    import inspect

    import mod25d_variance as dv
    import sim09_f2 as f2

    src = inspect.getsource(f2.install_f2)
    old = "return int(nb[rng.integers(len(nb))])"
    assert src.count(old) == 1
    src = src.replace(old, "return WPICK(nb, rng)")
    ns = dict(f2.__dict__)
    ns["sys"] = sys
    ns["dv"] = dv
    exec(WPICK_SRC, ns)
    exec(src, ns)
    return ns["install_f2"]


def patch_f2():
    import mod25d_variance as dv
    import sim09_f2 as f2

    orig_install = weighted_install() if os.environ.get("RS_W") == "1" else f2.install_f2

    def install():
        ns = dv._G["ns"]
        code = np.asarray(dv._G["tables"]["arrays"]["play_type_code"])
        base = ns["DECIDE"]
        last = {"idx": None}

        def inner(idx, *a, **k):
            r = base(idx, *a, **k)
            last["idx"] = r
            return r

        ns["DECIDE"] = inner
        orig_install()
        outer = ns["DECIDE"]

        def counted(idx, *a, **k):
            r = outer(idx, *a, **k)
            i0 = last["idx"]
            if int(code[i0]) in (0, 1):
                C["calls"] += 1
                if int(r) != int(i0):
                    C["f2_changed"] += 1
                if C["calls"] % 4000 == 0:
                    dump(C)
            return r

        ns["DECIDE"] = counted

    f2.install_f2 = install


def patch_f3():
    import sim09_f3 as f3

    orig = f3.Overlay.apply

    def apply(self, d, st):
        out = orig(self, d, st)
        if int(d["play_type_code"]) in (0, 1):
            C["ov_calls"] += 1
            if out is not d:
                if int(out["play_type_code"]) == 6:
                    C["ov_noplay"] += 1
                else:
                    C["ov_counted"] += 1
        return out

    f3.Overlay.apply = apply


def rs_budget(setting):
    import mod25e_crG as crG

    patch_f2()
    patch_f3()
    crG.crG_budget(setting)


def cmd_sim(a):
    import mod25e_crG as crG

    crG.crG_budget = rs_budget
    crG.cmd_sim(a)
    dump(C)


def rs_ss(setting):
    import mod25e_crG as crG

    patch_f2()
    crG.crG_ss(setting)


def cmd_e5(a):
    import mod25d_variance as dv
    import mod25e_crG as crG
    import mod25e_scorestate as ss

    crG.register(a.off)
    dv.DV["crW"] = dict(dv.DV["crG"])
    outd = ss.OUT / f"e5_crW_s{a.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    os.environ["BUD_OUT"] = str(outd)
    os.environ["RS_W"] = "1"
    ss.DV["crW"] = dv.DV["crW"]
    ss.s_init = rs_ss
    a.variant = "crW"
    a.scale = 1.0
    ss.cmd_e5(a)


def cmd_cnt(a):
    tot = {}
    for p in OUT.glob("cnt_*.json"):
        for k, v in json.loads(p.read_text()).items():
            tot[k] = tot.get(k, 0) + v
    print(tot)
    if tot.get("calls"):
        print("f2 replaced share of run/pass draws", tot["f2_changed"] / tot["calls"])
    if tot.get("ov_calls"):
        print("f3 accepted no-play share", tot["ov_noplay"] / tot["ov_calls"], "counted share", tot["ov_counted"] / tot["ov_calls"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("real")
    sub.add_parser("cnt")
    s = sub.add_parser("sim")
    s.add_argument("--off", default="")
    s.add_argument("--worlds", type=int, default=1)
    s.add_argument("--seasons", type=int, default=1)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=31)
    s.add_argument("--out-dir", dest="out_dir", required=True)
    e = sub.add_parser("e5")
    e.add_argument("--off", default="")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    {"real": cmd_real, "sim": cmd_sim, "cnt": cmd_cnt, "e5": cmd_e5}[a.cmd](a)


if __name__ == "__main__":
    main()
