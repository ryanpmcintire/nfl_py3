import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402

OUT = Path(os.environ.get("BUD_OUT", str(REPO / "artifacts" / "mod25e")))
CAP = {}


def e_init(setting):
    dv.d_gen_init(setting)
    ns = dv._G["ns"]
    orig = ns["run_one_game"]
    CAP["rows"] = []

    def wrapped(*a, **k):
        rec, cap = orig(*a, **k)
        lg = np.array(dv._G["log"], dtype=np.float64).reshape(-1, 12)
        pl = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
        CAP["rows"].append((lg, pl, float(rec["total"]), float(rec["margin"])))
        return rec, cap

    ns["run_one_game"] = wrapped


def e_play_season(task):
    CAP["rows"].clear()
    res = dv.d_play_season(task)
    world, sidx = task[0], task[1]
    frames = []
    for gi, (lg, pl, tot, mar) in enumerate(CAP["rows"]):
        n = min(len(lg), len(pl))
        f = np.column_stack([np.full(n, gi), lg[:n, :11], pl[:n, 0], np.full(n, tot), np.full(n, mar)])
        frames.append(f)
    a = np.concatenate(frames)
    cols = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "el", "offhome", "total", "margin"]
    d = pd.DataFrame(a, columns=cols)
    d.to_parquet(OUT / f"play_{world}_{sidx}.parquet")
    return res


def cmd_sim(args):
    import mod25_generator as gen

    OUT.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    cfgj = json.dumps(dict(dv.DV[args.variant], seed=args.seed, name=f"{args.variant}_s{args.scale:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": args.scale, "yard_gain": args.yard_gain, "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = e_init
    gen.play_season = e_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def build_real(seasons):
    pbp = dv.sim.load_reg_seasons(tuple(seasons))
    tr = dv.sim.build_transition_frame(pbp).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    sch = pd.read_parquet(REPO / "data" / "raw" / "20260908T162105Z" / "schedules.parquet")
    gi = pbp.drop_duplicates("game_id").set_index("game_id")[["season", "home_team", "away_team"]]
    gids = pd.Index(tr["game_id"].unique())
    gm = gi.loc[gids].join(sch.set_index("game_id")[["home_score", "away_score"]], how="left")
    gnum, _ = pd.factorize(tr["game_id"])
    pp = pbp[["game_id", "play_id", "posteam", "home_team"]].drop_duplicates(["game_id", "play_id"])
    mm = tr[["game_id", "play_id"]].merge(pp, on=["game_id", "play_id"], how="left")
    oh = (mm["posteam"] == mm["home_team"]).to_numpy().astype(int)
    T = pd.DataFrame({"sid": gm["season"].to_numpy()[gnum], "g": gnum, "qtr": tr["qtr_actual"].to_numpy(), "offhome": oh, "sd": tr["sc_raw"].to_numpy(float), "po": tr["points_off"].to_numpy(float), "pdf": tr["points_def"].to_numpy(float), "flip": tr["possession_flip"].to_numpy().astype(bool), "gsr": tr["gsr_actual"].to_numpy()})
    chk = float(np.mean(np.isclose(T["sd"] * np.where(T["offhome"] == 1, 1, -1), tr["home_margin_pre"])))
    print("real rows", len(T), "games", len(gm), "missing scores", int(gm["home_score"].isna().sum()), "hm_pre match", chk, flush=True)
    G = gm.reset_index(drop=True).rename(columns={"season": "sid"})
    G["g"] = np.arange(len(G))
    return T, G.dropna(subset=["home_score"])


def build_sim(minsid):
    gms = pd.read_parquet(OUT / "sim_games.parquet")
    frames = []
    for f in sorted(OUT.glob("play_*_*.parquet")):
        _, w, s = f.stem.split("_")
        w, s = int(w), int(s)
        if s < minsid:
            continue
        d = pd.read_parquet(f)
        d["sid"] = w * 1000 + s + 1
        frames.append(d)
    T = pd.concat(frames, ignore_index=True)
    T["g"] = T["g"].astype(np.int64)
    T["offhome"] = T["offhome"].astype(int)
    T["flip"] = T["flip"].astype(bool)
    gms = gms[gms["season"].isin(T["sid"].unique())].copy()
    gms["g"] = gms["game_id"].str[-3:].astype(int)
    return T, gms.rename(columns={"season": "sid"})


def cmd_budget(args):
    from mod25e_analysis import per_game, season_re, stats

    rng = np.random.default_rng(7)
    sets = {}
    for nm, ss in (("real_eval", dv.EVAL), ("real_train", dv.TRAIN)):
        T, G = build_real(ss)
        D, X = per_game(T, G)
        sets[nm] = (D, X, season_re(G))
    T, G = build_sim(2)
    D, X = per_game(T, G)
    sets["sim"] = (D, X, season_re(G))
    for nm, (D, X, re_) in sets.items():
        print(nm, len(X), "seasons", D["sid"].nunique(), flush=True)
    point = {nm: stats(D, X, list(re_.values())) for nm, (D, X, re_) in sets.items()}
    sidx = {}
    for nm, (D, X, re_) in sets.items():
        Dg = {k: v for k, v in D.groupby("sid")}
        Xg = {k: v for k, v in X.groupby("sid")}
        sidx[nm] = (Dg, Xg, re_)
    boots = {nm: [] for nm in sets}
    for b in range(args.boot):
        for nm, (Dg, Xg, re_) in sidx.items():
            ks = list(Dg.keys())
            sel = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            Db = pd.concat([Dg[k] for k in sel], ignore_index=True)
            Xb = pd.concat([Xg[k] for k in sel], ignore_index=True)
            boots[nm].append(stats(Db, Xb, [re_[k] for k in sel if k in re_]))
    rows = []
    for ref in ("real_eval", "real_train"):
        for k in point["sim"]:
            br = np.array([x[k] for x in boots[ref]])
            bs = np.array([x[k] for x in boots["sim"]])
            d = bs - br
            d = d[np.isfinite(d)]
            rows.append({"ref": ref, "term": k, "real": point[ref][k], "sim": point["sim"][k], "diff": point["sim"][k] - point[ref][k], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "p_pos": float((d > 0).mean())})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "budget.csv", index=False)
    pd.set_option("display.width", 200)
    for ref in ("real_eval", "real_train"):
        print(ref)
        print(R[R["ref"] == ref].drop(columns="ref").round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crj")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--worlds", type=int, default=8)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=9)
    s.add_argument("--yard-gain", dest="yard_gain", type=float, default=0.0)
    b = sub.add_parser("budget")
    b.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    args = ap.parse_args()
    if args.out_dir:
        os.environ["BUD_OUT"] = str(Path(args.out_dir).resolve())
        global OUT
        OUT = Path(os.environ["BUD_OUT"])
    {"sim": cmd_sim, "budget": cmd_budget}[args.cmd](args)


if __name__ == "__main__":
    main()
