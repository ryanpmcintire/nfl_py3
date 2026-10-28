import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25_generator as gen  # noqa: E402
import mod25_pipeline as pipe  # noqa: E402
import sim10_drive as sd  # noqa: E402

SYN = REPO / "data" / "processed" / "synthetic" / "sim11_v3"
OUT = REPO / "artifacts" / "mod25_pipeline_v3"
STRENGTH = Path(
    os.environ.get("MOD25_STRENGTH", str(REPO / "artifacts" / "sim11" / "strength_terms.json"))
)
BANDWIDTHS = Path(os.environ.get("MOD25_BANDWIDTHS", str(sd.OUT / "bandwidths_week.json")))
LATENT = Path(os.environ.get("MOD25_LATENT", str(sd.OUT / "latent_fit_2009_25.json")))
Q4DEC = Path(os.environ.get("MOD25_Q4DEC", str(sd.OUT / "q4dec_fg.json")))
SEED_BASE = 8000
ENGINE_FILES = ("sim10_drive.py", "mod25_generator.py", "mod25_pipeline.py")


def engine_hashes():
    out = {
        f: hashlib.sha256((REPO / "scripts" / f).read_bytes()).hexdigest()[:16]
        for f in ENGINE_FILES
    }
    out["strength_terms.json"] = hashlib.sha256(STRENGTH.read_bytes()).hexdigest()[:16]
    return out


def engine_params():
    h, ho, g, hr = sd.load_h(str(BANDWIDTHS))
    strength = json.loads(STRENGTH.read_text())
    return (
        h,
        ho,
        g,
        hr,
        (5,),
        True,
        True,
        (1, 4),
        True,
        True,
        "outcome",
        strength,
        True,
        False,
        True,
        False,
    )


def sim_world_truth(task):
    w, wseed, n_seasons = task
    eng = sd._W["eng"]
    fit = sd.load_latent_fit()
    setting = dict(gen.SETTING_DEFAULTS)
    grows, prow, ptm = [], [], []
    rng = np.random.default_rng(wseed)
    lat = gen.gen_world_latents(rng, n_seasons, fit, setting)
    gseeds = wseed.spawn(n_seasons)
    for sidx, (weekly, qb_out) in enumerate(lat):
        season = sd.SEASONS[(w * n_seasons + sidx) % len(sd.SEASONS)]
        sched = gen.make_schedule(rng)
        ratings = gen.season_ratings(weekly, qb_out, fit, setting, 0.0, 0.0)
        grng = np.random.default_rng(gseeds[sidx])
        sid = w * 1000 + sidx + 1
        for gi, (week, h, a) in enumerate(sched):
            rh, ra = ratings[week][h], ratings[week][a]
            hs, as_, ms, reg, ot, pl = eng.game(grng, (rh, ra), season, week)
            gidn = f"w{w:04d}s{sidx:02d}g{gi:03d}"
            dl = (rh[0] + ra[1]) - (ra[0] + rh[1])
            grows.append(
                (
                    gidn,
                    sid,
                    week,
                    f"W{w}T{h}",
                    f"W{w}T{a}",
                    hs,
                    as_,
                    ms[0],
                    ms[1],
                    ms[2],
                    reg if ot else np.nan,
                    ot,
                    season,
                    dl,
                    rh[0] - ra[0],
                    rh[1] - ra[1],
                )
            )
            if len(pl):
                prow.append(pl[:, 0].astype(np.int64))
                ptm.append((gidn, f"W{w}T{h}", f"W{w}T{a}", pl[:, 1].astype(np.int8)))
    return w, grows, prow, ptm


def sim_chunk(n_worlds, n_seasons, seed, workers):
    import multiprocessing as mp

    sd.os.environ.setdefault("QBC", "1")
    sd.os.environ["SIM10_FORM"] = "0"
    sd.os.environ["SIM10_HECARRY"] = "1"
    sd.os.environ["SIM10_LATENT"] = str(LATENT)
    sd.os.environ["SIM10_Q4DEC"] = str(Q4DEC)
    params = engine_params()
    ws = np.random.SeedSequence(seed).spawn(n_worlds)
    tasks = [(w, ws[w], n_seasons) for w in range(n_worlds)]
    if workers <= 1:
        sd._init(*params)
        out = [sim_world_truth(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=sd._init, initargs=params) as pool:
            out = pool.map(sim_world_truth, tasks, chunksize=1)
    out.sort(key=lambda r: r[0])
    grows, prow, ptm = [], [], []
    for _, g_, p_, t_ in out:
        grows += g_
        prow += p_
        ptm += t_
    cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "m1",
        "m2",
        "m3",
        "mreg",
        "went_ot",
        "real_season",
        "lat_d",
        "dxo",
        "dxd",
    ]
    G = pd.DataFrame(grows, columns=cols)
    G["gameday"] = pd.Timestamp("2000-09-05") + pd.to_timedelta(
        (G["season"] % 1000 - 1) * 365 + (G["week"] - 1) * 7, unit="D"
    )
    rows = np.concatenate(prow)
    P = (
        pd.read_parquet(sd.OUT / "plays.parquet")
        .set_index("row")
        .reindex(rows)
        .reset_index(drop=True)
    )
    n = np.array([len(x[3]) for x in ptm])
    P["game_id"] = np.repeat([x[0] for x in ptm], n)
    side = np.concatenate([x[3] for x in ptm])
    P["posteam"] = np.where(
        side == 0, np.repeat([x[1] for x in ptm], n), np.repeat([x[2] for x in ptm], n)
    )
    return G, P


def produce_chunk(k, worlds, seasons, workers):
    seed = SEED_BASE + k
    t0 = time.time()
    games, plays = sim_chunk(worlds, seasons, seed, workers)
    elapsed = time.time() - t0
    pre = f"c{k:02d}"
    games["game_id"] = pre + games["game_id"]
    games["home_team"] = pre + games["home_team"]
    games["away_team"] = pre + games["away_team"]
    plays["game_id"] = pre + plays["game_id"]
    plays["posteam"] = pre + plays["posteam"]
    ts = gen.team_stats_from_plays(plays)
    d_ = SYN / f"chunk_{k:02d}"
    d_.mkdir(parents=True, exist_ok=True)
    games["world"] = games["game_id"].str.slice(0, 8)
    games["sidx"] = games["season"] % 1000 - 1
    games["chunk"] = k
    games.to_parquet(d_ / "games.parquet")
    ts.to_parquet(d_ / "team_stats.parquet")
    plays.astype({"epa": "float32", "yards_gained": "float32"}).to_parquet(d_ / "plays.parquet")
    games[["game_id", "world", "sidx", "lat_d", "dxo", "dxd"]].to_parquet(d_ / "latents.parquet")
    settings = {
        "locked_defaults": "sim --tilt 5 --models 1 --pat 1 --mufree 1,4 --dr 1 --qstrata 1"
        " --eoh outcome --droof 1 --drmu 0 --mulabel 1 --realsched 0 --form 0 --hecarry 1"
        " --latent latent_fit_2009_25.json --q4dec q4dec_fg.json",
        "strength": str(STRENGTH.relative_to(REPO)),
        "bandwidths": str(BANDWIDTHS.relative_to(REPO)),
        "latent": str(LATENT.relative_to(REPO)),
        "q4dec": str(Q4DEC.relative_to(REPO)),
        "window": list(sd.SEASONS),
        "latent_setting": gen.SETTING_DEFAULTS,
    }
    meta = {
        "k": k,
        "seed": seed,
        "worlds": worlds,
        "seasons": seasons,
        "games": len(games),
        "plays": len(plays),
        "elapsed": elapsed,
        "engine": "sim11",
        "engine_sha": engine_hashes(),
        "settings": settings,
    }
    (d_ / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def cmd_produce(args):
    t0 = time.time()
    for k in range(args.start, args.start + args.chunks):
        t1 = time.time()
        m = produce_chunk(k, args.worlds, args.seasons, args.workers)
        print(
            f"chunk {k} games {m['games']} plays {m['plays']} {(time.time() - t1) / 60:.2f} min",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "produce_log.json").write_text(
        json.dumps(
            {"chunks": args.chunks, "start": args.start, "minutes": (time.time() - t0) / 60},
            indent=1,
        )
    )


def cmd_truth(args):
    parts = []
    for d_ in sorted(SYN.glob("chunk_*")):
        g = pd.read_parquet(d_ / "games.parquet")
        lat = pd.read_parquet(d_ / "latents.parquet")
        parts.append(g[["game_id", "home_score", "away_score"]].merge(lat, on="game_id"))
    t = pd.concat(parts, ignore_index=True)
    y = (t["home_score"] - t["away_score"]).to_numpy(float)
    X = np.column_stack([np.ones(len(t)), t["dxo"], t["dxd"]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    t["true_em"] = X @ coef
    t[["game_id", "true_em", "dxo", "dxd"]].to_parquet(SYN / "truth.parquet")
    rep = {
        "games": len(t),
        "intercept_hfa": float(coef[0]),
        "coef_off_diff": float(coef[1]),
        "coef_def_diff": float(coef[2]),
        "true_em_sd": float(t["true_em"].std()),
        "true_em_mean": float(t["true_em"].mean()),
        "result_sd": float(y.std()),
        "r2_true_em_vs_result": float(
            1 - ((y - t["true_em"]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truth_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("produce")
    p.add_argument("--worlds", type=int, default=6)
    p.add_argument("--seasons", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--chunks", type=int, default=8)
    sub.add_parser("truth")
    q = sub.add_parser("features")
    q.add_argument("--workers", type=int, default=6)
    sub.add_parser("student2")
    sub.add_parser("diag")
    gr = sub.add_parser("grade")
    gr.add_argument("--arm", required=True)
    ap.add_argument("--syn", default=str(SYN))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    globals()["SYN"] = Path(args.syn)
    globals()["OUT"] = Path(args.out)
    pipe.SYN = SYN
    pipe.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "produce":
        cmd_produce(args)
    elif args.cmd == "truth":
        cmd_truth(args)
    else:
        getattr(pipe, "cmd_" + args.cmd)(args)


if __name__ == "__main__":
    main()
