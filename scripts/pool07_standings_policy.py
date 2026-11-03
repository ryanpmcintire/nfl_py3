import argparse
import json
from pathlib import Path

import numpy as np
from pool01_forward_sim import K, load_games, past_state, rival_table, share_given_key

OUT = Path("artifacts/pool07")
GAMES = 15


def policies(cuts):
    pol = [("a_max_prob", None, "prob", False), ("c_bp_leverage", None, "lev", False)]
    for name, cut in cuts.items():
        pol.append((f"b_tilt_{name}", cut, "prob", False))
    pol.append(("d_trailing_tilt_q50", cuts["q50"], "prob", True))
    return pol


def play(g, rng, state, pol, weeks, dm, dsd, our_points, our_bp0):
    P, E = state[0].shape
    delta = dm + dsd * rng.standard_normal((P, E))
    p, sh, uo, ut = (g[k] for k in ("p", "share", "u_o", "u_t"))
    n = len(p)
    res = {}
    for name in [x[0] for x in pol]:
        res[name] = {
            "pts": np.full(P, float(our_points)),
            "bp": np.full(P, float(our_bp0)),
            "wins": np.zeros(P),
        }
    fp, fb = state[0].copy(), state[1].copy()
    for _ in range(weeks):
        idx = rng.integers(0, n, size=(P, GAMES))
        pw, shw, uow, utw = p[idx], sh[idx], uo[idx], ut[idx]
        cover = rng.random(pw.shape) < pw
        thr = np.where(cover, shw, 1.0 - shw)
        tp = np.clip(thr[:, None, :] + delta[:, :, None], 0.0, 1.0)
        fp += (rng.random(tp.shape, dtype=np.float32) < tp).sum(axis=2)
        eo, et = np.exp(uow), np.exp(utw)
        z = (eo + et).sum(axis=1, keepdims=True)
        rate = (np.where(cover, eo, et) / z).sum(axis=1)
        fb += rng.random((P, E)) < rate[:, None]
        for name, cut, bpmode, trailing in pol:
            r = res[name]
            if cut is None:
                pick_o = np.ones(pw.shape, bool)
            else:
                near = np.abs(pw - 0.5) < cut
                tilt = near & (shw > 0.5)
                if trailing:
                    ours = r["pts"] * K + r["bp"]
                    top = (fp * K + fb).max(axis=1)
                    behind = ours < top
                    tilt = tilt & behind[:, None]
                pick_o = ~tilt
            win = np.where(pick_o, cover, ~cover)
            wp = np.where(pick_o, pw, 1.0 - pw)
            if bpmode == "prob":
                score = wp
            else:
                own = np.where(pick_o, eo, et) / z
                score = wp * (1.0 - own)
            j = np.argmax(score, axis=1)
            rows = np.arange(P)
            r["pts"] += win.sum(axis=1)
            r["bp"] += win[rows, j]
            r["wins"] += win.sum(axis=1)
    out = {}
    fkey = fp * K + fb
    for name, r in res.items():
        ours = r["pts"] * K + r["bp"]
        first = share_given_key(fkey, ours)
        greater = (fkey > ours[:, None]).sum(axis=1)
        equal = (fkey == ours[:, None]).sum(axis=1)
        out[name] = {
            "first": first,
            "top10": (greater < 10).astype(float),
            "rank": 1.0 + greater + 0.5 * equal,
            "wins": r["wins"],
        }
    return out


def summ(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]


def main():
    ap = argparse.ArgumentParser(description="POOL-07 standings-conditional policy challenger")
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--weeks", type=int, default=9)
    ap.add_argument("--points", type=int, default=65)
    ap.add_argument("--rank", type=int, default=19)
    ap.add_argument("--entries", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    df = load_games()
    rng = np.random.default_rng(a.seed)
    E = a.entries - 1
    table = rival_table(a.points)
    kp, kb, kd, have = [], [], [], 0
    while have < a.paths:
        pp, bb, dd, dsd, _ = past_state(df, rng, E, 2000, a.points, a.rank, table)
        kp.append(pp)
        kb.append(bb)
        kd.append(dd)
        have += len(pp)
    state = (np.concatenate(kp)[: a.paths], np.concatenate(kb)[: a.paths])
    dm = np.concatenate(kd)[: a.paths]
    our_bp = float(
        sum(g["cov_o"].to_numpy()[int(np.argmax(g["p"].to_numpy()))] for _, g in df.groupby("week"))
    )
    arms = {
        "base": (df, dm, dsd),
        "no_skill": (df, np.zeros_like(dm), 0.0),
        "shares_weeks_1_4": (df[df["week"] <= 4], dm, dsd),
        "shares_weeks_5_8": (df[df["week"] >= 5], dm, dsd),
    }
    result = {"start": vars(a), "dsd": dsd, "paid_places": 1, "arms": {}}
    looks = 0
    for arm, (sub, dmm, dss) in arms.items():
        g = {k: sub[k].to_numpy() for k in ("p", "share", "u_o", "u_t")}
        dev = np.abs(g["p"] - 0.5)
        cuts = {f"q{int(q * 100)}": float(np.quantile(dev, q)) for q in (0.25, 0.5, 0.75)}
        pol = policies(cuts)
        out = play(g, rng, state, pol, a.weeks, dmm, dss, a.points, our_bp)
        base = out["a_max_prob"]
        rows = {}
        for name, o in out.items():
            looks += 1
            rows[name] = {
                "p_first": summ(o["first"]),
                "p_top10": summ(o["top10"]),
                "exp_rank": summ(o["rank"]),
                "exp_wins": summ(o["wins"]),
                "d_first_vs_a": summ(o["first"] - base["first"]),
                "d_top10_vs_a": summ(o["top10"] - base["top10"]),
                "d_wins_vs_a": summ(o["wins"] - base["wins"]),
            }
        result["arms"][arm] = {"cuts": cuts, "n_games": len(sub), "policies": rows}
    result["looks"] = looks
    result["paths"] = len(dm)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pool07_standings_policy.json").write_text(json.dumps(result, indent=1) + "\n")
    for arm, v in result["arms"].items():
        print(arm, v["cuts"])
        for n, r in v["policies"].items():
            print(
                f"  {n:22s} first {r['p_first'][0]:.4f}+-{r['p_first'][1]:.4f} "
                f"top10 {r['p_top10'][0]:.4f}+-{r['p_top10'][1]:.4f} "
                f"rank {r['exp_rank'][0]:.1f} wins {r['exp_wins'][0]:.2f} "
                f"dFirst {r['d_first_vs_a'][0]:+.4f}+-{r['d_first_vs_a'][1]:.4f} "
                f"dWins {r['d_wins_vs_a'][0]:+.2f}"
            )


if __name__ == "__main__":
    main()
