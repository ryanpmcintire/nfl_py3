import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_budget as bg  # noqa: E402
import mod25e_e96 as e96  # noqa: E402
from mod25e_analysis import per_game, season_re  # noqa: E402

E3 = REPO / "artifacts" / "mod25e3"
OUTD = E3 / os.environ.get("E100_OUT", "e100")
BASE = os.environ.get("E100_BASE", "crHpqokgndecsmfwtjo2as2ypw2")
CAND = os.environ.get("E100_CAND", BASE + "xfdcd")
SEEDS = (11, 12, 13)
NBOOT = 1000
BURN_SID = 2
KEYS_STAT = ["margin_sd", "noise_re_var", "strength_re_var", "xq_cov_sum", "q4_slope", "mass_3", "pts_game"]
rng = np.random.default_rng(100)
LINES = []


def say(s=""):
    LINES.append(s)
    print(s, flush=True)


def lean(X, re_vals):
    m = X["margin"].to_numpy(float)
    mr = np.where(np.isnan(X["mreg"].to_numpy(float)), m, X["mreg"].to_numpy(float))
    incs = []
    prev = np.zeros(len(X))
    for q in (1, 2, 3):
        d = np.where(np.isnan(X[f"m{q}"].to_numpy(float)), mr, X[f"m{q}"].to_numpy(float))
        incs.append(d - prev)
        prev = d
    incs.append(mr - prev)
    m3 = prev
    inc4 = mr - m3
    return {
        "margin_sd": float(m.std(ddof=1)),
        "noise_re_var": float(np.mean([v[1] for v in re_vals])),
        "strength_re_var": float(np.mean([v[0] for v in re_vals])),
        "xq_cov_sum": float(mr.var(ddof=1) - sum(i.var(ddof=1) for i in incs)),
        "q4_slope": float(np.cov(inc4, m3)[0, 1] / m3.var(ddof=1)),
        "mass_3": float(np.mean(np.abs(m) == 3)),
        "pts_game": float((X["home_score"].to_numpy(float) + X["away_score"].to_numpy(float)).mean()),
    }


def load(label, seed):
    bg.OUT = E3 / f"e5_{label}_s{seed}"
    T, G = bg.build_sim(BURN_SID)
    D, X = per_game(T, G)
    re_ = season_re(G)
    Xg = {k: v for k, v in X.groupby("sid")}
    return Xg, re_


def boot_means(data, nb):
    out = np.zeros((nb, len(KEYS_STAT)))
    for b in range(nb):
        acc = []
        for Xg, re_ in data:
            ks = list(Xg.keys())
            sel = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            import pandas as pd

            Xb = pd.concat([Xg[k] for k in sel], ignore_index=True)
            r = lean(Xb, [re_[k] for k in sel if k in re_])
            acc.append([r[k] for k in KEYS_STAT])
        out[b] = np.mean(acc, axis=0)
    return out


def part1():
    say("PART 1 base " + BASE + " -> cand " + CAND + f"; season bootstrap within seed, seeds {SEEDS}, {NBOOT} reps, seed-mean of 3 per rep")
    data, point, perseed = {}, {}, {}
    for nm, lab in (("base", BASE), ("cand", CAND)):
        data[nm] = [load(lab, s) for s in SEEDS]
        perseed[nm] = [lean(__import__("pandas").concat(list(Xg.values()), ignore_index=True), list(re_.values())) for Xg, re_ in data[nm]]
        point[nm] = {k: float(np.mean([p[k] for p in perseed[nm]])) for k in KEYS_STAT}
    bb = boot_means(data["base"], NBOOT)
    bc = boot_means(data["cand"], NBOOT)
    d = bc - bb
    res = {}
    say(f"{'metric':16s} {'base':>9s} {'cand':>9s} {'delta':>8s} {'90% interval':>20s} {'P(delta>0)':>10s} {'sd_base':>8s} {'sd_cand':>8s}  per-seed base | per-seed cand | per-seed delta")
    for j, k in enumerate(KEYS_STAT):
        lo, hi = np.quantile(d[:, j], [0.05, 0.95])
        pb = [p[k] for p in perseed["base"]]
        pc = [p[k] for p in perseed["cand"]]
        dl = point["cand"][k] - point["base"][k]
        res[k] = {"base": point["base"][k], "cand": point["cand"][k], "delta": dl, "lo90": float(lo), "hi90": float(hi), "p_pos": float((d[:, j] > 0).mean()), "per_seed_base": pb, "per_seed_cand": pc}
        say(f"{k:16s} {point['base'][k]:9.4f} {point['cand'][k]:9.4f} {dl:+8.4f} [{lo:+9.4f},{hi:+9.4f}] {(d[:, j] > 0).mean():10.3f} {bb[:, j].std():8.4f} {bc[:, j].std():8.4f}  " + " ".join(f"{x:.3f}" for x in pb) + " | " + " ".join(f"{x:.3f}" for x in pc) + " | " + " ".join(f"{y - x:+.3f}" for x, y in zip(pb, pc)))
    say("seed-pair spread (3 paired seed deltas) for late r2 from e5.json (no play-level epa saved, season bootstrap unavailable):")
    r2 = {}
    for nm, lab in (("base", BASE), ("cand", CAND)):
        r2[nm] = [json.loads((E3 / f"e5_{lab}_s{s}" / "e5.json").read_text())["gate"]["r2_w10_18"] for s in SEEDS]
    dd = np.array(r2["cand"]) - np.array(r2["base"])
    sb = rng.integers(0, 3, (NBOOT, 3))
    bm = np.array([np.mean(np.array(r2["cand"])[i]) - np.mean(np.array(r2["base"])[j]) for i, j in zip(sb, rng.integers(0, 3, (NBOOT, 3)))])
    lo, hi = np.quantile(bm, [0.05, 0.95])
    say(f"r2_w10_18 base {np.mean(r2['base']):.4f} cand {np.mean(r2['cand']):.4f} delta {dd.mean():+.4f} seed-resample 90% [{lo:+.4f},{hi:+.4f}] P(>0) {(bm > 0).mean():.3f} (only 3 seeds) per-seed base {np.round(r2['base'], 3).tolist()} cand {np.round(r2['cand'], 3).tolist()}")
    res["r2_w10_18"] = {"base": float(np.mean(r2["base"])), "cand": float(np.mean(r2["cand"])), "delta": float(dd.mean()), "lo90": float(lo), "hi90": float(hi), "p_pos": float((bm > 0).mean()), "per_seed_base": r2["base"], "per_seed_cand": r2["cand"]}
    (OUTD / "part1.json").write_text(json.dumps(res, indent=1))


def cells(ev, g, G, W):
    am = np.abs(g.margin.to_numpy())
    reg = np.abs(np.bincount(ev.gi.to_numpy()[ev.q.to_numpy() <= 4], weights=ev.spts.to_numpy()[ev.q.to_numpy() <= 4], minlength=G))
    ent = np.abs(e96.entering(ev, G, 300.0))
    tie = (reg == 0).astype(float)
    tl5 = (ent == 0).astype(float)
    m3 = (am == 3).astype(float)
    out = {
        "mass3": (W @ m3) / G,
        "P(tied after regulation)": (W @ tie) / G,
        "mass3 from OT games per game": (W @ (tie * m3)) / G,
        "OT share of mass3": (W @ (tie * m3)) / (W @ m3),
        "mass3 regulation-decided per game": (W @ ((1 - tie) * m3)) / G,
        "P(final3 | tied after reg)": (W @ (tie * m3)) / (W @ tie),
        "P(final3 | tied entering last 5 min)": (W @ (tl5 * m3)) / (W @ tl5),
        "P(tied entering last 5 min)": (W @ tl5) / G,
    }
    return out


def part2():
    say("")
    say("PART 2 E96 cells; real REG 2011-17, sim burn 2, game bootstrap " + str(e96.NB))
    evr, gr = e96.real_events()
    evr, gr, _ = e96.build(e96.classify(evr), gr)
    res = {"real": cells(evr, gr, len(gr), e96.boot_w(len(gr)))}
    for nm, lab in (("base", BASE), ("cand", CAND)):
        e96.LABEL = lab
        evs, gs = e96.sim_events()
        evs, gs, _ = e96.build(e96.classify(evs), gs)
        res[nm] = cells(evs, gs, len(gs), e96.boot_w(len(gs)))
    say(f"{'cell':40s} {'real':>26s} {'base ypw2':>26s} {'cand xfdcd':>26s}  cand-base 90% interval, P(>0)")
    for k in res["real"]:
        dc = res["cand"][k] - res["base"][k]
        lo, hi = np.quantile(dc[1:], [0.05, 0.95])
        say(f"{k:40s} {e96.fmt(res['real'][k]):>26s} {e96.fmt(res['base'][k]):>26s} {e96.fmt(res['cand'][k]):>26s}  {dc[0]:+.4f} [{lo:+.4f},{hi:+.4f}] {np.mean(dc[1:] > 0):.2f}")
    (OUTD / "part2.json").write_text(json.dumps({a: {k: float(v[0]) for k, v in b.items()} for a, b in res.items()}, indent=1))


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    part1()
    part2()
    (OUTD / "e100.txt").write_text("\n".join(LINES))


if __name__ == "__main__":
    main()
