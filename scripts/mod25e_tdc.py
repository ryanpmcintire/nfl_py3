import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_clk as ck  # noqa: E402

OUTDIR = REPO / "artifacts" / "mod25e3" / "tdc"
FIT = OUTDIR / "fit.json"
SALT_TD = 4303
LABEL = os.environ.get("CLK_LABEL", "crHpqokgndecsmfw")


def enabled():
    return os.environ.get("TDC") == "1"


def td_rows():
    R = ck.real_rows()
    R = R[(R.cls == 1) & (R.po > 0)].reset_index(drop=True)
    return R


def term_rows():
    R = ck.real_rows()
    return R[R.cls == 1].reset_index(drop=True)


def build(R, edges, sign, to, a, qs):
    hs, el = R.hs.to_numpy(float), R.el.to_numpy(float)
    q4 = (R.qtr.to_numpy() == 4).astype(int)
    h = ck.hbin(hs, edges) * 2 ** qs + (q4 if qs else 0)
    c = ck.cell(hs, R.sd.to_numpy(), R.oto.to_numpy(), R.dto.to_numpy(), edges, sign, to)
    if qs:
        c = c * 2 + q4
    table = {int(v): el[c == v] for v in np.unique(c)}
    margh = {int(v): el[h == v] for v in np.unique(h)}

    def draw(hs_v, sd, oto, dto, is_q4, u, v):
        q = int(is_q4) if qs else 0
        cv = int(ck.cell([hs_v], [sd], [oto], [dto], edges, sign, to)[0])
        if qs:
            cv = cv * 2 + q
        pool = table.get(cv)
        n = 0 if pool is None else len(pool)
        hv = int(ck.hbin([hs_v], edges)[0]) * 2 ** qs + q
        src = pool if (n and u < n / (n + a)) else margh.get(hv, el)
        return float(min(src[int(v * len(src))], hs_v))

    return draw


def loso_ll(Rtr, Rte, edges, sign, to, a, qs):
    seasons = np.unique(Rte.season.to_numpy())
    tot, n = 0.0, 0
    for s in seasons:
        te = Rte[Rte.season == s]
        tr = Rtr[Rtr.season != s]
        hb_tr = ck.hbin(tr.hs.to_numpy(), edges)
        q_tr = (tr.qtr.to_numpy() == 4).astype(int)
        c_tr = ck.cell(tr.hs.to_numpy(), tr.sd.to_numpy(), tr.oto.to_numpy(), tr.dto.to_numpy(), edges, sign, to)
        b_tr = ck.ebin(tr.el.to_numpy())
        if qs:
            hb_tr = hb_tr * 2 + q_tr
            c_tr = c_tr * 2 + q_tr
        mh = np.zeros((int(hb_tr.max()) + 1, ck.NBIN))
        np.add.at(mh, (hb_tr, b_tr), 1.0)
        margh = (mh + 1.0) / (mh.sum(axis=1, keepdims=True) + ck.NBIN)
        uc, inv = np.unique(c_tr, return_inverse=True)
        cnt = np.zeros((len(uc), ck.NBIN))
        np.add.at(cnt, (inv, b_tr), 1.0)
        nn = cnt.sum(axis=1)
        hb_te = ck.hbin(te.hs.to_numpy(), edges)
        q_te = (te.qtr.to_numpy() == 4).astype(int)
        c_te = ck.cell(te.hs.to_numpy(), te.sd.to_numpy(), te.oto.to_numpy(), te.dto.to_numpy(), edges, sign, to)
        if qs:
            hb_te = hb_te * 2 + q_te
            c_te = c_te * 2 + q_te
        idx = np.minimum(np.searchsorted(uc, c_te), len(uc) - 1)
        hit = uc[idx] == c_te
        be = ck.ebin(te.el.to_numpy())
        cc = np.where(hit, cnt[idx, be], 0.0)
        nh = np.where(hit, nn[idx], 0.0)
        hb_te = np.minimum(hb_te, margh.shape[0] - 1)
        p = (cc + a * margh[hb_te, be]) / (nh + a)
        tot += float(np.log(p).sum())
        n += len(be)
    return tot / n


def fit():
    T, R = td_rows(), term_rows()
    sp = json.loads(ck.FIT.read_text(encoding="utf-8"))["classes"]["term"]["best"]
    base = loso_ll(R, T, ck.EDGES[sp["edges"]], sp["sign"], sp["to"], float(sp["a"]), 0)
    print("td rows", len(T), "term rows", len(R), "mixed-term model scored on td rows", round(base, 4), flush=True)
    res = []
    for en, sg, to, a, qs in itertools.product(ck.EDGES, ck.SIGNS, ck.TOS, ck.SMOOTH, (0, 1)):
        res.append({"edges": en, "sign": sg, "to": to, "a": a, "qs": qs, "ll": loso_ll(T, T, ck.EDGES[en], sg, to, a, qs)})
    res.sort(key=lambda r: -r["ll"])
    pooled = max((r for r in res if r["qs"] == 0), key=lambda r: r["ll"])
    out = {"n_td": int(len(T)), "n_term": int(len(R)), "baseline_mixed_ll": base, "best": res[0], "pooled_best": pooled, "top5": res[:5], "looks": len(res) + 1, "mixed_spec": sp}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("best", res[0])
    print("pooled best", pooled)
    print("looks", out["looks"])


def spec_draw():
    f = json.loads(FIT.read_text(encoding="utf-8"))["best"]
    return build(td_rows(), ck.EDGES[f["edges"]], f["sign"], f["to"], float(f["a"]), int(f["qs"]))


def install_tdc():
    import mod25d_variance as dv
    import sim09_u4g as u4

    draw = spec_draw()
    probs = u4.load_probs()
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        offense, off_to, def_to = fr.f_locals["offense"], fr.f_locals["off_to"], fr.f_locals["def_to"]  # noqa: F841
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr not in (2, 4) or code not in (0, 1) or not drawn["points_off"] > 0:
            return drawn
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if hs <= 0:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT_TD, cseed)
        rng = st["rng"]
        u, v = rng.random(), rng.random()
        new = draw(hs, score_diff, float(off_to), float(def_to), qtr == 4, u, v)
        t = u4.WARN_AT.get(qtr)
        if t is not None and clock_val > t and clock_val - new < t and rng.random() < probs[u4.cell_key(qtr, code, True)]:
            new = float(clock_val - t)
        old = float(drawn["clock_elapsed"])
        drawn = dict(drawn)
        drawn["clock_elapsed"] = new
        lg = dv._G.get("log")
        if lg:
            tp = list(lg[-1])
            tp[10] = new
            lg[-1] = tuple(tp)
        es = dv._G.get("egst")
        if es is not None and es["ps"][0] is not None and abs(es["ps"][0] - (clock_val - old)) < 1e-6:
            import mod25e_endgame as eg

            stn = float(eg.stop_after(code, float(drawn["yards_gained"]), bool(drawn["flip"]), True, float(drawn["off_to_used"]), float(drawn["def_to_used"]), new))
            es["ps"] = (clock_val - new, stn)
        return drawn

    dv._G["pol"] = pol


def validate():
    import pandas as pd

    art = REPO / "artifacts" / "mod25e3"
    T, R = td_rows(), term_rows()
    sp = json.loads(ck.FIT.read_text(encoding="utf-8"))["classes"]["term"]["best"]
    old = build(R, ck.EDGES[sp["edges"]], sp["sign"], sp["to"], float(sp["a"]), 0)
    new = spec_draw()
    rows = []
    for f in sorted((art / f"e5_{LABEL}_s11").glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < 2:
            continue
        d = pd.read_parquet(f, columns=["qtr", "gsr", "code", "po", "sd", "oto", "dto"])
        d = d[(d.qtr.isin([2, 4])) & (d.code <= 1) & (d.po > 0)]
        rows.append(d)
    S = pd.concat(rows)
    S["hs"] = np.where(S.qtr == 2, S.gsr - 1800.0, S.gsr)
    S = S[S.hs > 0].reset_index(drop=True)
    rng = np.random.default_rng(SALT_TD)
    reps = 20
    S["el_old"] = [np.mean([old(h, sd, o, dd, q == 4, rng.random(), rng.random()) for _ in range(reps)]) for h, sd, o, dd, q in zip(S.hs, S.sd, S.oto, S.dto, S.qtr)]
    S["el_new"] = [np.mean([new(h, sd, o, dd, q == 4, rng.random(), rng.random()) for _ in range(reps)]) for h, sd, o, dd, q in zip(S.hs, S.sd, S.oto, S.dto, S.qtr)]
    S["cut_old"] = [np.mean([old(h, sd, o, dd, q == 4, rng.random(), rng.random()) >= h for _ in range(reps)]) for h, sd, o, dd, q in zip(S.hs, S.sd, S.oto, S.dto, S.qtr)]
    S["cut_new"] = [np.mean([new(h, sd, o, dd, q == 4, rng.random(), rng.random()) >= h for _ in range(reps)]) for h, sd, o, dd, q in zip(S.hs, S.sd, S.oto, S.dto, S.qtr)]
    T = T.assign(cut=(T.el >= T.hs - 1e-9).astype(float))
    lines = [f"sim TD states {len(S)} (s11, qtr 2/4), real TD rows {len(T)}; reps {reps}"]
    for q in (2, 4):
        for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 120), (120, 300)):
            ms = S[(S.qtr == q) & (S.hs > lo) & (S.hs <= hi)]
            mr = T[(T.qtr == q) & (T.hs > lo) & (T.hs <= hi)]
            if len(ms) and len(mr):
                lines.append(f"q{q} hs{lo}-{hi} n sim/real {len(ms)}/{len(mr)}  half-ends after TD  old {ms.cut_old.mean():.3f} new {ms.cut_new.mean():.3f} real {mr.cut.mean():.3f}   mean elapsed old {ms.el_old.mean():.1f} new {ms.el_new.mean():.1f} real {mr.el.mean():.1f}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "validate.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"fit": fit, "validate": validate}[sys.argv[1]]()
