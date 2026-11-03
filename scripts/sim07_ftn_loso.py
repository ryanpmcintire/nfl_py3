import json
import time

import numpy as np
import sim07_ftn_state as s
import sim07_personnel as m
import sim07_power as p

BOOT_GAME = 300
OUT = m.OUT / "ftn_loso_summary.json"


def build():
    raw = m.build()
    raw = raw[raw.season.isin(s.SEASONS)].copy()
    j = raw.merge(s.load_ftn(), on=["game_id", "play_id"], how="left")
    j = j[j.n_defense_box.notna()]
    j = j.merge(p.extra(), on=["game_id", "play_id"], how="left")
    j = j[j.xpass.notna() & j.wp.notna() & j.fumble_lost.notna()].reset_index(drop=True)
    d, _, _ = m.prepare(j)
    d = d.reset_index(drop=True)
    d["dp"] = s.clip_rule(d.n_defense_box).astype(str)
    d["y"] = d.success.astype(float)
    return d


def fitter(d):
    f = p.Aug(d, ["xpass", "wp"])
    f.y[:] = d.y.values
    return f


def deviations(f, w, cells):
    cell, sw, *_ = f.fit(w)
    pos = {c: i for i, c in enumerate(f.cells)}
    ok = [c for c in cells if c in pos]
    idx = np.array([pos[c] for c in ok])
    proj, _ = p.projector(ok, sw[idx])
    return ok, proj @ cell[idx], sw[idx]


def align(a, b):
    ca, da, wa = a
    cb, db, wb = b
    common = [c for c in ca if c in cb]
    ia = [ca.index(c) for c in common]
    ib = [cb.index(c) for c in common]
    return common, da[ia], db[ib], wa[ia], wb[ib]


def one_pair(fa, fb, wa, wb, cells):
    ra = deviations(fa, wa, cells)
    rb = deviations(fb, wb, cells)
    common, da, db, na, nb = align(ra, rb)
    return common, da, db, na, nb


def main():
    t0 = time.time()
    d = build()
    rng = np.random.default_rng(p.SEED)
    allcells = sorted(set((d.og + "|" + d.dp).unique()))
    folds = {}
    for hold in s.SEASONS:
        tr = d[d.season != hold].reset_index(drop=True)
        te = d[d.season == hold].reset_index(drop=True)
        ftr, fte = fitter(tr), fitter(te)
        common, dp_, dt_, ntr, nte = one_pair(
            ftr, fte, np.ones(len(ftr.y)), np.ones(len(fte.y)), allcells
        )
        r = m.wcorr(dp_, dt_, np.minimum(ntr, nte))[0]
        draws = []
        for _ in range(BOOT_GAME):
            _, a, b, na, nb = one_pair(ftr, fte, ftr.boot(rng), fte.boot(rng), allcells)
            draws.append(m.wcorr(a, b, np.minimum(na, nb))[0])
        draws = np.array(draws)
        draws = draws[~np.isnan(draws)]
        order = np.argsort(-np.abs(dp_ * dt_))
        folds[hold] = dict(  # noqa: C408
            corr=r,
            cells=len(common),
            game_boot_ci=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            game_boot_p_positive=float((draws > 0).mean()),
            sign_agree=int(((dp_ > 0) == (dt_ > 0)).sum()),
            rows=[
                dict(  # noqa: C408
                    cell=common[i],
                    train_pp=float(100 * dp_[i]),
                    heldout_pp=float(100 * dt_[i]),
                    heldout_n=int(nte[i]),
                )
                for i in order
            ],
        )
        print(hold, r, folds[hold]["game_boot_ci"], flush=True)
    rs = np.array([folds[h]["corr"] for h in s.SEASONS])
    bb = np.array([rng.choice(rs, len(rs)).mean() for _ in range(20000)])
    odd = d[d.week % 2 == 1].reset_index(drop=True)
    even = d[d.week % 2 == 0].reset_index(drop=True)
    fo, fe = fitter(odd), fitter(even)
    common, da, db, na, nb = one_pair(fo, fe, np.ones(len(fo.y)), np.ones(len(fe.y)), allcells)
    sh = m.wcorr(da, db, np.minimum(na, nb))[0]
    sd = []
    for _ in range(BOOT_GAME):
        _, a, b, x, y = one_pair(fo, fe, fo.boot(rng), fe.boot(rng), allcells)
        sd.append(m.wcorr(a, b, np.minimum(x, y))[0])
    sd = np.array(sd)
    sd = sd[~np.isnan(sd)]
    pooled_cells = {}
    for h in s.SEASONS:
        for r in folds[h]["rows"]:
            pooled_cells.setdefault(r["cell"], []).append((r["train_pp"], r["heldout_pp"]))
    carry = sorted(
        (
            dict(  # noqa: C408
                cell=c,
                mean_pred_pp=float(np.mean([a for a, _ in v])),
                mean_heldout_pp=float(np.mean([b for _, b in v])),
                folds=len(v),
            )
            for c, v in pooled_cells.items()
        ),
        key=lambda x: -abs(x["mean_heldout_pp"]),
    )
    out = dict(  # noqa: C408
        folds=folds,
        season_block=dict(  # noqa: C408
            mean_corr=float(rs.mean()),
            ci=[float(np.quantile(bb, 0.025)), float(np.quantile(bb, 0.975))],
            p_positive=float((bb > 0).mean()),
            fold_corrs=rs.tolist(),
        ),
        split_half=dict(  # noqa: C408
            corr=sh,
            cells=len(common),
            ci=[float(np.quantile(sd, 0.025)), float(np.quantile(sd, 0.975))],
            p_positive=float((sd > 0).mean()),
        ),
        carrying=carry,
        seconds=time.time() - t0,
    )
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("season_block", "split_half", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
