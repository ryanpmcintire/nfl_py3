import json

import numpy as np
import sky03_def_traits as d
import sky03_matchups as m
from sky04_lead_lag import fit_b, logit

DRAWS = 4000


def interval(vals):
    v = np.asarray(vals)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float((v < 0).mean())]


def main():
    rng = np.random.default_rng(m.SEED)
    t, _ = d.load()
    rr = d.build(t)
    z, r, w = d.CAND["B_oN_x_completions_per_target_allowed"]
    _res, sub, xs = d.term(rr, z, r, w, m.EVAL_SEASONS, rng)
    y = sub.home_covered.astype(float).values
    off = logit(sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values)
    se = sub.season.values
    wk = (sub.season * 100 + sub.week).values
    out = {"games": len(sub), "pooled_coefficient": float(fit_b(off, xs, y))}
    for lab, keys in (("season_block", se), ("week_block", wk), ("game", np.arange(len(sub)))):
        u = np.unique(keys)
        idxs = [np.where(keys == k)[0] for k in u]
        bs = []
        for _ in range(DRAWS):
            pick = rng.integers(0, len(u), len(u))
            ix = np.concatenate([idxs[i] for i in pick])
            bs.append(fit_b(off[ix], xs[ix], y[ix]))
        lo, hi, pneg = interval(bs)
        out[lab] = {"blocks": len(u), "low": lo, "high": hi, "p_negative": pneg}
    json.dump(out, open(d.OUTF / "sign.json", "w"), indent=1)  # noqa: SIM115
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
