import os

import numpy as np

COUNT = {"calls": 0, "fallback": 0}


def enabled():
    return os.environ.get("GSTOP") == "1"


def install_gstop():
    import mod25d_variance as dv
    import mod25e_geo as ge

    if getattr(ge.Geo.fix, "_gstop", False):
        return
    orig = ge.Geo.fix

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        h = dv._G["f2h"]
        stop = np.asarray(h["stop"])
        cls = np.asarray(h["cls"])
        s0 = stop[int(i)]
        c0 = cls[int(i)]

        def wf(near):
            w = np.asarray(weight_fn(near), dtype=float)
            m = (stop[near] == s0) & (cls[near] == c0)
            COUNT["calls"] += 1
            if not m.any() or float((w * m).sum()) <= 0.0:
                COUNT["fallback"] += 1
                return w
            return w * m

        return orig(self, drawn, i, down, q, f3q, net_sim, wf, rng, force)

    fix._gstop = True
    ge.Geo.fix = fix
