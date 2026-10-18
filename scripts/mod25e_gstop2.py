import os

import numpy as np

COUNT = {"calls": 0, "same_cls": 0, "stop_only": 0, "self": 0}
_STATE = {"k": None, "rng": None}
SALT = 9903


def enabled():
    return os.environ.get("GSTOP2") == "1"


def install_gstop2():
    import mod25d_variance as dv
    import mod25e_geo as ge

    if getattr(ge.Geo.fix, "_gstop2", False):
        return
    orig = ge.Geo.fix

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        cap = {}

        def wf(near):
            cap["near"] = near
            return weight_fn(near)

        out = orig(self, drawn, i, down, q, f3q, net_sim, wf, rng, force)
        if out is None or "near" not in cap:
            return out
        new, j, shift = out
        h = dv._G["f2h"]
        stop = np.asarray(h["stop"])
        cls = np.asarray(h["cls"])
        near = cap["near"]
        sj = stop[int(j)]
        ci = cls[int(i)]
        if _STATE["k"] != dv._G.get("task_key"):
            _STATE["k"] = dv._G.get("task_key")
            _STATE["rng"] = dv.task_rng(SALT, int(dv._G["cfg"].get("seed", 3)))
        r = _STATE["rng"]
        COUNT["calls"] += 1
        m = near[(stop[near] == sj) & (cls[near] == ci)]
        if len(m):
            COUNT["same_cls"] += 1
        else:
            m = near[stop[near] == sj]
            if len(m):
                COUNT["stop_only"] += 1
        if len(m) == 0:
            COUNT["self"] += 1
            k = int(j)
        else:
            k = int(m[r.integers(len(m))])
        new = dict(new)
        new["clock_elapsed"] = self.a["clock_elapsed"][k]
        return new, j, shift

    fix._gstop2 = True
    ge.Geo.fix = fix
