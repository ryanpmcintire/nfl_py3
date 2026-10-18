import inspect

SITE_DIST = "dist_sq = (off_row - off_sim) ** 2 + (def_row - def_sim) ** 2"
SITE_REPL = "dist_sq = (off_row - off_sim) ** 2 + ((def_row - def_sim) * AXH_DEF_RATIO) ** 2"


def axis_ratio(eo, ef):
    import numpy as np

    return float(np.std(eo)) / float(np.std(ef))


def patched_source(dv, upstream=None):
    orig = upstream or inspect.getsource

    def gs(o):
        s = orig(o)
        if o is dv.sim.pick_index_nn_conditioned:
            assert SITE_DIST in s
            s = s.replace(SITE_DIST, SITE_REPL)
        return s

    return orig, gs


def install(dv, eo, ef):
    dv._G["ns"]["AXH_DEF_RATIO"] = axis_ratio(eo, ef)
    dv._G["tables"]["nn_weight_cache_cond"].clear()
