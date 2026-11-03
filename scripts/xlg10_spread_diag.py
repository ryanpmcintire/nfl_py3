import contextlib
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pooled_screen as ps  # noqa: E402
import xlg10_addone as x  # noqa: E402

CAPTURED = {}
BINS = 5


def stash(g):
    CAPTURED[len(CAPTURED)] = g
    return original(g)


def slope(z, y):
    def nll(t):
        eta = t[0] + t[1] * z
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    r = minimize(nll, np.array([0.0, 1.0]), method="BFGS")
    return float(r.x[1]), float(np.sqrt(r.hess_inv[1, 1]))


def summarize(label, g):
    rows = {}
    for arm in ("base", "variant"):
        a = g[g["arm"] == arm]
        z = (a["edge"] / a["sigma"]).to_numpy()
        y = (a["cm"] > 0).astype(float).to_numpy()
        p = np.clip(
            0.5 * (1 + np.vectorize(__import__("math").erf)(z / np.sqrt(2))), 1e-6, 1 - 1e-6
        )
        s, se = slope(z, y)
        rows[arm] = {
            "n": len(a),
            "edge_sd": float(a["edge"].std()),
            "sigma_mean": float(a["sigma"].mean()),
            "z_sd": float(z.std()),
            "abs_p_dev_mean": float(np.abs(p - 0.5).mean()),
            "slope": s,
            "slope_se": se,
            "acc": float(a["correct"].mean() * 100),
            "ll": float(a["ll"].mean()),
            "sign_agree_with_base": None,
        }
    b = g[g["arm"] == "base"].set_index("game_id")
    v = g[g["arm"] == "variant"].set_index("game_id")
    idx = b.index.intersection(v.index)
    rows["variant"]["sign_agree_with_base"] = float(
        (np.sign(b.loc[idx, "edge"]) == np.sign(v.loc[idx, "edge"])).mean()
    )
    rows["edge_corr"] = float(np.corrcoef(b.loc[idx, "edge"], v.loc[idx, "edge"])[0, 1])
    print(label, rows, flush=True)


original = ps.contrast_seeded
ps.contrast_seeded = stash
if __name__ == "__main__":
    sys.argv = ["x", "bye_diff"]
    with contextlib.suppress(SystemExit):
        x.main()
    pops = list(CAPTURED.values())
    names = ["nfl_opener", "nfl_proxy", "nfl_all", "cfb"]
    for n, g in zip(names, pops, strict=True):
        if n in ("nfl_opener", "cfb"):
            summarize(n, g)
