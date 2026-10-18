import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "v1ad" / "fit.json"
ESS21 = REPO / "tests" / "scratch" / "e161" / "ess21.py"
COUNT = {"calls": 0, "round_up": 0, "censor": 0, "censor_td": 0, "no_ipw": 0}


def enabled():
    return os.environ.get("GEOV1") == "1"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_command():
    src = ESS21.read_text(encoding="utf-8").split("tdA = np.zeros(nq)")[0]
    src = src.replace("FORMS = {", 'FORMS = {"F1_type": (2, mkform(1))}\n_all_forms = {', 1)
    ns = {"__name__": "ess21_fit"}
    exec(compile(src, "ess21fit", "exec"), ns)
    form = "F1_type"
    out = {"form": "sigmoid(a + b * is_pass)", "seasons": {}, "loso_heldout_ll": {}}
    for s in ns["seasons"]:
        out["seasons"][str(int(s))] = [float(v) for v in ns["THS"][(form, s)]]
    out["all"] = [float(v) for v in ns["THS"][(form, None)]]
    out["fit_queries"] = int(len(ns["fitq"]))
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


def pi_of(spec, play_type_code):
    a, b = spec["all"]
    return float(sigmoid(a + b * (1.0 if int(play_type_code) == 1 else 0.0)))


def install_v1ad():
    import mod25e_geo as ge

    spec = json.loads(FIT.read_text(encoding="utf-8"))

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        if not force and not self.violates(drawn, i, q):
            return None
        d = down if down in (1, 2, 3, 4) else 4
        P = self.pools.get((d, int(drawn["play_type_code"])))
        if P is None:
            return None
        pos = self.candidates(P, q, net_sim, float(f3q[0]))
        if len(pos) == 0:
            return None
        k = min(self.k, len(pos))
        dd = ((P["f3"][pos] - f3q[None, :]) ** 2).sum(axis=1)
        near = pos[np.argpartition(dd, k - 1)[:k]] if k < len(pos) else pos
        ids = P["ids"][near]
        w = weight_fn(ids)
        if d == 4:
            ipw = None
            try:
                ipw = np.asarray(ge_ns()["IPW"])[ids]
            except KeyError:
                ipw = None
            if ipw is not None:
                w = w / ipw
                COUNT["no_ipw"] += 1
        c = np.cumsum(w)
        if not np.isfinite(c[-1]) or c[-1] <= 0.0:
            jp = int(rng.integers(len(ids)))
        else:
            jp = min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(ids) - 1)
        j = int(ids[jp])
        jpos = int(near[jp]) if k < len(pos) else int(pos[jp])
        shift = self.bias + self.gain * (net_sim - float(self.a["off_row"][j] + self.sign * self.a["def_row"][j]))
        fl = float(np.floor(shift))
        up = rng.random() < shift - fl
        COUNT["round_up"] += int(up)
        yards = float(self.a["yards_gained"][j]) + fl + (1.0 if up else 0.0)
        new = dict(drawn)
        keys = [k_ for k_ in ge.REPICK_KEYS if k_ not in ge.ENGINE_KEEP]
        for key in keys:
            new[key] = self.a[key][j]
        new["flip"] = bool(self.a["possession_flip"][j])
        new["auto_first"] = bool(self.a["auto_first"][j])
        new["repeat_down"] = bool(self.a["repeat_down"][j])
        new["yards_gained"] = yards
        new["idx"] = j
        if os.environ.get("FLIPA") == "1" and new["flip"]:
            new["next_yardline"] = float(min(max(float(self.a["next_yardline"][j]) + (float(P["fp"][jpos]) - float(q)) + (yards - float(self.a["yards_gained"][j])), 1.0), 99.0))
        if not P["td"][jpos] and not P["free"][jpos] and yards >= q:
            COUNT["censor"] += 1
            if rng.random() < pi_of(spec, drawn["play_type_code"]):
                COUNT["censor_td"] += 1
                tdp = pos[P["td"][pos] & (P["fp"][pos] >= q)]
                if len(tdp) == 0:
                    tdp = np.flatnonzero(P["td"] & (P["fp"] >= q))
                if len(tdp):
                    t = int(P["ids"][tdp[int(rng.integers(len(tdp)))]])
                    for key in keys:
                        new[key] = self.a[key][t]
                    new["flip"] = bool(self.a["possession_flip"][t])
                    new["auto_first"] = bool(self.a["auto_first"][t])
                    new["repeat_down"] = bool(self.a["repeat_down"][t])
                    new["yards_gained"] = float(q)
        return new, j, shift

    ge.Geo.fix = fix


def ge_ns():
    import mod25d_variance as dv

    return dv._G["ns"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit_command()
