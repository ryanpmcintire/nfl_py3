import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "clsdraw"
FIT = OUT / "fit.json"
FP_BAND_YD = 10
N_BANDS = 10
CLASS_NAMES = ("returned_td", "other_turnover", "none")
STATE = {"fit": None, "cls": None, "count": np.zeros(3)}


def enabled():
    return os.environ.get("CLSDRAW") == "1"


def features(down, q, code, f3, o_rt, d_rt):
    down = np.atleast_1d(down)
    q = np.atleast_1d(q).astype(float)
    n = len(q)
    band = np.clip((q // FP_BAND_YD).astype(int), 0, N_BANDS - 1)
    cols = [(band == b).astype(float) for b in range(1, N_BANDS)]
    cols += [(down == d).astype(float) for d in (2, 3, 4)]
    cols += [(np.atleast_1d(code) == 1).astype(float)]
    f3 = np.asarray(f3, dtype=float).reshape(n, 3)
    cols += [f3[:, 0], f3[:, 1], f3[:, 2]]
    cols += [np.atleast_1d(o_rt).astype(float), np.atleast_1d(d_rt).astype(float)]
    return np.column_stack(cols)


def class_probs(fit, X):
    Z = (X - np.asarray(fit["mu"])) / np.asarray(fit["sd"])
    L = Z @ np.asarray(fit["coef"]).T + np.asarray(fit["intercept"])
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def class_of(a):
    rp = np.isin(a["play_type_code"], (0, 1))
    flip = a["possession_flip"].astype(bool) & rp
    rtd = (a["points_def"] >= 6) & rp
    return np.where(rtd, 0, np.where(flip, 1, 2)).astype(np.int8)


def load_fit():
    if STATE["fit"] is None:
        STATE["fit"] = json.loads(FIT.read_text(encoding="utf-8"))
    return STATE["fit"]


def install_clsdraw():
    import mod25e_geo as ge

    if getattr(ge.Geo.fix, "_clsdraw", False):
        return
    fit = load_fit()

    def rating_frame():
        fr = sys._getframe(2)
        while fr is not None and not ("offense" in fr.f_locals and "off_sim" in fr.f_locals):
            fr = fr.f_back
        return fr.f_locals

    def fix(self, drawn, i, down, q, f3q, net_sim, weight_fn, rng, force=False):
        if not force and not self.violates(drawn, i, q):
            return None
        d = down if down in (1, 2, 3, 4) else 4
        code = int(drawn["play_type_code"])
        P = self.pools.get((d, code))
        if P is None:
            return None
        pos = self.candidates(P, q, net_sim, float(f3q[0]))
        if len(pos) == 0:
            return None
        if STATE["cls"] is None:
            STATE["cls"] = class_of(self.a)
        cls = STATE["cls"]
        k = min(self.k, len(pos))
        dd = ((P["f3"][pos] - f3q[None, :]) ** 2).sum(axis=1)
        near = pos[np.argpartition(dd, k - 1)[:k]] if k < len(pos) else pos
        L = rating_frame()
        X = features(d, q, code, np.asarray(f3q, dtype=float)[None, :], L["off_sim"], L["def_sim"])
        p = class_probs(fit, X)[0]
        cn = cls[P["ids"][near]]
        cw = cls[P["ids"][pos]]
        present = np.array([(cn == c).any() or (cw == c).any() for c in range(3)])
        pr = np.where(present, p, 0.0)
        if pr.sum() <= 0.0:
            pr = present.astype(float)
        c = int(np.searchsorted(np.cumsum(pr / pr.sum()), rng.random(), side="right"))
        c = min(c, 2)
        STATE["count"][c] += 1
        if (cn == c).any():
            sel = near[cn == c]
        else:
            sel = pos[cw == c]
        ids = P["ids"][sel]
        w = weight_fn(ids)
        cum = np.cumsum(w)
        if not np.isfinite(cum[-1]) or cum[-1] <= 0.0:
            jp = int(rng.integers(len(ids)))
        else:
            jp = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), len(ids) - 1)
        j = int(ids[jp])
        jpos = int(sel[jp])
        shift = self.bias + self.gain * (net_sim - float(self.a["off_row"][j] + self.sign * self.a["def_row"][j]))
        new = dict(drawn)
        for key in (k_ for k_ in ge.REPICK_KEYS if k_ not in ge.ENGINE_KEEP):
            new[key] = self.a[key][j]
        new["flip"] = bool(self.a["possession_flip"][j])
        new["auto_first"] = bool(self.a["auto_first"][j])
        new["repeat_down"] = bool(self.a["repeat_down"][j])
        new["yards_gained"] = ge.sim_dither(self.a["yards_gained"][j], shift, rng) if os.environ.get("YDITH") == "1" else self.a["yards_gained"][j] + shift
        new["idx"] = j
        if os.environ.get("FLIPA") == "1" and new["flip"]:
            new["next_yardline"] = float(min(max(float(self.a["next_yardline"][j]) + (float(P["fp"][jpos]) - float(q)) + (float(new["yards_gained"]) - float(self.a["yards_gained"][j])), 1.0), 99.0))
        if os.environ.get("GEOX") in ("1", "2") and not P["td"][jpos] and not P["free"][jpos] and float(self.a["yards_gained"][j]) + shift >= q:
            tdp = pos[P["td"][pos] & (P["fp"][pos] >= q)]
            if len(tdp) == 0:
                tdp = np.flatnonzero(P["td"] & (P["fp"] >= q))
            if len(tdp):
                t = int(P["ids"][tdp[int(rng.integers(len(tdp)))]])
                for key in (k_ for k_ in ge.REPICK_KEYS if k_ not in ge.ENGINE_KEEP):
                    new[key] = self.a[key][t]
                new["flip"] = bool(self.a["possession_flip"][t])
                new["auto_first"] = bool(self.a["auto_first"][t])
                new["repeat_down"] = bool(self.a["repeat_down"][t])
                new["yards_gained"] = float(q)
        return new, j, shift

    fix._clsdraw = True
    ge.Geo.fix = fix
