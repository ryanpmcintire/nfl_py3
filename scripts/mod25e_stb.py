import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUTD = REPO / "artifacts" / "mod25e3" / "stb"
FIT = OUTD / "fit.json"
SALT = 9703
POOL = tuple(range(2009, 2018))
TB_SPOTS = {"punt": (80.0,), "kick": (75.0, 80.0)}
FORMS = {"punt": {"const": (), "fp": ("fp",), "fp2": ("fp", "fp2")}, "kick": {"const": ()}}


def enabled():
    return os.environ.get("STB") == "1"


def kick_class(code, flip, po, pdf):
    punt = flip & (code == 2) & (po == 0) & (pdf == 0)
    kick = flip & ((po >= 3) | (pdf >= 6))
    return punt, kick


def real_rows():
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(POOL)
    t = sim.build_transition_frame(pbp)
    meta = pbp.drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", "posteam"]]
    t = t.merge(meta, on=["game_id", "play_id"], how="left")
    t["season"] = t.game_id.str[:4].astype(int)
    punt, kick = kick_class(t.play_type_code.to_numpy().astype(int), t.possession_flip.to_numpy().astype(bool), t.points_off.to_numpy(float), t.points_def.to_numpy(float))
    t["cls"] = np.where(punt, "punt", np.where(kick, "kick", ""))
    d = t[t.cls != ""].copy()
    d["fp"] = np.clip(np.round(d.fp_raw.to_numpy(float)), 1, 99)
    d["fp2"] = d.fp**2
    ny = d.next_yardline.to_numpy(float)
    d["tb"] = np.where(d.cls == "punt", np.isin(ny, TB_SPOTS["punt"]), np.isin(ny, TB_SPOTS["kick"])).astype(int)
    d["spot"] = np.where(d.tb == 1, ny, np.nan)
    return d.reset_index(drop=True)


def design(d, cols):
    return np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])


def fit_logit(X, y):
    from sklearn.linear_model import LogisticRegression

    if X.shape[1] == 1:
        p = float(y.mean())
        return np.array([np.log(p / (1.0 - p))])
    return LogisticRegression(C=1e9, max_iter=5000, fit_intercept=False).fit(X, y).coef_[0]


def loglik(X, y, b):
    z = X @ b
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


def loso(d, cols):
    per = {}
    for s in POOL:
        te, tr = d[d.season == s], d[d.season != s]
        b = fit_logit(design(tr, cols), tr.tb.to_numpy(float))
        per[s] = loglik(design(te, cols), te.tb.to_numpy(float), b)
    return sum(per.values()), per


def team_form(d):
    g = d.groupby(["season", "posteam"])
    tot, n = g.tb.transform("sum"), g.tb.transform("size")
    gg = d.groupby(["season", "posteam", "game_id"])
    m = (n - gg.tb.transform("size")).to_numpy(float)
    return np.where(m > 0, (tot - gg.tb.transform("sum")).to_numpy(float) / np.where(m > 0, m, 1.0), np.nan)


def cmd_fit():
    d = real_rows()
    OUTD.mkdir(parents=True, exist_ok=True)
    out = {"seasons": list(POOL), "classes": {}}
    lines = []
    for cls, forms in FORMS.items():
        c = d[d.cls == cls]
        res = {}
        for nm, cols in forms.items():
            ll, per = loso(c, cols)
            res[nm] = dict(ll=ll, per_season_ll=per, cols=list(cols))
        base = res["const"]
        for nm, r in res.items():
            r["dll"] = r["ll"] - base["ll"]
            r["seasons_better"] = int(sum(r["per_season_ll"][s] > base["per_season_ll"][s] for s in POOL))
            lines.append(f"{cls} form {nm} n {len(c)} LOSO ll {r['ll']:.2f} ({r['ll'] / len(c):.5f}/row) dll v const {r['dll']:+.2f}, seasons better {r['seasons_better']}/9")
        best = max(res, key=lambda k: res[k]["ll"])
        b = fit_logit(design(c, res[best]["cols"]), c.tb.to_numpy(float))
        spots = c.spot.dropna().to_numpy(float)
        out["classes"][cls] = dict(form=best, cols=res[best]["cols"], coef=b.tolist(), n=int(len(c)), tb=int(c.tb.sum()), tb_rate=float(c.tb.mean()), loso={k: {kk: vv for kk, vv in v.items() if kk != "per_season_ll"} for k, v in res.items()}, spot_share={str(int(s)): float((spots == s).mean()) for s in np.unique(spots)})
        lines.append(f"{cls} chosen {best} coef {np.round(b, 6).tolist()} tb {int(c.tb.sum())}/{len(c)} = {c.tb.mean():.4f} spots {out['classes'][cls]['spot_share']}")
        f = c.assign(form=team_form(c))
        f = f[f.form.notna()]
        f = f.assign(form=f.form - f.groupby("season").form.transform("mean"))
        l0, _ = loso(f, ())
        l1, _ = loso(f, ("form",))
        out["classes"][cls]["team_rate_dll"] = l1 - l0
        lines.append(f"{cls} diagnostic kicking-team season rate excl own game, n {len(f)}: dll v const {l1 - l0:+.2f}")
    FIT.write_text(json.dumps(out, indent=1))
    (OUTD / "an.txt").write_text(chr(10).join(lines))
    print(chr(10).join(lines))


def install_stb():
    import mod25d_variance as dv

    fit = json.loads(FIT.read_text(encoding="utf-8"))
    a = dv._G["tables"]["arrays"]
    code = np.asarray(a["play_type_code"]).astype(int)
    nyl = np.asarray(a["next_yardline"], float)
    fp = np.clip(np.round(np.asarray(a["fp_raw"], float)), 1, 99).astype(int)
    punt, kick = kick_class(code, np.asarray(a["possession_flip"], bool), np.asarray(a["points_off"], float), np.asarray(a["points_def"], float))
    ok = np.isfinite(nyl)
    tabs = {}
    for cls, m in (("punt", punt), ("kick", kick)):
        rows = np.flatnonzero(m & ok & ~np.isin(nyl, TB_SPOTS[cls]))
        rows = rows[np.argsort(fp[rows], kind="stable")]
        cnt = np.bincount(fp[rows], minlength=100)
        tabs[cls] = dict(rows=rows, edges=np.r_[0, np.cumsum(cnt)], present=np.flatnonzero(cnt))
    spots = {k: (np.array([float(s) for s in v["spot_share"]]), np.cumsum(list(v["spot_share"].values()))) for k, v in fit["classes"].items()}
    state = {"k": None, "rng": None}
    base = dv._G["pol"]

    def prob(cls, y):
        c = fit["classes"][cls]
        x = [1.0] + [{"fp": float(y), "fp2": float(y) ** 2}[n] for n in c["cols"]]
        return 1.0 / (1.0 + np.exp(-float(np.dot(c["coef"], x))))

    def pol(dn, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(dn, distance, yardline, score_diff, qtr, clock_val, drawn)
        if not bool(drawn["flip"]) or qtr > 4:
            return drawn
        pts_o, pts_d = float(drawn["points_off"]), float(drawn["points_def"])
        if pts_o >= 3 or pts_d >= 6:
            cls = "kick"
        elif int(drawn.get("play_type_code", -1)) == 2 and pts_o == 0.0 and pts_d == 0.0:
            cls = "punt"
        else:
            return drawn
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(SALT, int(dv._G["cfg"].get("seed", 3)))
        rng = state["rng"]
        y = int(min(max(round(float(yardline)), 1), 99))
        u, v, w = rng.random(), rng.random(), rng.random()
        new = dict(drawn)
        if u < prob(cls, y):
            sv, cs = spots[cls]
            spot = float(sv[min(int(np.searchsorted(cs, v, side="right")), len(sv) - 1)])
            new["next_yardline"] = spot
            new["next_distance"] = min(10.0, spot)
            new["next_down"] = 1
            new["stb_tb"] = True
            return new
        new["stb_tb"] = False
        if float(drawn["next_yardline"]) in TB_SPOTS[cls]:
            T = tabs[cls]
            if cls == "punt":
                yb = int(T["present"][np.argmin(np.abs(T["present"] - y))])
                lo, hi = T["edges"][yb], T["edges"][yb + 1]
            else:
                lo, hi = 0, len(T["rows"])
            j = T["rows"][lo + min(int(w * (hi - lo)), hi - lo - 1)]
            new["next_yardline"] = float(a["next_yardline"][j])
            new["next_down"] = a["next_down"][j]
            new["next_distance"] = a["next_distance"][j]
        return new

    dv._G["pol"] = pol


if __name__ == "__main__":
    cmd_fit()
