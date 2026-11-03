import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402
import mod24_pooled_levers as lev  # noqa: E402
import pooled_screen as ps  # noqa: E402
import xlg11_cfb_centered_u2 as u2  # noqa: E402

OUT = ROOT / "artifacts" / "xlg11_cfb_centered"
RULES = ("home", "away", "fav", "dog", "icpt")


def arms3(train, test, cols, _arms):
    out = u2.arms(train, test, cols, _arms)
    y = train["ats_margin"].to_numpy(float)
    out["icpt"] = (np.full(len(test), float(y.mean())), float(np.std(y, ddof=1)))
    return out


def boot(d, rng):
    n = len(d)
    idx = rng.integers(0, n, size=(base.DRAWS, n))
    est = d[idx].mean(1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {
        "effect": float(d.mean()),
        "low": float(lo),
        "high": float(hi),
        "se": float(est.std(ddof=1)),
        "p_pos": float((est > 0).mean()),
    }


def side(frame, arm):
    a = frame.loc[frame["arm"].eq(arm)].set_index("game_id")
    return np.sign(a["edge"]), a


def population(g, sv, rng):
    ref = g.loc[g["arm"].eq("u1")].set_index("game_id")
    cm = ref["cm"]
    fav = np.sign(ref["open"])
    sides = {
        "home": pd.Series(1.0, index=ref.index),
        "away": pd.Series(-1.0, index=ref.index),
        "fav": fav,
        "dog": -fav,
    }
    sides["icpt"] = side(g, "icpt")[0].reindex(ref.index)
    for arm in ("base", "u1", "free"):
        sides[arm] = side(g, arm)[0].reindex(ref.index)
    if sv is not None:
        sides["served"] = (
            sv["margin_vs_open"].apply(np.sign) * (2 * sv["correct_at_open"] - 1)
        ).reindex(ref.index)
    win = {k: (v == np.sign(cm)).astype(float) for k, v in sides.items() if v.notna().all()}
    n = len(ref)
    rec = {k: {"wins": int(v.sum()), "n": n, "acc": float(100 * v.mean())} for k, v in win.items()}
    agree = {
        arm: {
            k: float((sides[arm] == sides[k]).mean())
            for k in sides
            if k != arm and sides[k].notna().all()
        }
        for arm in ("u1", "free")
    }
    best = max(RULES, key=lambda k: win[k].sum())
    paired = {}
    for arm in ("u1", "free"):
        for k in (best, "icpt", "fav", "home"):
            paired[f"{arm}_minus_{k}"] = boot(100 * (win[arm] - win[k]).to_numpy(), rng)
    return {"records": rec, "agreement": agree, "best_constant": best, "paired": paired, "n": n}


def main():
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cols = ps.shared_columns()
    u2.STATE["cfb"] = base.cfb_prior_rows(cfb)
    lev.predict_arms = arms3
    g = ps.nfl_graded(cols)
    sv = pd.read_parquet(u2.SERVED)
    sv = sv.loc[sv["margin_vs_open"].ne(0)].set_index("game_id")
    rng = np.random.default_rng(base.SEED)
    res = {}
    for name, (lo, hi), s in (
        ("nfl_opener_2020_25", ps.NFL_OPENER_SEASONS, sv),
        ("nfl_proxy_2011_19", ps.NFL_PROXY_SEASONS, None),
    ):
        sub = g.loc[g["season"].between(lo, hi)]
        if s is not None:
            sub = sub.loc[sub["game_id"].isin(s.index)]
        res[name] = population(sub, s, rng)
    (OUT / "report_u3.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v["n"], "best", v["best_constant"])
        for r, c in v["records"].items():
            print("  rec", r, c["wins"], round(c["acc"], 2))
        for a, c in v["agreement"].items():
            print("  agree", a, {x: round(y, 3) for x, y in c.items()})
        for p, c in v["paired"].items():
            print("  pair", p, {x: round(y, 4) for x, y in c.items()})


if __name__ == "__main__":
    main()
