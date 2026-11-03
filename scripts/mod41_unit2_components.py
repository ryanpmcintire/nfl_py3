import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_cumulative_workload as m  # noqa: E402
from per07_first_game_fitted_term import (  # noqa: E402
    GAME_FEATURES,
    block_bootstrap,
    fit,
    metrics,
    summarize,
)
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

SERVED = REPO / "artifacts" / "opener_evaluation" / "20261101T151614Z" / "per_game.parquet"
OUT = REPO / "artifacts" / "mod41" / "unit2_components.json"
FAMILY = "mod41_cumulative_workload_u2"
NULL_DRAWS = 500
PERM_SEED = 4101
WINDOW = m.WINDOW


def components(t):
    key = [t["team"], t["season"]]
    g = t.groupby(key)
    t["own_td"] = g["run"].transform(lambda s: s.shift(1).expanding().mean())
    t["opp3"] = g["defended"].transform(lambda s: s.shift(1).rolling(WINDOW).mean())
    return t[["game_id", "team", "own_td", "opp3"]]


def build():
    d, t = m.build()
    served = pd.read_parquet(SERVED, columns=["game_id", "home_cover_probability_at_open"])
    d = d.merge(served.rename(columns={"home_cover_probability_at_open": "p_served"}), on="game_id")
    ps = d["p_served"].clip(1e-6, 1 - 1e-6)
    d["logit_s"] = np.log(ps / (1 - ps))
    rest = pd.read_parquet(GAME_FEATURES, columns=["game_id", "rest_diff"])
    d = d.merge(rest, on="game_id", how="left")
    d["rest_diff"] = d["rest_diff"].fillna(0.0)
    c = components(t)
    for side in ("home", "away"):
        f = c.rename(
            columns={"team": f"{side}_team", "own_td": f"{side}_own", "opp3": f"{side}_opp"}
        )
        d = d.merge(f, on=["game_id", f"{side}_team"], how="left")
    u = d["usable"]
    d["s_own"] = (d["home_own"] - d["away_own"]).where(u, 0.0).fillna(0.0)
    d["s_opp"] = (d["home_opp"] - d["away_opp"]).where(u, 0.0).fillna(0.0)
    x = np.column_stack([np.ones(u.sum()), d.loc[u, "s_own"], d.loc[u, "s_opp"]])
    beta, *_ = np.linalg.lstsq(x, d.loc[u, "s_load_total"].to_numpy(), rcond=None)
    d["s_resid"] = 0.0
    d.loc[u, "s_resid"] = d.loc[u, "s_load_total"].to_numpy() - x @ beta
    return d, [float(b) for b in beta]


def loso(d, cols):
    p = pd.Series(np.nan, index=d.index)
    co = {}
    for season in sorted(d["season"].unique()):
        trm = d["season"].ne(season)
        mod = fit(d.loc[trm], cols)
        p[~trm] = mod.predict_proba(d.loc[~trm, cols])[:, 1]
        co[int(season)] = [float(v) for v in mod.coef_[0]]
    return p, co


def run(d, base, add, name):
    bp, _ = loso(d, base)
    tp, co = loso(d, base + add)
    ip = pd.Series(np.nan, index=d.index)
    for season in sorted(d["season"].unique()):
        ip[d["season"].eq(season)] = d.loc[d["season"].ne(season), "y"].mean()
    sc = d.assign(p_base=bp, p_term=tp, p_int=ip)

    def gain(f):
        return (
            ((f["p_term"] >= 0.5).astype(int) == f["y"]).mean()
            - ((f["p_base"] >= 0.5).astype(int) == f["y"]).mean()
        ) * 100.0

    def llg(f):
        return metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cols = base + add
    ins = d.assign(
        p_base=fit(d, base).predict_proba(d[base])[:, 1],
        p_term=fit(d, cols).predict_proba(d[cols])[:, 1],
    )
    dec = sc.loc[(sc["p_term"] >= 0.5) != (sc["p_base"] >= 0.5)]
    return {
        "name": name,
        "base": base,
        "add": add,
        "decisive": len(dec),
        "decisive_term_correct": int(((dec["p_term"] >= 0.5).astype(int) == dec["y"]).sum()),
        "decisive_base_correct": int(((dec["p_base"] >= 0.5).astype(int) == dec["y"]).sum()),
        "intercept_only": metrics(sc, "p_int"),
        "base_oos": metrics(sc, "p_base"),
        "term_oos": metrics(sc, "p_term"),
        "acc_gain": summarize(block_bootstrap(sc, gain, rng), float(gain(sc))),
        "ll_gain": summarize(block_bootstrap(sc, llg, rng), float(llg(sc))),
        "in_ll_gain": float(llg(ins)),
        "gap_ll": float(llg(ins) - llg(sc)),
        "coef_folds": co,
        "coef_full": [float(v) for v in fit(d, cols).coef_[0]],
    }


def perm_null(d, base, col):
    bp, _ = loso(d, base)
    obs = None
    rng = np.random.default_rng(PERM_SEED)
    u = d.index[d["usable"]]
    blocks = [g.index.to_numpy() for _, g in d.loc[u].groupby("block")]
    gains = []
    for i in range(NULL_DRAWS + 1):
        dd = d.copy()
        if i:
            v = dd[col].to_numpy().copy()
            for idx in blocks:
                v[idx] = v[rng.permutation(idx)]
            dd[col] = v
        tp, _ = loso(dd, [*base, col])
        f = dd.assign(p_base=bp, p_term=tp)
        g = metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"]
        if i:
            gains.append(g)
        else:
            obs = g
    gains = np.array(gains)
    return {
        "col": col,
        "observed_ll_gain": float(obs),
        "null_mean": float(gains.mean()),
        "null_p_ge": float((1 + (gains >= obs).sum()) / (NULL_DRAWS + 1)),
    }


def unusable(d):
    u = d.loc[~d["usable"]]
    k = d.loc[d["usable"]]
    out = {
        "n_unusable": len(u),
        "weeks_unusable": {
            int(a): int(b) for a, b in u["week"].value_counts().sort_index().items()
        },
        "min_week_usable": int(k["week"].min()),
    }
    for lab, f in (("unusable", u), ("usable", k)):
        out[lab] = {
            "n": len(f),
            "served_acc": float(((f["p_served"] >= 0.5).astype(int) == f["y"]).mean()),
            "mean_abs_logit_s": float(f["logit_s"].abs().mean()),
            "mean_abs_spread": float(f["tue_open_home_spread"].abs().mean()),
        }
    return out


def main():
    d, beta = build()
    S = ["logit_s"]
    SR = ["logit_s", "rest_diff"]
    arms = [
        ("total_vs_logit_p", ["logit_p"], ["s_load_total"]),
        ("total", S, ["s_load_total"]),
        ("own_tempo", S, ["s_own"]),
        ("opp_tempo", S, ["s_opp"]),
        ("resid_load", S, ["s_resid"]),
        ("joint_abc", S, ["s_own", "s_opp", "s_resid"]),
        ("rest_only", S, ["rest_diff"]),
        ("total_given_rest", SR, ["s_load_total"]),
        ("own_given_rest", SR, ["s_own"]),
        ("opp_given_rest", SR, ["s_opp"]),
        ("resid_given_rest", SR, ["s_resid"]),
    ]
    out = {
        "family": FAMILY,
        "grade": "opener",
        "looks": len(arms),
        "games": len(d),
        "usable": int(d["usable"].sum()),
        "decomp_beta": beta,
        "served_all_correct": int(((d["p_served"] >= 0.5).astype(int) == d["y"]).sum()),
        "unusable": unusable(d),
        "arms": {},
        "null": {},
    }
    for name, base, add in arms:
        out["arms"][name] = run(d, base, add, name)
    for col in ("s_load_total", "s_own", "s_opp", "s_resid"):
        out["null"][col] = perm_null(d, S, col)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print({k: v for k, v in out.items() if k not in ("arms", "null")})
    for name, o in out["arms"].items():
        print(name, "dec", o["decisive"], o["decisive_term_correct"], o["decisive_base_correct"])
        print(" acc", o["acc_gain"], "\n ll", o["ll_gain"])
        print(" base", o["base_oos"], "\n term", o["term_oos"])
        print(" int", o["intercept_only"]["log_loss"], "in", o["in_ll_gain"], "gap", o["gap_ll"])
        print(" coef", o["coef_full"], [round(v[-1], 4) for v in o["coef_folds"].values()])
    print(out["null"])


if __name__ == "__main__":
    main()
