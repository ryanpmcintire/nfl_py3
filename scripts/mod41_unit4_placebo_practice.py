import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_cumulative_workload as m  # noqa: E402
import mod41_unit2_components as u2  # noqa: E402
import mod41_unit3_injury_mechanism as u3  # noqa: E402
from per07_first_game_fitted_term import block_bootstrap, metrics, summarize  # noqa: E402
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

OUT = REPO / "artifacts" / "mod41" / "unit4_placebo_practice.json"
FAMILY = "mod41_residual_load_u4_placebo_practice"
DRAWS = 500
SEED = 4104
STATUSES = ("Did Not Participate In Practice", "Limited Participation in Practice")
S = ["logit_s"]


def future_resid(d, beta):
    _, t = m.build()
    u2.components(t)
    t["r"] = t["load_total"] - beta[1] * t["own_td"] - beta[2] * t["opp3"]
    t = t.sort_values(["team", "kickoff"])
    g = t.groupby(["team", "season"])
    t["r_next"] = g["r"].shift(-1)
    t["ok_next"] = g["ok"].shift(-1).fillna(False).astype(bool) & t["r_next"].notna()
    for side in ("home", "away"):
        f = t[["game_id", "team", "r_next", "ok_next"]].rename(
            columns={
                "team": f"{side}_team",
                "r_next": f"{side}_rn",
                "ok_next": f"{side}_okn",
            }
        )
        d = d.merge(f, on=["game_id", f"{side}_team"], how="left")
    d["usable_f"] = d["usable"] & d["home_okn"].fillna(False).astype(bool)
    d["usable_f"] &= d["away_okn"].fillna(False).astype(bool)
    d["s_future"] = ((d["home_rn"] - d["away_rn"]) - beta[0]).where(d["usable_f"], 0.0)
    d["s_future"] = d["s_future"].fillna(0.0)
    d["s_resid_f"] = d["s_resid"].where(d["usable_f"], 0.0)
    return d


def swap_null(d, col):
    bp, _ = u2.loso(d, S)
    blocks = [g.index.to_numpy() for _, g in d.loc[d["usable"]].groupby("block")]
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(DRAWS):
        dd = d.copy()
        v = dd[col].to_numpy().copy()
        for idx in blocks:
            v[idx] = v[rng.permutation(idx)]
        dd["s_swap"] = v
        tp, _ = u2.loso(dd, [*S, "s_swap"])
        f = dd.assign(p_base=bp, p_term=tp)
        gains.append(metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"])
    g = np.array(gains)
    lo, hi = np.percentile(g, [2.5, 97.5])
    return {
        "draws": DRAWS,
        "mean": float(g.mean()),
        "ci95": [float(lo), float(hi)],
        "probability_positive": float((g > 0).mean()),
    }


def practice_counts():
    i = pd.read_parquet(u3.INJ)
    i = i[(i.game_type == "REG") & i.practice_status.isin(STATUSES)]
    pos = {p: g for g, ps in u3.GROUPS.items() for p in ps}
    i = i.assign(group=i.position.map(pos)).dropna(subset=["group"])
    c = i.groupby(["season", "week", "team", "group"])["gsis_id"].nunique().rename("n")
    return c.reset_index()


def attach(d):
    c = practice_counts()
    for g in u3.GROUPS:
        cg = c[c.group == g].set_index(["season", "week", "team"])["n"]
        for side in ("home", "away"):
            idx = pd.MultiIndex.from_frame(d[["season", "week", f"{side}_team"]])
            d[f"{side}_p{g}"] = cg.reindex(idx).fillna(0).to_numpy()
        d[f"prac_{g}"] = d[f"home_p{g}"] - d[f"away_p{g}"]
    d["prac_all"] = sum(d[f"prac_{g}"] for g in u3.GROUPS)
    return d


def first_stage(d, col, rng):
    u = d.loc[d["usable"]]

    def slope(f):
        x = f["s_resid"] - f["s_resid"].mean()
        return float((x * (f[col] - f[col].mean())).sum() / (x * x).sum())

    return summarize(block_bootstrap(u, slope, rng), slope(u))


def main():
    d, beta = u2.build()
    d = future_resid(d, beta)
    d = attach(d)
    out = {"family": FAMILY, "beta": beta, "usable_f": int(d["usable_f"].sum()), "arms": {}}
    out["arms"]["resid_same_rows"] = u2.run(d, S, ["s_resid_f"], "resid_same_rows")
    out["arms"]["future_placebo"] = u2.run(d, S, ["s_future"], "future_placebo")
    out["corr_future_resid"] = float(
        d.loc[d["usable_f"], "s_future"].corr(d.loc[d["usable_f"], "s_resid"])
    )
    out["swap_resid"] = swap_null(d, "s_resid")
    out["swap_resid_frozen_arm"] = u2.run(d, S, ["s_resid"], "resid_frozen")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cols = [f"prac_{g}" for g in u3.GROUPS] + ["prac_all"]
    out["first_stage"] = {c: first_stage(d, c, rng) for c in cols}
    arms = [("none", S)] + [(c, [*S, c]) for c in cols[:-1]]
    arms.append(("all_groups", [*S, *cols[:-1]]))
    out["looks"] = 3 + len(arms) + len(cols)
    out["mediation"] = {}
    for name, base in arms:
        out["mediation"][name] = u2.run(d, base, ["s_resid"], name)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for k in ("resid_same_rows", "future_placebo", "swap_resid_frozen_arm"):
        o = out["arms"].get(k) or out[k]
        print(k, o["ll_gain"], o["decisive_term_correct"], o["decisive_base_correct"])
        print(
            " gap", o["gap_ll"], "coef", o["coef_full"][-1], "int", o["intercept_only"]["log_loss"]
        )
    print("corr", out["corr_future_resid"], "usable_f", out["usable_f"])
    print("swap", out["swap_resid"])
    for c, v in out["first_stage"].items():
        print("fs", c, v)
    for k, o in out["mediation"].items():
        print("med", k, o["ll_gain"], o["coef_full"][-1], o["gap_ll"])


if __name__ == "__main__":
    main()
