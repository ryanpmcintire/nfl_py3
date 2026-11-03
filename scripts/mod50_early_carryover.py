import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_unit2_components as u2  # noqa: E402
from per07_first_game_fitted_term import fit, summarize  # noqa: E402
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

PICK = REPO / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
OUT = REPO / "artifacts" / "mod50" / "unit1_early_carryover.json"
FAMILY = "mod50_mod41_early"
WINDOW = u2.WINDOW
INTERACTION_DRAWS = 1000


def variant_features(t):
    t = t.sort_values(["team", "kickoff"]).reset_index(drop=True)
    rows = []
    for team, g in t.groupby("team"):
        g = g.copy()
        sm = g.groupby("season")[["total", "run"]].mean()
        prev = sm.shift(1)
        g["prev_total"] = g["season"].map(prev["total"])
        g["prev_run"] = g["season"].map(prev["run"])
        gk = g.groupby("season")
        g["in_total"] = gk["total"].transform(lambda s: s.shift(1).expanding().mean())
        g["in_run"] = gk["run"].transform(lambda s: s.shift(1).expanding().mean())
        g["norm_a"] = g["in_total"].fillna(g["prev_total"])
        g["own_a"] = g["in_run"].fillna(g["prev_run"])
        g["win_a"] = g["total"].shift(1).rolling(WINDOW).mean()
        g["opp_a"] = g["defended"].shift(1).rolling(WINDOW).mean()
        tot = g["total"].to_numpy()
        dfd = g["defended"].to_numpy()
        win_b = np.full(len(g), np.nan)
        opp_b = np.full(len(g), np.nan)
        for _, idx in g.groupby("season").indices.items():
            for pos, i in enumerate(idx):
                kk = min(WINDOW, pos - 1)
                if kk >= 1:
                    win_b[i] = tot[idx[pos - kk : pos]].mean()
                    opp_b[i] = dfd[idx[pos - kk : pos]].mean()
        g["win_b"] = win_b
        g["opp_b"] = opp_b
        g["load_a"] = g["win_a"] - g["norm_a"]
        g["load_b"] = g["win_b"] - g["in_total"]
        g["team"] = team
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def side_diff(d, f, cols):
    for side in ("home", "away"):
        m = f[["game_id", "team", *cols]].rename(
            columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in cols}}
        )
        d = d.merge(m, on=["game_id", f"{side}_team"], how="left")
    return d


def build():
    d, beta = u2.build()
    _, t = u2.m.build()
    pk = pd.read_parquet(PICK, columns=["game_id", "out_of_season_home_probability"])
    d = d.merge(pk, on="game_id", how="left")
    ps = d["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6)
    d["logit_s"] = np.log(ps / (1 - ps))
    v = variant_features(t)
    cols = ["load_a", "own_a", "opp_a", "load_b", "in_run", "opp_b"]
    d = side_diff(d, v, cols)
    d["early"] = ~d["usable"]
    for tag, (lo, ow, op) in {
        "a": ("load_a", "own_a", "opp_a"),
        "b": ("load_b", "in_run", "opp_b"),
    }.items():
        s_load = d[f"home_{lo}"] - d[f"away_{lo}"]
        s_own = d[f"home_{ow}"] - d[f"away_{ow}"]
        s_opp = d[f"home_{op}"] - d[f"away_{op}"]
        r = s_load - (beta[0] + beta[1] * s_own + beta[2] * s_opp)
        d[f"has_{tag}"] = r.notna() & d["early"]
        d[f"s_{tag}"] = r.where(d["early"], 0.0).fillna(0.0)
        d[f"m_{tag}"] = np.where(d["early"], d[f"s_{tag}"], d["s_resid"])
        d[f"mi_{tag}"] = np.where(d["early"], d[f"s_{tag}"], 0.0)
    return d, beta


def interaction(d, tag, base):
    cols = [*base, f"m_{tag}", f"mi_{tag}"]
    pt = float(fit(d, cols).coef_[0][-1])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = [g for _, g in d.groupby("block")]
    draws = np.empty(INTERACTION_DRAWS)
    for i in range(INTERACTION_DRAWS):
        idx = rng.integers(0, len(groups), len(groups))
        draws[i] = fit(pd.concat([groups[j] for j in idx]), cols).coef_[0][-1]
    out = summarize(draws, pt)
    out["main_coef"] = float(fit(d, cols).coef_[0][-2])
    out["early_total_coef"] = out["main_coef"] + pt
    return out


def main():
    d, beta = build()
    S = ["logit_s"]
    e = d.loc[d["early"]].reset_index(drop=True)
    out = {
        "family": FAMILY,
        "grade": "opener",
        "games": len(d),
        "early_games": len(e),
        "early_by_week": {int(a): int(b) for a, b in e["week"].value_counts().sort_index().items()},
        "early_served_correct": int(
            ((e["out_of_season_home_probability"] >= 0.5).astype(int) == e["y"]).sum()
        ),
        "beta": beta,
        "variants": {},
    }
    for tag in ("a", "b"):
        v = {
            "rows_with_value": int(e[f"has_{tag}"].sum()),
            "early_only": u2.run(e, S, [f"s_{tag}"], f"early_only_{tag}"),
            "merged": u2.run(d, S, [f"m_{tag}"], f"merged_{tag}"),
            "interaction": interaction(d, tag, S),
        }
        out["variants"][tag] = v
    base_all = u2.run(d, S, ["s_resid"], "current_main")
    out["current_main"] = base_all
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print({k: x for k, x in out.items() if k not in ("variants", "current_main")})
    for k, x in out["variants"].items():
        print(k, "rows", x["rows_with_value"])
        for part in ("early_only", "merged"):
            o = x[part]
            print(
                " ",
                part,
                "dec",
                o["decisive"],
                o["decisive_term_correct"],
                o["decisive_base_correct"],
            )
            print("   acc", o["acc_gain"], "\n   ll", o["ll_gain"])
            print(
                "   base",
                o["base_oos"]["log_loss"],
                "term",
                o["term_oos"]["log_loss"],
                "gap",
                o["gap_ll"],
            )
            print(
                "   coef_full", o["coef_full"], [round(v[-1], 4) for v in o["coef_folds"].values()]
            )
        print("  interaction", x["interaction"])
    o = out["current_main"]
    print(
        "main ll",
        o["ll_gain"],
        "dec",
        o["decisive"],
        o["decisive_term_correct"],
        o["decisive_base_correct"],
    )


if __name__ == "__main__":
    main()
