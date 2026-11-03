import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_unit2_components as u2  # noqa: E402
from per07_first_game_fitted_term import block_bootstrap, summarize  # noqa: E402
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

INJ = sorted(glob.glob(str(REPO / "data" / "players" / "raw" / "*" / "injuries.parquet")))[-1]
OUT = REPO / "artifacts" / "mod41" / "unit3_injury_mechanism.json"
FAMILY = "mod41_residual_load_u3_injury_mechanism"
DESIGNATIONS = ("Out", "Doubtful")
GROUPS = {
    "QB": ("QB",),
    "RB": ("RB", "FB"),
    "receivers": ("WR", "TE"),
    "OL": ("T", "G", "C"),
    "DL": ("DE", "DT"),
    "LB": ("LB",),
    "DB": ("CB", "S"),
}


def counts():
    i = pd.read_parquet(INJ)
    i = i[(i.game_type == "REG") & i.report_status.isin(DESIGNATIONS)]
    pos = {p: g for g, ps in GROUPS.items() for p in ps}
    i = i.assign(group=i.position.map(pos)).dropna(subset=["group"])
    c = i.groupby(["season", "week", "team", "group"])["gsis_id"].nunique().rename("n")
    return c.reset_index()


def attach(d):
    c = counts()
    for g in GROUPS:
        cg = c[c.group == g].set_index(["season", "week", "team"])["n"]
        for side in ("home", "away"):
            idx = pd.MultiIndex.from_frame(d[["season", "week", f"{side}_team"]])
            d[f"{side}_{g}"] = cg.reindex(idx).fillna(0).to_numpy()
        d[f"inj_{g}"] = d[f"home_{g}"] - d[f"away_{g}"]
    d["inj_all"] = sum(d[f"inj_{g}"] for g in GROUPS)
    return d


def first_stage(d, g, rng):
    u = d.loc[d["usable"]]

    def slope(f):
        x = f["s_resid"] - f["s_resid"].mean()
        return float((x * (f[f"inj_{g}"] - f[f"inj_{g}"].mean())).sum() / (x * x).sum())

    s = slope(u)
    return summarize(block_bootstrap(u, slope, rng), s)


def main():
    d, _ = u2.build()
    d = attach(d)
    S = ["logit_s"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"family": FAMILY, "designations": DESIGNATIONS, "injury_file": INJ, "arms": {}}
    out["first_stage"] = {g: first_stage(d, g, rng) for g in GROUPS}
    arms = [("none", S)]
    for g in GROUPS:
        arms.append((g, [*S, f"inj_{g}"]))
    allc = [f"inj_{g}" for g in GROUPS]
    arms.append(("all_groups", [*S, *allc]))
    out["looks"] = len(arms) + len(out["first_stage"])
    for name, base in arms:
        o = u2.run(d, base, ["s_resid"], name)
        out["arms"][name] = o
        print(
            name, "ll", o["ll_gain"], "dec", o["decisive_term_correct"], o["decisive_base_correct"]
        )
        print(" gap", o["gap_ll"], "resid coef", o["coef_full"][-1])
    for g, v in out["first_stage"].items():
        print("first_stage", g, v)
    OUT.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
