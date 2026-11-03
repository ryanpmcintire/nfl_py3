import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_unit2_components as u2  # noqa: E402

PICK = REPO / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
OUT = REPO / "artifacts" / "mod41" / "unit6_served_base.json"
FAMILY = "mod41_residual_load_u6_served_base"


def main():
    d, _beta = u2.build()
    pk = pd.read_parquet(PICK, columns=["game_id", "out_of_season_home_probability"])
    d = d.merge(pk, on="game_id", how="left")
    ps = d["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6)
    d["logit_s"] = np.log(ps / (1 - ps))
    y = d["y"]
    stored = (d["out_of_season_home_probability"] >= 0.5).astype(int)
    raw = (d["p_served"] >= 0.5).astype(int)
    pr = d["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6)
    ll_stored = float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean())
    out = {
        "family": FAMILY,
        "pick_artifact": str(PICK.relative_to(REPO)),
        "games": len(d),
        "missing_served": int(d["out_of_season_home_probability"].isna().sum()),
        "stored_served_correct": int((stored == y).sum()),
        "stored_served_log_loss": ll_stored,
        "raw_model_correct": int((raw == y).sum()),
        "usable": int(d["usable"].sum()),
        "arms": {},
        "null": {},
    }
    S = ["logit_s"]
    out["arms"]["resid_load_served"] = u2.run(d, S, ["s_resid"], "resid_load_served")
    out["null"]["s_resid"] = u2.perm_null(d, S, "s_resid")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print({k: v for k, v in out.items() if k not in ("arms", "null")})
    o = out["arms"]["resid_load_served"]
    print("dec", o["decisive"], o["decisive_term_correct"], o["decisive_base_correct"])
    print("acc", o["acc_gain"], "\nll", o["ll_gain"])
    print("base", o["base_oos"], "\nterm", o["term_oos"])
    print("int", o["intercept_only"], "in", o["in_ll_gain"], "gap", o["gap_ll"])
    print("coef", o["coef_full"])
    for k, v in o["coef_folds"].items():
        print(k, [round(x, 4) for x in v])
    print(out["null"])


if __name__ == "__main__":
    main()
