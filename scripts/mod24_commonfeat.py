import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.clv import _ACTIVE_MODEL_FALLBACK_CONFIG  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402

OPENER_REGIME = "sbr_multibook"


def cell(a, n_arm, c_arm, key):
    ne, ce = n_arm[key], c_arm[key]
    z = (ce["effect"] - ne["effect"]) / np.sqrt(ce["se"] ** 2 + ne["se"] ** 2)
    mult = (1 / ne["se"] ** 2 + 1 / ce["se"] ** 2) / (1 / ne["se"] ** 2)
    return {"nfl": ne, "cfb": ce, "het_z": float(z), "multiplier": float(mult)}


def main():
    nfl_cols = list(
        margin_feature_columns("market_residual", _ACTIVE_MODEL_FALLBACK_CONFIG["feature_profile"])
    )
    common = [c for c in CFB_MODEL_FEATURE_COLUMNS if c in nfl_cols]
    nfl = pd.read_parquet(base.NFL_FEATURES)
    nfl = nfl.loc[nfl["game_type"].eq("REG")].copy()
    nfl["gameday"] = pd.to_datetime(nfl["gameday"])
    pg = pd.read_parquet(base.PER_GAME)[["game_id", "tue_open_home_spread"]]
    nfl = nfl.merge(pg, on="game_id", how="left").rename(columns={"tue_open_home_spread": "open"})
    ng = base.grade(
        base.walk(
            nfl,
            common,
            lambda f: f["season"].between(2020, 2025) & f["open"].notna() & f["result"].notna(),
            base.NFL_MIN_TRAIN,
        )
    )
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cfb["open"] = cfb["spread_open"]
    mask = lambda f: (  # noqa: E731
        f["season"].between(2013, 2025)
        & f["season"].ne(2020)
        & f["open"].notna()
        & f["result"].notna()
    )
    cp = base.walk(cfb, common, mask, base.CFB_MIN_TRAIN)
    reg = cfb[["game_id", "source_regime"]].drop_duplicates("game_id")
    cp_o = cp.merge(reg, on="game_id").query("source_regime == @OPENER_REGIME")
    rn = base.league("nfl", ng)
    out = {"common_features": common, "n_common": len(common), "nfl": rn}
    for tag, pred in (("commonfeat", cp), ("openermatch", cp_o)):
        rc = base.league("cfb", base.grade(pred))
        accn = [rn[str(a)]["accuracy"] for a in base.ALPHAS]
        accc = [rc[str(a)]["accuracy"] for a in base.ALPHAS]
        lln = [rn[str(a)]["log_loss"] for a in base.ALPHAS]
        llc = [rc[str(a)]["log_loss"] for a in base.ALPHAS]
        out[tag] = {
            "cfb": rc,
            "spearman_acc": float(spearmanr(accn, accc).statistic),
            "spearman_ll": float(spearmanr(lln, llc).statistic),
            "contrasts": {
                str(a): {
                    m: cell(a, rn[str(a)], rc[str(a)], k)
                    for m, k in (("accuracy", "acc_vs_served"), ("log_loss", "ll_vs_served"))
                }
                for a in base.ALPHAS
                if a != base.SERVED_ALPHA
            },
        }
    (base.OUT / "commonfeat_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
