import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xlg12_cfb_forward"
VERSIONS = ("v1", "v2")
INVALID = ART / "forward_invalid.json"
FEATURES = ART / "features_2026.parquet"


def ledger_path(version):
    return (ART if version == "v1" else ART / version) / "forward_ledger.jsonl"


def load(version):
    rows = [json.loads(x) for x in ledger_path(version).read_text().splitlines() if x.strip()]
    d = pd.DataFrame(rows)
    if version == "v1" and INVALID.exists():
        bad = json.loads(INVALID.read_text()).get("invalid_rows", [])
        if bad:
            key = d["game_id"].astype(str) + "|" + d["provider"] + "|" + d["variant"]
            d = d.loc[~key.isin({f"{b['game_id']}|{b['provider']}|{b['variant']}" for b in bad})]
    f = pd.read_parquet(FEATURES, columns=["game_id", "completed", "home_points", "away_points"])
    f = f.loc[f["completed"].astype(bool)]
    d = d.merge(f[["game_id", "home_points", "away_points"]], on="game_id", how="inner")
    d["result"] = d["home_points"] - d["away_points"]
    d["ats"] = d["result"] - d["line_home_spread"]
    return d


def summarize(t):
    dec = t.loc[t["ats"] != 0]
    p = dec["p_home_cover_line"].clip(1e-9, 1 - 1e-9).to_numpy()
    y = (dec["ats"] > 0).to_numpy().astype(float)
    hit = (p > 0.5) == (y > 0.5)
    n = len(dec)
    k = int(hit.sum())
    acc = k / n if n else float("nan")
    se = math.sqrt(0.25 / n) if n else float("nan")
    return {
        "n": n,
        "pushes": int((t["ats"] == 0).sum()),
        "wins": k,
        "losses": n - k,
        "acc": acc,
        "acc_se_vs_50": se,
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if n else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if n else float("nan"),
        "logloss_coinflip": math.log(2),
        "brier_coinflip": 0.25,
    }


def score_version(version, weeks):
    if not ledger_path(version).exists():
        print(version, "no ledger")
        return
    d = load(version)
    if weeks:
        d = d.loc[d["week"].isin(weeks)]
    if d.empty:
        print(version, "no finished ledger rows")
        return
    wide = d.assign(pick=(d["p_home_cover_line"] > 0.5)).pivot_table(
        index=["game_id", "provider"], columns="variant", values="pick", aggfunc="first"
    )
    disagree = wide.nunique(axis=1).gt(1)
    d = d.merge(
        disagree.rename("variants_disagree").reset_index(), on=["game_id", "provider"], how="left"
    )
    out = []
    for (prov, var), t in d.groupby(["provider", "variant"]):
        out.append(
            {"version": version, "provider": prov, "variant": var, "subset": "all", **summarize(t)}
        )
        s = t.loc[t["variants_disagree"].fillna(False)]
        sub = "variants_disagree"
        out.append(
            {"version": version, "provider": prov, "variant": var, "subset": sub, **summarize(s)}
        )
    res = pd.DataFrame(out)
    res.to_csv((ART if version == "v1" else ART / version) / "forward_scores.csv", index=False)
    print(res.to_string())


def main():
    ap = argparse.ArgumentParser(description="Score XLG-12 forward ledger rows after games finish")
    ap.add_argument("--weeks", type=int, nargs="*")
    a = ap.parse_args()
    for version in VERSIONS:
        score_version(version, a.weeks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
