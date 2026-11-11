import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.availability import resolve_unavailability

REPO = Path(r"F:\Repos\nfl_py3")
LEAD = REPO / "artifacts" / "lead106_injury_language"
OUT = REPO / "artifacts" / "sim29"
BOOT = 2000


def main():
    s = pd.read_parquet(LEAD / "scored.parquet")
    pg = pd.read_parquet(REPO / "artifacts" / "per03" / "played_by_gsis.parquet")
    pg = pg[pg["season"] == 2026][["week", "team", "gsis_id", "played"]].rename(
        columns={"played": "played_gsis"}
    )
    d = s.merge(pg, on=["week", "team", "gsis_id"], how="left")
    d = d.dropna(subset=["played_gsis", "rp"])
    d["ladder"] = [
        1.0
        - resolve_unavailability(
            None,
            target_season=None,
            report_status=r.report_status,
            practice_status=r.practice_status,
            position=r.position,
        )[0]
        for r in d.itertuples()
    ]
    d["miss_real"] = 1.0 - d["played_gsis"].astype(float)
    d["miss_read"] = 1.0 - d["rp"].astype(float)
    d["miss_ladder"] = 1.0 - d["ladder"]
    g = d.groupby(["week", "team"])[["miss_real", "miss_read", "miss_ladder"]].sum()
    n = len(g)
    res = {"rows": len(d), "team_weeks": int(n), "weeks": sorted(d["week"].unique().tolist())}
    for k in ("miss_read", "miss_ladder"):
        res[k] = {
            "mae": float((g[k] - g["miss_real"]).abs().mean()),
            "corr": float(np.corrcoef(g[k], g["miss_real"])[0, 1]),
        }
    res["mean_realised"] = float(g["miss_real"].mean())
    rng = np.random.default_rng(29)
    diffs, cd = [], []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        x = g.iloc[i]
        diffs.append(
            (x["miss_ladder"] - x["miss_real"]).abs().mean()
            - (x["miss_read"] - x["miss_real"]).abs().mean()
        )
        cd.append(
            np.corrcoef(x["miss_read"], x["miss_real"])[0, 1]
            - np.corrcoef(x["miss_ladder"], x["miss_real"])[0, 1]
        )
    for name, v in (("mae_gain_read_vs_ladder", diffs), ("corr_gain_read_vs_ladder", cd)):
        v = np.array(v)
        res[name] = {
            "point": float(
                (g["miss_ladder"] - g["miss_real"]).abs().mean()
                - (g["miss_read"] - g["miss_real"]).abs().mean()
            )
            if name.startswith("mae")
            else float(
                np.corrcoef(g["miss_read"], g["miss_real"])[0, 1]
                - np.corrcoef(g["miss_ladder"], g["miss_real"])[0, 1]
            ),
            "ci": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "probability_positive": float((v > 0).mean()),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lineup_expectation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
