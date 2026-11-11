import argparse
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

MIN_DRIVES = 10
N_BOOT = 2000
STREAM_KEY = 100018
BIN_SECS = 300.0
BANDS = {"1-8": (1, 8), "9-16": (9, 16), "17+": (17, 99)}


def load():
    seg = pd.read_parquet(REPO / "artifacts" / "sim10" / "segments.parquet")
    sched = pd.read_parquet(
        sorted(glob.glob(str(REPO / "data" / "raw" / "*" / "schedules.parquet")))[-1]
    )[["game_id", "home_team", "away_team"]]
    seg = seg.merge(sched, on="game_id")
    seg["team"] = np.where(seg["home"] == 1, seg["home_team"], seg["away_team"])
    seg = seg[(seg["half"] == 2) & (seg["he"] == 0) & (seg["el"] < seg["secs"]) & (seg["el"] > 0)]
    seg["bin"] = (seg["secs"] // BIN_SECS).astype(int)
    seg["lel"] = np.log(seg["el"])
    return seg


def staff():
    ss = pd.read_parquet(
        sorted(glob.glob(str(REPO / "artifacts" / "staff_state" / "*" / "staff_state.parquet")))[-1]
    )
    ss = ss[ss.state.eq("filled") & ss.role.eq("OC")]
    return (
        ss.sort_values("first_observed_at").groupby(["season", "team"]).person.first().reset_index()
    )


def band_offsets(seg, oc, lo, hi):
    z = seg[(seg["sd"] >= lo) & (seg["sd"] <= hi)].copy()
    z["mu"] = np.nan
    for s in z["season"].unique():
        m = z["season"] == s
        cell = z[~m].groupby("bin")["lel"].mean()
        z.loc[m, "mu"] = z.loc[m, "bin"].map(cell)
    z["r"] = z["lel"] - z["mu"]
    z = z.dropna(subset=["r"]).merge(oc, on=["season", "team"])
    t = z.groupby(["season", "team", "person"])["r"].agg(["mean", "size"]).reset_index()
    t = t[t["size"] >= MIN_DRIVES].rename(columns={"mean": "a"})
    nxt = t.assign(season=t.season - 1)
    m = t.merge(nxt, on=["season", "team", "person"], suffixes=("0", "1"))
    m["spell"] = m.team + "|" + m.person
    return m, len(z), float(z["r"].std())


def interval(v):
    q = np.quantile(v, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def boot(pairs, stat, rng):
    groups = [g for _, g in pairs.groupby("spell")]
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(stat(pd.concat([groups[i] for i in pick])))
    return np.array(out)


def cov_stat(m):
    return float(np.cov(m.a0.to_numpy(), m.a1.to_numpy())[0, 1])


def slope_stat(m):
    return float(np.polyfit(m.a0.to_numpy(), m.a1.to_numpy(), 1)[0])


def oc_hazard(oc, rng):
    nxt = oc.assign(season=oc.season - 1)
    m = oc.merge(nxt, on=["season", "team"], suffixes=("0", "1"))
    m["changed"] = (m.person0 != m.person1).astype(float)
    teams = [g.changed.to_numpy() for _, g in m.groupby("team")]
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(teams), len(teams))
        bs.append(float(np.concatenate([teams[i] for i in pick]).mean()))
    return float(m.changed.mean()), interval(bs), len(m)


def main():
    ap = argparse.ArgumentParser(
        description="Fit the OC-spell lead-protection drive-length latent for the engine hook"
    )
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sim18" / "hook.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(18)
    seg = load()
    oc = staff()
    res = {"bands": {}}
    for name, (lo, hi) in BANDS.items():
        pairs, n, sdr = band_offsets(seg, oc, lo, hi)
        cv = boot(pairs, cov_stat, rng)
        sl = boot(pairs, slope_stat, rng)
        c0 = cov_stat(pairs)
        res["bands"][name] = {
            "drives": n,
            "log_ratio_sd": sdr,
            "spell_pairs": len(pairs),
            "trait_variance_log": c0,
            "trait_variance_interval": interval(cv),
            "trait_sd_log": float(np.sqrt(max(c0, 0.0))),
            "trait_sd_interval": [float(np.sqrt(max(x, 0.0))) for x in interval(cv)],
            "probability_variance_positive": float((cv > 0).mean()),
            "next_season_slope": slope_stat(pairs),
            "next_season_slope_interval": interval(sl),
        }
    hz, hz_int, hz_n = oc_hazard(oc, rng)
    res["oc_change_hazard"] = {"value": hz, "interval": hz_int, "team_season_pairs": hz_n}
    res["lead_band"] = "1-8"
    res["lead_lo"] = BANDS["1-8"][0]
    res["lead_hi"] = BANDS["1-8"][1]
    res["stream_key"] = STREAM_KEY
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
