import argparse
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FOURTH_COLS = ["yardline_100", "ydstogo", "score_differential", "game_seconds_remaining", "qtr"]
TWO_COLS = ["score_differential", "game_seconds_remaining", "qtr"]
MIN_PLAYS = 10
N_BOOT = 2000
STREAM_KEY = 100014
NVD = REPO / "data" / "raw" / "nflverse_pbp_full" / "20261103T131922Z"


def newton_offset(y, p):
    return float((y - p).sum() / (p * (1.0 - p)).sum())


def loso_prob(df, cols, params):
    out = np.full(len(df), np.nan)
    x = df[cols].to_numpy(float)
    y = df["go"].to_numpy(int)
    ssn = df["season"].to_numpy()
    for s in np.unique(ssn):
        m = ssn == s
        clf = H(random_state=0, **params)
        clf.fit(x[~m], y[~m])
        out[m] = clf.predict_proba(x[m])[:, 1]
    return np.clip(out, 1e-4, 1 - 1e-4)


def schedules():
    path = sorted(glob.glob(str(REPO / "data" / "raw" / "*" / "schedules.parquet")))[-1]
    return pd.read_parquet(path)[
        ["game_id", "season", "home_team", "away_team", "home_coach", "away_coach"]
    ]


def fourth_downs(sched):
    d = pd.read_parquet(REPO / "data" / "pbp" / "team_style" / "raw_pbp_narrow.parquet")
    d = d[
        (d.down == 4)
        & d.play_type.isin(["pass", "run", "punt", "field_goal"])
        & (d.qtr <= 4)
        & d.season_type.eq("REG")
    ]
    d = d[~d.qb_kneel.fillna(0).astype(bool) & ~d.qb_spike.fillna(0).astype(bool)]
    d = d.dropna(subset=FOURTH_COLS).copy()
    d["go"] = d.play_type.isin(["pass", "run"]).astype(int)
    d = d.drop(
        columns=[c for c in ("home_team", "season") if c in d.columns and c in sched.columns]
    )
    d = d.merge(sched, on="game_id")
    d["coach"] = np.where(d.posteam == d.home_team, d.home_coach, d.away_coach)
    return d.dropna(subset=["coach"]).reset_index(drop=True)


def two_point_tries(sched):
    cols = [
        "game_id",
        "season",
        "week",
        "season_type",
        "posteam",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
        "two_point_attempt",
        "extra_point_attempt",
    ]
    f = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(NVD.glob("pbp_20*.parquet"))])
    f = f[f.season_type.eq("REG") & ((f.two_point_attempt == 1) | (f.extra_point_attempt == 1))]
    f = f.dropna(subset=["score_differential", "game_seconds_remaining", "posteam"])
    f = f.drop(columns=["season"]).merge(sched, on="game_id")
    f["coach"] = np.where(f.posteam == f.home_team, f.home_coach, f.away_coach)
    f["go"] = (f.two_point_attempt == 1).astype(int)
    return f.dropna(subset=["coach"]).reset_index(drop=True)


def with_offsets(df, cols, params):
    df = df.copy()
    p = loso_prob(df, cols, params)
    lg = np.log(p / (1.0 - p))
    for s in df["season"].unique():
        m = (df["season"] == s).to_numpy()
        a = newton_offset(df["go"].to_numpy(float)[m], p[m])
        lg[m] = lg[m] + a
    df["p"] = np.clip(1.0 / (1.0 + np.exp(-lg)), 1e-4, 1 - 1e-4)
    return df


def spell_offsets(df):
    g = df.groupby(["season", "posteam", "coach"])
    t = g.apply(
        lambda z: pd.Series(
            {"a": newton_offset(z.go.to_numpy(float), z.p.to_numpy()), "n": len(z)}
        ),
        include_groups=False,
    ).reset_index()
    return t[t.n >= MIN_PLAYS]


def consecutive_pairs(t):
    nxt = t.assign(season=t.season - 1)
    m = t.merge(nxt, on=["season", "posteam", "coach"], suffixes=("0", "1"))
    m["spell"] = m.posteam + "|" + m.coach
    return m


def cluster_boot(pairs, stat, rng):
    groups = [g for _, g in pairs.groupby("spell")]
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(pd.concat([groups[i] for i in pick])))
    return np.array(vals)


def cov_stat(m):
    return float(np.cov(m.a0.to_numpy(), m.a1.to_numpy())[0, 1])


def slope_stat(m):
    return float(np.polyfit(m.a0.to_numpy(), m.a1.to_numpy(), 1)[0])


def interval(v):
    q = np.quantile(v, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def hazard(sched, rng):
    s = pd.concat(
        [
            sched.rename(columns={"home_team": "team", "home_coach": "coach"})[
                ["season", "team", "coach", "game_id"]
            ],
            sched.rename(columns={"away_team": "team", "away_coach": "coach"})[
                ["season", "team", "coach", "game_id"]
            ],
        ]
    ).dropna(subset=["coach"])
    modal = (
        s.groupby(["season", "team", "coach"])
        .size()
        .reset_index(name="n")
        .sort_values(["season", "team", "n"], ascending=[True, True, False])
        .drop_duplicates(["season", "team"])
    )
    nxt = modal.assign(season=modal.season - 1)
    m = modal.merge(nxt, on=["season", "team"], suffixes=("0", "1"))
    m["changed"] = (m.coach0 != m.coach1).astype(float)
    teams = [g.changed.to_numpy() for _, g in m.groupby("team")]
    boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(teams), len(teams))
        boot.append(float(np.concatenate([teams[i] for i in pick]).mean()))
    return float(m.changed.mean()), interval(boot), len(m)


def two_point_trait(f, other_coach_offsets, rng):
    f = f.sort_values(["coach", "season", "week"]).copy()
    gk = (
        f[["coach", "season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["coach", "season", "week"])
    )
    gk["half"] = gk.groupby("coach").cumcount() % 2
    f = f.merge(gk[["game_id", "coach", "half"]], on=["game_id", "coach"])
    rows = []
    for (coach, half), z in f.groupby(["coach", "half"]):
        if len(z) >= 15:
            rows.append((coach, half, newton_offset(z.go.to_numpy(float), z.p.to_numpy())))
    r = (
        pd.DataFrame(rows, columns=["coach", "half", "a"])
        .pivot(index="coach", columns="half", values="a")
        .dropna()
    )
    r = r.rename(columns={0: "a0", 1: "a1"}).reset_index()
    r["spell"] = r.coach
    cov = cov_stat(r)
    boot = cluster_boot(r, cov_stat, rng)
    both = r.merge(other_coach_offsets, on="coach")
    return r, cov, boot, both


def go_flags(out_path):
    import sim10_drive as sd

    seg = pd.read_parquet(sd.OUT / "segments.parquet")
    p = sd.load_pbp()
    r = seg["row"].to_numpy()
    assert (p["game_id"].to_numpy()[r] == seg["game_id"].to_numpy()).all()
    fd = p["fixed_drive"].to_numpy()[r]
    goes = p[(p.down == 4) & p.play_type.isin(["run", "pass"]) & p.posteam.notna()]
    key = set(zip(goes.game_id, goes.fixed_drive, strict=True))
    flag = np.array([(g, d) in key for g, d in zip(seg["game_id"], fd, strict=True)], dtype=np.int8)
    np.save(out_path, flag)
    return len(flag), int(flag.sum())


def main():
    ap = argparse.ArgumentParser(
        description="Fit the coach-spell fourth-down and two-point go latents for the engine hook"
    )
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sim14" / "hook.json"))
    ap.add_argument("--skip-flags", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(14)
    sched = schedules()
    d = with_offsets(
        fourth_downs(sched),
        FOURTH_COLS,
        {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 12, "min_samples_leaf": 60},
    )
    t4 = spell_offsets(d)
    pairs = consecutive_pairs(t4)
    cov4 = cov_stat(pairs)
    cov4_boot = cluster_boot(pairs, cov_stat, rng)
    slope4 = slope_stat(pairs)
    slope4_boot = cluster_boot(pairs, slope_stat, rng)
    hz, hz_int, hz_n = hazard(sched, rng)
    f = with_offsets(
        two_point_tries(sched),
        TWO_COLS,
        {"max_iter": 60, "learning_rate": 0.08, "max_leaf_nodes": 6, "min_samples_leaf": 80},
    )
    coach4 = (
        d.groupby("coach")
        .apply(lambda z: newton_offset(z.go.to_numpy(float), z.p.to_numpy()), include_groups=False)
        .rename("a4")
        .reset_index()
    )
    r2, cov2, cov2_boot, both = two_point_trait(f, coach4, rng)
    mid = (both.a0 + both.a1) / 2.0
    corr = float(np.corrcoef(mid, both.a4)[0, 1]) if len(both) > 3 else float("nan")
    cb = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(both), len(both))
        cb.append(np.corrcoef(mid.to_numpy()[ix], both.a4.to_numpy()[ix])[0, 1])
    res = {
        "fourth_down": {
            "plays": len(d),
            "go_rate": float(d.go.mean()),
            "spell_pairs": len(pairs),
            "trait_variance_logit": cov4,
            "trait_variance_interval": interval(cov4_boot),
            "trait_sd_logit": float(np.sqrt(max(cov4, 0.0))),
            "trait_sd_interval": [float(np.sqrt(max(x, 0.0))) for x in interval(cov4_boot)],
            "probability_variance_positive": float((cov4_boot > 0).mean()),
            "next_season_slope": slope4,
            "next_season_slope_interval": interval(slope4_boot),
            "observed_offset_sd": float(t4.a.std()),
        },
        "coach_change_hazard": {"value": hz, "interval": hz_int, "team_season_pairs": hz_n},
        "two_point": {
            "tries": len(f),
            "go_rate": float(f.go.mean()),
            "coaches": len(r2),
            "trait_variance_logit": cov2,
            "trait_variance_interval": interval(cov2_boot),
            "trait_sd_logit": float(np.sqrt(max(cov2, 0.0))),
            "trait_sd_interval": [float(np.sqrt(max(x, 0.0))) for x in interval(cov2_boot)],
            "probability_variance_positive": float((cov2_boot > 0).mean()),
            "corr_with_fourth_down_offset": corr,
            "corr_interval": interval(np.array(cb)),
            "corr_coaches": len(both),
        },
        "stream_key": STREAM_KEY,
        "go_flags": None,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not a.skip_flags:
        fp = out.parent / "go_flags.npy"
        n, k = go_flags(fp)
        res["go_flags"] = {"path": str(fp), "segments": n, "flagged": k}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
