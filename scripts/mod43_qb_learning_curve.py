from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
PBP_ROOT = REPO / "data" / "pbp" / "raw"
OPEN_PATH = REPO / "artifacts" / "opener_evaluation" / "20261103T195331Z" / "per_game.parquet"
OUT = REPO / "tests" / "scratch" / "mod43"
FIRST_SEASON = 2009
EARLY_SEASONS_SINCE = 1
TEST_SEASONS = tuple(range(2020, 2026))
DRAWS = 2000
SEED = 20261103
INDICES = ("career", "team")
FLAGS = {"career": ("debut_observed", "early_career"), "team": ("stint_observed", "early_team")}


def newest(root):
    return sorted(p.name for p in root.iterdir() if p.is_dir())[-1]


def load_pbp():
    stamp = newest(PBP_ROOT)
    cols = [
        "game_id",
        "season",
        "season_type",
        "week",
        "home_team",
        "away_team",
        "posteam",
        "qb_dropback",
        "passer_player_id",
        "epa",
    ]
    frames = []
    for path in sorted(glob.glob(str(PBP_ROOT / stamp / "season=*" / "*.parquet"))):
        s = int(os.path.basename(os.path.dirname(path)).split("=")[1])
        if s < FIRST_SEASON:
            continue
        d = pd.read_parquet(path, columns=cols)
        d = d.loc[
            (d["season_type"] == "REG")
            & (d["qb_dropback"] == 1)
            & d["passer_player_id"].notna()
            & d["epa"].notna()
        ]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def starters(pbp):
    per = pbp.groupby(
        ["game_id", "season", "week", "posteam", "passer_player_id"], as_index=False
    ).agg(db=("epa", "size"), epa=("epa", "sum"))
    per = per.sort_values(["game_id", "posteam", "db"], ascending=[True, True, False])
    s = per.groupby(["game_id", "posteam"], as_index=False).first()
    s["t"] = s["season"] * 100 + s["week"]
    s = s.sort_values(["passer_player_id", "t"]).reset_index(drop=True)
    g = s.groupby("passer_player_id")
    s["career_n"] = g.cumcount()
    s["debut_season"] = g["season"].transform("min")
    s["debut_observed"] = s["debut_season"] > FIRST_SEASON
    new_stint = s["posteam"] != g["posteam"].shift(1)
    s["stint"] = new_stint.groupby(s["passer_player_id"]).cumsum()
    gs = s.groupby(["passer_player_id", "stint"])
    s["team_n"] = gs.cumcount()
    s["stint_start_season"] = gs["season"].transform("min")
    s["stint_observed"] = s["debut_observed"] | (s["stint"] > 1)
    s["epa_db"] = s["epa"] / s["db"]
    s["epa_c"] = s["epa_db"] - s.groupby("season")["epa_db"].transform("mean")
    s["early_career"] = s["debut_observed"] & (
        s["season"] - s["debut_season"] <= EARLY_SEASONS_SINCE
    )
    s["early_team"] = s["stint_observed"] & (
        s["season"] - s["stint_start_season"] <= EARLY_SEASONS_SINCE
    )
    for idx in INDICES:
        s[f"x_{idx}"] = np.log1p(s[f"{idx}_n"])
    return s


def fe_slope(d, x):
    xm = d[x] - d.groupby("passer_player_id")[x].transform("mean")
    ym = d["epa_c"] - d.groupby("passer_player_id")["epa_c"].transform("mean")
    w = d["db"]
    den = (w * xm * xm).sum()
    return float((w * xm * ym).sum() / den) if den > 0 else np.nan


def boot_slope(d, x, rng):
    ids = d["passer_player_id"].unique()
    parts = {k: g for k, g in d.groupby("passer_player_id")}  # noqa: C416
    out = []
    for _ in range(DRAWS):
        pick = rng.choice(ids, len(ids))
        b = pd.concat([parts[k].assign(passer_player_id=f"{k}_{i}") for i, k in enumerate(pick)])
        out.append(fe_slope(b, x))
    return np.array(out)


def groups(s, idx):
    obs_col, early_col = FLAGS[idx]
    obs, early = s[obs_col], s[early_col]
    out = {}
    for name, mask in (("early", obs & early), ("veteran", obs & ~early)):
        d = s[mask]
        out[name] = d[d.groupby("passer_player_id")["epa_c"].transform("size") >= 2]
    return out


def curve_cells(s):
    rng = np.random.default_rng(SEED)
    rows = []
    for idx in INDICES:
        for name, d in groups(s, idx).items():
            x = f"x_{idx}"
            b = boot_slope(d, x, rng)
            rows.append(
                {
                    "index": idx,
                    "group": name,
                    "n_games": len(d),
                    "n_qbs": d["passer_player_id"].nunique(),
                    "slope": fe_slope(d, x),
                    "lo": float(np.percentile(b, 2.5)),
                    "hi": float(np.percentile(b, 97.5)),
                    "p_pos": float((b > 0).mean()),
                    "odd_seasons": fe_slope(d[d["season"] % 2 == 1], x),
                    "even_seasons": fe_slope(d[d["season"] % 2 == 0], x),
                }
            )
    return pd.DataFrame(rows)


def slopes_excluding(s, season):
    tr = s[s["season"] != season]
    return {
        (idx, name): fe_slope(d, f"x_{idx}")
        for idx in INDICES
        for name, d in groups(tr, idx).items()
    }


def feature_table(s):
    by_qb = {k: g for k, g in s.groupby("passer_player_id")}  # noqa: C416
    ts = s.sort_values(["posteam", "t"]).copy()
    ts["prior_qb"] = ts.groupby("posteam")["passer_player_id"].shift(1)
    recs = []
    for r in ts.itertuples():
        rec = {"game_id": r.game_id, "team": r.posteam}
        q = r.prior_qb
        prior = by_qb[q][by_qb[q]["t"] < r.t] if isinstance(q, str) else None
        if prior is None or prior.empty:
            recs.append(rec)
            continue
        last = prior.iloc[-1]
        ss = prior[prior["season"] < r.season]
        n_c0 = ss["career_n"].iloc[-1] + 1 if len(ss) else 0
        st = prior[(prior["posteam"] == r.posteam) & (prior["stint"] == last["stint"])]
        st0 = st[st["season"] < r.season]
        n_t0 = st0["team_n"].iloc[-1] + 1 if len(st0) else 0
        rec.update(
            {
                "dcareer": np.log1p(last["career_n"] + 1) - np.log1p(n_c0),
                "dteam": np.log1p(last["team_n"] + 1) - np.log1p(n_t0),
                "obs_career": bool(last["debut_observed"]),
                "early_career": bool(
                    last["debut_observed"]
                    and (r.season - last["debut_season"] <= EARLY_SEASONS_SINCE)
                ),
                "obs_team": bool(last["stint_observed"]),
                "early_team": bool(
                    last["stint_observed"]
                    and (r.season - last["stint_start_season"] <= EARLY_SEASONS_SINCE)
                ),
            }
        )
        recs.append(rec)
    return pd.DataFrame(recs)


def project(f, slopes, idx):
    e = f[f"early_{idx}"].fillna(False).astype(bool)
    ob = f[f"obs_{idx}"].fillna(False).astype(bool)
    v = np.where(
        e,
        slopes[(idx, "early")] * f[f"d{idx}"],
        np.where(ob, slopes[(idx, "veteran")] * f[f"d{idx}"], 0.0),
    )
    return np.nan_to_num(v)


def ll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pbp = load_pbp()
    games = pbp.drop_duplicates("game_id")[["game_id", "home_team", "away_team"]]
    s = starters(pbp)
    cells = curve_cells(s)
    cells.to_csv(OUT / "curve_cells.csv", index=False)
    print(cells.round(4).to_string())
    ft = feature_table(s)
    o = pd.read_parquet(OPEN_PATH)
    d = o.merge(games, on="game_id")
    d = d[d.season.between(2020, 2025) & d.margin_vs_open.ne(0)].copy()
    d["y"] = (d.margin_vs_open > 0).astype(int)
    d["z"] = logit(d.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6))
    hf = d[["game_id", "home_team"]].merge(
        ft, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left"
    )
    af = d[["game_id", "away_team"]].merge(
        ft, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left"
    )
    hf.index = d.index
    af.index = d.index
    for idx in INDICES:
        d[f"g_{idx}"] = 0.0
    for sea in TEST_SEASONS:
        sl = slopes_excluding(s, sea)
        m = (d.season == sea).values
        for idx in INDICES:
            d.loc[m, f"g_{idx}"] = project(hf[m], sl, idx) - project(af[m], sl, idx)
    print("raw sd", d[["g_career", "g_team"]].std().round(5).to_dict())
    print("nonzero", int((d.g_career != 0).sum()), int((d.g_team != 0).sum()))
    for c in ("g_career", "g_team"):
        d[c] = d[c] / d[c].std()
    arms = {
        "intercept": [],
        "served": ["z"],
        "served+career": ["z", "g_career"],
        "served+team": ["z", "g_team"],
    }
    P, coef, insamp = {}, {}, {}
    for a, cols in arms.items():
        p = pd.Series(0.5, index=d.index)
        coef[a] = {}
        for sea in TEST_SEASONS:
            tr = d[d.season != sea]
            te = d.season == sea
            if not cols:
                p[te] = tr.y.mean()
                continue
            mdl = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.y)
            p[te] = mdl.predict_proba(d.loc[te, cols])[:, 1]
            coef[a][sea] = np.round(mdl.coef_[0], 3).tolist()
        P[a] = p.clip(1e-6, 1 - 1e-6)
        if cols:
            mdl = LogisticRegression(C=1e6, max_iter=1000).fit(d[cols], d.y)
            insamp[a] = float(ll(d.y, mdl.predict_proba(d[cols])[:, 1]).mean())
        else:
            insamp[a] = float(ll(d.y, d.y.mean()).mean())
    rng = np.random.default_rng(SEED)
    res = {"n": len(d), "arms": {}, "coef": coef, "curve": cells.to_dict("records")}
    for a in arms:
        w = int(((P[a] > 0.5) == (d.y == 1)).sum())
        res["arms"][a] = {
            "W": w,
            "L": len(d) - w,
            "ll_oos": float(ll(d.y, P[a]).mean()),
            "ll_in": insamp[a],
            "brier": float(((P[a] - d.y) ** 2).mean()),
        }
        print(a, res["arms"][a])
    for a in ("served+career", "served+team"):
        flip = (P[a] > 0.5) != (P["served"] > 0.5)
        yy = d.y[flip] == 1
        wa = int(((P[a][flip] > 0.5) == yy).sum())
        ws = int(((P["served"][flip] > 0.5) == yy).sum())
        diff = (ll(d.y, P["served"]) - ll(d.y, P[a])).values
        b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(DRAWS)])
        sb = pd.Series(diff).groupby(d["season"].values).mean()
        res[a] = {
            "decisive_n": int(flip.sum()),
            "decisive_aug_wins": wa,
            "decisive_served_wins": ws,
            "dLL": float(diff.mean()),
            "lo": float(np.percentile(b, 2.5)),
            "hi": float(np.percentile(b, 97.5)),
            "se": float(b.std()),
            "p_pos": float((b > 0).mean()),
            "seasons_pos": int((sb > 0).sum()),
            "gap_in_minus_oos_ll": insamp[a] - res["arms"][a]["ll_oos"],
        }
        print(a, res[a])
    (OUT / "result.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
