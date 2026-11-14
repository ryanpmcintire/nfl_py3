import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per16_team_designation_habit as H
import per16_unit2_confounds as U

ROOT = H.ROOT
ART = H.ART
FAMILY = "per16_habit_availability_addone"
SEED_SOURCE = "per16_unit4_availability_addone"
SERVED = "out_of_season_home_probability"
TERMS = ("league_avail", "habit_avail", "habit_adj")
PRIMARY = "habit_adj"
COMPARISONS = (
    ("habit_adj", "served"),
    ("habit_adj", "intercept_only"),
    ("habit_avail", "league_avail"),
    ("habit_avail", "served"),
    ("league_avail", "served"),
    ("intercept_only", "served"),
)
STAMPED_COMPARISONS = COMPARISONS[:2]
FORWARD_COMPARISONS = (
    ("habit_adj", "served"),
    ("habit_avail", "served"),
    ("intercept_only", "served"),
)
TEAM_SUMS = ("n_q", "miss_league", "miss_habit", "adj")
BOOT_REPS = 5000
SHUFFLE_REPS = 500
NEWTON_STEPS = 100
NEWTON_TOL = 1e-10
INTERVAL = (2.5, 97.5)
SHUFFLE_BAND = (5.0, 95.0)
EVEN = 0.5


def seed():
    return int(hashlib.sha256(SEED_SOURCE.encode()).hexdigest()[:8], 16)


def served_games():
    path = sorted((ROOT / "artifacts" / "pick_probability").glob("*/per_game.parquet"))[-1]
    g = pd.read_parquet(
        path,
        columns=[
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_covered",
            SERVED,
        ],
    )
    g = g.dropna(subset=[SERVED, "home_covered"]).reset_index(drop=True)
    g["cut"] = pd.to_datetime(g["gameday"]).dt.tz_localize("UTC")
    g["p_served"] = g[SERVED].astype(float)
    return g, str(path)


def forward_games():
    led = pd.read_parquet(
        ROOT / "artifacts" / "forward_ledger" / "ledger.parquet",
        columns=[
            "game_id",
            "week",
            "deadline_utc",
            "served_side",
            "served_p",
            "card_line",
            "home_cover",
        ],
    )
    parts = led["game_id"].str.split("_", expand=True)
    led["season"] = parts[0].astype(int)
    led["away_team"] = parts[2]
    led["home_team"] = parts[3]
    home = led["served_side"].str.upper() == "HOME"
    side_probability = bool((led["served_p"] >= EVEN).all() and (~home).any())
    led["p_served"] = np.where(home | (not side_probability), led["served_p"], 1 - led["served_p"])
    led["cut"] = led["deadline_utc"].dt.normalize()
    led["home_covered"] = led["home_cover"]
    info = {
        "ledger_games": len(led),
        "served_p_is_side_probability": side_probability,
        "home_side_share": float(home.mean()),
        "home_cover_values": {
            str(k): int(v) for k, v in led["home_cover"].value_counts(dropna=False).items()
        },
    }
    g = led[led["home_cover"].isin([0.0, 1.0]) & led["p_served"].between(0, 1, inclusive="neither")]
    return g.reset_index(drop=True), info


def train_rows(coaches):
    x = pd.read_parquet(ROOT / "artifacts" / "per03" / "played_by_gsis.parquet")
    x = x[
        x["team_game"]
        & x["source"].str.startswith("participation")
        & (x["report_status"] == U.DESIG)
    ]
    return U.prep(x.merge(coaches, on=["season", "team"], how="left"))


def listed_rows(coaches):
    inj = pd.read_parquet(H.latest("data/raw/nflverse_injuries/*/injuries.parquet"))
    inj = inj[(inj["game_type"] == "REG") & inj["week"].notna()]
    inj = inj.sort_values("date_modified").drop_duplicates(
        ["season", "week", "gsis_id"], keep="last"
    )
    q = inj[inj["report_status"] == U.DESIG].copy()
    q["week"] = q["week"].astype(int)
    return U.prep(q.merge(coaches, on=["season", "team"], how="left"))


def habit_probabilities(train, target):
    train = train.copy()
    cells = U.fit_cells(train)
    train["p_dxp"], train["p_ctrl"] = U.predict_cells(cells, train)
    offset, tau2 = U.eb_offset(train, "coach", "p_ctrl")
    t = target.copy()
    _, ctrl = U.predict_cells(cells, t)
    clip = 1.0 / (len(train) + 2)
    t["offset"] = t["coach"].map(offset).fillna(0.0)
    t["p_ctrl"] = ctrl.clip(clip, 1 - clip)
    t["p_coach"] = (ctrl + t["offset"]).clip(clip, 1 - clip)
    meta = {
        "train_rows": len(train),
        "train_first_season": int(train["season"].min()),
        "train_last_season": int(train["season"].max()),
        "league_play_rate": float(cells["lg"]),
        "coach_tau2": float(tau2),
        "coaches_fitted": len(offset),
        "offset_sd_across_coaches": float(offset.std(ddof=1)),
        "target_rows": len(t),
        "target_rows_with_offset": int(t["coach"].isin(offset.index).sum()),
    }
    return t, meta


def build(games, listed, train):
    rows = []
    meta = []
    for s in sorted(games["season"].unique()):
        t, m = habit_probabilities(train[train["season"] < s], listed[listed["season"] == s])
        rows.append(t)
        meta.append({"season": int(s), **m})
    long = pd.concat(
        [
            games[["season", "week", f"{side}_team", "cut"]].rename(
                columns={f"{side}_team": "team"}
            )
            for side in ("home", "away")
        ]
    )
    r = pd.concat(rows).merge(long, on=["season", "week", "team"], how="inner")
    r["stamped"] = r["date_modified"].notna()
    r["late"] = r["stamped"] & (r["date_modified"] >= r["cut"])
    used = r[~r["late"]].assign(
        n_q=1.0,
        miss_league=lambda d: 1 - d["p_ctrl"],
        miss_habit=lambda d: 1 - d["p_coach"],
        adj=lambda d: d["p_coach"] - d["p_ctrl"],
    )
    agg = used.groupby(["season", "week", "team"], as_index=False)[list(TEAM_SUMS)].sum()
    out = games.copy()
    for side in ("home", "away"):
        a = agg.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in TEAM_SUMS}})
        out = out.merge(a, on=["season", "week", f"{side}_team"], how="left")
        for c in TEAM_SUMS:
            out[f"{side}_{c}"] = out[f"{side}_{c}"].fillna(0.0)
    out["league_avail"] = -(out["home_miss_league"] - out["away_miss_league"])
    out["habit_avail"] = -(out["home_miss_habit"] - out["away_miss_habit"])
    out["habit_adj"] = out["home_adj"] - out["away_adj"]
    by_season = r.groupby("season").agg(
        rows=("late", "size"), stamped=("stamped", "sum"), late=("late", "sum")
    )
    info = {
        "habit_fits": meta,
        "rows_by_season": {
            int(s): {k: int(v) for k, v in row.items()} for s, row in by_season.iterrows()
        },
        "stamped_seasons": [int(s) for s in by_season.index[by_season["stamped"] > 0]],
        "term_stats": {
            t: {
                "mean": float(out[t].mean()),
                "sd": float(out[t].std(ddof=1)),
                "nonzero_games": int((out[t] != 0).sum()),
            }
            for t in TERMS
        },
        "listed_per_team_game": float(
            (out["home_n_q"].sum() + out["away_n_q"].sum()) / (2 * len(out))
        ),
    }
    return out, info


def fit(z, x, y):
    cols = [np.ones(len(y))] if x is None else [np.ones(len(y)), x]
    design = np.column_stack(cols)
    b = np.zeros(design.shape[1])
    for _ in range(NEWTON_STEPS):
        p = expit(z + design @ b)
        step = np.linalg.solve(design.T @ (design * (p * (1 - p))[:, None]), design.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < NEWTON_TOL:
            break
    return b


def fitted(z, x, y):
    if x is None:
        b = fit(z, None, y)
        return {"intercept": float(b[0])}
    sd = float(np.std(x, ddof=1))
    b = fit(z, x / sd, y)
    return {"intercept": float(b[0]), "coef_per_sd": float(b[1]), "sd": sd}


def predict(z, x, c):
    if x is None:
        return expit(z + c["intercept"])
    return expit(z + c["intercept"] + c["coef_per_sd"] * x / c["sd"])


def loso(z, y, season, x):
    p = np.empty(len(y))
    folds = []
    for s in np.unique(season):
        tr = season != s
        c = fitted(z[tr], None if x is None else x[tr], y[tr])
        p[~tr] = predict(z[~tr], None if x is None else x[~tr], c)
        folds.append({"held_out": int(s), **c})
    return p, folds


def scores(p, y):
    return {
        "correct": ((p > EVEN) == (y == 1)).astype(float),
        "ll": -(y * np.log(p) + (1 - y) * np.log1p(-p)),
        "brier": (p - y) ** 2,
    }


def deltas(pa, pb, y):
    a = scores(pa, y)
    b = scores(pb, y)
    return {
        "accuracy_points": 100 * (a["correct"] - b["correct"]),
        "log_loss_improvement": b["ll"] - a["ll"],
        "brier_improvement": b["brier"] - a["brier"],
    }


def effects(pa, pb, y):
    return {k: float(v.mean()) for k, v in deltas(pa, pb, y).items()}


def compare(pa, pb, y, rng):
    idx = rng.integers(0, len(y), size=(BOOT_REPS, len(y)))
    out = {"n": len(y)}
    for k, v in deltas(pa, pb, y).items():
        m = v[idx].mean(axis=1)
        lo, hi = np.percentile(m, INTERVAL)
        out[k] = {
            "effect": float(v.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "probability_positive": float((m > 0).mean() + EVEN * (m == 0).mean()),
        }
    flip = (pa > EVEN) != (pb > EVEN)
    a = scores(pa, y)["correct"]
    b = scores(pb, y)["correct"]
    out["decisive"] = {
        "games": int(flip.sum()),
        "new_right": int(a[flip].sum()),
        "base_right": int(b[flip].sum()),
    }
    return out


def shuffled_control(z, y, season, x, probs, real, rng):
    rows = []
    for _ in range(SHUFFLE_REPS):
        xs = x.copy()
        for s in np.unique(season):
            m = season == s
            xs[m] = rng.permutation(xs[m])
        p, folds = loso(z, y, season, xs)
        row = {"coef_per_sd": float(np.mean([f["coef_per_sd"] for f in folds]))}
        for base in ("served", "intercept_only"):
            for k, v in effects(p, probs[base], y).items():
                row[f"{base}:{k}"] = v
        rows.append(row)
    d = pd.DataFrame(rows)
    out = {}
    for c in d.columns:
        lo, hi = np.percentile(d[c], SHUFFLE_BAND)
        out[c] = {"mean": float(d[c].mean()), "p05": float(lo), "p95": float(hi)}
        if c in real:
            out[c]["real"] = real[c]
            out[c]["share_of_shuffles_at_or_above_real"] = float((d[c] >= real[c]).mean())
    return out


def grade(g, comparisons, rng, shuffle):
    z = logit(g["p_served"].to_numpy(float))
    y = g["home_covered"].to_numpy(float)
    season = g["season"].to_numpy()
    probs = {"served": expit(z)}
    inside = {"served": expit(z)}
    folds = {}
    coefs = {}
    for arm in ("intercept_only", *TERMS):
        x = None if arm == "intercept_only" else g[arm].to_numpy(float)
        probs[arm], folds[arm] = loso(z, y, season, x)
        coefs[arm] = fitted(z, x, y)
        inside[arm] = predict(z, x, coefs[arm])
    cells = {}
    for a, b in comparisons:
        c = compare(probs[a], probs[b], y, rng)
        ins = effects(inside[a], inside[b], y)
        for k, v in ins.items():
            c[k]["in_sample"] = v
            c[k]["gap_in_minus_out"] = v - c[k]["effect"]
        cells[f"{a}__vs__{b}"] = c
    out = {
        "games": len(g),
        "seasons": [int(s) for s in np.unique(season)],
        "served_accuracy": float(scores(probs["served"], y)["correct"].mean()),
        "served_log_loss": float(scores(probs["served"], y)["ll"].mean()),
        "served_brier": float(scores(probs["served"], y)["brier"].mean()),
        "cells": cells,
        "folds": folds,
        "in_sample_coefficients": coefs,
        "fold_sign_share": {
            t: {
                "positive_folds": int(sum(f["coef_per_sd"] > 0 for f in folds[t])),
                "folds": len(folds[t]),
            }
            for t in TERMS
        },
    }
    if shuffle:
        real = {"coef_per_sd": float(np.mean([f["coef_per_sd"] for f in folds[PRIMARY]]))}
        for base in ("served", "intercept_only"):
            for k, v in effects(probs[PRIMARY], probs[base], y).items():
                real[f"{base}:{k}"] = v
        out["shuffled_control"] = shuffled_control(
            z, y, season, g[PRIMARY].to_numpy(float), probs, real, rng
        )
    return out, probs


def forward(coefs, listed, train, rng):
    g, info = forward_games()
    if g.empty:
        return {**info, "graded_games": 0}
    g, binfo = build(g, listed, train)
    z = logit(g["p_served"].to_numpy(float))
    y = g["home_covered"].to_numpy(float)
    probs = {"served": expit(z), "intercept_only": predict(z, None, coefs["intercept_only"])}
    for t in TERMS:
        probs[t] = predict(z, g[t].to_numpy(float), coefs[t])
    cells = {f"{a}__vs__{b}": compare(probs[a], probs[b], y, rng) for a, b in FORWARD_COMPARISONS}
    keep = g[["game_id", "season", "week", "home_covered", "p_served", *TERMS]].copy()
    for k, v in probs.items():
        keep[f"p_{k}"] = v
    keep.to_parquet(ART / "unit4_forward_games.parquet", index=False)
    return {
        **info,
        "graded_games": len(g),
        "weeks": [int(w) for w in sorted(g["week"].unique())],
        "served_accuracy": float(scores(probs["served"], y)["correct"].mean()),
        "build": binfo,
        "cells": cells,
    }


def line(name, c):
    d = c["decisive"]
    parts = [f"{name} n={c['n']} decisive new {d['new_right']}-{d['base_right']} base"]
    for k in ("accuracy_points", "log_loss_improvement", "brier_improvement"):
        v = c[k]
        s = f"{k} {v['effect']:+.5f} [{v['lo']:+.5f},{v['hi']:+.5f}]"
        s += f" P+ {v['probability_positive']:.3f}"
        if "in_sample" in v:
            s += f" in {v['in_sample']:+.5f} gap {v['gap_in_minus_out']:+.5f}"
        parts.append(s)
    return "\n    ".join(parts)


def main():
    rng = np.random.default_rng(seed())
    coaches = H.head_coaches()
    listed = listed_rows(coaches)
    train = train_rows(coaches)
    games, source = served_games()
    g, info = build(games, listed, train)
    full, probs = grade(g, COMPARISONS, rng, True)
    stamped, _ = grade(
        g[g["season"].isin(info["stamped_seasons"])], STAMPED_COMPARISONS, rng, False
    )
    fwd = forward(full["in_sample_coefficients"], listed, train, rng)
    keep = g[
        [
            "game_id",
            "season",
            "week",
            "home_covered",
            "p_served",
            *TERMS,
            "home_n_q",
            "away_n_q",
        ]
    ].copy()
    for k, v in probs.items():
        keep[f"p_{k}"] = v
    keep.to_parquet(ART / "unit4_games.parquet", index=False)
    out = {
        "family": FAMILY,
        "served_source": source,
        "served_column": SERVED,
        "seed": seed(),
        "build": info,
        "full": full,
        "stamped_only": stamped,
        "forward_2026": fwd,
    }
    (ART / "unit4_addone.json").write_text(json.dumps(out, indent=1, default=float))
    print("source", source, "games", full["games"], "served acc", round(full["served_accuracy"], 5))
    print("rows", info["rows_by_season"])
    print(
        "terms", info["term_stats"], "listed per team-game", round(info["listed_per_team_game"], 3)
    )
    print(
        "habit",
        [
            (
                m["season"],
                m["train_rows"],
                round(m["coach_tau2"], 5),
                m["coaches_fitted"],
                m["target_rows"],
                m["target_rows_with_offset"],
            )
            for m in info["habit_fits"]
        ],
    )
    for k, c in full["cells"].items():
        print(line(k, c))
    for t in ("intercept_only", *TERMS):
        print(
            "folds",
            t,
            [
                (f["held_out"], round(f["intercept"], 4), round(f.get("coef_per_sd", 0.0), 4))
                for f in full["folds"][t]
            ],
            "all",
            full["in_sample_coefficients"][t],
        )
    print("signs", full["fold_sign_share"])
    for k, v in full["shuffled_control"].items():
        print("shuffle", k, {a: round(b, 5) for a, b in v.items()})
    print("STAMPED", stamped["seasons"], stamped["games"])
    for k, c in stamped["cells"].items():
        print(line(k, c))
    print(
        "stamped folds",
        [(f["held_out"], round(f["coef_per_sd"], 4)) for f in stamped["folds"][PRIMARY]],
    )
    print("FORWARD", {k: v for k, v in fwd.items() if k not in ("cells", "build")})
    if fwd.get("graded_games"):
        print("forward rows", fwd["build"]["rows_by_season"], fwd["build"]["term_stats"])
        for k, c in fwd["cells"].items():
            print(line(k, c))


if __name__ == "__main__":
    main()
