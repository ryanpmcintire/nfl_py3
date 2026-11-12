import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nfl_ats.evidence_conventions import probability_positive_from_draws

FIELD = Path("data/splash/field")
OUT = Path("artifacts/pool_rank_replay")
SIDE = re.compile(r"([WL])([A-Z]{2,3})(-?\d+(?:\.\d+)?):(\d+):(\d+)(?::a(\d+))?")
ALIAS = {"JAC": "JAX"}
FEATURES = ["fav", "abs_spread", "p_centered"]


def parse_w1_2():
    rows = []
    for w in (1, 2):
        t = pd.read_csv(FIELD / f"2026_week0{w}_field_distribution.tsv", sep="\t", comment="#")
        t = t.replace(ALIAS)
        for r in t.itertuples():
            rows.append(
                {
                    "season": 2026,
                    "week": w,
                    "game_id": f"2026_{w:02d}_{r.away}_{r.home}",
                    "team": r.team,
                    "home": r.team == r.home,
                    "line": float(r.line),
                    "picks": int(r.picks),
                    "best_picks": int(r.best_picks),
                    "autopicks": np.nan,
                    "covered": r.result == "W",
                }
            )
    return rows


def parse_w3_8():
    rows = []
    raw = sorted(FIELD.glob("2026_week03-*_pick_distribution_raw.txt"))[-1]
    for line in raw.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head, body = line.split(" ", 1)
        w = int(head[1:])
        pool = sorted(Path("data/splash").glob(f"2026_week{w:02d}_*.json"))[-1]
        games = json.loads(pool.read_text())["games"]
        lookup = {frozenset((g["away"], g["home"])): g for g in games}
        for blob in body.split("; "):
            m = re.match(r"(\d+)-(\d+) (.*)", blob)
            sides = SIDE.findall(m.group(3))
            assert len(sides) == 2, blob
            teams = frozenset(ALIAS.get(s[1], s[1]) for s in sides)
            g = lookup[teams]
            for res, team, ln, picks, bp, auto in sides:
                rows.append(
                    {
                        "season": 2026,
                        "week": w,
                        "game_id": g["game_id"],
                        "team": ALIAS.get(team, team),
                        "home": ALIAS.get(team, team) == g["home"],
                        "line": float(ln),
                        "picks": int(picks),
                        "best_picks": int(bp),
                        "autopicks": int(auto) if auto else 0,
                        "covered": res == "W",
                    }
                )
    return rows


def attach_model_probability(df):
    p = []
    for w, g in df.groupby("week"):
        d = next(
            x
            for x in sorted(Path("artifacts/margin_predictions").glob(f"2026-week-{w:02d}-*"))
            if (x / "pool_card.csv").exists()
        )
        card = pd.read_csv(d / "pool_card.csv").set_index("game_id")
        for i in g.index:
            c = card.loc[g.at[i, "game_id"]]
            q = float(c["pick_probability"])
            p.append((i, q if g.at[i, "team"] == c["pool_pick"] else 1.0 - q))
    s = pd.Series(dict(p))
    df["model_p"] = s.reindex(df.index)
    return df


def design(df, with_picks):
    x = np.column_stack(
        [
            (df["line"] < 0).astype(float),
            df["line"].abs(),
            df["model_p"] - 0.5,
        ]
    )
    if with_picks:
        x = np.column_stack([x, np.log(df["picks"].clip(lower=1))])
    return x


def nll(beta, x, y, groups):
    u = x @ beta
    out = 0.0
    for g in groups:
        ug = u[g]
        m = ug.max()
        out -= float((y[g] * (ug - m - np.log(np.exp(ug - m).sum()))).sum())
    return out


def fit(df, with_picks):
    x = design(df, with_picks)
    y = df["best_picks"].to_numpy(float)
    groups = [np.flatnonzero((df["week"] == w).to_numpy()) for w in sorted(df["week"].unique())]
    r = minimize(nll, np.zeros(x.shape[1]), args=(x, y, groups), method="BFGS")
    return r.x


def shares(df, beta, with_picks):
    u = design(df, with_picks) @ beta
    e = np.exp(u - u.max())
    return e / e.sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(parse_w1_2() + parse_w3_8())
    df = df.sort_values(["week", "game_id", "home"]).reset_index(drop=True)
    df = attach_model_probability(df)
    tidy = FIELD / f"2026_field_best_picks_w01-{int(df['week'].max()):02d}.parquet"
    df.to_parquet(tidy, index=False)
    print("rows", len(df), "games", df.game_id.nunique(), tidy)

    weeks = sorted(df["week"].unique())
    result = {"rows": len(df), "weeks": [int(w) for w in weeks], "arms": {}}
    for name, wp in (("prelock_features", False), ("with_lock_pick_counts", True)):
        game_pred, game_obs, ll_model, ll_unif, per_week = [], [], 0.0, 0.0, []
        betas = []
        for w in weeks:
            tr, te = df[df.week != w], df[df.week == w].copy()
            beta = fit(tr, wp)
            betas.append(beta.tolist())
            te["q"] = shares(te, beta, wp)
            y = te["best_picks"].to_numpy(float)
            lm = float((y * np.log(te["q"])).sum() / y.sum())
            lu = float((y * np.log(1.0 / len(te))).sum() / y.sum())
            ll_model += lm * y.sum()
            ll_unif += lu * y.sum()
            per_week.append({"week": int(w), "loglik_model": lm, "loglik_uniform": lu})
            gg = te.groupby("game_id").agg(q=("q", "sum"), b=("best_picks", "sum"))
            game_pred += gg["q"].tolist()
            game_obs += (gg["b"] / y.sum()).tolist()
        gp, go = np.array(game_pred), np.array(game_obs)
        slope, intercept = np.polyfit(gp, go, 1)
        bins = np.quantile(gp, np.linspace(0, 1, 6))
        idx = np.clip(np.digitize(gp, bins[1:-1]), 0, 4)
        table = [
            {
                "bin": int(b),
                "n_games": int((idx == b).sum()),
                "predicted_share": float(gp[idx == b].mean()),
                "observed_share": float(go[idx == b].mean()),
            }
            for b in range(5)
        ]
        diffs = np.array([r["loglik_model"] - r["loglik_uniform"] for r in per_week])
        rng = np.random.default_rng(20261103)
        reps = np.array(
            [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(10000)]
        )
        full = fit(df, wp)
        result["arms"][name] = {
            "heldout_loglik_per_best_pick": ll_model / df["best_picks"].sum(),
            "uniform_loglik_per_best_pick": ll_unif / df["best_picks"].sum(),
            "gain_over_uniform_per_week": {
                "mean": float(diffs.mean()),
                "interval_95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
                "probability_positive": float(probability_positive_from_draws(reps)),
            },
            "heldout_game_share_calibration_slope": float(slope),
            "intercept": float(intercept),
            "reliability_quintiles": table,
            "per_week": per_week,
            "fold_coefficients": betas,
            "full_fit": full.tolist(),
            "features": FEATURES + (["log_picks"] if wp else []),
        }
    (OUT / "field_best_pick_model.json").write_text(json.dumps(result, indent=2) + "\n")
    for k, v in result["arms"].items():
        print(
            k,
            v["heldout_loglik_per_best_pick"],
            v["uniform_loglik_per_best_pick"],
            v["gain_over_uniform_per_week"],
            v["heldout_game_share_calibration_slope"],
        )
        print(v["reliability_quintiles"])
        print(np.round(np.array(v["fold_coefficients"]), 3).tolist())


if __name__ == "__main__":
    sys.exit(main())
