import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import per13_locked_seed_rest as u1  # noqa: E402
import sim09_stakes as ss  # noqa: E402

OUT = ROOT / "artifacts" / "per13_locked_seed_rest"
FWD = ROOT / "artifacts" / "forward_terms" / "per13_locked_rest"
WEEKS = (16, 17, 18)
DRAWS = 2000
SEED = 20261103
OUTER = 600
INNER = 300


def rows():
    u1.WEEKS = WEEKS
    t = u1.team_rows()
    t["x_lock"] = np.where(t.seed_modal > 0, t.p_seed_modal, 0.0)
    t["x_elim"] = 1.0 - t.p_make
    opp = t[["game_id", "team", "x_lock", "x_elim"]].rename(
        columns={"team": "opp", "x_lock": "o_lock", "x_elim": "o_elim"}
    )
    t["opp"] = np.where(t.side > 0, t.away_team, t.home_team)
    t = t.merge(opp, on=["game_id", "opp"], how="left")
    t["d_lock"] = t.x_lock - t.o_lock
    t["d_elim"] = t.x_elim - t.o_elim
    return t.dropna(subset=["open", "close", "d_lock", "d_elim"]).reset_index(drop=True)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def loso(t, cols, target):
    pred = np.zeros(len(t))
    for s in sorted(t.season.unique()):
        m = (t.season == s).to_numpy()
        b = ols(t.loc[~m, cols].to_numpy(), t.loc[~m, target].to_numpy())
        pred[m] = t.loc[m, cols].to_numpy() @ b
    return pred


def boot_coef(t, cols, target):
    codes, ids = pd.factorize(t.game_id)
    n = len(ids)
    X = t[cols].to_numpy()
    y = t[target].to_numpy()
    k = len(cols)
    xx = np.zeros((n, k, k))
    xy = np.zeros((n, k))
    for i in range(k):
        xy[:, i] = np.bincount(codes, weights=X[:, i] * y, minlength=n)
        for j in range(k):
            xx[:, i, j] = np.bincount(codes, weights=X[:, i] * X[:, j], minlength=n)
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=DRAWS)
    return np.array([np.linalg.solve(np.tensordot(wi, xx, 1), wi @ xy) for wi in w])


def summarize(draws, est):
    return {
        "est": float(est),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "p_neg": float((draws < 0).mean()),
        "p_pos": float((draws > 0).mean()),
    }


def lock_probabilities():
    s = pd.read_parquet(ss.SCHED)
    s = s[(s.season >= 2009) & (s.season <= 2026) & (s.game_type == "REG")].copy()
    s["home_team"] = ss.norm_team(s.home_team)
    s["away_team"] = ss.norm_team(s.away_team)
    s = s.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    s["hi"] = s.home_team.map(ss.TI)
    s["ai"] = s.away_team.map(ss.TI)
    s["margin"] = s.home_score - s.away_score
    res = pd.read_parquet(ROOT / "data" / "processed" / "game_features.parquet")
    res = res[["game_id", "result"]].rename(columns={"result": "played_margin"})
    s = s.merge(res, on="game_id", how="left")
    s["margin"] = s.margin.fillna(s.played_margin)
    s["neutral"] = (s.location == "Neutral").astype(float)
    tabs, last = ss.strength_tables(s)
    P = json.loads((ROOT / "artifacts" / "sim09" / "stakes_summary.json").read_text())[
        "rating_params"
    ]
    g = s[s.season == 2026].reset_index(drop=True)
    w0 = int(g[g.margin.isna()].week.min())
    r = ss.rating(tabs, last, 2026, w0, P["k"], P["c"])
    played = g[g.margin.notna()]
    W0 = np.zeros(32)
    for h, a, m in zip(played.hi, played.ai, played.margin, strict=True):
        W0[h] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
        W0[a] += 1.0 if m < 0 else (0.5 if m == 0 else 0.0)
    rng = np.random.default_rng(SEED)
    rem = g[g.margin.isna()]

    def play(df, n):
        e = P["hfa"] * (1 - df.neutral.values) + P["slope"] * (r[df.hi.values] - r[df.ai.values])
        x = rng.random((n, len(df))) < norm.cdf(e / P["sigma"])
        H = np.zeros((len(df), 32))
        A = np.zeros((len(df), 32))
        H[np.arange(len(df)), df.hi.values] = 1
        A[np.arange(len(df)), df.ai.values] = 1
        return x @ H + (~x) @ A

    ss.RNG = rng
    res = {}
    for wk in (17, 18):
        pre = rem[rem.week < wk]
        post = rem[rem.week >= wk]
        wins_outer = W0 + play(pre, OUTER)
        locked = np.zeros(32)
        elim = np.zeros(32)
        for i in range(OUTER):
            wins = np.tile(wins_outer[i], (INNER, 1)) + play(post, INNER)
            seed = ss.seeds_from_wins(wins, 7)
            pk = np.stack([(seed == q).mean(axis=0) for q in range(8)], axis=1)
            locked += (pk.argmax(axis=1) > 0) & (pk.max(axis=1) >= 0.99)
            elim += (1.0 - pk[:, 1:].sum(axis=1)) >= 0.99
        res[wk] = (locked / OUTER, elim / OUTER)
    df = pd.DataFrame(
        {
            "team": ss.TEAMS,
            "wins_now": W0,
            "p_locked_at_week17_tuesday": res[17][0],
            "p_locked_at_week18_tuesday": res[18][0],
            "p_elim_at_week17_tuesday": res[17][1],
            "p_elim_at_week18_tuesday": res[18][1],
        }
    )
    return df.sort_values("p_locked_at_week17_tuesday", ascending=False), w0


def main():
    t = rows().sort_values(["season", "week"]).reset_index(drop=True)
    cols = ["d_lock", "d_elim"]
    y = t.move_toward.to_numpy()
    full = ols(t[cols].to_numpy(), y)
    draws = boot_coef(t, cols, "move_toward")
    t["pred"] = loso(t, cols, "move_toward")
    sse0 = float((y**2).sum())
    cells = {
        "n_rows": len(t),
        "n_games": int(t.game_id.nunique()),
        "coef_lock": summarize(draws[:, 0], full[0]),
        "coef_elim": summarize(draws[:, 1], full[1]),
        "loso_oos_r2_vs_zero": 1.0 - float(((y - t.pred) ** 2).sum()) / sse0,
        "in_sample_r2_vs_zero": 1.0 - float(((y - t[cols].to_numpy() @ full) ** 2).sum()) / sse0,
    }
    per_season = {}
    for s in sorted(t.season.unique()):
        m = (t.season == s).to_numpy()
        per_season[int(s)] = float(ols(t.loc[m, cols].to_numpy(), y[m])[0])
    cells["season_lock_coef_negative_share"] = float(np.mean([v < 0 for v in per_season.values()]))
    dec = t[(t.locked == 1) & t.week.isin((17, 18))]
    cells["decisive"] = {
        "n_locked_rows": len(dec),
        "n_games": int(dec.game_id.nunique()),
        "pred_move_mean_loso": float(dec.pred.mean()),
        "realized_move_mean": float(dec.move_toward.mean()),
        "realized_ats_open_mean": float(dec.ats_open.mean()),
        "realized_ats_close_mean": float(dec.ats_close.mean()),
    }
    codes, ids = pd.factorize(dec.game_id)
    n = len(ids)
    w = np.random.default_rng(SEED).multinomial(n, np.full(n, 1.0 / n), size=DRAWS)
    a = np.bincount(codes, weights=dec.ats_open.to_numpy(), minlength=n)
    p = np.bincount(codes, weights=dec.pred.to_numpy(), minlength=n)
    c = np.bincount(codes, minlength=n).astype(float)
    cells["decisive"]["ats_open_minus_pred"] = summarize(
        (w @ a) / (w @ c) - (w @ p) / (w @ c), dec.ats_open.mean() - dec.pred.mean()
    )
    cells["decisive"]["ats_open_mean"] = summarize((w @ a) / (w @ c), dec.ats_open.mean())
    sl_draws = boot_coef(t, ["pred"], "ats_open")
    sl = ols(t[["pred"]].to_numpy(), t.ats_open.to_numpy())
    cells["cover_slope_on_pred_all_w16_18"] = summarize(sl_draws[:, 0], sl[0])
    cells["looks"] = {"coefficient_cells": 2, "decisive_cells": 2, "slope_cells": 1}
    cells["per_season_lock_coef"] = per_season
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "term_design_cells.json").write_text(json.dumps(cells, indent=1))
    FWD.mkdir(parents=True, exist_ok=True)
    (FWD / "frozen_coefficients.json").write_text(
        json.dumps(
            {
                "term": "per13_locked_rest_expected_point_cost",
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "target": "open_to_close_move_toward_team_points",
                "features": {
                    "d_lock": "own minus opponent p_seed_modal if modal seed is a playoff seed",
                    "d_elim": "own minus opponent (1 - p_make)",
                },
                "coef": {"d_lock": float(full[0]), "d_elim": float(full[1])},
                "fit_seasons": [int(t.season.min()), int(t.season.max())],
                "weeks": list(WEEKS),
                "n_rows": len(t),
                "status": "forward-logged challenger only; not served",
                "source_script": "scripts/per13_term_design.py",
            },
            indent=1,
        )
    )
    steps = [
        "Tuesday of Weeks 17 and 18, before the pool lock:",
        "1. Run the as-of standings simulation (sim09_stakes team_weeks logic) with prior",
        "   games played; take p_make, modal seed and its probability per team.",
        "2. x_lock = modal-seed probability if the modal seed is a playoff seed else 0;",
        "   x_elim = 1 - p_make; d = own minus opponent.",
        "3. term = coef_lock*d_lock + coef_elim*d_elim (frozen_coefficients.json): the",
        "   expected adverse opener-to-close move in points, unpriced at the opener.",
        "4. Forward-log challenger: shift the opener margin by the term, log the shifted",
        "   pick and cover probability beside the served ones, no effect on the served side.",
        "5. Serving needs a pinned pick_probability.py edit adding the term as a fitted",
        "   coefficient of the one calibrated probability, plus re-enrolled validation:",
        "   LOSO refit, reliability table, log loss against market-only and model-only",
        "   baselines, 2026 forward games as held-out.",
    ]
    (FWD / "procedure.md").write_text(chr(10).join(steps) + chr(10))
    lp, w0 = lock_probabilities()
    lp.to_csv(OUT / "lock_probabilities_2026.csv", index=False)
    print(json.dumps(cells, indent=1))
    print("next unplayed week", w0)
    print(lp.head(14).round(3).to_string())
    top = lp.sort_values("p_elim_at_week17_tuesday", ascending=False)
    print(top.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
