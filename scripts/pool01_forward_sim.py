import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.evidence_conventions import probability_positive_from_draws

K = 1000.0
OUT = Path("artifacts/pool_rank_replay")
FIELD = Path("data/splash/field")
FIELD_FILE = sorted(FIELD.glob("2026_field_best_picks_w01-*.parquet"))[-1]
WEEKLY_FILE = sorted(FIELD.glob("2026_weekly_points_w01-*.parquet"))[-1]
WEEKLY_JSON = OUT / "weekly_points_field.json"


def load_games():
    df = pd.read_parquet(FIELD_FILE)
    model = json.loads((OUT / "field_best_pick_model.json").read_text())
    beta = np.array(model["arms"]["prelock_features"]["full_fit"])
    x = np.column_stack([(df["line"] < 0).astype(float), df["line"].abs(), df["model_p"] - 0.5])
    df["u"] = x @ beta
    rows = []
    for gid, g in df.groupby("game_id"):
        o = g.loc[g["model_p"].idxmax()]
        t = g.drop(o.name).iloc[0]
        rows.append(
            {
                "game_id": gid,
                "week": int(o["week"]),
                "p": float(o["model_p"]),
                "share": float(o["picks"] / (o["picks"] + t["picks"])),
                "u_o": float(o["u"]),
                "u_t": float(t["u"]),
                "bp_o": int(o["best_picks"]),
                "bp_t": int(t["best_picks"]),
                "cov_o": bool(o["covered"]),
            }
        )
    return pd.DataFrame(rows)


def draw_week(games, rng, S, E, idx=None, delta=None):
    if idx is None:
        idx = rng.integers(0, len(games["p"]), size=(S, 15))
    p, sh, uo, ut = (games[k][idx] for k in ("p", "share", "u_o", "u_t"))
    cover = rng.random(p.shape) < p
    thr = np.where(cover, sh, 1.0 - sh)
    tp = thr[:, None, :]
    if delta is not None:
        tp = np.clip(tp + delta[None, :, None], 0.0, 1.0)
    pts = (rng.random((S, E, 15), dtype=np.float32) < tp).sum(axis=2).astype(float)
    eo, et = np.exp(uo), np.exp(ut)
    z = (eo + et).sum(axis=1, keepdims=True)
    rate = (np.where(cover, eo, et) / z).sum(axis=1)
    bp = (rng.random((S, E)) < rate[:, None]).astype(float)
    served = np.argmax(p, axis=1)
    return {
        "cover": cover,
        "pts": pts,
        "bp": bp,
        "served": served,
        "served_cov": cover[np.arange(S), served],
    }


def share_given_key(field_key, ours_key):
    top = field_key.max(axis=1)
    level = (field_key == ours_key[:, None]).sum(axis=1)
    return np.where(ours_key >= top, 1.0 / (1.0 + level), 0.0)


def rival_table(our_points):
    w = pd.read_parquet(WEEKLY_FILE)
    meta = json.loads(WEEKLY_JSON.read_text())
    tot = w["total"].to_numpy(float)
    mine = int(np.where(tot == our_points)[0][0])
    tot = np.delete(tot, mine)
    n_games = float(sum(meta["games_per_week"]))
    rate = tot / n_games
    r = max(float(meta["split_half_spearman_brown"]), 0.0)
    var_obs = float(rate.var(ddof=1))
    dmean = r * (rate - rate.mean())
    dsd = float(np.sqrt(r * (1.0 - r) * var_obs))
    return tot, dmean, dsd


def past_state(df, rng, E, S, our_points, our_rank, table):
    tot, dmean, dsd = table
    unknown = E - len(tot)
    our_bp = 0.0
    bp = np.zeros((S, E))
    for _, g in df.groupby("week"):
        cov = g["cov_o"].to_numpy()
        won = np.where(cov, g["bp_o"], g["bp_t"]).sum()
        rate = won / float((g["bp_o"] + g["bp_t"]).sum())
        bp += rng.random((S, E)) < rate
        our_bp += float(cov[int(np.argmax(g["p"].to_numpy()))])
    src = np.concatenate(
        [np.tile(np.arange(len(tot)), (S, 1)), rng.integers(0, len(tot), (S, unknown))], axis=1
    )
    pts = tot[src]
    key = pts * K + bp
    ours = our_points * K + our_bp
    ok = (key > ours).sum(axis=1) == (our_rank - 1)
    return pts[ok], bp[ok], dmean[src][ok], dsd, our_bp


def run_path(games, pts0, bp0, dm, dsd, our_bp0, our_points, rng, weeks, E, inner):
    delta = dm + dsd * rng.standard_normal(E)
    draws = [draw_week(games, rng, 1, E, delta=delta) for _ in range(weeks)]
    idxs = None
    out = {}
    for policy in ("served", "aware"):
        rp, rb = pts0.copy(), bp0.copy()
        op, ob = float(our_points), float(our_bp0)
        for k in range(weeks):
            d = draws[k]
            choice = int(d["served"][0])
            if policy == "aware":
                inn = [
                    {
                        "cover": np.repeat(d["cover"], inner, axis=0),
                        "pts": np.repeat(d["pts"], inner, axis=0),
                        "bp": np.repeat(d["bp"], inner, axis=0),
                    }
                ]
                inn[0]["cover"] = np.repeat(d["cover"], inner, axis=0)
                for _ in range(k + 1, weeks):
                    inn.append(draw_week(games, rng, inner, E, delta=delta))
                fkey = (
                    (rp[None, :] + sum(i["pts"] for i in inn)) * K
                    + rb[None, :]
                    + sum(i["bp"] for i in inn)
                )
                opts = op + sum(i["cover"].sum(axis=1) for i in inn)
                later = sum(i["served_cov"].astype(float) for i in inn[1:]) if len(inn) > 1 else 0.0
                base = opts * K + ob + later
                p1 = share_given_key(fkey, base + 1.0)
                p0 = share_given_key(fkey, base)
                value = np.where(inn[0]["cover"], p1[:, None], p0[:, None]).mean(axis=0)
                if float(value.max()) > value[choice]:
                    choice = int(np.argmax(value))
            op += float(d["cover"][0].sum())
            ob += float(d["cover"][0][choice])
            rp = rp + d["pts"][0]
            rb = rb + d["bp"][0]
        out[policy] = (op, ob, rp, rb)
    res = {}
    for policy, (op, ob, rp, rb) in out.items():
        fk = rp * K + rb
        res[policy] = float(share_given_key(fk[None, :], np.array([op * K + ob]))[0])
        if policy == "served":
            res["served_points_only"] = float(share_given_key(rp[None, :], np.array([op]))[0])
    return res, idxs


def summarize(values, rng):
    v = np.asarray(values, float)
    reps = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)])
    return {
        "mean": float(v.mean()),
        "interval_95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
        "probability_positive": float(probability_positive_from_draws(reps)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Forward Best Pick simulation from the current standing"
    )
    ap.add_argument("--paths", type=int, default=240)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--weeks", type=int, required=True, help="weeks still to play")
    ap.add_argument("--points", type=int, required=True, help="our season points on the page")
    ap.add_argument("--rank", type=int, required=True, help="our rank on the page")
    ap.add_argument("--entries", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    df = load_games()
    games = {k: df[k].to_numpy() for k in ("p", "share", "u_o", "u_t")}
    rng = np.random.default_rng(a.seed)
    E = a.entries - 1
    table = rival_table(a.points)
    kept_p, kept_b, kept_d, drawn, have = [], [], [], 0, 0
    while have < a.paths:
        pp, bb, dd, dsd, our_bp = past_state(df, rng, E, 2000, a.points, a.rank, table)
        kept_p.append(pp)
        kept_b.append(bb)
        kept_d.append(dd)
        have += len(pp)
        drawn += 2000
    pts0 = np.concatenate(kept_p)[: a.paths]
    bp0 = np.concatenate(kept_b)[: a.paths]
    dm0 = np.concatenate(kept_d)[: a.paths]
    rows = []
    for i in range(a.paths):
        r, _ = run_path(
            games, pts0[i], bp0[i], dm0[i], dsd, our_bp, a.points, rng, a.weeks, E, a.inner
        )
        rows.append(r)
        if a.smoke and i >= 1:
            break
    res = pd.DataFrame(rows)
    res["gain"] = res["aware"] - res["served"]
    res["bp_value"] = res["served"] - res["served_points_only"]
    brng = np.random.default_rng(a.seed + 1)
    measured_reliability = float(json.loads(WEEKLY_JSON.read_text())["split_half_spearman_brown"])
    out = {
        "start": {
            "points": a.points,
            "rank": a.rank,
            "entries": a.entries,
            "weeks_remaining": a.weeks,
            "games_per_week": 15,
            "our_best_pick_wins_so_far": our_bp,
            "games_played": int(len(df)),
            "our_stated_cover_chance": float(df["p"].mean()),
            "our_realised_cover_rate": float(df["cov_o"].mean()),
        },
        "rival_skill_reliability": {
            "measured": measured_reliability,
            "used": max(measured_reliability, 0.0),
        },
        "past_acceptance": {"accepted_pool": have, "drawn": drawn},
        "paths": len(res),
        "inner_samples": a.inner,
        "p_first_served": summarize(res["served"], brng),
        "p_first_aware": summarize(res["aware"], brng),
        "gain_aware_minus_served": summarize(res["gain"], brng),
        "p_first_served_points_only": summarize(res["served_points_only"], brng),
        "best_pick_tiebreak_value_served": summarize(res["bp_value"], brng),
        "paths_where_policies_differ": int((res["gain"] != 0).sum()),
    }
    if not a.smoke:
        (OUT / "forward_sim_real_rivals_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
