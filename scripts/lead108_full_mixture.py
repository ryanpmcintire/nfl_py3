import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import injury_scenario_grade as IG
import lead106_forward_score as F
import lead106_injury_language as L6
import lead108_wording_mixture as U1
import lead110_absorption as A

from nfl_ats.mass_preserving_lattice import band_read

OUT = L6.ROOT / "artifacts" / "lead108_full_mixture"
SEASON = 2026
LAST_WEEK = 10
RES = 100
GRID_STEP = 0.05
SEED = 1082
NULL_REPS = 500
POWER_Z = 2.8
SUBSETS = {"wk2_8": (2, 8), "wk9_10": (9, 10), "wk2_10": (2, 10)}
AGG = {"n": ("played", "size"), "k": ("played", "sum"), "m": ("snap_share", "mean")}


def verified_population():
    pop = L6.id_outcomes(pd.read_parquet(L6.ART / "population.parquet"))
    ver = pop[pop["verified"]].copy()
    ver["practice_status"] = ver["practice_status"].fillna("none")
    return ver


def week10_rows(ver):
    w9 = pd.read_parquet(L6.ART / "forward_score" / f"{SEASON}_week09_scored.parquet")
    w9 = w9[w9["verified"].astype(bool)].copy()
    w9["practice_status"] = w9["practice_status"].fillna("none")
    fw, files, excluded = F.load_forward(LAST_WEEK)
    snaps, snap_file = F.snap_frame(LAST_WEEK)
    fo = F.outcomes(fw, snaps, LAST_WEEK, LAST_WEEK)
    fv = fo[fo["verified"].astype(bool)].copy()
    fv["practice_status"] = fv["practice_status"].fillna("none")
    clip = 1.0 / (len(fv) + 2)
    fv["rp"] = fv["rp_raw"].clip(clip, 1 - clip)
    cols = ["week", "designation", "practice_status", "played", "snap_share"]
    prior = pd.concat([ver[cols], w9[cols]], ignore_index=True)
    prior = prior[prior["week"] < LAST_WEEK]
    tab = {
        LAST_WEEK: (
            prior.groupby(U1.KEYS).agg(**AGG),
            prior.groupby("designation").agg(**AGG),
        )
    }
    fv["lp"] = [L6.ladder_pred(r, tab, U1.KEYS)[0] for _, r in fv.iterrows()]
    out = fv[["team", "week", "gsis_id", "position", "designation", "lp", "rp"]].copy()
    out["grp"] = out["position"].map(A.POS_TO_GROUP)
    meta = {
        "files": files,
        "excluded": excluded,
        "snap_file": str(snap_file),
        "rows_read": len(fo),
        "rows_verified": len(fv),
        "clip": clip,
    }
    return out, meta


def prior_share(p):
    _, pmap = L6.roster_map()
    s, _ = F.snap_frame(LAST_WEEK)
    s["sh"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    scale = 100.0 if s["sh"].max() > 1 else 1.0
    s["sh"] = s["sh"] / scale
    s = s.sort_values("week")
    by = {k: g[["week", "sh"]].to_numpy(float) for k, g in s.groupby("pfr_player_id")}
    out = []
    for gid, w in zip(p["gsis_id"], p["week"], strict=True):
        a = by.get(pmap.get(gid))
        a = a[a[:, 0] < w] if a is not None else None
        out.append(float(a[-1, 1]) if a is not None and len(a) else np.nan)
    return np.array(out)


def dist(ks, qs):
    span = int(np.abs(ks).sum())
    d = np.zeros(2 * span + 1)
    d[span] = 1.0
    for k, q in zip(ks, qs, strict=True):
        d = (1 - q) * d + q * np.roll(d, int(k))
    return d, (np.arange(d.size) - span) / RES


def mixtures(g, p, val, pool_line, pool_margin):
    p = p.copy()
    p["v"] = p["grp"].map(val)
    p = p[p["v"].notna()]
    p["pr"] = p["rp"].fillna(p["lp"])
    by = {k: x for k, x in p.groupby(["team", "week"])}  # noqa: C416
    rows = []
    for r in g.itertuples(index=False):
        ks, ql, qr = [], [], []
        for team, sgn in ((r.home_team, 1), (r.away_team, -1)):
            x = by.get((team, r.week))
            if x is None:
                continue
            ks += list(sgn * np.round(x["v"].to_numpy() * RES).astype(int))
            ql += list(np.clip(1 - x["lp"].to_numpy(), 0, 1))
            qr += list(np.clip(1 - x["pr"].to_numpy(), 0, 1))
        ps = float(r.p_served)
        if not ks:
            rows.append((ps, ps, 0.0, 0.0, 0, 0.0, True))
            continue
        dl, dx = dist(np.array(ks), np.array(ql))
        dr, _ = dist(np.array(ks), np.array(qr))
        el = float((dl * dx).sum())
        er = float((dr * dx).sum())
        center, ok = IG.find_center(pool_line, pool_margin, float(r.open_line), ps)
        if not ok:
            rows.append((ps, ps, el, er, len(ks), np.nan, False))
            continue
        live = (dl > 0) | (dr > 0)
        lo, hi = dx[live].min() - el, dx[live].max() - el
        grid = np.arange(lo, hi + GRID_STEP, GRID_STEP)
        grid = np.union1d(grid, [0.0])
        reads = np.array(
            [
                band_read(
                    pool_line,
                    pool_margin,
                    float(r.open_line),
                    center + d,
                    conditioning_line=float(r.open_line),
                ).home_cover_probability
                for d in grid
            ]
        )
        rd = np.interp(dx - el, grid, reads)
        zero_err = abs(float(reads[grid == 0.0][0]) - ps)
        mix_l = float((dl * rd).sum())
        mix_r = float((dr * rd).sum())
        rows.append((mix_l, mix_r, el, er, len(ks), zero_err, True))
    cols = ["p_mix_lad", "p_mix_read", "e_lad", "e_read", "n_players", "zero_err", "center_ok"]
    return pd.DataFrame(rows, columns=cols, index=g.index)


def acc(p, y):
    return 100.0 * ((p > 0.5) == (y == 1)).astype(float)


def cell(pa, pb, y, wk, rng):
    out = {"games": len(y), "record_a_vs_b": U1.record(pa, pb, y)}
    for mname, d in (
        ("log_loss", U1.ll(pb, y) - U1.ll(pa, y)),
        ("accuracy_points", acc(pa, y) - acc(pb, y)),
    ):
        bs = U1.boot(d, wk, rng)
        out[mname] = {
            "gain": float(d.mean()),
            "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_positive": float((bs > 0).mean()),
            "se": float(bs.std()),
            "detectable_80pct": float(POWER_Z * bs.std()),
        }
    return out


def contrast(pa, pb, y, wk, rng):
    res = {}
    for sname, (lo, hi) in SUBSETS.items():
        m = (wk >= lo) & (wk <= hi)
        res[sname] = cell(pa[m], pb[m], y[m], wk[m], rng)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    val = json.loads((A.OUT / "report.json").read_text())["weights_spread_pts_per_out"]
    p9, lp_check = U1.players()
    ver = verified_population()
    p10, meta10 = week10_rows(ver)
    p = pd.concat([p9, p10], ignore_index=True)
    p["prior_share"] = prior_share(p)
    p["starter"] = p["prior_share"] >= 0.5
    U1.LAST_WEEK = LAST_WEEK
    tt = U1.team_terms(p, val)
    g = U1.games(tt)
    gf = pd.read_parquet(
        L6.ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "spread_line", "result"],
    )
    gf = gf[(gf["season"] < SEASON) & gf["spread_line"].notna() & gf["result"].notna()]
    pool_line = gf["spread_line"].to_numpy(float)
    pool_margin = gf["result"].to_numpy(float)
    mx = mixtures(g, p, val, pool_line, pool_margin)
    g = pd.concat([g, mx], axis=1)
    y = g["y"].to_numpy()
    wk = g["week"].to_numpy()
    ps = g["p_served"].to_numpy()
    off = U1.logit(ps)
    pl = g["p_mix_lad"].to_numpy()
    pr = g["p_mix_read"].to_numpy()
    zl = U1.logit(pl) - off
    zr = U1.logit(pr) - off
    plf, cl = U1.lowo(zl[:, None], y, off, wk)
    prf, cr = U1.lowo(zr[:, None], y, off, wk)
    one = np.ones((len(g), 1))
    zero = np.zeros(len(g))
    p_int, _ = U1.lowo(one, y, zero, wk)
    p_move, _ = U1.lowo(np.column_stack([one, g["move"].to_numpy()]), y, zero, wk)
    bl_in = U1.fit(zl[:, None], y, off)
    br_in = U1.fit(zr[:, None], y, off)

    def pooled_ll(pa, pb):
        return float((U1.ll(pb, y) - U1.ll(pa, y)).mean())

    obs = {
        "read_raw_vs_ladder_raw": pooled_ll(pr, pl),
        "read_fit_vs_served": pooled_ll(prf, ps),
        "ladder_fit_vs_served": pooled_ll(plf, ps),
    }
    nulls = {k: [] for k in obs}
    for _ in range(NULL_REPS):
        zp = zl + U1.perm_x(zr - zl, wk, rng)
        nulls["read_raw_vs_ladder_raw"].append(pooled_ll(U1.sig(off + zp), pl))
        for nm, z in (("read_fit_vs_served", zr), ("ladder_fit_vs_served", zl)):
            pp, _ = U1.lowo(U1.perm_x(z, wk, rng)[:, None], y, off, wk)
            nulls[nm].append(pooled_ll(pp, ps))
    null_pct = {k: float((np.array(v) < obs[k]).mean()) for k, v in nulls.items()}

    starters = (
        p[p["starter"]]
        .groupby(["team", "week"])
        .agg(st=("gsis_id", "size"), st_read=("rp", lambda s: int(s.notna().sum())))
    )
    cov = {}
    for w, x in g.groupby("week"):
        st = [
            starters["st"].get((t, w), 0) + starters["st"].get((a, w), 0)
            for t, a in zip(x["home_team"], x["away_team"], strict=True)
        ]
        sr = [
            starters["st_read"].get((t, w), 0) + starters["st_read"].get((a, w), 0)
            for t, a in zip(x["home_team"], x["away_team"], strict=True)
        ]
        cov[int(w)] = {
            "games": len(x),
            "listed_verified_per_game": float((x["h_listed"] + x["a_listed"]).mean()),
            "with_read_per_game": float((x["h_read"] + x["a_read"]).mean()),
            "listed_starters_per_game": float(np.mean(st)),
            "starters_with_read_per_game": float(np.mean(sr)),
            "games_any_starter_read": int((np.array(sr) > 0).sum()),
            "value_mass_share_read": float(
                (x["h_mass_read"] + x["a_mass_read"]).sum() / (x["h_mass"] + x["a_mass"]).sum()
            ),
        }
    rep = {
        "family": "lead108_wording_mixture",
        "unit": "full scenario mixture, PER-10 discrete lattice re-read, ladder vs read weights",
        "looks": 28,
        "look_family": (
            "4 contrasts (read raw vs ladder raw, read fitted vs ladder fitted, read raw vs "
            "served, ladder raw vs served) x 2 metrics (log loss, accuracy) x 3 subsets (wk2_8, "
            "wk9_10, wk2_10) + 2 controls (intercept, line move) x 2 metrics pooled"
        ),
        "centering": (
            "scenario shift minus the ladder expected shift for both arms; served assumed to "
            "price the ladder expectation (PER-10 centred on the served value-lost expectation)"
        ),
        "lattice": (
            "band_read conditioned on the Tuesday opener band, centre bisected so a zero shift "
            "reproduces the served probability, scenario shift moves the tilt centre; pool = "
            "game_features seasons before 2026"
        ),
        "joint": (
            "independent sits by exact convolution on a 0.01 point grid "
            "(PER-10 pairwise coupling not used)"
        ),
        "starter_rule": (
            "max(offense_pct, defense_pct) in the player's latest 2026 game before the week "
            ">= 0.5 (majority of the unit's snaps, definitional)"
        ),
        "week10": meta10,
        "ladder_week9_check": lp_check,
        "games": len(g),
        "games_by_week": {int(k): int(v) for k, v in g.groupby("week").size().items()},
        "center_failures": int((~g["center_ok"]).sum()),
        "max_zero_shift_error": float(g["zero_err"].max()),
        "max_expectation_vs_unit1_term": float(np.abs((g["e_lad"] - g["x_lad"]).to_numpy()).max()),
        "coverage": cov,
        "shift_sd": {
            "ladder_raw": float((pl - ps).std()),
            "read_raw": float((pr - ps).std()),
            "read_minus_ladder": float((pr - pl).std()),
        },
        "side_changes_vs_served": {
            "ladder_raw": int(((pl > 0.5) != (ps > 0.5)).sum()),
            "read_raw": int(((pr > 0.5) != (ps > 0.5)).sum()),
            "read_fit": int(((prf > 0.5) != (ps > 0.5)).sum()),
            "ladder_fit": int(((plf > 0.5) != (ps > 0.5)).sum()),
        },
        "coef_per_fold": {"ladder": cl, "read": cr},
        "insample_vs_oos": {
            "ladder_coef": float(bl_in[0]),
            "read_coef": float(br_in[0]),
            "ladder_ll_gain_in": pooled_ll(U1.sig(off + zl * bl_in[0]), ps),
            "read_ll_gain_in": pooled_ll(U1.sig(off + zr * br_in[0]), ps),
            "ladder_ll_gain_oos": pooled_ll(plf, ps),
            "read_ll_gain_oos": pooled_ll(prf, ps),
        },
        "mean_log_loss": {
            k: float(U1.ll(v, y).mean())
            for k, v in (
                ("served", ps),
                ("ladder_raw", pl),
                ("read_raw", pr),
                ("ladder_fit", plf),
                ("read_fit", prf),
                ("intercept", p_int),
                ("move", p_move),
            )
        },
        "accuracy": {
            k: float(acc(v, y).mean())
            for k, v in (("served", ps), ("ladder_raw", pl), ("read_raw", pr), ("read_fit", prf))
        },
        "matched_null_percentile_pooled_ll": null_pct,
        "reliability": {
            "served": U1.reliability(ps, y, 5),
            "read_raw": U1.reliability(pr, y, 5),
        },
        "read_raw_vs_ladder_raw": contrast(pr, pl, y, wk, rng),
        "read_fit_vs_ladder_fit": contrast(prf, plf, y, wk, rng),
        "read_raw_vs_served": contrast(pr, ps, y, wk, rng),
        "ladder_raw_vs_served": contrast(pl, ps, y, wk, rng),
        "controls_vs_served": {
            "intercept": cell(p_int, ps, y, wk, rng),
            "move": cell(p_move, ps, y, wk, rng),
        },
    }
    g.to_parquet(OUT / "games.parquet")
    p.to_parquet(OUT / "players.parquet")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
