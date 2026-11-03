import argparse
import hashlib
import json
import time
from math import erf, sqrt

import numpy as np
import pandas as pd
import sky03_matchup as sm
import sky03_matchups as m
from sky04_lead_lag import fit_b, logit

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["control", "freeze"], required=True)
ap.add_argument("--multipliers", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
ap.add_argument("--reps", type=int, default=200)
args = ap.parse_args()
seasons = m.EVAL_SEASONS
t = m.load_games().merge(sm.load_script(), on=["game_id", "off"], how="left")
t["sc"] = t.sc.fillna(0.5)
assert t.season.max() <= max(seasons)
rr = sm.build_rows(t, tuple(range(m.TRAIT_START + 1, max(seasons) + 1)))

if args.stage == "freeze":
    term_path = sm.OUTD / "term.json"
    term = json.loads(term_path.read_text())
    body = {
        "term": "sky03_separated_pair_residual",
        "c_A_epa_per_unit_product": float(m.fit_c(rr, "zA", "rA", "n_db")),
        "c_B_epa_per_unit_product": float(m.fit_c(rr, "zB", "rB", "n_ns")),
        "gamma_epa": float(rr.gam_epa.iloc[-1]),
        "gamma_epa_ns": float(rr.gam_epa_ns.iloc[-1]),
        "b_logit_per_point": term["pooled_coefficient"],
        "c_fit_rows": len(rr),
        "c_fit_seasons": [int(rr.season.min()), int(rr.season.max())],
        "b_fit_seasons": list(map(int, seasons)),
        "b_source_file": "artifacts/sky03/separated/term.json",
        "b_source_sha256": hashlib.sha256(term_path.read_bytes()).hexdigest(),
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = hashlib.sha256(
        json.dumps(
            {
                k: body[k]
                for k in (
                    "c_A_epa_per_unit_product",
                    "c_B_epa_per_unit_product",
                    "b_logit_per_point",
                )
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    body["created_at_utc"] = pd.Timestamp.now("UTC").isoformat()
    d = m.ROOT / "artifacts" / "forward_terms" / "sky03_separated"
    d.mkdir(parents=True, exist_ok=True)
    (d / "frozen_coefficients.json").write_text(json.dumps(body, indent=1))
    print(json.dumps(body, indent=1))
    raise SystemExit(0)

g = pd.read_parquet(m.PERGAME)
g = g[(g.game_type == "REG") & g.season.isin(seasons)]
sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
gx = m.game_x(rr, seasons)
hp = gx.merge(sub[["game_id", "home_team"]], on="game_id")
hp["sp"] = np.where(hp.off == hp.home_team, 1.0, -1.0) * hp.pts
sub["x"] = sub.game_id.map(hp.groupby("game_id").sp.sum())
sub = sub[sub.x.notna()].reset_index(drop=True)
pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
off = logit(pr)
xs = sub.x.values
se = sub.season.values
y0 = sub.home_covered.astype(float).values
bobs = float(fit_b(off, xs, y0))


def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def oos(y):
    pooled = np.zeros(len(y))
    for s in seasons:
        te = se == s
        b = fit_b(off[~te], xs[~te], y[~te])
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * xs[te])))
    d = (pr - y) ** 2 - (pooled - y) ** 2
    return d.mean(), d.mean() / (d.std(ddof=1) / sqrt(len(d)))


rng = np.random.default_rng(20261103)
out = {
    "script": "scripts/sky03_separated_control.py",
    "n": len(sub),
    "pooled_obs": bobs,
    "x_sd": float(xs.std()),
}
for mult in [0.0, *args.multipliers]:
    b0 = mult * bobs
    gains, pp = [], []
    t0 = time.time()
    for _ in range(args.reps):
        p = 1 / (1 + np.exp(-(off + b0 * xs)))
        y = (rng.random(len(p)) < p).astype(float)
        gm, z = oos(y)
        gains.append(gm)
        pp.append(phi(z))
    pp = np.array(pp)
    out[f"mult_{mult}"] = {
        "b0": b0,
        "mean_gain": float(np.mean(gains)),
        "frac_Ppos_ge_0.95": float((pp >= 0.95).mean()),
        "mean_Ppos": float(pp.mean()),
        "seconds": time.time() - t0,
    }
    print(mult, out[f"mult_{mult}"], flush=True)
(sm.OUTD / "positive_control.json").write_text(json.dumps(out, indent=1))
