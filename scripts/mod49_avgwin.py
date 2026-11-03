import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_unit2_components as u2  # noqa: E402
import mod49_construction as c49  # noqa: E402
from per07_first_game_fitted_term import fit, metrics  # noqa: E402

WINDOWS = (2, 3, 4, 5)
MIN_PRIOR = max(WINDOWS) + 1
AVG = "s_avg"
FAMILY = c49.FAMILY
OUT = REPO / "artifacts" / "mod49" / "avgwin.json"
FROZEN = REPO / "artifacts" / "forward_terms" / "mod41_avgwin_served" / "frozen_coefficients.json"
RAW = {w: f"r_w{w}" for w in WINDOWS}
ORIG_LOSO = u2.loso


def average(d, sds):
    z = sum(d[RAW[w]] / sds[w] for w in WINDOWS) / len(WINDOWS)
    return z.where(d["usable"], 0.0)


def train_sds(d, mask):
    u = mask & d["usable"]
    return {w: float(d.loc[u, RAW[w]].std(ddof=1)) for w in WINDOWS}


def loso_avg(d, cols):
    if AVG not in cols:
        return ORIG_LOSO(d, cols)
    p = pd.Series(np.nan, index=d.index)
    co = {}
    for season in sorted(d["season"].unique()):
        trm = d["season"].ne(season)
        dd = d.copy()
        dd[AVG] = average(dd, train_sds(dd, trm))
        mod = fit(dd.loc[trm], cols)
        p[~trm] = mod.predict_proba(dd.loc[~trm, cols])[:, 1]
        co[int(season)] = [float(v) for v in mod.coef_[0]]
    return p, co


def perm_null_joint(d, base):
    bp, _ = ORIG_LOSO(d, base)
    rng = np.random.default_rng(u2.PERM_SEED)
    u = d.index[d["usable"]]
    blocks = [g.index.to_numpy() for _, g in d.loc[u].groupby("block")]
    gains = []
    obs = None
    for i in range(u2.NULL_DRAWS + 1):
        dd = d.copy()
        if i:
            for idx in blocks:
                perm = rng.permutation(idx)
                for w in WINDOWS:
                    v = dd[RAW[w]].to_numpy().copy()
                    v[idx] = v[perm]
                    dd[RAW[w]] = v
        tp, _ = loso_avg(dd, [*base, AVG])
        f = dd.assign(p_base=bp, p_term=tp)
        g = metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"]
        if i:
            gains.append(g)
        else:
            obs = g
    gains = np.array(gains)
    return {
        "observed_ll_gain": float(obs),
        "null_mean": float(gains.mean()),
        "null_p_ge": float((1 + (gains >= obs).sum()) / (u2.NULL_DRAWS + 1)),
    }


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def main():
    base = c49.prepare()
    parts = base["game_id"].str.split("_", expand=True)
    base["away_team"], base["home_team"] = parts[2], parts[3]
    pk = pd.read_parquet(c49.PICK, columns=["game_id", "out_of_season_home_probability"])
    base = base.merge(pk, on="game_id", how="left")
    ps = base["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6)
    base["logit_s"] = np.log(ps / (1 - ps))
    t = c49.team_table()
    t["prior_n"] = np.where(t["prior_n"] >= MIN_PRIOR, t["prior_n"], 0)
    d = None
    betas = {}
    for w in WINDOWS:
        spec = {"window": w, "load": "total", "norm": "season", "unit": "plays"}
        dw, beta = c49.build_variant(base, t, spec)
        betas[w] = beta
        if d is None:
            d = dw.drop(columns=["s_resid"])
        d[RAW[w]] = dw["s_resid"].to_numpy()
        assert d["usable"].equals(dw["usable"])
    sds_full = train_sds(d, pd.Series(True, index=d.index))
    d[AVG] = average(d, sds_full)
    u2.loso = loso_avg
    S = ["logit_s"]
    r = u2.run(d, S, [AVG], "avgwin_served")
    r["null"] = perm_null_joint(d, S)
    r["usable"] = int(d["usable"].sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"family": FAMILY, "grade": "opener", "result": r}, indent=2, default=str)
    )
    print(
        "usable",
        r["usable"],
        "dec",
        r["decisive"],
        r["decisive_term_correct"],
        r["decisive_base_correct"],
    )
    print("ll", r["ll_gain"], "\nacc", r["acc_gain"], "\ngap", r["gap_ll"])
    print("coef_full", r["coef_full"])
    for k, v in r["coef_folds"].items():
        print(k, [round(x, 4) for x in v])
    print("null", r["null"])
    mod = fit(d, [*S, AVG])
    folds = {int(s): v[1] for s, v in r["coef_folds"].items()}
    served_folds = {int(s): v[0] for s, v in r["coef_folds"].items()}
    body = {
        "term": "mod49_avgwin_recent_workload_served",
        "term_definition": "home minus away of the equal-weight average over windows 2, 3, 4 and 5 of the standardized residual recent workload: per window w, [last-w-game mean of plays run plus plays defended minus the team-season mean of the earlier games] minus its linear projection on the home-minus-away season mean of plays run and the home-minus-away last-w-game mean of plays defended, divided by that window's frozen SD; zero when either team has fewer than 6 prior games; added to the logit of the served home probability",  # noqa: E501
        "windows": list(WINDOWS),
        "min_prior_games": MIN_PRIOR,
        "decomposition_by_window": {str(w): betas[w] for w in WINDOWS},
        "sd_by_window": {str(w): sds_full[w] for w in WINDOWS},
        "b_logit_per_unit": float(mod.coef_[0][1]),
        "served_logit_coefficient": float(mod.coef_[0][0]),
        "intercept": float(mod.intercept_[0]),
        "loso_avg_coefficient_by_heldout_season": folds,
        "loso_avg_coefficient_min": float(min(folds.values())),
        "loso_avg_coefficient_max": float(max(folds.values())),
        "loso_served_logit_coefficient_by_heldout_season": served_folds,
        "fit_seasons": [int(d.season.min()), int(d.season.max())],
        "fit_games": len(d),
        "fit_usable_games": r["usable"],
        "fit_base": f"logit of {c49.PICK.parent.name} out_of_season_home_probability; opener grade; unusable rows carry zero",  # noqa: E501
        "base_artifact": str(c49.PICK.relative_to(REPO)),
        "base_artifact_sha256": hashlib.sha256(c49.PICK.read_bytes()).hexdigest(),
        "fit_script": "scripts/mod49_avgwin.py",
        "score_result": "artifacts/mod49/avgwin.json",
        "target_season": 2026,
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj(
        {
            k: body[k]
            for k in ("b_logit_per_unit", "intercept", "decomposition_by_window", "sd_by_window")
        }
    )
    body["created_at_utc"] = datetime.now(UTC).isoformat()
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(body, indent=1))
    print("frozen", FROZEN)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
