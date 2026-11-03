import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import per07_first_game_fitted_term as _per07  # noqa: E402
import pooled_screen as ps  # noqa: E402
import pooled_sweep_variants as sv  # noqa: E402
from per07_first_game_fitted_term import (  # noqa: E402
    block_bootstrap,
    fit,
    metrics,
    summarize,
)
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

from nfl_ats.cfb_features import cfb_season_partitions  # noqa: E402

FEATURES = REPO / "data" / "processed" / "cfb_game_features.parquet"
OUT = REPO / "artifacts" / "xlg14" / "mod41_cfb.json"
FAMILY = "xlg14_mod41_cfb_u1"
DRAWS = 1000
WINDOW = 3
SEASONS = [*range(2012, 2020), *range(2021, 2026)]
CLOSE_COLUMNS = {"spread_line", "spread_dispersion", "total_line"}
_per07.BOOTSTRAP_SAMPLES = DRAWS
TERMS = [
    ("total", ["logit_p", "s_load_total"]),
    ("resid", ["logit_p", "s_resid"]),
    ("own", ["logit_p", "s_own"]),
    ("opp", ["logit_p", "s_opp"]),
    ("joint", ["logit_p", "s_own", "s_opp", "s_resid"]),
    ("resid_placebo", ["logit_p", "s_resid_pl"]),
]


def team_games():
    parts = cfb_season_partitions(REPO / "data" / "cfb", "pbp")
    cols = ["game_id", "season", "week", "seasonType", "pos_team_id", "rush", "pass", "kneel_down"]
    frames = []
    for s in SEASONS:
        f = pd.read_parquet(parts[s], columns=cols)
        for c in ("rush", "pass", "kneel_down"):
            f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0).astype(bool)
        f = f[pd.to_numeric(f["seasonType"], errors="coerce").eq(2)]
        f = f[(f["rush"] | f["pass"]) & ~f["kneel_down"] & f["pos_team_id"].notna()]
        frames.append(f)
    p = pd.concat(frames, ignore_index=True)
    p["team_id"] = pd.to_numeric(p["pos_team_id"]).astype("int64")
    run = p.groupby(["game_id", "team_id"]).agg(
        season=("season", "first"), week=("week", "first"), run=("rush", "size")
    )
    run = run.reset_index()
    pairs = run.merge(run[["game_id", "team_id", "run"]], on="game_id", suffixes=("", "_o"))
    pairs = pairs[pairs["team_id"].ne(pairs["team_id_o"])]
    counts = pairs.groupby(["game_id", "team_id"]).size()
    pairs = pairs.set_index(["game_id", "team_id"]).loc[counts.index[counts.eq(1)]].reset_index()
    t = pairs.rename(columns={"run_o": "defended"})
    t["week"] = pd.to_numeric(t["week"])
    t = t.sort_values(["team_id", "season", "week", "game_id"]).reset_index(drop=True)
    key = [t["team_id"], t["season"]]
    g = t.groupby(key)
    t["total"] = t["run"] + t["defended"]
    t["prior_n"] = g.cumcount()
    norm_t = g["total"].transform(lambda s: s.shift(1).expanding().mean())
    t["load"] = g["total"].transform(lambda s: s.shift(1).rolling(WINDOW).mean()) - norm_t
    t["load_pl"] = g["total"].transform(lambda s: s.shift(2).rolling(WINDOW).mean()) - norm_t
    t["own_td"] = g["run"].transform(lambda s: s.shift(1).expanding().mean())
    t["opp3"] = g["defended"].transform(lambda s: s.shift(1).rolling(WINDOW).mean())
    t["ok"] = t["prior_n"] >= WINDOW + 1
    return t


def base_logit(d, cols):
    p = pd.Series(np.nan, index=d.index)
    x = d[cols].to_numpy(float)
    y = d["result"].to_numpy(float) - d["spread_open"].to_numpy(float)
    for season in sorted(d["season"].unique()):
        trm = d["season"].ne(season).to_numpy()
        imp = SimpleImputer(strategy="median").fit(x[trm])
        sc = StandardScaler().fit(imp.transform(x[trm]))
        a = sc.transform(imp.transform(x[trm]))
        r = Ridge(alpha=sv.ALPHA).fit(a, y[trm])
        sd = float(np.std(y[trm] - r.predict(a), ddof=1))
        pred = r.predict(sc.transform(imp.transform(x[~trm])))
        p[~trm] = norm.cdf(pred / sd)
    p = p.clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build():
    d = pd.read_parquet(FEATURES)
    d = d[d["season"].isin(SEASONS) & d["spread_open"].notna() & d["result"].notna()].copy()
    d = d[d["result"].ne(d["spread_open"])].reset_index(drop=True)
    d["y"] = (d["result"] > d["spread_open"]).astype(int)
    cols = [c for c in ps.shared_columns() if c not in CLOSE_COLUMNS] + ["spread_open"]
    d["logit_p"] = base_logit(d, cols)
    t = team_games()
    keep = ["game_id", "team_id", "ok", "load", "load_pl", "own_td", "opp3"]
    for side in ("home", "away"):
        f = t[keep].rename(
            columns={c: f"{side}_{c}" for c in keep if c != "game_id"} | {"team_id": f"{side}_id"}
        )
        d = d.merge(f, on=["game_id", f"{side}_id"], how="left")
    d["usable"] = d["home_ok"].fillna(False).astype(bool) & d["away_ok"].fillna(False).astype(bool)
    for new, c in (
        ("s_load_total", "load"),
        ("s_resid_pl_raw", "load_pl"),
        ("s_own", "own_td"),
        ("s_opp", "opp3"),
    ):
        d[new] = (d[f"home_{c}"] - d[f"away_{c}"]).where(d["usable"], 0.0).fillna(0.0)
    u = d["usable"]
    x = np.column_stack([np.ones(u.sum()), d.loc[u, "s_own"], d.loc[u, "s_opp"]])
    beta, *_ = np.linalg.lstsq(x, d.loc[u, "s_load_total"].to_numpy(), rcond=None)
    d["s_resid"] = 0.0
    d.loc[u, "s_resid"] = d.loc[u, "s_load_total"].to_numpy() - x @ beta
    d["s_resid_pl"] = 0.0
    d.loc[u, "s_resid_pl"] = d.loc[u, "s_resid_pl_raw"].to_numpy() - x @ beta
    d["block"] = d["season"] * 100 + pd.to_numeric(d["week"]).astype(int)
    return d, [float(b) for b in beta]


def loso(d, cols):
    p = pd.Series(np.nan, index=d.index)
    co = {}
    for season in sorted(d["season"].unique()):
        trm = d["season"].ne(season)
        m = fit(d.loc[trm], cols)
        p[~trm] = m.predict_proba(d.loc[~trm, cols])[:, 1]
        co[int(season)] = [float(v) for v in m.coef_[0]]
    return p, co


def intercept_loso(d):
    p = pd.Series(np.nan, index=d.index)
    for season in sorted(d["season"].unique()):
        trm = d["season"].ne(season)
        p[~trm] = d.loc[trm, "y"].mean()
    return p


def analyze(d, cols, name):
    base_p, _ = loso(d, ["logit_p"])
    term_p, co = loso(d, cols)
    sc = d.assign(p_base=base_p, p_term=term_p, p_int=intercept_loso(d))

    def gain(f):
        return (
            ((f["p_term"] >= 0.5).astype(int) == f["y"]).mean()
            - ((f["p_base"] >= 0.5).astype(int) == f["y"]).mean()
        ) * 100.0

    def llg(f):
        return metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ins = d.assign(
        p_base=fit(d, ["logit_p"]).predict_proba(d[["logit_p"]])[:, 1],
        p_term=fit(d, cols).predict_proba(d[cols])[:, 1],
    )
    dec = sc.loc[(sc["p_term"] >= 0.5) != (sc["p_base"] >= 0.5)]
    return {
        "name": name,
        "cols": cols,
        "decisive_pick_changed_oos": len(dec),
        "decisive_term_correct": int(((dec["p_term"] >= 0.5).astype(int) == dec["y"]).sum()),
        "decisive_base_correct": int(((dec["p_base"] >= 0.5).astype(int) == dec["y"]).sum()),
        "intercept_only_oos": metrics(sc, "p_int"),
        "baseline_oos": metrics(sc, "p_base"),
        "term_oos": metrics(sc, "p_term"),
        "oos_accuracy_points_gain": summarize(block_bootstrap(sc, gain, rng), float(gain(sc))),
        "oos_log_loss_gain": summarize(block_bootstrap(sc, llg, rng), float(llg(sc))),
        "in_sample_log_loss_gain": float(llg(ins)),
        "in_minus_out_log_loss_gain": float(llg(ins) - llg(sc)),
        "coef_full": [float(v) for v in fit(d, cols).coef_[0]],
        "coef_per_heldout_season": co,
    }


def main():
    d, beta = build()
    out = {
        "family": FAMILY,
        "grade": "opener",
        "looks": len(TERMS),
        "games": len(d),
        "usable_games": int(d["usable"].sum()),
        "games_per_season": {
            int(k): int(v) for k, v in d["season"].value_counts().sort_index().items()
        },
        "decomposition_beta": beta,
    }
    for name, cols in TERMS:
        out[name] = analyze(d, cols, name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print({k: v for k, v in out.items() if not isinstance(v, dict) or k == "games_per_season"})
    for name, _ in TERMS:
        o = out[name]
        print(
            name,
            "decisive",
            o["decisive_pick_changed_oos"],
            o["decisive_term_correct"],
            o["decisive_base_correct"],
        )
        print(" int", o["intercept_only_oos"], "\n base", o["baseline_oos"])
        print(" term", o["term_oos"], "\n acc", o["oos_accuracy_points_gain"])
        print(" ll", o["oos_log_loss_gain"])
        print(" in", o["in_sample_log_loss_gain"], "gap", o["in_minus_out_log_loss_gain"])
        print(
            " coef_full",
            o["coef_full"],
            "folds",
            [round(v[-1], 5) for v in o["coef_per_heldout_season"].values()],
        )


if __name__ == "__main__":
    main()
