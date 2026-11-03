import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import per07_first_game_fitted_term as _per07  # noqa: E402
from per07_first_game_fitted_term import (  # noqa: E402
    GAME_FEATURES,
    block_bootstrap,
    fit,
    metrics,
    prepare,
    summarize,
)
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

OUT = REPO / "artifacts" / "mod29" / "hour_rest.json"
DRAWS = 1000
_per07.BOOTSTRAP_SAMPLES = DRAWS
ET, CT, MT, PT = "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"
TEAM_TZ = {
    "ARI": "America/Phoenix",
    "ATL": ET,
    "BAL": ET,
    "BUF": ET,
    "CAR": ET,
    "CHI": CT,
    "CIN": ET,
    "CLE": ET,
    "DAL": CT,
    "DEN": MT,
    "DET": ET,
    "GB": CT,
    "HOU": CT,
    "IND": ET,
    "JAX": ET,
    "KC": CT,
    "LA": PT,
    "LAC": PT,
    "LAR": PT,
    "LV": PT,
    "MIA": ET,
    "MIN": CT,
    "NE": ET,
    "NO": CT,
    "NYG": ET,
    "NYJ": ET,
    "OAK": PT,
    "PHI": ET,
    "PIT": ET,
    "SD": PT,
    "SEA": PT,
    "SF": PT,
    "STL": CT,
    "TB": ET,
    "TEN": CT,
    "WAS": ET,
}
TERMS = [
    ("hour_resid", ["logit_p", "s_resid"]),
    ("clock_gap", ["logit_p", "s_clock"]),
    ("both", ["logit_p", "s_resid", "s_clock"]),
]


def local_hour(utc: pd.Series, tz: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=utc.index)
    for z in tz.unique():
        m = tz.eq(z)
        t = utc[m].dt.tz_convert(z)
        out[m] = t.dt.hour + t.dt.minute / 60.0
    return out


def team_rows() -> pd.DataFrame:
    g = pd.read_parquet(
        GAME_FEATURES, columns=["game_id", "season", "kickoff", "home_team", "away_team"]
    )
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True)
    g["venue_tz"] = g["home_team"].map(TEAM_TZ)
    rows = []
    for side in ("home", "away"):
        rows.append(
            g.assign(team=g[f"{side}_team"], side=side)[
                ["game_id", "season", "kickoff", "venue_tz", "team", "side"]
            ]
        )
    r = pd.concat(rows).sort_values(["team", "kickoff"]).reset_index(drop=True)
    r["hour"] = local_hour(r["kickoff"], r["venue_tz"])
    k = r.groupby(["team", "season"])
    r["prev_kick"] = k["kickoff"].shift(1)
    r["prev_hour"] = k["hour"].shift(1)
    r["rest_hours"] = (r["kickoff"] - r["prev_kick"]).dt.total_seconds() / 3600.0
    gap = r["hour"] - r["prev_hour"]
    r["clock_gap"] = (gap + 12.0) % 24.0 - 12.0
    return r


def build() -> pd.DataFrame:
    d = prepare()
    g = pd.read_parquet(GAME_FEATURES, columns=["game_id", "rest_diff"])
    d = d.merge(g, on="game_id", how="left")
    r = team_rows()
    for side in ("home", "away"):
        s = r.loc[r["side"].eq(side), ["game_id", "rest_hours", "clock_gap"]]
        d = d.merge(
            s.rename(
                columns={"rest_hours": f"{side}_rest_hours", "clock_gap": f"{side}_clock_gap"}
            ),
            on="game_id",
            how="left",
        )
    d["usable"] = d[["home_rest_hours", "away_rest_hours", "rest_diff"]].notna().all(axis=1)
    d["hours_diff"] = d["home_rest_hours"] - d["away_rest_hours"]
    d["s_clock_raw"] = d["home_clock_gap"] - d["away_clock_gap"]
    d["s_clock"] = d["s_clock_raw"].where(d["usable"], 0.0).fillna(0.0)
    return d


def residualize(d: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    tr = d.loc[train_mask & d["usable"]]
    x = np.column_stack([np.ones(len(tr)), tr["rest_diff"]])
    beta = np.linalg.lstsq(x, tr["hours_diff"], rcond=None)[0]
    res = d["hours_diff"] - (beta[0] + beta[1] * d["rest_diff"])
    return res.where(d["usable"], 0.0)


def loso(d, cols):
    p = pd.Series(np.nan, index=d.index)
    co = {}
    for season in sorted(d["season"].unique()):
        trm = d["season"].ne(season)
        d = d.assign(s_resid=residualize(d, trm))
        m = fit(d.loc[trm], cols)
        te = ~trm
        p[te] = m.predict_proba(d.loc[te, cols])[:, 1]
        co[int(season)] = [float(x) for x in m.coef_[0]]
    return p, co


def analyze(d, cols, name):
    base_p, _ = loso(d, ["logit_p"])
    term_p, co = loso(d, cols)
    sc = d.assign(p_base=base_p, p_term=term_p)

    def gain(f):
        return (
            ((f["p_term"] >= 0.5).astype(int) == f["y"]).mean()
            - ((f["p_base"] >= 0.5).astype(int) == f["y"]).mean()
        ) * 100.0

    def llg(f):
        return metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    di = d.assign(s_resid=residualize(d, pd.Series(True, index=d.index)))
    full = fit(di, cols)
    ins = di.assign(
        p_base=fit(di, ["logit_p"]).predict_proba(di[["logit_p"]])[:, 1],
        p_term=full.predict_proba(di[cols])[:, 1],
    )
    dec = sc.loc[(sc["p_term"] >= 0.5) != (sc["p_base"] >= 0.5)]
    return {
        "name": name,
        "cols": cols,
        "decisive_pick_changed_oos": len(dec),
        "decisive_term_correct": int(((dec["p_term"] >= 0.5).astype(int) == dec["y"]).sum()),
        "decisive_base_correct": int(((dec["p_base"] >= 0.5).astype(int) == dec["y"]).sum()),
        "baseline_oos": metrics(sc, "p_base"),
        "term_oos": metrics(sc, "p_term"),
        "oos_accuracy_points_gain": summarize(block_bootstrap(sc, gain, rng), float(gain(sc))),
        "oos_log_loss_gain": summarize(block_bootstrap(sc, llg, rng), float(llg(sc))),
        "in_sample_accuracy_points_gain": float(gain(ins)),
        "in_sample_log_loss_gain": float(llg(ins)),
        "in_minus_out_log_loss_gain": float(llg(ins) - llg(sc)),
        "coef_full": [float(x) for x in full.coef_[0]],
        "coef_per_heldout_season": co,
    }


def main():
    d = build()
    u = d.loc[d["usable"]]
    out = {
        "family": "mod29_hour_rest",
        "grade": "opener",
        "looks": len(TERMS),
        "games": len(d),
        "usable_games": int(d["usable"].sum()),
        "corr_hours_diff_rest_diff": float(u["hours_diff"].corr(u["rest_diff"])),
        "r2_hours_on_days": float(u["hours_diff"].corr(u["rest_diff"]) ** 2),
        "hours_diff_sd": float(u["hours_diff"].std()),
        "resid_sd_full": float(residualize(d, pd.Series(True, index=d.index))[d["usable"]].std()),
        "clock_gap_diff_sd": float(u["s_clock_raw"].std()),
    }
    for name, cols in TERMS:
        out[name] = analyze(d, cols, name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in [t[0] for t in TERMS]}, indent=1))
    for name, _ in TERMS:
        o = out[name]
        print(
            name,
            o["decisive_pick_changed_oos"],
            o["decisive_term_correct"],
            o["decisive_base_correct"],
            o["baseline_oos"],
            o["term_oos"],
            o["oos_accuracy_points_gain"],
            o["oos_log_loss_gain"],
            o["in_sample_log_loss_gain"],
            o["in_minus_out_log_loss_gain"],
            o["coef_full"],
        )


if __name__ == "__main__":
    main()
