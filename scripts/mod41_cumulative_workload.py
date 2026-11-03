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

PBP = REPO / "data" / "pbp" / "raw" / "20260929T191306Z"
OUT = REPO / "artifacts" / "mod41" / "cumulative_workload.json"
FAMILY = "mod41_cumulative_workload"
DRAWS = 1000
WINDOW = 3
FULL_WEEK_HOURS = 6 * 24
SEASONS = range(2019, 2026)
_per07.BOOTSTRAP_SAMPLES = DRAWS
FEATS = {"total": "load_total", "defended": "load_def", "flag": "flags"}
TERMS = [
    ("load_total", ["logit_p", "s_load_total"]),
    ("load_total_placebo", ["logit_p", "s_load_total_pl"]),
    ("load_defended", ["logit_p", "s_load_def"]),
    ("load_defended_placebo", ["logit_p", "s_load_def_pl"]),
    ("ot_short_week_flags", ["logit_p", "s_flags"]),
]


def team_games() -> pd.DataFrame:
    cols = [
        "game_id",
        "season_type",
        "posteam",
        "defteam",
        "play_type",
        "qb_kneel",
        "qb_spike",
        "qtr",
    ]
    p = pd.concat(
        [pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols) for s in SEASONS],
        ignore_index=True,
    )
    p = p[(p.season_type == "REG") & p.play_type.isin(["pass", "run"])]
    p = p[(p.qb_kneel != 1) & (p.qb_spike != 1) & p.posteam.notna() & p.defteam.notna()]
    run = (
        p.groupby(["game_id", "posteam"])
        .size()
        .rename("run")
        .rename_axis(["game_id", "team"])
        .reset_index()
    )
    dfn = (
        p.groupby(["game_id", "defteam"])
        .size()
        .rename("defended")
        .rename_axis(["game_id", "team"])
        .reset_index()
    )
    ot = (p.qtr > 4).groupby(p.game_id).any().rename("ot").reset_index()
    t = run.merge(dfn, on=["game_id", "team"], how="outer").fillna(0).merge(ot, on="game_id")
    g = pd.read_parquet(GAME_FEATURES, columns=["game_id", "kickoff"])
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True)
    t = t.merge(g, on="game_id").sort_values(["team", "kickoff"]).reset_index(drop=True)
    t["season"] = t["game_id"].str[:4].astype(int)
    t["total"] = t["run"] + t["defended"]
    key = [t["team"], t["season"]]
    t["short"] = (
        t.groupby(key)["kickoff"].diff().dt.total_seconds() / 3600.0 < FULL_WEEK_HOURS
    ).astype(float)
    t["flag"] = t["ot"].astype(float) + t["short"]
    t["prior_n"] = t.groupby(key).cumcount()
    for c, name in FEATS.items():
        prior = t.groupby(key)[c].shift(1)
        norm = prior.groupby(key).expanding().mean().reset_index(level=[0, 1], drop=True)
        for tag, lag in (("", 1), ("_pl", 2)):
            s = t.groupby(key)[c].shift(lag)
            roll = s.groupby(key).rolling(WINDOW).mean().reset_index(level=[0, 1], drop=True)
            t[name + tag] = roll - (norm if c != "flag" else 0.0)
    t["ok"] = t["prior_n"] >= WINDOW + 1
    return t


def build() -> pd.DataFrame:
    d = prepare()
    parts = d["game_id"].str.split("_", expand=True)
    d["away_team"], d["home_team"] = parts[2], parts[3]
    t = team_games()
    names = [n + tag for n in FEATS.values() for tag in ("", "_pl")]
    for side in ("home", "away"):
        f = t[["game_id", "team", "ok", *names]].rename(
            columns={
                "team": f"{side}_team",
                "ok": f"{side}_ok",
                **{n: f"{side}_{n}" for n in names},
            }
        )
        d = d.merge(f, on=["game_id", f"{side}_team"], how="left")
    d["usable"] = d["home_ok"].fillna(False).astype(bool) & d["away_ok"].fillna(False).astype(bool)
    for n in names:
        d[f"s_{n}"] = (d[f"home_{n}"] - d[f"away_{n}"]).where(d["usable"], 0.0).fillna(0.0)
    d["const"] = 0.0
    return d, t


def loso(d, cols):
    p = pd.Series(np.nan, index=d.index)
    co = {}
    for season in sorted(d["season"].unique()):
        trm = d["season"].ne(season)
        m = fit(d.loc[trm], cols)
        p[~trm] = m.predict_proba(d.loc[~trm, cols])[:, 1]
        co[int(season)] = [float(x) for x in m.coef_[0]]
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
    a = summarize(block_bootstrap(sc, gain, rng), float(gain(sc)))
    ll = summarize(block_bootstrap(sc, llg, rng), float(llg(sc)))
    return {
        "name": name,
        "cols": cols,
        "decisive_pick_changed_oos": len(dec),
        "decisive_term_correct": int(((dec["p_term"] >= 0.5).astype(int) == dec["y"]).sum()),
        "decisive_base_correct": int(((dec["p_base"] >= 0.5).astype(int) == dec["y"]).sum()),
        "intercept_only_oos": metrics(sc, "p_int"),
        "baseline_oos": metrics(sc, "p_base"),
        "term_oos": metrics(sc, "p_term"),
        "oos_accuracy_points_gain": a,
        "oos_log_loss_gain": ll,
        "in_sample_accuracy_points_gain": float(gain(ins)),
        "in_sample_log_loss_gain": float(llg(ins)),
        "in_minus_out_log_loss_gain": float(llg(ins) - llg(sc)),
        "coef_full": [float(x) for x in fit(d, cols).coef_[0]],
        "coef_per_heldout_season": co,
    }


def main():
    d, t = build()
    out = {
        "family": FAMILY,
        "grade": "opener",
        "looks": len(TERMS),
        "games": len(d),
        "usable_games": int(d["usable"].sum()),
        "ot_game_share": float(t.loc[t["ok"], "ot"].mean()),
        "short_week_share": float(t.loc[t["ok"], "short"].mean()),
        "corr_total_defended": float(
            d.loc[d["usable"], "s_load_total"].corr(d.loc[d["usable"], "s_load_def"])
        ),
    }
    for name, cols in TERMS:
        out[name] = analyze(d, cols, name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print({k: v for k, v in out.items() if not isinstance(v, dict)})
    for name, _ in TERMS:
        o = out[name]
        print(
            name,
            o["decisive_pick_changed_oos"],
            o["decisive_term_correct"],
            o["decisive_base_correct"],
        )
        print(
            " int", o["intercept_only_oos"], "\n base", o["baseline_oos"], "\n term", o["term_oos"]
        )
        print(" acc", o["oos_accuracy_points_gain"], "\n ll", o["oos_log_loss_gain"])
        print(
            " in",
            o["in_sample_accuracy_points_gain"],
            o["in_sample_log_loss_gain"],
            "gap",
            o["in_minus_out_log_loss_gain"],
        )
        print(
            " coef_full",
            o["coef_full"],
            "folds",
            [round(v[-1], 5) for v in o["coef_per_heldout_season"].values()],
        )


if __name__ == "__main__":
    main()
