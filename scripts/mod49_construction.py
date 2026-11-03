import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_cumulative_workload as m  # noqa: E402
import mod41_unit2_components as u2  # noqa: E402
from per07_first_game_fitted_term import prepare  # noqa: E402

PICK = m.REPO / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
OUT = REPO / "artifacts" / "mod49" / "construction.json"
FAMILY = "mod49_mod41_construction"
BASE_WINDOW = m.WINDOW
VARIANTS = [
    ("V0_baseline", {"window": BASE_WINDOW, "load": "total", "norm": "season", "unit": "plays"}),
    ("W2", {"window": 2, "load": "total", "norm": "season", "unit": "plays"}),
    ("W4", {"window": 4, "load": "total", "norm": "season", "unit": "plays"}),
    ("W5", {"window": 5, "load": "total", "norm": "season", "unit": "plays"}),
    ("OFF_only", {"window": BASE_WINDOW, "load": "off", "norm": "season", "unit": "plays"}),
    ("PRIOR_norm", {"window": BASE_WINDOW, "load": "total", "norm": "prior", "unit": "plays"}),
    ("SEC", {"window": BASE_WINDOW, "load": "total", "norm": "season", "unit": "sec"}),
    (
        "PLACEBO_opp_opp",
        {
            "window": BASE_WINDOW,
            "load": "total",
            "norm": "season",
            "unit": "plays",
            "placebo": True,
        },
    ),
]


def possession_seconds() -> pd.DataFrame:
    cols = [
        "game_id",
        "play_id",
        "season_type",
        "fixed_drive",
        "game_seconds_remaining",
        "posteam",
        "defteam",
    ]
    p = pd.concat(
        [pd.read_parquet(m.PBP / f"season={s}" / "plays.parquet", columns=cols) for s in m.SEASONS],
        ignore_index=True,
    )
    p = p[(p.season_type == "REG") & p.posteam.notna() & p.defteam.notna()]
    p = p.sort_values(["game_id", "play_id"])
    g = p.groupby("game_id")
    nxt = g["game_seconds_remaining"].shift(-1)
    same = g["fixed_drive"].shift(-1).eq(p["fixed_drive"])
    p["dt"] = (p["game_seconds_remaining"] - nxt).where(same, 0.0).clip(lower=0.0).fillna(0.0)
    off = p.groupby(["game_id", "posteam"])["dt"].sum().rename("run_s")
    off = off.rename_axis(["game_id", "team"]).reset_index()
    dfn = p.groupby(["game_id", "defteam"])["dt"].sum().rename("def_s")
    dfn = dfn.rename_axis(["game_id", "team"]).reset_index()
    return off.merge(dfn, on=["game_id", "team"], how="outer").fillna(0.0)


def team_table() -> pd.DataFrame:
    t = m.team_games()[["game_id", "team", "run", "defended", "kickoff", "season", "prior_n"]]
    t = t.merge(possession_seconds(), on=["game_id", "team"], how="left")
    t = t.sort_values(["team", "kickoff"]).reset_index(drop=True)
    parts = t["game_id"].str.split("_", expand=True)
    t["opp"] = np.where(t["team"].eq(parts[2]), parts[3], parts[2])
    return t


def load_feature(t, spec):
    key = [t["team"], t["season"]]
    sec = spec["unit"] == "sec"
    run, dfn = ("run_s", "def_s") if sec else ("run", "defended")
    load = t[run] + t[dfn] if spec["load"] == "total" else t[run]
    prior = load.groupby(key).shift(1)
    if spec["norm"] == "season":
        norm = prior.groupby(key).expanding().mean().reset_index(level=[0, 1], drop=True)
    else:
        season_mean = load.groupby([t["team"], t["season"]]).mean().rename("sm").reset_index()
        season_mean["season"] += 1
        norm = t.merge(season_mean, on=["team", "season"], how="left")["sm"]
        norm.index = t.index
    roll = prior.groupby(key).rolling(spec["window"]).mean().reset_index(level=[0, 1], drop=True)
    return (
        roll - norm,
        t.groupby(key)[run].transform(lambda s: s.shift(1).expanding().mean()),
        (t.groupby(key)[dfn].transform(lambda s: s.shift(1).rolling(spec["window"]).mean())),
    )


def build_variant(base, t, spec):
    t = t.copy()
    f, own, opp = load_feature(t, spec)
    t["f"], t["own"], t["opp3"] = f, own, opp
    t["ok"] = t["prior_n"] >= spec["window"] + 1
    if spec.get("placebo"):
        prev = t.groupby(["team", "season"]).shift(1)
        t["pg"], t["pq"] = prev["game_id"], prev["opp"]
        lookup = t[["game_id", "team", "f"]].rename(
            columns={"game_id": "pg", "team": "pq", "f": "pf"}
        )
        po = t[["game_id", "team", "pg", "pq"]].rename(
            columns={"team": "opp", "pg": "o_pg", "pq": "o_pq"}
        )
        t = t.merge(po, on=["game_id", "opp"], how="left")
        t = t.merge(
            lookup.rename(columns={"pg": "o_pg", "pq": "o_pq", "pf": "pf"}),
            on=["o_pg", "o_pq"],
            how="left",
        )
    d = base.copy()
    cols = ["f", "own", "opp3", "ok"] + (["pf"] if spec.get("placebo") else [])
    for side in ("home", "away"):
        x = t[["game_id", "team", *cols]].rename(
            columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in cols}}
        )
        d = d.merge(x, on=["game_id", f"{side}_team"], how="left")
    ok = d["home_ok"].fillna(False).astype(bool) & d["away_ok"].fillna(False).astype(bool)
    for c in ("f", "own", "opp3"):
        d[f"s_{c}"] = (d[f"home_{c}"] - d[f"away_{c}"]).where(ok, np.nan)
    ok &= d[["s_f", "s_own", "s_opp3"]].notna().all(axis=1)
    target = "s_f"
    if spec.get("placebo"):
        d["s_pf"] = d["home_pf"] - d["away_pf"]
        ok &= d["s_pf"].notna()
        target = "s_pf"
    d["usable"] = ok
    x = np.column_stack([np.ones(ok.sum()), d.loc[ok, "s_own"], d.loc[ok, "s_opp3"]])
    beta, *_ = np.linalg.lstsq(x, d.loc[ok, target].to_numpy(), rcond=None)
    d["s_resid"] = 0.0
    d.loc[ok, "s_resid"] = d.loc[ok, target].to_numpy() - x @ beta
    return d, [float(b) for b in beta]


def main():
    base = prepare()
    parts = base["game_id"].str.split("_", expand=True)
    base["away_team"], base["home_team"] = parts[2], parts[3]
    pk = pd.read_parquet(PICK, columns=["game_id", "out_of_season_home_probability"])
    base = base.merge(pk, on="game_id", how="left")
    ps = base["out_of_season_home_probability"].clip(1e-6, 1 - 1e-6)
    base["logit_s"] = np.log(ps / (1 - ps))
    t = team_table()
    out = {"family": FAMILY, "grade": "opener", "looks": 2 * len(VARIANTS), "variants": {}}
    for name, spec in VARIANTS:
        d, beta = build_variant(base, t, spec)
        r = u2.run(d, ["logit_s"], ["s_resid"], name)
        r["null"] = u2.perm_null(d, ["logit_s"], "s_resid")
        r["spec"] = spec
        r["usable"] = int(d["usable"].sum())
        r["decomp_beta"] = beta
        out["variants"][name] = r
        print(
            name,
            r["usable"],
            "dec",
            r["decisive"],
            r["decisive_term_correct"],
            r["decisive_base_correct"],
            "ll",
            r["ll_gain"],
            "acc",
            r["acc_gain"],
            "gap",
            r["gap_ll"],
            "coef",
            r["coef_full"],
            [round(v[-1], 5) for v in r["coef_folds"].values()],
            "null",
            r["null"],
            flush=True,
        )
    OUT.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
