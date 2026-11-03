import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = sorted(glob.glob(str(ROOT / "data" / "raw" / "*" / "schedules.parquet")))[-1]
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
OPENER = sorted(
    glob.glob(str(ROOT / "artifacts" / "opener_evaluation" / "*" / "per_game.parquet"))
)[-1]
CAPTURES = sorted(glob.glob(str(ROOT / "data" / "market" / "raw" / "*" / "quotes.parquet")))
KAGGLE = ROOT / "data" / "market" / "historical" / "open_close" / "raw"
REVISIONS = ROOT / "artifacts" / "prospective" / "pick_revisions.graded.parquet"
OUT = ROOT / "artifacts" / "eng59"
FIRST_SEASON = 2011
SPREAD_SOURCES = [
    "nflv",
    "sbr_open",
    "sbr_close",
    "tue_open",
    "tue_close",
    "cap_sp_first",
    "cap_sp_last",
    "kg_sp",
]
ML_SOURCES = ["nflv_ml", "sbr_ml", "cap_ml_first", "cap_ml_last", "kg_ml"]
ML_COLUMN = {
    "nflv_ml": "nflv_p",
    "sbr_ml": "sbr_p",
    "cap_ml_first": "cap_p_first",
    "cap_ml_last": "cap_p_last",
    "kg_ml": "kg_p",
}
PAIRS = {
    "nflv_ml": "nflv",
    "sbr_ml": "sbr_close",
    "cap_ml_last": "cap_sp_last",
    "cap_ml_first": "cap_sp_first",
    "kg_ml": "kg_sp",
}
OPEN_STAGE = ["sbr_open", "cap_sp_first"]
CLOSE_STAGE = ["nflv", "sbr_close", "cap_sp_last", "kg_sp"]
PEER_TARGETS = (
    ("tue_open", [*OPEN_STAGE, "nflv"]),
    ("tue_close", CLOSE_STAGE),
    ("nflv", CLOSE_STAGE[1:]),
)


def implied(m):
    m = pd.Series(m, dtype=float)
    return np.where(m < 0, -m / (-m + 100.0), 100.0 / (m + 100.0))


def devig(home_ml, away_ml):
    ph = implied(home_ml)
    pa = implied(away_ml)
    return ph / (ph + pa)


def robust_z(x):
    x = pd.Series(x, dtype=float)
    med = x.median()
    zc = z_crit(len(x))
    scale = x.std()
    while True:
        inside = x[(x - med).abs() <= zc * scale]
        updated = inside.std()
        if updated == scale or updated == 0:
            break
        scale = updated
    return (x - med) / scale


def z_crit(n):
    return float(norm.ppf(1.0 - 0.5 / n))


def load_captures():
    sp_parts = []
    ml_parts = []
    cols = [
        "nflverse_game_id",
        "observed_at_utc",
        "commence_time_utc",
        "bookmaker_key",
        "market",
        "outcome_side",
        "price",
        "home_spread_line",
    ]
    for p in CAPTURES:
        d = pd.read_parquet(p, columns=cols)
        d = d[d["market"].isin(["spreads", "h2h"]) & d["nflverse_game_id"].notna()]
        d = d[d["observed_at_utc"] < d["commence_time_utc"]]
        if d.empty:
            continue
        s = d[(d["market"] == "spreads") & d["home_spread_line"].notna()]
        if not s.empty:
            sp_parts.append(
                s.groupby(["nflverse_game_id", "observed_at_utc", "bookmaker_key"], as_index=False)[
                    "home_spread_line"
                ].first()
            )
        h = d[(d["market"] == "h2h") & d["outcome_side"].isin(["HOME", "AWAY"])]
        if not h.empty:
            w = h.pivot_table(
                index=["nflverse_game_id", "observed_at_utc", "bookmaker_key"],
                columns="outcome_side",
                values="price",
                aggfunc="first",
            ).dropna()
            if not w.empty:
                w["p"] = devig(w["HOME"], w["AWAY"])
                ml_parts.append(
                    w.reset_index()[["nflverse_game_id", "observed_at_utc", "bookmaker_key", "p"]]
                )
    sp = pd.concat(sp_parts)
    ml = pd.concat(ml_parts)
    book_range = sp.groupby("nflverse_game_id")["home_spread_line"].agg(["min", "max"])
    sp = sp.groupby(["nflverse_game_id", "observed_at_utc"], as_index=False)[
        "home_spread_line"
    ].median()
    ml = ml.groupby(["nflverse_game_id", "observed_at_utc"], as_index=False)["p"].median()
    ids = sorted(set(sp["nflverse_game_id"]) | set(ml["nflverse_game_id"]))
    out = pd.DataFrame(index=pd.Index(ids, name="game_id"))
    sp = sp.sort_values("observed_at_utc")
    ml = ml.sort_values("observed_at_utc")
    g = sp.groupby("nflverse_game_id")["home_spread_line"]
    out["cap_sp_book_min"] = book_range["min"]
    out["cap_sp_book_max"] = book_range["max"]
    out["cap_sp_first"] = g.first()
    out["cap_sp_last"] = g.last()
    out["cap_n_spread"] = g.size()
    g = ml.groupby("nflverse_game_id")["p"]
    out["cap_p_first"] = g.first()
    out["cap_p_last"] = g.last()
    out["cap_n_ml"] = g.size()
    return out


def load_kaggle():
    parts = []
    for p in glob.glob(str(KAGGLE / "*" / "quotes.parquet")):
        d = pd.read_parquet(p)
        parts.append(d[(d["quote_stage"] == "reported_closing") & d["nflverse_game_id"].notna()])
    d = pd.concat(parts)
    sp = (
        d[d["market"] == "spreads"]
        .groupby("nflverse_game_id")["home_spread_line"]
        .median()
        .rename("kg_sp")
    )
    h = d[(d["market"] == "h2h") & d["outcome_side"].isin(["HOME", "AWAY"])]
    w = h.pivot_table(
        index=["nflverse_game_id", "bookmaker_key"],
        columns="outcome_side",
        values="price",
        aggfunc="first",
    ).dropna()
    w["p"] = devig(w["HOME"], w["AWAY"])
    ml = w.groupby("nflverse_game_id")["p"].median().rename("kg_p")
    out = pd.concat([sp, ml], axis=1)
    out.index.name = "game_id"
    return out


def build():
    s = pd.read_parquet(SCHEDULES)
    s = s[s["season"] >= FIRST_SEASON].drop_duplicates("game_id").set_index("game_id")
    base = pd.DataFrame(index=s.index)
    base["season"] = s["season"]
    base["week"] = s["week"]
    base["nflv"] = s["spread_line"]
    base["nflv_p"] = devig(s["home_moneyline"], s["away_moneyline"])
    base["result"] = s["result"]
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id"]).drop_duplicates("game_id")
    sbr = sbr.set_index("game_id")
    base["sbr_open"] = sbr["open_home_spread"]
    base["sbr_close"] = sbr["close_home_spread"]
    base["sbr_p"] = pd.Series(devig(sbr["home_moneyline"], sbr["away_moneyline"]), index=sbr.index)
    o = pd.read_parquet(OPENER).drop_duplicates("game_id").set_index("game_id")
    base["tue_open"] = o["tue_open_home_spread"]
    base["tue_close"] = o["close_home_spread"]
    base = base.join(load_captures()).join(load_kaggle())
    return base, o


def fit_relation(df, ml_col, sp_col):
    d = df[[ml_col, sp_col]].dropna()
    d = d[(d[ml_col] > 0) & (d[ml_col] < 1)]
    y = np.log(d[ml_col] / (1 - d[ml_col]))
    x = d[sp_col]
    b, a = np.polyfit(x, y, 1)
    return a, b, y - (a + b * x)


def regrade(res, pick, line):
    m = res - line
    cover_home = np.where(m > 0, 1.0, np.where(m < 0, 0.0, np.nan))
    return np.where(np.isnan(cover_home) | np.isnan(pick), np.nan, (pick == cover_home) * 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, opener = build()
    summary = {}
    fits = {}
    for mlsrc, spsrc in PAIRS.items():
        a, b, r = fit_relation(df, ML_COLUMN[mlsrc], spsrc)
        z = robust_z(r)
        zc = z_crit(len(r))
        summary[mlsrc] = {
            "n": len(r),
            "intercept": float(a),
            "slope_logit_per_point": float(b),
            "resid_sd": float(r.std()),
            "z_crit": zc,
        }
        fits[mlsrc] = (z, zc)
    votes = pd.DataFrame(index=df.index)
    for s in SPREAD_SOURCES:
        votes[s] = np.sign(df[s])
    for m in ML_SOURCES:
        votes[m] = np.sign(df[ML_COLUMN[m]] - 0.5)
    votes = votes.replace(0, np.nan)
    tally = votes.sum(axis=1)
    cnt = votes.notna().sum(axis=1)
    df["n_votes"] = cnt
    df["majority_fav_home"] = np.where(tally > 0, 1.0, np.where(tally < 0, 0.0, np.nan))
    df["majority_margin"] = tally.abs() / cnt
    maj = np.sign(tally)
    dissent = votes.apply(lambda c: c.notna() & (c != maj) & (maj != 0))
    df["dissenting_sources"] = dissent.apply(lambda r: ";".join(r.index[r.values]), axis=1)
    out_rows = []
    for mlsrc, spsrc in PAIRS.items():
        z, zc = fits[mlsrc]
        pcol = ML_COLUMN[mlsrc]
        d = df[df[pcol].notna() & df[spsrc].notna()]
        sg_s = np.sign(d[spsrc])
        sg_m = np.sign(d[pcol] - 0.5)
        fav_dis = (sg_s != 0) & (sg_m != 0) & (sg_s != sg_m)
        zz = z.reindex(d.index)
        for gid in d.index:
            reasons = []
            if fav_dis.loc[gid]:
                reasons.append("favourite_disagrees")
            if abs(zz.loc[gid]) > zc:
                reasons.append("ml_not_monotone_in_spread")
            for reason in reasons:
                out_rows.append((gid, mlsrc, spsrc, reason, float(zz.loc[gid]), zc))
    for stage, members in (
        ("open_stage", ["sbr_open", "tue_open", "cap_sp_first"]),
        ("close_stage", ["nflv", "sbr_close", "tue_close", "cap_sp_last", "kg_sp"]),
    ):
        for gid, r in votes[members].iterrows():
            if r.dropna().nunique() > 1:
                out_rows.append(
                    (gid, "", stage, "spread_sign_conflict_within_stage", np.nan, np.nan)
                )
    for tgt, peers in PEER_TARGETS:
        dv = (df[tgt] - df[peers].median(axis=1)).dropna()
        z = robust_z(dv)
        zc = z_crit(len(dv))
        summary[f"peer_{tgt}"] = {
            "n": len(dv),
            "z_crit": zc,
            "mad_points": float((dv - dv.median()).abs().median()),
        }
        for gid in z.index[z.abs() > zc]:
            out_rows.append(
                (gid, "", tgt, "spread_magnitude_outlier_vs_peers", float(z.loc[gid]), zc)
            )
    flags = pd.DataFrame(
        out_rows, columns=["game_id", "ml_source", "spread_source", "reason", "z", "z_crit"]
    )
    keep = [
        "cap_sp_book_min",
        "cap_sp_book_max",
        "season",
        "week",
        "result",
        *SPREAD_SOURCES,
        "nflv_p",
        "sbr_p",
        "cap_p_first",
        "cap_p_last",
        "kg_p",
        "n_votes",
        "majority_fav_home",
        "majority_margin",
        "dissenting_sources",
    ]
    audit = flags.merge(df[keep], left_on="game_id", right_index=True, how="left")
    op = opener.reindex(audit["game_id"])
    audit["opener_pick_home"] = op["pick_home_at_open"].values
    audit["opener_correct_at_open"] = op["correct_at_open"].values
    consensus_open = df[[*OPEN_STAGE, "nflv"]].median(axis=1)
    audit["alt_open"] = consensus_open.reindex(audit["game_id"]).values
    pick = audit["opener_pick_home"].astype(float).values
    audit["alt_correct_at_open"] = regrade(audit["result"].values, pick, audit["alt_open"].values)
    own = audit["opener_correct_at_open"].astype(float).values
    audit["tue_open_sign_dissent_raw"] = audit["dissenting_sources"].str.contains("tue_open")
    bmin = df["cap_sp_book_min"].reindex(audit["game_id"]).values
    bmax = df["cap_sp_book_max"].reindex(audit["game_id"]).values
    tue = audit["tue_open"].values
    audit["tue_open_supported_by_captures"] = np.where(
        np.isnan(bmin) | np.isnan(tue), np.nan, ((tue >= bmin) & (tue <= bmax)) * 1.0
    )
    audit["tue_open_sign_dissent"] = audit["tue_open_sign_dissent_raw"] & (
        audit["tue_open_supported_by_captures"] != 1.0
    )
    tue_wrong = audit["tue_open_sign_dissent"] | (
        (audit["reason"] == "spread_magnitude_outlier_vs_peers")
        & (audit["spread_source"] == "tue_open")
        & (audit["tue_open_supported_by_captures"] != 1.0)
    )
    audit["tue_open_flagged_wrong"] = tue_wrong
    alt = audit["alt_correct_at_open"].values
    audit["opener_grade_changes"] = tue_wrong & ~np.isnan(own) & ~np.isnan(alt) & (own != alt)
    audit["opener_pick_side_may_change"] = tue_wrong & audit["opener_pick_home"].notna()
    audit = audit.sort_values(["season", "week", "game_id", "reason"])
    audit.to_csv(OUT / "audit.csv", index=False)
    rev = pd.read_parquet(REVISIONS)
    rv = rev.merge(
        df[["nflv", "majority_fav_home", "dissenting_sources", "sbr_open", "cap_sp_first"]],
        left_on="game_id",
        right_index=True,
        how="left",
    )
    rv["spread_gap_vs_nflv"] = rv["decision_home_spread"] - rv["nflv"]
    rv.to_csv(OUT / "scorecard_2026_crosscheck.csv", index=False)
    cols = SPREAD_SOURCES + [ML_COLUMN[m] for m in ML_SOURCES]
    print("schedules", SCHEDULES)
    print(
        "games",
        len(df),
        "with any ML",
        int(df[[ML_COLUMN[m] for m in ML_SOURCES]].notna().any(axis=1).sum()),
    )
    print("coverage", df[cols].notna().sum().to_dict())
    print(json.dumps(summary, indent=1))
    print(flags.groupby(["reason", "ml_source", "spread_source"]).size().to_string())
    print("unique flagged games", flags["game_id"].nunique())
    sd = audit["tue_open_sign_dissent"]
    print(
        "raw tue_open dissent games",
        audit.loc[audit["tue_open_sign_dissent_raw"], "game_id"].nunique(),
    )
    print(
        "raw dissent but inside capture book range (genuine moves)",
        audit.loc[
            audit["tue_open_sign_dissent_raw"] & ~audit["tue_open_sign_dissent"], "game_id"
        ].nunique(),
    )
    print("tue_open sign dissent games", audit.loc[sd, "game_id"].nunique())
    print(
        "sign-dissent grade changes",
        audit.loc[sd & audit["opener_grade_changes"], "game_id"].unique().tolist(),
    )
    print("tue_open wrong games", audit.loc[tue_wrong, "game_id"].nunique())
    print("grade changes", audit.loc[audit["opener_grade_changes"], "game_id"].unique().tolist())
    print("dissent", df["dissenting_sources"].str.split(";").explode().value_counts().to_dict())
    for g in ("2020_13_BUF_SF", "2024_06_JAX_CHI"):
        print(
            g,
            df.loc[
                g, [*cols, "majority_fav_home", "n_votes", "dissenting_sources", "result"]
            ].to_dict(),
        )
    print("scorecard rows", len(rv), "gap!=0:", int((rv["spread_gap_vs_nflv"].abs() > 0).sum()))
    print(
        rv.loc[
            rv["dissenting_sources"].fillna("") != "",
            ["game_id", "decision_home_spread", "nflv", "dissenting_sources"],
        ].to_string()
    )


if __name__ == "__main__":
    main()
