import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
OPENER_GLOB = str(ROOT / "artifacts" / "opener_evaluation" / "*" / "per_game.parquet")
PROVIDER_GLOB = str(
    ROOT / "data" / "market" / "historical" / "open_close" / "raw" / "*" / "games.parquet"
)
SPLASH_GLOB = str(ROOT / "data" / "splash" / "2026_week*.json")
REVISIONS = ROOT / "artifacts" / "prospective" / "pick_revisions.graded.parquet"
OUT = ROOT / "artifacts" / "eng60"
FIRST_SEASON = 2011
SIGN_FLIP_MIN_LINE = 3.0


def wl(series):
    s = series.dropna()
    return f"{int((s == 1).sum())}-{int((s == 0).sum())}"


def mean_se(x):
    x = pd.Series(x).dropna()
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(FEATURES)
    parts = f["game_id"].str.split("_", expand=True)
    f["id_away"] = parts[2]
    f["id_home"] = parts[3]
    f["margin_vs_spread"] = f["result"] - f["spread_line"]
    n = f[(f["neutral_site"] == 1) & (f["season"] >= FIRST_SEASON)].copy()
    print(f"neutral games {FIRST_SEASON}-2026: {len(n)}; graded: {int(n['result'].notna().sum())}")
    print(n.groupby("season").size().to_dict())
    print("by game_type", n["game_type"].value_counts().to_dict())
    print(
        "home_team != id_home mismatches (all games):",
        int((f["home_team"] != f["id_home"]).sum()),
    )

    print("--- (a) features")
    if {"home_elo", "away_elo"}.issubset(f.columns):
        gap = f["elo_diff"] - (f["home_elo"] - f["away_elo"])
        print(
            "elo_diff - (home_elo-away_elo): neutral",
            sorted(set(gap[f["neutral_site"] == 1].round(6))),
            "non-neutral",
            sorted(set(gap[f["neutral_site"] == 0].round(6)))[:3],
        )
    else:
        print("home_elo/away_elo not in table")
    if "home_score" in f.columns and "away_score" in f.columns:
        diff = (f["home_score"] - f["away_score"] - f["result"]).abs()
        print(
            "result != home_score-away_score:",
            int((diff > 0).sum()),
            "neutral:",
            int((diff[f["neutral_site"] == 1] > 0).sum()),
        )
    played = f[f["result"].notna() & (f["season"] >= FIRST_SEASON)]
    nn = played[played["neutral_site"] == 1]
    hh = played[played["neutral_site"] == 0]
    for label, g in (("neutral", nn), ("home", hh)):
        m, se, k = mean_se(g["margin_vs_spread"])
        r, rse, _ = mean_se(g["result"])
        print(
            f"{label}: n={k} mean(result-spread)={m:.3f} se={se:.3f} "
            f"mean result={r:.2f} se={rse:.2f}"
        )
    for label, g in (("neutral", nn), ("home", hh)):
        print(
            f"{label} nominal-home cover W-L (non-push): "
            f"{int((g['margin_vs_spread'] > 0).sum())}-{int((g['margin_vs_spread'] < 0).sum())}"
        )

    o = pd.read_parquet(sorted(glob.glob(OPENER_GLOB))[-1])
    o = o.merge(
        f[["game_id", "neutral_site", "spread_line", "home_team", "id_home", "id_away"]],
        on="game_id",
        how="left",
    )
    on = o[o["neutral_site"] == 1].copy()
    print("--- (b) opener orientation")
    print(f"neutral in opener-graded artifact: {len(on)} of {len(o)}")
    on["open_minus_nflverse"] = on["tue_open_home_spread"] - on["spread_line"]
    on["sign_flip"] = np.sign(on["tue_open_home_spread"]) * np.sign(on["spread_line"]) < 0
    print("sign flips (neutral):", on.loc[on["sign_flip"], "game_id"].tolist())
    print("open_minus_nflverse:", on["open_minus_nflverse"].describe().round(2).to_dict())
    allgap = (o["tue_open_home_spread"] - o["spread_line"]).abs()
    print(
        "mean |open-nflverse| non-neutral:",
        round(float(allgap[o["neutral_site"] == 0].mean()), 3),
        "neutral:",
        round(float(allgap[o["neutral_site"] == 1].mean()), 3),
    )
    flip_all = o[
        (np.sign(o["tue_open_home_spread"]) * np.sign(o["spread_line"]) < 0)
        & (o["tue_open_home_spread"].abs() >= SIGN_FLIP_MIN_LINE)
    ]
    print(
        "sign flips any game |open|>=3:",
        flip_all[["game_id", "neutral_site", "tue_open_home_spread", "spread_line"]].to_dict(
            "records"
        ),
    )

    prov = pd.concat([pd.read_parquet(p) for p in glob.glob(PROVIDER_GLOB)], ignore_index=True)
    prov = prov.dropna(subset=["nflverse_game_id"]).drop_duplicates("nflverse_game_id")
    prov = prov.merge(
        f[["game_id", "neutral_site", "id_home", "id_away"]],
        left_on="nflverse_game_id",
        right_on="game_id",
        how="left",
    )
    pn = prov[prov["neutral_site"] == 1]
    print(
        f"provider rows {len(prov)}, neutral {len(pn)}; provider home != nflverse home: "
        f"all {int((prov['home_team'] != prov['id_home']).sum())}, "
        f"neutral {int((pn['home_team'] != pn['id_home']).sum())}"
    )
    print(
        pn[
            ["nflverse_game_id", "home_team", "opening_home_spread", "nflverse_spread_line"]
        ].to_string(index=False)
    )

    print("--- (c) grader")
    recomputed = on["result"] - on["tue_open_home_spread"]
    print(
        "margin_vs_open != result - open (neutral):",
        int((recomputed - on["margin_vs_open"]).abs().gt(0).sum()),
    )
    side_ok = np.where(on["pick_home_at_open"], on["margin_vs_open"] > 0, on["margin_vs_open"] < 0)
    expected = pd.Series(
        np.where(on["margin_vs_open"] == 0, np.nan, side_ok.astype(float)), index=on.index
    )
    print(
        "correct_at_open mismatches vs independent recompute:",
        int((expected.fillna(-1) != on["correct_at_open"].fillna(-1)).sum()),
    )

    print("--- opener record, neutral vs rest")
    for label, g in (("neutral", on), ("non-neutral", o[o["neutral_site"] == 0])):
        print(
            f"{label}: n={len(g)} correct_at_open W-L={wl(g['correct_at_open'])} "
            f"pick_home share={g['pick_home_at_open'].astype(float).mean():.3f} "
            f"mean residual_at_open={g['residual_at_open'].mean():.3f} "
            f"prob_rule W-L={wl(g['correct_at_open_probability_rule'])}"
        )
    print(
        "neutral pick-home: nominal home dog",
        int(((on["tue_open_home_spread"] < 0) & on["pick_home_at_open"]).sum()),
        "of",
        int((on["tue_open_home_spread"] < 0).sum()),
        "; nominal home fav",
        int(((on["tue_open_home_spread"] > 0) & on["pick_home_at_open"]).sum()),
        "of",
        int((on["tue_open_home_spread"] > 0).sum()),
    )
    cols = [
        "game_id",
        "tue_open_home_spread",
        "spread_line",
        "residual_at_open",
        "pick_home_at_open",
        "margin_vs_open",
        "correct_at_open",
    ]
    print(on[cols].to_string(index=False))

    print("--- (d) splash 2026")
    rows = []
    for p in sorted(glob.glob(SPLASH_GLOB)):
        with open(p) as handle:
            j = json.load(handle)
        for g in j["games"]:
            rows.append(
                {
                    "file": Path(p).name,
                    "game_id": g["game_id"],
                    "home": g["home"],
                    "away": g["away"],
                    "home_spread": g["home_spread"],
                }
            )
    s = pd.DataFrame(rows).merge(
        f[["game_id", "neutral_site", "id_home", "id_away", "spread_line"]],
        on="game_id",
        how="left",
    )
    s["orient_ok"] = (s["home"] == s["id_home"]) & (s["away"] == s["id_away"])
    print(
        f"splash rows {len(s)}; unmatched {int(s['neutral_site'].isna().sum())}; "
        f"orientation mismatches {int((~s['orient_ok']).sum())}"
    )
    print(
        s.loc[~s["orient_ok"], ["file", "game_id", "home", "away", "home_spread"]].to_string(
            index=False
        )
    )
    sn = s[s["neutral_site"] == 1].copy()
    sn["spread_vs_schedule"] = sn["home_spread"] - sn["spread_line"]
    print(
        sn[
            ["file", "game_id", "home", "away", "home_spread", "spread_line", "spread_vs_schedule"]
        ].to_string(index=False)
    )
    last = s.sort_values("file").groupby("game_id").tail(1)
    flips = last[
        (np.sign(last["home_spread"]) * np.sign(last["spread_line"]) < 0)
        & (last["home_spread"].abs() >= SIGN_FLIP_MIN_LINE)
    ]
    print(
        "splash vs schedule sign flips (last capture):",
        flips[["game_id", "home_spread", "spread_line"]].to_dict("records"),
    )

    print("--- (f) SBR third source")
    sbr = pd.read_parquet(ROOT / "data" / "processed" / "sbr_odds.parquet")
    sbr = sbr.dropna(subset=["game_id"]).drop_duplicates("game_id")
    m = sbr.merge(
        f[["game_id", "neutral_site", "spread_line", "id_home"]],
        on="game_id",
        how="inner",
        suffixes=("_sbr", ""),
    )
    print(f"sbr rows matched {len(m)}; seasons {int(m['season'].min())}-{int(m['season'].max())}")
    print(
        "sbr neutral flag vs nflverse neutral crosstab:",
        pd.crosstab(m["neutral_site_sbr"].astype(bool), m["neutral_site"].astype(bool)).to_dict(),
    )
    print("sbr home_team != nflverse id home:", int((m["home_team"] != m["id_home"]).sum()))
    m["close_vs_nflverse"] = m["close_home_spread"] - m["spread_line"]
    m["flip"] = np.sign(m["close_home_spread"]) * np.sign(m["spread_line"]) < 0
    for label, g in (
        ("neutral", m[m["neutral_site"] == 1]),
        ("non-neutral", m[m["neutral_site"] == 0]),
    ):
        print(
            f"{label}: n={len(g)} sign flips close-vs-nflverse={int(g['flip'].sum())} "
            f"mean |gap|={g['close_vs_nflverse'].abs().mean():.3f}"
        )
    mn = m[m["neutral_site"] == 1]
    print(
        mn[
            [
                "game_id",
                "home_team",
                "open_home_spread",
                "close_home_spread",
                "spread_line",
                "close_vs_nflverse",
            ]
        ].to_string(index=False)
    )

    print("--- (e) served ledger")
    r = pd.read_parquet(REVISIONS).merge(
        f[["game_id", "neutral_site", "spread_line"]], on="game_id", how="left"
    )
    graded = r["result"].notna() & r["settle_margin"].notna()
    won = np.where(r["pick_side"].eq("HOME"), r["settle_margin"] > 0, r["settle_margin"] < 0)
    exp_outcome = pd.Series(np.where(won, "won", "lost"), index=r.index)
    bad_outcome = graded & (r["settle_margin"] != 0) & (exp_outcome != r["outcome"])
    bad_settle = graded & ((r["result"] - r["decision_home_spread"] - r["settle_margin"]).abs() > 0)
    print(
        f"ledger rows {len(r)}; graded {int(graded.sum())}; "
        f"outcome mismatches {int(bad_outcome.sum())}; settle mismatches {int(bad_settle.sum())}"
    )
    rn = r[r["neutral_site"] == 1]
    print(
        rn[
            [
                "game_id",
                "arm",
                "pick_side",
                "decision_home_spread",
                "spread_line",
                "result",
                "settle_margin",
                "outcome",
            ]
        ].to_string(index=False)
    )
    keep = [
        c for c in n.columns if not c.startswith(("home_off", "home_def", "away_off", "away_def"))
    ]
    n[keep].to_csv(OUT / "neutral_games.csv", index=False)


main()
