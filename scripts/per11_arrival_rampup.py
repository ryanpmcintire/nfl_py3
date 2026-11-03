import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.transaction_flag_features import HIGH_SNAP_SHARE_THRESHOLD, default_snap_counts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "per11_arrival_rampup"
PICK = ROOT / "artifacts" / "pick_probability" / "20261101T171318Z" / "per_game.parquet"
PLACEBO_SHIFT = 3
GAMES = 3
SAMPLES = 4000
SEED = 20261103
STAT = ("effect", "lo", "hi", "p_pos", "se")


def team_games():
    latest = sorted(glob.glob(str(ROOT / "data/pbp/raw/*")))[-1]
    parts = []
    for f in glob.glob(latest + "/season=*/plays.parquet"):
        d = pd.read_parquet(
            f, columns=["game_id", "season_type", "week", "posteam", "defteam", "epa"]
        )
        parts.append(d.loc[d["season_type"].eq("REG") & d["epa"].notna() & d["posteam"].notna()])
    p = pd.concat(parts)
    p["season"] = p["game_id"].str[:4].astype(int)
    off = (
        p.groupby(["season", "week", "posteam"])["epa"]
        .mean()
        .rename("off")
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    de = (
        p.groupby(["season", "week", "defteam"])["epa"]
        .mean()
        .rename("dfn")
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    tg = off.merge(de, on=["season", "week", "team"]).sort_values(["team", "season", "week"])
    g = tg.groupby(["team", "season"])
    tg["off_base"] = g["off"].transform(lambda s: s.shift().expanding().mean())
    tg["dfn_base"] = g["dfn"].transform(lambda s: s.shift().expanding().mean())
    tg["off_d"] = tg["off"] - tg["off_base"]
    tg["dfn_d"] = tg["dfn_base"] - tg["dfn"]
    return tg.reset_index(drop=True)


def arrivals():
    sn = default_snap_counts()
    sn = sn.loc[sn["game_type"].eq("REG") & sn["season"].ge(2013)].sort_values(
        ["player", "season", "week"]
    )
    rows = []
    for (pl, se), g in sn.groupby(["player", "season"]):
        teams = g["team"].to_numpy()
        if len(set(teams)) < 2:
            continue
        for i in range(1, len(g)):
            if teams[i] != teams[i - 1]:
                prior = g.iloc[:i]
                prior = prior.loc[prior["team"].eq(teams[i - 1])]
                share = float(prior["snap_share"].mean())
                if share < HIGH_SNAP_SHARE_THRESHOLD:
                    break
                r = g.iloc[i]
                rows.append(
                    {
                        "player": pl,
                        "position": r["position"],
                        "season": int(se),
                        "from_team": teams[i - 1],
                        "team": teams[i],
                        "arrival_week": int(r["week"]),
                        "prior_share": share,
                        "new_off": float(r["offense_pct"]),
                        "new_def": float(r["defense_pct"]),
                    }
                )
                break
    return pd.DataFrame(rows)


def boot(x, rng):
    x = np.asarray(x, float)
    b = rng.choice(x, size=(SAMPLES, len(x))).mean(axis=1)
    return dict(
        zip(
            STAT,
            (
                float(x.mean()),
                float(np.percentile(b, 2.5)),
                float(np.percentile(b, 97.5)),
                float((b > 0).mean()),
                float(x.std(ddof=1) / np.sqrt(len(x))),
            ),
            strict=True,
        )
    )


def events(arr, tg):
    out = []
    key = tg.set_index(["team", "season"])
    for (team, season, wk), a in arr.groupby(["team", "season", "arrival_week"]):
        if (team, season) not in key.index:
            continue
        g = key.loc[(team, season)].reset_index().sort_values("week")
        g = g.loc[g["week"].ge(wk)].reset_index(drop=True)
        if len(g) < GAMES + PLACEBO_SHIFT:
            continue
        side = "off" if a["new_off"].max() >= a["new_def"].max() else "dfn"
        for k in range(GAMES):
            gm, pl = g.iloc[k], g.iloc[k + PLACEBO_SHIFT]
            if pd.isna(gm["off_d"]) or pd.isna(pl["off_d"]):
                continue
            out.append(
                {
                    "team": team,
                    "season": season,
                    "arrival_week": wk,
                    "k": k + 1,
                    "side": side,
                    "n_players": len(a),
                    "pos": ",".join(sorted(set(a["position"]))),
                    "week": int(gm["week"]),
                    "pweek": int(pl["week"]),
                    "off_d": gm["off_d"],
                    "dfn_d": gm["dfn_d"],
                    "p_off_d": pl["off_d"],
                    "p_dfn_d": pl["dfn_d"],
                }
            )
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tg = team_games()
    arr = arrivals()
    arr.to_csv(OUT / "arrivals.csv", index=False)
    ev = events(arr, tg)
    ev.to_csv(OUT / "events.csv", index=False)
    ev["own"] = np.where(ev["side"].eq("off"), ev["off_d"], ev["dfn_d"])
    ev["p_own"] = np.where(ev["side"].eq("off"), ev["p_off_d"], ev["p_dfn_d"])
    ev["d_own"] = ev["own"] - ev["p_own"]
    ev["x_off"] = ev["off_d"] - ev["p_off_d"]
    ev["x_dfn"] = ev["dfn_d"] - ev["p_dfn_d"]
    res = {
        "arrival_players": len(arr),
        "arrival_team_weeks": int(
            ev[["team", "season", "arrival_week"]].drop_duplicates().shape[0]
        ),
        "mean_off_d_all_team_games": float(tg["off_d"].mean()),
        "mean_dfn_d_all_team_games": float(tg["dfn_d"].mean()),
    }
    cells = {}

    def cell(name, d, col):
        d = d.dropna(subset=[col])
        m = d.groupby(["team", "season", "arrival_week"])[col].mean().rename("v").reset_index()
        c = boot(m["v"].to_numpy(), rng)
        c["n_events"] = len(m)
        c["parity_even"] = float(m.loc[m["season"] % 2 == 0, "v"].mean())
        c["parity_odd"] = float(m.loc[m["season"] % 2 == 1, "v"].mean())
        cells[name] = c

    for nm, sub in {"all_k123": ev, "pregame_k23": ev.loc[ev["k"].ge(2)]}.items():
        cell(f"own_side_vs_placebo_{nm}", sub, "d_own")
        cell(f"own_side_vs_asof_{nm}", sub, "own")
        cell(f"offense_arrivals_off_epa_vs_placebo_{nm}", sub.loc[sub["side"].eq("off")], "x_off")
        cell(f"defense_arrivals_def_epa_vs_placebo_{nm}", sub.loc[sub["side"].eq("dfn")], "x_dfn")
    res["cells"] = cells

    p = pd.read_parquet(PICK)
    p["season"] = p["season"].astype(int)
    p["week"] = p["week"].astype(int)
    rows = []
    for _, e in ev.iterrows():
        for wk, tag in ((e["week"], "arr"), (e["pweek"], "pl")):
            m = p.loc[
                (p["season"] == e["season"])
                & (p["week"] == wk)
                & ((p["home_team"] == e["team"]) | (p["away_team"] == e["team"]))
            ]
            if m.empty:
                continue
            r = m.iloc[0]
            home = r["home_team"] == e["team"]
            cover = (r["result"] - r["tue_open_home_spread"]) * (1 if home else -1)
            pick_team = bool(r["model_pick_home"] == 1) == home
            rows.append(
                {
                    "team": e["team"],
                    "season": e["season"],
                    "arrival_week": e["arrival_week"],
                    "k": e["k"],
                    "tag": tag,
                    "cover_open": cover,
                    "pick_on_team": pick_team,
                    "correct": r["model_correct"],
                    "decisive": cover != 0,
                }
            )
    o = pd.DataFrame(rows)
    o.to_csv(OUT / "opener_rows.csv", index=False)
    dec = o.loc[o["decisive"]]
    rec = {}
    for tag in ("arr", "pl"):
        d = dec.loc[dec["tag"] == tag]
        on = d.loc[d["pick_on_team"]]
        off = d.loc[~d["pick_on_team"]]
        rec[tag] = {
            "decisive_team_games": len(d),
            "team_covered": int((d["cover_open"] > 0).sum()),
            "model_correct": int(d["correct"].sum()),
            "picked_arrival_team_w_l": [
                int(on["correct"].sum()),
                int(len(on) - on["correct"].sum()),
            ],
            "picked_against_arrival_team_w_l": [
                int(off["correct"].sum()),
                int(len(off) - off["correct"].sum()),
            ],
        }
    res["decisive_record_2020_25"] = rec
    a = o.loc[o["tag"] == "arr"].groupby(["team", "season", "arrival_week"])["cover_open"].mean()
    q = o.loc[o["tag"] == "pl"].groupby(["team", "season", "arrival_week"])["cover_open"].mean()
    j = pd.concat([a.rename("a"), q.rename("p")], axis=1).dropna()
    res["opener_cover_margin_arrival_vs_zero"] = {**boot(a.to_numpy(), rng), "n_events": len(a)}
    res["opener_cover_margin_arrival_minus_placebo"] = {
        **boot((j["a"] - j["p"]).to_numpy(), rng),
        "n_events": len(j),
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
