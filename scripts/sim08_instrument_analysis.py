import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim08_terminal_fit as ttf

ROOT = Path(__file__).resolve().parents[1]
INSTR = ROOT / "artifacts" / "sim10" / "instrument"
ERAS = (
    ("2009-11", 2009, 2011),
    ("2012-16", 2012, 2016),
    ("2017-24", 2017, 2024),
    ("2025", 2025, 2025),
)
DIST_EDGES = (0.0, 40.0, 60.0, 80.0, 100.0)
DIST_LABELS = ("<=40", "40-60", "60-80", ">80")
M3_BUCKETS = ("<=-4", "-3..-1", "0", "1..3", ">=4")
HALF_SECS = 1800.0
MARGIN_PATH = 3
SCORE_CATS = ("td", "fg")


def era_of(season):
    for name, lo, hi in ERAS:
        if lo <= season <= hi:
            return name
    return "other"


def m3_bucket(x):
    if x <= -4:
        return M3_BUCKETS[0]
    if x < 0:
        return M3_BUCKETS[1]
    if x == 0:
        return M3_BUCKETS[2]
    if x <= 3:
        return M3_BUCKETS[3]
    return M3_BUCKETS[4]


def dist_label(yl):
    i = int(np.searchsorted(DIST_EDGES[1:], yl, side="left"))
    return DIST_LABELS[min(i, len(DIST_LABELS) - 1)]


def read_sim(label):
    rows = []
    for f in sorted((INSTR / label).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sim_frames(rows):
    games, ots, drives = [], [], []
    for g in rows:
        era = era_of(g["season"])
        reg = g["reg"]
        tied_end = g["end"][0] == g["end"][1]
        went_ot = g["oo"] is not None
        games.append(
            {
                "era": era,
                "m300": g["m300"],
                "m120": g["m120"],
                "reg": 0.0 if went_ot else reg,
                "ot": went_ot,
            }
        )
        for hf, pos, yl, secs, po, pdf, el, tr in g["dr"]:
            drives.append((era, int(hf), int(pos), yl, secs, po, pdf, el, tr))
        if went_ot:
            _recv, oyl, oel, o4, o5, _rule, _tot = g["oo"]
            odr = [d for d in g["dr"] if d[0] == 3]
            open_score = (o4 or o5) != 0
            poss = (
                [
                    {
                        "yl": oyl,
                        "score": bool(o4 > 0 or o5 > 0),
                        "el": oel,
                    }
                ]
                if open_score
                else []
            )
            poss += [{"yl": d[2], "score": bool(d[4] > 0 or d[5] > 0), "el": d[6]} for d in odr]
            ots.append(
                {
                    "era": era,
                    "n_poss": len(poss),
                    "tie": bool(tied_end),
                    "open_score": open_score,
                    "poss": poss,
                }
            )
    return (
        pd.DataFrame(games),
        ots,
        pd.DataFrame(drives, columns=["era", "hf", "pos", "yl", "secs", "po", "pd", "el", "trunc"]),
    )


def real_ot_drives():
    rows = []
    cols = [
        "game_id",
        "season",
        "season_type",
        "qtr",
        "game_seconds_remaining",
        "fixed_drive",
        "fixed_drive_result",
        "posteam",
        "play_type",
        "yardline_100",
    ]
    for s in ttf.SEASONS:
        path = (
            ttf.EARLY / f"pbp_{s}.parquet"
            if s <= 2017
            else ttf.LATE / f"season={s}" / "plays.parquet"
        )
        d = pd.read_parquet(path)
        if "season" not in d.columns:
            d["season"] = s
        d = d[(d["season_type"] == "REG") & (d["qtr"] >= 5)]
        d = d[d["posteam"].notna() & d["play_type"].isin(ttf.SCRIMMAGE) & d["fixed_drive"].notna()]
        d = d[[c for c in cols if c in d.columns]]
        for gid, g in d.groupby("game_id", sort=False):
            gb = g.groupby("fixed_drive", sort=True)
            first = gb.first()
            starts = first["game_seconds_remaining"].to_numpy(float)
            lasts = gb["game_seconds_remaining"].min().to_numpy(float)
            res = first["fixed_drive_result"].to_numpy()
            yl = first["yardline_100"].to_numpy(float)
            for i in range(len(first)):
                end = starts[i + 1] if i + 1 < len(first) else lasts[i]
                cat = ttf.RESULT_CAT.get(res[i], "other")
                rows.append((gid, int(s), i, yl[i], max(starts[i] - end, 0.0), cat))
    return pd.DataFrame(rows, columns=["game_id", "season", "k", "yl", "el", "cat"])


def real_frames():
    dr = ttf.all_drives()
    reg = dr[~dr["ot"]]
    gm = []
    for gid, g in reg.groupby("game_id", sort=False):
        season = int(g["season"].iloc[0])
        m300 = g[g["s"] <= 300.0]
        m120 = g[g["s"] <= 120.0]
        gm.append(
            {
                "game_id": gid,
                "era": era_of(season),
                "m300": float(m300["mh"].iloc[0]) if len(m300) else np.nan,
                "m120": float(m120["mh"].iloc[0]) if len(m120) else np.nan,
                "final": float(g["final_mh"].iloc[0]),
            }
        )
    games = pd.DataFrame(gm)
    ot_ids = set(dr.loc[dr["ot"], "game_id"])
    games["ot"] = games["game_id"].isin(ot_ids)
    games["reg"] = np.where(games["ot"], 0.0, games["final"])
    tie = dr.groupby("game_id")["final_mh"].first() == 0
    ot = real_ot_drives()
    ot["era"] = ot["season"].map(era_of)
    ot["tie"] = ot["game_id"].map(tie).fillna(False).astype(bool)
    h = reg.copy()
    h["era"] = h["season"].map(era_of)
    h["hf"] = np.where(h["s"] > HALF_SECS, 1, 2)
    return games, ot, h


def ot_possession_summary(sim_ots, real_ot):
    out = []
    for era, _, _ in ERAS:
        sg = [g for g in sim_ots if g["era"] == era]
        sp = [p for g in sg for p in g["poss"]]
        rg = real_ot[real_ot["era"] == era]
        rn = rg["game_id"].nunique()
        out.append(
            {
                "era": era,
                "sim_ot_games": len(sg),
                "real_ot_games": rn,
                "sim_poss_per_game": np.mean([g["n_poss"] for g in sg]) if sg else np.nan,
                "real_poss_per_game": len(rg) / rn if rn else np.nan,
                "sim_sec_per_poss": np.mean([p["el"] for p in sp]) if sp else np.nan,
                "real_sec_per_poss": rg["el"].mean() if rn else np.nan,
                "sim_open_score_share": np.mean([g["open_score"] for g in sg]) if sg else np.nan,
                "sim_tie": np.mean([g["tie"] for g in sg]) if sg else np.nan,
                "real_tie": rg.groupby("game_id")["tie"].first().mean() if rn else np.nan,
                "real_ties": int(rg.groupby("game_id")["tie"].first().sum()) if rn else 0,
            }
        )
    return pd.DataFrame(out)


def ot_score_by_distance(sim_ots, real_ot):
    rows = []
    for era, _, _ in ERAS:
        sp = [p for g in sim_ots if g["era"] == era for p in g["poss"]]
        rg = real_ot[real_ot["era"] == era]
        for lab in DIST_LABELS:
            ss = [p["score"] for p in sp if dist_label(p["yl"]) == lab]
            rr = rg[rg["yl"].map(dist_label) == lab]
            rr = rr["cat"].isin(SCORE_CATS)
            rows.append(
                {
                    "era": era,
                    "start": lab,
                    "sim_n": len(ss),
                    "sim_rate": np.mean(ss) if ss else np.nan,
                    "real_n": len(rr),
                    "real_rate": rr.mean() if len(rr) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def margin3_paths(sim_g, real_g):
    rows = []
    for era, _, _ in ERAS:
        for src, g in (("sim", sim_g), ("real", real_g)):
            ge = g[g["era"] == era]
            ge = ge[ge["m300"].notna()]
            if not len(ge):
                continue
            hit = ge[ge["reg"].abs() == MARGIN_PATH]
            sgn = np.sign(hit["reg"])
            row = {"era": era, "src": src, "games": len(ge), "m3_mass": len(hit) / len(ge)}
            b = (hit["m300"] * sgn).map(m3_bucket)
            for lab in M3_BUCKETS:
                row[lab] = (b == lab).mean() if len(hit) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def drives_per_half(sim_d, real_h, sim_n_games, real_n_games):
    rows = []
    for era, _, _ in ERAS:
        for hf in (1, 2):
            sd = sim_d[(sim_d["era"] == era) & (sim_d["hf"] == hf)]
            rd = real_h[(real_h["era"] == era) & (real_h["hf"] == hf)]
            sn, rn = sim_n_games.get(era, 0), real_n_games.get(era, 0)
            rows.append(
                {
                    "era": era,
                    "half": hf,
                    "sim_drives_per_team": len(sd) / (2 * sn) if sn else np.nan,
                    "real_drives_per_team": len(rd) / (2 * rn) if rn else np.nan,
                }
            )
    return pd.DataFrame(rows)


def analyse(sim_rows, real=None):
    sg, so, sd = sim_frames(sim_rows)
    if real is None:
        real = real_frames()
    rg, rot, rh = real
    out = {
        "ot_possessions": ot_possession_summary(so, rot),
        "ot_score_by_distance": ot_score_by_distance(so, rot),
        "margin3_paths": margin3_paths(sg, rg),
        "drives_per_half": drives_per_half(
            sd,
            rh,
            sg.groupby("era").size().to_dict(),
            rg.groupby("era").size().to_dict(),
        ),
    }
    return out


def synthetic_rows():
    base = {"season": 2013, "m300": 3.0, "m120": 3.0, "last": [[100.0, 3.0, 0.0]], "reg": 3.0}
    g1 = {
        **base,
        "id": "a",
        "dr": [[2, 0, 70.0, 400.0, 3.0, 0.0, 120.0, 0]],
        "oo": None,
        "end": [24.0, 21.0],
    }
    g2 = {
        **base,
        "id": "b",
        "reg": 0.0,
        "m300": 0.0,
        "dr": [[3, 0, 71.0, 600.0, 0.0, 0.0, 100.0, 0], [3, 1, 30.0, 500.0, 0.0, 3.0, 50.0, 0]],
        "oo": [0, 25.0, 5.0, 0.0, 0.0, "mod", 900.0],
        "end": [20.0, 23.0],
    }
    g3 = {
        **base,
        "id": "c",
        "reg": 0.0,
        "m300": 0.0,
        "dr": [[3, 0, 71.0, 600.0, 0.0, 0.0, 200.0, 0], [3, 1, 65.0, 400.0, 0.0, 0.0, 400.0, 1]],
        "oo": [0, 25.0, 5.0, 0.0, 0.0, "mod", 900.0],
        "end": [20.0, 20.0],
    }
    return [g1, g2, g3]


def synthetic_real():
    games = pd.DataFrame(
        {
            "game_id": ["r1", "r2"],
            "era": ["2012-16", "2012-16"],
            "m300": [3.0, 0.0],
            "m120": [3.0, 0.0],
            "final": [3.0, 0.0],
            "ot": [False, True],
            "reg": [3.0, 0.0],
        }
    )
    ot = pd.DataFrame(
        {
            "game_id": ["r2", "r2"],
            "season": [2013, 2013],
            "k": [0, 1],
            "yl": [71.0, 30.0],
            "el": [150.0, 60.0],
            "cat": ["punt", "fg"],
            "era": ["2012-16", "2012-16"],
            "tie": [False, False],
        }
    )
    h = pd.DataFrame({"era": ["2012-16"], "hf": [2]})
    return games, ot, h


def selftest():
    res = analyse(synthetic_rows(), synthetic_real())
    op = res["ot_possessions"].set_index("era").loc["2012-16"]
    assert op["sim_ot_games"] == 2 and op["real_ot_games"] == 1
    assert op["sim_poss_per_game"] == 2.0 and op["real_poss_per_game"] == 2.0
    assert abs(op["sim_sec_per_poss"] - 187.5) < 1e-9
    assert op["sim_tie"] == 0.5 and op["real_tie"] == 0.0
    sc = res["ot_score_by_distance"].set_index(["era", "start"]).loc[("2012-16", "<=40")]
    assert sc["sim_n"] == 1 and sc["real_rate"] == 1.0
    m3 = res["margin3_paths"].set_index(["era", "src"])
    assert abs(m3.loc[("2012-16", "sim"), "m3_mass"] - 1 / 3) < 1e-9
    assert m3.loc[("2012-16", "sim"), "1..3"] == 1.0
    return res


def cmd_run(a):
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    rows = []
    for lab in a.labels:
        rows += read_sim(lab)
    print(f"sim games {len(rows)} from labels {a.labels}")
    for k, v in analyse(rows).items():
        print(f"\n{k}\n{v.to_string(index=False)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("labels", nargs="+")
    sub.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "selftest":
        selftest()
        print("selftest ok (synthetic, in memory only)")
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
