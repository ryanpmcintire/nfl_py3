import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_injury_language as L

LADDERS = {
    "designation": ["designation"],
    "designation_x_practice": ["designation", "practice_status"],
}
SNAPS = L.ROOT / "data" / "players" / "forward_snaps"
INJ = L.ROOT / "data" / "raw" / "nflverse_injuries"
OUT = L.ART / "forward_score"
AGG = {"n": ("played", "size"), "k": ("played", "sum"), "m": ("snap_share", "mean")}
METRICS = ("log_loss_improvement", "brier_improvement", "mae_improvement")
PRIOR_COLS = ["week", "designation", "practice_status", "played", "snap_share"]
SHORT_METRIC = {
    "log_loss_improvement": "ll",
    "brier_improvement": "brier",
    "mae_improvement": "mae",
}
SHORT_LADDER = {"designation": "des", "designation_x_practice": "desprac"}
METRIC_TEXT = {
    "log_loss_improvement": "log loss on played or not",
    "brier_improvement": "Brier score on played or not",
    "mae_improvement": "mean absolute error of snap share",
}
METRIC_PLAIN = {
    "log_loss_improvement": "who would play",
    "brier_improvement": "who would play",
    "mae_improvement": "how much each player would play",
}
LADDER_TEXT = {
    "designation": "designation ladder",
    "designation_x_practice": "designation x practice ladder",
}
LADDER_PLAIN = {
    "designation": "the official injury tag",
    "designation_x_practice": "the official tag plus practice participation",
}


def stamp(s):
    return pd.Timestamp(datetime.strptime(s, L.STAMP).replace(tzinfo=UTC))


def snap_frame(week):
    f = SNAPS / f"snap_counts_{L.SEASON}_before_week_{week + 1:02d}.parquet"
    if not f.exists():
        raise SystemExit(f"missing snap capture for week {week}: {f}")
    s = pd.read_parquet(f)
    return s[s["season"] == L.SEASON].copy(), f


def outcomes(x, snaps, lo, hi):
    r, pmap = L.roster_map()
    teamweeks = set(map(tuple, snaps[["team", "week"]].drop_duplicates().to_numpy()))
    s = snaps[snaps["week"].between(lo, hi)].copy()
    s["total"] = s[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    s["share"] = s[["offense_pct", "defense_pct"]].fillna(0).max(axis=1)
    s = s.groupby(["pfr_player_id", "week"], as_index=False).agg(
        total=("total", "sum"), share=("share", "max"), st=("st_snaps", "sum")
    )
    rw = r.drop_duplicates(["gsis_id", "week"], keep="last").set_index(["gsis_id", "week"])[
        "status"
    ]
    x = x.copy()
    x["pfr"] = x["gsis_id"].map(pmap)
    x = x.merge(s, left_on=["pfr", "week"], right_on=["pfr_player_id", "week"], how="left")
    x["rost"] = [rw.get((g, w)) for g, w in zip(x["gsis_id"], x["week"], strict=True)]
    cat = []
    for a in x.itertuples():
        if (a.team, a.week) not in teamweeks:
            cat.append("unverified_team_week_not_in_snaps")
        elif pd.notna(a.total) and a.total > 0:
            cat.append("played")
        elif pd.notna(a.total):
            cat.append("in_snaps_zero_snaps")
        elif a.designation == "Out":
            cat.append("out_absent")
        elif pd.isna(a.pfr):
            cat.append("unverified_no_pfr_id")
        elif pd.notna(a.rost):
            cat.append("absent_rostered_" + str(a.rost))
        else:
            cat.append("unverified_absent_not_on_roster_week")
    x["category"] = cat
    x["verified"] = ~x["category"].str.startswith("unverified")
    x["played"] = (x["category"] == "played").astype(int)
    x["snap_share"] = np.where(x["played"] == 1, x["share"].fillna(0.0), 0.0)
    return x


def load_forward(week):
    fz = json.loads((L.ART / "frozen.json").read_text())
    g = pd.read_parquet(
        L.ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "week", "home_team", "away_team", "kickoff"],
    )
    g = g[(g["season"] == L.SEASON) & (g["week"] == week)]
    sched = {}
    for a in g.itertuples():
        sched[a.home_team] = a.kickoff
        sched[a.away_team] = a.kickoff
    files, rows, excluded = [], [], []
    seen = set()
    for f in sorted(L.FORWARD.glob(f"{L.SEASON}_week{week:02d}_*.json")):
        j = json.loads(f.read_text())
        rec = stamp(j["recorded_utc"])
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        enrolled = j["prompt_sha256"] == fz["prompt_sha256"] and j["model"] == fz["model"]
        inj = pd.read_parquet(INJ / j["capture_utc"] / "injuries.parquet")
        inj = inj[(inj["season"] == L.SEASON) & (inj["week"] == week)]
        inj = inj.drop_duplicates(["gsis_id", "week"], keep="last").set_index("gsis_id")
        kept = 0
        for o in j["rows"]:
            kick = stamp(o["kickoff_utc"])
            actual = sched.get(o["team"])
            why = None
            if not enrolled:
                why = "prompt_or_model_differs_from_enrollment"
            elif not rec < kick:
                why = "recorded_at_or_after_row_kickoff"
            elif actual is None or not rec < actual:
                why = "recorded_at_or_after_schedule_kickoff"
            elif not mtime < min(kick, actual):
                why = "file_modified_at_or_after_kickoff"
            elif o["gsis_id"] not in inj.index:
                why = "not_in_capture_snapshot"
            elif o["gsis_id"] in seen:
                why = "duplicate_player_in_week"
            if why:
                excluded.append({"file": f.name, "player": o["player"], "reason": why})
                continue
            seen.add(o["gsis_id"])
            p = inj.loc[o["gsis_id"]]
            ps = p["practice_status"]
            snap_des = p["report_status"] if pd.notna(p["report_status"]) else "none"
            rows.append(
                {
                    "gsis_id": o["gsis_id"],
                    "player": o["player"],
                    "team": o["team"],
                    "game_id": o["game_id"],
                    "week": week,
                    "view": j["view"],
                    "designation": o["designation"],
                    "snapshot_designation": snap_des,
                    "practice_status": ps if pd.notna(ps) else "none",
                    "position": p["position"],
                    "rp_raw": float(o["p_plays"]),
                    "rs_raw": float(o["snap_share"]),
                    "minutes_before_kickoff": float((min(kick, actual) - rec).total_seconds() / 60),
                }
            )
            kept += 1
        files.append(
            {
                "file": f.name,
                "capture_utc": j["capture_utc"],
                "recorded_utc": j["recorded_utc"],
                "modified_utc": mtime.strftime(L.STAMP),
                "rows": len(j["rows"]),
                "kept": kept,
            }
        )
    return pd.DataFrame(rows), files, excluded


def historical():
    pop = L.id_outcomes(pd.read_parquet(L.ART / "population.parquet"))
    s0 = pd.read_parquet(L.ART / "sample.parquet")[["rid", "gsis_id", "week"]]
    ver = pop[pop["verified"]].copy()
    ver["practice_status"] = ver["practice_status"].fillna("none")
    s = s0.merge(ver.drop(columns=["rid"], errors="ignore"), on=["gsis_id", "week"], how="inner")
    reads = L.parse_reads()
    s = s[s["rid"].isin(reads)].copy()
    clip = 1.0 / (len(s) + 2)
    s["rp"] = s["rid"].map(lambda r: min(max(reads[r][0], clip), 1 - clip))
    s["rs"] = s["rid"].map(lambda r: min(max(reads[r][1], 0.0), 1.0))
    popn = ver[ver["week"] > L.FIRST_WEEK].groupby("designation").size()
    samn = s.groupby("designation").size()
    s["w"] = s["designation"].map(lambda d: popn[d] / samn[d])
    for nm, keys in LADDERS.items():
        tab = L.ladder_tables(ver, keys)
        pr = [L.ladder_pred(r, tab, keys) for _, r in s.iterrows()]
        s[f"lp_{nm}"] = [min(max(p[0], clip), 1 - clip) for p in pr]
        s[f"ls_{nm}"] = [p[1] for p in pr]
    return pop, ver, s, clip


def score_cells(s, subsets):
    y = s["played"].to_numpy()
    sh = s["snap_share"].to_numpy()
    w = s["w"].to_numpy()
    cl = (s["team"] + s["week"].astype(str)).to_numpy()
    rp = s["rp"].to_numpy()
    rs = s["rs"].to_numpy()

    def ll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def br(p):
        return (p - y) ** 2

    out = []
    for nm in LADDERS:
        lp = s[f"lp_{nm}"].to_numpy()
        ls = s[f"ls_{nm}"].to_numpy()
        rng = np.random.default_rng(L.seed_value())
        deltas = (ll(lp) - ll(rp), br(lp) - br(rp), np.abs(ls - sh) - np.abs(rs - sh))
        for sub, mask in subsets:
            if mask.sum() < 2:
                continue
            for metric, d in zip(METRICS, deltas, strict=True):
                r = L.summarize(metric, d[mask], w[mask], cl[mask], rng)
                r.update(
                    baseline=nm,
                    subset=sub,
                    rows=int(mask.sum()),
                    clusters=len(set(cl[mask])),
                )
                out.append(r)
    return out


def strata_subsets(s, headline):
    n = len(s)
    d = s["designation"].to_numpy()
    subsets = [("all", np.ones(n, bool))]
    if headline:
        subsets.append(("headline_rows", s["has_headline"].to_numpy()))
    return subsets + [(st, d == st) for st in L.STRATA]


def forward_frames(week):
    out = []
    for w in range(L.LAST_WEEK + 1, week):
        f = OUT / f"{L.SEASON}_week{w:02d}_scored.parquet"
        if not f.exists():
            raise SystemExit(f"score week {w} before week {week}: missing {f}")
        out.append(pd.read_parquet(f))
    return out


def game_gain(fv, nm):
    y = fv["played"].to_numpy()

    def nll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = pd.Series(nll(fv[f"lp_{nm}"].to_numpy()) - nll(fv["rp"].to_numpy()), index=fv.index)
    g = d.groupby(fv["game_id"]).sum().sort_values(ascending=False)
    return {str(k): round(float(v), 3) for k, v in g.items()}


def keep_split(c):
    if c["subset"] == "all":
        return True
    return c["metric"] == "log_loss_improvement" and c["subset"] != "headline_rows"


def keep_pooled(c):
    return c["subset"] == "all" and c["metric"] == "log_loss_improvement"


def batch_cells(cells, prefix, scope, keep, note):
    out = []
    for c in cells:
        if not keep(c):
            continue
        better = "better" if c["estimate"] > 0 else "worse"
        out.append(
            {
                "name": f"{prefix}_{SHORT_METRIC[c['metric']]}_{SHORT_LADDER[c['baseline']]}_"
                f"{c['subset'].lower()}",
                "description": f"{scope}: LLM wording read vs {LADDER_TEXT[c['baseline']]} fit on "
                f"prior weeks, {METRIC_TEXT[c['metric']]}, subset {c['subset']}, id-joined "
                "verified outcomes",
                "effect": round(c["estimate"], 4),
                "effect_units": c["metric"],
                "interval_low": round(c["low"], 4),
                "interval_high": round(c["high"], 4),
                "probability_positive": round(c["probability_positive"], 4),
                "sample_blocks": c["clusters"],
                "notes": f"{c['rows']} rows in {c['clusters']} team-week clusters; {note}",
                "plain_summary": f"Reading the wording of injury reports predicted "
                f"{METRIC_PLAIN[c['metric']]} {better} than {LADDER_PLAIN[c['baseline']]} in "
                "this sample. Too few weeks to settle it, so the question stays open.",
            }
        )
    return out


def batch(rep, week):
    first = L.LAST_WEEK + 1
    total = sum(v for k, v in rep["looks"].items() if k != "family")
    cum = rep["forward_cumulative"]
    note = (
        f"scored {datetime.now(UTC).date()} by scripts/lead106_forward_score.py --week {week}; "
        f"reads scored as frozen, no model calls; ladders expand over every prior week "
        f"({rep['ladder_prior_rows']} rows); one of {total} looks this run, family lead106"
    )
    cells = batch_cells(
        rep["forward"]["cells"],
        f"lead106_forward_wk{week}",
        f"Forward 2026 Week {week} census",
        keep_split,
        note,
    )
    if week > first:
        cells += batch_cells(
            cum["cells"],
            f"lead106_forward_wk{first}_{week}",
            f"Forward 2026 Weeks {first}-{week} census, cumulative",
            keep_split,
            note,
        )
    cells += batch_cells(
        rep["pooled_weeks_2_to_forward"]["cells"],
        f"lead106_pooled_wk2_{week}",
        f"Pooled 2026 weeks 2-8 stratified sample (weighted to population) plus forward Weeks "
        f"{first}-{week} census",
        keep_pooled,
        note,
    )
    return {
        "family": "lead106",
        "league": "nfl",
        "season_start": L.SEASON,
        "season_end": L.SEASON,
        "classification": "unresolved_below_power",
        "classification_evidence": (
            f"{cum['rows']} id-verified forward rows in {cum['clusters']} team-week clusters over "
            f"{len(cum['weeks'])} forward week(s); {total} looks this run in family lead106; no "
            "split-half reliability on forward weeks and no positive control; week, cumulative and "
            "pooled cells overlap and are not independent votes; sub-cells with the whole "
            "interval below zero are listed in the lane for adjudication, not closed here"
        ),
        "source": f"artifacts/lead106_injury_language/forward_score/{L.SEASON}_week{week:02d}.json",
        "category": "health",
        "cells": cells,
    }


def line(c):
    return (
        f"{c['subset']:<14}{c['baseline']:<24}{c['metric']:<22}rows {c['rows']:>4} "
        f"clusters {c['clusters']:>4}  {c['estimate']:+.4f} "
        f"[{c['low']:+.4f}, {c['high']:+.4f}] P+ {c['probability_positive']:.4f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--week", type=int, required=True)
    a = p.parse_args()
    week = a.week
    snaps, snap_file = snap_frame(week)
    if not (snaps["week"] == week).any():
        raise SystemExit(f"snap capture {snap_file} holds no week {week} rows")
    pop, ver, hs, hclip = historical()
    prev, _ = snap_frame(L.LAST_WEEK)
    base = pd.read_parquet(L.ART / "population.parquet").drop(
        columns=["total", "share", "matched", "played", "snap_share"]
    )
    mine = outcomes(base, prev, L.FIRST_WEEK, L.LAST_WEEK)
    parity = bool((mine["category"].to_numpy() == pop["category"].to_numpy()).all())
    fw, files, excluded = load_forward(week)
    if fw.empty:
        raise SystemExit(f"no admissible forward reads for week {week}")
    fo = outcomes(fw, snaps, week, week)
    fv = fo[fo["verified"]].copy()
    if len(fv) < 2:
        raise SystemExit(f"no verified forward rows for week {week}")
    fclip = 1.0 / (len(fv) + 2)
    fv["rp"] = fv["rp_raw"].clip(fclip, 1 - fclip)
    fv["rs"] = fv["rs_raw"].clip(0.0, 1.0)
    fv["w"] = 1.0
    filed = fo.groupby("game_id")["designation"].agg(lambda v: bool((v != "none").any()))
    filed = filed.astype(bool)
    fv["status_filed"] = fv["game_id"].map(filed).astype(bool)
    fsubsets = strata_subsets(fv, False)
    fsubsets.append(("status_filed_games", fv["status_filed"].to_numpy()))
    earlier = forward_frames(week)
    prior = pd.concat(
        [ver[ver["week"] < week][PRIOR_COLS], *[f[PRIOR_COLS] for f in earlier]],
        ignore_index=True,
    )
    for nm, keys in LADDERS.items():
        tab = {week: (prior.groupby(keys).agg(**AGG), prior.groupby("designation").agg(**AGG))}
        pr = [L.ladder_pred(r, tab, keys) for _, r in fv.iterrows()]
        fv[f"lp_{nm}"] = [min(max(q[0], fclip), 1 - fclip) for q in pr]
        fv[f"ls_{nm}"] = [q[1] for q in pr]
    cols = ["team", "week", "designation", "played", "snap_share", "rp", "rs", "w"]
    cols += [f"{k}_{nm}" for nm in LADDERS for k in ("lp", "ls")]
    cum = pd.concat([*earlier, fv], ignore_index=True)
    cum["status_filed"] = cum["status_filed"].astype(bool)
    csubsets = strata_subsets(cum, False)
    csubsets.append(("status_filed_games", cum["status_filed"].to_numpy()))
    both = pd.concat([hs[cols], cum[cols]], ignore_index=True)
    stored = json.loads((L.ART / "score_ids.json").read_text())["cells"]
    rep = {
        "season": L.SEASON,
        "week": week,
        "snap_file": str(snap_file),
        "files": files,
        "excluded": excluded,
        "designation_differs_from_snapshot": int(
            (fw["designation"] != fw["snapshot_designation"]).sum()
        ),
        "outcome_rule_parity_weeks_1_8": parity,
        "least_minutes_before_kickoff": float(fw["minutes_before_kickoff"].min()),
        "forward": {
            "rows_read": len(fo),
            "rows_verified": len(fv),
            "games": int(fv["game_id"].nunique()),
            "clip": fclip,
            "verified_rate_by_designation": fo.groupby("designation")["verified"].mean().to_dict(),
            "category_by_designation": {
                d: g["category"].value_counts().to_dict() for d, g in fo.groupby("designation")
            },
            "by_designation": {
                d: {
                    "rows": len(g),
                    "played_rate": float(g["played"].mean()),
                    "mean_read_p": float(g["rp"].mean()),
                    "mean_designation_ladder_p": float(g["lp_designation"].mean()),
                    "mean_practice_ladder_p": float(g["lp_designation_x_practice"].mean()),
                    "mean_realised_share": float(g["snap_share"].mean()),
                    "mean_read_share": float(g["rs"].mean()),
                    "mean_designation_ladder_share": float(g["ls_designation"].mean()),
                }
                for d, g in fv.groupby("designation")
            },
            "games_without_filed_status": sorted(filed.index[~filed]),
            "log_loss_gain_by_game_vs_designation": game_gain(fv, "designation"),
            "log_loss_gain_by_game_vs_practice": game_gain(fv, "designation_x_practice"),
            "cells": score_cells(fv, fsubsets),
        },
        "ladder_prior_weeks": sorted(int(w) for w in prior["week"].unique()),
        "ladder_prior_rows": len(prior),
        "forward_cumulative": {
            "weeks": sorted(int(w) for w in cum["week"].unique()),
            "rows": len(cum),
            "clusters": int((cum["team"] + cum["week"].astype(str)).nunique()),
            "games": int(cum["game_id"].nunique()),
            "cells": score_cells(cum, csubsets),
        },
        "historical_weeks_2_8": {
            "rows": len(hs),
            "clip": hclip,
            "cells": score_cells(hs, strata_subsets(hs, True)),
            "stored_cells": [c for c in stored if c.get("subset") == "all"],
        },
        "pooled_weeks_2_to_forward": {
            "rows": len(both),
            "cells": score_cells(both, strata_subsets(both, False)),
        },
    }
    rep["looks"] = {
        "family": "lead106",
        "forward": len(rep["forward"]["cells"]),
        "forward_cumulative": len(rep["forward_cumulative"]["cells"]) if earlier else 0,
        "pooled": len(rep["pooled_weeks_2_to_forward"]["cells"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fv.to_parquet(OUT / f"{L.SEASON}_week{week:02d}_scored.parquet")
    (OUT / f"{L.SEASON}_week{week:02d}.json").write_text(json.dumps(rep, indent=1))
    reg = batch(rep, week)
    (OUT / f"{L.SEASON}_week{week:02d}_registry_batch.json").write_text(json.dumps(reg, indent=1))
    skip = ("forward", "historical_weeks_2_8", "pooled_weeks_2_to_forward", "forward_cumulative")
    head = {k: v for k, v in rep.items() if k not in skip}
    print(json.dumps(head, indent=1))
    fwd = {k: v for k, v in rep["forward"].items() if k != "cells"}
    print(json.dumps(fwd, indent=1))
    print("FORWARD")
    for c in rep["forward"]["cells"]:
        print(line(c))
    cm = rep["forward_cumulative"]
    print("CUMULATIVE", cm["weeks"], cm["rows"], cm["clusters"], cm["games"])
    for c in cm["cells"]:
        print(line(c))
    print("BATCH CELLS", len(reg["cells"]))
    print("HISTORICAL recomputed", len(hs))
    for c in rep["historical_weeks_2_8"]["cells"]:
        if c["subset"] == "all":
            print(line(c))
    print("HISTORICAL stored")
    for c in rep["historical_weeks_2_8"]["stored_cells"]:
        print(
            f"{c['baseline']:<24}{c['metric']:<22}{c['estimate']:+.4f} "
            f"[{c['low']:+.4f}, {c['high']:+.4f}] P+ {c['probability_positive']:.4f}"
        )
    print("POOLED", len(both))
    for c in rep["pooled_weeks_2_to_forward"]["cells"]:
        print(line(c))


if __name__ == "__main__":
    main()
