import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import features as F
from nfl_ats.constants import STATE_METRICS, TEAM_ABBREVIATION_ALIASES

HARNESS_PATH = Path(__file__).resolve().parent / "mod24_u1.py"
HARNESS = {"__file__": str(HARNESS_PATH), "__name__": "mod24_u1_harness"}
exec(compile(HARNESS_PATH.read_text().rsplit("\nmain()", 1)[0], str(HARNESS_PATH), "exec"), HARNESS)

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod23_unit7"
UNIT6B = ROOT / "artifacts" / "mod23_unit6b"
UNIT6 = ROOT / "artifacts" / "mod23_unit6"
BASE5 = ROOT / "artifacts" / "mod24_u5" / "base_per_game.parquet"
SNAP_GLOB = str(ROOT / "data" / "players" / "raw" / "*" / "snap_counts.parquet")
ALIAS = TEAM_ABBREVIATION_ALIASES
FIRST_SNAP_SEASON = 2013
SIDES = ("home", "away", "diff")
STATE_COLS = [f"{s}_{m}" for m in STATE_METRICS for s in SIDES]
FULL = HARNESS["FULL"]
BASE_ALPHA = HARNESS["BASE_ALPHA"]
BOOT = 20000
SEED = 20261016
LEAK_SEASON = 2022
LEAK_WEEK = 10
CTX = {"mode": "raw", "flags": None, "coef": None, "fit_seasons": {}, "zero_after": None}


OL_POS = {"T", "G", "C", "OL", "OT", "OG", "C/G", "G/T", "G/C", "T/G", "G/OT"}
SKILL_POS = {"WR", "TE", "WR/R"}
RB_POS = {"RB", "HB"}
GROUP_SIZE = {"ol": 5, "sk": 3, "df": 11, "rb": 2}
GROUP_METRICS = {
    "ol": ("off_epa_per_play", "off_pass_epa_per_play", "off_rush_epa_per_play", "off_yards_per_play", "off_sack_rate", "point_diff", "ats_residual"),
    "sk": ("off_epa_per_play", "off_pass_epa_per_play", "off_cpoe", "off_yards_per_play", "point_diff", "ats_residual"),
    "df": ("def_epa_per_play", "def_pass_epa_per_play", "def_rush_epa_per_play", "def_yards_per_play", "def_takeaway_rate", "def_sack_rate", "point_diff", "ats_residual"),
    "rb": ("off_rush_epa_per_play", "off_yards_per_play", "ats_residual"),
}
GROUPS = tuple(GROUP_METRICS)
MODE_OWN = {"B": "flag", "B2": "dose", "B3": "flag", "B5": "flag"}
MODE_GROUPS = {"B": (), "B2": (), "B3": ("ol", "sk", "df"), "B5": GROUPS}
ARMS = ("B3", "B5")


def group_missing(src, src_plays, game, game_plays, pool, k):
    if not src or src_plays <= 0 or game_plays <= 0:
        return 0.0
    cands = [p for p in src if p in pool]
    top = sorted(cands, key=lambda p: -src[p])[:k]
    exp = np.array([src[p] / src_plays * game_plays for p in top])
    act = np.array([game.get(p, 0.0) for p in top])
    tot = exp.sum()
    return float(np.maximum(exp - act, 0.0).sum() / tot) if tot > 0 else 0.0


def team_game_flags():
    path = sorted(glob.glob(SNAP_GLOB))[-1]
    snaps = pd.read_parquet(path)
    snaps = snaps.loc[snaps["game_type"].eq("REG")].copy()
    snaps["team"] = snaps["team"].replace(ALIAS)
    snaps["offense_snaps"] = pd.to_numeric(snaps["offense_snaps"], errors="coerce").fillna(0.0)
    snaps["defense_snaps"] = pd.to_numeric(snaps["defense_snaps"], errors="coerce").fillna(0.0)
    rows = []
    for (gid, team), g in snaps.groupby(["game_id", "team"]):
        q = g.loc[g["position"].eq("QB")]
        if q.empty:
            continue
        top = q.sort_values("offense_snaps", ascending=False).iloc[0]
        pid = g["pfr_player_id"].astype(object)
        pos = g["position"].astype(object)
        rows.append(
            {
                "game_id": gid,
                "team": team,
                "season": int(g["season"].iloc[0]),
                "week": int(g["week"].iloc[0]),
                "qb1": top["pfr_player_id"],
                "off": dict(zip(pid, g["offense_snaps"], strict=True)),
                "dfn": dict(zip(pid, g["defense_snaps"], strict=True)),
                "pos": dict(zip(pid, pos, strict=True)),
                "qb_snaps": dict(zip(q["pfr_player_id"], q["offense_snaps"], strict=True)),
            }
        )
    tg = pd.DataFrame(rows).sort_values(["team", "season", "week"]).reset_index(drop=True)
    n = len(tg)
    out = {k: np.zeros(n) for k in ("flag", "dose", "ol", "sk", "df", "rb")}
    for _, idx in tg.groupby("team").groups.items():
        cur_season = None
        last_qb = None
        last_prev = None
        qcum, ocum, dcum = {}, {}, {}
        oplays = dplays = 0.0
        prev = None
        posmap = {}
        for i in idx:
            r = tg.loc[i]
            if r["season"] != cur_season:
                if cur_season is not None and ocum:
                    prev = (dict(ocum), oplays, dict(dcum), dplays)
                last_prev = last_qb
                cur_season = r["season"]
                qcum, ocum, dcum = {}, {}, {}
                oplays = dplays = 0.0
            starter = max(qcum, key=qcum.get) if qcum else last_prev
            game_off_plays = max(r["off"].values()) if r["off"] else 0.0
            game_def_plays = max(r["dfn"].values()) if r["dfn"] else 0.0
            if starter is not None:
                out["flag"][i] = float(r["qb1"] != starter)
                if game_off_plays > 0:
                    out["dose"][i] = 1.0 - min(1.0, r["qb_snaps"].get(starter, 0.0) / game_off_plays)
            if ocum:
                so, sop, sd, sdp = ocum, oplays, dcum, dplays
            elif prev is not None:
                so, sop, sd, sdp = prev
            else:
                so, sop, sd, sdp = {}, 0.0, {}, 0.0
            ol_pool = {p for p, ps in posmap.items() if ps in OL_POS}
            sk_pool = {p for p, ps in posmap.items() if ps in SKILL_POS}
            df_pool = set(sd)
            out["ol"][i] = group_missing(so, sop, r["off"], game_off_plays, ol_pool, GROUP_SIZE["ol"])
            out["sk"][i] = group_missing(so, sop, r["off"], game_off_plays, sk_pool, GROUP_SIZE["sk"])
            rb_pool = {p for p, ps in posmap.items() if ps in RB_POS}
            out["rb"][i] = group_missing(so, sop, r["off"], game_off_plays, rb_pool, GROUP_SIZE["rb"])
            out["df"][i] = group_missing(sd, sdp, r["dfn"], game_def_plays, df_pool, GROUP_SIZE["df"])
            for p, v in r["off"].items():
                ocum[p] = ocum.get(p, 0.0) + v
            for p, v in r["dfn"].items():
                dcum[p] = dcum.get(p, 0.0) + v
            for p, v in r["qb_snaps"].items():
                qcum[p] = qcum.get(p, 0.0) + v
            posmap.update(r["pos"])
            oplays += game_off_plays
            dplays += game_def_plays
            last_qb = r["qb1"]
    for k, v in out.items():
        tg[k] = v
    return tg[["game_id", "team", "season", "week", "qb1", "flag", "dose", "ol", "sk", "df", "rb"]], path


def regressors(mode, m):
    return [g for g in MODE_GROUPS[mode] if m in GROUP_METRICS[g]]


def fit_coefficients(frame, mode):
    coef = {}
    metrics = list(STATE_METRICS)
    seasons = sorted(frame["season"].unique())
    for s in seasons:
        coef[int(s)] = {m: {} for m in metrics}
        fit = frame.loc[frame["season"].ge(FIRST_SNAP_SEASON) & frame["season"].lt(s)]
        CTX["fit_seasons"][int(s)] = int(fit["season"].max()) if len(fit) else None
        if fit.empty or fit["own"].sum() == 0:
            continue
        dm = {c: fit[c] - fit.groupby("season")[c].transform("mean") for c in ["own", *GROUPS]}
        for m in metrics:
            y = pd.to_numeric(fit[m], errors="coerce")
            ok = y.notna()
            y = y[ok] - y[ok].groupby(fit.loc[ok, "season"]).transform("mean")
            names = ["own", *regressors(mode, m)]
            X = np.column_stack([dm[c][ok].to_numpy() for c in names])
            beta = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
            coef[int(s)][m] = dict(zip(names, (float(b) for b in beta), strict=True))
    return coef


def wrapped(orig):
    def fn(schedules, team_stats, game_types=("REG",)):
        out = orig(schedules, team_stats, game_types)
        if CTX["mode"] == "raw":
            return out
        mode = CTX["mode"]
        fl = CTX["flags"].copy()
        if CTX["zero_after"] is not None:
            zs, zw = CTX["zero_after"]
            late = fl["season"].gt(zs) | (fl["season"].eq(zs) & fl["week"].ge(zw))
            fl.loc[late, ["flag", "dose", *GROUPS]] = 0.0
        fl["own"] = fl[MODE_OWN[mode]]
        o = out.merge(fl[["game_id", "team", "own", *GROUPS]], on=["game_id", "team"], how="left")
        o[["own", *GROUPS]] = o[["own", *GROUPS]].fillna(0.0)
        coef = fit_coefficients(o, mode)
        CTX["coef"] = coef
        for s, table in coef.items():
            m = o["season"].eq(s).to_numpy()
            for metric, betas in table.items():
                adj = 0.0
                for name, b in betas.items():
                    adj = adj + b * o.loc[m, name]
                o.loc[m, metric] = pd.to_numeric(o.loc[m, metric], errors="coerce") - adj
        if len(o) != len(out):
            raise RuntimeError("row mismatch")
        return o[list(out.columns)].sort_values(["gameday", "game_id", "team"])

    return fn


def build_all():
    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(u3.SNAPSHOT / "team_stats.parquet")
    flags, snap_path = team_game_flags()
    CTX["flags"] = flags
    print("snap file", snap_path, "team-games", len(flags), flush=True)
    print(flags.groupby("season")[["flag", "dose", "ol", "sk", "df", "rb"]].mean().round(4).to_dict(), flush=True)
    orig = F.build_team_game_metrics
    F.build_team_game_metrics = wrapped(orig)
    res = {}
    coefs = {}
    for mode in ("raw", *ARMS):
        CTX["mode"] = mode
        g = F._build_features_pass(schedules, team_stats, ("REG",))
        res[mode] = g.set_index("game_id")
        coefs[mode] = CTX["coef"]
        if mode != "raw":
            for s, fit_max in CTX["fit_seasons"].items():
                if fit_max is not None and fit_max >= s:
                    raise SystemExit(f"coefficient leak: season {s} fit through {fit_max}")
        print("built", mode, len(g), flush=True)
    CTX["mode"] = "B5"
    CTX["zero_after"] = (LEAK_SEASON, LEAK_WEEK)
    g = F._build_features_pass(schedules, team_stats, ("REG",)).set_index("game_id")
    CTX["zero_after"] = None
    F.build_team_game_metrics = orig
    feats = pd.read_parquet(u3.FEATURES).set_index("game_id")
    cols = [c for c in STATE_COLS if c in feats.columns]
    common = feats.index.intersection(res["raw"].index)
    d = (res["raw"].loc[common, cols] - feats.loc[common, cols]).abs()
    print("raw repro max abs diff", float(d.max().max()), "common", len(common), "of", len(feats), flush=True)
    keep = res["B5"].index[
        res["B5"]["season"].lt(LEAK_SEASON) | (res["B5"]["season"].eq(LEAK_SEASON) & res["B5"]["week"].le(LEAK_WEEK))
    ]
    a_early = res["B5"].loc[keep, cols].astype(float)
    z_early = g.loc[keep, cols].astype(float)
    leak_diff = float(np.nanmax(np.abs(a_early.to_numpy() - z_early.to_numpy())))
    nan_mismatch = int((a_early.isna().to_numpy() != z_early.isna().to_numpy()).sum())
    print("leak check games", len(keep), "max abs diff", leak_diff, "nan mismatch", nan_mismatch, flush=True)
    if leak_diff != 0.0 or nan_mismatch:
        raise SystemExit("LEAK CHECK FAILED")
    later = res["B5"]["season"].gt(LEAK_SEASON)
    sensitive = res["B5"].loc[later, cols].astype(float).to_numpy() != g.loc[later[later].index, cols].astype(float).to_numpy()
    print("leak positive control: later games differ in", int(np.nansum(sensitive)), "cells", flush=True)
    for mode in ARMS:
        dl = res[mode].loc[common, cols].astype(float) - res["raw"].loc[common, cols].astype(float)
        dl.to_parquet(OUT / f"delta_{mode}.parquet")
        print(mode, "mean abs delta", {c: round(float(v), 5) for c, v in dl.abs().mean().head(6).items()}, flush=True)
    coef_report = {mode: {str(s): t for s, t in coefs[mode].items() if s in (2014, 2020, 2025)} for mode in ARMS}
    (OUT / "coefficients.json").write_text(json.dumps(coef_report, indent=1))
    flags.to_parquet(OUT / "flags.parquet")


def variant(features, delta):
    f = features.copy()
    idx = f["game_id"].to_numpy()
    for c in delta.columns:
        f[c] = f[c] + delta[c].reindex(idx).fillna(0.0).to_numpy()
    return f


def boot_season(df, col):
    g = df.groupby("season")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(s), (BOOT, len(s)))
    d = s[idx].sum(1) / n[idx].sum(1)
    return {
        "low": float(np.percentile(d, 2.5)),
        "high": float(np.percentile(d, 97.5)),
        "pp": float((d > 0).mean()),
        "blocks": len(s),
    }


def compare(arm, base):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["d_ll"] = j["ll"] - j["ll_b"]
    j["d_brier"] = j["brier"] - j["brier_b"]
    out = {}
    for name, mask in (
        ("proxy_2011_2019", j["season"].le(2019)),
        ("true_2020_2025", j["season"].ge(2020)),
        ("pooled_2011_2025", j["season"].ge(0)),
    ):
        s = j.loc[mask]
        w = int(s["correct"].sum())
        flips = s["pick_home"] != s["pick_home_b"]
        per = {}
        for season, t in s.groupby("season"):
            per[int(season)] = {
                "record": f"{int(t['correct'].sum())}-{len(t) - int(t['correct'].sum())}",
                "base": f"{int(t['correct_b'].sum())}-{len(t) - int(t['correct_b'].sum())}",
                "d_acc": float(t["d_acc"].mean()),
            }
        out[name] = {
            "n": len(s),
            "record": f"{w}-{len(s) - w}",
            "record_base": f"{int(s['correct_b'].sum())}-{len(s) - int(s['correct_b'].sum())}",
            "flips": int(flips.sum()),
            "flips_W_L": f"{int((flips & s['correct'].eq(1.0)).sum())}-{int((flips & s['correct'].eq(0.0)).sum())}",
            "d_acc": float(s["d_acc"].mean()),
            "d_acc_season_boot": boot_season(s, "d_acc"),
            "d_ll_arm_minus_base": float(s["d_ll"].mean()),
            "d_brier_arm_minus_base": float(s["d_brier"].mean()),
            "ll_arm": float(s["ll"].mean()),
            "ll_base": float(s["ll_b"].mean()),
            "brier_arm": float(s["brier"].mean()),
            "brier_base": float(s["brier_b"].mean()),
            "per_season": per,
        }
    return out


def grade():
    features = pd.read_parquet(u3.FEATURES)
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base true", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    results = {}
    cells = []
    for arm in ("B", "B3", "B5"):
        path = (UNIT6 if arm == "B" else UNIT6B if arm == "B3" and (UNIT6B / "B3_run.parquet").exists() else OUT) / f"{arm}_run.parquet"
        if path.exists():
            run = pd.read_parquet(path)
        else:
            delta = pd.read_parquet(OUT / f"delta_{arm}.parquet")
            run = HARNESS["run_arm"](variant(features, delta), pop, lambda s: (FULL, BASE_ALPHA))
            run.to_parquet(path)
        run = run.loc[run.index.isin(base.index)]
        results[arm] = compare(run, base)
        for era in ("proxy_2011_2019", "true_2020_2025"):
            r = results[arm][era]
            if arm in ("B", "B3"):
                continue
            print(arm, era, r["record"], r["record_base"], round(r["d_acc"], 3), r["d_acc_season_boot"], flush=True)
            lo, hi = (2011, 2019) if era.startswith("proxy") else (2020, 2025)
            cells.append(
                {
                    "source": "artifacts/mod23_unit7/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit7_availability_depth",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "B5 adds a top-2 RB absence group to unit 6b B3; two eras (two new looks, family total 21); snap counts of completed games; coefficients fit on earlier seasons only.",
                    "name": f"mod23_unit7_availability_{arm}_{era}_accuracy_diff",
                    "description": f"Base model with rolling team inputs adjusted for own {'QB snap share and non-QB starter absences' if arm == 'B3' else 'QB snap share'} in each past game, paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Also discounting past games by running-back absence gave a small unresolved change in picks against the opener.",
                }
            )
    (OUT / "report.json").write_text(json.dumps(results, indent=1, default=str))
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))
    print(json.dumps(results, indent=1, default=str))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "grade"
    if stage == "prep":
        build_all()
    else:
        grade()


main()
