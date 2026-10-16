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
OUT = ROOT / "artifacts" / "mod23_unit6"
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


def team_game_flags():
    path = sorted(glob.glob(SNAP_GLOB))[-1]
    snaps = pd.read_parquet(path)
    snaps = snaps.loc[snaps["game_type"].eq("REG") & snaps["position"].eq("QB")].copy()
    snaps["team"] = snaps["team"].replace(ALIAS)
    snaps["offense_snaps"] = pd.to_numeric(snaps["offense_snaps"], errors="coerce").fillna(0.0)
    rows = []
    for (gid, team), g in snaps.groupby(["game_id", "team"]):
        top = g.sort_values("offense_snaps", ascending=False).iloc[0]
        rows.append(
            {
                "game_id": gid,
                "team": team,
                "season": int(g["season"].iloc[0]),
                "week": int(g["week"].iloc[0]),
                "qb1": top["pfr_player_id"],
                "snaps": dict(zip(g["pfr_player_id"], g["offense_snaps"], strict=True)),
            }
        )
    tg = pd.DataFrame(rows).sort_values(["team", "season", "week"]).reset_index(drop=True)
    flags = np.zeros(len(tg))
    for _, idx in tg.groupby("team").groups.items():
        last_prev = None
        cur_season = None
        cum = {}
        last_qb = None
        for i in idx:
            r = tg.loc[i]
            if r["season"] != cur_season:
                last_prev = last_qb
                cur_season = r["season"]
                cum = {}
            starter = max(cum, key=cum.get) if cum else last_prev
            flags[i] = float(starter is not None and r["qb1"] != starter)
            for k, v in r["snaps"].items():
                cum[k] = cum.get(k, 0.0) + v
            last_qb = r["qb1"]
    tg["flag"] = flags
    return tg[["game_id", "team", "season", "week", "qb1", "flag"]], path


def fit_coefficients(frame, use_opp):
    coef = {}
    metrics = list(STATE_METRICS)
    seasons = sorted(frame["season"].unique())
    for s in seasons:
        coef[int(s)] = {m: (0.0, 0.0) for m in metrics}
        fit = frame.loc[frame["season"].ge(FIRST_SNAP_SEASON) & frame["season"].lt(s)]
        CTX["fit_seasons"][int(s)] = int(fit["season"].max()) if len(fit) else None
        if fit.empty or fit["own"].sum() == 0:
            continue
        own = fit["own"] - fit.groupby("season")["own"].transform("mean")
        opp = fit["opp"] - fit.groupby("season")["opp"].transform("mean")
        for m in metrics:
            y = pd.to_numeric(fit[m], errors="coerce")
            ok = y.notna()
            y = y[ok] - y[ok].groupby(fit.loc[ok, "season"]).transform("mean")
            cols = [own[ok]] + ([opp[ok]] if use_opp else [])
            X = np.column_stack([c.to_numpy() for c in cols])
            beta = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
            coef[int(s)][m] = (float(beta[0]), float(beta[1]) if use_opp else 0.0)
    return coef


def wrapped(orig):
    def fn(schedules, team_stats, game_types=("REG",)):
        out = orig(schedules, team_stats, game_types)
        if CTX["mode"] == "raw":
            return out
        fl = CTX["flags"].copy()
        if CTX["zero_after"] is not None:
            zs, zw = CTX["zero_after"]
            late = fl["season"].gt(zs) | (fl["season"].eq(zs) & fl["week"].ge(zw))
            fl.loc[late, "flag"] = 0.0
        own = fl[["game_id", "team", "flag"]].rename(columns={"flag": "own"})
        o = out.merge(own, on=["game_id", "team"], how="left")
        oppf = own.rename(columns={"team": "opp_team", "own": "opp"})
        pair = o[["game_id", "team"]].merge(oppf, on="game_id")
        pair = pair.loc[pair["team"].ne(pair["opp_team"])][["game_id", "team", "opp"]]
        o = o.merge(pair, on=["game_id", "team"], how="left")
        o[["own", "opp"]] = o[["own", "opp"]].fillna(0.0)
        use_opp = CTX["mode"] == "A"
        coef = fit_coefficients(o, use_opp)
        CTX["coef"] = coef
        for s, table in coef.items():
            m = o["season"].eq(s).to_numpy()
            for metric, (b_own, b_opp) in table.items():
                o.loc[m, metric] = (
                    pd.to_numeric(o.loc[m, metric], errors="coerce")
                    - b_own * o.loc[m, "own"]
                    - b_opp * o.loc[m, "opp"]
                )
        if len(o) != len(out):
            raise RuntimeError("row mismatch")
        return o[list(out.columns)].sort_values(["gameday", "game_id", "team"])

    return fn


def build_all():
    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(u3.SNAPSHOT / "team_stats.parquet")
    flags, snap_path = team_game_flags()
    CTX["flags"] = flags
    print("snap file", snap_path, "team-games", len(flags), "flagged", int(flags["flag"].sum()), flush=True)
    print(flags.groupby("season")["flag"].mean().round(4).to_dict(), flush=True)
    orig = F.build_team_game_metrics
    F.build_team_game_metrics = wrapped(orig)
    res = {}
    coefs = {}
    for mode in ("raw", "A", "B"):
        CTX["mode"] = mode
        g = F._build_features_pass(schedules, team_stats, ("REG",))
        res[mode] = g.set_index("game_id")
        coefs[mode] = CTX["coef"]
        if mode != "raw":
            for s, fit_max in CTX["fit_seasons"].items():
                if fit_max is not None and fit_max >= s:
                    raise SystemExit(f"coefficient leak: season {s} fit through {fit_max}")
        print("built", mode, len(g), flush=True)
    CTX["mode"] = "A"
    CTX["zero_after"] = (LEAK_SEASON, LEAK_WEEK)
    g = F._build_features_pass(schedules, team_stats, ("REG",)).set_index("game_id")
    CTX["zero_after"] = None
    F.build_team_game_metrics = orig
    feats = pd.read_parquet(u3.FEATURES).set_index("game_id")
    cols = [c for c in STATE_COLS if c in feats.columns]
    common = feats.index.intersection(res["raw"].index)
    d = (res["raw"].loc[common, cols] - feats.loc[common, cols]).abs()
    print("raw repro max abs diff", float(d.max().max()), "common", len(common), "of", len(feats), flush=True)
    keep = res["A"].index[
        res["A"]["season"].lt(LEAK_SEASON) | (res["A"]["season"].eq(LEAK_SEASON) & res["A"]["week"].le(LEAK_WEEK))
    ]
    a_early = res["A"].loc[keep, cols].astype(float)
    z_early = g.loc[keep, cols].astype(float)
    leak_diff = float(np.nanmax(np.abs(a_early.to_numpy() - z_early.to_numpy())))
    nan_mismatch = int((a_early.isna().to_numpy() != z_early.isna().to_numpy()).sum())
    print("leak check games", len(keep), "max abs diff", leak_diff, "nan mismatch", nan_mismatch, flush=True)
    if leak_diff != 0.0 or nan_mismatch:
        raise SystemExit("LEAK CHECK FAILED")
    sensitive = (res["A"].loc[res["A"]["season"].gt(LEAK_SEASON), cols].astype(float).to_numpy()
                 != g.loc[res["A"]["season"].gt(LEAK_SEASON).pipe(lambda m: m[m].index), cols].astype(float).to_numpy())
    print("leak positive control: later games differ in", int(np.nansum(sensitive)), "cells", flush=True)
    deltas = {}
    for mode in ("A", "B"):
        dl = (res[mode].loc[common, cols].astype(float) - res["raw"].loc[common, cols].astype(float))
        dl.to_parquet(OUT / f"delta_{mode}.parquet")
        deltas[mode] = dl
        print(mode, "mean abs delta", {c: round(float(v), 5) for c, v in dl.abs().mean().head(6).items()}, flush=True)
    coef_report = {
        mode: {str(s): {m: list(v) for m, v in t.items()} for s, t in coefs[mode].items() if s >= 2014}
        for mode in ("A", "B")
    }
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
    for arm in ("A", "B"):
        path = OUT / f"{arm}_run.parquet"
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
            print(arm, era, r["record"], r["record_base"], round(r["d_acc"], 3), r["d_acc_season_boot"], flush=True)
            lo, hi = (2011, 2019) if era.startswith("proxy") else (2020, 2025)
            cells.append(
                {
                    "source": "artifacts/mod23_unit6/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit6_availability_history",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "Two arms, two eras (four accuracy looks); starter-QB availability flags from snap counts of completed games; coefficients fit on earlier seasons only.",
                    "name": f"mod23_unit6_availability_{arm}_{era}_accuracy_diff",
                    "description": f"Base model with rolling team inputs adjusted for own{' and opponent' if arm == 'A' else ''} starting-QB availability in each past game, paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Adjusting a team's recent stats for games the starting quarterback missed did not reliably change picks against the opener.",
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
