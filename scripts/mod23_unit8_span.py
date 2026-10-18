import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import clv
from nfl_ats.constants import STATE_METRICS
from nfl_ats.features import attach_team_states, build_team_game_metrics, build_team_states
from nfl_ats.margin import make_margin_estimator
from nfl_ats.modeling import regular_season_rows

HARNESS_PATH = Path(__file__).resolve().parent / "mod24_u1.py"
HARNESS = {"__file__": str(HARNESS_PATH), "__name__": "mod24_u1_harness"}
exec(compile(HARNESS_PATH.read_text().rsplit("\nmain()", 1)[0], str(HARNESS_PATH), "exec"), HARNESS)

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod23_unit8_span"
BASE5 = ROOT / "artifacts" / "mod24_u5" / "base_per_game.parquet"
FULL = HARNESS["FULL"]
BASE_ALPHA = HARNESS["BASE_ALPHA"]
SUPP = HARNESS["SUPP"]
SPANS = (3, 4, 6, 8, 12, 16, 24)
CONTROL = 8
SELECT_SEASONS = tuple(range(2011, 2026))
BOOT = 20000
SEED = 20261018
LEAK_SEASON = 2022
LEAK_WEEK = 10
FAMILIES = {
    "scoring": ("point_diff", "ats_residual"),
    "epa": (
        "off_epa_per_play", "off_pass_epa_per_play", "off_rush_epa_per_play", "off_cpoe",
        "def_epa_per_play", "def_pass_epa_per_play", "def_rush_epa_per_play",
    ),
    "yards": ("off_yards_per_play", "def_yards_per_play"),
    "turnover": ("off_turnover_rate", "def_takeaway_rate"),
    "sack": ("off_sack_rate", "def_sack_rate"),
}
SIDES = ("home", "away", "diff")


def metric_columns(metrics):
    return [f"{s}_{m}" for m in metrics for s in SIDES]


STATE_COLS = metric_columns(STATE_METRICS)


def build_tables(features, team_games):
    dated = features[["game_id", "season", "gameday", "home_team", "away_team"]].copy()
    dated["gameday"] = pd.to_datetime(dated["gameday"])
    tables = {}
    for k in SPANS:
        path = OUT / f"state_span_{k}.parquet"
        if path.exists():
            tables[k] = pd.read_parquet(path)
            continue
        att = attach_team_states(dated, build_team_states(team_games, span=k))
        tab = att[["game_id", *STATE_COLS]].set_index("game_id")
        tab.to_parquet(path)
        tables[k] = tab
        print("built span", k, flush=True)
    return tables


def compose(features, tables, spans):
    f = features.copy()
    idx = f["game_id"].to_numpy()
    for fam, metrics in FAMILIES.items():
        cols = metric_columns(metrics)
        src = tables[spans[fam]].reindex(idx)
        for c in cols:
            f[c] = src[c].to_numpy()
    return f


def uniform(k):
    return {fam: k for fam in FAMILIES}


def season_sse(frame, spans_tables, spans, hold, test_season):
    f = compose(frame, spans_tables, spans)
    reg = regular_season_rows(f)
    reg = reg.loc[pd.to_numeric(reg["ats_margin"], errors="coerce").notna()]
    train = reg.loc[~reg["season"].isin([hold, test_season])]
    test = reg.loc[reg["season"].eq(test_season)]
    est = make_margin_estimator("ridge", 42, ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=SUPP)
    est.fit(train.loc[:, list(FULL)], pd.to_numeric(train["ats_margin"]))
    pred = np.asarray(est.predict(test.loc[:, list(FULL)]), dtype=float)
    err = pd.to_numeric(test["ats_margin"]).to_numpy(dtype=float) - pred
    return float(np.sum(err**2)), len(err)


def inner_mse(frame, tables, spans, hold):
    tot, n = 0.0, 0
    for t in SELECT_SEASONS:
        if t == hold:
            continue
        s, c = season_sse(frame, tables, spans, hold, t)
        tot += s
        n += c
    return tot / n


def select_fold(frame, tables, hold):
    fam_choice, fam_table = {}, {}
    for fam in FAMILIES:
        sc = {}
        for k in SPANS:
            spans = uniform(CONTROL)
            spans[fam] = k
            sc[k] = inner_mse(frame, tables, spans, hold)
        fam_table[fam] = sc
        fam_choice[fam] = min(SPANS, key=lambda k: (sc[k], abs(k - CONTROL)))
    gsc = {k: inner_mse(frame, tables, uniform(k), hold) for k in SPANS}
    gk = min(SPANS, key=lambda k: (gsc[k], abs(k - CONTROL)))
    return {
        "family": fam_choice,
        "family_table": {f: {str(k): v for k, v in t.items()} for f, t in fam_table.items()},
        "global": gk,
        "global_table": {str(k): v for k, v in gsc.items()},
    }


def eval_fold(features, tables, spans, season, pop):
    f = compose(features, tables, spans)
    f = f.loc[f["season"].le(season)].reset_index(drop=True)
    if season <= 2019:
        run = HARNESS["proxy_eval"](f, pop.loc[pop["season"].le(season)], lambda s: (FULL, BASE_ALPHA))
    else:
        run = HARNESS["true_eval"](f, FULL, BASE_ALPHA)
    return run.loc[run["season"].eq(season)]


def eval_arm(features, tables, spans_by_season, pop, tag):
    parts = []
    cache = {}
    for s in SELECT_SEASONS:
        spans = spans_by_season[s]
        key = (s, tuple(spans[f] for f in FAMILIES))
        path = OUT / f"fold_{tag}_{s}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
            continue
        if key in cache:
            r = cache[key]
        else:
            r = eval_fold(features, tables, spans, s, pop)
            cache[key] = r
        r.to_parquet(path)
        parts.append(r)
        print("fold", tag, s, tuple(spans.values()), int(r["correct"].sum()), len(r), flush=True)
    return pd.concat(parts)


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


def market_only(j):
    y = np.where(j["pick_home_b"], j["correct_b"], 1.0 - j["correct_b"])
    j = j.assign(y=y)
    ll, br = [], []
    for s, t in j.groupby("season"):
        prior = j.loc[j["season"].lt(s) & (j["season"].le(2019) == (s <= 2019)), "y"]
        p = float(prior.mean()) if len(prior) else 0.5
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll.append(-(t["y"] * np.log(p) + (1 - t["y"]) * np.log(1 - p)))
        br.append((p - t["y"]) ** 2)
    return pd.concat(ll), pd.concat(br)


def compare(arm, base):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    mll, mbr = market_only(j)
    j["ll_m"] = mll
    j["brier_m"] = mbr
    out = {}
    for name, mask in (("sbr_2011_2019", j["season"].le(2019)), ("tuesday_2020_2025", j["season"].ge(2020))):
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
            "ll_arm": float(s["ll"].mean()),
            "ll_base": float(s["ll_b"].mean()),
            "ll_market": float(s["ll_m"].mean()),
            "brier_arm": float(s["brier"].mean()),
            "brier_base": float(s["brier_b"].mean()),
            "brier_market": float(s["brier_m"].mean()),
            "per_season": per,
        }
    return out


def leak_check(features, team_games, tables, frame):
    reg = regular_season_rows(features)
    cut = pd.Timestamp(reg.loc[reg["season"].eq(LEAK_SEASON) & reg["week"].eq(LEAK_WEEK), "gameday"].min())
    tg = team_games.copy()
    tg["gameday"] = pd.to_datetime(tg["gameday"])
    late = tg["gameday"].ge(cut)
    for m in STATE_METRICS:
        tg.loc[late, m] = 1.0e6
    dated = features[["game_id", "season", "gameday", "home_team", "away_team"]].copy()
    dated["gameday"] = pd.to_datetime(dated["gameday"])
    results = {}
    for k in (3, 24):
        att = attach_team_states(dated, build_team_states(tg, span=k))
        tab = att[["game_id", *STATE_COLS]].set_index("game_id")
        early = dated.loc[dated["gameday"].le(cut), "game_id"]
        a = tab.loc[early].to_numpy(dtype=float)
        b = tables[k].reindex(early)[STATE_COLS].to_numpy(dtype=float)
        same = bool(np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)))
        later = dated.loc[dated["gameday"].gt(cut), "game_id"]
        al = tab.loc[later].to_numpy(dtype=float)
        bl = tables[k].reindex(later)[STATE_COLS].to_numpy(dtype=float)
        control = not np.array_equal(np.nan_to_num(al, nan=-9e9), np.nan_to_num(bl, nan=-9e9))
        results[k] = (same, control)
    corrupted = frame.copy()
    corrupted.loc[corrupted["season"].eq(LEAK_SEASON), "ats_margin"] = 999.0
    sel_hold = LEAK_SEASON
    t_ref = {k: season_sse(frame, tables, uniform(k), sel_hold, 2013)[0] for k in (3, 24)}
    t_cor = {k: season_sse(corrupted, tables, uniform(k), sel_hold, 2013)[0] for k in (3, 24)}
    sel_same = t_ref == t_cor
    t_ref2 = season_sse(frame, tables, uniform(8), 2014, LEAK_SEASON)[0]
    t_cor2 = season_sse(corrupted, tables, uniform(8), 2014, LEAK_SEASON)[0]
    sel_control = t_ref2 != t_cor2
    print("leak span check (features before cutoff identical, later differ):", results, flush=True)
    print("leak selection check (held-out season outcomes unused, positive control differs):", sel_same, sel_control, flush=True)
    if not all(a and b for a, b in results.values()) or not sel_same or not sel_control:
        raise SystemExit("LEAK CHECK FAILED")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(u3.SNAPSHOT / "team_stats.parquet")
    team_games = build_team_game_metrics(schedules, team_stats)
    tables = build_tables(features, team_games)
    ref = features.set_index("game_id")[STATE_COLS].reindex(tables[CONTROL].index)
    keep = (features.set_index("game_id").reindex(tables[CONTROL].index)["game_type"].eq("REG") & features.set_index("game_id").reindex(tables[CONTROL].index)["season"].le(2025)).to_numpy()
    diff = (tables[CONTROL] - ref).abs()[keep].max().max()
    print("span 8 rebuild max abs diff vs stored features", float(diff), flush=True)
    if not diff < 1e-9:
        raise SystemExit("span 8 rebuild does not reproduce stored features")
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    frame = features
    leak_check(features, team_games, tables, frame)
    sel_path = OUT / "selection.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
    else:
        sel = {}
    for s in SELECT_SEASONS:
        if str(s) not in sel:
            sel[str(s)] = select_fold(frame, tables, s)
            sel_path.write_text(json.dumps(sel, indent=1))
            print("selected", s, sel[str(s)]["family"], sel[str(s)]["global"], flush=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    fam_spans = {s: dict(sel[str(s)]["family"]) for s in SELECT_SEASONS}
    glob_spans = {s: uniform(sel[str(s)]["global"]) for s in SELECT_SEASONS}
    control_spans = {s: uniform(CONTROL) for s in SELECT_SEASONS}
    results = {}
    arms = [("repro", control_spans), ("fam", fam_spans), ("glob", glob_spans)]
    if stage != "all":
        arms = [a for a in arms if a[0] == stage]
    for tag, spans in arms:
        run = eval_arm(features, tables, spans, pop, tag)
        run.to_parquet(OUT / f"{tag}_run.parquet")
        results[tag] = compare(run.loc[run.index.isin(base.index)], base)
        for era, r in results[tag].items():
            print(tag, era, r["record"], "base", r["record_base"], "d_acc", round(r["d_acc"], 3), r["d_acc_season_boot"], "flips", r["flips"], r["flips_W_L"], flush=True)
    (OUT / f"report_{stage}.json").write_text(json.dumps(results, indent=1, default=str))
    if "fam" in results and "glob" in results:
        cells = []
        for tag, label in (("fam", "per-family span chosen leave-one-season-out"), ("glob", "one global span chosen leave-one-season-out")):
            for era, r in results[tag].items():
                lo, hi = (2011, 2019) if era.startswith("sbr") else (2020, 2025)
                cells.append(
                    {
                        "source": "artifacts/mod23_unit8_span/report_all.json",
                        "effect_units": "accuracy_points",
                        "classification": "unresolved_below_power",
                        "league": "nfl",
                        "season_start": lo,
                        "season_end": hi,
                        "family": "mod23_unit8_span",
                        "category": "modeling",
                        "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                        "notes": f"Base ridge alpha 10 with the exponentially weighted team-state span ({label}) over a grid of 3 to 24 games, scored on the held-out season; four looks (two eras by family and global arms); the declared penalty and special-teams families have no state input and were not tested.",
                        "name": f"mod23_unit8_span_{tag}_{era}_accuracy_diff",
                        "description": f"Base model with {label}, paired vs span 8 base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                        "effect": r["d_acc"],
                        "interval_low": r["d_acc_season_boot"]["low"],
                        "interval_high": r["d_acc_season_boot"]["high"],
                        "probability_positive": r["d_acc_season_boot"]["pp"],
                        "sample_games": r["n"],
                        "sample_blocks": r["d_acc_season_boot"]["blocks"],
                        "plain_summary": "Letting the memory length of team form be chosen from other seasons moved picks against the opener by an unresolved amount.",
                    }
                )
        (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
