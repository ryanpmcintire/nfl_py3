from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mod57_p2q_market_reaction_speed as p2q  # noqa: E402
import xlg14_family_screen as x14  # noqa: E402
import xlg14_mod41_cfb as u1  # noqa: E402

HARNESS = ROOT / "scripts" / "xlg_harness_power.py"
NFL_TERM = ROOT / "artifacts" / "mod57_p2r" / "results.json"
NFL_TERM_NAME = "early_surprise"
OUT = ROOT / "artifacts" / "xlg_early_overreaction_cfb"
RESULTS = OUT / "results.json"
FAMILY = "xlg_early_overreaction_cfb"
COLLEGE_REGULAR_GAMES = 12
EARLY_COUNTS = (p2q.EARLY[0] - 1, p2q.EARLY[1] - 1)
REST_COUNTS = (EARLY_COUNTS[1] + 1, COLLEGE_REGULAR_GAMES - 1)
SPLIT = REST_COUNTS[0] + (REST_COUNTS[1] - REST_COUNTS[0] + 1) // 2 - 1
WINDOWS = {
    "early": (EARLY_COUNTS[0], EARLY_COUNTS[1]),
    "mid": (REST_COUNTS[0], SPLIT),
    "late": (SPLIT + 1, None),
}
LINES = {"close": "close_line", "open": "open_line"}
SEEN_BY = ("ll", "acc")
ADDED_METRICS = (
    ("oos_accuracy_points_gain", "accuracy_points"),
    ("oos_log_loss_gain", "log_loss_improvement"),
)


def load_harness():
    text = HARNESS.read_text(encoding="utf-8")
    body, call, rest = text.rpartition("\nmain()")
    if not call or rest.strip():
        raise SystemExit("harness power script no longer ends with a bare main() call")
    mod = types.ModuleType("xlg_harness_power")
    mod.__file__ = str(HARNESS)
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        exec(compile(body, str(HARNESS), "exec"), mod.__dict__)
    finally:
        sys.argv = argv
    return mod


XP = load_harness()
M36 = XP.M36


def plain(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def nfl_reference():
    r = json.loads(NFL_TERM.read_text(encoding="utf-8"))
    frozen = r["primary"]["frozen_T"]
    per_sd = float(frozen["beta"][-1])
    sd = float(frozen["stds"][NFL_TERM_NAME])
    return {
        "coefficient_per_sd": per_sd,
        "term_sd": sd,
        "coefficient_per_point": per_sd / sd,
        "sign": float(np.sign(per_sd)),
    }


def load_reg():
    full = pd.read_parquet(u1.FEATURES)
    g = full[full["season"].isin(u1.SEASONS) & full["result"].notna() & full["spread_line"].notna()]
    reg = pd.DataFrame(
        {
            "game_id": g["game_id"].to_numpy(),
            "season": g["season"].to_numpy(int),
            "week": pd.to_numeric(g["week"]).to_numpy(int),
            "gameday": pd.to_datetime(g["gameday"]).to_numpy(),
            "home_team": g["home_id"].to_numpy(),
            "away_team": g["away_id"].to_numpy(),
            "margin": g["result"].to_numpy(float),
            "close_line": g["spread_line"].to_numpy(float),
            "open_line": g["spread_open"].to_numpy(float),
        }
    )
    return reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def prior_counts(reg):
    parts = []
    for side in ("home", "away"):
        part = reg[["game_id", "season", "gameday", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        parts.append(part.assign(side=side))
    long = pd.concat(parts, ignore_index=True)
    long = long.sort_values(["season", "team", "gameday", "game_id"]).reset_index(drop=True)
    long["n"] = long.groupby(["season", "team"]).cumcount()
    home = long.loc[long["side"] == "home"].set_index("game_id")["n"]
    away = long.loc[long["side"] == "away"].set_index("game_id")["n"]
    return np.maximum(home, away.reindex(home.index))


def window_labels(reg):
    out = np.full(len(reg), "none", dtype=object)
    ok = reg["D2"].notna().to_numpy()
    count = reg["count"].to_numpy()
    for name, (lo, hi) in WINDOWS.items():
        mask = ok & (count >= lo)
        if hi is not None:
            mask = mask & (count <= hi)
        out[mask] = name
    return out


def truncation_max(reg, full):
    sample = reg.loc[reg["D2"].notna()].sample(p2q.CHECK_GAMES, random_state=p2q.CHECK_SEED)
    worst = 0.0
    cols = ["game_id", "season", "gameday", "home_team", "away_team", "margin", "close_line"]
    for g in sample.itertuples():
        keep = reg["gameday"] < g.gameday
        same = (reg["gameday"] == g.gameday) & (reg["season"] == g.season)
        cut = pd.concat(
            [reg.loc[keep, cols], reg.loc[same, cols].assign(margin=0.0, close_line=0.0)]
        ).reset_index(drop=True)
        again = p2q.surprise_values(cut)
        worst = max(worst, abs(float(again.loc[g.game_id]) - float(full.loc[g.game_id])))
    return worst


def block_read(season, value, scale, rng):
    block, labels = pd.factorize(season)
    count = len(labels)
    sums = np.bincount(block, weights=value, minlength=count)
    sizes = np.bincount(block, minlength=count).astype(float)
    idx = rng.integers(0, count, (p2q.BOOTSTRAP_SAMPLES, count))
    draws = scale * sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    low, high = np.percentile(draws, M36.INTERVAL)
    return {
        "effect": float(scale * value.mean()),
        "ci95": [float(low), float(high)],
        "probability_positive": float((draws > 0).mean()),
    }


def record_of(hit):
    wins = int(hit.sum())
    return f"{wins}-{len(hit) - wins}"


def pick_read(f, pick_home, line, rng):
    covered = (f["margin"] > f[line]).to_numpy()
    hit = (np.asarray(pick_home, dtype=bool) == covered).astype(float)
    season = f["season"].to_numpy()
    out = {"games": len(hit), "record": record_of(hit)}
    out.update(block_read(season, hit - 0.5, 100.0, rng))
    by = {}
    for s in np.unique(season):
        h = hit[season == s]
        by[int(s)] = {"record": record_of(h), "accuracy_points": float(100.0 * (h.mean() - 0.5))}
    out["by_season"] = by
    out["seasons_above_half"] = sum(v["accuracy_points"] > 0 for v in by.values())
    out["seasons_below_half"] = sum(v["accuracy_points"] < 0 for v in by.values())
    out["seasons"] = len(by)
    return out


def gradable(frame, window, line):
    f = frame.loc[(frame["window"] == window) & (frame["D2"] != 0) & frame[line].notna()]
    return f.loc[f["margin"] != f[line]]


def fade(frame, window, line, sign, rng):
    f = gradable(frame, window, line)
    return pick_read(f, (sign * f["D2"]).to_numpy() > 0, line, rng)


def learned_signs(frame, window):
    w = frame.loc[frame["window"] == window]
    real = w["margin"] - w["close_line"]
    signs = {}
    for test in sorted(int(s) for s in w["season"].unique() if s >= p2q.FIRST_TEST):
        train = w["season"] < test
        cov = np.cov(w.loc[train, "D2"].to_numpy(float), real.loc[train].to_numpy(float))
        signs[test] = float(np.sign(cov[0, 1]))
    return signs


def learned(frame, window, line, signs, rng):
    parts = []
    for test, sign in signs.items():
        te = gradable(frame.loc[frame["season"] == test], window, line)
        parts.append(te.assign(pick_home=(sign * te["D2"]).to_numpy() > 0))
    f = pd.concat(parts, ignore_index=True)
    return pick_read(f, f["pick_home"].to_numpy(bool), line, rng)


def slope_read(frame, window, line, rng):
    f = frame.loc[(frame["window"] == window) & frame[line].notna()]
    x = f["D2"].to_numpy(float)
    y = (f["margin"] - f[line]).to_numpy(float)
    block, labels = pd.factorize(f["season"].to_numpy())
    stats = np.zeros((len(labels), 5))
    np.add.at(stats, block, np.column_stack([np.ones(len(x)), x, y, x * x, x * y]))

    def beta(t):
        n, sx, sy, sxx, sxy = t.T
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)

    idx = rng.integers(0, len(labels), (p2q.BOOTSTRAP_SAMPLES, len(labels)))
    draws = beta(stats[idx].sum(axis=1))
    low, high = np.percentile(draws, M36.INTERVAL)
    each = beta(stats)
    return {
        "games": len(x),
        "slope": float(beta(stats.sum(axis=0, keepdims=True))[0]),
        "ci95": [float(low), float(high)],
        "probability_positive": float((draws > 0).mean()),
        "seasons_negative": int((each < 0).sum()),
        "seasons": len(labels),
        "by_season": {int(s): float(v) for s, v in zip(labels, each, strict=True)},
    }


def harness_frame(reg):
    d = x14.cfb_frame()
    t = reg.set_index("game_id")
    d["window"] = t["window"].reindex(d["game_id"]).fillna("none").to_numpy()
    d2 = t["D2"].reindex(d["game_id"]).to_numpy(float)
    for name in WINDOWS:
        d[f"T_{name}"] = np.where(d["window"].to_numpy() == name, d2, 0.0)
    d["in_reg"] = d["game_id"].isin(t.index).to_numpy()
    return d


def permuted_term(d, rng):
    out = d["T_early"].to_numpy().copy()
    season = d["season"].to_numpy()
    early = d["window"].to_numpy() == "early"
    for s in np.unique(season):
        idx = np.flatnonzero((season == s) & early)
        out[idx] = out[rng.permutation(idx)]
    return out


def addone(d, col):
    cols = ["logit_p", col]
    o = u1.analyze(d, cols, col)
    y = d["y"].to_numpy()
    pb = u1.fit(d, ["logit_p"]).predict_proba(d[["logit_p"]])[:, 1]
    pt = u1.fit(d, cols).predict_proba(d[cols])[:, 1]
    ins = 100.0 * float(((pt >= 0.5) == (y == 1)).mean() - ((pb >= 0.5) == (y == 1)).mean())
    folds = [v[-1] for v in o["coef_per_heldout_season"].values()]
    o["in_sample_accuracy_points_gain"] = ins
    o["in_minus_out_accuracy_points_gain"] = ins - o["oos_accuracy_points_gain"]["point"]
    o["fold_coefficient"] = {
        "mean": float(np.mean(folds)),
        "min": float(np.min(folds)),
        "max": float(np.max(folds)),
        "negative_folds": int(sum(v < 0 for v in folds)),
        "folds": len(folds),
    }
    o["nonzero_games"] = int((d[col] != 0).sum())
    return o


def matched_null(frame, tpl, index):
    st = M36.setup(frame, tpl, index)
    real = M36.real_read(st, tpl)
    rng = np.random.default_rng([XP.SEED, XP.STREAM_ADDONE, index, 0])
    rows = []
    for i in range(XP.DRAWS):
        M36.draw_rows(st, tpl, "null", 0.0, i, M36.permuted(st, rng), rows)
    null = pd.DataFrame(rows)
    out = {
        "shape": st["shape"],
        "real": real,
        "null": {k: M36.band(null[k].to_numpy()) for k in (*M36.SHORT, "coef")},
        "percentile": {
            k: M36.percentile_of(null[k].to_numpy(), real["served"][k]["effect"]) for k in M36.SHORT
        },
        "null_share_notable": {
            k: float((null[f"pp_{k}"] >= M36.NOTABLE[0]).mean()) for k in M36.SHORT
        },
        "null_changed_mean": float(null["changed"].mean()),
        "null_coefficient_sd": float(null["coef"].std(ddof=1)),
        "draws": len(null),
    }
    out["percentile"]["coef"] = M36.percentile_of(
        null["coef"].to_numpy(), real["full_coefficient_per_sd"]
    )
    return st, out


def detectable(st, tpl, index, null_sd, nfl_size, sign):
    reported = json.loads(XP.MOD36_SUMMARY.read_text(encoding="utf-8"))
    sizes = {abs(float(v)) for v in reported["reported_sizes_logit_per_sd"].values()}
    steps = {k * null_sd for k in range(1, M36.LADDER_RUNGS + 1)}
    grid = sorted({*sizes, *steps, nfl_size} - {0.0})
    cells, reached = [], {}
    for c, size in enumerate(grid):
        if len(reached) == len(SEEN_BY) and size != nfl_size:
            continue
        rng = np.random.default_rng([XP.SEED, XP.STREAM_ADDONE, index, 1 + c])
        tabs = XP.plant_tables(st, sign * size)
        rows = []
        for i in range(XP.DRAWS):
            x = XP.planted_from(st, tabs, rng)
            M36.draw_rows(st, tpl, "planted", sign * size, i, x, rows)
        g = pd.DataFrame(rows)
        cell = {
            "size_logit_per_sd": float(size),
            "true_accuracy_points": float(g["oracle_acc"].mean()),
            "true_log_loss_gain": float(g["oracle_ll"].mean()),
            "share_seen_ll": float((g["pp_ll"] >= M36.NOTABLE[0]).mean()),
            "share_seen_acc": float((g["pp_acc"] >= M36.NOTABLE[0]).mean()),
            "mean_ll": float(g["ll"].mean()),
            "mean_acc": float(g["acc"].mean()),
        }
        cells.append(cell)
        for k in SEEN_BY:
            if k not in reached and cell[f"share_seen_{k}"] >= M36.POWER_TARGET:
                reached[k] = cell
    at_nfl = next(c for c in cells if c["size_logit_per_sd"] == float(nfl_size))
    return {"grid": cells, "smallest": reached, "at_nfl_coefficient": at_nfl}


def compute():
    started = time.time()
    nfl = nfl_reference()
    sign = nfl["sign"]
    reg = load_reg()
    d2 = p2q.surprise_values(reg)
    reg["D2"] = d2.reindex(reg["game_id"]).to_numpy()
    reg["count"] = prior_counts(reg).reindex(reg["game_id"]).to_numpy()
    reg["window"] = window_labels(reg)
    rng = np.random.default_rng(p2q.BOOTSTRAP_SEED)
    res = {
        "family": FAMILY,
        "nfl_reference": nfl,
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "games": len(reg),
        "seasons": sorted(int(s) for s in reg["season"].unique()),
        "truncation_max_abs_diff": truncation_max(reg, d2),
        "one_term": {},
    }
    for w in WINDOWS:
        signs = learned_signs(reg, w)
        entry = {
            "games_in_window": int((reg["window"] == w).sum()),
            "term_sd_in_window": float(reg.loc[reg["window"] == w, "D2"].std(ddof=1)),
            "fade": {k: fade(reg, w, col, sign, rng) for k, col in LINES.items()},
            "slope": {k: slope_read(reg, w, col, rng) for k, col in LINES.items()},
            "learned_signs": signs,
            "learned_signs_matching_nfl": sum(v == sign for v in signs.values()),
        }
        if w == "early":
            entry["learned"] = {k: learned(reg, w, col, signs, rng) for k, col in LINES.items()}
        res["one_term"][w] = entry
    RESULTS.write_text(json.dumps(res, indent=1, default=plain), encoding="utf-8")
    print("one-term arms done", f"{time.time() - started:.0f}s", flush=True)

    d = harness_frame(reg)
    d["T_permuted"] = permuted_term(d, rng)
    res["harness_games"] = len(d)
    res["harness_games_without_close"] = int((~d["in_reg"]).sum())
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            M36.SERVED_COLUMN: expit(d["logit_p"].to_numpy(float)),
        }
    )
    res["matched_null"] = {}
    for position, w in enumerate(WINDOWS):
        index = len(XP.TEMPLATES) + position
        col = f"T_{w}"
        frame[col] = d[col].to_numpy(float)
        tpl = {"name": f"early_overreaction_{w}", "column": col, "intercept": False}
        st, out = matched_null(frame, tpl, index)
        if w == "early":
            nfl_size = abs(nfl["coefficient_per_point"]) * st["sd"]
            out["nfl_coefficient_at_college_scale_per_sd"] = float(nfl_size)
            out["detectable"] = detectable(
                st, tpl, index, out["null_coefficient_sd"], nfl_size, sign
            )
        res["matched_null"][w] = out
        del st
        print("matched null", w, f"{time.time() - started:.0f}s", flush=True)
    RESULTS.write_text(json.dumps(res, indent=1, default=plain), encoding="utf-8")

    res["added_term"] = {}
    for w in (*WINDOWS, "permuted"):
        res["added_term"][w] = addone(d, f"T_{w}")
        print("added term", w, f"{time.time() - started:.0f}s", flush=True)
    res["elapsed_seconds"] = time.time() - started
    RESULTS.write_text(json.dumps(res, indent=1, default=plain), encoding="utf-8")


def cells(res):
    out = []
    first, last = min(res["seasons"]), max(res["seasons"])
    for w, e in res["one_term"].items():
        lo, hi = res["windows"][w]
        span = f"{lo} to {hi}" if hi is not None else f"{lo} or more"
        for k, f in e["fade"].items():
            out.append(
                {
                    "name": f"xlg_eo_cfb_fade_{w}_{k}",
                    "description": (
                        f"College: fade the surprise to date (direction fixed from the NFL "
                        f"term), games where teams bring {span} earlier lined games ({w} "
                        f"window), accuracy minus 50 against the {k} line, season-block "
                        f"bootstrap; record {f['record']}, {f['seasons_above_half']} of "
                        f"{f['seasons']} seasons above half"
                    ),
                    "effect": f["effect"],
                    "effect_units": "accuracy_points",
                    "interval_low": f["ci95"][0],
                    "interval_high": f["ci95"][1],
                    "probability_positive": f["probability_positive"],
                    "season_start": first,
                    "season_end": last,
                    "sample_games": f["games"],
                }
            )
        for k, s in e["slope"].items():
            out.append(
                {
                    "name": f"xlg_eo_cfb_slope_{w}_{k}",
                    "description": (
                        f"College: slope of result minus the {k} line on surprise to date, "
                        f"games where teams bring {span} earlier lined games ({w} window), "
                        f"season-block bootstrap; negative in {s['seasons_negative']} of "
                        f"{s['seasons']} seasons; the NFL early slope is negative"
                    ),
                    "effect": s["slope"],
                    "effect_units": "slope_difference",
                    "interval_low": s["ci95"][0],
                    "interval_high": s["ci95"][1],
                    "probability_positive": s["probability_positive"],
                    "season_start": first,
                    "season_end": last,
                    "sample_games": s["games"],
                }
            )
    for k, f in res["one_term"]["early"]["learned"].items():
        out.append(
            {
                "name": f"xlg_eo_cfb_learned_sign_early_{k}",
                "description": (
                    "College: follow the direction chosen by college's own earlier seasons "
                    f"(walk-forward), early window, accuracy minus 50 against the {k} line, "
                    f"season-block bootstrap; record {f['record']}"
                ),
                "effect": f["effect"],
                "effect_units": "accuracy_points",
                "interval_low": f["ci95"][0],
                "interval_high": f["ci95"][1],
                "probability_positive": f["probability_positive"],
                "season_start": p2q.FIRST_TEST,
                "season_end": last,
                "sample_games": f["games"],
            }
        )
    for w, a in res["added_term"].items():
        for metric, units in ADDED_METRICS:
            g = a[metric]
            cell = {
                "name": f"xlg_eo_cfb_addone_{w}_{units}",
                "description": (
                    f"College-first harness: surprise to date in the {w} window as one added "
                    f"fitted term on the opener-input cover probability, leave-one-season-out, "
                    f"{units} against the base, season-week block bootstrap; changed picks "
                    f"{a['decisive_term_correct']}-{a['decisive_base_correct']}"
                ),
                "effect": g["point"],
                "effect_units": units,
                "interval_low": g["ci95"][0],
                "interval_high": g["ci95"][1],
                "probability_positive": g["probability_positive"],
                "season_start": first,
                "season_end": last,
                "sample_games": res["harness_games"],
            }
            if w == "permuted":
                cell["category"] = "control"
            if w == "early":
                cell.update(control_bound(res))
            out.append(cell)
    return out


def control_bound(res):
    m = res["matched_null"]["early"]
    det = m["detectable"]
    seen, nfl = det["smallest"]["ll"], det["at_nfl_coefficient"]
    return {
        "classification": "bounded_by_control",
        "closing_ground": "positive_control_bound",
        "classification_evidence": (
            f"Positive control on this term's own shape ({m['draws']} planted draws per size "
            f"through the college add-one): a linear effect of {seen['size_logit_per_sd']:.3f} "
            f"logit per SD ({seen['true_accuracy_points']:+.2f} accuracy points over all games) "
            f"is seen at P+ >= {M36.NOTABLE[0]:.2f} in {100 * seen['share_seen_ll']:.0f}% of "
            f"draws by log loss, and the NFL per-point coefficient carried to the college scale "
            f"({nfl['size_logit_per_sd']:.3f} per SD) in {100 * nfl['share_seen_ll']:.0f}%; the "
            f"real term sits at the {m['percentile']['ll']:.1f}th percentile of its matched "
            f"shuffle null on log loss. Bounds the linear fitted form in college below that "
            f"size; does not bound the sign-rule fade cells and says nothing about the NFL term."
        ),
    }


def report():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    batch = {
        "league": "cfb",
        "classification": "unresolved_below_power",
        "source": "artifacts/xlg_early_overreaction_cfb/results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "College test of the NFL early-season overreaction term (MOD-57 P2-Q, P2-R): "
            "surprise to date built from earlier closing lines, windows by count of earlier "
            "games, direction fixed from the NFL fit, declared in "
            "docs/lanes/xlg-early-overreaction-cfb-2026-11-14.md before outcomes were joined."
        ),
        "classification_evidence": (
            "Season-block and season-week block bootstrap with probability_positive; matched "
            "within-season shuffle null on the college add-one; no closing ground claimed."
        ),
        "plain_summary": (
            "Whether a college team that has beaten or missed the betting line early in a "
            "season gets a next line that is too high or too low."
        ),
        "cells": cells(res),
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    nfl = res["nfl_reference"]
    lines = [
        f"NFL coefficient per point {nfl['coefficient_per_point']:+.4f} per SD "
        f"{nfl['coefficient_per_sd']:+.4f} term SD {nfl['term_sd']:.2f}",
        f"games {res['games']} seasons {len(res['seasons'])} harness {res['harness_games']} "
        f"without close {res['harness_games_without_close']} truncation "
        f"{res['truncation_max_abs_diff']:.2e} elapsed {res['elapsed_seconds']:.0f}s",
    ]
    for w, e in res["one_term"].items():
        lines.append(
            f"WINDOW {w} {res['windows'][w]} games {e['games_in_window']} "
            f"term SD {e['term_sd_in_window']:.2f}"
        )
        for k, f in e["fade"].items():
            lines.append(
                f" fade {k}: {f['record']} {f['effect']:+.2f} [{f['ci95'][0]:+.2f}, "
                f"{f['ci95'][1]:+.2f}] P+ {f['probability_positive']:.3f} seasons above half "
                f"{f['seasons_above_half']}/{f['seasons']} below {f['seasons_below_half']}"
            )
        for k, s in e["slope"].items():
            lines.append(
                f" slope {k}: {s['slope']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                f"P+ {s['probability_positive']:.3f} seasons negative "
                f"{s['seasons_negative']}/{s['seasons']} games {s['games']}"
            )
        lines.append(
            f" college-learned sign matches NFL in {e['learned_signs_matching_nfl']}/"
            f"{len(e['learned_signs'])} test seasons"
        )
        for k, f in e.get("learned", {}).items():
            lines.append(
                f" learned-sign {k}: {f['record']} {f['effect']:+.2f} [{f['ci95'][0]:+.2f}, "
                f"{f['ci95'][1]:+.2f}] P+ {f['probability_positive']:.3f}"
            )
    for w, a in res["added_term"].items():
        acc, ll, fc = a["oos_accuracy_points_gain"], a["oos_log_loss_gain"], a["fold_coefficient"]
        lines.append(
            f"ADDED {w}: nonzero {a['nonzero_games']} changed {a['decisive_pick_changed_oos']} "
            f"term {a['decisive_term_correct']} base {a['decisive_base_correct']} | acc "
            f"{acc['point']:+.3f} [{acc['ci95'][0]:+.3f}, {acc['ci95'][1]:+.3f}] P+ "
            f"{acc['probability_positive']:.3f} | ll {ll['point']:+.6f} [{ll['ci95'][0]:+.6f}, "
            f"{ll['ci95'][1]:+.6f}] P+ {ll['probability_positive']:.3f}"
        )
        lines.append(
            f"  in-sample ll {a['in_sample_log_loss_gain']:+.6f} gap "
            f"{a['in_minus_out_log_loss_gain']:+.6f} | in-sample acc "
            f"{a['in_sample_accuracy_points_gain']:+.3f} gap "
            f"{a['in_minus_out_accuracy_points_gain']:+.3f} | coef {a['coef_full'][-1]:+.5f} "
            f"folds {fc['mean']:+.5f} [{fc['min']:+.5f}, {fc['max']:+.5f}] negative "
            f"{fc['negative_folds']}/{fc['folds']}"
        )
    first = next(iter(res["added_term"].values()))
    lines.append(f"CONTROL intercept-only {first['intercept_only_oos']}")
    lines.append(f"CONTROL base {first['baseline_oos']}")
    for w, m in res["matched_null"].items():
        r, n, p = m["real"], m["null"], m["percentile"]
        s = r["served"]
        lines.append(
            f"NULL {w}: sd {m['shape']['sd']:.2f} nonzero {m['shape']['nonzero_share']:.3f} | "
            f"real coef/SD {r['full_coefficient_per_sd']:+.4f} changed {s['changed']} "
            f"{s['arm_right']}-{s['ref_right']} acc {s['acc']['effect']:+.3f} P+ "
            f"{s['acc']['pp']:.3f} ll {s['ll']['effect']:+.6f} P+ {s['ll']['pp']:.3f} | "
            f"percentile acc {p['acc']:.1f} ll {p['ll']:.1f} coef {p['coef']:.1f} | null ll "
            f"{n['ll']['mean']:+.6f} [{n['ll']['q05']:+.6f}, {n['ll']['q95']:+.6f}] acc "
            f"{n['acc']['mean']:+.3f} [{n['acc']['q05']:+.3f}, {n['acc']['q95']:+.3f}] coef sd "
            f"{m['null_coefficient_sd']:.4f} | null P+ notable acc "
            f"{m['null_share_notable']['acc']:.3f} ll {m['null_share_notable']['ll']:.3f}"
        )
    det = res["matched_null"]["early"]["detectable"]
    for k, c in det["smallest"].items():
        lines.append(
            f"DETECT by {k}: {c['size_logit_per_sd']:.4f} per SD = "
            f"{c['true_accuracy_points']:+.2f} pts, ll {c['true_log_loss_gain']:+.5f}"
        )
    c = det["at_nfl_coefficient"]
    lines.append(
        f"DETECT at NFL per-point coefficient: {c['size_logit_per_sd']:.4f} per SD = "
        f"{c['true_accuracy_points']:+.2f} pts, seen by ll {c['share_seen_ll']:.3f} by acc "
        f"{c['share_seen_acc']:.3f}"
    )
    for c in det["grid"]:
        lines.append(
            f"  grid {c['size_logit_per_sd']:.4f} true {c['true_accuracy_points']:+.2f} "
            f"ll {c['share_seen_ll']:.3f} acc {c['share_seen_acc']:.3f}"
        )
    lines.append(f"cells {len(batch['cells'])}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] != ["report"]:
        compute()
    report()


if __name__ == "__main__":
    main()
