from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_candidate as v4  # noqa: E402
from mod27_fenced_repairs_replay import FOUR, Fit, log_loss, paired, record  # noqa: E402

from nfl_ats import pick_probability_fit as ppf  # noqa: E402
from nfl_ats.pick_probability import (  # noqa: E402
    market_move_toward_home,
    signed_composition_flags,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts/inseason_refit"
PICK = v4.PICK
QW = v4.OUT / "opener_QW.parquet"
REPLAY = v4.OUT / "replay_2026.csv"
SCORED = (2021, 2022, 2023, 2024, 2025)
CHRONO = (2022, 2023, 2024, 2025)
LIVE = v4.LIVE_SEASON
LIVE_GRADED_FIRST_WEEK = 2


def find_key(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def walk(frame, seasons, inseason):
    p = np.full(len(frame), np.nan)
    coefs = {}
    for s in seasons:
        if not inseason:
            fit = Fit(FOUR, frame[frame["season"] < s])
            m = (frame["season"] == s).to_numpy()
            p[m] = fit.predict(frame[m])
            coefs[str(s)] = fit.natural()
            continue
        for w in sorted(frame.loc[frame["season"] == s, "week"].unique()):
            train = frame[(frame["season"] < s) | ((frame["season"] == s) & (frame["week"] < w))]
            m = ((frame["season"] == s) & (frame["week"] == w)).to_numpy()
            fit = Fit(FOUR, train)
            p[m] = fit.predict(frame[m])
            coefs[f"{s}-{int(w):02d}"] = fit.natural()
    return p, coefs


def reliability(p, y, bands):
    pick = np.where(p >= 0.5, p, 1 - p)
    hit = (p >= 0.5) == (y == 1)
    rows = []
    for lo, hi in bands:
        m = (pick >= lo) & (pick < hi)
        if m.any():
            rows.append(
                {
                    "band": f"{lo:.2f}-{hi:.2f}",
                    "games": int(m.sum()),
                    "mean_pick_probability": round(float(pick[m].mean()), 4),
                    "hit_rate": round(float(hit[m].mean()), 4),
                }
            )
    return rows


def compare(frame, pa, pb, bands, season_block=True):
    y = frame["home_covered"].to_numpy(float)
    changed = (pa >= 0.5) != (pb >= 0.5)
    out = {
        "games": int(len(frame)),
        "P_record": record(pa, y),
        "I_record": record(pb, y),
        "P_log_loss": float(log_loss(pa, y).mean()),
        "I_log_loss": float(log_loss(pb, y).mean()),
        "P_brier": float(((pa - y) ** 2).mean()),
        "I_brier": float(((pb - y) ** 2).mean()),
        "week_block": paired(frame, pa, pb, "week_block"),
        "changed_picks": int(changed.sum()),
        "changed_I_record": record(pb[changed], y[changed]) if changed.any() else "0-0",
        "changed_P_record": record(pa[changed], y[changed]) if changed.any() else "0-0",
        "per_season": {
            str(int(s)): {
                "P": record(pa[m], y[m]),
                "I": record(pb[m], y[m]),
            }
            for s in sorted(frame["season"].unique())
            for m in [(frame["season"] == s).to_numpy()]
        },
        "reliability_I": reliability(pb, y, bands),
        "reliability_P": reliability(pa, y, bands),
    }
    if season_block:
        out["season_block"] = paired(frame, pa, pb, "season")
    return out


def history_frame(model_column):
    pop = pd.read_parquet(PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    if model_column == "QW":
        qw = pd.read_parquet(QW)
        qw.index = qw.index.astype(str)
        pop["model_logit"] = v4.logit(
            qw["home_cover_probability_at_open"].reindex(pop["game_id"]).to_numpy(float)
        )
    pop["week_block"] = pop["season"] * 100 + pop["week"]
    return pop


def live_frame():
    table = pd.read_parquet(v4.TABLE)
    table["game_id"] = table["game_id"].astype(str)
    rep = pd.read_csv(REPLAY)
    rep["game_id"] = rep["game_id"].astype(str)
    rep = rep.drop_duplicates("game_id").set_index("game_id")
    parts = []
    for week in v4.LIVE_WEEKS:
        d = v4.served_dir(table, week)
        pr = pd.read_csv(d / "predictions.csv", low_memory=False)
        pr = pr.loc[pr["method"].eq("market_residual")].copy()
        pr["game_id"] = pr["game_id"].astype(str)
        pr = pr.drop_duplicates("game_id")
        pr["served_dir"] = d.name
        parts.append(pr)
    live = pd.concat(parts, ignore_index=True)
    live = live.loc[live["game_id"].isin(rep.index)].reset_index(drop=True)
    schedules, _ = ppf.load_snapshot(ppf.latest_snapshot(DATA / "raw"))
    flags = signed_composition_flags(
        live,
        schedules,
        incidents=ppf._arrest_incidents(DATA),
        forecasts_tuesday_noon=ppf._forecast_temperatures(DATA),
        protection_back_side=ppf._protection_back_side(schedules, DATA),
    ).set_index("game_id")
    meta = json.loads((PICK / "metadata.json").read_text(encoding="utf-8"))
    counted = list(find_key(meta, "counted_flag_columns"))
    version = json.loads((ROOT / "artifacts/active_pick_probability.json").read_text())[
        "market_move_feature_version"
    ]
    move = market_move_toward_home(live, DATA, feature_version=version)
    move["game_id"] = move["game_id"].astype(str)
    move = move.drop_duplicates("game_id").set_index("game_id")
    ids = live["game_id"]
    frame = pd.DataFrame(
        {
            "game_id": ids,
            "season": LIVE,
            "week": live["week"].astype(int).to_numpy(),
            "served_dir": live["served_dir"].to_numpy(),
        }
    )
    served_logit = v4.logit(pd.to_numeric(live["home_cover_probability"], errors="coerce"))
    delta = rep["QW_model_logit_delta"].reindex(ids).to_numpy(float)
    frame["model_logit"] = served_logit + np.nan_to_num(delta)
    frame[ppf.FLAG_SUM_COLUMN] = (
        flags[counted].reindex(ids).astype(float).fillna(0.0).sum(axis=1).to_numpy()
    )
    if ppf.MOVE_COLUMN in move.columns and ppf.MOVE_AVAILABLE_COLUMN in move.columns:
        frame[ppf.MOVE_COLUMN] = move[ppf.MOVE_COLUMN].reindex(ids).astype(float).fillna(0.0).to_numpy()
        frame[ppf.MOVE_AVAILABLE_COLUMN] = (
            move[ppf.MOVE_AVAILABLE_COLUMN].reindex(ids).astype(float).fillna(0.0).to_numpy()
        )
    else:
        raw = pd.to_numeric(move[ppf.MARKET_MOVE_COLUMN].reindex(ids), errors="coerce")
        frame[ppf.MOVE_AVAILABLE_COLUMN] = raw.notna().astype(float).to_numpy()
        frame[ppf.MOVE_COLUMN] = raw.fillna(0.0).to_numpy()
    result = table.drop_duplicates("game_id").set_index("game_id")["result"].reindex(ids)
    home_line = rep["home_line"].reindex(ids).to_numpy(float)
    edge = pd.to_numeric(result, errors="coerce").to_numpy(float) + home_line
    frame["home_line"] = home_line
    frame["pool_edge"] = edge
    frame["home_covered"] = (edge > 0).astype(float)
    frame["card_home"] = rep["card_home"].reindex(ids).to_numpy(float)
    frame["week_block"] = frame["season"] * 100 + frame["week"]
    coverage = {
        "games": int(len(frame)),
        "graded": int(np.isfinite(edge).sum()),
        "pool_pushes": int((edge == 0).sum()),
        "move_available": int(frame[ppf.MOVE_AVAILABLE_COLUMN].sum()),
        "flag_nonzero": int(frame[ppf.FLAG_SUM_COLUMN].ne(0).sum()),
        "qw_delta_missing": int(np.isnan(delta).sum()),
        "model_missing": int(np.isnan(frame["model_logit"]).sum()),
        "counted_flags": counted,
        "move_version": version,
        "move_columns": list(move.columns),
    }
    return frame, coverage


def main():
    parser = argparse.ArgumentParser(description="In-season refit of the four-term stage.")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    coef = json.loads((PICK / "coefficients.json").read_text(encoding="utf-8"))
    bands = [(b["lower"], b["upper"]) for b in coef["confidence_bands"]]
    results = {}

    stored = history_frame("stored")
    p_stored, _ = walk(stored, CHRONO, inseason=False)
    m = stored["season"].isin(CHRONO).to_numpy()
    results["reproduction_chronological_max_abs"] = float(
        np.nanmax(np.abs(p_stored[m] - stored["chronological_home_probability"].to_numpy(float)[m]))
    )

    hist = history_frame("QW")
    qw_all = pd.read_parquet(QW)
    results["opener_pushes_2021_2025"] = int(
        (qw_all["margin_vs_open"].eq(0.0) & qw_all["season"].isin(SCORED)).sum()
    )
    p_prior, prior_coefs = walk(hist, SCORED, inseason=False)
    p_in, in_coefs = walk(hist, SCORED, inseason=True)
    sc = hist["season"].isin(SCORED).to_numpy()
    results["history_2021_2025"] = compare(
        hist.loc[sc].reset_index(drop=True), p_prior[sc], p_in[sc], bands
    )
    results["history_2021_2025"]["P_coefficients"] = prior_coefs
    ml = {k: v["model_logit"] for k, v in in_coefs.items()}
    mv = {k: v[ppf.MOVE_COLUMN] for k, v in in_coefs.items()}
    results["history_2021_2025"]["I_coefficient_ranges"] = {
        "model_logit": [min(ml.values()), max(ml.values())],
        "move_toward_home": [min(mv.values()), max(mv.values())],
        "weekly_fits": len(in_coefs),
    }
    hist_out = hist.loc[sc, ["game_id", "season", "week", "home_covered"]].copy()
    hist_out["P"] = p_prior[sc]
    hist_out["I"] = p_in[sc]
    hist_out["grade"] = "opener_2021_2025"

    live, coverage = live_frame()
    results["live_coverage"] = coverage
    usable = live.loc[np.isfinite(live["pool_edge"]) & live["pool_edge"].ne(0.0)].copy()
    usable = usable.loc[np.isfinite(usable["model_logit"])].reset_index(drop=True)
    cols = ["game_id", "season", "week", "week_block", "home_covered", *FOUR]
    combined = pd.concat([hist[cols], usable[cols]], ignore_index=True)
    lp, lp_coefs = walk(combined, (LIVE,), inseason=False)
    li, li_coefs = walk(combined, (LIVE,), inseason=True)
    lm = (combined["season"].eq(LIVE) & combined["week"].ge(LIVE_GRADED_FIRST_WEEK)).to_numpy()
    graded = combined.loc[lm].reset_index(drop=True)
    results["live_2026_w2_10_pool_line"] = compare(graded, lp[lm], li[lm], bands, season_block=False)
    card = usable.set_index("game_id")["card_home"].reindex(graded["game_id"]).to_numpy(float)
    y = graded["home_covered"].to_numpy(float)
    results["live_2026_w2_10_pool_line"]["card_record"] = record(card, y)
    results["live_2026_w2_10_pool_line"]["P_side_agrees_with_card"] = int(
        ((lp[lm] >= 0.5) == (card >= 0.5)).sum()
    )
    results["live_2026_w2_10_pool_line"]["P_coefficients"] = lp_coefs
    results["live_2026_w2_10_pool_line"]["I_last_week_coefficients"] = li_coefs[
        max(li_coefs)
    ]
    live_out = graded[["game_id", "season", "week", "home_covered"]].copy()
    live_out["P"] = lp[lm]
    live_out["I"] = li[lm]
    live_out["card"] = card
    live_out["grade"] = "pool_line_2026_w2_10"
    pd.concat([hist_out, live_out], ignore_index=True).to_parquet(OUT / "per_game.parquet")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    brief = {
        k: (
            {kk: vv for kk, vv in v.items() if "coefficients" not in kk and "reliability" not in kk}
            if isinstance(v, dict)
            else v
        )
        for k, v in results.items()
    }
    print(json.dumps(brief, default=str))


if __name__ == "__main__":
    main()
