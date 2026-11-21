import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lead101_delta_read", HERE / "lead101_delta_read.py")
L101 = importlib.util.module_from_spec(spec)
sys.modules["lead101_delta_read"] = L101
spec.loader.exec_module(L101)
L99 = L101.L99

ROOT = L99.ROOT
RUNS = L99.ART / "runs"
REPEATS = L99.ART / "repeats"
BLIND = L99.ART / "blinded"
COEFFICIENTS = ROOT / "artifacts" / "pick_probability" / "20261111T011859Z" / "coefficients.json"
MOVE_TERM = "move_toward_home"
OUT = ROOT / "artifacts" / "lead107_dispersion_error"
FAMILY = "lead107_read_dispersion_weight"
MIN_READS = 3
BOOTSTRAP_DRAWS = 10000
PERMUTATION_DRAWS = 10000
INTERVAL_PERCENTILES = (2.5, 97.5)
SEED = 107
ARM_WORDS = {
    "named": "repeated reads of the same packet with team and player names",
    "blinded": "repeated reads of the same packet with names replaced by role labels",
}


def instant(s):
    for fmt in ("%Y%m%dT%H%M%S%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt).replace(tzinfo=UTC))
        except ValueError:
            pass
    return L101.utc(s)


def logit(p):
    p = float(p)
    return float(np.log(p / (1 - p)))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_sets():
    sources = {}
    for f in sorted(REPEATS.glob("*.json")):
        d = load(f)
        if d.get("valid") and d.get("parsed_rows"):
            sources.setdefault(d["source_run"], []).append(d)
    store = load(BLIND / "blinded_packet.json")
    blinded = [load(f) for f in sorted((BLIND / "reads").glob("*.json"))]
    blinded = [d for d in blinded if d.get("valid") and d.get("parsed_rows")]
    sets = []
    for name, repeats in sources.items():
        src = load(RUNS / name)
        named = [src, *repeats]
        arms = {"named": [{r["game_id"]: r for r in d["parsed_rows"]} for d in named]}
        stamps = [instant(d["recorded_at_utc"]) for d in named]
        if store["source_packet_sha256"] == src["packet_sha256"]:
            mapped = [
                {store["mapping"][r["game_id"]]: r for r in d["parsed_rows"]}
                for d in blinded
                if d["source_packet_sha256"] == src["packet_sha256"]
            ]
            if mapped:
                arms["blinded"] = mapped
                stamps += [
                    instant(d["recorded_at_utc"])
                    for d in blinded
                    if d["source_packet_sha256"] == src["packet_sha256"]
                ]
        sets.append((name, src, arms, max(stamps)))
    return sets


def later_lines(src, start, quotes):
    by_game = {gid: g.sort_values("seen") for gid, g in quotes.groupby("game_id")}
    out = {}
    for g in src["packet"]["games"]:
        q = by_game.get(g["game_id"])
        deadline = L101.utc(g["deadline_et"])
        window = None if q is None else q.loc[q["seen"].gt(start) & q["seen"].lt(deadline)]
        if window is None or window.empty:
            continue
        last = window.groupby("book").tail(1)
        out[g["game_id"]] = {
            "pool_line": float(g["pool_line"]["home_spread"]),
            "later_line": float(last["line"].median()),
            "later_books": len(last),
            "later_seen_utc": last["seen"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "deadline_utc": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    return out


def rank_corr(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def split_gap(d, e):
    high = d > np.median(d)
    if high.all() or not high.any():
        return np.nan
    return float(e[high].mean() - e[~high].mean())


def interval(draws):
    d = draws[np.isfinite(draws)]
    low, high = np.percentile(d, INTERVAL_PERCENTILES)
    return {
        "interval_low": float(low),
        "interval_high": float(high),
        "probability_positive": float((d > 0).mean() + (d == 0).mean() / 2),
        "valid_draws": len(d),
    }


def arm_scores(frame, rng):
    d = frame["dispersion"].to_numpy()
    e = frame["error"].to_numpy()
    weeks = frame["week"].to_numpy()
    strata = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    rho = rank_corr(d, e)
    gap = split_gap(d, e)
    n = len(frame)
    boot_rho = np.full(BOOTSTRAP_DRAWS, np.nan)
    boot_gap = np.full(BOOTSTRAP_DRAWS, np.nan)
    for i in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, n)
        boot_rho[i] = rank_corr(d[idx], e[idx])
        boot_gap[i] = split_gap(d[idx], e[idx])
    null_rho = np.empty(PERMUTATION_DRAWS)
    null_gap = np.empty(PERMUTATION_DRAWS)
    for i in range(PERMUTATION_DRAWS):
        s = d.copy()
        for st in strata:
            s[st] = d[rng.permutation(st)]
        null_rho[i] = rank_corr(s, e)
        null_gap[i] = split_gap(s, e)
    high = d > np.median(d)
    above = e > np.median(e)
    return {
        "games": n,
        "weeks": len(strata),
        "reads_per_game": sorted({int(k) for k in frame["reads"]}),
        "mean_dispersion": float(d.mean()),
        "mean_error": float(e.mean()),
        "record": {
            "high_dispersion_games": int(high.sum()),
            "high_dispersion_above_median_error": int((high & above).sum()),
            "low_dispersion_games": int((~high).sum()),
            "low_dispersion_above_median_error": int((~high & above).sum()),
        },
        "spearman": {
            "effect": rho,
            **interval(boot_rho),
            "permutation_mean": float(np.nanmean(null_rho)),
            "permutation_p_upper": float((1 + np.sum(null_rho >= rho)) / (1 + PERMUTATION_DRAWS)),
        },
        "median_split_error_gap": {
            "effect": gap,
            **interval(boot_gap),
            "permutation_mean": float(np.nanmean(null_gap)),
            "permutation_p_upper": float((1 + np.sum(null_gap >= gap)) / (1 + PERMUTATION_DRAWS)),
        },
    }


def batch_payload(cells, looks):
    out = []
    for arm, c in cells.items():
        s = c["spearman"]
        r = c["record"]
        hits = r["high_dispersion_above_median_error"]
        high = r["high_dispersion_games"]
        wrong = s["interval_high"] < 0
        out.append(
            {
                "source": "artifacts/lead107_dispersion_error/summary.json",
                "league": "nfl",
                "season_start": L99.SEASON,
                "season_end": L99.SEASON,
                "classification": "unresolved_below_power",
                "category": "modeling",
                "family": FAMILY,
                "name": f"lead107_dispersion_vs_error_{arm}_spearman",
                "description": (
                    f"LEAD-107 unit 1, {ARM_WORDS[arm]}: Spearman correlation between the "
                    "between-read SD of the read logits and the absolute gap between the mean "
                    "read logit and the later line read at the pool line through the served move "
                    f"coefficient; positive means disagreeing reads miss more; {hits} of {high} "
                    "high-dispersion games had above-median error; "
                    f"within-week permutation p {s['permutation_p_upper']:.3f}; {looks} looks in "
                    "the family this unit"
                ),
                "plain_summary": (
                    "A language model read each game several times. This checks whether the games "
                    "where its repeated reads disagreed are the games where its read ended up "
                    "furthest from where the betting line later went."
                ),
                "effect": s["effect"],
                "effect_units": "correlation",
                "interval_low": s["interval_low"],
                "interval_high": s["interval_high"],
                "probability_positive": s["probability_positive"],
                "sample_games": c["games"],
                "sample_blocks": c["weeks"],
                "classification_evidence": (
                    f"{c['games']} games in {c['weeks']} week, game-cluster bootstrap; "
                    + (
                        "interval wholly below zero on one week, held as unresolved until a "
                        "second week repeats it"
                        if wrong
                        else "interval spans zero and no positive control exists, so no closing "
                        "ground applies"
                    )
                ),
            }
        )
    return {"family": FAMILY, "cells": out}


def main():
    ap = argparse.ArgumentParser(description="LEAD-107 read dispersion against read error")
    ap.parse_args()
    c = float(json.loads(COEFFICIENTS.read_text())["coefficients"][MOVE_TERM])
    quotes = L101.served_quotes()
    rows = []
    for name, src, arms, start in read_sets():
        lines = later_lines(src, start, quotes)
        for arm, reads in arms.items():
            for gid, ln in lines.items():
                vals = [logit(r[gid]["home_cover_probability"]) for r in reads if gid in r]
                if len(vals) < MIN_READS:
                    continue
                m = float(np.mean(vals))
                q = c * (ln["pool_line"] - ln["later_line"])
                rows.append(
                    {
                        "source_run": name,
                        "week": int(src["week"]),
                        "arm": arm,
                        "game_id": gid,
                        "reads": len(vals),
                        "mean_logit": m,
                        "dispersion": float(np.std(vals, ddof=1)),
                        **ln,
                        "later_read_logit": q,
                        "error": abs(m - q),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("no game with enough stored reads and a later quote")
    rng = np.random.default_rng(SEED)
    cells = {arm: arm_scores(g.reset_index(drop=True), rng) for arm, g in frame.groupby("arm")}
    looks = 2 * len(cells)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_game.csv", index=False)
    summary = {
        "scored_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "move_coefficient": c,
        "move_coefficient_source": str(COEFFICIENTS.relative_to(ROOT)),
        "looks": looks,
        "family": FAMILY,
        "cells": cells,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps(batch_payload(cells, looks), indent=1), encoding="utf-8"
    )
    pd.set_option("display.width", 250)
    print("move coefficient", c)
    cols = ["arm", "game_id", "reads", "mean_logit", "dispersion", "pool_line", "later_line"]
    print(frame[[*cols, "later_books", "later_seen_utc", "later_read_logit", "error"]].to_string())
    print(json.dumps(cells, indent=1))
    print("looks", looks, "family", FAMILY)


if __name__ == "__main__":
    main()
