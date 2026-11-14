import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, pearsonr, spearmanr

SEED = 13
DRAWS = 10000
PERCENTILES = (2.5, 97.5)
POPULATION = "nfl_opener"
METRIC = "accuracy"
SIGN_NULL = 0.5
TOLERANCE = 1e-9
KINDS = ("A", "B")
METRICS = ("accuracy_points", "log_loss_improvement")
SUBSET_KEYS = ("cheap_type", "served_type")


def origin_slope(x, y):
    return (float(np.sum(x * y) / np.sum(x * x)),)


def ols_slope(x, y):
    return (float(np.polyfit(x, y, 1)[0]),)


def bootstrap(p, s, f, rng, need_spread=True):
    k = len(p)
    out = []
    for _ in range(DRAWS):
        i = rng.integers(0, k, k)
        if need_spread and (len(set(p[i])) < 2 or len(set(s[i])) < 2):
            continue
        if not need_spread and not np.any(p[i]):
            continue
        out.append(f(p[i], s[i])[0])
    out = np.array(out)
    return np.percentile(out, PERCENTILES).tolist(), float((out > 0).mean()), len(out), out


def stat_block(x, y, f, rng, need_spread=True):
    lo_hi, ppos, kept, draws = bootstrap(x, y, f, rng, need_spread)
    block = {
        "value": float(f(x, y)[0]),
        "boot_interval": lo_hi,
        "probability_positive": ppos,
        "draws_kept": kept,
    }
    return block, draws


def permutation_ge(x, y, rng):
    obs = spearmanr(x, y)[0]
    exact = math.factorial(len(y)) <= DRAWS
    if exact:
        perms = (np.array(q) for q in itertools.permutations(y.tolist()))
    else:
        perms = (rng.permutation(y) for _ in range(DRAWS))
    return float(np.mean([spearmanr(x, q)[0] >= obs - TOLERANCE for q in perms])), exact


def has_both(row, metric):
    cell = row.get(metric) or {}
    return cell.get("cheap") is not None and cell.get("served") is not None


def sign_counts(rows, metric):
    x = np.array([r[metric]["cheap"] for r in rows], dtype=float)
    y = np.array([r[metric]["served"] for r in rows], dtype=float)
    pos = x > 0
    res = {
        "pairs": [r["id"] for r in rows],
        "n_pairs": len(rows),
        "cheap": x.tolist(),
        "served": y.tolist(),
        "sign_agreement": int(np.sum(np.sign(x) == np.sign(y))),
        "cheap_positive": int(pos.sum()),
        "cheap_positive_served_positive": int(np.sum(pos & (y > 0))),
        "served_positive": int(np.sum(y > 0)),
    }
    if res["cheap_positive"]:
        res["binomial_tail_stayed_positive"] = float(
            binom.cdf(res["cheap_positive_served_positive"], res["cheap_positive"], SIGN_NULL)
        )
    return res, x, y


def summarise(rows, metric, rng, detectable):
    res, x, y = sign_counts(rows, metric)
    k = len(rows)
    if k < 2:
        res["note"] = "one pair: no slope or rank statistic"
        return res
    res["slope_through_origin"], draws = stat_block(x, y, origin_slope, rng, need_spread=False)
    if detectable:
        value = res["slope_through_origin"]["value"]
        res["expected_served_at_detectable_gain"] = [
            {
                "cheap_gain": g,
                "served": value * g,
                "boot_interval": np.percentile(draws * g, PERCENTILES).tolist(),
            }
            for g in detectable["range"]
        ]
    if k < 3:
        res["note"] = "two pairs: rank and product-moment correlation are 1 or -1 by construction"
        return res
    for nm, f in (("spearman", spearmanr), ("pearson", pearsonr), ("ols_slope", ols_slope)):
        res[nm], _ = stat_block(x, y, f, rng)
    res["ols_intercept"] = float(np.polyfit(x, y, 1)[1])
    res["permutation_spearman_ge_obs"], res["permutation_exact"] = permutation_ge(x, y, rng)
    return res


def pairs_report(table):
    pairs = table["pairs"]
    detectable = table.get("detectable_cheap_gain", {})
    res = {"declared": table.get("declared"), "kinds": {}, "subsets": {}, "excluded": []}
    for a, kind in enumerate(KINDS):
        for b, metric in enumerate(METRICS):
            rows = [
                r for r in pairs if r["kind"] == kind and r.get("admitted") and has_both(r, metric)
            ]
            if not rows:
                continue
            rng = np.random.default_rng([SEED, a, b])
            res["kinds"].setdefault(kind, {})[metric] = summarise(
                rows, metric, rng, detectable.get(metric) if kind == KINDS[0] else None
            )
    for key in SUBSET_KEYS:
        for metric in METRICS:
            rows = [
                r
                for r in pairs
                if r["kind"] == KINDS[0]
                and has_both(r, metric)
                and (r.get("admitted") or (r.get("subset_only") and key == SUBSET_KEYS[0]))
            ]
            for value in sorted({r[key] for r in rows}):
                group = [r for r in rows if r[key] == value]
                counts, _, _ = sign_counts(group, metric)
                res["subsets"].setdefault(key, {}).setdefault(value, {})[metric] = counts
    res["excluded"] = [
        {"id": r["id"], "kind": r["kind"], "reason": r.get("reason")}
        for r in pairs
        if not r.get("admitted")
    ]
    return res


def legacy_report(screens, served_path):
    served = json.loads(Path(served_path).read_text())
    names = list(served)
    p = np.array(
        [
            json.loads((Path(screens) / f"{n}.json").read_text())["populations"][POPULATION][
                METRIC
            ]["effect"]
            for n in names
        ]
    )
    s = np.array([served[n] for n in names])
    rng = np.random.default_rng(SEED)
    res = {"levers": names, "screen": p.tolist(), "served": s.tolist()}
    for nm, f in (("spearman", spearmanr), ("pearson", pearsonr)):
        res[nm], _ = stat_block(p, s, f, rng)
    obs = spearmanr(p, s)[0]
    res["permutation_spearman_le_obs"] = float(
        np.mean(
            [
                spearmanr(p, np.array(q))[0] <= obs + TOLERANCE
                for q in itertools.permutations(s.tolist())
            ]
        )
    )
    res["sign_agreement"] = int(np.sum(np.sign(p) == np.sign(s)))
    return res


def main():
    ap = argparse.ArgumentParser(
        description="Screen-to-served transfer: how a gain on a cheaper harness or a non-served "
        "base predicts the gain on the served NFL model, from pairs of existing results"
    )
    ap.add_argument("--pairs", default=None, help="JSON pair table (kinds A and B)")
    ap.add_argument("--screens", default=None, help="directory of pooled_screen JSON reports")
    ap.add_argument("--served", default=None, help="JSON map of lever name to served gain")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.pairs:
        res = pairs_report(json.loads(Path(args.pairs).read_text(encoding="utf-8")))
    elif args.screens and args.served:
        res = legacy_report(args.screens, args.served)
    else:
        ap.error("give --pairs, or --screens with --served")
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
