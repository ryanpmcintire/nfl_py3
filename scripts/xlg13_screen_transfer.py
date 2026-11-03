import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

SEED = 13
DRAWS = 10000
PERCENTILES = (2.5, 97.5)
POPULATION = "nfl_opener"
METRIC = "accuracy"


def bootstrap(p, s, f, rng):
    k = len(p)
    out = []
    for _ in range(DRAWS):
        i = rng.integers(0, k, k)
        if len(set(p[i])) < 2 or len(set(s[i])) < 2:
            continue
        out.append(f(p[i], s[i])[0])
    out = np.array(out)
    return np.percentile(out, PERCENTILES).tolist(), float((out > 0).mean()), len(out)


def main():
    ap = argparse.ArgumentParser(
        description="Screen-to-served transfer: rank and Pearson correlation of pooled-screen "
        "NFL opener accuracy gains against served-model gains"
    )
    ap.add_argument("--screens", required=True, help="directory of pooled_screen JSON reports")
    ap.add_argument("--served", required=True, help="JSON map of lever name to served gain")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    served = json.loads(Path(args.served).read_text())
    names = list(served)
    p = np.array(
        [
            json.loads((Path(args.screens) / f"{n}.json").read_text())["populations"][POPULATION][
                METRIC
            ]["effect"]
            for n in names
        ]
    )
    s = np.array([served[n] for n in names])
    rng = np.random.default_rng(SEED)
    res = {"levers": names, "screen": p.tolist(), "served": s.tolist()}
    for nm, f in (("spearman", spearmanr), ("pearson", pearsonr)):
        lo_hi, ppos, kept = bootstrap(p, s, f, rng)
        res[nm] = {
            "value": float(f(p, s)[0]),
            "boot_interval": lo_hi,
            "probability_positive": ppos,
            "draws_kept": kept,
        }
    obs = spearmanr(p, s)[0]
    res["permutation_spearman_le_obs"] = float(
        np.mean(
            [spearmanr(p, np.array(q))[0] <= obs + 1e-9 for q in itertools.permutations(s.tolist())]
        )
    )
    res["sign_agreement"] = int(np.sum(np.sign(p) == np.sign(s)))
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
