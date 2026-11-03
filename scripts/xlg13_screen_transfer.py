import itertools

import numpy as np
from scipy.stats import pearsonr, spearmanr

n = ["ridgecv", "enet", "accalpha", "huber"]
p = np.array([0.93, 1.40, 0.80, 0.67])
s = np.array([-0.73, -3.19, -1.80, -0.87])
rng = np.random.default_rng(13)


def st(f):
    o = []
    for _ in range(10000):
        i = rng.integers(0, 4, 4)
        if len(set(p[i])) < 2 or len(set(s[i])) < 2:
            continue
        o.append(f(p[i], s[i])[0])
    o = np.array(o)
    return np.percentile(o, [2.5, 97.5]), (o > 0).mean(), len(o)


for nm, f in (("spearman", spearmanr), ("pearson", pearsonr)):
    print(nm, f(p, s)[0], st(f))
perm = []

print(
    "perm spearman <=obs",
    np.mean(
        [
            spearmanr(p, q)[0] <= spearmanr(p, s)[0] + 1e-9
            for q in map(np.array, itertools.permutations(s))
        ]
    ),
)
