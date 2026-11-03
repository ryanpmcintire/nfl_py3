import json
import math
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parents[1] / "artifacts" / "lead99_llm_read"
SEED = 20261103
N_BOOT = 10000


def logit(p):
    return math.log(p / (1 - p))


def rows_of(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["game_id"]: logit(r["home_cover_probability"]) for r in d["parsed_rows"]}


def main():
    base = ART / "runs" / "2026-week-09-tuesday-20261103T183210Z.json"
    reps = sorted((ART / "repeats").glob("2026-week-09-tuesday-*-repeat-*.json"))
    reads = [rows_of(base)] + [
        rows_of(p) for p in reps if json.loads(p.read_text(encoding="utf-8"))["valid"]
    ]
    ids = sorted(set.intersection(*[set(r) for r in reads]))
    m = np.array([[r[g] for g in ids] for r in reads])
    print("reads", m.shape[0], "games", m.shape[1])
    sd = m.std(axis=0, ddof=1)
    print("per-game between-read SD (logit):", dict(zip(ids, np.round(sd, 4), strict=True)))
    print("mean between-read SD logit", sd.mean())
    p = 1 / (1 + np.exp(-m))
    psd = p.std(axis=0, ddof=1)
    print("mean between-read SD prob", psd.mean(), "max", psd.max())
    print("between-game SD of mean logit", m.mean(axis=0).std(ddof=1))
    a = m[0]
    b = m[1:].mean(axis=0)
    r = np.corrcoef(a, b)[0, 1]
    rng = np.random.default_rng(SEED)
    n = len(ids)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if a[i].std() == 0 or b[i].std() == 0:
            continue
        bs.append(np.corrcoef(a[i], b[i])[0, 1])
    bs = np.array(bs)
    print(
        "split-half r",
        r,
        "boot 2.5/97.5",
        np.percentile(bs, [2.5, 97.5]),
        "P(r>0)",
        (bs > 0).mean(),
    )
    sb = 2 * r / (1 + r)
    print("spearman-brown", sb)
    views = sorted((ART / "runs").glob("*.json"))
    by_game = {}
    for v in views:
        for g, x in rows_of(v).items():
            by_game.setdefault(g, []).append(x)
    multi = {g: np.std(x, ddof=1) for g, x in by_game.items() if len(x) > 1}
    print(
        "across-view SD (logit), games with >1 view:",
        len(multi),
        "mean",
        np.mean(list(multi.values())),
    )


main()
