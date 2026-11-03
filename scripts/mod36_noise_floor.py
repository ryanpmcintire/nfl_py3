import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U8 = Path(__file__).resolve().parent / "mod23_unit8_span.py"
NS = {"__file__": str(U8), "__name__": "mod23_unit8_span_import"}
exec(compile(U8.read_text().rsplit("\nmain()", 1)[0], str(U8), "exec"), NS)

from nfl_ats import clv  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
OUT = u3.ROOT / "artifacts" / "mod36_noise"
SEED = 36
KS = [int(k) for k in sys.argv[2].split(",")]
N = int(sys.argv[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    path = OUT / "draws.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["k"], r["i"]))
    for k in KS:
        for i in range(N):
            if (k, i) in done:
                continue
            t0 = time.time()
            rng = np.random.default_rng([SEED, k, i])
            f = features.copy()
            new = tuple(f"noise_{j}" for j in range(k))
            for c in new:
                f[c] = rng.standard_normal(len(f))
            run = HARNESS["true_eval"](f, tuple(FULL) + new, BASE_ALPHA)
            run = run.loc[run["season"].ge(2020)]
            c = compare(run.loc[run.index.isin(base.index)], base)["tuesday_2020_2025"]
            r = {
                "k": k, "i": i, "d_acc": c["d_acc"], "pp": c["d_acc_season_boot"]["pp"],
                "ll_gain": c["ll_base"] - c["ll_arm"], "flips": c["flips"],
                "record": c["record"], "sec": time.time() - t0,
            }
            with path.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(r, flush=True)


main()
