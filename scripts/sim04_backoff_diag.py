import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim04_engine as eng

TOTAL = {"l0": 0, "l1": 0, "l2": 0, "l3": 0}
Q4_LE300 = {"l0": 0, "l1": 0, "l2": 0, "l3": 0, "n": 0}
Q2_LE120 = {"l0": 0, "l1": 0, "l2": 0, "l3": 0, "n": 0}

_orig_pick_index = eng.pick_index


def instrumented(rng, tables, k0, k1, k2, k3, min_cell_n):
    level_used = "l3"
    for name, level, key in (("l0", tables["l0"], k0), ("l1", tables["l1"], k1), ("l2", tables["l2"], k2)):
        idxs = level.get(key)
        if idxs is not None and len(idxs) >= min_cell_n:
            level_used = name
            break
    TOTAL[level_used] += 1
    tb_fine = k0[4]
    if tb_fine in (5, 6):
        Q4_LE300[level_used] += 1
        Q4_LE300["n"] += 1
    if tb_fine == 2:
        Q2_LE120[level_used] += 1
        Q2_LE120["n"] += 1
    return _orig_pick_index(rng, tables, k0, k1, k2, k3, min_cell_n)


eng.pick_index = instrumented

tables = eng.build_tables(eng.TRAIN_SEASONS)
rng = np.random.default_rng(eng.RNG_SEED)
frame = eng.simulate(3000, rng, tables)

total_n = sum(TOTAL.values())
print("overall", {k: v / total_n for k, v in TOTAL.items()}, "n=", total_n)
q4n = Q4_LE300["n"]
print("q4_le300", {k: Q4_LE300[k] / q4n for k in ("l0", "l1", "l2", "l3")}, "n=", q4n)
q2n = Q2_LE120["n"]
print("q2_le120", {k: Q2_LE120[k] / q2n for k in ("l0", "l1", "l2", "l3")}, "n=", q2n)
