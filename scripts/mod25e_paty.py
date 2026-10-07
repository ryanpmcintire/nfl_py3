import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
OUTD = REPO / "artifacts" / "mod25e3" / "paty"
FIT = OUTD / "fit.json"
TRAIN = tuple(range(2009, 2018))
RULE_YEAR = 2015


def enabled():
    return os.environ.get("PATY") == "1"


def regime(year):
    return "post" if year >= RULE_YEAR else "pre"


def cmd_fit():
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(TRAIN)
    full = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    m = full["play_type"].notna() & (full["play_type"] != "no_play")
    pl = full.loc[m].copy()
    grp = pl.groupby("game_id", sort=False)
    nt = grp["play_type"].shift(-1)
    nn = grp["play_type_nfl"].shift(-1)
    npost = grp["posteam"].shift(-1)
    gain = grp["posteam_score_post"].shift(-1) - grp["posteam_score"].shift(-1)
    td = (pl["touchdown"] == 1) & pl["play_type"].isin(["run", "pass"]) & ((pl["posteam_score_post"] - pl["posteam_score"]) >= 6)
    is_pat = (nt == "extra_point") | (nn == "PAT2")
    rows = td & is_pat & (npost == pl["posteam"]) & pl["score_differential"].notna()
    d = pl.loc[rows]
    two = (nn.loc[rows] == "PAT2").to_numpy()
    g = gain.loc[rows].to_numpy()
    seas = d["season"].to_numpy()
    out = {"train": list(TRAIN), "rule_year": RULE_YEAR}
    for name, mk in (("pre", seas < RULE_YEAR), ("post", seas >= RULE_YEAR)):
        k = mk & ~two
        t = mk & two
        out[name] = dict(kick_good=float(np.mean(g[k] == 1)), n_kick=int(k.sum()), two_share=float(t.sum() / mk.sum()), two_good=float(np.mean(g[t] == 2)), n_two=int(t.sum()), n_td=int(mk.sum()))
    OUTD.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def pat_cell(fn):
    seen = set()
    todo = [fn]
    while todo:
        f = todo.pop(0)
        if id(f) in seen or not callable(f) or not hasattr(f, "__code__"):
            continue
        seen.add(id(f))
        names = f.__code__.co_freevars
        cells = f.__closure__ or ()
        if "pat" in names:
            return cells[names.index("pat")]
        todo.extend(c.cell_contents for c in cells if callable(getattr(c, "cell_contents", None)))
    raise RuntimeError("no pol in the wrapper chain closes over pat")


def install_paty():
    import mod25d_variance as dv
    import mod25e_ot as ot

    fit = json.loads(FIT.read_text(encoding="utf-8"))
    base = dv._G["pol"]
    cell = pat_cell(base)
    cur = {"y": None}
    orig_ps = dv.d_play_season
    spec = os.environ["OTY"]

    def ps(task):
        cur["y"] = ot.season_year(spec, task[1])
        return orig_ps(task)

    dv.d_play_season = ps
    served = cell.cell_contents
    by = {r: (np.clip(served[0] * fit[r]["two_share"] / served[4], 0.0, 1.0), fit[r]["kick_good"], served[2], served[3], served[4]) for r in ("pre", "post")}

    def pol(dn, distance, yardline, score_diff, qtr, clock_val, drawn):
        cell.cell_contents = by[regime(cur["y"])]
        return base(dn, distance, yardline, score_diff, qtr, clock_val, drawn)

    dv._G["pol"] = pol


if __name__ == "__main__":
    cmd_fit()
