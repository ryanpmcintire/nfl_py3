import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import sim04_loso as base  # noqa: E402
import sim08_evaluate as ev  # noqa: E402

from nfl_ats.mass_preserving_lattice import (  # noqa: E402
    BAND_HALF_WIDTH,
    BAND_STEP,
    MAX_BAND,
    MIN_BAND_GAMES,
    tilted_atoms,
)

SERVED_PATH = REPO / "artifacts" / "pick_probability" / "20261111T011859Z" / "per_game.parquet"
OUT = REPO / "artifacts" / "sim30"
KEYS = (3, 7, 10, 14, 17)
SHAPE_BANDS = (
    ("0_2.5", 0.0, 2.5),
    ("3_3.5", 3.0, 3.5),
    ("4_6.5", 4.0, 6.5),
    ("7_9.5", 7.0, 9.5),
    ("10plus", 10.0, 1e9),
)
N_BOOT = 2000
SEED = 20261111


def lattice_masses(frame, pool):
    n = len(frame)
    key = np.zeros((n, len(KEYS)))
    push = np.zeros(n)
    for i, (s_, line) in enumerate(zip(frame["season"], frame["line"], strict=True)):
        p = pool[pool["season"] < s_]
        pl = p["spread_line"].to_numpy(float)
        pm = np.rint(p["result"].to_numpy(float))
        band = BAND_HALF_WIDTH
        while True:
            sel = np.abs(pl - line) <= band
            if int(sel.sum()) >= MIN_BAND_GAMES or band >= MAX_BAND:
                break
            band = min(band + BAND_STEP, MAX_BAND)
        values, counts = np.unique(pm[sel], return_counts=True)
        mass, _ = tilted_atoms(values, counts.astype(float), float(line), float(line))
        sgn = 1.0 if line >= 0 else -1.0
        for j, k in enumerate(KEYS):
            key[i, j] = mass[np.abs(values - sgn * k) < 1e-9].sum()
        push[i] = mass[np.abs(values - line) < 1e-9].sum()
    return key, push


def gboot(d, rng, mask):
    x = d[mask]
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    b = x[idx].mean(1)
    return {
        "effect": float(x.mean()),
        "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
        "probability_positive": float(np.mean(b > 0.0)),
        "n_games": int(mask.sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    frame = base.load_games()
    g = pd.read_parquet(base.GAME_FEATURES_PATH)
    g = g.loc[g["game_type"] == "REG"]
    pool = g.loc[
        g["spread_line"].notna() & g["result"].notna(), ["season", "spread_line", "result"]
    ]
    S = ev.load_arm("main", "primary", frame)
    line = frame["line"].to_numpy(float)
    result = frame["result"].to_numpy(float)
    outcome = frame["outcome"].to_numpy()
    sgn = np.where(line >= 0, 1.0, -1.0)
    fav_sim = S["margins"] * sgn[:, None]
    fav_real = result * sgn
    sim_key = np.column_stack([(np.abs(fav_sim - k) < 1e-9).mean(1) for k in KEYS])
    real_key = np.column_stack([(np.abs(fav_real - k) < 1e-9).astype(float) for k in KEYS])
    lat_key, lat_push = lattice_masses(frame, pool)
    sim_push = (np.abs(S["margins"] - line[:, None]) < 1e-9).mean(1)
    real_push = (outcome == "push").astype(float)
    absl = np.abs(line)
    shape = {}
    for name, lo, hi in SHAPE_BANDS:
        m = (absl >= lo - 1e-9) & (absl <= hi + 1e-9)
        cell = {"n_games": int(m.sum())}
        for j, k in enumerate(KEYS):
            cell[str(k)] = {
                "sim": float(sim_key[m, j].mean()),
                "real": float(real_key[m, j].mean()),
                "lattice": float(lat_key[m, j].mean()),
                "sim_minus_real": gboot(sim_key[:, j] - real_key[:, j], rng, m),
                "lattice_minus_real": gboot(lat_key[:, j] - real_key[:, j], rng, m),
                "sim_minus_lattice": gboot(sim_key[:, j] - lat_key[:, j], rng, m),
            }
        shape[name] = cell
    whole = np.abs(line - np.rint(line)) < 1e-9
    pushcell = {
        "n_games": int(whole.sum()),
        "sim": float(sim_push[whole].mean()),
        "real": float(real_push[whole].mean()),
        "lattice": float(lat_push[whole].mean()),
        "sim_minus_real": gboot(sim_push - real_push, rng, whole),
        "lattice_minus_real": gboot(lat_push - real_push, rng, whole),
    }
    served_df = pd.read_parquet(SERVED_PATH)
    served_df = served_df.set_index("game_id")
    has = frame["game_id"].isin(served_df.index).to_numpy()
    sv = served_df.reindex(frame["game_id"])
    sv_cov = sv["home_cover_probability_excluding_push_at_open"].to_numpy(float)
    sv_push = sv["push_probability_at_open"].to_numpy(float)
    sv_loss = sv["home_loss_probability_at_open"].to_numpy(float)
    c, pu, lo_ = ev.sim_counts(S["margins"], line)
    n = S["draws"]
    sim3 = ((c + 0.5) / (n + 1.5), (pu + 0.5) / (n + 1.5), (lo_ + 0.5) / (n + 1.5))
    sim_cond = (c + 0.5) / (c + lo_ + 1.0)
    mk3 = ev.market_hist_three_way(frame)
    mk_cond = mk3[0] / np.clip(mk3[0] + mk3[2], base.EPS, None)
    sv_cond = sv_cov / np.clip(sv_cov + sv_loss, base.EPS, None)
    sv_cov, sv_push, sv_loss, sv_cond = (
        np.nan_to_num(a, nan=0.5) for a in (sv_cov, sv_push, sv_loss, sv_cond)
    )
    inv, nb = ev.blocks_of(frame)
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)

    def two_way(cond):
        p = np.clip(np.where(y > 0.5, cond, 1 - cond), base.EPS, 1 - base.EPS)
        return -np.log(p)

    reads3 = {
        "sim": base.three_way_score(*sim3, outcome)[0],
        "served": base.three_way_score(sv_cov, sv_push, sv_loss, outcome)[0],
        "market": base.three_way_score(*mk3, outcome)[0],
    }
    reads2 = {"sim": two_way(sim_cond), "served": two_way(sv_cond), "market": two_way(mk_cond)}
    grade = {"n_common": int(has.sum()), "n_blocks": nb}
    mask3 = has
    mask2 = has & nonpush
    pairs = (
        ("served_vs_sim", "served", "sim"),
        ("market_vs_sim", "market", "sim"),
        ("market_vs_served", "market", "served"),
    )
    for nm, a, b in pairs:
        grade[nm + "_3way"] = ev.boot_mean(reads3[a] - reads3[b], inv, nb, rng, mask3)
        grade[nm + "_2way"] = ev.boot_mean(reads2[a] - reads2[b], inv, nb, rng, mask2)
    grade["mean_3way_ll"] = {k: float(v[mask3].mean()) for k, v in reads3.items()}
    grade["mean_2way_ll"] = {k: float(v[mask2].mean()) for k, v in reads2.items()}
    wp = has & whole
    pm = wp & (outcome == "push")
    grade["push_pricing_whole_lines"] = {
        "n_games": int(wp.sum()),
        "n_push_games": int(pm.sum()),
        "mean_push_prob": {
            "sim": float(sim3[1][wp].mean()),
            "served": float(sv_push[wp].mean()),
            "market": float(mk3[1][wp].mean()),
            "real_rate": float(pm.sum() / wp.sum()),
        },
        "served_minus_sim_3way_ll": ev.boot_mean(
            reads3["served"] - reads3["sim"], inv, nb, rng, wp
        ),
        "market_minus_sim_3way_ll": ev.boot_mean(
            reads3["market"] - reads3["sim"], inv, nb, rng, wp
        ),
    }
    sim_side = sim_cond > 0.5
    sv_side = sv_cond > 0.5
    dec = has & nonpush & (sim_side != sv_side) & (c != lo_)
    sim_right = int((sim_side[dec] == (y[dec] > 0.5)).sum())
    grade["decisive_sim_vs_served"] = {
        "n_decisive": int(dec.sum()),
        "sim_correct": sim_right,
        "served_correct": int(dec.sum()) - sim_right,
        "binom_p_two_sided": float(binomtest(sim_right, int(dec.sum()), 0.5).pvalue)
        if dec.sum()
        else None,
        "n_all_nonpush_common": int(mask2.sum()),
        "sim_pick_accuracy_all": float((sim_side[mask2] == (y[mask2] > 0.5)).mean()),
        "served_pick_accuracy_all": float((sv_side[mask2] == (y[mask2] > 0.5)).mean()),
    }
    sv8 = base.baseline_three_way(frame)
    ll_sv8 = base.three_way_score(*sv8, outcome)[0]
    whole_all = np.abs(line - np.rint(line)) < 1e-9
    push_all = whole_all & (outcome == "push")
    grade["sim08_reproduction_1537"] = {
        "served_sim08_minus_sim_3way": ev.boot_mean(ll_sv8 - reads3["sim"], inv, nb, rng),
        "n_games": len(frame),
        "push_lines_whole_n": int(whole_all.sum()),
        "push_games_n": int(push_all.sum()),
        "served_sim08_minus_sim_3way_whole_lines": ev.boot_mean(
            ll_sv8 - reads3["sim"], inv, nb, rng, whole_all
        ),
        "market_minus_sim_3way_whole_lines": ev.boot_mean(
            reads3["market"] - reads3["sim"], inv, nb, rng, whole_all
        ),
        "served_sim08_minus_sim_3way_push_games": ev.boot_mean(
            ll_sv8 - reads3["sim"], inv, nb, rng, push_all
        ),
        "market_minus_sim_3way_push_games": ev.boot_mean(
            reads3["market"] - reads3["sim"], inv, nb, rng, push_all
        ),
        "mean_push_prob_whole_lines": {
            "sim": float(sim3[1][whole_all].mean()),
            "served_sim08": float(sv8[1][whole_all].mean()),
            "market": float(mk3[1][whole_all].mean()),
            "real": float(push_all.sum() / whole_all.sum()),
        },
    }
    res = {
        "shape": shape,
        "push_mass_whole_lines": pushcell,
        "grade": grade,
        "n_games_total": len(frame),
    }
    (OUT / "sim30_margin_mass.json").write_text(json.dumps(res, indent=1))
    for name, cell in shape.items():
        for k in KEYS:
            c_ = cell[str(k)]
            print(
                name,
                cell["n_games"],
                k,
                round(c_["sim"], 3),
                round(c_["real"], 3),
                round(c_["lattice"], 3),
                [
                    round(c_["sim_minus_real"]["effect"], 3),
                    *[round(v, 3) for v in c_["sim_minus_real"]["ci95"]],
                ],
            )
    print(json.dumps(pushcell))
    print(json.dumps(grade, indent=1))


if __name__ == "__main__":
    main()
