import json

import mod10_graph_blitz as bl
import mod24_ftn_turnover_luck as m
import numpy as np
import pandas as pd

PART = m.ROOT / "data" / "players" / "participation" / "raw" / "20260813T131635Z"
OUT = m.ROOT / "artifacts" / "mod10_graph_coverage"
MIN_HALF = 30
SEEDS = 20261103


def load():
    part = pd.concat(
        [
            pd.read_parquet(
                PART / f"season={s}" / "participation.parquet",
                columns=["game_id", "play_id", "defense_man_zone_type", "defense_coverage_type"],
            )
            for s in range(2022, 2026)
        ],
        ignore_index=True,
    )
    cols = [
        "game_id",
        "play_id",
        "season",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "qb_dropback",
        "epa",
    ]
    pbp = pd.concat(
        [pd.read_parquet(m.PBP_ROOT / f"season={s}", columns=cols) for s in range(2022, 2026)],
        ignore_index=True,
    )
    pbp = pbp[(pbp.season_type == "REG") & (pbp.qb_dropback == 1) & pbp.epa.notna()]
    pbp = pbp.assign(play_id=pbp.play_id.astype(float))
    part = part.assign(play_id=part.play_id.astype(float))
    t = pbp.merge(part, on=["game_id", "play_id"], how="inner")
    t["man"] = (t.defense_man_zone_type == "MAN_COVERAGE").astype(float)
    t["known_mz"] = t.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])
    t["c3"] = (t.defense_coverage_type == "COVER_3").astype(float)
    t["known_cv"] = t.defense_coverage_type.isin(
        ["COVER_0", "COVER_1", "COVER_2", "COVER_3", "COVER_4", "COVER_6"]
    )
    t["half"] = np.where(t.week % 2 == 1, "odd", "even")
    return t


def pack(a, b, rng, n_full):
    r, rho = bl.sb(a, b)
    ci, p = bl.boot_sb(a, b, rng)
    return {
        "sb": float(r),
        "rho": float(rho),
        "ci": ci,
        "p_above_gate_436": p,
        "n": len(a),
        "n_full": n_full,
    }


def share(t, known, col, by):
    s = (
        t[t[known]]
        .groupby(["season", by, "half"])[col]
        .agg(["mean", "size"])
        .unstack("half")
        .dropna()
    )
    s = s[(s[("size", "odd")] >= MIN_HALF) & (s[("size", "even")] >= MIN_HALF)]
    return (
        s[("mean", "odd")].to_numpy(),
        s[("mean", "even")].to_numpy(),
        float((s[("size", "odd")] + s[("size", "even")]).median()),
    )


def gap(t):
    s = t[t.known_mz].groupby(["season", "posteam", "half", "man"]).epa.agg(["mean", "size"])
    s = s.unstack(["half", "man"]).dropna()
    ok = np.ones(len(s), bool)
    for h in ("odd", "even"):
        for v in (0.0, 1.0):
            ok &= (s[("size", h, v)] >= MIN_HALF).to_numpy()
    s = s[ok]
    go = (s[("mean", "odd", 1.0)] - s[("mean", "odd", 0.0)]).to_numpy()
    ge = (s[("mean", "even", 1.0)] - s[("mean", "even", 0.0)]).to_numpy()
    n = float(sum(s[("size", h, v)] for h in ("odd", "even") for v in (0.0, 1.0)).median())
    return go, ge, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    rng = np.random.default_rng(SEEDS)
    res = {
        "rows": len(t),
        "known_man_zone": float(t.known_mz.mean()),
        "known_cover": float(t.known_cv.mean()),
    }
    for name, known, col, by in (
        ("def_man_share", "known_mz", "man", "defteam"),
        ("def_cover3_share", "known_cv", "c3", "defteam"),
    ):
        a, b, n = share(t, known, col, by)
        res[name] = pack(a, b, rng, n)
    go, ge, n = gap(t)
    res["off_man_minus_zone_epa_gap"] = pack(go, ge, rng, n)
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
