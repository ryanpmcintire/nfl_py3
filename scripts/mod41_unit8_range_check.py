import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod41_cumulative_workload as m  # noqa: E402
import mod41_freeze_residual_load as f41  # noqa: E402
import mod41_unit2_components as u2  # noqa: E402

TERM = "mod41_residual_load_served"
RECORD = REPO / "artifacts" / "forward_terms" / TERM / "2026_week09_20261103T223235Z.json"
FROZEN = REPO / "artifacts" / "forward_terms" / TERM / "frozen_coefficients.json"
OUT = REPO / "artifacts" / "mod41" / "unit8_range_check.json"
SEASON = 2026
WEEK = 9
FIT_SEASONS = range(2020, 2026)
BYE_GAP = 2
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def team_table():
    t = m.team_games()
    t = t.merge(u2.components(t.copy()), on=["game_id", "team"], how="left")
    t["week"] = t["game_id"].str.split("_").str[1].astype(int)
    key = [t["team"], t["season"]]
    t["gap_weeks"] = t.groupby(key)["week"].diff()
    return t


def dist(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        **{f"q{int(q * 100):03d}": float(np.quantile(x, q)) for q in QUANTILES},
    }


def pct_rank(ref, v):
    return float((np.asarray(ref) <= v).mean())


def training(t, beta):
    d, _ = u2.build()
    d = d[d.season.isin(FIT_SEASONS)].copy()
    d["week"] = d["game_id"].str.split("_").str[1].astype(int)
    g = t[["game_id", "team", "gap_weeks", "short"]]
    for side in ("home", "away"):
        d = d.merge(
            g.rename(
                columns={
                    "team": f"{side}_team",
                    "gap_weeks": f"{side}_gap",
                    "short": f"{side}_short",
                }
            ),
            on=["game_id", f"{side}_team"],
            how="left",
        )
    d["off_bye"] = (d.home_gap >= BYE_GAP) | (d.away_gap >= BYE_GAP)
    d["home_bye"] = d.home_gap >= BYE_GAP
    d["away_bye"] = d.away_gap >= BYE_GAP
    return d[d.usable]


def forward_replay(d, pbp, beta):
    rows = []
    for season in sorted(d.season.unique()):
        tg = f41.season_team_games(pbp, int(season))
        for week in sorted(d.loc[d.season == season, "week"].unique()):
            st = f41.team_state(tg[tg.week < week])
            gw = d[(d.season == season) & (d.week == week)]
            for r in gw.itertuples():
                if r.home_team not in st.index or r.away_team not in st.index:
                    continue
                h, a = st.loc[r.home_team], st.loc[r.away_team]
                if not (bool(h.ok) and bool(a.ok)):
                    continue
                b0, bo, bp = beta
                rows.append(
                    {
                        "game_id": r.game_id,
                        "fwd": float(
                            (h.load_total - a.load_total)
                            - (b0 + bo * (h.own_td - a.own_td) + bp * (h.opp3 - a.opp3))
                        ),
                        "train": float(r.s_resid),
                    }
                )
    return pd.DataFrame(rows)


def week9_parts(pbp, fr):
    tg = f41.season_team_games(pbp, SEASON)
    prior = tg[tg.week < WEEK]
    st = f41.team_state(prior)
    last2 = prior.sort_values("week").groupby("team")["week"].apply(list)
    b0, bo, bp = fr["decomposition_beta_const_own_opp"]
    rec = json.loads(RECORD.read_text())
    out = []
    for r in rec["games"]:
        h, a = st.loc[r["home_team"]], st.loc[r["away_team"]]
        wk = {s: last2[r[f"{s}_team"]] for s in ("home", "away")}
        gap = {s: WEEK - wk[s][-1] for s in wk}
        win = {s: wk[s][-3:] for s in wk}
        bye_in_window = {s: bool(np.any(np.diff(win[s]) >= BYE_GAP)) for s in wk}
        s_total = float(h.load_total - a.load_total)
        s_own = float(h.own_td - a.own_td)
        s_opp = float(h.opp3 - a.opp3)
        resid = s_total - (b0 + bo * s_own + bp * s_opp)
        out.append(
            {
                "game_id": r["game_id"],
                "home_load": float(h.load_total),
                "away_load": float(a.load_total),
                "s_total": s_total,
                "home_own": float(h.own_td),
                "away_own": float(a.own_td),
                "s_own": s_own,
                "home_opp3": float(h.opp3),
                "away_opp3": float(a.opp3),
                "s_opp": s_opp,
                "resid": resid,
                "record_term": r["term_value"],
                "home_gap_weeks": int(gap["home"]),
                "away_gap_weeks": int(gap["away"]),
                "home_window_weeks": win["home"],
                "away_window_weeks": win["away"],
                "home_bye_in_window": bye_in_window["home"],
                "away_bye_in_window": bye_in_window["away"],
                "base_p": r["base_home_probability"],
                "combined_p": r["combined_home_probability"],
                "delta_logit": r["delta_logit"],
                "flip": bool(
                    (r["base_home_probability"] >= 0.5) != (r["combined_home_probability"] >= 0.5)
                ),
            }
        )
    return pd.DataFrame(out)


def main():
    fr = json.loads(FROZEN.read_text())
    beta = fr["decomposition_beta_const_own_opp"]
    pbp = f41.latest_pbp(REPO / "data" / "pbp" / "raw")
    t = team_table()
    d = training(t, beta)
    ref = d["s_resid"].to_numpy()
    tt = t[t.ok & t.season.isin(FIT_SEASONS)]
    off = tt[tt.gap_weeks >= BYE_GAP]
    on = tt[tt.gap_weeks < BYE_GAP]
    replay = forward_replay(d, pbp, beta)
    replay["abs_diff"] = (replay.fwd - replay.train).abs()
    w9 = week9_parts(pbp, fr)
    lo, hi = float(ref.min()), float(ref.max())
    w9["beyond_train_range"] = (w9.resid < lo) | (w9.resid > hi)
    w9["pct_rank_all"] = [pct_rank(ref, v) for v in w9.resid]
    w9["off_bye"] = (w9.home_gap_weeks >= BYE_GAP) | (w9.away_gap_weeks >= BYE_GAP)
    w9["team_load_beyond"] = [
        bool(max(h, a) > tt.load_total.max() or min(h, a) < tt.load_total.min())
        for h, a in zip(w9.home_load, w9.away_load, strict=True)
    ]
    wk9 = d[d.week == WEEK]
    out = {
        "train_games_usable": len(d),
        "train_resid": dist(ref),
        "train_resid_week9": dist(wk9.s_resid),
        "train_resid_off_bye_games": dist(d.loc[d.off_bye, "s_resid"]),
        "train_resid_home_off_bye": dist(d.loc[d.home_bye, "s_resid"]),
        "train_resid_away_off_bye": dist(d.loc[d.away_bye, "s_resid"]),
        "train_resid_no_bye": dist(d.loc[~d.off_bye, "s_resid"]),
        "train_team_load_all": dist(tt.load_total),
        "train_team_load_off_bye": dist(off.load_total),
        "train_team_load_on_bye_none": dist(on.load_total),
        "train_range": [lo, hi],
        "week9_resid": dist(w9.resid),
        "week9_off_bye_games": int(w9.off_bye.sum()),
        "week9_share_off_bye_train": float(d.off_bye.mean()),
        "parity_replay_n": len(replay),
        "parity_replay_max_abs_diff": float(replay.abs_diff.max()),
        "parity_replay_n_over_1e-9": int((replay.abs_diff > 1e-9).sum()),
        "week9_record_vs_recomputed_max_abs_diff": float((w9.resid - w9.record_term).abs().max()),
        "week9_games": w9.to_dict(orient="records"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    for k, v in out.items():
        if k != "week9_games":
            print(k, v)
    cols = [
        "game_id",
        "home_load",
        "away_load",
        "s_own",
        "s_opp",
        "resid",
        "home_gap_weeks",
        "away_gap_weeks",
        "home_bye_in_window",
        "away_bye_in_window",
        "pct_rank_all",
        "beyond_train_range",
        "flip",
    ]
    print(w9[cols].round(2).to_string())


if __name__ == "__main__":
    main()
