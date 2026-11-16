from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_candidate import LIVE_SEASON, LIVE_WEEKS, kickoff_and_cutoff  # noqa: E402

from nfl_ats.coach_fade_overlay import team_season_primary_coach, year_one_by_game  # noqa: E402
from nfl_ats.pick_probability import COACH_WEEK_MAX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts/coach_flag_parity"
G26 = ROOT / "artifacts/mod27_replay_inputs/g26.parquet"
STAMP = "%Y%m%dT%H%M%SZ"
GAME_COLUMNS = ["game_id", "season", "week", "gameday", "home_team", "away_team"]


def stamped():
    rows = []
    for path in DATA.glob("raw/*/schedules.parquet"):
        stamp = pd.to_datetime(path.parent.name, format=STAMP, errors="coerce", utc=True)
        if pd.notna(stamp) and (path.parent / "manifest.json").is_file():
            rows.append((stamp, path))
    return sorted(rows, key=lambda row: row[0])


def as_of(snapshots, moment):
    eligible = [path for stamp, path in snapshots if stamp < moment]
    return eligible[-1] if eligible else None


def reg(path, cache):
    key = str(path)
    if key not in cache:
        frame = pd.read_parquet(path)
        frame["game_id"] = frame["game_id"].astype(str)
        cache[key] = frame.loc[frame["game_type"].astype(str).eq("REG")].copy()
    return cache[key]


def coach_inputs(games, schedule):
    yo = year_one_by_game(schedule).merge(
        schedule[["game_id", "home_coach", "away_coach"]].drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )
    primary = team_season_primary_coach(schedule)
    cols = ["game_id", "home_coach", "away_coach", "year_one_home", "year_one_away"]
    out = games[["game_id", "season", "week", "home_team", "away_team"]].merge(
        yo[cols].drop_duplicates("game_id"), on="game_id", how="left"
    )
    prior = primary.assign(season=primary["season"] + 1)
    for side in ("home", "away"):
        p = prior.rename(columns={"team": f"{side}_team", "primary_coach": f"{side}_prior"})
        out = out.merge(p, on=[f"{side}_team", "season"], how="left")
    yh = out["year_one_home"].fillna(False).astype(bool)
    ya = out["year_one_away"].fillna(False).astype(bool)
    el = out["week"].astype(int).le(COACH_WEEK_MAX)
    out["flag_coach"] = np.where(el & ya & ~yh, 1, np.where(el & yh & ~ya, -1, 0))
    return out


def cmd_audit(_a):
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = stamped()
    cache = {}
    latest = snaps[-1][1]
    sched = reg(latest, cache)
    live = sched.loc[sched["season"].eq(LIVE_SEASON) & sched["week"].isin(list(LIVE_WEEKS))]
    live = live.reset_index(drop=True)
    _k, deadline = kickoff_and_cutoff(live)
    live["deadline_utc"] = deadline.to_numpy()
    live["sched_asof"] = [as_of(snaps, d) for d in live["deadline_utc"]]
    lat = coach_inputs(live, sched)
    parts = []
    for path, grp in live.groupby("sched_asof", sort=False):
        part = coach_inputs(grp, reg(path, cache))
        part["sched_asof"] = path.parent.name
        parts.append(part)
    asof = pd.concat(parts, ignore_index=True).set_index("game_id").reindex(lat["game_id"])
    g26 = pd.read_parquet(G26)
    g26["game_id"] = g26["game_id"].astype(str)
    t = lat.set_index("game_id").join(
        asof[["home_coach", "away_coach", "home_prior", "away_prior", "flag_coach", "sched_asof"]],
        rsuffix="_asof",
    )
    t = t.join(g26.set_index("game_id")[["flag_coach"]].rename(columns={"flag_coach": "g26"}))
    t["result"] = sched.set_index("game_id")["result"].reindex(t.index)
    t.to_csv(OUT / "per_game_2026.csv")
    both = t[t["g26"].notna()]
    mism = both[both["g26"].astype(float) != both["flag_coach"].astype(float)]
    cov = []
    for stamp, path in snaps:
        s = reg(path, cache)
        s26 = s[s["season"].eq(LIVE_SEASON)]
        if s26.empty:
            continue
        cov.append(
            {
                "snapshot": path.parent.name,
                "weeks_with_result": int(s26.loc[s26["result"].notna(), "week"].max())
                if s26["result"].notna().any()
                else 0,
                "coach_nonnull_w1_8": int(s26.loc[s26["week"].le(8), "home_coach"].notna().sum()),
                "games_w1_8": int(s26["week"].le(8).sum()),
                "flag_nonzero_w1_8": int(
                    coach_inputs(s26[s26["week"].le(8)], s)["flag_coach"].ne(0).sum()
                ),
            }
        )
    g26_cols = list(g26.columns)
    summary = {
        "latest_schedule_snapshot": latest.parent.name,
        "coach_week_max": int(COACH_WEEK_MAX),
        "g26_columns": g26_cols[:40],
        "g26_weeks": sorted(int(w) for w in g26["week"].unique()) if "week" in g26 else None,
        "g26_flag_coach_counts": g26["flag_coach"].value_counts().to_dict(),
        "games_compared": len(both),
        "agree_latest_v_g26": int(len(both) - len(mism)),
        "agree_asof_v_g26": int((both["g26"].astype(float) == both["flag_coach_asof"]).sum()),
        "agree_latest_v_asof": int((t["flag_coach"] == t["flag_coach_asof"]).sum()),
        "games_live": len(t),
        "mismatch_by_week": mism["week"].value_counts().sort_index().to_dict(),
        "mismatch_pairs": mism.groupby(["g26", "flag_coach"]).size().rename("n").reset_index().to_dict("records"),
        "snapshot_coverage_2026": cov,
    }
    (OUT / "audit.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "snapshot_coverage_2026"}, default=str))
    print(pd.DataFrame(cov).to_string())
    show = ["week", "home_team", "away_team", "home_coach", "home_prior", "away_coach", "away_prior",
            "flag_coach", "flag_coach_asof", "g26"]
    print(mism[show].to_string())


def signed_record(flag, edge):
    s = np.sign(flag) * np.sign(edge)
    nz = np.sign(flag) != 0
    return f"{int((s > 0).sum())}-{int((s < 0).sum())}" + (
        f"-{int((nz & (s == 0)).sum())}p" if (nz & (s == 0)).any() else ""
    ), int(nz.sum())


def first_coach(schedule):
    rows = []
    for side in ("home", "away"):
        rows.append(
            schedule[["season", "week", f"{side}_team", f"{side}_coach"]].set_axis(
                ["season", "week", "team", "coach"], axis=1
            )
        )
    long = pd.concat(rows).sort_values(["season", "week"])
    return long.groupby(["team", "season"])["coach"].first()


def cmd_explain(_a):
    from nfl_ats.splash_lines import load_splash_capture, splash_decision_lines

    snaps = stamped()
    cache = {}
    first, latest = snaps[0][1], snaps[-1][1]
    out = {}
    g26 = pd.read_parquet(G26)
    flags = [c for c in g26.columns if c.startswith("flag_")]
    out["g26_nonzero_by_flag"] = {c: int(g26[c].ne(0).sum()) for c in flags}
    t = pd.read_csv(OUT / "per_game_2026.csv")
    lines = {}
    for w in LIVE_WEEKS:
        cap = load_splash_capture(DATA, LIVE_SEASON, int(w))
        if cap is not None:
            lines.update(splash_decision_lines(cap))
    t["pool_line"] = t["game_id"].map(lines)
    edge = (t["result"] - t["pool_line"]).to_numpy(float)
    ok = ~np.isnan(edge)
    for col in ("flag_coach", "flag_coach_asof"):
        out[f"record_2026_{col}_pool_line"] = signed_record(t[col].to_numpy()[ok], edge[ok])
    diff = t[t["flag_coach"] != t["flag_coach_asof"]]
    out["latest_v_asof_differ"] = len(diff)
    out["latest_v_asof_teams"] = sorted(
        set(
            diff.apply(
                lambda r: r["home_team"]
                if r["home_coach"] != r["home_coach_asof"] or r["home_prior"] != r["home_prior_asof"]
                else r["away_team"],
                axis=1,
            )
        )
    )
    out["latest_v_asof_coach_names"] = (
        diff.apply(
            lambda r: [
                (r[f"{s}_team"], r[f"{s}_coach_asof"], r[f"{s}_coach"], r[f"{s}_prior_asof"], r[f"{s}_prior"])
                for s in ("home", "away")
                if r[f"{s}_coach"] != r[f"{s}_coach_asof"] or r[f"{s}_prior"] != r[f"{s}_prior_asof"]
            ],
            axis=1,
        )
        .explode()
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
    )
    out["asof_snapshots_used"] = t["sched_asof"].value_counts().to_dict()
    sig = pd.read_parquet(ROOT / "artifacts/pooled_stack/signals.parquet")
    sig["game_id"] = sig["game_id"].astype(str)
    hist = {}
    for name, path in (("first", first), ("latest", latest)):
        s = reg(path, cache)
        g = s[s["season"].between(2020, 2025)]
        hist[name] = coach_inputs(g, s).set_index("game_id")
    h = hist["latest"].reindex(sig["game_id"])
    out["history_games"] = len(sig)
    out["history_stored_v_latest_rebuild_agree"] = int((sig["flag_coach"].to_numpy() == h["flag_coach"].to_numpy()).sum())
    out["history_stored_v_first_snapshot_agree"] = int(
        (sig["flag_coach"].to_numpy() == hist["first"]["flag_coach"].reindex(sig["game_id"]).to_numpy()).sum()
    )
    fc = first_coach(reg(latest, cache))
    nz = h[h["flag_coach"] != 0].copy()
    side = np.where(nz["flag_coach"] > 0, "away", "home")
    team = np.where(side == "away", nz["away_team"], nz["home_team"])
    coach = np.where(side == "away", nz["away_coach"], nz["home_coach"])
    opening = [fc.get((tm, int(se))) for tm, se in zip(team, nz["season"], strict=True)]
    nz["midseason_change"] = [c != o for c, o in zip(coach, opening, strict=True)]
    pg = pd.read_parquet(ROOT / "artifacts/pooled_stack/per_game.parquet")
    out["per_game_columns"] = list(pg.columns)[:30]
    ycol = next((c for c in ("y", "home_covered", "label") if c in pg.columns), None)
    if ycol is not None:
        pg["game_id"] = pg["game_id"].astype(str)
        y = pg.set_index("game_id")[ycol].reindex(sig["game_id"]).to_numpy(float)
        e = np.where(y == 0.5, 0.0, np.where(y == 1, 1.0, -1.0))
        f = sig["flag_coach"].to_numpy(float)
        out["record_history_open"] = signed_record(f, e)
        mid = h["flag_coach"].ne(0).to_numpy() & h.index.isin(nz.index[nz["midseason_change"]])
        out["record_history_midseason_only"] = signed_record(np.where(mid, f, 0), e)
        out["record_history_by_season"] = {
            int(se): signed_record(f[sig["season"].to_numpy() == se], e[sig["season"].to_numpy() == se])
            for se in range(2020, 2026)
        }
    out["history_nonzero"] = len(nz)
    out["history_midseason_change_nonzero"] = int(nz["midseason_change"].sum())
    out["history_midseason_examples"] = (
        nz.loc[nz["midseason_change"], ["season", "week", "home_team", "away_team", "home_coach", "away_coach"]]
        .astype(str)
        .values.tolist()[:20]
    )
    (OUT / "explain.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, default=str))


def cell(t, col, label, text):
    from scipy.stats import beta

    edge = (t["result"] - t["pool_line"]).to_numpy(float)
    f = t[col].to_numpy(float)
    s = np.sign(f) * np.sign(edge)
    use = (np.sign(f) != 0) & ~np.isnan(edge) & (s != 0)
    w, n = int((s[use] > 0).sum()), int(use.sum())
    lo = beta.ppf(0.025, w, n - w + 1) if w > 0 else 0.0
    hi = beta.ppf(0.975, w + 1, n - w) if w < n else 1.0
    pp = float(1 - beta.cdf(0.5, 1 + w, 1 + n - w))
    cls = "wrong_sign_resolved" if hi < 0.5 else "unresolved_below_power"
    return cls, {
        "name": f"coach_flag_parity_2026_11_16_{label}_pool_line_w1_8_accuracy_points",
        "description": text + f" Record {w}-{n - w} on {n} decided nonzero games, 2026 Weeks 1-8 at the Splash pool line; declared 2026-11-16T23:13:33Z before intervals; PARTLY SEEN (records viewed in explain.json first).",
        "effect": 100 * (w / n - 0.5),
        "effect_units": "accuracy_points",
        "interval_low": 100 * (lo - 0.5),
        "interval_high": 100 * (hi - 0.5),
        "probability_positive": pp,
        "season_start": LIVE_SEASON,
        "season_end": LIVE_SEASON,
        "sample_games": n,
        "sample_blocks": int(t.loc[use, "week"].nunique()),
        "notes": "Exact Clopper-Pearson 95% interval; probability_positive = P(rate > .5) under Beta(1+W,1+L). 2 looks in the family. History 2020-25 at the opener: 236 nonzero, 136-96-4 (stored fit flag, equal to every as-of rebuild). The flag is zero after Week 8.",
    }


def cmd_record(_a):
    from nfl_ats.splash_lines import load_splash_capture, splash_decision_lines

    t = pd.read_csv(OUT / "per_game_2026.csv")
    lines = {}
    for w in LIVE_WEEKS:
        cap = load_splash_capture(DATA, LIVE_SEASON, int(w))
        if cap is not None:
            lines.update(splash_decision_lines(cap))
    t["pool_line"] = t["game_id"].map(lines)
    t = t[t["week"].le(COACH_WEEK_MAX)]
    a = cell(t, "flag_coach", "latest_snapshot", "Year-one head-coach flag (history definition: the game's coach of record against the prior season's modal coach; +1 away year one, -1 home year one), values from the latest schedules snapshot.")
    b = cell(t, "flag_coach_asof", "deadline_asof", "Year-one head-coach flag as the live feed served it: values from the latest schedules snapshot stamped before each game's pick deadline (nflverse coach fields stale for ATL, ARI and BUF before 2026-11-11).")
    classes = {a[0], b[0]}
    batch = {
        "family": "coach_flag_parity_2026_11_16",
        "league": "nfl",
        "category": "modeling",
        "classification": "unresolved_below_power" if classes == {"unresolved_below_power"} else "mixed",
        "source": "scripts/coach_flag_parity.py; artifacts/coach_flag_parity/; docs/lanes/coach-flag-parity-2026-11-16.md",
        "classification_evidence": "Exact 95% intervals: "
        + "; ".join(f"{c['name']} [{c['interval_low']:+.2f},{c['interval_high']:+.2f}]" for _, c in (a, b))
        + ". Neither interval lies wholly below zero and no positive control exists, so unresolved_below_power is the only admissible classification.",
        "cells": [a[1], b[1]],
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1))
    print(json.dumps({"classes": [a[0], b[0]], "cells": [{k: c[k] for k in ("name", "effect", "interval_low", "interval_high", "probability_positive", "sample_games")} for _, c in (a, b)]}, indent=1))


def main():
    p = argparse.ArgumentParser(description="Coach flag train/serve parity")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    sub.add_parser("explain").set_defaults(func=cmd_explain)
    sub.add_parser("record").set_defaults(func=cmd_record)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
