import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nfelo_forward as NF
import outside_model_gap as O
from xlg14_fumble_luck_nfl_confirmation import load_mod36

from nfl_ats.published_picks import game_deadlines

ROOT = O.ROOT
SEASON = 2025
OUT = ROOT / "artifacts" / "mod30"
COMMITS = OUT / "commits_2025.json"
STAMPED = OUT / "stamped"
LANE = "docs/lanes/mod-30-mod30-public-ratings.md"
FAMILY = "mod30_nfelo_stamped_2026_11_21"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
LOCAL = "America/New_York"
CAPTURE_HOUR = 9
CAPTURE_WEEKDAYS = (2, 5)
SINCE = "2025-08-25T00:00:00Z"
UNTIL = "2026-01-06T00:00:00Z"
DELAY_SECONDS = 2
SEED = 20261121
DRAWS = 200
BOOT = 2000
INTERVAL = (2.5, 97.5)
BAR = 95.0


def polite(url):
    time.sleep(DELAY_SECONDS)
    try:
        return NF.get(url)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"stop: HTTP {e.code} on {url}") from e


def schedule():
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "week", "game_id", "kickoff", "game_type"],
    )
    g = g[(g["season"] == SEASON) & (g["game_type"] == "REG")].copy()
    g["kickoff_utc"] = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    dl = {}
    for _, wk in g.groupby("week"):
        dl.update(dict(game_deadlines(wk[["game_id"]].assign(kickoff=wk["kickoff_utc"]))))
    g["pick_deadline_utc"] = g["game_id"].map(dl)
    g["cutoff_utc"] = g[["kickoff_utc", "pick_deadline_utc"]].min(axis=1)
    return g.reset_index(drop=True)


def capture_times(sched):
    out = {}
    for week, k in sched.groupby("week")["kickoff_utc"].min().items():
        local = k.tz_convert(LOCAL)
        back = (local.weekday() - CAPTURE_WEEKDAYS[0]) % 7
        first = (local.normalize() - pd.Timedelta(days=back)).date()
        times = []
        for wd in CAPTURE_WEEKDAYS:
            day = pd.Timestamp(first) + pd.Timedelta(days=wd - CAPTURE_WEEKDAYS[0])
            t = pd.Timestamp(day.year, day.month, day.day, CAPTURE_HOUR, tz=LOCAL).tz_convert("UTC")
            times.append(t)
        out[int(week)] = times
    return out


def commits():
    rows, page = [], 1
    while True:
        url = (
            f"https://api.github.com/repos/{NF.REPO}/commits?path={NF.PATH}"
            f"&since={SINCE}&until={UNTIL}&per_page=100&page={page}"
        )
        batch = json.loads(polite(url))
        if not batch:
            break
        rows += [{"sha": c["sha"], "commit_utc": c["commit"]["committer"]["date"]} for c in batch]
        page += 1
    return rows


def cmd_fetch(a):
    OUT.mkdir(parents=True, exist_ok=True)
    STAMPED.mkdir(parents=True, exist_ok=True)
    if not COMMITS.exists():
        c = commits()
        COMMITS.write_text(
            json.dumps({"fetched_at_utc": datetime.now(UTC).isoformat(), "commits": c}, indent=1)
        )
    c = pd.DataFrame(json.loads(COMMITS.read_text())["commits"])
    c["t"] = pd.to_datetime(c["commit_utc"], utc=True)
    c = c.sort_values("t")
    sched = schedule()
    need = set()
    for times in capture_times(sched).values():
        for t in times:
            before = c[c["t"] < t]
            if not before.empty:
                need.add(before.iloc[-1]["sha"])
    print(f"commits {len(c)}, snapshots needed {len(need)}", flush=True)
    for sha in sorted(need):
        f = STAMPED / f"{sha}.json"
        if f.exists():
            continue
        body = polite(f"https://raw.githubusercontent.com/{NF.REPO}/{sha}/{NF.PATH}")
        doc = {"sha": sha, "file_sha256": hashlib.sha256(body).hexdigest(), "rows": []}
        try:
            n = pd.read_csv(io.BytesIO(body), usecols=["game_id", "nfelo_dif_base"])
        except ValueError as e:
            doc["error"] = str(e)
            n = pd.DataFrame(columns=["game_id", "nfelo_dif_base"])
        n = n[n["game_id"].astype(str).str.startswith(f"{SEASON}_")].dropna(
            subset=["nfelo_dif_base"]
        )
        for r in n.itertuples():
            p = str(r.game_id).split("_")
            doc["rows"].append(
                {
                    "nfelo_game_id": r.game_id,
                    "game_id": f"{p[0]}_{p[1]}_{NF.code(p[2])}_{NF.code(p[3])}",
                    "nfelo_dif_base": float(r.nfelo_dif_base),
                }
            )
        f.write_text(json.dumps(doc))
        print(sha[:8], len(doc["rows"]), doc.get("error", ""), flush=True)


def stamped_values(sched):
    c = pd.DataFrame(json.loads(COMMITS.read_text())["commits"])
    c["t"] = pd.to_datetime(c["commit_utc"], utc=True)
    c = c.sort_values("t")
    caps = capture_times(sched)
    best, cover = {}, {}
    for r in sched.itertuples():
        for t in caps[int(r.week)]:
            before = c[c["t"] < t]
            if before.empty or t >= r.cutoff_utc:
                continue
            sha, ct = before.iloc[-1]["sha"], before.iloc[-1]["t"]
            if ct >= r.cutoff_utc:
                continue
            rows = {
                x["game_id"]: x["nfelo_dif_base"]
                for x in json.loads((STAMPED / f"{sha}.json").read_text())["rows"]
            }
            if r.game_id in rows:
                best[r.game_id] = rows[r.game_id]
                cover[r.game_id] = {
                    "capture_utc": t.isoformat(),
                    "sha": sha[:8],
                    "commit_utc": ct.isoformat(),
                }
    return best, cover


def loss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gains(y, z, x, beta):
    p0, p1 = expit(z), expit(z + beta * x)
    right0, right1 = (p0 > 0.5) == (y == 1), (p1 > 0.5) == (y == 1)
    return {
        "log_loss_improvement": loss(p0, y) - loss(p1, y),
        "brier_improvement": (p0 - y) ** 2 - (p1 - y) ** 2,
        "accuracy_points": 100.0 * (right1.astype(float) - right0.astype(float)),
        "changed": (p0 > 0.5) != (p1 > 0.5),
        "right1": right1,
        "right0": right0,
    }


def read(t, x, beta, rng):
    y, z, wk = t["home_covered"].to_numpy(float), t["z"].to_numpy(), t["week"].to_numpy()
    g = gains(y, z, x, beta)
    weeks = np.unique(wk)
    idx = [np.flatnonzero(wk == w) for w in weeks]
    res = {"games": len(y), "nonzero_games": int((x != 0).sum()), "weeks": len(weeks)}
    ch = g["changed"]
    res["changed"] = int(ch.sum())
    res["record"] = f"{int((g['right1'] & ch).sum())}-{int((g['right0'] & ch).sum())}"
    for k in ("log_loss_improvement", "brier_improvement", "accuracy_points"):
        v = g[k]
        sums, ns = np.array([v[i].sum() for i in idx]), np.array([len(i) for i in idx])
        draws = rng.integers(0, len(weeks), size=(BOOT, len(weeks)))
        boot = sums[draws].sum(axis=1) / ns[draws].sum(axis=1)
        null = []
        for _ in range(DRAWS):
            xs = x.copy()
            for i in idx:
                xs[i] = rng.permutation(xs[i])
            null.append(gains(y, z, xs, beta)[k].mean())
        null = np.array(null)
        eff = float(v.mean())
        res[k] = {
            "effect": eff,
            "low": float(np.percentile(boot, INTERVAL[0])),
            "high": float(np.percentile(boot, INTERVAL[1])),
            "probability_positive": float((boot > 0).mean()),
            "null_percentile": float(100 * ((null < eff).mean() + 0.5 * (null == eff).mean())),
            "null_median": float(np.median(null)),
            "null_p95": float(np.percentile(null, BAR)),
            "detectable_size": float(np.percentile(null, BAR) - np.median(null)),
        }
    return res


def reliability(t, x, beta):
    p0 = expit(t["z"].to_numpy())
    p1 = expit(t["z"].to_numpy() + beta * x)
    q = pd.qcut(p0, 5, labels=False, duplicates="drop")
    df = pd.DataFrame(
        {"bin": q, "served": p0, "combined": p1, "home_covered": t["home_covered"].to_numpy(float)}
    )
    return (
        df.groupby("bin")
        .agg(
            n=("served", "size"),
            served=("served", "mean"),
            combined=("combined", "mean"),
            observed=("home_covered", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )


def cmd_score(a):
    fz = json.loads(NF.FROZEN.read_text())
    m36 = load_mod36()
    pick, g = m36.games()
    g = g[["game_id", "season", "week", "home_covered", m36.SERVED_COLUMN]]
    d, _ = O.frame()
    g = g.merge(
        d[["game_id", "tue_open_home_spread", "nfelo_margin"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    g["retro_gap"] = (g["nfelo_margin"] - g["tue_open_home_spread"]).fillna(0.0)
    g["z"] = logit(np.clip(g[m36.SERVED_COLUMN].to_numpy(float), 1e-9, 1 - 1e-9))
    train = g[g["season"].between(2020, SEASON - 1)].reset_index(drop=True)
    test = g[g["season"] == SEASON].reset_index(drop=True)
    beta = float(
        m36.fit_offset(
            train[["retro_gap"]].to_numpy(),
            train["home_covered"].to_numpy(float),
            train["z"].to_numpy(),
        )[0]
    )
    sched = schedule()
    best, cover = stamped_values(sched)
    test["stamped_dif_base"] = test["game_id"].map(best)
    test["stamped_gap"] = (
        fz["points_per_elo"] * test["stamped_dif_base"] - test["tue_open_home_spread"]
    ).fillna(0.0)
    both = test.dropna(subset=["stamped_dif_base", "nfelo_margin"])
    stamped_margin = fz["points_per_elo"] * both["stamped_dif_base"]
    rng = np.random.default_rng(SEED)
    ins = gains(
        train["home_covered"].to_numpy(float),
        train["z"].to_numpy(),
        train["retro_gap"].to_numpy(),
        beta,
    )
    res = {
        "pick_artifact": str(pick.relative_to(ROOT)),
        "commits_file": str(COMMITS.relative_to(ROOT)),
        "coefficient_logit_per_point_fit_2020_2024": beta,
        "coefficient_per_sd_train": beta * float(train["retro_gap"].std(ddof=1)),
        "train_games": len(train),
        "frozen_forward_coefficient_logit_per_point": fz["coefficient_logit_per_point"],
        "points_per_elo": fz["points_per_elo"],
        "schedule_games_2025": len(sched),
        "stamped_games": len(best),
        "served_games_2025": len(test),
        "served_games_with_stamped_value": int(test["stamped_dif_base"].notna().sum()),
        "stamped_vs_current_margin_corr": float(
            np.corrcoef(stamped_margin, both["nfelo_margin"])[0, 1]
        ),
        "stamped_vs_current_mean_abs_diff_points": float(
            (stamped_margin - both["nfelo_margin"]).abs().mean()
        ),
        "stamped_gap_sd_points": float(
            test["stamped_gap"][test["stamped_dif_base"].notna()].std(ddof=1)
        ),
        "in_sample_train_log_loss_improvement": float(ins["log_loss_improvement"].mean()),
        "stamped": read(test, test["stamped_gap"].to_numpy(), beta, rng),
        "retro_control": read(test, test["retro_gap"].to_numpy(), beta, rng),
        "reliability_stamped": reliability(test, test["stamped_gap"].to_numpy(), beta),
        "capture_examples": dict(list(cover.items())[:3]),
    }
    res["in_sample_minus_held_out_log_loss"] = (
        res["in_sample_train_log_loss_improvement"]
        - res["stamped"]["log_loss_improvement"]["effect"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stamped_2025.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    cells = []
    for look, key, units, what in (
        ("stamped", "stamped", "log_loss_improvement", "commit-stamped pregame nfelo values"),
        ("stamped", "stamped", "accuracy_points", "commit-stamped pregame nfelo values"),
        (
            "current_file_control",
            "retro_control",
            "log_loss_improvement",
            "current-file (retroactively rebuilt) nfelo values, control",
        ),
    ):
        r = res[key][units]
        n = res[key]
        refuted = r["high"] < 0
        cell = {
            "name": f"mod30_nfelo_gap_2025_{look}_{units}",
            "description": (
                "nfelo model margin (points per Elo x nfelo_dif_base) minus the Tuesday "
                f"opener, home minus away, points, {what}; offset logistic on the served "
                "out-of-season probability, coefficient fitted 2020-2024 "
                f"({beta:+.5f} logit per point) and applied unchanged to 2025. "
                f"Games {n['games']}, nonzero {n['nonzero_games']}, changed {n['changed']} "
                f"record {n['record']}; null percentile {r['null_percentile']:.1f} of {DRAWS} "
                f"within-week shuffles; detectable size {r['detectable_size']:.5f}."
            ),
            "source": LANE,
            "effect": round(r["effect"], 6),
            "effect_units": units,
            "classification": "refuted_mechanism" if refuted else "unresolved_below_power",
            "league": "nfl",
            "season_start": SEASON,
            "season_end": SEASON,
            "family": FAMILY,
            "category": "market",
            "classification_evidence": (
                "Whole week-block interval below zero."
                if refuted
                else "No resolved wrong sign; one season; no positive control bounds an "
                f"effect this small ({POWER_LANE})."
            ),
            "notes": f"Family {FAMILY}: 2 looks (stamped, current-file control), 1 season.",
            "interval_low": round(r["low"], 6),
            "interval_high": round(r["high"], 6),
            "probability_positive": round(r["probability_positive"], 4),
            "sample_games": int(n["games"]),
            "sample_blocks": int(n["weeks"]),
            "plain_summary": (
                "Checks whether the nfelo model's number, as published before each 2025 "
                "game, improves the card's probability against the opening line. "
                "No pick changes from this check."
            ),
        }
        if refuted:
            cell["closing_ground"] = "wrong_sign_resolved"
        cells.append(cell)
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


def main():
    ap = argparse.ArgumentParser(
        description="MOD-30: 2025 commit-stamped nfelo gap on the served probability"
    )
    ap.add_argument("cmd", choices=["fetch", "score"])
    a = ap.parse_args()
    {"fetch": cmd_fetch, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
