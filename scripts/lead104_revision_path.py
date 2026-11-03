import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "lead99_llm_read" / "runs"
REPEATS = ROOT / "artifacts" / "lead99_llm_read" / "repeats"
SEED = 20261103
RESAMPLES = 10000
BETWEEN_READ_SD = None


def logit(p):
    return math.log(p / (1 - p))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def view_rows(run):
    market = {g["game_id"]: g for g in run["packet"]["games"]}
    out = {}
    for r in run["parsed_rows"]:
        g = market[r["game_id"]]
        m = g.get("market")
        out[r["game_id"]] = {
            "p": r["home_cover_probability"],
            "spread": m["home_spread_latest"] if m else None,
            "pool": g["pool_line"]["home_spread"],
        }
    return out


def between_read_sd():
    runs = [load(p) for p in sorted(REPEATS.glob("*.json"))]
    runs.append(load(next(RUNS.glob("2026-week-09-tuesday-*.json"))))
    stacks = {}
    for run in runs:
        for r in run.get("parsed_rows") or run.get("rows") or []:
            stacks.setdefault(r["game_id"], []).append(logit(r["home_cover_probability"]))
    sds = [np.std(v, ddof=1) for v in stacks.values() if len(v) > 1]
    return float(np.mean(sds)) if sds else None


def paired(week):
    files = sorted(RUNS.glob(f"2026-week-{week:02d}-*.json"))
    runs = [load(p) for p in files]
    tue = [r for r in runs if r["view"] == "tuesday"]
    dl = sorted([r for r in runs if r["view"] == "deadline"], key=lambda r: r["run_at_utc"])
    if not tue or not dl:
        return None
    t = view_rows(tue[0])
    last = {}
    for r in dl:
        last.update(view_rows(r))
    rows = []
    for gid, v in last.items():
        if gid in t and v["spread"] is not None and t[gid]["spread"] is not None:
            rows.append(
                {
                    "week": week,
                    "game_id": gid,
                    "revision": logit(v["p"]) - logit(t[gid]["p"]),
                    "move": v["spread"] - t[gid]["spread"],
                    "pool": t[gid]["pool"],
                    "p_dl": v["p"],
                }
            )
    return rows


def main():
    rows = []
    weeks = []
    for w in range(1, 19):
        r = paired(w)
        if r:
            rows += r
            weeks.append(w)
    df = pd.DataFrame(rows)
    print("weeks", weeks, "games", len(df))
    print(df.to_string(index=False))
    noise = between_read_sd()
    print(
        "between-read SD (logit)",
        noise,
        "revision noise SD sqrt2*",
        None if noise is None else noise * math.sqrt(2),
    )
    rev = df["revision"].to_numpy()
    mv = df["move"].to_numpy()
    n = len(df)
    rng = np.random.default_rng(SEED)

    def stats(r, m):
        if np.std(m) == 0 or np.std(r) == 0:
            return np.nan, np.nan, np.nan
        c = np.corrcoef(r, m)[0, 1]
        s = np.polyfit(m, r, 1)[0]
        return c, s, c * c

    c, s, r2 = stats(rev, mv)
    boot = np.array(
        [stats(rev[i], mv[i]) for i in (rng.integers(0, n, n) for _ in range(RESAMPLES))]
    )
    for k, name in enumerate(["corr", "slope", "R2"]):
        b = boot[:, k]
        b = b[~np.isnan(b)]
        pt = (c, s, r2)[k]
        print(
            name,
            round(pt, 4),
            "CI",
            np.percentile(b, [2.5, 97.5]).round(4),
            "P(>0)",
            round(float((b > 0).mean()), 4),
            "valid",
            len(b),
        )
    coef = np.polyfit(mv, rev, 1)
    resid = rev - np.polyval(coef, mv)
    print(
        "intercept", coef[1], "residual SD", resid.std(ddof=2), "raw revision SD", rev.std(ddof=1)
    )
    rs = [np.std(resid[i], ddof=1) for i in (rng.integers(0, n, n) for _ in range(RESAMPLES))]
    print("residual SD CI", np.percentile(rs, [2.5, 97.5]).round(4))
    if noise:
        print(
            "resid SD / read noise",
            resid.std(ddof=2) / noise,
            "/ revision noise",
            resid.std(ddof=2) / (noise * math.sqrt(2)),
        )
    g = pd.read_parquet(ROOT / "data" / "processed" / "game_features.parquet")
    g = g[g["season"].eq(2026)].set_index("game_id")
    df["resid"] = resid
    df["margin"] = df["game_id"].map(g["home_score"] - g["away_score"])
    df["cover"] = df["margin"] + df["pool"]
    played = df[df["margin"].notna() & df["cover"].ne(0) & df["resid"].ne(0)].copy()
    played["agree"] = np.sign(played["resid"]) == np.sign(played["cover"])
    print("outcome look (1 look): decisive", len(played), "agree", int(played["agree"].sum()))
    if len(played):
        print(played[["game_id", "resid", "cover", "agree"]].to_string(index=False))
        a = played["agree"].to_numpy()
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(RESAMPLES)])
        print(
            "agree rate",
            a.mean(),
            "CI",
            np.percentile(bs, [2.5, 97.5]).round(3),
            "P(>0.5)",
            float((bs > 0.5).mean()),
        )


if __name__ == "__main__":
    main()
