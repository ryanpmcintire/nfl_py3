import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
Z95 = 1.959964
QUANT_SD = 1.0 / math.sqrt(12.0)


def min_latency_sample():
    return math.ceil((Z95 / (2.0 * QUANT_SD)) ** 2)


def _secs(series):
    return (
        (pd.to_datetime(series, utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
        .dt.total_seconds()
        .to_numpy()
    )


def load_tapes(paths):
    frames = []
    for p in paths:
        f = pd.read_json(Path(p), lines=True)
        f["tape"] = Path(p).parent.name
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "tape": d["tape"],
            "market_id": d["market_id"].astype(str),
            "event_id": d["provider_event_id"].astype(str),
            "home": d["home_team"],
            "away": d["away_team"],
            "t_obs": (_secs(d["request_sent_utc"]) + _secs(d["response_received_utc"])) / 2.0,
            "lm": d["provider_last_updated_ms"].astype(float) / 1000.0,
            "line": d["home_spread_line"].astype(float),
        }
    )
    return out.sort_values(["tape", "market_id", "t_obs"]).reset_index(drop=True)


def poll_gaps(tape):
    g = tape.groupby(["tape", "market_id"])["t_obs"].diff().dropna()
    return g.to_numpy()


def change_events(tape):
    rows = []
    for (_, mid), g in tape.groupby(["tape", "market_id"], sort=False):
        g = g.sort_values("t_obs")
        t = g["t_obs"].to_numpy()
        ln = g["line"].to_numpy()
        lm = g["lm"].to_numpy()
        for i in range(1, len(g)):
            if ln[i] != ln[i - 1] and not (np.isnan(ln[i]) or np.isnan(ln[i - 1])):
                lm_adv = lm[i] > lm[i - 1]
                if lm_adv and lm[i] <= t[i]:
                    lo = hi = lm[i]
                else:
                    lo, hi = t[i - 1], t[i]
                rows.append(
                    {
                        "market_id": mid,
                        "home": g["home"].iloc[i],
                        "away": g["away"].iloc[i],
                        "t_prev": t[i - 1],
                        "t_obs": t[i],
                        "old": ln[i - 1],
                        "new": ln[i],
                        "lm_advanced": bool(lm_adv),
                        "T_lo": lo,
                        "T_hi": hi,
                    }
                )
    return pd.DataFrame(rows)


def latency_estimate(tape):
    rows = []
    for _, g in tape.groupby(["tape", "event_id"], sort=False):
        g = g.drop_duplicates("t_obs").sort_values("t_obs")
        t = g["t_obs"].to_numpy()
        lm = g["lm"].to_numpy()
        for i in range(1, len(g)):
            if lm[i] > lm[i - 1]:
                rows.append((t[i] - lm[i], t[i] - t[i - 1]))
    n = len(rows)
    nmin = min_latency_sample()
    out = {"n_lastmodified_advances": n, "n_min": nmin}
    gaps = poll_gaps(tape)
    out["poll_interval_median_s"] = float(np.median(gaps)) if len(gaps) else None
    if n == 0:
        out["status"] = "no_advances"
        return out
    d = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows])
    delta = float(np.median(w))
    out["observation_delay_quantiles_s"] = {
        str(q): float(np.quantile(d, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }
    out["resolution_s"] = delta
    est = float(np.median(d) - delta / 2.0)
    se = delta / (2.0 * math.sqrt(n))
    out["provider_lag_s"] = {
        "estimate": est,
        "standard_error": se,
        "interval_low": est - Z95 * se,
        "interval_high": est + Z95 * se,
        "floor_min_delay_s": float(d.min()),
    }
    out["status"] = "estimate" if n >= nmin else "below_minimum_sample"
    return out


def load_book_scans(patterns):
    frames = []
    for pat in patterns:
        for f in glob.glob(pat):
            q = pd.read_parquet(
                f,
                columns=[
                    "source_scan_at_utc",
                    "home_team",
                    "away_team",
                    "bookmaker_key",
                    "market",
                    "outcome_side",
                    "home_spread_line",
                ],
            )
            q = q[(q["market"] == "spreads") & (q["outcome_side"] == "HOME")]
            frames.append(q)
    if not frames:
        return pd.DataFrame(columns=["s", "home", "away", "book", "line"])
    d = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "s": _secs(d["source_scan_at_utc"]),
            "home": d["home_team"],
            "away": d["away_team"],
            "book": d["bookmaker_key"],
            "line": d["home_spread_line"].astype(float),
        }
    )
    out = out.drop_duplicates(["s", "home", "away", "book"]).sort_values("s")
    return out.reset_index(drop=True)


def book_changes(scans):
    rows = []
    for (h, a, b), g in scans.groupby(["home", "away", "book"], sort=False):
        s = g["s"].to_numpy()
        ln = g["line"].to_numpy()
        for i in range(1, len(g)):
            if ln[i] != ln[i - 1]:
                rows.append(
                    {
                        "home": h,
                        "away": a,
                        "book": b,
                        "s_prev": s[i - 1],
                        "s_cur": s[i],
                        "old": ln[i - 1],
                        "new": ln[i],
                    }
                )
    return pd.DataFrame(rows)


def lead_lag(events, bchg, span):
    if events.empty or bchg.empty:
        return {"status": "no_overlap", "pairs": []}, pd.DataFrame()
    pairs = []
    for _, e in events.iterrows():
        c = bchg[
            (bchg["home"] == e["home"])
            & (bchg["away"] == e["away"])
            & (bchg["old"] == e["old"])
            & (bchg["new"] == e["new"])
        ]
        for b, g in c.groupby("book"):
            mid_b = ((g["s_prev"] + g["s_cur"]) / 2.0 - (e["T_lo"] + e["T_hi"]) / 2.0).abs()
            r = g.loc[mid_b.idxmin()]
            if mid_b.min() > span:
                continue
            lo = r["s_prev"] - e["T_hi"]
            hi = r["s_cur"] - e["T_lo"]
            pairs.append(
                {
                    "book": b,
                    "lag_lo": lo,
                    "lag_hi": hi,
                    "lag_mid": (lo + hi) / 2.0,
                    "width": hi - lo,
                }
            )
    pdf = pd.DataFrame(pairs)
    if pdf.empty:
        return {"status": "no_overlap", "pairs": []}, pdf
    res = []
    for b, g in pdf.groupby("book"):
        n = len(g)
        w = float(g["width"].median())
        sd = float(g["lag_mid"].std(ddof=1)) if n > 1 else float("nan")
        row = {
            "book": b,
            "n": n,
            "bracket_width_median_s": w,
            "lag_mid_sd_s": sd,
        }
        if n > 1 and w > 0:
            nmin = max(2, math.ceil((Z95 * sd / (w * QUANT_SD)) ** 2))
            row["n_min"] = nmin
            row["status"] = "estimate" if n >= nmin else "below_minimum_sample"
        else:
            row["n_min"] = None
            row["status"] = "below_minimum_sample"
        if row["status"] == "estimate":
            se = sd / math.sqrt(n)
            m = float(g["lag_mid"].median())
            row.update(
                {
                    "lag_median_s": m,
                    "standard_error_s": se,
                    "interval_low": m - Z95 * se,
                    "interval_high": m + Z95 * se,
                    "share_book_after_bovada": float((g["lag_mid"] > 0).mean()),
                }
            )
        res.append(row)
    return {"status": "ok", "pairs": res}, pdf


def analyse(tape, scans):
    ev = change_events(tape)
    span = float(tape["t_obs"].max() - tape["t_obs"].min())
    hours = max(span / 3600.0, 1e-9)
    summ = {
        "tapes": sorted(tape["tape"].unique().tolist()),
        "markets": int(tape["market_id"].nunique()),
        "polls_rows": len(tape),
        "span_s": span,
        "line_change_events": len(ev),
        "line_changes_per_market_hour": float(len(ev) / (tape["market_id"].nunique() * hours))
        if len(tape)
        else None,
        "line_change_resolution_s": float(np.median(poll_gaps(tape))) if len(tape) > 1 else None,
        "line_changes_with_provider_stamp": int(ev["lm_advanced"].sum()) if len(ev) else 0,
    }
    lat = latency_estimate(tape)
    if scans is None or scans.empty:
        ll = {"status": "no_other_book_scans", "pairs": []}
    else:
        win = scans[
            (scans["s"] >= tape["t_obs"].min() - span) & (scans["s"] <= tape["t_obs"].max() + span)
        ]
        bchg = book_changes(win)
        ll, _ = lead_lag(ev, bchg, span)
    return {"summary": summ, "latency": lat, "lead_lag": ll}


def selftest():
    rng = np.random.default_rng(20261108)
    delta = 8.5
    planted_d = 17.0
    planted_follow = 45.0
    n_mk = 14
    dur = 3600.0
    t0 = 1.8e9
    polls = np.arange(0.0, dur, delta)
    rows = []
    srows = []
    scan_t = np.arange(-600.0, dur + 600.0, 60.0)
    for m in range(n_mk):
        k = rng.poisson(6)
        u = np.sort(rng.uniform(60.0, dur - 120.0, k))
        lines = 3.0 + 0.5 * np.cumsum(rng.choice([-1, 1], k))
        lines = np.where(np.diff(np.r_[3.0, lines]) == 0, lines + 0.5, lines)
        seq = np.r_[3.0, lines]
        for t in polls + rng.uniform(0, 0.3, len(polls)):
            vis = u + planted_d <= t
            j = int(vis.sum())
            lm = t0 + (u[j - 1] if j else -3600.0)
            rows.append((f"m{m}", f"e{m}", f"H{m}", f"A{m}", t0 + t, lm, seq[j]))
        for s in scan_t:
            vis = u + planted_follow <= s
            srows.append((t0 + s, f"H{m}", f"A{m}", "follower", seq[int(vis.sum())]))
    tape = pd.DataFrame(
        rows, columns=["market_id", "event_id", "home", "away", "t_obs", "lm", "line"]
    )
    tape["tape"] = "synthetic"
    scans = pd.DataFrame(srows, columns=["s", "home", "away", "book", "line"])
    res = analyse(tape, scans)
    lat = res["latency"]["provider_lag_s"]
    err = abs(lat["estimate"] - planted_d)
    ll = {p["book"]: p for p in res["lead_lag"]["pairs"]}["follower"]
    ev = change_events(tape)
    _, pdf = lead_lag(ev, book_changes(scans), float(tape["t_obs"].max() - tape["t_obs"].min()))
    cover = float(((pdf["lag_lo"] <= planted_follow) & (pdf["lag_hi"] >= planted_follow)).mean())
    out = {
        "planted_provider_lag_s": planted_d,
        "recovered_provider_lag_s": lat["estimate"],
        "abs_error_s": err,
        "polling_resolution_s": delta,
        "latency_within_resolution": bool(err <= delta),
        "latency_interval": [lat["interval_low"], lat["interval_high"]],
        "n_advances": res["latency"]["n_lastmodified_advances"],
        "planted_follower_lag_s": planted_follow,
        "recovered_follower_lag_s": ll.get("lag_median_s"),
        "follower_bracket_width_s": ll["bracket_width_median_s"],
        "follower_n": ll["n"],
        "follower_status": ll["status"],
        "share_brackets_containing_planted": cover,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="SKY-04 sub-minute tape latency and lead-lag")
    ap.add_argument("--tape", nargs="*", default=None)
    ap.add_argument(
        "--scans",
        nargs="*",
        default=[str(REPO / "data" / "market" / "raw" / "*odds-gap-private" / "quotes.parquet")],
    )
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sky04" / "tape_lag.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=1))
        return 0
    paths = a.tape or sorted(
        glob.glob(str(REPO / "data" / "raw" / "odds_tape" / "*" / "tape.jsonl"))
    )
    if not paths:
        print("no tapes", file=sys.stderr)
        return 1
    tape = load_tapes(paths)
    scans = load_book_scans(a.scans)
    res = analyse(tape, scans)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
