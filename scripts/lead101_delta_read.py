import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lead99_llm_read", HERE / "lead99_llm_read.py")
L99 = importlib.util.module_from_spec(spec)
sys.modules["lead99_llm_read"] = L99
spec.loader.exec_module(L99)

ART = L99.ROOT / "artifacts" / "lead101"
PROMPT = HERE / "lead101_prompt.md"
FROZEN = ART / "frozen.json"
TUESDAY_RUNS = L99.ART / "runs"
SCORE = ART / "score"
FAMILY = "lead101_news_delta_read"
VIEW_WEEKDAY = {"thursday": 3, "sunday": 6}
BASELINES = ("zero", "nochange", "momentum")
BOOTSTRAP_DRAWS = 10000
PERMUTATION_DRAWS = 10000
INTERVAL_PERCENTILES = (2.5, 97.5)
ROUND_DIGITS = 9
SEED = 101
VIEW_WORDS = {
    "thursday": "Thursday evening reads",
    "sunday": "Sunday morning reads",
    "pooled": "Thursday and Sunday reads together",
}
BASELINE_WORDS = {
    "zero": "a guess that the line ends where the pool line froze",
    "nochange": "a guess that the line stays at the latest quote the model was shown",
    "momentum": "a guess that the move so far keeps going at the same pace",
}
DIRECTION_WORDS = {
    "total": "direction of the move from the frozen pool line to the deadline line",
    "remaining": "direction of the move from the latest quote shown to the deadline line",
}


def prompt_sha():
    return L99.sha256_bytes(PROMPT.read_bytes())


def utc(s):
    t = pd.Timestamp(s)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def tuesday_freeze(week):
    fs = sorted(TUESDAY_RUNS.glob(f"{L99.SEASON}-week-{week:02d}-tuesday-*.json"))
    if not fs:
        raise SystemExit("no stored Tuesday packet for this week")
    d = json.loads(fs[-1].read_text(encoding="utf-8"))
    return fs[-1], utc(d["run_at_utc"])


def after(ts, freeze):
    return ts is not None and utc(ts) > freeze


def delta_game(g, freeze):
    h, a = g["home"], g["away"]
    out = {}
    m = g["market"]
    if m and after(m["latest_capture_at_utc"], freeze):
        out["market"] = {
            "home_spread_latest": m["home_spread_latest"],
            "latest_capture_at_utc": m["latest_capture_at_utc"],
            "book": m["book"],
        }
    inj = {}
    for t in (h, a):
        rows = [r for r in g["injuries"][t]["rows"] if after(r["date_modified_utc"], freeze)]
        if rows:
            inj[t] = rows
    if inj:
        out["injuries"] = inj
    hl = {
        t: [x for x in g["injury_headlines"][t] if after(x["published_at_utc"], freeze)]
        for t in (h, a)
    }
    hl = {t: v for t, v in hl.items() if v}
    if hl:
        out["injury_headlines"] = hl
    ina = [x for x in g["inactives"] if after(x["captured_at_utc"], freeze)]
    if ina:
        out["inactives"] = ina
    beat = {
        t: [p for p in g["beat_posts"][t]["posts"] if after(p["at_utc"], freeze)] for t in (h, a)
    }
    beat = {t: v for t, v in beat.items() if v}
    if beat:
        out["beat_posts"] = beat
    wx = [x for x in g["weather"] if after(x["captured_at_utc"], freeze)]
    if wx:
        out["weather"] = wx
    return out


def build_delta(week, run_at):
    path, freeze = tuesday_freeze(week)
    full = L99.build_packet(week, run_at)
    games = []
    for g in full["games"]:
        d = delta_game(g, freeze)
        games.append(
            {
                "game_id": g["game_id"],
                "away": g["away"],
                "home": g["home"],
                "kickoff_et": g["kickoff_et"],
                "deadline_et": g["deadline_et"],
                "frozen_pool_line": g["pool_line"],
                "items_after_freeze": d,
                "n_item_groups": len(d),
            }
        )
    packet = {
        "week": week,
        "season": L99.SEASON,
        "freeze_utc": freeze.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_at_utc": run_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "convention": full["convention"],
        "games": games,
    }
    return packet, path


def validate(parsed, ids):
    rows = parsed["games"]
    assert sorted(r["game_id"] for r in rows) == sorted(ids), "game set mismatch"
    for r in rows:
        v = r["expected_home_spread_move"]
        assert isinstance(v, (int, float)) and not isinstance(v, bool)
        assert isinstance(r["mechanism"], str) and r["mechanism"].strip()
        assert isinstance(r["key_facts"], list)
    return rows


def call_no_tools(prompt_text, packet_text, cwd):
    cmd = [
        L99.shutil.which("claude"),
        "-p",
        "--output-format",
        "json",
        "--model",
        L99.MODEL,
        "--tools",
        "",
        "--no-session-persistence",
        "--setting-sources",
        "local",
        "--permission-mode",
        "dontAsk",
    ]
    p = subprocess.run(
        cmd,
        input=prompt_text + "\n\nPACKET\n" + packet_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=1800,
    )
    if p.returncode != 0:
        raise SystemExit(f"claude failed rc={p.returncode}: {p.stderr[-800:]}")
    env = json.loads(p.stdout)
    return env, env.get("result", "")


def cmd_enroll(args):
    ART.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(
        json.dumps(
            {
                "prompt_sha256": prompt_sha(),
                "model": L99.MODEL,
                "tools": "",
                "enrolled_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        )
    )
    print(FROZEN.read_text())


def cmd_run(args):
    fz = json.loads(FROZEN.read_text())
    if fz["prompt_sha256"] != prompt_sha() or fz["model"] != L99.MODEL:
        raise SystemExit("prompt or model differs from frozen enrollment")
    run_at = datetime.now(UTC).replace(microsecond=0)
    week = args.week if args.week is not None else L99.current_week(run_at)
    packet, tue_path = build_delta(week, run_at)
    live = [g for g in packet["games"] if g["n_item_groups"] > 0]
    ids = [g["game_id"] for g in live]
    out = {
        "week": week,
        "view": args.view,
        "packet": packet,
        "packet_sha256": L99.sha256_bytes(
            json.dumps(packet, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ),
        "prompt_sha256": prompt_sha(),
        "model": L99.MODEL,
        "tuesday_packet": tue_path.name,
        "items_by_game": {g["game_id"]: g["n_item_groups"] for g in packet["games"]},
        "attempts": [],
        "parsed_rows": [],
        "run_at_utc": packet["run_at_utc"],
    }
    if ids:
        sub = dict(packet, games=live)
        cwd = ART / "cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        for attempt in (1, 2):
            env, raw = call_no_tools(
                PROMPT.read_text(encoding="utf-8"),
                json.dumps(sub, ensure_ascii=False, sort_keys=True),
                str(cwd),
            )
            rec = {"attempt": attempt, "raw_response": raw, "envelope": env}
            try:
                out["parsed_rows"] = validate(L99.extract_json(raw), ids)
                rec["valid"] = True
            except Exception as e:
                rec["valid"] = False
                rec["error"] = repr(e)
            out["attempts"].append(rec)
            if rec["valid"]:
                break
    recorded = datetime.now(UTC)
    out["recorded_at_utc"] = recorded.strftime("%Y-%m-%dT%H:%M:%SZ")
    dl = {g["game_id"]: datetime.fromisoformat(g["deadline_et"]) for g in packet["games"]}
    late = [r["game_id"] for r in out["parsed_rows"] if recorded >= dl[r["game_id"]]]
    out["dropped_past_deadline"] = late
    out["parsed_rows"] = [r for r in out["parsed_rows"] if r["game_id"] not in late]
    runs = ART / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = recorded.strftime("%Y%m%dT%H%M%SZ")
    path = runs / f"{L99.SEASON}-week-{week:02d}-{args.view}-{stamp}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)
    print(out["items_by_game"])
    print("rows", len(out["parsed_rows"]), "model_called", bool(ids))


def served_quotes():
    from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes
    from nfl_ats.sharp_book_movement_features import LEADERSHIP_WEIGHTS

    q = load_decision_quotes(L99.ROOT / "data" / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    q = q.loc[q["market"].eq("spreads") & q["bookmaker_key"].isin(LEADERSHIP_WEIGHTS)]
    blank = pd.Series(pd.NA, index=q.index, dtype="object")
    seen = pd.to_datetime(q["observed_at_utc"], utc=True, errors="coerce")
    updated = pd.to_datetime(q["bookmaker_last_update_utc"], utc=True, errors="coerce")
    scan = pd.to_datetime(q.get("source_scan_at_utc", blank), utc=True, errors="coerce")
    basis = q.get("quote_timestamp_basis", blank).astype("string")
    line = pd.to_numeric(q["home_spread_line"], errors="coerce")
    valid = (
        seen.notna()
        & np.isfinite(line)
        & (scan.isna() | scan.le(seen))
        & (updated.le(seen) | (updated.isna() & basis.eq("capture_observed_utc").fillna(False)))
    )
    out = pd.DataFrame(
        {
            "game_id": q["nflverse_game_id"].astype(str),
            "book": q["bookmaker_key"],
            "seen": seen,
            "line": line,
        }
    ).loc[valid]
    return out.drop_duplicates(["game_id", "book", "seen"])


def declared_records():
    fz = json.loads(FROZEN.read_text())
    kept, excluded = [], []
    for f in sorted((ART / "runs").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        view = d["view"]
        weekday = utc(d["recorded_at_utc"]).tz_convert("America/New_York").weekday()
        if view not in VIEW_WEEKDAY:
            reason = "view is not a declared forward view"
        elif weekday != VIEW_WEEKDAY[view]:
            reason = "recorded on another weekday than its view"
        elif d["prompt_sha256"] != fz["prompt_sha256"] or d["model"] != fz["model"]:
            reason = "prompt or model differs from the frozen enrollment"
        else:
            kept.append((f.name, d))
            continue
        excluded.append(
            {
                "file": f.name,
                "view": view,
                "recorded_at_utc": d["recorded_at_utc"],
                "reason": reason,
            }
        )
    return kept, excluded


def score_rows(kept, quotes, now):
    by_game = {gid: g.sort_values("seen") for gid, g in quotes.groupby("game_id")}
    rows, skipped = [], []
    for name, d in kept:
        run_at = utc(d["run_at_utc"])
        recorded = utc(d["recorded_at_utc"])
        games = {g["game_id"]: g for g in d["packet"]["games"]}
        for r in d["parsed_rows"]:
            g = games[r["game_id"]]
            deadline = utc(g["deadline_et"])
            base = {"file": name, "week": d["week"], "view": d["view"], "game_id": r["game_id"]}
            q = by_game.get(r["game_id"])
            if recorded >= deadline:
                skipped.append({**base, "reason": "recorded at or after the deadline"})
                continue
            if deadline >= now:
                skipped.append({**base, "reason": "deadline not reached"})
                continue
            window = None if q is None else q.loc[q["seen"].gt(run_at) & q["seen"].lt(deadline)]
            if window is None or window.empty:
                skipped.append({**base, "reason": "no quote between the record and the deadline"})
                continue
            last = window.groupby("book").tail(1)
            pool = float(g["frozen_pool_line"]["home_spread"])
            pooled_at = utc(g["frozen_pool_line"]["captured_at_et"])
            market = g["items_after_freeze"].get("market")
            so_far, momentum = 0.0, 0.0
            if market:
                quoted_at = utc(market["latest_capture_at_utc"])
                so_far = float(market["home_spread_latest"]) - pool
                momentum = so_far * float((deadline - pooled_at) / (quoted_at - pooled_at))
            closing = float(last["line"].median())
            rows.append(
                {
                    **base,
                    "recorded_at_utc": d["recorded_at_utc"],
                    "deadline_utc": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "deadline_line_seen_utc": last["seen"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "deadline_books": len(last),
                    "pool_line": pool,
                    "deadline_line": closing,
                    "realised": closing - pool,
                    "read": float(r["expected_home_spread_move"]),
                    "zero": 0.0,
                    "nochange": so_far,
                    "momentum": momentum,
                }
            )
    return pd.DataFrame(rows), skipped


def direction(call, target, grid):
    moved = np.abs(target) >= grid
    agree = np.sign(np.round(call, ROUND_DIGITS)) * np.sign(np.round(target, ROUND_DIGITS))
    return moved, np.where(moved, agree, 0.0)


def spread(draws):
    d = draws[np.isfinite(draws)]
    low, high = np.percentile(d, INTERVAL_PERCENTILES)
    return {
        "interval_low": float(low),
        "interval_high": float(high),
        "probability_positive": float((d > 0).mean() + (d == 0).mean() / 2),
    }


def cell_scores(frame, grid, rng):
    target = frame["realised"].to_numpy()
    read = frame["read"].to_numpy()
    so_far = frame["nochange"].to_numpy()
    err = np.abs(read - target)
    rival_err = {b: np.abs(frame[b].to_numpy() - target) for b in BASELINES}
    calls = {"total": (read, target), "remaining": (read - so_far, target - so_far)}
    marks = {k: direction(c, t, grid) for k, (c, t) in calls.items()}
    game_ids = frame["game_id"].to_numpy()
    uniq = np.unique(game_ids)
    members = [np.flatnonzero(game_ids == g) for g in uniq]
    draws = {k: np.full(BOOTSTRAP_DRAWS, np.nan) for k in (*BASELINES, *calls)}
    for i in range(BOOTSTRAP_DRAWS):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
        for b in BASELINES:
            draws[b][i] = (rival_err[b][idx] - err[idx]).mean()
        for k, (moved, agree) in marks.items():
            n = moved[idx].sum()
            if n:
                draws[k][i] = agree[idx].sum() / n
    keys = frame["week"].astype(str).to_numpy() + frame["view"].to_numpy()
    strata = [np.flatnonzero(keys == k) for k in np.unique(keys)]
    out = {
        "rows": len(frame),
        "games": len(uniq),
        "weeks": int(frame["week"].nunique()),
        "mae_read": float(err.mean()),
        "errors": {},
        "direction": {},
    }
    for b in BASELINES:
        out["errors"][b] = {
            "mae_baseline": float(rival_err[b].mean()),
            "gain": float((rival_err[b] - err).mean()),
            **spread(draws[b]),
        }
    for k, (call, goal) in calls.items():
        moved, agree = marks[k]
        rival = direction(so_far, goal, grid)[1]
        cell = {
            "moved": int(moved.sum()),
            "hits": int((agree > 0).sum()),
            "misses": int((agree < 0).sum()),
            "baseline_hits": int((rival > 0).sum()),
            "baseline_misses": int((rival < 0).sum()),
        }
        cell["no_call"] = cell["moved"] - cell["hits"] - cell["misses"]
        if cell["moved"]:
            cell["net_rate"] = float(agree.sum() / cell["moved"])
            cell.update(spread(draws[k]))
            beat = 0
            for _ in range(PERMUTATION_DRAWS):
                shuffled = call.copy()
                for s in strata:
                    shuffled[s] = call[rng.permutation(s)]
                beat += int(direction(shuffled, goal, grid)[1].sum() >= agree.sum())
            cell["permutation_p"] = (1 + beat) / (1 + PERMUTATION_DRAWS)
        out["direction"][k] = cell
    return out


def batch_payload(cells):
    shared = {
        "source": "artifacts/lead101/score/scores.json",
        "league": "nfl",
        "season_start": L99.SEASON,
        "season_end": L99.SEASON,
        "classification": "unresolved_below_power",
        "category": "market",
        "family": FAMILY,
    }
    out = []
    for name, c in cells.items():
        evidence = (
            f"{c['weeks']} forward week(s), {c['games']} games, game bootstrap; no split-half "
            "reliability and no positive control exist yet, so no closing ground applies"
        )
        for b in BASELINES:
            e = c["errors"][b]
            out.append(
                {
                    **shared,
                    "name": f"lead101_delta_read_{name}_error_gain_vs_{b}",
                    "description": (
                        f"LEAD-101 news-delta read, {VIEW_WORDS[name]}: mean absolute error of "
                        f"{BASELINE_WORDS[b]} minus the read's error, in line points, target = "
                        "deadline line minus frozen pool line; positive means the read is closer"
                    ),
                    "plain_summary": (
                        "A language model sees only the news that arrived after the pool line "
                        "froze and guesses how far each line will move by the pick deadline. "
                        f"This compares the size of its misses with {BASELINE_WORDS[b]}."
                    ),
                    "effect": e["gain"],
                    "effect_units": "mae_improvement",
                    "interval_low": e["interval_low"],
                    "interval_high": e["interval_high"],
                    "probability_positive": e["probability_positive"],
                    "sample_games": c["games"],
                    "sample_blocks": c["weeks"],
                    "classification_evidence": evidence,
                }
            )
        for k, d in c["direction"].items():
            if not d["moved"]:
                continue
            out.append(
                {
                    **shared,
                    "name": f"lead101_delta_read_{name}_direction_{k}",
                    "description": (
                        f"LEAD-101 news-delta read, {VIEW_WORDS[name]}: {DIRECTION_WORDS[k]}, "
                        "right calls minus wrong calls per moved game, in percentage points; "
                        f"record {d['hits']}-{d['misses']} with {d['no_call']} no call, "
                        f"within-week permutation p {d['permutation_p']:.3f}"
                    ),
                    "plain_summary": (
                        "A language model sees only the news that arrived after the pool line "
                        "froze and guesses which way each line will move by the pick deadline. "
                        "This counts right guesses minus wrong guesses on the games that moved."
                    ),
                    "effect": 100 * d["net_rate"],
                    "effect_units": "rate_pp",
                    "interval_low": 100 * d["interval_low"],
                    "interval_high": 100 * d["interval_high"],
                    "probability_positive": d["probability_positive"],
                    "sample_games": d["moved"],
                    "sample_blocks": c["weeks"],
                    "classification_evidence": evidence,
                }
            )
    return {"family": FAMILY, "cells": out}


def cmd_score(args):
    kept, excluded = declared_records()
    quotes = served_quotes()
    grid = float(np.diff(np.unique(quotes["line"].to_numpy())).min())
    now = pd.Timestamp.now(tz="UTC")
    frame, skipped = score_rows(kept, quotes, now)
    if frame.empty:
        raise SystemExit("no settled row with a quote between the record and the deadline")
    rng = np.random.default_rng(SEED)
    views = [v for v in VIEW_WEEKDAY if frame["view"].eq(v).any()]
    cells = {v: cell_scores(frame.loc[frame["view"].eq(v)], grid, rng) for v in views}
    cells["pooled"] = cell_scores(frame, grid, rng)
    SCORE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SCORE / "rows.csv", index=False)
    payload = {
        "scored_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quote_grid_measured": grid,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "permutation_draws": PERMUTATION_DRAWS,
        "records_scored": [name for name, _ in kept],
        "records_excluded": excluded,
        "rows_skipped": skipped,
        "cells": cells,
    }
    (SCORE / "scores.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    (SCORE / "weak_signals_batch.json").write_text(
        json.dumps(batch_payload(cells), indent=1), encoding="utf-8"
    )
    print("excluded", [(e["file"], e["reason"]) for e in excluded])
    reasons = pd.DataFrame(skipped)
    if not reasons.empty:
        print(reasons.groupby(["week", "view", "reason"]).size().to_string())
        print(reasons.loc[~reasons["reason"].eq("deadline not reached")].to_string(index=False))
    print("quote grid measured", grid)
    shown = ["week", "view", "game_id", "deadline_books", "deadline_line_seen_utc", "pool_line"]
    print(frame[[*shown, "deadline_line", "realised", "read", *BASELINES]].to_string(index=False))
    for name, c in cells.items():
        print(name, "rows", c["rows"], "games", c["games"], "weeks", c["weeks"])
        print(" mae read", round(c["mae_read"], 4))
        for b, e in c["errors"].items():
            print(
                f" vs {b}: baseline mae {e['mae_baseline']:.4f} gain {e['gain']:+.4f} "
                f"[{e['interval_low']:+.4f}, {e['interval_high']:+.4f}] "
                f"probability_positive {e['probability_positive']:.3f}"
            )
        for k, d in c["direction"].items():
            print(f" direction {k}:", json.dumps(d))


def main():
    ap = argparse.ArgumentParser(description="LEAD-101 news-delta read")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enroll").set_defaults(fn=cmd_enroll)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    s = sub.add_parser("run")
    s.add_argument("--week", type=int)
    s.add_argument("--view", choices=["thursday", "sunday", "dry"], required=True)
    s.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
