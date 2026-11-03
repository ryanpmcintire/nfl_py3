import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser(description="LEAD-101 news-delta read")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enroll").set_defaults(fn=cmd_enroll)
    s = sub.add_parser("run")
    s.add_argument("--week", type=int)
    s.add_argument("--view", choices=["thursday", "sunday", "dry"], required=True)
    s.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
