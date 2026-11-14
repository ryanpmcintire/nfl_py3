import argparse
import json
import math
import statistics
from datetime import UTC, datetime

import lead99_llm_read as parent
import live_kickoff
import pandas as pd

ROOT = parent.ROOT
FAMILY = "lead105_tiebreak_total_read"
ART = ROOT / "artifacts" / FAMILY
PROMPT = ROOT / "scripts" / "lead105_prompt.md"
FROZEN = ART / "frozen.json"
TIEBREAKER = ROOT / "tiebreaker.json"
MAX_ATTEMPTS = 2
TIEBREAKER_MARKET_KEYS = ("current_median_total", "market_total")
SERVED_TOTAL_KEY = "projected_total"
TIEBREAKER_FIELDS = (
    "game_id",
    "season",
    "week",
    "generated_at_utc",
    "forecast_artifact",
    "guess_home",
    "guess_away",
    "projected_total",
    "served_total",
    "served_total_method",
    "market_total",
    "current_median_total",
    "tuesday_total",
    "total_move_status",
    "total_move_deadline_utc",
    "total_move_refreshed_at_utc",
    "total_move_quote",
)
QUOTE_COLUMNS = [
    "observed_at_utc",
    "nflverse_game_id",
    "bookmaker_key",
    "market",
    "outcome_side",
    "line",
]
READ_NUMBERS = ("total_points", "total_sd")
READ_SCORE = ("home_points", "away_points")


def prompt_sha():
    return parent.sha256_bytes(PROMPT.read_bytes())


def stamp(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def schedule():
    live = live_kickoff.live_kickoffs()
    return live[live["season"] == parent.SEASON]


def pending_week(run_at):
    live = schedule()
    pending = live[live["live_kickoff"] > pd.Timestamp(run_at)]
    weeks = sorted({int(w) for w in pending["week"]})
    return next((w for w in weeks if parent.splash_files(w)), None)


def last_game(week):
    games = schedule()
    games = games[games["week"] == week]
    last = games[games["live_kickoff"] == games["live_kickoff"].max()]
    if len(last) != 1:
        raise SystemExit(f"schedule names {len(last)} last games for week {week}")
    row = last.iloc[0]
    return row["game_id"], row["live_kickoff"].to_pydatetime()


def build_packet(week, run_at):
    game_id, live_kick = last_game(week)
    full = parent.build_packet(week, run_at)
    games = [g for g in full["games"] if g["game_id"] == game_id]
    if len(games) != 1:
        raise SystemExit(f"last game {game_id} is not in the captured pool week")
    kickoff = min(datetime.fromisoformat(games[0]["kickoff_et"]), live_kick)
    return {**full, "games": games}, games[0], kickoff


def latest_quote_total(game, run_at):
    floor = datetime.fromisoformat(game["pool_line"]["captured_at_et"])
    for d in reversed(parent.stamp_dirs(ROOT / "data" / "market" / "raw", run_at)):
        if parent.dir_time(d.name) < floor:
            return None
        f = d / "quotes.parquet"
        if not f.is_file():
            continue
        q = pd.read_parquet(f, columns=QUOTE_COLUMNS)
        q = q[
            (q["market"] == "totals")
            & (q["outcome_side"] == "OVER")
            & (q["nflverse_game_id"] == game["game_id"])
            & (q["observed_at_utc"] <= pd.Timestamp(run_at))
            & q["line"].notna()
        ]
        if q.empty:
            continue
        q = q[q["observed_at_utc"] == q["observed_at_utc"].max()]
        return {
            "total": float(statistics.median(q["line"].astype(float))),
            "books": int(q["bookmaker_key"].nunique()),
            "observed_at_utc": parent.iso(q["observed_at_utc"].max()),
            "snapshot": d.name,
        }
    return None


def tiebreaker_snapshot(week, game_id):
    raw = TIEBREAKER.read_bytes()
    tb = json.loads(raw)
    if (tb.get("season"), tb.get("week"), tb.get("game_id")) != (parent.SEASON, week, game_id):
        raise SystemExit("tiebreaker.json does not describe the week's last game")
    market_key = next((k for k in TIEBREAKER_MARKET_KEYS if tb.get(k) is not None), None)
    if market_key is None or tb.get(SERVED_TOTAL_KEY) is None:
        raise SystemExit("tiebreaker.json carries no market total or no served total")
    return {
        "sha256": parent.sha256_bytes(raw),
        "market_total_field": market_key,
        "market_total": float(tb[market_key]),
        "served_total_field": SERVED_TOTAL_KEY,
        "served_tiebreaker_total": float(tb[SERVED_TOTAL_KEY]),
        "fields": {k: tb.get(k) for k in TIEBREAKER_FIELDS},
    }


def market_snapshot(game, run_at, served):
    quote = latest_quote_total(game, run_at)
    if quote is not None:
        return quote["total"], "latest_captured_quote", quote
    return served["market_total"], f"tiebreaker.json:{served['market_total_field']}", None


def validate(parsed, game_id):
    assert parsed["game_id"] == game_id, "game mismatch"
    for key in READ_NUMBERS:
        v = parsed[key]
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"bad {key}"
        assert math.isfinite(v) and v > 0, f"bad {key}"
    for key in READ_SCORE:
        v = parsed[key]
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"bad {key}"
        assert float(v).is_integer() and v >= 0, f"bad {key}"
    assert isinstance(parsed["mechanism"], str) and parsed["mechanism"].strip()
    assert isinstance(parsed["key_facts"], list) and isinstance(parsed["sources"], list)
    return parsed


def write_once(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=1))


def resolve(args, run_at):
    week = args.week if args.week is not None else pending_week(run_at)
    if week is None:
        return None
    packet, game, kickoff = build_packet(week, run_at)
    return week, packet, game, kickoff


def cmd_enroll(args):
    if FROZEN.exists():
        raise SystemExit(f"already enrolled: {FROZEN}")
    write_once(
        FROZEN,
        {
            "family": FAMILY,
            "prompt_sha256": prompt_sha(),
            "model": parent.MODEL,
            "tools": parent.TOOLS,
            "packet_fields": parent.PACKET_FIELDS,
            "enrolled_at_utc": stamp(datetime.now(UTC)),
        },
    )
    print(FROZEN.read_text(encoding="utf-8"))


def cmd_packet(args):
    run_at = datetime.now(UTC).replace(microsecond=0)
    resolved = resolve(args, run_at)
    if resolved is None:
        raise SystemExit("no captured pool week has a game still to kick off")
    week, packet, game, kickoff = resolved
    served = tiebreaker_snapshot(week, game["game_id"])
    market_total, market_source, _ = market_snapshot(game, run_at, served)
    print("week", week, game["game_id"], "kickoff", kickoff.astimezone(parent.ET).isoformat())
    print("pool deadline", game["deadline_et"], "kicked off", run_at >= kickoff)
    print(len(json.dumps(packet, ensure_ascii=False, sort_keys=True)), "chars")
    print("market total", market_total, market_source)
    print("served tiebreaker total", served["served_tiebreaker_total"])
    print("prompt sha256", prompt_sha())


def cmd_run(args):
    fz = json.loads(FROZEN.read_text(encoding="utf-8"))
    if (fz["prompt_sha256"], fz["model"], fz["tools"]) != (
        prompt_sha(),
        parent.MODEL,
        parent.TOOLS,
    ):
        raise SystemExit("prompt, model or tools differ from frozen enrollment")
    run_at = datetime.now(UTC).replace(microsecond=0)
    resolved = resolve(args, run_at)
    if resolved is None:
        if args.game_day_only:
            print("skip: no captured pool week has a game still to kick off")
            return
        raise SystemExit("no captured pool week has a game still to kick off")
    week, packet, game, kickoff = resolved
    game_id = game["game_id"]
    if args.game_day_only and (
        kickoff.astimezone(parent.ET).date() != run_at.astimezone(parent.ET).date()
    ):
        print("skip: the last game", game_id, "is not today")
        return
    if run_at >= kickoff:
        raise SystemExit(f"refused: {game_id} has kicked off")
    served = tiebreaker_snapshot(week, game_id)
    market_total, market_source, quote = market_snapshot(game, run_at, served)
    packet_text = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    prompt_text = PROMPT.read_text(encoding="utf-8")
    cwd = ART / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    attempts = []
    read = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        env, raw = parent.call_claude(prompt_text, packet_text, str(cwd))
        rec = {"attempt": attempt, "raw_response": raw, "envelope": env}
        try:
            read = validate(parent.extract_json(raw), game_id)
            rec["valid"] = True
        except Exception as e:
            rec["valid"] = False
            rec["error"] = repr(e)
        attempts.append(rec)
        if read is not None:
            break
    recorded = datetime.now(UTC)
    if recorded >= kickoff:
        raise SystemExit(f"refused: {game_id} kicked off before the read returned")
    deadline = datetime.fromisoformat(game["deadline_et"])
    out = {
        "family": FAMILY,
        "season": parent.SEASON,
        "week": week,
        "game_id": game_id,
        "away": game["away"],
        "home": game["home"],
        "kickoff_et": kickoff.astimezone(parent.ET).isoformat(),
        "pool_deadline_et": deadline.isoformat(),
        "recorded_before_pool_deadline": recorded < deadline,
        "run_at_utc": packet["run_at_utc"],
        "recorded_at_utc": stamp(recorded),
        "prompt_sha256": prompt_sha(),
        "packet_sha256": parent.sha256_bytes(packet_text.encode("utf-8")),
        "model": parent.MODEL,
        "tools": parent.TOOLS,
        "market_total": market_total,
        "market_total_source": market_source,
        "latest_quote": quote,
        "served_tiebreaker_total": served["served_tiebreaker_total"],
        "tiebreaker": served,
        "read": read,
        "attempts": attempts,
        "retried": len(attempts) > 1,
        "packet": packet,
    }
    name = f"{parent.SEASON}-week-{week:02d}-{recorded.strftime('%Y%m%dT%H%M%SZ')}.json"
    if read is None:
        write_once(ART / "failed" / name, out)
        raise SystemExit("no valid output after retry")
    path = ART / "runs" / name
    write_once(path, out)
    print(path)
    print(
        game_id,
        "read total",
        read["total_points"],
        "sd",
        read["total_sd"],
        "score",
        f"{game['away']} {read['away_points']} {game['home']} {read['home_points']}",
        "market",
        market_total,
        "served",
        served["served_tiebreaker_total"],
        "before pool deadline",
        out["recorded_before_pool_deadline"],
    )


def main():
    ap = argparse.ArgumentParser(description="LEAD-105 tiebreak game total read runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enroll").set_defaults(fn=cmd_enroll)
    for n, fn in (("run", cmd_run), ("packet", cmd_packet)):
        s = sub.add_parser(n)
        s.add_argument(
            "--week",
            type=int,
            help="pool week; defaults to the captured week with a game still to kick off",
        )
        if n == "run":
            s.add_argument(
                "--game-day-only",
                action="store_true",
                help="exit quietly unless the week's last game kicks off today",
            )
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
