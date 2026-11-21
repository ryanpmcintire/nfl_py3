import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead99_llm_read as L

ART = L.ROOT / "artifacts" / "lead101" / "market_excluded"
RUNS = ART / "runs"
FROZEN = ART / "frozen.json"
PROMPT = L.ROOT / "scripts" / "lead101_market_excluded_prompt.md"
REMOVED_PROMPT_TEXT = "the latest market spread and its move since the Tuesday opener, "
DROPPED_BLOCKS = ("market",)
LINE_PATTERN = re.compile(
    r"\b\d{1,2}(?:\.5)?[- ]point (?:favou?rite|underdog|dog)s?\b"
    r"|\bfavou?red by \d"
    r"|\b(?:spread|line) (?:of|is|at|to) [+-]?\d"
    r"|(?<![\w.])[+-]\d{1,2}\.5(?![\w.])"
    r"|\bpick'?em\b",
    re.I,
)
NUMBER = re.compile(r"\d{1,2}(?:\.5)?")
ODDS_HOSTS = (
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "bovada",
    "pinnacle",
    "espnbet",
    "sportsbook",
    "covers.com",
    "vegasinsider",
    "actionnetwork",
    "oddsshark",
    "oddschecker",
)
K_DEFAULT = 3
SEED = 20261121
N_BOOT = 10000


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def logit(p):
    return math.log(p / (1 - p))


def rows_of(d):
    return {r["game_id"]: logit(r["home_cover_probability"]) for r in d["parsed_rows"]}


def stamp(d):
    s = d.get("recorded_at_utc") or d.get("run_at_utc")
    return datetime.strptime("".join(c for c in s if c.isdigit())[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def derived_prompt_bytes():
    text = L.PROMPT.read_bytes().decode("utf-8")
    if text.count(REMOVED_PROMPT_TEXT) != 1:
        raise SystemExit("market clause not found exactly once in the LEAD-99 prompt")
    return text.replace(REMOVED_PROMPT_TEXT, "").encode("utf-8")


def enroll(args):
    if L.prompt_sha() != json.loads(L.FROZEN.read_text())["prompt_sha256"]:
        raise SystemExit("LEAD-99 prompt differs from its frozen enrollment")
    want = derived_prompt_bytes()
    if PROMPT.exists() and PROMPT.read_bytes() != want:
        raise SystemExit("existing market-excluded prompt differs from the derivation")
    if FROZEN.exists():
        raise SystemExit(f"already enrolled: {FROZEN.read_text()}")
    PROMPT.write_bytes(want)
    ART.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(
        json.dumps(
            {
                "prompt_file": PROMPT.name,
                "prompt_sha256": L.sha256_bytes(want),
                "derived_from_prompt_sha256": L.prompt_sha(),
                "removed_prompt_text": REMOVED_PROMPT_TEXT,
                "model": L.MODEL,
                "tools": L.TOOLS,
                "dropped_blocks": list(DROPPED_BLOCKS),
                "line_pattern": LINE_PATTERN.pattern,
                "line_pattern_applies_to": ["beat_posts[*].posts[*].text", "injury_headlines[*][*].headline"],
                "reads_per_packet": K_DEFAULT,
                "enrolled_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        )
    )
    print(FROZEN.read_text())


def ablate(packet):
    removed = []
    for g in packet["games"]:
        for b in DROPPED_BLOCKS:
            del g[b]
        for team, v in g["beat_posts"].items():
            hit = [x for x in v["posts"] if LINE_PATTERN.search(x["text"])]
            removed += [{"game_id": g["game_id"], "block": "beat_posts", "team": team, "text": x["text"]} for x in hit]
            v["posts"] = [x for x in v["posts"] if not LINE_PATTERN.search(x["text"])]
            v["n_posts"] = len(v["posts"])
        for team, v in g["injury_headlines"].items():
            if isinstance(v, list):
                hit = [x for x in v if LINE_PATTERN.search(x.get("headline", ""))]
                removed += [{"game_id": g["game_id"], "block": "injury_headlines", "team": team, "text": x["headline"]} for x in hit]
                g["injury_headlines"][team] = [x for x in v if not LINE_PATTERN.search(x.get("headline", ""))]
    return packet, removed


def latest_source():
    week = L.current_week(datetime.now(UTC))
    c = sorted((L.ART / "runs").glob(f"{L.SEASON}-week-{week:02d}-deadline-*.json"))
    if not c:
        raise SystemExit(f"no LEAD-99 deadline packet for week {week}")
    return c[-1]


def run(args):
    fz = json.loads(FROZEN.read_text())
    if fz["prompt_sha256"] != L.sha256_bytes(PROMPT.read_bytes()) or fz["model"] != L.MODEL or fz["tools"] != L.TOOLS:
        raise SystemExit("prompt, model or tools differ from the frozen market-excluded enrollment")
    source = args.source or latest_source()
    d = load(source)
    full_text = json.dumps(d["packet"], ensure_ascii=False, sort_keys=True)
    if L.sha256_bytes(full_text.encode("utf-8")) != d["packet_sha256"]:
        raise SystemExit("stored packet hash mismatch")
    if d["prompt_sha256"] != fz["derived_from_prompt_sha256"]:
        raise SystemExit("source run used a different LEAD-99 prompt")
    done = [p for p in sorted(RUNS.glob(f"{source.stem}-mx-*.json")) if load(p)["valid"]]
    k = args.k - len(done)
    if k <= 0:
        print(source.name, "already has", len(done), "valid market-excluded reads")
        return
    packet, removed = ablate(json.loads(full_text))
    text = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    ids = [g["game_id"] for g in packet["games"]]
    deadlines = {g["game_id"]: datetime.fromisoformat(g["deadline_et"]) for g in packet["games"]}
    prompt_text = PROMPT.read_text(encoding="utf-8")
    cwd = ART / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    print(source.name, "line-bearing items removed", len(removed), [(r["game_id"], r["text"][:80]) for r in removed], flush=True)
    for _ in range(k):
        attempts = []
        rows = None
        for attempt in (1, 2):
            try:
                env, raw = L.call_claude(prompt_text, text, str(cwd))
            except SystemExit as e:
                attempts.append({"attempt": attempt, "valid": False, "error": str(e)})
                continue
            rec = {"attempt": attempt, "raw_response": raw, "envelope": env}
            try:
                rows = L.validate(L.extract_json(raw), ids)
                rec["valid"] = True
            except Exception as e:
                rec["valid"] = False
                rec["error"] = repr(e)
            attempts.append(rec)
            if rows is not None:
                break
        recorded = datetime.now(UTC)
        t = recorded.strftime("%Y%m%dT%H%M%S%fZ")
        path = RUNS / f"{source.stem}-mx-{t}.json"
        path.write_text(
            json.dumps(
                {
                    "source_run": source.name,
                    "source_packet_sha256": d["packet_sha256"],
                    "packet_sha256": L.sha256_bytes(text.encode("utf-8")),
                    "dropped_blocks": list(DROPPED_BLOCKS),
                    "removed_line_items": removed,
                    "prompt_sha256": fz["prompt_sha256"],
                    "model": L.MODEL,
                    "tools": L.TOOLS,
                    "attempts": attempts,
                    "valid": rows is not None,
                    "parsed_rows": rows,
                    "kept_before_deadline": [r["game_id"] for r in rows or [] if recorded < deadlines[r["game_id"]]],
                    "recorded_at_utc": t,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(path.name, "valid", rows is not None, "attempts", len(attempts), flush=True)


def boot_corr(x, y, idx):
    cs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx if np.std(x[i]) > 0 and np.std(y[i]) > 0])
    return float(np.corrcoef(x, y)[0, 1]), float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)), float((cs > 0).mean())


def within_sd(M):
    return float(np.sqrt(M.var(axis=0, ddof=1).mean())) if len(M) > 1 else float("nan")


def contamination(d, pool):
    hosts = 0
    rows = 0
    for r in d["parsed_rows"]:
        hosts += sum(any(h in (s.get("url") or "").lower() for h in ODDS_HOSTS) for s in r["sources"])
        txt = " ".join([r["mechanism"], *map(str, r["key_facts"])])
        nums = [float(n) for m in LINE_PATTERN.finditer(txt) for n in NUMBER.findall(m.group(0))]
        if any(n != abs(pool[r["game_id"]]) for n in nums):
            rows += 1
    turns = max((a.get("envelope") or {}).get("num_turns") or 0 for a in d["attempts"])
    return {"num_turns": turns, "odds_host_sources": hosts, "rows_with_other_line_numbers": rows}


def summarize(args):
    source = args.source
    src = load(source)
    arm_all = [load(p) for p in sorted(RUNS.glob(f"{source.stem}-mx-*.json"))]
    arm = [d for d in arm_all if d["valid"]]
    if not arm:
        raise SystemExit("no valid market-excluded read")
    inc = [src] + [
        d for d in (load(p) for p in sorted((L.ART / "repeats").glob(source.stem + "-repeat-*.json"))) if d["valid"]
    ]
    abl = [
        d for d in (load(p) for p in sorted((L.ART / "ablations").glob(f"{source.stem}-drop-market-*.json"))) if d["valid"]
    ]
    last = max(stamp(d) for d in arm)
    kick = {g["game_id"]: datetime.fromisoformat(g["kickoff_et"]) for g in src["packet"]["games"]}
    pool = {g["game_id"]: g["pool_line"]["home_spread"] for g in src["packet"]["games"]}
    quote = {g["game_id"]: (g.get("market") or {}).get("home_spread_latest") for g in src["packet"]["games"]}
    ids = sorted(g for g in rows_of(src) if kick[g] > last)
    out = {
        "source": source.name,
        "arm_reads": len(arm_all),
        "arm_valid": len(arm),
        "arm_attempts": sum(len(d["attempts"]) for d in arm_all),
        "excluded_kicked_off": sorted(set(rows_of(src)) - set(ids)),
        "games": len(ids),
        "included_reads": len(inc),
        "drop_market_ablation_reads": len(abl),
    }
    A = np.array([[rows_of(d)[g] for g in ids] for d in arm])
    I = np.array([[rows_of(d)[g] for g in ids] for d in inc])
    a, i, s = A.mean(axis=0), I.mean(axis=0), I[0]
    out["within_read_sd_arm"] = within_sd(A)
    out["within_read_sd_included"] = within_sd(I)
    out["between_game_sd_arm"] = float(a.std(ddof=1))
    out["between_game_sd_included"] = float(i.std(ddof=1))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), (N_BOOT, len(ids)))
    out["corr_arm_v_included_mean"] = boot_corr(a, i, idx)
    out["corr_arm_v_included_scheduled_read"] = boot_corr(a, s, idx)
    if abl:
        B = np.array([[rows_of(d)[g] for g in ids] for d in abl])
        out["corr_arm_v_drop_market_ablation"] = boot_corr(a, B.mean(axis=0), idx)
    dd = a - i
    sb = dd[idx].mean(axis=1)
    out["mean_abs_shift"] = float(np.abs(dd).mean())
    out["signed_shift_toward_home"] = [float(dd.mean()), float(np.percentile(sb, 2.5)), float(np.percentile(sb, 97.5)), float((sb > 0).mean())]
    flips = [ids[j] for j in range(len(ids)) if np.sign(a[j]) != np.sign(i[j])]
    out["side_differs_from_included_mean"] = flips
    out["side_differs_from_scheduled_read"] = [ids[j] for j in range(len(ids)) if np.sign(a[j]) != np.sign(s[j])]
    out["per_game"] = [
        {
            "game_id": g,
            "pool_line": pool[g],
            "quote_shown_to_included": quote[g],
            "p_included_mean": round(1 / (1 + math.exp(-i[j])), 4),
            "p_arm_mean": round(1 / (1 + math.exp(-a[j])), 4),
            "arm_read_sd_logit": round(float(A[:, j].std(ddof=1)), 4) if len(A) > 1 else None,
        }
        for j, g in enumerate(ids)
    ]
    out["contamination"] = [contamination(d, pool) for d in arm]
    if args.expected_n:
        f = len(flips) / len(ids)
        n = args.expected_n
        z = NormalDist().inv_cdf(1 - args.alpha / 2) + NormalDist().inv_cdf(args.power)
        out["detectable_accuracy_points"] = {
            "expected_n": n,
            "alpha_two_sided": args.alpha,
            "power": args.power,
            "arm_vs_50": round(100 * z * 0.5 / math.sqrt(n), 1),
            "discordant_fraction_measured": round(f, 3),
            "arm_vs_included_paired": round(100 * z * math.sqrt(f * n) / n, 1) if f else None,
        }
    path = ART / f"summary-{source.stem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_game"}, indent=1))
    for r in out["per_game"]:
        print(r)
    print(path)


def main():
    ap = argparse.ArgumentParser(description="LEAD-101 market-excluded read arm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enroll").set_defaults(fn=enroll)
    r = sub.add_parser("run")
    r.add_argument("--source", type=Path)
    r.add_argument("--k", type=int, default=K_DEFAULT)
    r.set_defaults(fn=run)
    s = sub.add_parser("summarize")
    s.add_argument("--source", type=Path, required=True)
    s.add_argument("--expected-n", type=int)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--power", type=float, default=0.8)
    s.set_defaults(fn=summarize)
    a = ap.parse_args()
    a.fn(a)


main()
