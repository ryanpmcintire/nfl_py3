import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^2026_week(\d{2})_field_(\d{8}_\d{4})\.json$")
TOP_REQUIRED = (
    "capture_method",
    "captured_at_et",
    "contest",
    "week",
    "tiebreak_game_id",
    "entrants",
)
ENTRANT_REQUIRED = ("entrant", "tiebreaker_guess")
GUESS_KEYS = ("away_score", "home_score", "total")
ISO_ET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?[+-]\d{2}:\d{2}$")


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(path):
    errs = []
    m = NAME.match(path.name)
    if not m:
        errs.append("filename must match 2026_week<NN>_field_<YYYYMMDD_HHMM>.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in TOP_REQUIRED:
        if k not in d:
            errs.append(f"missing top-level key {k}")
    if errs:
        return errs, d
    if d["capture_method"] != "browser_read_by_agent":
        errs.append("capture_method must be browser_read_by_agent")
    if not ISO_ET.match(str(d["captured_at_et"])):
        errs.append("captured_at_et must be ISO-8601 with offset")
    if m and d["week"] != int(m.group(1)):
        errs.append("week does not match filename")
    for k in ("contest_id", "entries"):
        if k not in d["contest"]:
            errs.append(f"contest.{k} missing")
    if not re.match(r"^\d{4}_\d{2}_[A-Z]+_[A-Z]+$", str(d["tiebreak_game_id"])):
        errs.append("tiebreak_game_id must look like 2026_09_JAX_BAL")
    ents = d["entrants"]
    if not isinstance(ents, list) or not ents:
        errs.append("entrants must be a non-empty list")
        return errs, d
    seen = set()
    for i, e in enumerate(ents):
        for k in ENTRANT_REQUIRED:
            if k not in e:
                errs.append(f"entrants[{i}] missing {k}")
        name = e.get("entrant")
        if name in seen:
            errs.append(f"entrants[{i}] duplicate entrant {name}")
        seen.add(name)
        g = e.get("tiebreaker_guess")
        if g is None:
            continue
        if not isinstance(g, dict) or not all(number(g.get(k)) for k in GUESS_KEYS):
            errs.append(f"entrants[{i}].tiebreaker_guess needs numeric {GUESS_KEYS}")
        elif g["away_score"] + g["home_score"] != g["total"]:
            errs.append(f"entrants[{i}] total != away_score + home_score")
        if "weekly_score" in e and e["weekly_score"] is not None and not number(e["weekly_score"]):
            errs.append(f"entrants[{i}].weekly_score must be numeric or null")
    if number(d["contest"].get("entries")) and len(ents) > d["contest"]["entries"]:
        errs.append("more entrants listed than contest.entries")
    return errs, d


def main():
    ap = argparse.ArgumentParser(
        description="Validate a Splash pool field capture (entrant tiebreaker guesses)"
    )
    ap.add_argument("path", nargs="?")
    ap.add_argument("--schema", action="store_true")
    a = ap.parse_args()
    if a.schema or not a.path:
        print(
            json.dumps(
                {
                    "filename": "data/splash/2026_week<NN>_field_<YYYYMMDD_HHMM>.json",
                    "capture_method": "browser_read_by_agent",
                    "captured_at_et": "ISO-8601 with offset",
                    "contest": {"contest_id": "str", "entries": "int"},
                    "week": "int, equals NN in filename",
                    "tiebreak_game_id": "e.g. 2026_09_JAX_BAL",
                    "entrants": [
                        {
                            "entrant": "display name as shown",
                            "tiebreaker_guess": {
                                "away_score": "number",
                                "home_score": "number",
                                "total": "away+home",
                            },
                            "weekly_score": "number or null (null when not shown)",
                            "best_pick_game_id": "optional str",
                        }
                    ],
                    "rule": "silence is missing: omit entrants not shown; use null guess only when the page shows none",
                },
                indent=1,
            )
        )
        return 0
    p = Path(a.path)
    errs, d = validate(p)
    n = len(d.get("entrants", [])) if isinstance(d, dict) else 0
    print(json.dumps({"file": p.name, "entrants": n, "errors": errs, "valid": not errs}, indent=1))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
