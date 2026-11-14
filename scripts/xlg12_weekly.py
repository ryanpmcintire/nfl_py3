import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ats import cfb_features as cf  # noqa: E402

ART = ROOT / "artifacts" / "xlg12_cfb_forward"
CFB_ROOT = ROOT / "data" / "cfb"
SEASON = 2026
KEY_ENV = "CFBD_API_KEY"
VERSIONS = ("v1", "v2")
PY = sys.executable
SCRIPTS = ROOT / "scripts"
EXIT_NO_KEY = 2
EXIT_NOTHING_DONE = 3
SPAN = ["--start-season", str(SEASON), "--end-season", str(SEASON)]
INGEST_SCHEDULES = [PY, "-m", "nfl_ats.cli", "cfb-ingest", "--source", "schedules", *SPAN]
INGEST_PBP = [PY, "-m", "nfl_ats.cli", "cfb-ingest", "--source", "pbp", *SPAN]
FEATURES = [PY, str(SCRIPTS / "xlg12_features_2026.py")]


def ensure_key():
    if os.environ.get(KEY_ENV, "").strip():
        return True
    if sys.platform != "win32":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            value, _ = winreg.QueryValueEx(k, KEY_ENV)
    except OSError:
        return False
    if str(value).strip():
        os.environ[KEY_ENV] = str(value).strip()
        return True
    return False


def shown(argv):
    return " ".join(Path(a).name if os.sep in a else a for a in argv)


def run(argv):
    print("run", shown(argv), flush=True)
    done = subprocess.run(argv, cwd=ROOT, check=False)
    if done.returncode != 0:
        print("failed", done.returncode, flush=True)
        sys.exit(done.returncode)


def ledger_path(version):
    base = ART if version == "v1" else ART / version
    return base / "forward_ledger_open.jsonl"


def ledger_rows(version):
    path = ledger_path(version)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def frozen_games(version, week):
    return {r["game_id"] for r in ledger_rows(version) if r["week"] == week}


def schedule():
    s = cf.load_cfb_seasons(
        CFB_ROOT, "schedules", [SEASON], columns=list(cf._SCHEDULE_LOAD_COLUMNS)
    )
    cf.cast_int_columns(s, ("game_id", "season", "week"))
    s = s.loc[s["season_type"].astype(str).eq("regular")].copy()
    s["kickoff"] = pd.to_datetime(s["start_date"], utc=True, format="ISO8601")
    return s


def resolve_weeks(s, now):
    last = s.groupby("week")["kickoff"].max()
    open_weeks = sorted(w for w, t in last.items() if t > now)
    if not open_weeks:
        return None, None
    current = int(open_weeks[0])
    following = int(open_weeks[1]) if len(open_weeks) > 1 else current + 1
    return current, following


def score_report():
    run([PY, str(SCRIPTS / "xlg12_score.py")])
    import xlg12_score

    now = pd.Timestamp(datetime.now(UTC))
    s = schedule()
    final = {int(g) for g in s.loc[s["completed"].eq(True), "game_id"]}
    kicked = {int(g) for g in s.loc[s["kickoff"].le(now), "game_id"]}
    unscored = 0
    lines = []
    for v in VERSIONS:
        frozen = {int(r["game_id"]) for r in ledger_rows(v)}
        scored = {int(g) for g in xlg12_score.load(v)["game_id"]} if frozen else set()
        missing = (frozen & final) - scored
        unscored = max(unscored, len(missing))
        lines.append(
            f"{v} scored {len(scored)} of {len(frozen)} frozen games, "
            f"{len((frozen & kicked) - final)} kicked off without a final score yet, "
            f"{len(frozen - kicked)} not kicked off"
        )
        if missing:
            print(v, "final in the schedule but not scored", sorted(missing))
    return "; ".join(lines), unscored


def fail_unscored(unscored):
    print(
        f"xlg12 score FAILED: {unscored} frozen games are final in the schedule "
        "but were not scored",
        flush=True,
    )
    sys.exit(EXIT_NOTHING_DONE)


def score(_args):
    for st in (INGEST_SCHEDULES, INGEST_PBP, FEATURES):
        run(st)
    text, unscored = score_report()
    print(text, flush=True)
    if unscored:
        fail_unscored(unscored)


def predict_argv(week, version):
    return [
        PY,
        str(SCRIPTS / "xlg12_predict.py"),
        "--forward-week",
        str(week),
        "--open-input",
        "--version",
        version,
    ]


def freeze(args):
    if not args.dry:
        if not ensure_key():
            print(f"{KEY_ENV} not found in environment or HKCU Environment")
            sys.exit(EXIT_NO_KEY)
        run(INGEST_SCHEDULES)
    now = pd.Timestamp(datetime.now(UTC))
    s = schedule()
    current, following = resolve_weeks(s, now)
    if current is None:
        print(
            f"no regular-season game left to kick off in the {SEASON} schedule; nothing to freeze"
        )
        return
    wk = s.loc[s["week"].eq(current)]
    unkicked = {int(g) for g in wk.loc[wk["kickoff"].gt(now), "game_id"]}
    print("current_week", current, "next_week", following, "unkicked_games", len(unkicked))
    before = {v: len(ledger_rows(v)) for v in VERSIONS}
    for v in VERSIONS:
        print(v, "already_frozen_games_week", current, len(frozen_games(v, current)))
    steps = [
        [
            PY,
            str(SCRIPTS / "capture_cfbd_lines.py"),
            "--season",
            str(SEASON),
            "--weeks",
            str(current),
            str(following),
            "--refresh",
        ],
        INGEST_PBP,
        FEATURES,
        *[predict_argv(current, v) for v in VERSIONS],
    ]
    if args.dry:
        for st in (INGEST_SCHEDULES, *steps):
            print("would run", shown(st))
        return
    for st in steps:
        run(st)
    scored_text, unscored = score_report()
    print(scored_text, flush=True)
    feats = pd.read_parquet(ART / "features_2026.parquet", columns=["game_id", "week"])
    import xlg12_predict

    late = unkicked & xlg12_predict.late_games(current)
    freezable = (
        unkicked & {int(g) for g in feats.loc[feats["week"].eq(current), "game_id"]}
    ) - late
    if late:
        print(
            f"week {current} skipped {len(late)} games as non-opener: their earliest line "
            f"capture came after the freeze window of {xlg12_predict.FREEZE_JOB}",
            flush=True,
        )
    silent = 0
    lines = []
    for v in VERSIONS:
        written = len(ledger_rows(v)) - before[v]
        left = freezable - frozen_games(v, current)
        lines.append(
            f"{v} week {current} wrote {written} rows, "
            f"{len(freezable) - len(left)} of {len(freezable)} games not yet kicked off are frozen"
        )
        if written == 0 and left:
            silent = max(silent, len(left))
    print("; ".join(lines), flush=True)
    if silent:
        print(
            f"xlg12 freeze FAILED: wrote 0 rows for week {current} of the regular season "
            f"while {silent} games that had not kicked off have no frozen prediction",
            flush=True,
        )
        sys.exit(EXIT_NOTHING_DONE)
    if unscored:
        fail_unscored(unscored)


def main():
    ap = argparse.ArgumentParser(description="XLG-12 weekly CFB shadow forward score and freeze")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score").set_defaults(fn=score)
    f = sub.add_parser("freeze")
    f.add_argument("--dry", action="store_true")
    f.set_defaults(fn=freeze)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
