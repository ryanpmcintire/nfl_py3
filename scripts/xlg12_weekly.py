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


def run(argv):
    print("run", " ".join(Path(a).name if os.sep in a else a for a in argv), flush=True)
    done = subprocess.run(argv, cwd=ROOT, check=False)
    if done.returncode != 0:
        print("failed", done.returncode, flush=True)
        sys.exit(done.returncode)


def ledger_path(version):
    base = ART if version == "v1" else ART / version
    return base / "forward_ledger_open.jsonl"


def frozen_games(version, week):
    path = ledger_path(version)
    if not path.exists():
        return set()
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r["game_id"] for r in rows if r["week"] == week}


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


def score(_args):
    run([PY, str(SCRIPTS / "xlg12_score.py")])


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
    now = pd.Timestamp(datetime.now(UTC))
    current, following = resolve_weeks(schedule(), now)
    if current is None:
        print("no unplayed week in the 2026 schedule")
        return
    s = schedule()
    wk = s.loc[s["week"].eq(current)]
    unkicked = wk.loc[wk["kickoff"].gt(now), "game_id"].nunique()
    print("current_week", current, "next_week", following, "unkicked_games", unkicked)
    for v in VERSIONS:
        print(v, "already_frozen_games_week", current, len(frozen_games(v, current)))
    span = ["--start-season", str(SEASON), "--end-season", str(SEASON)]
    steps = [
        [
            PY,
            str(SCRIPTS / "capture_cfbd_lines.py"),
            "--season",
            str(SEASON),
            "--weeks",
            str(current),
            str(following),
        ],
        [PY, "-m", "nfl_ats.cli", "cfb-ingest", "--source", "schedules", *span],
        [PY, "-m", "nfl_ats.cli", "cfb-ingest", "--source", "pbp", *span],
        [PY, str(SCRIPTS / "xlg12_features_2026.py")],
        *[predict_argv(current, v) for v in VERSIONS],
    ]
    if args.dry:
        for st in steps:
            print("would run", " ".join(Path(a).name if os.sep in a else a for a in st))
        return
    if not ensure_key():
        print(f"{KEY_ENV} not found in environment or HKCU Environment")
        sys.exit(2)
    for st in steps:
        run(st)


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
