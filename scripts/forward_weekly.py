import argparse
import sys
from pathlib import Path

import eng50_online_stacker
import forward_ledger
import forward_monitor
import lead101_delta_read
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _settled_weeks():
    games = pd.read_parquet(
        ROOT / "data/processed/game_features.parquet", columns=["season", "week", "result"]
    )
    games = games.loc[games["season"].eq(forward_ledger.SEASON)]
    settled = games.groupby("week")["result"].apply(lambda column: bool(column.notna().all()))
    return [str(int(week)) for week in settled.index[settled]]


def _run(module, *arguments):
    sys.argv = [module.__name__, *arguments]
    module.main()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Weekly refresh of the matched forward ledger over fully settled weeks, followed "
            "by the evidence monitor and the online stacker. Research only; serves nothing."
        )
    )
    parser.parse_args()
    weeks = _settled_weeks()
    if not weeks:
        raise SystemExit("no fully settled week in game_features")
    _run(forward_ledger, "--weeks", *weeks, "--out", "artifacts/forward_ledger_allweeks")
    _run(
        forward_ledger,
        "--weeks",
        *weeks,
        "--require",
        *forward_monitor.CANDIDATES,
        "--out",
        "artifacts/forward_ledger_core",
    )
    _run(forward_monitor)
    _run(eng50_online_stacker)
    _run(lead101_delta_read, "score")


if __name__ == "__main__":
    main()
