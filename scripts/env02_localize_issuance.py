import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SOURCE = Path("data/raw/forecast_archive/pool_decision_2009_2025")
TARGET = Path("data/raw/forecast_archive/pool_decision_2009_2025_aware")
TIME_COLUMNS = (
    "issuance_runtime_utc",
    "forecast_valid_utc",
    "forecast_precip_prob_valid_utc",
    "decision_cutoff_utc",
    "kickoff_utc",
)


def localize(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize("UTC")
    return parsed.dt.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()
    frame = pd.read_parquet(args.source / "forecasts.parquet")
    ok = frame["fetch_status"] == "ok"
    out = frame.copy()
    for column in TIME_COLUMNS:
        if column in out.columns:
            out[column] = localize(out[column])
    steps = out["lookback_steps_used"]
    cutoff = out["decision_cutoff_utc"]
    base = cutoff.dt.floor("12h")
    expected = base - pd.to_timedelta(12 * steps.fillna(0), unit="h")
    derived = (out["issuance_runtime_utc"] == expected) & ok
    pre_cutoff = (out["issuance_runtime_utc"] <= cutoff) & ok
    pre_kickoff = (cutoff <= out["kickoff_utc"]) & ok
    summary = {
        "ok_rows": int(ok.sum()),
        "issuance_equals_requested_utc_runtime": int(derived.sum()),
        "issuance_le_cutoff_pass": int(pre_cutoff.sum()),
        "issuance_le_cutoff_fail": int((ok & ~pre_cutoff).sum()),
        "cutoff_le_kickoff_pass": int(pre_kickoff.sum()),
        "cutoff_le_kickoff_fail": int((ok & ~pre_kickoff).sum()),
    }
    if summary["issuance_equals_requested_utc_runtime"] != summary["ok_rows"]:
        raise SystemExit(json.dumps(summary))
    if summary["issuance_le_cutoff_fail"] or summary["cutoff_le_kickoff_fail"]:
        raise SystemExit(json.dumps(summary))
    args.target.mkdir(parents=True, exist_ok=False)
    path = args.target / "forecasts.parquet"
    out.to_parquet(path, index=False)
    summary["source_sha256"] = hashlib.sha256(
        (args.source / "forecasts.parquet").read_bytes()
    ).hexdigest()
    summary["target_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.target / "localization.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
