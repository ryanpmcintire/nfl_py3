import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forward2026_ftn_terms as f
import mod24_ftn_turnover_luck as m
import mod25_wf_forward as mwf
import numpy as np
import pandas as pd

from nfl_ats.margin import (
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    _target_values,
    make_margin_estimator,
)

OUT = m.ROOT / "artifacts" / "mod24_postseason_forward"
SEASON = 2026
FROZEN = OUT / "frozen_rule.json"
RULE = (
    "arm = base REG rows plus every labelled non-REG game with season < 2026; "
    "base = REG rows only; both expanding past-only, refit each week W on 2026 REG weeks < W; "
    "same ridge, same columns; arm delta = arm margin minus base margin, "
    "mapped through the locked line sweep from the Tuesday opener"
)


def train_mask(table, week, with_post):
    labelled = table["ats_margin"].notna()
    reg = table["game_type"].eq("REG")
    past_reg = reg & (
        table["season"].lt(SEASON) | (table["season"].eq(SEASON) & table["week"].lt(week))
    )
    mask = labelled & past_reg
    if with_post:
        mask = mask | (labelled & ~reg & table["season"].lt(SEASON))
    return mask


def fit(table, week, with_post):
    columns = list(m.BASE_COLUMNS)
    suppressed = tuple(
        c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(m.PROFILE, ()) if c in columns
    )
    train = table.loc[train_mask(table, week, with_post)]
    model = make_margin_estimator(
        "ridge", ridge_alpha=m.BASE_ALPHA, suppressed_indicator_columns=suppressed
    )
    model.fit(train.loc[:, columns], _target_values(train, "market_residual"))
    return model, train


def stage_freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if FROZEN.exists():
        raise SystemExit("frozen rule already exists; refusing to overwrite")
    served = pd.read_parquet(m.SERVED_FEATURES)
    post = served.loc[
        ~served["game_type"].eq("REG") & served["ats_margin"].notna() & served["season"].lt(SEASON)
    ]
    reg = served.loc[
        served["game_type"].eq("REG") & served["ats_margin"].notna() & served["season"].lt(SEASON)
    ]
    ids = sorted(post["game_id"])
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "rule": RULE,
        "ridge_alpha": m.BASE_ALPHA,
        "profile": m.PROFILE,
        "target": "market_residual",
        "columns": list(m.BASE_COLUMNS),
        "postseason_rows": len(post),
        "postseason_max_season": int(post["season"].max()),
        "postseason_ids_sha256": hashlib.sha256("|".join(ids).encode()).hexdigest(),
        "reg_rows_before_2026": len(reg),
        "tuned_parameters": 0,
        "looks": "3 metrics (accuracy, log loss, Brier), family mod24_postseason_rows",
    }
    FROZEN.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


def stage_score(first, last):
    spec = json.loads(FROZEN.read_text())
    weeks = tuple(range(first, last + 1))
    suffix = f"_forward2026_w{first:02d}_{last:02d}"
    served = pd.read_parquet(m.SERVED_FEATURES)
    served["gameday"] = pd.to_datetime(served["gameday"])
    served26 = served.loc[served["season"].eq(SEASON) & served["week"].isin(weeks)]
    openers = mwf.tuesday_open(served26)
    rows = f.locked_inputs(served26)
    deltas = {}
    counts = {}
    for week in weeks:
        wk = served26.loc[served26["week"].eq(week) & served26["game_id"].isin(rows.keys())]
        if wk.empty:
            continue
        base_model, base_train = fit(served, week, False)
        arm_model, arm_train = fit(served, week, True)
        cols = list(m.BASE_COLUMNS)
        d = np.ravel(arm_model.predict(wk.loc[:, cols])) - np.ravel(
            base_model.predict(wk.loc[:, cols])
        )
        deltas.update(dict(zip(wk["game_id"], d, strict=True)))
        counts[week] = {"base_rows": len(base_train), "arm_rows": len(arm_train)}
    f.OUT = OUT
    res = f.score("postseason" + suffix, deltas, rows, served26, openers)
    report = {
        "frozen_created_at": spec["created_at"],
        "scored_at": datetime.now(UTC).isoformat(),
        "weeks": list(weeks),
        "locked_games": len(rows),
        "openers": len(openers),
        "train_counts": counts,
        "result": res,
    }
    path = OUT / f"report{suffix}.json"
    path.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main():
    args = sys.argv[1:]
    if args == ["freeze"]:
        stage_freeze()
    elif len(args) == 3 and args[0] == "score":
        stage_score(int(args[1]), int(args[2]))
    else:
        raise SystemExit("usage: mod24_postseason_forward.py freeze | score FIRST LAST")


if __name__ == "__main__":
    main()
