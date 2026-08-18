"""Load every accuracy-scale effect measurement recorded in artifacts/.

Scans ``artifacts/**/*paired*.csv`` (27 files) for rows on the forced-pick
*accuracy* scale -- excluding Brier, log-loss, and MAE rows, which are a
different currency and would silently distort a prior meant to answer "how
big is a typical accuracy effect in this project's history." That filter
alone recovers exactly 282 rows, which is the count of "recorded effect
measurements" this task cites.

Those 282 rows are not 282 independent draws. Most artifacts report the same
point estimate twice -- once resampled by week block, once by season block,
for the same arm/window/metric -- purely to get two versions of the interval,
not because the underlying comparison ran twice. Treating both as separate
"studies" would pseudo-replicate the sample and understate the prior's
between-study variance (this is checked explicitly in
``docs/decision_rule.md``). This loader collapses exact-duplicate point
estimates (same source file, same identifying columns, same evaluation
window, same estimate to 6 decimal places) down to one row, keeping the
wider (more conservative) of the two intervals. That leaves 210 usable,
independent measurements (two more rows are dropped outright: a same-file
zero-width-CI sanity check where a feature set was compared against itself,
which is a pipeline no-op, not a measurement of an effect).

Run directly to regenerate the canonical CSV:

    uv run --no-sync python scripts/decision_load_measurements.py <out.csv>
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

_Z95 = 1.959963984540054

# Column-name pairs tried in order to build a human-readable label for a row.
_ID_COLUMN_PAIRS = (
    ("baseline_feature_set", "candidate_feature_set"),
    ("baseline_feature_profile", "candidate_feature_profile"),
    ("baseline_availability_method", "candidate_availability_method"),
    ("arm", "baseline_arm"),
)


def _label_for_row(row: pd.Series, columns: set[str]) -> str:
    for left, right in _ID_COLUMN_PAIRS:
        if left in columns and right in columns:
            return f"{row[left]}~{row[right]}"
    return "unlabeled"


def load_raw_accuracy_rows(artifacts_root: Path) -> pd.DataFrame:
    """Every accuracy-scale row across the paired-comparison artifacts, undeduplicated."""

    files = sorted(glob.glob(str(artifacts_root / "**" / "*paired*.csv"), recursive=True))
    records: list[dict[str, object]] = []

    for file_path in files:
        df = pd.read_csv(file_path)
        columns = set(df.columns)
        rel_source = str(file_path)

        if "metric" in columns:
            estimate_col = "estimate" if "estimate" in columns else "improvement"
            sub = df[df["metric"].isin(["accuracy_improvement", "accuracy"])]
            for _, row in sub.iterrows():
                lower = float(row["lower"]) * 100.0
                upper = float(row["upper"]) * 100.0
                n_raw = row.get("paired_games", row.get("games"))
                records.append(
                    {
                        "source": rel_source,
                        "label": _label_for_row(row, columns),
                        "window": str(row.get("evaluation_window", row.get("block", ""))),
                        "block": row.get("block"),
                        "estimate": float(row[estimate_col]) * 100.0,
                        "lower": lower,
                        "upper": upper,
                        "n_games": None if pd.isna(n_raw) else int(n_raw),
                        "probability_positive": (
                            None
                            if pd.isna(row.get("probability_positive"))
                            else float(row["probability_positive"])
                        ),
                    }
                )
        elif "accuracy_diff_points" in columns:
            for _, row in df.iterrows():
                n = row.get("non_push_games")
                p_bar = (float(row["base_accuracy"]) + float(row["pbp_accuracy"])) / 2.0
                se = _paired_proportion_se(p_bar, n)
                records.append(
                    {
                        "source": rel_source,
                        "label": f"season_{int(row['season'])}",
                        "window": "season_table",
                        "block": "season",
                        "estimate": float(row["accuracy_diff_points"]),
                        "lower": (float(row["accuracy_diff_points"]) - _Z95 * se) if se else None,
                        "upper": (float(row["accuracy_diff_points"]) + _Z95 * se) if se else None,
                        "n_games": None if pd.isna(n) else int(n),
                        "probability_positive": None,
                    }
                )
        elif "diff_vs_base_alpha10_points" in columns:
            for _, row in df.iterrows():
                n = row.get("non_push_games")
                p_bar = (
                    float(row["base_alpha10_accuracy"]) + float(row["candidate_accuracy"])
                ) / 2.0
                se = _paired_proportion_se(p_bar, n)
                estimate = float(row["diff_vs_base_alpha10_points"])
                records.append(
                    {
                        "source": rel_source,
                        "label": f"season_{int(row['season'])}",
                        "window": "season_table",
                        "block": "season",
                        "estimate": estimate,
                        "lower": (estimate - _Z95 * se) if se else None,
                        "upper": (estimate + _Z95 * se) if se else None,
                        "n_games": None if pd.isna(n) else int(n),
                        "probability_positive": None,
                    }
                )

    return pd.DataFrame.from_records(records)


def _paired_proportion_se(p_bar: float, n: float | None) -> float | None:
    """Conservative SE for a paired-accuracy difference on ``n`` shared games.

    Assumes zero correlation between the two arms' per-game correctness,
    which overstates the SE (a real positive correlation, which two arms
    scored on the identical games will have, only shrinks it) -- deliberately
    the conservative direction absent the raw per-game data needed to compute
    the true paired covariance.
    """

    if n is None or (isinstance(n, float) and math.isnan(n)) or n <= 0:
        return None
    # Var(p1 - p2) ~= Var(p1) + Var(p2) ~= 2 * p_bar * (1 - p_bar) / n when
    # p1 ~= p2 ~= p_bar and correlation is (conservatively) ignored -- the
    # same approximation used for penalty_discipline in scripts/decision_apply.py.
    return math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n) * 100.0


def deduplicate(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse exact-duplicate point estimates from week/season block resampling.

    Also drops zero-width-CI rows (``lower == upper == estimate == 0`` with a
    genuinely zero, not just rounded, standard error) -- these are pipeline
    self-comparisons, not measurements.
    """

    usable = raw.copy()
    usable["se"] = (usable["upper"] - usable["lower"]) / (2.0 * _Z95)
    usable = usable[(usable["se"].notna()) & (usable["se"] > 1e-9)].copy()

    usable["_estimate_round"] = usable["estimate"].round(6)
    dedup_keys = ["source", "label", "window", "_estimate_round"]
    # Keep the row with the wider (more conservative) interval per group.
    usable = usable.sort_values("se", ascending=False)
    deduped = usable.drop_duplicates(subset=dedup_keys, keep="first")
    deduped = deduped.drop(columns=["_estimate_round"]).sort_values(["source", "label"])
    return deduped.reset_index(drop=True)


def load_measurements(artifacts_root: Path | None = None) -> pd.DataFrame:
    """The canonical, deduplicated table: 210 rows, columns estimate/se/n_games/source."""

    root = artifacts_root if artifacts_root is not None else (REPO_ROOT / "artifacts")
    raw = load_raw_accuracy_rows(root)
    return deduplicate(raw)


def main(argv: list[str]) -> int:
    out_path = Path(argv[0]) if argv else REPO_ROOT / "artifacts" / "decision_rule_measurements.csv"
    raw = load_raw_accuracy_rows(REPO_ROOT / "artifacts")
    deduped = load_measurements()
    print(f"raw accuracy-scale rows: {len(raw)}")
    print(f"after deduplication: {len(deduped)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
