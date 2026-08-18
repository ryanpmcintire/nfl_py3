"""The 2026 weekly monitoring rule: anytime-valid, checked every week, for free.

Picks are recorded pre-kickoff every week starting the 2026-09-08 lock
(``docs/prospective_evidence.md``); ``weekly-run``'s step 11
(``prospective-score``) settles them into
``artifacts/prospective_scoring/<run>/settled_decisions.parquet``, one row
per (entrant, game) with ``correct_at_decision_line`` already computed. This
script is what a reader (or a dashboard) runs against that file each week to
get an always-valid readout of "is the challenger beating the baseline" --
without the multiple-testing penalty of treating a fixed-sample bootstrap as
if it were being checked only once, when in fact it will be checked every
single week of the season.

**What this computes, every week:** the anytime-valid confidence sequence
(``nfl_ats.anytime.paired_anytime_comparisons``) for the challenger's paired
per-game ``accuracy_improvement`` over the baseline, one look per completed
week, at ``per_game_variance_proxy=0.55`` (measured on real CFB data) and
``intraclass_correlation=0.0`` -- this project's standing decision that
games within a week are independent events (disjoint teams, no shared
outcome mechanism), not a padded or estimated value; see
``docs/anytime_valid.md``.

**What a reader is entitled to conclude, at ANY week:** read
``plain_english_reading``'s output directly, or the three-way rule it
encodes:

1. Interval excludes zero on the POSITIVE side -- strong, formally valid
   evidence the challenger is ahead. Rare within one season for anything but
   a large effect (see the instrument-property appendix in
   ``docs/anytime_valid.md``); if it happens, it is real and needs no
   additional confirmation to act on.
2. Interval excludes zero on the NEGATIVE side -- strong evidence AGAINST
   the challenger. Different from "unresolved"; worth reconsidering the
   challenger's promotion.
3. Interval still contains zero -- the expected outcome most weeks,
   especially early in the season, and NOT a negative result
   (``AGENTS.md``'s binding rule applies here exactly as it does to the
   fixed-sample method). Report the current point estimate and interval
   width as "how much the evidence has narrowed," and check again next
   week at no validity cost -- that is the entire point of this instrument.

This script does not touch ``registry/rotation_registry.json`` and requires
no NFL rotation window: prospective 2026 scoring "needs no window at all"
(``docs/rotation_registry.md``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.anytime import (
    DEFAULT_ALPHA,
    DEFAULT_TARGET_GAMES,
    anytime_summary,
    paired_anytime_comparisons,
)
from nfl_ats.reporting import artifact_directories

REPO = Path(__file__).resolve().parents[1]
BASELINE_ENTRANT = "active_model"
DEFAULT_CHALLENGER_ID = "mod07_weak_signal_stack"

# docs/anytime_valid.md. per_game_variance_proxy is measured on real CFB
# paired comparisons. intraclass_correlation is a standing project decision
# (independence -- disjoint teams, no shared outcome mechanism), not a
# measured or padded value; keep it at 0.0 unless deliberately stress-testing.
MEASURED_PER_GAME_VARIANCE_PROXY = 0.55
OPERATING_INTRACLASS_CORRELATION = 0.0

_REQUIRED_SETTLED_COLUMNS = frozenset(
    {"entrant", "game_id", "season", "week", "pick_side", "correct_at_decision_line"}
)


def latest_settled_decisions_path(artifacts_root: Path) -> Path:
    """The most recent ``prospective-score`` run's settled ledger."""

    directories = artifact_directories(
        artifacts_root / "prospective_scoring", "settled_decisions.parquet"
    )
    if not directories:
        raise FileNotFoundError(
            f"No settled_decisions.parquet under {artifacts_root / 'prospective_scoring'}. "
            "Run `nfl-ats prospective-score` (weekly-run's step 11) at least once first."
        )
    return directories[0] / "settled_decisions.parquet"


def build_paired_frame(
    settled: pd.DataFrame, *, baseline_entrant: str, challenger_id: str
) -> pd.DataFrame:
    """Adapt the settled prospective ledger to ``paired_anytime_comparisons``'s contract.

    The settled ledger already carries ``correct_at_decision_line`` (whether
    that entrant's pick won, 0/1, NaN if push/pending) and ``pick_side``
    ("HOME"/"AWAY"), not a continuous probability. ``home_cover_probability``
    is reconstructed as exactly 1.0/0.0 from ``pick_side`` -- accuracy_
    improvement only ever tests ``(probability >= 0.5) == actual``, so this
    reconstruction reproduces ``correct_at_decision_line`` exactly, with no
    information lost for this metric. ``home_cover`` (the actual outcome) is
    recovered from ``pick_side`` and ``correct_at_decision_line`` together:
    a HOME pick that was correct means HOME covered, and so on.
    """

    missing = sorted(_REQUIRED_SETTLED_COLUMNS.difference(settled.columns))
    if missing:
        raise ValueError(f"Settled decisions are missing columns: {', '.join(missing)}")

    frames: list[pd.DataFrame] = []
    for entrant, feature_set in ((baseline_entrant, "baseline"), (challenger_id, "challenger")):
        subset = settled.loc[settled["entrant"].astype(str).eq(entrant)].copy()
        subset = subset.loc[subset["correct_at_decision_line"].notna()]
        if subset.empty:
            raise ValueError(
                f"No settled, gradeable (non-push, non-pending) decisions for entrant "
                f"{entrant!r} in the settled ledger"
            )
        pick_home = subset["pick_side"].astype(str).eq("HOME").to_numpy()
        correct = pd.to_numeric(subset["correct_at_decision_line"], errors="raise").to_numpy(
            dtype=float
        )
        home_cover = np.where(pick_home, correct, 1.0 - correct)
        frames.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set,
                    "game_id": subset["game_id"].astype(str).to_numpy(),
                    "season": subset["season"].astype(int).to_numpy(),
                    "week": subset["week"].astype(int).to_numpy(),
                    "home_cover": home_cover,
                    "home_cover_probability": np.where(pick_home, 1.0, 0.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plain_english_reading(summary_row: pd.Series) -> str:
    """Translate one ``anytime_summary`` row into the three-way rule readers need."""

    mean_points = float(summary_row["final_estimate"]) * 100.0
    lower_points = float(summary_row["final_lower"]) * 100.0
    upper_points = float(summary_row["final_upper"]) * 100.0
    games = int(summary_row["games"])
    weeks = int(summary_row["looks"])

    if bool(summary_row["final_excludes_zero"]) and lower_points > 0.0:
        return (
            f"STRONG EVIDENCE FOR the challenger, formally valid despite being checked "
            f"every week so far: current estimate {mean_points:+.2f} accuracy points, "
            f"95% interval [{lower_points:+.2f}, {upper_points:+.2f}] over {games} games "
            f"({weeks} weeks). This is a rare outcome within one season unless the true "
            f"effect is large -- it does not need a second season to be trusted."
        )
    if bool(summary_row["final_excludes_zero"]) and upper_points < 0.0:
        return (
            f"STRONG EVIDENCE AGAINST the challenger: current estimate {mean_points:+.2f} "
            f"accuracy points, 95% interval [{lower_points:+.2f}, {upper_points:+.2f}] over "
            f"{games} games ({weeks} weeks). Different from 'unresolved' -- worth "
            f"reconsidering the challenger's promotion."
        )
    return (
        f"NOT YET RESOLVED -- the expected outcome most weeks, and NOT a negative result "
        f"(AGENTS.md: an interval containing zero is never grounds for rejection). Current "
        f"best estimate {mean_points:+.2f} accuracy points, 95% interval "
        f"[{lower_points:+.2f}, {upper_points:+.2f}] over {games} games ({weeks} weeks). "
        f"This instrument will not formally exclude zero within one ~285-game NFL season "
        f"for anything under roughly 2 accuracy points (docs/anytime_valid.md's "
        f"instrument-property appendix) -- a still-inconclusive reading in the season's "
        f"final week is the expected property of the instrument for a small-to-moderate "
        f"effect, not evidence the effect is absent. Checking again next week costs "
        f"nothing; that is the entire point of this instrument."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO / "artifacts",
        help="repo artifacts root (matches NFL_ATS_ARTIFACTS_DIR)",
    )
    parser.add_argument("--settled-decisions", type=Path, default=None)
    parser.add_argument("--challenger-id", default=DEFAULT_CHALLENGER_ID)
    parser.add_argument("--baseline-entrant", default=BASELINE_ENTRANT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--per-game-variance-proxy", type=float, default=MEASURED_PER_GAME_VARIANCE_PROXY
    )
    parser.add_argument(
        "--intraclass-correlation", type=float, default=OPERATING_INTRACLASS_CORRELATION
    )
    parser.add_argument("--target-games", type=int, default=DEFAULT_TARGET_GAMES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settled_path = args.settled_decisions or latest_settled_decisions_path(args.artifacts_root)
    settled = pd.read_parquet(settled_path)
    paired_input = build_paired_frame(
        settled, baseline_entrant=args.baseline_entrant, challenger_id=args.challenger_id
    )
    trace = paired_anytime_comparisons(
        paired_input,
        baseline_feature_set="baseline",
        metric="accuracy_improvement",
        block="week",
        alpha=args.alpha,
        target_games=args.target_games,
        per_game_variance_proxy=args.per_game_variance_proxy,
        intraclass_correlation=args.intraclass_correlation,
    )
    summary = anytime_summary(trace)
    summary_row = summary.iloc[0]

    reading = plain_english_reading(summary_row)
    result: dict[str, Any] = {
        "settled_decisions": str(settled_path),
        "baseline_entrant": args.baseline_entrant,
        "challenger_id": args.challenger_id,
        "per_game_variance_proxy": args.per_game_variance_proxy,
        "intraclass_correlation": args.intraclass_correlation,
        "summary": json.loads(summary.iloc[[0]].to_json(orient="records"))[0],
        "reading": reading,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n{reading}")

    if args.output is not None:
        trace.to_csv(args.output / "trace.csv", index=False)
        (args.output / "reading.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=float), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
