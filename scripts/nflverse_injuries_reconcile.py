"""Reconcile the new full-column nflverse injuries snapshot
(``scripts/nflverse_injuries_ingest.py``) against the repo's existing NFL.com
league-injury-report scrape (``data/raw/nflcom_injuries/*/injuries.parquet``)
on their three overlapping seasons, 2022-2024.

This answers the question ``docs/new_lead_classes_20260826.md`` section 1
left open: are the two sources interchangeable, in particular on the
``illness`` designation the illness battery (``docs/illness_battery.md``)
depends on? It matters because the repo's existing injury-derived features
were built against the NFL.com scrape (via
``nfl_ats.prospective.latest_nflcom_injuries_snapshot``), not this new
source.

Join key: ``(season, week, team, normalized full name)`` exact match first,
then a fallback ``(season, week, team, first-initial+last-name)`` match when
that key is unique on both sides -- the identical two-tier matching
``scripts/ingest_nflcom_injuries.py``'s own ``agreement()`` function already
uses (name normalization functions imported from there, not reimplemented).

nflverse carries the FULL per-row revision history (one row per report
issued, ``date_modified``-stamped); the NFL.com scrape is a single
per-week snapshot representing that week's FINAL reported state. To compare
like with like, each nflverse (season, week, team, gsis_id) group is
collapsed to its own latest (``date_modified``-max) row before joining --
the fair "final state" comparison, not a full-history vs single-snapshot
mismatch.

Not an ATS experiment: no hypothesis, no cell, no verdict, nothing recorded
to ``registry/weak_signals.json``. This is a data-quality reconciliation
report, the same category ``ingest_nflcom_injuries.py --agreement`` already
occupies (and is allowlisted in ``tests/test_experiment_registry.py`` for
the same reason: a plain ``atomic_json`` write to ``artifacts/...``, not a
``write_experiment_artifact`` provenance-registry row). This script is
added to that same allowlist.

Writes ``artifacts/nflverse_injuries_reconcile/<UTC>/agreement.json``.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

ILLNESS_TEXT_COLS = (
    "report_primary_injury",
    "report_secondary_injury",
    "practice_primary_injury",
    "practice_secondary_injury",
)
OVERLAP_SEASONS = (2022, 2023, 2024)


def _load_nflcom_module() -> Any:
    """Import ``scripts/ingest_nflcom_injuries.py`` by path (it is not a
    package module) purely for its ``normalize_name``/``initial_last_key``
    helpers -- reused verbatim, not reimplemented, so this script's matching
    is provably identical to the repo's existing agreement check."""

    spec = importlib.util.spec_from_file_location(
        "ingest_nflcom_injuries_for_reconcile", REPO / "scripts" / "ingest_nflcom_injuries.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _latest_nonempty_under(root: Path, pattern: str, label: str, seasons: tuple[int, ...]) -> Path:
    """Newest snapshot matching ``pattern`` that actually has rows for the
    requested seasons -- the plain lexicographically-last snapshot is NOT
    safe to assume here: the nflcom_injuries source has newer snapshots
    (in-season ``--current`` captures) that hold zero rows because they were
    taken in the offseason, and picking one of those silently reconciles
    against nothing rather than failing loudly."""

    candidates = sorted(root.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {pattern!r} under {root}")
    for candidate in candidates:
        frame = pd.read_parquet(candidate, columns=["season"])
        if frame["season"].isin(seasons).any():
            return candidate
    raise FileNotFoundError(
        f"no {label} under {root} has any row for seasons {list(seasons)} "
        f"(checked {len(candidates)} snapshot(s))"
    )


def _has_illness(row: pd.Series) -> bool:
    for col in ILLNESS_TEXT_COLS:
        value = row.get(col)
        if isinstance(value, str) and "illness" in value.lower():
            return True
    return False


def load_nflverse_final_state(injuries_path: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    """Collapse the full nflverse revision history to one row per
    (season, week, team, gsis_id): the row with the latest ``date_modified``
    -- the "final reported state" comparable to the NFL.com scrape's single
    per-week snapshot."""

    frame = pd.read_parquet(injuries_path)
    frame = frame.loc[
        (frame["game_type"].astype(str) == "REG") & (frame["season"].isin(seasons))
    ].copy()
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], errors="coerce", utc=True)
    frame["is_illness"] = frame.apply(_has_illness, axis=1)
    # Rows with a null date_modified sort last under NaT-goes-first ascending
    # sort unless handled explicitly; put them FIRST (oldest) so a real
    # timestamped revision always wins the "latest" slot when one exists.
    frame = frame.sort_values(
        ["season", "week", "team", "gsis_id", "date_modified"], na_position="first"
    )
    final = frame.drop_duplicates(subset=["season", "week", "team", "gsis_id"], keep="last")
    return final.reset_index(drop=True)


def _anti_join(frame: pd.DataFrame, keys_df: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    merged = frame.merge(keys_df, on=on, how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only"].drop(columns="_merge")


def run_reconcile(
    nflverse_path: Path, nflcom_path: Path, artifacts_root: Path, seasons: tuple[int, ...]
) -> dict[str, Any]:
    """Fully vectorized (merge-based, no row-wise apply/iterrows) two-tier
    match: exact normalized-name, then fallback first-initial+last-name when
    unique on both remaining sides -- same rule as
    ``ingest_nflcom_injuries.py``'s ``agreement()``, just implemented with
    ``pd.merge`` instead of per-row loops for tractable runtime at this
    row count (~17k nflcom rows x ~17k nflverse final-state rows)."""

    nflcom_module = _load_nflcom_module()

    nflverse_full = load_nflverse_final_state(nflverse_path, seasons)
    nflverse_full["norm_name"] = nflverse_full["full_name"].map(nflcom_module.normalize_name)
    init_last_v = nflverse_full["full_name"].map(nflcom_module.initial_last_key)
    nflverse_full["init"] = [pair[0] for pair in init_last_v]
    nflverse_full["last"] = [pair[1] for pair in init_last_v]

    nflcom_full = pd.read_parquet(nflcom_path)
    nflcom_full = nflcom_full.loc[nflcom_full["season"].isin(seasons)].copy()
    nflcom_full["norm_name"] = nflcom_full["player"].map(nflcom_module.normalize_name)
    init_last_c = nflcom_full["player"].map(nflcom_module.initial_last_key)
    nflcom_full["init"] = [pair[0] for pair in init_last_c]
    nflcom_full["last"] = [pair[1] for pair in init_last_c]
    nflcom_full["is_illness"] = (
        nflcom_full["injury"].astype(str).str.lower().str.contains("illness", na=False)
    )

    key4 = ["season", "week", "team", "norm_name"]
    key5 = ["season", "week", "team", "init", "last"]
    v_cols = [*key4, "init", "last", "is_illness", "report_status"]
    c_cols = [*key4, "init", "last", "is_illness", "game_status"]
    nflverse = nflverse_full[v_cols].copy()
    nflcom = nflcom_full[c_cols].copy()

    # Tier 1: exact (season, week, team, normalized name).
    exact_pairs = nflverse.merge(nflcom, on=key4, how="inner", suffixes=("_v", "_c"))
    matched_keys_df = exact_pairs[key4].drop_duplicates()

    # Tier 2: fallback fuzzy (first-initial + last name) match, only when
    # that key is unique on both remaining sides within (season, week, team).
    nflverse_remaining = _anti_join(nflverse, matched_keys_df, key4)
    nflcom_remaining = _anti_join(nflcom, matched_keys_df, key4)

    v_counts = nflverse_remaining.groupby(key5).size()
    c_counts = nflcom_remaining.groupby(key5).size()
    fuzzy_keys = sorted(
        key
        for key in (set(v_counts[v_counts == 1].index) & set(c_counts[c_counts == 1].index))
        if key[4] != ""  # non-empty last name required
    )
    fuzzy_keys_df = pd.DataFrame(fuzzy_keys, columns=key5)

    nflverse_fuzzy_subset = nflverse_remaining.merge(fuzzy_keys_df, on=key5, how="inner")
    nflcom_fuzzy_subset = nflcom_remaining.merge(fuzzy_keys_df, on=key5, how="inner")
    fuzzy_pairs = nflverse_fuzzy_subset.merge(
        nflcom_fuzzy_subset, on=key5, how="inner", suffixes=("_v", "_c")
    )

    pair_cols = ["season", "week", "team", "is_illness_v", "is_illness_c"]
    status_cols_v = "report_status"
    status_cols_c = "game_status"
    all_pairs = pd.concat(
        [
            exact_pairs[[*pair_cols, status_cols_v, status_cols_c]],
            fuzzy_pairs[[*pair_cols, status_cols_v, status_cols_c]],
        ],
        ignore_index=True,
    )

    all_pairs["report_status_norm"] = all_pairs["report_status"].fillna("-").astype(str)
    all_pairs["game_status_norm"] = all_pairs["game_status"].fillna("-").astype(str)
    status_agree = int((all_pairs["report_status_norm"] == all_pairs["game_status_norm"]).sum())
    status_confusion = Counter(
        (all_pairs["report_status_norm"] + "|" + all_pairs["game_status_norm"]).tolist()
    )

    illness_agree = int((all_pairs["is_illness_v"] == all_pairs["is_illness_c"]).sum())
    illness_confusion = Counter(
        (
            "nflverse="
            + all_pairs["is_illness_v"].astype(str)
            + "|nflcom="
            + all_pairs["is_illness_c"].astype(str)
        ).tolist()
    )
    n_pairs = len(all_pairs)
    n_illness_either = int((all_pairs["is_illness_v"] | all_pairs["is_illness_c"]).sum())
    n_illness_both = int((all_pairs["is_illness_v"] & all_pairs["is_illness_c"]).sum())

    matched_keys = set(matched_keys_df.itertuples(index=False, name=None))
    nflverse = nflverse_full
    nflcom = nflcom_full
    coverage = {
        "seasons": list(seasons),
        "nflcom_rows_total": len(nflcom),
        "nflcom_rows_per_season": {str(s): int((nflcom["season"] == s).sum()) for s in seasons},
        "nflverse_final_state_rows_total": len(nflverse),
        "nflverse_final_state_rows_per_season": {
            str(s): int((nflverse["season"] == s).sum()) for s in seasons
        },
        "nflcom_illness_rows": int(nflcom["is_illness"].sum()),
        "nflverse_illness_rows_final_state": int(nflverse["is_illness"].sum()),
        "matched_exact_name": len(matched_keys),
        "matched_fuzzy_initial_last": len(fuzzy_keys),
        "matched_pairs_total": n_pairs,
        "match_rate_vs_nflcom": float(round(n_pairs / max(len(nflcom), 1), 4)),
        "match_rate_vs_nflverse_final_state": float(round(n_pairs / max(len(nflverse), 1), 4)),
    }
    status_comparison = {
        "comparable_matched_rows": n_pairs,
        "exact_agreement": status_agree,
        "agreement_rate": float(round(status_agree / max(n_pairs, 1), 4)),
        "confusion_top": dict(status_confusion.most_common(25)),
    }
    illness_comparison = {
        "comparable_matched_rows": n_pairs,
        "n_illness_by_either_source": n_illness_either,
        "n_illness_by_both_sources": n_illness_both,
        "exact_agreement": illness_agree,
        "agreement_rate": float(round(illness_agree / max(n_pairs, 1), 4)),
        "disagreement_rate": float(round(1.0 - illness_agree / max(n_pairs, 1), 4)),
        "confusion": dict(illness_confusion.most_common(10)),
    }

    result = {
        "schema": "nflverse_injuries_reconcile/1",
        "nflverse_source": str(nflverse_path.relative_to(REPO)),
        "nflcom_source": str(nflcom_path.relative_to(REPO)),
        "normalization": (
            "casefold, ASCII-fold accents, drop punctuation and suffix tokens "
            "(jr/sr/ii/iii/iv/v); join season+week+team+normalized full name, then "
            "first-initial+last-name when that key is unique on both sides within "
            "the same season+week+team (identical rule to "
            "scripts/ingest_nflcom_injuries.py's own agreement() function)."
        ),
        "coverage": coverage,
        "status_comparison": status_comparison,
        "illness_comparison": illness_comparison,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = artifacts_root / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    write_stamped_artifact(result, out_dir / "agreement.json")  # ENG-38
    print(f"wrote {out_dir / 'agreement.json'}")
    return result


def _latest_nflverse_injuries(root: Path, seasons: tuple[int, ...]) -> Path:
    return _latest_nonempty_under(root, "*/injuries.parquet", "nflverse_injuries snapshot", seasons)


def _latest_nflcom_injuries(root: Path, seasons: tuple[int, ...]) -> Path:
    return _latest_nonempty_under(root, "*/injuries.parquet", "nflcom_injuries snapshot", seasons)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nflverse", type=Path, default=None, help="path to a nflverse_injuries injuries.parquet"
    )
    parser.add_argument(
        "--nflcom", type=Path, default=None, help="path to a nflcom_injuries injuries.parquet"
    )
    parser.add_argument(
        "--artifacts-root", type=Path, default=REPO / "artifacts" / "nflverse_injuries_reconcile"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(OVERLAP_SEASONS))
    args = parser.parse_args()

    seasons_tuple = tuple(args.seasons)
    nflverse_path = args.nflverse or _latest_nflverse_injuries(
        REPO / "data" / "raw" / "nflverse_injuries", seasons_tuple
    )
    nflcom_path = args.nflcom or _latest_nflcom_injuries(
        REPO / "data" / "raw" / "nflcom_injuries", seasons_tuple
    )
    print(f"nflverse source: {nflverse_path}")
    print(f"nflcom source:   {nflcom_path}")

    result = run_reconcile(nflverse_path, nflcom_path, args.artifacts_root, tuple(args.seasons))
    print("\n=== coverage ===")
    for key, value in result["coverage"].items():
        print(f"  {key}: {value}")
    print("\n=== status_comparison ===")
    for key, value in result["status_comparison"].items():
        if key != "confusion_top":
            print(f"  {key}: {value}")
    print("\n=== illness_comparison ===")
    for key, value in result["illness_comparison"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
