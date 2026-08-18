"""Overlap audit for the player-availability / player-value evidence family.

ROADMAP RWB-16 records: "The one family where it accumulates is player
availability and value -- five measurements, all positive, p = 0.0625 within
the family." That p-value is a two-sided sign test, and a sign test assumes the
measurements are INDEPENDENT bits. This script checks that assumption instead of
inheriting it, three ways:

1. **Game overlap.** Which games did each measurement actually score?
2. **Error correlation.** Per-game paired correctness differences
   (candidate_correct - baseline_correct) for every contrast in the family, then
   the correlation matrix over the games they share. Two contrasts that move
   together on the same games are not two bits of evidence.
3. **Effective independent count.** Cheverud/Nyholt ``M_eff`` from the
   eigenvalues of that correlation matrix, and the sign-test p-value recomputed
   at ``M_eff`` instead of at the nominal count.

Everything here reads artifacts that already exist. Nothing is re-fit, no window
is touched, no registry entry is written.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/availability_overlap_audit.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

PLAYER_ABLATION = REPO / "artifacts/player_experiments/20260813T122348Z/predictions.parquet"
AVAILABILITY_ABLATION = (
    REPO / "artifacts/availability_experiments/20260813T133345Z/predictions.parquet"
)
PARTICIPATION_ABLATION = (
    REPO / "artifacts/participation_experiments/20260813T132030Z/predictions.parquet"
)
MOD07_AVAILABILITY = (
    REPO
    / "artifacts/availability_experiments"
    / "mod07_ablation_2020_2021.B_minus_A_availability_and_value.parquet"
)
MOD07_RECORDED = (
    REPO
    / "artifacts/availability_experiments/mod07_ablation_2020_2021.C_minus_A_recorded_look.parquet"
)


def _correct(frame: pd.DataFrame) -> pd.Series:
    """Forced-pick correctness, using the evaluator's own tie convention.

    ``>= 0.5`` (a coin-flip probability picks HOME), not ``> 0.5``. The two
    differ on the exactly-0.5 rows, which the profiles produce often enough to
    move a contrast by ~0.2 accuracy points; ``>= 0.5`` is the convention that
    reproduces every ``cover_accuracy`` in the recorded ``summary.csv`` files
    (checked against player_value 0.5214457831 and player_participation
    0.5171084337 before this audit reported anything).
    """

    scored = frame.loc[frame["home_cover"].notna()]
    pick_home = scored["home_cover_probability"].astype(float) >= 0.5
    return pd.Series(
        (pick_home.to_numpy() == scored["home_cover"].astype(bool).to_numpy()).astype(float),
        index=scored["game_id"].to_numpy(),
    )


def _profile_correct(path: Path, column: str, value: str) -> pd.Series:
    frame = pd.read_parquet(path)
    return _correct(frame.loc[frame[column] == value])


def build_contrasts() -> dict[str, pd.Series]:
    """Per-game (candidate - baseline) correctness for every family measurement."""

    player = pd.read_parquet(PLAYER_ABLATION)
    base = _correct(player.loc[player["feature_profile"] == "base"])
    contrasts: dict[str, pd.Series] = {}
    for profile, label in (
        ("player_injuries", "M1 base->player_injuries (fixed injury priors)"),
        ("player_injury_value", "M2 base->player_injury_value (value-weighted injuries)"),
        ("player_value", "M3 base->player_value (full player + value composite)"),
        ("player", "M8 base->player (frozen active profile, injuries included)"),
        ("player_injuries_continuity", "M9 base->player_injuries_continuity"),
        ("player_qb_injuries", "M10 base->player_qb_injuries"),
    ):
        candidate = _correct(player.loc[player["feature_profile"] == profile])
        contrasts[label] = candidate.sub(base, fill_value=np.nan).dropna()

    fixed = _profile_correct(AVAILABILITY_ABLATION, "availability_method", "fixed")
    learned = _profile_correct(AVAILABILITY_ABLATION, "availability_method", "learned")
    contrasts["M4 fixed->learned availability (ATS)"] = learned.sub(
        fixed, fill_value=np.nan
    ).dropna()

    participation = pd.read_parquet(PARTICIPATION_ABLATION)
    pv = _correct(participation.loc[participation["feature_profile"] == "player_value"])
    pp = _correct(participation.loc[participation["feature_profile"] == "player_participation"])
    contrasts["M6 player_value->participation RAPM"] = pp.sub(pv, fill_value=np.nan).dropna()

    if MOD07_AVAILABILITY.exists():
        mod07 = pd.read_parquet(MOD07_AVAILABILITY)
        contrasts["M7 MOD-07 availability half (opener)"] = pd.Series(
            (mod07["right_correct"] - mod07["left_correct"]).to_numpy(),
            index=mod07["game_id"].to_numpy(),
        )
    return contrasts


def game_weeks() -> pd.Series:
    """(season, week) label per game_id, taken from the artifacts themselves."""

    frame = pd.read_parquet(PLAYER_ABLATION, columns=["game_id", "season", "week"])
    frame = frame.drop_duplicates("game_id")
    return pd.Series(
        (frame["season"].astype(int) * 100 + frame["week"].astype(int)).to_numpy(),
        index=frame["game_id"].to_numpy(),
    )


def joint_sign_null(contrasts: dict[str, pd.Series], *, samples: int, seed: int) -> dict[str, Any]:
    """How often do ALL these contrasts land the same sign when the truth is zero?

    A sign test at p = 0.0625 assumes five independent coin flips. This measures
    the real thing instead: each contrast's per-game differences are CENTERED (so
    the true effect is exactly zero by construction), then whole weeks are
    resampled ONCE per draw and every contrast is recomputed on that same
    resample -- which is precisely the dependence the shared games create. The
    fraction of draws where all five land positive is the honest analogue of the
    0.5**5 the sign test assumes.
    """

    weeks = game_weeks()
    names = list(contrasts)
    centered: dict[str, pd.Series] = {}
    week_of: dict[str, np.ndarray] = {}
    for name in names:
        series = contrasts[name]
        centered[name] = series - series.mean()
        week_of[name] = weeks.reindex(series.index).to_numpy()

    all_weeks = np.unique(np.concatenate([week_of[name] for name in names]))
    rng = np.random.default_rng(seed)
    # Pre-bucket each contrast's values by week so a draw is a cheap gather.
    buckets: dict[str, dict[int, np.ndarray]] = {}
    for name in names:
        values = centered[name].to_numpy()
        buckets[name] = {int(w): values[week_of[name] == w] for w in np.unique(week_of[name])}

    all_positive = 0
    all_same_sign = 0
    for _ in range(samples):
        drawn = rng.choice(all_weeks, size=len(all_weeks), replace=True)
        means = []
        for name in names:
            parts = [buckets[name][int(w)] for w in drawn if int(w) in buckets[name]]
            pooled = np.concatenate(parts) if parts else np.zeros(0)
            means.append(float(pooled.mean()) if pooled.size else 0.0)
        signs = np.sign(means)
        if np.all(signs > 0):
            all_positive += 1
        if np.all(signs > 0) or np.all(signs < 0):
            all_same_sign += 1

    n = len(names)
    return {
        "measurements": names,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "independent_all_positive_probability": 0.5**n,
        "observed_all_positive_probability": all_positive / samples,
        "independent_two_sided_p": 2.0 * 0.5**n,
        "observed_two_sided_p": all_same_sign / samples,
        "inflation_factor": (all_same_sign / samples) / (2.0 * 0.5**n),
    }


def m_eff(correlation: np.ndarray) -> float:
    """Cheverud/Nyholt effective number of independent tests from a correlation matrix."""

    m = correlation.shape[0]
    if m <= 1:
        return float(m)
    eigenvalues = np.linalg.eigvalsh(correlation)
    variance = float(np.var(eigenvalues, ddof=1))
    return float(1.0 + (m - 1.0) * (1.0 - variance / m))


def sign_test_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial sign test at p=0.5, the statistic RWB-16 quotes."""

    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    contrasts = build_contrasts()
    report: dict[str, Any] = {"measurements": [], "pairwise_game_overlap": [], "correlation": {}}

    for name, series in contrasts.items():
        seasons = sorted({int(str(g)[:4]) for g in series.index})
        report["measurements"].append(
            {
                "name": name,
                "games": len(series),
                "seasons": [seasons[0], seasons[-1]],
                "season_count": len(seasons),
                "mean_delta_points": float(series.mean() * 100.0),
                "sign": "positive" if series.mean() > 0 else "negative",
            }
        )

    names = list(contrasts)
    for a, b in itertools.combinations(names, 2):
        shared = contrasts[a].index.intersection(contrasts[b].index)
        smaller = min(len(contrasts[a]), len(contrasts[b]))
        report["pairwise_game_overlap"].append(
            {
                "a": a,
                "b": b,
                "shared_games": len(shared),
                "share_of_smaller": float(len(shared) / smaller) if smaller else 0.0,
            }
        )

    # Correlation over the games EVERY contrast scored (the honest common ground).
    common = contrasts[names[0]].index
    for name in names[1:]:
        common = common.intersection(contrasts[name].index)
    matrix = pd.DataFrame({name: contrasts[name].reindex(common) for name in names})
    correlation = matrix.corr()
    report["correlation"] = {
        "common_games": len(common),
        "matrix": json.loads(correlation.round(4).to_json(orient="split")),
    }

    # The same correlation on the FULL 2,075-game close-graded set, which is where
    # five of the six contrasts actually live (the 453-game intersection above is
    # forced small only by the opener window).
    wide_names = [n for n in names if len(contrasts[n]) > 1000]
    wide_common = contrasts[wide_names[0]].index
    for name in wide_names[1:]:
        wide_common = wide_common.intersection(contrasts[name].index)
    wide = pd.DataFrame({name: contrasts[name].reindex(wide_common) for name in wide_names}).corr()
    report["correlation_full_close_set"] = {
        "measurements": wide_names,
        "common_games": len(wide_common),
        "matrix": json.loads(wide.round(4).to_json(orient="split")),
        "effective_independent_measurements": round(m_eff(wide.to_numpy()), 3),
    }

    # The family boundary is NOT recorded anywhere. RWB-16 says "five
    # measurements, all positive, p = 0.0625" without naming them, so the
    # sensitivity of that p-value to where the boundary is drawn is itself the
    # finding. Three defensible boundaries, scored the same way.
    narrow = [n for n in names if n.split()[0] in {"M1", "M2", "M3", "M4", "M7"}]
    same_kind = narrow + [n for n in names if n.split()[0] == "M6"]
    boundaries = {
        "narrow_reconstructed_five": narrow,
        "narrow_plus_same_kind_negative": same_kind,
        "broad_every_injury_or_value_contrast": names,
    }
    report["family_boundary_sensitivity"] = []
    for label, members in boundaries.items():
        positives = [n for n in members if contrasts[n].mean() > 0]
        entry: dict[str, Any] = {
            "boundary": label,
            "members": members,
            "positive": len(positives),
            "total": len(members),
            "nominal_two_sided_sign_test_p": sign_test_two_sided(len(positives), len(members)),
        }
        if len(positives) == len(members):
            entry["dependence_adjusted"] = joint_sign_null(
                {n: contrasts[n] for n in members}, samples=4000, seed=20260818
            )
        report["family_boundary_sensitivity"].append(entry)

    positive_names = narrow
    positive_correlation = correlation.loc[positive_names, positive_names].to_numpy()
    report["sign_test"] = {
        "reconstructed_five": positive_names,
        "nominal_two_sided_p": sign_test_two_sided(len(narrow), len(narrow)),
        "effective_independent_measurements_453_game_basis": round(m_eff(positive_correlation), 3),
    }
    report["joint_block_bootstrap_null"] = joint_sign_null(
        {name: contrasts[name] for name in positive_names}, samples=4000, seed=20260818
    )
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
