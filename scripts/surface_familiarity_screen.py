"""Follow-up mechanism screen sharpening ``weather_battery_surface_switch_grass_to_turf``
(``registry/weak_signals.json``; ``artifacts/nfl_weather_battery/20260819T155124Z/results.json``).

**Why this script exists.** That cell's flagged subset (away team's modal home
surface normalizes to grass AND this game's surface normalizes to turf) is
entirely turf-venue games: every flagged row shares one venue-side property
(the stadium is turf) and one visitor-side property (the visitor's own home
surface is grass). The original comparison (flag vs. everything else) cannot
tell those two apart -- the +1.16pt effect could be genuine visitor surface
-familiarity, or it could just be "turf venues run a bit hot for the home
side" with the grass-modal-visitor condition riding along for free, since the
complement group mixes in every grass-venue game too.

This script holds the venue fixed and varies only the visitor, in both
directions:

- **R1** (decisive): WITHIN turf-venue games, grass-modal visitors vs.
  turf-modal visitors. If the gap survives here, it is not pure venue --
  the venue is identical in both arms.
- **R2** (mirror): WITHIN grass-venue games, turf-modal visitors vs.
  grass-modal visitors. A true familiarity mechanism predicts a symmetric
  positive gap here too (mismatched visitor helps the home side either way).
- **R3**: R1 rerun separately on 2009-2017 and 2018-2025 (era stability).
- **R4**: R1's gap, re-intervaled with a franchise-blocked (not week-blocked)
  bootstrap, plus an additive per-franchise contribution decomposition
  (robustness only, not written to the registry).
- **R5**: R1's pair rows split by roof (dome/closed vs. outdoor-turf)
  (report only, not written to the registry).

Every design choice (arm definitions, exclusions, the fraction-of-slate
denominator for each read, seed) was frozen BEFORE any read was scored in
``<scratchpad>/agent_surface/predeclaration.json``. This script implements
exactly what that document specifies.

**Machinery reused, not reimplemented**: ``load_population`` (surface
normalization, modal-home-surface derivation, cover computation via
``nfl_ats.features.add_ats_outcomes``, week_block ids) is imported verbatim
from ``scripts/nfl_weather_battery_screen.py`` so this follow-up scores the
IDENTICAL population the lead was measured on. The two-arm joint block
bootstrap generalizes ``penalty_discipline_interval.py::
block_bootstrap_quartile_gap`` (itself the two-group generalization of
``nfl_weather_battery_screen.py::block_bootstrap_two_group``) to an
arbitrary block column, so R4's franchise-blocked resample is the same
algorithm as R1's week-blocked one with one argument changed.

**Mined-family discipline (binding, AGENTS.md).** This is a follow-up read on
a lead surfaced by mining (the 8-cell weather/environment battery already
spent multiplicity). Every registry entry this script's numbers feed records
``unresolved_below_power`` regardless of interval shape -- an interval
containing zero is never grounds to reject, and a resolved-shaped interval on
a mined-family follow-up is not auto-promoted either, per the precedent this
very cell's own registry entry already sets.

Writes JSON to ``<scratchpad>/agent_surface/results.json`` and prints a
summary to stdout. Never writes to ``registry/`` -- recording happens via
separate ``nfl-ats weak-signals record`` invocations using this script's
printed numbers, under the repository's registry write-lock protocol.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import latest_schedules  # noqa: E402

_BATTERY_PATH = REPO / "scripts" / "nfl_weather_battery_screen.py"
_spec = importlib.util.spec_from_file_location("nfl_weather_battery_screen", _BATTERY_PATH)
assert _spec is not None and _spec.loader is not None
_battery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_battery)

load_population = _battery.load_population
OUTDOOR_ROOFS = _battery.OUTDOOR_ROOFS
DOME_CLOSED_ROOFS = _battery.DOME_CLOSED_ROOFS

OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\26042060-ffd8-45a7-b2e7-a9b30b87bd34\scratchpad\agent_surface"
)
PREDECLARATION_PATH = OUT_DIR / "predeclaration.json"

READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/ -- its only two
# write sites are under OUT_DIR, a scratchpad temp directory (see module
# docstring), never artifacts/ or registry/; the module mentions an
# artifacts/... path only as a citation of another script's prior output.
READ_ONLY_EXCEPTIONS: dict[int, str] = {
    267: "OUT_DIR is the scratchpad temp directory defined above, not artifacts/",
    432: "output_path == OUT_DIR / 'results.json', the scratchpad temp directory",
}

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
ERA_1 = (2009, 2017)
ERA_2 = (2018, 2025)


def build_pair(df: pd.DataFrame, *, venue_surface: str, mismatch_surface: str) -> pd.DataFrame:
    """Two-arm pair rows for a venue-controlled visitor-surface split.

    ``venue_surface`` fixes ``surface_norm`` (the game's actual venue).
    ``arm`` is True for the MISMATCHED visitor (away_modal_surface equal to
    the surface the venue is NOT), False for the MATCHED visitor
    (away_modal_surface equal to the venue surface). Rows whose
    away_modal_surface is neither grass nor turf are dropped from the pair
    (predeclared, mirrors the quartile-design precedent's Q2/Q3 exclusion).
    """

    subset = df.loc[df["surface_norm"] == venue_surface].copy()
    known = subset.loc[subset["away_modal_surface"].isin(["grass", "turf"])].copy()
    known["arm_mismatched"] = known["away_modal_surface"] == mismatch_surface
    return known


def block_bootstrap_two_arm(
    pair_df: pd.DataFrame,
    *,
    arm_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Joint block bootstrap of ``100*(mean(value|arm=True) - mean(value|arm=False))``.

    Direct generalization of ``penalty_discipline_interval.py::
    block_bootstrap_quartile_gap`` to an arbitrary two-valued arm column and
    an arbitrary block column (week_block, season, or home_team here): one
    multinomial draw over the shared block ids feeds both arms' means each
    draw, so a single resample never mixes blocks across arms.
    """

    blocks, block_index = np.unique(pair_df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = pair_df[value_col].to_numpy(dtype=np.float64)
    arm = pair_df[arm_col].to_numpy(dtype=bool)

    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        mask = arm == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    a_count = drawn @ counts[True]
    b_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = (drawn @ sums[True]) / a_count
        mean_b = (drawn @ sums[False]) / b_count
    gap = (mean_a - mean_b) * 100.0
    valid = (a_count > 0) & (b_count > 0)
    return gap[valid]


def score_pair(
    pair_df: pd.DataFrame,
    *,
    n_total_for_fraction: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    n_a = int(pair_df["arm_mismatched"].sum())
    n_b = int((~pair_df["arm_mismatched"]).sum())
    cover_a = float(pair_df.loc[pair_df["arm_mismatched"], "home_cover"].mean())
    cover_b = float(pair_df.loc[~pair_df["arm_mismatched"], "home_cover"].mean())
    raw_gap_pts = (cover_a - cover_b) * 100.0
    fraction_of_slate = (n_a + n_b) / n_total_for_fraction
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    week_draws = block_bootstrap_two_arm(
        pair_df,
        arm_col="arm_mismatched",
        value_col="home_cover",
        block_col="week_block",
        samples=samples,
        seed=seed,
    )
    season_draws = block_bootstrap_two_arm(
        pair_df,
        arm_col="arm_mismatched",
        value_col="home_cover",
        block_col="season",
        samples=samples,
        seed=seed,
    )

    def _interval(draws: np.ndarray) -> dict[str, Any]:
        scaled = draws * fraction_of_slate
        lower, upper = np.quantile(scaled, [0.025, 0.975]) if len(scaled) else (np.nan, np.nan)
        return {
            "estimate": float(np.mean(scaled)) if len(scaled) else float("nan"),
            "ci95": [float(lower), float(upper)],
            "probability_positive": float(np.mean(draws > 0)) if len(draws) else float("nan"),
            "samples": len(scaled),
        }

    week_summary = _interval(week_draws)
    season_summary = _interval(season_draws)

    return {
        "n_mismatched": n_a,
        "n_matched": n_b,
        "n_pair": n_a + n_b,
        "n_total_for_fraction": n_total_for_fraction,
        "cover_mismatched": cover_a,
        "cover_matched": cover_b,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "n_week_blocks": int(pair_df["week_block"].nunique()),
        "n_seasons": int(pair_df["season"].nunique()),
        "week_blocked": week_summary,
        "season_blocked_secondary": season_summary,
    }


def franchise_contribution(pair_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Additive per-home-franchise decomposition of R1's raw gap.

    cover_arm = sum_f (n_arm_f/n_arm) * cover_arm_f, so
    raw_gap = sum_f [ (n_a_f/n_a)*cover_a_f - (n_b_f/n_b)*cover_b_f ].
    """

    n_a = int(pair_df["arm_mismatched"].sum())
    n_b = int((~pair_df["arm_mismatched"]).sum())
    rows = []
    for team, grp in pair_df.groupby("home_team"):
        a = grp.loc[grp["arm_mismatched"]]
        b = grp.loc[~grp["arm_mismatched"]]
        contrib = 0.0
        if len(a):
            contrib += (len(a) / n_a) * float(a["home_cover"].mean())
        if len(b):
            contrib -= (len(b) / n_b) * float(b["home_cover"].mean())
        rows.append(
            {
                "home_team": team,
                "n_mismatched": len(a),
                "n_matched": len(b),
                "contribution_pts": contrib * 100.0,
            }
        )
    rows.sort(key=lambda r: abs(r["contribution_pts"]), reverse=True)
    return rows


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedules_path = latest_schedules()
    print(f"=== loading {schedules_path} ===")
    df = load_population(schedules_path)
    n_total = len(df)
    print(
        f"REG {_battery.SEASON_START}-{_battery.SEASON_END} scored population: {n_total} "
        f"(pushes/missing dropped: {df.attrs['pushes_or_missing']})"
    )

    results: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_total_population": n_total,
        "predeclaration": str(PREDECLARATION_PATH),
    }

    # ---- R1: venue-controlled, within turf-venue games ----
    print("\n=== R1: turf-venue games, grass-modal vs turf-modal visitor ===")
    r1_pair = build_pair(df, venue_surface="turf", mismatch_surface="grass")
    r1 = score_pair(
        r1_pair, n_total_for_fraction=n_total, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    results["R1_turf_venue_visitor_split"] = r1
    wb = r1["week_blocked"]
    print(
        f"  n_pair={r1['n_pair']} (mismatched/grass-modal={r1['n_mismatched']}, "
        f"matched/turf-modal={r1['n_matched']}) of n_total={n_total}"
    )
    print(
        f"  cover: mismatched(grass-modal visitor)={r1['cover_mismatched']:.4f} "
        f"matched(turf-modal visitor)={r1['cover_matched']:.4f} "
        f"raw_gap={r1['raw_gap_pts']:+.4f}pts frac_of_slate={r1['fraction_of_slate']:.4f}"
    )
    print(
        f"  full_slate_effect={r1['full_slate_effect_pts']:+.4f}pts week-blocked 95% "
        f"[{wb['ci95'][0]:+.4f}, {wb['ci95'][1]:+.4f}] P+={wb['probability_positive']:.4f} "
        f"n_week_blocks={r1['n_week_blocks']}"
    )

    # ---- R2: mirror, within grass-venue games ----
    print("\n=== R2: grass-venue games, turf-modal vs grass-modal visitor (mirror) ===")
    r2_pair = build_pair(df, venue_surface="grass", mismatch_surface="turf")
    r2 = score_pair(
        r2_pair, n_total_for_fraction=n_total, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    results["R2_grass_venue_mirror"] = r2
    wb2 = r2["week_blocked"]
    print(
        f"  n_pair={r2['n_pair']} (mismatched/turf-modal={r2['n_mismatched']}, "
        f"matched/grass-modal={r2['n_matched']}) of n_total={n_total}"
    )
    print(
        f"  cover: mismatched(turf-modal visitor)={r2['cover_mismatched']:.4f} "
        f"matched(grass-modal visitor)={r2['cover_matched']:.4f} "
        f"raw_gap={r2['raw_gap_pts']:+.4f}pts frac_of_slate={r2['fraction_of_slate']:.4f}"
    )
    print(
        f"  full_slate_effect={r2['full_slate_effect_pts']:+.4f}pts week-blocked 95% "
        f"[{wb2['ci95'][0]:+.4f}, {wb2['ci95'][1]:+.4f}] P+={wb2['probability_positive']:.4f} "
        f"n_week_blocks={r2['n_week_blocks']}"
    )

    # ---- R3: era split of R1 ----
    results["R3_era_split"] = {}
    for label, (start, end) in (("2009_2017", ERA_1), ("2018_2025", ERA_2)):
        print(f"\n=== R3 era {start}-{end}: R1 rerun within era, own denominator ===")
        era_df = df.loc[df["season"].between(start, end)].copy()
        era_pair = build_pair(era_df, venue_surface="turf", mismatch_surface="grass")
        era_result = score_pair(
            era_pair,
            n_total_for_fraction=len(era_df),
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        era_result["era"] = [start, end]
        results["R3_era_split"][label] = era_result
        wbe = era_result["week_blocked"]
        print(
            f"  n_pair={era_result['n_pair']} of era n_total={len(era_df)} "
            f"raw_gap={era_result['raw_gap_pts']:+.4f}pts "
            f"full_slate_effect={era_result['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wbe['ci95'][0]:+.4f}, {wbe['ci95'][1]:+.4f}] "
            f"P+={wbe['probability_positive']:.4f}"
        )

    # ---- distinct turf-venue home franchises per era (report only) ----
    turf_franchises_by_era = {}
    for label, (start, end) in (("2009_2017", ERA_1), ("2018_2025", ERA_2)):
        era_df = df.loc[df["season"].between(start, end)]
        teams = sorted(era_df.loc[era_df["surface_norm"] == "turf", "home_team"].unique().tolist())
        turf_franchises_by_era[label] = teams
        print(f"\nturf-venue home franchises {label} (n={len(teams)}): {teams}")
    results["turf_venue_home_franchises_by_era"] = turf_franchises_by_era

    # ---- R4: franchise-cluster robustness of R1 (not recorded) ----
    print("\n=== R4: R1 re-intervaled with a home-franchise-blocked bootstrap ===")
    r4_draws = block_bootstrap_two_arm(
        r1_pair,
        arm_col="arm_mismatched",
        value_col="home_cover",
        block_col="home_team",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    r4_scaled = r4_draws * r1["fraction_of_slate"]
    r4_lower, r4_upper = (
        np.quantile(r4_scaled, [0.025, 0.975]) if len(r4_scaled) else (np.nan, np.nan)
    )
    n_franchise_blocks = int(r1_pair["home_team"].nunique())
    r4 = {
        "block_col": "home_team",
        "n_blocks": n_franchise_blocks,
        "estimate": float(np.mean(r4_scaled)) if len(r4_scaled) else float("nan"),
        "ci95": [float(r4_lower), float(r4_upper)],
        "probability_positive": float(np.mean(r4_draws > 0)) if len(r4_draws) else float("nan"),
        "samples": len(r4_scaled),
    }
    results["R4_franchise_cluster_robustness"] = r4
    print(
        f"  n_home_franchise_blocks={n_franchise_blocks} estimate={r4['estimate']:+.4f}pts "
        f"franchise-blocked 95% [{r4['ci95'][0]:+.4f}, {r4['ci95'][1]:+.4f}] "
        f"P+={r4['probability_positive']:.4f}  (vs week-blocked 95% "
        f"[{wb['ci95'][0]:+.4f}, {wb['ci95'][1]:+.4f}])"
    )
    contributions = franchise_contribution(r1_pair)
    results["R4_franchise_contributions_top5"] = contributions[:5]
    print("  top-5 |contribution| home franchises to R1's raw gap:")
    for row in contributions[:5]:
        print(
            f"    {row['home_team']}: {row['contribution_pts']:+.4f}pts "
            f"(n_mismatched={row['n_mismatched']}, n_matched={row['n_matched']})"
        )

    # ---- R5: dome interaction within R1 (report only) ----
    print("\n=== R5: R1 split by roof (dome/closed vs outdoor-turf), report only ===")
    r5: dict[str, Any] = {}
    for roof_label, roof_set in (("dome_closed", DOME_CLOSED_ROOFS), ("outdoor", OUTDOOR_ROOFS)):
        roof_pair = r1_pair.loc[r1_pair["roof"].isin(roof_set)].copy()
        n_a = int(roof_pair["arm_mismatched"].sum())
        n_b = int((~roof_pair["arm_mismatched"]).sum())
        if n_a == 0 or n_b == 0:
            r5[roof_label] = {"n_mismatched": n_a, "n_matched": n_b, "insufficient_data": True}
            print(f"  {roof_label}: insufficient data (n_mismatched={n_a}, n_matched={n_b})")
            continue
        roof_result = score_pair(
            roof_pair,
            n_total_for_fraction=n_total,
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        r5[roof_label] = roof_result
        wbr = roof_result["week_blocked"]
        print(
            f"  {roof_label}: n_pair={roof_result['n_pair']} "
            f"raw_gap={roof_result['raw_gap_pts']:+.4f}pts "
            f"full_slate_effect={roof_result['full_slate_effect_pts']:+.4f}pts "
            f"week-blocked 95% [{wbr['ci95'][0]:+.4f}, {wbr['ci95'][1]:+.4f}] "
            f"P+={wbr['probability_positive']:.4f}"
        )
    results["R5_dome_interaction_report_only"] = r5

    results["elapsed_seconds"] = time.time() - started
    output_path = OUT_DIR / "results.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
