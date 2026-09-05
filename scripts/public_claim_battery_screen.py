"""LEAD-57: Public-handicapper claim replication battery.

Re-runs ten of the twelve predeclared public-betting-folklore claims listed
in ``docs/public_claim_battery.md`` on this project's own 2009-2025 archive
(the other two -- dome-cold and Thursday-road -- duplicate cells already in
``registry/weak_signals.json`` and are cited there, not re-scored, per this
task's instruction). One predeclared boolean subset flag per claim, scored
against the closing nflverse spread with the same week-blocked joint block
bootstrap, full-slate scaling, and 2009-2017/2018-2025 era split as
``scripts/nfl_bias_battery_screen.py``.

**Measure-only.** This script never writes to ``registry/weak_signals.json``
and never edits a tracked doc. It writes an automatic experiment-provenance
stamp to ``registry/experiments/`` via ``write_experiment_artifact`` (a run
log, not a verdict) -- recording to the weak-signals registry happens via
separate ``nfl-ats weak-signals record`` calls against this script's output,
per AGENTS.md ("verdicts flow only through nfl-ats weak-signals record,
never through prose").

**Lead-generation screen, not a rotation-registry confirmation.** Per
``docs/rotation_registry.md`` rule 8 ("CFB and non-reserved seasons stay
free... The registry governs NFL confirmation looks only"), this full-
population close-graded screen spends no rotation window; nothing here calls
``nfl_ats.rotation.assign_window``/``record_look``.

The full predeclaration (hypothesis, exact subset definition, mechanism,
predicted direction) is frozen in ``docs/public_claim_battery.md``, written
before this script scored anything. This module implements exactly what
that document specifies.

**Reuses ``scripts/nfl_bias_battery_screen.py`` by import** (data loading,
long-table construction, ``block_bootstrap_two_group``,
``summarize_population``, ``score_hypothesis``, ``ERA_SPLITS``) -- not
copy-pasted, not edited -- to guarantee bit-identical team-game long-table
values (``team_covered``, ``team_spread``, ``team_is_favorite``, ``div_game``,
``weekday``, ``gametime_hour``, ``own_rest``, ``prior_win_pct``,
``opp_prior_win_pct``, ``prior_score_margin``) to the parent battery. Only
two genuinely new pregame-safe columns are added on top here:
``prior_team_spread`` (claim 7) and ``ats_streak_len`` (claim 12).

Data: ``data/processed/game_features.parquet`` inner-joined on ``game_id``
with the newest ``data/raw/*/schedules.parquet`` snapshot -- identical join
to the parent battery. REG season only, 2009-2025. Grading line is the
CLOSING nflverse spread (disclosed, per the parent battery's own
convention): this is a screen, not a play/no-play decision, which per
AGENTS.md must be graded at the opener.

Writes JSON to ``artifacts/public_claim_battery/<UTC timestamp>/results.json``
via ``write_experiment_artifact`` and prints a plain-text summary table plus
the ``nfl-ats weak-signals record`` commands for every cell.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260905


def _load_parent_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "nfl_bias_battery_screen_for_public_claims",
        REPO / "scripts" / "nfl_bias_battery_screen.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_parent_module()


def _ats_losing_streak(group: pd.DataFrame) -> pd.Series:
    """Entering-this-game count of consecutive ``team_covered == 0`` results,
    strictly prior, reset at the season boundary (``group`` is one
    ``(team, season)`` slice sorted by ``gameday``) and on any
    ``team_covered == 1``. Pushes are already dropped from the long table
    before this runs, so ``team_covered`` here is always exactly 0.0 or 1.0
    -- a push neither extends nor breaks the streak by construction.
    """

    streak = 0.0
    out: list[float] = []
    for covered in group["team_covered"]:
        out.append(streak)
        if covered == 1.0:
            streak = 0.0
        elif covered == 0.0:
            streak += 1.0
    return pd.Series(out, index=group.index)


def add_claim_history_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add the two pregame-safe columns claims 7 and 12 need beyond what the
    parent's own ``add_history_features`` already computes. Both are
    strictly-prior, within-``(team, season)`` derived columns -- the current
    game's own outcome never enters either.
    """

    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)

    grouped = long_df.groupby(["team", "season"], sort=False)
    long_df["prior_team_spread"] = grouped["team_spread"].shift(1)

    long_df["ats_streak_len"] = 0.0
    for _, group in long_df.groupby(["team", "season"], sort=False):
        long_df.loc[group.index, "ats_streak_len"] = _ats_losing_streak(group).to_numpy()

    return long_df


def build_claims(long_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return {name: {"flag", "sign", "mechanism_class", "description",
    optional "eligible"}} for the 10 freshly-scored claims (2, 3, 4, 5, 6, 7,
    8, 9, 10, 12 in ``docs/public_claim_battery.md``'s numbering; claims 1
    and 11 duplicate already-recorded cells and are cited, not built here).
    Field shape matches ``nfl_bias_battery_screen.build_hypotheses`` exactly
    so ``base.score_hypothesis`` can be reused unmodified.
    """

    claims: dict[str, dict[str, Any]] = {}

    primetime_mask = long_df["weekday"].isin(["Thursday", "Monday"]) | (
        long_df["weekday"].eq("Sunday") & (long_df["gametime_hour"] >= 20)
    )
    claims["public_claim_primetime_dog"] = {
        "flag": primetime_mask & (long_df["team_spread"] < 0),
        "sign": 1,
        "mechanism_class": "market",
        "description": "Claim 2: Thu/Mon or Sunday-night underdog (Saturday excluded, "
        "stated limitation, matching the parent battery's primetime_favorite cell) -- "
        "BACK primetime dogs.",
    }

    claims["public_claim_post_bye_back"] = {
        "flag": long_df["own_rest"] >= 12,
        "sign": 1,
        "mechanism_class": "schedule",
        "description": "Claim 3: team (home or road) off a strict >=12-day rest gap -- "
        "BACK post-bye teams. Predeclared sign is the OPPOSITE of this project's own "
        "established bye-fade family (bye_overval_home_edge_post2011 etc.); see doc.",
        "eligible": long_df["own_rest"].notna(),
    }

    claims["public_claim_division_dog"] = {
        "flag": (long_df["div_game"] == 1) & (long_df["team_spread"] < 0),
        "sign": 1,
        "mechanism_class": "schedule",
        "description": "Claim 4: divisional game AND this team is the underdog -- BACK "
        "division dogs.",
    }

    claims["public_claim_road_fav_big_fade"] = {
        "flag": (~long_df["is_home"]) & (long_df["team_spread"] >= 7),
        "sign": -1,
        "mechanism_class": "market",
        "description": "Claim 5: road team favored by >=7 points -- FADE big road favorites.",
    }

    claims["public_claim_home_dog_3plus"] = {
        "flag": long_df["is_home"] & (long_df["team_spread"] <= -3),
        "sign": 1,
        "mechanism_class": "market",
        "description": "Claim 6: home team getting 3+ points -- BACK home dogs of 3+.",
    }

    claims["public_claim_upset_letdown_fade"] = {
        "flag": (long_df["prior_team_spread"] < 0) & (long_df["prior_score_margin"] > 0),
        "sign": -1,
        "mechanism_class": "onfield",
        "description": "Claim 7: immediately preceding game (this season) this team was "
        "the underdog by any margin and won straight-up -- FADE last week's upset "
        "winner.",
    }

    claims["public_claim_blowout_loss_bounce_21"] = {
        "flag": long_df["prior_score_margin"] <= -21,
        "sign": 1,
        "mechanism_class": "onfield",
        "description": "Claim 8: immediately preceding game (this season) was a loss by "
        ">=21 raw points -- BACK post-blowout losers.",
    }

    claims["public_claim_week1_dog"] = {
        "flag": (long_df["week"] == 1) & (long_df["team_spread"] < 0),
        "sign": 1,
        "mechanism_class": "market",
        "description": "Claim 9: Week 1 AND this team is the underdog -- BACK Week 1 dogs.",
    }

    reliable_record = (long_df["week"].isin([17, 18])) & (long_df["prior_games"] >= 13)
    reliable_opp_record = long_df["opp_prior_games"] >= 13
    claims["public_claim_eliminated_fade_wk17_18"] = {
        "flag": (long_df["prior_win_pct"] <= 0.400) & (long_df["opp_prior_win_pct"] >= 0.600),
        "sign": -1,
        "mechanism_class": "onfield",
        "description": "Claim 10: weeks 17-18, this team has a losing record (<=.400) "
        "AND the opponent is still >=.600 -- FADE the proxy-eliminated team. "
        "Elimination is proxied by record (no tiebreaker-aware standings "
        "reconstruction), per this task's stated fallback.",
        "eligible": reliable_record & reliable_opp_record,
    }

    claims["public_claim_ats_streak_regress"] = {
        "flag": long_df["ats_streak_len"] >= 3,
        "sign": 1,
        "mechanism_class": "market",
        "description": "Claim 12: entering this game on a 3+ game ATS losing streak "
        "(strictly prior, season-bounded) -- BACK teams on an ATS losing streak.",
    }

    return claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=base.DEFAULT_FEATURES)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.schedules is None:
        args.schedules = base.default_schedules()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "public_claim_battery" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.features} + {args.schedules} ===")
    merged = base.load_merged(args.features, args.schedules)
    print(f"REG games: {len(merged)}")

    long_df = base.build_long_table(merged)
    long_df = base.add_history_features(long_df)
    long_df = add_claim_history_features(long_df)
    print(f"team-game rows (pushes dropped): {len(long_df)}")

    claims = build_claims(long_df)
    assert len(claims) == 10, f"expected 10 freshly-scored claims, got {len(claims)}"

    results = []
    for name, spec in claims.items():
        print(f"\n=== {name} ({spec['mechanism_class']}) ===")
        cell = base.score_hypothesis(long_df, name, spec, samples=args.samples, seed=args.seed)
        results.append(cell)
        full = cell["full_period"]
        if full.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={full['n_flag']} n_total={full['n_total']} "
            f"subset_cover={full['subset_cover']:.4f} "
            f"complement_cover={full['complement_cover']:.4f}"
        )
        print(
            f"  raw_gap={full['raw_gap_pts']:+.3f}pts "
            f"frac_of_slate={full['fraction_of_slate']:.4f} "
            f"full_slate_effect={full['full_slate_effect_pts']:+.4f}pts"
        )
        print(
            f"  week-blocked 95% scaled [{full['week_blocked_ci95_scaled'][0]:+.4f}, "
            f"{full['week_blocked_ci95_scaled'][1]:+.4f}] P+={full['probability_positive']:.4f}"
        )
        for era_label, _, _ in base.ERA_SPLITS:
            era = cell["era_split"][era_label]
            if era.get("insufficient_data"):
                print(f"  [{era_label}] insufficient data")
                continue
            print(
                f"  [{era_label}] n_flag={era['n_flag']} cover={era['subset_cover']:.4f} "
                f"vs {era['complement_cover']:.4f} "
                f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
                f"P+={era['probability_positive']:.4f}"
            )

    print("\n=== ranked by |full-slate effect|, full period 2009-2025 ===")
    ranked = sorted(
        (r for r in results if not r["full_period"].get("insufficient_data")),
        key=lambda r: abs(r["full_period"]["full_slate_effect_pts"]),
        reverse=True,
    )
    for rank, cell in enumerate(ranked, start=1):
        full = cell["full_period"]
        print(
            f"{rank:>2}. {cell['name']:<32} {full['full_slate_effect_pts']:+.4f}pts "
            f"P+={full['probability_positive']:.4f} n_flag={full['n_flag']}"
        )

    print("\n=== proposed nfl-ats weak-signals record commands (not executed) ===")
    for cell in results:
        full = cell["full_period"]
        if full.get("insufficient_data"):
            continue
        print(
            f"nfl-ats weak-signals record --name {cell['name']} "
            f"--effect {full['full_slate_effect_pts']:.6f} --effect-units accuracy_points "
            f"--interval-low {full['week_blocked_ci95_scaled'][0]:.6f} "
            f"--interval-high {full['week_blocked_ci95_scaled'][1]:.6f} "
            f"--probability-positive {full['probability_positive']:.4f} "
            f"--classification unresolved_below_power --league nfl "
            f"--season-start 2009 --season-end 2025 --family public_claim_battery "
            f"--category {cell['mechanism_class']}"
        )

    configuration = {
        "command": "public-claim-battery-screen",
        "features": str(args.features),
        "schedules": str(args.schedules),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "n_claims": len(claims),
        "n_reg_games": len(merged),
        "n_team_game_rows": len(long_df),
        "predeclaration": "docs/public_claim_battery.md (frozen before scoring)",
        "results": results,
        "ranked_by_abs_full_slate_effect": [cell["name"] for cell in ranked],
        "provenance": artifact_provenance(configuration, args.schedules, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="public-claim-battery-screen",
        metrics=payload,
        source="docs/public_claim_battery.md",
        notes=(
            "LEAD-57 public-handicapper claim replication battery: 10 freshly-scored "
            "predeclared cells (claims 1 and 11 duplicate already-recorded registry "
            "entries and are cited there instead, per docs/public_claim_battery.md). "
            "Lead-generation screen, close-graded, no rotation window spent "
            "(docs/rotation_registry.md rule 8). Every cell records "
            "unresolved_below_power via a separate nfl-ats weak-signals record call "
            "regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
