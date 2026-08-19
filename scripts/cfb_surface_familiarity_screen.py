"""Cross-league replication (CFB) of the NFL surface-switch / surface-familiarity
leads measured today on 2009-2025 REG NFL data:

- ``weather_battery_surface_switch_grass_to_turf`` (registry/weak_signals.json):
  away team's modal home surface is grass AND this game's surface is artificial
  turf -> home side covers 52.25% vs 47.74% complement, +1.16 full-slate pts,
  week-blocked 95% [+0.29, +2.04], P+ 0.995, n=1,112.
- ``surface_familiarity_r1_turf_venue_visitor_split`` (venue-controlled
  follow-up, within turf venues only): grass-modal visitors cover .5225 vs
  turf-modal visitors' .4886, +1.46 full-slate pts, P+ 0.933. NOT corroborated
  by the grass-venue mirror (P+ 0.320).

This script asks whether the SAME construct leans the same way on college
football: C1 mirrors the original subset-vs-complement screen, C2 mirrors the
venue-controlled turf-venue visitor split (R1), C3 mirrors the grass-venue
mirror (R2). No NFL evaluation window is spent here -- CFB is this project's
sanctioned free cross-league replication ground.

**Surface truth, exactly where it came from (measure-once, documented).** No
local CFB snapshot carries venue surface: ``CFB_SCHEDULE_SNAPSHOT_COLUMNS`` in
``src/nfl_ats/cfb.py`` has ``venue_id``/``venue`` (a name) but no
surface/grass/dome field, confirmed by reading
``data/cfb/schedules/raw/*/season=2024`` directly. The CollegeFootballData API
does carry it: ``GET /venues`` returns one row per venue with a ``grass``
boolean (True=natural grass, False=artificial turf, None=unknown). This is
NOT one of XLG-02's six CFBD gap-fillers, so a one-off, read-only, uncached
(no manifest, not a tracked ingestion source) authenticated fetch was made
this session via ``<scratchpad>/agent_cfb_surface/fetch_venues.py`` (reusing
``nfl_ats.cfb._cfbd_get``/``_cfbd_records``/``cfbd_api_key`` verbatim -- the
same low-level plumbing XLG-02's other CFBD sources use, just without the
season-partitioned manifest machinery a tracked source needs). It fetched 852
venues in one call; the raw JSON is cached at
``<scratchpad>/agent_cfb_surface/venues_raw.json`` and the trimmed
id/name/grass/dome/city/state table at
``<scratchpad>/agent_cfb_surface/venues.parquet``. Joined against every FBS
home game in the clean-core window (``schedules.venue_id -> venues.id``,
CFBD's own id space, so no name join), only 21 of 10,632 rows (0.2%) fail to
resolve a known grass/turf value -- surface truth is effectively complete for
this population. The API key is read from the ``CFBD_API_KEY`` environment
variable only and is never printed, logged, or written to any output file.

**Population.** The XLG-03 canonical benchmark table
(``data/processed/cfb_game_features.parquet``, built by
``nfl-ats cfb-build-features``): completed regular-season FBS-vs-FBS games
with an orientable spread, ATS semantics identical to the NFL convention
(``ats_margin = result - spread_line``, ``home_cover`` 1/0/NaN-on-push).
Restricted to the clean core (``nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS``
= 2012-2019 + 2021-2025, reused verbatim, not redeclared), pushes/missing
covers dropped, neutral-site games excluded (no meaningful "home venue" arm).
This game's own venue surface and the away team's modal home surface are
joined on afterward from the schedule snapshot + the fetched venues table;
neither join can leak postgame information (venue identity and grass/turf are
stadium facts fixed long before kickoff).

**Modal home surface**, same derivation logic as the NFL script: for each
(team, season), the mode of that team's OWN home games' surface (grass/turf),
computed from the fullest available same-season home-game sample -- every FBS
home game in the schedule (any opponent division), regular season, excluding
neutral-site "home" games -- not just the clean-core scored subset, so a
thinner scored population never starves the mode of data it doesn't need to.

**Method**, reused verbatim from today's NFL scripts
(``scripts/nfl_weather_battery_screen.py``, ``scripts/surface_familiarity_
screen.py``): a joint week-blocked bootstrap (season-blocked secondary),
full-slate effect scaling (raw gap x fraction-of-slate), 20,000 draws, fixed
seed, ``probability_positive`` never reported as a zero/nonzero binary. C1 is
a subset-vs-complement screen (``block_bootstrap_two_group``); C2/C3 are
two-arm splits within a venue-fixed subset (``block_bootstrap_two_arm``) --
both are the SAME algorithm (joint multinomial block resample, one arm's mean
minus the other's) applied to a different pair of groups, written once here
and shared.

**Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim).** An
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is unresolved_below_power: record it with ``nfl-ats weak-signals record``,
report probability_positive, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator.

Writes the predeclaration to
``<scratchpad>/agent_cfb_surface/predeclaration.json`` BEFORE any cover rate
is computed, and results to
``<scratchpad>/agent_cfb_surface/cfb_results.json``. Never writes to
``registry/`` -- recording happens via separate
``nfl-ats weak-signals record`` invocations under the repository's registry
write-lock protocol.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\26042060-ffd8-45a7-b2e7-a9b30b87bd34\scratchpad\agent_cfb_surface"
)
PREDECLARATION_PATH = OUT_DIR / "predeclaration.json"
VENUES_PATH = OUT_DIR / "venues.parquet"
RESULTS_PATH = OUT_DIR / "cfb_results.json"

FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"
SCHEDULES_ROOT = REPO / "data" / "cfb" / "schedules" / "raw"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
ERA_1 = (2012, 2019)
ERA_2 = (2021, 2025)

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "home_division",
    "away_division",
    "home_team",
    "away_team",
    "neutral_site",
    "venue_id",
]


def _latest_cfb_schedule_snapshot() -> Path:
    candidates = sorted(p for p in SCHEDULES_ROOT.glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no CFB schedule snapshot found under {SCHEDULES_ROOT}")
    return candidates[-1]


def load_cfb_schedules_all(snapshot_dir: Path) -> pd.DataFrame:
    """Concatenate every season partition's schedule rows, minimal columns only."""

    paths = sorted(snapshot_dir.glob("season=*/schedules.parquet"))
    if not paths:
        raise FileNotFoundError(f"no season=*/schedules.parquet files under {snapshot_dir}")
    frames = [pd.read_parquet(path, columns=SCHEDULE_COLUMNS) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_id"] = pd.to_numeric(df["game_id"], errors="raise").astype("int64")
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["venue_id"] = pd.to_numeric(df["venue_id"], errors="coerce")
    df["season_type"] = df["season_type"].astype(str).str.lower()
    df["home_division"] = df["home_division"].astype(str).str.lower()
    df["neutral_site"] = df["neutral_site"].astype(bool)
    return df


def load_venues() -> pd.DataFrame:
    if not VENUES_PATH.exists():
        raise FileNotFoundError(
            f"{VENUES_PATH} is missing; run fetch_venues.py first (CFBD /venues endpoint "
            "is the only surface-truth source -- no local CFB snapshot carries grass/turf)"
        )
    venues = pd.read_parquet(VENUES_PATH)
    venues["id"] = pd.to_numeric(venues["id"], errors="raise").astype("int64")
    return venues


def _surface_from_grass(value: object) -> str | None:
    if value is True:
        return "grass"
    if value is False:
        return "turf"
    return None


def compute_modal_home_surface(
    schedules: pd.DataFrame, venues: pd.DataFrame, seasons: tuple[int, ...]
) -> pd.Series:
    """Per (home_team, season) mode of that team's own home-game surface.

    Uses every FBS home game in the schedule (any away-team division), regular
    season, excluding neutral-site "home" games -- the fullest same-season
    sample of "what surface does this team's stadium actually have", mirroring
    the NFL script's use of the full REG population rather than the (smaller)
    scored subset for this stadium-fact derivation.
    """

    home = schedules.loc[
        schedules["season"].isin(seasons)
        & schedules["season_type"].eq("regular")
        & schedules["home_division"].eq("fbs")
        & ~schedules["neutral_site"]
    ].copy()
    home = home.merge(venues.loc[:, ["id", "grass"]], left_on="venue_id", right_on="id", how="left")
    home["surface_norm"] = home["grass"].map(_surface_from_grass)

    def _mode(values: pd.Series) -> str | None:
        modes = values.mode(dropna=True)
        return str(modes.iat[0]) if not modes.empty else None

    modal = home.groupby(["home_team", "season"])["surface_norm"].agg(_mode)
    return modal.rename("away_modal_surface")


def load_population() -> pd.DataFrame:
    """Build the scored CFB game-level table with surface columns attached."""

    features = pd.read_parquet(FEATURES_PATH)
    features["season"] = pd.to_numeric(features["season"], errors="raise").astype(int)
    features["week"] = pd.to_numeric(features["week"], errors="raise").astype(int)
    features["game_id"] = pd.to_numeric(features["game_id"], errors="raise").astype("int64")

    cc = features.loc[features["season"].isin(CFB_CLEAN_CORE_SEASONS)].copy()
    n_clean_core_all = len(cc)

    cc = cc.loc[cc["home_cover"].notna()].copy()
    n_pushes_dropped = n_clean_core_all - len(cc)

    neutral_mask = pd.to_numeric(cc["neutral_site"], errors="coerce").fillna(0).astype(int).eq(1)
    n_neutral_excluded = int(neutral_mask.sum())
    cc = cc.loc[~neutral_mask].reset_index(drop=True)
    n_total = len(cc)

    snapshot_dir = _latest_cfb_schedule_snapshot()
    schedules = load_cfb_schedules_all(snapshot_dir)
    venues = load_venues()

    cc = cc.merge(schedules.loc[:, ["game_id", "venue_id"]], on="game_id", how="left")
    cc = cc.merge(venues.loc[:, ["id", "grass"]], left_on="venue_id", right_on="id", how="left")
    cc["surface_norm"] = cc["grass"].map(_surface_from_grass)
    cc = cc.drop(columns=["id", "grass"])

    modal = compute_modal_home_surface(schedules, venues, CFB_CLEAN_CORE_SEASONS)
    cc = cc.merge(modal, left_on=["away_team", "season"], right_index=True, how="left")

    cc["week_block"] = cc["season"] * 100 + cc["week"]

    cc.attrs["n_clean_core_all"] = n_clean_core_all
    cc.attrs["n_pushes_dropped"] = n_pushes_dropped
    cc.attrs["n_neutral_excluded"] = n_neutral_excluded
    cc.attrs["n_total"] = n_total
    cc.attrs["n_missing_surface_norm"] = int(cc["surface_norm"].isna().sum())
    cc.attrs["n_missing_away_modal_surface"] = int(cc["away_modal_surface"].isna().sum())
    cc.attrs["schedule_snapshot"] = str(snapshot_dir)
    return cc


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Joint block bootstrap of ``100*(mean(value|flag=True)-mean(value|flag=False))``.

    One algorithm serves both C1 (subset vs. its full-population complement)
    and C2/C3 (mismatched vs. matched visitor within a venue-fixed pair): a
    single multinomial draw over the shared block ids feeds both groups' means
    each draw, so a resample never mixes blocks across groups. Reused verbatim
    (mechanically identical) from ``nfl_weather_battery_screen.py::
    block_bootstrap_two_group`` / ``surface_familiarity_screen.py::
    block_bootstrap_two_arm``.
    """

    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag = df[flag_col].to_numpy(dtype=bool)

    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        mask = flag == group
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


def _interval(draws: np.ndarray, fraction_of_slate: float) -> dict[str, Any]:
    scaled = draws * fraction_of_slate
    lower, upper = np.quantile(scaled, [0.025, 0.975]) if len(scaled) else (np.nan, np.nan)
    return {
        "estimate": float(np.mean(scaled)) if len(scaled) else float("nan"),
        "ci95": [float(lower), float(upper)],
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else float("nan"),
        "samples": len(scaled),
    }


def score_subset(
    df: pd.DataFrame,
    *,
    flag: pd.Series,
    n_total_for_fraction: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Subset-vs-complement scorer for C1 (mirrors nfl_weather_battery_screen.summarize)."""

    work = df.copy()
    work["_flag"] = flag.to_numpy()
    n_flag = int(work["_flag"].sum())
    n_complement = len(work) - n_flag
    if n_flag == 0 or n_complement == 0:
        return {"n_flag": n_flag, "n_complement": n_complement, "insufficient_data": True}

    subset_cover = float(work.loc[work["_flag"], "home_cover"].mean())
    complement_cover = float(work.loc[~work["_flag"], "home_cover"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total_for_fraction
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    week_draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="home_cover",
        block_col="week_block",
        samples=samples,
        seed=seed,
    )
    season_draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="home_cover",
        block_col="season",
        samples=samples,
        seed=seed,
    )
    return {
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_total_for_fraction": n_total_for_fraction,
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "n_week_blocks": int(work["week_block"].nunique()),
        "n_seasons": int(work["season"].nunique()),
        "week_blocked": _interval(week_draws, fraction_of_slate),
        "season_blocked_secondary": _interval(season_draws, fraction_of_slate),
        "insufficient_data": False,
    }


def build_pair(df: pd.DataFrame, *, venue_surface: str, mismatch_surface: str) -> pd.DataFrame:
    """Two-arm pair rows for a venue-controlled visitor-surface split (C2/C3).

    ``venue_surface`` fixes ``surface_norm`` (the game's actual venue).
    ``arm_mismatched`` is True for the visitor whose own modal home surface
    equals the surface the venue is NOT (the mismatched visitor). Rows whose
    away_modal_surface is neither grass nor turf are dropped from the pair.
    """

    subset = df.loc[df["surface_norm"] == venue_surface].copy()
    known = subset.loc[subset["away_modal_surface"].isin(["grass", "turf"])].copy()
    known["arm_mismatched"] = known["away_modal_surface"] == mismatch_surface
    return known


def score_pair(
    pair_df: pd.DataFrame, *, n_total_for_fraction: int, samples: int, seed: int
) -> dict[str, Any]:
    n_a = int(pair_df["arm_mismatched"].sum())
    n_b = int((~pair_df["arm_mismatched"]).sum())
    if n_a == 0 or n_b == 0:
        return {"n_mismatched": n_a, "n_matched": n_b, "insufficient_data": True}

    cover_a = float(pair_df.loc[pair_df["arm_mismatched"], "home_cover"].mean())
    cover_b = float(pair_df.loc[~pair_df["arm_mismatched"], "home_cover"].mean())
    raw_gap_pts = (cover_a - cover_b) * 100.0
    fraction_of_slate = (n_a + n_b) / n_total_for_fraction
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    week_draws = block_bootstrap_two_group(
        pair_df,
        flag_col="arm_mismatched",
        value_col="home_cover",
        block_col="week_block",
        samples=samples,
        seed=seed,
    )
    season_draws = block_bootstrap_two_group(
        pair_df,
        flag_col="arm_mismatched",
        value_col="home_cover",
        block_col="season",
        samples=samples,
        seed=seed,
    )
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
        "week_blocked": _interval(week_draws, fraction_of_slate),
        "season_blocked_secondary": _interval(season_draws, fraction_of_slate),
        "insufficient_data": False,
    }


def write_predeclaration() -> None:
    """Freeze the exact design BEFORE any cover rate is computed."""

    payload = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "replication_of": [
            "weather_battery_surface_switch_grass_to_turf",
            "surface_familiarity_r1_turf_venue_visitor_split",
            "surface_familiarity_r2_grass_venue_mirror (per its registry entry)",
        ],
        "population": (
            "data/processed/cfb_game_features.parquet restricted to "
            "nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS (2012-2019, 2021-2025), "
            "home_cover not null (pushes/missing dropped), neutral_site==0 excluded"
        ),
        "surface_truth_source": (
            "CFBD GET /venues 'grass' boolean, fetched this session via "
            "fetch_venues.py; True=grass, False=turf, None=unknown; joined onto "
            "schedules.venue_id (CFBD's own id space, no name join)"
        ),
        "modal_home_surface": (
            "per (home_team, season) mode of surface_norm over that team's own "
            "FBS home games (any opponent division), regular season, "
            "neutral_site excluded, computed from the schedule snapshot directly "
            "(not the scored subset)"
        ),
        "reads": {
            "C1_cfb_surface_switch_grass_to_turf": (
                "subset vs. complement over the full scored population: "
                "away_modal_surface=='grass' AND surface_norm=='turf'"
            ),
            "C2_cfb_surface_familiarity_turf_venue_visitor_split": (
                "within surface_norm=='turf' games: away_modal_surface=='grass' "
                "(mismatched) vs. =='turf' (matched), two-arm"
            ),
            "C3_cfb_surface_familiarity_grass_venue_mirror": (
                "within surface_norm=='grass' games: away_modal_surface=='turf' "
                "(mismatched) vs. =='grass' (matched), two-arm (the mirror)"
            ),
        },
        "era_split_report_only": {"era_1": list(ERA_1), "era_2": list(ERA_2)},
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "primary_block": "week_block (season*100+week)",
            "secondary_block": "season",
            "scaling": "raw_gap_pts * fraction_of_slate (fraction of the full scored population)",
        },
        "closing_grounds_taxonomy": (
            "An interval or CI that contains zero is NEVER grounds to reject, fail, "
            "or close an experiment. At this evaluator's ~2-point resolution, "
            "'contains zero' is the EXPECTED outcome for a real small signal. Only "
            "two grounds ever close a line of work: (1) refuted mechanism -- a "
            "RESOLVED wrong sign (whole interval on the wrong side of zero) or zero "
            "split-half reliability; (2) bounded by a positive control proven able "
            "to detect an effect that size. Everything else is "
            "unresolved_below_power: record it with `nfl-ats weak-signals record`, "
            "report probability_positive, never the binary 'contains zero'. The "
            "registry code hard-rejects inadmissible closures; if a record command "
            "errors, the verdict is wrong, not the validator."
        ),
        "default_classification": (
            "unresolved_below_power for all three reads unless a read's full "
            "week-blocked interval sits entirely below zero (opposite of the "
            "replicated sign), in which case refuted_mechanism / "
            "wrong_sign_resolved is attempted and, if the validator rejects it, "
            "unresolved_below_power is recorded instead"
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDECLARATION_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote predeclaration to {PREDECLARATION_PATH}")


def _print_subset(label: str, result: dict[str, Any]) -> None:
    if result.get("insufficient_data"):
        print(f"  {label}: insufficient data")
        return
    wb = result["week_blocked"]
    print(
        f"  {label}: n_flag={result['n_flag']} n_complement={result['n_complement']} "
        f"subset_cover={result['subset_cover']:.4f} "
        f"complement_cover={result['complement_cover']:.4f}"
    )
    print(
        f"    raw_gap={result['raw_gap_pts']:+.4f}pts "
        f"frac_of_slate={result['fraction_of_slate']:.4f} "
        f"full_slate_effect={result['full_slate_effect_pts']:+.4f}pts"
    )
    print(
        f"    week-blocked 95% [{wb['ci95'][0]:+.4f}, {wb['ci95'][1]:+.4f}] "
        f"P+={wb['probability_positive']:.4f} n_week_blocks={result['n_week_blocks']}"
    )


def _print_pair(label: str, result: dict[str, Any]) -> None:
    if result.get("insufficient_data"):
        print(f"  {label}: insufficient data")
        return
    wb = result["week_blocked"]
    print(
        f"  {label}: n_pair={result['n_pair']} (mismatched={result['n_mismatched']}, "
        f"matched={result['n_matched']})"
    )
    print(
        f"    cover: mismatched={result['cover_mismatched']:.4f} "
        f"matched={result['cover_matched']:.4f} "
        f"raw_gap={result['raw_gap_pts']:+.4f}pts frac_of_slate={result['fraction_of_slate']:.4f}"
    )
    print(
        f"    full_slate_effect={result['full_slate_effect_pts']:+.4f}pts week-blocked 95% "
        f"[{wb['ci95'][0]:+.4f}, {wb['ci95'][1]:+.4f}] P+={wb['probability_positive']:.4f} "
        f"n_week_blocks={result['n_week_blocks']}"
    )


def main() -> None:
    started = time.time()
    write_predeclaration()  # frozen BEFORE any cover rate below is computed

    print(f"\n=== loading CFB clean-core population (features: {FEATURES_PATH}) ===")
    df = load_population()
    n_total = df.attrs["n_total"]
    print(
        f"clean-core rows before exclusions: {df.attrs['n_clean_core_all']}; "
        f"pushes/missing dropped: {df.attrs['n_pushes_dropped']}; "
        f"neutral-site excluded: {df.attrs['n_neutral_excluded']}; "
        f"scored population n_total={n_total}"
    )
    print(
        f"surface_norm missing (unresolved venue): {df.attrs['n_missing_surface_norm']}; "
        f"away_modal_surface missing: {df.attrs['n_missing_away_modal_surface']}"
    )

    results: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_clean_core_all": df.attrs["n_clean_core_all"],
        "n_pushes_dropped": df.attrs["n_pushes_dropped"],
        "n_neutral_excluded": df.attrs["n_neutral_excluded"],
        "n_total_population": n_total,
        "n_missing_surface_norm": df.attrs["n_missing_surface_norm"],
        "n_missing_away_modal_surface": df.attrs["n_missing_away_modal_surface"],
        "schedule_snapshot": df.attrs["schedule_snapshot"],
        "predeclaration": str(PREDECLARATION_PATH),
    }

    # ---- descriptives ----
    known_surface = df.loc[df["surface_norm"].notna()]
    turf_games = int((known_surface["surface_norm"] == "turf").sum())
    grass_games = int((known_surface["surface_norm"] == "grass").sum())
    turf_fraction_known = turf_games / len(known_surface) if len(known_surface) else float("nan")
    print(
        f"\n=== descriptives: game-surface split (known-surface games only) ===\n"
        f"  turf={turf_games} grass={grass_games} unknown={df.attrs['n_missing_surface_norm']} "
        f"turf_fraction_of_known={turf_fraction_known:.4f}"
    )
    results["descriptives"] = {
        "turf_games": turf_games,
        "grass_games": grass_games,
        "unknown_surface_games": df.attrs["n_missing_surface_norm"],
        "turf_fraction_of_known_surface": turf_fraction_known,
        "nfl_comparison_measured_today": (
            "NFL REG 2009-2025 weather-battery population: 1857 turf / 2418 grass "
            "/ 42 unknown of 4317 games = 43.4% turf of known-surface games "
            "(measured via nfl_weather_battery_screen.load_population this session)"
        ),
    }

    # ---- C1: subset vs. complement, full population ----
    print("\n=== C1 cfb_surface_switch_grass_to_turf ===")
    c1_flag = (df["away_modal_surface"] == "grass") & (df["surface_norm"] == "turf")
    c1 = score_subset(
        df,
        flag=c1_flag,
        n_total_for_fraction=n_total,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    results["C1_cfb_surface_switch_grass_to_turf"] = c1
    _print_subset("C1", c1)

    # ---- C2: turf-venue visitor split ----
    print("\n=== C2 cfb_surface_familiarity_turf_venue_visitor_split ===")
    c2_pair = build_pair(df, venue_surface="turf", mismatch_surface="grass")
    c2 = score_pair(
        c2_pair, n_total_for_fraction=n_total, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    results["C2_cfb_surface_familiarity_turf_venue_visitor_split"] = c2
    _print_pair("C2", c2)

    # ---- C3: grass-venue mirror ----
    print("\n=== C3 cfb_surface_familiarity_grass_venue_mirror ===")
    c3_pair = build_pair(df, venue_surface="grass", mismatch_surface="turf")
    c3 = score_pair(
        c3_pair, n_total_for_fraction=n_total, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    results["C3_cfb_surface_familiarity_grass_venue_mirror"] = c3
    _print_pair("C3", c3)

    # ---- era split of C1 (report-only, no extra registry entries) ----
    results["era_split_C1_report_only"] = {}
    for label, (start, end) in (("2012_2019", ERA_1), ("2021_2025", ERA_2)):
        print(f"\n=== era split (C1 only, report-only) {start}-{end} ===")
        era_df = df.loc[df["season"].between(start, end)].copy()
        era_flag = (era_df["away_modal_surface"] == "grass") & (era_df["surface_norm"] == "turf")
        era_result = score_subset(
            era_df,
            flag=era_flag,
            n_total_for_fraction=len(era_df),
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        era_result["era"] = [start, end]
        era_result["era_n_total"] = len(era_df)
        results["era_split_C1_report_only"][label] = era_result
        _print_subset(f"era {label}", era_result)

    results["elapsed_seconds"] = time.time() - started
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
