"""Revives the backup-QB concept killed in `docs/prospective_evidence.md`'s
Tuesday-visibility audit: `home_qb_name`/`away_qb_name` in `schedules.parquet`
are populated only AFTER a game is played, so the schedule columns can never
supply a pregame SIGNAL -- but they still supply valid historical GROUND
TRUTH (an outcome fact), and the pool's picks are editable up to each game's
own deadline (owner correction, 2026-08-20), not just Tuesday noon --
**second owner correction, same day**: that per-game deadline is
`min(kickoff, that week's own Sunday 16:00 ET)`, not raw kickoff, since
Sunday-night and Monday-night games lock EARLY at Sunday 4pm ET rather than
at their own later kickoff. This script measures whether the two now-ingested
timestamped news corpora (`docs/injury_news_sourcing.md` ProFootballTalk,
`docs/pfr_transactions_sourcing.md` Pro Football Rumors transactions)
foreshadow a backup start BEFORE that deadline, and if so, by when (Tuesday
noon / Friday / Sunday 10:00 ET-or-deadline / later-but-still-pregame /
never).

Predeclaration: `docs/qb_news_channel.md` (matching rule, keyword list, and
time buckets frozen there BEFORE this script computed any cover rate).

Three phases:

1. Ground truth: `backup_start` team-games 2015-2025, reusing
   `scripts/nfl_bias_battery_screen.py`'s exact long-table construct (the
   same one `bias_battery_backup_qb_start`/`_opener` are registered against)
   -- not re-derived.
2. Visibility: PFT match is a pure local computation (299,739 rows, real
   per-article `lastmod`, no fetch needed). PFR match needs a live per-
   article JSON-LD fetch for day/hour precision beyond the free url_year/
   url_month bound (`Crawl-delay: 1`, budgeted).
3. Screen: IF pre-kickoff coverage is materially better than the already-
   measured 2.39%-by-Tuesday-noon baseline (`docs/injury_news_sourcing.md`
   sec 5.1), runs the ONE predeclared screen cell from
   `docs/qb_news_channel.md` sec 5 and records to `registry/weak_signals.json`.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/qb_backup_news_visibility.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from nfl_bias_battery_screen import (  # noqa: E402
    DEFAULT_FEATURES,
    add_history_features,
    build_long_table,
    default_schedules,
    load_merged,
)

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

PFT_INDEX = REPO / "data/raw/injury_news/20260819T191639Z/index.parquet"
PFR_INDEX = REPO / "data/raw/pfr_transactions/20260820T011126Z/index.parquet"

SEASONS = (2015, 2025)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819

# Task's ~15-minute politeness budget, mirroring ingest_transaction_news.py's
# own Crawl-delay: 1 (measured, docs/pfr_transactions_sourcing.md sec 1).
PFR_CRAWL_DELAY_SECONDS = 1.0
PFR_FETCH_WALLCLOCK_BUDGET_SECONDS = 12 * 60
PFR_MAX_FETCHES = 700
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"

# Frozen in docs/qb_news_channel.md sec 3 BEFORE any cover rate was computed.
# Matches against the raw, hyphenated slug text (same convention
# ingest_injury_news.py / ingest_transaction_news.py already use for their
# own keyword flags).
QB_NEWS_KEYWORDS: tuple[str, ...] = (
    "injured-reserve",
    "-on-ir",
    "-ir-",
    "questionable",
    "doubtful",
    "ruled-out",
    "will-miss",
    "out-for-the-season",
    "out-for-season",
    "concussion",
    "hamstring",
    "quadricep",
    "groin",
    "ankle",
    "knee-injury",
    "torn-acl",
    "torn-mcl",
    "shoulder-injury",
    "achilles",
    "surgery",
    "fracture",
    "fractured",
    "sprain",
    "sprained",
    "day-to-day",
    "week-to-week",
    "injury-report",
    "injury-designation",
    "placed-on-ir",
    "pup-list",
    "exits-",
    "exited-",
    "starting",
    "starts",
    "to-start",
    "named-starter",
    "gets-the-start",
    "in-for",
    "replace",
    "replaces",
    "replacing",
    "bench",
    "benched",
    "benching",
    "demoted",
    "new-starter",
    "backup",
    "takes-over",
    "relief",
    "subs-in",
    "activated",
    "activate-",
    "elevate",
    "elevated",
    "elevation",
    "practice-squad",
    "signs",
    "signed",
    "waived",
    "released",
    "claimed",
    "suspended",
    "suspension",
    "inactive",
    "inactives",
)
QB_NEWS_KEYWORD_RE = re.compile("|".join(re.escape(kw) for kw in QB_NEWS_KEYWORDS))

# docs/qb_news_channel.md sec 3 addendum: a match only counts within
# [deadline - LOOKBACK_DAYS, deadline), where deadline = min(kickoff, that
# week's Sunday 16:00 ET) (owner-corrected 2026-08-20). Without this bound
# a journeyman
# backup QB's own name recurs across his whole career against these generic
# keywords, producing a name-collision artifact, not foreshadowing.
LOOKBACK_DAYS = 10.0

_CURLY_APOSTROPHE = chr(0x2019)
NAME_PUNCTUATION = re.compile("[." + "'" + _CURLY_APOSTROPHE + "]")
NAME_NONALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: object) -> str:
    """Same normalization as scripts/injury_tuesday_cutoff_experiment.py."""

    text = str(name).lower()
    text = NAME_PUNCTUATION.sub("", text)
    text = NAME_NONALNUM.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Ground truth: reuse the bias battery's own construct, not re-derived.
# ---------------------------------------------------------------------------


def ground_truth_long_table() -> pd.DataFrame:
    merged = load_merged(DEFAULT_FEATURES, default_schedules())
    long_df = build_long_table(merged)
    long_df = add_history_features(long_df)
    return long_df


def kickoff_utc_by_game(schedules_path: Path) -> pd.DataFrame:
    """One row per game_id with kickoff (UTC), home_team, away_team, weekday.

    Same construction as nfl_ats.features._kickoff_utc: combine nflverse
    gameday + Eastern gametime, localize America/New_York, convert to UTC.
    """

    schedules = pd.read_parquet(schedules_path)
    date_text = pd.to_datetime(schedules["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = schedules["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    kickoff_utc = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    return pd.DataFrame(
        {
            "game_id": schedules["game_id"].astype(str),
            "gameday": pd.to_datetime(schedules["gameday"], errors="raise"),
            "weekday": schedules["weekday"].astype(str),
            "kickoff_utc": kickoff_utc,
        }
    )


def game_cutoffs(kickoffs: pd.DataFrame) -> pd.DataFrame:
    """T (Tuesday noon ET), F' (Friday 23:59:59 ET, clamped <= D),
    S' (Sunday 10:00 ET if Sunday else D, clamped <= D), D (deadline).

    **Owner-corrected 2026-08-20**: the pool's actual per-game pick deadline
    is `min(kickoff, that week's own Sunday 16:00 ET)`, not raw kickoff --
    SNF and MNF lock EARLY at Sunday 4pm ET, not at their own (later)
    kickoffs. `D` (deadline) replaces `K` (raw kickoff) as the boundary
    everywhere pregame-safety is enforced. This affects Monday games most
    (their true deadline is the PRECEDING day, not their own kickoff) and
    marginally affects Sunday-night/late-afternoon games (deadline a few
    minutes to ~4 hours before their own kickoff). `kickoff_utc` is still
    carried for reference/reporting but is no longer the enforcement
    boundary. docs/qb_news_channel.md sec 4 -- frozen before any cover rate
    was seen; this deadline correction was applied before any FINAL cover
    rate was recorded (see docs/qb_news_channel.md sec 4 addendum).
    """

    frame = kickoffs.copy()
    kickoff_et = frame["kickoff_utc"].dt.tz_convert("US/Eastern")
    days_since_tuesday = (kickoff_et.dt.weekday - 1) % 7
    tuesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_tuesday, unit="D")
    tuesday_noon_et = tuesday_date_et + pd.Timedelta(hours=12)
    friday_et = tuesday_date_et + pd.Timedelta(days=3, hours=23, minutes=59, seconds=59)
    # The week's own Sunday (Tuesday + 5 days), NOT the game's own calendar
    # date -- for a Monday game this is the day BEFORE kickoff, which is the
    # whole point of the owner's correction.
    sunday_date_et = tuesday_date_et + pd.Timedelta(days=5)
    sunday_10am_et = sunday_date_et + pd.Timedelta(hours=10)
    sunday_1600_et = sunday_date_et + pd.Timedelta(hours=16)

    frame["T_utc"] = tuesday_noon_et.dt.tz_convert("UTC")
    friday_utc = friday_et.dt.tz_convert("UTC")
    sunday_10am_utc = sunday_10am_et.dt.tz_convert("UTC")
    sunday_1600_utc = sunday_1600_et.dt.tz_convert("UTC")

    frame["deadline_utc"] = frame["kickoff_utc"].where(
        frame["kickoff_utc"] <= sunday_1600_utc, sunday_1600_utc
    )
    is_sunday = frame["weekday"].eq("Sunday")
    sunday_or_deadline_utc = sunday_10am_utc.where(is_sunday, frame["deadline_utc"])

    frame["F_utc"] = friday_utc.clip(upper=frame["deadline_utc"])
    frame["S_utc"] = sunday_or_deadline_utc.clip(upper=frame["deadline_utc"])
    return frame[["game_id", "kickoff_utc", "deadline_utc", "T_utc", "F_utc", "S_utc"]]


def assign_bucket(match_utc: pd.Timestamp, row: pd.Series) -> str:
    if pd.isna(match_utc):
        return "never"
    if match_utc >= row["deadline_utc"]:
        return "never"  # not pregame-safe (past the pool's own deadline); must not count
    if match_utc <= row["T_utc"]:
        return "by_tuesday_noon"
    if match_utc <= row["F_utc"]:
        return "by_friday"
    if match_utc <= row["S_utc"]:
        return "by_sunday_10am_or_kickoff"
    return "before_deadline"


# ---------------------------------------------------------------------------
# PFT matching (pure local computation, no fetch: lastmod is already a real
# per-article UTC timestamp, docs/injury_news_sourcing.md sec 3).
# ---------------------------------------------------------------------------


def build_last_name_index(headlines_norm: pd.Series) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for pos, text in enumerate(headlines_norm.to_numpy()):
        for token in set(str(text).split()):
            if len(token) >= 3:
                index.setdefault(token, []).append(pos)
    return index


def match_pft(backup_rows: pd.DataFrame, pft: pd.DataFrame) -> pd.Series:
    """Earliest qualifying PFT match time (UTC) per backup_rows row, or NaT.

    Restricted to ``[deadline - LOOKBACK_DAYS, deadline)`` per row, where
    ``deadline = min(kickoff, that week's Sunday 16:00 ET)`` (owner-corrected
    2026-08-20: SNF/MNF lock at Sunday 4pm ET, not their own later kickoff)
    -- see module-level ``LOOKBACK_DAYS`` docstring for why an unbounded
    search is a name-collision artifact, not a foreshadowing measurement.
    """

    candidates = pft.loc[
        pft["slug"].astype(str).str.lower().map(QB_NEWS_KEYWORD_RE.search).notna()
    ].copy()
    candidates["headline_norm"] = candidates["headline_guess"].map(_normalize_name)
    headlines = candidates["headline_norm"].to_numpy()
    # Naive UTC datetime64[ns] throughout, matching
    # injury_tuesday_cutoff_experiment.py's own convention (avoids tz-aware
    # object-array pitfalls while staying an exact UTC instant comparison).
    lastmods = (
        candidates["lastmod"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    last_name_index = build_last_name_index(candidates["headline_norm"])

    deadlines_naive = (
        backup_rows["deadline_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    lookback = np.timedelta64(int(LOOKBACK_DAYS * 86400), "s")

    results = np.full(len(backup_rows), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    names = backup_rows["own_qb_name_norm"].to_numpy(dtype=object)
    for row_pos, name in enumerate(names):
        if not name:
            continue
        deadline = deadlines_naive[row_pos]
        if np.isnat(deadline):
            continue
        window_start = deadline - lookback
        last_token = name.split()[-1]
        cand_idx = last_name_index.get(last_token)
        if not cand_idx:
            continue
        best = None
        for idx in cand_idx:
            if name not in headlines[idx]:
                continue
            lm = lastmods[idx]
            if np.isnat(lm) or lm < window_start or lm >= deadline:
                continue
            if best is None or lm < best:
                best = lm
        if best is not None:
            results[row_pos] = best
    return pd.Series(pd.to_datetime(results, utc=True), index=backup_rows.index)


# ---------------------------------------------------------------------------
# PFR candidate identification (local) + budgeted per-article fetch.
# ---------------------------------------------------------------------------


def pfr_candidates(backup_rows: pd.DataFrame, pfr: pd.DataFrame) -> pd.DataFrame:
    """(backup_row_index, url, url_year, url_month) for every name+keyword hit
    whose url_year/url_month falls within the [deadline - LOOKBACK_DAYS,
    deadline] month range, where ``deadline = min(kickoff, that week's
    Sunday 16:00 ET)`` (owner-corrected 2026-08-20). Month granularity is
    coarser than the 10-day lookback window, so every surviving candidate
    still needs a per-article fetch to confirm day/hour precision -- the
    month bound here only prunes candidates that CANNOT possibly be
    in-window (too early or after the deadline), the same free
    100%-verified month proxy (docs/pfr_transactions_sourcing.md sec 3)
    used elsewhere in this module.
    """

    keyword_hit = pfr["slug"].astype(str).str.lower().map(QB_NEWS_KEYWORD_RE.search).notna()
    candidates = pfr.loc[keyword_hit].copy()
    candidates["headline_norm"] = candidates["headline_from_slug"].map(_normalize_name)
    last_name_index = build_last_name_index(candidates["headline_norm"])
    headlines = candidates["headline_norm"].to_numpy()
    cand_url = candidates["url"].to_numpy(dtype=object)
    cand_year = pd.to_numeric(candidates["url_year"], errors="coerce").to_numpy()
    cand_month = pd.to_numeric(candidates["url_month"], errors="coerce").to_numpy()

    rows: list[dict[str, Any]] = []
    names = backup_rows["own_qb_name_norm"].to_numpy(dtype=object)
    deadlines = backup_rows["deadline_utc"].dt.tz_convert("UTC")
    earliest_allowed = deadlines - pd.Timedelta(days=LOOKBACK_DAYS)
    game_ym = (deadlines.dt.year * 12 + deadlines.dt.month).to_numpy()
    earliest_ym = (earliest_allowed.dt.year * 12 + earliest_allowed.dt.month).to_numpy()
    for row_pos, name in enumerate(names):
        if not name:
            continue
        last_token = name.split()[-1]
        cand_idx = last_name_index.get(last_token)
        if not cand_idx:
            continue
        lo_ym, hi_ym = earliest_ym[row_pos], game_ym[row_pos]
        for idx in cand_idx:
            if name not in headlines[idx]:
                continue
            cy, cm = cand_year[idx], cand_month[idx]
            if np.isnan(cy) or np.isnan(cm):
                continue
            cand_ym = int(cy) * 12 + int(cm)
            if cand_ym < lo_ym or cand_ym > hi_ym:
                continue
            rows.append(
                {
                    "backup_row_pos": row_pos,
                    "url": cand_url[idx],
                    "url_year": int(cy),
                    "url_month": int(cm),
                }
            )
    return pd.DataFrame(rows)


DATE_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')


@dataclass
class RateLimiter:
    delay_seconds: float
    _last: float | None = field(default=None, init=False)

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.delay_seconds - (time.monotonic() - self._last)
            if remaining > 0:
                time.sleep(remaining)
        self._last = time.monotonic()


def fetch_pfr_dates(
    urls: list[str], *, wallclock_budget_seconds: float, max_fetches: int
) -> dict[str, pd.Timestamp | None]:
    """Fetch JSON-LD datePublished for each url; budgeted, polite (Crawl-delay 1)."""

    limiter = RateLimiter(PFR_CRAWL_DELAY_SECONDS)
    started = time.monotonic()
    out: dict[str, pd.Timestamp | None] = {}
    for i, url in enumerate(urls):
        if i >= max_fetches or (time.monotonic() - started) >= wallclock_budget_seconds:
            print(
                f"  PFR fetch budget reached ({i}/{len(urls)} attempted); "
                "remaining candidates left unfetched (month-proxy-only, excluded from "
                "day/hour bucketing).",
                file=sys.stderr,
            )
            break
        limiter.wait()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8", errors="ignore")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            print(f"  fetch failed: {url}: {error}", file=sys.stderr)
            out[url] = None
            continue
        match = DATE_PUBLISHED_RE.search(raw)
        out[url] = pd.Timestamp(match.group(1)) if match else None
    return out


# ---------------------------------------------------------------------------
# Full-slate scaled week-blocked bootstrap (nfl_bias_battery_screen.py convention)
# ---------------------------------------------------------------------------


def block_bootstrap_two_group(
    df: pd.DataFrame, *, flag_col: str, value_col: str, block_col: str, samples: int, seed: int
) -> np.ndarray:
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
    subset_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_subset = (drawn @ sums[True]) / subset_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_subset - mean_complement) * 100.0
    valid = (subset_count > 0) & (complement_count > 0)
    return gap[valid]


def summarize_population(
    df: pd.DataFrame, *, flag: pd.Series, sign: int, samples: int, seed: int
) -> dict[str, Any]:
    n_total = len(df)
    n_flag = int(flag.sum())
    n_complement = n_total - n_flag
    work = df.copy()
    work["_flag"] = flag.to_numpy()
    subset_cover = float(work.loc[work["_flag"], "team_covered"].mean())
    complement_cover = float(work.loc[~work["_flag"], "team_covered"].mean())
    raw_gap_pts = sign * (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="team_covered",
        block_col="week_block",
        samples=samples,
        seed=seed,
    )
    signed_draws = sign * draws
    scaled_draws = signed_draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )
    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_week_blocks": int(work["week_block"].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "week_blocked_ci95_scaled": [float(lower), float(upper)],
        "standard_error": float(np.std(scaled_draws, ddof=1)) if len(scaled_draws) > 1 else np.nan,
        "probability_positive": float(np.mean(signed_draws > 0)) if len(signed_draws) else np.nan,
        "bootstrap_samples": samples,
        "sample_blocks": int(work["week_block"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfr-max-fetches", type=int, default=PFR_MAX_FETCHES)
    parser.add_argument(
        "--pfr-budget-seconds", type=float, default=PFR_FETCH_WALLCLOCK_BUDGET_SECONDS
    )
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/qb_news_channel")
    args = parser.parse_args()

    print("=== Phase 1: ground truth ===")
    long_df = ground_truth_long_table()
    scoped = long_df.loc[long_df["season"].between(*SEASONS)].reset_index(drop=True)
    eligible = scoped["backup_qb_flag"].notna()
    eligible_pop = scoped.loc[eligible].reset_index(drop=True)
    backup_flag = eligible_pop["backup_qb_flag"] == 1.0
    print(f"Eligible team-games {SEASONS[0]}-{SEASONS[1]}: {len(eligible_pop)}")
    print(f"Backup starts: {int(backup_flag.sum())}")
    by_season = eligible_pop.loc[backup_flag].groupby("season").size()
    print(by_season.to_string())

    kickoffs = kickoff_utc_by_game(default_schedules())
    cutoffs = game_cutoffs(kickoffs)
    eligible_pop = eligible_pop.merge(cutoffs, on="game_id", how="left", validate="many_to_one")
    missing_kickoff = eligible_pop["kickoff_utc"].isna().sum()
    if missing_kickoff:
        print(f"WARNING: {missing_kickoff} rows missing a kickoff timestamp", file=sys.stderr)

    backup_rows = eligible_pop.loc[backup_flag].reset_index(drop=True)
    backup_rows["own_qb_name_norm"] = backup_rows["own_qb_name"].map(_normalize_name)

    print("\n=== Phase 2a: PFT match (local, no fetch) ===")
    pft = pd.read_parquet(PFT_INDEX)
    pft_match = match_pft(backup_rows, pft)
    backup_rows["pft_match_utc"] = pft_match

    print("\n=== Phase 2b: PFR candidate identification (local, no fetch) ===")
    pfr = pd.read_parquet(PFR_INDEX)
    candidates = pfr_candidates(backup_rows, pfr)
    print(f"PFR name+keyword+month-window-bounded candidates: {len(candidates)}")

    print("\n=== Phase 2c: PFR per-article fetch (budgeted, Crawl-delay 1s) ===")
    fetch_urls = sorted(set(candidates["url"])) if len(candidates) else []
    print(f"Unique urls needing a fetch: {len(fetch_urls)}")
    fetched = fetch_pfr_dates(
        fetch_urls,
        wallclock_budget_seconds=args.pfr_budget_seconds,
        max_fetches=args.pfr_max_fetches,
    )
    n_fetch_ok = sum(1 for v in fetched.values() if v is not None)
    print(f"Fetched OK: {n_fetch_ok}/{len(fetched)}")

    # Earliest PFR match per backup row: only candidates with a successfully
    # fetched JSON-LD datePublished, strictly inside
    # [deadline - LOOKBACK_DAYS, deadline), count as a match, where
    # deadline = min(kickoff, that week's Sunday 16:00 ET) (owner-corrected
    # 2026-08-20). Month-only (unfetched) candidates are dropped -- no date
    # evidence precise enough to place them in a bucket or even confirm they
    # are in-window.
    pfr_match_utc = pd.Series(pd.NaT, index=backup_rows.index, dtype="datetime64[ns, UTC]")
    if len(candidates):
        lookback = pd.Timedelta(days=LOOKBACK_DAYS)
        for _, cand in candidates.iterrows():
            row_pos = int(cand["backup_row_pos"])
            deadline = backup_rows.loc[row_pos, "deadline_utc"]
            fetched_dt = fetched.get(cand["url"])
            if fetched_dt is None or pd.isna(deadline):
                continue
            candidate_time = (
                fetched_dt.tz_convert("UTC") if fetched_dt.tzinfo else fetched_dt.tz_localize("UTC")
            )
            if candidate_time < (deadline - lookback) or candidate_time >= deadline:
                continue
            current = pfr_match_utc.iloc[row_pos]
            if pd.isna(current) or candidate_time < current:
                pfr_match_utc.iloc[row_pos] = candidate_time
    backup_rows["pfr_match_utc"] = pfr_match_utc

    earliest = backup_rows[["pft_match_utc", "pfr_match_utc"]].min(axis=1)
    backup_rows["earliest_match_utc"] = earliest
    backup_rows["bucket"] = [
        assign_bucket(earliest.iloc[i], backup_rows.iloc[i]) for i in range(len(backup_rows))
    ]

    print("\n=== Phase 2 results: visibility buckets (backup starts only) ===")
    bucket_counts = backup_rows["bucket"].value_counts()
    print(bucket_counts.to_string())
    n_backup = len(backup_rows)
    bucket_order = [
        "by_tuesday_noon",
        "by_friday",
        "by_sunday_10am_or_kickoff",
        "before_deadline",
        "never",
    ]
    cumulative = {}
    running = 0
    for b in bucket_order:
        running += int(bucket_counts.get(b, 0))
        cumulative[b] = running
    print("\nCumulative coverage by bucket:")
    for b in bucket_order:
        print(f"  by {b}: {cumulative[b]}/{n_backup} = {cumulative[b] / n_backup:.4%}")

    pre_sunday_bucket_flag = backup_rows["bucket"].isin(
        ["by_tuesday_noon", "by_friday", "by_sunday_10am_or_kickoff"]
    )
    any_pregame_flag = backup_rows["bucket"] != "never"

    tuesday_rate = cumulative["by_tuesday_noon"] / n_backup
    sunday_rate = cumulative["by_sunday_10am_or_kickoff"] / n_backup
    any_pregame_rate = int(any_pregame_flag.sum()) / n_backup
    print(
        f"\nMaterially-better check: by-Tuesday-noon={tuesday_rate:.4%} "
        f"vs. by-Sunday10am/deadline={sunday_rate:.4%} "
        f"vs. any-pregame={any_pregame_rate:.4%} "
        "(docs/injury_news_sourcing.md sec 5.1 comparator: 2.39% by "
        "Tuesday noon, official designations)"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "seasons": list(SEASONS),
        "n_eligible_team_games": len(eligible_pop),
        "n_backup_starts": n_backup,
        "backup_starts_by_season": {int(k): int(v) for k, v in by_season.items()},
        "bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
        "cumulative_coverage": {
            k: {"n": cumulative[k], "rate": cumulative[k] / n_backup} for k in bucket_order
        },
        "pfr_candidates": {
            "total": len(candidates),
            "unique_urls_fetched_attempted": len(fetch_urls),
            "unique_urls_fetched_ok": n_fetch_ok,
        },
    }

    materially_better = (
        cumulative["by_sunday_10am_or_kickoff"] / n_backup
        > 3
        * (cumulative["by_tuesday_noon"] / n_backup if cumulative["by_tuesday_noon"] else 0.0001)
        or (cumulative["by_sunday_10am_or_kickoff"] / n_backup) >= 0.10
    )

    if materially_better:
        print("\n=== Phase 3: predeclared screen cell (docs/qb_news_channel.md sec 5) ===")
        screen_pop = eligible_pop.copy()
        screen_pop["_news_visible_flag"] = False
        visible_row_ids = (
            backup_rows.loc[pre_sunday_bucket_flag, "game_id"].astype(str)
            + "|"
            + backup_rows.loc[pre_sunday_bucket_flag].apply(lambda r: r["team"], axis=1)
        )
        screen_key = screen_pop["game_id"].astype(str) + "|" + screen_pop["team"].astype(str)
        screen_pop["_news_visible_flag"] = screen_key.isin(set(visible_row_ids)).to_numpy()

        summary = summarize_population(
            screen_pop,
            flag=screen_pop["_news_visible_flag"],
            sign=1,
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        print(json.dumps(summary, indent=2, default=float))
        results["screen_cell"] = summary
    else:
        print(
            "\nPhase 3 SKIPPED: pre-kickoff coverage is not materially better than the "
            "2.39%-by-Tuesday-noon comparator by this run's own threshold; see "
            "docs/qb_news_channel.md sec 5 for the predeclared trigger."
        )
        results["screen_cell"] = None

    configuration = {
        "command": "qb-backup-news-visibility",
        "seasons": list(SEASONS),
        "pft_index": str(PFT_INDEX),
        "pfr_index": str(PFR_INDEX),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    results["provenance"] = artifact_provenance(configuration, DEFAULT_FEATURES, project_root=REPO)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_artifact(
        run_dir,
        "result.json",
        results,
        command="qb-backup-news-visibility",
        metrics=results,
        notes=(
            "Backup-QB news-visibility measurement (PFT+PFR corpora) and, if warranted, "
            "the predeclared news-visible screen cell vs. bias_battery_backup_qb_start. "
            "No rotation-registry window drawn."
        ),
    )
    print(f"\nwrote {run_dir / 'result.json'}")

    backup_rows_out = backup_rows.drop(columns=[c for c in ("_flag",) if c in backup_rows.columns])
    backup_rows_out.to_parquet(run_dir / "backup_rows_detail.parquet", index=False)
    print(f"wrote {run_dir / 'backup_rows_detail.parquet'}")


if __name__ == "__main__":
    main()
