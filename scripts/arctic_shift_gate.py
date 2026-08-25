"""Shared-variance feasibility gate (scout v5 Section C #3): does weekly
team-subreddit post/comment volume carry variance the Wikipedia-pageview
attention feature does not?

Frozen gate rule (docs/arctic_shift_gate.md, frozen before computing):
proceed to an ATS battery only if pooled weekly subreddit-vs-Wikipedia
Pearson r < 0.7 AND subreddit volume year-over-year reliability >= 0.2.

Fetches daily post/comment aggregates for 6 fixed team subreddits x REG
2019-2021 from the Arctic Shift API (no auth), saves every raw response plus
a sha256 manifest under data/raw/arctic_shift/, aligns Tuesday-ending 7-day
windows identical to scripts/attention_battery_screen.py, compares against
Wikipedia pageviews from that script's scratchpad raw JSONs, evaluates the
gate, and writes artifacts/arctic_shift_gate/results.json with an
experiment-provenance stamp rooted inside that same directory (the shared
registry/experiments/ tree is never touched). Measure-only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.append(str(REPO / "scripts"))

from _common import latest_schedules  # noqa: E402

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

API_BASE = "https://arctic-shift.photon-reddit.com"
REQUEST_DELAY_SECONDS = 3.0
MAX_ATTEMPTS = 6
BACKOFF_BASE_SECONDS = 5.0

SUBREDDITS = {
    "DAL": "cowboys",
    "NE": "Patriots",
    "GB": "GreenBayPackers",
    "PIT": "steelers",
    "JAX": "Jaguars",
    "TEN": "Tennesseetitans",
}

WIKI_SCRATCH_RAW = Path(
    "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/"
    "26042060-ffd8-45a7-b2e7-a9b30b87bd34/scratchpad/agent_attention/raw"
)

WIKI_ARTICLES = {
    "DAL": ["Dallas_Cowboys"],
    "NE": ["New_England_Patriots"],
    "GB": ["Green_Bay_Packers"],
    "PIT": ["Pittsburgh_Steelers"],
    "JAX": ["Jacksonville_Jaguars"],
    "TEN": ["Tennessee_Titans"],
}

SEASONS = [2019, 2020, 2021]

GATE_R_THRESHOLD = 0.7
GATE_RELIABILITY_THRESHOLD = 0.2


def fetch_with_retries(url: str) -> bytes:
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nfl-ats-gate/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
            payload = json.loads(body.decode("utf-8"))
            if isinstance(payload, dict) and payload.get("error"):
                last_error = str(payload["error"])
                raise RuntimeError(last_error)
            return body
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            reset = exc.headers.get("X-RateLimit-Reset") if exc.headers else None
            wait = float(reset) + 5.0 if reset else BACKOFF_BASE_SECONDS * attempt
            print(f"  attempt {attempt} failed ({last_error}); waiting {wait:.0f}s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = str(exc)
            wait = BACKOFF_BASE_SECONDS * attempt
            print(f"  attempt {attempt} failed ({last_error}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up after {MAX_ATTEMPTS} attempts: {url}: {last_error}")


def fetch_subreddit_daily_counts(raw_dir: Path) -> dict[str, dict[str, pd.Series]]:
    """Return {team: {'volume': daily series, 'posts': ..., 'comments': ...}}."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest_entries: list[dict[str, object]] = []
    out: dict[str, dict[str, pd.Series]] = {}
    window_after = f"{min(SEASONS)}-08-01"
    window_before = f"{max(SEASONS) + 1}-02-15"

    for team, subreddit in SUBREDDITS.items():
        out[team] = {}
        for kind in ("posts", "comments"):
            params = urllib.parse.urlencode(
                {
                    "key": f"r/{subreddit}/{kind}/count",
                    "precision": "day",
                    "after": window_after,
                    "before": window_before,
                }
            )
            url = f"{API_BASE}/api/time_series?{params}"
            time.sleep(REQUEST_DELAY_SECONDS)
            print(f"fetch {kind} r/{subreddit}")
            started = time.time()
            body = fetch_with_retries(url)
            filename = f"{subreddit}_{kind}_timeseries_{min(SEASONS)}_{max(SEASONS)}.json"
            (raw_dir / filename).write_bytes(body)
            manifest_entries.append(
                {
                    "path": f"data/raw/arctic_shift/{filename}",
                    "url": url,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "elapsed_seconds": round(time.time() - started, 2),
                }
            )
            items = json.loads(body.decode("utf-8"))["data"] or []
            if items:
                dates = pd.to_datetime([int(it["date"]) for it in items], unit="s")
                counts = pd.Series(
                    [float(it["value"]) for it in items], index=dates, dtype="float64"
                )
                counts = counts.groupby(counts.index).sum()
            else:
                counts = pd.Series(dtype="float64")
            out[team][kind] = counts
        out[team]["volume"] = out[team]["posts"].add(out[team]["comments"], fill_value=0.0)

    manifest_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")
    return out


def load_wiki_daily_views(team: str) -> pd.Series:
    summed: pd.Series | None = None
    for article in WIKI_ARTICLES[team]:
        body = json.loads((WIKI_SCRATCH_RAW / f"{article}.json").read_text())
        items = body["items"]
        dates = pd.to_datetime([it["timestamp"][:8] for it in items], format="%Y%m%d")
        views = pd.Series([float(it["views"]) for it in items], index=dates, dtype="float64")
        views = views.groupby(views.index).sum()
        summed = views if summed is None else summed.add(views, fill_value=0.0)
    assert summed is not None
    return summed.sort_index()


def build_team_weeks(schedules_path: Path) -> pd.DataFrame:
    schedules = pd.read_parquet(schedules_path)
    games = schedules.loc[
        (schedules["game_type"] == "REG")
        & schedules["season"].isin(SEASONS)
        & (schedules["spread_line"].notna())
    ].copy()
    sides = []
    for team_col in ("home_team", "away_team"):
        side = pd.DataFrame(
            {
                "team": games[team_col],
                "season": games["season"].astype(int),
                "week": games["week"].astype(int),
                "gameday": pd.to_datetime(games["gameday"]),
            }
        )
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)
    long_df["team"] = long_df["team"].map(lambda c: TEAM_ABBREVIATION_ALIASES.get(c, c))
    long_df = long_df.loc[long_df["team"].isin(SUBREDDITS)]
    long_df = long_df.drop_duplicates(subset=["team", "season", "week"], keep="first")
    weekday = long_df["gameday"].dt.weekday
    window_end = long_df["gameday"] - pd.to_timedelta((weekday - 1) % 7, unit="D")
    long_df["window_end"] = window_end
    long_df["window_start"] = window_end - pd.Timedelta(days=6)
    iso = window_end.dt.isocalendar()
    long_df["iso_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    return long_df.sort_values(["team", "gameday"]).reset_index(drop=True)


def window_sum(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if series.empty:
        return 0.0
    idx = series.index
    mask = (idx >= start) & (idx <= end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return float(series.loc[mask].sum())


def pearson_with_ci(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": float("nan"), "n": n}
    r = float(np.corrcoef(x, y)[0, 1])
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return {"r": r, "n": n, "ci95_low": float(lo), "ci95_high": float(hi)}


def main() -> None:
    raw_dir = REPO / "data" / "raw" / "arctic_shift"
    output_dir = REPO / "artifacts" / "arctic_shift_gate"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"=== fetching Arctic Shift aggregates into {raw_dir} ===")
    subreddit_daily = fetch_subreddit_daily_counts(raw_dir)

    schedules_path = latest_schedules()
    print(f"=== building team-weeks from {schedules_path} ===")
    team_weeks = build_team_weeks(schedules_path)
    print(f"team-weeks: {len(team_weeks)}")

    vol_vals = np.empty(len(team_weeks))
    wiki_vals = np.empty(len(team_weeks))
    wiki_cache = {team: load_wiki_daily_views(team) for team in SUBREDDITS}
    for i, row in enumerate(team_weeks.itertuples(index=False)):
        vol_vals[i] = window_sum(
            subreddit_daily[row.team]["volume"], row.window_start, row.window_end
        )
        wiki_vals[i] = window_sum(wiki_cache[row.team], row.window_start, row.window_end)
    team_weeks["sub_volume"] = vol_vals
    team_weeks["wiki_views"] = wiki_vals

    log_vol = np.log1p(team_weeks["sub_volume"].to_numpy(dtype="float64"))
    log_wiki = np.log1p(team_weeks["wiki_views"].to_numpy(dtype="float64"))
    shared_variance = pearson_with_ci(log_vol, log_wiki)
    spearman_rho = pearson_with_ci(
        pd.Series(log_vol).rank().to_numpy(), pd.Series(log_wiki).rank().to_numpy()
    )["r"]
    raw_scale_r = pearson_with_ci(
        team_weeks["sub_volume"].to_numpy(), team_weeks["wiki_views"].to_numpy()
    )["r"]

    per_team: dict[str, dict[str, float | int]] = {}
    for team, group in team_weeks.groupby("team"):
        per_team[str(team)] = pearson_with_ci(
            np.log1p(group["sub_volume"].to_numpy()), np.log1p(group["wiki_views"].to_numpy())
        )

    season_means = (
        team_weeks.assign(log_vol=log_vol)
        .groupby(["team", "season"])["log_vol"]
        .mean()
        .unstack("season")
    )
    pair_corrs = []
    reliability_pairs = {}
    for a, b in ((2019, 2020), (2020, 2021)):
        pair = season_means[[a, b]].dropna()
        stat = pearson_with_ci(pair[a].to_numpy(), pair[b].to_numpy())
        reliability_pairs[f"{a}_vs_{b}"] = stat
        if not np.isnan(stat["r"]):
            pair_corrs.append(float(stat["r"]))
    reliability = float(np.mean(pair_corrs)) if pair_corrs else float("nan")

    leg1_pass = bool(shared_variance["r"] < GATE_R_THRESHOLD)
    leg2_pass = bool(reliability >= GATE_RELIABILITY_THRESHOLD)
    gate_pass = leg1_pass and leg2_pass
    verdict = "GATE PASS" if gate_pass else "GATE FAIL"

    print("\n=== shared variance (weekly subreddit volume vs Wikipedia views) ===")
    print(
        f"  pooled log-scale Pearson r={shared_variance['r']:.4f} "
        f"95% CI [{shared_variance['ci95_low']:.4f}, {shared_variance['ci95_high']:.4f}] "
        f"(n={shared_variance['n']})"
    )
    print(f"  Spearman rho={spearman_rho:.4f}; raw-scale r={raw_scale_r:.4f}")
    for team, stat in per_team.items():
        print(f"  {team}: r={stat['r']:.4f} (n={stat['n']})")
    print("\n=== year-over-year reliability of subreddit volume ===")
    for name, stat in reliability_pairs.items():
        print(f"  {name}: r={stat['r']:.4f} (n_teams={stat['n']})")
    print(f"  headline reliability (mean of pairs) = {reliability:.4f}")
    print(
        f"\n{verdict}: shared-variance leg (<{GATE_R_THRESHOLD}): {leg1_pass}; "
        f"reliability leg (>={GATE_RELIABILITY_THRESHOLD}): {leg2_pass}"
    )

    payload = {
        "gate_rule_frozen_in": "docs/arctic_shift_gate.md",
        "frozen_thresholds": {
            "shared_variance_r_below": GATE_R_THRESHOLD,
            "yoy_reliability_at_or_above": GATE_RELIABILITY_THRESHOLD,
        },
        "sampled": {
            "subreddits": SUBREDDITS,
            "seasons": SEASONS,
            "api_base": API_BASE,
            "raw_dir": "data/raw/arctic_shift",
        },
        "schedules": str(schedules_path),
        "n_team_weeks": len(team_weeks),
        "shared_variance_pooled_log_pearson": shared_variance,
        "shared_variance_spearman_rho": spearman_rho,
        "shared_variance_raw_scale_r": raw_scale_r,
        "per_team_log_pearson": per_team,
        "yoy_reliability_pairs": reliability_pairs,
        "yoy_reliability_headline": reliability,
        "leg_results": {"shared_variance_leg_pass": leg1_pass, "reliability_leg_pass": leg2_pass},
        "verdict": verdict,
        "caveats": [
            "6 teams sampled; YoY reliability rests on 6 teams per adjacent pair.",
            "Arctic Shift /api/time_series dates are UTC-midnight epochs; the "
            "search/aggregate endpoint's ~T22:00Z bucket labels were NOT used.",
            "Correlation here measures construct overlap, not predictive value; "
            "an ATS battery remains a separate decision.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    configuration = {
        "command": "arctic-shift-gate",
        "schedules": str(schedules_path),
        "api_base": API_BASE,
        "seasons": SEASONS,
        "subreddits": SUBREDDITS,
        "gate_thresholds": payload["frozen_thresholds"],
    }
    payload["provenance"] = artifact_provenance(configuration, schedules_path, project_root=REPO)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="arctic-shift-gate",
        metrics=payload,
        notes=(
            "Measure-only shared-variance gate (scout v5 Section C #3); no "
            "weak-signal registry writes. The experiment stamp is rooted under "
            "artifacts/arctic_shift_gate/experiment_registry via registry_root so "
            "the shared registry/experiments/ tree stays untouched."
        ),
        project_root=REPO,
        registry_root=output_dir / "experiment_registry",
    )
    team_weeks.to_csv(output_dir / "team_weeks.csv", index=False)
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
