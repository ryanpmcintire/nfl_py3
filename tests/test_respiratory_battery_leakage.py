"""Leakage regression test for the total-respiratory-illness battery
(``docs/respiratory_battery.md``): proves a revision issued AFTER a game's
decision-cutoff Tuesday can never reach that game's ``respiratory_total``
AS-OF feature, and that a missing pathogen poisons the summed feature
instead of being silently treated as 0.

Modeled on ``tests/test_gdelt_attention_screen.py``'s canary pattern: load
the real script module (not a reimplementation) and feed it a synthetic
fixture with a deliberately extreme "future" value that must not leak.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("respiratory_battery_screen_test", "respiratory_battery_screen.py")


def test_epiweek_to_release_date_round_trips_through_cdc_epiweek() -> None:
    """``epiweek_to_release_date`` must be the true inverse of
    ``fluview_battery_screen.cdc_epiweek``, landing on the LAST day (a
    Saturday) of that epiweek -- not merely some day inside it."""

    # Derived from real calendar dates (never hand-picked WW numbers) so
    # every epiweek tested is guaranteed to actually exist -- some years
    # have 52 CDC weeks and some have 53, so a hardcoded "...53" can silently
    # overflow into week 1 of the following year.
    anchor_dates = pd.to_datetime(
        ["2017-10-05", "2020-01-02", "2023-01-05", "2024-12-30", "2026-08-22"]
    )
    for anchor in anchor_dates:
        epiweek = screen.cdc_epiweek(anchor)
        release_date = screen.epiweek_to_release_date(epiweek)
        assert release_date.dayofweek == 5  # Saturday
        assert screen.cdc_epiweek(release_date) == epiweek
        # The following day belongs to the NEXT epiweek -- confirms this is
        # the epiweek's LAST day, the conservative (latest-possible) anchor
        # docs/respiratory_battery.md section 3 requires.
        next_day = release_date + pd.Timedelta(days=1)
        assert screen.cdc_epiweek(next_day) != epiweek


def test_asof_lookup_never_sees_a_revision_issued_after_the_cutoff() -> None:
    """Canary: epiweek 202301's value is revised from 1.0 to 99.0 (an
    extreme, unmistakable value) by a MUCH later issue. A decision cutoff
    strictly before that later issue's release date must see the OLD value,
    never 99.0; a cutoff on/after that release date must see the revision.
    This is the exact property ``docs/respiratory_battery.md`` section 3
    claims for the reused FluView checkpoint/merge_asof machinery."""

    raw = pd.DataFrame(
        {
            "region": ["zz", "zz"],
            "pathogen_signal": ["pct_ed_visits_covid", "pct_ed_visits_covid"],
            "time_value": [202301, 202301],
            "issue": [202305, 202320],  # early issue, then a much later revision
            "lag": [4, 19],
            "value": [1.0, 99.0],
        }
    )
    checkpoints = screen.build_pathogen_checkpoints(raw, "pct_ed_visits_covid")
    checkpoint = checkpoints["zz"]

    release_early = screen.epiweek_to_release_date(202305)
    release_late = screen.epiweek_to_release_date(202320)
    assert release_late > release_early

    # String round-trip (not a hardcoded dtype=...): matches how
    # ``_to_fluview_shape`` derives ``release_date`` and how
    # ``load_schedules`` derives real games' ``cutoff_date`` (schedules.
    # parquet's ``gameday`` is stored as plain strings), so this test's
    # dtype matches production by the same construction, not by luck.
    cutoffs = pd.to_datetime(
        pd.Series([release_late - pd.Timedelta(days=1), release_late]).astype(str)
    )
    looked_up = screen.asof_lookup(checkpoint, cutoffs)

    assert looked_up.loc[0, "known_ili"] == 1.0, "revision leaked BEFORE its release date"
    assert looked_up.loc[1, "known_ili"] == 99.0, "revision not visible on its own release date"


def test_respiratory_total_requires_all_three_pathogens_non_missing() -> None:
    """docs/respiratory_battery.md section 3: ``respiratory_total`` sums
    covid+flu+rsv AS-OF values ONLY when all three are present for that
    side. A pathogen with zero rows for a state (e.g. RSV never reporting
    for the home team's market) must poison the sum to missing, never be
    silently treated as 0."""

    covid, influenza, rsv = screen.PATHOGEN_SIGNALS

    def _single_row_raw(signal: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "region": ["hs"],
                "pathogen_signal": [signal],
                "time_value": [202301],
                "issue": [202301],
                "lag": [0],
                "value": [1.0],
            }
        )

    def _empty_raw(signal: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "region": pd.Series(dtype="object"),
                "pathogen_signal": pd.Series(dtype="object"),
                "time_value": pd.Series(dtype="int64"),
                "issue": pd.Series(dtype="int64"),
                "lag": pd.Series(dtype="int64"),
                "value": pd.Series(dtype="float64"),
            }
        )

    # covid and influenza both report a real as-of value for the home
    # state; RSV never reports at all for that state (an upstream coverage
    # gap, not a "zero illness" reading).
    raw = pd.concat(
        [_single_row_raw(covid), _single_row_raw(influenza), _empty_raw(rsv)],
        ignore_index=True,
    )

    cutoff = screen.epiweek_to_release_date(202301) + pd.Timedelta(days=30)
    games = pd.DataFrame(
        {
            "home_state": ["hs"],
            "away_state": ["as"],  # a state with no data at all, either side
            "cutoff_date": pd.to_datetime(pd.Series([cutoff]).astype(str)),
        }
    )
    scored = screen.attach_asof_respiratory(games, raw)

    assert scored.loc[0, f"home_{covid}"] == 1.0
    assert scored.loc[0, f"home_{influenza}"] == 1.0
    assert pd.isna(scored.loc[0, f"home_{rsv}"])
    assert pd.isna(scored.loc[0, "home_respiratory_total"]), (
        "a missing pathogen (RSV) was silently treated as 0 instead of poisoning the sum"
    )
    assert pd.isna(scored.loc[0, "away_respiratory_total"])
