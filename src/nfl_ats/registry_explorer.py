"""Read-only registry and overlap explorer (ROADMAP.md Phase 13, ENG-07).

Both registries already have write paths, validation, and one-off report
commands (``nfl-ats weak-signals status/pool``, ``nfl-ats rotation status``).
What is missing is a single place that answers "what should be looked at
next" without opening ``docs/pool_edge_plan.md`` and re-deriving it by hand
every session -- which is exactly what the 2026-08-31 "registry state and
next shots" addendum in that doc is: a manual survey pass that goes stale the
moment the registry grows. This module is the mechanical replacement for
that survey. It is a pure reporting layer: every function here takes an
already-loaded :class:`nfl_ats.weak_signals.Registry` and/or
:class:`nfl_ats.rotation.Registry` and returns plain dicts/lists. Nothing in
this module calls either registry's ``save_registry``/``record_*`` writer,
and nothing here should ever be extended to do so -- see
``tests/test_registry_explorer.py``'s byte-identical-file assertion, which
exists specifically to catch a future edit that adds one.

**Binding taxonomy this module's callers must respect (verbatim, since a
module has no access to AGENTS.md/CLAUDE.md's session context injection):**

    An interval or CI that contains zero is NEVER grounds to reject, fail,
    or close an experiment. At this evaluator's ~2-point resolution,
    "contains zero" is the EXPECTED outcome for a real small signal. Only
    two grounds ever close a line of work: (1) refuted mechanism -- a
    RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
    split-half reliability; (2) bounded by a positive control proven able
    to detect an effect that size. Everything else is
    `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
    report `probability_positive`, never the binary "contains zero". If a
    record command errors, the verdict is wrong, not the validator.

This module never closes or reclassifies anything itself; every view below
either reports what is already recorded or computes a bounded, clearly
labelled aggregate over it (see :func:`shared_population_groups`'s
"effective sample size" bounds). It also never computes "games needed" --
that quantity is banned project-wide (within-week correlation is fixed at
zero by owner mandate, so there is no sound way to derive one) -- and never
prints or reads API keys.

Five views, matching the ENG-07 definition of done:

- :func:`unresolved_signals` -- (a) unresolved weak-signal entries, filtered.
- :func:`repeated_windows` -- (b) rotation-registry season blocks touched by
  more than one family, and the mined-2018-2025 discount rule.
- :func:`shared_population_groups` -- (c) weak-signal groups whose game
  windows overlap, with a bounded effective-sample-size read.
- :func:`source_availability` -- (d) per-family capture-job/source mapping.
- :func:`next_shots` -- (e) the ranked prioritisation output built from (a),
  (c), and the rotation registry's remaining capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nfl_ats import rotation, weak_signals

VIEWS: tuple[str, ...] = (
    "unresolved",
    "repeated-windows",
    "shared-populations",
    "source-availability",
    "next-shots",
)


# ---------------------------------------------------------------------------
# (a) Unresolved signals
# ---------------------------------------------------------------------------


def unresolved_signals(
    registry: weak_signals.Registry,
    *,
    league: str | None = None,
    effect_units: str | None = None,
    family: str | None = None,
) -> list[dict[str, Any]]:
    """Every ``unresolved_below_power`` entry, filtered and self-contained.

    A refuted mechanism or control-bounded null is excluded on purpose --
    those are closed, not open research questions, matching
    ``weak_signals.poolable_signals``'s own eligibility rule. Sorted by
    ``probability_positive`` descending (entries with no recorded value sort
    last, never treated as zero) so the most promising open leads read
    first even before ranking against rotation-window availability in
    :func:`next_shots`.
    """

    rows: list[dict[str, Any]] = []
    for signal in registry.signals.values():
        if signal.classification != weak_signals.POOLABLE_CLASSIFICATION:
            continue
        if league is not None and signal.league != league:
            continue
        if effect_units is not None and signal.effect_units != effect_units:
            continue
        signal_family = weak_signals.signal_family(signal)
        if family is not None and signal_family != family:
            continue
        rows.append(
            {
                "name": signal.name,
                "league": signal.league,
                "family": signal_family,
                "category": signal.category,
                "effect": signal.effect,
                "effect_units": signal.effect_units,
                "interval": None if signal.interval is None else list(signal.interval),
                "probability_positive": signal.probability_positive,
                "sample_games": signal.sample_games,
                "sample_blocks": signal.sample_blocks,
                "seasons": list(signal.seasons),
                "reliability": signal.reliability,
                "source": signal.source,
                "recorded_at": signal.recorded_at,
            }
        )
    rows.sort(
        key=lambda row: (
            row["probability_positive"] is None,
            -(row["probability_positive"] or 0.0),
            row["name"],
        )
    )
    return rows


# ---------------------------------------------------------------------------
# (b) Repeated windows
# ---------------------------------------------------------------------------


def repeated_windows(reg: rotation.Registry) -> dict[str, Any]:
    """Season blocks the rotation registry has drawn more than once.

    A single family cannot re-look at its own window -- ``record_look``
    marks a window spent forever, and ``rotation._validate`` refuses a
    family that overlaps its own or its inheritance chain's prior windows
    (``src/nfl_ats/rotation.py``, the block starting at the
    "A family must not re-look at seasons" comment). So "repeated" here
    means the other thing rule 4 explicitly allows and the ledger tracks
    for visibility: **two or more independent families drawing the same
    season(s)** (``docs/rotation_registry.md`` rule 4: "Windows retire
    per-family, not globally... two different families MAY draw
    overlapping seasons -- their hypotheses are independent -- but the
    ledger records global usage per season so accumulating cross-family
    multiplicity stays visible instead of silent"). ``rotation.season_usage``
    already computes the count; this reports the season-by-season detail
    plus the specific windows involved.

    Separately, rule 6 (same doc) singles out one particular kind of reuse
    as carrying a mandatory disclosure: any window intersecting the mined
    2018-2025 seasons requires ``acknowledges_mined_2018_2025`` and "a
    result there carries a discount that the write-up must state" -- not a
    ban (the project's "opener windows are not scarce" correction is
    explicit that reuse does not "dilute" a window and blocks may be
    redrawn), a disclosed penalty on how much weight a decision should put
    on the result. Both facts are reported together here because the
    reuse-discount rule a caller needs to cite differs by which kind of
    repetition it is looking at.
    """

    season_touches: dict[int, list[dict[str, Any]]] = {}
    mined_windows: list[dict[str, Any]] = []
    mined_seasons = frozenset(range(rotation.MINED_SEASONS[0], rotation.MINED_SEASONS[1] + 1))

    for name, family in reg.families.items():
        for window in family.windows:
            for season in window.covered_seasons:
                season_touches.setdefault(season, []).append(
                    {
                        "family": name,
                        "grade": family.grade,
                        "state": window.state,
                        "window_kind": window.window_kind,
                        "seasons": list(window.seasons),
                        "verdict": window.verdict,
                        "probability_positive": window.probability_positive,
                    }
                )
            if set(window.covered_seasons) & mined_seasons:
                mined_windows.append(
                    {
                        "family": name,
                        "grade": family.grade,
                        "seasons": list(window.seasons),
                        "window_kind": window.window_kind,
                        "state": window.state,
                        "verdict": window.verdict,
                        "acknowledges_mined_2018_2025": family.acknowledges_mined_2018_2025,
                    }
                )

    multi_family_seasons: list[dict[str, Any]] = []
    usage = rotation.season_usage(reg)
    for season in sorted(season_touches):
        touches = season_touches[season]
        families_here = sorted({touch["family"] for touch in touches})
        if len(families_here) > 1:
            multi_family_seasons.append(
                {
                    "season": season,
                    "families": families_here,
                    "family_count": len(families_here),
                    "spent_family_count": usage.get(str(season), 0),
                    "touches": touches,
                }
            )

    return {
        "multi_family_seasons": multi_family_seasons,
        "mined_era_windows": sorted(mined_windows, key=lambda row: (row["family"], row["seasons"])),
        "reuse_discount_rule": (
            "docs/rotation_registry.md rule 4 (src/nfl_ats/rotation.py): windows "
            "retire PER-FAMILY, not globally -- overlapping draws by independent "
            "families are allowed by design and reported here for visibility, "
            "not as an error. Rule 6: a window intersecting the mined "
            f"{rotation.MINED_SEASONS[0]}-{rotation.MINED_SEASONS[1]} seasons requires "
            "the family to declare acknowledges_mined_2018_2025 and 'carries a "
            "discount that the write-up must state' -- a disclosed penalty on "
            "how much a result should move a decision, never a ban on drawing it."
        ),
    }


# ---------------------------------------------------------------------------
# (c) Shared populations
# ---------------------------------------------------------------------------


def shared_population_groups(
    registry: weak_signals.Registry,
    *,
    league: str | None = None,
    effect_units: str | None = None,
) -> dict[str, Any]:
    """Groups of unresolved signals whose measurement windows overlap.

    Grouping key and the overlap test both mirror
    ``weak_signals.family_overlap_warnings`` exactly (same
    ``signal_family`` grouping, same "do these two entries' ``[seasons[0],
    seasons[1]]`` ranges intersect" pairwise test) -- reimplemented rather
    than called directly because that function reports only group-level
    counts, and this view additionally needs each overlapping member's
    identity to compute a bounded effective-sample-size read per group.
    ``pool_summary`` below is the direct, unmodified output of
    ``family_overlap_warnings`` on the same input, included so every number
    here is traceable back to the exact function ``nfl-ats weak-signals
    pool`` already uses.

    **Effective sample size is reported as a bound, not a point estimate.**
    Members of one group are correlated decompositions of the same window
    (AGENTS.md), so summing their ``sample_games``/``sample_blocks`` treats
    them as independent information, which the overlap makes false --
    that sum is reported as ``naive_sum_upper_bound``. The amount of
    information a single best-covered member alone already carries is
    reported as ``max_single_member_lower_bound``. The true effective N for
    the group lies somewhere between the two; this module does not invent a
    single number for it (and never computes a "games needed" figure --
    that quantity is banned project-wide).
    """

    signals = weak_signals.poolable_signals(registry, league=league, effect_units=effect_units)
    pool_summary = weak_signals.family_overlap_warnings(signals)

    families: dict[tuple[str, str], list[weak_signals.WeakSignal]] = {}
    for signal in signals:
        families.setdefault((signal.league, weak_signals.signal_family(signal)), []).append(signal)

    groups: list[dict[str, Any]] = []
    group_index = 0
    for (league_name, family), members in sorted(families.items()):
        overlapping_names: set[str] = set()
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                low = max(first.seasons[0], second.seasons[0])
                high = min(first.seasons[1], second.seasons[1])
                if low <= high:
                    overlapping_names.add(first.name)
                    overlapping_names.add(second.name)
        if not overlapping_names:
            continue
        group_index += 1
        group_members = sorted(
            (member for member in members if member.name in overlapping_names),
            key=lambda member: member.name,
        )
        games = [member.sample_games for member in group_members if member.sample_games is not None]
        blocks = [
            member.sample_blocks for member in group_members if member.sample_blocks is not None
        ]
        groups.append(
            {
                "group_id": f"{league_name}:{family}:{group_index}",
                "league": league_name,
                "family": family,
                "members": [member.name for member in group_members],
                "member_count": len(group_members),
                "shared_seasons": [
                    min(member.seasons[0] for member in group_members),
                    max(member.seasons[1] for member in group_members),
                ],
                "effective_sample_size_games": {
                    "naive_sum_upper_bound": sum(games) if games else None,
                    "max_single_member_lower_bound": max(games) if games else None,
                    "members_missing_sample_games": sum(
                        1 for member in group_members if member.sample_games is None
                    ),
                },
                "effective_sample_size_blocks": {
                    "naive_sum_upper_bound": sum(blocks) if blocks else None,
                    "max_single_member_lower_bound": max(blocks) if blocks else None,
                },
            }
        )

    groups.sort(key=lambda group: (-group["member_count"], group["group_id"]))
    return {
        "groups": groups,
        "pool_summary": pool_summary,
        "note": (
            "Members within one group are correlated decompositions of a shared "
            "window (AGENTS.md, docs/registry_correlation_audit_20260822.md section 3): "
            "treat each group as one dependent vote, not len(members) independent "
            "ones. effective_sample_size_* bounds the group's true information "
            "content; it does not assert a point value, and this module never "
            "computes a 'games needed' number from it."
        ),
    }


# ---------------------------------------------------------------------------
# (d) Source availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRule:
    """One family-prefix rule, with the citation for how it was established.

    ``status`` is one of ``captured_scheduled`` (an enabled job in
    ``scripts/capture_scheduler.py``'s ``SCHEDULE`` feeds this family),
    ``paused_scheduled`` (a job exists but is disabled, or its dependency is
    environment-conditional), ``derived_no_separate_capture`` (built purely
    from data already ingested through the main feature pipeline -- schedule
    fields, static reference tables -- with no distinct capture job),
    ``bulk_ingest_unscheduled`` (a manual/periodic ingest script exists under
    ``scripts/`` but is not in the scheduler's ``SCHEDULE``), or ``mixed``
    (part of the family is captured, part is not; see ``detail``).
    """

    prefix: str
    status: str
    detail: str
    citation: str


#: Every entry below was verified this session by reading the cited file(s)
#: -- see each ``citation``. A family not matched by any prefix here falls
#: through to :data:`_CATEGORY_FALLBACK` (a lower-confidence, category-level
#: inference, explicitly labelled as such) and finally to ``"unknown"``. Do
#: not add an entry without a citation: this table exists specifically so
#: source availability is never guessed (ENG-07's own requirement).
FAMILY_SOURCE_RULES: tuple[SourceRule, ...] = (
    SourceRule(
        "public_betting",
        "captured_scheduled",
        "Action Network public bet%/money% snapshot; scheduled jobs "
        "public_betting_sat (Sat 12:00 ET) and public_betting_sun (Sun 12:00 "
        "ET), both enabled=True.",
        "scripts/capture_scheduler.py:328-351 (read)",
    ),
    SourceRule(
        "odds_microstructure",
        "captured_scheduled",
        "Market microstructure over The Odds API point-in-time snapshots; six "
        "scheduled jobs (odds_tue_open, odds_thu_tnf, odds_sat, odds_sun_close, "
        "odds_sun_late, odds_mon_mnf), all enabled=True.",
        "scripts/capture_scheduler.py:213-326 (read)",
    ),
    SourceRule(
        "transaction_wire_battery",
        "captured_scheduled",
        "Pro Football Rumors transaction wire; scheduled jobs "
        "pfr_transactions_wed / pfr_transactions_sat (Wed/Sat 07:00 ET), both "
        "enabled=True.",
        "scripts/capture_scheduler.py:795-836 (read)",
    ),
    SourceRule(
        "player_arrest",
        "captured_scheduled",
        "USA Today public arrests table; scheduled job player_arrests_tue "
        "(Tue 07:00 ET), enabled=True.",
        "scripts/capture_scheduler.py:486-511 (read)",
    ),
    SourceRule(
        "referee_battery",
        "mixed",
        "Historical crew-tendency construction is derived from the existing "
        "feature/pbp table (no separate capture); the CURRENT week's "
        "officiating assignment used to join it prospectively is captured by "
        "referee_assignments_wed (Wed 15:00 ET, enabled=True).",
        "scripts/capture_scheduler.py:761-785 (read)",
    ),
    SourceRule(
        "penalty_crew_tendencies",
        "mixed",
        "Same basis as referee_battery: historical crew tendencies are "
        "derived, not captured; the current week's assignment feed is "
        "referee_assignments_wed (Wed 15:00 ET, enabled=True).",
        "scripts/capture_scheduler.py:761-785 (read)",
    ),
    SourceRule(
        "sagarin",
        "bulk_ingest_unscheduled",
        "Jeff Sagarin power ratings backfilled from the Wayback Machine via "
        "scripts/ingest_sagarin_ratings.py; no job referencing sagarin exists "
        "in capture_scheduler.py's SCHEDULE.",
        "scripts/ingest_sagarin_ratings.py:1-20 (read); "
        "scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "fluview",
        "bulk_ingest_unscheduled",
        "CDC Delphi FluView state-level ILI history, bulk-ingested via "
        "scripts/fluview_battery_ingest.py (full multi-issue revision "
        "history, pulled once/periodically rather than on a weekly job); no "
        "fluview job in capture_scheduler.py's SCHEDULE.",
        "scripts/fluview_battery_ingest.py:1-25 (read); "
        "scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "illness_on_production",
        "bulk_ingest_unscheduled",
        "Built on the same FluView bulk ingest as the fluview_* family "
        "(docs/pool_edge_plan.md WP3 read); no scheduled job.",
        "scripts/fluview_battery_ingest.py:1-25 (read); "
        "scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "arctic_shift",
        "bulk_ingest_unscheduled",
        "Reddit history via scripts/arctic_shift_battery_fetch.py; no "
        "arctic_shift job in capture_scheduler.py's SCHEDULE.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "reddit_attention",
        "bulk_ingest_unscheduled",
        "Reddit-derived attention metric; same Arctic Shift bulk-ingest basis "
        "as the arctic_shift_battery family, no scheduled job.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "attention_battery",
        "bulk_ingest_unscheduled",
        "GDELT/media-attention bulk ingest via scripts/ingest_gdelt_attention.py "
        "or scripts/ingest_sports_media_watch.py; no matching job in "
        "capture_scheduler.py's SCHEDULE.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "attention_followup",
        "bulk_ingest_unscheduled",
        "Follow-up cells on the same attention_battery bulk ingest; no scheduled job.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "gdelt",
        "bulk_ingest_unscheduled",
        "GDELT attention data via scripts/ingest_gdelt_attention.py; no scheduled job.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "weather_battery",
        "mixed",
        "The AQI portion is captured by airnow_tue_checkpoint (Tue 11:40 ET, "
        "enabled=True); broader forecast weather (temp/wind/precip) is "
        "ingested via scripts/ingest_forecast_archive.py, which has no "
        "corresponding scheduled job.",
        "scripts/capture_scheduler.py:245-264 (read); "
        "scheduler job-name grep (measured, no 'weather'/'forecast' match)",
    ),
    SourceRule(
        "environmental_battery",
        "mixed",
        "Same split as weather_battery: AQI is captured (airnow_tue_checkpoint, "
        "enabled=True); other environmental fields are bulk-ingested with no "
        "scheduled job.",
        "scripts/capture_scheduler.py:245-264 (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "weak_stack_v4_forecast_weather",
        "bulk_ingest_unscheduled",
        "NWS-style forecast weather via scripts/ingest_forecast_archive.py; "
        "no scheduled job (distinct from the AQI-only airnow job).",
        "scheduler job-name grep (measured, no 'forecast' match)",
    ),
    SourceRule(
        "body_clock",
        "derived_no_separate_capture",
        "Kickoff time plus stadium lat/lon/timezone, pregame-known schedule "
        "facts already in the ingested schedules.parquet snapshot plus the "
        "static registry/stadium_coordinates.json reference table; no "
        "separate capture job.",
        "scripts/nfl_travel_rest_battery_screen.py:1-25 (read)",
    ),
    SourceRule(
        "bye_overval",
        "derived_no_separate_capture",
        "Bye-week/rest flags (home_rest/away_rest) already present in the "
        "ingested schedules.parquet snapshot; no separate capture job.",
        "scripts/nfl_bias_battery_screen.py:27-31 (read, confirms "
        "home_rest/away_rest/roof/surface are schedules.parquet columns)",
    ),
    SourceRule(
        "roof_battery",
        "derived_no_separate_capture",
        "Roof/surface fields already present in the ingested "
        "schedules.parquet snapshot; no separate capture job.",
        "scripts/nfl_bias_battery_screen.py:27-31 (read)",
    ),
    SourceRule(
        "dst_transition",
        "derived_no_separate_capture",
        "Clock-change shock measured against schedule dates already ingested; "
        "no external source beyond the schedule.",
        "scripts/dst_transition_battery_screen.py:1-20 (read)",
    ),
    SourceRule(
        "graph_input_screen",
        "derived_no_separate_capture",
        "Built from the existing nflverse-play-by-play-derived feature table "
        "(data/processed/game_features.parquet) via a graph/team-stat "
        "transform; not a distinct capture source. No matching job in "
        "capture_scheduler.py's SCHEDULE.",
        "scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "graph_ratings_v2",
        "derived_no_separate_capture",
        "Same basis as graph_input_screen: built from the existing feature "
        "table, no separate capture job.",
        "scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "graph_team_stat",
        "derived_no_separate_capture",
        "Same basis as graph_input_screen: built from the existing feature "
        "table, no separate capture job.",
        "scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "unit_apm",
        "derived_no_separate_capture",
        "Adjusted plus-minus built from existing nflverse pbp/roster "
        "participation data already in the feature pipeline; no separate "
        "capture job.",
        "registry/weak_signals.json unit_apm_* 'source' fields (read this "
        "session); scheduler job-name grep (measured, no 'apm' match)",
    ),
    SourceRule(
        "surface_familiarity",
        "derived_no_separate_capture",
        "Turf/venue history derived from the schedules.parquet venue+surface "
        "columns, the same basis confirmed for roof_battery/bye_overval.",
        "registry/weak_signals.json surface_familiarity_* 'source' field (read this session)",
    ),
    SourceRule(
        "altitude_deficit",
        "derived_no_separate_capture",
        "Venue elevation vs. an away team's modal home elevation; static "
        "reference table registry/stadium_elevations.json plus the schedule's "
        "venue field, no live capture.",
        "registry/weak_signals.json altitude_deficit_4000ft 'notes' field (read this session)",
    ),
    SourceRule(
        "ffc_adp",
        "bulk_ingest_unscheduled",
        "Fantasy football ADP via scripts/ingest_ffc_adp.py; no scheduled job.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "combine",
        "bulk_ingest_unscheduled",
        "NFL Combine data via scripts/ingest_combine.py; no scheduled job.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "drought",
        "bulk_ingest_unscheduled",
        "US Drought Monitor via scripts/ingest_drought_monitor.py; no scheduled job.",
        "scripts/ directory listing (read); scheduler job-name grep (measured, no match)",
    ),
    SourceRule(
        "injury_value_lost",
        "mixed",
        "NFL.com injury-report jobs (injuries_wed/thu/fri/sat) exist but are "
        "PAUSED (enabled=False, MKT-09 source policy: NFL.com terms require "
        "express consent). Licensed Sportradar replacement jobs exist but are "
        "enabled only when SPORTRADAR_API_KEY is set in the running "
        "environment (dynamic; this table reports the dependency, not "
        "today's environment state, and never inspects or prints the key).",
        "scripts/capture_scheduler.py:356-403 (read, nflcom paused); "
        ":404-418 (read, sportradar conditional)",
    ),
    SourceRule(
        "injury_signal",
        "mixed",
        "Same dependency as injury_value_lost: NFL.com paused, Sportradar "
        "conditional on SPORTRADAR_API_KEY.",
        "scripts/capture_scheduler.py:356-403, :404-418 (read)",
    ),
    SourceRule(
        "movement_attribution",
        "mixed",
        "Injury-news-attributed line movement; same paused/conditional "
        "injury-capture dependency as injury_value_lost, joined against the "
        "captured_scheduled odds snapshots.",
        "scripts/capture_scheduler.py:356-403, :404-418, :213-326 (read)",
    ),
    SourceRule(
        "cfb_",
        "bulk_ingest_unscheduled",
        "College Football Data (CFBD) historical feature table; "
        "capture_scheduler.py's SCHEDULE has no CFB-named job -- it covers "
        "the NFL weekly cadence only.",
        "scheduler grep for 'cfb' in job names (measured, no match)",
    ),
)

#: Second-tier, lower-confidence fallback keyed by the registry's own
#: ``category`` field (an existing structured field, not inferred by this
#: module) when no :data:`FAMILY_SOURCE_RULES` prefix matches. Every detail
#: string says "not verified for this specific family" so a reader can never
#: mistake this tier for the citation-backed rows above.
_CATEGORY_FALLBACK: dict[str, tuple[str, str]] = {
    "schedule": (
        "inferred_from_category",
        "No family-specific rule matched. Category='schedule': verified "
        "schedule-derived families (body_clock, bye_overval, roof, "
        "dst_transition) all read pregame-known fields already in the "
        "ingested schedules.parquet snapshot with no separate capture job; "
        "not verified for this specific family.",
    ),
    "environment": (
        "inferred_from_category",
        "No family-specific rule matched. Category='environment': verified "
        "examples split between derived-from-schedule (altitude, surface, "
        "roof) and a mixed AQI-captured/forecast-unscheduled split "
        "(weather_battery); not verified for this specific family.",
    ),
    "market": (
        "inferred_from_category",
        "No family-specific rule matched. Category='market': likely tied to "
        "a captured_scheduled odds or public-betting job (verified for "
        "odds_microstructure and public_betting above) or a historical odds "
        "archive; not verified for this specific family.",
    ),
    "health": (
        "inferred_from_category",
        "No family-specific rule matched. Category='health': likely tied to "
        "the paused NFL.com / conditional Sportradar injury jobs, the "
        "captured player-arrests job, or the unscheduled FluView bulk "
        "ingest (all verified above for their own families); not verified "
        "for this specific family.",
    ),
    "attention": (
        "inferred_from_category",
        "No family-specific rule matched. Category='attention': likely an "
        "unscheduled bulk attention/media ingest (gdelt/reddit/ffc-adp, "
        "verified above for their own families); not verified for this "
        "specific family.",
    ),
    "offfield": (
        "inferred_from_category",
        "No family-specific rule matched. Category='offfield': likely the "
        "captured player-arrests or transaction-wire jobs, or an unscheduled "
        "bulk ingest; not verified for this specific family.",
    ),
    "onfield": (
        "inferred_from_category",
        "No family-specific rule matched. Category='onfield': typically "
        "derived from the existing nflverse-play-by-play feature table with "
        "no separate capture job (verified for graph_* and unit_apm above); "
        "not verified for this specific family.",
    ),
    "modeling": (
        "not_applicable_modeling",
        "Category='modeling': an internal comparison/construction over data "
        "already in the feature table or a prior artifact, not a claim "
        "about a new external source. No capture job applies.",
    ),
    "control": (
        "not_applicable_modeling",
        "Category='control': a placebo/oracle/instrument check, not a claim "
        "about a new external source. No capture job applies.",
    ),
}


def _source_for_family(family: str, category: str | None) -> tuple[str, str, str | None]:
    """Return (status, detail, citation) for one family. Never guesses."""

    matches = [
        rule
        for rule in FAMILY_SOURCE_RULES
        if family == rule.prefix or family.startswith(rule.prefix)
    ]
    if matches:
        best = max(matches, key=lambda rule: len(rule.prefix))
        return best.status, best.detail, best.citation
    if category is not None and category in _CATEGORY_FALLBACK:
        status, detail = _CATEGORY_FALLBACK[category]
        return status, detail, None
    return (
        "unknown",
        "No family-specific rule matched and no category-level inference "
        "applies (category is missing or not in the fallback table).",
        None,
    )


def source_availability(
    registry: weak_signals.Registry,
    *,
    league: str | None = None,
) -> list[dict[str, Any]]:
    """Per-family source/capture-job classification for every recorded family.

    One row per distinct ``(league, family)`` pair present in the registry
    (not per signal -- every member of a family shares its source). See
    :data:`FAMILY_SOURCE_RULES` for the citation behind every non-``unknown``
    row; nothing here is a guess.
    """

    seen: dict[tuple[str, str], str | None] = {}
    for signal in registry.signals.values():
        if league is not None and signal.league != league:
            continue
        key = (signal.league, weak_signals.signal_family(signal))
        # Prefer a signal that actually carries a category, in case members
        # of the same family were recorded inconsistently.
        if key not in seen or seen[key] is None:
            seen[key] = signal.category

    rows: list[dict[str, Any]] = []
    for (league_name, family), category in seen.items():
        status, detail, citation = _source_for_family(family, category)
        rows.append(
            {
                "league": league_name,
                "family": family,
                "category": category,
                "status": status,
                "detail": detail,
                "citation": citation,
            }
        )
    rows.sort(key=lambda row: (row["league"], row["family"]))
    return rows


# ---------------------------------------------------------------------------
# ENG-27: rotation-registry coverage plan.
# ---------------------------------------------------------------------------

COVERAGE_ACTION_STUB = "declare_stub"
COVERAGE_ACTION_NO_ROTATION_NEEDED = "no_rotation_needed"


def coverage_plan(
    weak_registry: weak_signals.Registry,
    rotation_registry: rotation.Registry,
) -> list[dict[str, Any]]:
    """Read-only ENG-27 coverage plan: what ``rotation declare-coverage`` WOULD do.

    ROADMAP.md Phase 13, ENG-27: measured 2026-09-04, the rotation registry
    declared ~29 families against 350+ weak-signal families, so
    :func:`next_shots` reported ``unspent_rotation_window: None`` for nearly
    every top row. This computes, for every distinct ``(league, family)``
    already present in ``weak_registry`` (the same grouping
    :func:`source_availability` uses) that :func:`matching_rotation_families`
    finds no rotation-family match for AND that ``rotation_registry.
    no_rotation_needed`` does not already excuse, one of two actions:

    - ``declare_stub``: no admissible :data:`rotation.NO_ROTATION_FIXED_REASONS`
      applies (:func:`rotation.classify_no_rotation_reason` returned
      ``None``), so the plan reserves a rotation-family stub named after the
      weak-signal family (falling back to a ``<family>__<league>`` suffix on
      a name collision, which cannot happen with the registry measured this
      session but is handled defensively for a future one).
    - ``no_rotation_needed``: the classifier found an admissible reason
      (CFB out-of-scope, reliability measurement, positive control, oracle,
      or retired profile), so the plan records that reason instead of a stub.
      ``league`` is passed to the classifier (ENG-37, ROADMAP.md Phase 13,
      2026-09-05): the rotation registry governs NFL confirmation looks only
      (rule 8, docs/rotation_registry.md), so every non-NFL family classifies
      to ``"cfb_out_of_scope"`` before any name/category rule is even
      consulted, and never gets a stub -- 54 CFB families had already been
      given one before this fix (measured 2026-09-04).

    Never guessed: a family only gets ``no_rotation_needed`` when the
    classifier names one of the fixed reasons; every other unmatched family
    gets a stub. This function reads both registries and writes to neither --
    the write path is ``rotation.declare_coverage_stub`` /
    ``rotation.record_no_rotation_needed``, driven by the CLI. Naturally
    idempotent: a family with either an existing rotation-family match or an
    existing ``no_rotation_needed`` record is skipped, so re-running this
    against an already-covered registry returns an empty plan.
    """

    seen_category: dict[tuple[str, str], str | None] = {}
    effect_units_by_family: dict[tuple[str, str], set[str]] = {}
    for signal in weak_registry.signals.values():
        key = (signal.league, weak_signals.signal_family(signal))
        if key not in seen_category or seen_category[key] is None:
            seen_category[key] = signal.category
        effect_units_by_family.setdefault(key, set()).add(signal.effect_units)

    reserved_names = set(rotation_registry.families)
    rows: list[dict[str, Any]] = []
    for league, family in sorted(seen_category):
        if matching_rotation_families(family, rotation_registry):
            continue
        if family in rotation_registry.no_rotation_needed:
            continue
        category = seen_category[(league, family)]
        units = sorted(effect_units_by_family[(league, family)])
        reason = rotation.classify_no_rotation_reason(family, category, league=league)
        if reason is not None:
            rows.append(
                {
                    "action": COVERAGE_ACTION_NO_ROTATION_NEEDED,
                    "weak_signal_family": family,
                    "league": league,
                    "category": category,
                    "effect_units": units,
                    "reason": reason,
                    "stub_name": None,
                }
            )
            continue
        stub_name = family
        suffix = 0
        while stub_name in reserved_names:
            suffix += 1
            stub_name = f"{family}__{league}" if suffix == 1 else f"{family}__{league}_{suffix}"
        reserved_names.add(stub_name)
        rows.append(
            {
                "action": COVERAGE_ACTION_STUB,
                "weak_signal_family": family,
                "league": league,
                "category": category,
                "effect_units": units,
                "reason": None,
                "stub_name": stub_name,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# (e) Next shots
# ---------------------------------------------------------------------------


def matching_rotation_families(
    weak_signal_family: str, rotation_registry: rotation.Registry
) -> list[str]:
    """Fuzzy-match one weak-signal family name against declared rotation families.

    Extracted from :func:`next_shots` (ENG-27, ROADMAP.md Phase 13) so the
    coverage tooling in ``nfl-ats rotation declare-coverage`` reuses this
    exact matcher instead of a second, drift-prone copy of the same rule.

    ``weak_signals.signal_family`` and the rotation registry's declared
    family names are independent naming conventions with no guaranteed
    correspondence -- a name is treated as a match if it equals or is a
    prefix/suffix superstring of the other. Best-effort, and every caller
    should say so rather than treating a match (or its absence) as certain.
    """

    return sorted(
        name
        for name in rotation_registry.families
        if name == weak_signal_family
        or name.startswith(weak_signal_family)
        or weak_signal_family.startswith(name)
    )


def _rotation_has_capacity(reg: rotation.Registry, name: str) -> bool:
    """Whether ``name`` currently holds an unspent window or could draw one."""

    family = reg.families[name]
    if family.assigned_window is not None:
        return True
    try:
        return len(rotation.eligible_blocks(reg, name)) > 0
    except rotation.RegistryError:
        return False


def next_shots(
    weak_registry: weak_signals.Registry,
    rotation_registry: rotation.Registry,
    *,
    league: str | None = None,
    effect_units: str | None = None,
    top: int | None = None,
) -> list[dict[str, Any]]:
    """The ranked prioritisation view: what to look at next, and why.

    Sort order: ``probability_positive`` descending (missing values sort
    last, never coerced to zero), then whether a matching rotation family
    still has an unspent window (an assigned-but-unspent window, or at
    least one eligible block left to draw), then name for determinism.

    **Family matching between the two registries is best-effort, and says
    so.** ``weak_signals.signal_family`` and the rotation registry's
    declared family names are independent naming conventions with no
    guaranteed correspondence (e.g. the weak-signal family
    ``graph_ratings_v2_team_stat`` has no rotation-registry counterpart at
    all, while ``fluview_home_market_elevated`` corresponds to
    ``fluview_home_elevated_opener``). A name is treated as a match if it
    equals or is a prefix/suffix superstring of the other; every row reports
    the exact ``matching_rotation_families`` list it matched against, and
    ``unspent_rotation_window`` is ``None`` (never a guessed ``False``) when
    no rotation family matched at all.
    """

    rows = unresolved_signals(weak_registry, league=league, effect_units=effect_units)
    shared = shared_population_groups(weak_registry, league=league, effect_units=effect_units)
    group_of: dict[str, str] = {
        member: group["group_id"] for group in shared["groups"] for member in group["members"]
    }

    enriched: list[dict[str, Any]] = []
    for row in rows:
        family = row["family"]
        matches = matching_rotation_families(family, rotation_registry)
        unspent: bool | None = None
        if matches:
            unspent = any(_rotation_has_capacity(rotation_registry, name) for name in matches)
        enriched.append(
            {
                **row,
                "overlap_group_id": group_of.get(row["name"]),
                "matching_rotation_families": matches or None,
                "unspent_rotation_window": unspent,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        probability = row["probability_positive"]
        has_probability = probability is not None
        unspent_state = row["unspent_rotation_window"]
        unspent_rank = 0 if unspent_state is True else (1 if unspent_state is None else 2)
        return (not has_probability, -(probability or 0.0), unspent_rank, row["name"])

    enriched.sort(key=sort_key)
    return enriched[:top] if top else enriched
