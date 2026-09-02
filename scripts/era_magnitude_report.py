"""Read-only era-magnitude report over ``registry/weak_signals.json``.

Item 6 of the ranked agenda in ``docs/pool_edge_plan.md``'s "2026-08-31
registry state and next shots" section asks for per-era magnitude reporting
of the registry's already-recorded era-split constructs (per this project's
standing "era magnitude, not presence" rule: effects vary in MAGNITUDE
across eras, a weaker-era reading is never absence, and a sign flip between
eras must be reported as two magnitudes, never averaged away). This script
is the mechanical half of that: it finds every group of registry entries
that share a construct stem and differ only by an era suffix (season-range
in the name), and prints one Markdown table per group.

**Binding taxonomy this script and its output must respect (verbatim,
because a script has no access to AGENTS.md/CLAUDE.md's session context
injection):**

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

**This script records nothing and closes nothing.** It never calls
``nfl-ats weak-signals record`` / ``rotation record-look``, never writes to
``registry/``, and never writes an artifact file -- it only reads
``registry/weak_signals.json`` and prints. Grouping and heterogeneity are
purely mechanical (regex era-suffix stripping + disjoint-season detection);
which groups deserve a mechanism write-up is a judgement made in
``docs/era_magnitude_report.md``, not by this script.

**Grouping rule.** A construct's own name minus one era-suffix pattern
(``_era_YYYY_YYYY``, bare ``_YYYY_YYYY``, ``_preYYYY``/``_postYYYY``, or
``_pre_YYYY``/``_post_YYYY``) is its "stem". Two or more entries sharing a
stem, whose ``seasons`` ranges are pairwise disjoint, form an era-split
group. A stem-only entry with no suffix (e.g. ``pt_post_mnf_sunday``
alongside ``pt_post_mnf_sunday_era_2009_2017``/``_era_2018_2025``) is
reported separately as the group's full-range parent -- it overlaps every
era member's seasons at once, so it is marked "overlapping, not an
independent vote" rather than folded into the group's own table. This
mirrors the disjointness test that already gates ``family`` inference in
``nfl_ats.weak_signals.signal_family`` and the exclusion of full-range
parents from the sub-pools reported in ``docs/pool_edge_plan.md``'s
"2026-08-31 registry state and next shots" section 2 (read this session).

**Heterogeneity.** Between-era tau-squared is the DerSimonian-Laird
estimator, computed by calling ``nfl_ats.weak_signals.pooled_effect``
(method="random") directly on the group's era members -- not
reimplemented, so this script's numbers are guaranteed to agree with
``nfl-ats weak-signals pool``'s own arithmetic. The formula, stated because
the task requires it stated: for k era estimates theta_i with variances
v_i = SE_i^2 (SE recovered from the 95% interval as
width / (2 * 1.959963984540054) when no standard_error is recorded, exactly
``WeakSignal.resolved_standard_error``'s own rule), fixed-effect weights
w_i = 1/v_i, fixed-effect mean theta_F = sum(w_i * theta_i) / sum(w_i),
Q = sum(w_i * (theta_i - theta_F)^2), C = sum(w_i) - sum(w_i^2)/sum(w_i),
tau^2 = max(0, (Q - (k-1)) / C). The random-effects pooled interval then
reweights with w_i' = 1/(v_i + tau^2). tau^2 = 0 means the eras agree once
their own sampling noise is accounted for; tau^2 >> 0 means real detected
heterogeneity beyond what sampling noise explains -- the two eras are not
measuring the same thing.

Usage::

    ./.tools/uv.exe run --no-sync python scripts/era_magnitude_report.py [--markdown] [--league nfl]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.weak_signals import (  # noqa: E402
    WeakSignal,
    default_registry_path,
    load_registry,
    pooled_effect,
)

# Applied in order; the first pattern that matches and leaves a non-empty
# remainder wins. Ordered longest/most-specific first so
# "_era_2009_2017" is not mistaken for a bare "_2009_2017"-style match by a
# looser pattern trying first.
ERA_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"_era_\d{4}_\d{4}$"),
    re.compile(r"_era_\d{4}$"),
    re.compile(r"_\d{4}_\d{4}$"),
    re.compile(r"_pre_\d{4}$"),
    re.compile(r"_post_\d{4}$"),
    re.compile(r"_pre\d{4}$"),
    re.compile(r"_post\d{4}$"),
)

Z95 = 1.959963984540054


def era_stem(name: str) -> tuple[str, bool]:
    """Strip one era suffix from ``name``. Returns (stem, had_suffix)."""

    for pattern in ERA_SUFFIX_PATTERNS:
        stripped = pattern.sub("", name)
        if stripped and stripped != name:
            return stripped, True
    return name, False


@dataclass(frozen=True)
class EraGroup:
    stem: str
    league: str
    members: tuple[WeakSignal, ...]  # sorted by seasons[0]
    parent: WeakSignal | None  # exact-stem full-range entry, if one exists


def find_era_groups(signals: dict[str, WeakSignal], *, league: str) -> list[EraGroup]:
    """Every stem sharing >=2 same-league entries with pairwise-disjoint seasons."""

    by_stem: dict[str, list[WeakSignal]] = {}
    for signal in signals.values():
        if signal.league != league:
            continue
        stem, had_suffix = era_stem(signal.name)
        if not had_suffix:
            continue
        by_stem.setdefault(stem, []).append(signal)

    groups: list[EraGroup] = []
    for stem, members in sorted(by_stem.items()):
        if len(members) < 2:
            continue
        ordered = tuple(sorted(members, key=lambda s: s.seasons[0]))
        disjoint = all(
            ordered[i].seasons[1] < ordered[i + 1].seasons[0] for i in range(len(ordered) - 1)
        )
        if not disjoint:
            continue
        parent = signals.get(stem)
        if parent is not None and parent.league != league:
            parent = None
        groups.append(EraGroup(stem=stem, league=league, members=ordered, parent=parent))
    return groups


def signal_sign(signal: WeakSignal) -> str:
    if signal.effect > 0.0:
        return "+"
    if signal.effect < 0.0:
        return "-"
    return "0"


def group_sign_flip(group: EraGroup) -> bool:
    """True iff at least one member is strictly positive and one strictly negative.

    An exact-zero member (no split-half sign to flip from/to) never counts as
    either side, so a "-, 0" group is reported as flat, not a flip -- this
    deliberately does not decide whether a near-zero straddle (e.g. -0.04 vs
    +0.09) is a *meaningful* flip; that judgement belongs in the write-up,
    not this mechanical flag.
    """

    signs = {signal_sign(m) for m in group.members}
    return "+" in signs and "-" in signs


def fmt_interval(signal: WeakSignal) -> str:
    if signal.interval is None:
        return "n/a"
    low, high = signal.interval
    return f"[{low:+.4f}, {high:+.4f}]"


def fmt_p(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_member_row(signal: WeakSignal, *, markdown: bool) -> list[str]:
    seasons = f"{signal.seasons[0]}-{signal.seasons[1]}"
    n = "n/a" if signal.sample_games is None else str(signal.sample_games)
    effect = f"{signal.effect:+.4f}"
    return [
        signal.name,
        seasons,
        n,
        effect,
        fmt_interval(signal),
        fmt_p(signal.probability_positive),
        signal_sign(signal),
    ]


HEADERS = ["entry", "seasons", "n games", "effect (pts)", "95% interval", "P+", "sign"]


def render_table(rows: list[list[str]], *, markdown: bool) -> str:
    if not markdown:
        widths = [
            max(len(HEADERS[i]), *(len(row[i]) for row in rows)) if rows else len(HEADERS[i])
            for i in range(len(HEADERS))
        ]
        lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(HEADERS))]
        lines.append("-+-".join("-" * w for w in widths))
        for row in rows:
            lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        return "\n".join(lines)
    lines = ["| " + " | ".join(HEADERS) + " |"]
    lines.append("|" + "|".join("---" for _ in HEADERS) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_group(group: EraGroup, *, markdown: bool) -> str:
    rows = [render_member_row(m, markdown=markdown) for m in group.members]
    pieces = [f"### `{group.stem}`  ({group.league})", ""]
    pieces.append(render_table(rows, markdown=markdown))
    pieces.append("")

    pool = pooled_effect(list(group.members), method="random")
    tau2 = pool["heterogeneity_tau_squared"]
    interval = pool["interval"]
    interval_str = "n/a" if interval is None else f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
    flips = group_sign_flip(group)
    pieces.append(
        f"- Between-era heterogeneity (DerSimonian-Laird): **tau^2 = {tau2:.4f}** "
        f"({pool['effect_units']}^2); random-effects pooled point (informational, "
        f"NOT an independent confirmatory look) = {pool['pooled_effect']:+.4f} "
        f"{interval_str}, `excludes_zero`={pool['excludes_zero']}."
    )
    pieces.append(f"- Sign flips across the era boundary: **{flips}**.")

    if group.parent is not None:
        p = group.parent
        pieces.append(
            f"- Full-range parent (OVERLAPPING both eras at once -- not an "
            f"independent vote): `{p.name}`, seasons {p.seasons[0]}-{p.seasons[1]}, "
            f"n={p.sample_games}, effect {p.effect:+.4f}, interval {fmt_interval(p)}, "
            f"P+ {fmt_p(p.probability_positive)}."
        )
    else:
        pieces.append(
            "- No exact-stem full-range parent entry exists in the registry for this construct."
        )

    notes = [m for m in group.members if m.notes.strip()]
    if notes:
        pieces.append("- Registry notes (read directly, verbatim):")
        for m in notes:
            pieces.append(f"  - `{m.name}`: {m.notes.strip()}")

    return "\n".join(pieces)


def build_report(*, league: str, markdown: bool, registry_path: Path | None = None) -> str:
    path = registry_path if registry_path is not None else default_registry_path()
    registry = load_registry(path)
    groups = find_era_groups(registry.signals, league=league)

    out = [
        "# Era-magnitude report (mechanical, read-only)",
        "",
        f"Source: `{path}`. League: `{league}`. {len(groups)} era-split group(s) found.",
        "",
        "This script records nothing and closes nothing. Per AGENTS.md, an interval "
        "crossing zero is never grounds to close a line of work; every table below "
        "reports probability_positive alongside the interval, never a binary "
        '"contains zero" read.',
        "",
    ]
    for group in groups:
        out.append(render_group(group, markdown=markdown))
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit GitHub-flavoured Markdown pipe tables (for pasting into a doc) "
        "instead of the default fixed-width text tables.",
    )
    parser.add_argument("--league", default="nfl", choices=("nfl", "cfb"))
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override registry path (default: registry/weak_signals.json, honouring "
        "NFL_ATS_REGISTRY_DIR).",
    )
    args = parser.parse_args(argv)
    print(build_report(league=args.league, markdown=args.markdown, registry_path=args.registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
