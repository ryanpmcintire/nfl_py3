"""What the honest widening does to every verdict already on the books (estvar).

Research item: ``docs/estimation_variance.md``. Defect D2 says every recorded
interval in this project is conditional on one model fit and is therefore too
narrow; defect D4 says an interval built on too few blocks is not an interval
at all. Both are properties of the machinery, so they apply to results already
recorded, not only to future ones.

This script answers the only question that matters about the back catalogue --
**which recorded verdicts change** -- without re-running a single experiment.
It reads ``registry/weak_signals.json`` and ``registry/rotation_registry.json``
(READ ONLY; it never writes either file), recovers each entry's conditional
standard error from whichever of ``standard_error`` / ``interval`` /
``probability_positive`` it recorded, widens it by the measured inflation
factor, and reports the honest ``probability_positive`` next to the recorded
one.

Four things it deliberately does NOT do:

- It does not re-estimate any effect. Widening changes the interval, never the
  point estimate, so ``effect`` is carried through untouched.
- It does not treat "the honest interval now contains zero" as a negative.
  Per ``AGENTS.md`` that is never grounds for rejection, and at this
  evaluator's ~2-point resolution it is the EXPECTED outcome for a real,
  small signal. The reported column is ``probability_positive``, and the
  classification column asks only whether the RECORDED classification still
  follows from its own recorded evidence.
- It does not apply the refit widening to entries outside its scope. D2 is a
  property of comparisons between differently FITTED models. A comparison that
  holds the mean model fixed and changes only how a fixed residual sample is
  READ carries no refit variance, so its factor is exactly 1.0.
- It does not report a widened ``probability_positive`` for an entry whose
  block count is below the measured D4 floor. There is nothing to widen: the
  recorded bounds were never a 95% interval, so scaling them would launder a
  degenerate number into a plausible-looking one.

Both registries are compared on the SAME normal reference distribution: the
recorded bootstrap ``probability_positive`` and the honest one are not directly
comparable (one is a bootstrap tail fraction, the other a normal tail), so the
output carries ``normal_at_factor_1`` -- the same normal calculation with no
widening -- and the honest-vs-recorded gap should be read against THAT, not
against the recorded bootstrap value. The two agree to about 0.01-0.02 on every
entry that records both, which is the sanity check that the approximation is
not doing any of the work.

Usage::

    ./.tools/uv.exe run --no-sync python scripts/estvar_blast_radius.py
    ./.tools/uv.exe run --no-sync python scripts/estvar_blast_radius.py --markdown
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nfl_ats.estimation_variance import (
    MIN_BLOCKS_FOR_INTERVAL,
    RELIABLE_BLOCKS_FOR_INTERVAL,
    block_count_verdict,
    inflate_recorded_interval,
)

REPO = Path(__file__).resolve().parents[1]
WEAK_SIGNALS = REPO / "registry/weak_signals.json"
ROTATION = REPO / "registry/rotation_registry.json"
Z95 = 1.959963985


@dataclass(frozen=True)
class Mechanism:
    """Which resampling a family's uncertainty actually needs, and how big it is.

    **One factor does not fit all families.** The inflation factor is a property
    of the mechanism the comparison varies, not a global constant, so applying a
    single number across the registry would be exactly the underived-constant
    defect this work exists to remove.
    """

    name: str
    low: float
    central: float
    high: float
    provenance: str


#: Comparisons between differently FITTED models (different feature columns or
#: different ridge alpha). MEASURED this session on real CFB clean core (8,933
#: games, 199 week blocks, f=19.8%, 120 paired refits):
#: ``scripts/estvar_refit_intervals.py --study cfb`` gives an interaction-free
#: training component of 0.042 accuracy points against a conditional SD of
#: 0.512, i.e. a factor of **1.003 with a one-sided 95% upper bound of 1.099**.
#: The published 17-58% band came from adding the FIXED-GAMES refit spread
#: (0.419 pts, factor 1.293), which double-counts the training-by-game
#: interaction the game bootstrap already carries. See
#: ``docs/estimation_variance.md`` Part II.
REFIT_MECHANISM = Mechanism(
    name="refit (differently fitted models)",
    low=1.000,
    central=1.003,
    high=1.099,
    provenance="measured 2026-08-18 on CFB clean core, this session",
)

#: Comparisons that hold the fitted mean model FIXED and vary only how the
#: residual sample is READ. ``docs/estimation_variance.md`` sec 7 already
#: disclaimed the refit bootstrap for these; a parallel agent built the
#: mechanism-appropriate bootstrap and measured 2.07x / 2.29x. REPORTED, not
#: verified here -- I did not re-run it.
READER_MECHANISM = Mechanism(
    name="reader (fixed mean model, residual sample re-read)",
    low=2.07,
    central=2.07,
    high=2.29,
    provenance="reported by a parallel agent 2026-08-18, UNVERIFIED by this script",
)

#: CFB role-continuity, measured family-specifically by a parallel agent on a
#: cheaper refit cadence. REPORTED, not verified here.
ROLE_MECHANISM = Mechanism(
    name="refit, family-specific",
    low=1.438,
    central=1.438,
    high=1.438,
    provenance="reported by a parallel agent 2026-08-18, UNVERIFIED by this script",
)

_RESIDUAL_LOCATION_SIGNALS = frozenset(
    {
        "residual_location_recency_hl100_cfb",
        "residual_location_recency_hl200_cfb",
        "residual_location_recency_hl400_cfb",
        "residual_location_recency_hl800_cfb",
        "residual_location_shrink_025_cfb",
        "residual_location_shrink_050_cfb",
        "residual_location_shrink_075_cfb",
        "residual_location_shrink_100_cfb",
    }
)


def mechanism_for(identifier: str) -> tuple[Mechanism, str]:
    """Pick the mechanism a family's uncertainty actually needs, with a caveat."""

    if identifier in _RESIDUAL_LOCATION_SIGNALS:
        return READER_MECHANISM, ""
    if identifier == "ecdf_smoothing_accuracy":
        return READER_MECHANISM, (
            "reader mechanism, but the 2.07-2.29x factor was measured on the recency/shrink "
            "screens, not on ECDF smoothing -- treat as an extrapolation"
        )
    if identifier.startswith("cfb_role_continuity"):
        return ROLE_MECHANISM, ""
    return REFIT_MECHANISM, ""


_POINTS = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*pts")
#: Only an interval the notes explicitly label as block-bootstrapped is trusted;
#: a bare bracket may be in any units (see rotation_window_evidence).
_BLOCKED_INTERVAL = re.compile(
    r"(week|season)-blocked\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]"
)
_PAIRED_SE = re.compile(r"paired\s+SE\s+([\d.]+)\s*pts")
_PROBABILITY = re.compile(r"P\+\s*=?\s*([01]?\.\d+)")
_WEEKS = re.compile(r"(\d+)\s*weeks")


@dataclass(frozen=True)
class Evidence:
    """The continuous evidence an entry recorded, however it recorded it."""

    effect: float | None
    lower: float | None
    upper: float | None
    probability_positive: float | None
    blocks: int | None
    source: str


@dataclass(frozen=True)
class BlastRow:
    identifier: str
    registry: str
    classification: str
    effect: float | None
    blocks: int | None
    games: int | None
    recorded_probability_positive: float | None
    evidence_source: str
    conditional_sd: float | None
    normal_at_factor_1: float | None
    mechanism: str
    mechanism_provenance: str
    factor: float
    honest_probability_positive: float | None
    honest_low_factor: float | None
    honest_high_factor: float | None
    d4_degenerate: bool
    d4_marginal: bool
    classification_holds: str
    note: str


def weak_signal_evidence(entry: dict[str, Any]) -> Evidence:
    effect = entry.get("effect")
    interval = entry.get("interval")
    standard_error = entry.get("standard_error")
    probability = entry.get("probability_positive")
    lower = upper = None
    source = "none"
    if standard_error is not None and effect is not None:
        lower = float(effect) - Z95 * float(standard_error)
        upper = float(effect) + Z95 * float(standard_error)
        source = "standard_error"
    elif isinstance(interval, list) and len(interval) == 2 and interval[1] > interval[0]:
        lower, upper = float(interval[0]), float(interval[1])
        source = "interval"
    elif probability is not None and 0.0 < float(probability) < 1.0 and effect is not None:
        source = "probability_positive"
    return Evidence(
        effect=None if effect is None else float(effect),
        lower=lower,
        upper=upper,
        probability_positive=None if probability is None else float(probability),
        blocks=entry.get("sample_blocks"),
        source=source,
    )


def rotation_window_evidence(window: dict[str, Any]) -> Evidence:
    """Recover a window's evidence, falling back to parsing its ``notes`` prose.

    The rotation-registry schema has fields for ``probability_positive`` and
    ``verdict`` but NONE for the effect size, the interval, the standard error
    or the block count, so on every spent window those numbers exist only inside
    a free-text ``notes`` string. Parsing prose to recover them is not a design
    anyone would choose -- it is the finding: **the schema does not retain
    enough to re-read its own verdicts.**

    The parsing is deliberately strict, because a loose version got this wrong
    once already. ``pbp_drive_bundle``'s notes contain a Brier interval
    ``[-0.00553, +0.00072]``; pairing that with the same note's -0.08
    ACCURACY-POINT effect produced a confident-looking ``probability_positive``
    of 0.0000 out of two different units. So a bare bracket is never accepted:
    an interval must be introduced by the project's own ``week-blocked`` /
    ``season-blocked`` phrasing, which is only ever used for the accuracy-point
    metric.
    """

    notes = str(window.get("notes") or "")
    probability = window.get("probability_positive")
    points = _POINTS.search(notes)
    blocked = _BLOCKED_INTERVAL.search(notes)
    standard_error = _PAIRED_SE.search(notes)
    prose_probability = _PROBABILITY.search(notes)
    weeks = _WEEKS.search(notes)

    effect = float(points.group(1)) if points else None
    if probability is None and prose_probability is not None:
        probability = float(prose_probability.group(1))

    lower = upper = None
    source = "none"
    if effect is not None and standard_error is not None:
        sigma = float(standard_error.group(1))
        lower, upper = effect - Z95 * sigma, effect + Z95 * sigma
        source = "notes_paired_se"
    elif effect is not None and blocked is not None:
        lower, upper = float(blocked.group(2)), float(blocked.group(3))
        source = "notes_blocked_interval"
    elif effect is not None and probability is not None:
        source = "notes_effect_and_probability"
    elif effect is not None:
        source = "notes_effect_only"
    elif probability is not None:
        source = "probability_positive_only"
    return Evidence(
        effect=effect,
        lower=lower,
        upper=upper,
        probability_positive=None if probability is None else float(probability),
        blocks=int(weeks.group(1)) if weeks else None,
        source=source,
    )


def widen(evidence: Evidence, factor: float) -> tuple[float, float] | None:
    """Return (conditional_sd, honest probability_positive) under ``factor``."""

    if evidence.effect is None:
        return None
    try:
        widened = inflate_recorded_interval(
            evidence.effect,
            evidence.lower,
            evidence.upper,
            inflation_factor=factor,
            probability_positive=evidence.probability_positive,
        )
    except ValueError:
        return None
    conditional_sd = (widened.upper - widened.lower) / (2.0 * Z95 * factor)
    return conditional_sd, widened.probability_positive


def _verdict_text(
    classification: str,
    degenerate: bool,
    mechanism: Mechanism,
    recorded: float | None,
    honest: float | None,
) -> tuple[str, str]:
    """Return (classification_holds, note)."""

    if degenerate:
        return (
            "HOLDS, BUT ITS INTERVAL IS VOID",
            f"below the measured {MIN_BLOCKS_FOR_INTERVAL}-block floor, so the recorded bounds "
            "were never a 95% interval and are not widened here. The verdict has to rest on its "
            "non-interval evidence or be re-run.",
        )
    if classification == "unresolved_below_power":
        if honest is None:
            return (
                "HOLDS, ON NO RECORDED EVIDENCE",
                "the entry records no interval, no standard error and no probability_positive, "
                "so there is nothing to widen -- and nothing to read either. 'Unresolved' is "
                "still the right bucket, but the row carries no continuous evidence at all.",
            )
        return (
            "HOLDS",
            "'unresolved' is where a wider interval lands anyway; only the stated confidence "
            "moves, and it was never a decision gate (AGENTS.md: forced picks).",
        )
    if classification in {"refuted_mechanism", "closed_negative"}:
        if honest is None:
            return (
                "CANNOT BE CHECKED",
                "terminal verdict with no interval and no probability_positive recorded -- "
                "there is no continuous evidence to widen.",
            )
        return (
            "HOLDS ONLY IF MECHANISTIC",
            f"terminal verdict; honest P+ {honest:.4f} vs recorded {recorded} under the "
            f"{mechanism.name} factor {mechanism.central:g}x. If the closure rested on the "
            "interval rather than on a mechanism, it is no longer supported.",
        )
    if classification == "open":
        return (
            "HOLDS",
            "family is open with no spent window -- there is no recorded verdict for a wider "
            "interval to disturb.",
        )
    if classification in {"confirmed", "unresolved"}:
        if honest is None:
            return (
                "CANNOT BE CHECKED",
                "the window records no interval, no standard error and no probability_positive, "
                "so it carries no continuous evidence to widen.",
            )
        return (
            "HOLDS",
            f"the predeclared gate was set against the recorded P+; honest P+ is {honest:.4f}. "
            "Per AGENTS.md a promotion bar governs what the docs may CLAIM, never which card is "
            "PLAYED.",
        )
    return ("UNMAPPED", "no rule maps this classification")


def build_row(
    *,
    identifier: str,
    registry: str,
    classification: str,
    evidence: Evidence,
    games: int | None,
    mechanism: Mechanism,
    caveat: str = "",
) -> BlastRow:
    verdict = block_count_verdict(evidence.blocks) if evidence.blocks else None
    degenerate = bool(verdict and verdict.degenerate)
    baseline = widen(evidence, 1.0)
    central = None if degenerate else widen(evidence, mechanism.central)
    low = None if degenerate else widen(evidence, mechanism.low)
    high = None if degenerate else widen(evidence, mechanism.high)
    honest = None if central is None else central[1]
    holds, note = _verdict_text(
        classification, degenerate, mechanism, evidence.probability_positive, honest
    )
    if caveat:
        note = f"{note} [{caveat}]"
    return BlastRow(
        identifier=identifier,
        registry=registry,
        classification=classification,
        effect=evidence.effect,
        blocks=evidence.blocks,
        games=games,
        recorded_probability_positive=evidence.probability_positive,
        evidence_source=evidence.source,
        conditional_sd=None if baseline is None else baseline[0],
        normal_at_factor_1=None if baseline is None else baseline[1],
        mechanism=mechanism.name,
        mechanism_provenance=mechanism.provenance,
        factor=mechanism.central,
        honest_probability_positive=honest,
        honest_low_factor=None if low is None else low[1],
        honest_high_factor=None if high is None else high[1],
        d4_degenerate=degenerate,
        d4_marginal=bool(verdict and verdict.marginal),
        classification_holds=holds,
        note=note,
    )


def weak_signal_rows(payload: dict[str, Any]) -> list[BlastRow]:
    rows: list[BlastRow] = []
    for identifier, entry in sorted(payload["signals"].items()):
        mechanism, caveat = mechanism_for(identifier)
        rows.append(
            build_row(
                identifier=identifier,
                registry="weak_signals",
                classification=str(entry.get("classification")),
                evidence=weak_signal_evidence(entry),
                games=entry.get("sample_games"),
                mechanism=mechanism,
                caveat=caveat,
            )
        )
    return rows


def rotation_rows(payload: dict[str, Any]) -> list[BlastRow]:
    rows: list[BlastRow] = []
    for family, entry in sorted(payload["families"].items()):
        mechanism, caveat = mechanism_for(family)
        windows = entry.get("windows") or []
        if not windows:
            rows.append(
                build_row(
                    identifier=family,
                    registry="rotation",
                    classification=str(entry.get("status")),
                    evidence=Evidence(None, None, None, None, None, "none"),
                    games=None,
                    mechanism=mechanism,
                    caveat=caveat,
                )
            )
            continue
        for index, window in enumerate(windows):
            rows.append(
                build_row(
                    identifier=f"{family}[{index}] seasons={window.get('seasons')}",
                    registry="rotation",
                    classification=str(window.get("verdict")),
                    evidence=rotation_window_evidence(window),
                    games=None,
                    mechanism=mechanism,
                    caveat=caveat,
                )
            )
    return rows


def _fmt(value: float | None, digits: int = 4) -> str:
    return "none" if value is None else f"{value:.{digits}f}"


def render_markdown(rows: list[BlastRow]) -> str:
    lines = [
        "| signal_id | classification | recorded P+ | honest P+ | band | mechanism | "
        "classification still holds? |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        band = (
            "-"
            if row.honest_low_factor is None or row.honest_high_factor is None
            else f"{row.honest_low_factor:.3f}-{row.honest_high_factor:.3f}"
        )
        lines.append(
            f"| `{row.identifier}` | {row.classification} | "
            f"{_fmt(row.recorded_probability_positive)} | "
            f"{_fmt(row.honest_probability_positive)} | {band} | "
            f"{row.mechanism} {row.factor:g}x | "
            f"**{row.classification_holds}** -- {row.note} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-signals", type=Path, default=WEAK_SIGNALS)
    parser.add_argument("--rotation", type=Path, default=ROTATION)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    weak = json.loads(args.weak_signals.read_text(encoding="utf-8"))
    rotation = json.loads(args.rotation.read_text(encoding="utf-8"))
    rows = weak_signal_rows(weak) + rotation_rows(rotation)

    if args.markdown:
        print(render_markdown(rows))
    else:
        for row in rows:
            print(f"--- {row.identifier} [{row.registry}] {row.classification}")
            print(
                f"    effect={row.effect} blocks={row.blocks} games={row.games} "
                f"evidence={row.evidence_source} conditional_sd={_fmt(row.conditional_sd)}"
            )
            print(
                f"    recorded P+={_fmt(row.recorded_probability_positive)} "
                f"normal@1.0={_fmt(row.normal_at_factor_1)} "
                f"honest P+={_fmt(row.honest_probability_positive)} (factor {row.factor}, band "
                f"{_fmt(row.honest_low_factor, 3)}-{_fmt(row.honest_high_factor, 3)})"
            )
            print(
                f"    D4: degenerate={row.d4_degenerate} marginal={row.d4_marginal} "
                f"(floor {MIN_BLOCKS_FOR_INTERVAL}, reliable {RELIABLE_BLOCKS_FOR_INTERVAL})"
            )
            print(f"    {row.classification_holds}: {row.note}")

    print("\n== Summary ==")
    changed = [
        row
        for row in rows
        if row.honest_probability_positive is not None
        and row.normal_at_factor_1 is not None
        and (row.honest_probability_positive >= 0.90) != (row.normal_at_factor_1 >= 0.90)
    ]
    print(f"Entries crossing the 0.90 claim threshold under the honest widening: {len(changed)}")
    for row in changed:
        print(f"  {row.identifier}")
    reclassified = [row for row in rows if row.classification_holds != "HOLDS"]
    print(f"Entries whose recorded classification does NOT simply hold: {len(reclassified)}")
    for row in reclassified:
        print(f"  {row.identifier}: {row.classification_holds}")
    print(f"Entries with no widenable evidence: {sum(1 for r in rows if r.effect is None)}")
    for row in rows:
        if row.effect is None:
            print(f"  {row.identifier} [{row.registry}] evidence={row.evidence_source}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
