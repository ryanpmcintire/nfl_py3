"""BET-09 guard: paper mode default; no automated wager-placement path.

Enforces AGENTS.md's research invariant ("This is a research and
paper-decision project; do not add automated wagering.") in code, not just
prose, per this project's "directives now enforced in code" discipline
(memory: directives-now-enforced-in-code -- a rule that only lives in prose
does not bind).

Three checks:

(a) No `pyproject.toml` dependency (any dependency group) matches a
    denylist of sportsbook/exchange wagering-client package names.
(b) No wager-PLACEMENT verb (``place_bet``, ``place_wager``, ``submit_bet``,
    a POST to a ``/bets``-shaped endpoint) appears anywhere in ``src/`` or
    ``scripts/``, outside the explicitly allowlisted read-only odds modules.
(c) A paper-only/limitations statement exists in ``docs/``.

Kept fast (<1s) and deterministic: static text scanning of files already on
disk, no imports of the package under test, no network, no fixtures.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# (a) No dependency matches a known wagering-client package name.
# ---------------------------------------------------------------------------

# inferred: plausible PyPI/package names for wagering clients of the major
# US sportsbooks and exchanges, plus generic wagering-client names, guessed
# from the brand/product names themselves -- none of these packages is
# claimed to exist on PyPI or to have ever been considered as a dependency.
# The point of the denylist is to fail loudly the day a dependency matching
# one of these names is EVER added, not to prove today's absence of
# something nobody proposed.
WAGER_CLIENT_DENYLIST = {
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "caesarssportsbook",
    "pointsbet",
    "kalshi",
    "polymarket",
    "betfair",
    "pinnacle",
    "sportsbook",
    "wager",
    "bet365",
    "unibet",
    "bovada",
    "mybookie",
    "prizepicks",
    "underdogfantasy",
    "espnbet",
    "barstoolsportsbook",
    "wynnbet",
    "betrivers",
    "betway",
    "williamhill",
    "ladbrokes",
    "skybook",
    "888sport",
    "betonline",
    "sxbet",
    "novig",
}


def _dependency_names() -> set[str]:
    """Every declared dependency's bare distribution name, from every group.

    Covers ``[project.dependencies]``, ``[project.optional-dependencies]``,
    and every group under ``[dependency-groups]`` -- the last is
    `[read, pyproject.toml:27]`'s ``dev`` group, and this stays generic so a
    future group is covered without editing this test.
    """

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    raw_specs: list[str] = list(project.get("dependencies", []))
    for extra_deps in project.get("optional-dependencies", {}).values():
        raw_specs.extend(extra_deps)
    for group_deps in data.get("dependency-groups", {}).values():
        raw_specs.extend(group_deps)

    normalized: set[str] = set()
    for spec in raw_specs:
        # PEP 508 requirement string: keep the bare distribution name, drop
        # any version specifier/marker/extras, fold - and _ together (PyPI
        # treats them as equivalent).
        name = re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].strip()
        normalized.add(name.lower().replace("_", "-"))
    return normalized


def test_no_dependency_matches_a_known_wagering_client_name() -> None:
    deps = _dependency_names()
    assert deps, "expected at least one declared dependency to check"
    hits = {dep: banned for dep in deps for banned in WAGER_CLIENT_DENYLIST if banned in dep}
    assert not hits, (
        "pyproject.toml declares a dependency matching the wagering-client "
        f"denylist: {hits}. This is a research and paper-decision project "
        "(AGENTS.md); do not add automated wagering."
    )


# ---------------------------------------------------------------------------
# (b) No wager-PLACEMENT verb outside the read-only odds client.
# ---------------------------------------------------------------------------

WAGER_PLACEMENT_PATTERN = re.compile(
    r"\bplace_bet\b|\bplace_wager\b|\bsubmit_bet\b|\bplace_order\b"
    r"""|['"]/bets['"]|\.post\([^)]*['"]/bets"""
)

# nfl_ats.odds: pure payout/vig ARITHMETIC on paper decisions (`choose_bet`,
# `settle_bet`, `implied_probability`) -- no network call anywhere in the
# module [read, src/nfl_ats/odds.py].
# nfl_ats.odds_backfill / nfl_ats.market_data: The Odds API historical/live
# quote FETCH only -- `urllib.request.Request` with no `data=` argument,
# i.e. always a GET, never a POST [read, src/nfl_ats/odds_backfill.py:224-226,
# src/nfl_ats/market_data.py:249-251]. This project's only odds integration
# is read-only market quotes; it never places a wager.
ALLOWLISTED_READONLY_ODDS_MODULES = {
    SRC_DIR / "nfl_ats" / "odds.py",
    SRC_DIR / "nfl_ats" / "odds_backfill.py",
    SRC_DIR / "nfl_ats" / "market_data.py",
}


def _python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_no_wager_placement_verb_outside_the_readonly_odds_client() -> None:
    candidates = _python_files(SRC_DIR) + _python_files(SCRIPTS_DIR)
    assert candidates, "expected to scan at least one Python file under src/ or scripts/"

    offenders: list[str] = []
    for path in candidates:
        if path in ALLOWLISTED_READONLY_ODDS_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        if WAGER_PLACEMENT_PATTERN.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "wager-placement verb found outside the allowlisted read-only odds "
        f"client: {offenders}. This is a research and paper-decision "
        "project; do not add automated wagering."
    )


def test_allowlisted_odds_modules_actually_exist_and_stay_read_only() -> None:
    """Guards the allowlist itself: if one of these files is ever renamed or
    deleted, or ever grows a wager-placement verb, this must fail rather
    than silently allowlisting nothing (an empty allowlist would make the
    check above vacuous for these exact files)."""

    for path in ALLOWLISTED_READONLY_ODDS_MODULES:
        assert path.is_file(), f"expected allowlisted odds module to exist: {path}"
        text = path.read_text(encoding="utf-8")
        assert not WAGER_PLACEMENT_PATTERN.search(text), (
            f"{path} is allowlisted as read-only-quotes but now contains a "
            "wager-placement verb; re-audit before keeping it allowlisted"
        )


# ---------------------------------------------------------------------------
# (c) A paper-only/limitations statement exists in docs/.
# ---------------------------------------------------------------------------

RESPONSIBLE_USE_DOC = REPO_ROOT / "docs" / "responsible_use.md"


def test_a_paper_only_limitations_statement_exists() -> None:
    """No sentence in README.md or docs/*.md said, in so many words, "paper
    mode is the default and there is no automated wager-placement path"
    [read, README.md:548-553 -- that "Responsible use" section advises a
    reader who might wager real money; it does not describe this codebase's
    own paper-only architecture]. `docs/responsible_use.md` was added by
    this work package to say so explicitly, and is asserted on here per
    instructions rather than editing README.md."""

    assert RESPONSIBLE_USE_DOC.is_file(), (
        "docs/responsible_use.md is required as the paper-only/limitations "
        "statement this guard checks"
    )
    lowered = RESPONSIBLE_USE_DOC.read_text(encoding="utf-8").lower()
    assert "paper" in lowered
    assert "no code path from a model prediction to money" in lowered
    assert "wager" in lowered
    assert "automated wagering" in lowered
