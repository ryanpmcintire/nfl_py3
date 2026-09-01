"""Builds the full ATS Terminal site: This Week, The Model, and What We've
Learned -- three pages, at the site root.

This is the real-publish-path replacement for
:func:`nfl_ats.public_board.build_public_site`'s old single-skin, seven-page
output. It reuses that module's loaders and data-integrity guards
exclusively through :func:`nfl_ats.board_site_content.load_site_content`
(which in turn reuses :func:`nfl_ats.board_content.load_board_content` for
the This Week page) -- this module itself never opens an artifact. The
Terminal renderer (:mod:`nfl_ats.board_terminal`) is a pure function over
the shared content dataclasses; this module's only job is to call the right
renderer for the right page and lay out the site tree.

Site layout (flat, no subdirectory)::

    index.html      -- This Week
    model.html      -- The Model
    findings.html   -- What We've Learned

2026-08-31 owner redirect: the Cover Desk skin (a second, parallel
``desk/`` tree plus a ``terminal/`` subdirectory, a top-level redirect page,
and a header skin-toggle control) is dropped entirely -- "Let's drop the
Desk theme altogether and just focus on the Terminal theme." This module
used to be named ``build_two_skin_site`` and return a
``terminal/``/``desk/``-prefixed path per page; it is now ``build_site`` and
returns exactly ``{"index.html", "model.html", "findings.html"}``, each
mapped straight to the site root.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nfl_ats import board_terminal
from nfl_ats.board_site_content import SiteContent, load_site_content


def build_site(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    registry_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> dict[str, str]:
    """Build every page of the ATS Terminal site.

    Returns ``{relative_path: complete HTML document}`` -- one entry per
    :data:`nfl_ats.board_terminal.SITE_PAGES` filename, each a bare
    site-root filename (``"index.html"``, ``"model.html"``,
    ``"findings.html"``). Raises exactly when
    :func:`nfl_ats.board_site_content.load_site_content` raises (no
    synchronized active model, a stale player-arrests snapshot under
    ``require_fresh_arrest_overlay=True``, a curation/ledger drift, etc.) --
    the same fail-loud contract ``public_board.build_public_site`` has
    always had for these conditions; every OTHER optional artifact still
    degrades quietly, per each page's own content loader.

    ``require_fresh_arrest_overlay`` defaults to ``True`` to match
    ``build_public_site``'s real-publish default; pass ``False`` for a
    rehearsal/scratch build (see ``scripts/build_full_site.py``).
    """

    content: SiteContent = load_site_content(
        artifacts_root,
        data_root=data_root,
        registry_root=registry_root,
        generated_at=generated_at,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
    )

    return {
        board_terminal.PICKS_PAGE: board_terminal.render(content.board),
        board_terminal.MODEL_PAGE: board_terminal.render_model_page(content.model),
        board_terminal.FINDINGS_PAGE: board_terminal.render_findings_page(content.findings),
    }


__all__ = ["build_site"]
