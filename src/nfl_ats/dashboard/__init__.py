"""Shared design system for the public GitHub Pages site.

The site generator (:mod:`nfl_ats.public_board`) composes
:mod:`nfl_ats.dashboard.theme`, :mod:`nfl_ats.dashboard.viz`, and
:mod:`nfl_ats.dashboard.findings_content` into the pages served from ``docs/``.
Every module here is pure HTML/CSS/text: nothing imports a web-framework
runtime, and nothing writes to ``data/`` or ``artifacts/``.
"""

from __future__ import annotations

__all__: list[str] = []
