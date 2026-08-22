"""Alternate "Gridiron Observatory" skin assets for the generated public site.

Nothing here changes default rendering: the integrator lane injects
:func:`render_theme_toggle_head` output and ships the two asset files; until
then this package is inert. String constants only -- no I/O at import time.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
OBSERVATORY_CSS_PATH = PACKAGE_DIR / "observatory.css"
TOGGLE_JS_PATH = PACKAGE_DIR / "toggle.js"

THEME_CLASS = "theme-obs"
DAY_MODE_ATTRIBUTE = 'data-mode="day"'
STORAGE_KEY = "site-theme-pref"
TOGGLE_BUTTON_ID = "theme-toggle"
TOGGLE_MOUNT_ID = "theme-toggle-mount"

_CSS_NAME = "observatory.css"
_JS_NAME = "toggle.js"


def render_theme_toggle_head(asset_prefix: str = ".") -> str:
    """The exact snippets the integrator injects: link + deferred script + mount."""

    return (
        f'<link rel="stylesheet" href="{asset_prefix}/{_CSS_NAME}">\n'
        f'<script src="{asset_prefix}/{_JS_NAME}" defer></script>\n'
        f'<div id="{TOGGLE_MOUNT_ID}" aria-live="polite"></div>'
    )
