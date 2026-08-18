"""Small values shared by more than one dashboard page.

Streamlit page scripts under ``app_pages/`` execute as top-level script code
on every rerun, not as importable modules (see ``picks.py``'s own note on
this) -- so importing one page from another to reuse a bare module-level
constant would re-execute that page's Streamlit calls outside a page context
and break. A constant more than one page needs has to live somewhere that is
not itself a page; this module is that "somewhere."
"""

from __future__ import annotations

# A "strong lean" is a fair-line disagreement with the market of at least this
# many points. It gates the market-disagreement explanation on picks.py and
# the strong-lean badge on workbench.py, never the pick itself: every pool
# pick is forced, and our confidence ordering has NOT proven predictive (see
# the findings page), so the lean is a narrative marker, never a weighting.
# Shared here so the two pages cannot silently drift to different thresholds.
STRONG_LEAN_POINTS = 1.5
