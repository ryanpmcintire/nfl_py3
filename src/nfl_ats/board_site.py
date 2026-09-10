from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nfl_ats import board_interactive, board_terminal
from nfl_ats.board_site_content import SiteContent, load_site_content


def build_site(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    registry_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> dict[str, str]:

    content: SiteContent = load_site_content(
        artifacts_root,
        data_root=data_root,
        registry_root=registry_root,
        generated_at=generated_at,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
    )

    pages = {
        board_terminal.PICKS_PAGE: board_terminal.render(content.board),
        board_terminal.MODEL_PAGE: board_terminal.render_model_page(content.model),
        board_terminal.HISTORY_PAGE: board_terminal.render_history_page(content.history),
        board_terminal.FINDINGS_PAGE: board_terminal.render_findings_page(content.findings),
    }
    return {
        page: board_interactive.enhance(document, page=page, board=content.board)
        for page, document in pages.items()
    }


__all__ = ["build_site"]
