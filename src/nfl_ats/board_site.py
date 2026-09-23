from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from nfl_ats import board_interactive, board_terminal, board_week_navigation
from nfl_ats.board_site_content import SiteContent, load_site_content


def build_site(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    registry_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> dict[str, str]:

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    content: SiteContent = load_site_content(
        artifacts_root,
        data_root=data_root,
        registry_root=registry_root,
        generated_at=generated,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
    )

    archives = board_week_navigation.archive_boards(content, data_root, generated)
    pages = {
        board_terminal.PICKS_PAGE: board_terminal.render(content.board),
        board_terminal.MODEL_PAGE: board_terminal.render_model_page(content.model),
        board_terminal.HISTORY_PAGE: board_terminal.render_history_page(content.history),
        board_terminal.FINDINGS_PAGE: board_terminal.render_findings_page(content.findings),
    }
    enhanced = {
        page: board_interactive.enhance(
            document, page=page, board=content.board, archived_boards=archives
        )
        for page, document in pages.items()
    }
    enhanced[board_terminal.PICKS_PAGE] = board_week_navigation.enhance(
        enhanced[board_terminal.PICKS_PAGE],
        content,
        data_root=data_root,
        generated_at=generated,
        archived_boards=archives,
    )
    return enhanced


__all__ = ["build_site"]
